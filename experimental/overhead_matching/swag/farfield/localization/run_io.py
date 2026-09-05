"""Self-describing run directory for localization runs (design doc §7.5).

Layout (all consumers — plots, tests, viewers — read only this):
  manifest.json             typed artifact identity/completion manifest
  run_manifest.json         RunManifest (config echo, provenance, history hash)
  tier0_health.jsonl        HealthRecord per keyframe
  tier1_odometry.jsonl      OdometryDelta per keyframe
  tier1_measurements.jsonl  TrackletMeasurement events
  tier1_tables.json         CompatibilityTable list
  truth.jsonl               TruthPose per keyframe (diagnostics)
  events.jsonl              ProposalEvent index (§7.3 auto-bookmarks)
  mode_events.jsonl         ModeEvent index (birth/death/merge)
  checkpoints/index.json    sorted checkpoint keyframe indices
  checkpoints/kf_00042.npz  particle arrays

Tier 1 plus the manifest's config re-runs the filter bit-exactly *in the
same environment* (the §7.1 replay contract). Bit-exactness is not promised
across numpy/BLAS versions; the manifest records the history hash so a
divergence is at least detectable, and records git commit / argv / created
so the environment is at least identifiable.

`write_run` validates the manifest's provenance before writing anything —
a run directory that cannot name its inputs is worse than no run directory,
because every downstream consumer treats what is written here as true.

The JSONL helpers (`read_jsonl` / `write_jsonl`) are the shared run-record
serialization boundary.
"""

import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Any

import msgspec
import numpy as np

from common.python.serialization import msgspec_dec_hook, msgspec_enc_hook
from experimental.overhead_matching.swag.farfield import (
    artifact,
    paths,
    publication,
)
from experimental.overhead_matching.swag.farfield.localization import structs


RUN_KIND = "localization_run"
RUN_MANIFEST_NAME = "run_manifest.json"
RUN_CONTRACT_CONFIG_KEY = "localization_run_contract"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_ARRAYS = frozenset({
    "east_m", "north_m", "heading_rad", "log_weight",
    "proposal_event_id", "proposal_hypothesis", "mode_id",
})


@dataclasses.dataclass
class RunData:
    manifest: structs.RunManifest
    truth: list
    odometry: list
    measurements: list
    tables: dict
    health: list
    checkpoints: dict  # keyframe_idx -> dict[str, np.ndarray]
    proposal_events: list = dataclasses.field(default_factory=list)
    mode_events: list = dataclasses.field(default_factory=list)
    artifact_ref: artifact.ArtifactRef | None = None


def write_jsonl(path: Path, records) -> None:
    payload = b"".join(
        msgspec.json.encode(record, enc_hook=msgspec_enc_hook) + b"\n"
        for record in records)
    artifact.atomic_write_file(path, payload)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value!r}")
    return parsed


def _strict_json_document(payload: bytes, where: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_finite_json_float,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value!r}")),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON in {where}: {error}") from error


def _reject_unknown_shape(source: Any, normalized: Any, where: str) -> None:
    """Reject fields a typed msgspec decoder would otherwise ignore."""
    if isinstance(source, dict):
        if not isinstance(normalized, dict):
            raise ValueError(f"{where} has the wrong JSON shape")
        unknown = sorted(set(source) - set(normalized))
        missing = sorted(set(normalized) - set(source))
        if unknown or missing:
            details = []
            if unknown:
                details.append(f"unknown fields {unknown}")
            if missing:
                details.append(f"missing fields {missing}")
            raise ValueError(f"{where} has " + ", ".join(details))
        for key, value in source.items():
            _reject_unknown_shape(value, normalized[key], f"{where}.{key}")
        return
    if isinstance(source, list):
        if not isinstance(normalized, list) or len(source) != len(normalized):
            raise ValueError(f"{where} has the wrong JSON list shape")
        for index, (value, decoded) in enumerate(zip(source, normalized)):
            _reject_unknown_shape(value, decoded, f"{where}[{index}]")
        return
    # msgspec legitimately accepts an integer spelling for a float field.
    if (isinstance(source, (int, float)) and not isinstance(source, bool)
            and isinstance(normalized, (int, float))
            and not isinstance(normalized, bool)):
        if source != normalized:
            raise ValueError(f"{where} changed value while decoding")
        return
    if type(source) is not type(normalized) or source != normalized:
        raise ValueError(f"{where} changed type or value while decoding")


def _decode_typed_json(payload: bytes, record_type, where: str):
    document = _strict_json_document(payload, where)
    try:
        value = msgspec.json.decode(
            payload, type=record_type, dec_hook=msgspec_dec_hook)
    except (msgspec.DecodeError, msgspec.ValidationError) as error:
        error.add_note(f"while decoding typed run JSON in {where}")
        raise
    normalized = _strict_json_document(
        msgspec.json.encode(value, enc_hook=msgspec_enc_hook), where)
    _reject_unknown_shape(document, normalized, where)
    return value


_RETIRED_NOOP_FILTER_FIELDS = {
    "measurement_damage_cap_nats": None,
}
_RETIRED_NOOP_PROPOSAL_FIELDS = {
    "revival_enabled": False,
    "revival_match_radius_m": None,
    "revival_margin_nats": 0.0,
}
def _without_retired_noop_filter_fields(document: dict, where: str) -> dict:
    """Normalize recorded runs across filter-config schema changes.

    Retired fields: several 2026-08-27 runs were stamped with damage-cap/
    revival settings while all four were disabled. The experiment was removed
    before commit, leaving valid no-op runs that the strict reader could no
    longer inspect. Accept only the exact inactive spellings; a non-default
    value shaped a run and is still unknown science, so it remains a hard
    error.

    Fields added later are not back-filled: a run recorded before a setting
    existed fails the strict shape check and has to be re-run.
    """
    normalized = dict(document)
    filter_config = document.get("filter_config")
    if not isinstance(filter_config, dict):
        return normalized
    filter_config = dict(filter_config)
    for name, expected in _RETIRED_NOOP_FILTER_FIELDS.items():
        if name not in filter_config:
            continue
        actual = filter_config[name]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"{where}.filter_config.{name} used retired non-noop value "
                f"{actual!r}")
        filter_config.pop(name)
    proposal = filter_config.get("proposal")
    if isinstance(proposal, dict):
        proposal = dict(proposal)
        for name, expected in _RETIRED_NOOP_PROPOSAL_FIELDS.items():
            if name not in proposal:
                continue
            actual = proposal[name]
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"{where}.filter_config.proposal.{name} used retired "
                    f"non-noop value {actual!r}")
            proposal.pop(name)
        filter_config["proposal"] = proposal
    normalized["filter_config"] = filter_config
    return normalized


def _decode_run_manifest(payload: bytes, where: str) -> structs.RunManifest:
    document = _strict_json_document(payload, where)
    if not isinstance(document, dict):
        raise ValueError(f"{where} must be a JSON object")
    normalized = _without_retired_noop_filter_fields(document, where)
    return _decode_typed_json(
        json.dumps(normalized, separators=(",", ":")).encode("utf-8"),
        structs.RunManifest, where)


def _read_regular_file(path: Path) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"required run input is missing: {path}")
    return path.read_bytes()


def read_jsonl(path: Path, record_type) -> list:
    payload = _read_regular_file(path)
    records = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL record in {path}:{line_number}")
        records.append(_decode_typed_json(
            line, record_type, f"{path}:{line_number}"))
    return records


def _finite_tree(value: Any, where: str, problems: list[str]) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            problems.append(f"{where} is non-finite")
        return
    if isinstance(value, msgspec.Struct):
        for field in value.__struct_fields__:
            _finite_tree(getattr(value, field), f"{where}.{field}", problems)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{where}[{index}]", problems)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{where}.{key}", problems)


def _manifest_contract(manifest: structs.RunManifest) -> dict[str, Any]:
    document = _strict_json_document(
        msgspec.json.encode(manifest, enc_hook=msgspec_enc_hook),
        "encoded RunManifest")
    return {
        key: document[key]
        for key in (
            "schema_version", "dataset", "run_kind",
            "initialization_kind", "bearings_consumed", "proposal_enabled",
            "localization_inputs_manifest_sha256", "n_keyframes",
            "filter_config", "matcher_version", "max_visible_range_m",
            "ablation_tags", "truth_position_artifact",
            "truth_position_schema", "position_mass_metric",
        )
    }


def validate_manifest(manifest: structs.RunManifest) -> None:
    """Refuse to write a run that cannot name its own inputs."""
    problems = []
    if manifest.schema_version != structs.SCHEMA_VERSION:
        problems.append(
            f"schema_version must be {structs.SCHEMA_VERSION!r}")
    try:
        artifact.require_identifier(manifest.dataset, "run dataset")
    except artifact.ArtifactValidationError as error:
        problems.append(str(error))
    if not manifest.scenario_name:
        problems.append("scenario_name is empty")
    if not manifest.export_dir:
        problems.append(
            "export_dir is empty — record the export the run consumed, or "
            "'synthetic:<scenario>' for a generated run")
    if manifest.max_visible_range_m is None or \
            manifest.max_visible_range_m <= 0.0:
        problems.append("max_visible_range_m must be the positive radius "
                        "the catalog was built with")
    if not manifest.git_commit:
        problems.append("git_commit is empty (use provenance.git_commit())")
    if not manifest.created:
        problems.append("created is empty")
    if not manifest.dataset:
        problems.append("dataset is empty")
    if manifest.run_kind not in (
            "evaluation", "diagnostic_control", "synthetic"):
        problems.append("run_kind must be evaluation, diagnostic_control, "
                        "or synthetic")
    if not manifest.initialization_kind:
        problems.append("initialization_kind is empty")
    tags = manifest.ablation_tags
    if (tags != sorted(set(tags))
            or any(not isinstance(tag, str) or not tag for tag in tags)):
        problems.append("ablation_tags must be sorted unique non-empty strings")
    if not manifest.bearings_consumed and "no_bearings" not in tags:
        problems.append(
            "a bearings-withheld control must carry ablation tag no_bearings")
    if manifest.bearings_consumed and "no_bearings" in tags:
        problems.append("no_bearings tag disagrees with bearings_consumed")
    if manifest.initialization_kind == "truth":
        problems.append(
            "initialization_kind 'truth' is ambiguous; use 'truth_position'")
    if (manifest.initialization_kind == "truth_position"
            and "truth_position_initialization" not in tags):
        problems.append(
            "truth_position initialization must carry its explicit ablation tag")
    if ("truth_position_initialization" in tags
            and manifest.initialization_kind != "truth_position"):
        problems.append(
            "truth_position_initialization tag disagrees with initialization")
    if manifest.run_kind == "evaluation" and tags:
        problems.append("evaluation runs cannot carry ablation_tags")
    if manifest.run_kind == "evaluation" and (
            manifest.initialization_kind != "uniform"
            or not manifest.bearings_consumed):
        problems.append(
            "evaluation classification requires uniform init and bearings")
    truth_schema = manifest.truth_position_schema
    if truth_schema is not None and (
            not isinstance(truth_schema, str) or not truth_schema):
        problems.append("truth_position_schema must be null or non-empty")
    truth_source = manifest.truth_position_artifact
    if truth_source is not None and (
            not isinstance(truth_source, dict)
            or not truth_source
            or any(not isinstance(key, str) or not key
                   or not isinstance(value, str) or not value
                   for key, value in truth_source.items())):
        problems.append(
            "truth_position_artifact must contain non-empty string fields")
    metric_config = manifest.position_mass_metric
    if metric_config is not None:
        if not metric_config.metric_id or not metric_config.metric_version:
            problems.append("position_mass_metric identity/version is empty")
        radii = metric_config.radii_m
        if (not radii or radii != sorted(set(radii))
                or any(not math.isfinite(radius) or radius <= 0.0
                       for radius in radii)):
            problems.append(
                "position_mass_metric radii must be finite, positive, sorted, "
                "and unique")
        if not truth_schema:
            problems.append(
                "position_mass_metric requires truth_position_schema")
    if (isinstance(manifest.n_keyframes, bool)
            or not isinstance(manifest.n_keyframes, int)
            or manifest.n_keyframes < 2):
        problems.append("n_keyframes must be an integer >= 2")
    if (not math.isfinite(manifest.anchor_lat_deg)
            or not -90.0 <= manifest.anchor_lat_deg <= 90.0
            or not math.isfinite(manifest.anchor_lon_deg)
            or not -180.0 <= manifest.anchor_lon_deg <= 180.0):
        problems.append("anchor latitude/longitude is invalid")
    if not manifest.matcher_version:
        problems.append("matcher_version is empty")
    if manifest.proposal_enabled != manifest.filter_config.proposal.enabled:
        problems.append("proposal_enabled disagrees with filter_config")
    digest = manifest.localization_inputs_manifest_sha256
    if manifest.run_kind == "synthetic":
        if digest is not None:
            problems.append("synthetic run must not claim a localization "
                            "input manifest digest")
    elif not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        problems.append("real run must record the localization_inputs "
                        "manifest SHA-256")
    if not _SHA256_RE.fullmatch(manifest.particle_history_sha256):
        problems.append("particle_history_sha256 must be a SHA-256 digest")
    if not manifest.argv or not all(
            isinstance(value, str) and value for value in manifest.argv):
        problems.append("argv must contain non-empty strings")
    if manifest.filter_config.n_particles <= 0:
        problems.append("filter_config.n_particles must be positive")
    if manifest.filter_config.checkpoint_every <= 0:
        problems.append("filter_config.checkpoint_every must be positive")
    if manifest.run_kind == "evaluation":
        if not isinstance(manifest.filter_config.init, structs.UniformBoxInit):
            problems.append("evaluation run must use a uniform-box prior")
        if not manifest.bearings_consumed:
            problems.append("evaluation run must consume bearings")
    landmark_ids = [landmark.landmark_id for landmark in manifest.landmarks]
    if not landmark_ids:
        problems.append("landmark catalog is empty")
    elif not all(isinstance(value, str) and value for value in landmark_ids):
        problems.append("landmark ids must be non-empty strings")
    elif len(landmark_ids) != len(set(landmark_ids)):
        problems.append("landmark ids are not unique")
    sigmas = [landmark.position_sigma_m for landmark in manifest.landmarks]
    if sigmas and ((manifest.run_kind != "synthetic" and sigmas[0] <= 0.0)
                   or sigmas[0] < 0.0
                   or any(value != sigmas[0] for value in sigmas)):
        problems.append(
            "landmark position uncertainty must be one uniform value "
            "(positive for real runs)")
    if any(not -90.0 <= landmark.lat_deg <= 90.0
           or not -180.0 <= landmark.lon_deg <= 180.0
           for landmark in manifest.landmarks):
        problems.append("landmark latitude/longitude is out of bounds")
    for landmark in manifest.landmarks:
        east = landmark.hull_east_m
        north = landmark.hull_north_m
        if len(east) != len(north):
            problems.append(
                f"landmark {landmark.landmark_id!r} hull arrays differ in length")
        elif len(east) == 1:
            problems.append(
                f"landmark {landmark.landmark_id!r} hull has only one vertex")
    _finite_tree(manifest, "run_manifest", problems)
    if problems:
        raise ValueError("run manifest fails provenance validation:\n  - "
                         + "\n  - ".join(problems))


def _expected_checkpoint_keyframes(manifest: structs.RunManifest) -> list[int]:
    every = manifest.filter_config.checkpoint_every
    return sorted(set(range(0, manifest.n_keyframes, every))
                  | {manifest.n_keyframes - 1})


def _checkpoint_arrays(value: Any) -> dict[str, np.ndarray]:
    if isinstance(value, dict):
        return value
    return {name: getattr(value, name) for name in _CHECKPOINT_ARRAYS}


def _validate_checkpoint(kf: int, value: Any,
                         manifest: structs.RunManifest,
                         proposal_events: dict[int, structs.ProposalEvent],
                         problems: list[str]) -> None:
    try:
        arrays = _checkpoint_arrays(value)
    except AttributeError as error:
        problems.append(f"checkpoint {kf} lacks {error.name}")
        return
    if set(arrays) != _CHECKPOINT_ARRAYS:
        problems.append(
            f"checkpoint {kf} arrays are not exactly "
            f"{sorted(_CHECKPOINT_ARRAYS)}")
        return
    n_particles = manifest.filter_config.n_particles
    for name, raw in arrays.items():
        array = np.asarray(raw)
        if array.ndim != 1 or array.shape != (n_particles,):
            problems.append(
                f"checkpoint {kf}.{name} must have shape ({n_particles},)")
            continue
        if name in ("proposal_event_id", "proposal_hypothesis", "mode_id"):
            if array.dtype.kind not in "iu":
                problems.append(f"checkpoint {kf}.{name} must be integer")
        elif array.dtype.kind != "f" or not np.isfinite(array).all():
            problems.append(
                f"checkpoint {kf}.{name} must be finite floating point")
    event_ids = np.asarray(arrays["proposal_event_id"])
    hypothesis_ids = np.asarray(arrays["proposal_hypothesis"])
    if (event_ids.shape == hypothesis_ids.shape == (n_particles,)
            and event_ids.dtype.kind in "iu"
            and hypothesis_ids.dtype.kind in "iu"):
        if np.any(event_ids < -1) or np.any(hypothesis_ids < -1):
            problems.append(
                f"checkpoint {kf} has invalid negative proposal provenance")
        if np.any((event_ids == -1) != (hypothesis_ids == -1)):
            problems.append(
                f"checkpoint {kf} has unpaired proposal provenance")
        for event_id in np.unique(event_ids[event_ids >= 0]):
            event = proposal_events.get(int(event_id))
            if event is None:
                problems.append(
                    f"checkpoint {kf} references unknown proposal event "
                    f"{int(event_id)}")
                continue
            selected = hypothesis_ids[event_ids == event_id]
            if np.any(selected >= event.n_hypotheses):
                problems.append(
                    f"checkpoint {kf} references an out-of-range hypothesis "
                    f"for proposal event {int(event_id)}")


def _validate_payloads(manifest: structs.RunManifest, truth: list,
                       odometry: list, measurements: list, tables: dict,
                       health: list, checkpoints: dict,
                       proposal_events: list, mode_events: list) -> None:
    problems = []
    _finite_tree(truth, "truth", problems)
    _finite_tree(odometry, "odometry", problems)
    _finite_tree(measurements, "measurements", problems)
    _finite_tree(tables, "tables", problems)
    _finite_tree(health, "health", problems)
    _finite_tree(proposal_events, "proposal_events", problems)
    _finite_tree(mode_events, "mode_events", problems)

    expected_health = list(range(manifest.n_keyframes))
    health_indices = [record.keyframe_idx for record in health]
    if health_indices != expected_health:
        problems.append("health keyframes must be contiguous 0..N-1")
    metric_config = manifest.position_mass_metric
    if metric_config is None:
        if any(record.position_probability_mass for record in health):
            problems.append(
                "health records contain position mass without metric config")
    else:
        expected_metric_keys = {
            f"{metric_config.metric_id}@{metric_config.metric_version}:"
            f"radius_m={float(radius):g}"
            for radius in metric_config.radii_m
        }
        if not truth:
            problems.append(
                "position-mass metric config requires truth positions")
        for record in health:
            values = record.position_probability_mass
            if set(values) != expected_metric_keys:
                problems.append(
                    f"health keyframe {record.keyframe_idx} does not contain "
                    "every configured position-mass metric exactly once")
            elif any(not math.isfinite(value) or not 0.0 <= value <= 1.0
                     for value in values.values()):
                problems.append(
                    f"health keyframe {record.keyframe_idx} has invalid "
                    "position probability mass")
    expected_odometry = list(range(1, manifest.n_keyframes))
    odometry_indices = [record.keyframe_idx for record in odometry]
    if odometry_indices != expected_odometry:
        problems.append("odometry keyframes must be contiguous 1..N-1")
    for record in odometry:
        if record.sigma_m <= 0.0 or record.sigma_yaw_rad <= 0.0:
            problems.append(
                f"odometry at keyframe {record.keyframe_idx} has "
                "non-positive uncertainty")
    truth_indices = [record.keyframe_idx for record in truth]
    if truth_indices and truth_indices != expected_health:
        problems.append("truth must be empty or contiguous 0..N-1")
    for record in truth:
        if not 0.0 <= record.course_world_cw_deg < 360.0:
            problems.append(
                f"truth course at keyframe {record.keyframe_idx} is not "
                "world-CW [0, 360)")

    measurement_order = [
        (record.anchor_keyframe_idx, record.tracklet_id)
        for record in measurements]
    if measurement_order != sorted(measurement_order):
        problems.append(
            "measurements must be sorted by (anchor_keyframe_idx, tracklet_id)")
    measurement_keys = [
        (record.tracklet_id, record.anchor_keyframe_idx)
        for record in measurements]
    if len(measurement_keys) != len(set(measurement_keys)):
        problems.append("measurement information epochs are not unique")
    counts = [0] * manifest.n_keyframes
    for record in measurements:
        if not 0 <= record.anchor_keyframe_idx < manifest.n_keyframes:
            problems.append(
                f"measurement {record.tracklet_id!r} is outside the run")
        else:
            counts[record.anchor_keyframe_idx] += 1
        if (not 0.0 <= record.bearing_forward_cw_deg < 360.0
                or record.kappa <= 0.0):
            problems.append(
                f"measurement {record.tracklet_id!r} has invalid bearing "
                "or concentration")
    if health_indices == expected_health:
        recorded_counts = [record.n_measurements for record in health]
        if recorded_counts != counts:
            problems.append(
                "health n_measurements disagrees with measurement epochs")
    if not manifest.bearings_consumed and (measurements or tables):
        problems.append(
            "bearings-withheld run must have empty measurements and tables")
    if manifest.run_kind == "evaluation" and not measurements:
        problems.append("evaluation run must contain a bearing measurement")

    table_map = tables if isinstance(tables, dict) else {}
    if not isinstance(tables, dict) or any(
            key != table.tracklet_id for key, table in table_map.items()):
        problems.append("table mapping keys must equal table tracklet ids")
    measurement_tracklets = {record.tracklet_id for record in measurements}
    if set(table_map) != measurement_tracklets:
        problems.append(
            "compatibility tables must exactly cover measured tracklets")
    known_landmarks = {item.landmark_id for item in manifest.landmarks}
    for tracklet_id, table in table_map.items():
        if table.matcher_version != manifest.matcher_version:
            problems.append(
                f"table {tracklet_id!r} matcher version disagrees with run")
        if (table.status not in ("fast", "refined")
                or table.clip_lo >= table.clip_hi):
            problems.append(f"table {tracklet_id!r} has invalid policy")
        entry_ids = [entry.landmark_id for entry in table.entries]
        if len(entry_ids) != len(set(entry_ids)):
            problems.append(f"table {tracklet_id!r} repeats a landmark")
        unknown = set(entry_ids) - known_landmarks
        if unknown:
            problems.append(
                f"table {tracklet_id!r} names unknown landmarks "
                f"{sorted(unknown)}")

    event_ids = [event.event_id for event in proposal_events]
    if event_ids != list(range(len(proposal_events))):
        problems.append("proposal event ids must be contiguous from zero")
    event_order = [(event.keyframe_idx, event.event_id)
                   for event in proposal_events]
    if event_order != sorted(event_order) or any(
            not 0 <= event.keyframe_idx < manifest.n_keyframes
            for event in proposal_events):
        problems.append("proposal events are out of keyframe order or range")
    event_by_id = {event.event_id: event for event in proposal_events}
    if health_indices == expected_health:
        health_events = [
            (record.keyframe_idx, record.proposal_event_id)
            for record in health if record.proposal_event_id is not None]
        expected_events = [
            (event.keyframe_idx, event.event_id) for event in proposal_events]
        if health_events != expected_events:
            problems.append(
                "health proposal references disagree with proposal events")
    mode_order = [record.keyframe_idx for record in mode_events]
    if mode_order != sorted(mode_order) or any(
            not 0 <= keyframe < manifest.n_keyframes
            for keyframe in mode_order):
        problems.append("mode events are out of keyframe order or range")

    expected_checkpoints = _expected_checkpoint_keyframes(manifest)
    actual_checkpoints = sorted(checkpoints)
    if actual_checkpoints != expected_checkpoints:
        problems.append(
            f"checkpoint keyframes must be exactly {expected_checkpoints}")
    for kf, value in checkpoints.items():
        _validate_checkpoint(kf, value, manifest, event_by_id, problems)
    if problems:
        raise ValueError("localization run payload fails validation:\n  - "
                         + "\n  - ".join(problems))


def write_run(run_dir: Path, manifest: structs.RunManifest, truth: list,
              odometry: list, measurements: list, tables: dict,
              history, *, dataset: str, version: str,
              upstreams: tuple[artifact.ArtifactRef, ...] = (),
              artifact_config: dict | None = None,
              generator: str = "farfield.localization.run_io",
              arguments: tuple[str, ...] | None = None,
              extra_outputs: dict[str, bytes] | None = None
              ) -> artifact.ArtifactRef:
    """`history` is a filter.FilterHistory (duck-typed to avoid the dep).

    `measurements`/`tables` must be the ones the filter actually consumed:
    an odometry-only control run passes its empty lists, never the full
    inputs it chose to ignore (writing the unconsumed ones once produced run
    directories describing runs that never happened).
    """
    validate_manifest(manifest)
    if dataset != manifest.dataset:
        raise ValueError("run artifact dataset disagrees with RunManifest")
    localization_inputs = [
        ref for ref in upstreams if ref.kind == paths.LOCALIZATION_INPUTS]
    if manifest.run_kind != "synthetic":
        if len(localization_inputs) != 1:
            raise ValueError(
                "real run requires exactly one localization_inputs upstream")
        if (localization_inputs[0].manifest_digest
                != manifest.localization_inputs_manifest_sha256):
            raise ValueError(
                "RunManifest input digest disagrees with artifact upstream")
    run_upstreams = [ref for ref in upstreams if ref.kind == RUN_KIND]
    allowed_kinds = ({RUN_KIND} if manifest.run_kind == "synthetic"
                     else {paths.LOCALIZATION_INPUTS, RUN_KIND})
    unknown_kinds = sorted({ref.kind for ref in upstreams} - allowed_kinds)
    if unknown_kinds:
        raise ValueError(
            f"run artifact has unexpected upstream kinds {unknown_kinds}")
    run_dir = Path(run_dir)
    extra_outputs = dict(extra_outputs or {})
    if not all(isinstance(name, str) and isinstance(payload, bytes)
               for name, payload in extra_outputs.items()):
        raise TypeError("extra_outputs must map relative names to bytes")
    keyframes = sorted(history.checkpoints.keys())
    _validate_payloads(
        manifest, truth, odometry, measurements, tables, history.health,
        history.checkpoints, history.proposal_events, history.mode_events)
    artifact_config = dict(artifact_config or {})
    managed_config = {
        "run_kind": manifest.run_kind,
        "localization_inputs_manifest_sha256": (
            manifest.localization_inputs_manifest_sha256),
        "ablation_tags": list(manifest.ablation_tags),
        "truth_position_artifact": manifest.truth_position_artifact,
        "truth_position_schema": manifest.truth_position_schema,
        "position_mass_metric": (
            None if manifest.position_mass_metric is None
            else msgspec.to_builtins(manifest.position_mass_metric)),
    }
    for key, value in managed_config.items():
        if key in artifact_config and artifact_config[key] != value:
            raise ValueError(
                f"artifact config {key!r} disagrees with RunManifest")
        artifact_config[key] = value
    contract = _manifest_contract(manifest)
    if (RUN_CONTRACT_CONFIG_KEY in artifact_config
            and artifact_config[RUN_CONTRACT_CONFIG_KEY] != contract):
        raise ValueError(
            f"artifact config {RUN_CONTRACT_CONFIG_KEY!r} is producer-owned")
    artifact_config[RUN_CONTRACT_CONFIG_KEY] = contract
    if run_upstreams:
        source = artifact_config.get("source_run")
        if (len(run_upstreams) != 1 or not isinstance(source, dict)
                or source != run_upstreams[0].to_dict()):
            raise ValueError(
                "a run upstream must be recorded exactly as config.source_run")
    elif "source_run" in artifact_config:
        raise ValueError("config.source_run has no matching run upstream")
    outputs = (
        RUN_MANIFEST_NAME,
        "tier0_health.jsonl",
        "tier1_odometry.jsonl",
        "tier1_measurements.jsonl",
        "tier1_tables.json",
        "truth.jsonl",
        "events.jsonl",
        "mode_events.jsonl",
        "checkpoints/index.json",
        *(f"checkpoints/kf_{kf:05d}.npz" for kf in keyframes),
        *extra_outputs,
    )
    with publication.published_artifact(
            run_dir, kind=RUN_KIND, dataset=dataset, version=version,
            generator=generator, git_commit=manifest.git_commit,
            arguments=arguments, upstreams=upstreams,
            config=artifact_config, declared_outputs=outputs) as builder:
        artifact.atomic_write_file(
            builder.output_path(RUN_MANIFEST_NAME),
            msgspec.json.encode(manifest, enc_hook=msgspec_enc_hook))
        write_jsonl(builder.output_path("tier0_health.jsonl"), history.health)
        write_jsonl(builder.output_path("tier1_odometry.jsonl"), odometry)
        write_jsonl(
            builder.output_path("tier1_measurements.jsonl"), measurements)
        artifact.atomic_write_file(
            builder.output_path("tier1_tables.json"),
            msgspec.json.encode(
                sorted(tables.values(), key=lambda table: table.tracklet_id),
                enc_hook=msgspec_enc_hook))
        write_jsonl(builder.output_path("truth.jsonl"), truth)
        write_jsonl(
            builder.output_path("events.jsonl"), history.proposal_events)
        write_jsonl(
            builder.output_path("mode_events.jsonl"), history.mode_events)
        artifact.atomic_write_file(
            builder.output_path("checkpoints/index.json"),
            msgspec.json.encode(keyframes))
        for kf in keyframes:
            belief = history.checkpoints[kf]
            checkpoint_path = builder.output_path(
                f"checkpoints/kf_{kf:05d}.npz")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(checkpoint_path,
                     east_m=belief.east_m, north_m=belief.north_m,
                     heading_rad=belief.heading_rad,
                     log_weight=belief.log_weight,
                     proposal_event_id=belief.proposal_event_id,
                     proposal_hypothesis=belief.proposal_hypothesis,
                     mode_id=belief.mode_id)
        for name, payload in extra_outputs.items():
            artifact.atomic_write_file(builder.output_path(name), payload)
    if builder.artifact_ref is None:
        raise RuntimeError("run artifact publication returned no identity")
    return builder.artifact_ref


def read_run(run_dir: Path) -> RunData:
    run_dir = Path(run_dir)
    reference = artifact.open_artifact(run_dir, expected_kind=RUN_KIND)
    artifact_manifest = artifact.load_manifest(run_dir)
    manifest = _decode_run_manifest(
        _read_regular_file(run_dir / RUN_MANIFEST_NAME),
        str(run_dir / RUN_MANIFEST_NAME))
    validate_manifest(manifest)
    if reference.dataset != manifest.dataset:
        raise ValueError(
            "artifact manifest dataset disagrees with RunManifest")
    if artifact_manifest.git_commit != manifest.git_commit:
        raise ValueError(
            "artifact manifest git commit disagrees with RunManifest")
    expected_contract = _manifest_contract(manifest)
    recorded_contract = artifact_manifest.config.get(RUN_CONTRACT_CONFIG_KEY)
    if isinstance(recorded_contract, dict):
        recorded_contract = _without_retired_noop_filter_fields(
            recorded_contract,
            f"{run_dir / artifact.MANIFEST_NAME} config "
            f"{RUN_CONTRACT_CONFIG_KEY}")
    if recorded_contract != expected_contract:
        raise ValueError(
            "artifact manifest does not contain the exact run contract")
    for key, expected in (
            ("run_kind", manifest.run_kind),
            ("localization_inputs_manifest_sha256",
             manifest.localization_inputs_manifest_sha256),
            ("ablation_tags", list(manifest.ablation_tags)),
            ("truth_position_artifact", manifest.truth_position_artifact),
            ("truth_position_schema", manifest.truth_position_schema),
            ("position_mass_metric", (
                None if manifest.position_mass_metric is None
                else msgspec.to_builtins(manifest.position_mass_metric)))):
        if artifact_manifest.config.get(key) != expected:
            raise ValueError(
                f"artifact config {key!r} disagrees with RunManifest")

    localization_inputs = [
        ref for ref in artifact_manifest.upstreams
        if ref.kind == paths.LOCALIZATION_INPUTS]
    run_upstreams = [
        ref for ref in artifact_manifest.upstreams if ref.kind == RUN_KIND]
    allowed_kinds = ({RUN_KIND} if manifest.run_kind == "synthetic"
                     else {paths.LOCALIZATION_INPUTS, RUN_KIND})
    actual_kinds = {ref.kind for ref in artifact_manifest.upstreams}
    if not actual_kinds <= allowed_kinds:
        raise ValueError("run artifact contains unexpected upstream kinds")
    if manifest.run_kind == "synthetic":
        if localization_inputs:
            raise ValueError(
                "synthetic run cannot claim localization-input upstreams")
    elif (len(localization_inputs) != 1
          or localization_inputs[0].manifest_digest
          != manifest.localization_inputs_manifest_sha256):
        raise ValueError(
            "real run does not name its exact localization-input upstream")
    source = artifact_manifest.config.get("source_run")
    if run_upstreams:
        if (len(run_upstreams) != 1 or not isinstance(source, dict)
                or source != run_upstreams[0].to_dict()):
            raise ValueError(
                "run upstream disagrees with artifact config.source_run")
    elif "source_run" in artifact_manifest.config:
        raise ValueError("config.source_run has no matching run upstream")

    tables_list = _decode_typed_json(
        _read_regular_file(run_dir / "tier1_tables.json"),
        list[structs.CompatibilityTable],
        str(run_dir / "tier1_tables.json"))
    table_ids = [table.tracklet_id for table in tables_list]
    if table_ids != sorted(set(table_ids)):
        raise ValueError(
            "tier1_tables.json must have unique sorted tracklet ids")

    checkpoint_dir = run_dir / "checkpoints"
    keyframes = _decode_typed_json(
        _read_regular_file(checkpoint_dir / "index.json"), list[int],
        str(checkpoint_dir / "index.json"))
    expected_keyframes = _expected_checkpoint_keyframes(manifest)
    if keyframes != expected_keyframes:
        raise ValueError(
            f"checkpoint index must be exactly {expected_keyframes}")
    expected_core_outputs = {
        RUN_MANIFEST_NAME,
        "tier0_health.jsonl",
        "tier1_odometry.jsonl",
        "tier1_measurements.jsonl",
        "tier1_tables.json",
        "truth.jsonl",
        "events.jsonl",
        "mode_events.jsonl",
        "checkpoints/index.json",
        *(f"checkpoints/kf_{kf:05d}.npz" for kf in keyframes),
    }
    declared = set(artifact_manifest.declared_outputs)
    missing = expected_core_outputs - declared
    unexpected_checkpoints = {
        name for name in declared
        if name.startswith("checkpoints/") and name not in expected_core_outputs
    }
    if missing or unexpected_checkpoints:
        raise ValueError(
            "run artifact declared outputs disagree with its run contract")
    checkpoints = {}
    for kf in keyframes:
        checkpoint_path = checkpoint_dir / f"kf_{kf:05d}.npz"
        if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"required run input is missing: {checkpoint_path}")
        with np.load(checkpoint_path, allow_pickle=False) as npz:
            checkpoints[kf] = {key: npz[key] for key in npz.files}

    data = RunData(
        manifest=manifest,
        truth=read_jsonl(run_dir / "truth.jsonl", structs.TruthPose),
        odometry=read_jsonl(run_dir / "tier1_odometry.jsonl",
                            structs.OdometryDelta),
        measurements=read_jsonl(run_dir / "tier1_measurements.jsonl",
                                structs.TrackletMeasurement),
        tables={t.tracklet_id: t for t in tables_list},
        health=read_jsonl(run_dir / "tier0_health.jsonl",
                          structs.HealthRecord),
        checkpoints=checkpoints,
        proposal_events=read_jsonl(run_dir / "events.jsonl",
                                   structs.ProposalEvent),
        mode_events=read_jsonl(run_dir / "mode_events.jsonl",
                               structs.ModeEvent),
        artifact_ref=reference)
    _validate_payloads(
        data.manifest, data.truth, data.odometry, data.measurements,
        data.tables, data.health, data.checkpoints, data.proposal_events,
        data.mode_events)
    return data
