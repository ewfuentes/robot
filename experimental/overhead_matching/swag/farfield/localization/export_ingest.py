"""Validate and load a completed ``localization_inputs`` artifact.

The manifest is the authority.  This reader verifies every declared byte,
the exact upstream artifact kinds, the copied nominal-forward calibration and
motion-source digests, complete matching attestation, uniform map uncertainty,
clockwise angle fields, and all serialized index/coverage relationships before
constructing filter inputs.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import msgspec
import numpy as np

from common.python.serialization import msgspec_dec_hook
from experimental.overhead_matching.swag.farfield import (
    artifact,
    geometry as geo,
    nominal_forward,
    paths as paths_lib,
)
from experimental.overhead_matching.swag.farfield.localization import (
    filter_catalog,
    run_io,
    structs,
)
from experimental.overhead_matching.swag.farfield.matching import identity_review


EXPORT_SCHEMA = "farfield_localization_inputs/v1"
_NOMINAL_KEYS = frozenset({
    "file",
    "source_path",
    "content_sha256",
    "schema",
    "frame",
    "dataset",
    "version",
    "mounting_id",
    "panorama_column",
    "panorama_width",
    "bearing_camera_cw_deg",
    "uncertainty_deg",
    "evidence_frame_ids",
    "operator",
    "approved_at",
    "notes",
})
_MOTION_KEYS = frozenset({
    "file",
    "source_path",
    "content_sha256",
    "course_heading_status",
    "reverse_annotation_source",
})
_REDUCER_KEYS = frozenset({
    "name",
    "epoch_keyframes",
    "input_frame",
    "output_frame",
})


class ExportMeta(msgspec.Struct, forbid_unknown_fields=True):
    schema_version: str
    message_schema_version: str
    dataset: str
    scenario_name: str
    anchor_lat_deg: float
    anchor_lon_deg: float
    n_keyframes: int
    matcher_version: str
    matching_coverage: str
    max_visible_range_m: float
    landmark_position_sigma_m: float
    nominal_forward: dict
    motion: dict
    reducer: dict


@dataclasses.dataclass
class ExportData:
    artifact_ref: artifact.ArtifactRef
    manifest: artifact.ArtifactManifest
    meta: ExportMeta
    frame: geo.RegionFrame
    catalog: filter_catalog.LandmarkCatalog
    landmarks: list
    odometry: list
    measurements: list
    tables: dict
    truth: list

    @property
    def n_keyframes(self) -> int:
        return self.meta.n_keyframes


def _finite(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _exact_dict(value, expected: frozenset[str], name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}")
    return value


def _exact_upstream(manifest: artifact.ArtifactManifest,
                    kind: str) -> artifact.ArtifactRef:
    refs = [item for item in manifest.upstreams if item.kind == kind]
    if len(refs) != 1:
        raise ValueError(
            f"localization_inputs must have exactly one {kind} upstream")
    return refs[0]


def _validate_nominal_forward(export_dir: Path, meta: ExportMeta,
                              manifest: artifact.ArtifactManifest):
    recorded = _exact_dict(
        meta.nominal_forward, _NOMINAL_KEYS, "nominal_forward")
    if recorded["file"] != "nominal_forward.json":
        raise ValueError(
            "nominal_forward.file must be 'nominal_forward.json'")
    copied = export_dir / recorded["file"]
    digest = artifact.sha256_file(copied)
    if digest != recorded["content_sha256"]:
        raise ValueError("copied nominal-forward digest does not match meta")
    if manifest.config.get("nominal_forward_sha256") != digest:
        raise ValueError(
            "localization manifest does not bind the nominal-forward bytes")
    calibration = nominal_forward.load(
        copied, expected_dataset=meta.dataset)
    expected = {
        "schema": nominal_forward.SCHEMA,
        "frame": nominal_forward.FRAME,
        "dataset": calibration.dataset,
        "version": calibration.version,
        "mounting_id": calibration.mounting_id,
        "panorama_column": calibration.panorama_column,
        "panorama_width": calibration.panorama_width,
        "bearing_camera_cw_deg": calibration.bearing_camera_cw_deg,
        "uncertainty_deg": calibration.uncertainty_deg,
        "evidence_frame_ids": list(calibration.evidence_frame_ids),
        "operator": calibration.operator,
        "approved_at": calibration.approved_at,
        "notes": calibration.notes,
    }
    for key, value in expected.items():
        if recorded.get(key) != value:
            raise ValueError(
                f"nominal_forward.{key} disagrees with the copied approved "
                "record")
    if not isinstance(recorded["source_path"], str) \
            or not recorded["source_path"]:
        raise ValueError("nominal_forward.source_path must be non-empty")
    return calibration


def _validate_motion(export_dir: Path, meta: ExportMeta,
                     manifest: artifact.ArtifactManifest) -> None:
    recorded = _exact_dict(meta.motion, _MOTION_KEYS, "motion")
    if recorded["file"] != "motion_source.csv":
        raise ValueError("motion.file must be 'motion_source.csv'")
    digest = artifact.sha256_file(export_dir / recorded["file"])
    if digest != recorded["content_sha256"]:
        raise ValueError("copied motion-source digest does not match meta")
    if manifest.config.get("motion_source_sha256") != digest:
        raise ValueError(
            "localization manifest does not bind the motion-source bytes")
    if (not isinstance(recorded["source_path"], str)
            or not recorded["source_path"]):
        raise ValueError("motion.source_path must be non-empty")
    if recorded["course_heading_status"] not in (
            "gps_course_diagnostic_only",
            "abstained_insufficient_displacement"):
        raise ValueError("motion.course_heading_status is invalid")
    if (not isinstance(recorded["reverse_annotation_source"], str)
            or not recorded["reverse_annotation_source"].strip()):
        raise ValueError(
            "motion.reverse_annotation_source must be non-empty")


def _validate_reducer(meta: ExportMeta,
                      manifest: artifact.ArtifactManifest) -> None:
    recorded = _exact_dict(meta.reducer, _REDUCER_KEYS, "reducer")
    expected_static = {
        "name": "epoch_fused_compat_v1",
        "input_frame": "camera_cw_deg",
        "output_frame": "nominal_forward_cw_deg",
    }
    for key, expected in expected_static.items():
        if recorded[key] != expected:
            raise ValueError(f"reducer.{key} must be {expected!r}")
    epoch = recorded["epoch_keyframes"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("reducer.epoch_keyframes must be a positive integer")
    if manifest.config.get("reducer") != recorded:
        raise ValueError("localization manifest does not bind reducer config")


def load(export_dir: Path, *, expected_dataset: str | None = None) \
        -> ExportData:
    """Validate and decode one completed localization-input artifact."""
    export_dir = Path(export_dir)
    reference = artifact.open_artifact(
        export_dir,
        expected_kind=paths_lib.LOCALIZATION_INPUTS,
        expected_dataset=expected_dataset)
    manifest = artifact.load_manifest(export_dir)
    required_kinds = {
        paths_lib.BEARING_OBSERVATIONS,
        paths_lib.LANDMARK_MATCHES,
        paths_lib.CATALOGS,
    }
    actual_kinds = [item.kind for item in manifest.upstreams]
    allowed_kinds = required_kinds | {identity_review.IDENTITY_REVIEW_KIND}
    if (any(actual_kinds.count(kind) != 1 for kind in required_kinds)
            or actual_kinds.count(identity_review.IDENTITY_REVIEW_KIND) > 1
            or not set(actual_kinds) <= allowed_kinds):
        raise ValueError(
            "localization_inputs upstreams must be exactly "
            "bearing_observations, landmark_matches, and catalogs, plus at "
            "most one typed identity_reviews artifact")
    _exact_upstream(manifest, paths_lib.BEARING_OBSERVATIONS)
    _exact_upstream(manifest, paths_lib.LANDMARK_MATCHES)
    _exact_upstream(manifest, paths_lib.CATALOGS)
    review_refs = [
        item for item in manifest.upstreams
        if item.kind == identity_review.IDENTITY_REVIEW_KIND]
    review_config = manifest.config.get("identity_review")
    if review_refs:
        review_ref = review_refs[0]
        if (not isinstance(review_config, dict)
                or not artifact.records_same_artifact(
                    review_config.get("artifact"), review_ref)
                or review_config.get("content_digest")
                != review_ref.content_digest):
            raise ValueError(
                "localization manifest does not bind its identity review "
                "upstream exactly")
    elif review_config is not None:
        raise ValueError(
            "localization manifest records an identity review without its "
            "typed upstream")

    try:
        meta = msgspec.json.decode(
            (export_dir / "export_meta.json").read_bytes(), type=ExportMeta)
    except (OSError, msgspec.DecodeError, msgspec.ValidationError) as exc:
        raise ValueError(f"invalid localization export metadata: {exc}") \
            from exc
    if meta.schema_version != EXPORT_SCHEMA:
        raise ValueError(
            f"export schema must be {EXPORT_SCHEMA!r}, got "
            f"{meta.schema_version!r}")
    if meta.message_schema_version != structs.SCHEMA_VERSION:
        raise ValueError(
            "export message schema differs from localization structs: "
            f"{meta.message_schema_version!r} != {structs.SCHEMA_VERSION!r}")
    if meta.dataset != reference.dataset:
        raise ValueError(
            "export metadata dataset disagrees with the artifact manifest")
    if not isinstance(meta.scenario_name, str) or not meta.scenario_name:
        raise ValueError("scenario_name must be non-empty")
    if (isinstance(meta.n_keyframes, bool)
            or not isinstance(meta.n_keyframes, int)
            or meta.n_keyframes < 2):
        raise ValueError("n_keyframes must be an integer >= 2")
    _finite(meta.anchor_lat_deg, "anchor_lat_deg")
    _finite(meta.anchor_lon_deg, "anchor_lon_deg")
    if not -90.0 <= meta.anchor_lat_deg <= 90.0 \
            or not -180.0 <= meta.anchor_lon_deg <= 180.0:
        raise ValueError("export anchor is outside latitude/longitude bounds")
    _finite(meta.max_visible_range_m, "max_visible_range_m", positive=True)
    _finite(meta.landmark_position_sigma_m,
            "landmark_position_sigma_m", positive=True)
    if meta.matching_coverage != "complete" \
            or manifest.config.get("matching_coverage") != "complete":
        raise ValueError("localization input requires complete matching coverage")
    n_expected = manifest.config.get("matching_n_expected")
    n_successful = manifest.config.get("matching_n_successful")
    if (isinstance(n_expected, bool) or not isinstance(n_expected, int)
            or isinstance(n_successful, bool)
            or not isinstance(n_successful, int)
            or n_expected < 0 or n_successful != n_expected):
        raise ValueError(
            "manifest does not attest complete successful matching requests")
    selected = manifest.config.get("localization_inputs")
    if not isinstance(selected, dict):
        raise ValueError(
            "manifest.localization_inputs must record the selected config")
    for key, expected in (
            ("max_visible_range_m", meta.max_visible_range_m),
            ("landmark_position_sigma_m",
             meta.landmark_position_sigma_m),
            ("reducer_epoch_keyframes", meta.reducer.get("epoch_keyframes"))):
        if selected.get(key) != expected:
            raise ValueError(
                f"export metadata {key} disagrees with manifest config")
    _validate_nominal_forward(export_dir, meta, manifest)
    _validate_motion(export_dir, meta, manifest)
    _validate_reducer(meta, manifest)

    try:
        landmarks = msgspec.json.decode(
            (export_dir / "landmarks.json").read_bytes(),
            type=list[structs.LandmarkEntry], dec_hook=msgspec_dec_hook)
        table_list = msgspec.json.decode(
            (export_dir / "tier1_tables.json").read_bytes(),
            type=list[structs.CompatibilityTable], dec_hook=msgspec_dec_hook)
    except (OSError, msgspec.DecodeError, msgspec.ValidationError) as exc:
        raise ValueError(f"cannot decode localization JSON inputs: {exc}") \
            from exc
    if not landmarks:
        raise ValueError("localization catalog is empty")
    frame = geo.RegionFrame(meta.anchor_lat_deg, meta.anchor_lon_deg)
    east, north = frame.enu_from_latlon(
        np.asarray([item.lat_deg for item in landmarks]),
        np.asarray([item.lon_deg for item in landmarks]))
    catalog = filter_catalog.LandmarkCatalog(
        [item.landmark_id for item in landmarks], east, north,
        max_visible_range_m=meta.max_visible_range_m,
        position_sigma_m=np.asarray(
            [item.position_sigma_m for item in landmarks], dtype=np.float64))
    table_ids = [item.tracklet_id for item in table_list]
    if len(table_ids) != len(set(table_ids)):
        raise ValueError("localization tables repeat a tracklet_id")
    data = ExportData(
        artifact_ref=reference,
        manifest=manifest,
        meta=meta,
        frame=frame,
        catalog=catalog,
        landmarks=landmarks,
        odometry=run_io.read_jsonl(
            export_dir / "tier1_odometry.jsonl", structs.OdometryDelta),
        measurements=run_io.read_jsonl(
            export_dir / "tier1_measurements.jsonl",
            structs.TrackletMeasurement),
        tables={item.tracklet_id: item for item in table_list},
        truth=run_io.read_jsonl(
            export_dir / "truth.jsonl", structs.TruthPose))
    validate(data)
    return data


def _global_tracklet_id(value: str) -> bool:
    return (isinstance(value, str) and "@sha256:" in value
            and "#T" in value and not value.startswith("T"))


def validate(data: ExportData) -> None:
    """Fail at the artifact boundary rather than inside a filter run."""
    problems = []
    expected_odometry = list(range(1, data.meta.n_keyframes))
    actual_odometry = [item.keyframe_idx for item in data.odometry]
    if actual_odometry != expected_odometry:
        problems.append("odometry keyframe indices are not contiguous 1..N-1")
    for item in data.odometry:
        values = (item.forward_m, item.left_m, item.delta_yaw_cw_rad,
                  item.sigma_m, item.sigma_yaw_rad)
        if not all(math.isfinite(value) for value in values):
            problems.append(
                f"odometry at keyframe {item.keyframe_idx} is non-finite")
        if item.sigma_m <= 0.0 or item.sigma_yaw_rad <= 0.0:
            problems.append(
                f"odometry at keyframe {item.keyframe_idx} has non-positive "
                "uncertainty")

    if data.truth:
        truth_indices = [item.keyframe_idx for item in data.truth]
        if truth_indices != list(range(data.meta.n_keyframes)):
            problems.append("truth keyframe indices are not contiguous 0..N-1")
        for item in data.truth:
            if not all(math.isfinite(value) for value in
                       (item.east_m, item.north_m,
                        item.course_world_cw_deg)):
                problems.append(
                    f"truth at keyframe {item.keyframe_idx} is non-finite")
            if not 0.0 <= item.course_world_cw_deg < 360.0:
                problems.append(
                    f"truth course at keyframe {item.keyframe_idx} is not "
                    "world CW [0, 360)")
        if data.meta.motion["course_heading_status"] != \
                "gps_course_diagnostic_only":
            problems.append("truth exists despite a course abstention")
    elif data.meta.motion["course_heading_status"] != \
            "abstained_insufficient_displacement":
        problems.append("truth is absent without a recorded course abstention")

    measurement_order = [
        (item.anchor_keyframe_idx, item.tracklet_id)
        for item in data.measurements]
    if measurement_order != sorted(measurement_order):
        problems.append(
            "measurements are not sorted by (anchor_keyframe_idx, tracklet_id)")
    seen = set()
    for measurement in data.measurements:
        key = (measurement.tracklet_id, measurement.anchor_keyframe_idx)
        if key in seen:
            problems.append(f"duplicate information epoch {key}")
        seen.add(key)
        if not _global_tracklet_id(measurement.tracklet_id):
            problems.append(
                f"measurement {key} does not use a global tracklet id")
        if measurement.tracklet_id not in data.tables:
            problems.append(f"no table for tracklet {measurement.tracklet_id!r}")
        if not 0 <= measurement.anchor_keyframe_idx < data.meta.n_keyframes:
            problems.append(f"measurement anchored outside the run: {key}")
        if not math.isfinite(measurement.kappa) or measurement.kappa <= 0.0:
            problems.append(f"non-positive kappa on {key}")
        cap = measurement.range_max_m
        if cap is not None and (isinstance(cap, bool)
                                or not isinstance(cap, (int, float))
                                or not math.isfinite(cap) or cap <= 0.0):
            problems.append(f"range_max_m must be null or positive on {key}")
        if (not math.isfinite(measurement.bearing_forward_cw_deg)
                or not 0.0 <= measurement.bearing_forward_cw_deg < 360.0):
            problems.append(
                "bearing_forward_cw_deg outside finite [0, 360) on "
                f"{key}: {measurement.bearing_forward_cw_deg}")

    known = {item.landmark_id for item in data.landmarks}
    if len(known) != len(data.landmarks):
        problems.append("landmark ids are not unique")
    sigmas = []
    for landmark in data.landmarks:
        if not all(math.isfinite(value) for value in
                   (landmark.lat_deg, landmark.lon_deg,
                    landmark.position_sigma_m)):
            problems.append(f"landmark {landmark.landmark_id!r} is non-finite")
        if landmark.position_sigma_m <= 0.0:
            problems.append(
                f"landmark {landmark.landmark_id!r} has non-positive sigma")
        sigmas.append(landmark.position_sigma_m)
    if (not sigmas
            or any(value != data.meta.landmark_position_sigma_m
                   for value in sigmas)):
        problems.append(
            "landmark position sigma is not the one uniform recorded value")

    for table in data.tables.values():
        if not _global_tracklet_id(table.tracklet_id):
            problems.append(
                f"table {table.tracklet_id!r} does not use a global tracklet id")
        if table.matcher_version != data.meta.matcher_version:
            problems.append(
                f"table {table.tracklet_id!r} matcher version disagrees "
                "with export meta")
        numeric = (table.default_log_lr, table.clip_lo, table.clip_hi)
        if not all(math.isfinite(value) for value in numeric) \
                or table.clip_lo >= table.clip_hi:
            problems.append(
                f"table {table.tracklet_id!r} has invalid numeric bounds")
        entry_ids = [entry.landmark_id for entry in table.entries]
        if len(entry_ids) != len(set(entry_ids)):
            problems.append(
                f"table {table.tracklet_id!r} repeats a landmark")
        unknown = [landmark_id for landmark_id in entry_ids
                   if landmark_id not in known]
        if unknown:
            problems.append(
                f"table {table.tracklet_id!r} scores a catalog-absent "
                f"landmark {unknown[0]!r}")
        if any(not math.isfinite(entry.log_lr) for entry in table.entries):
            problems.append(
                f"table {table.tracklet_id!r} has non-finite log-LR")
    if problems:
        raise ValueError("export failed validation:\n  - "
                         + "\n  - ".join(problems))


def region_box(data: ExportData, margin_m: float) -> structs.UniformBoxInit:
    """A uniform prior spanning everything the catalog could explain."""
    margin_m = _finite(margin_m, "margin_m")
    if margin_m < 0.0:
        raise ValueError("margin_m must be nonnegative")
    return structs.UniformBoxInit(
        east_min_m=float(data.catalog.east_m.min()) - margin_m,
        east_max_m=float(data.catalog.east_m.max()) + margin_m,
        north_min_m=float(data.catalog.north_m.min()) - margin_m,
        north_max_m=float(data.catalog.north_m.max()) + margin_m)


def describe(data: ExportData) -> str:
    box = region_box(data, 0.0)
    kappas = [item.kappa for item in data.measurements]
    sigmas = [math.degrees(1.0 / math.sqrt(value)) for value in kappas]
    calibration = data.meta.nominal_forward
    return "\n".join([
        f"export      : {data.meta.scenario_name}",
        f"dataset     : {data.meta.dataset}",
        f"matcher     : {data.meta.matcher_version} (complete coverage)",
        f"nominal fwd : camera {calibration['bearing_camera_cw_deg']:.3f} deg "
        f"CW ({calibration['mounting_id']}, approved "
        f"{calibration['approved_at']})",
        f"anchor      : {data.meta.anchor_lat_deg:.6f}, "
        f"{data.meta.anchor_lon_deg:.6f}",
        f"keyframes   : {data.n_keyframes}",
        f"catalog     : {data.catalog.n} landmarks spanning "
        f"{(box.east_max_m - box.east_min_m) / 1000:.1f} x "
        f"{(box.north_max_m - box.north_min_m) / 1000:.1f} km; "
        f"uniform sigma {data.meta.landmark_position_sigma_m:.1f} m",
        f"measurements: {len(data.measurements)} over "
        f"{len({item.tracklet_id for item in data.measurements})} tracklets"
        + (f"; bearing sigma {min(sigmas):.1f}-{max(sigmas):.1f} deg"
           if sigmas else ""),
        f"tables      : {len(data.tables)}",
        f"truth       : {len(data.truth)} poses "
        f"({data.meta.motion['course_heading_status']})",
    ])
