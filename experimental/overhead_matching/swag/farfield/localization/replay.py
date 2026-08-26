"""Tier-3 replay service: the production filter re-run from a run directory.

Design doc §7.1 stores three tiers and recomputes the rest. This module is the
recompute. It reads a run directory, rebuilds the exact filter inputs, and
calls `filter.run_filter` — *the same function that produced the run*, not a
reimplementation of it (§7.5 [CONTRACT]). Divergence between "what the viewer
says happened" and "what happened" is therefore not a bug that can be
introduced; it would require the filter to disagree with itself.

Two things about this are worth stating plainly, because both differ from what
§7.1 describes:

**Replay is from keyframe 0, not from the nearest checkpoint.** §7.1 promises
bounded recompute by resuming from a Tier-2 checkpoint, but the checkpoints as
written cannot support that: they carry particle pose and weight, and not the
RNG state, the per-tracklet association arrays (dropped on purpose), or the
ModeTracker's lineage registry. Resuming from one would produce a *plausible*
continuation rather than the run's actual one, which is exactly the class of
quiet infidelity this module exists to rule out. Measured cost of the honest
alternative on the whole-map harbour run — 13,210 landmarks, 20k particles,
379 keyframes, GPU backend — is 32 s, so bounded recompute buys little here
and would cost the exactness guarantee. If runs get long enough for this to
hurt, the fix is to checkpoint the missing state, not to resume without it.

**A run is only replayable if its manifest fully determines the filter.** The
config is stored as a msgspec struct, so fields added to `FilterConfig` after
a run was written are silently filled with *today's* defaults on read — a
replay of such a run runs different filter semantics under the original run's
name. `replayability(run_dir)` detects this by diffing the stored JSON's keys
against the current struct, and `replay(verify=True)` refuses to claim
fidelity it cannot demonstrate. The tell that everything worked is
`particle_history_sha256`: the manifest records the original, replay
recomputes it, and a mismatch is reported rather than swallowed.

`max_visible_range_m` is READ FROM THE MANIFEST, never guessed: schema 0.3
made it a required field, so every readable run records the radius its
proposal geometry was built with. (An earlier fallback constant here claimed
to match the catalog default and did not — 15 km vs 10 km — which is exactly
the silent divergence a recorded value exists to prevent.)

Usage:
  # faithful reconstruction, hash-verified
  result = replay.replay(run_dir)

  # counterfactual: what if the matcher had never scored this tracklet?
  result = replay.replay(run_dir, edits=replay.Edits(
      drop_tracklets=["LT267"]))
"""

import dataclasses
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import msgspec
import numpy as np

from experimental.overhead_matching.swag.farfield import artifact, provenance
from common.python.serialization import MSGSPEC_STRUCT_OPTS, msgspec_enc_hook
from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.localization import (
    filter as pf,
    filter_catalog as catalog_mod,
    metrics,
    run_io,
    runner,
    structs,
)

# Counterfactuals are mutable diagnostic products, not outputs declared by the
# completed source-run artifact. Keep them in a deterministic sibling so a
# replay never changes the source artifact's content digest or identity.
# forensics_cli and viewer_server both build paths through
# default_counterfactual_dir.
COUNTERFACTUAL_DIR_SUFFIX = ".counterfactuals"


class _ObserverGroup(pf.RunObserver):
    """Fan each read-only filter hook out to independent observers."""

    def __init__(self, observers):
        self._observers = tuple(observer for observer in observers
                                if observer is not None)

    def keyframe_start(self, keyframe_idx, belief):
        for observer in self._observers:
            observer.keyframe_start(keyframe_idx, belief)

    def measurement(self, keyframe_idx, meas, log_weight_before, belief,
                    pass_index):
        for observer in self._observers:
            observer.measurement(keyframe_idx, meas, log_weight_before, belief,
                                 pass_index)

    def injection(self, keyframe_idx, event, n_injected):
        for observer in self._observers:
            observer.injection(keyframe_idx, event, n_injected)

    def resample(self, keyframe_idx, log_weight_before, mode_id_before, belief):
        for observer in self._observers:
            observer.resample(keyframe_idx, log_weight_before, mode_id_before,
                              belief)

    def keyframe_end(self, keyframe_idx, belief, health):
        for observer in self._observers:
            observer.keyframe_end(keyframe_idx, belief, health)


def _edits_document(edits: "Edits") -> dict:
    """Plain JSON value used as the one canonical edit identity."""
    return msgspec.json.decode(msgspec.json.encode(
        edits, enc_hook=msgspec_enc_hook))


class Edits(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """A counterfactual: the §7.4 view-5 what-if console, as data.

    Every field leaves the run's inputs otherwise untouched, so a replay with
    edits differs from the original in exactly the stated way and the ghost
    trajectory it produces is attributable to that one change. Serialized into
    the counterfactual's own manifest, so a ghost run directory records what
    made it a ghost.
    """
    # Silence tracklets entirely: their measurements never reach the filter.
    # The "was this tracklet actually the culprit" confirmation of §7.4.
    drop_tracklets: tuple[str, ...] = ()
    # Keep only these (applied after drop_tracklets). Isolates a subset.
    keep_only_tracklets: tuple[str, ...] | None = None
    # Rewrite specific log-LRs: tracklet_id -> landmark_id -> log_lr. Entries
    # not already in the table are appended, so this can add evidence as well
    # as edit it. Values are still clipped by the filter.
    log_lr: dict[str, dict[str, float]] = {}
    # Rewrite a tracklet's table to endorse ONE landmark at clip_hi and
    # nothing else: "what if the matcher had simply got this right?". Paired
    # with the truth triage, this is how a matcher-fault verdict gets tested
    # rather than asserted.
    force_landmark: dict[str, str] = {}
    # Filter knobs the console exposes directly.
    pi0: float | None = None
    matcher_recall: float | None = None
    seed: int | None = None
    n_particles: int | None = None
    measurement_backend: str | None = None
    disable_proposal: bool = False
    disable_persistence: bool = False
    disable_modes: bool = False
    checkpoint_every: int | None = None

    @property
    def is_empty(self) -> bool:
        """True when this asks for the original run back."""
        return self == Edits()

    def describe(self) -> str:
        """One line naming the counterfactual, for labels and filenames."""
        parts = []
        if self.drop_tracklets:
            parts.append("without " + ",".join(self.drop_tracklets))
        if self.keep_only_tracklets is not None:
            parts.append("only " + ",".join(self.keep_only_tracklets))
        for tid, landmark_id in sorted(self.force_landmark.items()):
            parts.append(f"{tid}:={landmark_id}")
        for tid, overrides in sorted(self.log_lr.items()):
            parts.append(f"{tid} llr[" + ",".join(
                f"{lid}={value:+.1f}"
                for lid, value in sorted(overrides.items())) + "]")
        for name in ("pi0", "matcher_recall", "seed", "n_particles",
                     "measurement_backend", "checkpoint_every"):
            value = getattr(self, name)
            if value is not None:
                parts.append(f"{name}={value}")
        for name in ("disable_proposal", "disable_persistence",
                     "disable_modes"):
            if getattr(self, name):
                parts.append(name)
        return "; ".join(parts) or "unmodified"

    def slug(self) -> str:
        """Readable filesystem name plus collision-resistant edit identity."""
        text = self.describe().replace("unmodified", "baseline")
        keep = [c if (c.isalnum() or c in "-_") else "_" for c in text]
        readable = "".join(keep)[:60].strip("_") or "baseline"
        identity = artifact.sha256_json(_edits_document(self))[:16]
        return f"{readable}--{identity}"


def default_counterfactual_dir(run_dir: Path, edits: Edits) -> Path:
    """Return ``<run>.counterfactuals/<slug>`` beside the immutable run."""
    run_dir = Path(run_dir)
    root = run_dir.with_name(run_dir.name + COUNTERFACTUAL_DIR_SUFFIX)
    return root / edits.slug()


@dataclasses.dataclass
class Replayability:
    """Whether a run directory fully determines a replay, and what is missing.

    `missing_config_keys` is the important one: those are `FilterConfig`
    fields the run was written without, which today's struct fills with
    defaults that may not match the semantics the run actually used.
    """
    replayable: bool
    schema_version: str
    missing_config_keys: tuple[str, ...]
    has_max_visible_range: bool
    has_particle_history_hash: bool
    notes: tuple[str, ...]

    def report(self) -> str:
        lines = [f"schema {self.schema_version}; "
                 + ("replayable" if self.replayable
                    else "NOT faithfully replayable")]
        if self.missing_config_keys:
            lines.append(
                "  filter config fields absent from the stored manifest "
                "(today's defaults would be substituted, changing filter "
                "semantics): " + ", ".join(self.missing_config_keys))
        lines.extend(f"  {note}" for note in self.notes)
        return "\n".join(lines)


def _missing_keys(stored: dict, struct_type, prefix: str = "") -> list:
    """Config keys the current struct declares that the stored JSON lacks."""
    missing = []
    info = msgspec.inspect.type_info(struct_type)
    if not isinstance(info, msgspec.inspect.StructType):
        return missing
    for field in info.fields:
        name = f"{prefix}{field.name}"
        if field.name not in stored:
            missing.append(name)
            continue
        value = stored[field.name]
        if not isinstance(value, dict):
            continue
        # Nested tagged struct: recurse into whichever union member the
        # stored "kind" tag names.
        candidates = [c for c in (getattr(field.type, "types", None)
                                  or [field.type])
                      if isinstance(c, msgspec.inspect.StructType)]
        for candidate in candidates:
            tag = getattr(candidate, "tag", None)
            if len(candidates) > 1 and value.get("kind") != tag:
                continue
            missing.extend(_missing_keys(value, candidate.cls, f"{name}."))
            break
    return missing


def replayability(run_dir: Path) -> Replayability:
    """Can this run directory be replayed as the run it records?

    Reads the manifest as raw JSON on purpose: decoding it into the current
    struct is precisely the step that hides the problem, by filling absent
    fields with present-day defaults.
    """
    run_dir = Path(run_dir)
    raw = msgspec.json.decode(
        (run_dir / run_io.RUN_MANIFEST_NAME).read_bytes())
    stored_config = raw.get("filter_config", {})
    missing = tuple(_missing_keys(stored_config, structs.FilterConfig))
    notes = []
    schema = raw.get("schema_version", "?")
    if schema != structs.SCHEMA_VERSION:
        notes.append(f"schema {schema!r} but this build reads "
                     f"{structs.SCHEMA_VERSION!r}")
    has_range = raw.get("max_visible_range_m") is not None
    if not has_range:
        notes.append(
            "max_visible_range_m is not recorded; the proposal's visibility "
            "geometry is undetermined, so this run cannot be replayed by "
            "this build (schema 0.3 made the field required)")
    has_history_hash = bool(raw.get("particle_history_sha256"))
    if not has_history_hash:
        notes.append("no particle_history_sha256, so a replay cannot be "
                     "verified against the original")
    return Replayability(
        replayable=(not missing and schema == structs.SCHEMA_VERSION
                    and has_range and has_history_hash),
        schema_version=schema, missing_config_keys=missing,
        has_max_visible_range=has_range,
        has_particle_history_hash=has_history_hash, notes=tuple(notes))


@dataclasses.dataclass
class ReplayInputs:
    """Everything `run_filter` needs, rebuilt from a run directory alone."""
    config: structs.FilterConfig
    catalog: catalog_mod.LandmarkCatalog
    odometry: list
    measurements: list
    tables: dict
    data: run_io.RunData
    frame: geo.RegionFrame
    max_visible_range_m: float

    @property
    def manifest(self) -> structs.RunManifest:
        return self.data.manifest


def region_frame(manifest: structs.RunManifest) -> geo.RegionFrame:
    return geo.RegionFrame(manifest.anchor_lat_deg, manifest.anchor_lon_deg)


def build_catalog(manifest: structs.RunManifest, max_visible_range_m: float,
                  frame: geo.RegionFrame | None = None
                  ) -> catalog_mod.LandmarkCatalog:
    """The run's catalog, in the run's own ENU frame.

    The manifest carries every catalog row's lat/lon, so this needs no access
    to the export the run was built from — which is what makes a run directory
    a self-sufficient replay unit.
    """
    frame = frame or region_frame(manifest)
    east, north = frame.enu_from_latlon(
        np.array([lm.lat_deg for lm in manifest.landmarks]),
        np.array([lm.lon_deg for lm in manifest.landmarks]))
    return catalog_mod.LandmarkCatalog(
        [lm.landmark_id for lm in manifest.landmarks], east, north,
        position_sigma_m=[lm.position_sigma_m for lm in manifest.landmarks],
        max_visible_range_m=max_visible_range_m)


def load_inputs(run_dir: Path,
                data: run_io.RunData | None = None) -> ReplayInputs:
    """Rebuild the filter's inputs from a run directory.

    The visibility radius comes from the manifest — required since schema
    0.3, so there is no fallback and no override: the replay's geometry is
    the run's geometry.
    """
    run_dir = Path(run_dir)
    data = data if data is not None else run_io.read_run(run_dir)
    visible_range = data.manifest.max_visible_range_m
    frame = region_frame(data.manifest)
    return ReplayInputs(
        config=data.manifest.filter_config,
        catalog=build_catalog(data.manifest, visible_range, frame),
        odometry=data.odometry, measurements=data.measurements,
        tables=data.tables, data=data, frame=frame,
        max_visible_range_m=visible_range)


def _edit_table(table: structs.CompatibilityTable, overrides: dict
                ) -> structs.CompatibilityTable:
    """Apply validated landmark_id -> log_lr overrides."""
    entries = {entry.landmark_id: entry.log_lr for entry in table.entries}
    entries.update(overrides)
    return msgspec.structs.replace(table, entries=[
        structs.CompatibilityEntry(landmark_id=landmark_id, log_lr=log_lr)
        for landmark_id, log_lr in entries.items()])


def apply_edits(inputs: ReplayInputs, edits: Edits) -> ReplayInputs:
    """A counterfactual's inputs. Never mutates `inputs`."""
    measurement_ids = {item.tracklet_id for item in inputs.measurements}
    table_ids = set(inputs.tables)
    catalog_ids = set(inputs.catalog.landmark_ids)

    def _unique_known(values, known, where):
        values = tuple(values)
        if len(values) != len(set(values)):
            raise ValueError(f"{where} contains duplicate IDs")
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"{where} contains unknown IDs: {unknown}")
        return set(values)

    dropped = _unique_known(
        edits.drop_tracklets, measurement_ids, "drop_tracklets")
    kept = None
    if edits.keep_only_tracklets is not None:
        kept = _unique_known(
            edits.keep_only_tracklets, measurement_ids,
            "keep_only_tracklets")
        overlap = dropped & kept
        if overlap:
            raise ValueError(
                "tracklets cannot be both dropped and retained: "
                f"{sorted(overlap)}")
    override_tracklets = set(edits.log_lr) | set(edits.force_landmark)
    unknown_tables = sorted(override_tracklets - table_ids)
    if unknown_tables:
        raise ValueError(
            f"table edits contain unknown tracklet IDs: {unknown_tables}")
    overlap_edits = set(edits.log_lr) & set(edits.force_landmark)
    if overlap_edits:
        raise ValueError(
            "a tracklet cannot have both force_landmark and log_lr edits: "
            f"{sorted(overlap_edits)}")
    inactive = override_tracklets & dropped
    if inactive or (kept is not None and override_tracklets - kept):
        raise ValueError(
            "table edits must refer to tracklets retained by the "
            "counterfactual")
    for tracklet_id, landmark_id in edits.force_landmark.items():
        if landmark_id not in catalog_ids:
            raise ValueError(
                f"force_landmark[{tracklet_id!r}] references unknown "
                f"landmark {landmark_id!r}")
    for tracklet_id, overrides in edits.log_lr.items():
        if not overrides:
            raise ValueError(f"log_lr[{tracklet_id!r}] must not be empty")
        unknown = sorted(set(overrides) - catalog_ids)
        if unknown:
            raise ValueError(
                f"log_lr[{tracklet_id!r}] references unknown landmarks: "
                f"{unknown}")
        for landmark_id, value in overrides.items():
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))):
                raise ValueError(
                    f"log_lr[{tracklet_id!r}][{landmark_id!r}] must be "
                    "finite numeric")

    config = inputs.config
    replacements = {}
    for name, field in (("pi0", "pi0"), ("matcher_recall", "matcher_recall"),
                        ("seed", "seed"), ("n_particles", "n_particles"),
                        ("measurement_backend", "measurement_backend"),
                        ("checkpoint_every", "checkpoint_every")):
        value = getattr(edits, name)
        if value is not None:
            replacements[field] = value
    if edits.disable_persistence:
        replacements["association_persistence"] = False
    if edits.disable_proposal:
        replacements["proposal"] = msgspec.structs.replace(
            config.proposal, enabled=False)
    if edits.disable_modes:
        replacements["modes"] = msgspec.structs.replace(
            config.modes, enabled=False)
    if replacements:
        config = msgspec.structs.replace(config, **replacements)

    measurements = inputs.measurements
    if dropped:
        measurements = [m for m in measurements
                        if m.tracklet_id not in dropped]
    if kept is not None:
        measurements = [m for m in measurements if m.tracklet_id in kept]

    tables = inputs.tables
    if edits.log_lr or edits.force_landmark:
        tables = dict(tables)
        for tracklet_id, landmark_id in edits.force_landmark.items():
            table = tables[tracklet_id]
            tables[tracklet_id] = msgspec.structs.replace(
                table, entries=[structs.CompatibilityEntry(
                    landmark_id=landmark_id, log_lr=table.clip_hi)],
                matcher_version=f"{table.matcher_version}+forced")
        for tracklet_id, overrides in edits.log_lr.items():
            tables[tracklet_id] = _edit_table(tables[tracklet_id], overrides)

    result = dataclasses.replace(inputs, config=config,
                                 measurements=measurements, tables=tables)
    if (not edits.is_empty and result.config == inputs.config
            and result.measurements == inputs.measurements
            and result.tables == inputs.tables):
        raise ValueError("counterfactual edits are a no-op")
    return result


@dataclasses.dataclass
class ReplayResult:
    history: pf.FilterHistory
    inputs: ReplayInputs
    edits: Edits
    elapsed_s: float
    # None when there is nothing to compare against (an edited replay, or a
    # run with no recorded hash). True/False otherwise, and False is a
    # finding: the reconstruction is not the run.
    hash_match: bool | None
    replayability: Replayability
    recorded_sha256: str
    replayed_sha256: str
    # Edited runs are valid counterfactuals only after this build first
    # reproduces the unedited source run exactly.
    baseline_hash_match: bool | None = None

    @property
    def faithful(self) -> bool:
        """A verified reconstruction of the original run."""
        return bool(self.edits.is_empty and self.hash_match)

    def report(self) -> str:
        lines = [f"replayed {self.inputs.manifest.scenario_name} "
                 f"({self.inputs.manifest.n_keyframes} keyframes, "
                 f"{self.inputs.catalog.n} landmarks, "
                 f"{len(self.inputs.measurements)} measurements) in "
                 f"{self.elapsed_s:.1f}s"]
        if not self.edits.is_empty:
            lines.append(f"  counterfactual: {self.edits.describe()}")
            if self.baseline_hash_match:
                lines.append("  unedited baseline hash MATCHES")
        if self.hash_match is True:
            lines.append(f"  history hash MATCHES "
                         f"{self.recorded_sha256[:12]} — bit-exact "
                         f"reconstruction")
        elif self.hash_match is False:
            lines.append(f"  history hash DIVERGED: recorded "
                         f"{self.recorded_sha256[:12]}, replayed "
                         f"{self.replayed_sha256[:12]}")
        if self.replayability.missing_config_keys:
            lines.append("  " + self.replayability.report().replace(
                "\n", "\n  "))
        return "\n".join(lines)


def replay(run_dir: Path, edits: Edits | None = None,
           observer: pf.RunObserver | None = None,
           verify: bool = True,
           data: run_io.RunData | None = None) -> ReplayResult:
    """Re-run the production filter over a run directory's inputs.

    With `edits=None` this reconstructs the run and checks the reconstruction
    against the recorded `particle_history_sha256`. With `verify=True` (the
    default) a divergence raises, because a silent near-miss is worse than no
    replay at all: every Tier-3 number downstream would describe a run that
    never happened. Pass `verify=False` to inspect a divergence instead.
    """
    edits = edits or Edits()
    inputs = load_inputs(run_dir, data=data)
    status = replayability(run_dir)
    if verify and not status.replayable:
        raise ValueError(
            "the run directory does not fully determine a verified replay\n"
            + status.report())
    edited = apply_edits(inputs, edits) if not edits.is_empty else inputs
    metric_config = edited.manifest.position_mass_metric
    mass_recorder = (
        runner.PositionMassRecorder(edited.data.truth, metric_config)
        if edited.data.truth and metric_config is not None else None)
    observers = [value for value in (observer, mass_recorder)
                 if value is not None]
    run_observer = (_ObserverGroup(observers) if len(observers) > 1 else
                    (observers[0] if observers else None))

    recorded = inputs.manifest.particle_history_sha256 or ""
    baseline_hash_match = None
    if not edits.is_empty and verify:
        baseline = pf.run_filter(
            inputs.config, inputs.catalog, inputs.odometry,
            inputs.measurements, inputs.tables)
        baseline_hash_match = baseline.particle_history_sha256 == recorded
        if not baseline_hash_match:
            baseline_result = ReplayResult(
                history=baseline, inputs=inputs, edits=Edits(), elapsed_s=0.0,
                hash_match=False, replayability=status,
                recorded_sha256=recorded,
                replayed_sha256=baseline.particle_history_sha256,
                baseline_hash_match=False)
            raise ReplayDivergence(
                baseline_result,
                reason="unedited baseline diverged; counterfactual refused")

    start = time.perf_counter()
    history = pf.run_filter(edited.config, edited.catalog, edited.odometry,
                            edited.measurements, edited.tables,
                            observer=run_observer)
    if mass_recorder is not None:
        for record in history.health:
            record.position_probability_mass = mass_recorder.by_keyframe[
                record.keyframe_idx]
    elapsed = time.perf_counter() - start

    hash_match = None
    if edits.is_empty and recorded:
        hash_match = history.particle_history_sha256 == recorded
    result = ReplayResult(
        history=history, inputs=edited, edits=edits, elapsed_s=elapsed,
        hash_match=hash_match, replayability=status, recorded_sha256=recorded,
        replayed_sha256=history.particle_history_sha256,
        baseline_hash_match=(hash_match if edits.is_empty
                             else baseline_hash_match))

    if verify and hash_match is False:
        raise ReplayDivergence(result)
    return result


class ReplayDivergence(RuntimeError):
    """A replay did not reproduce the run it claims to reconstruct."""

    def __init__(self, result: ReplayResult, reason: str | None = None):
        self.result = result
        message = reason or "replay diverged from the recorded run"
        super().__init__(f"{message}\n{result.report()}")


def write_counterfactual(output_dir: Path, source_run_dir: Path,
                         result: ReplayResult) -> Path:
    """Persist a counterfactual as a first-class run directory.

    A ghost that is a run directory can be opened in the viewer, diffed,
    replayed again, and archived, which a transient in-memory result cannot.
    The manifest's scenario_name records what made it a ghost.
    """
    output_dir = Path(output_dir)
    # A generated scenario has no LOCALIZATION_INPUTS artifact to claim. Its
    # counterfactual therefore remains synthetic; only counterfactuals of real
    # runs become diagnostic controls.
    run_kind = ("synthetic"
                if result.inputs.manifest.run_kind == "synthetic"
                else "diagnostic_control")
    manifest = msgspec.structs.replace(
        result.inputs.manifest,
        scenario_name=f"{result.inputs.manifest.scenario_name}"
                      f" [{result.edits.describe()}]",
        filter_config=result.inputs.config,
        max_visible_range_m=result.inputs.max_visible_range_m,
        particle_history_sha256=result.history.particle_history_sha256,
        run_kind=run_kind,
        git_commit=provenance.git_commit(),
        argv=list(sys.argv),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    source_ref = artifact.open_artifact(source_run_dir)
    source_artifact = artifact.load_manifest(source_run_dir)
    counterfactual = msgspec.json.encode({
        "source_run_dir": str(Path(source_run_dir).resolve()),
        "edits": result.edits,
        "edit_digest": artifact.sha256_json(_edits_document(result.edits)),
        "describe": result.edits.describe(),
        "elapsed_s": round(result.elapsed_s, 2),
        "baseline_hash_verified": result.baseline_hash_match is True,
    }, enc_hook=msgspec_enc_hook)
    consumed_tracklets = {
        measurement.tracklet_id for measurement in result.inputs.measurements
    }
    consumed_tables = {
        tracklet_id: table
        for tracklet_id, table in result.inputs.tables.items()
        if tracklet_id in consumed_tracklets
    }
    extra_outputs = {"counterfactual.json": counterfactual}
    if manifest.position_mass_metric is not None:
        summary = metrics.position_mass_summary(
            result.history.health, manifest.position_mass_metric)
        extra_outputs[metrics.POSITION_MASS_SUMMARY_NAME] = (
            msgspec.json.encode(summary) + b"\n")
    run_io.write_run(output_dir, manifest, result.inputs.data.truth,
                     result.inputs.odometry, result.inputs.measurements,
                     consumed_tables, result.history,
                     dataset=manifest.dataset, version=output_dir.name,
                     upstreams=(*source_artifact.upstreams, source_ref),
                     artifact_config={
                         "run_kind": run_kind,
                         "source_run": source_ref.to_dict(),
                         "counterfactual_edits": _edits_document(
                             result.edits),
                         "counterfactual_edit_digest": artifact.sha256_json(
                             _edits_document(result.edits)),
                     },
                     generator=("//experimental/overhead_matching/swag/"
                                "farfield/localization:replay"),
                     extra_outputs=extra_outputs)
    return output_dir
