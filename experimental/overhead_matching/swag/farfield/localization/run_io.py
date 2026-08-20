"""Self-describing run directory for localization runs (design doc §7.5).

Layout (all consumers — plots, tests, viewers — read only this):
  manifest.json             RunManifest (config echo, provenance, history hash)
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

This module was called `run_log.py` and sat one underscore away from the
`runlog` forensics CLI; renamed so the I/O library and the CLI cannot be
confused again.

The JSONL helpers (`read_jsonl` / `write_jsonl`) are public: they were
re-implemented five times across the old package.
"""

import dataclasses
from pathlib import Path

import msgspec
import numpy as np

from common.python.serialization import msgspec_dec_hook, msgspec_enc_hook
from experimental.overhead_matching.swag.farfield.localization import structs


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


def write_jsonl(path: Path, records) -> None:
    with open(path, "wb") as f:
        for record in records:
            f.write(msgspec.json.encode(record, enc_hook=msgspec_enc_hook))
            f.write(b"\n")


def read_jsonl(path: Path, record_type) -> list:
    if not Path(path).exists():
        return []
    return [
        msgspec.json.decode(line, type=record_type,
                            dec_hook=msgspec_dec_hook)
        for line in Path(path).read_bytes().splitlines() if line.strip()]


def validate_manifest(manifest: structs.RunManifest) -> None:
    """Refuse to write a run that cannot name its own inputs."""
    problems = []
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
    if problems:
        raise ValueError("run manifest fails provenance validation:\n  - "
                         + "\n  - ".join(problems))


def write_run(run_dir: Path, manifest: structs.RunManifest, truth: list,
              odometry: list, measurements: list, tables: dict,
              history) -> None:
    """`history` is a filter.FilterHistory (duck-typed to avoid the dep).

    `measurements`/`tables` must be the ones the filter actually consumed:
    an odometry-only control run passes its empty lists, never the full
    inputs it chose to ignore (writing the unconsumed ones once produced run
    directories describing runs that never happened).
    """
    validate_manifest(manifest)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "manifest.json", "wb") as f:
        f.write(msgspec.json.encode(manifest, enc_hook=msgspec_enc_hook))
    write_jsonl(run_dir / "tier0_health.jsonl", history.health)
    write_jsonl(run_dir / "tier1_odometry.jsonl", odometry)
    write_jsonl(run_dir / "tier1_measurements.jsonl", measurements)
    with open(run_dir / "tier1_tables.json", "wb") as f:
        f.write(msgspec.json.encode(
            sorted(tables.values(), key=lambda t: t.tracklet_id),
            enc_hook=msgspec_enc_hook))
    write_jsonl(run_dir / "truth.jsonl", truth)
    write_jsonl(run_dir / "events.jsonl", history.proposal_events)
    write_jsonl(run_dir / "mode_events.jsonl", history.mode_events)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    keyframes = sorted(history.checkpoints.keys())
    with open(checkpoint_dir / "index.json", "wb") as f:
        f.write(msgspec.json.encode(keyframes))
    for kf in keyframes:
        belief = history.checkpoints[kf]
        np.savez(checkpoint_dir / f"kf_{kf:05d}.npz",
                 east_m=belief.east_m, north_m=belief.north_m,
                 heading_rad=belief.heading_rad,
                 log_weight=belief.log_weight,
                 proposal_event_id=belief.proposal_event_id,
                 proposal_hypothesis=belief.proposal_hypothesis,
                 mode_id=belief.mode_id)


def read_run(run_dir: Path) -> RunData:
    run_dir = Path(run_dir)
    manifest = msgspec.json.decode(
        (run_dir / "manifest.json").read_bytes(), type=structs.RunManifest,
        dec_hook=msgspec_dec_hook)
    if manifest.schema_version != structs.SCHEMA_VERSION:
        raise ValueError(
            f"run directory {run_dir} has schema version "
            f"{manifest.schema_version!r}, this build reads "
            f"{structs.SCHEMA_VERSION!r}")
    tables_list = msgspec.json.decode(
        (run_dir / "tier1_tables.json").read_bytes(),
        type=list[structs.CompatibilityTable], dec_hook=msgspec_dec_hook)

    checkpoint_dir = run_dir / "checkpoints"
    keyframes = msgspec.json.decode(
        (checkpoint_dir / "index.json").read_bytes(), type=list[int])
    checkpoints = {}
    for kf in keyframes:
        with np.load(checkpoint_dir / f"kf_{kf:05d}.npz") as npz:
            checkpoints[kf] = {key: npz[key] for key in npz.files}

    return RunData(
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
                               structs.ModeEvent))
