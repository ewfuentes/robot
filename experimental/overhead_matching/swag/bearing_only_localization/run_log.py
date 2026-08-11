"""Self-describing run directory for localization runs (design doc §7.5).

Layout (all consumers — plots, tests, future viewers — read only this):
  manifest.json             RunManifest (config echo, versions, history hash)
  tier0_health.jsonl        HealthRecord per keyframe
  tier1_odometry.jsonl      OdometryDelta per keyframe
  tier1_measurements.jsonl  TrackletMeasurement events
  tier1_tables.json         CompatibilityTable list
  truth.jsonl               TruthPose per keyframe (synthetic runs)
  checkpoints/index.json    sorted checkpoint keyframe indices
  checkpoints/kf_00042.npz  particle arrays (east_m/north_m/heading_rad/
                            log_weight)

Tier 1 plus the manifest's config re-runs the filter bit-exactly *in the
same environment* (the §7.1 replay contract). Bit-exactness is not promised
across numpy/BLAS/scipy versions, which can change reduction order; the
manifest records the history hash so a divergence is at least detectable.
Checkpoints are the sparse Tier 2, and hold the weighted posterior at their
keyframe (pre-resample, pre-roughening).
"""

import dataclasses
from pathlib import Path

import msgspec
import numpy as np

from common.python.serialization import msgspec_dec_hook, msgspec_enc_hook
from experimental.overhead_matching.swag.bearing_only_localization import (
    structs,
)


@dataclasses.dataclass
class RunData:
    manifest: structs.RunManifest
    truth: list
    odometry: list
    measurements: list
    tables: dict
    health: list
    checkpoints: dict  # keyframe_idx -> dict[str, np.ndarray]


def _write_jsonl(path: Path, records) -> None:
    with open(path, "wb") as f:
        for record in records:
            f.write(msgspec.json.encode(record, enc_hook=msgspec_enc_hook))
            f.write(b"\n")


def _read_jsonl(path: Path, record_type) -> list:
    if not path.exists():
        return []
    return [
        msgspec.json.decode(line, type=record_type,
                            dec_hook=msgspec_dec_hook)
        for line in path.read_bytes().splitlines() if line.strip()]


def write_run(run_dir: Path, manifest: structs.RunManifest, truth: list,
              odometry: list, measurements: list, tables: dict,
              history) -> None:
    """`history` is a filter.FilterHistory (duck-typed to avoid the dep)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "manifest.json", "wb") as f:
        f.write(msgspec.json.encode(manifest, enc_hook=msgspec_enc_hook))
    _write_jsonl(run_dir / "tier0_health.jsonl", history.health)
    _write_jsonl(run_dir / "tier1_odometry.jsonl", odometry)
    _write_jsonl(run_dir / "tier1_measurements.jsonl", measurements)
    with open(run_dir / "tier1_tables.json", "wb") as f:
        f.write(msgspec.json.encode(
            sorted(tables.values(), key=lambda t: t.tracklet_id),
            enc_hook=msgspec_enc_hook))
    _write_jsonl(run_dir / "truth.jsonl", truth)

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
                 log_weight=belief.log_weight)


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
        truth=_read_jsonl(run_dir / "truth.jsonl", structs.TruthPose),
        odometry=_read_jsonl(run_dir / "tier1_odometry.jsonl",
                             structs.OdometryDelta),
        measurements=_read_jsonl(run_dir / "tier1_measurements.jsonl",
                                 structs.TrackletMeasurement),
        tables={t.tracklet_id: t for t in tables_list},
        health=_read_jsonl(run_dir / "tier0_health.jsonl",
                           structs.HealthRecord),
        checkpoints=checkpoints)
