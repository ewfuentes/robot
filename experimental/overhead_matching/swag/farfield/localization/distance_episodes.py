"""Portable distance windows over immutable, forward-only tracking outputs."""

import argparse
import bisect
import dataclasses
import json
import math
from pathlib import Path

import msgspec

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    release_schedule,
)

SCHEMA = "farfield_distance_episodes/v1"


def windows(truth, count):
    """Inclusive keyframe bounds, approximately 50% overlapping in arc length.

    Boundaries snap forward to recorded keyframes. One window means the
    original full recording. No minimum duration or truth-centered prior.
    """
    if type(count) is not int or count < 1 or len(truth) < 2:
        raise ValueError("need a positive window count and at least two poses")
    if [p.keyframe_idx for p in truth] != list(range(len(truth))):
        raise ValueError("truth keyframes must be contiguous from zero")
    distance = [0.0]
    for a, b in zip(truth, truth[1:]):
        step = math.hypot(b.east_m - a.east_m, b.north_m - a.north_m)
        if not math.isfinite(step):
            raise ValueError("truth positions must be finite")
        distance.append(distance[-1] + step)
    if count == 1:
        return [(0, len(truth) - 1)]
    if distance[-1] <= 0:
        raise ValueError("cannot split a zero-distance recording")
    stride = distance[-1] / (count + 1)
    bounds = [(bisect.bisect_left(distance, i * stride),
               min(bisect.bisect_left(distance, (i + 2) * stride),
                   len(truth) - 1)) for i in range(count)]
    bounds[-1] = (bounds[-1][0], len(truth) - 1)
    if any(a >= b for a, b in bounds) or len(set(bounds)) != count:
        raise ValueError("recording has too few distinct keyframes for windows")
    return bounds


def _upstream(manifest, kind):
    refs = [r for r in manifest.upstreams if r.kind == kind]
    if len(refs) != 1:
        raise ValueError(f"expected exactly one {kind} upstream")
    return refs[0]


def build_plan(data, releases, count, artifacts_dir, schedule_path):
    """Verify source tracks once; workers need only the bound portable plan."""
    bearing_ref = _upstream(data.manifest, "bearing_observations")
    bearing_dir = artifacts_dir / bearing_ref.kind / bearing_ref.dataset / bearing_ref.version
    if artifact.open_artifact(bearing_dir) != bearing_ref:
        raise ValueError("bearing artifact differs from input binding")
    tracks_ref = _upstream(artifact.load_manifest(bearing_dir), "object_tracks")
    tracks_dir = artifacts_dir / tracks_ref.kind / tracks_ref.dataset / tracks_ref.version
    if artifact.open_artifact(tracks_dir) != tracks_ref:
        raise ValueError("source tracks differ from bearing binding")
    records = json.loads((tracks_dir / "tracks_full.json").read_text())["tracks"]
    tracks = {f"T{r['track_id']}": r for r in records}
    if len(tracks) != len(records):
        raise ValueError("source track IDs repeat")
    births = {}
    for release in releases:
        track = tracks[release.tracklet_id.split("#")[-1]]
        birth = track["birth_keyframe"]
        expected_close = (track["last_keyframe"] if track["status"] == "closed"
                          else data.n_keyframes - 1)
        if track["status"] not in ("closed", "alive") or expected_close != release.release_keyframe_idx:
            raise ValueError("release disagrees with source track closure")
        births[release.tracklet_id] = birth
    plan = {
        "schema": SCHEMA,
        "localization_inputs": data.artifact_ref.to_dict(),
        "release_schedule_sha256": artifact.sha256_file(schedule_path),
        "source_tracks": tracks_ref.to_dict(),
        "source_track_births": births,
        "windows": [list(b) for b in windows(data.truth, count)],
        "boundary_policy": "fully_contained_source_track_inclusive_end",
    }
    # Apply the same validation as remote readers before publishing.
    select(data, releases, plan, 0, schedule_path)
    return plan


def select(data, releases, plan, index, schedule_path):
    """Select whole tracks, reindex poses/releases, preserve catalog and prior."""
    view, metadata = select_trajectory(data, plan, index)
    if (plan.get("boundary_policy")
            != "fully_contained_source_track_inclusive_end"
            or plan.get("release_schedule_sha256")
            != artifact.sha256_file(schedule_path)):
        raise ValueError("episode track policy or release binding differs")
    births = plan["source_track_births"]
    if set(births) != {r.tracklet_id for r in releases}:
        raise ValueError("episode births must cover every released track")
    for release in releases:
        birth = births[release.tracklet_id]
        if type(birth) is not int or not 0 <= birth <= min(m.anchor_keyframe_idx for m in release.measurements):
            raise ValueError("invalid source track birth")
    start = metadata["parent_keyframe_start"]
    end = metadata["parent_keyframe_end_inclusive"]
    kept = [r for r in releases if births[r.tracklet_id] >= start
            and r.release_keyframe_idx <= end]
    selected = tuple(dataclasses.replace(
        r, release_keyframe_idx=r.release_keyframe_idx - start,
        measurements=tuple(msgspec.structs.replace(
            m, anchor_keyframe_idx=m.anchor_keyframe_idx - start)
                           for m in r.measurements)) for r in kept)
    ids = {r.tracklet_id for r in selected}
    view = dataclasses.replace(
        view,
        measurements=sorted([m for r in selected for m in r.measurements],
                            key=lambda m: (m.anchor_keyframe_idx, m.tracklet_id)),
        tables={key: value for key, value in data.tables.items() if key in ids})
    metadata.update({
        "kept_tracklets": len(kept), "parent_tracklets": len(releases),
        "kept_tracklet_ids": sorted(ids),
    })
    return view, selected, metadata


def select_trajectory(data, plan, index):
    """Select and reindex only the trajectory window named by an episode plan."""
    if (plan.get("schema") != SCHEMA
            or not artifact.records_same_artifact(
                plan.get("localization_inputs"), data.artifact_ref)):
        raise ValueError("episode plan identity differs")
    bounds = plan.get("windows")
    if (not isinstance(bounds, list) or not bounds
            or any(not isinstance(pair, list) or len(pair) != 2
                   for pair in bounds)
            or any(type(v) is not int for pair in bounds for v in pair)
            or bounds != [list(b) for b in windows(data.truth, len(bounds))]):
        raise ValueError("episode bounds differ from distance windows")
    if type(index) is not int or not 0 <= index < len(bounds):
        raise ValueError("episode index is outside the plan")
    start, end = bounds[index]
    view = dataclasses.replace(
        data, meta=msgspec.structs.replace(
            data.meta, n_keyframes=end - start + 1),
        truth=[msgspec.structs.replace(p, keyframe_idx=p.keyframe_idx - start)
               for p in data.truth[start:end + 1]],
        odometry=[], measurements=[], tables={})
    metadata = {
        "schema": SCHEMA, "index": index, "count": len(bounds),
        "parent_keyframe_start": start, "parent_keyframe_end_inclusive": end,
        "parent_dataset": data.artifact_ref.dataset,
        "direction": "forward", "boundary_policy": plan.get("boundary_policy"),
        "plan_sha256": artifact.sha256_json(plan),
        "initialization": "fresh_uniform_parent_region_and_fresh_imu_error",
    }
    return view, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--release_schedule", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--artifacts_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = export_ingest.load(args.input_dir)
    releases = release_schedule.load_sidecar(args.release_schedule, data)
    plan = build_plan(data, releases, args.count,
                      args.artifacts_dir or args.input_dir.parents[2],
                      args.release_schedule)
    if args.out.exists():
        parser.error("refusing to overwrite an episode plan")
    artifact.atomic_write_json(args.out, plan)
    print(f"wrote {len(plan['windows'])} windows to {args.out}")


if __name__ == "__main__":
    main()
