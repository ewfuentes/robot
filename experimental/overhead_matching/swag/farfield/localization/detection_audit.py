"""Bind cached detections to the audited source tracks that later replace them."""
import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path

import msgspec

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.localization import export_ingest
from experimental.overhead_matching.swag.farfield.tracking import tracklets

SCHEMA = "farfield_detection_audit_replacement/v1"


def source_members(track):
    """Use the canonical positive-support classes, not overlapping bystanders."""
    members = {track["birth_obs_id"]}
    members.update(s["obs_id"] for r in track["records"]
                   for s in r.get("supports", ())
                   if s.get("class") in tracklets.EVIDENCE_SUPPORT_CLASSES)
    return members


def member_mapping(tracks, detections, tracked_ids, detection_ids):
    by_observation = {}
    raw_suffix = {k.rsplit("#", 1)[-1]: k for k in detection_ids}
    for detection in detections:
        if (detection["status"] != "closed"
                or detection["birth_keyframe"] != detection["last_keyframe"]):
            raise ValueError("provisional source must contain single-frame detections")
        obs = detection["birth_obs_id"]
        if obs in by_observation:
            raise ValueError("a detection observation appears more than once")
        by_observation[obs] = raw_suffix.get(f'T{detection["track_id"]}')
    by_track = {f'T{t["track_id"]}': t for t in tracks}
    if len(by_track) != len(tracks):
        raise ValueError("source track IDs repeat")
    mapping, missing = {}, {}
    for track_id in sorted(tracked_ids):
        members = source_members(by_track[track_id.rsplit("#", 1)[-1]])
        mapping[track_id] = sorted({by_observation[o] for o in members
                                    if by_observation.get(o) is not None})
        missing[track_id] = sorted(o for o in members if by_observation.get(o) is None)
    return mapping, missing


def select(plan, tracked, detections, releases, provisional_releases, start, end):
    if (plan.get("schema") != SCHEMA
            or not artifact.records_same_artifact(plan["tracked_inputs"], tracked.artifact_ref)
            or not artifact.records_same_artifact(plan["detection_inputs"], detections.artifact_ref)):
        raise ValueError("replacement plan does not bind these inputs")
    selected = []
    for r in provisional_releases:
        if (len(r.measurements) != 1
                or r.measurements[0].anchor_keyframe_idx != r.release_keyframe_idx):
            raise ValueError("provisional evidence must arrive at its detection frame")
        if start <= r.release_keyframe_idx <= end:
            selected.append(dataclasses.replace(
                r, release_keyframe_idx=r.release_keyframe_idx - start,
                measurements=(msgspec.structs.replace(
                    r.measurements[0], anchor_keyframe_idx=r.release_keyframe_idx - start),)))
    raw = {r.tracklet_id: r for r in selected}
    removals = {}
    for release in releases:
        ids = set(plan["replacements"][release.tracklet_id]) & raw.keys()
        if any(raw[k].release_keyframe_idx > release.release_keyframe_idx for k in ids):
            raise ValueError("an audit cannot replace a future detection")
        removals[release.tracklet_id] = sorted(ids)
    claims = Counter(k for ids in removals.values() for k in ids)
    return tuple(selected), removals, {
        "schema": SCHEMA,
        "plan_sha256": artifact.sha256_json(plan),
        "policy": "all_detections_then_replace_accepted_source_supports_once",
        "unclaimed_detections": "retained_including_rejected_or_boundary_crossing_tracks",
        "partial_audits": "all_source_supports_replaced_by_audit_valid_track_geometry",
        "cross_track_overlap": "inherited_from_non_deduplicated_track_baseline",
        "n_detections": len(raw), "n_claimed_detections": len(claims),
        "n_shared_detection_claims": sum(n > 1 for n in claims.values()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracked_input", type=Path, required=True)
    p.add_argument("--detection_input", type=Path, required=True)
    p.add_argument("--tracked_episode_plan", type=Path, required=True)
    p.add_argument("--detection_episode_plan", type=Path, required=True)
    p.add_argument("--artifacts_dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError("choose a new replacement plan path")
    data, raw = export_ingest.load(a.tracked_input), export_ingest.load(a.detection_input)
    for name in ("truth.jsonl", "landmarks.json", "motion_source.csv"):
        if artifact.sha256_file(a.tracked_input / name) != artifact.sha256_file(a.detection_input / name):
            raise ValueError(f"paired {name} differs")
    sources = []
    for path, source in ((a.tracked_episode_plan, data), (a.detection_episode_plan, raw)):
        plan = json.loads(path.read_text())
        if not artifact.records_same_artifact(plan["localization_inputs"], source.artifact_ref):
            raise ValueError("episode plan input binding differs")
        ref = plan["source_tracks"]
        directory = a.artifacts_dir / ref["kind"] / ref["dataset"] / ref["version"]
        if not artifact.records_same_artifact(ref, artifact.open_artifact(directory)):
            raise ValueError("source tracking artifact differs")
        sources.append((ref, json.loads((directory / "tracks_full.json").read_text())["tracks"]))
    mapping, missing = member_mapping(sources[0][1], sources[1][1], data.tables, raw.tables)
    result = {
        "schema": SCHEMA, "tracked_inputs": data.artifact_ref.to_dict(),
        "detection_inputs": raw.artifact_ref.to_dict(),
        "source_tracks": [s[0] for s in sources], "replacements": mapping,
        "unrepresented_source_observations": missing,
        "episode_plan_sha256": [artifact.sha256_file(a.tracked_episode_plan),
                                 artifact.sha256_file(a.detection_episode_plan)],
    }
    a.out.write_text(json.dumps(result, indent=2))
    print("wrote", a.out, "tracks", len(mapping))


if __name__ == "__main__":
    main()
