"""Consolidate tracks into landmarks and emit localization-ready bearings.

Reads a tracking run (plus the semantic audit, when present) and writes

  <run_dir>/merged/landmarks.json     merged landmarks + evidence + links
  <run_dir>/merged/pair_stats.json    every co-visibility verdict
  <run_dir>/merged/measurements.json  fused per-epoch bearings per landmark
  <run_dir>/merged/index.html         review page

Merging is decided by geometry alone (track_merge.py); semantics ride along
for the matcher. Bearings are CAMERA-frame azimuths: converting them to the
body frame the localization filter expects is a single fixed mount offset,
which is NOT applied here because it has not been measured for this rig.

Run:
  bazel run //...object_tracking:m6_merge_tracks -- --run_dir <runs>/r002_full_leg1
"""

import argparse
import html
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from experimental.overhead_matching.swag.landmark_filtering import ingest
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    semantic_audit as sa,
    track_merge as tm,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    IngestConfig,
)

DEFAULT_DATASET = Path("/data/farfield_matching/boston_harbor_dataset/processed/leg1")
DEFAULT_LANDMARKS = Path(
    "/data/farfield_matching/boston_harbor_dataset/panorama_landmarks/boston_harbor_leg1")


def load_audits(run_dir: Path) -> dict:
    """track_id -> audit dict, or {} when the audit stage has not run."""
    audit_dir = run_dir / "semantic_audit"
    results, meta_path = audit_dir / "results.jsonl", audit_dir / "audit_meta.json"
    if not (results.exists() and meta_path.exists()):
        return {}
    meta = json.loads(meta_path.read_text())
    audits = {}
    with open(results) as f:
        for line in f:
            if not line.strip():
                continue
            key, audit, _ = sa.parse_result_line(json.loads(line))
            if audit is not None and key in meta:
                audits[meta[key]["track_id"]] = audit
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--dataset_base", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--landmark_base", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--min_supports", type=int, default=2,
                        help="Tracks below this never enter consolidation. "
                             "Matches the audit bar on purpose: a track that "
                             "was never audited has no canonical semantics, "
                             "so it cannot be matched and must not reach the "
                             "filter.")
    parser.add_argument("--epoch_keyframes", type=int, default=5,
                        help="Keyframes fused into one bearing measurement")
    parser.add_argument("--bearing_sigma_deg", type=float, default=1.0)
    args = parser.parse_args()

    artifact = json.loads(
        next(args.run_dir.glob("tracks_*.json")).read_text())
    result = ingest.run_ingest(args.dataset_base, args.landmark_base,
                               IngestConfig())
    obs_by_id = {o.obs_id: o for o in result.observations}
    probe = Image.open(args.dataset_base / "panorama"
                       / f"{result.frames[0].pano_stem}.jpg")
    pano_w = probe.size[0]

    cfg = sa.AuditConfig()
    audits = load_audits(args.run_dir)
    tracks, dossiers, evidences = {}, {}, {}
    for track in artifact["tracks"]:
        if not track["records"]:
            continue
        dossier = sa.build_dossier(track, obs_by_id, cfg)
        if dossier["n_supports"] < args.min_supports:
            continue
        tid = track["track_id"]
        tracks[tid] = track
        dossiers[tid] = dossier
        evidences[tid] = sa.build_evidence(track, dossier, pano_w)

    print(f"{len(tracks)} tracks enter consolidation "
          f"({len(audits)} have audits)")

    landmarks, pair_stats = tm.merge_tracks(
        tracks, dossiers, evidences, audits, pano_w, tm.MergeConfig())
    verdicts = Counter(p.verdict for p in pair_stats)
    multi = [lm for lm in landmarks if len(lm.track_ids) > 1]
    print(f"pair verdicts: {dict(verdicts)}")
    print(f"{len(landmarks)} landmarks from {len(tracks)} tracks "
          f"({len(multi)} merged from 2+ tracks)")

    # Fused bearings per landmark: every member track contributes its own
    # valid-segment bearings, all tagged with the landmark id the filter
    # will use as its tracklet id.
    measurements = []
    for lm in landmarks:
        for tid in lm.track_ids:
            segments = (audits.get(tid) or {}).get("valid_segments")
            series = tm.bearing_series(tracks[tid], pano_w, segments)
            for anchor, az, kappa in tm.fuse_bearings(
                    series, args.epoch_keyframes, args.bearing_sigma_deg):
                measurements.append({
                    "tracklet_id": lm.landmark_id,
                    "source_track_id": tid,
                    "anchor_keyframe_idx": anchor,
                    "bearing_camera_deg": az,
                    "kappa": kappa})
    measurements.sort(key=lambda m: (m["anchor_keyframe_idx"],
                                     m["tracklet_id"]))
    print(f"{len(measurements)} fused bearing measurements "
          f"(epoch={args.epoch_keyframes} keyframes)")

    out = args.run_dir / "merged"
    out.mkdir(exist_ok=True)
    lm_records = [{
        "landmark_id": lm.landmark_id, "track_ids": lm.track_ids,
        "n_supports": lm.n_supports,
        "n_supported_keyframes": lm.n_supported_keyframes,
        "keyframe_span": lm.keyframe_span, "name_votes": lm.name_votes,
        "tag_votes": lm.tag_votes, "name_contested": lm.name_contested,
        "parent_of": lm.parent_of, "child_of": lm.child_of,
        "handoff_proposals": lm.handoff_proposals,
        "review_pairs": lm.review_pairs,
        "merge_conflicts": lm.merge_conflicts} for lm in landmarks]
    (out / "landmarks.json").write_text(json.dumps(lm_records, indent=1))
    (out / "pair_stats.json").write_text(json.dumps([
        {"track_a": p.track_a, "track_b": p.track_b, "verdict": p.verdict,
         "n_covisible": p.n_covisible, "median_iou": p.median_iou,
         "median_sep_deg": p.median_sep_deg,
         "median_containment": p.median_containment,
         "parent": p.parent, "child": p.child,
         "gap_keyframes": p.gap_keyframes}
        for p in pair_stats if p.verdict != tm.DISJOINT], indent=1))
    (out / "measurements.json").write_text(json.dumps(measurements, indent=1))

    n_review = sum(len(lm.review_pairs) for lm in landmarks)
    n_props = sum(len(lm.handoff_proposals) for lm in landmarks)
    n_conf = sum(len(lm.merge_conflicts) for lm in landmarks)
    parts = [
        "<html><head><title>merged landmarks</title><style>",
        "body{font-family:sans-serif;background:#161616;color:#ddd;margin:16px}",
        "a{color:#8bf}table{border-collapse:collapse;margin-top:10px}",
        "td,th{padding:3px 10px;font-size:13px;border-bottom:1px solid #333;",
        "text-align:left;vertical-align:top}",
        "th{color:#89a}.merged{color:#3c3}.contested{color:#fa2}",
        "</style></head><body><h1>merged landmarks</h1>",
        f"<p>{len(landmarks)} landmarks from {len(tracks)} tracks | "
        f"<span class='merged'>{len(multi)} merged from 2+ tracks</span> | "
        f"<b>{n_review} ambiguous pairs needing adjudication</b> | "
        f"{n_props} handoff proposals (not merged) | "
        f"{n_conf} merge conflicts</p>",
        "<p>pair verdicts: " + ", ".join(
            f"{k} {v}" for k, v in verdicts.most_common()) + "</p>",
        "<p>Merging uses geometry only: tracks co-visible at a shared "
        "keyframe with separated masks cannot be one object, whatever their "
        "names agree on. Handoff candidates (never co-visible) are listed "
        "but never auto-merged.</p>",
        "<table><tr><th>landmark</th><th>tracks</th><th>supports</th>"
        "<th>keyframes</th><th>top tags</th><th>names</th><th>links</th>"
        "</tr>"]
    for lm in sorted(landmarks, key=lambda l: -l.n_supports):
        cls = " class='merged'" if len(lm.track_ids) > 1 else ""
        names = ", ".join(f"{html.escape(n)} x{c}"
                          for n, c in list(lm.name_votes.items())[:3])
        if lm.name_contested:
            names = f"<span class='contested'>{names} (contested)</span>"
        tags = ", ".join(f"{html.escape(t)} x{c}"
                         for t, c in list(lm.tag_votes.items())[:3])
        links = []
        if lm.parent_of:
            links.append(f"parent of {len(lm.parent_of)}")
        if lm.child_of:
            links.append(f"child of {len(lm.child_of)}")
        if lm.review_pairs:
            links.append(f"<b>{len(lm.review_pairs)} review</b>")
        if lm.handoff_proposals:
            links.append(f"{len(lm.handoff_proposals)} handoff?")
        if lm.merge_conflicts:
            links.append(f"<b>{len(lm.merge_conflicts)} conflict</b>")
        track_links = " ".join(
            f"<a href='../track_full_leg1_T{t}.html'>T{t}</a>"
            for t in lm.track_ids)
        parts.append(
            f"<tr{cls}><td>{html.escape(lm.landmark_id)}</td>"
            f"<td>{track_links}</td><td>{lm.n_supports}</td>"
            f"<td>{lm.keyframe_span[0]}-{lm.keyframe_span[1]}</td>"
            f"<td>{tags}</td><td>{names}</td>"
            f"<td>{', '.join(links)}</td></tr>")
    parts.append("</table></body></html>")
    (out / "index.html").write_text("\n".join(parts))
    print(f"wrote {out}/index.html")


if __name__ == "__main__":
    main()
