"""Pairing stage: merged landmarks x wedge candidates -> Gemini label requests.

Mirrors LOCI's `landmark_pairing_cli` exactly - same system prompt, same
hard/easy negative definitions, same uniqueness 1-5 score, same JSON schema,
same "Set 1 / Set 2" prompt body - and changes only how Set 2 is gathered.
LOCI takes the OSM landmarks on the satellite tiles covering a panorama; we
take the ones in the **bearing wedge**, intersected across every observation
of the tracklet.

Set 1 is our merged landmarks (one entry per tracklet, so the model sees the
whole leg's inventory at once, mirroring "all landmarks in this panorama").

Outputs under <run_dir>/pairing/:
  requests.jsonl   Vertex batch requests (submit with vertex_batch_manager)
  prompts/*.txt    the exact filled-in prompt per request
  figures/*.png    map of the trajectory, wedges and candidates
  index.html       review page

Run:
  bazel run //...object_tracking:m7_build_pairing_requests -- \\
      --run_dir <runs>/r003_full_leg1 --max_tracklets 6
"""

import argparse
import csv
import html
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (  # noqa: E402
    bearing_matcher as bm,
    harbor_catalog as hc,
)

DEFAULT_DATASET = Path("/data/farfield_matching/boston_harbor_dataset/processed/leg1")
DEFAULT_FEATHER = Path(
    "/data/farfield_matching/boston_harbor_dataset/landmarks/harbor_osm_enc_v1.feather")

# Camera->body yaw, calibrated MAP-FREE by sweeping the offset and minimising
# the median triangulation residual over 26 well-conditioned tracklets
# (smooth unimodal curve: 5.95 deg at 180, 1.33 deg at 214, 4.34 deg at 270).
# A wrong offset rotates every bearing by a constant, which stops the rays of
# a static object from intersecting - so the residual identifies the offset
# without any map, any assumed match, or any hand-read image.
#
# The 202.5 deg bow estimate was ~12 deg off: the deckhouse occludes the bow,
# so it could not be read directly. Independent corroboration: the wake in the
# temporal-median image implies ~218 deg, and LT20 matched to One
# International Place implies 222 deg.
DEFAULT_MOUNT_OFFSET_DEG = 214.0

# Set 2 size. 60 was an arbitrary carry-over from "a tile's worth of
# landmarks" and it cost recall: a Conley Terminal container crane, the
# correct answer for one tracklet, sat outside it. At ~25 tokens per entry,
# 500 entries is ~12 k tokens per request - the same order as the per-track
# audit already costs, and trivial against the leg's budget. Recall matters
# more than brevity here: a candidate that is not in Set 2 can never be
# labelled, and every missing true match is a training example we never get.
DEFAULT_MAX_SET2 = 500

# Structural classes a distant observer can actually resolve. Used ONLY to
# order Set 2 within the prompt's display budget - nothing is removed from
# the catalog, which stays complete for the filter's own spatial gating. The
# first run put benches, waste baskets and toilets in the visible 60 while
# real candidates fell off the end; that is a presentation bug, not a reason
# to filter the map.
SALIENT_KEYS = ("seamark:type", "object_class", "man_made", "historic",
                "place", "natural", "landuse", "bridge", "power", "aeroway",
                "leisure", "tourism", "waterway", "height")


# Structural classes that are conspicuous at range regardless of whether OSM
# bothered to tag a height. Container cranes are 50-80 m tall and visible
# across the harbor, but carry only `man_made=crane`.
TALL_STRUCTURE_VALUES = frozenset({
    "crane", "lighthouse", "water_tower", "storage_tank", "chimney", "tower",
    "mast", "silo", "gasometer", "windmill", "obelisk", "bridge",
})


def set2_salience(cand) -> float:
    """How likely a distant observer is to pick this candidate out.

    A single score rather than lexicographic tiers. The tiered version kept
    producing cliff effects: first it buried a 46-storey tower under a named
    bus stop, then - once prominence gated the tier - it dumped every
    untagged-height container crane into the bottom bucket with the benches.
    A sum degrades gracefully where a tier ordering falls off a cliff.

    Ordering only. Nothing is removed from the catalog, and this deliberately
    ignores Set 1: ranking Set 2 by similarity to the query would bias the
    labels toward whatever the baseline scorer already believes.
    """
    tags = cand.entry.tags
    score = 0.0
    if cand.entry.source == "enc":
        score += 3.0
    if "conspicuous" in tags.get("description", "").lower():
        score += 3.0
    if any(v in TALL_STRUCTURE_VALUES for v in tags.values()):
        score += 2.5
    if any(k in tags for k in SALIENT_KEYS):
        score += 1.0
    if "name" in tags:
        score += 1.5
    try:
        levels = float(str(tags.get("building:levels", "0")).split(";")[0])
    except ValueError:
        levels = 0.0
    try:
        height = float(str(tags.get("height", "0")).rstrip("m").split(";")[0])
    except ValueError:
        height = 0.0
    score += min(3.0, max(height, levels * 3.5) / 60.0)
    return score


def set2_rank(cand):
    """Sort key for prompt display: salience first, then wedge agreement."""
    return (-set2_salience(cand), -cand.support_frac,
            cand.median_abs_residual_deg)

SYSTEM_PROMPT = """You are a landmark matching expert. Given two sets of OpenStreetMap-style tag bundles,
identify which landmarks from Set 1 (extracted from imagery captured from a boat) represent the same physical object as a
landmark in Set 2 (from an OpenStreetMap / nautical chart database). Both sets use key=value tag notation.

For each Set 1 landmark that has a match, rate the uniqueness of that landmark's tag set (1-5).
The score describes how distinctive the observed landmark is on its own - NOT the quality
of the match or how similar the two sides are.
  1 = extremely generic (e.g., building=yes)
  2 = common category (e.g., man_made=pier)
  3 = moderately specific (e.g., man_made=water_tower)
  4 = quite distinctive (e.g., historic=fort; name=Fort Independence)
  5 = highly unique/unmistakable (e.g., man_made=lighthouse; name=Boston Light)

For each Set 1 landmark, also provide 0-2 negative examples from Set 2 - landmarks that
are NOT a match. Label each as "hard" or "easy":
  - hard: same general category but UNAMBIGUOUSLY a different landmark. Valid reasons:
      * Names that refer to different entities (e.g., "Deer Island Light" vs "Long Island Head Light")
      * Vastly different scale (e.g., a 60 m standpipe vs a 3 m daybeacon)
    Do NOT treat these as conflicts - they do not make valid hard negatives:
      * Small numeric differences (height=40 vs 45) - the extractor is often off
      * Different tag specificity for the same thing (man_made=tower vs man_made=water_tower,
        place=island vs natural=coastline)
      * One name being a substring/variant of another (Tobin Bridge vs Maurice J. Tobin Memorial Bridge)
      * Features that could be spatially contained (a fort on an island; a light on a pier)
      * Missing tags on one side
    When in doubt, do NOT include it as a negative.
  - easy: obviously unrelated (completely different type, e.g., a lighthouse vs a parking lot).

For every match, also say what KIND of match it is:
  - "instance": this is that exact physical object. Something identifies it uniquely -
    a matching name, or a combination of tags no other nearby candidate shares.
  - "category": the right kind of object, but the tags cannot tell you WHICH one.
    Matching "man_made=tower" against a charted tower is a category match when several
    charted towers sit along the same direction.
Both are useful and both should be reported. They are not degrees of confidence: a
category match can be certain and still fail to identify which object it is.

Set 2 is restricted to landmarks lying along the direction the Set 1 landmark was observed in,
so a plausible match is usually present - but not always. Only propose matches you are confident
about. Some landmarks have no match."""

JSON_SCHEMA = {
    "type": "object",
    "required": ["matches"],
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["set_1_id", "set_2_matches", "uniqueness_score",
                             "negatives"],
                "properties": {
                    "set_1_id": {"type": "integer"},
                    "set_2_matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["set_2_id", "match_type"],
                            "properties": {
                                "set_2_id": {"type": "integer"},
                                "match_type": {
                                    "type": "string",
                                    "enum": ["instance", "category"]},
                            },
                        },
                    },
                    "uniqueness_score": {"type": "integer"},
                    "negatives": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["set_2_id", "difficulty"],
                            "properties": {
                                "set_2_id": {"type": "integer"},
                                "difficulty": {"type": "string",
                                               "enum": ["hard", "easy"]},
                            },
                        },
                    },
                },
            },
        }
    },
}


def format_tags(tags) -> str:
    """'key=value; key=value' - LOCI's format_tags."""
    return "; ".join(f"{k}={v}" for k, v in tags)


def itemized_list(items) -> str:
    return "\n".join(f" {i}. {v}" for i, v in enumerate(items))


def load_poses(dataset_base: Path, anchor_lat, anchor_lon):
    """keyframe -> (east_m, north_m, course_deg, speed_mps).

    frames_gps.csv carries no course column, so course comes from the GPS
    delta to the next fix. It is only meaningful when moving: below the speed
    gate the heading is left as None rather than reporting the direction of
    GPS jitter.
    """
    rows = list(csv.DictReader(open(dataset_base / "frames_gps.csv")))
    lat = np.array([float(r["latitude"]) for r in rows])
    lon = np.array([float(r["longitude"]) for r in rows])
    east, north = hc.enu_from_latlon(lat, lon, anchor_lat, anchor_lon)
    poses = {}
    for i, row in enumerate(rows):
        j = min(i + 1, len(rows) - 1)
        k = max(i - 1, 0)
        d_east = float(east[j] - east[k])
        d_north = float(north[j] - north[k])
        step = math.hypot(d_east, d_north)
        course = (math.degrees(math.atan2(d_east, d_north)) % 360.0
                  if step > 1.0 else None)
        poses[int(row["idx"])] = (float(east[i]), float(north[i]), course,
                                  float(row["speed_mps"]))
    return poses


def observations_for(tracklet_id, measurements, poses, mount_offset_deg):
    """Fused bearings -> world-frame Observations at known poses."""
    out = []
    for m in measurements:
        if m["tracklet_id"] != tracklet_id:
            continue
        pose = poses.get(m["anchor_keyframe_idx"])
        if pose is None or pose[2] is None:
            continue
        east, north, course, _ = pose
        body = (m["bearing_camera_deg"] - mount_offset_deg) % 360.0
        # Half-width from the fused concentration: kappa = 1/sigma^2.
        sigma_deg = math.degrees(1.0 / math.sqrt(max(m["kappa"], 1e-9)))
        out.append(bm.Observation(
            anchor_keyframe_idx=m["anchor_keyframe_idx"],
            east_m=east, north_m=north,
            bearing_world_deg=(course + body) % 360.0,
            half_width_deg=min(sigma_deg, 20.0)))
    return out


def bearing_quality(observations):
    """(median_residual_deg, condition_number, range_m) via triangulation.

    Replaces an earlier "circular std of the world bearing" metric, which was
    simply wrong: a static object 700 m away sweeps ~74 deg of genuine
    bearing as the vessel passes it, so spread measured parallax, not error.
    Tracklets it called failures were the best-conditioned ones in the run.

    Residual asks the right question - are these bearings consistent with a
    single static point - and the condition number guards the other end: a
    short observing arc yields a tiny residual while leaving position along
    the line of sight almost free.
    """
    result = bm.triangulate(observations)
    if result is None:
        return float("nan"), float("nan"), float("nan")
    east, north, residual, condition = result
    range_m = math.hypot(east - observations[0].east_m,
                         north - observations[0].north_m)
    return residual, condition, range_m


def landmark_tag_list(landmark, audit):
    """Set 1 tag bundle for a merged landmark: audited tags + names."""
    tags = []
    if audit:
        for t in audit["primary_object"]["tags"]:
            key, _, value = t["tag"].partition("=")
            if key and value:
                tags.append((key, value))
        for c in audit["primary_object"].get("name_candidates", []):
            if c.get("name") and c.get("weight", 0) >= 0.5:
                tags.append(("name", c["name"]))
        extent = audit["primary_object"].get("extent")
        if extent:
            tags.append(("extent", extent))
    if not tags:
        for tag in list(landmark["tag_votes"])[:4]:
            key, _, value = tag.partition("=")
            if key and value:
                tags.append((key, value))
    seen, unique = set(), []
    for pair in tags:
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)
    return unique


def render_figure(path, poses, observations, candidates, matched_ids,
                  tracklet_id, label):
    """Trajectory + per-observation wedges + candidates pulled out."""
    fig, ax = plt.subplots(figsize=(9, 8))
    track_e = [p[0] for p in poses.values()]
    track_n = [p[1] for p in poses.values()]
    ax.plot(track_e, track_n, color="#888", lw=1.0, zorder=1,
            label="vessel track")

    ray_len = 20000.0
    for i, obs in enumerate(observations):
        half = 6.0 + obs.half_width_deg
        edges = []
        for sign in (-1, 1):
            edge = obs.bearing_world_deg + sign * half
            edges.append((obs.east_m + ray_len * math.sin(math.radians(edge)),
                          obs.north_m + ray_len * math.cos(math.radians(edge))))
        ax.fill([obs.east_m, edges[0][0], edges[1][0]],
                [obs.north_m, edges[0][1], edges[1][1]],
                color="#3a86ff", alpha=0.06, zorder=2,
                label="wedge" if i == 0 else None)
        ax.plot([obs.east_m,
                 obs.east_m + ray_len * math.sin(
                     math.radians(obs.bearing_world_deg))],
                [obs.north_m,
                 obs.north_m + ray_len * math.cos(
                     math.radians(obs.bearing_world_deg))],
                color="#3a86ff", lw=0.5, alpha=0.30, zorder=2)
        ax.plot(obs.east_m, obs.north_m, "o", ms=3, color="#3a86ff", zorder=3)

    if candidates:
        ce = [c.entry.east_m for c in candidates]
        cn = [c.entry.north_m for c in candidates]
        ax.scatter(ce, cn, s=14, color="#ffb703", edgecolor="#7a5200",
                   linewidth=0.3, zorder=4,
                   label=f"wedge candidates ({len(candidates)})")
        for c in candidates:
            if c.entry.landmark_id in matched_ids:
                ax.scatter([c.entry.east_m], [c.entry.north_m], s=120,
                           facecolor="none", edgecolor="#e63946", lw=1.8,
                           zorder=5)
                name = c.entry.tags.get("name") or c.entry.tags.get(
                    "man_made") or c.entry.source
                ax.annotate(name, (c.entry.east_m, c.entry.north_m),
                            fontsize=7, color="#e63946",
                            xytext=(4, 4), textcoords="offset points")

    focus_e = [o.east_m for o in observations] + [
        c.entry.east_m for c in candidates]
    focus_n = [o.north_m for o in observations] + [
        c.entry.north_m for c in candidates]
    if focus_e:
        pad = 0.15 * max(1000.0, max(max(focus_e) - min(focus_e),
                                     max(focus_n) - min(focus_n)))
        ax.set_xlim(min(focus_e) - pad, max(focus_e) + pad)
        ax.set_ylim(min(focus_n) - pad, max(focus_n) + pad)

    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title(f"{tracklet_id}: {label}", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--dataset_base", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--feather", type=Path, default=DEFAULT_FEATHER)
    parser.add_argument("--mount_offset_deg", type=float,
                        default=DEFAULT_MOUNT_OFFSET_DEG)
    parser.add_argument("--max_tracklets", type=int, default=None)
    parser.add_argument("--min_supports", type=int, default=5)
    parser.add_argument("--bearing_slack_deg", type=float, default=25.0,
                        help="Wedge half-width added to the tracklet's own "
                             "angular width. Default is wide because the "
                             "mount offset is only known to ~20 deg and the "
                             "GPS-course heading reference drifts; tighten it "
                             "once the offset is calibrated from matches.")
    parser.add_argument("--min_observations", type=int, default=1,
                        help="Fused bearings required to build a request. "
                             "Triangulation quality is reported per tracklet "
                             "but no longer gates: the mount offset is a rig "
                             "constant, so bearings are valid with or without "
                             "an intersecting geometry. NOTE the real risk for "
                             "a 1-observation tracklet is an unbounded single "
                             "ray sweeping tens of thousands of candidates - "
                             "bound it with the detector's distance_estimate "
                             "rather than by refusing the tracklet.")
    parser.add_argument("--max_set2", type=int, default=DEFAULT_MAX_SET2,
                        help="Candidates shown per tracklet in the prompt")
    parser.add_argument("--thinking_level", default="LOW")
    parser.add_argument("--max_bearing_residual_deg", type=float,
                        default=None,
                        help="Skip tracklets whose bearings are this "
                             "inconsistent with any single static point "
                             "(see bearing_quality)")
    args = parser.parse_args()

    merged = args.run_dir / "merged"
    landmarks = json.loads((merged / "landmarks.json").read_text())
    measurements = json.loads((merged / "measurements.json").read_text())
    audits = {}
    meta_path = args.run_dir / "semantic_audit" / "audit_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        track_to_key = {v["track_id"]: k for k, v in meta.items()}
        from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (  # noqa: E501
            semantic_audit as sa)
        raw = {}
        with open(args.run_dir / "semantic_audit" / "results.jsonl") as f:
            for line in f:
                if line.strip():
                    key, audit, _ = sa.parse_result_line(json.loads(line))
                    if audit:
                        raw[key] = audit
        for lm in landmarks:
            for tid in lm["track_ids"]:
                key = track_to_key.get(tid)
                if key in raw:
                    audits[lm["landmark_id"]] = raw[key]
                    break

    rows = list(csv.DictReader(open(args.dataset_base / "frames_gps.csv")))
    anchor_lat = sum(float(r["latitude"]) for r in rows) / len(rows)
    anchor_lon = sum(float(r["longitude"]) for r in rows) / len(rows)
    print(f"anchor {anchor_lat:.5f},{anchor_lon:.5f}")
    poses = load_poses(args.dataset_base, anchor_lat, anchor_lon)
    entries = hc.load_catalog_cached(args.feather, anchor_lat, anchor_lon)
    print(f"catalog {len(entries)} entries; poses {len(poses)}")

    chosen = [lm for lm in landmarks if lm["n_supports"] >= args.min_supports]
    chosen.sort(key=lambda lm: -lm["n_supports"])
    if args.max_tracklets:
        chosen = chosen[:args.max_tracklets]
    print(f"{len(chosen)} tracklets (>= {args.min_supports} supports)")

    out = args.run_dir / "pairing"
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cfg = bm.WedgeConfig(bearing_slack_deg=args.bearing_slack_deg)
    scorer = bm.TagRuleScorer()

    records, page_rows = [], []
    for lm in chosen:
        tid = lm["landmark_id"]
        observations = observations_for(tid, measurements, poses,
                                        args.mount_offset_deg)
        if not observations:
            print(f"  {tid}: no usable observations (speed-gated), skipping")
            continue
        if len(observations) < args.min_observations:
            print(f"  {tid}: only {len(observations)} fused observation(s) "
                  f"(< {args.min_observations}) - skipping")
            continue
        # Triangulation is an ANNOTATION, not an entry requirement. It earned
        # its place calibrating the mount offset, which is a rig constant: once
        # known it applies to every track, so a track that cannot triangulate
        # still has correctly-referenced bearings - it just cannot self-verify.
        # Gating on it previously discarded 29 labellable tracklets, which is
        # also not how LOCI works (it labels from tag bundles inside a coarse
        # spatial gate, triangulating nothing).
        consistency, condition, tri_range = bearing_quality(observations)
        if (args.max_bearing_residual_deg is not None
                and not math.isnan(consistency)
                and consistency > args.max_bearing_residual_deg):
            print(f"  {tid}: triangulation residual {consistency:.1f} deg "
                  "exceeds gate, skipping")
            continue
        all_candidates = bm.gather_candidates(entries, observations, cfg)
        candidates = sorted(all_candidates, key=set2_rank)[:args.max_set2]
        audit = audits.get(tid)
        set1 = landmark_tag_list(lm, audit)
        scores = scorer.score_candidates(
            {"tags": (audit or {}).get("primary_object", {}).get("tags", []),
             "name_candidates": (audit or {}).get(
                 "primary_object", {}).get("name_candidates", [])},
            candidates)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        matched_ids = {lid for lid, s in top if s > 0}

        prompt = (
            f"Set 1 (observed from the vessel):\n"
            f"{itemized_list([format_tags(set1)])}\n\n"
            f"Set 2 (map database, along the observed bearing):\n"
            f"{itemized_list(format_tags(sorted(c.entry.tags.items())) for c in candidates)}")
        (out / "prompts" / f"{tid}.txt").write_text(
            f"=== SYSTEM ===\n{SYSTEM_PROMPT}\n\n=== USER ===\n{prompt}\n")

        fig_name = f"{tid}.png"
        label = (", ".join(f"{k}={v}" for k, v in set1[:3])
                 + ("  |  not triangulable"
                    if math.isnan(consistency) else
                    f"  |  resid {consistency:.2f}deg, cond {condition:.0f}, "
                    f"range {tri_range:.0f}m"))
        render_figure(out / "figures" / fig_name, poses, observations,
                      candidates, matched_ids, tid, label)

        records.append({
            "key": tid,
            "request": {
                "contents": [{"parts": [{"text": prompt}], "role": "user"}],
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": JSON_SCHEMA,
                    "thinkingConfig": {"thinkingLevel": args.thinking_level},
                },
            },
        })
        page_rows.append((lm, set1, observations, candidates, top, fig_name,
                          prompt, consistency))
        print(f"  {tid}: resid {consistency:5.2f}deg cond {condition:6.0f} "
              f"range {tri_range:6.0f}m | "
              f"{len(observations)} obs -> {len(all_candidates)} in wedge, "
              f"{len(candidates)} shown; "
              f"candidates; top={[f'{i.split(chr(58))[-1]}:{s:.1f}' for i, s in top]}")

    with open(out / "requests.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    parts = [
        "<html><head><title>pairing wedges</title><style>",
        "body{font-family:sans-serif;background:#161616;color:#ddd;margin:16px}",
        "img{max-width:900px;border-radius:6px;background:#fff}",
        "pre{background:#1d1d1d;padding:10px;border-radius:6px;font-size:12px;",
        "white-space:pre-wrap;max-width:1100px}",
        "h2{margin-top:40px;border-top:1px solid #333;padding-top:16px}",
        "table{border-collapse:collapse}td,th{padding:2px 10px;font-size:12px;",
        "border-bottom:1px solid #333;text-align:left}",
        "</style></head><body><h1>Set 2 by bearing wedge</h1>",
        f"<p>{len(records)} tracklets | mount offset "
        f"{args.mount_offset_deg}&deg; (bootstrap) | catalog "
        f"{len(entries)} entries, unfiltered</p>"]
    page_rows.sort(key=lambda row: (math.isnan(row[7]), row[7]))
    for (lm, set1, observations, candidates, top, fig_name, prompt,
         consistency) in page_rows:
        parts.append(f"<h2>{html.escape(lm['landmark_id'])}</h2>")
        parts.append(f"<p>Set 1: <code>{html.escape(format_tags(set1))}</code>"
                     f" | {len(observations)} observations | "
                     f"{len(candidates)} candidates in wedge | "
                     + ("not triangulable (single ray)</p>"
                        if math.isnan(consistency) else
                        f"triangulation residual <b>{consistency:.2f}&deg;</b> "
                        f"(cond {condition:.0f}, range {tri_range:.0f} m)</p>"))
        parts.append(f"<img src='figures/{fig_name}'>")
        parts.append("<table><tr><th>rank</th><th>landmark</th>"
                     "<th>score</th><th>tags</th></tr>")
        by_id = {c.entry.landmark_id: c for c in candidates}
        for rank, (lid, score) in enumerate(top, 1):
            cand = by_id[lid]
            parts.append(
                f"<tr><td>{rank}</td><td>{html.escape(lid)}</td>"
                f"<td>{score:.2f}</td>"
                f"<td>{html.escape(format_tags(sorted(cand.entry.tags.items())))}"
                "</td></tr>")
        parts.append("</table>")
        parts.append("<details><summary>filled prompt</summary>"
                     f"<pre>{html.escape(prompt)}</pre></details>")
    parts.append("</body></html>")
    (out / "index.html").write_text("\n".join(parts))
    print(f"wrote {out}/index.html and {len(records)} requests")


if __name__ == "__main__":
    main()
