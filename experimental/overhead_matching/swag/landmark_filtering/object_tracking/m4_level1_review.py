"""Level-1 (within-track evidence hygiene) manual review site.

Pulls every supporting detection for ~20 stratified tracks from a tracking
run, renders each support as an image chip (keyframe pano crop around the
detection box), and annotates it with a verdict from candidate Level-1
rules - include / flag / exclude, with the rule that fired. The page is the
working document for deciding which rules suffice and where (if anywhere)
an LLM adjudicator is needed.

Support classes are RECOMPUTED from the recorded metrics with the current
classifier (track_builder.classify_support), so this page always reflects
the present rules even on artifacts from older runs. Where the effective
class differs from the recorded one, both are shown.

Candidate rules (also rendered on the page):
  R1 tag-mismatch-weak: primary tag differs from the track's modal tag AND
     the support is weak with marginal overlap -> exclude.
  R2 tag-mismatch: primary tag differs from modal but overlap is solid ->
     flag (real tag variance vs contamination - review).
  R3 name-conflict: non-empty name differs from the track's modal name ->
     flag (LLM adjudication candidate).
  R5 drift-tail: support within the last keyframes of a drift_alarm track
     -> flag (geometry suspect; the mask may already have slid).
  R6 reclassified-none: the current classifier's mutual-agreement floors
     reject the overlap entirely (T185 occluder case) -> exclude.
  R7 context: contained-but-incoherent (mask fills <10% of a containing
     box; T172 island-theft case) -> not support; kept as Level-2
     merge/occlusion evidence. Replaces old R4, whose "giant boxes are
     often correct" reading was an artifact of chips not drawing the mask.
  Everything else -> include.

Chips draw the detection box (verdict color) AND the track's mask bbox at
that keyframe (red) so speck-in-giant-box relations are visible.

Run:
  bazel run //...object_tracking:m4_level1_review -- --run_dir <runs>/r002_full_leg1
"""

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experimental.overhead_matching.swag.data import farfield_paths
from experimental.overhead_matching.swag.landmark_filtering import ingest
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    pano_geometry as pg,
    track_builder as tb,
    viz_common as vc,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    IngestConfig,
)


CHIP_H = 170
DRIFT_TAIL_KF = 5

VERDICT_CSS = {"include": "#3c3", "flag": "#fa2", "exclude": "#e55",
               "context": "#99b"}
CLASSIFIER_CFG = tb.TrackBuilderConfig()


def effective_class(s) -> str:
    """Recompute the support class from recorded metrics with the current
    classifier, so old artifacts are reviewed under present rules."""
    return tb.classify_support(
        {"iou": s["iou"], "inter_over_mask": s["inter_over_mask"],
         "inter_over_box": s["inter_over_box"]}, CLASSIFIER_CFG)


def obs_tags(obs):
    tags = dict(tuple(t) for t in obs.additional_tags)
    return f"{obs.primary_tag_key}={obs.primary_tag_value}", tags.get("name", "")


def recompute_votes(track, obs_by_id):
    tag_votes, name_votes = Counter(), Counter()
    for rec in track["records"]:
        for s in rec.get("supports", []):
            if effective_class(s) not in tb.SUPPORT_CLASSES:
                continue
            obs = obs_by_id.get(s["obs_id"])
            if obs is None:
                continue
            tag, name = obs_tags(obs)
            tag_votes[tag] += 1
            if name:
                name_votes[name] += 1
    return tag_votes, name_votes


def judge(s, eff, obs, modal_tag, modal_name, rec_keyframe, track):
    """(verdict, rule, reason) for one supporting detection, judged on its
    effective (recomputed) class."""
    tag, name = obs_tags(obs)
    is_drift = track["close_reason"] == "drift_alarm"
    tail_start = (track["end_keyframe"] or 0) - DRIFT_TAIL_KF
    if eff == "none":
        return ("exclude", "R6",
                f"reclassified 'none' by current classifier: iou "
                f"{s['iou']:.2f}, i/m={s['inter_over_mask']:.2f}, "
                f"i/b={s['inter_over_box']:.2f} fail the mutual-agreement "
                "floors (occluder/neighbor box grazing the mask)")
    if eff == "context":
        return ("context", "R7",
                f"contained but incoherent: mask fills only "
                f"{s['inter_over_box']:.2f} of the box - likely a "
                "different, larger object (T172 island-theft pattern); "
                "not support, kept for Level-2 merge/occlusion evidence")
    s = dict(s, **{"class": eff})
    if tag != modal_tag:
        if s["class"] == "weak" and s["iou"] < 0.25:
            return ("exclude", "R1",
                    f"tag {tag} != modal {modal_tag} and overlap is "
                    f"marginal (iou {s['iou']:.2f})")
        return ("flag", "R2",
                f"tag {tag} != modal {modal_tag} but overlap is solid - "
                "tag variance or contamination?")
    if name and modal_name and name != modal_name:
        return ("flag", "R3",
                f"name '{name}' conflicts with modal '{modal_name}' - "
                "LLM adjudication candidate")
    if is_drift and rec_keyframe >= tail_start:
        return ("flag", "R5",
                "inside the pre-drift-alarm window; the mask may already "
                "have slid off the object")
    return ("include", "-", "")


def select_tracks(tracks, obs_by_id, n=20):
    tracks = [t for t in tracks if t["n_supported_keyframes"] >= 1]
    picks = {}

    def add(t, why):
        if t["track_id"] not in picks and len(picks) < n:
            picks[t["track_id"]] = (t, why)

    by_sup = sorted(tracks, key=lambda t: -t["n_supported_keyframes"])
    for t in by_sup[:6]:
        add(t, "top by support")
    for t in by_sup:
        _, names = recompute_votes(t, obs_by_id)
        if len(names) >= 2:
            add(t, "multiple names in votes")
    for reason in ("drift_alarm", "mask_dead", "starved"):
        found = 0
        for t in by_sup:
            if t["close_reason"] == reason and 3 <= t["n_supported_keyframes"] <= 15:
                add(t, f"mid-tier {reason}")
                found += 1
                if found == 2:
                    break
    for t in by_sup:
        if t["status"] == "alive":
            add(t, "alive at leg end")
            break
    low = [t for t in by_sup if 1 <= t["n_supported_keyframes"] <= 2]
    for t in low[:2]:
        add(t, "low-support (1-2 kf)")
    for t in by_sup:
        add(t, "fill by support")
    return list(picks.values())


def render_chip(pano, pano_box, mask_pano_box, out_path, color):
    """Crop around the union of detection box and (if known) mask bbox;
    detection box in the verdict color, mask bbox in red."""
    pano_w = pano.shape[1]
    x0, y0, x1, y1 = pano_box
    ux0, uy0, ux1, uy1 = x0, y0, x1, y1
    mask_rel = None
    if mask_pano_box is not None:
        # Re-anchor the mask bbox near the detection box across the wrap.
        dx = pg.signed_x_offset(mask_pano_box[0], x0, pano_w)
        mxa = x0 + dx
        mask_rel = (mxa, mask_pano_box[1],
                    mxa + (mask_pano_box[2] - mask_pano_box[0]),
                    mask_pano_box[3])
        ux0, uy0 = min(ux0, mask_rel[0]), min(uy0, mask_rel[1])
        ux1, uy1 = max(ux1, mask_rel[2]), max(uy1, mask_rel[3])
    w, h = ux1 - ux0, uy1 - uy0
    mx, my = max(30, 0.25 * w), max(30, 0.25 * h)
    cw, ch = int(w + 2 * mx), int(h + 2 * my)
    crop, cy0 = pg.extract_window(pano, ux0 - mx, uy0 - my, cw, ch)
    img = Image.fromarray(crop)
    draw = ImageDraw.Draw(img)
    line_w = max(2, int(ch / 120))
    cx0 = pg.signed_x_offset(ux0, ux0 - mx, pano_w)  # = mx, wrap-safe
    draw.rectangle([cx0 + (x0 - ux0), y0 - cy0,
                    cx0 + (x1 - ux0), y1 - cy0],
                   outline=color, width=line_w)
    if mask_rel is not None:
        draw.rectangle([cx0 + (mask_rel[0] - ux0), mask_rel[1] - cy0,
                        cx0 + (mask_rel[2] - ux0), mask_rel[3] - cy0],
                       outline=(255, 60, 60), width=line_w)
    scale = CHIP_H / img.height
    img = img.resize((max(60, int(img.width * scale)), CHIP_H),
                     Image.BILINEAR)
    img.save(out_path, quality=85)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    farfield_paths.add_arguments(parser)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--n_tracks", type=int, default=20)
    parser.add_argument("--findings", type=Path, default=None,
                        help="JSON: {'global': str, 'tracks': {track_id: str}}"
                             " commentary merged into the page")
    args = parser.parse_args()
    paths = farfield_paths.resolve(
        parser, args, infer_from=args.run_dir,
        require=("dataset_base", "frame_landmarks"))

    artifact = json.loads(
        next(args.run_dir.glob("tracks_*.json")).read_text())
    range_name = artifact["range"]["name"]
    result = ingest.run_ingest(paths.dataset_base, paths.frame_landmarks,
                               IngestConfig())
    obs_by_id = {o.obs_id: o for o in result.observations}
    frames_by_idx = {f.frame_idx: f for f in result.frames}
    probe = Image.open(
        paths.dataset_base / "panorama" / f"{result.frames[0].pano_stem}.jpg")
    pano_w, pano_h = probe.size

    findings = {"global": "", "tracks": {}}
    if args.findings and args.findings.exists():
        findings.update(json.loads(args.findings.read_text()))

    picked = select_tracks([t for t in artifact["tracks"] if t["records"]],
                           obs_by_id, args.n_tracks)
    out = args.run_dir / "level1_review"
    (out / "chips").mkdir(parents=True, exist_ok=True)

    # Chip tasks grouped by keyframe so each pano decodes once.
    tasks = defaultdict(list)  # keyframe -> [(track, rec, support, verdict)]
    track_infos = {}
    stats = Counter()
    for t, why in picked:
        tag_votes, name_votes = recompute_votes(t, obs_by_id)
        modal_tag = tag_votes.most_common(1)[0][0] if tag_votes else "?"
        modal_name = name_votes.most_common(1)[0][0] if name_votes else ""
        track_infos[t["track_id"]] = {
            "track": t, "why": why, "tag_votes": tag_votes,
            "name_votes": name_votes, "modal_tag": modal_tag,
            "modal_name": modal_name, "rows": defaultdict(list), "gaps": []}
        run_start = None
        for rec in t["records"]:
            supports = [s for s in rec.get("supports", [])
                        if s["class"] != "none"]
            if not supports:
                if run_start is None:
                    run_start = rec["keyframe"]
                continue
            if run_start is not None:
                track_infos[t["track_id"]]["gaps"].append(
                    (run_start, rec["keyframe"] - 1))
                run_start = None
            for s in supports:
                obs = obs_by_id.get(s["obs_id"])
                if obs is None:
                    continue
                eff = effective_class(s)
                verdict = judge(s, eff, obs, modal_tag, modal_name,
                                rec["keyframe"], t)
                stats[(verdict[0], verdict[1])] += 1
                tasks[rec["keyframe"]].append(
                    (t["track_id"], rec, s, eff, obs, verdict))
        if run_start is not None:
            track_infos[t["track_id"]]["gaps"].append(
                (run_start, t["records"][-1]["keyframe"]))

    print(f"{len(picked)} tracks, "
          f"{sum(len(v) for v in tasks.values())} support chips, "
          f"{len(tasks)} keyframes to decode")
    for keyframe in sorted(tasks):
        frame = frames_by_idx.get(keyframe)
        if frame is None:
            continue
        pano = np.asarray(Image.open(
            paths.dataset_base / "panorama" / f"{frame.pano_stem}.jpg"))
        for track_id, rec, s, eff, obs, verdict in tasks[keyframe]:
            pano_box = pg.pano_bbox_for_observation(obs.boxes, pano_w, pano_h)
            mask_pano = None
            mb = rec.get("mask_bbox_window")
            if mb is not None:
                ox, oy = rec["window_origin"]
                mask_pano = (ox + mb[0], oy + mb[1], ox + mb[2], oy + mb[3])
            chip_rel = f"chips/T{track_id}_f{keyframe:04d}_{s['obs_id']}.jpg"
            color = tuple(int(VERDICT_CSS[verdict[0]][i:i + 1] * 2, 16)
                          for i in (1, 2, 3))
            render_chip(pano, pano_box, mask_pano, out / chip_rel, color)
            track_infos[track_id]["rows"][keyframe].append(
                (rec, s, eff, obs, verdict, chip_rel))

    # ---- page ----
    parts = [
        "<html><head><title>Level-1 evidence review</title><style>",
        "body{font-family:sans-serif;background:#161616;color:#ddd;margin:16px}",
        "a{color:#8bf}.chip{display:inline-block;background:#222;margin:3px;",
        "padding:4px;border-radius:4px;vertical-align:top;max-width:280px;",
        "font-size:12px}",
        ".chip img{height:170px;display:block;border-radius:3px}",
        "table{border-collapse:collapse}td,th{padding:2px 8px;font-size:13px;",
        "border-bottom:1px solid #333;text-align:left}",
        ".v_include{color:#3c3}.v_flag{color:#fa2}.v_exclude{color:#e55}",
        ".v_context{color:#99b}",
        ".notes{background:#26261c;border-left:4px solid #fa2;padding:8px 12px;",
        "border-radius:4px;margin:8px 0;max-width:1200px}",
        ".kf{color:#89a;font-weight:bold;margin-top:8px}",
        "h2{margin-top:44px;border-top:1px solid #333;padding-top:18px}",
        "</style></head><body>",
        "<h1>Level-1 review: supporting evidence for 20 tracks</h1>",
        f"<p>run: {html.escape(args.run_dir.name)} | chip border & verdict "
        "colors: <span class='v_include'>include</span> / "
        "<span class='v_flag'>flag</span> / "
        "<span class='v_exclude'>exclude</span> / "
        "<span class='v_context'>context (not support)</span> | "
        "red rectangle on chips = the track's mask bbox at that keyframe</p>",
        "<h3>Candidate rules</h3><pre style='color:#aaa'>"
        + html.escape(__doc__.split("Candidate rules")[1].split("Run:")[0])
        + "</pre>",
        "<h3>Rule firing counts</h3><table><tr><th>verdict</th><th>rule</th>"
        "<th>count</th></tr>"]
    for (v, rule), c in sorted(stats.items()):
        parts.append(f"<tr><td class='v_{v}'>{v}</td><td>{rule}</td>"
                     f"<td>{c}</td></tr>")
    parts.append("</table>")
    if findings["global"]:
        parts.append("<h3>Findings</h3><div class='notes'>"
                     + findings["global"] + "</div>")

    for t, why in picked:
        info = track_infos[t["track_id"]]
        # See m6_merge_tracks: the range name comes from the artifact, because
        # track pages are track_<range>_T<id>.html.
        key = f"{range_name}_T{t['track_id']}"
        status = t["close_reason"] if t["status"] == "closed" else "alive"
        parts.append(
            f"<h2 id='T{t['track_id']}'>T{t['track_id']} "
            f"{html.escape(t['modal_label'])}</h2>"
            f"<p>picked: {why} | sup={t['n_supported_keyframes']} | "
            f"f{t['birth_keyframe']:04d}..f{t['end_keyframe']:04d} | "
            f"{status} | <a href='../track_{key}.html'>track page + video"
            "</a></p>")
        parts.append("<p>tag votes: " + ", ".join(
            f"{k} x{v}" for k, v in info["tag_votes"].most_common())
            + (" | name votes: " + ", ".join(
                f"'{k}' x{v}" for k, v in info["name_votes"].most_common())
               if info["name_votes"] else " | (no names)") + "</p>")
        if info["gaps"]:
            gap_txt = ", ".join(f"f{a:04d}-f{b:04d}" for a, b in info["gaps"])
            parts.append(f"<p style='color:#888'>unsupported stretches: "
                         f"{gap_txt}</p>")
        commentary = findings["tracks"].get(str(t["track_id"]), "")
        if commentary:
            parts.append(f"<div class='notes'>{commentary}</div>")
        for keyframe in sorted(info["rows"]):
            parts.append(f"<div class='kf'>f{keyframe:04d}</div>")
            for rec, s, eff, obs, (v, rule, reason), chip_rel in \
                    info["rows"][keyframe]:
                tag, name = obs_tags(obs)
                label = html.escape(vc.obs_semantic_label(obs))
                desc = html.escape(obs.description[:110])
                why_txt = html.escape(reason)
                cls_txt = eff if eff == s["class"] \
                    else f"{eff} (was {s['class']})"
                mb = rec.get("mask_bbox_window")
                ratio = ""
                if mb is not None and mb[2] > mb[0]:
                    bw = s["box_window"][2] - s["box_window"][0]
                    ratio = f" box/mask={bw / (mb[2] - mb[0]):.1f}x"
                parts.append(
                    f"<div class='chip'><img src='{chip_rel}' loading='lazy'>"
                    f"<b class='v_{v}'>{v}"
                    + (f" ({rule})" if rule != "-" else "") + "</b> "
                    f"{cls_txt} iou={s['iou']:.2f} "
                    f"i/m={s['inter_over_mask']:.2f} "
                    f"i/b={s['inter_over_box']:.2f}{ratio}<br>{label}<br>"
                    f"<span style='color:#999'>{desc}</span>"
                    + (f"<br><i>{why_txt}</i>" if why_txt else "")
                    + "</div>")
    parts.append("</body></html>")
    (out / "index.html").write_text("\n".join(parts))
    print(f"wrote {out}/index.html")
    print("verdicts:", dict(stats))


if __name__ == "__main__":
    main()
