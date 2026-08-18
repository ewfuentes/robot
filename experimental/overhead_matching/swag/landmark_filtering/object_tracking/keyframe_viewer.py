"""Per-keyframe detection viewer.

One page per keyframe showing the full annotated panorama - every detection
box (colored by semantic identity, labeled) plus the mask bbox of every
track alive at that keyframe (red, labeled T##) - and a table with a zoom
chip per detection: its tags, description, and which track (if any) absorbed
it and as what support class.

This is the ground level of the viewer hierarchy (leg -> track -> keyframe):
track pages and the semantic-audit review link into these pages so any
detection mentioned elsewhere can be inspected in full context.

Outputs <run_dir>/keyframes/f####.html (+ index.html + assets).

Run:
  bazel run //...object_tracking:keyframe_viewer -- \\
      --run_dir <runs>/r002_full_leg1 [--kf_start N --kf_end M]
"""

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experimental.overhead_matching.swag.data import farfield_paths
from experimental.overhead_matching.swag.landmark_filtering import ingest
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    pano_geometry as pg,
    semantic_audit as sa,
    track_builder as tb,
    viz_common as vc,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    IngestConfig,
)


MASK_COLOR = (255, 60, 60)
CHIP_H = 200

STYLE = """
body{font-family:sans-serif;background:#161616;color:#ddd;margin:16px}
a{color:#8bf}
.panowrap{overflow-x:auto;border:1px solid #333;border-radius:4px}
.panowrap img{display:block}
table{border-collapse:collapse;margin-top:12px}
td,th{padding:4px 10px;font-size:13px;border-bottom:1px solid #333;
text-align:left;vertical-align:top}
td img{height:200px;border-radius:3px;display:block}
.kf{color:#89a;font-weight:bold}
.cls_context{color:#99b}.cls_none{color:#777}
.seeds{color:#fd6;font-weight:bold}
.cls_continue_clean,.cls_merge_superset,.cls_weak,.cls_split_child{color:#3c3}
.masklab{color:#e55}
"""


def draw_wrapped_rect(draw, box, scale, pano_w_scaled, color, width, label,
                      font):
    """Rect in scaled-pano coords; if it crosses the wrap (x1 > pano width),
    draw the overflow again at x - width."""
    x0, y0, x1, y1 = [v * scale for v in box]
    for dx in (0, -pano_w_scaled):
        if x0 + dx < pano_w_scaled and x1 + dx > 0:
            draw.rectangle([x0 + dx, y0, x1 + dx, y1], outline=color,
                           width=width)
            if label:
                ty = y0 - 15 if y0 >= 15 else y1 + 2
                draw.text((max(x0 + dx, 2) + 1, ty + 1), label,
                          fill=(0, 0, 0), font=font)
                draw.text((max(x0 + dx, 2), ty), label, fill=color, font=font)


def track_associations(artifact, classifier_cfg):
    """Returns (by_obs, masks, seeded, rejected).

    by_obs:   (keyframe, obs_id) -> [(track_id, effective_class)]
    masks:    keyframe -> [(track_id, action, mask_box_pano)]
    seeded:   obs_id -> track_id it founded. A birth is NOT recorded as a
              support entry, so without this a detection that created a
              track looks unclaimed on the page.
    rejected: obs_id -> health dict for births the birth gate refused.
    """
    by_obs = defaultdict(list)
    masks = defaultdict(list)
    seeded = {}
    rejected = {r["obs_id"]: r.get("health", {})
                for r in artifact.get("rejected_births", [])}
    for t in artifact["tracks"]:
        seeded[t["birth_obs_id"]] = t["track_id"]
        for rec in t["records"]:
            mb = rec.get("mask_bbox_window")
            if mb is not None:
                ox, oy = rec["window_origin"]
                masks[rec["keyframe"]].append(
                    (t["track_id"], rec["action"],
                     (ox + mb[0], oy + mb[1], ox + mb[2], oy + mb[3])))
            for s in rec.get("supports", []):
                eff = tb.classify_support(
                    {"iou": s["iou"], "inter_over_mask": s["inter_over_mask"],
                     "inter_over_box": s["inter_over_box"]}, classifier_cfg)
                by_obs[(rec["keyframe"], s["obs_id"])].append(
                    (t["track_id"], eff))
    return by_obs, masks, seeded, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    farfield_paths.add_arguments(parser)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--pano_width", type=int, default=3072)
    parser.add_argument("--kf_start", type=int, default=None)
    parser.add_argument("--kf_end", type=int, default=None)
    args = parser.parse_args()
    paths = farfield_paths.resolve(
        parser, args, infer_from=args.run_dir,
        require=("dataset_base", "frame_landmarks"))

    artifact = json.loads(
        next(args.run_dir.glob("tracks_*.json")).read_text())
    range_name = artifact.get("range", {}).get("name", "full_leg1")
    result = ingest.run_ingest(paths.dataset_base, paths.frame_landmarks,
                               IngestConfig())
    obs_by_frame = defaultdict(list)
    for o in result.observations:
        obs_by_frame[o.frame_idx].append(o)
    frames = sorted(result.frames, key=lambda f: f.frame_idx)
    if args.kf_start is not None:
        frames = [f for f in frames if f.frame_idx >= args.kf_start]
    if args.kf_end is not None:
        frames = [f for f in frames if f.frame_idx <= args.kf_end]

    classifier_cfg = tb.TrackBuilderConfig()
    assoc_by_obs, masks_by_kf, seeded_by_obs, rejected_births = \
        track_associations(artifact, classifier_cfg)

    out = args.run_dir / "keyframes"
    (out / "img").mkdir(parents=True, exist_ok=True)
    font = vc.load_font(14)

    def kf_name(idx):
        return f"f{idx:04d}"

    index_rows = []
    for n, frame in enumerate(frames):
        kf = frame.frame_idx
        obs_list = sorted(obs_by_frame.get(kf, []), key=lambda o: o.obs_id)
        masks = sorted(masks_by_kf.get(kf, []))
        pano = np.asarray(Image.open(
            paths.dataset_base / "panorama" / f"{frame.pano_stem}.jpg"))
        pano_h, pano_w = pano.shape[:2]
        scale = args.pano_width / pano_w

        # annotated pano
        anno = Image.fromarray(pano).resize(
            (args.pano_width, int(pano_h * scale)), Image.BILINEAR)
        draw = ImageDraw.Draw(anno)
        for o in obs_list:
            box = pg.pano_bbox_for_observation(o.boxes, pano_w, pano_h)
            draw_wrapped_rect(draw, box, scale, args.pano_width,
                              vc.obs_color(o), 2, vc.obs_semantic_label(o),
                              font)
        for track_id, action, box in masks:
            draw_wrapped_rect(draw, box, scale, args.pano_width, MASK_COLOR,
                              2, f"T{track_id}", font)
        anno.save(out / "img" / f"{kf_name(kf)}_pano.jpg", quality=85)

        # detection chips
        for o in obs_list:
            box = pg.pano_bbox_for_observation(o.boxes, pano_w, pano_h)
            sa.render_chip(pano, box, None,
                           out / "img" / f"{kf_name(kf)}_{o.obs_id}.jpg",
                           CHIP_H)

        prev_kf = kf_name(frames[n - 1].frame_idx) if n else None
        next_kf = kf_name(frames[n + 1].frame_idx) if n + 1 < len(frames) \
            else None
        parts = [
            f"<html><head><title>{kf_name(kf)}</title><style>{STYLE}</style>",
            "</head><body>",
            "<p><a href='index.html'>&larr; all keyframes</a>"
            + (f" | <a href='{prev_kf}.html'>&larr; {prev_kf}</a>"
               if prev_kf else "")
            + (f" | <a href='{next_kf}.html'>{next_kf} &rarr;</a>"
               if next_kf else "")
            + " | <a href='../index.html'>track boards</a></p>",
            f"<h1>{kf_name(kf)}</h1>",
            f"<p>{len(obs_list)} detections | {len(masks)} track masks "
            "(red). Scroll the panorama horizontally; box colors are stable "
            "per (tag, name) identity.</p>",
            f"<div class='panowrap'><img src='img/{kf_name(kf)}_pano.jpg'>"
            "</div>",
            "<h2>Detections</h2>",
            "<table><tr><th>chip</th><th>detection</th><th>description</th>"
            "<th>track</th></tr>"]
        for o in obs_list:
            tags = dict(tuple(t) for t in o.additional_tags)
            label = html.escape(vc.obs_semantic_label(o))
            dist = tags.get("distance_estimate", "?")
            lines = []
            seeded = seeded_by_obs.get(o.obs_id)
            if seeded is not None:
                lines.append(
                    f"<a href='../track_{range_name}_T{seeded}.html' "
                    f"class='seeds'>&#9733; seeds T{seeded}</a>")
            if o.obs_id in rejected_births:
                reason = rejected_births[o.obs_id].get("reason", "?")
                lines.append("<span class='cls_none'>birth rejected "
                             f"({html.escape(reason)})</span>")
            # Supports first, then classes that were considered and refused.
            assoc = assoc_by_obs.get((kf, o.obs_id), [])
            real = [(tid, c) for tid, c in assoc if c in tb.SUPPORT_CLASSES]
            other = [(tid, c) for tid, c in assoc
                     if c not in tb.SUPPORT_CLASSES]
            for tid, cls in real + other:
                note = "" if cls in tb.SUPPORT_CLASSES else " (not support)"
                lines.append(
                    f"<a href='../track_{range_name}_T{tid}.html'>T{tid}</a> "
                    f"<span class='cls_{cls}'>{cls}{note}</span>")
            if not lines:
                lines.append("<span class='cls_none'>unclaimed</span>")
            assoc_txt = "<br>".join(lines)
            parts.append(
                f"<tr id='{o.obs_id}'>"
                f"<td><img src='img/{kf_name(kf)}_{o.obs_id}.jpg' "
                "loading='lazy'></td>"
                f"<td><code>{o.obs_id}</code><br>{label}<br>"
                f"dist: {html.escape(dist)}</td>"
                f"<td style='max-width:420px'>{html.escape(o.description)}"
                "</td>"
                f"<td>{assoc_txt}</td></tr>")
        parts.append("</table>")
        if masks:
            parts.append("<h2>Tracks alive at this keyframe</h2>"
                         "<table><tr><th>track</th><th>action</th>"
                         "<th>mask bbox (pano px)</th></tr>")
            for track_id, action, box in masks:
                bb = ", ".join(f"{v:.0f}" for v in box)
                parts.append(
                    f"<tr id='T{track_id}'><td class='masklab'>"
                    f"<a href='../track_{range_name}_T{track_id}.html'>"
                    f"T{track_id}</a></td>"
                    f"<td>{action}</td><td>{bb}</td></tr>")
            parts.append("</table>")
        parts.append("</body></html>")
        (out / f"{kf_name(kf)}.html").write_text("\n".join(parts))
        index_rows.append((kf, len(obs_list), len(masks)))
        if n % 50 == 0:
            print(f"[{n + 1}/{len(frames)}] {kf_name(kf)}")

    parts = [
        f"<html><head><title>keyframes</title><style>{STYLE}</style></head>",
        "<body><h1>keyframes</h1>",
        "<p><a href='../index.html'>track boards</a></p>",
        "<table><tr><th>keyframe</th><th>detections</th>"
        "<th>track masks</th></tr>"]
    for kf, n_obs, n_masks in index_rows:
        parts.append(f"<tr><td class='kf'><a href='{kf_name(kf)}.html'>"
                     f"{kf_name(kf)}</a></td><td>{n_obs}</td>"
                     f"<td>{n_masks}</td></tr>")
    parts.append("</table></body></html>")
    (out / "index.html").write_text("\n".join(parts))
    print(f"wrote {len(index_rows)} keyframe pages to {out}")


if __name__ == "__main__":
    main()
