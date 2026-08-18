"""M3 track viewer: per-track videos, evidence tables, and navigation.

Re-runs tracking over the requested ranges with a media hook that captures
every propagation frame, then writes a static site:
- viewer/index.html: per-range stats, an alive-timeline (click a bar to open
  its track), a filterable gallery of tracks with representative thumbnails,
  and a section for landmarks that produced no track (rejected births and
  never-supported tracks).
- viewer/track_<range>_T<id>.html: looping video of the track's whole life
  (mask overlay, detection boxes at keyframes colored by class), a
  termination explanation, and a per-keyframe evidence table listing every
  nearby detection with include/exclude reasoning against the config
  thresholds.

Serve with any static file server, e.g. the http.server already running on
the output tree.

Run:
  bazel run //experimental/overhead_matching/swag/landmark_filtering/object_tracking:m3_track_viewer
"""

import argparse
import html
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experimental.overhead_matching.swag.data import farfield_paths
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    pano_geometry as pg,
    range_runner as rr,
    track_builder as tb,
    viz_common as vc,
)

STAGE_DIR = "m3_tracks"


def refresh_artifact_manifest(paths, runs_root: Path, run_name: str):
    """Write/refresh `manifest.json` for the object_tracks artifact.

    No stage used to write one: leg1's was authored by hand during the disk
    reorganisation, so a second dataset's artifact would have had none at all.
    Refreshed on every run because this artifact is a *working tree* -- its
    version bumps only when the pipeline changes shape, while runs accumulate
    inside it -- so the manifest's job is to name the producer and point at the
    per-run records that carry the settings.
    """
    if paths is None:
        return
    runs = sorted(p.name for p in runs_root.glob("*")
                  if p.is_dir() and (p / "run_meta.json").exists())
    root = paths.root
    paths.write_manifest(
        farfield_paths.OBJECT_TRACKS,
        generator=("//experimental/overhead_matching/swag/landmark_filtering/"
                   "object_tracking (m0-m3 + audit/merge/matching stages)"),
        config={
            "per_run": "m3_tracks/runs/<run>/run_meta.json records the resolved "
                       "dataset, frame_landmarks and video, plus the ranges and "
                       "notes; tracks_<range>.json carries the full "
                       "TrackBuilderConfig it was built with",
            "per_stage": {
                "semantic_audit": "<run>/semantic_audit/audit_meta.json",
                "merge": "<run>/merged/",
                "matching": "<run>/matching/settings.json",
                "mount_offset": "<run>/mount_offset_sweep.json",
            },
            "runs": runs,
            "latest_run": run_name,
        },
        inputs=[
            farfield_paths.relative_to_root(paths.dataset_base, root),
            farfield_paths.relative_to_root(paths.frame_landmarks, root),
            farfield_paths.relative_to_root(paths.video, root),
        ],
        notes=("Working tree of the tracking pipeline: products and their debug "
               "boards travel together, and the internal layout is the "
               "producer's. Immutability lives on the run ids - never "
               "regenerate an rNNN in place with different settings, mint the "
               "next one. Mint v<N+1> if the pipeline changes shape."),
    )

# Short dev ranges from boston_harbor_leg1. A real run passes --range covering
# the whole leg; these exist so an iteration on a rule change is cheap.
LEG1_DEV_RANGES = [("f0000_departure", 0, 30), ("f0122_port", 114, 144),
                   ("f0149_fort", 141, 171)]

CLASS_COLORS = {
    "continue_clean": (60, 220, 60),
    "merge_superset": (200, 90, 220),
    "split_child": (70, 220, 200),
    "weak": (240, 170, 40),
    "context": (150, 150, 170),
    "none": (230, 60, 60),
}
CLASS_CSS = {k: f"rgb{v}" for k, v in CLASS_COLORS.items()}
STATUS_CSS = {"alive": "#3c3", "starved": "#fa2", "drift_alarm": "#c7d",
              "mask_dead": "#e44", "mask_lost_in_window": "#e44"}
SUPPORT_SCORE = {"continue_clean": 3, "merge_superset": 2, "split_child": 2,
                 "weak": 1, "context": 0, "none": 0}
VIDEO_SIZE = 512
VIDEO_FPS = 6


class MediaSink:
    """TrackBuilder on_interval hook: captures annotated frames per track and
    encodes them into one mp4 per track at finalize()."""

    def __init__(self, out_dir: Path, range_name: str, font, pano_w: int):
        self.range_name = range_name
        self.font = font
        self.pano_w = pano_w
        self.frames_root = out_dir / "_frames" / range_name
        self.videos_dir = out_dir / "videos"
        self.thumbs_dir = out_dir / "thumbs"
        for d in (self.frames_root, self.videos_dir, self.thumbs_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.counters = defaultdict(int)
        self.best_thumb_score = {}

    def key(self, track):
        return f"{self.range_name}_T{track.track_id}"

    def on_interval(self, track, keyframe, crops, origins, masks):
        rec = track.records[-1] if track.records else None
        first_interval = (track.records
                          and track.records[0]["keyframe"] == keyframe)
        key = self.key(track)
        track_dir = self.frames_root / key
        track_dir.mkdir(exist_ok=True)
        action = rec["action"] if rec else "?"
        n = len(crops)
        for i, (crop, mask) in enumerate(zip(crops, masks)):
            if i == 0 and not first_interval:
                continue  # frame 0 duplicates the previous interval's end
            img = crop.copy()
            if mask is not None and mask.any():
                overlay = np.zeros_like(img)
                overlay[mask] = (255, 60, 60)
                img = (0.65 * img + 0.35 * overlay).astype(np.uint8)
            pil = Image.fromarray(img)
            draw = ImageDraw.Draw(pil)
            if i == 0 and first_interval and hasattr(track, "_birth_pano_box"):
                box = self._window_box(track._birth_pano_box, origins[0])
                draw.rectangle(box, outline=(60, 255, 60), width=4)
            if i == n - 1 and rec is not None:
                for s in rec.get("supports", []):
                    color = CLASS_COLORS.get(s["class"], (200, 200, 200))
                    draw.rectangle(s["box_window"], outline=color, width=3)
                    draw.text((s["box_window"][0], s["box_window"][1] - 16),
                              f"{s['class']} iou={s['iou']:.2f}",
                              fill=color, font=self.font)
            vc.draw_caption(
                draw, f"f{keyframe:04d}->f{keyframe + 1:04d}  {action}",
                self.font)
            pil = pil.resize((VIDEO_SIZE, VIDEO_SIZE), Image.BILINEAR)
            pil.save(track_dir / f"{self.counters[key]:06d}.jpg", quality=88)
            self.counters[key] += 1
            if i == n - 1 or (i == 0 and first_interval):
                self._maybe_thumb(key, pil, rec)

    def _window_box(self, pano_box, origin):
        rel = pg.signed_x_offset(pano_box[0], origin[0], self.pano_w)
        return [rel, pano_box[1] - origin[1],
                rel + (pano_box[2] - pano_box[0]), pano_box[3] - origin[1]]

    def _maybe_thumb(self, key, pil, rec):
        score = -1.0
        if rec is not None:
            for s in rec.get("supports", []):
                score = max(score,
                            SUPPORT_SCORE.get(s["class"], 0) + s["iou"])
        if score >= self.best_thumb_score.get(key, -2.0):
            self.best_thumb_score[key] = score
            pil.resize((256, 256), Image.BILINEAR).save(
                self.thumbs_dir / f"{key}.jpg", quality=85)

    def finalize(self):
        encoded = {}
        for track_dir in sorted(self.frames_root.iterdir()):
            key = track_dir.name
            n = self.counters.get(key, 0)
            if n == 0:
                continue
            out = self.videos_dir / f"{key}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-framerate", str(VIDEO_FPS),
                 "-i", str(track_dir / "%06d.jpg"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
                 str(out)], check=True)
            encoded[key] = n
        shutil.rmtree(self.frames_root, ignore_errors=True)
        return encoded


# --------------------------------------------------------------------------
# Explanations


def explain_support(s, cfg, near_miss):
    c, iou = s["class"], s["iou"]
    iom, iob = s["inter_over_mask"], s["inter_over_box"]
    if c == "continue_clean":
        return (f"included, clean 1:1: iou {iou:.2f} >= {cfg['clean_iou']}; "
                "eligible to re-anchor the track")
    if c == "merge_superset":
        return (f"included as superset evidence: box covers {iom:.2f} of the "
                f"mask but the mask fills only {iob:.2f} of the box "
                "(detection much wider than the tracked object; no re-anchor)")
    if c == "split_child":
        return (f"included as child evidence: box is {iob:.2f} inside the "
                f"mask but covers only {iom:.2f} of it (possible split)")
    if c == "weak":
        return (f"included weakly: some overlap (iou {iou:.2f}) but neither "
                "clean nor containment-shaped")
    if c == "context":
        return (f"context only: box contains the mask (inter/mask "
                f"{iom:.2f}) but the mask fills just {iob:.2f} of it - "
                "likely a different, larger object; not counted as support")
    base = (f"excluded: iou {iou:.2f} < {cfg['weak_min_iou']}, inter/mask "
            f"{iom:.2f} and inter/box {iob:.2f} < "
            f"{cfg['weak_min_containment']}")
    if near_miss:
        base += (" | NEAR MISS: same-tag detection within the drift gate; "
                 "counted toward the drift alarm")
    return base


def explain_close(t, cfg):
    reason = t["close_reason"]
    if t["status"] == "alive":
        return "Alive at the end of the range."
    if reason == "starved":
        return (f"Starved: {cfg['patience_keyframes']} consecutive keyframes "
                "passed with no supporting detection. Last support at "
                f"f{t['end_keyframe']:04d}.")
    if reason == "drift_alarm":
        return (f"Drift alarm: {cfg['drift_patience']} consecutive keyframes "
                "where a detection with the track's modal tag landed within "
                f"{cfg['drift_gate_px']:.0f} px of the mask without "
                "overlapping it - the mask most likely slid off the object.")
    if reason == "mask_dead":
        return (f"Mask died: propagated mask area fell below "
                f"{cfg['min_mask_area_px']} px - the object left the window "
                "or SAM lost it.")
    if reason == "mask_lost_in_window":
        return "The stored mask fell outside the re-centered window."
    if reason.startswith("birth_"):
        h = t["records"][0].get("health", {})
        detail = {
            "birth_fragmented": (
                f"largest component holds {h.get('dominant_cc_frac')} of the "
                f"mask (< {cfg['birth_min_dominant_cc']}), "
                f"{h.get('n_components')} components"),
            "birth_spill": (
                f"{h.get('spill_frac')} of the mask lies outside the "
                f"detection box (> {cfg['birth_max_spill']})"),
            "birth_coverage": (
                f"mask covers only {h.get('coverage')} of the detection box "
                f"(< {cfg['birth_min_coverage']})"),
            "birth_empty": "prompt produced an (almost) empty mask",
        }.get(reason, "")
        return f"Rejected at birth ({reason.removeprefix('birth_')}): {detail}."
    return reason


def modal_primary_tag(label):
    return label.split(" '")[0]


def near_miss_flag(rec, s, track_label, obs_by_id, cfg):
    if s["class"] != "none" or rec.get("mask_bbox_window") is None:
        return False
    obs = obs_by_id.get(s["obs_id"])
    if obs is None:
        return False
    tag = f"{obs.primary_tag_key}={obs.primary_tag_value}"
    if tag != modal_primary_tag(track_label):
        return False
    mb = rec["mask_bbox_window"]
    mc = ((mb[0] + mb[2]) / 2.0, (mb[1] + mb[3]) / 2.0)
    bb = s["box_window"]
    bc = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)
    return math.hypot(mc[0] - bc[0], mc[1] - bc[1]) <= cfg["drift_gate_px"]


# --------------------------------------------------------------------------
# HTML

STYLE = """
body{font-family:sans-serif;background:#161616;color:#ddd;margin:16px}
a{color:#8bf} img,video{max-width:100%}
.cards{display:flex;flex-wrap:wrap;gap:10px}
.card{background:#222;border-radius:6px;padding:8px;width:220px}
.card img{width:204px;height:204px;object-fit:cover;border-radius:4px}
.chip{display:inline-block;padding:1px 7px;border-radius:9px;font-size:12px;
      background:#333}
table{border-collapse:collapse;margin:8px 0}
td,th{padding:3px 9px;text-align:left;border-bottom:1px solid #333;
      font-size:14px;vertical-align:top}
.kf{background:#262626;font-weight:bold}
.controls{margin:10px 0}
.controls select,.controls input{background:#222;color:#ddd;border:1px solid
 #444;padding:3px;margin-right:8px}
h2{margin-top:34px}
.banner{background:#282828;border-left:5px solid #fa2;padding:8px 12px;
        border-radius:4px;margin:10px 0}
"""

GALLERY_JS = """
function applyFilters(){
  const range=document.getElementById('f_range').value;
  const status=document.getElementById('f_status').value;
  const minsup=parseInt(document.getElementById('f_minsup').value||'0');
  document.querySelectorAll('.card').forEach(c=>{
    const ok=(range=='all'||c.dataset.range==range)
      &&(status=='all'||(status=='alive'?c.dataset.status=='alive'
                         :c.dataset.status!='alive'))
      &&parseInt(c.dataset.sup)>=minsup;
    c.style.display=ok?'':'none';});
}
"""


def status_chip(t):
    s = t["close_reason"] if t["status"] == "closed" else "alive"
    color = STATUS_CSS.get(s, "#999")
    return f"<span class='chip' style='color:{color}'>{html.escape(s)}</span>"


def track_key(range_name, tid):
    return f"{range_name}_T{tid}"


def render_track_page(out, range_name, t, cfg, obs_by_id, seeded_by,
                      nav_prev, nav_next, video_exists):
    key = track_key(range_name, t["track_id"])
    span = (f"f{t['birth_keyframe']:04d}..f{t['end_keyframe']:04d}"
            if t["end_keyframe"] is not None else "-")
    parts = [f"<html><head><title>{key}</title><style>{STYLE}</style></head>",
             "<body>",
             f"<p><a href='index.html'>&larr; all tracks</a>"
             + (f" | <a href='track_{nav_prev}.html'>&larr; prev</a>"
                if nav_prev else "")
             + (f" | <a href='track_{nav_next}.html'>next &rarr;</a>"
                if nav_next else "") + "</p>",
             f"<h1>{key}: {html.escape(t['modal_label'])}</h1>",
             f"<p>born f{t['birth_keyframe']:04d} from "
             f"<code>{html.escape(t['birth_obs_id'])}</code> | supported span "
             f"{span} | {t['n_supported_keyframes']} supported keyframes | "
             f"{status_chip(t)}</p>",
             f"<div class='banner'>{html.escape(explain_close(t, cfg))}</div>"]
    if video_exists:
        parts.append(f"<video src='videos/{key}.mp4' controls loop muted "
                     "autoplay playsinline style='width:512px'></video>")
    parts.append("<h2>Evidence by keyframe</h2>")
    parts.append("<table><tr><th>keyframe</th><th>action</th>"
                 "<th>detection</th><th>class</th><th>iou</th>"
                 "<th>inter/mask</th><th>inter/box</th><th>why</th></tr>")
    for rec in t["records"]:
        kf = f"f{rec['keyframe']:04d}"
        action = rec["action"]
        supports = rec.get("supports", [])
        area = rec.get("mask_area")
        extra = f" (mask {area}px)" if area is not None else ""
        kf_cell = f"<a href='keyframes/{kf}.html'>{kf}</a>"
        if not supports:
            health = rec.get("health")
            note = (f"health: {json.dumps(health)}" if health
                    else "no detections near the window")
            parts.append(f"<tr><td class='kf'>{kf_cell}</td><td>{action}{extra}"
                         f"</td><td colspan='6'>{html.escape(note)}</td></tr>")
            continue
        for i, s in enumerate(supports):
            nm = near_miss_flag(rec, s, t["modal_label"], obs_by_id, cfg)
            why = explain_support(s, cfg, nm)
            color = CLASS_CSS.get(s["class"], "#999")
            obs = obs_by_id.get(s["obs_id"])
            obs_label = (vc.obs_semantic_label(obs) if obs else s["obs_id"])
            seeded = seeded_by.get(s["obs_id"])
            link = (f" <a href='track_{seeded}.html'>&rarr;{seeded.split('_T')[-1]}"
                    "</a>" if seeded and seeded != key else "")
            first = (f"<td class='kf'>{kf_cell}</td><td>{action}{extra}</td>"
                     if i == 0 else "<td></td><td></td>")
            parts.append(
                f"<tr>{first}"
                f"<td><code>{html.escape(s['obs_id'])}</code>{link}<br>"
                f"{html.escape(obs_label)}</td>"
                f"<td style='color:{color}'>{s['class']}</td>"
                f"<td>{s['iou']:.2f}</td><td>{s['inter_over_mask']:.2f}</td>"
                f"<td>{s['inter_over_box']:.2f}</td>"
                f"<td>{html.escape(why)}</td></tr>")
    parts.append("</table></body></html>")
    (out / f"track_{key}.html").write_text("\n".join(parts))


def timeline_svg(range_name, artifact):
    k0 = artifact["range"]["k_start"]
    k1 = artifact["range"]["k_end"]
    tracks = [t for t in artifact["tracks"] if t["records"]]
    tracks.sort(key=lambda t: (t["birth_keyframe"], t["track_id"]))
    px, row_h = 14, 14
    width = (k1 - k0 + 1) * px + 60
    height = len(tracks) * row_h + 24
    rows = [f"<svg width='{width}' height='{height}' "
            "xmlns='http://www.w3.org/2000/svg' style='background:#1c1c1c'>"]
    for j in range(k0, k1 + 1, 5):
        x = 50 + (j - k0) * px
        rows.append(f"<text x='{x}' y='12' fill='#888' font-size='10'>"
                    f"f{j:04d}</text>")
    for i, t in enumerate(tracks):
        y = 20 + i * row_h
        last = t["last_keyframe"] if t["last_keyframe"] is not None else \
            t["birth_keyframe"]
        x0 = 50 + (t["birth_keyframe"] - k0) * px
        w = max((last - t["birth_keyframe"]) * px, 4)
        status = t["close_reason"] if t["status"] == "closed" else "alive"
        color = STATUS_CSS.get(status, "#999")
        key = track_key(range_name, t["track_id"])
        rows.append(
            f"<a href='track_{key}.html'><rect x='{x0}' y='{y}' width='{w}' "
            f"height='10' rx='2' fill='{color}' fill-opacity='0.45'>"
            f"<title>T{t['track_id']} {html.escape(t['modal_label'])} "
            f"[{status}] sup={t['n_supported_keyframes']}</title></rect>")
        for rec in t["records"]:
            if any(s["class"] in tb.SUPPORT_CLASSES
                   for s in rec.get("supports", [])):
                x = 50 + (rec["keyframe"] - k0) * px
                rows.append(f"<rect x='{x}' y='{y}' width='4' height='10' "
                            f"fill='{color}'/>")
        rows.append(f"<text x='2' y='{y + 9}' fill='#aaa' font-size='9'>"
                    f"T{t['track_id']}</text></a>")
    rows.append("</svg>")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    farfield_paths.add_arguments(parser, video=True, checkpoint=True)
    parser.add_argument("--runs_root", type=Path, default=None,
                        help=f"default: <object_tracks artifact>/{STAGE_DIR}/runs")
    parser.add_argument("--run_name", required=True,
                        help="results land in <runs_root>/<run_name>/")
    parser.add_argument("--notes", default="",
                        help="what changed in this run and why (shown in "
                             "run_meta.json and the diff viewer)")
    parser.add_argument("--range", nargs=3, action="append", default=None,
                        metavar=("NAME", "K_START", "K_END"))
    parser.add_argument("--skip_existing_ranges", action="store_true",
                        help="reuse tracks_<range>.json + media already in "
                             "the run dir instead of re-tracking that range "
                             "(resume after a crash / partial iteration)")
    args = parser.parse_args()

    paths = farfield_paths.resolve(
        parser, args,
        require=("dataset_base", "frame_landmarks", "video", "sam2_checkpoint"))
    ctx = rr.load_context(paths.dataset_base, paths.frame_landmarks,
                          paths.video, paths.sam2_checkpoint)
    last_keyframe = max(f.frame_idx for f in ctx["result"].frames)
    if args.range:
        ranges = [(n, int(a), int(b)) for n, a, b in args.range]
        past_end = [n for n, _, b in ranges if b > last_keyframe]
        if past_end:
            parser.error(
                f"range(s) {', '.join(past_end)} end past this dataset's last "
                f"keyframe f{last_keyframe:04d} ({paths.dataset} has "
                f"{len(ctx['result'].frames)} frames)")
    else:
        ranges = [(n, a, min(b, last_keyframe)) for n, a, b in LEG1_DEV_RANGES
                  if a <= last_keyframe]
        if not ranges:
            parser.error("no default range fits this dataset; pass --range "
                         "NAME K_START K_END")
    font = vc.load_font(13)
    builder_cfg = tb.TrackBuilderConfig()
    runs_root = args.runs_root or paths.tracks_runs_root
    out = runs_root / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    import datetime
    (out / "run_meta.json").write_text(json.dumps({
        "run_name": args.run_name,
        "dataset": paths.dataset,
        "notes": args.notes,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "viewer_rel": ".",
        "inputs": {
            "dataset_base": str(paths.dataset_base),
            "frame_landmarks": str(paths.frame_landmarks),
            "video": str(paths.video),
        },
        "ranges": [{"name": n, "k_start": a, "k_end": b}
                   for n, a, b in ranges],
    }, indent=1))

    artifacts = {}
    videos = {}
    for range_name, k_start, k_end in ranges:
        art_path = out / f"tracks_{range_name}.json"
        if args.skip_existing_ranges and art_path.exists():
            artifacts[range_name] = json.loads(art_path.read_text())
            videos.update({p.stem: 1 for p in
                           (out / "videos").glob(f"{range_name}_T*.mp4")})
            print(f"range {range_name}: reusing existing artifact + media")
            continue
        print(f"range {range_name}: f{k_start:04d}..f{k_end:04d}")
        sink = MediaSink(out, range_name, font, ctx["pano_w"])
        _, artifact = rr.run_range(
            range_name, k_start, k_end, builder_cfg, ctx["backend"],
            ctx["provider"], ctx["model"], ctx["result"], ctx["obs_by_frame"],
            ctx["det_pano_boxes"], ctx["pano_w"], ctx["pano_h"],
            paths.dataset_base, on_interval=sink.on_interval)
        rr.write_artifact(artifact, out, range_name)
        artifacts[range_name] = artifact
        print("  encoding videos ...")
        videos.update(sink.finalize())

    cfg = next(iter(artifacts.values()))["config"]
    obs_by_id = ctx["obs_by_id"]

    # obs -> track it seeded (for cross links).
    seeded_by = {}
    for range_name, artifact in artifacts.items():
        for t in artifact["tracks"]:
            seeded_by[t["birth_obs_id"]] = track_key(range_name, t["track_id"])

    # Per-track pages, ordered for prev/next navigation.
    ordered = [(rn, t) for rn, a in artifacts.items() for t in a["tracks"]
               if t["records"]]
    keys = [track_key(rn, t["track_id"]) for rn, t in ordered]
    for i, (rn, t) in enumerate(ordered):
        render_track_page(
            out, rn, t, cfg, obs_by_id, seeded_by,
            keys[i - 1] if i > 0 else None,
            keys[i + 1] if i + 1 < len(keys) else None,
            track_key(rn, t["track_id"]) in videos)

    # Index page.
    parts = [f"<html><head><title>M3 track viewer</title>"
             f"<style>{STYLE}</style><script>{GALLERY_JS}</script></head>",
             "<body><h1>M3 track viewer</h1>"]
    parts.append("<h2>Alive timeline</h2><p>bar = track lifetime, solid "
                 "notches = supported keyframes; click to open</p>")
    for range_name, artifact in artifacts.items():
        parts.append(f"<h3>{html.escape(range_name)}</h3>")
        parts.append(timeline_svg(range_name, artifact))

    parts.append("<h2>Tracks</h2><div class='controls'>"
                 "range <select id='f_range' onchange='applyFilters()'>"
                 "<option value='all'>all</option>"
                 + "".join(f"<option>{rn}</option>" for rn in artifacts)
                 + "</select> status <select id='f_status' "
                 "onchange='applyFilters()'><option value='all'>all</option>"
                 "<option value='alive'>alive</option>"
                 "<option value='closed'>closed</option></select>"
                 " min supported <input id='f_minsup' type='number' value='0'"
                 " style='width:60px' oninput='applyFilters()'></div>")
    parts.append("<div class='cards'>")
    cards = []
    for range_name, artifact in artifacts.items():
        for t in artifact["tracks"]:
            if not t["records"]:
                continue
            cards.append((t["n_supported_keyframes"], range_name, t))
    cards.sort(key=lambda c: -c[0])
    for sup, range_name, t in cards:
        key = track_key(range_name, t["track_id"])
        status = t["close_reason"] if t["status"] == "closed" else "alive"
        thumb = f"thumbs/{key}.jpg"
        cards_html = (
            f"<div class='card' data-range='{range_name}' "
            f"data-status='{status}' data-sup='{sup}'>"
            f"<a href='track_{key}.html'><img src='{thumb}' loading='lazy'>"
            f"</a><br><b>T{t['track_id']}</b> "
            f"{html.escape(t['modal_label'][:40])}<br>"
            f"<small>{range_name} | b=f{t['birth_keyframe']:04d} | "
            f"sup={sup}</small><br>{status_chip(t)}</div>")
        parts.append(cards_html)
    parts.append("</div>")

    parts.append("<h2>Landmarks that produced no track</h2>"
                 "<p>Rejected at birth (mask health) or never supported "
                 "after birth.</p><div class='cards'>")
    for range_name, artifact in artifacts.items():
        rejected_ids = {r["obs_id"] for r in artifact["rejected_births"]}
        for t in artifact["tracks"]:
            if not t["records"]:
                continue
            is_reject = t["birth_obs_id"] in rejected_ids and \
                t["close_reason"].startswith("birth_")
            never_supported = (t["n_supported_keyframes"] == 0
                               and t["status"] == "closed")
            if not (is_reject or never_supported):
                continue
            key = track_key(range_name, t["track_id"])
            obs = obs_by_id.get(t["birth_obs_id"])
            label = vc.obs_semantic_label(obs) if obs else t["birth_obs_id"]
            parts.append(
                f"<div class='card' data-range='{range_name}' "
                f"data-status='{t['close_reason']}' data-sup='0'>"
                f"<a href='track_{key}.html'>"
                f"<img src='thumbs/{key}.jpg' loading='lazy'></a><br>"
                f"{html.escape(label[:48])}<br>"
                f"<small>{html.escape(t['birth_obs_id'])} ({range_name})"
                f"</small><br>{status_chip(t)}</div>")
    parts.append("</div></body></html>")
    (out / "index.html").write_text("\n".join(parts))
    print(f"wrote {out}/index.html ({len(keys)} track pages, "
          f"{len(videos)} videos)")

    # Only when writing into the artifact lane; a --runs_root pointed elsewhere
    # is scratch and should not claim to be an artifact version.
    if runs_root == paths.tracks_runs_root:
        refresh_artifact_manifest(paths, runs_root, args.run_name)
        print(f"refreshed {paths.object_tracks}/manifest.json")


if __name__ == "__main__":
    main()
