"""Tracking stage: SAM2 track building over keyframe ranges + track viewer.

Runs tracking over the requested ranges with a media hook that captures
every propagation frame, then writes a static site:
- <run_dir>/index.html: per-range stats, an alive-timeline (click a bar to
  open its track), a filterable gallery of tracks with representative
  thumbnails, and a section for landmarks that produced no track (rejected
  births and never-supported tracks).
- <run_dir>/track_<range>_T<id>.html: looping video of the track's whole life
  (mask overlay, detection boxes at keyframes colored by class), a
  termination explanation, and a per-keyframe evidence table listing every
  nearby detection with include/exclude reasoning against the RECORDED config
  thresholds.

Completion contract (the P0 fix for the crash-shaped hole the old stage had):
`run_meta.json` is written at the START of the run and records only inputs +
settings -- it is NOT a completion claim, and nothing may treat its presence
as one (the old stage wrote it before the range loop, so a crash mid-tracking
left a marker that made the orchestrator skip the stage). Completion lives in
`tracks_complete.json`, updated ONLY as each range finishes; a range is done
when its name appears there, and the run is done when every range declared in
run_meta.json appears there (`unfinished_ranges`). `--skip_existing_ranges`
reuses a finished range's tracks_<range>.json + media instead of re-tracking.

Serve with any static file server, e.g. the http.server already running on
the output tree.

Run:
  bazel run //experimental/overhead_matching/swag/farfield/tracking:run_tracking
"""

import argparse
import concurrent.futures as cf
import dataclasses
import datetime
import html
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from experimental.overhead_matching.swag.farfield import (
    dataset,
    geometry as geo,
    paths as paths_lib,
    provenance,
)
from experimental.overhead_matching.swag.farfield.tracking import (
    range_runner as rr,
    track_builder as tb,
    viz_common as vc,
)
from experimental.overhead_matching.swag.farfield.tracking.perf_profile import (
    PROFILE,
)
from experimental.overhead_matching.swag.farfield.viewers import (
    page as page_lib,
)

GENERATOR = "//experimental/overhead_matching/swag/farfield/tracking:run_tracking"

RUN_META = "run_meta.json"
TRACKS_COMPLETE = "tracks_complete.json"

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
MASK_TINT = (255, 60, 60)
ENCODE_WORKERS = 8
VIDEO_SIZE = 512
VIDEO_FPS = 6


# ---------------------------------------------------------------------------
# Run metadata + completion markers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def write_run_meta(run_dir: Path, *, run_name: str, dataset_name: str,
                   notes: str, inputs: dict, builder_cfg, ingest_params,
                   ranges: list) -> Path:
    """Record the run's inputs and settings, WITHOUT any completion claim.

    Written up front so a crash mid-run still leaves a record of what was
    attempted. Everything result-shaping is here verbatim: the full
    TrackBuilderConfig and IngestParams (so the recorded values are the used
    values, never a stale default), every resolved input including the SAM2
    checkpoint (the old stage omitted it), and the declared ranges that
    tracks_complete.json must eventually cover.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RUN_META
    path.write_text(json.dumps({
        "run_name": run_name,
        "dataset": dataset_name,
        "notes": notes,
        "created": _now(),
        "generator": ("//experimental/overhead_matching/swag/farfield/"
                      "tracking:run_tracking"),
        "git_commit": provenance.git_commit(),
        "argv": list(sys.argv),
        "viewer_rel": ".",
        # None values are recorded deliberately (e.g. video on keyframe-only
        # datasets): "known absent" is information, and
        # paths.recorded_run_inputs skips falsy entries on the read side.
        "inputs": {k: (str(v) if v is not None else None)
                   for k, v in inputs.items()},
        "config": {
            "track_builder": dataclasses.asdict(builder_cfg),
            "ingest": dataclasses.asdict(ingest_params),
        },
        "ranges": [{"name": n, "k_start": a, "k_end": b}
                   for n, a, b in ranges],
        "completion": (f"this file is NOT a completion marker; a range is "
                       f"complete only when it appears in {TRACKS_COMPLETE}"),
    }, indent=1))
    return path


def completed_ranges(run_dir: Path) -> dict:
    """{range_name: finished-timestamp} recorded so far, {} when none."""
    path = Path(run_dir) / TRACKS_COMPLETE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()).get("completed", {})
    except (json.JSONDecodeError, OSError):
        return {}


def mark_range_complete(run_dir: Path, range_name: str) -> Path:
    """Append one finished range to tracks_complete.json (atomic rewrite).

    Called ONLY after the range's tracks_<range>.json has been written; this
    ordering is the whole point of the marker (see the module docstring).
    """
    run_dir = Path(run_dir)
    completed = completed_ranges(run_dir)
    completed[range_name] = _now()
    doc = {
        "schema": "farfield_tracks_complete/v1",
        "note": ("a range listed here has its tracks_<range>.json fully "
                 "written; the run is complete when every range declared in "
                 f"{RUN_META} is listed"),
        "completed": completed,
    }
    path = run_dir / TRACKS_COMPLETE
    # Atomic replace: a crash mid-write must not corrupt the marker that
    # exists precisely to survive crashes.
    fd, tmp = tempfile.mkstemp(dir=run_dir, prefix=TRACKS_COMPLETE + ".")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(doc, indent=1))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def unfinished_ranges(run_dir: Path) -> list:
    """Ranges declared in run_meta.json but absent from tracks_complete.json.

    The orchestrator's (REORG.md PR 12) skip test: a stage is re-runnable
    while this is non-empty, and only an empty answer means done.
    """
    meta_path = Path(run_dir) / RUN_META
    declared = [r["name"] for r in
                json.loads(meta_path.read_text()).get("ranges", [])]
    done = completed_ranges(run_dir)
    return [name for name in declared if name not in done]


def refresh_artifact_manifest(paths, runs_root: Path, run_name: str):
    """Write/refresh `manifest.json` for the object_tracks artifact.

    Refreshed on every run because this artifact is a *working tree* -- its
    version bumps only when the pipeline changes shape, while runs accumulate
    inside it -- so the manifest's job is to name the producer and point at
    the per-run records that carry the settings.
    """
    runs = {}
    for run_dir in sorted(p for p in runs_root.glob("*") if p.is_dir()):
        if not (run_dir / RUN_META).exists():
            continue
        try:
            pending = unfinished_ranges(run_dir)
        except (json.JSONDecodeError, OSError):
            pending = ["<unreadable run_meta>"]
        runs[run_dir.name] = ("complete" if not pending
                              else f"incomplete: {', '.join(pending)}")
    inputs = {
        "dataset_base": paths_lib.relative_to_root(paths.dataset_base,
                                                   paths.root),
        "frame_landmarks": paths_lib.relative_to_root(paths.frame_landmarks,
                                                      paths.root),
    }
    try:
        inputs["video"] = paths_lib.relative_to_root(paths.video, paths.root)
    except paths_lib.MissingInput:
        pass  # keyframe-only dataset
    provenance.write(
        paths.object_tracks,
        generator=("//experimental/overhead_matching/swag/farfield/tracking"
                   ":run_tracking (+ later per-run stages)"),
        inputs=inputs,
        config={
            "per_run": f"runs/<run>/{RUN_META} records the resolved inputs "
                       f"(incl. the SAM2 checkpoint), the full "
                       f"TrackBuilderConfig + ingest params, and the declared "
                       f"ranges; runs/<run>/{TRACKS_COMPLETE} records which "
                       f"ranges actually finished",
            "runs": runs,
            "latest_run": run_name,
        },
        notes=("Working tree of the tracking pipeline: products and their "
               "debug boards travel together, and the internal layout is the "
               "producer's. Immutability lives on the run ids - never "
               "regenerate an rNNN in place with different settings, mint "
               "the next one. Mint v<N+1> if the pipeline changes shape."),
    )


# ---------------------------------------------------------------------------
# Media capture
# ---------------------------------------------------------------------------

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

    def on_interval(self, track, keyframe, crops, origins, masks,
                    previews=None):
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
            # Downscale before doing anything else. Every frame here ends up at
            # VIDEO_SIZE, so blending and annotating at the window's native size
            # (up to 3072px) copies and rewrites ~25x more pixels than survive:
            # measured 11.9 s of full-size `crop.copy()` plus blend and 7.9 s of
            # PIL resize per 20 intervals, for debug media.
            #
            # The backend hands back a GPU-made thumbnail per frame, because any
            # CPU pass over the full crop costs ~4.4 ms whatever the filter --
            # it is memory traffic, not interpolation. Falling back to cv2 keeps
            # this hook usable with a backend that supplies no previews.
            with PROFILE.phase("media_downscale", items=1):
                scale = VIDEO_SIZE / crop.shape[0]
                preview = previews[i] if previews else None
                small = (preview if preview is not None
                         else cv2.resize(crop, (VIDEO_SIZE, VIDEO_SIZE),
                                         interpolation=cv2.INTER_LINEAR))
                if small.shape[0] != VIDEO_SIZE:
                    scale = small.shape[0] / crop.shape[0]
            with PROFILE.phase("media_blend", items=1):
                if mask is not None and mask.any():
                    small_mask = cv2.resize(
                        mask.astype(np.uint8), (VIDEO_SIZE, VIDEO_SIZE),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
                    if small_mask.any():
                        selected = small[small_mask].astype(np.float32)
                        small[small_mask] = (
                            selected * 0.65
                            + np.asarray(MASK_TINT, np.float32) * 0.35
                        ).astype(np.uint8)
            with PROFILE.phase("media_annotate", items=1):
                pil = Image.fromarray(small)
                draw = ImageDraw.Draw(pil)
                if (i == 0 and first_interval
                        and hasattr(track, "_birth_pano_box")):
                    box = self._window_box(track._birth_pano_box, origins[0])
                    draw.rectangle([v * scale for v in box],
                                   outline=(60, 255, 60), width=2)
                if i == n - 1 and rec is not None:
                    for support in rec.get("supports", []):
                        color = CLASS_COLORS.get(support["class"],
                                                 (200, 200, 200))
                        box = [v * scale for v in support["box_window"]]
                        draw.rectangle(box, outline=color, width=2)
                        draw.text((box[0], max(box[1] - 14, 0)),
                                  f"{support['class']} iou={support['iou']:.2f}",
                                  fill=color, font=self.font)
                vc.draw_caption(
                    draw, f"f{keyframe:04d}->f{keyframe + 1:04d}  {action}",
                    self.font)
            with PROFILE.phase("media_save_jpeg", items=1):
                pil.save(track_dir / f"{self.counters[key]:06d}.jpg",
                         quality=88)
            self.counters[key] += 1
            if i == n - 1 or (i == 0 and first_interval):
                with PROFILE.phase("media_thumb", items=1):
                    self._maybe_thumb(key, pil, rec)

    def _window_box(self, pano_box, origin):
        rel = geo.signed_x_offset(pano_box[0], origin[0], self.pano_w)
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
        """Encode one mp4 per track, several at a time.

        Serially this was 883 s of charles' 84 min run -- 423 tracks x ~2.1 s,
        each a process launch plus a single-stream x264 encode of ~40 tiny
        512px frames, with the GPU idle and 31 of 32 cores unused. The encodes
        are independent (own input directory, own output file), so they pool
        cleanly; threads suffice because each worker is blocked in `ffmpeg`.
        """
        jobs = []
        for track_dir in sorted(self.frames_root.iterdir()):
            key = track_dir.name
            n = self.counters.get(key, 0)
            if n == 0:
                continue
            jobs.append((key, track_dir, n))
        if not jobs:
            shutil.rmtree(self.frames_root, ignore_errors=True)
            return {}

        def encode(job):
            key, track_dir, _ = job
            result = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-framerate", str(VIDEO_FPS),
                 "-i", str(track_dir / "%06d.jpg"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
                 str(self.videos_dir / f"{key}.mp4")],
                capture_output=True, text=True)
            return key, result.returncode, result.stderr.strip()

        encoded, failures = {}, []
        # Each ffmpeg already spreads over ~3 cores, so a handful saturates the
        # machine; more would only add contention.
        workers = max(1, min(ENCODE_WORKERS, (os.cpu_count() or 4) // 4))
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            for key, code, stderr in pool.map(encode, jobs):
                if code == 0:
                    encoded[key] = self.counters.get(key, 0)
                else:
                    failures.append((key, code, stderr[:200]))
        if failures:
            # Loud: a missing video is a dead link in the viewer, and the old
            # `check=True` at least stopped the run rather than hiding it.
            print(f"  WARNING: {len(failures)} track video(s) failed to encode")
            for key, code, stderr in failures[:5]:
                print(f"    {key}: ffmpeg exit {code}: {stderr}")
        shutil.rmtree(self.frames_root, ignore_errors=True)
        return encoded


# --------------------------------------------------------------------------
# Explanations


def explain_support(s, cfg, near_miss):
    """cfg is the RECORDED config dict from the range's tracks_*.json."""
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

# Pages go through the one shared farfield.viewers.page helper (one CSS, a
# provenance footer on every page); only the classes specific to this viewer
# live here.
EXTRA_STYLE = """
img,video{max-width:100%}
.cards{display:flex;flex-wrap:wrap;gap:10px}
.card{background:#222;border-radius:6px;padding:8px;width:220px}
.card img{width:204px;height:204px;object-fit:cover;border-radius:4px}
.chip{display:inline-block;padding:1px 7px;border-radius:9px;font-size:12px;
      background:#333}
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
    parts = ["<p><a href='index.html'>&larr; all tracks</a>"
             + (f" | <a href='track_{nav_prev}.html'>&larr; prev</a>"
                if nav_prev else "")
             + (f" | <a href='track_{nav_next}.html'>next &rarr;</a>"
                if nav_next else "") + "</p>",
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
    parts.append("</table>")
    (out / f"track_{key}.html").write_text(page_lib.page(
        f"{key}: {t['modal_label']}", "\n".join(parts),
        generator=GENERATOR, extra_style=EXTRA_STYLE))


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
    paths_lib.add_arguments(parser, video=True, checkpoint=True)
    parser.add_argument("--runs_root", type=Path, default=None,
                        help="default: <object_tracks artifact>/runs "
                             "(requires --object_tracks_version)")
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
    # TODO(run_config): the ingest values move into the run's recorded config
    # (run_config.json, REORG.md PR 12); required on the CLI until then. No
    # defaults on purpose -- these shape which detections exist at all.
    parser.add_argument("--fov_deg", type=float, required=True,
                        help="Pinhole-face FOV the extraction rendered with, "
                             "degrees (previously 90.0 on all datasets)")
    parser.add_argument("--seam_gap_norm", type=float, required=True,
                        help="Seam-merge margin in bbox units 0-1000 "
                             "(previously 25)")
    parser.add_argument("--seam_min_y_iou", type=float, required=True,
                        help="Vertical IoU to accept a seam continuation "
                             "(previously 0.3)")
    args = parser.parse_args()

    paths = paths_lib.resolve(
        parser, args,
        require=("dataset_base", "frame_landmarks", "sam2_checkpoint"))
    ingest_params = dataset.IngestParams(
        fov_deg=args.fov_deg, seam_gap_norm=args.seam_gap_norm,
        seam_min_y_iou=args.seam_min_y_iou)
    # No source video is a mode, not an error: Mapillary datasets retain only
    # the posted keyframes, so the tracker propagates across those directly.
    try:
        video = paths.video
    except paths_lib.MissingInput:
        video = None
        print(f"{paths.dataset}: no source video in metadata; tracking "
              f"across keyframes only")
    ctx = rr.load_context(paths.dataset_base, paths.frame_landmarks,
                          video, paths.sam2_checkpoint, ingest_params,
                          preview_size=VIDEO_SIZE)
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
        # The whole leg. A production stage defaulting to anything less than
        # the full leg silently truncates the run.
        ranges = [("full", 0, last_keyframe)]
    font = vc.load_font(13)
    # The whole config comes from the dataclass defaults and is recorded
    # verbatim (run_meta.json + every tracks_*.json); there is deliberately
    # no per-threshold CLI override. reference_pano_width is set from the
    # actual imagery -- see TrackBuilderConfig's resolution warning.
    builder_cfg = tb.TrackBuilderConfig(reference_pano_width=ctx["pano_w"])
    lane_runs_root = None
    try:
        lane_runs_root = paths.tracks_runs_root
    except paths_lib.MissingInput as exc:
        if args.runs_root is None:
            parser.error(str(exc))
    runs_root = args.runs_root or lane_runs_root
    out = runs_root / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    # Inputs + settings only -- completion lives in tracks_complete.json,
    # written per range below (see the module docstring).
    write_run_meta(
        out, run_name=args.run_name, dataset_name=paths.dataset,
        notes=args.notes,
        inputs={
            "dataset_base": paths.dataset_base,
            "frame_landmarks": paths.frame_landmarks,
            "video": video,
            "sam2_checkpoint": paths.sam2_checkpoint,
        },
        builder_cfg=builder_cfg, ingest_params=ingest_params, ranges=ranges)

    artifacts = {}
    videos = {}
    for range_name, k_start, k_end in ranges:
        art_path = out / f"tracks_{range_name}.json"
        if args.skip_existing_ranges and art_path.exists():
            artifacts[range_name] = json.loads(art_path.read_text())
            videos.update({p.stem: 1 for p in
                           (out / "videos").glob(f"{range_name}_T*.mp4")})
            # The artifact exists and parses, so the range is done; recording
            # it keeps a resumed run's completion marker consistent even when
            # the crash happened between the artifact write and the mark.
            mark_range_complete(out, range_name)
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
        mark_range_complete(out, range_name)
        artifacts[range_name] = artifact
        print("  encoding videos ...")
        videos.update(sink.finalize())

    obs_by_id = ctx["obs_by_id"]

    # obs -> track it seeded (for cross links).
    seeded_by = {}
    for range_name, artifact in artifacts.items():
        for t in artifact["tracks"]:
            seeded_by[t["birth_obs_id"]] = track_key(range_name, t["track_id"])

    # Per-track pages, ordered for prev/next navigation. Each page explains
    # thresholds from ITS range's recorded config, never a fresh default.
    ordered = [(rn, t) for rn, a in artifacts.items() for t in a["tracks"]
               if t["records"]]
    keys = [track_key(rn, t["track_id"]) for rn, t in ordered]
    for i, (rn, t) in enumerate(ordered):
        render_track_page(
            out, rn, t, artifacts[rn]["config"], obs_by_id, seeded_by,
            keys[i - 1] if i > 0 else None,
            keys[i + 1] if i + 1 < len(keys) else None,
            track_key(rn, t["track_id"]) in videos)

    # Index page.
    parts = [f"<script>{GALLERY_JS}</script>"]
    parts.append("<h2>Alive timeline</h2><p>bar = track lifetime, solid "
                 "notches = supported keyframes; click to open</p>")
    for range_name, artifact in artifacts.items():
        parts.append(f"<h3>{html.escape(range_name)}</h3>")
        parts.append(timeline_svg(range_name, artifact))

    # NOTE: the old viewer also chipped tracks by whether they would advance
    # to the semantic audit "at its default bar". That re-derived a decision
    # from a freshly-constructed default AuditConfig -- exactly the stale-
    # default failure mode this migration removes. Audit advancement is
    # decided (and recorded) by the audit stage itself (REORG.md PR 07); this
    # viewer shows only what this stage knows.
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
    parts.append("</div>")
    (out / "index.html").write_text(page_lib.page(
        f"Tracking run {args.run_name}", "\n".join(parts),
        generator=GENERATOR, extra_style=EXTRA_STYLE))
    print(f"wrote {out}/index.html ({len(keys)} track pages, "
          f"{len(videos)} videos)")
    pending = unfinished_ranges(out)
    if pending:
        print(f"WARNING: ranges never finished: {', '.join(pending)} "
              f"(see {TRACKS_COMPLETE})")

    # Only when writing into the artifact lane; a --runs_root pointed elsewhere
    # is scratch and should not claim to be an artifact version.
    if lane_runs_root is not None and runs_root == lane_runs_root:
        refresh_artifact_manifest(paths, runs_root, args.run_name)
        print(f"refreshed {paths.object_tracks}/manifest.json")


if __name__ == "__main__":
    main()
