"""Build one immutable ``object_tracks`` artifact for one configured range.

The production contract is one explicit range named ``full``, one
``tracks_full.json`` payload, and one completed typed manifest. Tracking,
media encoding, and the
static review pages all happen in ``<output_dir>.incomplete``.  Only after
every output is finalized is the manifest written and the directory renamed
atomically into place.  A failed invocation is therefore never reusable as a
completed scientific artifact.
"""

import argparse
import concurrent.futures as cf
import dataclasses
import html
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from experimental.overhead_matching.swag.farfield import (
    artifact as artifact_lib,
    build_config,
    dataset,
    geometry as geo,
    paths as paths_lib,
    provenance,
    publication as publication_lib,
)
from experimental.overhead_matching.swag.farfield.tracking import (
    keyframe_viewer,
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
TRACKS_FILE = "tracks_full.json"
RANGE_NAME = "full"

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
            details = "; ".join(
                f"{key}: ffmpeg exit {code}: {stderr}"
                for key, code, stderr in failures[:5])
            raise RuntimeError(
                f"{len(failures)} track video(s) failed to encode; "
                f"refusing to publish viewer with dead links ({details})")
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
        # Keyframe pages belonged to the retired mutable run workspace and
        # are not outputs of this immutable producer.  Keep the evidence
        # table self-contained instead of publishing a dead cross-artifact
        # link.
        kf_cell = kf
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


class TrackingContractError(ValueError):
    """An explicit input disagrees with the immutable tracking recipe."""


def _exact_keys(value, expected: set[str], where: str) -> None:
    if not isinstance(value, dict):
        raise TrackingContractError(f"{where} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise TrackingContractError(
            f"{where} has missing={missing}, unknown={unknown}")


def _flatten_config(value, prefix: str) -> dict[str, object]:
    if not isinstance(value, dict):
        return {prefix: value}
    if not value:
        return {prefix: value}
    flattened = {}
    for key, child in value.items():
        if not isinstance(key, str) or not key:
            raise TrackingContractError(
                f"{prefix} contains a non-string or empty key")
        flattened.update(_flatten_config(child, f"{prefix}.{key}"))
    return flattened


def orchestration_contract(document: dict) -> dict:
    """Recompute the pipeline's exact track-stage config selection."""
    config = document.get("config")
    if not isinstance(config, dict):
        raise TrackingContractError("build config has no config object")
    selected = {}
    for prefix in ("ingest", "tracking", "gps_course"):
        if prefix not in config:
            raise TrackingContractError(
                f"build config does not record {prefix!r}")
        selected.update(_flatten_config(config[prefix], prefix))
    selected["artifacts.object_tracks_version"] = build_config.value(
        document, "artifacts.object_tracks_version")
    return {
        "schema": "farfield_pipeline_stage/v1",
        "stage": "track",
        "config_digest": artifact_lib.sha256_json(selected),
    }


def _same_path(actual: Path, recorded: str, what: str) -> Path:
    actual = Path(actual)
    recorded_path = Path(recorded)
    resolved = actual.resolve()
    if resolved != recorded_path.resolve():
        raise TrackingContractError(
            f"{what} disagrees with immutable build config: "
            f"{resolved} != {recorded_path.resolve()}")
    return resolved


def load_tracking_config(args):
    """Validate explicit bindings and materialize every scientific setting."""
    config_path = Path(args.build_config)
    if (config_path.name != build_config.BUILD_CONFIG_NAME
            or not config_path.is_file() or config_path.is_symlink()):
        raise TrackingContractError(
            f"--build_config must name a regular, non-symlink "
            f"{build_config.BUILD_CONFIG_NAME} file")
    document = build_config.load(config_path.parent)
    if document["dataset"] != args.dataset:
        raise TrackingContractError(
            "--dataset disagrees with the immutable build config")

    inputs = document["inputs"]
    dataset_base = _same_path(
        args.dataset_base, inputs.get("dataset_base", ""), "--dataset_base")
    if not dataset_base.is_dir() or dataset_base.is_symlink():
        raise TrackingContractError(
            f"--dataset_base must be a regular directory: {dataset_base}")
    metadata = dataset.load_metadata(dataset_base)
    if metadata["dataset_name"] != args.dataset:
        raise TrackingContractError(
            "dataset metadata disagrees with --dataset")
    try:
        dataset_digests = paths_lib.dataset_source_digests(dataset_base)
    except paths_lib.MissingInput as exc:
        raise TrackingContractError(str(exc)) from exc
    mismatched_sources = [
        key for key in paths_lib.DATASET_SOURCE_DIGEST_KEYS
        if inputs.get(key) != dataset_digests[key]
    ]
    if mismatched_sources:
        raise TrackingContractError(
            "dataset source bytes differ from the immutable build recipe: "
            f"{mismatched_sources}")

    configured_checkpoint = build_config.value(
        document, "tracking.sam2_checkpoint")
    checkpoint = _same_path(
        args.checkpoint, configured_checkpoint, "--checkpoint")
    checkpoint = _same_path(
        checkpoint, inputs.get("sam2_checkpoint", ""), "--checkpoint")
    checkpoint_digest = artifact_lib.sha256_file(checkpoint)
    if checkpoint_digest != inputs.get("sam2_checkpoint_sha256"):
        raise TrackingContractError(
            "SAM2 checkpoint content digest disagrees with build config")

    ingest = document["config"].get("ingest")
    _exact_keys(
        ingest, {"fov_deg", "seam_gap_norm", "seam_min_y_iou"},
        "build config ingest")
    ingest_params = dataset.IngestParams(**ingest)

    tracking = document["config"].get("tracking")
    builder_fields = {field.name for field in dataclasses.fields(
        tb.TrackBuilderConfig)}
    _exact_keys(
        tracking, builder_fields | {"sam2_checkpoint", "range"},
        "build config tracking")
    range_config = tracking["range"]
    _exact_keys(range_config, {"k_start", "k_end"},
                "build config tracking.range")
    if (type(range_config["k_start"]) is not int
            or type(range_config["k_end"]) is not int
            or range_config["k_start"] > range_config["k_end"]):
        raise TrackingContractError(
            "tracking.range must contain integer k_start <= k_end")
    if (args.k_start != range_config["k_start"]
            or args.k_end != range_config["k_end"]):
        raise TrackingContractError(
            "--k_start/--k_end disagree with immutable build config")
    builder_cfg = tb.TrackBuilderConfig(**{
        field.name: tracking[field.name]
        for field in dataclasses.fields(tb.TrackBuilderConfig)
    })

    course = document["config"].get("gps_course")
    _exact_keys(course, {"min_displacement_m", "smooth_window_s"},
                "build config gps_course")
    for name, allow_zero in (("min_displacement_m", False),
                             ("smooth_window_s", True)):
        value = course[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0.0
                or (not allow_zero and value == 0.0)):
            qualifier = "nonnegative" if allow_zero else "positive"
            raise TrackingContractError(
                f"gps_course.{name} must be finite and {qualifier}")

    output_version = build_config.value(
        document, "artifacts.object_tracks_version")
    if Path(args.output_dir).name != output_version:
        raise TrackingContractError(
            f"--output_dir must end in configured version {output_version!r}")
    orchestration = orchestration_contract(document)
    if args.orchestration_config_digest != orchestration["config_digest"]:
        raise TrackingContractError(
            "--orchestration_config_digest does not match the immutable "
            "track-stage config selection")

    pinhole_ref = artifact_lib.open_artifact(
        args.pinhole_dir,
        expected_kind=paths_lib.PINHOLE_IMAGES,
        expected_dataset=args.dataset,
        expected_version=build_config.value(
            document, "artifacts.pinhole_images_version"))
    frame_ref = artifact_lib.open_artifact(
        args.frame_landmarks_dir,
        expected_kind=paths_lib.FRAME_LANDMARKS,
        expected_dataset=args.dataset,
        expected_version=build_config.value(
            document, "artifacts.frame_landmarks_version"))
    pinhole_manifest = artifact_lib.load_manifest(args.pinhole_dir)
    pinhole_input_digests = pinhole_manifest.config.get("input_digests")
    expected_extraction_digests = {
        "pipeline_metadata": dataset_digests[
            paths_lib.DATASET_PIPELINE_METADATA_SHA256],
        "frames_gps": dataset_digests[paths_lib.DATASET_FRAMES_GPS_SHA256],
        "panorama_directory": dataset_digests[
            paths_lib.DATASET_PANORAMA_SHA256],
    }
    if (not isinstance(pinhole_input_digests, dict)
            or any(pinhole_input_digests.get(key) != value
                   for key, value in expected_extraction_digests.items())):
        raise TrackingContractError(
            "pinhole artifact does not bind the current frozen dataset sources")
    frame_manifest = artifact_lib.load_manifest(args.frame_landmarks_dir)
    if frame_manifest.config.get(
            "build_identity") != document["build_identity"]:
        raise TrackingContractError(
            "frame_landmarks artifact belongs to a different immutable "
            "build identity")
    if frame_manifest.upstreams.count(pinhole_ref) != 1:
        raise TrackingContractError(
            "frame_landmarks does not bind the exact pinhole artifact once")

    video = Path(args.video) if args.video is not None else None
    video_digest = None
    if video is not None:
        video = _same_path(video, inputs.get("video", ""), "--video")
        video_digest = artifact_lib.sha256_file(video)
        if video_digest != inputs.get("video_sha256"):
            raise TrackingContractError(
                "source video content digest disagrees with build config")
    elif "video_sha256" in inputs:
        raise TrackingContractError(
            "immutable build config records a source video but --video is absent")
    return {
        "document": document,
        "build_config_sha256": artifact_lib.sha256_file(config_path),
        "dataset_base": dataset_base,
        "dataset_source_sha256": artifact_lib.sha256_json(dataset_digests),
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_digest,
        "video": video,
        "video_sha256": video_digest,
        "ingest_params": ingest_params,
        "builder_cfg": builder_cfg,
        "course": dict(course),
        "output_version": output_version,
        "orchestration": orchestration,
        # Pipeline dependency order is part of the artifact contract.
        "upstreams": (pinhole_ref, frame_ref),
    }


def dataset_tracking_source_digest(dataset_base: Path) -> str:
    """Bind the raw metadata/GPS/panorama bytes actually read by tracking."""
    try:
        return artifact_lib.sha256_json(
            paths_lib.dataset_source_digests(dataset_base))
    except paths_lib.MissingInput as exc:
        raise TrackingContractError(str(exc)) from exc


def _seal_staged_outputs(publication) -> None:
    """Freeze the dynamic viewer/media filenames before manifest publication."""
    outputs = []
    for path in publication.staging_dir.rglob("*"):
        if path.is_symlink():
            raise TrackingContractError(
                f"tracking output contains a symlink: {path}")
        if path.is_file():
            outputs.append(path.relative_to(
                publication.staging_dir).as_posix())
    required = {TRACKS_FILE, "index.html"}
    missing = sorted(required - set(outputs))
    if missing:
        raise TrackingContractError(
            f"tracking did not finalize required outputs: {missing}")
    # Track ids, and therefore page/media filenames, become known only after
    # the GPU run.  Seal the builder's public declaration immediately before
    # publish; ArtifactDirectoryBuilder still verifies the exact file set,
    # writes the manifest last, validates it, and performs the atomic rename.
    publication.declared_outputs = tuple(sorted(outputs))


def validate_tracks_document(value: dict, *, expected_range: dict,
                             expected_config: dict) -> None:
    """Reject a partial or stale range-runner payload before publication."""
    _exact_keys(
        value,
        {"range", "config", "tracks", "rejected_births", "track_overlaps"},
        "tracks_full.json")
    if value["range"] != expected_range:
        raise TrackingContractError(
            "range runner returned a payload for a different range")
    if value["config"] != expected_config:
        raise TrackingContractError(
            "range runner returned a payload with different tracking config")
    for name in ("tracks", "rejected_births", "track_overlaps"):
        if not isinstance(value[name], list):
            raise TrackingContractError(
                f"tracks_full.json.{name} must be a list")
    track_ids = [track.get("track_id") for track in value["tracks"]
                 if isinstance(track, dict)]
    if (len(track_ids) != len(value["tracks"])
            or any(type(track_id) is not int for track_id in track_ids)
            or len(track_ids) != len(set(track_ids))):
        raise TrackingContractError(
            "tracks_full.json tracks must have unique integer track_id values")
    for track in value["tracks"]:
        records = track.get("records")
        if not isinstance(records, list) or not records:
            raise TrackingContractError(
                "tracks_full.json track "
                f"{track['track_id']} has no records")


def render_viewer(out: Path, dataset_name: str, tracks_document: dict,
                  obs_by_id: dict, videos: dict) -> int:
    """Write every static page after media finalization; return page count."""
    range_name = RANGE_NAME
    tracks = tracks_document["tracks"]
    seeded_by = {
        track["birth_obs_id"]: track_key(range_name, track["track_id"])
        for track in tracks
    }
    ordered = [track for track in tracks if track["records"]]
    keys = [track_key(range_name, track["track_id"]) for track in ordered]
    for index, track in enumerate(ordered):
        render_track_page(
            out, range_name, track, tracks_document["config"], obs_by_id,
            seeded_by, keys[index - 1] if index > 0 else None,
            keys[index + 1] if index + 1 < len(keys) else None,
            track_key(range_name, track["track_id"]) in videos)

    parts = [f"<script>{GALLERY_JS}</script>",
             "<h2>Alive timeline</h2><p>bar = track lifetime, solid "
             "notches = supported keyframes; click to open</p>",
             timeline_svg(range_name, tracks_document),
             "<h2>Tracks</h2><div class='controls'>"
             "<input id='f_range' type='hidden' value='all'>"
             "status <select id='f_status' onchange='applyFilters()'>"
             "<option value='all'>all</option><option value='alive'>alive"
             "</option><option value='closed'>closed</option></select>"
             " min supported <input id='f_minsup' type='number' value='0'"
             " style='width:60px' oninput='applyFilters()'></div>",
             "<div class='cards'>"]
    cards = sorted(
        ((track["n_supported_keyframes"], track) for track in ordered),
        key=lambda item: -item[0])
    for supported, track in cards:
        key = track_key(range_name, track["track_id"])
        status = (track["close_reason"]
                  if track["status"] == "closed" else "alive")
        preview = (
            f"<img src='thumbs/{key}.jpg' loading='lazy'>"
            if (out / "thumbs" / f"{key}.jpg").is_file()
            else "<span class='chip'>no preview</span>")
        parts.append(
            f"<div class='card' data-range='{range_name}' "
            f"data-status='{status}' data-sup='{supported}'>"
            f"<a href='track_{key}.html'>{preview}</a><br>"
            f"<b>T{track['track_id']}</b> "
            f"{html.escape(track['modal_label'][:40])}<br>"
            f"<small>b=f{track['birth_keyframe']:04d} | sup={supported}"
            f"</small><br>{status_chip(track)}</div>")
    parts.append("</div><h2>Landmarks that produced no track</h2>"
                 "<p>Rejected at birth (mask health) or never supported "
                 "after birth.</p><div class='cards'>")
    rejected_ids = {
        item["obs_id"] for item in tracks_document["rejected_births"]
    }
    for track in ordered:
        is_reject = (track["birth_obs_id"] in rejected_ids
                     and track["close_reason"].startswith("birth_"))
        never_supported = (track["n_supported_keyframes"] == 0
                           and track["status"] == "closed")
        if not (is_reject or never_supported):
            continue
        key = track_key(range_name, track["track_id"])
        observation = obs_by_id.get(track["birth_obs_id"])
        label = (vc.obs_semantic_label(observation)
                 if observation else track["birth_obs_id"])
        preview = (
            f"<img src='thumbs/{key}.jpg' loading='lazy'>"
            if (out / "thumbs" / f"{key}.jpg").is_file()
            else "<span class='chip'>no preview</span>")
        parts.append(
            f"<div class='card' data-range='{range_name}' "
            f"data-status='{track['close_reason']}' data-sup='0'>"
            f"<a href='track_{key}.html'>{preview}</a><br>"
            f"{html.escape(label[:48])}<br>"
            f"<small>{html.escape(track['birth_obs_id'])}</small><br>"
            f"{status_chip(track)}</div>")
    parts.append("</div>")
    (out / "index.html").write_text(page_lib.page(
        f"Object tracks: {dataset_name}", "\n".join(parts),
        generator=GENERATOR, extra_style=EXTRA_STYLE))
    return len(keys)


def publish_tracking(args, *, arguments: tuple[str, ...] = ()):
    resolved = load_tracking_config(args)
    builder_cfg = resolved["builder_cfg"]
    source_digests = {
        "build_config": resolved["build_config_sha256"],
        "dataset_tracking_inputs": resolved["dataset_source_sha256"],
        "sam2_checkpoint": resolved["checkpoint_sha256"],
        paths_lib.PINHOLE_IMAGES: resolved["upstreams"][0].content_digest,
        paths_lib.FRAME_LANDMARKS: resolved["upstreams"][1].content_digest,
    }
    if resolved["video_sha256"] is not None:
        source_digests["video"] = resolved["video_sha256"]
    manifest_config = {
        "orchestration": resolved["orchestration"],
        "schema": "farfield_object_tracks/v1",
        "coverage": "complete",
        "build_identity": resolved["document"]["build_identity"],
        "range": {"name": RANGE_NAME, "k_start": args.k_start,
                  "k_end": args.k_end},
        "resolved": {
            "ingest": dataclasses.asdict(resolved["ingest_params"]),
            "tracking": {
                **dataclasses.asdict(builder_cfg),
                "sam2_checkpoint": str(resolved["checkpoint"].resolve()),
            },
            "gps_course": resolved["course"],
        },
        "source_digests": source_digests,
    }
    with publication_lib.published_artifact(
            args.output_dir,
            kind=paths_lib.OBJECT_TRACKS,
            dataset=args.dataset,
            version=resolved["output_version"],
            generator=GENERATOR,
            git_commit=provenance.git_commit(),
            arguments=arguments,
            upstreams=resolved["upstreams"],
            config=manifest_config,
            declared_outputs=()) as publication:
        out = publication.staging_dir
        ctx = rr.load_context(
            resolved["dataset_base"], Path(resolved["upstreams"][1].path),
            resolved["video"], resolved["checkpoint"],
            resolved["ingest_params"],
            course_min_displacement_m=(
                resolved["course"]["min_displacement_m"]),
            course_smooth_window_s=resolved["course"]["smooth_window_s"],
            preview_size=VIDEO_SIZE)
        frames = ctx["result"].frames
        frame_indices = {frame.frame_idx for frame in frames}
        expected_indices = set(range(args.k_start, args.k_end + 1))
        missing_indices = sorted(expected_indices - frame_indices)
        if missing_indices:
            raise TrackingContractError(
                f"configured tracking range is not covered by the dataset; "
                f"missing keyframes {missing_indices[:10]}")
        if builder_cfg.reference_pano_width != ctx["pano_w"]:
            raise TrackingContractError(
                "tracking.reference_pano_width disagrees with source "
                f"panoramas: {builder_cfg.reference_pano_width} != "
                f"{ctx['pano_w']}")
        font = vc.load_font(13)
        print(f"range {RANGE_NAME}: f{args.k_start:04d}..f{args.k_end:04d}")
        sink = MediaSink(out, RANGE_NAME, font, ctx["pano_w"])
        _, tracks_document = rr.run_range(
            RANGE_NAME, args.k_start, args.k_end, builder_cfg,
            ctx["backend"], ctx["provider"], ctx["model"], ctx["result"],
            ctx["obs_by_frame"], ctx["det_pano_boxes"], ctx["pano_w"],
            ctx["pano_h"], resolved["dataset_base"],
            on_interval=sink.on_interval)
        validate_tracks_document(
            tracks_document,
            expected_range=manifest_config["range"],
            expected_config=dataclasses.asdict(builder_cfg))
        artifact_lib.atomic_write_json(out / TRACKS_FILE, tracks_document)
        print("  encoding videos ...")
        videos = sink.finalize()
        page_count = render_viewer(
            out, args.dataset, tracks_document, ctx["obs_by_id"], videos)
        _seal_staged_outputs(publication)
    print(f"published {args.output_dir} ({page_count} track pages, "
          f"{len(videos)} videos)")
    assert publication.artifact_ref is not None
    return publication.artifact_ref


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame_landmarks_dir", type=Path, required=True)
    parser.add_argument("--pinhole_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_base", type=Path, required=True)
    parser.add_argument("--k_start", type=int, required=True)
    parser.add_argument("--k_end", type=int, required=True)
    parser.add_argument("--build_config", type=Path, required=True)
    parser.add_argument("--orchestration_config_digest", required=True)
    parser.add_argument("--video", type=Path, default=None)
    return parser


def publish_viewer_sidecar(args, tracks_ref):
    """Best-effort derived viewer publication after scientific tracking.

    The viewer is deliberately a separate artifact.  A rendering failure
    therefore never rolls back or contaminates the already-complete tracking
    artifact, and viewer-only code changes never invalidate scientific work.
    """
    viewer_args = argparse.Namespace(
        tracks_dir=Path(tracks_ref.path),
        dataset_base=args.dataset_base,
        frame_landmarks_dir=args.frame_landmarks_dir,
        output_dir=None,
        pano_width=3072,
        kf_start=None,
        kf_end=None,
        image_workers=keyframe_viewer.IMAGE_WORKERS,
    )
    try:
        return keyframe_viewer.publish_viewer(
            viewer_args,
            arguments=(keyframe_viewer.GENERATOR,
                       "--automatic_from_tracking", str(tracks_ref.path)))
    except (Exception, SystemExit) as error:  # Derived pages are non-scientific.
        print("WARNING: tracking is complete, but automatic keyframe viewer "
              f"publication failed: {error}", file=sys.stderr)
        return None


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    arguments = tuple(sys.argv if argv is None
                      else [GENERATOR, *(str(value) for value in argv)])
    try:
        reference = publish_tracking(args, arguments=arguments)
        publish_viewer_sidecar(args, reference)
        return reference
    except (artifact_lib.ArtifactError, dataset.ContractViolation,
            TrackingContractError, FileExistsError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
