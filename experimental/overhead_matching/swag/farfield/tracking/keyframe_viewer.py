"""Per-keyframe detection viewer.

One page per keyframe showing the full annotated panorama - every detection
box (colored by semantic identity, labeled) plus the mask bbox of every
track alive at that keyframe (red, labeled T##) - and a table with a zoom
chip per detection: its tags, description, and which track (if any) absorbed
it and as what support class.

This is the ground level of the viewer hierarchy (leg -> track -> keyframe):
track pages and the semantic-audit review link into these pages so any
detection mentioned elsewhere can be inspected in full context.

Reads one completed ``object_tracks`` artifact with its required
``tracks_full.json`` payload. Ingest and tracking settings come from that
artifact's recorded manifest; the viewer has no scientific-setting flags.

Outputs a separate derived directory and never mutates the tracks artifact.

Run:
  bazel run //experimental/overhead_matching/swag/farfield/tracking:keyframe_viewer -- \\
      --tracks_dir <object_tracks> --dataset_base <dataset> \\
      --frame_landmarks_dir <frame_landmarks> --output_dir <derived>
"""

import argparse
import concurrent.futures as cf
import html
import json
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experimental.overhead_matching.swag.farfield import (
    artifact as artifact_lib,
    dataset,
    geometry as geo,
    paths as paths_lib,
)
from experimental.overhead_matching.swag.farfield.tracking import (
    track_builder as tb,
    viz_common as vc,
)
from experimental.overhead_matching.swag.farfield.tracking.perf_profile import (
    PROFILE,
)
from experimental.overhead_matching.swag.farfield.viewers import (
    page as page_lib,
)

GENERATOR = ("//experimental/overhead_matching/swag/farfield/tracking"
             ":keyframe_viewer")

MASK_COLOR = (255, 60, 60)
IMAGE_WORKERS = 12
CHIP_H = 200

# Pages go through the one shared farfield.viewers.page helper (one CSS, a
# provenance footer on every page); only the classes specific to this viewer
# live here.
EXTRA_STYLE = """
.panowrap{overflow-x:auto;border:1px solid #333;border-radius:4px}
.panowrap img{display:block}
td img{height:200px;border-radius:3px;display:block}
.kf{color:#89a;font-weight:bold}
.cls_context{color:#99b}.cls_none{color:#777}
.seeds{color:#fd6;font-weight:bold}
.cls_continue_clean,.cls_merge_superset,.cls_weak,.cls_split_child{color:#3c3}
.masklab{color:#e55}
"""


@dataclass(frozen=True)
class ViewerInputs:
    tracks_ref: artifact_lib.ArtifactRef
    artifacts: dict
    ingest_params: dataset.IngestParams
    dataset_base: Path
    frame_landmarks_dir: Path


def _load_tracks_contract(tracks_dir: Path):
    tracks_dir = Path(tracks_dir)
    try:
        tracks_ref = artifact_lib.open_artifact(
            tracks_dir, expected_kind=paths_lib.OBJECT_TRACKS)
        manifest = artifact_lib.load_manifest(tracks_dir)
    except artifact_lib.ArtifactError as exc:
        raise SystemExit(
            f"invalid completed object_tracks artifact {tracks_dir}: {exc}") \
            from exc
    if (manifest.config.get("schema") != "farfield_object_tracks/v1"
            or manifest.config.get("coverage") != "complete"):
        raise SystemExit(
            f"{tracks_dir} is not a complete farfield_object_tracks/v1 artifact")
    payloads = sorted(tracks_dir.glob("tracks_*.json"))
    if [path.name for path in payloads] != ["tracks_full.json"]:
        raise SystemExit(
            f"{tracks_dir} must contain exactly one tracks_full.json payload")
    try:
        document = json.loads(payloads[0].read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid tracks_full.json: {exc}") from exc
    if not isinstance(document, dict):
        raise SystemExit("tracks_full.json must contain a JSON object")
    range_record = document.get("range")
    if (not isinstance(range_record, dict)
            or range_record.get("name") != "full"
            or range_record != manifest.config.get("range")):
        raise SystemExit(
            "tracks_full.json must bind the manifest's single full range")
    return tracks_ref, manifest, {"full": document}


def load_track_artifacts(tracks_dir: Path) -> dict:
    """Load the single payload from a validated ``object_tracks`` artifact."""
    return _load_tracks_contract(tracks_dir)[2]


def load_viewer_inputs(tracks_dir: Path, dataset_base: Path,
                       frame_landmarks_dir: Path) -> ViewerInputs:
    tracks_ref, manifest, artifacts = _load_tracks_contract(tracks_dir)
    dataset_base = Path(dataset_base)
    if dataset_base.is_symlink() or not dataset_base.is_dir():
        raise SystemExit(
            f"--dataset_base must be a regular directory: {dataset_base}")
    dataset_base = dataset_base.resolve()
    try:
        source_digest = artifact_lib.sha256_json(
            paths_lib.dataset_source_digests(dataset_base))
    except paths_lib.MissingInput as exc:
        raise SystemExit(str(exc)) from exc
    expected_source = (manifest.config.get("source_digests") or {}).get(
        "dataset_tracking_inputs")
    if source_digest != expected_source:
        raise SystemExit(
            "--dataset_base bytes do not match the object_tracks artifact")
    metadata = dataset.load_metadata(dataset_base)
    if metadata.get("dataset_name") != tracks_ref.dataset:
        raise SystemExit(
            "--dataset_base metadata names a different dataset")

    frame_refs = tuple(
        ref for ref in manifest.upstreams
        if ref.kind == paths_lib.FRAME_LANDMARKS)
    if len(frame_refs) != 1:
        raise SystemExit(
            "object_tracks must bind exactly one frame_landmarks artifact")
    try:
        supplied_frame_ref = artifact_lib.open_artifact(
            frame_landmarks_dir,
            expected_kind=paths_lib.FRAME_LANDMARKS,
            expected_dataset=tracks_ref.dataset,
            expected_version=frame_refs[0].version)
    except artifact_lib.ArtifactError as exc:
        raise SystemExit(
            f"invalid --frame_landmarks_dir artifact: {exc}") from exc
    if supplied_frame_ref != frame_refs[0]:
        raise SystemExit(
            "--frame_landmarks_dir is not the exact artifact bound by tracks")

    resolved = manifest.config.get("resolved")
    ingest = resolved.get("ingest") if isinstance(resolved, dict) else None
    expected_ingest_keys = {"fov_deg", "seam_gap_norm", "seam_min_y_iou"}
    if not isinstance(ingest, dict) or set(ingest) != expected_ingest_keys:
        raise SystemExit(
            "object_tracks manifest has no exact recorded ingest configuration")
    try:
        ingest_params = dataset.IngestParams(**ingest)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"object_tracks recorded invalid ingest configuration: {exc}") \
            from exc
    return ViewerInputs(
        tracks_ref=tracks_ref,
        artifacts=artifacts,
        ingest_params=ingest_params,
        dataset_base=dataset_base,
        frame_landmarks_dir=Path(frame_landmarks_dir).resolve(),
    )


def prepare_output_directory(tracks_dir: Path,
                             output_dir: Path) -> tuple[Path, Path]:
    tracks_dir = Path(tracks_dir).resolve()
    output_dir = Path(output_dir)
    resolved_output = output_dir.resolve(strict=False)
    if (resolved_output == tracks_dir or tracks_dir in resolved_output.parents):
        raise SystemExit(
            "--output_dir must not be the tracks artifact or a directory inside it")
    if output_dir.exists() or output_dir.is_symlink():
        raise SystemExit(f"--output_dir already exists: {output_dir}")
    if output_dir.parent.is_symlink():
        raise SystemExit(
            f"--output_dir parent must not be a symlink: {output_dir.parent}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(output_dir.name + ".incomplete")
    if staging.exists() or staging.is_symlink():
        raise SystemExit(f"incomplete viewer output already exists: {staging}")
    staging.mkdir()
    return output_dir, staging


def recorded_config(artifact) -> tb.TrackBuilderConfig:
    """The TrackBuilderConfig this artifact was BUILT with.

    Reconstructed from the dict the tracking stage stored via
    dataclasses.asdict. A fresh configuration could classify supports under
    different thresholds, so the recorded values are mandatory.
    """
    config = artifact.get("config")
    if not config:
        raise SystemExit(
            "tracking artifact records no 'config'; the viewer classifies "
            "supports under the RECORDED thresholds and refuses to substitute "
            "defaults")
    return tb.TrackBuilderConfig(**config)


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


def track_associations(artifacts: dict):
    """Returns (by_obs, masks, seeded, rejected), merged across all ranges.

    Track references are "<range>_T<id>" keys (matching the track pages), so
    two ranges may reuse a track_id without colliding. Support classes are
    recomputed under EACH artifact's recorded config.

    by_obs:   (keyframe, obs_id) -> [(track_key, effective_class)]
    masks:    keyframe -> [(track_key, action, mask_box_pano)]
    seeded:   obs_id -> [track_key it founded]. A birth is NOT recorded as a
              support entry, so without this a detection that created a
              track looks unclaimed on the page.
    rejected: obs_id -> health dict for births the birth gate refused.
    """
    by_obs = defaultdict(list)
    masks = defaultdict(list)
    seeded = defaultdict(list)
    rejected = {}
    for range_name, artifact in artifacts.items():
        classifier_cfg = recorded_config(artifact)
        rejected.update({r["obs_id"]: r.get("health", {})
                         for r in artifact.get("rejected_births", [])})
        for t in artifact["tracks"]:
            key = f"{range_name}_T{t['track_id']}"
            seeded[t["birth_obs_id"]].append(key)
            for rec in t["records"]:
                mb = rec.get("mask_bbox_window")
                if mb is not None:
                    ox, oy = rec["window_origin"]
                    masks[rec["keyframe"]].append(
                        (key, rec["action"],
                         (ox + mb[0], oy + mb[1], ox + mb[2], oy + mb[3])))
                for s in rec.get("supports", []):
                    eff = tb.classify_support(
                        {"iou": s["iou"],
                         "inter_over_mask": s["inter_over_mask"],
                         "inter_over_box": s["inter_over_box"]},
                        classifier_cfg)
                    by_obs[(rec["keyframe"], s["obs_id"])].append((key, eff))
    return by_obs, masks, dict(seeded), rejected


def kf_name(idx):
    """Page/file stem for a keyframe. Module scope because the pooled image
    renderer names its outputs with it too."""
    return f"f{idx:04d}"


_THREAD_LOCAL = threading.local()


def _worker_font():
    """A font per thread. PIL's FreeTypeFont is not documented thread-safe, and
    a shared one is the kind of thing that corrupts glyphs under load rather
    than failing outright."""
    font = getattr(_THREAD_LOCAL, "font", None)
    if font is None:
        font = _THREAD_LOCAL.font = vc.load_font(14)
    return font


def _render_task(task):
    """Pool entry point. Top-level (not a lambda) so it can be pickled to a
    worker process."""
    return render_keyframe_images(*task)


def render_keyframe_images(frame, obs_list, masks, dataset_base, out,
                           pano_width):
    """Everything image-shaped for one keyframe: the annotated panorama and one
    chip per detection.

    Split out of the page loop so it can run in a pool. Serially this stage was
    631 s for charles' 514 keyframes (1.23 s each), nearly all of it decoding a
    7680x3840 JPEG and writing derived images -- work that is per-keyframe
    independent and releases the GIL inside PIL, while 31 of 32 cores idled.
    The HTML that follows needs only the counts, so it stays serial and cheap.
    """
    kf = frame.frame_idx
    font = _worker_font()
    with PROFILE.phase("kf_pano_decode", items=1):
        pano = np.asarray(Image.open(
            dataset_base / "panorama" / f"{frame.pano_stem}.jpg"))
    pano_h, pano_w = pano.shape[:2]
    scale = pano_width / pano_w
    with PROFILE.phase("kf_pano_annotate", items=1):
        anno = Image.fromarray(pano).resize(
            (pano_width, int(pano_h * scale)), Image.BILINEAR)
        draw = ImageDraw.Draw(anno)
        for o in obs_list:
            box = geo.pano_bbox_for_observation(o.boxes, pano_w, pano_h)
            draw_wrapped_rect(draw, box, scale, pano_width,
                              vc.obs_color(o), 2, vc.obs_semantic_label(o),
                              font)
        for track_key, _action, box in masks:
            label = "T" + track_key.split("_T")[-1]
            draw_wrapped_rect(draw, box, scale, pano_width, MASK_COLOR,
                              2, label, font)
    with PROFILE.phase("kf_pano_save", items=1):
        anno.save(out / "img" / f"{kf_name(kf)}_pano.jpg", quality=85)
    with PROFILE.phase("kf_chips", items=len(obs_list)):
        for o in obs_list:
            box = geo.pano_bbox_for_observation(o.boxes, pano_w, pano_h)
            vc.render_chip(pano, box, None,
                           out / "img" / f"{kf_name(kf)}_{o.obs_id}.jpg",
                           CHIP_H)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks_dir", type=Path, required=True)
    parser.add_argument("--dataset_base", type=Path, required=True)
    parser.add_argument("--frame_landmarks_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--pano_width", type=int, default=3072)
    parser.add_argument("--kf_start", type=int, default=None)
    parser.add_argument("--kf_end", type=int, default=None)
    parser.add_argument("--image_workers", type=int, default=IMAGE_WORKERS,
                        help="threads rendering keyframe images (default: "
                             f"{IMAGE_WORKERS}); 1 to render serially")
    return parser


def main():
    args = build_parser().parse_args()
    inputs = load_viewer_inputs(
        args.tracks_dir, args.dataset_base, args.frame_landmarks_dir)
    artifacts = inputs.artifacts
    result = dataset.run_ingest(
        inputs.dataset_base, inputs.frame_landmarks_dir, inputs.ingest_params)
    obs_by_frame = defaultdict(list)
    for o in result.observations:
        obs_by_frame[o.frame_idx].append(o)
    frames = sorted(result.frames, key=lambda f: f.frame_idx)
    if args.kf_start is not None:
        frames = [f for f in frames if f.frame_idx >= args.kf_start]
    if args.kf_end is not None:
        frames = [f for f in frames if f.frame_idx <= args.kf_end]

    assoc_by_obs, masks_by_kf, seeded_by_obs, rejected_births = \
        track_associations(artifacts)

    final_out, out = prepare_output_directory(
        args.tracks_dir, args.output_dir)
    tracks_href = html.escape(os.path.relpath(
        Path(args.tracks_dir).resolve(), final_out.resolve(strict=False)),
        quote=True)
    (out / "img").mkdir(parents=True, exist_ok=True)

    index_rows = []
    per_kf = [(frame,
               sorted(obs_by_frame.get(frame.frame_idx, []),
                      key=lambda o: o.obs_id),
               sorted(masks_by_kf.get(frame.frame_idx, [])))
              for frame in frames]
    workers = max(1, min(args.image_workers, os.cpu_count() or 4))
    tasks = [(frame, obs_list, masks, inputs.dataset_base, out, args.pano_width)
             for frame, obs_list, masks in per_kf]
    print(f"rendering {len(tasks)} keyframe image sets over {workers} "
          f"process(es)")
    if workers == 1:
        for task in tasks:
            _render_task(task)
    else:
        # Each task performs an independent panorama decode and derived-image
        # write. Processes let those CPU-heavy tasks run in separate interpreters.
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(_render_task, tasks, chunksize=4):
                pass

    for n, (frame, obs_list, masks) in enumerate(per_kf):
        kf = frame.frame_idx
        prev_kf = kf_name(frames[n - 1].frame_idx) if n else None
        next_kf = kf_name(frames[n + 1].frame_idx) if n + 1 < len(frames) \
            else None
        parts = [
            "<p><a href='index.html'>&larr; all keyframes</a>"
            + (f" | <a href='{prev_kf}.html'>&larr; {prev_kf}</a>"
               if prev_kf else "")
            + (f" | <a href='{next_kf}.html'>{next_kf} &rarr;</a>"
               if next_kf else "")
            + f" | <a href='{tracks_href}/index.html'>track boards</a></p>",
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
            for seeded_key in seeded_by_obs.get(o.obs_id, []):
                lines.append(
                    f"<a href='{tracks_href}/track_{seeded_key}.html' "
                    f"class='seeds'>&#9733; seeds "
                    f"T{seeded_key.split('_T')[-1]}</a>")
            if o.obs_id in rejected_births:
                reason = rejected_births[o.obs_id].get("reason", "?")
                lines.append("<span class='cls_none'>birth rejected "
                             f"({html.escape(reason)})</span>")
            # Supports first, then classes that were considered and refused.
            assoc = assoc_by_obs.get((kf, o.obs_id), [])
            real = [(k, c) for k, c in assoc if c in tb.SUPPORT_CLASSES]
            other = [(k, c) for k, c in assoc
                     if c not in tb.SUPPORT_CLASSES]
            for track_key, cls in real + other:
                note = "" if cls in tb.SUPPORT_CLASSES else " (not support)"
                tid = track_key.split("_T")[-1]
                lines.append(
                    f"<a href='{tracks_href}/track_{track_key}.html'>T{tid}</a> "
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
            for track_key, action, box in masks:
                bb = ", ".join(f"{v:.0f}" for v in box)
                tid = track_key.split("_T")[-1]
                parts.append(
                    f"<tr id='{track_key}'><td class='masklab'>"
                    f"<a href='{tracks_href}/track_{track_key}.html'>T{tid}</a>"
                    "</td>"
                    f"<td>{action}</td><td>{bb}</td></tr>")
            parts.append("</table>")
        (out / f"{kf_name(kf)}.html").write_text(page_lib.page(
            kf_name(kf), "\n".join(parts), generator=GENERATOR,
            extra_style=EXTRA_STYLE))
        index_rows.append((kf, len(obs_list), len(masks)))
        if n % 50 == 0:
            print(f"[{n + 1}/{len(frames)}] {kf_name(kf)}")

    parts = [
        f"<p><a href='{tracks_href}/index.html'>track boards</a></p>",
        "<table><tr><th>keyframe</th><th>detections</th>"
        "<th>track masks</th></tr>"]
    for kf, n_obs, n_masks in index_rows:
        parts.append(f"<tr><td class='kf'><a href='{kf_name(kf)}.html'>"
                     f"{kf_name(kf)}</a></td><td>{n_obs}</td>"
                     f"<td>{n_masks}</td></tr>")
    parts.append("</table>")
    (out / "index.html").write_text(page_lib.page(
        "keyframes", "\n".join(parts), generator=GENERATOR,
        extra_style=EXTRA_STYLE))
    artifact_lib.publish_directory_no_clobber(out, final_out)
    print(f"wrote {len(index_rows)} keyframe pages to {final_out}")


if __name__ == "__main__":
    main()
