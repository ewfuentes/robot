"""M2 spike: SAM2 video propagation of VLM landmark boxes across one
keyframe interval, inside heading-compensated pano windows.

For each test case:
- cut a square window around the anchor detection at keyframe k,
  heading-compensated per intermediate video frame (M1),
- prompt SAM2's video predictor with the detection box at frame 0,
- propagate through the ~11 intermediate 3 fps frames to keyframe k+1,
- compare the propagated mask against every detection at k+1
  (IoU + containment both ways -> the association signal for M3),
- render a mask-overlay filmstrip for eyeballing.

SAM3 note: interface is deliberately narrow (frames dir in, per-frame masks
out) so the tracker backend can be swapped for SAM3 when checkpoint access
lands.

Run:
  bazel run //experimental/overhead_matching/swag/landmark_filtering/object_tracking:m2_sam_tracking
"""

import argparse
import html
import json
import tempfile
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401  (must precede torch)
import torch

import numpy as np
from PIL import Image, ImageDraw

from experimental.overhead_matching.swag.landmark_filtering import ingest
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    heading as heading_mod,
    pano_geometry as pg,
    video_frames,
    viz_common as vc,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    IngestConfig,
)

DEFAULT_DATASET = Path("/data/farfield_matching/datasets/boston_harbor_leg1")
DEFAULT_LANDMARKS = Path(
    "/data/farfield_matching/artifacts/frame_landmarks/boston_harbor_leg1/v1")
DEFAULT_VIDEO = Path(
    "/data/farfield_matching/raw_material/boston_harbor_20260712/videos/long_wharf_to_hull_wharf.mp4")
DEFAULT_OUTPUT = Path(
    "/data/farfield_matching/artifacts/object_tracks/boston_harbor_leg1/v1/m2_sam2")
DEFAULT_CHECKPOINT = Path(
    "/data/farfield_matching/models/sam2/sam2.1_hiera_large.pt")

TEST_CASES = [
    ("f0000_lm0_custom_house_tower", "f0000__lm0__box0"),
    ("f0000_lm1_one_international_place", "f0000__lm1__box0"),
    ("f0000_lm3_bridge", "f0000__lm3__box0"),
    ("f0122_lm2_crane_group", "f0122__lm2__box0"),
    ("f0149_all", "f0149__*"),
]

WIN = 1024  # square window == SAM2's internal resolution: no detail lost


def build_predictor(checkpoint: Path):
    from sam2.build_sam import build_sam2_video_predictor
    for config in ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2_hiera_l.yaml"):
        try:
            return build_sam2_video_predictor(config, str(checkpoint),
                                              device="cuda")
        except Exception as e:  # noqa: BLE001 - try older config layout
            last_err = e
    raise RuntimeError(f"could not build SAM2 predictor: {last_err}")


def window_box(pano_box, wx0, wy0, pano_w):
    """Unwrapped pano bbox -> [x0, y0, x1, y1] in window pixels."""
    x_min, y_min, x_max, y_max = pano_box
    rel_x = pg.signed_x_offset(x_min, wx0, pano_w)
    return [rel_x, y_min - wy0, rel_x + (x_max - x_min), y_max - wy0]


def box_iou_containment(mask: np.ndarray, box):
    """IoU and containment coefficients between a binary mask and a box."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h, w = mask.shape
    box_mask = np.zeros_like(mask, dtype=bool)
    box_mask[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
    inter = float(np.logical_and(mask, box_mask).sum())
    mask_area = float(mask.sum())
    box_area = float(box_mask.sum())
    union = mask_area + box_area - inter
    return {
        "iou": inter / union if union else 0.0,
        "inter_over_mask": inter / mask_area if mask_area else 0.0,
        "inter_over_box": inter / box_area if box_area else 0.0,
    }


def run_case(predictor, provider, model, frames_by_idx, obs_by_frame, anchor,
             pano_w, pano_h, out_dir, font, thumb_w=512):
    """Propagate one anchor detection across one keyframe interval."""
    k = anchor.frame_idx
    if k + 1 not in frames_by_idx:
        return None
    t0 = frames_by_idx[k].time_s
    t1 = frames_by_idx[k + 1].time_s
    pano_box = pg.pano_bbox_for_observation(anchor.boxes, pano_w, pano_h)
    cx = (pano_box[0] + pano_box[2]) / 2.0
    cy = (pano_box[1] + pano_box[3]) / 2.0
    az0, _ = pg.direction_from_pano_px(cx, cy, pano_w, pano_h)

    # Heading-compensated windows around the anchor bearing.
    columns = []
    window_origins = []
    for vidx, t, frame_rgb in provider.frames_between(t0, t1):
        az_w = az0 - model.delta(t, t0)
        wx, _ = pg.pano_px_from_direction(az_w, 0.0, pano_w, pano_h)
        wx0 = wx - WIN / 2.0
        crop, wy0 = pg.extract_window(frame_rgb, wx0, cy - WIN / 2.0, WIN, WIN)
        columns.append((vidx, t, crop))
        window_origins.append((wx0, wy0))

    prompt_box = window_box(pano_box, *window_origins[0], pano_w)

    # SAM2 wants a directory of %05d.jpg frames.
    masks = [None] * len(columns)
    with tempfile.TemporaryDirectory() as tmp:
        for i, (_, _, crop) in enumerate(columns):
            Image.fromarray(crop).save(f"{tmp}/{i:05d}.jpg", quality=95)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=tmp)
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=0,
                box=np.array(prompt_box, dtype=np.float32))
            for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                masks[fidx] = (mask_logits[0, 0] > 0.0).cpu().numpy()

    # Score the final mask against every detection at keyframe k+1.
    final_mask = masks[-1]
    end_scores = []
    for obs in obs_by_frame.get(k + 1, []):
        obox = window_box(
            pg.pano_bbox_for_observation(obs.boxes, pano_w, pano_h),
            *window_origins[-1], pano_w)
        if obox[0] > WIN or obox[0] + (obox[2] - obox[0]) < 0:
            continue
        end_scores.append({
            "obs_id": obs.obs_id,
            "label": vc.obs_semantic_label(obs),
            "box_window_px": [round(v, 1) for v in obox],
            **box_iou_containment(final_mask, obox),
        })
    end_scores.sort(key=lambda s: -s["iou"])

    # Filmstrip: overlay masks, prompt box on col 0, detections on last col.
    scale = thumb_w / WIN
    strip = Image.new("RGB", (thumb_w * len(columns), thumb_w + 18), (24, 24, 24))
    n_components = []
    for i, (vidx, t, crop) in enumerate(columns):
        img = crop.copy()
        if masks[i] is not None and masks[i].any():
            overlay = np.zeros_like(img)
            overlay[masks[i]] = (255, 60, 60)
            img = (0.65 * img + 0.35 * overlay).astype(np.uint8)
            n_components.append(_count_components(masks[i]))
        else:
            n_components.append(0)
        thumb = Image.fromarray(img).resize((thumb_w, thumb_w), Image.BILINEAR)
        draw = ImageDraw.Draw(thumb)
        if i == 0:
            draw.rectangle([v * scale for v in prompt_box],
                           outline=(60, 255, 60), width=2)
        if i == len(columns) - 1:
            for s in end_scores:
                draw.rectangle([v * scale for v in s["box_window_px"]],
                               outline=(60, 160, 255), width=2)
                draw.text((s["box_window_px"][0] * scale,
                           s["box_window_px"][1] * scale - 14),
                          f"iou={s['iou']:.2f}", fill=(60, 160, 255), font=font)
        area = int(masks[i].sum()) if masks[i] is not None else 0
        vc.draw_caption(draw, f"+{t - t0:.1f}s area={area} cc={n_components[i]}",
                        font)
        strip.paste(thumb, (i * thumb_w, 0))
    vc.draw_caption(ImageDraw.Draw(strip),
                    f"{anchor.obs_id}  prompt green, detections@k+1 blue",
                    font, xy=(5, thumb_w + 2))
    strip_rel = f"strip_{anchor.obs_id}.jpg"
    strip.save(out_dir / strip_rel, quality=90)

    def mask_bbox(mask):
        if mask is None or not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    return {
        "obs_id": anchor.obs_id,
        "label": vc.obs_semantic_label(anchor),
        "n_frames": len(columns),
        "heading_delta_deg": round(model.delta(t1, t0), 3),
        "window_x0_first_last": [round(window_origins[0][0], 1),
                                 round(window_origins[-1][0], 1)],
        "prompt_box_window_px": [round(v, 1) for v in prompt_box],
        "mask_areas": [int(m.sum()) if m is not None else 0 for m in masks],
        "mask_bboxes": [mask_bbox(m) for m in masks],
        "mask_components_final": n_components[-1] if n_components else 0,
        "end_scores": end_scores,
        "strip": strip_rel,
    }


def _count_components(mask: np.ndarray) -> int:
    """Connected components of a binary mask (4-connectivity), pure numpy
    flood fill; masks are small so simple is fine."""
    import scipy.ndimage
    _, n = scipy.ndimage.label(mask)
    return int(n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_base", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--landmark_base", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--probe", action="store_true",
                        help="only verify SAM2 imports and checkpoint load")
    args = parser.parse_args()

    predictor = build_predictor(args.checkpoint)
    print(f"SAM2 predictor ready on cuda "
          f"({torch.cuda.get_device_name(0)})")
    if args.probe:
        return

    result = ingest.run_ingest(args.dataset_base, args.landmark_base,
                               IngestConfig())
    frames_by_idx = {f.frame_idx: f for f in result.frames}
    obs_by_frame = {}
    obs_by_id = {}
    for obs in result.observations:
        obs_by_frame.setdefault(obs.frame_idx, []).append(obs)
        obs_by_id[obs.obs_id] = obs

    model = heading_mod.heading_model_from_positions(
        [f.x_m for f in result.frames], [f.y_m for f in result.frames],
        [f.time_s for f in result.frames])
    provider = video_frames.VideoFrameProvider(args.video)
    probe = Image.open(
        args.dataset_base / "panorama" / f"{result.frames[0].pano_stem}.jpg")
    pano_w, pano_h = probe.size
    font = vc.load_font(13)

    anchors = []
    for case_name, anchor_id in TEST_CASES:
        if anchor_id.endswith("__*"):
            frame_idx = int(anchor_id.split("__")[0][1:])
            anchors.extend(sorted(obs_by_frame[frame_idx],
                                  key=lambda o: o.obs_id))
        else:
            anchors.append(obs_by_id[anchor_id])

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for anchor in anchors:
        print(f"tracking {anchor.obs_id} ...")
        report = run_case(predictor, provider, model, frames_by_idx,
                          obs_by_frame, anchor, pano_w, pano_h, out, font)
        if report is not None:
            best = report["end_scores"][0] if report["end_scores"] else None
            print(f"  best end match: "
                  f"{best['obs_id'] if best else 'none'} "
                  f"iou={best['iou']:.2f}" if best else "  no detections at k+1")
            reports.append(report)

    (out / "results.json").write_text(json.dumps(reports, indent=2))
    parts = ["<html><head><title>M2: SAM2 interval propagation</title>",
             "<style>body{font-family:sans-serif;background:#181818;color:#ddd}"
             "img{display:block;margin:4px 0;max-width:100%}"
             "td,th{padding:2px 8px;text-align:left}</style></head><body>",
             "<h1>M2: SAM2 box-prompt propagation, one keyframe interval</h1>"]
    for r in reports:
        parts.append(f"<h2>{html.escape(r['label'])} "
                     f"({html.escape(r['obs_id'])})</h2>")
        parts.append(f"<img src='{r['strip']}' loading='lazy'>")
        parts.append("<table><tr><th>detection @k+1</th><th>iou</th>"
                     "<th>inter/mask</th><th>inter/box</th></tr>")
        for s in r["end_scores"]:
            parts.append(
                f"<tr><td>{html.escape(s['label'])}</td><td>{s['iou']:.2f}"
                f"</td><td>{s['inter_over_mask']:.2f}</td>"
                f"<td>{s['inter_over_box']:.2f}</td></tr>")
        parts.append("</table>")
    parts.append("</body></html>")
    (out / "index.html").write_text("\n".join(parts))
    print(f"wrote {out}/index.html and results.json")


if __name__ == "__main__":
    main()
