"""M0 visualization: reproject VLM landmark boxes into panorama space and
render them over time, so box placement and stability can be eyeballed
before any tracking is built.

For each test case (an anchor observation, or every landmark of an anchor
frame) this renders:
- a zoom strip: a fixed pano window centered on the anchor landmark's
  bearing, across keyframes around the anchor frame, with every observation
  intersecting the window drawn and labeled. No heading compensation yet
  (that is M1), so objects drift horizontally as the boat turns - that drift
  is itself worth seeing.
- a full-pano sheet for the anchor frame with all observations drawn.
- an index.html tying it together.

Box colors are keyed on (primary tag, name) so the same semantic object
keeps its color across frames.

Run:
  bazel run //experimental/overhead_matching/swag/landmark_filtering/object_tracking:m0_render_boxes
"""

import argparse
import html
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experimental.overhead_matching.swag.data import farfield_paths
from experimental.overhead_matching.swag.landmark_filtering import ingest
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    pano_geometry as pg,
    viz_common as vc,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    IngestConfig,
)

STAGE_DIR = "m0_boxes"

# (case_name, anchor_obs_id), curated on boston_harbor_leg1. "f0149__*" expands
# to one case per landmark in that frame.
#
# These names are leg1's, and observation ids are PURELY POSITIONAL
# (`f{frame}__lm{index}__box{n}`) -- they encode nothing about content. On another
# dataset the ids therefore resolve perfectly well and the tool renders that
# dataset's boxes under Boston's landmark names: pohang_canal_04 produced
# `f0000_lm0_custom_house_tower` for a Korean harbour. The boxes were right and
# the captions were fiction, and nothing said so, because the only warning fires
# when an id FAILS to resolve. `--auto_cases` (and the automatic fallback when
# the curated ids do not all resolve) picks anchors from the data instead, the
# same way m1_heading_windows does.
LEG1_DATASET = "boston_harbor_leg1"

LEG1_TEST_CASES = [
    ("f0000_lm0_custom_house_tower", "f0000__lm0__box0"),
    ("f0000_lm1_one_international_place", "f0000__lm1__box0"),
    ("f0000_lm3_bridge", "f0000__lm3__box0"),
    ("f0122_lm2_crane_group", "f0122__lm2__box0"),
    ("f0149_all", "f0149__*"),
]


def auto_cases(obs_by_frame, n_cases=4):
    """Anchor cases chosen from the data: the busiest frames, spread out.

    The question this viewer answers is "do the boxes land on the objects the
    descriptions name", so the useful anchors are simply frames carrying several
    detections; they are spread across the run so one local mis-stitch cannot
    make every case look fine. Named for the frame and the observed tag, never
    for a landmark this tool cannot identify.
    """
    ranked = sorted(obs_by_frame.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    if not ranked:
        return []
    picked, seen = [], []
    span = max(obs_by_frame) or 1
    for frame_idx, observations in ranked:
        # Keep anchors at least 5% of the run apart.
        if any(abs(frame_idx - other) < 0.05 * span for other in seen):
            continue
        seen.append(frame_idx)
        picked.append((f"f{frame_idx:04d}_auto_n{len(observations)}",
                       f"f{frame_idx:04d}__*"))
        if len(picked) >= n_cases:
            break
    return picked

def render_window_frame(pano: np.ndarray, frame, observations, window_x0: float,
                        window_y0: float, win_w: int, win_h: int, font,
                        highlight_obs_id: str | None):
    crop, y_start = pg.extract_window(pano, window_x0, window_y0, win_w, win_h)
    img = Image.fromarray(crop)
    draw = ImageDraw.Draw(img)
    for obs in observations:
        vc.draw_obs_box(draw, obs, pano.shape[1], pano.shape[0], window_x0,
                     y_start, win_w, win_h, 1.0, font,
                     highlight=(obs.obs_id == highlight_obs_id))
    caption = f"f{frame.frame_idx:04d}"
    if frame.time_s is not None:
        caption += f"  t={frame.time_s:.1f}s"
    draw.text((6, 6), caption, fill=(0, 0, 0), font=font)
    draw.text((5, 5), caption, fill=(255, 255, 60), font=font)
    return img


def render_full_pano(pano: np.ndarray, frame, observations, font,
                     out_width: int = 2400):
    scale = out_width / pano.shape[1]
    out_height = int(pano.shape[0] * scale)
    img = Image.fromarray(pano).resize((out_width, out_height), Image.BILINEAR)
    draw = ImageDraw.Draw(img)
    for obs in observations:
        vc.draw_obs_box(draw, obs, pano.shape[1], pano.shape[0], 0.0, 0.0,
                     out_width, out_height, scale, font)
    caption = f"f{frame.frame_idx:04d} all observations"
    draw.text((6, 6), caption, fill=(0, 0, 0), font=font)
    draw.text((5, 5), caption, fill=(255, 255, 60), font=font)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    farfield_paths.add_arguments(parser)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help=f"default: <object_tracks artifact>/{STAGE_DIR}")
    parser.add_argument("--frames_before", type=int, default=8)
    parser.add_argument("--frames_after", type=int, default=12)
    parser.add_argument("--window_deg", type=float, default=60.0)
    parser.add_argument("--auto_cases", action="store_true",
                        help="pick anchors from the data instead of the "
                             "curated leg1 ids (implied when they do not all "
                             "resolve)")
    args = parser.parse_args()

    paths = farfield_paths.resolve(
        parser, args, require=("dataset_base", "frame_landmarks"))
    dataset_base = paths.dataset_base
    out = args.output_dir or paths.tracks_stage(STAGE_DIR)

    result = ingest.run_ingest(dataset_base, paths.frame_landmarks,
                               IngestConfig())
    frames_by_idx = {f.frame_idx: f for f in result.frames}
    obs_by_frame = defaultdict(list)
    obs_by_id = {}
    for obs in result.observations:
        obs_by_frame[obs.frame_idx].append(obs)
        obs_by_id[obs.obs_id] = obs
    print(f"ingest: {len(result.frames)} frames, "
          f"{len(result.observations)} observations")

    # The curated names describe boston_harbor_leg1's f0000/f0122/f0149
    # specifically, so the dataset -- not whether the ids resolve -- is the test.
    # Gating on resolution was not enough: all five ids resolve on leg2 and leg3
    # (positional ids, similar frame counts), so those legs silently kept leg1's
    # captions, which is the original bug with extra steps.
    if args.auto_cases or paths.dataset != LEG1_DATASET:
        if not args.auto_cases:
            print(f"NOTE: the curated anchor cases name "
                  f"{LEG1_DATASET}'s landmarks and describe nothing in "
                  f"{paths.dataset}. Their ids are positional and resolve here "
                  f"regardless, which is exactly how they went unnoticed. "
                  f"Choosing anchors from this dataset's own detections.")
        selected = auto_cases(obs_by_frame)
        print(f"using {len(selected)} auto-selected anchor(s): "
              + ", ".join(name for name, _ in selected))
    else:
        selected = [c for c in LEG1_TEST_CASES
                    if c[1].endswith("__*") or c[1] in obs_by_id]
        if len(selected) < len(LEG1_TEST_CASES):
            print(f"WARNING: only {len(selected)} of {len(LEG1_TEST_CASES)} "
                  f"curated anchors resolve even on {LEG1_DATASET}")

    # Expand wildcard cases into one case per landmark in the anchor frame.
    cases = []
    for case_name, anchor_id in selected:
        if anchor_id.endswith("__*"):
            frame_idx = int(anchor_id.split("__")[0][1:])
            for obs in sorted(obs_by_frame[frame_idx], key=lambda o: o.obs_id):
                cases.append((
                    f"{case_name}_lm{obs.landmark_idx}_{obs.primary_tag_value}",
                    obs.obs_id))
        else:
            cases.append((case_name, anchor_id))

    # Resolve anchors and per-case frame ranges / windows.
    probe_stem = result.frames[0].pano_stem
    probe = Image.open(dataset_base / "panorama" / f"{probe_stem}.jpg")
    pano_w, pano_h = probe.size
    win_w = int(round(args.window_deg / 360.0 * pano_w))
    win_h = int(round(win_w * 9 / 16))

    case_specs = []  # (case_name, anchor_obs, frame_range, window_x0, window_y0)
    for case_name, anchor_id in cases:
        anchor = obs_by_id.get(anchor_id)
        if anchor is None:
            print(f"WARNING: anchor observation {anchor_id} not found, skipping")
            continue
        x_min, y_min, x_max, y_max = pg.pano_bbox_for_observation(
            anchor.boxes, pano_w, pano_h)
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        lo = max(0, anchor.frame_idx - args.frames_before)
        hi = min(max(frames_by_idx), anchor.frame_idx + args.frames_after)
        case_specs.append((case_name, anchor, range(lo, hi + 1),
                           cx - win_w / 2.0, cy - win_h / 2.0))

    # Frame-outer loop so each pano is decoded once.
    needed = defaultdict(list)  # frame_idx -> [case index]
    for i, (_, _, frame_range, _, _) in enumerate(case_specs):
        for frame_idx in frame_range:
            needed[frame_idx].append(i)

    font = vc.load_font(14)
    out.mkdir(parents=True, exist_ok=True)
    case_images = defaultdict(list)
    anchor_sheets = {}
    for frame_idx in sorted(needed):
        frame = frames_by_idx.get(frame_idx)
        if frame is None:
            continue
        pano_path = dataset_base / "panorama" / f"{frame.pano_stem}.jpg"
        pano = np.asarray(Image.open(pano_path))
        observations = obs_by_frame[frame_idx]
        for case_idx in needed[frame_idx]:
            case_name, anchor, _, wx0, wy0 = case_specs[case_idx]
            case_dir = out / case_name
            case_dir.mkdir(exist_ok=True)
            img = render_window_frame(
                pano, frame, observations, wx0, wy0, win_w, win_h, font,
                highlight_obs_id=anchor.obs_id
                if frame_idx == anchor.frame_idx else None)
            rel = f"{case_name}/f{frame_idx:04d}.jpg"
            img.save(out / rel, quality=88)
            case_images[case_name].append((frame_idx, rel))
            if frame_idx == anchor.frame_idx and anchor.frame_idx not in anchor_sheets:
                sheet = render_full_pano(pano, frame, observations, font)
                sheet_rel = f"full_pano_f{frame_idx:04d}.jpg"
                sheet.save(out / sheet_rel, quality=88)
                anchor_sheets[anchor.frame_idx] = sheet_rel
        print(f"rendered frame f{frame_idx:04d} "
              f"({len(needed[frame_idx])} case windows)")

    # Index page.
    parts = ["<html><head><title>M0: landmark boxes over time</title>",
             "<style>body{font-family:sans-serif;background:#181818;color:#ddd}"
             "img{display:block;margin:2px 0;max-width:100%}"
             "h2{margin-top:40px}</style></head><body>",
             "<h1>M0: VLM boxes reprojected to pano space, over time</h1>",
             "<p>Fixed window per case (no heading compensation yet); "
             "horizontal drift = boat yaw. Colors keyed on (tag, name).</p>"]
    parts.append("<h2>Anchor frame full-pano sheets</h2>")
    for frame_idx in sorted(anchor_sheets):
        parts.append(f"<h3>f{frame_idx:04d}</h3>"
                     f"<img src='{anchor_sheets[frame_idx]}' loading='lazy'>")
    for case_name, anchor, frame_range, _, _ in case_specs:
        parts.append(f"<h2>{html.escape(case_name)}</h2>")
        parts.append(
            f"<p>anchor: {html.escape(vc.obs_semantic_label(anchor))} — "
            f"{html.escape(anchor.description)}</p>")
        for frame_idx, rel in sorted(case_images[case_name]):
            parts.append(f"<img src='{rel}' loading='lazy'>")
    parts.append("</body></html>")
    (out / "index.html").write_text("\n".join(parts))
    print(f"\nwrote {out}/index.html "
          f"({sum(len(v) for v in case_images.values())} window images, "
          f"{len(anchor_sheets)} full-pano sheets)")


if __name__ == "__main__":
    main()
