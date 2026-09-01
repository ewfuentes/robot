"""Timelapse of the retrieval observation field over the candidate region.

Per sampled keyframe: embed the query crop ring, score it against the whole
reference database, reduce over heading (max over shift, the same reduction
retrieval uses), and paint the per-location field over the region and over a
zoom around the truth. No renders are needed, so this is fast enough to sweep
a whole dataset.

The field is what a Bayes stage would turn into an observation likelihood, so
its SHAPE is the point: one peak, a ridge, or a scatter of near-ties. Colour is
a per-frame relative scale (min..max printed in the caption) because absolute
descriptor scores are not calibrated across frames.

    bazel run //experimental/overhead_matching/swag/farfield/dem_baseline:export_likelihood_video -- \
        --dataset_dir /data/farfield_matching/datasets/charles_river_20260727 \
        --db_dir /data/farfield_matching/artifacts/depth_render_db/charles_river/v1_dsm_100m \
        --weights .../converted_weights.npz \
        --hillshade /data/farfield_matching/artifacts/dem_surfaces/charles_river/background_glo30_30m/surface \
        --out_dir /data/farfield_matching/runs/260901_dem_baseline_ma/likelihood_charles_dsm \
        --stride 5
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

import numpy as np
from PIL import Image, ImageDraw

from experimental.overhead_matching.swag.farfield.dem_baseline import (
    crosslocate_net,
    panorama_score,
    query_crops,
    render_db,
    terrain,
    truth_strips,
    viz,
)
from experimental.overhead_matching.swag.farfield.viewers import page

GENERATOR = "dem_baseline.export_likelihood_video"
PANEL_PX = 520
CAPTION_H = 30
BG_RGB = (18, 18, 18)
TRUTH_RGB = (60, 255, 110)
WIN_RGB = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--db_dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--hillshade", type=Path, default=None,
                        help="Optional coarse HeightField for a terrain "
                             "backdrop under the field (base path, no suffix)")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--zoom_span_m", type=float, default=6000.0)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--clip_percentile", type=float, default=95.0,
                        help="colour floor: this percentile of the field maps "
                             "to the bottom of the scale, so the top of the "
                             "field is visible instead of the noise floor")
    parser.add_argument("--alpha", type=float, default=0.78,
                        help="field opacity over the backdrop")
    parser.add_argument("--keep_frames", action="store_true")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class FieldGrid:
    """Maps lattice locations onto a raster, north up."""

    def __init__(self, x_m: np.ndarray, y_m: np.ndarray, spacing_m: float):
        self.spacing = spacing_m
        self.x0, self.y1 = x_m.min(), y_m.max()
        self.col = np.round((x_m - self.x0) / spacing_m).astype(int)
        self.row = np.round((self.y1 - y_m) / spacing_m).astype(int)
        self.shape = (self.row.max() + 1, self.col.max() + 1)

    def raster(self, values: np.ndarray) -> np.ndarray:
        out = np.full(self.shape, np.nan, dtype=np.float32)
        out[self.row, self.col] = values
        return out

    def pixel(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.x0) / self.spacing, (self.y1 - y) / self.spacing)


def turbo(values01: np.ndarray) -> np.ndarray:
    from matplotlib import cm
    lut = (cm.get_cmap("turbo")(np.linspace(0, 1, 256))[:, :3] * 255)
    filled = np.nan_to_num(values01, nan=0.0)
    rgb = lut[np.clip(filled * 255, 0, 255).astype(np.uint8)].astype(np.uint8)
    rgb[np.isnan(values01)] = 30
    return rgb


def backdrop(grid: FieldGrid, hillshade: Path | None) -> np.ndarray | None:
    """Grey terrain under the field, sampled onto the lattice raster."""
    if hillshade is None:
        return None
    hf = terrain.HeightField.load(hillshade)
    shade = viz.hillshade_u8(hf.elevation, hf.res, max_px=4000)
    step = max(1, int(np.ceil(max(hf.elevation.shape) / 4000)))
    res = hf.res * step
    northing = grid.y1 - np.arange(grid.shape[0]) * grid.spacing
    easting = grid.x0 + np.arange(grid.shape[1]) * grid.spacing
    src_row = np.clip(((hf.y0 - northing) / res).astype(int),
                      0, shade.shape[0] - 1)
    src_col = np.clip(((easting - hf.x0) / res).astype(int),
                      0, shade.shape[1] - 1)
    return shade[np.ix_(src_row, src_col)]


def panel(field: np.ndarray, base: np.ndarray | None, grid: FieldGrid, *,
          size_px: int, alpha: float, centre_xy=None, span_m=None,
          title: str, truth_xy=None, win_xy=None) -> Image.Image:
    rgb = turbo(field).astype(np.float32)
    if base is not None:
        grey = np.repeat(base[..., None].astype(np.float32), 3, axis=2)
        rgb = alpha * rgb + (1.0 - alpha) * grey
    image = Image.fromarray(rgb.astype(np.uint8))
    offset = (0.0, 0.0)
    if span_m is not None and centre_xy is not None:
        cx, cy = grid.pixel(*centre_xy)
        half = span_m / grid.spacing / 2.0
        box = (cx - half, cy - half, cx + half, cy + half)
        image = image.crop([int(round(v)) for v in box])
        offset = (box[0], box[1])
    scale = size_px / max(image.size)
    image = image.resize((max(int(image.width * scale), 1),
                          max(int(image.height * scale), 1)), Image.NEAREST)
    draw = ImageDraw.Draw(image)

    def place(xy):
        px, py = grid.pixel(*xy)
        return ((px - offset[0]) * scale, (py - offset[1]) * scale)

    if win_xy is not None:
        px, py = place(win_xy)
        draw.line([(px - 5, py - 5), (px + 5, py + 5)], fill=WIN_RGB, width=2)
        draw.line([(px - 5, py + 5), (px + 5, py - 5)], fill=WIN_RGB, width=2)
    if truth_xy is not None:
        px, py = place(truth_xy)
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], outline=TRUTH_RGB,
                     width=2)
    bar_m = 5000.0 if span_m is None else max(span_m / 5.0, 100.0)
    bar = bar_m / grid.spacing * scale
    y_bar = image.height - 12
    draw.line([(10, y_bar), (10 + bar, y_bar)], fill=(255, 255, 255), width=2)
    draw.text((10, y_bar - 13),
              f"{bar_m / 1000:g} km" if bar_m >= 1000 else f"{bar_m:.0f} m",
              fill=(255, 255, 255))
    draw.text((6, 4), title, fill=(245, 245, 245))
    return image


def main() -> None:
    args = parse_args()
    db = render_db.load_database(args.db_dir, device=args.device)
    n_theta = db["manifest"]["render_config"]["n_yaw"]
    spacing = db["manifest"]["lattice"]["spacing_m"]
    crop_config = query_crops.CropRingConfig(
        n_crops=n_theta, fov_deg=db["manifest"]["render_config"]["fov_deg"])
    db_xy = np.stack([db["x_m"], db["y_m"]], axis=1)
    grid = FieldGrid(db["x_m"], db["y_m"], spacing)
    base = backdrop(grid, args.hillshade)

    model = crosslocate_net.CrossLocateVGG16MAC().to(args.device).eval()
    crosslocate_net.load_converted_weights(model, args.weights)

    frames = truth_strips.load_frames(args.dataset_dir)
    course = truth_strips.GridCourse(frames, db["manifest"]["lattice"]["crs"])
    indices = list(range(0, len(frames), max(args.stride, 1)))

    frames_dir = args.out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.time()
    for n, i in enumerate(indices):
        frame = frames[i]
        fid = truth_strips.frame_id(frame)
        truth_x, truth_y = course.xy(i)
        pano = np.asarray(truth_strips.open_pano(
            args.dataset_dir / "panorama" / frame["frame_file"], 4096
        ).convert("RGB"))
        ring = query_crops.extract_crop_ring(pano, crop_config)
        with torch.inference_mode():
            batch = torch.stack([crosslocate_net.rgb_query_tensor(c)
                                 for c in ring]).to(args.device)
            joint = panorama_score.joint_scores(model(batch),
                                                db["descriptors"])
            per_loc = joint.scores.max(dim=1).values.cpu().numpy()
        lo, hi = float(per_loc.min()), float(per_loc.max())
        floor = float(np.percentile(per_loc, args.clip_percentile))
        shown = np.clip((per_loc - floor) / max(hi - floor, 1e-9), 0.0, 1.0)
        field = grid.raster(shown)
        win = int(np.argmax(per_loc))
        cand = int(np.argmin(np.hypot(db_xy[:, 0] - truth_x,
                                      db_xy[:, 1] - truth_y)))
        truth_pct = float((per_loc < per_loc[cand]).mean())

        left = panel(field, base, grid, size_px=PANEL_PX, alpha=args.alpha,
                     title="region", truth_xy=(truth_x, truth_y),
                     win_xy=db_xy[win])
        right = panel(field, base, grid, size_px=PANEL_PX, alpha=args.alpha,
                      centre_xy=(truth_x, truth_y), span_m=args.zoom_span_m,
                      title=f"{args.zoom_span_m / 1000:g} km around truth",
                      truth_xy=(truth_x, truth_y), win_xy=db_xy[win])
        canvas = Image.new("RGB", (left.width + right.width + 8,
                                   max(left.height, right.height) + CAPTION_H),
                           BG_RGB)
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width + 8, 0))
        ImageDraw.Draw(canvas).text(
            (6, canvas.height - 22),
            f"{args.dataset_dir.name}  {fid}  ({n + 1}/{len(indices)})  "
            f"score {lo:.3f}..{hi:.3f} (colour floor p{args.clip_percentile:g}"
            f" = {floor:.3f})  truth percentile "
            f"{100 * truth_pct:.1f}  top1 err "
            f"{np.hypot(db_xy[win, 0] - truth_x, db_xy[win, 1] - truth_y) / 1000:.2f} km"
            "   (circle = truth, cross = argmax)", fill=(238, 238, 238))
        canvas.save(frames_dir / f"{n:05d}.png")
        records.append(dict(sequence=n, keyframe_idx=i, frame_id=fid,
                            score_min=lo, score_max=hi,
                            truth_percentile=truth_pct,
                            top1_dist_m=float(np.hypot(
                                db_xy[win, 0] - truth_x,
                                db_xy[win, 1] - truth_y))))
        if (n + 1) % 20 == 0 or n + 1 == len(indices):
            rate = (time.time() - started) / (n + 1)
            print(f"{n + 1}/{len(indices)} frames  {rate:.2f} s/frame",
                  flush=True)

    video = args.out_dir / "likelihood.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{args.fps:g}",
         "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264",
         "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(video)], check=True)
    if not args.keep_frames:
        shutil.rmtree(frames_dir)

    pct = np.array([r["truth_percentile"] for r in records])
    (args.out_dir / "likelihood.json").write_text(json.dumps(dict(
        generator=GENERATOR, dataset=args.dataset_dir.name,
        db_dir=str(args.db_dir), stride=args.stride, fps=args.fps,
        n_frames=len(records), lattice_spacing_m=spacing,
        truth_percentile=dict(median=float(np.median(pct)),
                              p10=float(np.percentile(pct, 10)),
                              p90=float(np.percentile(pct, 90))),
        frames=records), indent=1, sort_keys=True))
    body = (
        f"<video src='{video.name}' controls loop style='width:100%'></video>"
        f"<p>{len(records)} frames at stride {args.stride}. Per-location "
        "retrieval score, reduced over heading exactly as retrieval does, on a "
        "per-frame relative colour scale (turbo), with the colour floor at "
        f"the {args.clip_percentile:g}th percentile so the top of the field is "
        "visible rather than the noise floor; each frame's caption carries the "
        "absolute range. Circle = truth, cross = argmax.</p>"
        f"<p>The truth's own candidate sits at the "
        f"{100 * float(np.median(pct)):.1f}th percentile of the field at the "
        "median frame. This is the field a Bayes stage would consume, so a "
        "wide plateau or a scatter of near-ties is what temporal accumulation "
        "would have to overcome.</p>"
        f"<p class='muted'>db: {page.esc(str(args.db_dir))}</p>")
    (args.out_dir / "viewer.html").write_text(page.page(
        f"observation field: {args.dataset_dir.name}", body,
        generator=GENERATOR))
    print(f"wrote {video}")


if __name__ == "__main__":
    main()
