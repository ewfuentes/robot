"""Per-frame retrieval forensics: what the query saw, what won, where the
winners are, and how the truth's own candidate scored.

For each sampled keyframe this renders one panel:

* the query photo strip, rolled into the map frame implied by the winning
  shift (shift k means the pano centre column points at map azimuth
  k * 360/n_theta -- see panorama_score and query_crops), with the
  truth-candidate's render horizon drawn over it;
* the depth strip at each of the top-k retrieved locations;
* the depth strip at the lattice candidate NEAREST THE TRUTH, with its score
  and its rank among all locations -- the number that separates "the map is
  ambiguous" (truth scores nearly as well as the winner) from "the query does
  not look like any render" (truth scores far worse);
* a region map with the truth, the retrieved winners, and the truth candidate,
  so a blob of near-tied winners is visible as a blob.

Truth is used only to pick that candidate and to label distances; it never
enters scoring.

    bazel run //experimental/overhead_matching/swag/farfield/dem_baseline:export_retrieval_forensics -- \
        --dataset_dir /data/farfield_matching/datasets/charles_river_20260727 \
        --db_dir /data/farfield_matching/artifacts/depth_render_db/charles_river/v1_dsm_100m \
        --surface /data/farfield_matching/artifacts/dem_surfaces/charles_river/v1_dsm/surface \
        --background /data/farfield_matching/artifacts/dem_surfaces/charles_river/background_glo30_30m/surface \
        --weights /data/farfield_matching/models/crosslocate/AlpsPhotosToDepthCompact_31_2/converted_weights.npz \
        --out_dir /data/farfield_matching/runs/260901_dem_baseline_ma/forensics_charles_dsm \
        --stride 8
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
    depth_render,
    panorama_score,
    query_crops,
    render_db,
    terrain,
    truth_strips,
    viz,
)
from experimental.overhead_matching.swag.farfield.viewers import page

GENERATOR = "dem_baseline.export_retrieval_forensics"
MAP_PX = 280  # each of two stacked map panels
STRIP_H = 118
LABEL_H = 15
BG_RGB = (18, 18, 18)
TRUTH_RGB = (80, 255, 120)
WIN_RGB = (255, 210, 60)
CAND_RGB = (255, 64, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--db_dir", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--background", type=Path, action="append", default=[])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--elev_min_deg", type=float, default=-12.0)
    parser.add_argument("--elev_max_deg", type=float, default=25.0)
    parser.add_argument("--n_az", type=int, default=880)
    parser.add_argument("--zoom_span_m", type=float, default=8000.0,
                        help="width of the zoomed map panel")
    parser.add_argument("--observer_height_m", type=float, default=3.0)
    parser.add_argument("--max_range_m", type=float, default=30000.0)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--keep_frames", action="store_true")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class BaseMap:
    """Region hillshade with water tinted, plus metric->pixel mapping.

    Water is tinted from the DSM provenance raster when it sits beside the
    surface (code 3 = the provider's hydro-flattened DEM), because whether a
    winning candidate stands on water or on land is the first thing to check
    about a blob of near-tied winners.
    """

    def __init__(self, surface: Path, hf: terrain.HeightField,
                 max_px: int = 1400):
        shade = viz.hillshade_u8(hf.elevation, hf.res, max_px=max_px)
        rgb = np.repeat(shade[..., None], 3, axis=2)
        provenance = surface.parent / "provenance.npz"
        if provenance.exists():
            with np.load(provenance) as data:
                prov = data["provenance"]
            step = max(1, int(np.ceil(max(hf.elevation.shape) / max_px)))
            water = prov[::step, ::step][:shade.shape[0], :shade.shape[1]] == 3
            water = water[:rgb.shape[0], :rgb.shape[1]]
            tint = np.array([40, 90, 170], dtype=np.float32)
            rgb[water] = (0.45 * rgb[water] + 0.55 * tint).astype(np.uint8)
            del prov
        self.image = Image.fromarray(rgb)
        self.scale = self.image.width / (hf.elevation.shape[1] * hf.res)
        self.x0, self.y0 = hf.x0, hf.y0

    def pixel(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.x0) * self.scale, (self.y0 - y) * self.scale)

    def panel(self, size_px: int, truth_xy, winners_xy, cand_xy, *,
              span_m: float | None, title: str) -> Image.Image:
        """One square map panel; span_m None means the whole region."""
        if span_m is None:
            crop = self.image.copy()
            offset = (0.0, 0.0)
            crop_scale = size_px / max(crop.size)
        else:
            cx, cy = self.pixel(*truth_xy)
            half = span_m * self.scale / 2.0
            box = (cx - half, cy - half, cx + half, cy + half)
            crop = self.image.crop([int(round(v)) for v in box])
            offset = (box[0], box[1])
            crop_scale = size_px / max(crop.size[0], 1)
        crop = crop.resize((size_px, size_px), Image.BILINEAR)

        def place(x, y):
            px, py = self.pixel(x, y)
            return ((px - offset[0]) * crop_scale,
                    (py - offset[1]) * crop_scale)

        draw = ImageDraw.Draw(crop)
        for rank, (x, y) in enumerate(winners_xy, start=1):
            px, py = place(x, y)
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], outline=WIN_RGB,
                         width=2)
            draw.text((px + 5, py - 6), str(rank), fill=WIN_RGB)
        px, py = place(*cand_xy)
        draw.line([(px - 6, py), (px + 6, py)], fill=CAND_RGB, width=2)
        draw.line([(px, py - 6), (px, py + 6)], fill=CAND_RGB, width=2)
        px, py = place(*truth_xy)
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=TRUTH_RGB)

        bar_m = 5000.0 if span_m is None else max(span_m / 5.0, 100.0)
        bar = bar_m * self.scale * crop_scale
        y_bar = crop.height - 12
        draw.line([(10, y_bar), (10 + bar, y_bar)], fill=(255, 255, 255),
                  width=2)
        label = (f"{bar_m / 1000:g} km" if bar_m >= 1000
                 else f"{bar_m:.0f} m")
        draw.text((10, y_bar - 13), label, fill=(255, 255, 255))
        draw.text((6, 4), title, fill=(240, 240, 240))
        return crop


def local_relief(hf: terrain.HeightField, block_m: float = 100.0) \
        -> tuple[np.ndarray, float, float, float]:
    """Per-block elevation standard deviation, a proxy for "how much geometry
    a view from here contains".

    The forensics panels suggest the released descriptor prefers renders
    packed with near-field structure over the correct low-relief water view.
    Relief is the cheap lattice-wide test of that: if score tracks relief
    rather than similarity, the winners are geometry-dense by construction.
    """
    step = max(int(round(block_m / hf.res)), 1)
    rows = hf.elevation.shape[0] // step
    cols = hf.elevation.shape[1] // step
    out = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        band = hf.elevation[r * step:(r + 1) * step, :cols * step]
        band = band.reshape(step, cols, step)
        out[r] = band.std(axis=(0, 2))
    return out, hf.x0, hf.y0, step * hf.res


def labelled(strip: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (strip.width, strip.height + LABEL_H), BG_RGB)
    draw = ImageDraw.Draw(out)
    draw.text((4, 2), text, fill=(232, 232, 232))
    out.paste(strip, (0, LABEL_H))
    return out


def main() -> None:
    args = parse_args()
    hf = terrain.HeightField.load(args.surface)
    tt = depth_render.TerrainTensor.chain_from_height_fields(
        [hf] + [terrain.HeightField.load(p) for p in args.background],
        device=args.device)
    config = depth_render.RenderConfig(
        max_range_m=args.max_range_m,
        observer_height_m=args.observer_height_m)
    base_map = BaseMap(args.surface, hf)
    relief, relief_x0, relief_y0, relief_res = local_relief(hf)

    db = render_db.load_database(args.db_dir, device=args.device)
    n_theta = db["manifest"]["render_config"]["n_yaw"]
    crop_config = query_crops.CropRingConfig(
        n_crops=n_theta, fov_deg=db["manifest"]["render_config"]["fov_deg"])
    db_xy = np.stack([db["x_m"], db["y_m"]], axis=1)
    relief_col = np.clip(((db_xy[:, 0] - relief_x0) / relief_res).astype(int),
                         0, relief.shape[1] - 1)
    relief_row = np.clip(((relief_y0 - db_xy[:, 1]) / relief_res).astype(int),
                         0, relief.shape[0] - 1)
    lattice_relief = relief[relief_row, relief_col]
    model = crosslocate_net.CrossLocateVGG16MAC().to(args.device).eval()
    crosslocate_net.load_converted_weights(model, args.weights)

    frames = truth_strips.load_frames(args.dataset_dir)
    course = truth_strips.GridCourse(frames, hf.crs)
    indices = list(range(0, len(frames), max(args.stride, 1)))

    def strip_at(x: float, y: float) -> tuple[Image.Image, np.ndarray]:
        cyl = depth_render.render_cylinder(
            tt, config, x, y, n_az=args.n_az, n_rows=STRIP_H,
            elev_min_deg=args.elev_min_deg, elev_max_deg=args.elev_max_deg)
        depth = cyl.depth_m.cpu().numpy()
        return viz.depth_image(depth, max_px=args.n_az).convert("RGB"), depth

    frames_dir = args.out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.time()
    for n, i in enumerate(indices):
        frame = frames[i]
        fid = truth_strips.frame_id(frame)
        truth_x, truth_y = course.xy(i)
        pano = np.asarray(Image.open(
            args.dataset_dir / "panorama" / frame["frame_file"]
        ).convert("RGB"))
        ring = query_crops.extract_crop_ring(pano, crop_config)
        with torch.inference_mode():
            batch = torch.stack([crosslocate_net.rgb_query_tensor(c)
                                 for c in ring]).to(args.device)
            descriptors = model(batch)
            joint = panorama_score.joint_scores(descriptors, db["descriptors"])
            scores = joint.scores
            flat_values, flat_idx = scores.reshape(-1).topk(args.top_k)
            loc_idx = (flat_idx // n_theta).cpu().numpy()
            shift_idx = (flat_idx % n_theta).cpu().numpy()
            per_loc = scores.max(dim=1)
            # The candidate nearest the truth, and how it ranks.
            cand = int(np.argmin(np.hypot(db_xy[:, 0] - truth_x,
                                          db_xy[:, 1] - truth_y)))
            cand_score = float(per_loc.values[cand])
            cand_shift = int(per_loc.indices[cand])
            cand_rank = int((per_loc.values > per_loc.values[cand]).sum()) + 1
        loc_scores = per_loc.values.cpu().numpy()
        top100 = np.argsort(-loc_scores)[:100]
        relief_r = float(np.corrcoef(loc_scores, lattice_relief)[0, 1])
        relief_top100 = float(lattice_relief[top100].mean())
        relief_truth = float(lattice_relief[cand])
        winners = [(int(l), int(s), float(v))
                   for l, s, v in zip(loc_idx, shift_idx,
                                      flat_values.cpu().numpy())]

        # Photo rolled into the map frame the truth candidate's match implies.
        centre_yaw = cand_shift * (360.0 / n_theta)
        cand_strip, cand_depth = strip_at(*db_xy[cand])
        photo = truth_strips.pano_strip(
            Image.fromarray(pano), centre_yaw,
            elev_min_deg=args.elev_min_deg, elev_max_deg=args.elev_max_deg,
            n_az=args.n_az, n_rows=STRIP_H)
        photo_img = truth_strips.add_compass_ticks(
            truth_strips.draw_horizon(Image.fromarray(photo),
                                      truth_strips.horizon_rows(cand_depth)),
            course_deg=course.course_deg(i))

        panels = [labelled(
            photo_img,
            f"{fid}  query photo, rolled to the truth candidate's matched "
            f"heading (centre column -> {centre_yaw:.0f} deg)")]
        for rank, (loc, shift, value) in enumerate(winners, start=1):
            dist = float(np.hypot(db_xy[loc, 0] - truth_x,
                                  db_xy[loc, 1] - truth_y))
            strip, _ = strip_at(*db_xy[loc])
            panels.append(labelled(
                truth_strips.add_compass_ticks(strip),
                f"rank {rank}  score {value:.4f}  {dist / 1000:.2f} km from "
                f"truth  heading {shift * 360.0 / n_theta:.0f} deg"))
        panels.append(labelled(
            truth_strips.add_compass_ticks(cand_strip),
            f"TRUTH candidate  score {cand_score:.4f}  rank {cand_rank} of "
            f"{len(db_xy)}  (winner scored {winners[0][2]:.4f})"))

        strips_h = sum(p.height for p in panels)
        canvas = Image.new("RGB", (MAP_PX + args.n_az,
                                   max(strips_h, 2 * MAP_PX) + 18), BG_RGB)
        winners_xy = [db_xy[loc] for loc, _, _ in winners]
        canvas.paste(base_map.panel(MAP_PX, (truth_x, truth_y), winners_xy,
                                    db_xy[cand], span_m=None,
                                    title="region"), (0, 0))
        canvas.paste(base_map.panel(MAP_PX, (truth_x, truth_y), winners_xy,
                                    db_xy[cand], span_m=args.zoom_span_m,
                                    title=f"{args.zoom_span_m / 1000:g} km "
                                          "around truth"), (0, MAP_PX))
        y = 0
        for panel in panels:
            canvas.paste(panel, (MAP_PX, y))
            y += panel.height
        ImageDraw.Draw(canvas).text(
            (4, canvas.height - 14),
            f"{args.dataset_dir.name}  {fid}  ({n + 1}/{len(indices)})  "
            f"top1 err {np.hypot(db_xy[winners[0][0], 0] - truth_x, db_xy[winners[0][0], 1] - truth_y) / 1000:.2f} km"
            f"  truth-candidate rank {cand_rank}  score gap "
            f"{winners[0][2] - cand_score:+.4f}", fill=(235, 235, 235))
        canvas.save(frames_dir / f"{n:05d}.png")

        records.append(dict(
            sequence=n, keyframe_idx=i, frame_id=fid,
            truth_xy=[truth_x, truth_y],
            top1_dist_m=float(np.hypot(db_xy[winners[0][0], 0] - truth_x,
                                       db_xy[winners[0][0], 1] - truth_y)),
            top1_score=winners[0][2],
            truth_candidate_score=cand_score,
            truth_candidate_rank=cand_rank,
            truth_candidate_shift=cand_shift,
            score_gap=winners[0][2] - cand_score,
            score_vs_relief_r=relief_r,
            relief_top100_m=relief_top100,
            relief_truth_m=relief_truth))
        if (n + 1) % 10 == 0 or n + 1 == len(indices):
            rate = (time.time() - started) / (n + 1)
            print(f"{n + 1}/{len(indices)} frames  {rate:.2f} s/frame",
                  flush=True)

    video = args.out_dir / "forensics.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{args.fps:g}",
         "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264",
         "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(video)], check=True)
    if not args.keep_frames:
        shutil.rmtree(frames_dir)

    ranks = np.array([r["truth_candidate_rank"] for r in records])
    gaps = np.array([r["score_gap"] for r in records])
    relief_rs = np.array([r["score_vs_relief_r"] for r in records])
    summary = dict(
        generator=GENERATOR, dataset=args.dataset_dir.name,
        db_dir=str(args.db_dir), surface=str(args.surface),
        backgrounds=[str(p) for p in args.background],
        n_frames=len(records), n_locations=len(db_xy),
        observer_height_m=args.observer_height_m,
        truth_candidate_rank=dict(
            median=float(np.median(ranks)), p10=float(np.percentile(ranks, 10)),
            p90=float(np.percentile(ranks, 90)),
            frac_top_100=float((ranks <= 100).mean()),
            frac_top_1000=float((ranks <= 1000).mean())),
        score_gap=dict(median=float(np.median(gaps)),
                       p90=float(np.percentile(gaps, 90))),
        score_vs_local_relief=dict(
            median_pearson_r=float(np.median(relief_rs)),
            p10=float(np.percentile(relief_rs, 10)),
            p90=float(np.percentile(relief_rs, 90)),
            lattice_mean_relief_m=float(lattice_relief.mean()),
            median_top100_relief_m=float(np.median(
                [r["relief_top100_m"] for r in records])),
            median_truth_relief_m=float(np.median(
                [r["relief_truth_m"] for r in records])),
            note="relief = per-100 m-block elevation std; a positive "
                 "correlation means the score rewards geometry density"),
        frames=records)
    (args.out_dir / "forensics.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True))

    r = summary["truth_candidate_rank"]
    body = (
        f"<video src='{video.name}' controls loop style='width:100%'></video>"
        f"<p>{len(records)} frames. Left: region hillshade with the truth "
        "(green), the retrieved winners (yellow, numbered) and the candidate "
        "nearest the truth (magenta); water is tinted blue. Right: the query photo rolled into the "
        "map frame that the truth candidate's own best match implies, with "
        "that candidate's render horizon over it; then the winners' depth "
        "strips; then the truth candidate's.</p>"
        f"<p>The truth's own candidate ranks {r['median']:.0f} of "
        f"{len(db_xy)} at the median (p10 {r['p10']:.0f}, p90 "
        f"{r['p90']:.0f}); it is in the top 100 on "
        f"{100 * r['frac_top_100']:.0f}% of frames and the top 1000 on "
        f"{100 * r['frac_top_1000']:.0f}%. Median score gap to the winner "
        f"{summary['score_gap']['median']:+.4f}. A small gap with a poor rank "
        "is map ambiguity; a large gap is the query not looking like any "
        "render.</p>"
        f"<p>Score vs local relief (100 m block elevation std) across the "
        f"whole lattice: median Pearson r "
        f"{summary['score_vs_local_relief']['median_pearson_r']:+.2f}. "
        f"Top-100 mean relief "
        f"{summary['score_vs_local_relief']['median_top100_relief_m']:.1f} m "
        f"vs {summary['score_vs_local_relief']['lattice_mean_relief_m']:.1f} m "
        f"over the lattice and "
        f"{summary['score_vs_local_relief']['median_truth_relief_m']:.1f} m at "
        "the truth. A positive correlation means the descriptor rewards "
        "geometry density rather than similarity.</p>"
        f"<p class='muted'>db: {page.esc(str(args.db_dir))}</p>")
    (args.out_dir / "viewer.html").write_text(page.page(
        f"retrieval forensics: {args.dataset_dir.name}", body,
        generator=GENERATOR))
    print(f"wrote {video}")


if __name__ == "__main__":
    main()
