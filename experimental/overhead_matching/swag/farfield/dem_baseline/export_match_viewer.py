"""Static viewer for framewise_eval results: click through evaluated frames
and see the query panorama, its crop ring, the top-1 candidate's depth ring
aligned crop-for-view at the matched heading shift, and a map of where the
top-k candidates sit relative to truth.

Reads frames.jsonl / summary.json from a framewise_eval output directory;
re-renders candidate rings from the database's surface + recorded config
(the database stores descriptors only).

    bazel run //experimental/overhead_matching/swag/farfield/dem_baseline:export_match_viewer -- \
        --eval_dir /data/farfield_matching/runs/dem_baseline_dev/leg2_framewise_dev100m \
        --db_dir /data/farfield_matching/artifacts/depth_render_db/mount_washington/v1_dev100m \
        --dataset mount_washington_20260815_leg2 --n_frames 40

Output: <eval_dir>/viewer/index.html, browsable via the data-root http.server.
"""

import argparse
import json
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.dem_baseline import (
    depth_render,
    query_crops,
    render_db,
    terrain,
    viz,
)
from experimental.overhead_matching.swag.farfield.viewers import page

GENERATOR = "dem_baseline.export_match_viewer"
RANK_COLORS = ("#e0e0e0", "#c8b25a", "#b07840")  # rank 1, 2-10, 11+


def frame_page_name(index: int) -> str:
    return f"frame_{index:04d}.html"


def candidate_marker_style(rank: int) -> dict:
    if rank == 0:
        return {"marker": "D", "s": 34, "color": RANK_COLORS[0],
                "zorder": 5}
    if rank < 10:
        return {"marker": "o", "s": 16, "color": RANK_COLORS[1], "zorder": 4}
    return {"marker": "o", "s": 7, "color": RANK_COLORS[2], "zorder": 3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    paths_lib.add_arguments(parser)
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--db_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="default: <eval_dir>/viewer")
    parser.add_argument("--n_frames", type=int, default=40,
                        help="evenly spread over the evaluated frames")
    parser.add_argument("--top_k_map", type=int, default=50)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    paths = paths_lib.resolve(parser, args, require=("panorama_dir",))

    records = [json.loads(line) for line in
               (args.eval_dir / "frames.jsonl").read_text().splitlines()]
    summary = json.loads((args.eval_dir / "summary.json").read_text())
    db = render_db.load_database(args.db_dir)
    manifest = db["manifest"]
    hf = terrain.HeightField.load(Path(manifest["height_field"]))
    config = depth_render.RenderConfig(**manifest["render_config"])
    crop_config = query_crops.CropRingConfig(
        n_crops=config.n_yaw, fov_deg=config.fov_deg)
    tt = depth_render.TerrainTensor.from_height_field(hf, device=args.device)
    mapper = viz.MapRenderer(hf)
    bounds = tuple(manifest["lattice"]["bounds_xy"])

    out_dir = args.out_dir or (args.eval_dir / "viewer")
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    viz.depth_colorbar().save(images / "colorbar.png")

    picks = sorted(set(
        int(i) for i in
        np.linspace(0, len(records) - 1, min(args.n_frames, len(records)))))
    yaws = config.yaw_degrees()

    for k, rec_idx in enumerate(picks):
        rec = records[rec_idx]
        stem = rec["stem"]
        pano_path = paths.panorama_dir / f"{stem}.jpg"
        pano = np.asarray(Image.open(pano_path).convert("RGB"))

        pano_thumb = Image.fromarray(pano)
        pano_thumb.thumbnail((1400, 700))
        pano_name = f"pano_{k:04d}.jpg"
        pano_thumb.save(images / pano_name, quality=82)

        # Query crop ring.
        ring = query_crops.extract_crop_ring(pano, crop_config)
        crop_entries = []
        for m in range(crop_config.n_crops):
            thumb = Image.fromarray(ring[m])
            thumb.thumbnail((150, 150))
            name = f"crop_{k:04d}_{m:02d}.jpg"
            thumb.save(images / name, quality=80)
            crop_entries.append(
                (f"images/{name}",
                 f"az {int(crop_config.azimuths_deg()[m])}&deg;"))

        # Top-1 candidate ring, aligned so view (m + shift) sits under crop m.
        top1 = rec["top_loc_idx"][0]
        shift = rec["top_shift_idx"][0]
        cx, cy = float(db["x_m"][top1]), float(db["y_m"][top1])
        cand_ring = depth_render.render_ring(tt, config, cx, cy)
        match_entries = []
        for m in range(config.n_yaw):
            view = (m + shift) % config.n_yaw
            name = f"match_{k:04d}_{m:02d}.png"
            viz.depth_image(cand_ring.depth_m[view].cpu().numpy(),
                            max_px=150).save(images / name)
            match_entries.append(
                (f"images/{name}", f"view yaw {int(yaws[view])}&deg;"))

        # Map: truth + candidates by rank.
        tx, ty = rec["truth_xy"]
        markers = [(tx, ty, {"marker": "*", "s": 130, "color": "#6fbf73",
                             "zorder": 6}, "truth")]
        for rank, loc in enumerate(rec["top_loc_idx"][:args.top_k_map]):
            label = ("top-1" if rank == 0 else
                     "rank 2-10" if rank < 10 else f"rank 11-{args.top_k_map}")
            markers.append((float(db["x_m"][loc]), float(db["y_m"][loc]),
                            candidate_marker_style(rank), label))
        map_name = f"map_{k:04d}.png"
        mapper.render(images / map_name, bounds_xy=bounds, markers=markers)

        rows = [[
            page.esc(rank + 1),
            page.esc(loc),
            f"{rec['top_scores'][rank]:.4f}",
            f"{rec['top_dist_m'][rank]:.0f}",
            f"{(rec['top_shift_idx'][rank] * 360 / config.n_yaw):.0f}&deg;",
        ] for rank, loc in enumerate(rec["top_loc_idx"][:10])]

        body = (
            viz.nav_html(k, len(picks), frame_page_name)
            + f'<div class="pano"><img src="images/{pano_name}"></div>'
            + f'<div class="muted">{page.esc(stem)} &middot; top-1 error '
            + f"{rec['top_dist_m'][0]:.0f} m &middot; joint-score entropy "
            + f"{rec['score_entropy']:.2f} nats</div>"
            + "<h2>query crops (body azimuth) vs top-1 matched depth views "
            + f"(implied heading {shift * 360 // config.n_yaw}&deg;)</h2>"
            + viz.thumb_strip(crop_entries)
            + viz.thumb_strip(match_entries)
            + viz.colorbar_html("images/colorbar.png")
            + '<div class="pane"><div>'
            + f"<h2>top candidates</h2>"
            + page.table(["rank", "lattice idx", "score", "dist to truth (m)",
                          "implied heading"], rows)
            + '</div><div class="mapimg"><h2>map</h2>'
            + f'<img src="images/{map_name}"></div></div>')
        (out_dir / frame_page_name(k)).write_text(page.page(
            f"{stem} — {args.eval_dir.name}", body, generator=GENERATOR,
            extra_style=viz.VIEWER_STYLE,
            crumbs=[("run", "../index.html"), ("viewer", None)]))

    # Index: summary metrics + per-frame table + truth-track overview map.
    mapper.render(images / "map_track.png", bounds_xy=bounds, markers=[
        (rec["truth_xy"][0], rec["truth_xy"][1],
         {"marker": ".", "s": 8, "color": "#6fbf73"}, "truth track")
        for rec in records])
    recall_rows = sorted(
        (k, v) for k, v in summary["recall"].items())
    frame_rows = [[
        f'<a href="{frame_page_name(k)}">{page.esc(records[i]["stem"])}</a>',
        f"{records[i]['top_dist_m'][0]:.0f}",
        f"{min(records[i]['top_dist_m'][:10]):.0f}",
        f"{records[i]['score_entropy']:.2f}",
    ] for k, i in enumerate(picks)]
    body = (
        f"<p>{summary['n_frames']} frames evaluated against "
        f"{page.esc(summary['db_dir'])} "
        f"(lattice {summary['lattice_spacing_m']:.0f} m); "
        f"{len(picks)} frames exported. Top-1 median error "
        f"{summary['top1_error_m']['median']:.0f} m.</p>"
        + "<div class='mapimg'><img src='images/map_track.png'></div>"
        + "<h2>recall</h2>"
        + page.table(["metric", "value"],
                     [[page.esc(k), f"{v:.3f}"] for k, v in recall_rows])
        + "<h2>frames</h2>"
        + page.table(["frame", "top-1 err (m)", "best top-10 err (m)",
                      "entropy (nats)"], frame_rows))
    (out_dir / "index.html").write_text(page.page(
        f"match viewer — {args.eval_dir.name}", body, generator=GENERATOR,
        extra_style=viz.VIEWER_STYLE))
    print(f"wrote {out_dir}/index.html ({len(picks)} frame pages)")


if __name__ == "__main__":
    main()
