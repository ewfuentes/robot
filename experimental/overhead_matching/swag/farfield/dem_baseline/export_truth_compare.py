"""Truth-pose QA viewer: each sampled query panorama next to the surface's
cylindrical depth render at the same (ground-truth) pose, heading-aligned.

Alignment: the photo strip is rolled so column 0 is grid north, CW eastward,
using the pipeline's GPS course model and the dataset's approved
`nominal_forward.json` (see truth_strips for the one statement of the
convention). The depth strip is rendered directly in that frame. Course is a
GPS estimate, so a few degrees of horizontal offset is expected -- this page
is for judging whether the surface contains the skyline the camera saw, not
for calibration. `export_truth_video` covers every keyframe and measures the
offset.

    bazel run //experimental/overhead_matching/swag/farfield/dem_baseline:export_truth_compare -- \
        --dataset_dir /data/farfield_matching/datasets/mount_washington_20260815_leg2 \
        --surface /data/farfield_matching/artifacts/dem_surfaces/mount_washington/v2/surface \
        --out_dir /data/farfield_matching/runs/dem_baseline_dev/truth_compare/leg2_v2 \
        --n_frames 10
"""

import argparse
import json
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.farfield.dem_baseline import (
    depth_render,
    terrain,
    truth_strips,
    viz,
)
from experimental.overhead_matching.swag.farfield.viewers import page

GENERATOR = "dem_baseline.export_truth_compare"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True,
                        help="HeightField base path (no suffix)")
    parser.add_argument("--background", type=Path, action="append",
                        default=[],
                        help="Coarse HeightField for the far field and for "
                             "holes in --surface (base path, no suffix); "
                             "repeatable, fine to coarse")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--n_frames", type=int, default=10)
    parser.add_argument("--frame_ids", nargs="*", default=None,
                        help="explicit frame ids (e.g. f0123) instead")
    parser.add_argument("--elev_min_deg", type=float, default=-20.0)
    parser.add_argument("--elev_max_deg", type=float, default=20.0)
    parser.add_argument("--n_az", type=int, default=1440)
    parser.add_argument("--n_rows", type=int, default=160)
    parser.add_argument("--observer_height_m", type=float, default=3.0)
    parser.add_argument("--max_range_m", type=float, default=30000.0)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    hf = terrain.HeightField.load(args.surface)
    tt = depth_render.TerrainTensor.chain_from_height_fields(
        [hf] + [terrain.HeightField.load(p) for p in args.background],
        device=args.device)
    config = depth_render.RenderConfig(
        max_range_m=args.max_range_m,
        observer_height_m=args.observer_height_m)

    nominal = json.loads(
        (args.dataset_dir / "nominal_forward.json").read_text())
    bearing_camera_cw = float(nominal["bearing_camera_cw_deg"])
    frames = truth_strips.load_frames(args.dataset_dir)
    course_track = truth_strips.GridCourse(frames, hf.crs)

    if args.frame_ids:
        wanted = set(args.frame_ids)
        picked = [(i, f) for i, f in enumerate(frames)
                  if f["frame_file"].split(",")[0] in wanted]
    else:
        idx = np.unique(np.linspace(0, len(frames) - 1,
                                    args.n_frames).astype(int))
        picked = [(int(i), frames[int(i)]) for i in idx]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_html = []
    for i, frame in picked:
        frame_id = truth_strips.frame_id(frame)
        course = course_track.course_deg(i)
        if course is None:
            rows_html.append(
                f"<p class='muted'>{page.esc(frame_id)}: skipped "
                "(the GPS course model abstained)</p>")
            continue
        center_yaw = (course - bearing_camera_cw) % 360.0
        x, y = course_track.xy(i)
        cyl = depth_render.render_cylinder(
            tt, config, x, y, n_az=args.n_az,
            elev_min_deg=args.elev_min_deg, elev_max_deg=args.elev_max_deg,
            n_rows=args.n_rows)
        depth = cyl.depth_m.cpu().numpy()
        depth_img = viz.depth_image(depth, max_px=args.n_az)

        pano = truth_strips.open_pano(
            args.dataset_dir / "panorama" / frame["frame_file"],
            args.n_az * 2)
        photo = truth_strips.pano_strip(
            pano, center_yaw, elev_min_deg=args.elev_min_deg,
            elev_max_deg=args.elev_max_deg, n_az=args.n_az,
            n_rows=args.n_rows)

        combined = Image.new(
            "RGB", (args.n_az, args.n_rows * 2 + 4), (24, 24, 24))
        combined.paste(
            truth_strips.add_compass_ticks(
                truth_strips.draw_horizon(
                    Image.fromarray(photo),
                    truth_strips.horizon_rows(depth)),
                course_deg=course),
            (0, 0))
        combined.paste(
            truth_strips.add_compass_ticks(depth_img.convert("RGB")),
            (0, args.n_rows + 4))
        png_name = f"{frame_id}.png"
        combined.save(args.out_dir / png_name)

        rows_html.append(
            f"<h3>{page.esc(frame_id)}</h3>"
            f"<p class='muted'>lat/lon {float(frame['latitude']):.6f}, "
            f"{float(frame['longitude']):.6f} &middot; course "
            f"{course:.1f}&deg; &middot; pano centre yaw {center_yaw:.1f}"
            f"&deg; &middot; observer z {cyl.observer_z_m:.1f} m &middot; "
            f"source coverage {cyl.coverage:.2f}</p>"
            f"<img src='{png_name}' style='width:100%'>")

    body = (
        "<p>Photo strip (top) vs cylindrical depth render at the truth pose "
        "(bottom), both rolled to grid north with CW azimuth; elevation band "
        f"[{args.elev_min_deg:.0f}&deg;, {args.elev_max_deg:.0f}&deg;]. "
        "Heading from course-over-ground + approved nominal forward "
        f"({bearing_camera_cw:.1f}&deg; camera CW), so expect a few degrees "
        "of horizontal offset.</p>"
        f"<p class='muted'>surface: {page.esc(str(args.surface))}</p>"
        + "".join(rows_html))
    (args.out_dir / "viewer.html").write_text(page.page(
        f"truth compare: {args.dataset_dir.name}", body,
        generator=GENERATOR))
    print(f"wrote {args.out_dir}/viewer.html ({len(rows_html)} frames)")


if __name__ == "__main__":
    main()
