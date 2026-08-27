"""Surveyed-pose renderer QA (CLD-1): render a ring at a known pose and check
that a named far landmark appears on the rendered horizon at its geodetically
expected azimuth and elevation.

    bazel run //experimental/overhead_matching/swag/farfield/dem_baseline:qa_render -- \
        --height_field .../dem_surfaces/mount_washington/v1/surface \
        --observer_latlon 44.2592 -71.3188 \
        --target_latlon 44.2705 -71.3032 --max_range_m 30000
"""

import argparse
import math
import time
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

import numpy as np

from experimental.overhead_matching.swag.farfield.dem_baseline import (
    depth_render,
    terrain,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height_field", type=Path, required=True)
    parser.add_argument("--observer_latlon", type=float, nargs=2,
                        required=True, metavar=("LAT", "LON"))
    parser.add_argument("--target_latlon", type=float, nargs=2, required=True,
                        metavar=("LAT", "LON"),
                        help="A prominent far landmark (e.g. a summit)")
    parser.add_argument("--observer_height_m", type=float, default=1.7)
    parser.add_argument("--max_range_m", type=float, default=30000.0)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_npz", type=Path, default=None,
                        help="Optionally save the rendered ring for viewing")
    args = parser.parse_args()

    hf = terrain.HeightField.load(args.height_field)
    ox, oy = terrain.utm_from_latlon(*args.observer_latlon, hf.crs)
    tx, ty = terrain.utm_from_latlon(*args.target_latlon, hf.crs)
    obs_z = float(hf.sample(ox, oy)) + args.observer_height_m
    target_z = float(hf.sample(tx, ty))
    dist = math.hypot(tx - ox, ty - oy)
    expected_az = math.degrees(math.atan2(tx - ox, ty - oy)) % 360.0
    expected_el = math.degrees(math.atan2(target_z - obs_z, dist))
    print(f"observer z={obs_z:.1f} m, target z={target_z:.1f} m, "
          f"distance {dist:.0f} m")
    print(f"expected target az={expected_az:.2f} el={expected_el:.2f} deg")

    tt = depth_render.TerrainTensor.from_height_field(hf, device=args.device)
    config = depth_render.RenderConfig(max_range_m=args.max_range_m,
                                       observer_height_m=args.observer_height_m)
    started = time.monotonic()
    ring = depth_render.render_ring(tt, config, ox, oy)
    if args.device == "cuda":
        torch.cuda.synchronize()
    print(f"render time {time.monotonic() - started:.2f}s on {args.device}")

    # Rendered horizon (highest non-sky pixel) in the column nearest the
    # target's azimuth, in the view whose axis is closest to it. Nearby
    # terrain can legitimately stand taller elsewhere on the horizon, so the
    # check is at the target's azimuth, not a global max.
    pixel_elev = depth_render.row_elevation_angles_rad(config, args.device)
    az = depth_render.column_azimuths_rad(
        config, torch.deg2rad(torch.tensor(
            config.yaw_degrees(), device=args.device, dtype=torch.float32)))
    az_err = torch.remainder(
        az - math.radians(expected_az) + math.pi, 2.0 * math.pi) - math.pi
    view, col = divmod(int(az_err.abs().reshape(-1).argmin()), config.width)
    finite_col = torch.isfinite(ring.depth_m[view, :, col])
    if not bool(finite_col.any()):
        print("QA FAIL: all-sky column at the target azimuth")
        return
    col_elev = pixel_elev[:, col]
    horizon_row = int(torch.nonzero(finite_col)[0])  # rows go top-down
    found_el = math.degrees(float(col_elev[horizon_row]))
    depth_at = ring.depth_m[view, horizon_row, col].item()
    found_az = math.degrees(float(az[view, col])) % 360.0
    print(f"rendered horizon at az={found_az:.2f} (view {view}, col {col}): "
          f"el={found_el:.2f} deg, depth={depth_at:.0f} m")
    print(f"horizon-vs-target elevation error {found_el - expected_el:+.2f} "
          f"deg (positive can be a nearer ridge; large negative means the "
          f"target is missing from the render)")
    print(f"depth-vs-straight-line ratio {depth_at / dist:.3f}")

    d0 = ring.depth_m[0]
    finite0 = torch.isfinite(d0)
    print(f"view0: finite fraction {finite0.float().mean():.3f}, min "
          f"{d0[finite0].min():.1f} m, median {d0[finite0].median():.0f} m")
    print("coverage per view:", np.round(ring.coverage, 3))

    if args.save_npz:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_npz,
                            depth_m=ring.depth_m.cpu().numpy(),
                            yaw_deg=ring.yaw_deg)
        print(f"saved {args.save_npz}")


if __name__ == "__main__":
    main()
