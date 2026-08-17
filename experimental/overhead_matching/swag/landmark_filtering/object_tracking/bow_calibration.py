"""Bow azimuth in the camera frame, from the vessel's own structure.

**This does NOT measure `mount_offset_deg`.** Decided 2026-08-14: the offset the
pipeline consumes is the azimuth of the **direction of travel** in the camera
frame, because that is what `gps_to_odometry` declares as body x when it sets
`left_m = 0`, and it is what `bearing_matcher.estimate_mount_offset` already
solves for via `bearing_world = course + (bearing_camera - offset)`. The bow
differs from the direction of travel by the crab angle, so a bow reading is a
*corroborating* measurement, not a substitute. Feeding one in as the offset
injects the crab angle as a constant bearing bias, and under design doc 5.2
position routes through heading, so that bias rotates the whole dead-reckoned
track.

For boston_harbor leg1 this tool is doubly unsuitable: the deckhouse occludes
the bow, so the bow tip appears in no frame, and three hand readings disagreed
by 30 deg (deckhouse centreline ~190, wake ~218, LT20-to-One-International-Place
222). The leg's accepted offset, **214 deg**, came from sweeping the offset to
minimise the median triangulation residual over 26 tracklets and was verified
against a surveyed building over 72 keyframes (mean +0.6 deg, std 2.42 deg).
That sweep, not this tool, is the reference method.

What remains useful here: the temporal-median render itself, as a way to see
what is rigidly attached to the camera at full resolution. For an automatic,
thresholded version of the same idea see
`swag/scripts/detect_vehicle_anchor.py`, which normalises by local texture and
uses only the x-derivative so it does not also flag the horizon.

The world sweeps past across a leg while the bow, rails and canopy stay on the
same pixels, so a per-pixel temporal median renders the boat sharp and
everything else washed toward the median of whatever passed behind it.

Why bother: an eyeball estimate at 1280 px preview width is worth about
+/-2.8 deg, and position error floors at ~range * sigma_heading - roughly
245 m at a 5 km landmark. One pass over the panoramas buys ~0.1 deg.

This tool produces the median image plus a "staticness" map (per-pixel
temporal spread, low = rigidly attached to the camera). It does NOT
auto-detect the bow: the vessel's silhouette differs per rig, so the offset
is read off the rendered strip by a human and passed back as
--mount_offset_deg to the pipeline. What the tool guarantees is that the
reading is made on evidence at full resolution.

Run:
  bazel run //...object_tracking:bow_calibration -- \\
      --dataset_base /data/.../processed/leg1 --out_dir <dir>
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    pano_geometry as pg,
)

DEFAULT_DATASET = Path("/data/farfield_matching/datasets/boston_harbor_leg1")


def temporal_stats(pano_paths, max_frames: int, band: tuple[float, float],
                   column_block: int = 512, scratch_dir=None):
    """(median, spread) over a horizontal band of the panoramas.

    Memory discipline matters here and the obvious implementation is a trap.
    Holding 120 frames of a 7680-wide band is ~5 GB as uint8, but the naive
    spread computation - `abs(stack.astype(int16) - median)` - upcasts the
    WHOLE stack and then builds two more full-size temporaries, peaking near
    35 GB. That is enough to OOM a 62 GB machine that is also running the
    tracker, the viewers and several agents, and it does so silently: the
    same call succeeds when the box happens to be idle.

    So the stack goes to a memory-mapped scratch file (the OS pages it) and
    both statistics are computed in column blocks, accumulating the spread
    one frame at a time so no full-size intermediate is ever materialised.
    Peak RSS is one block: n_frames x band_rows x column_block x 3 bytes.
    """
    probe = np.asarray(Image.open(pano_paths[0]))
    h, w = probe.shape[:2]
    y0, y1 = int(band[0] * h), int(band[1] * h)
    rows = y1 - y0

    step = max(1, len(pano_paths) // max_frames)
    picked = pano_paths[::step][:max_frames]
    n = len(picked)

    scratch = Path(tempfile.mkdtemp(dir=scratch_dir, prefix="bowcal_"))
    stack_path = scratch / "band.u8"
    try:
        stack = np.memmap(stack_path, dtype=np.uint8, mode="w+",
                          shape=(n, rows, w, 3))
        for i, path in enumerate(picked):
            stack[i] = np.asarray(Image.open(path))[y0:y1]
            if i % 25 == 0:
                print(f"  loaded {i + 1}/{n}")
        stack.flush()

        median = np.empty((rows, w, 3), dtype=np.uint8)
        spread = np.empty((rows, w), dtype=np.float32)
        for x0 in range(0, w, column_block):
            x1 = min(w, x0 + column_block)
            block = np.asarray(stack[:, :, x0:x1, :])      # one block in RAM
            block_median = np.median(block, axis=0).astype(np.uint8)
            median[:, x0:x1] = block_median
            reference = block_median.astype(np.int16)
            accumulator = np.zeros((rows, x1 - x0), dtype=np.float32)
            for i in range(n):
                accumulator += np.abs(
                    block[i].astype(np.int16) - reference).mean(axis=2)
            spread[:, x0:x1] = accumulator / n
            del block
        del stack
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return median, spread, (y0, y1), (h, w), n


def azimuth_ruler(draw, width, height, pano_w, every_deg=10):
    """Tick marks labelled with camera azimuth, so the strip can be read
    directly in the units the offset is expressed in."""
    for az in range(0, 360, every_deg):
        x = pg.pano_px_from_direction(az, 0.0, pano_w, 2)[0]
        if not 0 <= x < width:
            continue
        major = az % 30 == 0
        draw.line([(x, 0), (x, 26 if major else 14)],
                  fill=(255, 230, 60), width=3 if major else 1)
        if major:
            draw.text((x + 4, 28), f"{az}", fill=(255, 230, 60))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_base", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_frames", type=int, default=150,
                        help="Panoramas sampled evenly across the leg")
    parser.add_argument("--band", type=float, nargs=2, default=[0.45, 0.95],
                        help="Vertical band (fractions) holding the vessel")
    parser.add_argument("--column_block", type=int, default=512,
                        help="Columns processed at once. Peak RSS is "
                             "n_frames x band_rows x column_block x 3 bytes; "
                             "lower it on a busy machine.")
    parser.add_argument("--scratch_dir", default=None,
                        help="Where the memory-mapped frame stack lives "
                             "(defaults to the system temp dir)")
    parser.add_argument("--mount_offset_deg", type=float, default=None,
                        help="If given, draw this azimuth on the strip to "
                             "check a candidate reading")
    args = parser.parse_args()

    pano_paths = sorted((args.dataset_base / "panorama").glob("*.jpg"))
    if not pano_paths:
        raise SystemExit(f"no panoramas under {args.dataset_base}/panorama")
    print(f"{len(pano_paths)} panoramas; sampling {args.max_frames}")

    median, spread, (y0, y1), (h, w), n_used = temporal_stats(
        pano_paths, args.max_frames, tuple(args.band),
        column_block=args.column_block, scratch_dir=args.scratch_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Staticness: low spread = same pixels every frame = attached to the rig.
    norm = spread / max(1e-6, np.percentile(spread, 99))
    static = np.clip(1.0 - norm, 0.0, 1.0)
    Image.fromarray((static * 255).astype(np.uint8)).save(
        args.out_dir / "staticness.png")
    Image.fromarray(median).save(args.out_dir / "median.jpg", quality=92)

    # Column-wise staticness profile: the vessel shows as a broad plateau.
    profile = static.mean(axis=0)
    az_of_col = np.array([pg.direction_from_pano_px(x, 0.0, w, 2)[0]
                          for x in range(w)])
    peak_col = int(np.argmax(profile))
    print(f"median/staticness written for band y=[{y0},{y1}] of {h}")
    print(f"most-static column x={peak_col} -> camera azimuth "
          f"{az_of_col[peak_col]:.2f} deg (peak of the profile, NOT "
          f"necessarily the bow - read the strip)")

    # Annotated strip: median image with an azimuth ruler.
    strip = Image.fromarray(median).convert("RGB")
    draw = ImageDraw.Draw(strip)
    azimuth_ruler(draw, strip.width, strip.height, w)
    if args.mount_offset_deg is not None:
        x = pg.pano_px_from_direction(args.mount_offset_deg, 0.0, w, 2)[0]
        draw.line([(x, 0), (x, strip.height)], fill=(60, 255, 90), width=5)
        draw.text((x + 8, 60), f"claimed bow {args.mount_offset_deg:.1f}",
                  fill=(60, 255, 90))
    strip.save(args.out_dir / "median_with_azimuth.jpg", quality=92)

    (args.out_dir / "calibration.json").write_text(json.dumps({
        "n_panoramas_total": len(pano_paths),
        "n_panoramas_used": n_used,
        "pano_size": [w, h],
        "band_px": [y0, y1],
        "most_static_column_px": peak_col,
        "most_static_column_azimuth_deg": float(az_of_col[peak_col]),
        "claimed_mount_offset_deg": args.mount_offset_deg,
        "note": "mount_offset_deg is READ from median_with_azimuth.jpg; the "
                "most-static column is a hint, not a bow detector",
    }, indent=1))
    print(f"wrote {args.out_dir}/median_with_azimuth.jpg")


if __name__ == "__main__":
    main()
