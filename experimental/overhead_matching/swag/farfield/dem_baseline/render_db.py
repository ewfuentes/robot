"""Build a reference descriptor database: render depth rings over a lattice
and extract CrossLocate descriptors (plan section 5.1, steps 5-8).

Storage design: descriptors only, (n_locations, n_theta, 512) float16, plus a
manifest and a small archived render sample for regression tests. Dense
renders for a full lattice would be hundreds of GB and are cheap to
regenerate on GPU from the surface + manifest.

Run via bazel:

    bazel run //experimental/overhead_matching/swag/farfield/dem_baseline:render_db -- \
        --height_field /data/farfield_matching/artifacts/dem_surfaces/mount_washington/v1/surface \
        --weights /data/farfield_matching/models/crosslocate/AlpsPhotosToDepthCompact_31_2/converted_weights.npz \
        --spacing_m 200 --max_range_m 30000 --sky_fill_m 0 \
        --output_dir /data/farfield_matching/artifacts/depth_render_db/mount_washington/v1
"""

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401  (must precede torch)
import torch

import numpy as np

from experimental.overhead_matching.swag.farfield.dem_baseline import (
    crosslocate_net,
    depth_render,
    lattice as lattice_lib,
    terrain,
)


def build_database(hf: terrain.HeightField, lat: lattice_lib.Lattice,
                   model: crosslocate_net.CrossLocateVGG16MAC,
                   render_config: depth_render.RenderConfig, *,
                   sky_fill_m: float, device: str = "cuda",
                   views_per_batch: int = 24,
                   progress_every: int = 200,
                   sample_render_indices: tuple[int, ...] = (),
                   backgrounds: Sequence[terrain.HeightField] = ()) -> dict:
    """Render + embed every lattice location.

    Returns {"descriptors": (N, n_theta, D) float16 array, "coverage":
    (N, n_theta) float32, "sample_renders": {loc_idx: (n_theta, H, W) f16}}.
    """
    tt = depth_render.TerrainTensor.chain_from_height_fields(
        [hf, *backgrounds], device=device)
    model = model.to(device).eval()
    n_theta = render_config.n_yaw
    descriptors = np.zeros((len(lat), n_theta, crosslocate_net.DESCRIPTOR_DIM),
                           dtype=np.float16)
    coverage = np.zeros((len(lat), n_theta), dtype=np.float32)
    sample_renders = {}
    started = time.monotonic()

    with torch.inference_mode():
        for i in range(len(lat)):
            ring = depth_render.render_ring(tt, render_config,
                                            float(lat.x_m[i]),
                                            float(lat.y_m[i]))
            coverage[i] = ring.coverage
            if i in sample_render_indices:
                sample_renders[i] = ring.depth_m.half().cpu().numpy()
            depth = torch.where(
                torch.isfinite(ring.depth_m), ring.depth_m,
                torch.full_like(ring.depth_m, sky_fill_m))
            views = depth.unsqueeze(1).expand(-1, 3, -1, -1)
            if views.shape[-2:] != crosslocate_net.NATIVE_INPUT_HW:
                views = torch.nn.functional.interpolate(
                    views, size=crosslocate_net.NATIVE_INPUT_HW, mode="area")
            for lo in range(0, n_theta, views_per_batch):
                batch = views[lo:lo + views_per_batch]
                descriptors[i, lo:lo + batch.shape[0]] = (
                    model(batch).half().cpu().numpy())
            if progress_every and (i + 1) % progress_every == 0:
                rate = (i + 1) / (time.monotonic() - started)
                print(f"{i + 1}/{len(lat)} locations "
                      f"({rate:.2f}/s, ~{(len(lat) - i - 1) / rate / 60:.0f} "
                      f"min left)", flush=True)

    return {"descriptors": descriptors, "coverage": coverage,
            "sample_renders": sample_renders}


def save_database(output_dir: Path, result: dict,
                  lat: lattice_lib.Lattice,
                  render_config: depth_render.RenderConfig,
                  manifest_extra: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "descriptors.npz",
        descriptors=result["descriptors"],
        coverage=result["coverage"],
        x_m=lat.x_m, y_m=lat.y_m)
    for idx, render in result["sample_renders"].items():
        np.savez_compressed(output_dir / f"sample_render_{idx:06d}.npz",
                            depth_m=render)
    manifest = {
        "schema": "dem_baseline_depth_render_db/v1",
        "n_locations": len(lat),
        "lattice": {
            "spacing_m": lat.spacing_m, "crs": lat.crs,
            "bounds_xy": list(lat.bounds_xy),
            "n_dropped_nodata": lat.n_dropped_nodata,
        },
        "render_config": {
            "n_yaw": render_config.n_yaw,
            "fov_deg": render_config.fov_deg,
            "width": render_config.width,
            "height": render_config.height,
            "observer_height_m": render_config.observer_height_m,
            "max_range_m": render_config.max_range_m,
            "min_range_m": render_config.min_range_m,
            "step_res_scale": render_config.step_res_scale,
            "step_angular": render_config.step_angular,
            "curvature": render_config.curvature,
            "refraction_k": render_config.refraction_k,
        },
    }
    manifest.update(manifest_extra)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


def load_database(db_dir: Path, device: str = "cpu") -> dict:
    db_dir = Path(db_dir)
    data = np.load(db_dir / "descriptors.npz")
    manifest = json.loads((db_dir / "manifest.json").read_text())
    return {
        "descriptors": torch.from_numpy(
            data["descriptors"]).float().to(device),
        "coverage": data["coverage"],
        "x_m": data["x_m"],
        "y_m": data["y_m"],
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", type=Path, action="append",
                        default=[],
                        help="Coarse HeightField consulted where the main "
                             "surface has no data (far field beyond its box, "
                             "and interior holes); base path, no suffix. "
                             "Repeatable, fine to coarse.")
    parser.add_argument("--height_field", type=Path, required=True,
                        help="Base path of a saved HeightField (no suffix)")
    parser.add_argument("--weights", type=Path, required=True,
                        help="converted_weights.npz: name-keyed dump of the release TF1 checkpoint")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--spacing_m", type=float, required=True)
    parser.add_argument("--bounds_xy", type=float, nargs=4, default=None,
                        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
                        help="Declared search bounds in the surface CRS "
                             "(default: full height field)")
    parser.add_argument("--max_range_m", type=float, default=30000.0)
    parser.add_argument("--observer_height_m", type=float, default=1.7)
    parser.add_argument("--sky_fill_m", type=float, required=True,
                        help="Finite depth standing in for sky, matching the "
                             "release encoding (a modeling choice: explicit)")
    parser.add_argument("--n_sample_renders", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    hf = terrain.HeightField.load(args.height_field)
    backgrounds = [terrain.HeightField.load(p) for p in args.background]
    lat = lattice_lib.build_lattice(
        hf, spacing_m=args.spacing_m,
        bounds_xy=tuple(args.bounds_xy) if args.bounds_xy else None,
        backgrounds=backgrounds)
    print(f"lattice: {len(lat)} locations "
          f"({lat.n_dropped_nodata} dropped for nodata)")

    render_config = depth_render.RenderConfig(
        max_range_m=args.max_range_m,
        observer_height_m=args.observer_height_m)
    model = crosslocate_net.CrossLocateVGG16MAC()
    crosslocate_net.load_converted_weights(model, args.weights)

    sample_indices = tuple(
        int(i) for i in np.linspace(0, max(len(lat) - 1, 0),
                                    args.n_sample_renders).round())
    result = build_database(hf, lat, model, render_config,
                            sky_fill_m=args.sky_fill_m, device=args.device,
                            sample_render_indices=sample_indices,
                            backgrounds=backgrounds)
    save_database(args.output_dir, result, lat, render_config, {
        "height_field": str(args.height_field),
        "backgrounds": [str(p) for p in args.background],
        "weights": str(args.weights),
        "sky_fill_m": args.sky_fill_m,
        "argv_bounds_xy": args.bounds_xy,
    })
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
