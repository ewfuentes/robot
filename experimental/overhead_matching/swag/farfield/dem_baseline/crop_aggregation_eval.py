"""A/B framewise eval of crop-ring aggregation rules (CLD occlusion study).

The joint score averages all M crops (panorama_score section 5.3). Under
partial occlusion (forest canopy, people, buildings) a few clean crops are
drowned by many occluded ones: on mount_washington leg1 truth sits in the
top-1% of candidates on 99% of frames but is top-1 on only 13%. This tool
scores every frame ONCE (the per-crop aligned tensor) and evaluates several
aggregation rules on it, so aggregation is compared with everything else
held fixed:

  mean        all M crops (the shipped rule; baseline)
  top<j>      mean of the j best-matching crops per (location, shift)
  trim<j>     mean after dropping the j worst crops
  soft<T>     softmax(T)-weighted mean over crops

Truth is read only to score results, never used by retrieval.

    bazel run //experimental/overhead_matching/swag/farfield/dem_baseline:crop_aggregation_eval -- \
        --dataset mount_washington_20260815_leg1 \
        --db_dir <artifacts>/depth_render_db/mount_washington/v1_dev100m \
        --weights <models>/crosslocate/AlpsPhotosToDepthCompact_31_2/converted_weights.npz
"""

import argparse
import time
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401  (must precede torch)
import torch

import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.dem_baseline import (
    crosslocate_net,
    framewise_eval,
    query_crops,
    render_db,
)

RADII_M = (100.0, 250.0, 500.0, 1000.0)


def aligned_tensor(query_descriptors: torch.Tensor,
                   database_descriptors: torch.Tensor) -> torch.Tensor:
    """(L, M, K): crop m against view (m+k) mod N at every location."""
    m = query_descriptors.shape[0]
    n_loc, n_theta, _ = database_descriptors.shape
    cos = torch.einsum("md,lnd->lmn", query_descriptors,
                       database_descriptors)
    m_idx = torch.arange(m, device=cos.device)
    k_idx = torch.arange(n_theta, device=cos.device)
    gather = (m_idx[:, None] + k_idx[None, :]) % n_theta
    return torch.gather(cos, 2, gather[None].expand(n_loc, -1, -1))


def aggregators(n_crops: int) -> dict:
    def top_j(j):
        return lambda a: a.topk(j, dim=1).values.mean(dim=1)

    def soft(temp):
        return lambda a: (torch.softmax(a / temp, dim=1) * a).sum(dim=1)

    rules = {"mean": lambda a: a.mean(dim=1)}
    for j in (2, 3, 4, 6):
        rules[f"top{j}"] = top_j(j)
    rules[f"trim4"] = top_j(n_crops - 4)
    rules["soft.05"] = soft(0.05)
    return rules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    paths_lib.add_arguments(parser)
    parser.add_argument("--db_dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    paths = paths_lib.resolve(parser, args,
                              require=("panorama_dir", "frames_gps"))

    db = render_db.load_database(args.db_dir, device=args.device)
    crs = db["manifest"]["lattice"]["crs"]
    n_theta = db["manifest"]["render_config"]["n_yaw"]
    crop_config = query_crops.CropRingConfig(
        n_crops=n_theta,
        fov_deg=db["manifest"]["render_config"]["fov_deg"])
    truth = framewise_eval.load_truth_utm(paths.frames_gps, crs)

    model = crosslocate_net.CrossLocateVGG16MAC().to(args.device).eval()
    crosslocate_net.load_converted_weights(model, args.weights)

    panos = [p for p in sorted(paths.panorama_dir.glob("*.jpg"))
             if p.stem in truth][::args.frame_stride]
    if not panos:
        raise SystemExit(f"no panoramas with truth in {paths.panorama_dir}")

    db_xy = np.stack([db["x_m"], db["y_m"]], axis=1)
    rules = aggregators(n_theta)
    errors = {name: [] for name in rules}
    started = time.monotonic()
    with torch.inference_mode():
        for i, pano_path in enumerate(panos):
            pano = np.asarray(Image.open(pano_path).convert("RGB"))
            ring = query_crops.extract_crop_ring(pano, crop_config)
            batch = torch.stack([
                crosslocate_net.rgb_query_tensor(crop) for crop in ring
            ]).to(args.device)
            aligned = aligned_tensor(model(batch), db["descriptors"])
            tx, ty = truth[pano_path.stem]
            dist = np.hypot(db_xy[:, 0] - tx, db_xy[:, 1] - ty)
            for name, rule in rules.items():
                scores = rule(aligned)  # (L, K)
                loc = int(torch.argmax(scores.max(dim=1).values))
                errors[name].append(dist[loc])
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.monotonic() - started)
                print(f"{i + 1}/{len(panos)} frames ({rate:.2f}/s)",
                      flush=True)

    print(f"\ndataset={paths.dataset}  n_frames={len(panos)}  "
          f"db={args.db_dir.name}")
    header = "rule     " + "".join(f"  r@1<{int(r)}m" for r in RADII_M) \
        + "   median_err"
    print(header)
    for name in rules:
        err = np.array(errors[name])
        cells = "".join(f"  {np.mean(err < r):8.3f}" for r in RADII_M)
        print(f"{name:8s} {cells}   {np.median(err):8.0f} m")


if __name__ == "__main__":
    main()
