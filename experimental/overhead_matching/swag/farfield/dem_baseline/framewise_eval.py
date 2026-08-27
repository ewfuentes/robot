"""Framewise CrossLocate-Depth retrieval over a farfield dataset (CLD-2).

For every panorama: extract the query crop ring, embed, jointly score against
the reference database, and record top-k candidates. Reports recall@k within
the plan's radii plus rank statistics against evaluation-only GPS truth.

Truth is never an input to retrieval; it is read only to score results. The
database and its lattice must have been built from the declared region, not
from the route (render_db records this in its manifest).

    bazel run //experimental/overhead_matching/swag/farfield/dem_baseline:framewise_eval -- \
        --dataset mount_washington_20260815_leg3 \
        --db_dir /data/farfield_matching/artifacts/depth_render_db/mount_washington/v1 \
        --weights .../converted_weights.npz \
        --out_dir /data/farfield_matching/runs/dem_baseline_dev/leg3_framewise_v1
"""

import argparse
import json
import time
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.dem_baseline import (
    crosslocate_net,
    panorama_score,
    query_crops,
    render_db,
    terrain,
)

RECALL_RADII_M = (25.0, 50.0, 100.0, 250.0, 500.0, 1000.0)
RECALL_KS = (1, 5, 10, 50, 100)


def load_truth_utm(frames_gps_csv: Path, crs: str) -> dict[str, tuple]:
    """frame_file stem -> (x, y) in the database CRS."""
    import csv
    import pyproj
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs,
                                              always_xy=True)
    truth = {}
    with open(frames_gps_csv) as file_in:
        for row in csv.DictReader(file_in):
            x, y = transformer.transform(float(row["longitude"]),
                                         float(row["latitude"]))
            truth[Path(row["frame_file"]).stem] = (x, y)
    return truth


def evaluate_frame(pano_path: Path, model, db, crop_config,
                   device: str) -> dict:
    pano = np.asarray(Image.open(pano_path).convert("RGB"))
    ring = query_crops.extract_crop_ring(pano, crop_config)
    with torch.inference_mode():
        batch = torch.stack([
            crosslocate_net.rgb_query_tensor(crop) for crop in ring
        ]).to(device)
        descriptors = model(batch)
        scores = panorama_score.joint_scores(descriptors, db["descriptors"])
        values, loc_idx, shift_idx = scores.top_k(max(RECALL_KS))
    return {
        "top_values": values.cpu().numpy(),
        "top_loc_idx": loc_idx.cpu().numpy(),
        "top_shift_idx": shift_idx.cpu().numpy(),
        "score_entropy": float(torch.special.entr(
            torch.softmax(scores.scores.reshape(-1).double(), dim=0)
        ).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    paths_lib.add_arguments(parser)
    parser.add_argument("--db_dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_frames", type=int, default=None)
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
    truth = load_truth_utm(paths.frames_gps, crs)

    model = crosslocate_net.CrossLocateVGG16MAC().to(args.device).eval()
    crosslocate_net.load_converted_weights(model, args.weights)

    panos = sorted(paths.panorama_dir.glob("*.jpg"))[::args.frame_stride]
    if args.max_frames:
        panos = panos[:args.max_frames]
    if not panos:
        raise SystemExit(f"no panoramas in {paths.panorama_dir}")

    db_xy = np.stack([db["x_m"], db["y_m"]], axis=1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_records = []
    started = time.monotonic()
    with open(args.out_dir / "frames.jsonl", "w") as frames_out:
        for i, pano_path in enumerate(panos):
            stem = pano_path.stem
            if stem not in truth:
                print(f"skip {stem}: no truth row")
                continue
            result = evaluate_frame(pano_path, model, db, crop_config,
                                    args.device)
            tx, ty = truth[stem]
            cand_xy = db_xy[result["top_loc_idx"]]
            dists = np.hypot(cand_xy[:, 0] - tx, cand_xy[:, 1] - ty)
            record = {
                "stem": stem,
                "truth_xy": [tx, ty],
                "top_loc_idx": result["top_loc_idx"].tolist(),
                "top_shift_idx": result["top_shift_idx"].tolist(),
                "top_scores": [round(float(v), 5)
                               for v in result["top_values"]],
                "top_dist_m": [round(float(d), 1) for d in dists],
                "score_entropy": round(result["score_entropy"], 3),
            }
            frames_out.write(json.dumps(record) + "\n")
            frame_records.append(record)
            if (i + 1) % 20 == 0:
                rate = (i + 1) / (time.monotonic() - started)
                print(f"{i + 1}/{len(panos)} frames ({rate:.2f}/s), "
                      f"last top1 err {dists[0]:.0f} m", flush=True)

    all_dists = np.array([r["top_dist_m"] for r in frame_records])
    recall = {
        f"recall@{k}_within_{int(radius)}m": float(
            (all_dists[:, :k].min(axis=1) <= radius).mean())
        for k in RECALL_KS for radius in RECALL_RADII_M
    }
    top1 = all_dists[:, 0]
    summary = {
        "schema": "dem_baseline_framewise_eval/v1",
        "dataset": paths.dataset,
        "db_dir": str(args.db_dir),
        "weights": str(args.weights),
        "n_frames": len(frame_records),
        "frame_stride": args.frame_stride,
        "lattice_spacing_m": db["manifest"]["lattice"]["spacing_m"],
        "top1_error_m": {
            "median": float(np.median(top1)),
            "p25": float(np.percentile(top1, 25)),
            "p75": float(np.percentile(top1, 75)),
            "mean": float(top1.mean()),
        },
        "recall": recall,
        "note": "diagnostic run; heading error not evaluated (needs "
                "mount-offset truth), protocol not frozen",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["top1_error_m"], indent=1))
    for k in RECALL_KS:
        line = "  ".join(
            f"r@{k}<{int(radius)}m={recall[f'recall@{k}_within_{int(radius)}m']:.3f}"
            for radius in RECALL_RADII_M)
        print(line)
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
