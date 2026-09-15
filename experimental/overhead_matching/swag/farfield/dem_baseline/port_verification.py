"""CLD-0 exit criterion: verify the PyTorch port against the released model's
own validation numbers on the released data.

The released ``output.log`` for AlpsPhotosToDepthCompact_31_2 records, for the
released epoch-39 checkpoint, validation recalls on GeoPose3K_v2
original->depth (Swiss) at 400 m:

    62.60 @ 1     85.27 @ 10     93.99 @ 100

This tool embeds the released 516 val query RGB photos and 6192 database
depth EXRs with the ported network, ranks by Euclidean descriptor distance,
and computes position recall@k within a radius from the released UTM
coordinates. Matching the logged numbers verifies weights, architecture,
preprocessing, and the depth-encoding contract end to end -- with no
TensorFlow environment.

    bazel run //experimental/overhead_matching/swag/farfield/dem_baseline:port_verification -- \
        --meta .../meta_structures/sparse_dataset/GeoPose3K_v2_original_to_depth_swiss_val.npy \
        --database_dir .../sparse_dataset/database_depth \
        --queries_dir .../sparse_dataset/query_original \
        --out .../models/crosslocate/AlpsPhotosToDepthCompact_31_2/port_verification.json
"""

import argparse
import json
import os
import time
from pathlib import Path

# Must precede the cv2 import; opencv's EXR reader is env-gated.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import common.torch.load_torch_deps  # noqa: F401
import torch

import cv2
import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.farfield.dem_baseline import (
    crosslocate_net,
)

RELEASED_EPOCH39_VAL_RECALL_400M = {1: 62.60, 10: 85.27, 100: 93.99}
SKY_VALUE = -1.0  # measured from released EXRs; see DEPTH_ENCODING.md


def load_meta(path: Path) -> dict:
    return np.load(path, allow_pickle=True).item()


def embed_depth_files(model, database_dir: Path, filenames: list,
                      device: str, batch_size: int = 32) -> torch.Tensor:
    out = torch.zeros(len(filenames), crosslocate_net.DESCRIPTOR_DIM)
    batch, idxs = [], []

    def flush():
        nonlocal batch, idxs
        if not batch:
            return
        with torch.inference_mode():
            out[idxs] = model(torch.stack(batch).to(device)).cpu()
        batch, idxs = [], []

    for i, fn in enumerate(filenames):
        depth = cv2.imread(str(database_dir / fn), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(database_dir / fn)
        if depth.ndim == 3:
            # cv2 returns BGR: the EXR's R channel lands at index 2; indices
            # 0/1 are zero-filled (verified against a direct OpenEXR read).
            depth = depth[:, :, 2]
        # Released EXRs already use the sky sentinel; pass through unchanged.
        batch.append(crosslocate_net.depth_render_tensor(
            depth.astype(np.float32), sky_fill_m=SKY_VALUE))
        idxs.append(i)
        if len(batch) == batch_size:
            flush()
            if (i + 1) % 512 == 0:
                print(f"  db {i + 1}/{len(filenames)}", flush=True)
    flush()
    return out


def embed_query_files(model, queries_dir: Path, filenames: list,
                      device: str, batch_size: int = 32) -> torch.Tensor:
    out = torch.zeros(len(filenames), crosslocate_net.DESCRIPTOR_DIM)
    batch, idxs = [], []

    def flush():
        nonlocal batch, idxs
        if not batch:
            return
        with torch.inference_mode():
            out[idxs] = model(torch.stack(batch).to(device)).cpu()
        batch, idxs = [], []

    for i, fn in enumerate(filenames):
        rgb = np.asarray(Image.open(queries_dir / fn).convert("RGB"))
        batch.append(crosslocate_net.rgb_query_tensor(rgb))
        idxs.append(i)
        if len(batch) == batch_size:
            flush()
    flush()
    return out


def position_recall(query_desc: torch.Tensor, db_desc: torch.Tensor,
                    q_utm: np.ndarray, db_utm: np.ndarray,
                    radius_m: float, ks: tuple) -> dict:
    dists = crosslocate_net.descriptor_distances(query_desc, db_desc)
    _, order = torch.sort(dists, dim=1)
    order = order.numpy()
    recall = {}
    for k in ks:
        hits = 0
        for qi in range(len(q_utm)):
            cand = db_utm[order[qi, :k]]
            err = np.hypot(cand[:, 0] - q_utm[qi, 0],
                           cand[:, 1] - q_utm[qi, 1])
            hits += bool((err <= radius_m).any())
        recall[k] = 100.0 * hits / len(q_utm)
    return recall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--database_dir", type=Path, required=True)
    parser.add_argument("--queries_dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path(
        "/data/farfield_matching/models/crosslocate/"
        "AlpsPhotosToDepthCompact_31_2/converted_weights.npz"))
    parser.add_argument("--radius_m", type=float, default=400.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    meta = load_meta(args.meta)
    q_utm = np.array([(x, y) for x, y in meta["qUTM"]])
    db_utm = np.array([(x, y) for x, y in meta["dbUTM"]])
    print(f"{len(meta['qImageFns'])} queries, {len(meta['dbImageFns'])} "
          f"database views")

    model = crosslocate_net.CrossLocateVGG16MAC().to(args.device).eval()
    crosslocate_net.load_converted_weights(model, args.weights)

    started = time.monotonic()
    db_desc = embed_depth_files(model, args.database_dir,
                                meta["dbImageFns"], args.device)
    q_desc = embed_query_files(model, args.queries_dir,
                               meta["qImageFns"], args.device)
    print(f"embedding took {time.monotonic() - started:.0f}s")

    ks = tuple(RELEASED_EPOCH39_VAL_RECALL_400M)
    recall = position_recall(q_desc, db_desc, q_utm, db_utm,
                             args.radius_m, ks)
    report = {
        "schema": "dem_baseline_port_verification/v1",
        "meta": str(args.meta),
        "weights": str(args.weights),
        "radius_m": args.radius_m,
        "ported_recall": {str(k): round(recall[k], 2) for k in ks},
        "released_epoch39_val_recall_400m":
            {str(k): v for k, v in RELEASED_EPOCH39_VAL_RECALL_400M.items()},
    }
    for k in ks:
        expected = RELEASED_EPOCH39_VAL_RECALL_400M[k]
        print(f"recall@{k:>3} within {args.radius_m:.0f} m: "
              f"ported {recall[k]:6.2f}  released {expected:6.2f}  "
              f"delta {recall[k] - expected:+.2f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
