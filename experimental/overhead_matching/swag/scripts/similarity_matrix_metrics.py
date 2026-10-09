"""Retrieval metrics (recall@k, MRR) for precomputed (num_panos, num_sats) similarity matrices.

Example:
  bazel run //experimental/overhead_matching/swag/scripts:similarity_matrix_metrics -- \
    --dataset_path $DATA_ROOT/veluwe --landmark_version netherlands_veluwe_v1_250101 \
    --similarity_matrix_path $DATA_ROOT/veluwe/similarity_matrices/wag_no_hinge.pt [more .pt files]
"""
import argparse
import json
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.evaluation import retrieval_metrics
from experimental.overhead_matching.swag.scripts.export_correspondence_similarity import load_vigor_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--landmark_version", required=True)
    parser.add_argument("--similarity_matrix_path", type=Path, nargs="+", required=True)
    parser.add_argument("--ks", default="1,5,10")
    args = parser.parse_args()

    dataset = load_vigor_dataset(args.dataset_path, args.landmark_version, 1.0)
    ks = [int(k) for k in args.ks.split(",")]
    for path in args.similarity_matrix_path:
        sim = torch.load(path, weights_only=False)
        sim = sim["similarity"] if isinstance(sim, dict) else sim
        metrics = retrieval_metrics.compute_top_k_metrics(sim.float(), dataset, ks=ks)
        print(path, json.dumps({k: round(float(v), 4) for k, v in metrics.items()}))


if __name__ == "__main__":
    main()
