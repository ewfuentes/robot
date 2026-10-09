"""Oracle top-k re-ranking of a (num_panos, num_sats) similarity matrix: an upper bound for CVG-Text's ERM.

ERM (GPT-4o re-ranking) only permutes each query's top-k candidates. The oracle does the best any such
permutation can: when a positive tile is among a panorama's top-k, it is moved to rank 1 and the tiles it
passes shift down one rank, keeping the row's multiset of top-k scores. Rows with no positive in the top-k
are unchanged, as is everything outside the top-k.

Example:
  bazel run //experimental/overhead_matching/swag/scripts:oracle_rerank_similarity -- \
    --dataset_path /data/overhead_matching/datasets/VIGOR/NewYork --landmark_version v4_202001 \
    --similarity_matrix_path .../NewYork/similarity_matrices/ct2l_chicago_osm.pt --output_path .../ct2l_chicago_osm_oracle5.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.scripts.export_ct2l_similarity import load_dataset


def oracle_rerank(sim: torch.Tensor, positives: list[list[int]], k: int = 5) -> tuple[torch.Tensor, int]:
    """Returns (reranked copy, number of rows changed)."""
    out = sim.clone()
    values, idxs = torch.topk(sim, k, dim=1)
    changed = 0
    for row, pos in enumerate(positives):
        pos = set(pos)
        rank = next((r for r, c in enumerate(idxs[row].tolist()) if c in pos), None)
        if not rank:  # None (no positive in top-k) or 0 (already first)
            continue
        new_order = [idxs[row, rank]] + [idxs[row, r] for r in range(k) if r != rank]
        out[row, torch.stack(new_order)] = values[row]
        changed += 1
    return out, changed


def _demo():
    sim = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.1], [0.9, 0.8, 0.7, 0.6, 0.5, 0.1], [0.9, 0.8, 0.7, 0.6, 0.5, 0.1]])
    out, changed = oracle_rerank(sim, [[2], [0], [5]], k=5)
    assert changed == 1
    assert torch.equal(out[0], torch.tensor([0.8, 0.7, 0.9, 0.6, 0.5, 0.1]))  # positive 2 -> rank 1, tiles 0,1 shift down
    assert torch.equal(out[1:], sim[1:])  # already first / outside top-k: untouched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--landmark_version", required=True)
    parser.add_argument("--similarity_matrix_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    _demo()
    dataset = load_dataset(args.dataset_path, args.landmark_version)
    positives = [list(p) for p in dataset._panorama_metadata.positive_satellite_idxs]
    sim = torch.load(args.similarity_matrix_path, weights_only=True)
    assert sim.shape[0] == len(positives), (sim.shape, len(positives))
    out, changed = oracle_rerank(sim, positives, args.k)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output_path)
    stats = {"source": str(args.similarity_matrix_path), "k": args.k, "num_panos": len(positives),
             "rows_reranked": changed, "frac_reranked": changed / len(positives)}
    args.output_path.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
