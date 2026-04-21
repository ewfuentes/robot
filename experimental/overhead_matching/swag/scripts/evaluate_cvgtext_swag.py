"""Run our WAG pano-to-satellite retrieval model on CVG-Text.

Experiment A of the CVG-Text comparison plan: a direct image-to-image
retrieval baseline, measured against the full per-city satellite gallery
(same galleries as Step 0's CrossText2Loc runs, no 100-nearest prior).

Loads a trained `WagPatchEmbedding` pano/sat pair from the directory layout
produced by our training pipeline (e.g.
`/data/overhead_matching/training_outputs/260215_baseline_retraining/260218_093851_all_chicago_dinov3_wag_bs18_v2/`),
builds a `CVGTextDataset` for the requested test city, and writes
`similarity.pt` + `metrics.json` + `config.json` under
`<output_base>/expA_test-<city>_<wag_tag>/`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.data.cvgtext_dataset import CVGTextDataset
from experimental.overhead_matching.swag.evaluation import evaluate_swag, retrieval_metrics
from experimental.overhead_matching.swag.scripts.export_similarity_matrix import (
    load_models_from_training_output,
)


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/data/overhead_matching/datasets/cvgtext"),
    )
    parser.add_argument("--test_city", required=True, choices=("Brisbane", "NewYork", "Tokyo"))
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--wag_checkpoint_dir",
        type=Path,
        required=True,
        help="Directory containing `<checkpoint>_panorama/` and `<checkpoint>_satellite/` subdirs.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Which checkpoint tag to load — typically 'best', 'last', or a numeric epoch tag.",
    )
    parser.add_argument(
        "--wag_tag",
        default=None,
        help="Output dir suffix. Defaults to the checkpoint dir's basename.",
    )
    parser.add_argument(
        "--output_base",
        type=Path,
        default=Path("/data/overhead_matching/evaluation/results/cvgtext"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    pano_model, sat_model, resolved_ckpt = load_models_from_training_output(
        args.wag_checkpoint_dir,
        device=device,
        checkpoint=args.checkpoint,
        fallback_to_config=True,
    )

    pano_patch_dims = tuple(pano_model.patch_dims)
    sat_patch_dims = tuple(sat_model.patch_dims)
    print(f"pano patch_dims={pano_patch_dims}  sat patch_dims={sat_patch_dims}")

    dataset = CVGTextDataset(
        root=args.dataset_root,
        city=args.test_city,
        split=args.split,
        gallery_kind="satellite",
        panorama_size=pano_patch_dims,
        satellite_patch_size=sat_patch_dims,
    )
    print(
        f"CVG-Text: test={args.test_city}  |  "
        f"{dataset.num_queries} queries, {dataset.num_gallery} gallery entries"
    )

    similarity = evaluate_swag.compute_similarity_matrix(
        sat_model=sat_model,
        pano_model=pano_model,
        dataset=dataset,
        device=device,
    )
    print(f"similarity shape: {tuple(similarity.shape)}")

    metrics = retrieval_metrics.compute_top_k_metrics(
        similarity.cpu(), dataset, ks=[1, 5, 10]
    )
    print("Metrics:", metrics)

    wag_tag = args.wag_tag or args.wag_checkpoint_dir.name
    out_dir = args.output_base / f"expA_test-{args.test_city}_{wag_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(similarity.cpu(), out_dir / "similarity.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                "test_city": args.test_city,
                "split": args.split,
                "wag_checkpoint_dir": str(args.wag_checkpoint_dir),
                "resolved_checkpoint": resolved_ckpt,
                "pano_patch_dims": list(pano_patch_dims),
                "sat_patch_dims": list(sat_patch_dims),
                "num_queries": dataset.num_queries,
                "num_gallery": dataset.num_gallery,
                "git_sha": _git_sha(),
            },
            f,
            indent=2,
        )
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
