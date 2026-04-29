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

from experimental.overhead_matching.swag.data import satellite_embedding_database as sed
from experimental.overhead_matching.swag.data import vigor_dataset as vd
from experimental.overhead_matching.swag.data.cvgtext_dataset import CVGTextDataset
from experimental.overhead_matching.swag.evaluation import retrieval_metrics
from experimental.overhead_matching.swag.scripts.export_similarity_matrix import (
    load_models_from_training_output,
)


def _check_self_retrieval(embeddings: torch.Tensor, label: str, tol: float = 1e-5) -> None:
    """Cosine-similarity self-retrieval sanity.

    The invariant we care about is indexing-alignment: embedding[i] must be the
    representation of the i-th metadata row. The strict "argmax == i" check is
    too strong whenever two rows have identical images (duplicate tiles in the
    dataset produce identical embeddings and therefore tied cosine similarities).
    We instead assert `sim[i, i] >= sim[i].max() - tol` — i.e. self-similarity
    is tied for max — which is exactly the alignment invariant.
    """
    emb = embeddings.squeeze()
    assert emb.ndim == 2, f"Expected 2D embedding tensor for {label}, got shape {tuple(emb.shape)}"
    n = emb.shape[0]
    sim = emb @ emb.T
    self_sim = sim.diagonal()
    row_max = sim.max(dim=1).values
    gap = row_max - self_sim
    failing = (gap > tol).nonzero(as_tuple=True)[0]
    assert failing.numel() == 0, (
        f"{label}: self-similarity fell below row max on {failing.numel()}/{n} rows. "
        f"First 5 rows: {failing[:5].tolist()} with gaps {gap[failing[:5]].tolist()}"
    )
    top1 = sim.argmax(dim=1)
    tied = (top1 != torch.arange(n, device=top1.device)).nonzero(as_tuple=True)[0]
    if tied.numel() > 0:
        print(
            f"[sanity] {label}: self-similarity == row-max on all {n} rows; "
            f"{tied.numel()} rows have tied top-1 (duplicate entries in the data)"
        )
    else:
        print(f"[sanity] {label}: self-retrieval R@1 = 1.0 across all {n} rows")


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

    # Build the two embedding databases separately so we can run self-retrieval
    # sanity checks on each before computing the cross-modal similarity.
    sat_loader = vd.get_dataloader(
        dataset.get_sat_patch_view(), batch_size=96, num_workers=8
    )
    pano_loader = vd.get_dataloader(
        dataset.get_pano_view(), batch_size=96, num_workers=8
    )
    with torch.no_grad():
        print("building satellite embedding database")
        sat_embeddings = sed.build_satellite_db(sat_model, sat_loader, device=device)
        print("building panorama embedding database")
        pano_embeddings = sed.build_panorama_db(pano_model, pano_loader, device=device)

    _check_self_retrieval(sat_embeddings, "sat")
    _check_self_retrieval(pano_embeddings, "pano")

    similarity = sed.calculate_cos_similarity_against_database(
        pano_embeddings.squeeze(), sat_embeddings.squeeze()
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
