"""Run CrossText2Loc (yejy53/CVG-Text, ICCV 2025) inference on a CVG-Text split.

The paper's published numbers use a 100-nearest-neighbor prior that restricts
the retrieval candidate set per query. This script intentionally bypasses that
filter: we compute cosine similarity across the *full* per-city gallery so the
numbers are directly comparable to image-side retrieval baselines that don't
assume a location prior.

Supports cross-city evaluation: `--train_city X --test_city Y` loads the
checkpoint trained on X and evaluates against Y's queries + gallery.

Outputs under `--output_base/crosstext2loc_train-<X>_test-<Y>_<kind>/`:
- `similarity.pt` — (num_queries, num_gallery) float tensor
- `metrics.json` — recall@{1,5,10}, MRR
- `config.json` — run metadata
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch
from tqdm import tqdm

from experimental.overhead_matching.swag.data.cvgtext_dataset import CVGTextDataset
from experimental.overhead_matching.swag.evaluation import retrieval_metrics
from experimental.overhead_matching.swag.model import crosstext2loc_encoder as enc


def _find_checkpoint(model_dir: Path, train_city: str, kind: str) -> Path:
    pattern = str(model_dir / f"long_model_{train_city}-mixed_1e-05_128_{kind}_epoch*_*.pth")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint matched {pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous checkpoints for {train_city}/{kind}: {matches}")
    return Path(matches[0])


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
    parser.add_argument("--train_city", required=True, choices=("Brisbane", "NewYork", "Tokyo"))
    parser.add_argument("--test_city", required=True, choices=("Brisbane", "NewYork", "Tokyo"))
    parser.add_argument("--gallery_kind", required=True, choices=("sat", "osm"))
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--output_base",
        type=Path,
        default=Path("/data/overhead_matching/evaluation/results/cvgtext"),
    )
    parser.add_argument("--text_batch_size", type=int, default=64)
    parser.add_argument("--image_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    # Map our CLI naming (sat/osm) to the on-disk folder naming (satellite/OSM).
    gallery_dir_kind = "satellite" if args.gallery_kind == "sat" else "OSM"

    dataset = CVGTextDataset(
        root=args.dataset_root,
        city=args.test_city,
        split=args.split,
        gallery_kind=gallery_dir_kind,
    )

    print(
        f"CVG-Text: train={args.train_city}, test={args.test_city}, gallery={args.gallery_kind}"
        f"  |  {dataset.num_queries} queries, {dataset.num_gallery} gallery entries"
    )

    checkpoint_dir = args.dataset_root / "models"
    checkpoint_path = _find_checkpoint(checkpoint_dir, args.train_city, args.gallery_kind)
    print(f"Loading CrossText2Loc checkpoint: {checkpoint_path.name}")
    model, preprocessor = enc.build_model(checkpoint_path, device)

    text_feats = enc.encode_texts(model, preprocessor, dataset.texts, device, args.text_batch_size, desc="text")
    image_feats = enc.encode_images(model, preprocessor, dataset.gallery_image_paths, device,
                                    args.image_batch_size, args.num_workers, desc=f"{args.gallery_kind} gallery")

    similarity = text_feats @ image_feats.T  # (num_queries, num_gallery)
    print(f"similarity shape: {tuple(similarity.shape)}")

    metrics = retrieval_metrics.compute_top_k_metrics(similarity, dataset, ks=[1, 5, 10])
    print("Metrics:", metrics)

    out_dir = (
        args.output_base
        / f"crosstext2loc_train-{args.train_city}_test-{args.test_city}_{args.gallery_kind}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(similarity, out_dir / "similarity.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump({
            "train_city": args.train_city,
            "test_city": args.test_city,
            "gallery_kind": args.gallery_kind,
            "split": args.split,
            "checkpoint": checkpoint_path.name,
            "num_queries": dataset.num_queries,
            "num_gallery": dataset.num_gallery,
            "git_sha": _git_sha(),
        }, f, indent=2)
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
