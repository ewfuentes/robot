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
import torch.nn.functional as F
from crosstext2loc import load_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from experimental.overhead_matching.swag.data.cvgtext_dataset import CVGTextDataset
from experimental.overhead_matching.swag.evaluation import retrieval_metrics


def _find_checkpoint(model_dir: Path, train_city: str, kind: str) -> Path:
    pattern = str(model_dir / f"long_model_{train_city}-mixed_1e-05_128_{kind}_epoch*_*.pth")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint matched {pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous checkpoints for {train_city}/{kind}: {matches}")
    return Path(matches[0])


def _build_model(checkpoint_path: Path, device: torch.device):
    """Load CrossText2Loc's CLIP-L/14@336 via the upstream `load_model`.

    Upstream hard-wires `expand_text=True` (77→300 text pos-embed interp) and
    `is_stv=False` (square 336×336 imagery) for the `sat` and `osm` retrieval
    checkpoints, so that's what we pass here. The returned `preprocessor` is
    the upstream callable that takes `(image, text)` and returns a tuple of
    tensors — we call it with a dummy text for image batches and a dummy
    image for text batches so both paths go through their canonical code.
    """
    model, preprocessor, _evaluator, _forward = load_model(
        "CLIP-L/14@336",
        expand_text=True,
        checkpoint_path=str(checkpoint_path),
        is_stv=False,
    )
    model = model.to(device).eval()
    return model, preprocessor


class _ImagePathDataset(Dataset):
    """Loads PNG paths and runs them through the upstream preprocessor.

    Upstream's `preprocessor` is a `(image, text) -> (image_tensor, text_id)`
    bundled callable; we pass an empty text so the tokenizer side is a no-op
    and we only keep the image tensor. Going through upstream for image
    preprocessing (rather than re-invoking `clip.load` ourselves) keeps the
    transform pipeline identical to what CrossText2Loc's `zeroshot.py` would
    have applied.
    """

    def __init__(self, paths: list[str], preprocessor):
        self.paths = paths
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        image_tensor, _text_id = self.preprocessor(img, "")
        return image_tensor


@torch.no_grad()
def _encode_images(
    model: torch.nn.Module,
    preprocessor,
    paths: list[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    desc: str,
) -> torch.Tensor:
    loader = DataLoader(
        _ImagePathDataset(paths, preprocessor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    feats = []
    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device, non_blocking=True)
        f = model.encode_image(batch)
        f = F.normalize(f, dim=-1)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def _encode_texts(
    model: torch.nn.Module,
    preprocessor,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    desc: str,
) -> torch.Tensor:
    """Tokenize via upstream's preprocessor so the tokenization (incl. the
    `context_length=MAX_TEXT_LEN, truncate=True` args) matches `zeroshot.py`.

    The upstream preprocessor needs an image too; we pass a tiny dummy and
    discard the image tensor.
    """
    dummy_image = Image.new("RGB", (336, 336))
    feats = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        chunk = texts[start : start + batch_size]
        _image, tokens = preprocessor(dummy_image, chunk)
        tokens = tokens.to(device)
        f = model.encode_text(tokens)
        f = F.normalize(f, dim=-1)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return "unknown"


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
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Optional cap on #queries for smoke tests.",
    )
    parser.add_argument(
        "--max_gallery",
        type=int,
        default=None,
        help="Optional cap on gallery size for smoke tests.",
    )
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
    if args.max_queries is not None:
        dataset._panorama_metadata = dataset._panorama_metadata.iloc[: args.max_queries].reset_index(drop=True)
    if args.max_gallery is not None:
        kept = dataset._satellite_metadata.iloc[: args.max_gallery].reset_index(drop=True)
        dataset._satellite_metadata = kept
        # Remap positive indices to the truncated gallery; drop queries whose GT was cut.
        kept_filenames = set(kept["filename"])
        filename_to_new_idx = {fn: i for i, fn in enumerate(kept["filename"])}
        keep_mask = []
        new_positive_idxs = []
        for _, row in dataset._panorama_metadata.iterrows():
            fn = row["filename"]
            if fn in kept_filenames:
                keep_mask.append(True)
                new_positive_idxs.append([filename_to_new_idx[fn]])
            else:
                keep_mask.append(False)
                new_positive_idxs.append([])
        dataset._panorama_metadata = dataset._panorama_metadata[keep_mask].reset_index(drop=True)
        dataset._panorama_metadata["positive_satellite_idxs"] = [
            p for p, k in zip(new_positive_idxs, keep_mask) if k
        ]

    print(
        f"CVG-Text: train={args.train_city}, test={args.test_city}, gallery={args.gallery_kind}"
        f"  |  {dataset.num_queries} queries, {dataset.num_gallery} gallery entries"
    )

    checkpoint_dir = args.dataset_root / "models"
    checkpoint_path = _find_checkpoint(checkpoint_dir, args.train_city, args.gallery_kind)
    print(f"Loading CrossText2Loc checkpoint: {checkpoint_path.name}")
    model, preprocessor = _build_model(checkpoint_path, device)

    text_feats = _encode_texts(
        model, preprocessor, dataset.texts, device, args.text_batch_size, desc="text"
    )
    image_feats = _encode_images(
        model,
        preprocessor,
        dataset.gallery_image_paths,
        device,
        args.image_batch_size,
        args.num_workers,
        desc=f"{args.gallery_kind} gallery",
    )

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
            "max_queries": args.max_queries,
            "max_gallery": args.max_gallery,
            "num_queries": dataset.num_queries,
            "num_gallery": dataset.num_gallery,
            "git_sha": _git_sha(),
        }, f, indent=2)
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
