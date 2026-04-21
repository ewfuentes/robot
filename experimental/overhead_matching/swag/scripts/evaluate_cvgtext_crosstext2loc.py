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
import clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from experimental.overhead_matching.swag.data.cvgtext_dataset import CVGTextDataset
from experimental.overhead_matching.swag.evaluation import retrieval_metrics


MAX_TEXT_LEN = 300


def _interpolate_text_positional_embedding(pos_embedding: torch.Tensor, new_len: int) -> torch.Tensor:
    """Linear-interpolate text positional embedding from its trained length to `new_len`.

    Standard ViT positional-embedding resize (Dosovitskiy et al.). CrossText2Loc
    fine-tuned ViT-L/14@336 with text pos embedding pre-interpolated from 77 → 300
    so longer captions fit; we reproduce that init here before loading the state dict.
    """
    # pos_embedding: (seq_len, dim) -> (1, dim, seq_len) for F.interpolate
    x = pos_embedding.unsqueeze(0).permute(0, 2, 1)
    x = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
    return x.permute(0, 2, 1).squeeze(0)


def _expand_clip_text_context(model: torch.nn.Module, new_len: int) -> None:
    """Mutate a `clip` model in place to accept up to `new_len` text tokens."""
    model.positional_embedding = torch.nn.Parameter(
        _interpolate_text_positional_embedding(model.positional_embedding.data, new_len)
    )
    mask = torch.empty(new_len, new_len, dtype=model.positional_embedding.dtype)
    mask.fill_(float("-inf"))
    mask.triu_(1)
    for block in model.transformer.resblocks:
        block.attn_mask = mask


def _find_checkpoint(model_dir: Path, train_city: str, kind: str) -> Path:
    pattern = str(model_dir / f"long_model_{train_city}-mixed_1e-05_128_{kind}_epoch*_*.pth")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint matched {pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous checkpoints for {train_city}/{kind}: {matches}")
    return Path(matches[0])


def _build_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, object]:
    model, preprocess = clip.load("ViT-L/14@336px", device="cpu", jit=False)
    model = model.float()
    _expand_clip_text_context(model, MAX_TEXT_LEN)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and not any(
        k.startswith("visual.") for k in state_dict
    ):
        state_dict = state_dict["state_dict"]
    msg = model.load_state_dict(state_dict, strict=False)
    if msg.missing_keys:
        raise RuntimeError(
            f"Missing keys when loading {checkpoint_path}: {msg.missing_keys[:5]}..."
        )
    if msg.unexpected_keys:
        print(f"[load_state_dict] {len(msg.unexpected_keys)} unexpected keys (ignored)")

    model = model.to(device).eval()
    return model, preprocess


class _ImagePathDataset(Dataset):
    def __init__(self, paths: list[str], preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.preprocess(img)


@torch.no_grad()
def _encode_images(
    model: torch.nn.Module,
    preprocess,
    paths: list[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    desc: str,
) -> torch.Tensor:
    loader = DataLoader(
        _ImagePathDataset(paths, preprocess),
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
    texts: list[str],
    device: torch.device,
    batch_size: int,
    desc: str,
) -> torch.Tensor:
    feats = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        chunk = texts[start : start + batch_size]
        tokens = clip.tokenize(chunk, context_length=MAX_TEXT_LEN, truncate=True).to(device)
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
    model, preprocess = _build_model(checkpoint_path, device)

    text_feats = _encode_texts(
        model, dataset.texts, device, args.text_batch_size, desc="text"
    )
    image_feats = _encode_images(
        model,
        preprocess,
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
