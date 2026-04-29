"""Extract DINO CLS tokens from pinhole images and save to .npz cache.

For each panorama directory, loads 4 yaw images (000, 090, 180, 270) at
full resolution (1024x1024), normalizes with ImageNet stats, extracts the
CLS token from a DINOv2 or DINOv3 model, and concatenates to a single
feature vector per panorama.

Supports both DINOv2 (torch.hub, no auth) and DINOv3 (local checkpoint).

For multi-machine parallelism, use --shard_idx and --num_shards to split
the work, then --merge to combine the outputs:

    # Machine 0:
    ... :extract_dino_cls_tokens -- --pinhole_dir ... --output /tmp/shard_0.npz --shard_idx 0 --num_shards 2
    # Machine 1:
    ... :extract_dino_cls_tokens -- --pinhole_dir ... --output /tmp/shard_1.npz --shard_idx 1 --num_shards 2
    # Merge:
    ... :extract_dino_cls_tokens -- --merge /tmp/shard_0.npz /tmp/shard_1.npz --output /tmp/dino_cls_tokens.npz
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import common.torch.load_torch_deps  # noqa: F401 — must precede torch import
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
YAW_SUFFIXES = ["yaw_000.jpg", "yaw_090.jpg", "yaw_180.jpg", "yaw_270.jpg"]


class PinholeImageDataset(Dataset):
    """Dataset that yields individual yaw images from panorama directories."""

    def __init__(self, pano_dirs: list[Path], transform: T.Compose):
        self._pano_dirs = pano_dirs
        self._transform = transform

    def __len__(self):
        return len(self._pano_dirs) * 4

    def __getitem__(self, idx):
        pano_idx = idx // 4
        yaw_idx = idx % 4
        path = self._pano_dirs[pano_idx] / YAW_SUFFIXES[yaw_idx]
        img = Image.open(path).convert("RGB")
        return self._transform(img)


def merge_shards(shard_paths: list[Path], output_path: Path):
    """Merge multiple .npz shard files into a single output."""
    all_ids = []
    all_features = []
    for p in shard_paths:
        data = np.load(p, allow_pickle=True)
        all_ids.append(data["pano_ids"])
        all_features.append(data["features"])
    pano_ids = np.concatenate(all_ids)
    features = np.concatenate(all_features, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, pano_ids=pano_ids, features=features.astype(np.float32))
    print(f"Merged {len(shard_paths)} shards -> {output_path}")
    print(f"  {len(pano_ids)} panoramas, {features.shape[1]}-dim features")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pinhole_dir",
        type=Path,
        default=None,
        help="Directory containing panorama folders with pinhole images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npz file for cached features",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for DINO inference",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers for image loading",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dinov3_vitb16",
        help="DINO model name (e.g. dinov3_vitb16, dinov2_vitb14)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to local checkpoint (required for DINOv3 if not in torch hub cache)",
    )
    parser.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="Shard index for multi-machine parallelism (0-indexed)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards for multi-machine parallelism",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        nargs="+",
        default=None,
        help="Merge multiple .npz shard files instead of extracting",
    )
    args = parser.parse_args()

    # Merge mode
    if args.merge:
        merge_shards(args.merge, args.output)
        return

    if args.pinhole_dir is None:
        parser.error("--pinhole_dir is required when not using --merge")

    pinhole_dir = args.pinhole_dir
    if not pinhole_dir.is_dir():
        raise FileNotFoundError(f"Pinhole directory not found: {pinhole_dir}")

    # Discover panorama folders — each folder name is "{pano_id},{lat},{lon},"
    pano_dirs = sorted(
        d for d in pinhole_dir.iterdir()
        if d.is_dir() and all((d / suffix).exists() for suffix in YAW_SUFFIXES)
    )
    print(f"Found {len(pano_dirs)} panorama folders with all 4 yaw images")

    # Apply sharding
    if args.num_shards > 1:
        pano_dirs = pano_dirs[args.shard_idx::args.num_shards]
        print(f"Shard {args.shard_idx}/{args.num_shards}: processing {len(pano_dirs)} panoramas")

    if not pano_dirs:
        print("No valid panorama folders found. Exiting.")
        return

    # Extract pano IDs from folder names
    pano_ids = [d.name.split(",")[0] for d in pano_dirs]

    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device}...")
    model_family = args.model.split("_")[0]  # "dinov2" or "dinov3"
    repo = f"facebookresearch/{model_family}"
    if args.weights:
        model = torch.hub.load(repo, args.model, source="local", weights=str(args.weights))
    else:
        model = torch.hub.load(repo, args.model)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    emb_dim = model.num_features
    print(f"Model embedding dim: {emb_dim}, total feature dim: {emb_dim * 4}")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    dataset = PinholeImageDataset(pano_dirs, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Extract CLS tokens
    all_cls = []
    n_images = len(dataset)
    print(f"Extracting CLS tokens from {n_images} images "
          f"(batch_size={args.batch_size}, workers={args.num_workers})...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing", total=len(loader)):
            cls_tokens = model(batch.to(device, non_blocking=True))
            all_cls.append(cls_tokens.cpu())

    all_cls = torch.cat(all_cls, dim=0).numpy()  # (N*4, emb_dim)

    # Reshape and concatenate: 4 CLS tokens per panorama -> (N, emb_dim*4)
    n_panos = len(pano_dirs)
    features = all_cls.reshape(n_panos, 4, emb_dim).reshape(n_panos, emb_dim * 4)
    print(f"Feature matrix shape: {features.shape}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        pano_ids=np.array(pano_ids, dtype=object),
        features=features.astype(np.float32),
    )
    print(f"Saved to {args.output}")
    print(f"  {n_panos} panoramas, {features.shape[1]}-dim features")


if __name__ == "__main__":
    main()
