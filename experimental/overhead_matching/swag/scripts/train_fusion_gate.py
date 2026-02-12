"""Train a learned gating MLP for fusing image and landmark similarity matrices.

The gate MLP takes 8 scalar features per panorama and predicts a mixing weight
alpha in [0, 1] such that:
    fused_sim = alpha * img_sim + (1 - alpha) * lm_sim

The loss encourages the fused similarity to place high probability on the
true positive satellite patches:
    loss = -log(softmax(fused_sim / sigma)[true_patch_idx])
"""

import argparse
from pathlib import Path

import common.torch.load_torch_deps
import torch
import torch.nn.functional as F

import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    _build_gate_mlp,
    _load_similarity_matrix,
    extract_gate_features,
)


class FusionGateDataset(torch.utils.data.Dataset):
    """Dataset of (features, img_sim_row, lm_sim_row, true_patch_indices) per panorama."""

    def __init__(self, features, img_sim_rows, lm_sim_rows, true_patch_indices):
        assert features.shape[0] == img_sim_rows.shape[0] == lm_sim_rows.shape[0]
        assert len(true_patch_indices) == features.shape[0]
        self.features = features
        self.img_sim_rows = img_sim_rows
        self.lm_sim_rows = lm_sim_rows
        self.true_patch_indices = true_patch_indices

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.img_sim_rows[idx],
            self.lm_sim_rows[idx],
            self.true_patch_indices[idx],
        )


def collate_fn(batch):
    """Custom collate that handles variable-length true_patch_indices."""
    features = torch.stack([b[0] for b in batch])
    img_sims = torch.stack([b[1] for b in batch])
    lm_sims = torch.stack([b[2] for b in batch])
    true_indices = [b[3] for b in batch]
    return features, img_sims, lm_sims, true_indices


def build_dataset_from_city(img_sim_path, lm_sim_path, dataset_path, landmark_version, sigma):
    """Build a FusionGateDataset for a single city."""
    print(f"Loading similarity matrices for {dataset_path.name}...")
    img_sim_matrix = _load_similarity_matrix(img_sim_path)
    lm_sim_matrix = _load_similarity_matrix(lm_sim_path)

    print(f"Loading VIGOR dataset from {dataset_path}...")
    dataset_config = vd.VigorDatasetConfig(
        panorama_tensor_cache_info=None,
        satellite_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=False,
        landmark_version=landmark_version,
    )
    vigor_dataset = vd.VigorDataset(dataset_path, dataset_config)
    pano_metadata = vigor_dataset._panorama_metadata

    num_panos = len(pano_metadata)
    assert img_sim_matrix.shape[0] == num_panos
    assert lm_sim_matrix.shape[0] == num_panos

    print(f"Extracting features for {num_panos} panoramas...")
    all_features = []
    all_true_indices = []
    for pano_idx in range(num_panos):
        img_sim = img_sim_matrix[pano_idx]
        lm_sim = lm_sim_matrix[pano_idx]
        features = extract_gate_features(img_sim, lm_sim, sigma)
        all_features.append(features)
        row = pano_metadata.iloc[pano_idx]
        true_indices = list(row["positive_satellite_idxs"]) + list(row["semipositive_satellite_idxs"])
        all_true_indices.append(true_indices)

    features_tensor = torch.stack(all_features)
    return FusionGateDataset(
        features=features_tensor,
        img_sim_rows=img_sim_matrix,
        lm_sim_rows=lm_sim_matrix,
        true_patch_indices=all_true_indices,
    )


def compute_loss(gate_mlp, features, img_sims, lm_sims, true_indices, sigma):
    """Compute the fusion gate loss for a batch."""
    alpha = torch.sigmoid(gate_mlp(features))  # (B, 1)
    fused_sim = alpha * img_sims + (1 - alpha) * lm_sims  # (B, num_patches)
    log_probs = F.log_softmax(fused_sim / sigma, dim=1)  # (B, num_patches)

    batch_losses = []
    for i, indices in enumerate(true_indices):
        if len(indices) == 0:
            continue
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=log_probs.device)
        sample_loss = -log_probs[i, idx_tensor].mean()
        batch_losses.append(sample_loss)

    if len(batch_losses) == 0:
        return torch.tensor(0.0, device=features.device, requires_grad=True)
    return torch.stack(batch_losses).mean()


def main():
    parser = argparse.ArgumentParser(
        description="Train a learned gating MLP for image-landmark fusion"
    )
    parser.add_argument("--img-sim-paths", type=str, required=True,
                        help="Comma-separated paths to image similarity matrices")
    parser.add_argument("--lm-sim-paths", type=str, required=True,
                        help="Comma-separated paths to landmark similarity matrices")
    parser.add_argument("--dataset-paths", type=str, required=True,
                        help="Comma-separated paths to VIGOR datasets")
    parser.add_argument("--landmark-version", type=str, required=True,
                        help="Landmark version string")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Where to save trained gate model weights")
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    img_sim_paths = [Path(p.strip()) for p in args.img_sim_paths.split(",")]
    lm_sim_paths = [Path(p.strip()) for p in args.lm_sim_paths.split(",")]
    dataset_paths = [Path(p.strip()) for p in args.dataset_paths.split(",")]
    assert len(img_sim_paths) == len(lm_sim_paths) == len(dataset_paths)

    # Build datasets for all cities
    city_datasets = []
    for img_path, lm_path, ds_path in zip(img_sim_paths, lm_sim_paths, dataset_paths):
        city_ds = build_dataset_from_city(
            img_path, lm_path, ds_path, args.landmark_version, args.sigma
        )
        city_datasets.append(city_ds)

    # Concatenate all city datasets
    all_features = torch.cat([ds.features for ds in city_datasets], dim=0)
    all_img_sims = torch.cat([ds.img_sim_rows for ds in city_datasets], dim=0)
    all_lm_sims = torch.cat([ds.lm_sim_rows for ds in city_datasets], dim=0)
    all_true_indices = []
    for ds in city_datasets:
        for indices in ds.true_patch_indices:
            all_true_indices.append(indices)

    # Handle different number of patches per city by padding
    num_patches_per_city = [ds.img_sim_rows.shape[1] for ds in city_datasets]
    if len(set(num_patches_per_city)) > 1:
        max_patches = max(num_patches_per_city)
        padded_img_sims = []
        padded_lm_sims = []
        for ds in city_datasets:
            n_panos, n_patches = ds.img_sim_rows.shape
            if n_patches < max_patches:
                pad_size = max_patches - n_patches
                padded_img_sims.append(torch.cat([ds.img_sim_rows, torch.zeros(n_panos, pad_size)], dim=1))
                padded_lm_sims.append(torch.cat([ds.lm_sim_rows, torch.zeros(n_panos, pad_size)], dim=1))
            else:
                padded_img_sims.append(ds.img_sim_rows)
                padded_lm_sims.append(ds.lm_sim_rows)
        all_img_sims = torch.cat(padded_img_sims, dim=0)
        all_lm_sims = torch.cat(padded_lm_sims, dim=0)

    combined_dataset = FusionGateDataset(
        features=all_features,
        img_sim_rows=all_img_sims,
        lm_sim_rows=all_lm_sims,
        true_patch_indices=all_true_indices,
    )

    # Split into train/val (90/10)
    num_total = len(combined_dataset)
    num_val = max(1, int(0.1 * num_total))
    num_train = num_total - num_val
    generator = torch.Generator().manual_seed(42)
    train_split, val_split = torch.utils.data.random_split(
        range(num_total), [num_train, num_val], generator=generator
    )
    train_dataset = torch.utils.data.Subset(combined_dataset, train_split.indices)
    val_dataset = torch.utils.data.Subset(combined_dataset, val_split.indices)
    print(f"Total panoramas: {num_total}, Train: {num_train}, Val: {num_val}")

    # Compute normalization statistics on training data only
    train_features = all_features[train_split.indices]
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0)
    print(f"Feature mean: {feature_mean}")
    print(f"Feature std:  {feature_std}")

    # Normalize all features in-place (using train stats)
    combined_dataset.features = (combined_dataset.features - feature_mean) / (feature_std + 1e-8)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Build model and optimizer
    gate_mlp = _build_gate_mlp()
    optimizer = torch.optim.Adam(gate_mlp.parameters(), lr=args.lr)
    best_val_loss = float("inf")
    best_state_dict = None

    print(f"\nStarting training for {args.epochs} epochs...")
    header = f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Best':>4}"
    print(header)
    print("-" * len(header))

    for epoch in range(args.epochs):
        gate_mlp.train()
        train_loss_sum = 0.0
        train_count = 0
        for features, img_sims, lm_sims, true_indices in train_loader:
            optimizer.zero_grad()
            loss = compute_loss(gate_mlp, features, img_sims, lm_sims, true_indices, args.sigma)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * features.shape[0]
            train_count += features.shape[0]
        train_loss = train_loss_sum / max(train_count, 1)

        gate_mlp.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for features, img_sims, lm_sims, true_indices in val_loader:
                loss = compute_loss(gate_mlp, features, img_sims, lm_sims, true_indices, args.sigma)
                val_loss_sum += loss.item() * features.shape[0]
                val_count += features.shape[0]
        val_loss = val_loss_sum / max(val_count, 1)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_state_dict = {k: v.clone() for k, v in gate_mlp.state_dict().items()}
        marker = " *" if is_best else ""
        print(f"{epoch + 1:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} |{marker}")

    # Save best model
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "gate_weights": best_state_dict,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }
    torch.save(checkpoint, output_path)
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
