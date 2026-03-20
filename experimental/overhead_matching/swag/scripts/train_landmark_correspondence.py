#!/usr/bin/env python3
"""Training script for landmark correspondence classifier.

Trains a twin-encoder model to predict whether a pano landmark and OSM landmark
refer to the same physical object.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:train_landmark_correspondence -- \
        --config /path/to/config.yaml

Config format (YAML):
    data_dir: /data/overhead_matching/datasets/landmark_correspondence/neg_v3_full
    text_embeddings_path: /data/overhead_matching/datasets/landmark_correspondence/text_embeddings.pkl
    output_dir: /tmp/landmark_correspondence_training
    train_city: Chicago
    val_city: Seattle
    include_difficulties: [positive, easy]
    batch_size: 512
    num_epochs: 20
    lr: 0.001
    weight_decay: 0.01
    warmup_fraction: 0.05
    gradient_clip_norm: 1.0
    seed: 42
    encoder:
        text_input_dim: 768
        text_proj_dim: 128
    classifier:
        mlp_hidden_dim: 128
        dropout: 0.1
"""

import argparse
import random
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401 - Must import before torch

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from experimental.overhead_matching.swag.data.landmark_correspondence_dataset import (
    NUM_CROSS_FEATURES_WITH_CATEGORY,
    NUM_CROSS_FEATURES_WITHOUT_CATEGORY,
    PARSE_REPORT_PATH,
    CorrespondenceBatch,
    LandmarkCorrespondenceDataset,
    collate_correspondence,
    load_pairs_from_directory,
    load_text_embeddings,
    scan_parse_failures,
    write_parse_report,
)
from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    CorrespondenceClassifier,
    CorrespondenceClassifierConfig,
    TagBundleEncoderConfig,
)
from experimental.overhead_matching.swag.scripts.correspondence_configs import (
    CorrespondenceTrainConfig,
    load_config,
    save_config,
)


def setup_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_lr_scheduler(
    optimizer: AdamW, warmup_steps: int, total_steps: int,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        remaining = total_steps - step
        total_decay = total_steps - warmup_steps
        return max(0.0, remaining / total_decay)
    return LambdaLR(optimizer, lr_lambda)


def compute_metrics(
    all_labels: list[float],
    all_probs: list[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics."""
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    preds = (probs >= threshold).astype(float)

    metrics = {}
    try:
        metrics["auc_roc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auc_roc"] = 0.0

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)
    metrics["accuracy"] = float((preds == labels).mean())

    return metrics


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    scaler: GradScaler | None,
    gradient_clip_norm: float,
) -> tuple[float, list[float], list[float]]:
    """Train for one epoch. Returns (avg_loss, all_labels, all_probs)."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    all_labels = []
    all_probs = []

    use_amp = scaler is not None
    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        batch: CorrespondenceBatch = batch.to(device)
        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(
                pano_key_indices=batch.pano_key_indices,
                pano_value_type=batch.pano_value_type,
                pano_boolean_values=batch.pano_boolean_values,
                pano_numeric_values=batch.pano_numeric_values,
                pano_numeric_nan_mask=batch.pano_numeric_nan_mask,
                pano_housenumber_values=batch.pano_housenumber_values,
                pano_housenumber_nan_mask=batch.pano_housenumber_nan_mask,
                pano_text_embeddings=batch.pano_text_embeddings,
                pano_tag_mask=batch.pano_tag_mask,
                osm_key_indices=batch.osm_key_indices,
                osm_value_type=batch.osm_value_type,
                osm_boolean_values=batch.osm_boolean_values,
                osm_numeric_values=batch.osm_numeric_values,
                osm_numeric_nan_mask=batch.osm_numeric_nan_mask,
                osm_housenumber_values=batch.osm_housenumber_values,
                osm_housenumber_nan_mask=batch.osm_housenumber_nan_mask,
                osm_text_embeddings=batch.osm_text_embeddings,
                osm_tag_mask=batch.osm_tag_mask,
                cross_features=batch.cross_features,
            ).squeeze(-1)  # (B,)
            loss = F.binary_cross_entropy_with_logits(logits, batch.labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().tolist()
            labels = batch.labels.cpu().tolist()
            all_labels.extend(labels)
            all_probs.extend(probs)

        pbar.set_postfix(loss=loss.item())

    return total_loss / max(num_batches, 1), all_labels, all_probs


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Evaluate model. Returns (avg_loss, metrics_dict)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_labels = []
    all_probs = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = batch.to(device)
        logits = model(
            pano_key_indices=batch.pano_key_indices,
            pano_value_type=batch.pano_value_type,
            pano_boolean_values=batch.pano_boolean_values,
            pano_numeric_values=batch.pano_numeric_values,
            pano_numeric_nan_mask=batch.pano_numeric_nan_mask,
            pano_housenumber_values=batch.pano_housenumber_values,
            pano_housenumber_nan_mask=batch.pano_housenumber_nan_mask,
            pano_text_embeddings=batch.pano_text_embeddings,
            pano_tag_mask=batch.pano_tag_mask,
            osm_key_indices=batch.osm_key_indices,
            osm_value_type=batch.osm_value_type,
            osm_boolean_values=batch.osm_boolean_values,
            osm_numeric_values=batch.osm_numeric_values,
            osm_numeric_nan_mask=batch.osm_numeric_nan_mask,
            osm_housenumber_values=batch.osm_housenumber_values,
            osm_housenumber_nan_mask=batch.osm_housenumber_nan_mask,
            osm_text_embeddings=batch.osm_text_embeddings,
            osm_tag_mask=batch.osm_tag_mask,
            cross_features=batch.cross_features,
        ).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, batch.labels)

        total_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(logits).cpu().tolist()
        labels = batch.labels.cpu().tolist()
        all_labels.extend(labels)
        all_probs.extend(probs)

    avg_loss = total_loss / max(num_batches, 1)
    metrics = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics


def train(config: CorrespondenceTrainConfig) -> None:
    """Main training function."""
    setup_reproducibility(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    save_config(config, output_dir / "config.yaml")

    # Load text embeddings
    text_embeddings = None
    text_input_dim = config.encoder.text_input_dim
    emb_path = config.text_embeddings_path
    print(f"Loading text embeddings from {emb_path}...")
    text_embeddings = load_text_embeddings(emb_path)
    print(f"Loaded {len(text_embeddings):,} text embeddings")
    # Infer dim from first embedding
    first_emb = next(iter(text_embeddings.values()))
    text_input_dim = first_emb.shape[0]
    print(f"Text embedding dim: {text_input_dim}")

    # Load data
    data_dir = config.data_dir
    train_city = config.train_city
    val_city = config.val_city
    include_difficulties = tuple(config.include_difficulties)

    print(f"\nLoading training data ({train_city})...")
    train_pairs = load_pairs_from_directory(data_dir / train_city)
    print(f"Loaded {len(train_pairs):,} total pairs")

    print(f"Loading validation data ({val_city})...")
    val_pairs = load_pairs_from_directory(data_dir / val_city)
    print(f"Loaded {len(val_pairs):,} total pairs")

    # Print pair statistics
    for name, pairs in [("Train", train_pairs), ("Val", val_pairs)]:
        pos = sum(1 for p in pairs if p.difficulty == "positive")
        hard = sum(1 for p in pairs if p.difficulty == "hard")
        easy = sum(1 for p in pairs if p.difficulty == "easy")
        print(f"  {name}: {pos} positive, {hard} hard neg, {easy} easy neg")

    # Scan for parse failures and write report (cleared each run)
    all_pairs = train_pairs + val_pairs
    print("\nScanning for parse failures...")
    failures = scan_parse_failures(all_pairs, text_embeddings)
    if failures:
        write_parse_report(failures, all_pairs, PARSE_REPORT_PATH)
        print(f"  {len(failures)} parse failures found (see {PARSE_REPORT_PATH})")
    else:
        # Clear any old report
        if PARSE_REPORT_PATH.exists():
            PARSE_REPORT_PATH.unlink()
        print("  No parse failures found")

    include_category = config.include_category_features
    train_dataset = LandmarkCorrespondenceDataset(
        train_pairs, text_embeddings, text_input_dim, include_difficulties,
        include_category_features=include_category,
    )
    val_dataset = LandmarkCorrespondenceDataset(
        val_pairs, text_embeddings, text_input_dim, include_difficulties,
        include_category_features=include_category,
    )
    print(f"After filtering: train={len(train_dataset):,}, val={len(val_dataset):,}")

    batch_size = config.batch_size
    num_workers = config.num_workers
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_correspondence, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_correspondence, num_workers=num_workers,
        pin_memory=True,
    )

    # Create model
    encoder_config = TagBundleEncoderConfig(
        text_input_dim=text_input_dim,
        text_proj_dim=config.encoder.text_proj_dim,
    )
    num_cross = NUM_CROSS_FEATURES_WITH_CATEGORY if include_category else NUM_CROSS_FEATURES_WITHOUT_CATEGORY
    classifier_config = CorrespondenceClassifierConfig(
        encoder=encoder_config,
        mlp_hidden_dim=config.classifier.mlp_hidden_dim,
        dropout=config.classifier.dropout,
        num_cross_features=num_cross,
    )
    model = CorrespondenceClassifier(classifier_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {num_params:,} parameters")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    num_epochs = config.num_epochs
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * config.warmup_fraction)
    scheduler = create_lr_scheduler(optimizer, warmup_steps, total_steps)

    gradient_clip_norm = config.gradient_clip_norm

    # Mixed precision
    scaler = None
    use_amp = config.use_amp and device.type == "cuda"
    if use_amp:
        scaler = GradScaler()
        print("Mixed precision training enabled (float16)")

    # TensorBoard
    tb_dir = output_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    # Training loop
    best_auc = 0.0
    print(f"\nTraining for {num_epochs} epochs ({total_steps} steps, {warmup_steps} warmup)")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_labels, train_probs = train_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, gradient_clip_norm,
        )
        train_metrics = compute_metrics(train_labels, train_probs)

        val_loss, val_metrics = evaluate(model, val_loader, device)

        # Log
        print(f"  Train loss: {train_loss:.4f}, AUC: {train_metrics['auc_roc']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"  Val   loss: {val_loss:.4f}, AUC: {val_metrics['auc_roc']:.4f}, "
              f"P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        for name, val in train_metrics.items():
            writer.add_scalar(f"train/{name}", val, epoch)
        for name, val in val_metrics.items():
            writer.add_scalar(f"val/{name}", val, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        # Save best model
        if val_metrics["auc_roc"] > best_auc:
            best_auc = val_metrics["auc_roc"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  New best AUC: {best_auc:.4f}")

        # Save checkpoint
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_metrics": val_metrics,
        }, output_dir / f"checkpoint_epoch_{epoch + 1}.pt")

    writer.close()
    print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train landmark correspondence classifier")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
