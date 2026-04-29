#!/usr/bin/env python3
"""Training script for the landmark correspondence classifier.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:train_landmark_correspondence -- \\
        --config /path/to/config.yaml
"""

import argparse
import functools
import math
import random
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401

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
    LandmarkCorrespondenceDataset,
    collate_correspondence,
    load_pairs_from_directory,
    load_text_embeddings,
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
    cosine: bool = False,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        if cosine:
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(0.0, 1.0 - progress)
    return LambdaLR(optimizer, lr_lambda)


def compute_metrics(
    all_labels: list[float], all_probs: list[float], threshold: float = 0.5,
) -> dict[str, float]:
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    preds = (probs >= threshold).astype(float)

    # Only swallow the specific "one class present" case; let other failures
    # (NaN probs from divergent training, etc.) propagate loudly.
    if len(np.unique(labels)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(labels, probs)

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {
        "auc_roc": float(auc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "accuracy": float((preds == labels).mean()),
    }


def _forward_batch(model, batch):
    return model(
        pano_key_indices=batch.pano_key_indices,
        pano_text_embeddings=batch.pano_text_embeddings,
        pano_tag_mask=batch.pano_tag_mask,
        osm_key_indices=batch.osm_key_indices,
        osm_text_embeddings=batch.osm_text_embeddings,
        osm_tag_mask=batch.osm_tag_mask,
        cross_features=batch.cross_features,
    ).squeeze(-1)


def train_epoch(model, dataloader, optimizer, scheduler, device, scaler, gradient_clip_norm):
    model.train()
    total_loss, num_batches = 0.0, 0
    all_labels, all_probs = [], []
    use_amp = scaler is not None
    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=use_amp):
            logits = _forward_batch(model, batch)
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
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(batch.labels.cpu().tolist())
        pbar.set_postfix(loss=loss.item())

    return total_loss / max(num_batches, 1), all_labels, all_probs


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss, num_batches = 0.0, 0
    all_labels, all_probs = [], []
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = batch.to(device)
        logits = _forward_batch(model, batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch.labels)
        total_loss += loss.item()
        num_batches += 1
        all_probs.extend(torch.sigmoid(logits).cpu().tolist())
        all_labels.extend(batch.labels.cpu().tolist())
    return total_loss / max(num_batches, 1), compute_metrics(all_labels, all_probs)


def train(config: CorrespondenceTrainConfig) -> None:
    setup_reproducibility(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    print(f"Loading text embeddings from {config.text_embeddings_path}...")
    text_embeddings = load_text_embeddings(config.text_embeddings_path)
    print(f"Loaded {len(text_embeddings):,} text embeddings")
    text_input_dim = next(iter(text_embeddings.values())).shape[0]

    include_difficulties = tuple(config.include_difficulties)
    print(f"\nLoading training data ({config.train_city})...")
    train_pairs = load_pairs_from_directory(config.data_dir / config.train_city)
    print(f"Loaded {len(train_pairs):,} total pairs")
    print(f"Loading validation data ({config.val_city})...")
    val_pairs = load_pairs_from_directory(config.data_dir / config.val_city)
    print(f"Loaded {len(val_pairs):,} total pairs")

    for name, pairs in [("Train", train_pairs), ("Val", val_pairs)]:
        pos = sum(1 for p in pairs if p.difficulty == "positive")
        hard = sum(1 for p in pairs if p.difficulty == "hard")
        easy = sum(1 for p in pairs if p.difficulty == "easy")
        print(f"  {name}: {pos} positive, {hard} hard neg, {easy} easy neg")

    train_dataset = LandmarkCorrespondenceDataset(
        train_pairs, text_embeddings, text_input_dim, include_difficulties,
    )
    val_dataset = LandmarkCorrespondenceDataset(
        val_pairs, text_embeddings, text_input_dim, include_difficulties,
    )
    print(f"After filtering: train={len(train_dataset):,}, val={len(val_dataset):,}")

    collate_fn = functools.partial(
        collate_correspondence, text_input_dim=text_input_dim,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=config.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=config.num_workers,
        pin_memory=True,
    )

    encoder_config = TagBundleEncoderConfig(
        text_input_dim=text_input_dim, text_proj_dim=config.encoder.text_proj_dim,
    )
    classifier_config = CorrespondenceClassifierConfig(
        encoder=encoder_config,
        mlp_hidden_dim=config.classifier.mlp_hidden_dim,
        dropout=config.classifier.dropout,
    )
    model = CorrespondenceClassifier(classifier_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {num_params:,} parameters")

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_fraction)
    scheduler = create_lr_scheduler(optimizer, warmup_steps, total_steps,
                                    cosine=config.cosine_schedule)

    scaler = None
    if config.use_amp and device.type == "cuda":
        scaler = GradScaler()
        print("Mixed precision training enabled")

    tb_dir = output_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    best_auc = -math.inf
    best_model_path = output_dir / "best_model.pt"
    print(f"\nTraining for {config.num_epochs} epochs ({total_steps} steps, "
          f"{warmup_steps} warmup)")

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        train_loss, train_labels, train_probs = train_epoch(
            model, train_loader, optimizer, scheduler, device, scaler,
            config.gradient_clip_norm,
        )
        train_metrics = compute_metrics(train_labels, train_probs)
        val_loss, val_metrics = evaluate(model, val_loader, device)

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

        val_auc = val_metrics["auc_roc"]
        if not math.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best AUC: {best_auc:.4f}")

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_metrics": val_metrics,
        }, output_dir / f"checkpoint_epoch_{epoch + 1}.pt")

    writer.close()
    if best_auc == -math.inf:
        print(
            f"\nTraining complete, but no best model was saved: val AUC was NaN "
            f"every epoch (likely one-class batches). {best_model_path} does "
            f"not exist."
        )
    else:
        print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train landmark correspondence classifier")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
