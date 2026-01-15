"""Training script for sentence embedding model.

Loads landmarks from SQLite database, generates sentences on-the-fly,
and trains a multi-task model with classification and contrastive heads.
"""

import argparse
import json
import os
import random
from pathlib import Path

# Silence tokenizer parallelism warning when using DataLoader with multiple workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import common.torch.load_torch_deps  # noqa: F401 - Must import before torch

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from experimental.overhead_matching.swag.data.osm_sentence_generator import (
    OSMSentenceGenerator,
)
from experimental.overhead_matching.swag.data.sentence_dataset import (
    ContrastiveBatchSampler,
    SentenceDataset,
    create_collate_fn,
    load_landmarks_from_db,
    load_tag_vocabularies_from_db,
    split_landmarks_by_id,
)
from experimental.overhead_matching.swag.model.sentence_embedding_model import (
    create_model_from_config,
)
from experimental.overhead_matching.swag.scripts.sentence_configs import (
    DEFAULT_CLASSIFICATION_TAGS,
    DEFAULT_CONTRASTIVE_TAGS,
    SentenceEmbeddingModelConfig,
    SentenceTrainConfig,
    TrainingConfig,
    load_config,
    save_config,
    LearningRateScheduleConfig,
)
from experimental.overhead_matching.swag.scripts.sentence_losses import (
    aggregate_losses,
    compute_classification_accuracy,
    compute_sentence_losses,
)


def log_example_sentences(
    writer: SummaryWriter,
    batch,  # SentenceBatch
    output: dict,
    tag_vocabs: dict[str, dict[str, int]],
    global_step: int,
    num_examples: int = 5,
) -> None:
    """Log example sentences with classification predictions and contrastive similarities.

    Args:
        writer: TensorBoard SummaryWriter
        batch: SentenceBatch with sentences and labels
        output: Model output dict with logits and embeddings
        tag_vocabs: Tag vocabularies for converting indices to names
        global_step: Current training step
        num_examples: Number of examples to log
    """
    # Build reverse vocabularies (index -> name)
    reverse_vocabs = {
        task: {idx: name for name, idx in vocab.items()}
        for task, vocab in tag_vocabs.items()
    }

    # Classification examples
    cls_text_parts = ["## Classification Examples\n"]
    for task, (mask, labels) in batch.classification_labels.items():
        if task not in output["classification_logits"]:
            continue
        if mask.sum() == 0:
            continue

        logits = output["classification_logits"][task]
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        # Get indices where this task applies
        valid_indices = mask.nonzero(as_tuple=True)[0][:num_examples]

        if len(valid_indices) == 0:
            continue

        cls_text_parts.append(f"\n### {task}\n")
        reverse_vocab = reverse_vocabs.get(task, {})

        for idx in valid_indices:
            i = idx.item()
            sentence = batch.sentences[i]
            true_label = labels[i].item()
            pred_label = preds[i].item()
            confidence = probs[i, pred_label].item()

            true_name = reverse_vocab.get(true_label, f"[{true_label}]")
            pred_name = reverse_vocab.get(pred_label, f"[{pred_label}]")

            status = "✓" if true_label == pred_label else "✗"

            # Get top-3 predictions
            top_probs, top_indices = probs[i].topk(3)
            top_preds = [
                f"{reverse_vocab.get(ti, f'[{ti}]')} ({tp:.2f})"
                for tp, ti in zip(top_probs.tolist(), top_indices.tolist())
            ]

            cls_text_parts.append(
                f"- **Sentence:** {sentence[:100]}...\n"
                f"  - **True:** {true_name} | **Pred:** {pred_name} ({confidence:.2f}) {status}\n"
                f"  - **Top-3:** {', '.join(top_preds)}\n"
            )

    writer.add_text("examples/classification", "".join(cls_text_parts), global_step)

    # Contrastive examples
    contrast_text_parts = ["## Contrastive Examples\n"]
    for task, (mask, positive_matrix) in batch.contrastive_labels.items():
        if task not in output["contrastive_embeddings"]:
            continue
        if mask.sum() < 2:
            continue

        embeddings = output["contrastive_embeddings"][task]
        # Compute similarity matrix
        similarity = embeddings @ embeddings.T

        # Get indices where this task applies
        valid_indices = mask.nonzero(as_tuple=True)[0][:num_examples]

        if len(valid_indices) == 0:
            continue

        contrast_text_parts.append(f"\n### {task}\n")

        for idx in valid_indices:
            i = idx.item()
            sentence = batch.sentences[i]

            # Find positives (same value)
            pos_indices = positive_matrix[i].nonzero(as_tuple=True)[0]

            # Get similarities to all other masked samples
            other_indices = [j for j in mask.nonzero(as_tuple=True)[0].tolist() if j != i]

            if not other_indices:
                continue

            contrast_text_parts.append(f"- **Anchor:** {sentence[:80]}...\n")

            # Show positives
            if len(pos_indices) > 0:
                contrast_text_parts.append("  - **Positives:**\n")
                for pi in pos_indices[:3]:
                    j = pi.item()
                    sim = similarity[i, j].item()
                    contrast_text_parts.append(
                        f"    - (sim={sim:.3f}) {batch.sentences[j][:60]}...\n"
                    )

            # Show top negative (highest similarity among negatives)
            neg_indices = [j for j in other_indices if j not in pos_indices.tolist()]
            if neg_indices:
                neg_sims = [(j, similarity[i, j].item()) for j in neg_indices]
                neg_sims.sort(key=lambda x: x[1], reverse=True)
                contrast_text_parts.append("  - **Hardest negatives:**\n")
                for j, sim in neg_sims[:2]:
                    contrast_text_parts.append(
                        f"    - (sim={sim:.3f}) {batch.sentences[j][:60]}...\n"
                    )

    writer.add_text("examples/contrastive", "".join(contrast_text_parts), global_step)


def setup_reproducibility(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_lr_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Create learning rate scheduler with warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return step / max(warmup_steps, 1)
        else:
            # Linear decay to 0
            remaining = total_steps - step
            total_decay = total_steps - warmup_steps
            return max(0.0, remaining / total_decay)

    return LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    config: TrainingConfig,
    writer: SummaryWriter | None,
    global_step: int,
    device: torch.device,
    tag_vocabs: dict[str, dict[str, int]] | None = None,
    log_examples_every_n_steps: int = 500,
) -> tuple[int, dict[str, float]]:
    """Train for one epoch.

    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        writer: TensorBoard writer (optional)
        global_step: Current global step
        device: Device to train on
        tag_vocabs: Tag vocabularies for logging examples
        log_examples_every_n_steps: How often to log example sentences

    Returns:
        Tuple of (final_global_step, epoch_metrics)
    """
    model.train()

    epoch_losses: dict[str, list[float]] = {}
    epoch_accuracies: dict[str, list[float]] = {}

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        batch = batch.to(device)

        # Forward pass
        output = model(batch.sentences)

        # Compute losses
        losses = compute_sentence_losses(output, batch, temperature=config.temperature)

        if not losses:
            continue

        # Aggregate and backward
        total_loss = aggregate_losses(losses)
        total_loss.backward()

        # Gradient clipping
        if config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Track losses
        for name, loss in losses.items():
            if name not in epoch_losses:
                epoch_losses[name] = []
            epoch_losses[name].append(loss.item())

        # Compute accuracies
        with torch.no_grad():
            accuracies = compute_classification_accuracy(output, batch)
            for name, acc in accuracies.items():
                if name not in epoch_accuracies:
                    epoch_accuracies[name] = []
                epoch_accuracies[name].append(acc)

        # Logging
        global_step += 1
        if writer is not None and global_step % config.log_every_n_steps == 0:
            writer.add_scalar("train/total_loss", total_loss.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            for name, loss in losses.items():
                writer.add_scalar(f"train/{name}", loss.item(), global_step)
            for name, acc in accuracies.items():
                writer.add_scalar(f"train/{name}", acc, global_step)

        # Log example sentences periodically (also at first step of epoch)
        is_first_step_of_epoch = (global_step - 1) % len(dataloader) == 0
        if (
            writer is not None
            and tag_vocabs is not None
            and (global_step % log_examples_every_n_steps == 0 or is_first_step_of_epoch)
        ):
            with torch.no_grad():
                log_example_sentences(
                    writer=writer,
                    batch=batch,
                    output=output,
                    tag_vocabs=tag_vocabs,
                    global_step=global_step,
                )

        # Update progress bar
        pbar.set_postfix({"loss": total_loss.item(), "step": global_step})

    # Compute epoch metrics
    metrics = {}
    for name, values in epoch_losses.items():
        metrics[name] = sum(values) / len(values)
    for name, values in epoch_accuracies.items():
        metrics[name] = sum(values) / len(values)

    return global_step, metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model on a dataset.

    Args:
        model: The model to evaluate
        dataloader: Evaluation data loader
        config: Training configuration
        device: Device to evaluate on

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_losses: dict[str, list[float]] = {}
    all_accuracies: dict[str, list[float]] = {}

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = batch.to(device)
        output = model(batch.sentences)

        losses = compute_sentence_losses(output, batch, temperature=config.temperature)
        accuracies = compute_classification_accuracy(output, batch)

        for name, loss in losses.items():
            if name not in all_losses:
                all_losses[name] = []
            all_losses[name].append(loss.item())

        for name, acc in accuracies.items():
            if name not in all_accuracies:
                all_accuracies[name] = []
            all_accuracies[name].append(acc)

    # Aggregate metrics
    metrics = {}
    for name, values in all_losses.items():
        metrics[f"val_{name}"] = sum(values) / len(values)
    for name, values in all_accuracies.items():
        metrics[f"val_{name}"] = sum(values) / len(values)

    return metrics


def train(
    db_path: Path,
    output_dir: Path,
    tag_vocabs_path: Path | None = None,
    tensorboard_dir: Path | None = None,
    model_config: SentenceEmbeddingModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    classification_tags: list[str] | None = None,
    contrastive_tags: list[str] | None = None,
    train_split: float = 0.9,
    seed: int = 42,
    limit: int | None = None,
) -> None:
    """Main training function.

    Args:
        db_path: Path to landmarks SQLite database
        output_dir: Directory to save model and logs
        tag_vocabs_path: Optional path to precomputed vocabularies JSON
        tensorboard_dir: Optional TensorBoard log directory
        model_config: Model configuration
        training_config: Training configuration
        classification_tags: Tags for classification tasks
        contrastive_tags: Tags for contrastive tasks
        train_split: Fraction of data for training
        seed: Random seed
        limit: Optional limit on number of landmarks
    """
    # Setup
    setup_reproducibility(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use defaults if not provided
    model_config = model_config or SentenceEmbeddingModelConfig()
    training_config = training_config or TrainingConfig()
    classification_tags = classification_tags or list(DEFAULT_CLASSIFICATION_TAGS)
    contrastive_tags = contrastive_tags or list(DEFAULT_CONTRASTIVE_TAGS)

    print(f"Classification tasks: {classification_tags}")
    print(f"Contrastive tasks: {contrastive_tags}")

    # Load or build vocabularies
    if tag_vocabs_path is not None and tag_vocabs_path.exists():
        print(f"Loading vocabularies from {tag_vocabs_path}")
        with open(tag_vocabs_path) as f:
            tag_vocabs = json.load(f)
    else:
        print(f"Building vocabularies from {db_path}")
        tag_vocabs = load_tag_vocabularies_from_db(
            db_path, classification_tags, min_count=100
        )
        # Save for future runs
        vocab_save_path = output_dir / "tag_vocabs.json"
        with open(vocab_save_path, "w") as f:
            json.dump(tag_vocabs, f, indent=2)
        print(f"Saved vocabularies to {vocab_save_path}")

    # Filter classification tags to those with vocabularies
    classification_tags = [t for t in classification_tags if t in tag_vocabs]
    print(f"Classification tasks (with vocab): {classification_tags}")

    for tag, vocab in tag_vocabs.items():
        print(f"  {tag}: {len(vocab)} classes")

    # Load landmarks
    print(f"\nLoading landmarks from {db_path}")
    landmarks = load_landmarks_from_db(db_path, limit=limit)
    print(f"Loaded {len(landmarks):,} landmarks")

    # Split data
    train_landmarks, test_landmarks = split_landmarks_by_id(
        landmarks, train_fraction=train_split, seed=seed
    )
    print(f"Train: {len(train_landmarks):,}, Test: {len(test_landmarks):,}")

    # Create sentence generator (shared across datasets)
    generator = OSMSentenceGenerator()

    # Create datasets
    train_dataset = SentenceDataset(train_landmarks, generator=generator, epoch=0)
    test_dataset = SentenceDataset(test_landmarks, generator=generator, epoch=0)

    # Create collate function
    collate_fn = create_collate_fn(
        tag_vocabs=tag_vocabs,
        classification_tasks=classification_tags,
        contrastive_tasks=contrastive_tags,
    )

    # Create data loaders with smart batching for contrastive learning
    train_batch_sampler = ContrastiveBatchSampler(
        landmarks=train_landmarks,
        contrastive_tags=contrastive_tags,
        batch_size=training_config.batch_size,
        groups_per_batch=32,  # Sample from 32 groups per batch
        samples_per_group=4,  # Up to 4 samples per group
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=collate_fn,
        num_workers=training_config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=training_config.num_workers,
        pin_memory=True,
    )

    # Create model
    print("\nCreating model...")
    model = create_model_from_config(
        model_config=model_config,
        tag_vocabs=tag_vocabs,
        classification_task_names=classification_tags,
        contrastive_task_names=contrastive_tags,
    )
    model = model.to(device)
    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Create optimizer with differential learning rates
    lr_config = training_config.lr_schedule
    encoder_lr = lr_config.encoder_lr
    heads_lr = lr_config.heads_lr if lr_config.heads_lr is not None else encoder_lr

    if heads_lr != encoder_lr:
        # Differential learning rates: lower for encoder, higher for heads
        encoder_params = list(model.encoder.parameters())
        head_params = (
            list(model.classification_heads.parameters())
            + list(model.contrastive_heads.parameters())
            + list(model.presence_heads.parameters())
        )
        param_groups = [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": head_params, "lr": heads_lr},
        ]
        optimizer = AdamW(param_groups)
        print(f"Using differential learning rates: encoder={encoder_lr}, heads={heads_lr}")
    else:
        # Single learning rate for all parameters
        optimizer = AdamW(model.parameters(), lr=encoder_lr)
        print(f"Using single learning rate: {encoder_lr}")

    total_steps = len(train_loader) * training_config.num_epochs
    scheduler = create_lr_scheduler(
        optimizer,
        warmup_steps=lr_config.warmup_steps,
        total_steps=total_steps,
    )

    # TensorBoard writer
    writer = None
    if tensorboard_dir is not None:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)

    # Save config
    config_dict = {
        "db_path": str(db_path),
        "model_config": {
            "encoder_name": model_config.encoder_name,
            "projection_dim": model_config.projection_dim,
            "freeze_encoder": model_config.freeze_encoder,
        },
        "training_config": {
            "batch_size": training_config.batch_size,
            "num_epochs": training_config.num_epochs,
            "encoder_lr": training_config.lr_schedule.encoder_lr,
            "heads_lr": training_config.lr_schedule.heads_lr,
            "temperature": training_config.temperature,
        },
        "classification_tags": classification_tags,
        "contrastive_tags": contrastive_tags,
        "train_split": train_split,
        "seed": seed,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(training_config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{training_config.num_epochs}")

        # Update epoch for sentence variety and batch sampling
        train_dataset.set_epoch(epoch)
        train_batch_sampler.set_epoch(epoch)

        # Train
        global_step, train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=training_config,
            writer=writer,
            global_step=global_step,
            device=device,
            tag_vocabs=tag_vocabs,
        )

        print(f"Train metrics: {train_metrics}")

        # Evaluate
        val_metrics = evaluate(
            model=model,
            dataloader=test_loader,
            config=training_config,
            device=device,
        )
        print(f"Val metrics: {val_metrics}")

        # Log to TensorBoard
        if writer is not None:
            for name, value in val_metrics.items():
                writer.add_scalar(f"val/{name}", value, epoch)

        # Save best model
        val_loss = sum(
            v
            for k, v in val_metrics.items()
            if k.startswith("val_loss_")
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
                "val_metrics": val_metrics,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to {checkpoint_path}")

    if writer is not None:
        writer.close()

    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Train sentence embedding model")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config file. CLI args override config values.",
    )
    parser.add_argument(
        "--db_path",
        type=Path,
        help="Path to landmarks SQLite database",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Output directory for model and logs",
    )
    parser.add_argument(
        "--tag_vocabs",
        type=Path,
        help="Path to precomputed tag_vocabs.json (optional)",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=Path,
        help="TensorBoard log directory (optional)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size (default: 256)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs (default: 10)",
    )
    parser.add_argument(
        "--encoder_lr",
        type=float,
        help="Learning rate for encoder (default: 2e-5)",
    )
    parser.add_argument(
        "--heads_lr",
        type=float,
        help="Learning rate for heads (default: 1e-4, use same as encoder_lr if not set)",
    )
    parser.add_argument(
        "--encoder_name",
        type=str,
        help="Sentence transformer encoder name",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        help="Fraction of data for training (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of landmarks to load (for testing)",
    )
    args = parser.parse_args()

    # Load config from file or use defaults
    if args.config is not None:
        print(f"Loading config from {args.config}")
        config = load_config(args.config)
    else:
        config = SentenceTrainConfig()

    # Override config with CLI args
    if args.db_path is not None:
        config.db_path = args.db_path
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.tag_vocabs is not None:
        config.tag_vocabs_path = args.tag_vocabs
    if args.tensorboard_dir is not None:
        config.tensorboard_dir = args.tensorboard_dir
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    if args.encoder_lr is not None:
        config.training.lr_schedule.encoder_lr = args.encoder_lr
    if args.heads_lr is not None:
        config.training.lr_schedule.heads_lr = args.heads_lr
    if args.encoder_name is not None:
        config.model.encoder_name = args.encoder_name
    if args.train_split is not None:
        config.train_split = args.train_split
    if args.seed is not None:
        config.seed = args.seed
    if args.limit is not None:
        config.limit = args.limit

    # Validate required fields
    if config.db_path is None:
        parser.error("--db_path is required (via config or CLI)")
    if config.output_dir is None:
        parser.error("--output_dir is required (via config or CLI)")

    # Set tensorboard dir default
    tensorboard_dir = config.tensorboard_dir
    if tensorboard_dir is None:
        tensorboard_dir = config.output_dir / "tensorboard"

    train(
        db_path=config.db_path,
        output_dir=config.output_dir,
        tag_vocabs_path=config.tag_vocabs_path,
        tensorboard_dir=tensorboard_dir,
        model_config=config.model,
        training_config=config.training,
        classification_tags=config.classification_tags,
        contrastive_tags=config.contrastive_tags,
        train_split=config.train_split,
        seed=config.seed,
        limit=config.limit,
    )


if __name__ == "__main__":
    main()
