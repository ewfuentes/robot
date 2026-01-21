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
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from experimental.overhead_matching.swag.data.llm_sentence_loader import (
    get_coverage_stats,
    load_llm_sentences_by_pruned_tags,
    load_llm_sentences_by_pruned_tags_from_dir,
    split_llm_sentences,
)
from experimental.overhead_matching.swag.data.osm_sentence_generator import (
    OSMSentenceGenerator,
)
from experimental.overhead_matching.swag.data.paired_sentence_dataset import (
    PairedBatchSampler,
    PairedContrastiveBatchSampler,
    PairedSentenceDataset,
    collate_paired_samples,
)
from experimental.overhead_matching.swag.data.sentence_dataset import (
    CombinedBatchSampler,
    CombinedDataset,
    ContrastiveBatchSampler,
    SentenceDataset,
    SentenceSample,
    create_collate_fn,
    get_unique_pruned_tags_from_landmarks,
    load_landmarks_from_db,
    load_tag_vocabularies_from_db,
    split_landmarks_by_id,
    split_pruned_tags,
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
    compute_contrastive_mrr,
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
                f"- **Sentence:** {sentence}\n"
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

            contrast_text_parts.append(f"- **Anchor:** {sentence}\n")

            # Show positives
            if len(pos_indices) > 0:
                contrast_text_parts.append("  - **Positives:**\n")
                for pi in pos_indices[:3]:
                    j = pi.item()
                    sim = similarity[i, j].item()
                    contrast_text_parts.append(
                        f"    - (sim={sim:.3f}) {batch.sentences[j]}\n"
                    )

            # Show top negative (highest similarity among negatives)
            neg_indices = [j for j in other_indices if j not in pos_indices.tolist()]
            if neg_indices:
                neg_sims = [(j, similarity[i, j].item()) for j in neg_indices]
                neg_sims.sort(key=lambda x: x[1], reverse=True)
                contrast_text_parts.append("  - **Hardest negatives:**\n")
                for j, sim in neg_sims[:2]:
                    contrast_text_parts.append(
                        f"    - (sim={sim:.3f}) {batch.sentences[j]}\n"
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


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: AdamW,
    scheduler: LambdaLR,
    config: TrainingConfig,
    device: torch.device,
    scaler: GradScaler | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor] | None:
    """Execute a single training step.

    Args:
        model: The model to train
        batch: Input batch
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        device: Device to train on
        scaler: GradScaler for mixed precision training (None to disable)

    Returns:
        Tuple of (losses, output, total_loss, batch) or None if no losses computed.
    """
    use_amp = scaler is not None

    with record_function("batch_to_device"):
        batch = batch.to(device)

    with record_function("forward"):
        # Use pre-tokenized input if available (parallel tokenization in workers)
        with autocast(device_type="cuda", enabled=use_amp):
            if batch.token_ids is not None:
                output = model(token_ids=batch.token_ids)
            else:
                output = model(sentences=batch.sentences)

    with record_function("compute_losses"):
        with autocast(device_type="cuda", enabled=use_amp):
            losses = compute_sentence_losses(output, batch, temperature=config.temperature)

    if not losses:
        return None

    with record_function("backward"):
        total_loss = aggregate_losses(losses)
        if use_amp:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

    with record_function("optimizer_step"):
        if use_amp:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            if config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            if config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return losses, output, total_loss, batch


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
    profile_dir: Path | None = None,
    scaler: GradScaler | None = None,
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
        profile_dir: If provided, enable PyTorch profiler and save traces here
        scaler: GradScaler for mixed precision training (None to disable)

    Returns:
        Tuple of (final_global_step, epoch_metrics)
    """
    model.train()

    epoch_losses: dict[str, list[float]] = {}
    epoch_accuracies: dict[str, list[float]] = {}

    def process_step_results(
        result: tuple | None,
        global_step: int,
        epoch_losses: dict,
        epoch_accuracies: dict,
    ) -> int:
        """Process results from a training step, update metrics, and log."""
        if result is None:
            return global_step

        losses, output, total_loss, batch = result

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

            # Log contrastive batch statistics
            for task, (mask, positive_matrix) in batch.contrastive_labels.items():
                n_samples_with_tag = mask.sum().item()
                n_positive_pairs = int(positive_matrix.sum().item() / 2)
                writer.add_scalar(f"train/contrastive_samples_{task}", n_samples_with_tag, global_step)
                writer.add_scalar(f"train/contrastive_pairs_{task}", n_positive_pairs, global_step)

        # Log example sentences periodically
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

        return global_step

    if profile_dir is not None:
        # Profile first 20 batches: 5 warmup, 15 active
        profile_dir.mkdir(parents=True, exist_ok=True)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            pbar = tqdm(dataloader, desc="Training (profiling)")
            for batch in pbar:
                with record_function("data_loading"):
                    pass  # Data already loaded by iterator

                result = train_step(model, batch, optimizer, scheduler, config, device, scaler)
                global_step = process_step_results(result, global_step, epoch_losses, epoch_accuracies)

                if result is not None:
                    pbar.set_postfix({"loss": result[2].item(), "step": global_step})

                # Stop after profiled batches
                if global_step >= 20:
                    break

        # Export Chrome trace
        trace_path = profile_dir / "trace.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"\nProfiler trace saved to {trace_path}")
        print("View with: chrome://tracing or https://ui.perfetto.dev")
    else:
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            result = train_step(model, batch, optimizer, scheduler, config, device, scaler)
            global_step = process_step_results(result, global_step, epoch_losses, epoch_accuracies)

            if result is not None:
                pbar.set_postfix({"loss": result[2].item(), "step": global_step})

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
    compute_global_mrr: bool = True,
) -> dict[str, float]:
    """Evaluate the model on a dataset.

    Args:
        model: The model to evaluate
        dataloader: Evaluation data loader
        config: Training configuration
        device: Device to evaluate on
        compute_global_mrr: If True, compute global MRR across all samples

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_losses: dict[str, list[float]] = {}
    all_accuracies: dict[str, list[float]] = {}

    # For global MRR computation
    all_base_embeddings: list[torch.Tensor] = []
    all_pruned_tags: list[frozenset] = []

    use_amp = config.use_amp and device.type == "cuda"

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = batch.to(device)
        # Use pre-tokenized input if available
        with autocast(device_type="cuda", enabled=use_amp):
            if batch.token_ids is not None:
                output = model(token_ids=batch.token_ids)
            else:
                output = model(sentences=batch.sentences)

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

        # Collect for global MRR
        if compute_global_mrr and batch.pruned_tags is not None:
            all_base_embeddings.append(output["base_embedding"].cpu())
            all_pruned_tags.extend(batch.pruned_tags)

    # Aggregate metrics
    metrics = {}
    for name, values in all_losses.items():
        metrics[f"val_{name}"] = sum(values) / len(values)
    for name, values in all_accuracies.items():
        metrics[f"val_{name}"] = sum(values) / len(values)

    # Compute global MRR across all samples
    if compute_global_mrr and all_base_embeddings:
        embeddings = torch.cat(all_base_embeddings, dim=0)
        normalized = F.normalize(embeddings, p=2, dim=-1)
        similarity = normalized @ normalized.T

        # Build positive matrix: same pruned_tags = positive
        n = len(all_pruned_tags)
        positive_matrix = torch.zeros(n, n, dtype=torch.float)
        for i in range(n):
            for j in range(n):
                if i != j and all_pruned_tags[i] == all_pruned_tags[j]:
                    positive_matrix[i, j] = 1.0

        global_mrr = compute_contrastive_mrr(similarity, positive_matrix)
        metrics["val_mrr_base_contrastive_global"] = global_mrr

    return metrics


def train(
    db_path: Path,
    output_dir: Path,
    tag_vocabs_path: Path | None = None,
    tensorboard_dir: Path | None = None,
    llm_sentences_path: Path | None = None,
    model_config: SentenceEmbeddingModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    classification_tags: list[str] | None = None,
    contrastive_tags: list[str] | None = None,
    required_tags: list[str] | None = None,
    train_split: float = 0.9,
    seed: int = 42,
    limit: int | None = None,
    profile: bool = False,
) -> None:
    """Main training function.

    Args:
        db_path: Path to landmarks SQLite database
        output_dir: Directory to save model and logs
        tag_vocabs_path: Optional path to precomputed vocabularies JSON
        tensorboard_dir: Optional TensorBoard log directory
        llm_sentences_path: Optional path to LLM sentences JSONL file for base contrastive loss
        model_config: Model configuration
        training_config: Training configuration
        classification_tags: Tags for classification tasks
        contrastive_tags: Tags for contrastive tasks
        required_tags: Tags that must always be included in sentences (defaults to contrastive_tags)
        train_split: Fraction of data for training
        seed: Random seed
        limit: Optional limit on number of landmarks
        profile: If True, enable PyTorch profiler for first epoch
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

    # Create model early to get tokenizer for parallel tokenization in DataLoader workers
    print("\nCreating model...")
    model = create_model_from_config(
        model_config=model_config,
        tag_vocabs=tag_vocabs,
        classification_task_names=classification_tags,
        contrastive_task_names=contrastive_tags,
    )
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    tokenizer = model.encoder.tokenize

    # Load landmarks to get all tags for LLM sentence matching
    print(f"\nLoading landmarks from {db_path}")
    landmarks = load_landmarks_from_db(db_path, limit=limit)
    print(f"Loaded {len(landmarks):,} landmarks")

    # Create sentence generator (shared across datasets)
    # Default required_tags to contrastive_tags to ensure contrastive learning has signal
    effective_required_tags = required_tags if required_tags is not None else contrastive_tags
    print(f"Required tags (always included): {effective_required_tags}")
    generator = OSMSentenceGenerator(required_tags=effective_required_tags)

    # Load LLM sentences if provided
    llm_sentences = None
    use_paired_dataset = False
    if llm_sentences_path is not None and llm_sentences_path.exists():
        print(f"\nLoading LLM sentences from {llm_sentences_path}")
        # Get all landmark tags for mapping
        all_tags = [l.tags for l in landmarks]
        # Support both file and directory paths
        if llm_sentences_path.is_dir():
            llm_sentences = load_llm_sentences_by_pruned_tags_from_dir(llm_sentences_path, all_tags)
        else:
            llm_sentences = load_llm_sentences_by_pruned_tags(llm_sentences_path, all_tags)
        coverage = get_coverage_stats(llm_sentences, len(all_tags))
        print(f"LLM sentences loaded: {coverage['llm_sentences_count']:,} sentences "
              f"({coverage['coverage_percent']:.1f}% coverage)")
        use_paired_dataset = len(llm_sentences) > 0

    # Extract unique pruned_tags from all landmarks for template-only dataset
    all_pruned_tags = get_unique_pruned_tags_from_landmarks(landmarks)
    print(f"Unique pruned_tags from landmarks: {len(all_pruned_tags):,}")

    # Create datasets
    use_combined_sampler = False
    if use_paired_dataset:
        # Use paired dataset indexed by pruned_tags
        print("\nUsing paired dataset (template + LLM sentences)")

        # Split LLM sentences into train/test
        train_llm, test_llm = split_llm_sentences(
            llm_sentences, train_fraction=train_split, seed=seed
        )
        print(f"LLM sentences split: Train {len(train_llm):,}, Test {len(test_llm):,}")

        # Create paired datasets (indexed by pruned_tags, not landmarks)
        train_paired_dataset = PairedSentenceDataset(
            train_llm, generator=generator, epoch=0
        )
        test_paired_dataset = PairedSentenceDataset(
            test_llm, generator=generator, epoch=0
        )
        print(f"Paired train set: {len(train_paired_dataset):,} unique pruned_tags")
        print(f"Paired test set: {len(test_paired_dataset):,} unique pruned_tags")

        # Check if we need combined sampling (mixed paired + template-only)
        paired_ratio = training_config.paired_ratio
        if paired_ratio < 1.0:
            use_combined_sampler = True
            print(f"\nUsing combined sampling with paired_ratio={paired_ratio}")

            # Get template-only pruned_tags (exclude those already in paired dataset)
            paired_tags_set = set(train_llm.keys())
            template_only_tags = [t for t in all_pruned_tags if t not in paired_tags_set]

            # Split template-only tags
            train_template_tags, test_template_tags = split_pruned_tags(
                template_only_tags, train_fraction=train_split, seed=seed
            )
            print(f"Template-only tags: {len(template_only_tags):,} "
                  f"(train: {len(train_template_tags):,}, test: {len(test_template_tags):,})")

            # Create template-only datasets
            train_template_dataset = SentenceDataset(
                train_template_tags, generator=generator, epoch=0
            )
            test_template_dataset = SentenceDataset(
                test_template_tags, generator=generator, epoch=0
            )

            # Create combined batch sampler
            train_batch_sampler = CombinedBatchSampler(
                paired_pruned_tags=train_paired_dataset.pruned_tags_list,
                template_pruned_tags=train_template_dataset.pruned_tags_list,
                contrastive_tags=contrastive_tags,
                batch_size=training_config.batch_size,
                paired_ratio=paired_ratio,
                groups_per_batch=training_config.groups_per_batch,
                samples_per_group=training_config.samples_per_group,
                seed=seed,
            )

            # Create collate function for combined batches
            base_collate_fn = create_collate_fn(
                tag_vocabs=tag_vocabs,
                classification_tasks=classification_tags,
                contrastive_tasks=contrastive_tags,
                include_base_contrastive=True,
                tokenizer=tokenizer,
                max_length=training_config.max_seq_length,
            )

            combined_dataset = CombinedDataset(
                paired_dataset=train_paired_dataset,
                template_dataset=train_template_dataset,
                sampler_len=len(train_batch_sampler),
            )

            def combined_collate_fn(nested_samples: list[list[SentenceSample]]):
                # Flatten the nested list of samples
                samples = [s for sample_list in nested_samples for s in sample_list]
                return base_collate_fn(samples)

            train_loader = DataLoader(
                combined_dataset,
                batch_sampler=train_batch_sampler,
                collate_fn=combined_collate_fn,
                num_workers=training_config.num_workers,
            )
        else:
            # Pure paired mode (paired_ratio = 1.0)
            train_dataset = train_paired_dataset
            test_dataset = test_paired_dataset

            # Create collate function that first flattens pairs, then collates
            base_collate_fn = create_collate_fn(
                tag_vocabs=tag_vocabs,
                classification_tasks=classification_tags,
                contrastive_tasks=contrastive_tags,
                include_base_contrastive=True,
                tokenizer=tokenizer,
                max_length=training_config.max_seq_length,
            )

            def paired_collate_fn(pairs: list[tuple[SentenceSample, SentenceSample]]):
                flat_samples = collate_paired_samples(pairs)
                return base_collate_fn(flat_samples)

            collate_fn = paired_collate_fn

            # Use contrastive batch sampler for tag-specific grouping
            train_batch_sampler = PairedContrastiveBatchSampler(
                pruned_tags_list=train_dataset.pruned_tags_list,
                contrastive_tags=contrastive_tags,
                batch_size=training_config.batch_size // 2,
                groups_per_batch=training_config.groups_per_batch,
                samples_per_group=training_config.samples_per_group,
                seed=seed,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_batch_sampler,
                collate_fn=collate_fn,
                num_workers=training_config.num_workers,
                pin_memory=True,
            )
    else:
        # Use standard template-only dataset indexed by pruned_tags
        print("\nUsing template-only dataset")

        # Split pruned_tags into train/test
        train_pruned_tags, test_pruned_tags = split_pruned_tags(
            all_pruned_tags, train_fraction=train_split, seed=seed
        )
        print(f"Split: Train {len(train_pruned_tags):,}, Test {len(test_pruned_tags):,}")

        # Create datasets indexed by pruned_tags
        train_dataset = SentenceDataset(train_pruned_tags, generator=generator, epoch=0)
        test_dataset = SentenceDataset(test_pruned_tags, generator=generator, epoch=0)

        # Create collate function
        collate_fn = create_collate_fn(
            tag_vocabs=tag_vocabs,
            classification_tasks=classification_tags,
            contrastive_tasks=contrastive_tags,
            include_base_contrastive=False,
            tokenizer=tokenizer,
            max_length=training_config.max_seq_length,
        )

        # Create data loaders with smart batching for contrastive learning
        train_batch_sampler = ContrastiveBatchSampler(
            pruned_tags_list=train_pruned_tags,
            contrastive_tags=contrastive_tags,
            batch_size=training_config.batch_size,
            groups_per_batch=training_config.groups_per_batch,
            samples_per_group=training_config.samples_per_group,
            seed=seed,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
            num_workers=training_config.num_workers,
            pin_memory=True,
        )

    # Test loader setup
    # For evaluation, we always use the paired test dataset if available (for global MRR)
    test_collate_fn = create_collate_fn(
        tag_vocabs=tag_vocabs,
        classification_tasks=classification_tags,
        contrastive_tasks=contrastive_tags,
        include_base_contrastive=True,  # Enable for global MRR computation
        tokenizer=tokenizer,
        max_length=training_config.max_seq_length,
    )

    if use_combined_sampler:
        # Use paired test dataset for evaluation (includes both template and LLM)
        test_dataset = test_paired_dataset

        def test_paired_collate_fn(pairs: list[tuple[SentenceSample, SentenceSample]]):
            flat_samples = collate_paired_samples(pairs)
            return test_collate_fn(flat_samples)
        test_collate = test_paired_collate_fn
        test_batch_size = training_config.batch_size // 2
    elif use_paired_dataset:
        # Pure paired mode - test_dataset already set
        def test_paired_collate_fn(pairs: list[tuple[SentenceSample, SentenceSample]]):
            flat_samples = collate_paired_samples(pairs)
            return test_collate_fn(flat_samples)
        test_collate = test_paired_collate_fn
        test_batch_size = training_config.batch_size // 2
    else:
        # Template-only mode - test_dataset already set
        test_collate = test_collate_fn
        test_batch_size = training_config.batch_size

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=test_collate,
        num_workers=training_config.num_workers,
        pin_memory=True,
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

    # Mixed precision training
    scaler = None
    if training_config.use_amp and device.type == "cuda":
        scaler = GradScaler()
        print("Mixed precision training enabled (FP16)")
    elif training_config.use_amp:
        print("Mixed precision training disabled (requires CUDA)")

    # TensorBoard writer
    writer = None
    if tensorboard_dir is not None:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)

    # Save config
    config_dict = {
        "db_path": str(db_path),
        "llm_sentences_path": str(llm_sentences_path) if llm_sentences_path else None,
        "use_paired_dataset": use_paired_dataset,
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
            "groups_per_batch": training_config.groups_per_batch,
            "samples_per_group": training_config.samples_per_group,
            "use_amp": training_config.use_amp,
        },
        "classification_tags": classification_tags,
        "contrastive_tags": contrastive_tags,
        "required_tags": effective_required_tags,
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
        if use_combined_sampler:
            train_paired_dataset.set_epoch(epoch)
            train_template_dataset.set_epoch(epoch)
        else:
            train_dataset.set_epoch(epoch)
        train_batch_sampler.set_epoch(epoch)

        # Train (enable profiler on first epoch if requested)
        profile_dir = output_dir / "profiler" if profile and epoch == 0 else None
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
            profile_dir=profile_dir,
            scaler=scaler,
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
        "--llm_sentences",
        type=Path,
        help="Path to LLM sentences JSONL file or directory of JSONL files (optional)",
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
    parser.add_argument(
        "--groups_per_batch",
        type=int,
        help="Number of contrastive groups to sample per batch (default: 32)",
    )
    parser.add_argument(
        "--samples_per_group",
        type=int,
        help="Target samples per contrastive group (default: 4)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler for first epoch (saves to output_dir/profiler)",
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
    if args.llm_sentences is not None:
        config.llm_sentences_path = args.llm_sentences
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
    if args.groups_per_batch is not None:
        config.training.groups_per_batch = args.groups_per_batch
    if args.samples_per_group is not None:
        config.training.samples_per_group = args.samples_per_group

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
        llm_sentences_path=config.llm_sentences_path,
        model_config=config.model,
        training_config=config.training,
        classification_tags=config.classification_tags,
        contrastive_tags=config.contrastive_tags,
        required_tags=config.required_tags,
        train_split=config.train_split,
        seed=config.seed,
        limit=config.limit,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
