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
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from experimental.overhead_matching.swag.data.llm_sentence_loader import (
    split_llm_sentences,
)
from experimental.overhead_matching.swag.data.osm_sentence_generator import (
    OSMSentenceGenerator,
)

from experimental.overhead_matching.swag.data.paired_sentence_dataset import (
    PairedSentenceDataset,
    flatten_samples,
    load_sentences_from_pickle,
)
from experimental.overhead_matching.swag.data.sentence_dataset import (
    CombinedDataset,
    ContrastiveBatchSampler,
    MixingSampler,
    SentenceDataset,
    SentenceSample,
    create_collate_fn,
    get_unique_pruned_tags_from_landmarks,
    load_landmarks_from_db,
    load_tag_vocabularies_from_db,
    split_pruned_tags,
)
from experimental.overhead_matching.swag.model.sentence_embedding_model import (
    create_model_from_config,
)
from experimental.overhead_matching.swag.scripts.sentence_configs import (
    DEFAULT_CLASSIFICATION_TAGS,
    DEFAULT_CONTRASTIVE_TAGS,
    DatasetConfig,
    OSMPairedDatasetConfig,
    SentenceEmbeddingModelConfig,
    SentenceTrainConfig,
    TemplateDatasetConfig,
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
        # Profile first 20 batches, then continue without profiling
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_batches = 20

        dataloader_iter = iter(dataloader)
        pbar = tqdm(total=len(dataloader), desc="Training (profiling)")

        # Phase 1: Profile first N batches
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(profile_batches):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    break

                with record_function("data_loading"):
                    pass  # Data already loaded by iterator

                result = train_step(model, batch, optimizer, scheduler, config, device, scaler)
                global_step = process_step_results(result, global_step, epoch_losses, epoch_accuracies)

                if result is not None:
                    pbar.set_postfix({"loss": result[2].item(), "step": global_step})
                pbar.update(1)

        # Export Chrome trace
        trace_path = profile_dir / "trace.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"\nProfiler trace saved to {trace_path}")
        print("View with: chrome://tracing or https://ui.perfetto.dev")

        # Phase 2: Continue with remaining batches without profiling
        pbar.set_description("Training")
        for batch in dataloader_iter:
            result = train_step(model, batch, optimizer, scheduler, config, device, scaler)
            global_step = process_step_results(result, global_step, epoch_losses, epoch_accuracies)

            if result is not None:
                pbar.set_postfix({"loss": result[2].item(), "step": global_step})
            pbar.update(1)

        pbar.close()
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
def evaluate_base_contrastive(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate base contrastive loss and global MRR on paired dataset.

    This is a focused evaluation that only computes metrics relevant to
    template <-> LLM alignment, skipping classification and tag-specific metrics.

    Args:
        model: The model to evaluate
        dataloader: Evaluation data loader (should be paired dataset)
        config: Training configuration
        device: Device to evaluate on

    Returns:
        Dictionary with 'val_loss_base_contrastive' and 'val_mrr_base_contrastive_global'
    """
    model.eval()

    all_base_losses: list[float] = []
    all_base_embeddings: list[torch.Tensor] = []
    all_pruned_tags: list[frozenset] = []

    use_amp = config.use_amp and device.type == "cuda"

    for batch in tqdm(dataloader, desc="Evaluating (base contrastive)"):
        batch = batch.to(device)

        with autocast(device_type="cuda", enabled=use_amp):
            if batch.token_ids is not None:
                output = model(token_ids=batch.token_ids)
            else:
                output = model(sentences=batch.sentences)

            # Compute only base contrastive loss
            if batch.base_contrastive_labels is not None:
                from experimental.overhead_matching.swag.scripts.sentence_losses import (
                    compute_base_contrastive_loss,
                )
                base_loss = compute_base_contrastive_loss(
                    base_embedding=output["base_embedding"],
                    base_contrastive_labels=batch.base_contrastive_labels,
                    temperature=config.temperature,
                )
                if base_loss.item() > 0:
                    all_base_losses.append(base_loss.item())

        # Collect embeddings for global MRR
        if batch.pruned_tags is not None:
            all_base_embeddings.append(output["base_embedding"].cpu())
            all_pruned_tags.extend(batch.pruned_tags)

    metrics = {}

    # Average base contrastive loss
    if all_base_losses:
        metrics["val_loss_base_contrastive"] = sum(all_base_losses) / len(all_base_losses)

    # Compute global MRR across all samples
    if all_base_embeddings:
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


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model on template dataset for classification and tag-specific metrics.

    For base contrastive / global MRR evaluation on paired data, use evaluate_base_contrastive().

    Args:
        model: The model to evaluate
        dataloader: Evaluation data loader (template dataset)
        config: Training configuration
        device: Device to evaluate on

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_losses: dict[str, list[float]] = {}
    all_accuracies: dict[str, list[float]] = {}

    use_amp = config.use_amp and device.type == "cuda"

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = batch.to(device)
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

    # Aggregate metrics
    metrics = {}
    for name, values in all_losses.items():
        metrics[f"val_{name}"] = sum(values) / len(values)
    for name, values in all_accuracies.items():
        metrics[f"val_{name}"] = sum(values) / len(values)

    return metrics


def train(
    config: SentenceTrainConfig,
    profile: bool = False,
) -> None:
    """Main training function.

    Args:
        config: Full training configuration
        profile: If True, enable PyTorch profiler for first epoch
    """
    # Setup
    setup_reproducibility(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Set tensorboard dir default if not provided
    tensorboard_dir = config.tensorboard_dir
    if tensorboard_dir is None:
        tensorboard_dir = config.output_dir / "tensorboard"

    print(f"Classification tasks: {config.classification_tags}")
    print(f"Contrastive tasks: {config.contrastive_tags}")

    # Validate datasets
    if not config.datasets:
        raise ValueError("At least one dataset must be configured in config.datasets")

    # Find first TemplateDatasetConfig for vocabulary building
    template_configs = [d for d in config.datasets if isinstance(d, TemplateDatasetConfig)]
    paired_configs = [d for d in config.datasets if isinstance(d, OSMPairedDatasetConfig)]

    # Load or build vocabularies
    if config.tag_vocabs_path is not None and config.tag_vocabs_path.exists():
        print(f"Loading vocabularies from {config.tag_vocabs_path}")
        with open(config.tag_vocabs_path) as f:
            tag_vocabs = json.load(f)
    elif template_configs:
        db_path = template_configs[0].db_path
        print(f"Building vocabularies from {db_path}")
        tag_vocabs = load_tag_vocabularies_from_db(
            db_path, config.classification_tags, min_count=100
        )
        # Save for future runs
        vocab_save_path = config.output_dir / "tag_vocabs.json"
        with open(vocab_save_path, "w") as f:
            json.dump(tag_vocabs, f, indent=2)
        print(f"Saved vocabularies to {vocab_save_path}")
    else:
        # No template configs and no vocab path - use empty vocabs
        print("No TemplateDatasetConfig found and no tag_vocabs_path - using empty vocabularies")
        tag_vocabs = {}

    # Filter classification tags to those with vocabularies
    classification_tags = [t for t in config.classification_tags if t in tag_vocabs]
    print(f"Classification tasks (with vocab): {classification_tags}")

    for tag, vocab in tag_vocabs.items():
        print(f"  {tag}: {len(vocab)} classes")

    # Create model early to get tokenizer for parallel tokenization in DataLoader workers
    print("\nCreating model...")
    model = create_model_from_config(
        model_config=config.model,
        tag_vocabs=tag_vocabs,
        classification_task_names=classification_tags,
        contrastive_task_names=config.contrastive_tags,
    )
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    tokenizer = model.encoder.tokenize

    # Create sentence generator (shared across datasets)
    # Default required_tags to contrastive_tags to ensure contrastive learning has signal
    effective_required_tags = (
        config.required_tags if config.required_tags is not None else config.contrastive_tags
    )
    print(f"Required tags (always included): {effective_required_tags}")
    generator = OSMSentenceGenerator(required_tags=effective_required_tags)

    # -------------------------------------------------------------------------
    # Build datasets and samplers from config.datasets
    # -------------------------------------------------------------------------
    train_datasets: dict[str, Dataset] = {}
    test_datasets: dict[str, Dataset] = {}
    train_sampler_configs: dict[str, tuple[Sampler, float, int]] = {}

    # Track whether we have any paired datasets (for base contrastive loss)
    has_paired_data = bool(paired_configs)

    # Process template datasets
    for i, tc in enumerate(template_configs):
        name = f"template_{i}" if len(template_configs) > 1 else "template"
        print(f"\nLoading template dataset from {tc.db_path}")
        landmarks = load_landmarks_from_db(tc.db_path, limit=config.limit)
        print(f"Loaded {len(landmarks):,} landmarks")

        # Extract unique pruned_tags
        all_pruned_tags = get_unique_pruned_tags_from_landmarks(landmarks)
        print(f"Unique pruned_tags: {len(all_pruned_tags):,}")

        # Split into train/test
        train_tags, test_tags = split_pruned_tags(
            all_pruned_tags, train_fraction=config.train_split, seed=config.seed
        )
        print(f"Split: train={len(train_tags):,}, test={len(test_tags):,}")

        # Create datasets
        train_dataset = SentenceDataset(train_tags, generator=generator, epoch=0)
        test_dataset = SentenceDataset(test_tags, generator=generator, epoch=0)
        train_datasets[name] = train_dataset
        test_datasets[name] = test_dataset

        # Create sampler with ContrastiveBatchSampler for grouping
        sampler = ContrastiveBatchSampler(
            pruned_tags_list=train_tags,
            contrastive_tags=config.contrastive_tags,
            batch_size=config.training.batch_size,
            groups_per_batch=config.training.groups_per_batch,
            samples_per_group=config.training.samples_per_group,
            seed=config.seed,
        )
        train_sampler_configs[name] = (sampler, tc.weight, 1)

    # Process paired datasets
    for i, pc in enumerate(paired_configs):
        name = f"osm_paired_{i}" if len(paired_configs) > 1 else "osm_paired"
        print(f"\nLoading OSM paired dataset from {pc.sentences_path}")
        llm_sentences = load_sentences_from_pickle(pc.sentences_path)
        print(f"Loaded {len(llm_sentences):,} sentence pairs")

        # Split into train/test
        train_llm, test_llm = split_llm_sentences(
            llm_sentences, train_fraction=config.train_split, seed=config.seed
        )
        print(f"Split: train={len(train_llm):,}, test={len(test_llm):,}")

        # Create datasets
        train_dataset = PairedSentenceDataset(train_llm, generator=generator, epoch=0)
        test_dataset = PairedSentenceDataset(test_llm, generator=generator, epoch=0)
        train_datasets[name] = train_dataset
        test_datasets[name] = test_dataset

        # Paired dataset uses simple shuffle sampler (positives come from pair structure)
        paired_batch_size = config.training.batch_size // 2  # Each index yields 2 samples
        sampler = BatchSampler(
            RandomSampler(train_dataset),
            batch_size=paired_batch_size,
            drop_last=False,
        )
        train_sampler_configs[name] = (sampler, pc.weight, 2)

    # Create combined dataset and mixing sampler
    combined_train_dataset = CombinedDataset(train_datasets)

    # Determine if we need mixing or single-source sampling
    use_mixing_sampler = len(train_sampler_configs) > 1
    include_base_contrastive = has_paired_data

    # Create collate function
    base_collate_fn = create_collate_fn(
        tag_vocabs=tag_vocabs,
        classification_tasks=classification_tags,
        contrastive_tasks=config.contrastive_tags,
        include_base_contrastive=include_base_contrastive,
        tokenizer=tokenizer,
        max_length=config.training.max_seq_length,
    )

    if use_mixing_sampler:
        # Mixed mode: use MixingSampler with multiple sources
        print(f"\nUsing MixingSampler with sources: {list(train_sampler_configs.keys())}")
        train_batch_sampler = MixingSampler(
            samplers=train_sampler_configs,
            batch_size=config.training.batch_size,
            seed=config.seed,
        )

        def mixing_collate_fn(samples: list):
            # Flatten samples (handles singles, pairs, tuples)
            flat = flatten_samples(samples)
            return base_collate_fn(flat)

        train_loader = DataLoader(
            combined_train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=mixing_collate_fn,
            num_workers=config.training.num_workers,
            pin_memory=True,
        )
    else:
        # Single source mode
        single_name = list(train_sampler_configs.keys())[0]
        single_sampler, _, _ = train_sampler_configs[single_name]
        print(f"\nUsing single source: {single_name}")
        train_batch_sampler = single_sampler

        single_dataset = train_datasets[single_name]

        # For paired datasets, need to flatten
        if isinstance(single_dataset, PairedSentenceDataset):
            def single_collate_fn(samples: list):
                flat = flatten_samples(samples)
                return base_collate_fn(flat)
            collate_fn = single_collate_fn
        else:
            collate_fn = base_collate_fn

        train_loader = DataLoader(
            single_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
            num_workers=config.training.num_workers,
            pin_memory=True,
        )

    # -------------------------------------------------------------------------
    # Test loader setup
    # -------------------------------------------------------------------------
    # Find template test datasets (for classification accuracy and tag-specific metrics)
    template_test_names = [n for n in test_datasets.keys() if n.startswith("template")]
    paired_test_names = [n for n in test_datasets.keys() if n.startswith("osm_paired")]

    # Template test loader: for classification accuracy and tag-specific contrastive metrics
    template_test_loader = None
    if template_test_names:
        template_test_collate_fn = create_collate_fn(
            tag_vocabs=tag_vocabs,
            classification_tasks=classification_tags,
            contrastive_tasks=config.contrastive_tags,
            include_base_contrastive=False,
            tokenizer=tokenizer,
            max_length=config.training.max_seq_length,
        )

        # Use the first template test dataset for evaluation
        template_test_loader = DataLoader(
            test_datasets[template_test_names[0]],
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=template_test_collate_fn,
            num_workers=config.training.num_workers,
            pin_memory=True,
        )

    # Paired test loader: for global MRR (template <-> LLM alignment)
    paired_test_loader = None
    if paired_test_names:
        paired_test_collate_fn = create_collate_fn(
            tag_vocabs=tag_vocabs,
            classification_tasks=classification_tags,
            contrastive_tasks=config.contrastive_tags,
            include_base_contrastive=True,
            tokenizer=tokenizer,
            max_length=config.training.max_seq_length,
        )

        def paired_collate_wrapper(pairs: list[tuple[SentenceSample, SentenceSample]]):
            flat = flatten_samples(pairs)
            return paired_test_collate_fn(flat)

        # Use the first paired test dataset for evaluation
        paired_test_loader = DataLoader(
            test_datasets[paired_test_names[0]],
            batch_size=config.training.batch_size // 2,
            shuffle=False,
            collate_fn=paired_collate_wrapper,
            num_workers=config.training.num_workers,
            pin_memory=True,
        )

    # Create optimizer with differential learning rates
    lr_config = config.training.lr_schedule
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

    total_steps = len(train_loader) * config.training.num_epochs
    scheduler = create_lr_scheduler(
        optimizer,
        warmup_steps=lr_config.warmup_steps,
        total_steps=total_steps,
    )

    # Mixed precision training
    scaler = None
    if config.training.use_amp and device.type == "cuda":
        scaler = GradScaler()
        print("Mixed precision training enabled (FP16)")
    elif config.training.use_amp:
        print("Mixed precision training disabled (requires CUDA)")

    # TensorBoard writer
    writer = None
    if tensorboard_dir is not None:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)

    # Save config
    save_config(config, config.output_dir / "config.yaml")

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(config.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")

        # Update epoch for sentence variety and batch sampling
        for dataset in train_datasets.values():
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
        train_batch_sampler.set_epoch(epoch)

        # Train (enable profiler on first epoch if requested)
        profile_dir = config.output_dir / "profiler" if profile and epoch == 0 else None
        global_step, train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config.training,
            writer=writer,
            global_step=global_step,
            device=device,
            tag_vocabs=tag_vocabs,
            profile_dir=profile_dir,
            scaler=scaler,
        )

        print(f"Train metrics: {train_metrics}")

        # Evaluate on template test set (classification, tag-specific contrastive)
        val_metrics = {}
        if template_test_loader is not None:
            val_metrics = evaluate(
                model=model,
                dataloader=template_test_loader,
                config=config.training,
                device=device,
            )

        # Evaluate on paired test set (base contrastive loss and global MRR)
        if paired_test_loader is not None:
            paired_metrics = evaluate_base_contrastive(
                model=model,
                dataloader=paired_test_loader,
                config=config.training,
                device=device,
            )
            val_metrics.update(paired_metrics)

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
            save_path = config.output_dir / "best_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

        # Save checkpoint
        checkpoint_path = config.output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
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
        required=True,
        help="Path to YAML config file. CLI args override config values.",
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

    # Load config from file
    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Override config with CLI args
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
    if args.groups_per_batch is not None:
        config.training.groups_per_batch = args.groups_per_batch
    if args.samples_per_group is not None:
        config.training.samples_per_group = args.samples_per_group

    # Validate required fields
    if config.output_dir is None:
        parser.error("--output_dir is required (via config or CLI)")

    train(config=config, profile=args.profile)


if __name__ == "__main__":
    main()
