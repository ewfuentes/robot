"""
Standalone training script for sentence embedding models.

This script trains a sentence embedding model on correspondence data,
following the training approach used by all-MiniLM-L6-v2:
- Cross-entropy loss with in-batch negatives
- Symmetric loss (both anchor->positive and positive->anchor)
- Mean pooling + L2 normalization

The trained model can then be loaded into the geolocalization pipeline
via the model_weights_path parameter in TrainableSentenceEmbedderConfig.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple
import tqdm

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import common.torch.load_torch_deps
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from experimental.overhead_matching.swag.model.trainable_sentence_embedder import TrainableSentenceEmbedder
from experimental.overhead_matching.swag.model.swag_config_types import TrainableSentenceEmbedderConfig
from experimental.overhead_matching.swag.model.semantic_landmark_utils import load_all_jsonl_from_folder, make_sentence_dict_from_json
from experimental.overhead_matching.swag.model.semantic_landmark_extractor import _custom_id_from_props
from common.torch.load_and_save_models import save_model as save_model_with_metadata


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class CorrespondenceDataset(Dataset):
    """Dataset that loads pano<->OSM correspondence pairs."""

    def __init__(
        self,
        correspondence_file: Path,
        sentence_directory: Path | None = None,
        use_natural_language: bool = True,
    ):
        """
        Args:
            correspondence_file: Path to JSON file with correspondences
            sentence_directory: Optional path to directory with natural language descriptions
            use_natural_language: Whether to use natural language descriptions from sentence_directory
        """
        logger.info(f"Loading correspondences from {correspondence_file}")
        with open(correspondence_file, "r") as f:
            self.data = json.load(f)

        self.pairs = []
        self.use_natural_language = use_natural_language
        self.sentence_dict = {}

        # Load sentences from directory if using natural language mode
        if self.use_natural_language and sentence_directory is not None:
            sentence_dir = Path(sentence_directory)
            if sentence_dir.exists():
                logger.info(f"Loading sentences from {sentence_dir}")
                sentence_jsons = load_all_jsonl_from_folder(sentence_dir)
                self.sentence_dict, _ = make_sentence_dict_from_json(sentence_jsons)
                logger.info(f"Loaded {len(self.sentence_dict)} sentences")
            else:
                logger.warning(f"Sentence directory not found: {sentence_dir}. Using tag format instead.")
                self.use_natural_language = False

        # Extract positive pairs from correspondences
        missing_sentences = 0
        for entry_id, entry in tqdm.tqdm(self.data.items(), desc="Processing correspondences"):
            pano_descs = entry["pano"]
            osm_items = entry["osm"]
            matches = entry["matches"]["matches"]

            for match in matches:
                pano_idx = match["set_1_id"]
                osm_indices = match["set_2_matches"]

                if pano_idx < len(pano_descs):
                    pano_text = pano_descs[pano_idx]

                    for osm_idx in osm_indices:
                        if osm_idx < len(osm_items):
                            osm_item = osm_items[osm_idx]
                            osm_text, found_sentence = self._process_osm_item(osm_item)
                            if osm_text:  # Skip if empty
                                self.pairs.append((pano_text, osm_text))
                                if not found_sentence and self.use_natural_language:
                                    missing_sentences += 1

        logger.info(f"Loaded {len(self.pairs)} positive pairs")
        if missing_sentences > 0 and self.use_natural_language:
            logger.warning(
                f"Could not find natural language sentences for {missing_sentences} OSM items. "
                f"Using tag format as fallback. This may happen if the correspondence file contains "
                f"OSM tags that were not in the original sentence generation."
            )

    def _parse_osm_tags(self, tags_str: str) -> dict:
        """Parse OSM tags string into a dictionary."""
        props = {}
        for tag_pair in tags_str.split(";"):
            tag_pair = tag_pair.strip()
            if "=" in tag_pair:
                key, value = tag_pair.split("=", 1)
                props[key.strip()] = value.strip()
        return props

    def _process_osm_item(self, osm_item: dict) -> tuple[str, bool]:
        """Convert OSM item to text representation.

        Returns:
            tuple of (text, found_sentence_in_dict)
        """
        if "tags" not in osm_item:
            return "", False

        tags_str = osm_item["tags"]
        pruned_props = self._parse_osm_tags(tags_str)

        if self.use_natural_language:
            # Try to load natural language description from sentences
            custom_id = _custom_id_from_props(pruned_props)
            if custom_id in self.sentence_dict:
                return self.sentence_dict[custom_id], True

        # Fall back to tag format: "key: value, key: value, ..."
        parts = [f"{k}: {v}" for k, v in sorted(pruned_props.items())]
        text = ", ".join(parts) if parts else ""
        return text, False

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def train_step(
    batch: List[Tuple[str, str]],
    model: TrainableSentenceEmbedder,
    optimizer,
    device,
    scale: float,
    loss_fn,
) -> Tuple[torch.Tensor, dict]:
    """Execute one training step with in-batch negatives."""

    # Separate anchor and positive texts
    anchor_texts = [pair[0] for pair in batch]
    positive_texts = [pair[1] for pair in batch]

    # Compute embeddings using TrainableSentenceEmbedder
    # (handles tokenization internally)
    anchor_embeddings = model(anchor_texts)
    positive_embeddings = model(positive_texts)

    # Compute similarity matrix: [batch_size, batch_size]
    # Scaled cosine similarity (embeddings are already normalized)
    scores = torch.mm(anchor_embeddings, positive_embeddings.transpose(0, 1)) * scale

    # Labels: diagonal elements are the true matches
    # anchor[i] should match with positive[i]
    labels = torch.arange(len(scores), dtype=torch.long, device=device)

    # Symmetric loss as in CLIP and all-MiniLM-L6-v2
    loss = (loss_fn(scores, labels) + loss_fn(scores.transpose(0, 1), labels)) / 2

    # Compute accuracy
    predictions = torch.argmax(scores, dim=1)
    accuracy = (predictions == labels).float().mean()

    metrics = {
        "loss": loss.item(),
        "accuracy": accuracy.item(),
        "avg_positive_sim": scores.diag().mean().item(),
        "avg_negative_sim": (scores.sum() - scores.diag().sum()).item()
        / (scores.numel() - len(scores)),
    }

    return loss, metrics


def extract_validation_data(correspondence_file: Path, sentence_directory: Path | None, use_natural_language: bool):
    """Extract unique texts and correspondence mappings from validation file.

    Returns:
        tuple: (pano_texts, osm_texts, pano_to_osm_indices)
            - pano_texts: List of unique pano descriptions
            - osm_texts: List of unique OSM descriptions
            - pano_to_osm_indices: Dict mapping pano_idx -> list of osm_indices
    """
    logger.info(f"Loading validation data from {correspondence_file}")

    # Load correspondence data
    with open(correspondence_file, 'r') as f:
        corr_data = json.load(f)

    # Load sentences if using natural language mode
    sentence_dict = {}
    if use_natural_language and sentence_directory is not None:
        sentence_dir = Path(sentence_directory)
        if sentence_dir.exists():
            sentence_jsons = load_all_jsonl_from_folder(sentence_dir)
            sentence_dict, _ = make_sentence_dict_from_json(sentence_jsons)
            logger.info(f"Loaded {len(sentence_dict)} validation sentences")

    def parse_osm_tags(tags_str: str) -> dict:
        """Parse OSM tags string into a dictionary."""
        props = {}
        for tag_pair in tags_str.split(";"):
            tag_pair = tag_pair.strip()
            if "=" in tag_pair:
                key, value = tag_pair.split("=", 1)
                props[key.strip()] = value.strip()
        return props

    def process_osm_item(osm_item: dict) -> str:
        """Convert OSM item to text representation."""
        if "tags" not in osm_item:
            return ""

        tags_str = osm_item["tags"]
        pruned_props = parse_osm_tags(tags_str)

        if use_natural_language and sentence_dict:
            custom_id = _custom_id_from_props(pruned_props)
            if custom_id in sentence_dict:
                return sentence_dict[custom_id]

        # Fall back to tag format
        parts = [f"{k}: {v}" for k, v in sorted(pruned_props.items())]
        return ", ".join(parts) if parts else ""

    # Build unique text lists and mapping
    pano_text_to_idx = {}
    osm_text_to_idx = {}
    pano_texts = []
    osm_texts = []
    pano_to_osm_indices = {}

    for entry_id, entry in corr_data.items():
        pano_descs = entry["pano"]
        osm_items = entry["osm"]
        matches = entry["matches"]["matches"]

        for match in matches:
            pano_idx_in_entry = match["set_1_id"]
            osm_indices_in_entry = match["set_2_matches"]

            if pano_idx_in_entry >= len(pano_descs):
                continue

            pano_desc = pano_descs[pano_idx_in_entry]

            # Handle different pano formats: dict with 'sentence' key or plain string
            if isinstance(pano_desc, dict):
                pano_text = pano_desc.get('sentence', '')
            else:
                pano_text = pano_desc

            if not pano_text:
                continue

            # Add pano text if not seen
            if pano_text not in pano_text_to_idx:
                pano_text_to_idx[pano_text] = len(pano_texts)
                pano_texts.append(pano_text)

            pano_idx = pano_text_to_idx[pano_text]

            # Process matched OSM items
            for osm_idx_in_entry in osm_indices_in_entry:
                if osm_idx_in_entry >= len(osm_items):
                    continue

                osm_item = osm_items[osm_idx_in_entry]
                osm_text = process_osm_item(osm_item)

                if not osm_text:
                    continue

                # Add OSM text if not seen
                if osm_text not in osm_text_to_idx:
                    osm_text_to_idx[osm_text] = len(osm_texts)
                    osm_texts.append(osm_text)

                osm_idx = osm_text_to_idx[osm_text]

                # Record correspondence
                if pano_idx not in pano_to_osm_indices:
                    pano_to_osm_indices[pano_idx] = []
                if osm_idx not in pano_to_osm_indices[pano_idx]:
                    pano_to_osm_indices[pano_idx].append(osm_idx)

    logger.info(f"Validation set: {len(pano_texts)} unique panos, {len(osm_texts)} unique OSMs")

    if len(pano_texts) == 0 or len(osm_texts) == 0:
        logger.warning("Validation set is empty! Validation will be skipped.")
        return None

    return pano_texts, osm_texts, pano_to_osm_indices


@torch.no_grad()
def compute_validation_metrics(
    model: TrainableSentenceEmbedder,
    pano_texts: List[str],
    osm_texts: List[str],
    pano_to_osm_indices: dict,
    device: torch.device,
    batch_size: int = 128
) -> dict:
    """Compute retrieval metrics on validation set.

    Args:
        model: The sentence embedding model
        pano_texts: List of pano descriptions
        osm_texts: List of OSM descriptions
        pano_to_osm_indices: Dict mapping pano_idx -> list of correct osm_indices
        device: Device to run on
        batch_size: Batch size for embedding computation

    Returns:
        dict: Validation metrics including MRR, recall@k, mean/median rank
    """
    model.eval()

    # Safety check
    if len(pano_texts) == 0 or len(osm_texts) == 0:
        logger.warning("Empty validation set, returning empty metrics")
        return {}

    logger.info("Computing validation embeddings...")

    # Compute pano embeddings in batches
    pano_embeddings_list = []
    for i in range(0, len(pano_texts), batch_size):
        batch_texts = pano_texts[i:i+batch_size]
        embeddings = model(batch_texts)
        pano_embeddings_list.append(embeddings.cpu())
    pano_embeddings = torch.cat(pano_embeddings_list, dim=0)

    # Compute OSM embeddings in batches
    osm_embeddings_list = []
    for i in range(0, len(osm_texts), batch_size):
        batch_texts = osm_texts[i:i+batch_size]
        embeddings = model(batch_texts)
        osm_embeddings_list.append(embeddings.cpu())
    osm_embeddings = torch.cat(osm_embeddings_list, dim=0)

    # Chunk size for computing rankings to avoid OOM
    chunk_size = 500

    # Move embeddings to device for computation
    pano_embeddings = pano_embeddings.to(device)
    osm_embeddings = osm_embeddings.to(device)

    # Compute rankings in chunks to avoid OOM
    # We only need the rankings, not the full similarity matrix
    all_ranks = []

    for chunk_idx, start_idx in enumerate(range(0, len(pano_embeddings), chunk_size)):
        end_idx = min(start_idx + chunk_size, len(pano_embeddings))
        pano_chunk = pano_embeddings[start_idx:end_idx]

        # Compute similarity for this chunk: [chunk_size, N_osm]
        similarity_chunk = torch.mm(pano_chunk, osm_embeddings.T)

        # Rank OSMs for this chunk (descending order)
        ranks_chunk = torch.argsort(similarity_chunk, dim=1, descending=True)
        all_ranks.append(ranks_chunk.cpu())

        # Free memory immediately
        del similarity_chunk, ranks_chunk
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    ranks = torch.cat(all_ranks, dim=0)  # [N_pano, N_osm]

    # Compute metrics
    reciprocal_ranks = []
    min_ranks = []
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    recall_at_100 = 0

    num_queries = len(pano_to_osm_indices)

    for pano_idx, correct_osm_indices in pano_to_osm_indices.items():
        # Get ranked OSM indices for this pano
        ranked_osm_indices = ranks[pano_idx].cpu().tolist()

        # Find positions of correct OSMs in the ranked list
        positions = []
        for correct_osm_idx in correct_osm_indices:
            try:
                pos = ranked_osm_indices.index(correct_osm_idx)
                positions.append(pos + 1)  # 1-indexed rank
            except ValueError:
                # This should not happen but handle gracefully
                continue

        if not positions:
            continue

        # MRR: reciprocal rank of first correct match
        min_rank = min(positions)
        min_ranks.append(min_rank)
        reciprocal_ranks.append(1.0 / min_rank)

        # Recall@k: is any correct match in top-k?
        if min_rank <= 1:
            recall_at_1 += 1
        if min_rank <= 5:
            recall_at_5 += 1
        if min_rank <= 10:
            recall_at_10 += 1
        if min_rank <= 100:
            recall_at_100 += 1

    # Compute final metrics
    mrr = sum(reciprocal_ranks) / num_queries if num_queries > 0 else 0.0
    mean_rank = sum(min_ranks) / num_queries if num_queries > 0 else 0.0
    median_rank = torch.tensor(min_ranks).float().median().item() if min_ranks else 0.0

    metrics = {
        "pano_to_osm/mrr": mrr,
        "pano_to_osm/recall@1": recall_at_1 / num_queries if num_queries > 0 else 0.0,
        "pano_to_osm/recall@5": recall_at_5 / num_queries if num_queries > 0 else 0.0,
        "pano_to_osm/recall@10": recall_at_10 / num_queries if num_queries > 0 else 0.0,
        "pano_to_osm/recall@100": recall_at_100 / num_queries if num_queries > 0 else 0.0,
        "pano_to_osm/mean_rank": mean_rank,
        "pano_to_osm/median_rank": median_rank,
    }

    # Compute reverse direction: osm -> pano
    # Build osm_to_pano mapping
    osm_to_pano_indices = {}
    for pano_idx, osm_indices in pano_to_osm_indices.items():
        for osm_idx in osm_indices:
            if osm_idx not in osm_to_pano_indices:
                osm_to_pano_indices[osm_idx] = []
            if pano_idx not in osm_to_pano_indices[osm_idx]:
                osm_to_pano_indices[osm_idx].append(pano_idx)

    # Compute reverse rankings in chunks (osm -> pano)
    all_ranks_reverse = []
    for chunk_idx, start_idx in enumerate(range(0, len(osm_embeddings), chunk_size)):
        end_idx = min(start_idx + chunk_size, len(osm_embeddings))
        osm_chunk = osm_embeddings[start_idx:end_idx]

        # Compute similarity for this chunk: [chunk_size, N_pano]
        similarity_chunk = torch.mm(osm_chunk, pano_embeddings.T)

        # Rank panos for this chunk (descending order)
        ranks_chunk = torch.argsort(similarity_chunk, dim=1, descending=True)
        all_ranks_reverse.append(ranks_chunk.cpu())

        # Free memory immediately
        del similarity_chunk, ranks_chunk
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    ranks_reverse = torch.cat(all_ranks_reverse, dim=0)  # [N_osm, N_pano]

    reciprocal_ranks_reverse = []
    min_ranks_reverse = []
    num_queries_reverse = len(osm_to_pano_indices)

    for osm_idx, correct_pano_indices in osm_to_pano_indices.items():
        ranked_pano_indices = ranks_reverse[osm_idx].cpu().tolist()

        positions = []
        for correct_pano_idx in correct_pano_indices:
            try:
                pos = ranked_pano_indices.index(correct_pano_idx)
                positions.append(pos + 1)
            except ValueError:
                continue

        if not positions:
            continue

        min_rank = min(positions)
        min_ranks_reverse.append(min_rank)
        reciprocal_ranks_reverse.append(1.0 / min_rank)

    mrr_reverse = sum(reciprocal_ranks_reverse) / num_queries_reverse if num_queries_reverse > 0 else 0.0
    mean_rank_reverse = sum(min_ranks_reverse) / num_queries_reverse if num_queries_reverse > 0 else 0.0

    metrics["osm_to_pano/mrr"] = mrr_reverse
    metrics["osm_to_pano/mean_rank"] = mean_rank_reverse

    return metrics


def train(
    model: TrainableSentenceEmbedder,
    train_dataloader: DataLoader,
    optimizer,
    lr_scheduler,
    device,
    args,
    writer: SummaryWriter,
    validation_data: tuple | None = None,
):
    """Main training loop.

    Args:
        validation_data: Optional tuple of (pano_texts, osm_texts, pano_to_osm_indices)
    """

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    max_grad_norm = 1.0

    global_step = 0
    total_loss = 0
    log_interval = 100

    # Track best validation MRR
    best_val_mrr = 0.0

    logger.info("Starting training...")
    logger.info(f"  Total steps: {args.steps}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  TensorBoard logs: {args.output}/tensorboard")
    if validation_data is not None:
        logger.info(f"  Validation will run every {args.validation_interval} steps")

    pbar = tqdm.tqdm(total=args.steps, desc="Training")

    while global_step < args.steps:
        for batch in train_dataloader:
            if global_step >= args.steps:
                break

            optimizer.zero_grad()

            loss, metrics = train_step(
                batch=batch,
                model=model,
                optimizer=optimizer,
                device=device,
                scale=args.scale,
                loss_fn=loss_fn,
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()

            total_loss += metrics["loss"]
            global_step += 1

            # TensorBoard logging (every step)
            writer.add_scalar("train/loss", metrics["loss"], global_step)
            writer.add_scalar("train/accuracy", metrics["accuracy"], global_step)
            writer.add_scalar("train/avg_positive_sim", metrics["avg_positive_sim"], global_step)
            writer.add_scalar("train/avg_negative_sim", metrics["avg_negative_sim"], global_step)
            writer.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], global_step)

            # Console logging
            if global_step % log_interval == 0:
                avg_loss = total_loss / log_interval
                pbar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "acc": f"{metrics['accuracy']:.3f}",
                        "pos_sim": f"{metrics['avg_positive_sim']:.3f}",
                        "neg_sim": f"{metrics['avg_negative_sim']:.3f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    }
                )
                total_loss = 0

            pbar.update(1)

            # Save checkpoint
            if (global_step + 1) % args.save_steps == 0:
                save_path = os.path.join(args.output, f"step_{global_step + 1}")
                save_model(model, save_path, vars(args))
                logger.info(f"Saved checkpoint to {save_path}")

            # Run validation
            if validation_data is not None and (global_step + 1) % args.validation_interval == 0:
                logger.info(f"\nRunning validation at step {global_step + 1}...")
                pano_texts, osm_texts, pano_to_osm_indices = validation_data

                val_metrics = compute_validation_metrics(
                    model=model,
                    pano_texts=pano_texts,
                    osm_texts=osm_texts,
                    pano_to_osm_indices=pano_to_osm_indices,
                    device=device,
                    batch_size=args.batch_size * 2  # Use larger batch for validation
                )

                # Log validation metrics
                logger.info("Validation Results:")
                for metric_name, metric_value in val_metrics.items():
                    logger.info(f"  {metric_name}: {metric_value:.4f}")
                    writer.add_scalar(f"val/{metric_name}", metric_value, global_step + 1)

                # Save best checkpoint based on validation MRR
                current_mrr = val_metrics["pano_to_osm/mrr"]
                if args.save_best_checkpoint and current_mrr > best_val_mrr:
                    best_val_mrr = current_mrr
                    best_path = os.path.join(args.output, "best_model")
                    save_model(model, best_path, vars(args))
                    logger.info(f"New best validation MRR: {best_val_mrr:.4f}! Saved to {best_path}")

                # Return model to training mode
                model.train()

    pbar.close()
    writer.close()

    # Save final model
    save_path = os.path.join(args.output, "final")
    save_model(model, save_path, vars(args))
    logger.info(f"Saved final model to {save_path}")


def save_model(model: TrainableSentenceEmbedder, output_path: str, training_args: dict):
    """Save model using the same logic as train.py."""
    # Create example input for the model
    example_texts = ["This is a test sentence."]

    # Save using common save_model function
    save_model_with_metadata(
        model=model,
        save_path=Path(output_path),
        example_model_inputs=(example_texts,),
        aux_information={
            "training_args": training_args,
            "model_type": "TrainableSentenceEmbedder",
        }
    )

    logger.info(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train sentence embedding model on correspondence data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/paraphrase-MiniLM-L3-v2",
        help="Pretrained model name from HuggingFace",
    )
    parser.add_argument(
        "--correspondence_file",
        type=str,
        required=True,
        help="Path to correspondence JSON file",
    )
    parser.add_argument(
        "--sentence_directory",
        type=str,
        default=None,
        help="Optional path to directory with natural language descriptions",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for trained model"
    )
    parser.add_argument(
        "--steps", type=int, default=10000, help="Total training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size per device"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=384,
        help="Output embedding dimension",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=500, help="Number of warmup steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=20.0,
        help="Similarity scale factor (20 for cosine similarity)",
    )
    parser.add_argument(
        "--use_natural_language",
        action="store_true",
        help="Use natural language descriptions from sentence_directory",
    )
    parser.add_argument(
        "--freeze_weights",
        action="store_true",
        help="Freeze transformer weights (only train projection layer)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="Path to validation correspondence JSON file",
    )
    parser.add_argument(
        "--validation_sentence_directory",
        type=str,
        default=None,
        help="Optional path to directory with natural language descriptions for validation",
    )
    parser.add_argument(
        "--validation_interval",
        type=int,
        default=1000,
        help="Run validation every N steps",
    )
    parser.add_argument(
        "--save_best_checkpoint",
        action="store_true",
        default=True,
        help="Save best model based on validation MRR",
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Save training arguments
    with open(os.path.join(args.output, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Create model config and initialize TrainableSentenceEmbedder
    logger.info(f"Loading model: {args.model}")
    model_config = TrainableSentenceEmbedderConfig(
        pretrained_model_name_or_path=args.model,
        output_dim=args.output_dim,
        max_sequence_length=args.max_length,
        freeze_weights=args.freeze_weights,
        model_weights_path=None,
    )
    model = TrainableSentenceEmbedder(model_config)
    model = model.to(device)

    # Load dataset
    sentence_dir = Path(args.sentence_directory) if args.sentence_directory else None
    dataset = CorrespondenceDataset(
        correspondence_file=Path(args.correspondence_file),
        sentence_directory=sentence_dir,
        use_natural_language=args.use_natural_language,
    )

    # Create dataloader with infinite iteration
    def infinite_dataloader(dataloader):
        """Create infinite dataloader by repeating."""
        while True:
            for batch in dataloader:
                yield batch

    # Use num_workers=0 to avoid tokenizer fork warnings
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    train_dataloader = infinite_dataloader(dataloader)

    # Setup optimizer and scheduler (use torch.optim.AdamW to avoid deprecation warning)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps,
    )

    # Setup TensorBoard
    tensorboard_dir = os.path.join(args.output, "tensorboard")
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Log hyperparameters
    writer.add_text("hyperparameters", json.dumps(vars(args), indent=2), 0)

    # Load validation data if provided
    validation_data = None
    if args.validation_file is not None:
        val_sentence_dir = Path(args.validation_sentence_directory) if args.validation_sentence_directory else None
        validation_data = extract_validation_data(
            correspondence_file=Path(args.validation_file),
            sentence_directory=val_sentence_dir,
            use_natural_language=args.use_natural_language,
        )

    # Train
    train(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        args=args,
        writer=writer,
        validation_data=validation_data,
    )

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
