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
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class CorrespondenceDataset(Dataset):
    """Dataset that loads pano<->OSM correspondence pairs."""

    def __init__(self, correspondence_file: Path, convert_osm_tags_to_nl: bool = True):
        """
        Args:
            correspondence_file: Path to JSON file with correspondences
            convert_osm_tags_to_nl: Whether to convert OSM tags to natural language
        """
        logger.info(f"Loading correspondences from {correspondence_file}")
        with open(correspondence_file, "r") as f:
            self.data = json.load(f)

        self.pairs = []
        self.convert_osm_tags = convert_osm_tags_to_nl

        # Extract positive pairs from correspondences
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
                            osm_text = self._process_osm_item(osm_item)
                            if osm_text:  # Skip if empty
                                self.pairs.append((pano_text, osm_text))

        logger.info(f"Loaded {len(self.pairs)} positive pairs")

    def _process_osm_item(self, osm_item: dict) -> str:
        """Convert OSM item to text representation."""
        if "tags" in osm_item:
            tags_str = osm_item["tags"]
            if self.convert_osm_tags:
                # Convert "name=Shell; building=roof" to natural language
                parts = []
                for tag_pair in tags_str.split(";"):
                    tag_pair = tag_pair.strip()
                    if "=" in tag_pair:
                        key, value = tag_pair.split("=", 1)
                        key = key.strip().replace("_", " ").replace(":", " ")
                        value = value.strip()
                        # Create simple natural language format
                        parts.append(f"{key}: {value}")
                return ", ".join(parts) if parts else tags_str
            else:
                return tags_str
        return ""

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def train_step(
    batch: List[Tuple[str, str]],
    model: nn.Module,
    tokenizer,
    optimizer,
    device,
    max_length: int,
    scale: float,
    loss_fn,
) -> Tuple[torch.Tensor, dict]:
    """Execute one training step with in-batch negatives."""

    # Separate anchor and positive texts
    anchor_texts = [pair[0] for pair in batch]
    positive_texts = [pair[1] for pair in batch]

    # Tokenize
    anchor_inputs = tokenizer(
        anchor_texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    ).to(device)

    positive_inputs = tokenizer(
        positive_texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    ).to(device)

    # Compute embeddings
    anchor_embeddings = model(anchor_inputs["input_ids"], anchor_inputs["attention_mask"])
    positive_embeddings = model(
        positive_inputs["input_ids"], positive_inputs["attention_mask"]
    )

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


def train(
    model: nn.Module,
    tokenizer,
    train_dataloader: DataLoader,
    optimizer,
    lr_scheduler,
    device,
    args,
    writer: SummaryWriter,
):
    """Main training loop."""

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    max_grad_norm = 1.0

    global_step = 0
    total_loss = 0
    log_interval = 100

    logger.info("Starting training...")
    logger.info(f"  Total steps: {args.steps}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Max sequence length: {args.max_length}")
    logger.info(f"  TensorBoard logs: {args.output}/tensorboard")

    pbar = tqdm.tqdm(total=args.steps, desc="Training")

    while global_step < args.steps:
        for batch in train_dataloader:
            if global_step >= args.steps:
                break

            optimizer.zero_grad()

            loss, metrics = train_step(
                batch=batch,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                device=device,
                max_length=args.max_length,
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
                save_model(model, tokenizer, save_path)
                logger.info(f"Saved checkpoint to {save_path}")

    pbar.close()
    writer.close()

    # Save final model
    save_path = os.path.join(args.output, "final")
    save_model(model, tokenizer, save_path)
    logger.info(f"Saved final model to {save_path}")


def save_model(model: nn.Module, tokenizer, output_path: str):
    """Save model and tokenizer."""
    os.makedirs(output_path, exist_ok=True)

    # Save tokenizer
    tokenizer.save_pretrained(output_path)

    # Save transformer model (not the wrapper)
    if hasattr(model, "transformer"):
        model.transformer.config.save_pretrained(output_path)
        torch.save(model.transformer.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
    else:
        # Fallback if structure is different
        torch.save(model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    logger.info(f"Model saved to {output_path}")


class SentenceEmbeddingModel(nn.Module):
    """Wrapper model for training that mimics TrainableSentenceEmbedder."""

    def __init__(self, model_name: str, normalize: bool = True):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.normalize = normalize

    def forward(self, input_ids, attention_mask):
        """Forward pass with mean pooling."""
        model_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pool(model_output, attention_mask)

        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def mean_pool(self, model_output, attention_mask):
        """Apply mean pooling over token embeddings."""
        token_embeddings = model_output[0]  # [batch, seq_len, hidden_dim]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


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
        "--convert_osm_tags",
        action="store_true",
        help="Convert OSM tags to natural language format",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
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

    # Load tokenizer and model
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = SentenceEmbeddingModel(args.model, normalize=True)
    model = model.to(device)

    # Load dataset
    dataset = CorrespondenceDataset(
        Path(args.correspondence_file), convert_osm_tags_to_nl=args.convert_osm_tags
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

    # Train
    train(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        args=args,
        writer=writer,
    )

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
