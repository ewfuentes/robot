"""
Standalone training script for sentence embedding models using sentence-transformers.

This script trains a sentence embedding model on correspondence data using
the sentence-transformers library and best practices from:
- https://github.com/huggingface/sentence-transformers
- https://sbert.net/docs/sentence_transformer/loss_overview.html
- https://huggingface.co/blog/train-sentence-transformers

Training approach:
- CachedMultipleNegativesRankingLoss for efficient in-batch negatives
- InformationRetrievalEvaluator for validation
- SentenceTransformerTrainer for training loop
- Automatic TensorBoard logging

The trained model can be loaded into the geolocalization pipeline
via the model_weights_path parameter in TrainableSentenceEmbedderConfig.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
import tqdm

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import common.torch.load_torch_deps
import torch
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers
from sentence_transformers.trainer import SentenceTransformerTrainer

from experimental.overhead_matching.swag.model.trainable_sentence_embedder import TrainableSentenceEmbedder
from experimental.overhead_matching.swag.model.swag_config_types import TrainableSentenceEmbedderConfig
from experimental.overhead_matching.swag.model.semantic_landmark_utils import load_all_jsonl_from_folder, make_sentence_dict_from_json
from experimental.overhead_matching.swag.model.semantic_landmark_extractor import custom_id_from_props
from common.torch.load_and_save_models import save_model as save_model_with_metadata


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_osm_tags(tags_str: str) -> dict:
    """Parse OSM tags string into a dictionary."""
    props = {}
    for tag_pair in tags_str.split(";"):
        tag_pair = tag_pair.strip()
        if "=" in tag_pair:
            key, value = tag_pair.split("=", 1)
            props[key.strip()] = value.strip()
    return props


def load_sentence_dict(sentence_directory: Path | None) -> Dict[str, str]:
    """Load natural language sentence dictionary from directory."""
    if sentence_directory is None or not sentence_directory.exists():
        return {}

    logger.info(f"Loading sentences from {sentence_directory}")
    sentence_jsons = load_all_jsonl_from_folder(sentence_directory)
    sentence_dict, _ = make_sentence_dict_from_json(sentence_jsons)
    logger.info(f"Loaded {len(sentence_dict)} sentences")
    return sentence_dict


def process_osm_item(
    osm_item: dict,
    sentence_dict: Dict[str, str],
    use_natural_language: bool
) -> str:
    """Convert OSM item to text representation.

    Args:
        osm_item: OSM item dict with 'tags' field
        sentence_dict: Dictionary mapping custom_id -> natural language sentence
        use_natural_language: Whether to use natural language descriptions

    Returns:
        Text representation of OSM item
    """
    if "tags" not in osm_item:
        return ""

    tags_str = osm_item["tags"]
    pruned_props = parse_osm_tags(tags_str)

    if use_natural_language and sentence_dict:
        custom_id = custom_id_from_props(pruned_props)
        if custom_id in sentence_dict:
            return sentence_dict[custom_id]
        else:
            print(f"Can't find custom id {custom_id}")
            return ""


def create_training_dataset(
    correspondence_file: Path,
    sentence_directory: Path | None,
    use_natural_language: bool
) -> Dataset:
    """Create HuggingFace Dataset with (anchor, positive) pairs from correspondence file.

    Args:
        correspondence_file: Path to correspondence JSON file
        sentence_directory: Optional directory with natural language descriptions
        use_natural_language: Whether to use natural language mode

    Returns:
        datasets.Dataset with columns ["anchor", "positive"]
    """
    logger.info(f"Loading correspondences from {correspondence_file}")
    with open(correspondence_file, "r") as f:
        corr_data = json.load(f)

    # Load sentence dictionary if using natural language
    sentence_dict = load_sentence_dict(sentence_directory) if use_natural_language else {}

    # Extract (anchor, positive) pairs
    anchors = []
    positives = []
    missing_sentences = 0

    for entry_id, entry in tqdm.tqdm(corr_data.items(), desc="Processing correspondences"):
        pano_descs = entry["pano"]
        osm_items = entry["osm"]
        matches = entry["matches"]["matches"]

        for match in matches:
            pano_idx = match["set_1_id"] - 1
            osm_indices = [x-1 for x in match["set_2_matches"]]

            if pano_idx >= len(pano_descs):
                continue

            # Handle different pano formats: dict with 'sentence' key or plain string
            pano_desc = pano_descs[pano_idx]
            if isinstance(pano_desc, dict):
                pano_text = pano_desc.get('sentence', '')
            else:
                pano_text = pano_desc

            if not pano_text:
                continue

            # Process matched OSM items
            for osm_idx in osm_indices:
                if osm_idx >= len(osm_items):
                    continue

                osm_item = osm_items[osm_idx]
                osm_text = process_osm_item(osm_item, sentence_dict, use_natural_language)

                if not osm_text:
                    print(f"Failed to find {osm_item}")
                    continue


                anchors.append(pano_text)
                positives.append(osm_text)

    logger.info(f"Created {len(anchors)} training pairs")
    if missing_sentences > 0 and use_natural_language:
        logger.warning(
            f"Could not find natural language sentences for {missing_sentences} OSM items. "
            f"Using tag format as fallback."
        )

    # Create HuggingFace Dataset
    # Column order matters! First column is anchor, second is positive
    dataset = Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
    })

    return dataset


def load_huggingface_dataset(
    dataset_name: str,
    target_size: int | None = None,
    seed: int = 42
) -> Dataset:
    """Load and prepare a HuggingFace benchmark dataset.

    Args:
        dataset_name: Name of the dataset (natural-questions, squad, all-nli)
        target_size: Optional target size to trim the dataset
        seed: Random seed for shuffling

    Returns:
        datasets.Dataset with columns ["anchor", "positive"]
    """
    logger.info(f"Loading HuggingFace dataset: sentence-transformers/{dataset_name}")

    # Map dataset names to their HuggingFace paths and column mappings
    dataset_configs = {
        "natural-questions": {
            "path": "sentence-transformers/natural-questions",
            "anchor_col": "query",
            "positive_col": "answer",
        },
        "squad": {
            "path": "sentence-transformers/squad",
            "anchor_col": "query",
            "positive_col": "answer",
        },
        "all-nli": {
            "path": "sentence-transformers/all-nli",
            "anchor_col": "anchor",
            "positive_col": "positive",
        },
    }

    if dataset_name not in dataset_configs:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {list(dataset_configs.keys())}"
        )

    config = dataset_configs[dataset_name]

    # Load dataset from HuggingFace
    # Most sentence-transformers datasets have a "train" split
    try:
        dataset = load_dataset(config["path"], split="train")
    except Exception as e:
        logger.error(f"Failed to load dataset {config['path']}: {e}")
        raise

    logger.info(f"Loaded {len(dataset)} examples from {dataset_name}")

    # Map columns to standard format: ["anchor", "positive"]
    anchor_col = config["anchor_col"]
    positive_col = config["positive_col"]

    # Check if columns exist
    if anchor_col not in dataset.column_names or positive_col not in dataset.column_names:
        raise ValueError(
            f"Expected columns '{anchor_col}' and '{positive_col}' not found in dataset. "
            f"Available columns: {dataset.column_names}"
        )

    # Rename columns if needed
    if anchor_col != "anchor" or positive_col != "positive":
        dataset = dataset.rename_column(anchor_col, "anchor")
        dataset = dataset.rename_column(positive_col, "positive")

    # Keep only anchor and positive columns
    columns_to_remove = [col for col in dataset.column_names if col not in ["anchor", "positive"]]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    # Filter out empty examples
    original_size = len(dataset)
    dataset = dataset.filter(lambda x: x["anchor"] and x["positive"])
    filtered_count = original_size - len(dataset)
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} empty examples")

    # Trim to target size if specified
    if target_size is not None and len(dataset) > target_size:
        logger.info(f"Trimming dataset from {len(dataset)} to {target_size} examples")
        # Shuffle before selecting to get a random subset
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(target_size))

    logger.info(f"Final dataset size: {len(dataset)} (anchor, positive) pairs")

    return dataset


def create_information_retrieval_evaluator(
    correspondence_file: Path,
    sentence_directory: Path | None,
    use_natural_language: bool,
    name: str = "validation"
) -> InformationRetrievalEvaluator | None:
    """Create InformationRetrievalEvaluator from correspondence file.

    Args:
        correspondence_file: Path to correspondence JSON file
        sentence_directory: Optional directory with natural language descriptions
        use_natural_language: Whether to use natural language mode
        name: Name for the evaluator

    Returns:
        InformationRetrievalEvaluator or None if no valid data
    """
    logger.info(f"Creating {name} evaluator from {correspondence_file}")

    with open(correspondence_file, 'r') as f:
        corr_data = json.load(f)

    # Load sentence dictionary if using natural language
    sentence_dict = load_sentence_dict(sentence_directory) if use_natural_language else {}

    # Build unique text lists and mappings
    # For InformationRetrievalEvaluator we need:
    # - queries: Dict[query_id, query_text]
    # - corpus: Dict[corpus_id, corpus_text]
    # - relevant_docs: Dict[query_id, Set[corpus_id]]

    queries = {}
    corpus = {}
    relevant_docs = {}

    pano_text_to_id = {}
    osm_text_to_id = {}

    for entry_id, entry in tqdm.tqdm(corr_data.items(), desc=f"Processing {name} data"):
        pano_descs = entry["pano"]
        osm_items = entry["osm"]
        matches = entry["matches"]["matches"]

        for match in matches:
            pano_idx_in_entry = match["set_1_id"]
            osm_indices_in_entry = match["set_2_matches"]

            if pano_idx_in_entry >= len(pano_descs):
                continue

            # Handle different pano formats
            pano_desc = pano_descs[pano_idx_in_entry]
            if isinstance(pano_desc, dict):
                pano_text = pano_desc.get('sentence', '')
            else:
                pano_text = pano_desc

            if not pano_text:
                continue

            # Add pano text to queries if not seen
            if pano_text not in pano_text_to_id:
                pano_id = f"pano_{len(pano_text_to_id)}"
                pano_text_to_id[pano_text] = pano_id
                queries[pano_id] = pano_text

            pano_id = pano_text_to_id[pano_text]

            # Process matched OSM items
            for osm_idx_in_entry in osm_indices_in_entry:
                if osm_idx_in_entry >= len(osm_items):
                    continue

                osm_item = osm_items[osm_idx_in_entry]
                osm_text = process_osm_item(osm_item, sentence_dict, use_natural_language)

                if not osm_text:
                    continue

                # Add OSM text to corpus if not seen
                if osm_text not in osm_text_to_id:
                    osm_id = f"osm_{len(osm_text_to_id)}"
                    osm_text_to_id[osm_text] = osm_id
                    corpus[osm_id] = osm_text

                osm_id = osm_text_to_id[osm_text]

                # Record correspondence in relevant_docs
                if pano_id not in relevant_docs:
                    relevant_docs[pano_id] = set()
                relevant_docs[pano_id].add(osm_id)

    logger.info(f"{name} set: {len(queries)} queries, {len(corpus)} corpus items")

    if len(queries) == 0 or len(corpus) == 0:
        logger.warning(f"{name} set is empty! Evaluator will not be created.")
        return None

    # Create InformationRetrievalEvaluator
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        show_progress_bar=True,
    )

    return evaluator


def save_model_for_pipeline(
    model: SentenceTransformer,
    output_path: Path,
    training_args: dict
):
    """Save SentenceTransformer in format compatible with pipeline loading.

    This function extracts the trained weights and saves them using the same
    format as the original TrainableSentenceEmbedder, ensuring compatibility
    with the geo-localization pipeline.

    Args:
        model: Trained SentenceTransformer model
        output_path: Path to save the model
        training_args: Dictionary of training arguments for metadata
    """
    logger.info(f"Saving model for pipeline compatibility to {output_path}")

    # Create a TrainableSentenceEmbedder config matching the trained model
    # We need to extract the base model name from the SentenceTransformer
    base_model_name = training_args.get("base_model", "sentence-transformers/all-MiniLM-L6-v2")
    max_seq_length = model.get_max_seq_length()

    # Create config
    model_config = TrainableSentenceEmbedderConfig(
        pretrained_model_name_or_path=base_model_name,
        max_sequence_length=max_seq_length,
        freeze_weights=False,  # Weights are already trained
        model_weights_path=None,
    )

    # Create a new TrainableSentenceEmbedder instance
    pipeline_model = TrainableSentenceEmbedder(model_config)

    # Copy weights from SentenceTransformer to TrainableSentenceEmbedder
    # The SentenceTransformer architecture is: [Transformer, Pooling, (optional) Dense, Normalize]
    # TrainableSentenceEmbedder has: transformer and does pooling/normalization in forward
    # We only copy the transformer weights - no projection layer

    # Copy transformer weights
    st_transformer = model[0]  # First module is the transformer
    pipeline_model.transformer.load_state_dict(st_transformer.auto_model.state_dict())

    logger.info("Copied transformer weights from SentenceTransformer to TrainableSentenceEmbedder")
    logger.info(f"Output dimension: {pipeline_model.output_dim}")

    # Save using common save_model function
    example_texts = ["This is a test sentence."]
    save_model_with_metadata(
        model=pipeline_model,
        save_path=output_path,
        example_model_inputs=(example_texts,),
        aux_information={
            "training_args": training_args,
            "model_type": "TrainableSentenceEmbedder",
            "trained_with_sentence_transformers": True,
            "sentence_embedding_dimension": pipeline_model.output_dim,
        }
    )

    logger.info(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train sentence embedding model using sentence-transformers"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base SentenceTransformer model from HuggingFace",
    )
    parser.add_argument(
        "--dataset_source",
        type=str,
        choices=["correspondence", "huggingface"],
        default="correspondence",
        help="Source of training data: 'correspondence' for custom correspondence files, 'huggingface' for benchmark datasets",
    )
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        choices=["natural-questions", "squad", "all-nli"],
        default="natural-questions",
        help="HuggingFace dataset to use when --dataset_source=huggingface (default: natural-questions)",
    )
    parser.add_argument(
        "--hf_dataset_size",
        type=int,
        default=None,
        help="Optional size to trim HuggingFace dataset to (default: use full dataset)",
    )
    parser.add_argument(
        "--train_correspondence_file",
        type=str,
        default=None,
        help="Path to training correspondence JSON file (required when --dataset_source=correspondence)",
    )
    parser.add_argument(
        "--val_correspondence_file",
        type=str,
        required=True,
        help="Path to validation correspondence JSON file (e.g., /tmp/minimal_historical_seattle.json for Seattle)",
    )
    parser.add_argument(
        "--train_sentence_directory",
        type=str,
        default=None,
        help="Optional path to directory with natural language descriptions for training",
    )
    parser.add_argument(
        "--val_sentence_directory",
        type=str,
        default=None,
        help="Optional path to directory with natural language descriptions for validation",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for trained model and logs",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (fraction of total steps)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--use_natural_language",
        action="store_true",
        help="Use natural language descriptions from sentence directories",
    )
    parser.add_argument(
        "--freeze_transformer",
        action="store_true",
        help="Freeze transformer weights (only train final layers)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training (FP16)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use mixed precision training (BF16)",
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.95,
        help="Fraction of training data to use for training (rest for validation subset)",
    )

    args = parser.parse_args()

    # Validate arguments based on dataset source
    if args.dataset_source == "correspondence":
        if not args.train_correspondence_file:
            parser.error("--train_correspondence_file is required when --dataset_source=correspondence")
    elif args.dataset_source == "huggingface":
        logger.info(f"Using HuggingFace dataset: {args.hf_dataset_name}")
        if args.hf_dataset_size:
            logger.info(f"Will trim to {args.hf_dataset_size} examples")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Save training arguments
    with open(os.path.join(args.output, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info(f"Loading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model)#,
                                # model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
                                # tokenizer_kwargs={"padding_side": "left"},)

    # Freeze transformer if requested
    if args.freeze_transformer:
        logger.info("Freezing transformer weights")
        # The first module is typically the transformer
        if len(model) > 0 and hasattr(model[0], 'auto_model'):
            model[0].auto_model.requires_grad_(False)
            logger.info(f"Frozen transformer: {model[0].__class__.__name__}")
        else:
            logger.warning("Could not find transformer module to freeze")

    # Load training dataset based on source
    if args.dataset_source == "correspondence":
        # Create full training dataset from correspondence files
        train_sentence_dir = Path(args.train_sentence_directory) if args.train_sentence_directory else None
        full_train_dataset = create_training_dataset(
            correspondence_file=Path(args.train_correspondence_file),
            sentence_directory=train_sentence_dir,
            use_natural_language=args.use_natural_language,
        )
    else:  # huggingface
        # Load HuggingFace benchmark dataset
        full_train_dataset = load_huggingface_dataset(
            dataset_name=args.hf_dataset_name,
            target_size=args.hf_dataset_size,
            seed=args.seed,
        )

    # Split training dataset into train and train_validation subsets
    train_val_split_idx = int(len(full_train_dataset) * args.train_val_split)
    train_dataset = full_train_dataset.select(range(train_val_split_idx))
    train_val_dataset = full_train_dataset.select(range(train_val_split_idx, len(full_train_dataset)))

    logger.info(f"Split training data: {len(train_dataset)} train, {len(train_val_dataset)} train_validation")

    # Create evaluator for training validation subset (Chicago)
    # Convert train_val_dataset back to correspondence format for evaluator
    train_val_queries = {}
    train_val_corpus = {}
    train_val_relevant_docs = {}

    # Build unique sets for queries and corpus
    anchor_to_id = {}
    positive_to_id = {}

    for idx, example in enumerate(train_val_dataset):
        anchor = example["anchor"]
        positive = example["positive"]

        # Add anchor to queries if not seen
        if anchor not in anchor_to_id:
            anchor_id = f"train_anchor_{len(anchor_to_id)}"
            anchor_to_id[anchor] = anchor_id
            train_val_queries[anchor_id] = anchor

        # Add positive to corpus if not seen
        if positive not in positive_to_id:
            positive_id = f"train_positive_{len(positive_to_id)}"
            positive_to_id[positive] = positive_id
            train_val_corpus[positive_id] = positive

        # Record correspondence
        anchor_id = anchor_to_id[anchor]
        positive_id = positive_to_id[positive]

        if anchor_id not in train_val_relevant_docs:
            train_val_relevant_docs[anchor_id] = set()
        train_val_relevant_docs[anchor_id].add(positive_id)

    logger.info(f"Train validation set: {len(train_val_queries)} queries, {len(train_val_corpus)} corpus items")

    # Use consistent names for TensorBoard plotting across different datasets
    train_val_evaluator = InformationRetrievalEvaluator(
        queries=train_val_queries,
        corpus=train_val_corpus,
        relevant_docs=train_val_relevant_docs,
        name="train_val",  # Consistent name for training subset validation
        show_progress_bar=True,
    )

    # Create validation evaluator for Seattle test set
    val_sentence_dir = Path(args.val_sentence_directory) if args.val_sentence_directory else None
    seattle_evaluator = create_information_retrieval_evaluator(
        correspondence_file=Path(args.val_correspondence_file),
        sentence_directory=val_sentence_dir,
        use_natural_language=args.use_natural_language,
        name="test",  # Consistent name for test set
    )

    # Combine evaluators using SequentialEvaluator
    evaluator = SequentialEvaluator([train_val_evaluator, seattle_evaluator])

    # Create loss function
    # CachedMultipleNegativesRankingLoss uses in-batch negatives with caching
    # for better performance
    loss = losses.CachedMultipleNegativesRankingLoss(model=model)

    # Setup training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # Required for in-batch negatives
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="test_cosine_ndcg@10",  # Use NDCG@10 on test set as metric
        greater_is_better=True,
        seed=args.seed,
        # TensorBoard logging is automatic
        report_to="tensorboard",
    )

    # Create trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    # Log training info
    logger.info("=" * 80)
    logger.info("Training Configuration:")
    logger.info(f"  Base model: {args.base_model}")
    logger.info(f"  Dataset source: {args.dataset_source}")
    if args.dataset_source == "huggingface":
        logger.info(f"  HuggingFace dataset: {args.hf_dataset_name}")
        if args.hf_dataset_size:
            logger.info(f"  Dataset size (trimmed): {args.hf_dataset_size}")
    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Train validation samples: {len(train_val_dataset)}")
    logger.info(f"  Test validation queries: {len(seattle_evaluator.queries) if seattle_evaluator else 0}")
    logger.info(f"  Epochs: {args.num_epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Warmup ratio: {args.warmup_ratio}")
    logger.info(f"  Train/val split: {args.train_val_split:.2%}")
    logger.info(f"  Freeze transformer: {args.freeze_transformer}")
    logger.info(f"  Output directory: {args.output}")
    logger.info(f"  TensorBoard logs: {args.output}/runs")
    logger.info("=" * 80)

    # Train!
    trainer.train()

    # Save final model in sentence-transformers format
    final_model_path = os.path.join(args.output, "final_sentence_transformer")
    model.save(final_model_path)
    logger.info(f"Saved final model (sentence-transformers format) to {final_model_path}")

    # Save model in format compatible with pipeline
    pipeline_model_path = Path(args.output) / "final_pipeline_compatible"
    save_model_for_pipeline(
        model=model,
        output_path=pipeline_model_path,
        training_args=vars(args),
    )

    # Also save best model in pipeline format if different from final
    if training_args.load_best_model_at_end:
        best_pipeline_path = Path(args.output) / "best_pipeline_compatible"
        save_model_for_pipeline(
            model=model,
            output_path=best_pipeline_path,
            training_args=vars(args),
        )
        logger.info(f"Saved best model (pipeline compatible) to {best_pipeline_path}")

    logger.info("Training completed!")
    logger.info(f"View training logs with: tensorboard --logdir {args.output}/runs")


if __name__ == "__main__":
    main()
