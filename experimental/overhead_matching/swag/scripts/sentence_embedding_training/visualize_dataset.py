"""
Visualize anchor/positive pairs from training datasets.

This script helps you inspect the quality of your training data by displaying
random samples of (anchor, positive) pairs from either correspondence files
or HuggingFace benchmark datasets.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict

from datasets import load_dataset
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    load_all_jsonl_from_folder,
    make_sentence_dict_from_json,
    custom_id_from_props
)


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

    print(f"Loading sentences from {sentence_directory}")
    # Import here to avoid dependency issues when not needed

    sentence_jsons = load_all_jsonl_from_folder(sentence_directory)
    sentence_dict, _ = make_sentence_dict_from_json(sentence_jsons)
    print(f"Loaded {len(sentence_dict)} sentences")
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

    # Fall back to tag format: "key: value, key: value, ..."
    parts = [f"{k}: {v}" for k, v in sorted(pruned_props.items())]
    return ", ".join(parts) if parts else ""


def visualize_correspondence_dataset(
    correspondence_file: Path,
    sentence_directory: Path | None = None,
    use_natural_language: bool = False,
    num_samples: int = 20,
    seed: int = 42,
):
    """Visualize samples from correspondence dataset."""
    print(f"\n{'='*80}")
    print(f"Correspondence Dataset: {correspondence_file.name}")
    if use_natural_language:
        print(f"Natural Language Mode: Enabled")
        if sentence_directory:
            print(f"Sentence Directory: {sentence_directory}")
    else:
        print(f"Natural Language Mode: Disabled (showing OSM tags)")
    print(f"{'='*80}\n")

    with open(correspondence_file, 'r') as f:
        corr_data = json.load(f)

    # Load sentence dictionary if using natural language
    sentence_dict = load_sentence_dict(sentence_directory) if use_natural_language else {}
    missing_sentences = 0

    # Extract all (anchor, positive) pairs
    pairs = []
    for entry_id, entry in corr_data.items():
        pano_descs = entry["pano"]
        osm_items = entry["osm"]
        matches = entry["matches"]["matches"]

        for match in matches:
            pano_idx = match["set_1_id"] - 1
            osm_indices = [x-1 for x in match["set_2_matches"]]

            if pano_idx >= len(pano_descs):
                continue

            # Handle different pano formats
            pano_desc = pano_descs[pano_idx]
            if isinstance(pano_desc, dict):
                pano_text = pano_desc.get('sentence', '')
            else:
                pano_text = pano_desc

            if not pano_text:
                continue

            for osm_idx in osm_indices:
                if osm_idx >= len(osm_items):
                    continue

                osm_item = osm_items[osm_idx]
                osm_text = process_osm_item(osm_item, sentence_dict, use_natural_language)

                if not osm_text:
                    continue

                # Track missing sentences
                if use_natural_language and sentence_dict and "tags" in osm_item:
                    props = parse_osm_tags(osm_item["tags"])
                    custom_id = custom_id_from_props(props)
                    if custom_id not in sentence_dict:
                        missing_sentences += 1

                pairs.append((pano_text, osm_text, entry_id))

    print(f"Total pairs in dataset: {len(pairs)}")
    if missing_sentences > 0 and use_natural_language:
        print(f"Warning: {missing_sentences} OSM items missing natural language sentences (using tag format as fallback)")
    print()

    # Sample random pairs
    random.seed(seed)
    samples = random.sample(pairs, min(num_samples, len(pairs)))

    for i, (anchor, positive, entry_id) in enumerate(samples, 1):
        print(f"Sample {i}/{len(samples)} (from entry {entry_id}):")
        print(f"  Anchor (pano):   {anchor}")
        print(f"  Positive (OSM):  {positive}")
        print()


def visualize_huggingface_dataset(
    dataset_name: str,
    num_samples: int = 20,
    seed: int = 42,
):
    """Visualize samples from HuggingFace dataset."""
    print(f"\n{'='*80}")
    print(f"HuggingFace Dataset: {dataset_name}")
    print(f"{'='*80}\n")

    # Map dataset names to their HuggingFace paths
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
            f"Available: {list(dataset_configs.keys())}"
        )

    config = dataset_configs[dataset_name]
    print(f"Loading dataset from: {config['path']}")

    # Load dataset
    dataset = load_dataset(config["path"], split="train")
    print(f"Total pairs in dataset: {len(dataset)}\n")

    # Sample random pairs
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    anchor_col = config["anchor_col"]
    positive_col = config["positive_col"]

    for i, idx in enumerate(indices, 1):
        example = dataset[idx]
        anchor = example[anchor_col]
        positive = example[positive_col]

        print(f"Sample {i}/{len(indices)}:")
        print(f"  Anchor:   {anchor}")
        print(f"  Positive: {positive}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize anchor/positive pairs from training datasets"
    )
    parser.add_argument(
        "--dataset_source",
        type=str,
        choices=["correspondence", "huggingface"],
        required=True,
        help="Source of data to visualize",
    )
    parser.add_argument(
        "--correspondence_file",
        type=str,
        default=None,
        help="Path to correspondence JSON file (required when --dataset_source=correspondence)",
    )
    parser.add_argument(
        "--sentence_directory",
        type=str,
        default=None,
        help="Optional path to directory with natural language descriptions for OSM items",
    )
    parser.add_argument(
        "--use_natural_language",
        action="store_true",
        help="Use natural language descriptions from sentence directory for OSM items",
    )
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        choices=["natural-questions", "squad", "all-nli"],
        default="natural-questions",
        help="HuggingFace dataset to visualize (when --dataset_source=huggingface)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of random samples to display",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.dataset_source == "correspondence":
        if not args.correspondence_file:
            parser.error("--correspondence_file is required when --dataset_source=correspondence")

        sentence_dir = Path(args.sentence_directory) if args.sentence_directory else None
        visualize_correspondence_dataset(
            correspondence_file=Path(args.correspondence_file),
            sentence_directory=sentence_dir,
            use_natural_language=args.use_natural_language,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    else:  # huggingface
        visualize_huggingface_dataset(
            dataset_name=args.hf_dataset_name,
            num_samples=args.num_samples,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
