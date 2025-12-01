"""Generate embeddings from a trained sentence embedding model.

This script takes a trained sentence embedding model and generates embeddings for all
sentences in a given directory structure. It supports both OSM landmark sentences and
panorama sentences, automatically detecting the input type.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:generate_embeddings_from_model -- \
        --model_checkpoint /path/to/checkpoint/dir \
        --parent_dir /data/overhead_matching/datasets/semantic_landmark_extractor/pano_v1 \
        --output_dir /tmp/embeddings_output \
        --batch_size 128
"""

import argparse
import pickle
import shutil
from pathlib import Path
from typing import Optional

import common.torch.load_torch_deps
import torch
from tqdm import tqdm

from common.torch.load_and_save_models import load_model
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    load_all_jsonl_from_folder,
    make_sentence_dict_from_json,
    make_sentence_dict_from_pano_jsons,
)
# Import model classes for unpickling
from experimental.overhead_matching.swag.model.trainable_sentence_embedder import TrainableSentenceEmbedder
from experimental.overhead_matching.swag.model.swag_config_types import TrainableSentenceEmbedderConfig


def detect_directory_structure(parent_dir: Path) -> tuple[str, list[Path]]:
    """Detect the directory structure type and return sentence directories.

    Args:
        parent_dir: Parent directory to analyze

    Returns:
        Tuple of (structure_type, sentence_directories) where:
        - structure_type: 'panorama_multi_city', 'panorama_flat', or 'osm_flat'
        - sentence_directories: List of paths to sentences/ directories
    """
    # Check for multi-city structure (Chicago/, Seattle/, etc.)
    city_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]
    city_sentence_dirs = []

    for city_dir in city_dirs:
        sentence_dir = city_dir / "sentences"
        if sentence_dir.exists() and sentence_dir.is_dir():
            city_sentence_dirs.append(sentence_dir)

    if city_sentence_dirs:
        print(f"Detected multi-city panorama structure with {len(city_sentence_dirs)} cities")
        return "panorama_multi_city", city_sentence_dirs

    # Check for flat structure with sentences/ directly in parent
    direct_sentence_dir = parent_dir / "sentences"
    if direct_sentence_dir.exists() and direct_sentence_dir.is_dir():
        # Try to determine if it's panorama or OSM by checking file content
        sample_files = list(direct_sentence_dir.glob("*.jsonl"))[:1]
        if sample_files:
            with open(sample_files[0], 'r') as f:
                import json
                first_line = json.loads(f.readline())
                # Panorama files have JSON content with "landmarks" key
                if first_line.get("response", {}).get("body", {}).get("choices"):
                    content = first_line["response"]["body"]["choices"][0]["message"]["content"]
                    try:
                        json.loads(content)
                        print("Detected flat panorama structure")
                        return "panorama_flat", [direct_sentence_dir]
                    except:
                        pass

        print("Detected flat OSM structure")
        return "osm_flat", [direct_sentence_dir]

    raise ValueError(f"No sentences/ directories found in {parent_dir}")


def load_sentences_from_directories(
    sentence_dirs: list[Path],
    structure_type: str
) -> tuple[dict[str, str], Optional[dict]]:
    """Load all sentences from the given directories.

    Args:
        sentence_dirs: List of paths to sentences/ directories
        structure_type: Type of directory structure

    Returns:
        Tuple of (sentences_dict, panorama_metadata) where:
        - sentences_dict: Maps custom_id to sentence text
        - panorama_metadata: For panoramas, maps pano_id to landmark metadata list
    """
    all_sentences = {}
    all_metadata = {} if structure_type.startswith("panorama") else None

    for sentence_dir in sentence_dirs:
        print(f"Loading sentences from {sentence_dir}")
        sentence_jsons = load_all_jsonl_from_folder(sentence_dir)

        if structure_type.startswith("panorama"):
            sentences, metadata, _ = make_sentence_dict_from_pano_jsons(sentence_jsons)
            all_sentences.update(sentences)
            if all_metadata is not None:
                all_metadata.update(metadata)
        else:
            sentences, _ = make_sentence_dict_from_json(sentence_jsons)
            all_sentences.update(sentences)

    print(f"Loaded {len(all_sentences)} sentences total")
    return all_sentences, all_metadata


def generate_embeddings_in_batches(
    model: torch.nn.Module,
    sentences_dict: dict[str, str],
    batch_size: int,
    device: str
) -> tuple[torch.Tensor, dict[str, int]]:
    """Generate embeddings for all sentences using the model.

    Args:
        model: Trained sentence embedding model
        sentences_dict: Dictionary mapping custom_id to sentence text
        batch_size: Number of sentences to process at once
        device: Device to run model on ('cuda' or 'cpu')

    Returns:
        Tuple of (embeddings_tensor, custom_id_to_idx) where:
        - embeddings_tensor: Tensor of shape [num_sentences, embedding_dim]
        - custom_id_to_idx: Dictionary mapping custom_id to index in tensor
    """
    model.eval()
    model.to(device)

    # Sort custom IDs for deterministic ordering
    custom_ids = sorted(sentences_dict.keys())
    custom_id_to_idx = {custom_id: idx for idx, custom_id in enumerate(custom_ids)}

    # Process in batches
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(custom_ids), batch_size), desc="Generating embeddings"):
            batch_ids = custom_ids[i:i + batch_size]
            batch_texts = [sentences_dict[custom_id] for custom_id in batch_ids]

            # Generate embeddings
            embeddings = model(batch_texts)
            all_embeddings.append(embeddings.cpu())

    # Concatenate all batches
    embeddings_tensor = torch.cat(all_embeddings, dim=0)

    print(f"Generated embeddings with shape {embeddings_tensor.shape}")
    return embeddings_tensor, custom_id_to_idx


def save_outputs(
    output_dir: Path,
    embeddings_tensor: torch.Tensor,
    custom_id_to_idx: dict[str, int],
    panorama_metadata: Optional[dict],
    structure_type: str,
    sentence_dirs: list[Path]
) -> None:
    """Save embeddings and metadata to output directory.

    Args:
        output_dir: Base output directory
        embeddings_tensor: Tensor of embeddings
        custom_id_to_idx: Mapping from custom_id to index
        panorama_metadata: Optional panorama metadata by pano_id
        structure_type: Type of directory structure
        sentence_dirs: Original sentence directories (for metadata copying)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if structure_type == "panorama_multi_city":
        # Create per-city output directories
        for sentence_dir in sentence_dirs:
            city_name = sentence_dir.parent.name
            city_output = output_dir / city_name
            embeddings_output = city_output / "embeddings"
            embeddings_output.mkdir(parents=True, exist_ok=True)

            # Filter embeddings for this city
            # Panorama custom IDs are formatted as "{pano_id}__landmark_{idx}"
            # We need to filter by pano_ids that belong to this city
            city_custom_ids = [cid for cid in custom_id_to_idx.keys()
                              if cid.split("__")[0] in panorama_metadata]

            # For multi-city, we actually want to save all embeddings in each city
            # because the model sees all cities. Let's save the full pkl in each city.
            pkl_path = embeddings_output / "embeddings.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump((embeddings_tensor, custom_id_to_idx), f)
            print(f"Saved embeddings.pkl to {pkl_path}")

            # Copy panorama_metadata.jsonl if it exists
            metadata_src = sentence_dir.parent / "embedding_requests" / "panorama_metadata.jsonl"
            if metadata_src.exists():
                embedding_requests_output = city_output / "embedding_requests"
                embedding_requests_output.mkdir(parents=True, exist_ok=True)
                metadata_dst = embedding_requests_output / "panorama_metadata.jsonl"
                shutil.copy2(metadata_src, metadata_dst)
                print(f"Copied panorama_metadata.jsonl to {metadata_dst}")

    else:
        # Flat structure - save directly in output_dir
        embeddings_output = output_dir / "embeddings"
        embeddings_output.mkdir(parents=True, exist_ok=True)

        pkl_path = embeddings_output / "embeddings.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump((embeddings_tensor, custom_id_to_idx), f)
        print(f"Saved embeddings.pkl to {pkl_path}")

        # For panorama flat, copy metadata if it exists
        if structure_type == "panorama_flat":
            metadata_src = sentence_dirs[0].parent / "embedding_requests" / "panorama_metadata.jsonl"
            if metadata_src.exists():
                embedding_requests_output = output_dir / "embedding_requests"
                embedding_requests_output.mkdir(parents=True, exist_ok=True)
                metadata_dst = embedding_requests_output / "panorama_metadata.jsonl"
                shutil.copy2(metadata_src, metadata_dst)
                print(f"Copied panorama_metadata.jsonl to {metadata_dst}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings from a trained sentence embedding model"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory (saved using common.torch.save_model)"
    )
    parser.add_argument(
        "--parent_dir",
        type=str,
        required=True,
        help="Parent directory containing sentences/ subdirectory (or city subdirectories for multi-city)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for embeddings/ and embedding_requests/"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for processing sentences (default: 128)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on (default: cuda if available, else cpu)"
    )

    args = parser.parse_args()

    # Convert paths
    model_checkpoint = Path(args.model_checkpoint).expanduser()
    parent_dir = Path(args.parent_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    # Validate inputs
    if not model_checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")
    if not parent_dir.exists():
        raise FileNotFoundError(f"Parent directory not found: {parent_dir}")

    # Load model
    print(f"Loading model from {model_checkpoint}")
    model = load_model(model_checkpoint, device=args.device)
    print(f"Model loaded successfully on {args.device}")

    # Detect directory structure
    structure_type, sentence_dirs = detect_directory_structure(parent_dir)

    # Load sentences
    sentences_dict, panorama_metadata = load_sentences_from_directories(
        sentence_dirs, structure_type
    )

    # Generate embeddings
    embeddings_tensor, custom_id_to_idx = generate_embeddings_in_batches(
        model, sentences_dict, args.batch_size, args.device
    )

    # Save outputs
    save_outputs(
        output_dir,
        embeddings_tensor,
        custom_id_to_idx,
        panorama_metadata,
        structure_type,
        sentence_dirs
    )

    print(f"\nSuccessfully generated embeddings!")
    print(f"Total sentences processed: {len(sentences_dict)}")
    print(f"Embedding dimension: {embeddings_tensor.shape[1]}")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
