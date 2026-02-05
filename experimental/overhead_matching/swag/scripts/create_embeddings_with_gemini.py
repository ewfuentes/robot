#!/usr/bin/env python3
"""Process batch job outputs and create Vertex AI embeddings.

Supports two modes:
1. Panorama Mode: Process Vertex AI Gemini batch outputs with structured JSON
2. OSM Mode: Process OpenAI batch outputs with plain text descriptions

Panorama mode stores in hierarchical pickle format (v2.0).
OSM mode stores in flat tuple format compatible with existing infrastructure.

Usage:
    # Panorama mode
    bazel run //experimental/overhead_matching/swag/scripts:create_embeddings_with_gemini -- \
      --mode panorama \
      --input_dir ~/Downloads/ \
      --output_file /path/to/embeddings.pkl

    # OSM mode
    bazel run //experimental/overhead_matching/swag/scripts:create_embeddings_with_gemini -- \
      --mode osm \
      --input_dir /data/.../sentences/ \
      --output_file /path/to/embeddings_gemini.pkl

Prerequisites:
    export GOOGLE_CLOUD_PROJECT=your-project-id
    gcloud auth application-default login
"""

import argparse
import json
import pickle
import sys
import time
import warnings
from collections import Counter
from typing import Optional
from pathlib import Path

# IMPORTANT: Must import torch deps before torch to load CUDA libraries
import common.torch.load_torch_deps
import torch
from tqdm import tqdm

from google import genai
from google.genai.types import EmbedContentConfig


class BatchOutputCollector:
    """Recursively scan directory and collect panorama data from JSONL files."""

    def __init__(self, input_dir: Path, recursive: bool = True):
        self.input_dir = Path(input_dir)
        self.recursive = recursive
        self.panoramas = {}  # pano_id → panorama data
        self.errors = []
        self.files_processed = 0

    def collect(self) -> dict:
        """Scan directory and parse all JSONL files."""
        pattern = "**/*.jsonl" if self.recursive else "*.jsonl"
        jsonl_files = list(self.input_dir.glob(pattern))

        if not jsonl_files:
            print(f"Warning: No JSONL files found in {self.input_dir}")
            return self.panoramas

        print(f"Found {len(jsonl_files)} JSONL file(s) to process")

        for file_path in tqdm(jsonl_files, desc="Scanning JSONL files"):
            self._process_file(file_path)

        if self.errors:
            print(f"\nWarning: {len(self.errors)} error(s) encountered during parsing")
            print("First 5 errors:")
            for error in self.errors[:5]:
                print(f"  {error}")

        return self.panoramas

    def _process_file(self, file_path: Path):
        """Parse a single JSONL file."""
        self.files_processed += 1

        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    self._extract_panorama(data)
                except json.JSONDecodeError as e:
                    self.errors.append(f"{file_path}:{line_num}: JSON decode error: {e}")
                except KeyError as e:
                    self.errors.append(f"{file_path}:{line_num}: Missing key: {e}")
                except Exception as e:
                    self.errors.append(f"{file_path}:{line_num}: {type(e).__name__}: {e}")

    def _extract_panorama(self, data: dict):
        """Extract panorama data from batch output JSON."""
        pano_id = data["key"]

        # Parse response text as JSON
        response = data.get("response", {})
        if not response:
            raise ValueError("No response field")

        candidates = response.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates in response")

        text_content = candidates[0]["content"]["parts"][0]["text"]
        parsed = json.loads(text_content)

        # Validate landmarks field exists
        if "landmarks" not in parsed:
            raise KeyError("'landmarks' key missing from response")
        landmarks = parsed["landmarks"]
        if not landmarks:
            warnings.warn(f"Panorama {pano_id} has empty landmarks list")

        # Add landmark_idx to each landmark if not present
        for idx, landmark in enumerate(landmarks):
            if "landmark_idx" not in landmark:
                landmark["landmark_idx"] = idx

        self.panoramas[pano_id] = {
            "location_type": parsed["location_type"],
            "landmarks": landmarks
        }


class StatisticsReporter:
    """Generate statistics and validate data."""

    def __init__(self, panoramas: dict):
        self.panoramas = panoramas

    def report(self):
        """Print comprehensive statistics."""
        total_landmarks = sum(len(p["landmarks"]) for p in self.panoramas.values())
        total_chars = sum(
            len(lm["description"])
            for p in self.panoramas.values()
            for lm in p["landmarks"]
        )

        # Collect unique location_types
        location_types = Counter(p["location_type"] for p in self.panoramas.values())

        # Collect unique proper_nouns
        all_nouns = [
            noun
            for p in self.panoramas.values()
            for lm in p["landmarks"]
            for noun in lm.get("proper_nouns", [])
        ]
        proper_nouns = Counter(all_nouns)

        # Validate yaw angles
        yaw_errors = self._validate_yaw_angles()

        # Print report
        print("\n" + "="*80)
        print("BATCH OUTPUT STATISTICS")
        print("="*80)
        print(f"Total panoramas: {len(self.panoramas):,}")
        print(f"Total landmarks: {total_landmarks:,}")
        print(f"Total characters in descriptions: {total_chars:,}")
        print(f"\nUnique location types: {len(location_types)}")
        for loc_type, count in location_types.most_common(10):
            print(f"  {loc_type}: {count}")
        if len(location_types) > 10:
            print(f"  ... and {len(location_types) - 10} more")

        print(f"\nUnique proper nouns: {len(proper_nouns):,}")
        for noun, count in proper_nouns.most_common(20):
            print(f"  {noun}: {count}")
        if len(proper_nouns) > 20:
            print(f"  ... and {len(proper_nouns) - 20:,} more")

        print(f"\nYaw validation errors: {len(yaw_errors)}")
        if yaw_errors:
            print("First 10 errors:")
            for error in yaw_errors[:10]:
                print(f"  {error}")
            if len(yaw_errors) > 10:
                print(f"  ... and {len(yaw_errors) - 10} more")

        print("="*80)

        return yaw_errors

    def _validate_yaw_angles(self) -> list[str]:
        """Validate all yaw_angle strings are in {"0", "90", "180", "270"}."""
        errors = []
        valid_yaws = {"0", "90", "180", "270"}

        for pano_id, pano_data in self.panoramas.items():
            for lm_idx, landmark in enumerate(pano_data["landmarks"]):
                for bbox in landmark.get("bounding_boxes", []):
                    yaw = bbox.get("yaw_angle")
                    if yaw not in valid_yaws:
                        errors.append(
                            f"{pano_id}, landmark {lm_idx}: "
                            f"invalid yaw '{yaw}' (expected {valid_yaws})"
                        )

        return errors


class VertexEmbeddingGenerator:
    """Generate embeddings using Vertex AI gemini-embedding-001."""

    def __init__(self, model: str = "gemini-embedding-001",
                 task_type: str = "SEMANTIC_SIMILARITY",
                 batch_size: int = 250,
                 output_dimensionality: int = 1536):
        self.model = model
        self.task_type = task_type
        self.batch_size = min(batch_size, 250)  # Max 250 per Vertex AI request
        self.output_dimensionality = output_dimensionality
        self.genai_client = genai.Client(vertexai=True)

    def embed_texts(self, texts: list[str], desc: str = "Embedding") -> torch.Tensor:
        """Embed a list of texts, handling batching and retries."""
        if not texts:
            # Return empty tensor with correct shape
            return torch.zeros((0, self.output_dimensionality), dtype=torch.float32)

        all_embeddings = []

        # Process in batches
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(texts), self.batch_size),
                     desc=desc,
                     total=num_batches):
            batch = texts[i:i+self.batch_size]
            batch_embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        # Convert to tensor
        return torch.tensor(all_embeddings, dtype=torch.float32)

    def _embed_batch_with_retry(self, batch: list[str], max_retries: int = 3) -> list:
        """Embed a batch with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                config = EmbedContentConfig(
                    task_type=self.task_type,
                    output_dimensionality=self.output_dimensionality
                )
                response = self.genai_client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=config
                )
                return [emb.values for emb in response.embeddings]
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\nFailed to embed batch after {max_retries} attempts: {e}")
                    raise
                wait_time = 2 ** attempt
                print(f"\nEmbedding failed (attempt {attempt+1}/{max_retries}), "
                      f"retrying in {wait_time}s: {e}")
                time.sleep(wait_time)


class HierarchicalEmbeddingStorage:
    """Build and save hierarchical pickle format."""

    def __init__(self, panoramas: dict, generator: Optional[VertexEmbeddingGenerator] = None):
        self.panoramas = panoramas
        self.generator = generator

    def build_and_save(self, output_file: Path):
        """Generate all embeddings and save to pickle."""
        if self.generator is None:
            raise ValueError("VertexEmbeddingGenerator required for embedding generation")

        # Collect all texts to embed
        print("\nCollecting texts for embedding...")
        descriptions, desc_ids = self._collect_descriptions()
        location_types, loc_type_to_panos = self._collect_location_types()
        proper_nouns, noun_to_landmarks = self._collect_proper_nouns()

        print(f"  Descriptions: {len(descriptions):,}")
        print(f"  Unique location types: {len(location_types)}")
        print(f"  Unique proper nouns: {len(proper_nouns):,}")

        # Generate embeddings
        print("\nGenerating embeddings with Vertex AI...")
        desc_embeddings = self.generator.embed_texts(descriptions, "Descriptions")
        loc_embeddings = self.generator.embed_texts(location_types, "Location types")
        noun_embeddings = self.generator.embed_texts(proper_nouns, "Proper nouns")

        print(f"\nEmbedding dimensions:")
        print(f"  Descriptions: {desc_embeddings.shape}")
        print(f"  Location types: {loc_embeddings.shape}")
        print(f"  Proper nouns: {noun_embeddings.shape}")

        # Build index mappings
        desc_id_to_idx = {desc_id: i for i, desc_id in enumerate(desc_ids)}
        loc_type_to_idx = {loc_type: i for i, loc_type in enumerate(location_types)}
        noun_to_idx = {noun: i for i, noun in enumerate(proper_nouns)}

        # Build hierarchical structure
        data = {
            "version": "2.0",
            "embedding_model": self.generator.model,
            "embedding_dim": desc_embeddings.shape[1],
            "task_type": self.generator.task_type,
            "panoramas": self.panoramas,
            "description_embeddings": desc_embeddings,
            "description_id_to_idx": desc_id_to_idx,
            "location_type_embeddings": loc_embeddings,
            "location_type_to_idx": loc_type_to_idx,
            "proper_noun_embeddings": noun_embeddings,
            "proper_noun_to_idx": noun_to_idx,
        }

        # Save pickle
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

        # Report
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        total_embeddings = len(desc_ids) + len(location_types) + len(proper_nouns)

        print("\n" + "="*80)
        print("EMBEDDING GENERATION COMPLETE")
        print("="*80)
        print(f"Output file: {output_file}")
        print(f"File size: {file_size_mb:.1f} MB")
        print(f"Total embeddings: {total_embeddings:,}")
        print(f"  Descriptions: {len(desc_ids):,}")
        print(f"  Location types: {len(location_types)}")
        print(f"  Proper nouns: {len(proper_nouns):,}")
        print("="*80)

    def _collect_descriptions(self) -> tuple[list[str], list[str]]:
        """Collect all landmark descriptions with custom IDs."""
        descriptions = []
        desc_ids = []

        for pano_id, pano_data in sorted(self.panoramas.items()):
            for landmark in pano_data["landmarks"]:
                landmark_idx = landmark["landmark_idx"]
                custom_id = f"{pano_id}__landmark_{landmark_idx}"
                descriptions.append(landmark["description"])
                desc_ids.append(custom_id)

        return descriptions, desc_ids

    def _collect_location_types(self) -> tuple[list[str], dict]:
        """Collect unique location types."""
        location_types = list(set(p["location_type"] for p in self.panoramas.values()))
        return sorted(location_types), {}

    def _collect_proper_nouns(self) -> tuple[list[str], dict]:
        """Collect unique proper nouns."""
        all_nouns = set()
        for pano_data in self.panoramas.values():
            for landmark in pano_data["landmarks"]:
                all_nouns.update(landmark.get("proper_nouns", []))
        return sorted(all_nouns), {}


class OSMLandmarkCollector:
    """Scan directory and collect OSM landmark descriptions from OpenAI batch output JSONL files."""

    def __init__(self, input_dir: Path):
        self.input_dir = Path(input_dir)
        self.landmarks = {}  # custom_id → description
        self.errors = []
        self.files_processed = 0

    def collect(self) -> dict:
        """Scan directory and parse all batch output JSONL files."""
        jsonl_files = list(self.input_dir.glob("*.jsonl"))

        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found in {self.input_dir}")

        print(f"Found {len(jsonl_files)} JSONL file(s) to process")

        for file_path in tqdm(jsonl_files, desc="Scanning JSONL files"):
            self._process_file(file_path)

        if self.errors:
            print(f"\nWarning: {len(self.errors)} error(s) encountered during parsing")
            print("First 5 errors:")
            for error in self.errors[:5]:
                print(f"  {error}")

        return self.landmarks

    def _process_file(self, file_path: Path):
        """Parse a single JSONL file."""
        self.files_processed += 1

        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    self._extract_landmark(data)
                except json.JSONDecodeError as e:
                    self.errors.append(f"{file_path}:{line_num}: JSON decode error: {e}")
                except Exception as e:
                    self.errors.append(f"{file_path}:{line_num}: {type(e).__name__}: {e}")

    def _extract_landmark(self, data: dict):
        """Extract OSM landmark data from batch output JSON."""
        # Skip if error is present
        if data.get("error"):
            self.errors.append(f"Entry has error field: {data.get('error')}")
            return

        # Extract custom_id
        custom_id = data.get("custom_id")
        if not custom_id:
            self.errors.append("Entry missing custom_id field")
            return

        # Extract description from OpenAI response format
        try:
            response = data["response"]
            status_code = response.get("status_code")

            # Skip non-200 responses
            if status_code != 200:
                self.errors.append(f"Entry {custom_id}: non-200 status code: {status_code}")
                return

            # Extract plain text description
            body = response["body"]
            choices = body["choices"]
            message = choices[0]["message"]
            description = message["content"]

            # Store mapping
            self.landmarks[custom_id] = description

        except (KeyError, IndexError, TypeError) as e:
            self.errors.append(f"Entry {custom_id}: missing expected fields: {e}")


class OSMStatisticsReporter:
    """Generate statistics for OSM landmark data."""

    def __init__(self, landmarks: dict, errors: list):
        self.landmarks = landmarks
        self.errors = errors

    def report(self):
        """Print comprehensive statistics."""
        total_chars = sum(len(desc) for desc in self.landmarks.values())

        print("\n" + "="*80)
        print("OSM LANDMARK STATISTICS")
        print("="*80)
        print(f"Total landmarks: {len(self.landmarks):,}")
        print(f"Total characters in descriptions: {total_chars:,}")
        print(f"Parsing errors: {len(self.errors)}")

        if self.errors:
            print(f"\nFirst 10 errors:")
            for error in self.errors[:10]:
                print(f"  {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")

        # Show sample descriptions
        print(f"\nSample descriptions:")
        for i, (custom_id, desc) in enumerate(list(self.landmarks.items())[:5]):
            print(f"  {custom_id[:20]}...: {desc[:60]}...")

        print("="*80)


class FlatEmbeddingStorage:
    """Build and save flat tuple pickle format for OSM landmarks."""

    def __init__(self, landmarks: dict, generator: Optional[VertexEmbeddingGenerator] = None):
        self.landmarks = landmarks
        self.generator = generator

    def build_and_save(self, output_file: Path):
        """Generate all embeddings and save to pickle in tuple format."""
        if self.generator is None:
            raise ValueError("VertexEmbeddingGenerator required for embedding generation")

        # Collect all texts to embed
        print("\nCollecting texts for embedding...")
        custom_ids = sorted(self.landmarks.keys())  # Sort for deterministic ordering
        descriptions = [self.landmarks[cid] for cid in custom_ids]

        print(f"  Descriptions: {len(descriptions):,}")

        # Generate embeddings
        print("\nGenerating embeddings with Vertex AI...")
        embeddings_tensor = self.generator.embed_texts(descriptions, "OSM Landmarks")

        print(f"\nEmbedding dimensions: {embeddings_tensor.shape}")

        # Build index mapping
        custom_id_to_idx = {cid: i for i, cid in enumerate(custom_ids)}

        # Create tuple format (matching existing OSM pickle format)
        data = (embeddings_tensor, custom_id_to_idx)

        # Save pickle
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

        # Report
        file_size_mb = output_file.stat().st_size / 1024 / 1024

        print("\n" + "="*80)
        print("EMBEDDING GENERATION COMPLETE")
        print("="*80)
        print(f"Output file: {output_file}")
        print(f"File size: {file_size_mb:.1f} MB")
        print(f"Total embeddings: {len(custom_ids):,}")
        print(f"Embedding dimensions: {embeddings_tensor.shape[1]}")
        print("="*80)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Process batch outputs and create Vertex AI embeddings (supports panorama and OSM modes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Panorama mode: Process Vertex AI Gemini batch outputs
  %(prog)s --mode panorama --input_dir ~/Downloads/ --output_file /tmp/embeddings.pkl

  # OSM mode: Process OpenAI batch outputs for OSM landmarks
  %(prog)s --mode osm --input_dir /data/.../sentences/ --output_file /tmp/embeddings_gemini.pkl

  # Only show statistics (no embedding generation)
  %(prog)s --mode panorama --input_dir ~/Downloads/ --stats_only

  # Validate yaw angles only (panorama mode)
  %(prog)s --mode panorama --input_dir ~/Downloads/ --validate_only

Environment variables:
  GOOGLE_CLOUD_PROJECT - GCP project ID (required for Vertex AI)
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['panorama', 'osm'],
        help='Processing mode: "panorama" for Vertex AI Gemini outputs, "osm" for OpenAI batch outputs'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing JSONL batch output files'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Path to output pickle file (required unless --stats_only or --validate_only)'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        default=None,
        help='Scan subdirectories for JSONL files (default: True for panorama, False for osm)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-embedding-001',
        help='Vertex AI embedding model (default: gemini-embedding-001)'
    )
    parser.add_argument(
        '--task_type',
        type=str,
        default='SEMANTIC_SIMILARITY',
        help='Embedding task type (default: SEMANTIC_SIMILARITY)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=250,
        help='Items per embedding request (default: 250, max: 250 for Vertex AI)'
    )
    parser.add_argument(
        '--output_dimensionality',
        type=int,
        default=1536,
        help='Embedding dimensionality (default: 1536 to match OpenAI). Gemini default is 3072.'
    )
    parser.add_argument(
        '--stats_only',
        action='store_true',
        help='Only report statistics, do not create embeddings'
    )
    parser.add_argument(
        '--validate_only',
        action='store_true',
        help='Only validate yaw angles, do not create embeddings'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompts (useful for batch processing)'
    )

    args = parser.parse_args()

    # Set default recursive based on mode
    if args.recursive is None:
        args.recursive = (args.mode == 'panorama')

    # Validate arguments
    if not args.stats_only and not args.validate_only and not args.output_file:
        parser.error("--output_file is required unless using --stats_only or --validate_only")

    # Validate mode-specific flags
    if args.validate_only and args.mode == 'osm':
        parser.error("--validate_only is only available for panorama mode")

    # Route to mode-specific pipeline
    if args.mode == 'panorama':
        return main_panorama(args)
    elif args.mode == 'osm':
        return main_osm(args)
    else:
        parser.error(f"Unknown mode: {args.mode}")


def main_panorama(args):
    """Panorama mode: Process Vertex AI Gemini batch outputs."""
    # Phase 1: Collection & Validation
    print("="*80)
    print("PHASE 1: COLLECTION & VALIDATION (Panorama Mode)")
    print("="*80)

    collector = BatchOutputCollector(Path(args.input_dir), recursive=args.recursive)
    panoramas = collector.collect()

    if not panoramas:
        print("\nNo panoramas found. Exiting.")
        return 0

    print(f"\nCollected {len(panoramas):,} panorama(s) from {collector.files_processed} file(s)")

    # Generate statistics
    reporter = StatisticsReporter(panoramas)
    yaw_errors = reporter.report()

    # Check for validation errors
    if args.validate_only:
        if yaw_errors:
            print(f"\n✗ Validation failed: {len(yaw_errors)} yaw angle error(s)")
            return 1
        else:
            print("\n✓ Validation passed: All yaw angles are valid")
            return 0

    # Exit early if only stats requested
    if args.stats_only:
        return 0

    # Phase 2 & 3: Embedding Generation & Storage
    if yaw_errors and not args.force:
        response = input(f"\nWarning: {len(yaw_errors)} yaw validation error(s) found. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return 1
    elif yaw_errors:
        print(f"\nWarning: {len(yaw_errors)} yaw validation error(s) found. Continuing with --force flag.")

    print("\n" + "="*80)
    print("PHASE 2 & 3: EMBEDDING GENERATION & STORAGE")
    print("="*80)

    try:
        generator = VertexEmbeddingGenerator(
            model=args.model,
            task_type=args.task_type,
            batch_size=args.batch_size,
            output_dimensionality=args.output_dimensionality
        )
    except Exception as e:
        print(f"\nError initializing Vertex AI: {e}")
        print("\nMake sure you have:")
        print("  1. Set GOOGLE_CLOUD_PROJECT environment variable")
        print("  2. Authenticated with: gcloud auth application-default login")
        return 1

    storage = HierarchicalEmbeddingStorage(panoramas, generator)

    try:
        storage.build_and_save(Path(args.output_file))
        print("\n✓ Successfully created embeddings!")
        return 0
    except Exception as e:
        print(f"\n✗ Error creating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main_osm(args):
    """OSM mode: Process OpenAI batch outputs for OSM landmarks."""
    # Phase 1: Collection & Validation
    print("="*80)
    print("PHASE 1: COLLECTION & VALIDATION (OSM Mode)")
    print("="*80)

    collector = OSMLandmarkCollector(Path(args.input_dir))
    landmarks = collector.collect()

    if not landmarks:
        print("\nNo landmarks found. Exiting.")
        return 0

    print(f"\nCollected {len(landmarks):,} landmark(s) from {collector.files_processed} file(s)")

    # Generate statistics
    reporter = OSMStatisticsReporter(landmarks, collector.errors)
    reporter.report()

    # Exit early if only stats requested
    if args.stats_only:
        return 0

    # Phase 2 & 3: Embedding Generation & Storage
    print("\n" + "="*80)
    print("PHASE 2 & 3: EMBEDDING GENERATION & STORAGE")
    print("="*80)

    try:
        generator = VertexEmbeddingGenerator(
            model=args.model,
            task_type=args.task_type,
            batch_size=args.batch_size,
            output_dimensionality=args.output_dimensionality
        )
    except Exception as e:
        print(f"\nError initializing Vertex AI: {e}")
        print("\nMake sure you have:")
        print("  1. Set GOOGLE_CLOUD_PROJECT environment variable")
        print("  2. Authenticated with: gcloud auth application-default login")
        return 1

    storage = FlatEmbeddingStorage(landmarks, generator)

    try:
        storage.build_and_save(Path(args.output_file))
        print("\n✓ Successfully created embeddings!")
        return 0
    except Exception as e:
        print(f"\n✗ Error creating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
