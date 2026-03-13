"""Verification script for OSMFieldExtractor.

Loads a VIGOR dataset, creates an OSMFieldExtractor, runs it on a batch, and
manually verifies each output embedding against the raw pickle file to ensure
correctness.
"""

import common.torch.load_torch_deps
import torch
import pickle
import argparse
from pathlib import Path

from experimental.overhead_matching.swag.data.vigor_dataset import (
    VigorDataset, VigorDatasetConfig, get_dataloader)
from experimental.overhead_matching.swag.model.osm_field_extractor import OSMFieldExtractor
from experimental.overhead_matching.swag.model.swag_config_types import (
    OSMFieldExtractorConfig, LandmarkType)
from experimental.overhead_matching.swag.model.swag_model_input_output import ModelInput
from experimental.overhead_matching.swag.model.semantic_landmark_utils import custom_id_from_props


def verify_tag(dataset, tag_name, embedding_version, embedding_base_path, batch_size=16):
    """Run extractor and verify each embedding against manual pickle lookup.

    Args:
        dataset: VigorDataset instance.
        tag_name: OSM tag to verify (e.g. "thing", "place").
        embedding_version: Version string for embedding directory.
        embedding_base_path: Base path containing embedding versions.
        batch_size: Batch size for dataloader.
    """
    config = OSMFieldExtractorConfig(
        tag=tag_name,
        openai_embedding_size=256,
        embedding_version=embedding_version,
        auxiliary_info_key="test",
        landmark_type=LandmarkType.POINT,
    )
    extractor = OSMFieldExtractor(config, embedding_base_path)

    dataloader = get_dataloader(dataset, batch_size=batch_size)
    batch = next(iter(dataloader))
    model_input = ModelInput(
        image=batch.satellite,
        metadata=batch.satellite_metadata)

    output = extractor(model_input)

    # Manually load pickle
    pickle_path = Path(embedding_base_path) / embedding_version / "embeddings" / "embeddings.pkl"
    assert pickle_path.exists(), f"Pickle not found: {pickle_path}"
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    assert tag_name in data, f"Tag '{tag_name}' not in pickle keys: {list(data.keys())}"
    tag_data = data[tag_name]
    tensor, id_to_idx = tag_data[0], tag_data[1]

    num_checked = 0
    for i, item in enumerate(batch.satellite_metadata):
        token_idx = 0
        for lm in item["landmarks"]:
            if lm["geometry"].geom_type.lower() != "point":
                continue
            cid = custom_id_from_props(lm["pruned_props"])
            if cid not in id_to_idx:
                continue

            # Manual lookup and normalize
            manual_emb = tensor[id_to_idx[cid]][:256].float()
            manual_emb = manual_emb / torch.norm(manual_emb)

            # Compare with extractor output
            extractor_emb = output.features[i, token_idx]
            assert not output.mask[i, token_idx], (
                f"[{tag_name}] batch item {i}, token {token_idx} is masked but should not be")
            assert torch.allclose(extractor_emb, manual_emb, atol=1e-5), (
                f"[{tag_name}] Mismatch at batch item {i}, token {token_idx}: "
                f"max diff = {(extractor_emb - manual_emb).abs().max().item():.6e}")

            token_idx += 1
            num_checked += 1

    assert num_checked > 0, f"[{tag_name}] No embeddings were checked — dataset may have no matching landmarks"
    print(f"[{tag_name}] Verified {num_checked} embeddings — all match!")


def main():
    parser = argparse.ArgumentParser(description="Verify OSMFieldExtractor correctness")
    parser.add_argument("--dataset_base", type=str, required=True,
                        help="Path to VIGOR dataset (e.g. /data/overhead_matching/datasets/VIGOR)")
    parser.add_argument("--embedding_base", type=str, required=True,
                        help="Base path for semantic landmark embeddings")
    parser.add_argument("--embedding_version", type=str, required=True,
                        help="Embedding version directory")
    parser.add_argument("--landmark_version", type=str, default="spoofed_v1",
                        help="Landmark version for dataset")
    parser.add_argument("--tags", type=str, nargs="+", default=["thing", "place"],
                        help="Tags to verify")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--factor", type=float, default=0.1,
                        help="Factor for dataset subset (smaller = faster)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_base) / "Chicago"
    config = VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        factor=args.factor,
        should_load_images=True,
        should_load_landmarks=True,
        landmark_version=args.landmark_version,
    )
    print(f"Loading dataset from {dataset_path} with factor={args.factor}...")
    dataset = VigorDataset(dataset_path, config)
    print(f"Dataset loaded: {len(dataset)} items")

    for tag in args.tags:
        print(f"\nVerifying tag: {tag}")
        verify_tag(dataset, tag, args.embedding_version, args.embedding_base,
                   batch_size=args.batch_size)

    print(f"\nAll {len(args.tags)} tags verified successfully!")


if __name__ == "__main__":
    main()
