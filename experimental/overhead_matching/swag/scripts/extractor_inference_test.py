#!/usr/bin/env python
"""Test script for SWAG model inference with extractor output visualization."""

import common.torch.load_torch_deps
import torch
import common.torch.load_and_save_models as lsm
import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.model.semantic_landmark_utils import custom_id_from_props
from pathlib import Path


def get_proper_nouns_for_batch(extractor, metadata):
    """Get proper nouns from PanoramaProperNounExtractor for each batch item."""
    results = []
    for item in metadata:
        pano_id = item['pano_id']
        item_nouns = []
        if pano_id in extractor.panorama_proper_nouns:
            for lm in extractor.panorama_proper_nouns[pano_id]:
                item_nouns.extend(lm['proper_nouns'])
        results.append(item_nouns)
    return results


def get_osm_field_values_for_batch(metadata, extractor):
    """Get OSM field values for landmarks that matched in OSMFieldExtractor.

    The extractor's tag determines what text was embedded:
    - "thing" -> name (e.g., "Summit Inn")
    - "place" -> street address (e.g., "302 South Lander Street")
    """
    tag = extractor.config.tag
    results = []
    for item in metadata:
        item_values = []
        for lm in item.get('landmarks', []):
            custom_id = custom_id_from_props(lm['pruned_props'])
            if custom_id in extractor.custom_id_to_idx:
                # pruned_props is a frozenset of (key, value) tuples
                props_dict = dict(lm['pruned_props'])

                if tag == "thing":
                    # "thing" embeddings are from name/brand
                    text = props_dict.get('name') or props_dict.get('brand') or "N/A"
                elif tag == "place":
                    # "place" embeddings are from street address
                    addr_street = props_dict.get('addr:street', '')
                    addr_housenumber = props_dict.get('addr:housenumber', '')
                    if addr_street:
                        text = f"{addr_housenumber} {addr_street}".strip() if addr_housenumber else addr_street
                    else:
                        text = props_dict.get('name', 'N/A')  # fallback
                else:
                    text = str(props_dict)

                item_values.append(text)
        results.append(item_values)
    return results


def main():
    MODEL_BASE = Path("/data/overhead_matching/training_outputs/260123_toy_problem_ramp/260123_184938_spoofed_hash_bit_v2")
    DATASET_BASE = Path("/data/overhead_matching/datasets/VIGOR/")
    CITY = "Chicago"
    BATCH_SIZE = 4

    print("=" * 60)
    print("Loading models...")
    print("=" * 60)

    pano_model = lsm.load_model(MODEL_BASE / "best_panorama", device="cuda")
    sat_model = lsm.load_model(MODEL_BASE / "best_satellite", device="cuda")
    pano_model.eval()
    sat_model.eval()

    print(f"Pano model extractors: {list(pano_model._extractor_by_name.keys())}")
    print(f"Sat model extractors: {list(sat_model._extractor_by_name.keys())}")

    print("\n" + "=" * 60)
    print("Creating dataset...")
    print("=" * 60)

    config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=True,
        should_load_landmarks=True,
        landmark_version="spoofed_v1",
        satellite_patch_size=(320, 320),
        panorama_size=(320, 640),
    )

    dataset = vd.VigorDataset(
        dataset_path=DATASET_BASE / CITY,
        config=config
    )

    dataloader = vd.get_dataloader(dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

    print(f"Dataset size: {len(dataset)}")

    print("\n" + "=" * 60)
    print("Getting batch and running inference...")
    print("=" * 60)

    batch = next(iter(dataloader))

    print(f"Batch panorama shape: {batch.panorama.shape}")
    print(f"Batch satellite shape: {batch.satellite.shape}")

    pano_input = pano_model.model_input_from_batch(batch).to("cuda")
    sat_input = sat_model.model_input_from_batch(batch).to("cuda")

    with torch.no_grad():
        pano_embeddings, pano_extractor_outputs = pano_model(pano_input)
        sat_embeddings, sat_extractor_outputs = sat_model(sat_input)

    print(f"Pano embeddings shape: {pano_embeddings.shape}")
    print(f"Sat embeddings shape: {sat_embeddings.shape}")

    print("\n" + "=" * 60)
    print("PANORAMA EXTRACTOR OUTPUTS")
    print("=" * 60)

    for extractor_name, extractor_output in pano_extractor_outputs.items():
        print(f"\n--- {extractor_name} ---")
        print(f"  Features shape: {extractor_output.features.shape}")
        print(f"  Positions shape: {extractor_output.positions.shape}")
        print(f"  Mask shape: {extractor_output.mask.shape}")
        valid_tokens_per_item = (~extractor_output.mask).sum(dim=1).tolist()
        print(f"  Valid tokens per item: {valid_tokens_per_item}")

        extractor = pano_model._extractor_by_name[extractor_name]

        # Get text for this extractor
        if hasattr(extractor, 'panorama_proper_nouns'):
            proper_nouns = get_proper_nouns_for_batch(extractor, pano_input.metadata)
            for i, nouns in enumerate(proper_nouns):
                pano_id = pano_input.metadata[i]['pano_id']
                print(f"  Item {i} (pano_id={pano_id}): {nouns}")

    print("\n" + "=" * 60)
    print("SATELLITE EXTRACTOR OUTPUTS")
    print("=" * 60)

    for extractor_name, extractor_output in sat_extractor_outputs.items():
        print(f"\n--- {extractor_name} ---")
        print(f"  Features shape: {extractor_output.features.shape}")
        print(f"  Positions shape: {extractor_output.positions.shape}")
        print(f"  Mask shape: {extractor_output.mask.shape}")
        valid_tokens_per_item = (~extractor_output.mask).sum(dim=1).tolist()
        print(f"  Valid tokens per item: {valid_tokens_per_item}")

        extractor = sat_model._extractor_by_name[extractor_name]

        # Get text for this extractor
        if hasattr(extractor, 'custom_id_to_idx'):
            osm_values = get_osm_field_values_for_batch(sat_input.metadata, extractor)
            for i, values in enumerate(osm_values):
                lat = sat_input.metadata[i]['lat']
                lon = sat_input.metadata[i]['lon']
                print(f"  Item {i} (lat={lat:.4f}, lon={lon:.4f}): {values}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
