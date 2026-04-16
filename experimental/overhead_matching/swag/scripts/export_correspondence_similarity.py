"""Export correspondence-based similarity matrix for a VIGOR city.

Loads a trained CorrespondenceClassifier, pano_v2 tags, and a VigorDataset
to build a (num_panos, num_sats) similarity matrix using bipartite matching.
Reports recall@k and MRR metrics.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:export_correspondence_similarity -- \
        --model_path /data/.../best_model.pt \
        --text_embeddings_path /data/.../eval_text_embeddings.pkl \
        --dataset_path /data/overhead_matching/datasets/VIGOR/mapillary/MiamiBeach \
        --pano_v2_base /data/.../semantic_landmark_embeddings/mapillary \
        --landmark_version MiamiBeach_v1_150101 \
        --output_path /tmp/miami_corr_sim.pt
"""

import argparse
import os
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401 — must precede torch import
import numpy as np
import torch

from experimental.overhead_matching.swag.data import vigor_dataset as vd
from experimental.overhead_matching.swag.data.landmark_correspondence_dataset import (
    load_text_embeddings,
)
from experimental.overhead_matching.swag.evaluation import (
    correspondence_matching as cm,
)
from experimental.overhead_matching.swag.evaluation import retrieval_metrics as rm
from experimental.overhead_matching.swag.model.additional_panorama_extractors import (
    extract_panorama_data_across_cities,
)
from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    CorrespondenceClassifier,
    CorrespondenceClassifierConfig,
    TagBundleEncoderConfig,
)
from experimental.overhead_matching.swag.scripts.landmark_pairing_cli import (
    extract_tags_from_pano_data,
)


def auto_detect_landmark_version(dataset_path: Path) -> str:
    """Auto-detect landmark version from single .feather file in landmarks/ dir."""
    landmarks_dir = dataset_path / "landmarks"
    if not landmarks_dir.exists():
        raise FileNotFoundError(f"No landmarks/ directory found at {landmarks_dir}")
    feather_files = list(landmarks_dir.glob("*.feather"))
    if len(feather_files) == 0:
        raise FileNotFoundError(f"No .feather files in {landmarks_dir}")
    if len(feather_files) > 1:
        names = [f.name for f in feather_files]
        raise ValueError(
            f"Multiple .feather files in {landmarks_dir}, specify --landmark_version. "
            f"Found: {names}"
        )
    return feather_files[0].stem


def main():
    parser = argparse.ArgumentParser(
        description="Export correspondence-based similarity matrix for a VIGOR city"
    )
    parser.add_argument("--model_path", type=Path, default=None,
                        help="Path to trained CorrespondenceClassifier .pt file")
    parser.add_argument("--text_embeddings_path", type=Path, default=None,
                        help="Path to text embeddings pickle file")
    parser.add_argument("--dataset_path", type=Path, required=True,
                        help="Path to VIGOR city dir")
    parser.add_argument("--pano_v2_base", type=Path, nargs="+", default=None,
                        help="Base path(s) for pano_v2 embeddings (contains city subdirs)")
    parser.add_argument("--landmark_version", type=str, default=None,
                        help="Landmark version string (default: auto-detect)")
    parser.add_argument("--output_path", type=Path, required=True,
                        help="Output .pt file path")
    parser.add_argument("--inflation_factor", type=float, default=1.0,
                        help="Satellite patch inflation factor (default: 1.0)")
    parser.add_argument("--method", type=str, default="hungarian",
                        choices=["hungarian", "greedy"],
                        help="Matching method (default: hungarian)")
    parser.add_argument("--aggregation", type=str, default="sum",
                        choices=["sum", "max", "log_odds"],
                        help="Aggregation mode (default: sum)")
    parser.add_argument("--prob_threshold", type=float, default=0.3,
                        help="Min P(match) to include in matching (default: 0.3)")
    parser.add_argument("--ks", type=str, default="1,5,10",
                        help="Comma-separated top-k values (default: 1,5,10)")
    parser.add_argument("--save_raw", action="store_true",
                        help="Save raw cost matrix data for correspondence_explorer "
                             "instead of the final similarity matrix")
    parser.add_argument("--from_raw", type=Path, default=None,
                        help="Build similarity matrix from precomputed raw .pt file "
                             "(skips model inference entirely)")
    parser.add_argument("--uniqueness_weighted", action="store_true",
                        help="Weight matched pairs by pano landmark uniqueness "
                             "(1/log2(1 + n_matches))")
    parser.add_argument("--simple", action="store_true",
                        help="Use SimpleCorrespondenceClassifier (text-only encoder, 4 cross features)")
    args = parser.parse_args()

    dataset_path = args.dataset_path.expanduser().resolve()
    ks = [int(k) for k in args.ks.split(",")]

    method = cm.MatchingMethod(args.method)
    aggregation = cm.AggregationMode(args.aggregation)

    # Fast path: build similarity from precomputed raw data (no model needed)
    if args.from_raw:
        print(f"Loading precomputed raw data from {args.from_raw}")
        data = torch.load(args.from_raw, weights_only=False)
        # Support both formats: inline cost_matrix or separate .npy file
        if 'cost_matrix' in data:
            cost_matrix = data['cost_matrix']
        else:
            cost_npy = data['cost_matrix_path']
            print(f"  Loading cost matrix from {cost_npy}")
            cost_matrix = np.load(cost_npy)
        raw = cm.RawCorrespondenceData(
            cost_matrix=cost_matrix,
            pano_id_to_lm_rows=data['pano_id_to_lm_rows'],
            pano_lm_tags=data['pano_lm_tags'],
            osm_lm_indices=data['osm_lm_indices'],
            osm_lm_tags=data['osm_lm_tags'],
        )
        print(f"  {raw.cost_matrix.shape[0]} pano landmarks × {raw.cost_matrix.shape[1]} OSM landmarks")

        landmark_version = args.landmark_version or auto_detect_landmark_version(dataset_path)
        config = vd.VigorDatasetConfig(
            satellite_tensor_cache_info=None, panorama_tensor_cache_info=None,
            should_load_images=False, should_load_landmarks=True,
            landmark_version=landmark_version,
            landmark_correspondence_inflation_factor=args.inflation_factor,
        )
        dataset = vd.VigorDataset(dataset_path, config)

        print(f"Building similarity (method={args.method}, agg={args.aggregation}, "
              f"threshold={args.prob_threshold}, uniqueness={args.uniqueness_weighted})")
        similarity = cm.similarity_from_raw_data(
            raw, dataset, method, aggregation, args.prob_threshold,
            uniqueness_weighted=args.uniqueness_weighted,
        )

        metrics = rm.compute_top_k_metrics(similarity, dataset, ks=ks)
        city_name = dataset_path.name
        print(f"\nMetrics for {city_name}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        output_path = args.output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(similarity, output_path)
        print(f"Saved similarity matrix {similarity.shape} to {output_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load text embeddings
    print(f"Loading text embeddings from {args.text_embeddings_path}")
    text_embeddings = load_text_embeddings(args.text_embeddings_path)
    text_input_dim = next(iter(text_embeddings.values())).shape[0]
    print(f"  {len(text_embeddings)} entries, dim={text_input_dim}")

    # 2. Load model
    print(f"Loading model from {args.model_path}")
    if args.simple:
        from experimental.overhead_matching.swag.model.simple_correspondence_model import (
            SimpleCorrespondenceClassifier,
            SimpleCorrespondenceClassifierConfig,
            SimpleTagBundleEncoderConfig,
        )
        enc = SimpleTagBundleEncoderConfig(text_input_dim=text_input_dim, text_proj_dim=128)
        cfg = SimpleCorrespondenceClassifierConfig(encoder=enc)
        model = SimpleCorrespondenceClassifier(cfg).to(device)
    else:
        encoder_config = TagBundleEncoderConfig(text_input_dim=text_input_dim, text_proj_dim=128)
        classifier_config = CorrespondenceClassifierConfig(encoder=encoder_config)
        model = CorrespondenceClassifier(classifier_config).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"  Model loaded ({'simple' if args.simple else 'standard'}), device={device}")

    # 3. Load VigorDataset
    landmark_version = args.landmark_version or auto_detect_landmark_version(dataset_path)
    print(f"Loading dataset from {dataset_path} (landmark_version={landmark_version})")
    config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version=landmark_version,
        landmark_correspondence_inflation_factor=args.inflation_factor,
    )
    dataset = vd.VigorDataset(dataset_path, config)
    print(f"  {len(dataset._panorama_metadata)} panos, "
          f"{len(dataset._satellite_metadata)} sats, "
          f"{len(dataset._landmark_metadata)} landmarks")

    # 4. Load pano_v2 tags from all base paths
    pano_tags_from_pano_id = {}
    for base in args.pano_v2_base:
        print(f"Loading pano_v2 tags from {base}")
        tags = extract_panorama_data_across_cities(base, extract_tags_from_pano_data)
        pano_tags_from_pano_id.update(tags)
        print(f"  {len(tags)} panoramas from {base.name}")
    print(f"  Total: {len(pano_tags_from_pano_id)} panoramas with tags")

    # 5. Precompute raw cost data or build similarity matrix
    if args.save_raw:
        if args.simple:
            print(f"Precomputing raw cost matrix data (simple)...")
            from experimental.overhead_matching.swag.evaluation.simple_correspondence_export import (
                simple_precompute_raw_cost_data,
            )
            raw_data = simple_precompute_raw_cost_data(
                model=model,
                text_embeddings=text_embeddings,
                text_input_dim=text_input_dim,
                dataset=dataset,
                pano_tags_from_pano_id=pano_tags_from_pano_id,
                device=device,
            )
            _ckpt_dir = None
        else:
            # Use checkpoint dir next to output for resumability
            _ckpt_dir = str(args.output_path.expanduser().resolve()) + ".checkpoints"
            print(f"Precomputing raw cost matrix data (checkpoints: {_ckpt_dir})...")
            raw_data = cm.precompute_raw_cost_data(
                model=model,
                text_embeddings=text_embeddings,
                text_input_dim=text_input_dim,
                dataset=dataset,
                pano_tags_from_pano_id=pano_tags_from_pano_id,
                device=device,
                checkpoint_dir=_ckpt_dir,
            )

        # Save raw data — cost matrix as .npy (handles large arrays without pickle OOM),
        # metadata as torch .pt
        output_path = args.output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cost_npy_path = output_path.parent / (output_path.stem + "_cost_matrix.npy")

        print(f"Saving cost matrix ({raw_data.cost_matrix.shape}) to {cost_npy_path}...")
        np.save(cost_npy_path, raw_data.cost_matrix)
        # Free the big array before saving metadata
        del raw_data.cost_matrix

        save_dict = {
            "cost_matrix_path": str(cost_npy_path),
            "pano_id_to_lm_rows": raw_data.pano_id_to_lm_rows,
            "pano_lm_tags": raw_data.pano_lm_tags,
            "osm_lm_indices": raw_data.osm_lm_indices,
            "osm_lm_tags": raw_data.osm_lm_tags,
        }
        torch.save(save_dict, output_path, pickle_protocol=4)
        print(f"Saved metadata to {output_path}")
        print(f"Saved raw cost data to {cost_npy_path} + {output_path}")

        # Clean up checkpoints after successful save
        if _ckpt_dir is not None:
            import shutil
            if os.path.exists(_ckpt_dir):
                shutil.rmtree(_ckpt_dir)
                print(f"Cleaned up checkpoints at {_ckpt_dir}")
    else:
        print(f"Building similarity matrix (method={args.method}, agg={args.aggregation}, "
              f"threshold={args.prob_threshold}, uniqueness={args.uniqueness_weighted})")
        similarity = cm.build_correspondence_similarity_matrix(
            model=model,
            text_embeddings=text_embeddings,
            text_input_dim=text_input_dim,
            dataset=dataset,
            pano_tags_from_pano_id=pano_tags_from_pano_id,
            device=device,
            method=method,
            aggregation=aggregation,
            prob_threshold=args.prob_threshold,
            uniqueness_weighted=args.uniqueness_weighted,
        )
        print(f"Similarity matrix shape: {similarity.shape}")

        # 6. Metrics
        metrics = rm.compute_top_k_metrics(similarity, dataset, ks=ks)
        city_name = dataset_path.name
        print(f"\nMetrics for {city_name}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        # 7. Save
        output_path = args.output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(similarity, output_path)
        print(f"\nSaved similarity matrix {similarity.shape} to {output_path}")


if __name__ == "__main__":
    main()
