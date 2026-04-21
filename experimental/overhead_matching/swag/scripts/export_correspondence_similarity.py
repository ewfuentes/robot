"""Export correspondence-based raw cost data + optional similarity matrix.

Loads a trained CorrespondenceClassifier, pano_v2 tags, and a VigorDataset
to precompute a flat (total_pano_lm × total_osm_lm) P(match) cost matrix.
The raw artifact is always saved to disk. Optionally, `--compute_similarity`
additionally folds the raw data into a (num_panos, num_sats) similarity
matrix via `similarity_from_raw_data` and saves that too.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:export_correspondence_similarity -- \\
        --model_path /data/.../best_model.pt \\
        --text_embeddings_path /data/.../eval_text_embeddings.pkl \\
        --dataset_path /data/overhead_matching/datasets/VIGOR/mapillary/MiamiBeach \\
        --pano_v2_base /data/.../semantic_landmark_embeddings/mapillary \\
        --output_path /tmp/miami_corr.pt \\
        --compute_similarity

Load an existing raw artifact and re-run only the similarity-matrix step:
    bazel run //experimental/overhead_matching/swag/scripts:export_correspondence_similarity -- \\
        --from_raw /tmp/miami_corr.pt \\
        --dataset_path /data/overhead_matching/datasets/VIGOR/mapillary/MiamiBeach \\
        --output_path /tmp/miami_corr.pt \\
        --compute_similarity
"""

import argparse
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
    """Auto-detect landmark version from the single .feather file in landmarks/."""
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


def load_vigor_dataset(dataset_path: Path, landmark_version: str | None,
                       inflation_factor: float):
    landmark_version = landmark_version or auto_detect_landmark_version(dataset_path)
    config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version=landmark_version,
        landmark_correspondence_inflation_factor=inflation_factor,
    )
    return vd.VigorDataset(dataset_path, config)


def build_raw_cost_data(args) -> cm.RawCorrespondenceData:
    """Run the model-inference half of the pipeline and return a RawCorrespondenceData.

    Requires --model_path, --text_embeddings_path, --pano_v2_base.
    """
    for required in ("model_path", "text_embeddings_path", "pano_v2_base"):
        if getattr(args, required) is None:
            raise ValueError(
                f"--{required} is required when not loading from --from_raw"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading text embeddings from {args.text_embeddings_path}")
    text_embeddings = load_text_embeddings(args.text_embeddings_path)
    text_input_dim = next(iter(text_embeddings.values())).shape[0]
    print(f"  {len(text_embeddings)} entries, dim={text_input_dim}")

    print(f"Loading model from {args.model_path}")
    encoder_config = TagBundleEncoderConfig(
        text_input_dim=text_input_dim, text_proj_dim=128,
    )
    classifier_config = CorrespondenceClassifierConfig(encoder=encoder_config)
    model = CorrespondenceClassifier(classifier_config).to(device)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"  Model loaded, device={device}")

    dataset_path = args.dataset_path.expanduser().resolve()
    print(f"Loading dataset from {dataset_path}")
    dataset = load_vigor_dataset(dataset_path, args.landmark_version,
                                 args.inflation_factor)
    print(
        f"  {len(dataset._panorama_metadata)} panos, "
        f"{len(dataset._satellite_metadata)} sats, "
        f"{len(dataset._landmark_metadata)} landmarks"
    )

    pano_tags_from_pano_id: dict[str, list[dict]] = {}
    for base in args.pano_v2_base:
        print(f"Loading pano_v2 tags from {base}")
        tags = extract_panorama_data_across_cities(
            base, extract_tags_from_pano_data,
        )
        pano_tags_from_pano_id.update(tags)
        print(f"  {len(tags)} panoramas from {base.name}")
    print(f"  Total: {len(pano_tags_from_pano_id)} panoramas with tags")

    print("Precomputing raw cost matrix data...")
    return cm.precompute_raw_cost_data(
        model=model,
        text_embeddings=text_embeddings,
        text_input_dim=text_input_dim,
        dataset=dataset,
        pano_tags_from_pano_id=pano_tags_from_pano_id,
        device=device,
        allow_missing_text_embeddings=args.allow_missing_text_embeddings,
    )


def save_raw_cost_data(raw: cm.RawCorrespondenceData, output_path: Path) -> None:
    """Write raw data as .npy (cost matrix) + .pt (metadata)."""
    cost_npy_path = output_path.parent / (output_path.stem + "_cost_matrix.npy")
    print(f"Saving cost matrix ({raw.cost_matrix.shape}) to {cost_npy_path}...")
    np.save(cost_npy_path, raw.cost_matrix)

    save_dict = {
        "cost_matrix_path": str(cost_npy_path),
        "pano_id_to_lm_rows": raw.pano_id_to_lm_rows,
        "pano_lm_tags": raw.pano_lm_tags,
        "osm_lm_indices": raw.osm_lm_indices,
        "osm_lm_tags": raw.osm_lm_tags,
    }
    torch.save(save_dict, output_path, pickle_protocol=4)
    print(f"Saved metadata to {output_path}")
    print(f"Raw cost data saved: {cost_npy_path} + {output_path}")


def load_raw_cost_data(raw_path: Path) -> cm.RawCorrespondenceData:
    print(f"Loading precomputed raw data from {raw_path}")
    data = torch.load(raw_path, weights_only=False)
    if "cost_matrix" in data:
        cost_matrix = data["cost_matrix"]
    else:
        cost_npy = data["cost_matrix_path"]
        print(f"  Loading cost matrix from {cost_npy}")
        cost_matrix = np.load(cost_npy)
    return cm.RawCorrespondenceData(
        cost_matrix=cost_matrix,
        pano_id_to_lm_rows=data["pano_id_to_lm_rows"],
        pano_lm_tags=data["pano_lm_tags"],
        osm_lm_indices=data["osm_lm_indices"],
        osm_lm_tags=data["osm_lm_tags"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export correspondence raw cost data + optional similarity matrix."
    )
    parser.add_argument("--model_path", type=Path, default=None,
                        help="Trained CorrespondenceClassifier .pt file.")
    parser.add_argument("--text_embeddings_path", type=Path, default=None,
                        help="Path to text embeddings pickle file.")
    parser.add_argument("--dataset_path", type=Path, required=True,
                        help="Path to VIGOR city dir.")
    parser.add_argument("--pano_v2_base", type=Path, nargs="+", default=None,
                        help="Base path(s) for pano_v2 embeddings.")
    parser.add_argument("--landmark_version", type=str, default=None,
                        help="Landmark version string (default: auto-detect).")
    parser.add_argument("--output_path", type=Path, required=True,
                        help="Output .pt path for the raw artifact. "
                             "If --compute_similarity, the similarity matrix is "
                             "written alongside as <stem>_similarity.pt.")
    parser.add_argument("--inflation_factor", type=float, default=1.0,
                        help="Satellite patch inflation factor (default: 1.0).")
    parser.add_argument("--from_raw", type=Path, default=None,
                        help="Skip model inference; load raw artifact from this "
                             "path instead. Useful for re-running similarity with "
                             "different matching settings.")
    parser.add_argument("--compute_similarity", action="store_true",
                        help="Additionally compute a similarity matrix from the "
                             "raw data and save it to <output_path>_similarity.pt.")
    parser.add_argument("--method", type=str, default="hungarian",
                        choices=["hungarian", "greedy"],
                        help="Matching method (used with --compute_similarity).")
    parser.add_argument("--aggregation", type=str, default="sum",
                        choices=["sum", "max", "log_odds"],
                        help="Aggregation mode (used with --compute_similarity).")
    parser.add_argument("--prob_threshold", type=float, default=0.3,
                        help="Min P(match) to include (used with --compute_similarity).")
    parser.add_argument("--uniqueness_weighted", action="store_true",
                        help="Weight matched pairs by pano landmark uniqueness.")
    parser.add_argument("--no_dustbin", action="store_true",
                        help="Disable the Hungarian dustbin (augment with "
                             "threshold-valued sink columns). With the "
                             "dustbin on (default), the threshold is baked "
                             "into the optimization so low-prob rows route "
                             "to the sink instead of saddling other rows "
                             "with bad assignments. Use --no_dustbin to "
                             "reproduce legacy post-hoc-threshold artifacts.")
    parser.add_argument("--ks", type=str, default="1,5,10",
                        help="Comma-separated top-k values for retrieval metrics "
                             "(used with --compute_similarity).")
    parser.add_argument("--allow_missing_text_embeddings", action="store_true",
                        help="Silently substitute zero vectors for text values "
                             "not found in the embeddings pickle. Not recommended.")
    args = parser.parse_args()

    dataset_path = args.dataset_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stage 1: obtain RawCorrespondenceData — either by loading or by running inference.
    if args.from_raw is not None:
        raw = load_raw_cost_data(args.from_raw.expanduser().resolve())
    else:
        raw = build_raw_cost_data(args)
        save_raw_cost_data(raw, output_path)

    print(
        f"  {raw.cost_matrix.shape[0]} pano landmarks × "
        f"{raw.cost_matrix.shape[1]} OSM landmarks"
    )

    # Stage 2: optionally compute the similarity matrix.
    if not args.compute_similarity:
        return

    print(
        f"Building similarity matrix (method={args.method}, "
        f"agg={args.aggregation}, threshold={args.prob_threshold}, "
        f"uniqueness={args.uniqueness_weighted}, "
        f"dustbin={not args.no_dustbin})"
    )
    dataset = load_vigor_dataset(dataset_path, args.landmark_version,
                                 args.inflation_factor)
    similarity = cm.similarity_from_raw_data(
        raw, dataset,
        cm.MatchingMethod(args.method),
        cm.AggregationMode(args.aggregation),
        args.prob_threshold,
        uniqueness_weighted=args.uniqueness_weighted,
        use_dustbin=not args.no_dustbin,
    )

    ks = [int(k) for k in args.ks.split(",")]
    metrics = rm.compute_top_k_metrics(similarity, dataset, ks=ks)
    city_name = dataset_path.name
    print(f"\nMetrics for {city_name}:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    sim_path = output_path.parent / (output_path.stem + "_similarity.pt")
    torch.save(similarity, sim_path)
    print(f"\nSaved similarity matrix {similarity.shape} to {sim_path}")


if __name__ == "__main__":
    main()
