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
        --dataset_path /data/overhead_matching/datasets/VIGOR/Middletown \\
        --pano_v2_base /data/.../semantic_landmark_embeddings/panov2_tuned_prompt \\
        --satellite_dir /data/.../loci_satellite/.../satellite \\
        --landmark_path /data/.../loci_osm_landmarks/.../landmarks.feather \\
        --output_path /tmp/middletown_corr.pt \\
        --compute_similarity

Load an existing raw artifact and re-run only the similarity-matrix step:
    bazel run //experimental/overhead_matching/swag/scripts:export_correspondence_similarity -- \\
        --from_raw /tmp/middletown_corr.pt \\
        --dataset_path /data/overhead_matching/datasets/VIGOR/Middletown \\
        --output_path /tmp/middletown_corr.pt \\
        --compute_similarity
"""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

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
    iter_city_directories,
)
from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    CorrespondenceClassifier,
    CorrespondenceClassifierConfig,
    TagBundleEncoderConfig,
)
from experimental.overhead_matching.swag.scripts.landmark_pairing_cli import (
    extract_tags_from_pano_data,
)


RAW_CORRESPONDENCE_IDENTITY_SCHEMA = "swag_raw_correspondence_identity/v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    """Hash JSON-shaped metadata after normalizing NumPy/container scalars."""
    def normalize(item):
        if isinstance(item, dict):
            return {str(key): normalize(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        if isinstance(item, (set, frozenset)):
            return sorted((normalize(val) for val in item), key=repr)
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return float(item)
        return item

    payload = json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_landmark_path(
        dataset_path: Path, landmark_version: str | None,
        landmark_path: Path | None) -> Path:
    if landmark_path is not None:
        return Path(landmark_path).expanduser().resolve()
    version = landmark_version or auto_detect_landmark_version(dataset_path)
    path = dataset_path / "landmarks" / f"{version}.feather"
    if not path.exists():
        path = path.with_suffix(".geojson")
    return path.resolve()


def _satellite_landmark_indices_sha256(dataset) -> str:
    """Hash the exact ordered landmark-index association used by aggregation."""
    digest = hashlib.sha256(b"swag_satellite_landmark_indices/v1\0")
    for value in dataset._satellite_metadata["landmark_idxs"]:
        indices = [] if value is None else [int(index) for index in value]
        digest.update(len(indices).to_bytes(8, "big"))
        for index in indices:
            digest.update(index.to_bytes(8, "big", signed=True))
    return digest.hexdigest()


def _cost_matrix_values_sha256(cost_matrix: np.ndarray) -> str:
    """Hash matrix identity and values in bounded row chunks."""
    if cost_matrix.ndim != 2:
        raise ValueError(
            f"raw correspondence cost matrix must be 2-D, got "
            f"{cost_matrix.shape}")
    digest = hashlib.sha256(b"swag_raw_correspondence_cost_matrix/v1\0")
    for value in (str(cost_matrix.dtype), *cost_matrix.shape):
        encoded = str(value).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    bytes_per_row = max(1, cost_matrix.shape[1] * cost_matrix.dtype.itemsize)
    rows_per_chunk = max(1, (16 * 1024 * 1024) // bytes_per_row)
    for start in range(0, cost_matrix.shape[0], rows_per_chunk):
        chunk = np.ascontiguousarray(
            cost_matrix[start:start + rows_per_chunk])
        digest.update(chunk.tobytes(order="C"))
    return digest.hexdigest()


def _raw_payload_identity(raw: cm.RawCorrespondenceData) -> dict:
    return {
        "cost_matrix_shape": list(raw.cost_matrix.shape),
        "cost_matrix_dtype": str(raw.cost_matrix.dtype),
        "cost_matrix_values_sha256": _cost_matrix_values_sha256(
            raw.cost_matrix),
        "panorama_landmarks_sha256": _canonical_sha256({
            "pano_id_to_lm_rows": raw.pano_id_to_lm_rows,
            "pano_lm_tags": raw.pano_lm_tags,
        }),
        "osm_landmarks_sha256": _canonical_sha256({
            "osm_lm_indices": raw.osm_lm_indices,
            "osm_lm_tags": raw.osm_lm_tags,
        }),
    }


def _raw_aggregation_identity(
        dataset, dataset_path: Path, landmark_path: Path) -> dict:
    return {
        "pano_id_mapping_sha256": _sha256_file(
            dataset_path / "pano_id_mapping.csv"),
        "matrix_identity": vd.similarity_matrix_identity(
            dataset._panorama_metadata, dataset._satellite_metadata),
        "satellite_landmark_indices_sha256": (
            _satellite_landmark_indices_sha256(dataset)),
        "landmark_source_sha256": _sha256_file(landmark_path),
        "landmark_correspondence_inflation_factor": (
            dataset._config.landmark_correspondence_inflation_factor),
    }


def _pano_v2_pickle_hashes(bases: list[Path]) -> list[str]:
    hashes = []
    for base in bases:
        for _, city_dir in iter_city_directories(base):
            pickle_path = city_dir / "embeddings" / "embeddings.pkl"
            if pickle_path.is_file():
                hashes.append(_sha256_file(pickle_path))
    return hashes


def build_raw_identity(
        raw: cm.RawCorrespondenceData, dataset, dataset_path: Path,
        landmark_path: Path, model_path: Path, text_embeddings_path: Path,
        pano_v2_bases: list[Path], allow_missing_text_embeddings: bool) -> dict:
    return {
        "schema": RAW_CORRESPONDENCE_IDENTITY_SCHEMA,
        "aggregation_inputs": _raw_aggregation_identity(
            dataset, dataset_path, landmark_path),
        "raw_payload": _raw_payload_identity(raw),
        "inference_inputs": {
            "classifier_sha256": _sha256_file(model_path),
            "text_embeddings_sha256": _sha256_file(text_embeddings_path),
            "pano_v2_embeddings_sha256": _pano_v2_pickle_hashes(
                pano_v2_bases),
            "allow_missing_text_embeddings": allow_missing_text_embeddings,
        },
    }


def validate_raw_identity(
        raw: cm.RawCorrespondenceData, dataset, dataset_path: Path,
        landmark_path: Path, allow_legacy: bool = False,
        require_identity: bool = False) -> None:
    """Prove persisted raw rows/columns belong to the aggregation inputs."""
    identity = raw.identity
    if identity is None:
        message = (
            "Raw correspondence artifact has no source/alignment identity. "
            "Regenerate it, or explicitly pass --allow_legacy_raw_identity "
            "for an independently audited legacy artifact.")
        if require_identity and not allow_legacy:
            raise ValueError(message)
        if allow_legacy:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return
    if identity.get("schema") != RAW_CORRESPONDENCE_IDENTITY_SCHEMA:
        raise ValueError(
            "Raw correspondence artifact has an unsupported identity schema")

    expected_aggregation = _raw_aggregation_identity(
        dataset, dataset_path, landmark_path)
    if identity.get("aggregation_inputs") != expected_aggregation:
        actual = identity.get("aggregation_inputs", {})
        mismatches = sorted(
            key for key in set(actual) | set(expected_aggregation)
            if actual.get(key) != expected_aggregation.get(key))
        raise ValueError(
            "Raw correspondence artifact belongs to different dataset/OSM/"
            f"satellite aggregation inputs; fields differ: {mismatches}")
    if identity.get("raw_payload") != _raw_payload_identity(raw):
        raise ValueError(
            "Raw correspondence artifact values or metadata do not match "
            "its identity")


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


def load_vigor_dataset(
    dataset_path: Path,
    landmark_version: str | None,
    inflation_factor: float,
    satellite_dir: Path | None = None,
    landmark_path: Path | None = None,
):
    if landmark_path is None:
        landmark_version = (
            landmark_version or auto_detect_landmark_version(dataset_path))
    else:
        landmark_version = landmark_version or "v1"
    config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version=landmark_version,
        landmark_correspondence_inflation_factor=inflation_factor,
        satellite_dir=satellite_dir,
        landmark_path=landmark_path,
    )
    return vd.VigorDataset(dataset_path, config)


def load_panorama_tags(pano_v2_bases: list[Path]) -> dict[str, list[dict]]:
    """Load VLM tags without silently mixing legs that reuse panorama IDs."""
    result = {}
    for base in pano_v2_bases:
        print(f"Loading pano_v2 tags from {base}")
        tags = extract_panorama_data_across_cities(
            base, extract_tags_from_pano_data,
        )
        duplicates = result.keys() & tags.keys()
        if duplicates:
            first = min(duplicates)
            raise ValueError(
                f"{base} duplicates panorama ID {first!r} from an earlier "
                "--pano_v2_base input"
            )
        result.update(tags)
        print(f"  {len(tags)} panoramas from {base.name}")
    print(f"  Total: {len(result)} panoramas with tags")
    return result


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
    dataset = load_vigor_dataset(
        dataset_path, args.landmark_version, args.inflation_factor,
        args.satellite_dir, args.landmark_path)
    print(
        f"  {len(dataset._panorama_metadata)} panos, "
        f"{len(dataset._satellite_metadata)} sats, "
        f"{len(dataset._landmark_metadata)} landmarks"
    )

    pano_tags_from_pano_id = load_panorama_tags(args.pano_v2_base)

    print("Precomputing raw cost matrix data...")
    cost_matrix_memmap_path = None
    if args.stream_cost_matrix:
        cost_matrix_memmap_path = (
            args.output_path.expanduser().resolve().parent
            / (args.output_path.stem + "_cost_matrix.npy")
        )
    raw = cm.precompute_raw_cost_data(
        model=model,
        text_embeddings=text_embeddings,
        text_input_dim=text_input_dim,
        dataset=dataset,
        pano_tags_from_pano_id=pano_tags_from_pano_id,
        device=device,
        allow_missing_text_embeddings=args.allow_missing_text_embeddings,
        cost_matrix_memmap_path=cost_matrix_memmap_path,
    )
    landmark_path = _resolve_landmark_path(
        dataset_path, args.landmark_version, args.landmark_path)
    raw.identity = build_raw_identity(
        raw, dataset, dataset_path, landmark_path,
        args.model_path, args.text_embeddings_path, args.pano_v2_base,
        args.allow_missing_text_embeddings)
    return raw


def save_raw_cost_data(
    raw: cm.RawCorrespondenceData,
    output_path: Path,
    model_path: Path,
    text_embeddings_path: Path,
) -> None:
    """Write raw data as .npy (cost matrix) + .pt (metadata).

    When `raw.cost_matrix_path` is set (stream-to-disk path), the cost matrix
    bytes are already at that location. Sibling matrices are recorded relative
    to the metadata so an artifact-directory rename does not stale the path.
    Otherwise we save the in-RAM array next to ``output_path``.

    `model_path` and `text_embeddings_path` are recorded in the .pt for
    provenance — they are the inputs that produced this cost matrix.
    """
    if raw.cost_matrix_path is not None:
        cost_npy_path = raw.cost_matrix_path
        print(f"Cost matrix already streamed to {cost_npy_path} "
              f"(shape={raw.cost_matrix.shape}); skipping np.save.")
    else:
        cost_npy_path = output_path.parent / (output_path.stem + "_cost_matrix.npy")
        print(f"Saving cost matrix ({raw.cost_matrix.shape}) to {cost_npy_path}...")
        np.save(cost_npy_path, raw.cost_matrix)

    try:
        recorded_cost_path = cost_npy_path.relative_to(output_path.parent).as_posix()
    except ValueError:
        recorded_cost_path = str(cost_npy_path)

    save_dict = {
        "cost_matrix_path": recorded_cost_path,
        "model_path": str(Path(model_path).expanduser().resolve()),
        "text_embeddings_path": str(Path(text_embeddings_path).expanduser().resolve()),
        "pano_id_to_lm_rows": raw.pano_id_to_lm_rows,
        "pano_lm_tags": raw.pano_lm_tags,
        "osm_lm_indices": raw.osm_lm_indices,
        "osm_lm_tags": raw.osm_lm_tags,
        "identity": raw.identity,
    }
    torch.save(save_dict, output_path, pickle_protocol=4)
    print(f"Saved metadata to {output_path}")
    print(f"Raw cost data saved: {cost_npy_path} + {output_path}")


def load_raw_cost_data(raw_path: Path) -> cm.RawCorrespondenceData:
    return cm.load_raw_cost_data(raw_path)


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
    parser.add_argument(
        "--satellite_dir", type=Path, default=None,
        help="External satellite payload directory (default: dataset layout).",
    )
    parser.add_argument(
        "--landmark_path", type=Path, default=None,
        help="External landmark Feather/GeoJSON path (default: dataset layout).",
    )
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
    parser.add_argument(
        "--allow_legacy_raw_identity", action="store_true",
        help="Explicitly aggregate an independently audited legacy raw "
             "artifact that predates source/alignment identity metadata.",
    )
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
    parser.add_argument("--stream_cost_matrix", action="store_true",
                        help="Stream the cost matrix directly to a memmapped "
                             ".npy on disk during precompute, instead of "
                             "accumulating rows in RAM and vstack'ing at the "
                             "end. Use when the cost matrix would exceed "
                             "available RAM.")
    args = parser.parse_args()

    dataset_path = args.dataset_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stage 1: obtain RawCorrespondenceData — either by loading or by running inference.
    if args.from_raw is not None:
        raw = load_raw_cost_data(args.from_raw.expanduser().resolve())
    else:
        raw = build_raw_cost_data(args)
        save_raw_cost_data(raw, output_path, args.model_path, args.text_embeddings_path)

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
    dataset = load_vigor_dataset(
        dataset_path, args.landmark_version, args.inflation_factor,
        args.satellite_dir, args.landmark_path)
    landmark_path = _resolve_landmark_path(
        dataset_path, args.landmark_version, args.landmark_path)
    validate_raw_identity(
        raw, dataset, dataset_path, landmark_path,
        allow_legacy=args.allow_legacy_raw_identity,
        require_identity=True)
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
    identity_path = sim_path.with_suffix(".json")
    identity_path.write_text(json.dumps({
        "matrix_identity": vd.similarity_matrix_identity(
            dataset._panorama_metadata, dataset._satellite_metadata),
        "method": args.method,
        "aggregation": args.aggregation,
        "prob_threshold": args.prob_threshold,
        "uniqueness_weighted": args.uniqueness_weighted,
        "use_dustbin": not args.no_dustbin,
    }, indent=2) + "\n")
    print(f"\nSaved similarity matrix {similarity.shape} to {sim_path}")
    print(f"Saved ordered matrix identity to {identity_path}")


if __name__ == "__main__":
    main()
