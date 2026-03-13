"""Export parquet tables for tag weight optimization.

Builds `{city}_pano_osm_matches.parquet` and `{city}_sat_osm_table.parquet`
by calling `create_osm_tag_extraction_dataset` and `compute_osm_tag_match_similarity`
from `osm_tag_similarity.py`, which produces the match tables as a side effect.
"""

import argparse
from pathlib import Path

import polars as pl

import common.torch.load_torch_deps
from experimental.overhead_matching.swag.data import vigor_dataset as vd
from experimental.overhead_matching.swag.evaluation import osm_tag_similarity as ots


def main():
    parser = argparse.ArgumentParser(description="Export landmark tables for tag weight optimization")
    parser.add_argument("--dataset_path", type=Path, required=True,
                        help="Path to VIGOR city dir, e.g. /data/.../VIGOR/Chicago")
    parser.add_argument("--extraction_path", type=Path, required=True,
                        help="Path to pano extraction dir containing JSONL predictions, "
                             "e.g. /data/.../pano_v2/Chicago/sentences/")
    parser.add_argument("--landmark_version", default="v4_202001")
    parser.add_argument("--output_dir", type=Path, default=Path("~/scratch/landmark_tables"))
    parser.add_argument("--city_name", default=None,
                        help="City name for output filenames (default: inferred from dataset_path)")
    parser.add_argument("--inflation_factor", type=float, default=1.0,
                        help="Satellite patch inflation factor for attracting nearby landmarks (default: 1.0)")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    city_name = args.city_name or args.dataset_path.name.lower()

    # Load VIGOR dataset with landmarks
    config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        landmark_version=args.landmark_version,
        landmark_correspondence_inflation_factor=args.inflation_factor,
    )
    dataset = vd.VigorDataset(args.dataset_path, config)
    print(f"Loaded {city_name}: {len(dataset._panorama_metadata)} panos, "
          f"{len(dataset._satellite_metadata)} sats, "
          f"{len(dataset._landmark_metadata)} landmarks")

    # Build dataset and compute matches (produces tables as side effect)
    osm_dataset = ots.create_osm_tag_extraction_dataset(args.extraction_path, dataset)
    print(f"Loaded {len(osm_dataset['pano_data_from_pano_id'])} panoramas with extractions")

    _, (matches, osm_to_sat, _schema) = ots.compute_osm_tag_match_similarity(osm_dataset)

    # Add lat/lon to osm_to_sat from satellite metadata
    sat_coords = pl.DataFrame({
        "sat_idx": list(range(len(dataset._satellite_metadata))),
        "lat": dataset._satellite_metadata["lat"].tolist(),
        "lon": dataset._satellite_metadata["lon"].tolist(),
    }, schema={"sat_idx": pl.Int32, "lat": pl.Float64, "lon": pl.Float64})
    osm_to_sat = osm_to_sat.join(sat_coords, on="sat_idx", how="left")

    # Write tables
    sat_osm_path = output_dir / f"{city_name}_sat_osm_table.parquet"
    osm_to_sat.write_parquet(sat_osm_path)
    print(f"Wrote {len(osm_to_sat)} rows to {sat_osm_path}")

    matches_path = output_dir / f"{city_name}_pano_osm_matches.parquet"
    matches.write_parquet(matches_path)
    print(f"Wrote {len(matches)} rows to {matches_path}")


if __name__ == "__main__":
    main()
