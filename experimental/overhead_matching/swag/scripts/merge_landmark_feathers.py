"""Merge landmark feathers from several sources into one pipeline feather.

Takes the union of tag columns, requires globally unique `id` values (so
cost-matrix columns stay addressable back to a source row), and collapses exact
duplicates — identical tag sets whose geometries touch — across all inputs.

`landmark_type` carries the provenance ("historical" for OSM, "enc" for NOAA
charts) and is not in _TAGS_TO_KEEP, so it never reaches the matcher.

Example:
    bazel run //experimental/overhead_matching/swag/scripts:merge_landmark_feathers -- \\
        --inputs /data/.../landmarks/sources/osm_harbor_v1.feather \\
                 /data/.../landmarks/sources/enc_harbor_v1.feather \\
        --output /data/.../landmarks/harbor_osm_enc_v1.feather
"""

import argparse
from pathlib import Path

import geopandas as gpd

from experimental.overhead_matching.swag.scripts.landmark_feather_utils import (
    dedupe_exact_duplicates,
    merge_feathers,
    report_cross_source_collisions,
)


def main(inputs: list[Path], output: Path, dedupe_tolerance_m: float,
         collision_radius_m: float) -> gpd.GeoDataFrame:
    frames = []
    for path in inputs:
        frame = gpd.read_feather(path)
        counts = frame["landmark_type"].value_counts().to_dict()
        print(f"{path.name}: {len(frame)} landmarks, "
              f"{len(frame.columns)} columns, landmark_type={counts}")
        frames.append(frame)

    merged = merge_feathers(frames)
    print(f"\nMerged: {len(merged)} landmarks, {len(merged.columns)} columns")
    if dedupe_tolerance_m > 0:
        merged = dedupe_exact_duplicates(merged, dedupe_tolerance_m)
    print()
    report_cross_source_collisions(merged, collision_radius_m)

    print(f"\nBy source: {merged['landmark_type'].value_counts().to_dict()}")
    if "name" in merged.columns:
        print(f"Named: {merged['name'].notna().sum()}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.with_suffix(".feather")
    merged.to_feather(output)
    print(f"Wrote {output}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dedupe_tolerance_m", type=float, default=10.0,
                        help="Merge identical-tag features whose geometries are "
                             "within this distance (0 disables)")
    parser.add_argument("--collision_radius_m", type=float, default=150.0,
                        help="Radius for the same-name cross-source report")
    args = parser.parse_args()
    main(args.inputs, args.output, args.dedupe_tolerance_m,
         args.collision_radius_m)
