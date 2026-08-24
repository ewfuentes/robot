"""Merge landmark feathers from several sources into one catalog feather.

Takes the union of tag columns, requires globally unique `id` values (so
cost-matrix columns stay addressable back to a source row), and collapses
exact duplicates — identical tag sets whose geometries touch — across all
inputs.

`landmark_type` carries the provenance ("osm", "enc" for NOAA charts) and is
frame metadata, not a tag, so it never reaches the matcher.

All Feather reading goes through `farfield.catalog.schema`. This source-build
tool writes a compact Feather plus a provenance sidecar recording its exact
inputs; publish the selected result separately as a typed CATALOGS artifact.

Example:
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:merge_landmark_feathers -- \\
        --inputs /data/.../sources/osm_harbor_v1.feather \\
                 /data/.../sources/enc_harbor_v1.feather \\
        --output /data/.../catalog_sources/merged_v1 \\
        --dedupe_tolerance_m 10
"""

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd

from experimental.overhead_matching.swag.farfield import provenance
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.dataset_tools.feather_utils import (  # noqa: E501
    dedupe_exact_duplicates,
    merge_feathers,
    report_cross_source_collisions,
)


def main(inputs: list, output: Path, dedupe_tolerance_m: float,
         collision_radius_m: float) -> gpd.GeoDataFrame:
    frames = []
    for path in inputs:
        frame = schema.read_frame(path)
        counts = frame["landmark_type"].value_counts().to_dict()
        print(f"{path.name}: {schema.summarize(frame)}, "
              f"landmark_type={counts}")
        frames.append(frame)

    merged = merge_feathers(frames)
    print(f"\nMerged: {len(merged)} landmarks, {len(merged.columns)} columns")
    if dedupe_tolerance_m > 0:
        merged = dedupe_exact_duplicates(merged, dedupe_tolerance_m)
    print()
    report_cross_source_collisions(merged, collision_radius_m)

    print(f"\nBy source: {merged['landmark_type'].value_counts().to_dict()}")
    named = sum(1 for t in schema.tag_dicts(merged) if t.get("name"))
    print(f"Named: {named}")

    output = output.with_suffix(".feather")
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_feather(output)
    sidecar = output.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps({
        "tool": "farfield/dataset_tools/merge_landmark_feathers.py",
        "git_commit": provenance.git_commit(),
        "argv": list(sys.argv),
        "arguments": {
            "inputs": [str(p) for p in inputs],
            "output": str(output),
            "dedupe_tolerance_m": dedupe_tolerance_m,
            "collision_radius_m": collision_radius_m,
        },
        "rows_in": {str(p): int(len(f)) for p, f in zip(inputs, frames)},
        "rows_out": int(len(merged)),
    }, indent=1))
    print(f"Wrote {output}")
    print(f"      {sidecar}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="output feather (typically under "
                             "artifacts/catalogs/<dataset>/)")
    # The tolerance decides which rows the output contains, so it is required
    # (REORG.md rule 2); the previous default was 10.0 m.
    parser.add_argument("--dedupe_tolerance_m", type=float, required=True,
                        help="Merge identical-tag features whose geometries "
                             "are within this distance; 0 disables "
                             "(previously 10.0)")
    parser.add_argument("--collision_radius_m", type=float, default=150.0,
                        help="Radius for the same-name cross-source report "
                             "(report only; changes nothing in the output)")
    args = parser.parse_args()
    main(args.inputs, args.output, args.dedupe_tolerance_m,
         args.collision_radius_m)
