"""Append a typed source Feather to a published catalog as a new derived
CATALOGS artifact.

A stage-5 full catalog is OSM (+ENC) only. This tool adds the rows of one more
source Feather -- for example Overture Places from
`extract_landmarks_from_overture` -- to a published CATALOGS artifact and
publishes the union as a new derived catalog whose single catalog upstream is
the input. `catalog.lineage.require_passed_source_coverage` therefore still
terminates at the full catalog's coverage attestation, and `trim_catalog`
consumes the result like any other catalog.

A source row that names something the catalog already has is not added. Two
rows are the same thing when any of their name-like tags agree after
normalisation (NFKC, casefold, everything but letters and digits removed) and
the source point lies within --dedupe_name_radius_m of the catalog geometry.
The same rule collapses duplicates inside the source itself; the source
Feather's row order decides the survivor (the Overture extractor writes
highest confidence first). Every dropped pair is recorded in the manifest
config, so what the catalog did not gain is auditable.

Example:
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:add_catalog_source -- \\
        --input_catalog_dir /data/farfield_matching/artifacts/catalogs/pohang_canal_04/stage3_d92f15c_full_v1 \\
        --source_feather /data/farfield_matching/raw_material/catalog_sources/pohang_canal_04/overture_2026-08-19.0_v1.feather \\
        --dedupe_name_radius_m 150 \\
        --output_dir /data/farfield_matching/artifacts/catalogs/pohang_canal_04/stage3_<commit>_full_overture_v1
"""

import argparse
import math
import sys
import unicodedata
from pathlib import Path

import geopandas as gpd
import shapely

from experimental.overhead_matching.swag.farfield import (
    artifact,
    geometry as geo,
    provenance,
    publication,
)
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    feather_utils,
    source_publication,
)
from experimental.overhead_matching.swag.farfield.dataset_tools.landmark_positive_set import (  # noqa: E501
    open_catalog_artifact,
)

GENERATOR = "farfield/dataset_tools/add_catalog_source.py"
NAME_KEYS = frozenset({"name", "alt_name", "official_name", "short_name",
                       "loc_name", "old_name", "brand"})


def normalised_names(tags: dict) -> set[str]:
    """Every name-like value of a tag bundle, reduced to letters and digits."""
    names = set()
    for key, value in tags.items():
        if key not in NAME_KEYS and not key.startswith("name:"):
            continue
        for part in str(value).split(";"):
            folded = unicodedata.normalize("NFKC", part).casefold()
            compact = "".join(ch for ch in folded if ch.isalnum())
            if compact:
                names.add(compact)
    return names


def _metres_between(point, geometry) -> float:
    """Distance from a WGS84 point to a WGS84 geometry, in metres."""
    scale = geo.METERS_PER_DEG_LAT * math.cos(math.radians(point.y))

    def local(coords):
        out = coords.copy()
        out[:, 0] = (coords[:, 0] - point.x) * scale
        out[:, 1] = (coords[:, 1] - point.y) * geo.METERS_PER_DEG_LAT
        return out

    return float(shapely.distance(shapely.Point(0.0, 0.0),
                                  shapely.transform(geometry, local)))


class _NameIndex:
    """Rows by normalised name, for radius-limited duplicate lookup."""

    def __init__(self):
        self._rows: dict[str, list[tuple[str, object, str]]] = {}

    def add(self, row_id: str, geometry, tags: dict) -> None:
        for name in normalised_names(tags):
            self._rows.setdefault(name, []).append((row_id, geometry, name))

    def nearest_duplicate(self, point, tags: dict, radius_m: float):
        best = None
        for name in normalised_names(tags):
            for row_id, geometry, _ in self._rows.get(name, ()):
                distance = _metres_between(point, geometry)
                if distance <= radius_m and (best is None
                                             or distance < best[2]):
                    best = (row_id, name, distance)
        return best


def _pair(source_id: str, match, source_name: str) -> dict:
    return {"source_id": source_id, "duplicate_of": match[0],
            "name": match[1], "source_name": source_name,
            "distance_m": round(match[2], 1)}


def select_new_rows(base: gpd.GeoDataFrame, source: gpd.GeoDataFrame,
                    radius_m: float) -> tuple[list[int], list[dict], list[dict]]:
    """Positions of source rows to add, plus the two duplicate ledgers."""
    base_tags = schema.tag_dicts(base)
    source_tags = schema.tag_dicts(source)
    catalog_index = _NameIndex()
    for position in range(len(base)):
        catalog_index.add(str(base["id"].iloc[position]),
                          base.geometry.iloc[position], base_tags[position])
    source_index = _NameIndex()
    keep, of_catalog, within_source = [], [], []
    for position in range(len(source)):
        row_id = str(source["id"].iloc[position])
        tags = source_tags[position]
        point = shapely.centroid(source.geometry.iloc[position])
        match = catalog_index.nearest_duplicate(point, tags, radius_m)
        if match is not None:
            of_catalog.append(_pair(row_id, match, tags.get("name", "")))
            continue
        match = source_index.nearest_duplicate(point, tags, radius_m)
        if match is not None:
            within_source.append(_pair(row_id, match, tags.get("name", "")))
            continue
        keep.append(position)
        source_index.add(row_id, source.geometry.iloc[position], tags)
    return keep, of_catalog, within_source


def main(input_catalog_dir: Path, source_feather: Path, output_dir: Path,
         dedupe_name_radius_m: float) -> gpd.GeoDataFrame:
    if not (math.isfinite(dedupe_name_radius_m) and dedupe_name_radius_m >= 0):
        raise SystemExit("--dedupe_name_radius_m must be finite and >= 0")
    try:
        input_ref, input_path = open_catalog_artifact(input_catalog_dir)
    except artifact.ArtifactError as exc:
        raise SystemExit(f"invalid input catalog artifact: {exc}") from exc
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise SystemExit(
            f"{output_dir} already exists; catalogs are immutable, publish a "
            "new version")
    source_feather = Path(source_feather)
    sidecar = source_publication.output_paths(source_feather)[1]
    try:
        source, source_document = source_publication.validate_completed_pair(
            source_feather, sidecar)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"invalid source Feather: {exc}") from exc
    source_sha256 = source_document["output_sha256"]
    base = schema.read_frame(input_path)

    keep, of_catalog, within_source = select_new_rows(
        base, source, dedupe_name_radius_m)
    added = source.iloc[keep].reset_index(drop=True)
    merged = feather_utils.merge_feathers([base, added])
    print(f"{len(base)} catalog rows + {len(source)} source rows -> "
          f"{len(merged)} ({len(added)} added; {len(of_catalog)} duplicate "
          f"the catalog, {len(within_source)} duplicate the source)")
    for pair in sorted(of_catalog, key=lambda p: p["distance_m"])[:15]:
        print(f"  {pair['distance_m']:6.1f} m  {pair['source_name']}  "
              f"[{pair['source_id']} = {pair['duplicate_of']}]")

    config = {
        "source_feather": str(source_feather.resolve()),
        "source_feather_sha256": source_sha256,
        "source_provenance_sha256": artifact.sha256_file(sidecar),
        "source_landmark_types": sorted(
            set(source["landmark_type"].astype(str))),
        "dedupe_name_radius_m": float(dedupe_name_radius_m),
        "rows_in_catalog": int(len(base)),
        "rows_in_source": int(len(source)),
        "rows_added": int(len(added)),
        "rows_out": int(len(merged)),
        "duplicates_of_catalog": of_catalog,
        "duplicates_within_source": within_source,
    }
    if artifact.sha256_file(source_feather) != source_sha256:
        raise SystemExit("source Feather changed during the merge; refusing "
                         "to publish")
    with publication.published_artifact(
            output_dir,
            kind=paths_lib.CATALOGS,
            dataset=input_ref.dataset,
            version=output_dir.name,
            generator=GENERATOR,
            git_commit=provenance.git_commit(),
            arguments=list(sys.argv),
            upstreams=(input_ref,),
            config=config,
            declared_outputs=("catalog.feather",)) as builder:
        merged.to_feather(builder.output_path("catalog.feather"))
    print(f"Wrote {output_dir}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_catalog_dir", required=True, type=Path,
                        help="published CATALOGS artifact to extend")
    parser.add_argument("--source_feather", required=True, type=Path,
                        help="completed source Feather (with its "
                             ".provenance.json sidecar) to append")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="new immutable CATALOGS artifact directory")
    parser.add_argument("--dedupe_name_radius_m", required=True, type=float,
                        help="a source row within this distance of a "
                             "same-named catalog row is a duplicate")
    args = parser.parse_args()
    main(args.input_catalog_dir, args.source_feather, args.output_dir,
         args.dedupe_name_radius_m)
