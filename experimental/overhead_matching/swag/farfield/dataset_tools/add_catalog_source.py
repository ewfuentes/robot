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

Nameless sources (FAA obstacles) cannot be deduplicated by name, so
--class_merge_radius_m enables a second rule: a source row whose structure
class (`structure_class`: tower, chimney, tank, building, solar, ...) is
compatible with a catalog row of that class within the radius is MERGED into
it -- the catalog row keeps its identity, geometry and every tag it already
has, and gains the source's tags it lacked (for FAA: `height` and the `faa:*`
facts). Pairs are assigned nearest-first across the whole source (an antenna
farm with three OSM masts and three FAA rows resolves to three pairs), and a
catalog row absorbs at most one source row, so a second obstacle beside an
already-merged tower is added as its own row. A merge with more than one
candidate in radius is recorded as `contested` with the alternatives and
their distances, so the review can judge it. --class_merge_tolerance_tag names
a source tag holding a +- position tolerance in metres; the search radius for
that row is the larger of the fixed radius and twice the tolerance (the
tolerance's full width).

Example:
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:add_catalog_source -- \\
        --input_catalog_dir /data/farfield_matching/artifacts/catalogs/pohang_canal_04/stage3_d92f15c_full_v1 \\
        --source_feather /data/farfield_matching/raw_material/catalog_sources/pohang_canal_04/overture_2026-08-19.0_v1.feather \\
        --dedupe_name_radius_m 150 \\
        --output_dir /data/farfield_matching/artifacts/catalogs/pohang_canal_04/stage3_<commit>_full_overture_v1

    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:add_catalog_source -- \\
        --input_catalog_dir .../catalogs/portland_flight_20260906_leg1/osm_20260910_full_v1 \\
        --source_feather .../catalog_sources/portland_flight_20260906/faa_dof_20260802_v1.feather \\
        --dedupe_name_radius_m 150 \\
        --class_merge_radius_m 50 --class_merge_tolerance_tag faa:position_tolerance_m \\
        --output_dir .../catalogs/portland_flight_20260906_leg1/osmfaa_20260910_full_v1
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
# Structure classes that may merge across class. Everything else merges only
# with its own class.
COMPATIBLE_CLASSES = {"tower": frozenset({"tower", "building"}),
                      "building": frozenset({"building", "tower"})}
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


def structure_class(tags: dict) -> str | None:
    """The kind of tall structure a tag bundle describes, or None.

    Shared by both sides of a class merge, so an FAA `man_made=tower` and an
    OSM `man_made=communications_tower` compare as the same class.
    """
    man_made = tags.get("man_made")
    if man_made in ("tower", "mast", "communications_tower", "antenna"):
        return "tower"
    if man_made == "chimney":
        return "chimney"
    if man_made in ("storage_tank", "water_tower", "silo", "gasometer"):
        return "tank"
    if man_made in ("crane", "cooling_tower", "works", "lighthouse",
                    "offshore_platform"):
        return man_made
    if man_made == "bridge" or tags.get("bridge") not in (None, "no"):
        return "bridge"
    power = tags.get("power")
    if power in ("tower", "portal"):
        return "power_tower"
    if power == "line":
        return "power_line"
    if power in ("plant", "generator"):
        source = tags.get("plant:source") or tags.get("generator:source")
        if source in ("solar", "wind"):
            return source
        return "power_plant"
    if tags.get("waterway") == "dam":
        return "dam"
    if tags.get("landuse") == "landfill":
        return "landfill"
    if tags.get("historic") in ("monument", "memorial"):
        return "monument"
    if tags.get("building") not in (None, "no"):
        return "building"
    return None


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


class _ClassIndex:
    """Catalog rows by structure class, for radius-limited class matching."""

    def __init__(self, geometries, tags: list[dict]):
        self._geometries = geometries
        self._classes = [structure_class(t) for t in tags]
        self._tree = shapely.STRtree(geometries)

    def class_of(self, position: int) -> str | None:
        return self._classes[position]

    def candidates(self, point, source_class: str, radius_m: float
                   ) -> list[tuple[int, float]]:
        """(catalog position, distance_m) of compatible rows within radius."""
        wanted = COMPATIBLE_CLASSES.get(source_class, frozenset({source_class}))
        dlat = radius_m / geo.METERS_PER_DEG_LAT
        dlon = dlat / max(1e-6, math.cos(math.radians(point.y)))
        box = shapely.box(point.x - dlon, point.y - dlat,
                          point.x + dlon, point.y + dlat)
        found = []
        for position in self._tree.query(box):
            position = int(position)
            if self._classes[position] not in wanted:
                continue
            distance = _metres_between(point, self._geometries[position])
            if distance <= radius_m:
                found.append((position, distance))
        return sorted(found, key=lambda item: item[1])


def _class_radius(tags: dict, radius_m: float, tolerance_tag: str | None
                  ) -> float:
    if tolerance_tag is None or tolerance_tag not in tags:
        return radius_m
    try:
        tolerance = float(tags[tolerance_tag])
    except ValueError as exc:
        raise ValueError(
            f"{tolerance_tag}={tags[tolerance_tag]!r} is not a number") from exc
    return max(radius_m, 2.0 * tolerance)


def _pair(source_id: str, match, source_name: str) -> dict:
    return {"source_id": source_id, "duplicate_of": match[0],
            "name": match[1], "source_name": source_name,
            "distance_m": round(match[2], 1)}


def select_new_rows(base: gpd.GeoDataFrame, source: gpd.GeoDataFrame,
                    radius_m: float, class_merge_radius_m: float | None = None,
                    class_merge_tolerance_tag: str | None = None) -> dict:
    """Decide every source row: add, name-duplicate, class-merge, ambiguous.

    Returns {"keep": [source positions], "duplicates_of_catalog": [...],
    "duplicates_within_source": [...], "merged": [...],
    "base_tags": [per-row catalog tags after merging]}.

    Class merging is a two-pass assignment: every eligible source row lists
    its compatible catalog candidates, then (source, catalog) pairs are taken
    nearest-first with each side used at most once. A source row whose every
    candidate went to a nearer source row is appended like any new row.
    """
    base_tags = schema.tag_dicts(base)
    source_tags = schema.tag_dicts(source)
    catalog_index = _NameIndex()
    for position in range(len(base)):
        catalog_index.add(str(base["id"].iloc[position]),
                          base.geometry.iloc[position], base_tags[position])
    class_index = (_ClassIndex(base.geometry.values, base_tags)
                   if class_merge_radius_m is not None else None)
    source_index = _NameIndex()
    keep, of_catalog, within_source, merged = [], [], [], []
    # Pass 1: name duplicates out; class candidates for the rest.
    remaining: list[int] = []
    candidate_lists: dict[int, tuple[float, list[tuple[int, float]]]] = {}
    pairs: list[tuple[float, int, int]] = []
    for position in range(len(source)):
        row_id = str(source["id"].iloc[position])
        tags = source_tags[position]
        point = shapely.centroid(source.geometry.iloc[position])
        match = catalog_index.nearest_duplicate(point, tags, radius_m)
        if match is not None:
            of_catalog.append(_pair(row_id, match, tags.get("name", "")))
            continue
        remaining.append(position)
        source_class = structure_class(tags) if class_index else None
        if source_class is None:
            continue
        search_radius = _class_radius(
            tags, class_merge_radius_m, class_merge_tolerance_tag)
        candidates = class_index.candidates(point, source_class, search_radius)
        if candidates:
            candidate_lists[position] = (search_radius, candidates)
            pairs.extend((distance, position, target)
                         for target, distance in candidates)
    # Pass 2: nearest-first one-to-one assignment.
    assigned_source: dict[int, int] = {}
    taken_catalog: set[int] = set()
    for _, position, target in sorted(pairs):
        if position in assigned_source or target in taken_catalog:
            continue
        assigned_source[position] = target
        taken_catalog.add(target)
    for position in remaining:
        row_id = str(source["id"].iloc[position])
        tags = source_tags[position]
        target = assigned_source.get(position)
        if target is not None:
            search_radius, candidates = candidate_lists[position]
            distance = next(d for p, d in candidates if p == target)
            existing = base_tags[target]
            added = sorted(k for k in tags if k not in existing)
            conflicts = {k: [existing[k], tags[k]] for k in tags
                         if k in existing and existing[k] != tags[k]}
            base_tags[target] = {**existing, **{k: tags[k] for k in added}}
            merged.append({
                "source_id": row_id,
                "merged_into": str(base["id"].iloc[target]),
                "class": structure_class(tags),
                "catalog_class": class_index.class_of(target),
                "distance_m": round(distance, 1),
                "radius_m": round(search_radius, 1),
                "contested": len(candidates) > 1,
                "alternatives": [
                    {"id": str(base["id"].iloc[p]),
                     "class": class_index.class_of(p),
                     "distance_m": round(d, 1)}
                    for p, d in candidates if p != target],
                "added_tags": added,
                "conflicting_tags": conflicts,
            })
            continue
        point = shapely.centroid(source.geometry.iloc[position])
        match = source_index.nearest_duplicate(point, tags, radius_m)
        if match is not None:
            within_source.append(_pair(row_id, match, tags.get("name", "")))
            continue
        keep.append(position)
        source_index.add(row_id, source.geometry.iloc[position], tags)
    return {"keep": keep, "duplicates_of_catalog": of_catalog,
            "duplicates_within_source": within_source, "merged": merged,
            "base_tags": base_tags}


def apply_merged_tags(base: gpd.GeoDataFrame, base_tags: list[dict]
                      ) -> gpd.GeoDataFrame:
    """Copy of the catalog with its tags column re-encoded from base_tags.

    Only the tags column changes, so structural columns such as ENC
    `object_class` survive untouched.
    """
    encoded = base[schema.TAGS_COLUMN].tolist()
    current = schema.tag_dicts(base)
    for position, tags in enumerate(base_tags):
        if tags != current[position]:
            encoded[position] = schema.encode_tags(tags, position)
    updated = base.copy()
    updated[schema.TAGS_COLUMN] = encoded
    return updated


def main(input_catalog_dir: Path, source_feather: Path, output_dir: Path,
         dedupe_name_radius_m: float, class_merge_radius_m: float | None = None,
         class_merge_tolerance_tag: str | None = None,
         dry_run: bool = False) -> gpd.GeoDataFrame:
    if not (math.isfinite(dedupe_name_radius_m) and dedupe_name_radius_m >= 0):
        raise SystemExit("--dedupe_name_radius_m must be finite and >= 0")
    if class_merge_radius_m is not None and not (
            math.isfinite(class_merge_radius_m) and class_merge_radius_m >= 0):
        raise SystemExit("--class_merge_radius_m must be finite and >= 0")
    if class_merge_tolerance_tag is not None and class_merge_radius_m is None:
        raise SystemExit("--class_merge_tolerance_tag needs "
                         "--class_merge_radius_m")
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

    decision = select_new_rows(
        base, source, dedupe_name_radius_m,
        class_merge_radius_m=class_merge_radius_m,
        class_merge_tolerance_tag=class_merge_tolerance_tag)
    of_catalog = decision["duplicates_of_catalog"]
    within_source = decision["duplicates_within_source"]
    class_merged = decision["merged"]
    contested = [m for m in class_merged if m["contested"]]
    if class_merged:
        base = apply_merged_tags(base, decision["base_tags"])
    added = source.iloc[decision["keep"]].reset_index(drop=True)
    merged = feather_utils.merge_feathers([base, added])
    print(f"{len(base)} catalog rows + {len(source)} source rows -> "
          f"{len(merged)} ({len(added)} added; {len(of_catalog)} duplicate "
          f"the catalog by name, {len(within_source)} duplicate the source; "
          f"{len(class_merged)} merged into catalog rows by class, "
          f"{len(contested)} of them contested)")
    for pair in sorted(of_catalog, key=lambda p: p["distance_m"])[:15]:
        print(f"  {pair['distance_m']:6.1f} m  {pair['source_name']}  "
              f"[{pair['source_id']} = {pair['duplicate_of']}]")
    for pair in class_merged[:15]:
        print(f"  merge {pair['source_id']} -> {pair['merged_into']} "
              f"({pair['class']}/{pair['catalog_class']}, "
              f"{pair['distance_m']} m, +{len(pair['added_tags'])} tags"
              f"{', conflicts ' + str(pair['conflicting_tags']) if pair['conflicting_tags'] else ''})")
    for row in contested[:15]:
        print(f"  contested {row['source_id']} -> {row['merged_into']}"
              f"@{row['distance_m']}m over "
              + ", ".join(f"{c['id']}@{c['distance_m']}m"
                          for c in row["alternatives"]))
    if dry_run:
        print("dry run: nothing published")
        return merged

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
        "class_merge_radius_m": class_merge_radius_m,
        "class_merge_tolerance_tag": class_merge_tolerance_tag,
        "rows_merged_into_catalog": len(class_merged),
        "rows_merged_contested": len(contested),
        "merged_into_catalog": class_merged,
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
    parser.add_argument("--class_merge_radius_m", type=float, default=None,
                        help="enable class merging: a nameless source row "
                             "with exactly one same-class catalog row within "
                             "this distance is merged into it")
    parser.add_argument("--class_merge_tolerance_tag", default=None,
                        help="source tag holding a +- position tolerance in "
                             "metres; that row's radius is max(radius, 2x "
                             "tolerance)")
    parser.add_argument("--dry_run", action="store_true",
                        help="print the decision ledgers; publish nothing")
    args = parser.parse_args()
    main(args.input_catalog_dir, args.source_feather, args.output_dir,
         args.dedupe_name_radius_m,
         class_merge_radius_m=args.class_merge_radius_m,
         class_merge_tolerance_tag=args.class_merge_tolerance_tag,
         dry_run=args.dry_run)
