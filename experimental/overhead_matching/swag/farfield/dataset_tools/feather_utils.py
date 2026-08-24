"""Shared helpers for compact landmark feathers.

All Feather reading and construction goes through `farfield.catalog.schema`,
the only reader in the farfield tree. Catalogs have four required compact
columns and may retain explicitly allowed structural source metadata. ENC's
``object_class`` is also mirrored in canonical JSON ``tags`` for tag-only
consumers. Geometry constants come from `farfield.geometry` (one owner per
convention).
"""

import json
import math

import geopandas as gpd
import pandas as pd
import shapely

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.catalog import schema

# Tags that carry source provenance rather than observable identity. They are
# persisted in the compact tag object, but deliberately excluded when deciding
# whether two source features describe the same physical landmark.
SOURCE_RECORDS_TAG = "source_records"
DEDUPE_METADATA_TAGS = frozenset({"object_class", SOURCE_RECORDS_TAG})


def tag_signatures(gdf: gpd.GeoDataFrame) -> list[tuple]:
    """Hashable tag set per row: populated string tags only, sorted.

    ``object_class`` is decoded from the compact tags object and excluded as
    source provenance, regardless of its mirrored structural column. Thus an
    ENC layer split does not prevent otherwise identical, touching features
    from deduplicating.
    """
    return [
        tuple(sorted((k, v) for k, v in props.items()
                     if k not in DEDUPE_METADATA_TAGS
                     and isinstance(v, str) and v))
        for props in schema.tag_dicts(gdf)
    ]


def _cluster_by_proximity(geometries: list, tolerance_deg: float) -> list[int]:
    """Union-find cluster of geometries that touch or nearly touch.

    Returns a cluster label per geometry.
    """
    parent = list(range(len(geometries)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    tree = shapely.STRtree(geometries)
    for i, geom in enumerate(geometries):
        for j in tree.query(geom.buffer(tolerance_deg)):
            if j != i and geometries[j].distance(geom) <= tolerance_deg:
                union(i, int(j))
    return [find(i) for i in range(len(geometries))]


def dedupe_exact_duplicates(
    gdf: gpd.GeoDataFrame, tolerance_m: float, verbose: bool = True,
) -> gpd.GeoDataFrame:
    """Collapse features that are the same physical thing recorded twice.

    A duplicate is a feature with an *identical tag set* whose geometry
    touches (within `tolerance_m` of) another such feature — e.g. one bridge
    stored as four abutting segments, or an island clipped at an ENC cell
    boundary into two polygons. Each such cluster becomes a single feature
    with a unioned geometry. Cross-source clusters prefer ENC explicitly and
    retain every merged record's id/source in the non-evidence
    ``source_records`` metadata tag; input ordering never chooses the winner.

    Proximity is required, not just matching tags: 1638 distinct Boston piers
    all carry `man_made=pier`, and two unrelated islands 13 km apart are both
    named "Deer Island". Tag-only dedup would erase them.
    """
    if len(gdf) == 0:
        return gdf
    tolerance_deg = tolerance_m / geo.METERS_PER_DEG_LAT

    by_signature: dict[tuple, list[int]] = {}
    for position, signature in enumerate(tag_signatures(gdf)):
        by_signature.setdefault(signature, []).append(position)

    keep_positions: list[int] = []
    merged_geometry: dict[int, object] = {}
    merged_provenance: dict[int, str] = {}
    n_dropped = 0
    n_repaired = 0
    n_union_failed = 0
    for positions in by_signature.values():
        if len(positions) == 1:
            keep_positions.append(positions[0])
            continue
        geometries = [gdf.geometry.iloc[p] for p in positions]
        clusters: dict[int, list[int]] = {}
        for local, label in enumerate(
                _cluster_by_proximity(geometries, tolerance_deg)):
            clusters.setdefault(label, []).append(local)
        for members in clusters.values():
            absolute = [positions[member] for member in members]
            sources = {str(gdf["landmark_type"].iloc[p]) for p in absolute}
            preferred_source = "enc" if "enc" in sources else min(sources)
            preferred = [p for p in absolute
                         if gdf["landmark_type"].iloc[p] == preferred_source]
            first = min(preferred,
                        key=lambda p: (str(gdf["id"].iloc[p]), p))
            keep_positions.append(first)
            if len(members) > 1:
                # A cross-source duplicate keeps the preferred source's
                # geometry as well as its identity. Unioning a less accurate
                # OSM point/shape into an ENC feature would undo the choice.
                geometry_positions = (preferred if len(sources) > 1
                                      else absolute)
                parts = [gdf.geometry.iloc[p] for p in geometry_positions]
                # OSM polygons are not always valid (self-touching rings are
                # common), and GEOS raises a TopologyException rather than
                # degrading. Repair on demand instead of paying make_valid on
                # every geometry: on a NY+NJ merge only a handful of the 3.2M
                # clusters need it.
                try:
                    union = shapely.union_all(parts)
                except shapely.errors.GEOSException:
                    try:
                        union = shapely.union_all(
                            [shapely.make_valid(p) for p in parts])
                        n_repaired += 1
                    except shapely.errors.GEOSException:
                        # Keeping the first member is the conservative
                        # outcome: the cluster is duplicates of one landmark,
                        # so its geometry is representative even un-unioned.
                        union = parts[0]
                        n_union_failed += 1
                merged_geometry[first] = union
                if len(sources) > 1:
                    records = sorted(
                        ({"id": str(gdf["id"].iloc[p]),
                          "landmark_type": str(
                              gdf["landmark_type"].iloc[p])}
                         for p in absolute),
                        key=lambda record: (record["landmark_type"],
                                            record["id"]))
                    merged_provenance[first] = json.dumps(
                        records, sort_keys=True, separators=(",", ":"))
                n_dropped += len(members) - 1

    keep_positions.sort()
    out = gdf.iloc[keep_positions].copy()
    if merged_geometry:
        geometry = out.geometry.tolist()
        for offset, position in enumerate(keep_positions):
            if position in merged_geometry:
                geometry[offset] = merged_geometry[position]
        out = out.set_geometry(
            gpd.GeoSeries(geometry, index=out.index, crs=gdf.crs))
    if merged_provenance:
        decoded = schema.tag_dicts(out)
        for offset, position in enumerate(keep_positions):
            if position not in merged_provenance:
                continue
            tags = dict(decoded[offset])
            tags[SOURCE_RECORDS_TAG] = merged_provenance[position]
            out.iloc[offset, out.columns.get_loc(schema.TAGS_COLUMN)] = (
                json.dumps(tags, sort_keys=True, separators=(",", ":")))
        # Keep the source-build helper behind the same strict persisted schema
        # boundary as every catalog reader.
        schema.tag_dicts(out)
    if verbose:
        print(f"Dedupe (identical tags, geometries within {tolerance_m:g} m):"
              f" {len(gdf)} -> {len(out)} landmarks ({n_dropped} merged away)")
        if n_repaired or n_union_failed:
            print(f"  {n_repaired} cluster(s) needed make_valid before "
                  f"union; {n_union_failed} kept a representative member "
                  f"because the union failed even after repair")
    return out.reset_index(drop=True)


def merge_feathers(frames: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Concatenate landmark frames, taking the union of their tag columns."""
    for frame in frames:
        if "pruned_props" in frame.columns:
            raise ValueError(
                "input has a pruned_props column; it must be computed at "
                "load time, never stored")
        if frame.crs is None or frame.crs.to_string() != "EPSG:4326":
            raise ValueError(f"expected EPSG:4326, got {frame.crs}")
        for column in ("id", "landmark_type"):
            if column not in frame.columns:
                raise ValueError(f"input is missing the {column!r} column")

    merged = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True, sort=False), crs="EPSG:4326")

    # The same OSM feature legitimately appears in two overlapping extracts:
    # Geofabrik ships complete ways, so anything crossing a border -- the
    # Channel Tunnel, submarine cables, maritime boundaries -- is in both
    # national files. It is one landmark, so collapse it rather than refusing
    # the merge. Each copy is clipped to its own extract, so keep whichever
    # retained more geometry.
    duplicated = merged["id"].duplicated(keep=False)
    if duplicated.any():
        # Only same-source duplicates are a border artifact. The id schemes
        # are source-prefixed -- ('way', 123) versus ('enc', 'LNAM') -- so one
        # id arriving with two different landmark_types means the schemes
        # have collided, which silently collapsing would hide.
        per_id_sources = merged.loc[duplicated].groupby(
            "id")["landmark_type"].nunique()
        conflicting = per_id_sources[per_id_sources > 1]
        if len(conflicting):
            raise ValueError(
                f"{len(conflicting)} id(s) appear with more than one "
                f"landmark_type, e.g. {list(conflicting.index[:3])}; the "
                f"source-prefixed id schemes have collided")
        counts = shapely.get_num_coordinates(
            shapely.from_wkb(merged.geometry.to_wkb()))
        order = pd.Series(counts, index=merged.index)
        keep = order.groupby(merged["id"]).idxmax()
        drop = merged.index.difference(pd.Index(keep))
        n_ids = int(merged.loc[duplicated, "id"].nunique())
        print(f"  collapsed {len(drop)} duplicate row(s) spanning {n_ids} "
              f"id(s) present in more than one input (cross-border "
              f"features); kept the copy with the most complete geometry")
        merged = merged.drop(index=drop).reset_index(drop=True)

    duplicate_ids = merged["id"].duplicated().sum()
    if duplicate_ids:
        raise ValueError(
            f"{duplicate_ids} duplicate id values across inputs; ids must be "
            "globally unique so cost-matrix columns stay addressable")
    return merged


def report_cross_source_collisions(
    gdf: gpd.GeoDataFrame, radius_m: float, limit: int = 15,
) -> None:
    """Print same-name landmarks from different sources that sit close by.

    These are not removed: an OSM and an ENC record of the same lighthouse
    usually carry different tags, and which one matches better is the
    correspondence model's call, not ours.
    """
    # Names live inside the compact tags object. Assign before slicing, or the
    # slice has no name column to group by.
    names = pd.Series([t.get("name") for t in schema.tag_dicts(gdf)],
                      index=gdf.index)
    if names.isna().all():
        return
    gdf = gdf.assign(name=names)
    named = gdf[gdf["name"].notna()]
    points = {i: named.geometry[i].representative_point()
              for i in named.index}
    collisions = []
    for name, group in named.groupby("name"):
        sources = group["landmark_type"].unique()
        if len(sources) < 2 or len(group) < 2:
            continue
        indices = list(group.index)
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                if named["landmark_type"][i] == named["landmark_type"][j]:
                    continue
                p, q = points[i], points[j]
                distance = math.hypot(
                    (q.x - p.x) * geo.METERS_PER_DEG_LAT
                    * math.cos(math.radians(p.y)),
                    (q.y - p.y) * geo.METERS_PER_DEG_LAT)
                if distance <= radius_m:
                    collisions.append((distance, name,
                                       named["landmark_type"][i],
                                       named["landmark_type"][j]))
    if not collisions:
        print(f"No same-name cross-source landmarks within {radius_m:g} m")
        return
    collisions.sort()
    print(f"{len(collisions)} same-name cross-source pairs within "
          f"{radius_m:g} m (kept, not merged):")
    for distance, name, left, right in collisions[:limit]:
        print(f"  {distance:6.1f} m  {name}  [{left} | {right}]")
    if len(collisions) > limit:
        print(f"  ... and {len(collisions) - limit} more")


def bbox_from_dataset(dataset_base) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a dataset's trajectory bbox.

    Reads the canonical `pipeline_metadata.json` bbox written by ingest and the
    collection pipeline. Callers may provide an explicit bbox when that record
    is unavailable.
    """
    import json
    from pathlib import Path

    dataset_base = Path(dataset_base)
    meta_path = dataset_base / "pipeline_metadata.json"
    if not meta_path.is_file() or meta_path.is_symlink():
        raise FileNotFoundError(
            f"{dataset_base} has no regular pipeline_metadata.json; pass an "
            "explicit --bbox instead")
    document = json.loads(meta_path.read_text())
    if not isinstance(document, dict) or not isinstance(document.get("bbox"), dict):
        raise ValueError(f"{meta_path} must contain an object-valued bbox")
    bbox = document["bbox"]
    try:
        values = tuple(float(bbox[key]) for key in ("west", "south", "east", "north"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{meta_path} has an invalid bbox: {exc}") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{meta_path} bbox values must be finite")
    west, south, east, north = values
    if not (-180.0 <= west < east <= 180.0 and -90.0 <= south < north <= 90.0):
        raise ValueError(f"{meta_path} bbox is outside WGS84 bounds or unordered")
    return values


def buffered_bbox(bbox: tuple[float, float, float, float],
                  buffer_frac: float) -> tuple[float, float, float, float]:
    """Grow a bbox by `buffer_frac` of its own span on every side."""
    west, south, east, north = bbox
    width, height = east - west, north - south
    return (west - buffer_frac * width, south - buffer_frac * height,
            east + buffer_frac * width, north + buffer_frac * height)
