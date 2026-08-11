"""Shared helpers for landmark feathers (the <city>/landmarks/<version>.feather format).

The pipeline format is a GeoDataFrame with `id`, `geometry` (EPSG:4326),
`landmark_type` provenance, and one column per tag key. `pruned_props` is
computed at load time by vigor_dataset.load_landmark_geojson and must never be
stored in the file.
"""

import math

import geopandas as gpd
import pandas as pd
import shapely

# Columns that describe the record rather than the landmark's tags.
META_COLUMNS = ("id", "geometry", "landmark_type", "object_class")

METERS_PER_DEG_LAT = 110574.0


def tag_columns(gdf: gpd.GeoDataFrame) -> list[str]:
    return [c for c in gdf.columns if c not in META_COLUMNS]


def tag_signature(row, columns: list[str]) -> tuple:
    """Hashable tag set for a row: only populated string tags, sorted."""
    return tuple(sorted(
        (c, row[c]) for c in columns if isinstance(row[c], str) and row[c]))


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
    gdf: gpd.GeoDataFrame, tolerance_m: float = 10.0, verbose: bool = True,
) -> gpd.GeoDataFrame:
    """Collapse features that are the same physical thing recorded more than once.

    A duplicate is a feature with an *identical tag set* whose geometry touches
    (within `tolerance_m` of) another such feature — e.g. one bridge stored as
    four abutting segments, or an island clipped at an ENC cell boundary into
    two polygons. Each such cluster becomes a single feature with the unioned
    geometry, keeping the first member's `id`.

    Proximity is required, not just matching tags: 1638 distinct Boston piers
    all carry `man_made=pier`, and two unrelated islands 13 km apart are both
    named "Deer Island". Tag-only dedup would erase them.
    """
    if len(gdf) == 0:
        return gdf
    columns = tag_columns(gdf)
    tolerance_deg = tolerance_m / METERS_PER_DEG_LAT

    by_signature: dict[tuple, list[int]] = {}
    for position, (_, row) in enumerate(gdf.iterrows()):
        by_signature.setdefault(tag_signature(row, columns), []).append(position)

    keep_positions: list[int] = []
    merged_geometry: dict[int, object] = {}
    n_dropped = 0
    for positions in by_signature.values():
        if len(positions) == 1:
            keep_positions.append(positions[0])
            continue
        geometries = [gdf.geometry.iloc[p] for p in positions]
        clusters: dict[int, list[int]] = {}
        for local, label in enumerate(_cluster_by_proximity(geometries, tolerance_deg)):
            clusters.setdefault(label, []).append(local)
        for members in clusters.values():
            first = positions[members[0]]
            keep_positions.append(first)
            if len(members) > 1:
                merged_geometry[first] = shapely.union_all(
                    [geometries[m] for m in members])
                n_dropped += len(members) - 1

    keep_positions.sort()
    out = gdf.iloc[keep_positions].copy()
    if merged_geometry:
        geometry = out.geometry.tolist()
        for offset, position in enumerate(keep_positions):
            if position in merged_geometry:
                geometry[offset] = merged_geometry[position]
        out = out.set_geometry(gpd.GeoSeries(geometry, index=out.index, crs=gdf.crs))
    if verbose:
        print(f"Dedupe (identical tags, geometries within {tolerance_m:g} m): "
              f"{len(gdf)} -> {len(out)} landmarks ({n_dropped} merged away)")
    return out.reset_index(drop=True)


def merge_feathers(frames: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Concatenate landmark frames, taking the union of their tag columns."""
    for frame in frames:
        if "pruned_props" in frame.columns:
            raise ValueError(
                "input has a pruned_props column; it must be computed at load "
                "time, never stored")
        if frame.crs is None or frame.crs.to_string() != "EPSG:4326":
            raise ValueError(f"expected EPSG:4326, got {frame.crs}")
        for column in ("id", "landmark_type"):
            if column not in frame.columns:
                raise ValueError(f"input is missing the {column!r} column")

    merged = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True, sort=False), crs="EPSG:4326")
    duplicate_ids = merged["id"].duplicated().sum()
    if duplicate_ids:
        raise ValueError(
            f"{duplicate_ids} duplicate id values across inputs; ids must be "
            "globally unique so cost-matrix columns stay addressable")
    return merged


def report_cross_source_collisions(
    gdf: gpd.GeoDataFrame, radius_m: float = 150.0, limit: int = 15,
) -> None:
    """Print same-name landmarks from different sources that sit close together.

    These are not removed: an OSM and an ENC record of the same lighthouse
    usually carry different tags, and which one matches better is the
    correspondence model's call, not ours.
    """
    if "name" not in gdf.columns:
        return
    named = gdf[gdf["name"].notna()]
    points = {i: named.geometry[i].representative_point() for i in named.index}
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
                    (q.x - p.x) * 111320.0 * math.cos(math.radians(p.y)),
                    (q.y - p.y) * METERS_PER_DEG_LAT)
                if distance <= radius_m:
                    collisions.append((distance, name,
                                       named["landmark_type"][i],
                                       named["landmark_type"][j]))
    if not collisions:
        print("No same-name cross-source landmarks within "
              f"{radius_m:g} m")
        return
    collisions.sort()
    print(f"{len(collisions)} same-name cross-source pairs within {radius_m:g} m "
          "(kept, not merged):")
    for distance, name, left, right in collisions[:limit]:
        print(f"  {distance:6.1f} m  {name}  [{left} | {right}]")
    if len(collisions) > limit:
        print(f"  ... and {len(collisions) - limit} more")
