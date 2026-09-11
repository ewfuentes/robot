"""Deterministic classified-LiDAR -> DSM rasterization (plan section 7.2).

The primary static DSM keeps ground, building, and bridge-deck returns,
excludes vegetation/noise/water/unclassified returns (so trees, ships, cars,
and cranes never enter the prior map), rasterizes an upper-surface statistic
per cell, fills small holes with a fixed neighborhood rule, and falls back to
the provider bare-earth DEM where LiDAR left no valid surface. Because the
provider DEM is hydro-flattened, that fallback IS the declared water-surface
treatment: water cells carry the provider's flattened water elevation.

Per-cell provenance is recorded so coverage audits can distinguish measured
from filled cells; nothing is ever repaired from test imagery.

Cross-tile composition: the region grid must be integer-aligned in the
destination CRS and tiles must not overlap (USGS LPC tiles are disjoint and
edge-aligned), so a running per-cell maximum over tile chunks is exact. The
high-quantile alternative is implemented only as a single-tile comparison
(`compare_statistics`) used to justify freezing the max rule on a validation
tile, as the plan requires.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield.dem_baseline import terrain

# ASPRS classes kept for the primary static DSM (plan section 7.2 step 3).
DSM_KEEP_CLASSES = (2, 6, 17)  # ground, building, bridge deck
# Bare-earth ablation from the same point cloud.
DEM_KEEP_CLASSES = (2,)

# Per-cell provenance codes.
PROV_EMPTY = 0  # no data from any source (also in nodata_mask)
PROV_LIDAR = 1  # upper statistic of kept LiDAR returns
PROV_HOLE_FILLED = 2  # neighborhood median of PROV_LIDAR cells
PROV_DEM_FALLBACK = 3  # provider bare-earth DEM (hydro-flattened water)


@dataclass
class RasterAccumulator:
    """Running per-cell maximum + return count over streamed LiDAR chunks."""

    x0: float
    y0: float
    res: float
    crs: str
    elevation: np.ndarray  # (H, W) float32, -inf where no return yet
    counts: np.ndarray  # (H, W) uint32

    @classmethod
    def create(cls, *, bounds_xy: tuple[float, float, float, float],
               resolution_m: float, crs: str) -> "RasterAccumulator":
        x_min, y_min, x_max, y_max = bounds_xy
        for name, value in (("x_min", x_min), ("y_min", y_min),
                            ("x_max", x_max), ("y_max", y_max)):
            if value != round(value):
                raise ValueError(
                    f"{name}={value} is not integer-aligned; exact cross-tile "
                    "composition needs a grid aligned to tile edges")
        width = int(round((x_max - x_min) / resolution_m))
        height = int(round((y_max - y_min) / resolution_m))
        if width <= 0 or height <= 0:
            raise ValueError(f"empty grid for bounds {bounds_xy}")
        return cls(
            x0=x_min, y0=y_max, res=resolution_m, crs=crs,
            elevation=np.full((height, width), -np.inf, dtype=np.float32),
            counts=np.zeros((height, width), dtype=np.uint32))

    def add_points(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> int:
        """Fold one batch of kept returns into the running max; returns the
        number of in-bounds points used."""
        h, w = self.elevation.shape
        col = np.floor((x - self.x0) / self.res).astype(np.int64)
        row = np.floor((self.y0 - y) / self.res).astype(np.int64)
        keep = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        if not keep.any():
            return 0
        flat = row[keep] * w + col[keep]
        np.maximum.at(self.elevation.reshape(-1), flat,
                      z[keep].astype(np.float32))
        np.add.at(self.counts.reshape(-1), flat, 1)
        return int(keep.sum())


def _horizontal_transformer(src_crs, dst_crs: str):
    """A pyproj (x, y) transform, or None when it is the identity (< 1 cm
    over the axis-aligned unit square scaled anywhere on Earth we test)."""
    import pyproj
    src = pyproj.CRS(src_crs)
    dst = pyproj.CRS(dst_crs)
    transformer = pyproj.Transformer.from_crs(src, dst, always_xy=True)
    probe = np.array([0.0, 3.3e5, 4.7e6, 9.0e5])
    px = probe[[0, 1, 1, 0]]
    py = probe[[2, 2, 3, 3]]
    qx, qy = transformer.transform(px, py)
    if max(np.abs(qx - px).max(), np.abs(qy - py).max()) < 0.01:
        return None
    return transformer


def stream_tile(laz_path: Path, accumulator: RasterAccumulator, *,
                keep_classes: tuple[int, ...],
                chunk_points: int = 4_000_000) -> dict:
    """Stream one LAZ tile into the accumulator. Returns tile statistics."""
    import laspy

    kept = used = total = 0
    with laspy.open(laz_path) as reader:
        header_crs = reader.header.parse_crs()
        if header_crs is None:
            raise ValueError(f"{laz_path} carries no CRS; refusing to guess")
        transformer = _horizontal_transformer(header_crs, accumulator.crs)
        # Skip tiles whose header bbox (with a one-cell pad after any
        # reprojection) cannot touch the grid, so region builds do not
        # decompress the whole collection.
        tx = np.array([reader.header.mins[0], reader.header.maxs[0]])
        ty = np.array([reader.header.mins[1], reader.header.maxs[1]])
        if transformer is not None:
            corner_x, corner_y = transformer.transform(
                tx[[0, 1, 1, 0]], ty[[0, 0, 1, 1]])
            tx = np.array([corner_x.min(), corner_x.max()])
            ty = np.array([corner_y.min(), corner_y.max()])
        h, w = accumulator.elevation.shape
        pad = accumulator.res
        if (tx[1] < accumulator.x0 - pad
                or tx[0] > accumulator.x0 + w * accumulator.res + pad
                or ty[1] < accumulator.y0 - h * accumulator.res - pad
                or ty[0] > accumulator.y0 + pad):
            return {"path": str(laz_path),
                    "points_total": int(reader.header.point_count),
                    "points_kept_class": 0, "points_in_bounds": 0,
                    "reprojected": transformer is not None,
                    "skipped_disjoint_bbox": True}
        class_mask = np.zeros(256, dtype=bool)
        class_mask[list(keep_classes)] = True
        for chunk in reader.chunk_iterator(chunk_points):
            total += len(chunk)
            select = class_mask[np.asarray(chunk.classification)]
            if not select.any():
                continue
            x = np.asarray(chunk.x)[select]
            y = np.asarray(chunk.y)[select]
            z = np.asarray(chunk.z)[select]
            kept += int(select.sum())
            if transformer is not None:
                x, y = transformer.transform(x, y)
            used += accumulator.add_points(x, y, z)
    return {"path": str(laz_path), "points_total": total,
            "points_kept_class": kept, "points_in_bounds": used,
            "reprojected": transformer is not None,
            "skipped_disjoint_bbox": False}


_FILL_SHIFTS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1))


def fill_holes(elevation: np.ndarray, valid: np.ndarray, passes: int,
               block_rows: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-rule small-hole filling: for `passes` iterations, an invalid
    cell with at least 5 of its 8 neighbors valid takes their median. Fills
    pinholes (missed pulses, glass roofs) without flooding water bodies or
    large gaps. Returns (filled elevation, mask of cells filled here).

    Each pass reads the pre-pass state only (Jacobi update), swept in row
    blocks so the 8-neighbor stack never materializes for the whole grid --
    a region grid is ~775M cells and a full stack OOMs a 64 GB host.
    """
    elevation = elevation.copy()
    valid = valid.copy()
    filled_any = np.zeros_like(valid)
    h, w = elevation.shape
    for _ in range(passes):
        src = np.where(valid, elevation, np.nan)
        updates = []
        for r0 in range(0, h, block_rows):
            r1 = min(r0 + block_rows, h)
            hole = ~valid[r0:r1]
            if not hole.any():
                continue
            neighbors = np.full((8, r1 - r0, w), np.nan, dtype=np.float32)
            for k, (dr, dc) in enumerate(_FILL_SHIFTS):
                s0 = max(r0 + dr, 0)
                s1 = min(r1 + dr, h)
                if s0 >= s1:
                    continue
                b0 = s0 - (r0 + dr)
                b1 = b0 + (s1 - s0)
                c_dst0, c_dst1 = max(0, -dc), w - max(0, dc)
                neighbors[k, b0:b1, c_dst0:c_dst1] = \
                    src[s0:s1, c_dst0 + dc:c_dst1 + dc]
            n_valid = np.sum(~np.isnan(neighbors), axis=0)
            fill = hole & (n_valid >= 5)
            if not fill.any():
                continue
            rows, cols = np.nonzero(fill)
            median = np.nanmedian(neighbors[:, rows, cols], axis=0)
            updates.append((rows + r0, cols, median.astype(np.float32)))
        if not updates:
            break
        for rows, cols, median in updates:
            elevation[rows, cols] = median
            valid[rows, cols] = True
            filled_any[rows, cols] = True
    return elevation, filled_any


def compose_surface(accumulator: RasterAccumulator, *,
                    fill_passes: int,
                    dem_fallback: terrain.HeightField | None
                    ) -> tuple[terrain.HeightField, np.ndarray]:
    """Assemble the final surface: LiDAR max -> hole fill -> DEM fallback ->
    recorded no-data. Returns (HeightField, provenance uint8 raster)."""
    lidar_valid = accumulator.counts > 0
    elevation = np.where(lidar_valid, accumulator.elevation,
                         np.nan).astype(np.float32)
    provenance = np.where(lidar_valid, PROV_LIDAR,
                          PROV_EMPTY).astype(np.uint8)

    if fill_passes > 0:
        elevation, filled = fill_holes(elevation, lidar_valid, fill_passes)
        provenance[filled] = PROV_HOLE_FILLED
    covered = provenance != PROV_EMPTY

    if dem_fallback is not None:
        rows, cols = np.nonzero(~covered)
        if rows.size:
            x, y = accumulator.x0 + (cols + 0.5) * accumulator.res, \
                accumulator.y0 - (rows + 0.5) * accumulator.res
            if dem_fallback.crs != accumulator.crs:
                import pyproj
                transformer = pyproj.Transformer.from_crs(
                    accumulator.crs, dem_fallback.crs, always_xy=True)
                x, y = transformer.transform(x, y)
            dem_z = dem_fallback.sample(x, y)
            # A DEM cell that is itself no-data must not masquerade as data.
            drow = np.clip(((dem_fallback.y0 - y) / dem_fallback.res)
                           .astype(np.int64), 0,
                           dem_fallback.elevation.shape[0] - 1)
            dcol = np.clip(((x - dem_fallback.x0) / dem_fallback.res)
                           .astype(np.int64), 0,
                           dem_fallback.elevation.shape[1] - 1)
            usable = ~np.isnan(dem_z) & ~dem_fallback.nodata_mask[drow, dcol]
            elevation[rows[usable], cols[usable]] = dem_z[usable]
            provenance[rows[usable], cols[usable]] = PROV_DEM_FALLBACK

    nodata_mask = provenance == PROV_EMPTY
    if nodata_mask.all():
        raise ValueError("no cell received data from any source")
    elevation[nodata_mask] = np.nanmin(elevation)
    field = terrain.HeightField(
        elevation=elevation, x0=accumulator.x0, y0=accumulator.y0,
        res=accumulator.res, crs=accumulator.crs, nodata_mask=nodata_mask)
    return field, provenance


def compare_statistics(laz_path: Path, *, resolution_m: float,
                       keep_classes: tuple[int, ...] = DSM_KEEP_CLASSES,
                       quantile: float = 0.98) -> dict:
    """Exact per-cell max vs upper-quantile on ONE tile, in the tile's own
    CRS/extent. This is the validation-tile evidence the plan requires before
    freezing the rasterization statistic; it never feeds the region build."""
    import laspy

    with laspy.open(laz_path) as reader:
        crs = reader.header.parse_crs()
        chunks = [c for c in reader.chunk_iterator(4_000_000)]
        classification = np.concatenate(
            [np.asarray(c.classification) for c in chunks])
        x = np.concatenate([np.asarray(c.x) for c in chunks])
        y = np.concatenate([np.asarray(c.y) for c in chunks])
        z = np.concatenate([np.asarray(c.z) for c in chunks])
    class_mask = np.zeros(256, dtype=bool)
    class_mask[list(keep_classes)] = True
    select = class_mask[classification]
    x, y, z = x[select], y[select], z[select]
    x0 = np.floor(x.min())
    y0 = np.ceil(y.max())
    col = np.floor((x - x0) / resolution_m).astype(np.int64)
    row = np.floor((y0 - y) / resolution_m).astype(np.int64)
    width = int(col.max()) + 1
    flat = row * width + col
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    z_sorted = z[order]
    starts = np.flatnonzero(np.diff(flat_sorted, prepend=-1))
    cell_max = np.maximum.reduceat(z_sorted, starts)
    cell_quantile = np.empty_like(cell_max)
    bounds = np.append(starts, z_sorted.size)
    for i in range(starts.size):
        cell_quantile[i] = np.quantile(z_sorted[bounds[i]:bounds[i + 1]],
                                       quantile)
    delta = cell_max - cell_quantile
    return {
        "path": str(laz_path),
        "crs": str(crs),
        "resolution_m": resolution_m,
        "quantile": quantile,
        "n_cells": int(starts.size),
        "points_kept": int(z.size),
        "delta_max_minus_quantile_m": {
            "mean": float(delta.mean()),
            "p50": float(np.percentile(delta, 50)),
            "p90": float(np.percentile(delta, 90)),
            "p99": float(np.percentile(delta, 99)),
            "max": float(delta.max()),
        },
        "cells_delta_over_1m": int((delta > 1.0).sum()),
    }
