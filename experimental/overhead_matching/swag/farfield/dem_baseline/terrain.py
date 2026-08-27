"""Metric height fields from georeferenced elevation rasters.

One mosaic per evaluation region: source tiles (e.g. USGS 3DEP GeoTIFFs) are
merged and reprojected ONCE into a metric projected CRS (UTM 19N for New
England, per the plan's surface recipe), and the result is saved as an .npz +
manifest that downstream rendering consumes without touching rasterio again.

Coordinates everywhere in this package are (easting_m, northing_m) in the
height field's CRS. The localization stack's anchored-ENU RegionFrame is a
different frame; conversion happens only at that boundary, via lat/lon.

Vertical datum: whatever the sources share (NAVD88 orthometric for 3DEP).
Renders only ever use height *differences*, so orthometric heights are
consistent as long as every tile and the observer-height convention agree.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class HeightField:
    """North-up regular elevation grid in a metric projected CRS.

    ``elevation[row, col]`` is the height at easting ``x0 + (col + 0.5) * res``
    and northing ``y0 - (row + 0.5) * res`` (pixel centers; row 0 is the
    northern edge). ``nodata_mask`` marks cells with no valid source data;
    their elevation values are filled but must not be trusted, so rendering
    reports coverage from the mask.
    """

    elevation: np.ndarray  # (H, W) float32
    x0: float  # west edge (m)
    y0: float  # north edge (m)
    res: float  # cell size (m), square
    crs: str  # e.g. "EPSG:26919"
    nodata_mask: np.ndarray  # (H, W) bool

    def __post_init__(self):
        if self.elevation.ndim != 2:
            raise ValueError("elevation must be 2-D")
        if self.elevation.shape != self.nodata_mask.shape:
            raise ValueError("elevation and nodata_mask shapes differ")
        if self.res <= 0:
            raise ValueError("res must be positive")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) outer edges."""
        h, w = self.elevation.shape
        return (self.x0, self.y0 - h * self.res,
                self.x0 + w * self.res, self.y0)

    def grid_xy_from_rowcol(self, row, col):
        return (self.x0 + (np.asarray(col) + 0.5) * self.res,
                self.y0 - (np.asarray(row) + 0.5) * self.res)

    def sample(self, x, y):
        """Bilinear elevation at (x, y); NaN outside the grid."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        col = (x - self.x0) / self.res - 0.5
        row = (self.y0 - y) / self.res - 0.5
        h, w = self.elevation.shape
        # Inside means within the grid's outer edges; between the last pixel
        # center and the edge the clamped interpolation holds the edge value.
        inside = ((x >= self.x0) & (x <= self.x0 + w * self.res)
                  & (y <= self.y0) & (y >= self.y0 - h * self.res))
        col_c = np.clip(col, 0, w - 1)
        row_c = np.clip(row, 0, h - 1)
        c0 = np.floor(col_c).astype(np.int64)
        r0 = np.floor(row_c).astype(np.int64)
        c1 = np.minimum(c0 + 1, w - 1)
        r1 = np.minimum(r0 + 1, h - 1)
        fc = col_c - c0
        fr = row_c - r0
        elev = self.elevation
        value = ((1 - fr) * ((1 - fc) * elev[r0, c0] + fc * elev[r0, c1])
                 + fr * ((1 - fc) * elev[r1, c0] + fc * elev[r1, c1]))
        return np.where(inside, value, np.nan)

    def save(self, base_path: Path, extra_manifest: dict | None = None) -> None:
        """Write ``<base>.npz`` and ``<base>.json`` side by side."""
        base_path = Path(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            base_path.with_suffix(".npz"),
            elevation=self.elevation.astype(np.float32),
            nodata_mask=self.nodata_mask)
        manifest = {
            "x0": self.x0, "y0": self.y0, "res": self.res, "crs": self.crs,
            "shape": list(self.elevation.shape),
            "elevation_min": float(np.nanmin(self.elevation)),
            "elevation_max": float(np.nanmax(self.elevation)),
            "nodata_cells": int(self.nodata_mask.sum()),
        }
        manifest.update(extra_manifest or {})
        base_path.with_suffix(".json").write_text(json.dumps(manifest, indent=1))

    @classmethod
    def load(cls, base_path: Path) -> "HeightField":
        base_path = Path(base_path)
        meta = json.loads(base_path.with_suffix(".json").read_text())
        data = np.load(base_path.with_suffix(".npz"))
        return cls(elevation=data["elevation"], nodata_mask=data["nodata_mask"],
                   x0=meta["x0"], y0=meta["y0"], res=meta["res"],
                   crs=meta["crs"])


def utm_from_latlon(lat_deg: float, lon_deg: float,
                    crs: str) -> tuple[float, float]:
    """(easting_m, northing_m) of a WGS84 point in a projected CRS."""
    import pyproj
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs,
                                              always_xy=True)
    return transformer.transform(lon_deg, lat_deg)


def latlon_from_utm(x_m: float, y_m: float, crs: str) -> tuple[float, float]:
    import pyproj
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326",
                                              always_xy=True)
    lon_deg, lat_deg = transformer.transform(x_m, y_m)
    return lat_deg, lon_deg


def build_height_field(tile_paths: list[Path], *, dst_crs: str,
                       resolution_m: float,
                       bounds_xy: tuple[float, float, float, float],
                       resampling: str = "bilinear") -> HeightField:
    """Mosaic + reproject source rasters into one HeightField.

    ``bounds_xy`` is (x_min, y_min, x_max, y_max) in ``dst_crs``. Source
    no-data becomes ``nodata_mask`` and is filled with the mosaic's minimum
    valid elevation so ray marching never reads NaN (coverage still reported
    from the mask). Reprojection happens exactly once, source -> dst, per the
    surface recipe.
    """
    import rasterio
    import rasterio.merge
    import rasterio.warp
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin

    if not tile_paths:
        raise ValueError("no tiles given")
    x_min, y_min, x_max, y_max = bounds_xy
    width = int(round((x_max - x_min) / resolution_m))
    height = int(round((y_max - y_min) / resolution_m))
    if width <= 0 or height <= 0:
        raise ValueError(f"empty output grid for bounds {bounds_xy}")
    dst_transform = from_origin(x_min, y_max, resolution_m, resolution_m)
    dst = np.full((height, width), np.nan, dtype=np.float32)

    resampling_enum = Resampling[resampling]
    for path in tile_paths:
        with rasterio.open(path) as src:
            src_data = src.read(1, masked=True).filled(np.nan).astype(
                np.float32)
            tile_out = np.full((height, width), np.nan, dtype=np.float32)
            rasterio.warp.reproject(
                source=src_data,
                destination=tile_out,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=resampling_enum)
        # Later tiles only fill cells earlier tiles left empty, so tile order
        # cannot change values in overlap regions with identical sources.
        take = np.isnan(dst) & ~np.isnan(tile_out)
        dst[take] = tile_out[take]

    nodata_mask = np.isnan(dst)
    if nodata_mask.all():
        raise ValueError("no tile covered any of the requested bounds")
    fill_value = float(np.nanmin(dst))
    dst[nodata_mask] = fill_value
    return HeightField(elevation=dst, x0=x_min, y0=y_max, res=resolution_m,
                       crs=dst_crs, nodata_mask=nodata_mask)
