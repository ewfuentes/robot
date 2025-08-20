
from pathlib import Path
from dataclasses import dataclass
import rasterio
import itertools
import pyproj
import numpy as np

LON_LAT_CRS = "EPSG:4326"
WEB_MERCATOR_CRS = "EPSG:3857"

EARTH_RADIUS = 6378137.0
ORIGIN_SHIFT = np.pi * EARTH_RADIUS  # ~20037508.34


def meters_to_pixel_coords(easting_m: float, northing_m: float, zoom_level: int):
    """
    Convert Web Mercator meters (EPSG:3857) to pixel coordinates at given zoom.
    Returns (row, col) = (y, x).
    """
    map_size = 256 * (2 ** zoom_level)

    col = (easting_m + ORIGIN_SHIFT) / (2 * ORIGIN_SHIFT) * map_size
    row = (ORIGIN_SHIFT - northing_m) / (2 * ORIGIN_SHIFT) * map_size

    return row, col


@dataclass
class PartitionInfo:
    path: Path
    bounds: rasterio.coords.BoundingBox
    crs: rasterio.crs.CRS


class AlphaEarthRegistry:
    def __init__(self, base_path: Path, version: str):
        tif_files_dir = base_path / version
        self._partitions = {}
        self._file_handles = {}
        for tp in tif_files_dir.glob("*.tif"):
            with rasterio.open(tp) as ds:
                self._partitions[tp.stem] = PartitionInfo(
                    path=tp, bounds=ds.bounds, crs=ds.crs)

    def _find_dataset(self, lat_deg: float, lon_deg: float):
        for name, info in self._partitions.items():
            utm_from_lonlat = pyproj.Transformer.from_crs(LON_LAT_CRS, info.crs, always_xy=True)
            easting, northing = utm_from_lonlat.transform(lon_deg, lat_deg)
            if (info.bounds.left <= easting and
                    easting <= info.bounds.right and
                    info.bounds.bottom <= northing and
                    northing <= info.bounds.top):
                return name, easting, northing
        raise ValueError(
                f'Point {lat_deg=} {lon_deg=} is not contained in partitions: {self._partitions}')

    def query(self, lat_deg: float, lon_deg: float, patch_size: (int, int), zoom_level: int):
        '''
        Extract a patch around the specified lat/lon. Note that patch size is in pixels 
        '''
        name, easting, northing = self._find_dataset(lat_deg, lon_deg)
        if name not in self._file_handles:
            self._file_handles[name] = rasterio.open(self._partitions[name].path)
        fh = self._file_handles[name]
        info = self._partitions[name]
        row, col = fh.index(easting, northing)
        half_height = patch_size[0] // 2
        half_width = patch_size[1] // 2
        window = rasterio.windows.Window(
                col_off=col-half_width,
                row_off=row-half_height,
                height=patch_size[0],
                width=patch_size[1])
        data = fh.read(window=window, masked=True).transpose(1, 2, 0)

        to_web_mercator = pyproj.Transformer.from_crs(info.crs, WEB_MERCATOR_CRS, always_xy=True)
        position_info = np.zeros((window.height, window.width, 2))
        for row_idx, col_idx in itertools.product(range(window.height), range(window.width)):
            row_in_image = window.row_off + row_idx
            col_in_image = window.col_off + col_idx
            utm_easting, utm_northing = fh.xy(row_in_image, col_in_image)
            web_easting, web_northing = to_web_mercator.transform(utm_easting, utm_northing)
            web_y, web_x = meters_to_pixel_coords(easting_m=web_easting, northing_m=web_northing, zoom_level=zoom_level)

            position_info[row_idx, col_idx] = (web_y, web_x)
        return data, position_info
