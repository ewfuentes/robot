
from pathlib import Path
from dataclasses import dataclass
import rasterio
import pyproj


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
            utm_from_lonlat = pyproj.Transformer.from_crs("EPSG:4326", info.crs, always_xy=True)
            easting, northing = utm_from_lonlat.transform(lon_deg, lat_deg)
            if (info.bounds.left <= easting and
                    easting <= info.bounds.right and
                    info.bounds.bottom <= northing and
                    northing <= info.bounds.top):
                return name, easting, northing
        raise ValueError(
                f'Point {lat_deg=} {lon_deg=} is not contained in partitions: {self._partitions}')

    def query(self, lat_deg: float, lon_deg: float, patch_size: (int, int)):
        name, easting, northing = self._find_dataset(lat_deg, lon_deg)
        if name not in self._file_handles:
            self._file_handles[name] = rasterio.open(self._partitions[name].path)
        fh = self._file_handles[name]
        row, col = fh.index(easting, northing)
        half_height = patch_size[0] // 2
        half_width = patch_size[1] // 2
        window = rasterio.windows.Window(
                col_off=col-half_width,
                row_off=row-half_height,
                height=patch_size[0],
                width=patch_size[1])
        data = fh.read(window=window)

        print(lat_deg, lon_deg, patch_size, name, easting, northing, data.shape)
