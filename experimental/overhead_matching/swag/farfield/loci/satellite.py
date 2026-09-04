#!/usr/bin/env python3
"""Build a strict, resumable LOCI satellite-imagery artifact.

The immutable region artifact owns the Web-Mercator grid.  This producer
audits an ArcGIS cached imagery service against that grid, downloads the
native source tiles into a persistent build cache, assembles VIGOR-compatible
overlapping patches, and publishes only after every source tile and patch has
been decoded and hashed successfully.

Large mutable work lives under ``builds/<dataset>/``.  Downloads and patch
writes use atomic replacement, and concurrency is deliberately bounded: the
producer never holds the complete source mosaic (or one future per tile) in
memory.  A stopped invocation can therefore be run again without discarding
valid source tiles or completed patches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, TypeVar

import requests
from PIL import Image, ImageDraw, ImageStat
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from experimental.overhead_matching.swag.farfield import (
    artifact,
    artifact_identity,
    artifact_recipe,
    geometry,
    paths as paths_lib,
    provenance,
    publication,
)
from experimental.overhead_matching.swag.farfield.loci import region


SCHEMA = "loci_satellite/v1"
SOURCE_MANIFEST_SCHEMA = "loci_source_tile_manifest/v1"
PATCH_MANIFEST_SCHEMA = "loci_satellite_patch_manifest/v1"
COVERAGE_SCHEMA = "loci_imagery_coverage_audit/v1"
BUILD_SCHEMA = "loci_satellite_build/v1"
ASSEMBLY_VERSION = "vigor_web_mercator_nearest_pixel_row_major_v2"

ARTIFACT_KIND = "loci_satellite"
GENERATOR = "//experimental/overhead_matching/swag/farfield/loci:satellite"

DEFAULT_SERVICE_NAME = "Massachusetts_Aerial_Imagery_2025"
DEFAULT_SERVICE_URL = (
    "https://tiles.arcgis.com/tiles/hGdibHYSPO59RG1h/arcgis/rest/services/"
    f"{DEFAULT_SERVICE_NAME}/MapServer")
DEFAULT_SOURCE_INDEX_URL = (
    "https://services1.arcgis.com/hGdibHYSPO59RG1h/arcgis/rest/services/"
    "Color_2025_Aerial_Imagery_Index/FeatureServer/0")
ESRI_WORLD_IMAGERY_SERVICE_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer")
ESRI_WAYBACK_TILE_SERVICE_URL = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
    "World_Imagery/WMTS/1.0.0/default028mm/MapServer")
DEFAULT_JPEG_QUALITY = 95
ARCGIS_BUNDLE_SIZE = 128

CACHED_MAP_PROVIDER = "arcgis_cached_map"
IMAGE_SERVER_PROVIDER = "arcgis_image_server_export"
DEFAULT_PROVIDER_MODE = CACHED_MAP_PROVIDER
PROVIDER_MODES = (CACHED_MAP_PROVIDER, IMAGE_SERVER_PROVIDER)

IMAGE_SERVER_EXPORT_FORMAT = "png"
IMAGE_SERVER_INTERPOLATION = "RSP_BilinearInterpolation"
IMAGE_SERVER_RASTER_FUNCTION = "NaturalColor"
IMAGE_SERVER_MOSAIC_OPERATION = "MT_FIRST"
IMAGE_SERVER_MAX_CHUNK_TILES = 15
IMAGE_SERVER_CHUNK_WORKERS = 2
IMAGE_SERVER_CHUNK_SCHEMA = "arcgis_image_server_source_chunk/v1"
IMAGE_SERVER_CHUNK_ALGORITHM = (
    "northwest_grid_anchored_exact_web_mercator_256px_crops_v1")
IMAGE_SERVER_SOURCE_TILE_ENCODING = (
    "pillow_rgb_png_compress6_no_optimize_v1")
IMAGE_SERVER_CATALOG_FIELDS = (
    "OBJECTID",
    "Name",
    "State",
    "Year",
    "raster_name",
    "download_url",
    "acquisition_date",
    "agency",
    "vendor",
    "resolution_value",
    "resolution_units",
    "band_count",
    "sensor_type",
    "Category",
)

SATELLITE_DIR = "satellite"
SOURCE_MANIFEST = "source_tile_manifest.json"
PATCH_MANIFEST = "patch_manifest.json"
TILE_METADATA = "tile_metadata.csv"
COVERAGE_AUDIT = "coverage_audit.json"
COVERAGE_CONTACT_SHEET = "coverage_contact_sheet.jpg"
COVERAGE_CONTACT_INDEX = "coverage_contact_sheet.json"
SATELLITE_BBOX = "satellite_bbox.json"
SUMMARY_OUTPUT = "satellite_summary.json"


class SatelliteError(RuntimeError):
    """A satellite input, cached file, or provider response is invalid."""


class MissingTileError(SatelliteError):
    """The provider explicitly reports that a required tile is absent."""


def _normalize_esri_wayback_release(value: str | None) -> str | None:
    if value is None:
        return None
    release = str(value).strip()
    if not release.isdigit():
        raise SatelliteError("ESRI Wayback release must contain only digits")
    return release


@dataclass(frozen=True)
class ImageInfo:
    size_bytes: int
    sha256: str
    decoded_pixel_sha256: str
    image_format: str
    mode: str
    width: int
    height: int
    mean_rgb: tuple[float, float, float]


@dataclass(frozen=True)
class ImageServerSourceTile:
    """One canonical child tile split from an ImageServer export."""

    tile_x: int
    tile_y: int
    value: bytes
    info: ImageInfo


@dataclass(frozen=True)
class ImageServerTileChunk:
    """A validated ImageServer export and its canonical child tiles."""

    zoom: int
    tile_x: int
    tile_y: int
    width: int
    height: int
    response_info: ImageInfo
    tiles: tuple[ImageServerSourceTile, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_image_bytes(value: bytes, expected_size: tuple[int, int],
                          what: str) -> tuple[Image.Image, ImageInfo]:
    if not value:
        raise SatelliteError(f"{what} is empty")
    try:
        with Image.open(io.BytesIO(value)) as encoded:
            image_format = encoded.format or "unknown"
            encoded_mode = encoded.mode
            encoded.load()
            if encoded.size != expected_size:
                raise SatelliteError(
                    f"{what} is {encoded.size}, expected {expected_size}")
            image = encoded.convert("RGB")
            image.load()
    except SatelliteError:
        raise
    except Exception as error:
        raise SatelliteError(f"cannot decode {what}: {error}") from error
    pixels = image.tobytes()
    stats = ImageStat.Stat(image)
    info = ImageInfo(
        size_bytes=len(value),
        sha256=_sha256_bytes(value),
        decoded_pixel_sha256=_sha256_bytes(pixels),
        image_format=image_format,
        mode=encoded_mode,
        width=image.width,
        height=image.height,
        mean_rgb=tuple(round(number, 3) for number in stats.mean),
    )
    return image, info


def validate_image_file(path: Path, expected_size: tuple[int, int]) \
        -> ImageInfo | None:
    """Return image metadata, or ``None`` for a missing/corrupt cache file."""
    path = Path(path)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise SatelliteError(f"cached image is not a regular file: {path}")
    try:
        value = path.read_bytes()
        _, info = _validate_image_bytes(value, expected_size, str(path))
        return info
    except (OSError, SatelliteError):
        return None


def _canonical_rgb_png_bytes(image: Image.Image) -> tuple[bytes, ImageInfo]:
    """Encode one RGB tile with the chunk-mode byte-level contract."""
    if image.mode != "RGB":
        raise SatelliteError(
            f"canonical source tile has mode {image.mode}, expected RGB")
    output = io.BytesIO()
    image.save(
        output, format="PNG", compress_level=6, optimize=False)
    value = output.getvalue()
    _, info = _validate_image_bytes(
        value, image.size, "canonical ImageServer source tile")
    if info.image_format != "PNG" or info.mode != "RGB":
        raise SatelliteError(
            "canonical ImageServer source tile is not an RGB PNG")
    return value, info


@contextmanager
def _atomic_text_writer(path: Path):
    """Yield a UTF-8 stream and atomically replace ``path`` on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") \
                as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class ArcGisTileClient:
    """Small retrying client for one ArcGIS cached map service."""

    def __init__(self, service_url: str = DEFAULT_SERVICE_URL, *,
                 esri_wayback_release: str | None = None,
                 connect_timeout_s: float = 10.0,
                 read_timeout_s: float = 60.0,
                 max_retries: int = 4,
                 retry_backoff_s: float = 0.5) -> None:
        service_url = service_url.rstrip("/")
        if not service_url.startswith("https://"):
            raise SatelliteError("ArcGIS service URL must use https")
        if connect_timeout_s <= 0.0 or read_timeout_s <= 0.0:
            raise SatelliteError("HTTP timeouts must be positive")
        if max_retries < 1:
            raise SatelliteError("max_retries must be at least one")
        self.service_url = service_url
        self.esri_wayback_release = _normalize_esri_wayback_release(
            esri_wayback_release)
        if (self.esri_wayback_release is not None
                and service_url != ESRI_WORLD_IMAGERY_SERVICE_URL):
            raise SatelliteError(
                "ESRI Wayback requires the standard World Imagery "
                "MapServer metadata URL")
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self._local = threading.local()

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            session = requests.Session()
            session.headers["User-Agent"] = (
                "robot-loci-baseline/farfield-loci-satellite")
            self._local.session = session
        return self._local.session

    def _get(self, url: str, *, params: dict | None = None,
             missing_is_error: bool = False) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._session().get(
                    url, params=params,
                    timeout=(self.connect_timeout_s, self.read_timeout_s))
                if response.status_code == 404 and missing_is_error:
                    raise MissingTileError(f"required tile is missing: {url}")
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(
                        f"transient HTTP {response.status_code}",
                        response=response)
                response.raise_for_status()
                return response
            except MissingTileError:
                raise
            except (requests.RequestException, OSError) as error:
                last_error = error
                if attempt + 1 == self.max_retries:
                    break
                delay = self.retry_backoff_s * (2 ** attempt)
                retry_after = getattr(
                    getattr(error, "response", None), "headers", {}).get(
                        "Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, min(float(retry_after), 60.0))
                    except ValueError:
                        pass
                time.sleep(delay)
        raise SatelliteError(
            f"provider request failed after {self.max_retries} attempts: "
            f"{url}: {last_error}") from last_error

    def get_json(self, url: str, *, params: dict | None = None) -> dict:
        response = self._get(url, params=params)
        try:
            document = response.json()
        except (ValueError, json.JSONDecodeError) as error:
            raise SatelliteError(f"provider returned invalid JSON: {url}") \
                from error
        if not isinstance(document, dict):
            raise SatelliteError(f"provider JSON is not an object: {url}")
        if "error" in document:
            raise SatelliteError(
                f"provider returned an ArcGIS error for {url}: "
                f"{document['error']}")
        return document

    def get_service_metadata(self) -> dict:
        return self.get_json(self.service_url, params={"f": "json"})

    def get_tilemap(self, zoom: int, tile_x: int, tile_y: int,
                    width: int, height: int) -> dict:
        tile_service_url = (
            ESRI_WAYBACK_TILE_SERVICE_URL
            if self.esri_wayback_release is not None else self.service_url)
        release_path = (f"/{self.esri_wayback_release}"
                        if self.esri_wayback_release is not None else "")
        url = (f"{tile_service_url}/tilemap{release_path}/"
               f"{zoom}/{tile_y}/{tile_x}/"
               f"{width}/{height}")
        return self.get_json(url, params={"f": "json"})

    def query_source_index(self, source_index_url: str,
                           bbox_wsen: Iterable[float]) -> dict:
        west, south, east, north = tuple(bbox_wsen)
        params = {
            "f": "json",
            "where": "1=1",
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "OBJECTID,TILE_NAME,ZONE,URL",
            "returnGeometry": "true",
            "resultRecordCount": "2000",
        }
        return self.get_json(
            source_index_url.rstrip("/") + "/query", params=params)

    def fetch_tile(self, zoom: int, tile_x: int, tile_y: int) -> bytes:
        tile_service_url = (
            ESRI_WAYBACK_TILE_SERVICE_URL
            if self.esri_wayback_release is not None else self.service_url)
        release_path = (f"/{self.esri_wayback_release}"
                        if self.esri_wayback_release is not None else "")
        url = (f"{tile_service_url}/tile{release_path}/"
               f"{zoom}/{tile_y}/{tile_x}")
        response = self._get(
            url, params={"blankTile": "false"}, missing_is_error=True)
        return response.content


def _normalize_lock_raster_ids(values: Iterable[int]) -> tuple[int, ...]:
    try:
        raster_ids = tuple(values)
    except TypeError as error:
        raise SatelliteError("lock raster IDs must be iterable") from error
    if not raster_ids:
        raise SatelliteError(
            "ImageServer export requires at least one lock raster ID")
    if any(type(value) is not int or value <= 0 for value in raster_ids):
        raise SatelliteError(
            "lock raster IDs must be positive integers")
    if len(raster_ids) != len(set(raster_ids)):
        raise SatelliteError("lock raster IDs must be unique")
    return tuple(sorted(raster_ids))


def _image_server_catalog_parameters(
        bbox_wsen: Iterable[float], catalog_where: str) -> dict:
    west, south, east, north = tuple(float(item) for item in bbox_wsen)
    catalog_where = str(catalog_where).strip()
    if not catalog_where:
        raise SatelliteError(
            "ImageServer export requires a non-empty catalog where clause")
    return {
        "f": "json",
        "where": catalog_where,
        "geometry": f"{west},{south},{east},{north}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": ",".join(IMAGE_SERVER_CATALOG_FIELDS),
        "returnGeometry": "true",
        "resultRecordCount": "1000",
    }


def _image_server_mosaic_rule(
        lock_raster_ids: Iterable[int]) -> dict:
    return {
        "mosaicMethod": "esriMosaicLockRaster",
        "lockRasterIds": list(_normalize_lock_raster_ids(lock_raster_ids)),
        "mosaicOperation": IMAGE_SERVER_MOSAIC_OPERATION,
    }


def _json_parameter(value: dict) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False)


def _decode_image_server_response(response, *,
                                  expected_size: tuple[int, int],
                                  what: str) \
        -> tuple[Image.Image, ImageInfo]:
    """Validate one ArcGIS export response and return decoded RGB pixels."""
    content_type = str(response.headers.get("Content-Type", "")) \
        .split(";", 1)[0].strip().lower()
    value = response.content
    stripped = value.lstrip()
    if content_type in {"application/json", "text/json"} \
            or stripped.startswith(b"{"):
        try:
            document = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError):
            document = None
        detail = document.get("error") if isinstance(document, dict) \
            else "invalid JSON error response"
        raise SatelliteError(
            f"ImageServer returned JSON instead of imagery for {what}: "
            f"{detail}")
    if content_type != "image/png":
        raise SatelliteError(
            f"ImageServer returned content type {content_type!r}, "
            "expected 'image/png'")
    image, info = _validate_image_bytes(value, expected_size, what)
    if info.image_format != "PNG":
        raise SatelliteError(
            f"ImageServer encoded {info.image_format}, expected PNG")
    if info.mode != "RGB":
        raise SatelliteError(
            f"ImageServer encoded PNG mode {info.mode}, expected RGB")
    if image.getbbox() is None:
        raise MissingTileError(
            f"ImageServer returned an all-black no-data image for {what}")
    return image, info


class ArcGisImageServerClient(ArcGisTileClient):
    """Retrying client for one pinned ArcGIS dynamic ImageServer export."""

    def __init__(self, service_url: str, *, catalog_where: str,
                 lock_raster_ids: Iterable[int],
                 connect_timeout_s: float = 10.0,
                 read_timeout_s: float = 60.0,
                 max_retries: int = 4,
                 retry_backoff_s: float = 0.5) -> None:
        super().__init__(
            service_url, connect_timeout_s=connect_timeout_s,
            read_timeout_s=read_timeout_s, max_retries=max_retries,
            retry_backoff_s=retry_backoff_s)
        catalog_where = str(catalog_where).strip()
        if not catalog_where:
            raise SatelliteError(
                "ImageServer export requires a non-empty catalog where "
                "clause")
        self.catalog_where = catalog_where
        self.lock_raster_ids = _normalize_lock_raster_ids(lock_raster_ids)

    def query_catalog(self, bbox_wsen: Iterable[float]) -> dict:
        return self.get_json(
            self.service_url + "/query",
            params=_image_server_catalog_parameters(
                bbox_wsen, self.catalog_where))

    def fetch_tile(self, zoom: int, tile_x: int, tile_y: int) -> bytes:
        url = self.service_url + "/exportImage"
        parameters = _image_server_export_parameters(
            zoom, tile_x, tile_y, self.lock_raster_ids)
        response = self._get(url, params=parameters)
        _decode_image_server_response(
            response,
            expected_size=(region.DEFAULT_TILE_PX, region.DEFAULT_TILE_PX),
            what=f"z{zoom}/{tile_x}/{tile_y}")
        # Preserve the legacy one-request-per-tile byte contract.  Chunk mode
        # deliberately uses the canonical child encoding below instead.
        return response.content

    def fetch_tile_chunk(self, zoom: int, tile_x: int, tile_y: int,
                         width: int, height: int) \
            -> ImageServerTileChunk:
        """Fetch and split one fixed, north-up rectangle of XYZ tiles."""
        parameters = _image_server_export_parameters(
            zoom, tile_x, tile_y, self.lock_raster_ids,
            width=width, height=height)
        response = self._get(
            self.service_url + "/exportImage", params=parameters)
        tile_px = region.DEFAULT_TILE_PX
        image, response_info = _decode_image_server_response(
            response, expected_size=(width * tile_px, height * tile_px),
            what=(f"chunk z{zoom}/{tile_x}/{tile_y} "
                  f"{width}x{height}"))
        tiles = []
        for offset_y in range(height):
            for offset_x in range(width):
                child_x = tile_x + offset_x
                child_y = tile_y + offset_y
                left = offset_x * tile_px
                top = offset_y * tile_px
                child = image.crop(
                    (left, top, left + tile_px, top + tile_px))
                if child.getbbox() is None:
                    raise MissingTileError(
                        "ImageServer chunk contains an all-black no-data "
                        f"tile at z{zoom}/{child_x}/{child_y}")
                value, info = _canonical_rgb_png_bytes(child)
                tiles.append(ImageServerSourceTile(
                    tile_x=child_x, tile_y=child_y,
                    value=value, info=info))
        return ImageServerTileChunk(
            zoom=zoom, tile_x=tile_x, tile_y=tile_y,
            width=width, height=height,
            response_info=response_info, tiles=tuple(tiles))


def _grid_tile_items(grid: dict) -> Iterator[tuple[int, int]]:
    tile_x_min, tile_y_min, tile_x_max, tile_y_max = (
        grid["source_tile_range_xyxy"])
    for tile_y in range(tile_y_min, tile_y_max + 1):
        for tile_x in range(tile_x_min, tile_x_max + 1):
            yield tile_x, tile_y


def _iter_image_server_source_chunks(
        grid: dict, chunk_tiles: int) \
        -> Iterator[tuple[int, int, int, int]]:
    """Yield a fixed NW-anchored, row-major partition of source tiles."""
    chunk_tiles = _normalize_image_server_chunk_tiles(chunk_tiles)
    tile_x_min, tile_y_min, tile_x_max, tile_y_max = (
        grid["source_tile_range_xyxy"])
    for tile_y in range(tile_y_min, tile_y_max + 1, chunk_tiles):
        height = min(chunk_tiles, tile_y_max - tile_y + 1)
        for tile_x in range(tile_x_min, tile_x_max + 1, chunk_tiles):
            width = min(chunk_tiles, tile_x_max - tile_x + 1)
            yield tile_x, tile_y, width, height


def _tile_cache_path(build_dir: Path, zoom: int,
                     tile_x: int, tile_y: int) -> Path:
    return (Path(build_dir) / "source_tiles" / str(zoom) / str(tile_x)
            / f"{tile_y}.tile")


def _source_chunk_receipt_path(
        build_dir: Path, zoom: int, tile_x: int, tile_y: int,
        width: int, height: int) -> Path:
    return (Path(build_dir) / "source_tile_chunks" / str(zoom)
            / f"{tile_x}_{tile_y}_{width}x{height}.json")


def _web_mercator_m_from_pixel(pixel_x: float, pixel_y: float,
                               zoom: int, tile_px: int) \
        -> tuple[float, float]:
    half_world = math.pi * geometry.EARTH_RADIUS_M
    scale = tile_px * (2 ** zoom)
    return (
        pixel_x / scale * (2.0 * half_world) - half_world,
        half_world - pixel_y / scale * (2.0 * half_world),
    )


def _web_mercator_tile_bbox(zoom: int, tile_x: int, tile_y: int, *,
                            tile_px: int = region.DEFAULT_TILE_PX) \
        -> tuple[float, float, float, float]:
    """Return one slippy tile's ``xmin,ymin,xmax,ymax`` metre envelope."""
    if type(zoom) is not int or zoom < 0:
        raise SatelliteError("tile zoom must be a non-negative integer")
    limit = 2 ** zoom
    if (type(tile_x) is not int or type(tile_y) is not int
            or not 0 <= tile_x < limit or not 0 <= tile_y < limit):
        raise SatelliteError(
            f"tile coordinates are outside z{zoom}: {tile_x},{tile_y}")
    west_m, north_m = _web_mercator_m_from_pixel(
        tile_x * tile_px, tile_y * tile_px, zoom, tile_px)
    east_m, south_m = _web_mercator_m_from_pixel(
        (tile_x + 1) * tile_px, (tile_y + 1) * tile_px, zoom, tile_px)
    return west_m, south_m, east_m, north_m


def _rendered_footprint_bbox_wsen(grid: dict) -> list[float]:
    """Return the exact WGS84 envelope touched by quantized patch crops."""
    min_x, min_y = grid["min_pixel_xy"]
    n_x, n_y = grid["shape_xy"]
    stride = grid["stride_px"]
    source_px = grid["source_px"]
    last_x = min_x + (n_x - 1) * stride
    last_y = min_y + (n_y - 1) * stride
    west_pixel = region.nearest_pixel_origin(min_x, source_px)
    north_pixel = region.nearest_pixel_origin(min_y, source_px)
    east_pixel = region.nearest_pixel_origin(last_x, source_px) + source_px
    south_pixel = region.nearest_pixel_origin(last_y, source_px) + source_px
    north, west = region.pixel_to_lat_lon(
        west_pixel, north_pixel, grid["zoom"])
    south, east = region.pixel_to_lat_lon(
        east_pixel, south_pixel, grid["zoom"])
    return [west, south, east, north]


def _web_mercator_tile_range_bbox(
        zoom: int, tile_x: int, tile_y: int, *,
        width: int = 1, height: int = 1) \
        -> tuple[float, float, float, float]:
    if (type(width) is not int or type(height) is not int
            or not 1 <= width <= IMAGE_SERVER_MAX_CHUNK_TILES
            or not 1 <= height <= IMAGE_SERVER_MAX_CHUNK_TILES):
        raise SatelliteError(
            "ImageServer export chunk dimensions must be integers in "
            f"1..{IMAGE_SERVER_MAX_CHUNK_TILES}")
    northwest = _web_mercator_tile_bbox(zoom, tile_x, tile_y)
    southeast = _web_mercator_tile_bbox(
        zoom, tile_x + width - 1, tile_y + height - 1)
    return northwest[0], southeast[1], southeast[2], northwest[3]


def _image_server_export_parameters(
        zoom: int, tile_x: int, tile_y: int,
        lock_raster_ids: Iterable[int], *,
        width: int = 1, height: int = 1) -> dict:
    bbox = _web_mercator_tile_range_bbox(
        zoom, tile_x, tile_y, width=width, height=height)
    tile_px = region.DEFAULT_TILE_PX
    return {
        "bbox": ",".join(format(value, ".17g") for value in bbox),
        "bboxSR": "3857",
        "imageSR": "3857",
        "size": f"{width * tile_px},{height * tile_px}",
        "format": IMAGE_SERVER_EXPORT_FORMAT,
        "interpolation": IMAGE_SERVER_INTERPOLATION,
        "renderingRule": _json_parameter({
            "rasterFunction": IMAGE_SERVER_RASTER_FUNCTION,
        }),
        "mosaicRule": _json_parameter(
            _image_server_mosaic_rule(lock_raster_ids)),
        "f": "image",
    }


def _cached_map_provider_contract(
        service_url: str, metadata: dict, grid: dict, *,
        esri_wayback_release: str | None = None) -> dict:
    required = ["capabilities", "fullExtent", "tileInfo"]
    if esri_wayback_release is None:
        required.append("name")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise SatelliteError(f"service metadata is missing {missing}")
    capabilities = {
        item.strip() for item in str(metadata["capabilities"]).split(",")}
    if "Tilemap" not in capabilities:
        raise SatelliteError("ArcGIS service does not advertise Tilemap")
    tile_info = metadata["tileInfo"]
    if not isinstance(tile_info, dict):
        raise SatelliteError("service tileInfo is not an object")
    tile_px = grid["tile_px"]
    if tile_info.get("rows") != tile_px or tile_info.get("cols") != tile_px:
        raise SatelliteError(
            f"service tile size is not {tile_px}x{tile_px}")
    spatial_reference = tile_info.get("spatialReference") or {}
    wkids = {spatial_reference.get("wkid"),
             spatial_reference.get("latestWkid")}
    if not ({3857, 102100} & wkids):
        raise SatelliteError(
            f"service tile grid is not Web Mercator: {spatial_reference}")
    zoom = grid["zoom"]
    min_lod = metadata.get("minLOD")
    max_lod = metadata.get("maxLOD")
    if esri_wayback_release is not None:
        levels = [entry.get("level") for entry in tile_info.get("lods", [])
                  if type(entry.get("level")) is int]
        if min_lod is None and levels:
            min_lod = min(levels)
        if max_lod is None and levels:
            max_lod = max(levels)
    if (type(min_lod) is not int or type(max_lod) is not int
            or not min_lod <= zoom <= max_lod):
        raise SatelliteError(
            f"service native LOD range {min_lod}..{max_lod} does not "
            f"include z{zoom}")
    lod = next(
        (entry for entry in tile_info.get("lods", [])
         if entry.get("level") == zoom), None)
    if lod is None:
        raise SatelliteError(f"service tileInfo has no z{zoom} LOD")

    tile_x_min, tile_y_min, tile_x_max, tile_y_max = (
        grid["source_tile_range_xyxy"])
    west_m, north_m = _web_mercator_m_from_pixel(
        tile_x_min * tile_px, tile_y_min * tile_px, zoom, tile_px)
    east_m, south_m = _web_mercator_m_from_pixel(
        (tile_x_max + 1) * tile_px, (tile_y_max + 1) * tile_px,
        zoom, tile_px)
    extent = metadata["fullExtent"]
    try:
        extent_values = (
            float(extent["xmin"]), float(extent["ymin"]),
            float(extent["xmax"]), float(extent["ymax"]))
    except (KeyError, TypeError, ValueError) as error:
        raise SatelliteError("service fullExtent is invalid") from error
    tolerance_m = 1.0
    if not (extent_values[0] <= west_m + tolerance_m
            and extent_values[1] <= south_m + tolerance_m
            and extent_values[2] >= east_m - tolerance_m
            and extent_values[3] >= north_m - tolerance_m):
        raise SatelliteError(
            "required source-tile rectangle lies outside service fullExtent")

    provider = {
        "schema": "arcgis_cached_imagery_provider/v1",
        "type": "arcgis_cached_map_service",
        "service_url": service_url.rstrip("/"),
        "service_name": metadata.get("name") or "World_Imagery",
        "service_item_id": metadata.get("serviceItemId"),
        "service_metadata_sha256": artifact.sha256_json(metadata),
        "capabilities": sorted(capabilities),
        "full_extent_web_mercator": list(extent_values),
        "min_lod": min_lod,
        "max_lod": max_lod,
        "tile_px": tile_px,
        "tile_format": tile_info.get("format"),
        "tile_compression_quality": tile_info.get("compressionQuality"),
        "lod_resolution_web_mercator_m_per_px": lod.get("resolution"),
        "source_description": metadata.get("description"),
        "document_info": metadata.get("documentInfo"),
        "copyright_text": metadata.get("copyrightText"),
        "export_tiles_allowed": metadata.get("exportTilesAllowed"),
        "max_export_tiles_count": metadata.get("maxExportTilesCount"),
    }
    if esri_wayback_release is not None:
        provider.update({
            "esri_wayback_release": esri_wayback_release,
            "tile_service_url": ESRI_WAYBACK_TILE_SERVICE_URL,
        })
    return provider


def _image_server_provider_contract(
        service_url: str, metadata: dict, grid: dict,
        catalog_audit: dict, *, catalog_where: str,
        lock_raster_ids: Iterable[int],
        image_server_chunk_tiles: int = 1) -> dict:
    chunk_tiles = _normalize_image_server_chunk_tiles(
        image_server_chunk_tiles)
    required = [
        "name", "capabilities", "fullExtent",
        "maxImageWidth", "maxImageHeight",
    ]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise SatelliteError(
            f"ImageServer metadata is missing {missing}")
    capabilities = {
        item.strip() for item in str(metadata["capabilities"]).split(",")}
    missing_capabilities = {"Catalog", "Image"} - capabilities
    if missing_capabilities:
        raise SatelliteError(
            "ImageServer does not advertise required capabilities: "
            f"{sorted(missing_capabilities)}")
    if (type(metadata["maxImageWidth"]) is not int
            or type(metadata["maxImageHeight"]) is not int
            or metadata["maxImageWidth"] < grid["tile_px"] * chunk_tiles
            or metadata["maxImageHeight"] < grid["tile_px"] * chunk_tiles):
        raise SatelliteError(
            "ImageServer maximum export dimensions are smaller than the "
            f"requested {chunk_tiles}x{chunk_tiles}-tile chunk")
    extent = metadata["fullExtent"]
    try:
        extent_values = (
            float(extent["xmin"]), float(extent["ymin"]),
            float(extent["xmax"]), float(extent["ymax"]))
    except (KeyError, TypeError, ValueError) as error:
        raise SatelliteError("ImageServer fullExtent is invalid") from error
    if not all(math.isfinite(value) for value in extent_values) \
            or extent_values[0] >= extent_values[2] \
            or extent_values[1] >= extent_values[3]:
        raise SatelliteError("ImageServer fullExtent is invalid")
    spatial_reference = extent.get("spatialReference") or {}
    wkids = {
        spatial_reference.get("wkid"),
        spatial_reference.get("latestWkid"),
    }
    if {3857, 102100} & wkids:
        tile_x_min, tile_y_min, tile_x_max, tile_y_max = (
            grid["source_tile_range_xyxy"])
        west_m, _, _, north_m = _web_mercator_tile_bbox(
            grid["zoom"], tile_x_min, tile_y_min,
            tile_px=grid["tile_px"])
        _, south_m, east_m, _ = _web_mercator_tile_bbox(
            grid["zoom"], tile_x_max, tile_y_max,
            tile_px=grid["tile_px"])
        tolerance_m = 1.0
        if not (extent_values[0] <= west_m + tolerance_m
                and extent_values[1] <= south_m + tolerance_m
                and extent_values[2] >= east_m - tolerance_m
                and extent_values[3] >= north_m - tolerance_m):
            raise SatelliteError(
                "required source-tile rectangle lies outside ImageServer "
                "fullExtent")

    raster_ids = _normalize_lock_raster_ids(lock_raster_ids)
    provider = {
        "schema": "arcgis_image_server_export_provider/v1",
        "type": IMAGE_SERVER_PROVIDER,
        "service_url": service_url.rstrip("/"),
        "service_name": metadata["name"],
        "service_item_id": metadata.get("serviceItemId"),
        "service_metadata_sha256": artifact.sha256_json(metadata),
        "capabilities": sorted(capabilities),
        "full_extent": list(extent_values),
        "full_extent_spatial_reference": spatial_reference,
        "max_image_width": metadata["maxImageWidth"],
        "max_image_height": metadata["maxImageHeight"],
        "native_pixel_size": {
            "x": metadata.get("pixelSizeX"),
            "y": metadata.get("pixelSizeY"),
            "mean": metadata.get("meanPixelSize"),
        },
        "band_count": metadata.get("bandCount"),
        "pixel_type": metadata.get("pixelType"),
        "source_description": metadata.get("description"),
        "copyright_text": metadata.get("copyrightText"),
        "catalog": {
            "endpoint": service_url.rstrip("/") + "/query",
            "where": str(catalog_where).strip(),
            "response_sha256": catalog_audit["response_sha256"],
            "feature_count": catalog_audit["feature_count"],
            "lock_raster_ids": list(raster_ids),
        },
        "export": {
            "endpoint": service_url.rstrip("/") + "/exportImage",
            "bbox_spatial_reference": 3857,
            "image_spatial_reference": 3857,
            "tile_px": grid["tile_px"],
            "format": IMAGE_SERVER_EXPORT_FORMAT,
            "interpolation": IMAGE_SERVER_INTERPOLATION,
            "rendering_rule": {
                "rasterFunction": IMAGE_SERVER_RASTER_FUNCTION,
            },
            "mosaic_rule": _image_server_mosaic_rule(raster_ids),
            "tile_envelope": {
                "schema": "web_mercator_xyz_bbox/v1",
                "earth_radius_m": geometry.EARTH_RADIUS_M,
                "row_zero": "north",
                "row_direction": "south",
            },
        },
    }
    if chunk_tiles > 1:
        provider["export"]["chunking"] = _image_server_chunk_contract(
            chunk_tiles)
    return provider


def _normalize_image_server_chunk_tiles(value: int) -> int:
    if (type(value) is not int
            or not 1 <= value <= IMAGE_SERVER_MAX_CHUNK_TILES):
        raise SatelliteError(
            "ImageServer chunk tile count must be an integer in "
            f"1..{IMAGE_SERVER_MAX_CHUNK_TILES}")
    return value


def _image_server_chunk_contract(chunk_tiles: int) -> dict:
    chunk_tiles = _normalize_image_server_chunk_tiles(chunk_tiles)
    return {
        "schema": IMAGE_SERVER_CHUNK_SCHEMA,
        "shape_tiles_xy": [chunk_tiles, chunk_tiles],
        "maximum_export_size_px": [
            chunk_tiles * region.DEFAULT_TILE_PX,
            chunk_tiles * region.DEFAULT_TILE_PX,
        ],
        "partition_anchor": "region_source_tile_range_northwest",
        "partition_order": "row_major",
        "bbox": "exact_web_mercator_xyz_tile_union",
        "crop_orientation": "north_up_row_major_256px",
        "algorithm": IMAGE_SERVER_CHUNK_ALGORITHM,
        "child_encoding": IMAGE_SERVER_SOURCE_TILE_ENCODING,
        "commit_receipts": "atomic_after_all_child_tiles_v1",
    }


def _provider_request_contract(
        provider_mode: str, service_url: str, *,
        source_index_url: str | None,
        catalog_where: str | None,
        lock_raster_ids: Iterable[int],
        esri_wayback_release: str | None = None,
        require_source_index_coverage: bool = True,
        image_server_chunk_tiles: int = 1) -> dict:
    chunk_tiles = _normalize_image_server_chunk_tiles(
        image_server_chunk_tiles)
    if provider_mode not in PROVIDER_MODES:
        raise SatelliteError(
            f"unsupported imagery provider mode {provider_mode!r}")
    service_url = str(service_url).rstrip("/")
    if not service_url.startswith("https://"):
        raise SatelliteError("ArcGIS service URL must use https")
    esri_wayback_release = _normalize_esri_wayback_release(
        esri_wayback_release)
    raster_ids = tuple(lock_raster_ids)
    if provider_mode == CACHED_MAP_PROVIDER:
        if chunk_tiles != 1:
            raise SatelliteError(
                "ImageServer chunking applies only to ImageServer export "
                "mode")
        if catalog_where is not None or raster_ids:
            raise SatelliteError(
                "catalog where/lock raster IDs apply only to ImageServer "
                "export mode")
        if (esri_wayback_release is not None
                and service_url != ESRI_WORLD_IMAGERY_SERVICE_URL):
            raise SatelliteError(
                "ESRI Wayback requires the standard World Imagery "
                "MapServer metadata URL")
        request = {
            "provider_mode": provider_mode,
            "service_url": service_url,
            "source_index_url": source_index_url,
        }
        if esri_wayback_release is not None:
            request["esri_wayback_release"] = esri_wayback_release
        return request
    if esri_wayback_release is not None:
        raise SatelliteError(
            "ESRI Wayback release applies only to cached-map mode")
    if source_index_url is not None:
        raise SatelliteError(
            "ImageServer export uses its own catalog; source_index_url must "
            "be unset")
    if not require_source_index_coverage:
        raise SatelliteError(
            "--allow_incomplete_source_index applies only to cached-map "
            "mode; ImageServer catalog coverage is always strict")
    catalog_where = str(catalog_where or "").strip()
    if not catalog_where:
        raise SatelliteError(
            "ImageServer export requires --catalog_where")
    normalized_ids = _normalize_lock_raster_ids(raster_ids)
    request = {
        "provider_mode": provider_mode,
        "service_url": service_url,
        "catalog_where": catalog_where,
        "lock_raster_ids": list(normalized_ids),
        "export_format": IMAGE_SERVER_EXPORT_FORMAT,
        "export_interpolation": IMAGE_SERVER_INTERPOLATION,
        "export_raster_function": IMAGE_SERVER_RASTER_FUNCTION,
        "export_mosaic_operation": IMAGE_SERVER_MOSAIC_OPERATION,
    }
    if chunk_tiles > 1:
        request["export_chunking"] = _image_server_chunk_contract(
            chunk_tiles)
    return request


def _iter_tilemap_chunks(grid: dict):
    tile_x_min, tile_y_min, tile_x_max, tile_y_max = (
        grid["source_tile_range_xyxy"])
    tile_y = tile_y_min
    while tile_y <= tile_y_max:
        height = min(
            tile_y_max - tile_y + 1,
            ARCGIS_BUNDLE_SIZE - tile_y % ARCGIS_BUNDLE_SIZE)
        tile_x = tile_x_min
        while tile_x <= tile_x_max:
            width = min(
                tile_x_max - tile_x + 1,
                ARCGIS_BUNDLE_SIZE - tile_x % ARCGIS_BUNDLE_SIZE)
            yield tile_x, tile_y, width, height
            tile_x += width
        tile_y += height


def _audit_source_index(client, source_index_url: str,
                        bbox_wsen: Iterable[float], *,
                        require_complete: bool = True) -> dict:
    west, south, east, north = tuple(float(item) for item in bbox_wsen)
    response = client.query_source_index(
        source_index_url, (west, south, east, north))
    if response.get("exceededTransferLimit"):
        raise SatelliteError(
            "source imagery index exceeded its transfer limit")
    features = response.get("features")
    if not isinstance(features, list) or not features:
        raise SatelliteError("source imagery index returned no features")
    polygons = []
    entries = []
    for feature in features:
        attributes = feature.get("attributes") or {}
        geometry_value = feature.get("geometry") or {}
        rings = geometry_value.get("rings")
        if not isinstance(rings, list) or len(rings) != 1:
            raise SatelliteError(
                "source imagery index feature is not one simple polygon")
        try:
            polygon = Polygon(rings[0])
        except Exception as error:
            raise SatelliteError(
                "source imagery index contains invalid geometry") from error
        if polygon.is_empty or not polygon.is_valid:
            raise SatelliteError(
                "source imagery index contains invalid geometry")
        polygons.append(polygon)
        entries.append({
            "object_id": attributes.get("OBJECTID"),
            "tile_name": attributes.get("TILE_NAME"),
            "zone": attributes.get("ZONE"),
            "download_url": attributes.get("URL"),
        })
    coverage = unary_union(polygons)
    requested = box(west, south, east, north)
    uncovered = requested.difference(coverage)
    covers = coverage.covers(requested)
    if not covers and require_complete:
        raise SatelliteError(
            "source orthophoto index does not cover the complete patch "
            f"footprint (uncovered coordinate area {uncovered.area})")
    query = {
        "f": "json",
        "where": "1=1",
        "geometry": f"{west},{south},{east},{north}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "OBJECTID,TILE_NAME,ZONE,URL",
        "returnGeometry": "true",
        "resultRecordCount": "2000",
    }
    return {
        "status": "passed" if covers else "informational_partial",
        "coverage_required": require_complete,
        "layer_url": source_index_url,
        "query_url": source_index_url.rstrip("/") + "/query",
        "query_parameters": query,
        "response_sha256": artifact.sha256_json(response),
        "feature_count": len(features),
        "zones": sorted({entry["zone"] for entry in entries
                         if entry["zone"] is not None}),
        "covers_footprint": covers,
        "uncovered_coordinate_area": uncovered.area,
        "source_tiles": sorted(
            entries,
            key=lambda item: (str(item["tile_name"]),
                              str(item["object_id"]))),
    }


def _audit_image_server_catalog(
        client, service_url: str, bbox_wsen: Iterable[float], *,
        catalog_where: str, lock_raster_ids: Iterable[int]) -> dict:
    west, south, east, north = tuple(float(item) for item in bbox_wsen)
    expected_ids = _normalize_lock_raster_ids(lock_raster_ids)
    response = client.query_catalog((west, south, east, north))
    if response.get("exceededTransferLimit"):
        raise SatelliteError(
            "ImageServer catalog exceeded its transfer limit")
    features = response.get("features")
    if not isinstance(features, list) or not features:
        raise SatelliteError("ImageServer catalog returned no features")

    polygons = []
    entries = []
    observed_ids = []
    for feature in features:
        if not isinstance(feature, dict):
            raise SatelliteError(
                "ImageServer catalog contains a non-object feature")
        attributes = feature.get("attributes")
        geometry_value = feature.get("geometry")
        if not isinstance(attributes, dict) \
                or not isinstance(geometry_value, dict):
            raise SatelliteError(
                "ImageServer catalog feature lacks attributes or geometry")
        object_id = attributes.get("OBJECTID")
        if type(object_id) is not int or object_id <= 0:
            raise SatelliteError(
                "ImageServer catalog OBJECTID must be a positive integer")
        if attributes.get("Category") != 1:
            raise SatelliteError(
                f"ImageServer raster {object_id} is not Category=1 primary "
                "imagery")
        rings = geometry_value.get("rings")
        if not isinstance(rings, list) or len(rings) != 1:
            raise SatelliteError(
                "ImageServer catalog feature is not one simple polygon")
        try:
            polygon = Polygon(rings[0])
        except Exception as error:
            raise SatelliteError(
                "ImageServer catalog contains invalid geometry") from error
        if polygon.is_empty or not polygon.is_valid:
            raise SatelliteError(
                "ImageServer catalog contains invalid geometry")
        polygons.append(polygon)
        observed_ids.append(object_id)
        entries.append({
            "object_id": object_id,
            "attributes": {
                field: attributes.get(field)
                for field in IMAGE_SERVER_CATALOG_FIELDS
            },
            "geometry": geometry_value,
            "footprint_sha256": artifact.sha256_json(geometry_value),
        })

    if len(observed_ids) != len(set(observed_ids)):
        raise SatelliteError(
            "ImageServer catalog returned duplicate OBJECTIDs")
    observed_ids = sorted(observed_ids)
    if tuple(observed_ids) != expected_ids:
        missing = sorted(set(expected_ids) - set(observed_ids))
        unexpected = sorted(set(observed_ids) - set(expected_ids))
        raise SatelliteError(
            "ImageServer catalog does not match lock raster IDs: "
            f"missing={missing}, unexpected={unexpected}")

    coverage = unary_union(polygons)
    requested = box(west, south, east, north)
    uncovered = requested.difference(coverage)
    if not coverage.covers(requested):
        raise SatelliteError(
            "ImageServer source catalog does not cover the complete patch "
            f"footprint (uncovered coordinate area {uncovered.area})")
    query = _image_server_catalog_parameters(
        (west, south, east, north), catalog_where)
    return {
        "status": "passed",
        "coverage_required": True,
        "query_url": service_url.rstrip("/") + "/query",
        "query_parameters": query,
        "response_sha256": artifact.sha256_json(response),
        "feature_count": len(features),
        "lock_raster_ids": list(expected_ids),
        "covers_footprint": True,
        "uncovered_coordinate_area": uncovered.area,
        "source_rasters": sorted(
            entries, key=lambda item: item["object_id"]),
    }


def audit_coverage(client, plan: dict, service_metadata: dict, *,
                   service_url: str = DEFAULT_SERVICE_URL,
                   source_index_url: str | None = DEFAULT_SOURCE_INDEX_URL,
                   require_source_index_coverage: bool = True,
                   provider_mode: str = DEFAULT_PROVIDER_MODE,
                   catalog_where: str | None = None,
                   lock_raster_ids: Iterable[int] = (),
                   esri_wayback_release: str | None = None,
                   image_server_chunk_tiles: int = 1) \
        -> dict:
    """Strictly prove cache and source-index coverage for a region plan."""
    grid = plan.get("grid")
    if not isinstance(grid, dict):
        raise SatelliteError("region plan has no grid")
    request = _provider_request_contract(
        provider_mode, service_url, source_index_url=source_index_url,
        catalog_where=catalog_where, lock_raster_ids=lock_raster_ids,
        esri_wayback_release=esri_wayback_release,
        require_source_index_coverage=require_source_index_coverage,
        image_server_chunk_tiles=image_server_chunk_tiles)
    esri_wayback_release = request.get("esri_wayback_release")
    if provider_mode == IMAGE_SERVER_PROVIDER:
        rendered_footprint = _rendered_footprint_bbox_wsen(grid)
        catalog = _audit_image_server_catalog(
            client, service_url, rendered_footprint,
            catalog_where=request["catalog_where"],
            lock_raster_ids=request["lock_raster_ids"])
        provider = _image_server_provider_contract(
            service_url, service_metadata, grid, catalog,
            catalog_where=request["catalog_where"],
            lock_raster_ids=request["lock_raster_ids"],
            image_server_chunk_tiles=image_server_chunk_tiles)
        return {
            "schema": COVERAGE_SCHEMA,
            "status": "passed",
            "provider": provider,
            "provider_request": request,
            "service_metadata": service_metadata,
            "footprint_bbox_wsen": grid["footprint_bbox_wsen"],
            "rendered_footprint_bbox_wsen": rendered_footprint,
            "tilemap": {
                "status": "not_applicable",
                "reason": "dynamic ImageServer exports have no Tilemap",
            },
            "catalog": catalog,
            "source_index": {"status": "not_applicable"},
        }

    provider = _cached_map_provider_contract(
        service_url, service_metadata, grid,
        esri_wayback_release=esri_wayback_release)
    chunks = []
    n_present = 0
    n_missing = 0
    missing_examples = []
    for tile_x, tile_y, width, height in _iter_tilemap_chunks(grid):
        response = client.get_tilemap(
            grid["zoom"], tile_x, tile_y, width, height)
        location = response.get("location") or {}
        expected_location = {
            "left": tile_x, "top": tile_y,
            "width": width, "height": height}
        if location != expected_location or response.get("adjusted") is True:
            raise SatelliteError(
                "ArcGIS Tilemap returned an adjusted or mismatched window: "
                f"expected {expected_location}, found {location}")
        data = response.get("data")
        if (not isinstance(data, list) or len(data) != width * height
                or any(value not in (0, 1) for value in data)):
            raise SatelliteError("ArcGIS Tilemap returned invalid presence data")
        present = sum(data)
        missing = len(data) - present
        n_present += present
        n_missing += missing
        if missing:
            for index, value in enumerate(data):
                if value == 0 and len(missing_examples) < 20:
                    missing_examples.append({
                        "tile_x": tile_x + index % width,
                        "tile_y": tile_y + index // width,
                    })
        chunks.append({
            "query": {
                "zoom": grid["zoom"], "tile_x": tile_x,
                "tile_y": tile_y, "width": width, "height": height,
            },
            "response_sha256": artifact.sha256_json(response),
            "present": present,
            "missing": missing,
        })
    expected_tiles = grid["n_source_tiles"]
    if n_present + n_missing != expected_tiles:
        raise SatelliteError(
            "Tilemap audit count differs from the region grid: "
            f"{n_present + n_missing} != {expected_tiles}")
    if n_missing:
        raise MissingTileError(
            f"ArcGIS Tilemap reports {n_missing}/{expected_tiles} required "
            f"z{grid['zoom']} tiles missing; examples: {missing_examples}")

    if source_index_url is None:
        source_index = {"status": "not_configured"}
    else:
        source_index = _audit_source_index(
            client, source_index_url, grid["footprint_bbox_wsen"],
            require_complete=require_source_index_coverage)
    tile_service_url = (
        ESRI_WAYBACK_TILE_SERVICE_URL
        if esri_wayback_release is not None else service_url.rstrip("/"))
    release_path = (f"/{esri_wayback_release}"
                    if esri_wayback_release is not None else "")
    return {
        "schema": COVERAGE_SCHEMA,
        "status": "passed",
        "provider": provider,
        "service_metadata": service_metadata,
        "footprint_bbox_wsen": grid["footprint_bbox_wsen"],
        "tilemap": {
            "status": "passed",
            "endpoint_template": (
                tile_service_url + "/tilemap" + release_path
                + "/{zoom}/{tile_y}/{tile_x}/{width}/{height}?f=json"),
            "bundle_size": ARCGIS_BUNDLE_SIZE,
            "n_queries": len(chunks),
            "n_required_tiles": expected_tiles,
            "n_present": n_present,
            "n_missing": n_missing,
            "chunks": chunks,
        },
        "source_index": source_index,
    }


T = TypeVar("T")
R = TypeVar("R")


def _bounded_map(function: Callable[[T], R], values: Iterable[T], *,
                 workers: int, max_in_flight: int | None = None) \
        -> Iterator[R]:
    if workers < 1:
        raise SatelliteError("worker count must be positive")
    limit = max_in_flight or workers * 2
    if limit < workers:
        raise SatelliteError("max_in_flight must be at least workers")
    iterator = iter(values)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = set()
        for _ in range(limit):
            try:
                pending.add(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                yield future.result()
                try:
                    pending.add(executor.submit(function, next(iterator)))
                except StopIteration:
                    pass


def _image_info_json(info: ImageInfo) -> dict:
    value = asdict(info)
    value["mean_rgb"] = list(value["mean_rgb"])
    return value


def _source_chunk_descriptor(
        grid: dict, chunk_tiles: int,
        tile_x: int, tile_y: int, width: int, height: int) -> dict:
    bbox = _web_mercator_tile_range_bbox(
        grid["zoom"], tile_x, tile_y, width=width, height=height)
    return {
        "schema": IMAGE_SERVER_CHUNK_SCHEMA,
        "chunking": _image_server_chunk_contract(chunk_tiles),
        "zoom": grid["zoom"],
        "tile_x": tile_x,
        "tile_y": tile_y,
        "width": width,
        "height": height,
        "export_bbox_web_mercator_m": list(bbox),
        "export_size_px": [width * grid["tile_px"],
                           height * grid["tile_px"]],
    }


def _valid_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _load_valid_source_chunk_receipt(
        build_dir: Path, grid: dict, chunk_tiles: int,
        tile_x: int, tile_y: int, width: int, height: int) \
        -> dict | None:
    """Return a receipt only if every committed child still matches it."""
    receipt_path = _source_chunk_receipt_path(
        build_dir, grid["zoom"], tile_x, tile_y, width, height)
    if not receipt_path.exists():
        return None
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise SatelliteError(f"invalid source chunk receipt: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    descriptor = _source_chunk_descriptor(
        grid, chunk_tiles, tile_x, tile_y, width, height)
    if not isinstance(receipt, dict) or any(
            receipt.get(key) != value for key, value in descriptor.items()):
        return None
    response = receipt.get("response")
    if (not isinstance(response, dict)
            or response.get("image_format") != "PNG"
            or response.get("mode") != "RGB"
            or response.get("width") != width * grid["tile_px"]
            or response.get("height") != height * grid["tile_px"]
            or not _valid_sha256(response.get("sha256"))
            or not _valid_sha256(response.get("decoded_pixel_sha256"))):
        return None
    records = receipt.get("tiles")
    expected_coordinates = [
        (x, y)
        for y in range(tile_y, tile_y + height)
        for x in range(tile_x, tile_x + width)
    ]
    if not isinstance(records, list) or len(records) != len(
            expected_coordinates):
        return None
    for record, (child_x, child_y) in zip(records, expected_coordinates):
        if (not isinstance(record, dict)
                or record.get("tile_x") != child_x
                or record.get("tile_y") != child_y):
            return None
        path = _tile_cache_path(
            build_dir, grid["zoom"], child_x, child_y)
        if path.is_symlink():
            raise SatelliteError(
                f"cached source tile cannot be a symlink: {path}")
        try:
            value = path.read_bytes()
            image, info = _validate_image_bytes(
                value, (grid["tile_px"], grid["tile_px"]), str(path))
        except (OSError, SatelliteError):
            return None
        if (info.image_format != "PNG" or info.mode != "RGB"
                or image.getbbox() is None):
            return None
        expected_record = {
            "tile_x": child_x,
            "tile_y": child_y,
            **_image_info_json(info),
        }
        if record != expected_record:
            return None
    return receipt


def _ensure_image_server_source_chunks(
        build_dir: Path, grid: dict, client, *, workers: int,
        progress_every: int, chunk_tiles: int) -> dict:
    fetch_chunk = getattr(client, "fetch_tile_chunk", None)
    if not callable(fetch_chunk):
        raise SatelliteError(
            "ImageServer chunking requires a client with fetch_tile_chunk")
    chunk_tiles = _normalize_image_server_chunk_tiles(chunk_tiles)
    chunk_workers = min(workers, IMAGE_SERVER_CHUNK_WORKERS)
    if chunk_workers < 1:
        raise SatelliteError("worker count must be positive")
    zoom = grid["zoom"]
    tile_px = grid["tile_px"]

    def ensure_chunk(item: tuple[int, int, int, int]) -> dict:
        tile_x, tile_y, width, height = item
        receipt = _load_valid_source_chunk_receipt(
            build_dir, grid, chunk_tiles,
            tile_x, tile_y, width, height)
        if receipt is not None:
            return {
                "total": width * height,
                "resumed": width * height,
                "downloaded": 0,
                "replaced": 0,
            }

        coordinates = [
            (x, y)
            for y in range(tile_y, tile_y + height)
            for x in range(tile_x, tile_x + width)
        ]
        existed = {
            coordinate: _tile_cache_path(
                build_dir, zoom, *coordinate).exists()
            for coordinate in coordinates
        }
        result = fetch_chunk(
            zoom, tile_x, tile_y, width, height)
        if (not isinstance(result, ImageServerTileChunk)
                or (result.zoom, result.tile_x, result.tile_y,
                    result.width, result.height)
                != (zoom, tile_x, tile_y, width, height)
                or [(tile.tile_x, tile.tile_y) for tile in result.tiles]
                != coordinates):
            raise SatelliteError(
                "ImageServer chunk client returned a mismatched tile set")

        records = []
        for tile in result.tiles:
            image, info = _validate_image_bytes(
                tile.value, (tile_px, tile_px),
                f"chunk child z{zoom}/{tile.tile_x}/{tile.tile_y}")
            if (info != tile.info or info.image_format != "PNG"
                    or info.mode != "RGB" or image.getbbox() is None):
                raise SatelliteError(
                    "ImageServer chunk client returned an invalid canonical "
                    f"child at z{zoom}/{tile.tile_x}/{tile.tile_y}")
            path = _tile_cache_path(
                build_dir, zoom, tile.tile_x, tile.tile_y)
            artifact.atomic_write_file(path, tile.value)
            written = validate_image_file(path, (tile_px, tile_px))
            if written != info:
                raise SatelliteError(
                    f"atomic chunk tile write did not validate: {path}")
            records.append({
                "tile_x": tile.tile_x,
                "tile_y": tile.tile_y,
                **_image_info_json(info),
            })

        receipt_document = {
            **_source_chunk_descriptor(
                grid, chunk_tiles, tile_x, tile_y, width, height),
            "response": _image_info_json(result.response_info),
            "tiles": records,
        }
        receipt_path = _source_chunk_receipt_path(
            build_dir, zoom, tile_x, tile_y, width, height)
        artifact.atomic_write_json(receipt_path, receipt_document)
        if _load_valid_source_chunk_receipt(
                build_dir, grid, chunk_tiles,
                tile_x, tile_y, width, height) is None:
            raise SatelliteError(
                f"atomic source chunk receipt did not validate: "
                f"{receipt_path}")
        downloaded = sum(not existed[coordinate]
                         for coordinate in coordinates)
        return {
            "total": len(coordinates),
            "resumed": 0,
            "downloaded": downloaded,
            "replaced": len(coordinates) - downloaded,
        }

    counts = {"total": 0, "resumed": 0, "downloaded": 0, "replaced": 0}
    last_report = 0
    chunks = _iter_image_server_source_chunks(grid, chunk_tiles)
    for result in _bounded_map(
            ensure_chunk, chunks, workers=chunk_workers,
            max_in_flight=chunk_workers):
        for key in counts:
            counts[key] += result[key]
        if (progress_every
                and counts["total"] // progress_every > last_report):
            last_report = counts["total"] // progress_every
            print(
                f"  source tiles {counts['total']}/{grid['n_source_tiles']} "
                f"(cached={counts['resumed']}, new={counts['downloaded']}, "
                f"repaired={counts['replaced']})")
    if counts["total"] != grid["n_source_tiles"]:
        raise SatelliteError(
            f"processed {counts['total']} source tiles, expected "
            f"{grid['n_source_tiles']}")
    return counts


def ensure_source_tiles(build_dir: Path, grid: dict, client, *,
                        workers: int = 32, progress_every: int = 1000,
                        image_server_chunk_tiles: int = 1) -> dict:
    """Download/repair every source tile, retaining hash-matched entries."""
    build_dir = Path(build_dir)
    chunk_tiles = _normalize_image_server_chunk_tiles(
        image_server_chunk_tiles)
    if chunk_tiles > 1:
        return _ensure_image_server_source_chunks(
            build_dir, grid, client, workers=workers,
            progress_every=progress_every, chunk_tiles=chunk_tiles)
    zoom = grid["zoom"]
    tile_px = grid["tile_px"]
    prior_hashes: dict[tuple[int, int], tuple[str, str]] = {}
    prior_manifest_path = build_dir / SOURCE_MANIFEST
    if prior_manifest_path.exists():
        if prior_manifest_path.is_symlink() \
                or not prior_manifest_path.is_file():
            raise SatelliteError(
                f"invalid prior source manifest: {prior_manifest_path}")
        try:
            prior_manifest = json.loads(
                prior_manifest_path.read_text(encoding="utf-8"))
            if (prior_manifest.get("schema") != SOURCE_MANIFEST_SCHEMA
                    or prior_manifest.get("grid") != grid):
                raise SatelliteError(
                    f"prior source manifest recipe mismatch: "
                    f"{prior_manifest_path}")
            for record in prior_manifest.get("tiles", []):
                key = (record["tile_x"], record["tile_y"])
                value = (record["sha256"], record["decoded_pixel_sha256"])
                if key in prior_hashes:
                    raise SatelliteError(
                        f"duplicate prior source manifest tile: {key}")
                prior_hashes[key] = value
            if len(prior_hashes) != grid["n_source_tiles"]:
                raise SatelliteError(
                    f"prior source manifest contains {len(prior_hashes)} "
                    f"tiles, expected {grid['n_source_tiles']}")
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError,
                TypeError) as error:
            raise SatelliteError(
                f"invalid prior source manifest {prior_manifest_path}: "
                f"{error}") from error
    counts = {"total": 0, "resumed": 0, "downloaded": 0, "replaced": 0}

    def ensure_one(item: tuple[int, int]) -> str:
        tile_x, tile_y = item
        path = _tile_cache_path(build_dir, zoom, tile_x, tile_y)
        existed = path.exists()
        info = validate_image_file(path, (tile_px, tile_px))
        prior = prior_hashes.get((tile_x, tile_y))
        if info is not None and (not prior_hashes or prior == (
                info.sha256, info.decoded_pixel_sha256)):
            return "resumed"
        value = client.fetch_tile(zoom, tile_x, tile_y)
        if not isinstance(value, bytes):
            raise SatelliteError(
                f"tile client returned non-bytes for z{zoom}/{tile_x}/{tile_y}")
        _validate_image_bytes(
            value, (tile_px, tile_px), f"z{zoom}/{tile_x}/{tile_y}")
        artifact.atomic_write_file(path, value)
        if validate_image_file(path, (tile_px, tile_px)) is None:
            raise SatelliteError(
                f"atomic tile write did not validate: {path}")
        return "replaced" if existed else "downloaded"

    for status in _bounded_map(
            ensure_one, _grid_tile_items(grid), workers=workers):
        counts["total"] += 1
        counts[status] += 1
        if progress_every and counts["total"] % progress_every == 0:
            print(
                f"  source tiles {counts['total']}/{grid['n_source_tiles']} "
                f"(cached={counts['resumed']}, new={counts['downloaded']}, "
                f"repaired={counts['replaced']})")
    if counts["total"] != grid["n_source_tiles"]:
        raise SatelliteError(
            f"processed {counts['total']} source tiles, expected "
            f"{grid['n_source_tiles']}")
    return counts


def _write_streamed_manifest(path: Path, header: dict,
                             collection_name: str,
                             records: Iterable[dict]) -> int:
    count = 0
    with _atomic_text_writer(path) as stream:
        stream.write("{")
        first = True
        for key in sorted(header):
            if not first:
                stream.write(",")
            first = False
            stream.write(json.dumps(key))
            stream.write(":")
            stream.write(json.dumps(
                header[key], sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False))
        if not first:
            stream.write(",")
        stream.write(json.dumps(collection_name))
        stream.write(":[")
        for record in records:
            if count:
                stream.write(",")
            stream.write(json.dumps(
                record, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False))
            count += 1
        stream.write("]}")
        stream.write("\n")
    return count


def write_source_tile_manifest(build_dir: Path, grid: dict,
                               provider: dict) -> dict:
    """Validate every cached source tile and write its exact byte hashes."""
    build_dir = Path(build_dir)
    zoom = grid["zoom"]
    tile_px = grid["tile_px"]
    path = build_dir / SOURCE_MANIFEST

    def records():
        for tile_x, tile_y in _grid_tile_items(grid):
            tile_path = _tile_cache_path(
                build_dir, zoom, tile_x, tile_y)
            info = validate_image_file(tile_path, (tile_px, tile_px))
            if info is None:
                raise SatelliteError(
                    f"required cached source tile is missing/corrupt: "
                    f"{tile_path}")
            yield {
                "zoom": zoom,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "cache_key": f"{zoom}/{tile_x}/{tile_y}.tile",
                **asdict(info),
            }

    count = _write_streamed_manifest(
        path,
        {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "provider": provider,
            "grid": grid,
            "n_source_tiles": grid["n_source_tiles"],
        },
        "tiles", records())
    if count != grid["n_source_tiles"]:
        raise SatelliteError(
            f"source manifest contains {count} tiles, expected "
            f"{grid['n_source_tiles']}")
    return {"path": path, "sha256": artifact.sha256_file(path),
            "n_tiles": count}


class _DecodedTileCache:
    """Thread-safe, bounded LRU of fully decoded RGB source tiles."""

    def __init__(self, build_dir: Path, grid: dict,
                 max_entries: int) -> None:
        if max_entries < 1:
            raise SatelliteError("decoded tile cache size must be positive")
        self._build_dir = Path(build_dir)
        self._grid = grid
        self._max_entries = max_entries
        self._items: OrderedDict[tuple[int, int], Image.Image] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, tile_x: int, tile_y: int) -> Image.Image:
        key = (tile_x, tile_y)
        with self._lock:
            found = self._items.pop(key, None)
            if found is not None:
                self._items[key] = found
                return found
        path = _tile_cache_path(
            self._build_dir, self._grid["zoom"], tile_x, tile_y)
        if path.is_symlink() or not path.is_file():
            raise SatelliteError(f"required source tile is absent: {path}")
        try:
            value = path.read_bytes()
        except OSError as error:
            raise SatelliteError(f"cannot read source tile {path}: {error}") \
                from error
        loaded, _ = _validate_image_bytes(
            value, (self._grid["tile_px"], self._grid["tile_px"]),
            str(path))
        with self._lock:
            existing = self._items.pop(key, None)
            if existing is not None:
                self._items[key] = existing
                return existing
            self._items[key] = loaded
            while len(self._items) > self._max_entries:
                self._items.popitem(last=False)
        return loaded


def assemble_patch(center_x: float, center_y: float, grid: dict,
                   tile_cache: _DecodedTileCache) -> Image.Image:
    """Assemble one patch after quantizing its origin to the nearest pixel.

    The release downloader truncated the fractional source and destination
    coordinates independently.  For non-integral Web-Mercator centres that
    left the last row and column unfilled.  Quantizing the origin exactly once
    keeps the sub-pixel location error below half a pixel and guarantees that
    all ``source_px`` rows and columns are copied.
    """
    source_px = grid["source_px"]
    tile_px = grid["tile_px"]
    origin_x = region.nearest_pixel_origin(center_x, source_px)
    origin_y = region.nearest_pixel_origin(center_y, source_px)
    tile_x_min = origin_x // tile_px
    tile_x_max = (origin_x + source_px - 1) // tile_px
    tile_y_min = origin_y // tile_px
    tile_y_max = (origin_y + source_px - 1) // tile_px
    patch = Image.new("RGB", (source_px, source_px))
    for tile_x in range(tile_x_min, tile_x_max + 1):
        for tile_y in range(tile_y_min, tile_y_max + 1):
            tile = tile_cache.get(tile_x, tile_y)
            tile_left = tile_x * tile_px
            tile_top = tile_y * tile_px
            source_left = max(0, origin_x - tile_left)
            source_top = max(0, origin_y - tile_top)
            source_right = min(
                tile_px, origin_x + source_px - tile_left)
            source_bottom = min(
                tile_px, origin_y + source_px - tile_top)
            cropped = tile.crop((
                source_left, source_top, source_right, source_bottom))
            paste_x = tile_left + source_left - origin_x
            paste_y = tile_top + source_top - origin_y
            patch.paste(cropped, (paste_x, paste_y))
    if source_px != grid["patch_px"]:
        patch = patch.resize(
            (grid["patch_px"], grid["patch_px"]), Image.Resampling.LANCZOS)
    return patch


def patch_filename(center_x: float, center_y: float, zoom: int) -> str:
    latitude, longitude = region.pixel_to_lat_lon(center_x, center_y, zoom)
    return f"satellite_{latitude:.8f}_{longitude:.8f}.jpg"


def _patch_config_digest(grid: dict, tile_manifest_sha256: str,
                         jpeg_quality: int) -> str:
    return artifact.sha256_json({
        "assembly_version": ASSEMBLY_VERSION,
        "grid": grid,
        "source_tile_manifest_sha256": tile_manifest_sha256,
        "jpeg_quality": jpeg_quality,
    })


def ensure_patches(build_dir: Path, grid: dict,
                   tile_manifest_sha256: str, *,
                   jpeg_quality: int = DEFAULT_JPEG_QUALITY,
                   workers: int = 8, progress_every: int = 1000,
                   decoded_tile_cache_entries: int = 1024) -> dict:
    """Assemble/repair every row-major LOCI patch, resuming valid JPEGs."""
    if not 1 <= jpeg_quality <= 100:
        raise SatelliteError("jpeg_quality must lie in [1, 100]")
    build_dir = Path(build_dir)
    patch_config_digest = _patch_config_digest(
        grid, tile_manifest_sha256, jpeg_quality)
    patch_root = build_dir / "patch_sets" / patch_config_digest
    patch_dir = patch_root / SATELLITE_DIR
    patch_dir.mkdir(parents=True, exist_ok=True)
    prior_hashes: dict[str, tuple[str, str]] = {}
    prior_manifest_path = patch_root / PATCH_MANIFEST
    if prior_manifest_path.exists():
        if prior_manifest_path.is_symlink() \
                or not prior_manifest_path.is_file():
            raise SatelliteError(
                f"invalid prior patch manifest: {prior_manifest_path}")
        try:
            prior_manifest = json.loads(
                prior_manifest_path.read_text(encoding="utf-8"))
            if (prior_manifest.get("schema") != PATCH_MANIFEST_SCHEMA
                    or prior_manifest.get("assembly_version")
                    != ASSEMBLY_VERSION
                    or prior_manifest.get("grid") != grid
                    or prior_manifest.get("source_tile_manifest_sha256")
                    != tile_manifest_sha256
                    or prior_manifest.get("jpeg_quality") != jpeg_quality):
                raise SatelliteError(
                    f"prior patch manifest recipe mismatch: "
                    f"{prior_manifest_path}")
            for record in prior_manifest.get("patches", []):
                filename = record["filename"]
                value = (record["sha256"], record["decoded_pixel_sha256"])
                if filename in prior_hashes:
                    raise SatelliteError(
                        f"duplicate prior patch manifest filename: {filename}")
                prior_hashes[filename] = value
            if len(prior_hashes) != grid["n_patches"]:
                raise SatelliteError(
                    f"prior patch manifest contains {len(prior_hashes)} "
                    f"patches, expected {grid['n_patches']}")
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError,
                TypeError) as error:
            raise SatelliteError(
                f"invalid prior patch manifest {prior_manifest_path}: "
                f"{error}") from error
    cache = _DecodedTileCache(
        build_dir, grid, max_entries=decoded_tile_cache_entries)
    expected_size = (grid["patch_px"], grid["patch_px"])
    counts = {"total": 0, "resumed": 0, "written": 0,
              "replaced": 0}

    def values():
        n_x, n_y = grid["shape_xy"]
        min_x, min_y = grid["min_pixel_xy"]
        stride = grid["stride_px"]
        for row_index in range(n_y):
            center_y = min_y + row_index * stride
            for column_index in range(n_x):
                center_x = min_x + column_index * stride
                yield row_index, column_index, center_x, center_y

    def ensure_one(value: tuple[int, int, float, float]) -> str:
        _, _, center_x, center_y = value
        filename = patch_filename(center_x, center_y, grid["zoom"])
        path = patch_dir / filename
        existed = path.exists()
        info = validate_image_file(path, expected_size)
        prior = prior_hashes.get(filename)
        if info is not None and (not prior_hashes or prior == (
                info.sha256, info.decoded_pixel_sha256)):
            return "resumed"
        patch = assemble_patch(center_x, center_y, grid, cache)
        buffer = io.BytesIO()
        patch.save(buffer, "JPEG", quality=jpeg_quality)
        value_bytes = buffer.getvalue()
        _validate_image_bytes(value_bytes, expected_size, filename)
        artifact.atomic_write_file(path, value_bytes)
        if validate_image_file(path, expected_size) is None:
            raise SatelliteError(f"atomic patch write did not validate: {path}")
        return "replaced" if existed else "written"

    for status in _bounded_map(ensure_one, values(), workers=workers):
        counts["total"] += 1
        counts[status] += 1
        if progress_every and counts["total"] % progress_every == 0:
            print(
                f"  patches {counts['total']}/{grid['n_patches']} "
                f"(cached={counts['resumed']}, new={counts['written']}, "
                f"repaired={counts['replaced']})")
    if counts["total"] != grid["n_patches"]:
        raise SatelliteError(
            f"processed {counts['total']} patches, expected "
            f"{grid['n_patches']}")
    return {
        **counts,
        "patch_config_digest": patch_config_digest,
        "patch_root": patch_root,
        "patch_dir": patch_dir,
    }


def _iter_patch_records(grid: dict, patch_dir: Path):
    n_x, n_y = grid["shape_xy"]
    min_x, min_y = grid["min_pixel_xy"]
    stride = grid["stride_px"]
    source_half = grid["source_px"] / 2.0
    expected_size = (grid["patch_px"], grid["patch_px"])
    index = 0
    for row_index in range(n_y):
        center_y = min_y + row_index * stride
        for column_index in range(n_x):
            center_x = min_x + column_index * stride
            filename = patch_filename(center_x, center_y, grid["zoom"])
            path = patch_dir / filename
            info = validate_image_file(path, expected_size)
            if info is None:
                raise SatelliteError(
                    f"required patch is missing/corrupt: {path}")
            center_lat, center_lon = region.pixel_to_lat_lon(
                center_x, center_y, grid["zoom"])
            north_lat, west_lon = region.pixel_to_lat_lon(
                center_x - source_half, center_y - source_half,
                grid["zoom"])
            south_lat, east_lon = region.pixel_to_lat_lon(
                center_x + source_half, center_y + source_half,
                grid["zoom"])
            width_m, height_m = region.metric_dimensions(
                (west_lon, south_lat, east_lon, north_lat))
            yield {
                "index": index,
                "row": row_index,
                "column": column_index,
                "filename": filename,
                "path": f"{SATELLITE_DIR}/{filename}",
                "center_pixel_xy": [center_x, center_y],
                "center_lat": center_lat,
                "center_lon": center_lon,
                "north_lat": north_lat,
                "south_lat": south_lat,
                "east_lon": east_lon,
                "west_lon": west_lon,
                "width_m": width_m,
                "height_m": height_m,
                **asdict(info),
            }
            index += 1


def write_patch_manifest(build_dir: Path, grid: dict,
                         tile_manifest_sha256: str, *,
                         jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> dict:
    """Strictly validate patches and write JSON + LOCI-compatible CSV."""
    patch_config_digest = _patch_config_digest(
        grid, tile_manifest_sha256, jpeg_quality)
    patch_root = Path(build_dir) / "patch_sets" / patch_config_digest
    patch_dir = patch_root / SATELLITE_DIR
    manifest_path = patch_root / PATCH_MANIFEST
    csv_path = patch_root / TILE_METADATA

    def records_and_csv():
        with _atomic_text_writer(csv_path) as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "index", "filename", "center_lat", "center_lon",
                "north_lat", "south_lat", "east_lon", "west_lon",
                "width_m", "height_m", "row", "column",
                "jpeg_sha256", "decoded_pixel_sha256", "size_bytes",
            ])
            for record in _iter_patch_records(grid, patch_dir):
                writer.writerow([
                    record["index"], record["filename"],
                    f"{record['center_lat']:.12f}",
                    f"{record['center_lon']:.12f}",
                    f"{record['north_lat']:.12f}",
                    f"{record['south_lat']:.12f}",
                    f"{record['east_lon']:.12f}",
                    f"{record['west_lon']:.12f}",
                    f"{record['width_m']:.6f}",
                    f"{record['height_m']:.6f}",
                    record["row"], record["column"], record["sha256"],
                    record["decoded_pixel_sha256"], record["size_bytes"],
                ])
                yield record

    count = _write_streamed_manifest(
        manifest_path,
        {
            "schema": PATCH_MANIFEST_SCHEMA,
            "assembly_version": ASSEMBLY_VERSION,
            "grid": grid,
            "source_tile_manifest_sha256": tile_manifest_sha256,
            "jpeg_quality": jpeg_quality,
            "image_format": "jpg",
            "n_patches": grid["n_patches"],
        },
        "patches", records_and_csv())
    if count != grid["n_patches"]:
        raise SatelliteError(
            f"patch manifest contains {count} entries, expected "
            f"{grid['n_patches']}")
    return {
        "patch_root": patch_root,
        "patch_dir": patch_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": artifact.sha256_file(manifest_path),
        "metadata_path": csv_path,
        "metadata_sha256": artifact.sha256_file(csv_path),
        "n_patches": count,
        "patch_config_digest": patch_config_digest,
    }


def _sample_axis(length: int, requested: int) -> list[int]:
    count = min(length, requested)
    if count == 1:
        return [0]
    return [round(index * (length - 1) / (count - 1))
            for index in range(count)]


def write_coverage_contact_sheet(patch_manifest: dict, grid: dict, *,
                                 rows: int = 5, columns: int = 5,
                                 cell_px: int = 180) -> dict:
    patch_dir = Path(patch_manifest["patch_dir"])
    patch_root = Path(patch_manifest["patch_root"])
    row_indices = _sample_axis(grid["shape_xy"][1], rows)
    column_indices = _sample_axis(grid["shape_xy"][0], columns)
    header_px = 32
    sheet = Image.new(
        "RGB", (len(column_indices) * cell_px,
                len(row_indices) * cell_px + header_px), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (8, 9),
        f"LOCI coverage: {grid['shape_xy'][0]}x{grid['shape_xy'][1]} "
        f"patches, z{grid['zoom']}", fill="black")
    samples = []
    min_x, min_y = grid["min_pixel_xy"]
    stride = grid["stride_px"]
    for contact_row, row_index in enumerate(row_indices):
        center_y = min_y + row_index * stride
        for contact_column, column_index in enumerate(column_indices):
            center_x = min_x + column_index * stride
            filename = patch_filename(center_x, center_y, grid["zoom"])
            path = patch_dir / filename
            info = validate_image_file(
                path, (grid["patch_px"], grid["patch_px"]))
            if info is None:
                raise SatelliteError(
                    f"contact-sheet sample is missing/corrupt: {path}")
            with Image.open(path) as source:
                tile = source.convert("RGB").resize(
                    (cell_px, cell_px), Image.Resampling.LANCZOS)
            left = contact_column * cell_px
            top = header_px + contact_row * cell_px
            sheet.paste(tile, (left, top))
            label = f"r{row_index} c{column_index}"
            draw.rectangle(
                (left, top, left + 86, top + 15), fill=(0, 0, 0))
            draw.text((left + 3, top + 2), label, fill=(255, 255, 255))
            latitude, longitude = region.pixel_to_lat_lon(
                center_x, center_y, grid["zoom"])
            samples.append({
                "contact_row": contact_row,
                "contact_column": contact_column,
                "grid_row": row_index,
                "grid_column": column_index,
                "filename": filename,
                "center_lat": latitude,
                "center_lon": longitude,
                "jpeg_sha256": info.sha256,
            })
    image_buffer = io.BytesIO()
    sheet.save(image_buffer, "JPEG", quality=92)
    image_path = patch_root / COVERAGE_CONTACT_SHEET
    index_path = patch_root / COVERAGE_CONTACT_INDEX
    artifact.atomic_write_file(image_path, image_buffer.getvalue())
    artifact.atomic_write_json(index_path, {
        "schema": "loci_coverage_contact_sheet/v1",
        "layout": {
            "rows": len(row_indices), "columns": len(column_indices),
            "cell_px": cell_px, "header_px": header_px},
        "samples": samples,
    })
    return {
        "image_path": image_path,
        "image_sha256": artifact.sha256_file(image_path),
        "index_path": index_path,
        "index_sha256": artifact.sha256_file(index_path),
        "n_samples": len(samples),
    }


def _prepare_build_directory(build_dir: Path, state: dict) -> None:
    build_dir = Path(build_dir)
    if build_dir.is_symlink():
        raise SatelliteError(f"build directory cannot be a symlink: {build_dir}")
    build_dir.mkdir(parents=True, exist_ok=True)
    state_path = build_dir / "build_state.json"
    if state_path.exists():
        if state_path.is_symlink() or not state_path.is_file():
            raise SatelliteError(f"invalid build state path: {state_path}")
        try:
            existing = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SatelliteError(f"invalid build state {state_path}: {error}") \
                from error
        # One source cache may safely feed more than one patch assembler.  The
        # patch-set digest contains ASSEMBLY_VERSION, so differing assemblers
        # cannot reuse each other's JPEGs.  All source/coverage identity fields
        # remain strict.
        existing_source_state = dict(existing)
        requested_source_state = dict(state)
        existing_source_state.pop("assembly_version", None)
        requested_source_state.pop("assembly_version", None)
        # A republished immutable region can carry a different manifest
        # (for example, corrected code provenance) while its content and grid
        # remain byte-identical.  The content digest below is the cache input.
        existing_source_state.pop("region_manifest_digest", None)
        requested_source_state.pop("region_manifest_digest", None)
        if existing_source_state != requested_source_state:
            raise SatelliteError(
                f"existing satellite build belongs to a different recipe: "
                f"{build_dir}")
        return
    unexpected = [path for path in build_dir.iterdir()]
    if unexpected:
        raise SatelliteError(
            f"unbound non-empty satellite build directory: {build_dir}")
    artifact.atomic_create_json(state_path, state)


def _copy_regular(source: Path, destination: Path) -> None:
    """Copy build output into the immutable staging tree.

    Build directories are intentionally mutable and resumable.  Hard-linking
    their files into an artifact would let an in-place cache edit mutate a
    supposedly immutable published payload through the shared inode.
    """
    if source.is_symlink() or not source.is_file():
        raise SatelliteError(f"publication source is not regular: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination, follow_symlinks=False)


def _existing_artifact(destination: Path, region_ref: artifact.ArtifactRef,
                       *, service_url: str,
                       source_index_url: str | None,
                       require_source_index_coverage: bool,
                       jpeg_quality: int,
                       provider_mode: str = DEFAULT_PROVIDER_MODE,
                       catalog_where: str | None = None,
                       lock_raster_ids: Iterable[int] = (),
                       esri_wayback_release: str | None = None,
                       image_server_chunk_tiles: int = 1) \
        -> artifact.ArtifactRef | None:
    if not destination.exists() and not destination.is_symlink():
        return None
    reference = artifact.open_artifact(
        destination, expected_kind=ARTIFACT_KIND,
        expected_dataset=region_ref.dataset,
        expected_version=destination.name)
    manifest = artifact.load_manifest(destination)
    request = _provider_request_contract(
        provider_mode, service_url, source_index_url=source_index_url,
        catalog_where=catalog_where, lock_raster_ids=lock_raster_ids,
        esri_wayback_release=esri_wayback_release,
        require_source_index_coverage=require_source_index_coverage,
        image_server_chunk_tiles=image_server_chunk_tiles)
    provider = manifest.config.get("provider", {})
    expected = {
        "region_manifest_digest": region_ref.manifest_digest,
        "service_url": service_url.rstrip("/"),
        "source_index_url": source_index_url,
        "jpeg_quality": jpeg_quality,
        "assembly_version": ASSEMBLY_VERSION,
        "provider_type": (
            "arcgis_cached_map_service"
            if provider_mode == CACHED_MAP_PROVIDER
            else IMAGE_SERVER_PROVIDER),
        "esri_wayback_release": request.get("esri_wayback_release"),
        "source_index_coverage_required": require_source_index_coverage,
    }
    actual = {
        "region_manifest_digest": manifest.config.get(
            "region_manifest_digest"),
        "service_url": provider.get("service_url"),
        "source_index_url": manifest.config.get("source_index_url"),
        "jpeg_quality": manifest.config.get("jpeg_quality"),
        "assembly_version": manifest.config.get("assembly_version"),
        "provider_type": provider.get("type"),
        "esri_wayback_release": provider.get("esri_wayback_release"),
        "source_index_coverage_required": manifest.config.get(
            "source_index_coverage_required", True),
    }
    if provider_mode == IMAGE_SERVER_PROVIDER:
        expected["provider_request"] = request
        actual["provider_request"] = manifest.config.get("provider_request")
    if actual != expected:
        raise SatelliteError(
            f"existing satellite artifact differs from request: {destination}")
    return reference


def materialize(*, farfield_root: Path, dataset: str, region_dir: Path,
                version: str,
                build_cache_version: str | None = None,
                service_url: str = DEFAULT_SERVICE_URL,
                source_index_url: str | None = DEFAULT_SOURCE_INDEX_URL,
                require_source_index_coverage: bool = True,
                provider_mode: str = DEFAULT_PROVIDER_MODE,
                catalog_where: str | None = None,
                lock_raster_ids: Iterable[int] = (),
                esri_wayback_release: str | None = None,
                image_server_chunk_tiles: int = 1,
                jpeg_quality: int = DEFAULT_JPEG_QUALITY,
                workers: int = 32, patch_workers: int = 8,
                decoded_tile_cache_entries: int = 1024,
                client=None) -> artifact.ArtifactRef:
    farfield_root = Path(farfield_root).resolve()
    dataset = artifact.require_identifier(dataset, "artifact dataset")
    version = artifact.require_identifier(version, "artifact version")
    build_cache_version = artifact.require_identifier(
        build_cache_version or version, "satellite build cache version")
    region_ref, plan = region.load_region(Path(region_dir))
    if region_ref.dataset != dataset:
        raise SatelliteError(
            f"region dataset {region_ref.dataset!r} does not match "
            f"satellite dataset {dataset!r}")
    # Preserve the historical MassGIS source-index default for cached map
    # services while making ImageServer mode self-contained.  ImageServer
    # coverage comes from its own pinned raster catalog, not this unrelated
    # FeatureServer default.
    if (provider_mode == IMAGE_SERVER_PROVIDER
            and source_index_url == DEFAULT_SOURCE_INDEX_URL):
        source_index_url = None
    if (esri_wayback_release is not None
            and source_index_url == DEFAULT_SOURCE_INDEX_URL):
        source_index_url = None
    request = _provider_request_contract(
        provider_mode, service_url, source_index_url=source_index_url,
        catalog_where=catalog_where, lock_raster_ids=lock_raster_ids,
        esri_wayback_release=esri_wayback_release,
        require_source_index_coverage=require_source_index_coverage,
        image_server_chunk_tiles=image_server_chunk_tiles)
    destination = (
        farfield_root / "artifacts" / ARTIFACT_KIND / dataset / version)
    existing = _existing_artifact(
        destination, region_ref, service_url=service_url,
        source_index_url=source_index_url,
        require_source_index_coverage=require_source_index_coverage,
        jpeg_quality=jpeg_quality, provider_mode=provider_mode,
        catalog_where=request.get("catalog_where"),
        lock_raster_ids=request.get("lock_raster_ids", ()),
        esri_wayback_release=request.get("esri_wayback_release"),
        image_server_chunk_tiles=image_server_chunk_tiles)
    if existing is not None:
        return existing

    if client is None:
        if provider_mode == CACHED_MAP_PROVIDER:
            client = ArcGisTileClient(
                service_url,
                esri_wayback_release=request.get("esri_wayback_release"))
        else:
            client = ArcGisImageServerClient(
                service_url, catalog_where=request["catalog_where"],
                lock_raster_ids=request["lock_raster_ids"])
    service_metadata = client.get_service_metadata()
    coverage = audit_coverage(
        client, plan, service_metadata, service_url=service_url,
        source_index_url=source_index_url,
        require_source_index_coverage=require_source_index_coverage,
        provider_mode=provider_mode,
        catalog_where=request.get("catalog_where"),
        lock_raster_ids=request.get("lock_raster_ids", ()),
        esri_wayback_release=request.get("esri_wayback_release"),
        image_server_chunk_tiles=image_server_chunk_tiles)
    coverage_sha256 = artifact.sha256_json(coverage)
    provider = coverage["provider"]
    grid = plan["grid"]
    build_dir = (farfield_root / "builds" / dataset
                 / f"loci_satellite_{build_cache_version}")
    build_state = {
        "schema": BUILD_SCHEMA,
        "dataset": dataset,
        "version": build_cache_version,
        "region_manifest_digest": region_ref.manifest_digest,
        "region_content_digest": region_ref.content_digest,
        "grid": grid,
        "provider": provider,
        "source_index_url": source_index_url,
        "coverage_audit_sha256": coverage_sha256,
        "jpeg_quality": jpeg_quality,
        "assembly_version": ASSEMBLY_VERSION,
    }
    if provider_mode == IMAGE_SERVER_PROVIDER:
        build_state["provider_request"] = request
    if not require_source_index_coverage:
        build_state["source_index_coverage_required"] = False
    _prepare_build_directory(build_dir, build_state)
    coverage_path = build_dir / COVERAGE_AUDIT
    artifact.atomic_write_json(coverage_path, coverage)

    source_summary = ensure_source_tiles(
        build_dir, grid, client, workers=workers,
        image_server_chunk_tiles=image_server_chunk_tiles)
    source_manifest = write_source_tile_manifest(build_dir, grid, provider)
    patch_summary = ensure_patches(
        build_dir, grid, source_manifest["sha256"],
        jpeg_quality=jpeg_quality, workers=patch_workers,
        decoded_tile_cache_entries=decoded_tile_cache_entries)
    patch_manifest = write_patch_manifest(
        build_dir, grid, source_manifest["sha256"],
        jpeg_quality=jpeg_quality)
    contact_sheet = write_coverage_contact_sheet(patch_manifest, grid)

    bbox_path = patch_manifest["patch_root"] / SATELLITE_BBOX
    artifact.atomic_write_json(bbox_path, {
        "schema": "loci_satellite_bbox/v1",
        "bbox_wsen": plan["bbox_wsen"],
        "center_bbox_wsen": grid["center_bbox_wsen"],
        "footprint_bbox_wsen": grid["footprint_bbox_wsen"],
        "actual_area_km2": plan["actual_area_km2"],
        "grid": grid,
        "provider": provider,
    })
    summary = {
        "schema": SCHEMA,
        "dataset": dataset,
        "version": version,
        "region": region_ref.to_dict(),
        "provider": provider,
        "coverage_audit_sha256": artifact.sha256_file(coverage_path),
        "source_tile_manifest_sha256": source_manifest["sha256"],
        "patch_manifest_sha256": patch_manifest["manifest_sha256"],
        "tile_metadata_sha256": patch_manifest["metadata_sha256"],
        "contact_sheet_sha256": contact_sheet["image_sha256"],
        # Resume-vs-new counts describe one invocation, not the immutable
        # imagery.  Record only completion facts so an interrupted/resumed
        # build publishes the same summary as a one-shot build.
        "source_tile_counts": {
            "expected": grid["n_source_tiles"],
            "validated": source_summary["total"],
        },
        "patch_counts": {
            "expected": grid["n_patches"],
            "validated": patch_summary["total"],
        },
        "n_source_tiles": grid["n_source_tiles"],
        "n_patches": grid["n_patches"],
    }
    summary_path = patch_manifest["patch_root"] / SUMMARY_OUTPUT
    artifact.atomic_write_json(summary_path, summary)

    config = {
        "schema": SCHEMA,
        "region_manifest_digest": region_ref.manifest_digest,
        "provider": provider,
        "source_index_url": source_index_url,
        "grid": grid,
        "jpeg_quality": jpeg_quality,
        "assembly_version": ASSEMBLY_VERSION,
    }
    if provider_mode == IMAGE_SERVER_PROVIDER:
        config["provider_request"] = request
    if not require_source_index_coverage:
        config["source_index_coverage_required"] = False
    build_inputs = {
        "coverage_audit_sha256": artifact.sha256_file(coverage_path),
        "source_tile_manifest_sha256": source_manifest["sha256"],
        "patch_manifest_sha256": patch_manifest["manifest_sha256"],
        "tile_metadata_sha256": patch_manifest["metadata_sha256"],
    }
    stage_config_digest = artifact.sha256_json(config)
    identity = artifact_identity.compute(
        kind=ARTIFACT_KIND, dataset=dataset,
        stage_config_digest=stage_config_digest,
        upstreams=(region_ref,), build_inputs=build_inputs)
    recipe = artifact_recipe.build(
        stage="loci_satellite", stage_config=config,
        build_inputs=build_inputs, identity_upstreams=(region_ref,))

    metadata_sources = {
        SOURCE_MANIFEST: source_manifest["path"],
        PATCH_MANIFEST: patch_manifest["manifest_path"],
        TILE_METADATA: patch_manifest["metadata_path"],
        COVERAGE_AUDIT: coverage_path,
        COVERAGE_CONTACT_SHEET: contact_sheet["image_path"],
        COVERAGE_CONTACT_INDEX: contact_sheet["index_path"],
        SATELLITE_BBOX: bbox_path,
        SUMMARY_OUTPUT: summary_path,
    }
    patch_files = []
    for center_x, center_y in region.iter_grid_centres(grid):
        patch_files.append(
            patch_filename(center_x, center_y, grid["zoom"]))
    if len(patch_files) != len(set(patch_files)):
        raise SatelliteError("formatted VIGOR patch filenames are not unique")
    declared_outputs = [
        *metadata_sources,
        *(f"{SATELLITE_DIR}/{name}" for name in patch_files),
    ]
    with publication.published_artifact(
            destination, kind=ARTIFACT_KIND, dataset=dataset,
            version=version, generator=GENERATOR,
            git_commit=provenance.git_commit(), upstreams=(region_ref,),
            config=config, artifact_identity=identity, recipe=recipe,
            declared_outputs=declared_outputs) as builder:
        for output_name, source_path in metadata_sources.items():
            _copy_regular(source_path, builder.staging_dir / output_name)
        published_satellite = builder.staging_dir / SATELLITE_DIR
        published_satellite.mkdir(parents=True, exist_ok=True)
        for filename in patch_files:
            _copy_regular(
                patch_manifest["patch_dir"] / filename,
                published_satellite / filename)
    return artifact.open_artifact(
        destination, expected_kind=ARTIFACT_KIND,
        expected_dataset=dataset, expected_version=version)


def audit_region(region_dir: Path, *,
                 service_url: str = DEFAULT_SERVICE_URL,
                 source_index_url: str | None = DEFAULT_SOURCE_INDEX_URL,
                 require_source_index_coverage: bool = True,
                 provider_mode: str = DEFAULT_PROVIDER_MODE,
                 catalog_where: str | None = None,
                 lock_raster_ids: Iterable[int] = (),
                 esri_wayback_release: str | None = None,
                 image_server_chunk_tiles: int = 1,
                 client=None) -> dict:
    """Run the provider coverage proof without creating build state."""
    _, plan = region.load_region(Path(region_dir))
    if (provider_mode == IMAGE_SERVER_PROVIDER
            and source_index_url == DEFAULT_SOURCE_INDEX_URL):
        source_index_url = None
    if (esri_wayback_release is not None
            and source_index_url == DEFAULT_SOURCE_INDEX_URL):
        source_index_url = None
    request = _provider_request_contract(
        provider_mode, service_url, source_index_url=source_index_url,
        catalog_where=catalog_where, lock_raster_ids=lock_raster_ids,
        esri_wayback_release=esri_wayback_release,
        require_source_index_coverage=require_source_index_coverage,
        image_server_chunk_tiles=image_server_chunk_tiles)
    if client is None:
        if provider_mode == CACHED_MAP_PROVIDER:
            client = ArcGisTileClient(
                service_url,
                esri_wayback_release=request.get("esri_wayback_release"))
        else:
            client = ArcGisImageServerClient(
                service_url, catalog_where=request["catalog_where"],
                lock_raster_ids=request["lock_raster_ids"])
    return audit_coverage(
        client, plan, client.get_service_metadata(),
        service_url=service_url, source_index_url=source_index_url,
        require_source_index_coverage=require_source_index_coverage,
        provider_mode=provider_mode,
        catalog_where=request.get("catalog_where"),
        lock_raster_ids=request.get("lock_raster_ids", ()),
        esri_wayback_release=request.get("esri_wayback_release"),
        image_server_chunk_tiles=image_server_chunk_tiles)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farfield_root", type=Path,
                        default=paths_lib.DEFAULT_ROOT)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--region_dir", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--provider_mode", choices=PROVIDER_MODES,
        default=DEFAULT_PROVIDER_MODE,
        help="cached MapServer tiles (default) or pinned ImageServer exports")
    parser.add_argument(
        "--build_cache_version",
        help="reuse the mutable cache belonging to another artifact version; "
             "source identity remains strict and patch sets are assembler-keyed")
    parser.add_argument("--service_url", default=DEFAULT_SERVICE_URL)
    parser.add_argument(
        "--esri_wayback_release",
        help="pinned numeric ESRI Wayback release; requires the standard "
             "World Imagery MapServer as --service_url")
    parser.add_argument(
        "--source_index_url",
        help="ArcGIS source-orthophoto index layer for cached-map mode "
             f"(default: {DEFAULT_SOURCE_INDEX_URL}); pass an empty string "
             "to disable it")
    parser.add_argument(
        "--catalog_where",
        help="required ImageServer source-catalog where clause")
    parser.add_argument(
        "--lock_raster_id", action="append", type=int, default=[],
        help="ImageServer primary raster OBJECTID to pin; repeat for every "
             "catalog feature covering the region")
    parser.add_argument(
        "--image_server_chunk_tiles", type=int, default=1,
        help="fixed ImageServer export chunk width/height in source tiles; "
             "1 preserves legacy single-tile exports, 15 uses 3840px "
             "exports (maximum 15)")
    parser.add_argument(
        "--allow_incomplete_source_index", action="store_true",
        help="record source-index gaps as informational while still requiring "
             "every rendered Tilemap entry; intended for water outside a "
             "land-flight index")
    parser.add_argument("--jpeg_quality", type=int,
                        default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--patch_workers", type=int, default=8)
    parser.add_argument("--decoded_tile_cache_entries", type=int,
                        default=1024)
    parser.add_argument("--connect_timeout_s", type=float, default=10.0)
    parser.add_argument("--read_timeout_s", type=float, default=60.0)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument(
        "--audit_only", action="store_true",
        help="print the strict Tilemap/source-index audit without downloading")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_index_url is None:
        source_index_url = (
            DEFAULT_SOURCE_INDEX_URL
            if (args.provider_mode == CACHED_MAP_PROVIDER
                and args.esri_wayback_release is None) else None)
    else:
        source_index_url = args.source_index_url or None
    request = _provider_request_contract(
        args.provider_mode, args.service_url,
        source_index_url=source_index_url,
        catalog_where=args.catalog_where,
        lock_raster_ids=args.lock_raster_id,
        esri_wayback_release=args.esri_wayback_release,
        require_source_index_coverage=(
            not args.allow_incomplete_source_index),
        image_server_chunk_tiles=args.image_server_chunk_tiles)
    client_kwargs = {
        "connect_timeout_s": args.connect_timeout_s,
        "read_timeout_s": args.read_timeout_s,
        "max_retries": args.max_retries,
    }
    if args.provider_mode == CACHED_MAP_PROVIDER:
        client = ArcGisTileClient(
            args.service_url,
            esri_wayback_release=request.get("esri_wayback_release"),
            **client_kwargs)
    else:
        client = ArcGisImageServerClient(
            args.service_url, catalog_where=request["catalog_where"],
            lock_raster_ids=request["lock_raster_ids"], **client_kwargs)
    if args.audit_only:
        coverage = audit_region(
            args.region_dir, service_url=args.service_url,
            source_index_url=source_index_url,
            require_source_index_coverage=(
                not args.allow_incomplete_source_index),
            provider_mode=args.provider_mode,
            catalog_where=request.get("catalog_where"),
            lock_raster_ids=request.get("lock_raster_ids", ()),
            esri_wayback_release=request.get("esri_wayback_release"),
            image_server_chunk_tiles=args.image_server_chunk_tiles,
            client=client)
        print(json.dumps(coverage, sort_keys=True, indent=2))
        return
    reference = materialize(
        farfield_root=args.farfield_root, dataset=args.dataset,
        region_dir=args.region_dir, version=args.version,
        build_cache_version=args.build_cache_version,
        service_url=args.service_url,
        source_index_url=source_index_url,
        require_source_index_coverage=(
            not args.allow_incomplete_source_index),
        provider_mode=args.provider_mode,
        catalog_where=request.get("catalog_where"),
        lock_raster_ids=request.get("lock_raster_ids", ()),
        esri_wayback_release=request.get("esri_wayback_release"),
        image_server_chunk_tiles=args.image_server_chunk_tiles,
        jpeg_quality=args.jpeg_quality, workers=args.workers,
        patch_workers=args.patch_workers,
        decoded_tile_cache_entries=args.decoded_tile_cache_entries,
        client=client)
    print(reference.path)


if __name__ == "__main__":
    main()
