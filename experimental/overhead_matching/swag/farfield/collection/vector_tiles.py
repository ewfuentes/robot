"""Mapillary coverage via vector tiles, for area-scale discovery.

Why this exists: `/images` cannot answer "where is there coverage like X". It
rejects any bbox over 0.010 square degrees, and separately rejects dense areas
on result volume, both as HTTP 500 (see `api.MapillaryQueryTooLarge`). Sweeping
a region with it subdivides exponentially -- the collection README records depth
10 in SF, Seattle and London, never finishing. It is the right endpoint for the
neighbourhood of a point you already have, which is exactly how stitching uses
it, and the wrong one for search.

The vector tiles answer the opposite question. One HTTP request returns every
sequence intersecting a whole tile, already as a line geometry, with no area cap
and no result cap. At z14 a tile is ~2.4 km across at the equator; at z10 it is
~39 km. A region the size of Lake Geneva is 9 tiles at z10 against roughly 1,200
at z14 and several thousand `/images` queries.

    tileset mly1_public/2/{z}/{x}/{y}
      z0-5    overview   point,      one per sequence
      z6-14   sequence   linestring, whole sequence geometry
      z14     image      point,      individual images

`probe_layers()` reports what a given tileset/zoom actually serves rather than
trusting that list.

What tiles do NOT give you, and why the two endpoints are complementary:

  * **Properties are a subset.** The sequence layer carries `id`, `captured_at`,
    `is_pano`, `creator_id` / `creator_username` and `organization_id` -- but no
    camera model, no per-image compass angle, no `camera_parameters`. Anything
    that decides *how to convert* a capture still needs `/images` or
    `get_full_sequence`.
  * **Geometry is simplified per zoom.** Vertices are quantised to the tile's
    4096-unit grid, so at z10 a position is good to ~10 m and at z14 to ~0.6 m.
    Track *length* from tiles is an estimate; treat it as a screening number and
    re-measure from the manifest.
  * **A sequence is clipped at tile edges**, so a long track appears as one
    feature per tile it crosses, all sharing the same sequence id. `merge_tile
    _features` reassembles them; without it every ferry looks like a 2 km hop.

So: tiles to *find* candidates, Graph API to *qualify* them. `discover_tracks.py`
is that pipeline.

No dependency beyond `requests` -- the Mapbox Vector Tile format is protobuf,
and the ~4 message types it uses are decoded inline below rather than pulling in
`mapbox-vector-tile` and a protobuf runtime.
"""

import gzip
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from experimental.overhead_matching.swag.farfield.collection.api import RateLimiter

TILE_URL = "https://tiles.mapillary.com/maps/vtp/{tileset}/2/{z}/{x}/{y}"

# `mly1_public` is the coverage tileset -- where imagery exists. Mapillary also
# serves detected objects as `mly_map_feature_point` and
# `mly_map_feature_traffic_sign`, which `fetch` accepts via its `tileset`
# argument; they are named here so a future caller does not have to rediscover
# them, and are not used by the coverage search.
TILESET_COVERAGE = "mly1_public"

GEOM_POINT, GEOM_LINESTRING, GEOM_POLYGON = 1, 2, 3

# Under bazel the module lives in an ephemeral runfiles tree, so the cache
# must anchor to the user, not to __file__. Cached blobs are public map tiles,
# not secrets; ~/.cache is the right lifetime (survives builds, safe to rm).
DEFAULT_CACHE = Path(
    os.environ.get("MLY_TILE_CACHE",
                   Path.home() / ".cache" / "mapillary_tiles"))


# --------------------------------------------------------------------------
# protobuf wire format
# --------------------------------------------------------------------------

def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _zigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def _iter_fields(buf: bytes, start: int, end: int):
    """Yield (field_number, wire_type, payload) over one protobuf message.

    For wire type 2 the payload is a (start, end) slice rather than the bytes,
    so nested messages are walked without copying. A tile is a few MB and
    layers nest three deep, so copying at every level is the difference
    between milliseconds and seconds on a continental scan.
    """
    pos = start
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field_no, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _read_varint(buf, pos)
            yield field_no, wire, value
        elif wire == 1:
            yield field_no, wire, buf[pos:pos + 8]
            pos += 8
        elif wire == 2:
            length, pos = _read_varint(buf, pos)
            yield field_no, wire, (pos, pos + length)
            pos += length
        elif wire == 5:
            yield field_no, wire, buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")


def _read_packed_varints(buf: bytes, start: int, end: int) -> list[int]:
    out = []
    pos = start
    while pos < end:
        value, pos = _read_varint(buf, pos)
        out.append(value)
    return out


def _decode_value(buf: bytes, start: int, end: int):
    """One MVT Value message -> a Python scalar.

    Exactly one field is set. Ordering matters only in that `bool_value` (7)
    must not be read as an int, which is why this dispatches on field number
    rather than taking whatever arrives last.
    """
    import struct
    for field_no, _wire, payload in _iter_fields(buf, start, end):
        if field_no == 1:
            s, e = payload
            return buf[s:e].decode("utf-8", errors="replace")
        if field_no == 2:
            return struct.unpack("<f", payload)[0]
        if field_no == 3:
            return struct.unpack("<d", payload)[0]
        if field_no in (4, 5):
            return payload
        if field_no == 6:
            return _zigzag(payload)
        if field_no == 7:
            return bool(payload)
    return None


# --------------------------------------------------------------------------
# tile geometry
# --------------------------------------------------------------------------

def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(max(-85.05112878, min(85.05112878, lat)))
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a tile, in degrees."""
    n = 2 ** z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


def tiles_for_bbox(bbox, z: int) -> list[tuple[int, int]]:
    """Every (x, y) at zoom z covering a (west, south, east, north) bbox."""
    west, south, east, north = bbox
    x0, y0 = lonlat_to_tile(west, north, z)
    x1, y1 = lonlat_to_tile(east, south, z)
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def _tile_pixel_to_lonlat(z, x, y, px, py, extent):
    n = 2 ** z
    lon = (x + px / extent) / n * 360.0 - 180.0
    ty = (y + py / extent) / n
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty))))
    return lon, lat


def _decode_geometry(commands: list[int], geom_type: int) -> list[list[tuple[int, int]]]:
    """MVT command stream -> rings of tile-local integer coordinates.

    A command integer packs an id in its low 3 bits and a repeat count in the
    rest; parameters that follow are zigzag deltas from a running cursor. The
    cursor persists across commands, which is what makes a MoveTo mid-stream
    start a new ring rather than reset to the origin.
    """
    rings, current = [], []
    cursor_x = cursor_y = 0
    i = 0
    while i < len(commands):
        header = commands[i]
        i += 1
        command, count = header & 0x7, header >> 3
        if command == 1:  # MoveTo
            for _ in range(count):
                cursor_x += _zigzag(commands[i])
                cursor_y += _zigzag(commands[i + 1])
                i += 2
                if current:
                    rings.append(current)
                current = [(cursor_x, cursor_y)]
        elif command == 2:  # LineTo
            for _ in range(count):
                cursor_x += _zigzag(commands[i])
                cursor_y += _zigzag(commands[i + 1])
                i += 2
                current.append((cursor_x, cursor_y))
        elif command == 7:  # ClosePath
            if current:
                current.append(current[0])
        else:
            raise ValueError(f"unknown MVT command {command}")
    if current:
        rings.append(current)
    return rings


# --------------------------------------------------------------------------
# decoded features
# --------------------------------------------------------------------------

@dataclass
class TileFeature:
    layer: str
    geom_type: int
    properties: dict
    coords: list[tuple[float, float]]  # (lon, lat), flattened across rings
    tile: tuple[int, int, int] = (0, 0, 0)

    @property
    def sequence_id(self) -> str | None:
        # The coverage tiles name this `sequence_id` on the image layer and
        # `id` on the sequence layer; callers should not have to care which.
        return self.properties.get("sequence_id") or self.properties.get("id")


def decode_tile(data: bytes, tile: tuple[int, int, int] = (0, 0, 0)) -> dict[str, list[TileFeature]]:
    """Decode one MVT blob into {layer_name: [TileFeature, ...]}.

    Coordinates come out as (lon, lat) degrees. Mapillary serves these gzipped
    without always setting Content-Encoding, so sniff the magic rather than
    trusting the header.
    """
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)

    z, x, y = tile
    layers: dict[str, list[TileFeature]] = {}

    for field_no, _wire, payload in _iter_fields(data, 0, len(data)):
        if field_no != 3:  # Tile.layers
            continue
        layer_start, layer_end = payload
        name, extent = None, 4096
        keys: list[str] = []
        values: list = []
        feature_spans: list[tuple[int, int]] = []

        for lf, _lw, lpayload in _iter_fields(data, layer_start, layer_end):
            if lf == 1:
                s, e = lpayload
                name = data[s:e].decode("utf-8")
            elif lf == 2:
                feature_spans.append(lpayload)
            elif lf == 3:
                s, e = lpayload
                keys.append(data[s:e].decode("utf-8"))
            elif lf == 4:
                s, e = lpayload
                values.append(_decode_value(data, s, e))
            elif lf == 5:
                extent = lpayload

        features = []
        for fstart, fend in feature_spans:
            props, geom_type, geometry = {}, GEOM_POINT, []
            for ff, _fw, fpayload in _iter_fields(data, fstart, fend):
                if ff == 1:
                    props["_id"] = fpayload
                elif ff == 2:
                    s, e = fpayload
                    tags = _read_packed_varints(data, s, e)
                    for ki, vi in zip(tags[0::2], tags[1::2]):
                        if ki < len(keys) and vi < len(values):
                            props[keys[ki]] = values[vi]
                elif ff == 3:
                    geom_type = fpayload
                elif ff == 4:
                    s, e = fpayload
                    geometry = _read_packed_varints(data, s, e)

            coords = []
            for ring in _decode_geometry(geometry, geom_type):
                coords.extend(
                    _tile_pixel_to_lonlat(z, x, y, px, py, extent) for px, py in ring
                )
            features.append(TileFeature(name, geom_type, props, coords, tile))

        layers.setdefault(name, []).extend(features)

    return layers


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------

class VectorTileClient:
    """Fetches and decodes coverage tiles, caching raw blobs on disk.

    Caching is not an optimisation here so much as a courtesy: a continental
    z6 overview tile is 16 MB, and a search run re-reads the same tiles every
    time its scoring parameters change.
    """

    def __init__(self, token: str = None, cache_dir: Path = DEFAULT_CACHE,
                 max_per_minute: int = 5000):
        if token is None:
            from experimental.overhead_matching.swag.farfield.collection.api import MapillaryClient
            token = MapillaryClient._read_token()
        self.token = token
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self._limiter = RateLimiter(max_per_minute)
        self._lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "empty": 0, "bytes": 0}

    def _cache_path(self, tileset: str, z: int, x: int, y: int) -> Path:
        return self.cache_dir / tileset / str(z) / str(x) / f"{y}.mvt"

    def fetch_raw(self, z: int, x: int, y: int, tileset: str = TILESET_COVERAGE,
                  max_retries: int = 4) -> bytes:
        path = self._cache_path(tileset, z, x, y)
        if path.exists():
            with self._lock:
                self.stats["hits"] += 1
            return path.read_bytes()

        url = TILE_URL.format(tileset=tileset, z=z, x=x, y=y)
        for attempt in range(max_retries):
            self._limiter.acquire()
            try:
                resp = self.session.get(url, params={"access_token": self.token}, timeout=60)
            except requests.exceptions.RequestException:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 200:
                data = resp.content
                path.parent.mkdir(parents=True, exist_ok=True)
                # Write via a temp name so a killed run cannot leave a
                # truncated tile that later reads treat as a real empty tile.
                tmp = path.with_suffix(".part")
                tmp.write_bytes(data)
                os.replace(tmp, path)
                with self._lock:
                    self.stats["misses"] += 1
                    self.stats["bytes"] += len(data)
                    if len(data) < 64:
                        self.stats["empty"] += 1
                return data
            if resp.status_code == 404:
                # A tile with no coverage at all. Cache the emptiness, or a
                # scan over ocean re-requests every tile on every run.
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")
                with self._lock:
                    self.stats["empty"] += 1
                return b""
            if resp.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"tile {z}/{x}/{y} failed after {max_retries} attempts")

    def fetch(self, z: int, x: int, y: int, tileset: str = TILESET_COVERAGE
              ) -> dict[str, list[TileFeature]]:
        data = self.fetch_raw(z, x, y, tileset)
        if len(data) < 8:
            return {}
        return decode_tile(data, (z, x, y))

    def probe_layers(self, lon: float, lat: float, zooms=range(0, 15),
                     tileset: str = TILESET_COVERAGE) -> dict[int, dict[str, int]]:
        """What layers does this tileset actually serve, per zoom?

        Written because the published layer/zoom table has changed before, and
        a scan that assumes `sequence` exists at a zoom where it does not comes
        back empty rather than failing.
        """
        found = {}
        for z in zooms:
            x, y = lonlat_to_tile(lon, lat, z)
            try:
                layers = self.fetch(z, x, y, tileset)
            except Exception as exc:
                found[z] = {"_error": str(exc)[:80]}
                continue
            found[z] = {name: len(feats) for name, feats in layers.items()}
        return found


def merge_tile_features(features: list[TileFeature]) -> dict[str, dict]:
    """Reassemble per-tile fragments of the same sequence.

    Tiles clip geometry at their edges, so a 30 km ferry crossing z14 arrives
    as a dozen separate features that share a sequence id. Merging by id is
    what makes track length mean anything.

    Fragments are concatenated in nearest-neighbour order from the westernmost
    endpoint, not left in tile-fetch order, which is row-major over the tile
    grid and would zig-zag a diagonal track. Length is computed after that
    ordering; a track whose fragments are genuinely disjoint still inflates,
    so `n_fragments` is returned for callers that want to distrust it.
    """
    by_id: dict[str, list[TileFeature]] = {}
    for feat in features:
        key = feat.sequence_id
        if key is None:
            continue
        by_id.setdefault(str(key), []).append(feat)

    merged = {}
    for seq_id, feats in by_id.items():
        pieces = [f.coords for f in feats if len(f.coords) >= 2]
        if not pieces:
            pieces = [f.coords for f in feats if f.coords]
        if not pieces:
            continue

        remaining = list(pieces)
        start = min(remaining, key=lambda p: p[0][0])
        remaining.remove(start)
        chain = list(start)
        while remaining:
            tail = chain[-1]
            best, best_d, flip = None, float("inf"), False
            for piece in remaining:
                for end, needs_flip in ((piece[0], False), (piece[-1], True)):
                    d = (end[0] - tail[0]) ** 2 + (end[1] - tail[1]) ** 2
                    if d < best_d:
                        best, best_d, flip = piece, d, needs_flip
            remaining.remove(best)
            chain.extend(reversed(best) if flip else best)

        props = dict(feats[0].properties)
        props.pop("_id", None)
        merged[seq_id] = {
            "sequence_id": seq_id,
            "properties": props,
            "coords": chain,
            "n_fragments": len(feats),
            "length_km": polyline_length_km(chain),
        }
    return merged


def polyline_length_km(coords: list[tuple[float, float]]) -> float:
    total = 0.0
    for (lon0, lat0), (lon1, lat1) in zip(coords, coords[1:]):
        mid = math.radians((lat0 + lat1) / 2)
        dx = (lon1 - lon0) * 111.320 * math.cos(mid)
        dy = (lat1 - lat0) * 110.574
        total += math.hypot(dx, dy)
    return total
