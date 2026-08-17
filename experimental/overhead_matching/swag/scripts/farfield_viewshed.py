"""Which mapped landmarks can an observer actually see, and is the geometry any good?

This answers the site-selection question that `docs/mapillary-dataset-creation.md`
currently leaves to judgement: ferries were picked because water gives an empty
foreground, but water is only one way to get a long sightline and it is not the
best one. Given a candidate observer position, this computes terrain
line-of-sight against a catalog of tall things and reports both how many are
visible and whether their bearings are arranged usefully.

Two numbers, not one, because far-field and near-field landmarks do different
jobs:

  * **Heading.** A bearing to a landmark at range R has sensitivity dtheta/dp =
    1/R to observer position but sensitivity 1 to heading, at any range. So a
    peak at 50 km says almost nothing about where you are and everything about
    which way you are pointing. Under design doc section 5.2, where position
    routes through the heading state, that is precisely the quantity that stops
    a dead-reckoned track from rotating -- so `n_far` and `axial_spread` are the
    headline far-field numbers.
  * **Position.** The 1/R^2-weighted information matrix, reported as the two
    axes of the resulting position covariance in metres. A harbour crossing
    lands most of its landmarks in one bearing wedge, which pins cross-range and
    leaves down-range nearly free; that shows up here as a large
    `pos_sigma_major_m` next to a small minor axis, which a landmark *count*
    would never reveal.

Scoring a site on visible-landmark count alone ranks a wall of distant peaks
level with a genuinely well-conditioned basin. That is the mistake this module
exists to avoid.

Terrain comes from the Mapzen/AWS "skadi" mirror of void-filled SRTM: 1x1 degree
tiles of big-endian int16, 3601x3601, public and unauthenticated. No GDAL, no
rasterio -- the format is a raw array with a filename-encoded corner.

    # one observer
    bazel run //experimental/overhead_matching/swag/scripts:farfield_viewshed -- \\
        --lat 46.5197 --lon 6.6323 --landmarks /tmp/geneva_landmarks.json

    # score every track a discovery run proposed
    bazel run //experimental/overhead_matching/swag/scripts:farfield_viewshed -- \\
        --tracks /tmp/candidates.json --landmarks /tmp/geneva_landmarks.json \\
        --output /tmp/scored.json
"""

import argparse
import gzip
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SKADI_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}{lat:02d}/{ns}{lat:02d}{ew}{lon:03d}.hgt.gz"
DEFAULT_DEM_CACHE = Path.home() / "scratch" / "dem_cache"

SRTM_VOID = -32768
EARTH_RADIUS_M = 6_371_000.0

# Optical refraction coefficient. R_eff = R / (1 - k) bends the ray by folding
# the bend into the earth's radius, so a straight-line test in the corrected
# frame is exact. k = 0.13 is the standard *visual* value (R_eff ~ 7320 km).
#
# Do not reach for the familiar 4/3 earth here: that is k = 0.25, the *radio*
# convention, and it would overstate optical sightlines by roughly 8% in range
# -- enough to promote sites whose landmarks are actually below the horizon.
REFRACTION_K = 0.13


@dataclass(frozen=True)
class Landmark:
    """A tall thing, with the height bookkeeping that far-field visibility needs.

    `in_dem` is the field that is easy to get wrong and expensive to get wrong.
    A mountain is *already* part of the terrain model, so its height must be
    read from the DEM and `structure_height_m` must be zero; adding an `ele` tag
    on top would place a 4 km peak at 8 km. A radio mast is not in the DEM at
    all, so its height must be added to the ground elevation there or it is
    invisible. Same catalog, opposite treatment.
    """
    lat: float
    lon: float
    kind: str
    name: str = ""
    structure_height_m: float = 0.0
    in_dem: bool = False
    osm_id: str = ""


@dataclass(frozen=True)
class Sighting:
    """One landmark as seen (or not) from one observer position."""
    index: int
    name: str
    kind: str
    range_km: float
    bearing_deg: float
    elevation_angle_deg: float
    # How far above the highest intervening terrain the target sits. Visibility
    # is grazing > 0, but a landmark clearing the skyline by 0.02 degrees is a
    # geometric technicality: haze, trees and DEM error all live in that band.
    # Callers should threshold on this rather than on the boolean.
    grazing_deg: float
    visible: bool


# --------------------------------------------------------------------------
# terrain
# --------------------------------------------------------------------------

def _skadi_name(lat_sw: int, lon_sw: int) -> str:
    ns = "N" if lat_sw >= 0 else "S"
    ew = "E" if lon_sw >= 0 else "W"
    return f"{ns}{abs(lat_sw):02d}{ew}{abs(lon_sw):03d}.hgt"


def download_dem_tile(lat_sw: int, lon_sw: int, cache_dir: Path) -> Path | None:
    """Fetch one 1x1 degree SRTM tile, cached. None if the tile does not exist.

    A missing tile is ocean, not an error -- the mirror simply has no file
    there. Callers get sea level for those cells, which is correct.
    """
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_skadi_name(lat_sw, lon_sw)}.gz"
    missing_marker = cache_dir / f"{_skadi_name(lat_sw, lon_sw)}.absent"
    if path.exists():
        return path
    if missing_marker.exists():
        return None

    ns = "N" if lat_sw >= 0 else "S"
    ew = "E" if lon_sw >= 0 else "W"
    url = SKADI_URL.format(ns=ns, lat=abs(lat_sw), ew=ew, lon=abs(lon_sw))
    resp = requests.get(url, timeout=180)
    if resp.status_code == 404:
        missing_marker.write_text("no skadi tile (ocean)\n")
        return None
    resp.raise_for_status()
    tmp = path.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.replace(path)
    return path


def _load_dem_tile(lat_sw: int, lon_sw: int, cache_dir: Path) -> np.ndarray | None:
    path = download_dem_tile(lat_sw, lon_sw, cache_dir)
    if path is None:
        return None
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    side = int(round(math.sqrt(len(raw) / 2)))
    if side * side * 2 != len(raw):
        raise ValueError(f"{path} is {len(raw)} bytes, not a square int16 grid")
    grid = np.frombuffer(raw, dtype=">i2").reshape(side, side).astype(np.int16)
    return grid


def _max_pool(grid: np.ndarray, stride: int) -> np.ndarray:
    """Decimate by taking the highest post in each stride x stride cell.

    See DemMosaic.for_bbox for why this is not `grid[::stride, ::stride]`.
    Trailing rows and columns that do not fill a whole cell are trimmed rather
    than partially pooled, so the geotransform stays exactly `stride` posts per
    cell; the loss is at most `stride - 1` posts at the far edge, well inside
    the margin any caller should already be leaving.
    """
    rows = (grid.shape[0] // stride) * stride
    cols = (grid.shape[1] // stride) * stride
    trimmed = grid[:rows, :cols]
    reshaped = trimmed.reshape(rows // stride, stride, cols // stride, stride)
    return np.ascontiguousarray(reshaped.max(axis=(1, 3)))


class DemMosaic:
    """Contiguous elevation grid over a bbox, assembled from 1-degree tiles.

    Assembling once and indexing into a single array beats per-point tile
    dispatch by a wide margin, because line-of-sight sampling touches millions
    of scattered posts and every one of them would otherwise pay a lookup.

    Row 0 is the *north* edge of the mosaic and column 0 the west edge, matching
    the .hgt convention. Elevation is returned by bilinear interpolation.
    """

    def __init__(self, grid: np.ndarray, north: float, west: float,
                 step_deg: float, stride: int):
        self.grid = grid
        self.north = north
        self.west = west
        self.step_deg = step_deg
        self.stride = stride
        # A true lower bound on terrain anywhere in the mosaic, which is what
        # makes the curvature cull in visible_landmarks exact. Computed once
        # because it is read on every observer position.
        self.min_elevation = float(grid.min()) if grid.size else 0.0

    @classmethod
    def for_bbox(cls, bbox, cache_dir: Path = DEFAULT_DEM_CACHE,
                 stride: int = 1, verbose: bool = True) -> "DemMosaic":
        """Build a mosaic covering (west, south, east, north) in degrees.

        `stride` decimates the 1 arc-second source, trading accuracy for memory
        on large regions; a 3x2 degree mosaic is 467 MB at stride 1 and 52 MB
        at stride 3.

        **Do not decimate unless memory forces it, and never compare scores
        across strides.** Decimation was measured over Lake Geneva against
        stride 1 and it moves far-landmark counts by tens of percent:

            site       stride 1   subsample s3   max-pool s3
            Lausanne        457            644           656
            Nyon            569            933          1114

        Subsampling (`grid[::3, ::3]`) steps over narrow ridge crests, so things
        behind them become spuriously visible -- which is why this max-pools
        instead. But max-pooling does not fix it either, and the reason is worth
        understanding: the pooled grid serves *both* roles. It supplies the
        occluding terrain, where taking the maximum is genuinely conservative,
        and it supplies target and observer ground elevations, where taking the
        maximum lifts every landmark on sloped ground by tens of metres. The
        second effect dominates, so pooling is *more* optimistic than
        subsampling, not less.

        A one-sided guarantee would need separate grids for occluders and for
        point lookups, which would give back the memory that decimation was for.
        Max-pooling is kept because its failure is at least explicable, and the
        default is stride 1, where neither effect exists.
        """
        west, south, east, north = bbox
        lat0, lat1 = math.floor(south), math.ceil(north) - 1
        lon0, lon1 = math.floor(west), math.ceil(east) - 1
        lat1 = max(lat1, lat0)
        lon1 = max(lon1, lon0)

        n_lat = lat1 - lat0 + 1
        n_lon = lon1 - lon0 + 1
        posts = None
        tiles = {}
        for lat_sw in range(lat0, lat1 + 1):
            for lon_sw in range(lon0, lon1 + 1):
                tile = _load_dem_tile(lat_sw, lon_sw, cache_dir)
                if tile is not None:
                    tiles[(lat_sw, lon_sw)] = tile
                    posts = tile.shape[0]
        if posts is None:
            # Every tile is ocean. A flat sea-level mosaic is the right answer,
            # and it must still have a usable geotransform.
            posts = 3601
        if verbose:
            present = len(tiles)
            print(f"  DEM: {present}/{n_lat * n_lon} tiles present, "
                  f"{posts}x{posts} posts, stride {stride}", file=sys.stderr)

        # Tiles overlap by one row/column (both edges are inclusive), so drop
        # the last row and column of each before stacking or every degree
        # boundary gets a duplicated post and the geotransform drifts.
        inner = posts - 1
        mosaic = np.zeros((n_lat * inner, n_lon * inner), dtype=np.int16)
        for (lat_sw, lon_sw), tile in tiles.items():
            row = (lat1 - lat_sw) * inner
            col = (lon_sw - lon0) * inner
            mosaic[row:row + inner, col:col + inner] = tile[:inner, :inner]

        voids = mosaic == SRTM_VOID
        if voids.any():
            mosaic[voids] = 0
            if verbose:
                print(f"  DEM: {voids.sum()} void posts zeroed", file=sys.stderr)

        if stride > 1:
            mosaic = _max_pool(mosaic, stride)
            if verbose:
                print(f"  DEM: stride {stride} -- counts are NOT comparable to "
                      f"a stride-1 run, see DemMosaic.for_bbox", file=sys.stderr)

        step_deg = stride / inner
        return cls(mosaic.astype(np.float32), north=float(lat1 + 1),
                   west=float(lon0), step_deg=step_deg, stride=stride)

    def elevation(self, lat, lon):
        """Bilinearly interpolated elevation, in metres. Vectorised over arrays.

        Positions outside the mosaic clamp to the edge rather than raising:
        a ray that leaves the loaded area is over terrain we did not fetch, and
        clamping keeps the profile finite. Callers wanting strictness should
        size the bbox to the query range.
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        rows = (self.north - lat) / self.step_deg
        cols = (lon - self.west) / self.step_deg

        max_row = self.grid.shape[0] - 1
        max_col = self.grid.shape[1] - 1
        rows = np.clip(rows, 0, max_row - 1e-6)
        cols = np.clip(cols, 0, max_col - 1e-6)

        r0 = np.floor(rows).astype(np.int32)
        c0 = np.floor(cols).astype(np.int32)
        r1 = np.minimum(r0 + 1, max_row)
        c1 = np.minimum(c0 + 1, max_col)
        fr = rows - r0
        fc = cols - c0

        g = self.grid
        top = g[r0, c0] * (1 - fc) + g[r0, c1] * fc
        bot = g[r1, c0] * (1 - fc) + g[r1, c1] * fc
        return top * (1 - fr) + bot * fr


# --------------------------------------------------------------------------
# line of sight
# --------------------------------------------------------------------------

def _local_scale(lat_deg: float) -> tuple[float, float]:
    """Metres per degree of (longitude, latitude) at this latitude."""
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111_132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)
    m_per_deg_lon = 111_412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat)
    return m_per_deg_lon, m_per_deg_lat


def horizon_range_km(observer_height_m: float, target_height_m: float,
                     refraction_k: float = REFRACTION_K) -> float:
    """Curvature-limited maximum range, ignoring terrain.

    This is the gate that decides which landmark *classes* are worth cataloguing
    at all: at a 2 m eye height a 30 m water tower is capped near 27 km no
    matter how flat the ground, while a 3000 m peak reaches ~230 km. It is why
    peaks and masts dominate any far-field catalog.
    """
    r_eff = EARTH_RADIUS_M / (1.0 - refraction_k)
    return (math.sqrt(2 * r_eff * max(observer_height_m, 0.0))
            + math.sqrt(2 * r_eff * max(target_height_m, 0.0))) / 1000.0


def visible_landmarks(dem: DemMosaic, obs_lat: float, obs_lon: float,
                      landmarks: list[Landmark], observer_height_agl_m: float = 2.0,
                      max_range_km: float = 80.0, min_range_km: float = 0.2,
                      step_m: float = 90.0, refraction_k: float = REFRACTION_K,
                      exclude_near_target_m: float = 250.0,
                      chunk: int = 400) -> list[Sighting]:
    """Terrain line-of-sight from one observer to every landmark in range.

    Method is the standard reference-plane sweep: fold curvature and refraction
    into the earth radius, express every terrain post as an elevation angle
    above the observer, and take a running maximum along the ray. The target is
    visible exactly when its own elevation angle exceeds that maximum. The
    margin between the two is `grazing_deg`, which is far more useful than the
    boolean it implies.

    `exclude_near_target_m` drops terrain within a short distance of the target
    from the occluder set. Without it every mountain occludes itself: the DEM
    post at a summit *is* the summit, so it returns the target's own elevation
    angle and grazing collapses to zero for the entire peak catalog. The
    excluded ground is the landmark's own massif, not an occluder.
    """
    if not landmarks:
        return []

    m_per_lon, m_per_lat = _local_scale(obs_lat)
    lats = np.array([lm.lat for lm in landmarks])
    lons = np.array([lm.lon for lm in landmarks])

    dx = (lons - obs_lon) * m_per_lon
    dy = (lats - obs_lat) * m_per_lat
    ranges_m = np.hypot(dx, dy)

    in_band = (ranges_m >= min_range_km * 1000) & (ranges_m <= max_range_km * 1000)
    keep = np.flatnonzero(in_band)
    if keep.size == 0:
        return []

    obs_ground = float(dem.elevation(obs_lat, obs_lon))
    obs_elev = obs_ground + observer_height_agl_m
    r_eff = EARTH_RADIUS_M / (1.0 - refraction_k)

    struct_h = np.array([lm.structure_height_m for lm in landmarks])
    in_dem = np.array([lm.in_dem for lm in landmarks])
    ground = dem.elevation(lats, lons)
    # A peak's height is the DEM's; a mast's is the DEM plus its own structure.
    # `in_dem` therefore selects, it does not add.
    target_elev = np.where(in_dem, ground, ground + struct_h)

    bearings = np.degrees(np.arctan2(dx, dy)) % 360.0

    # Cull what curvature alone already hides, before paying for a ray march.
    # Worth ~8x here, because it removes the short classes (15-35 m bridges,
    # tanks, spires) at long range, which are most of the catalog's rows.
    #
    # The datum is the whole mosaic's minimum elevation, and that choice is what
    # makes the cull *exact* rather than merely plausible. The two-height
    # horizon sqrt(2R h1) + sqrt(2R h2) measures both heights above a datum
    # sphere, and the curvature bulge that does the hiding sits on that sphere.
    # Since no terrain anywhere is below `grid.min()` by construction, the bulge
    # can only be lower than assumed, so the reach is over-estimated and the
    # cull can only drop targets that are genuinely unreachable.
    #
    # Two tempting datums that are both wrong:
    #
    #   * the *observer's* elevation -- caps everything below the observer at
    #     the observer's own ~5 km horizon, deleting the entire view from a
    #     ridge road or overlook, which is the geometry those are chosen for;
    #   * the *lower of the two endpoints* -- looks safe, and is not. Where the
    #     ground between them dips below both (a valley, a lake basin) the
    #     sightline passes under the assumed bulge and reaches further than the
    #     bound. Measured on Lake Geneva this quietly dropped 19 of 457 genuinely
    #     visible landmarks at Lausanne and 40 of 569 at Nyon.
    datum = dem.min_elevation
    obs_reach = math.sqrt(2 * r_eff * max(obs_elev - datum, 0.0))
    horizon_m = obs_reach + np.sqrt(2 * r_eff * np.maximum(target_elev - datum, 0.0))
    keep = keep[ranges_m[keep] <= horizon_m[keep]]
    if keep.size == 0:
        return []

    n_steps = max(8, int(math.ceil(max_range_km * 1000 / step_m)))
    t = np.linspace(0.0, 1.0, n_steps + 1)[1:-1]  # exclusive of both endpoints

    sightings: list[Sighting] = []
    for start in range(0, keep.size, chunk):
        idx = keep[start:start + chunk]
        rng = ranges_m[idx]

        # (n_targets, n_steps) sample grid along each ray.
        sample_lat = obs_lat + np.outer(lats[idx] - obs_lat, t)
        sample_lon = obs_lon + np.outer(lons[idx] - obs_lon, t)
        sample_d = np.outer(rng, t)

        terrain = dem.elevation(sample_lat, sample_lon)
        # Reference-plane correction: everything measured against the tangent
        # plane at the observer, with the curvature drop subtracted.
        rise = terrain - obs_elev - sample_d ** 2 / (2 * r_eff)
        with np.errstate(divide="ignore", invalid="ignore"):
            angles = np.arctan2(rise, sample_d)

        usable = sample_d < (rng[:, None] - exclude_near_target_m)
        angles = np.where(usable, angles, -np.inf)
        max_terrain_angle = angles.max(axis=1)
        # A target closer than the exclusion buffer has no occluder samples at
        # all; -inf there is correct (nothing blocks it), not a bug.
        max_terrain_angle = np.where(np.isfinite(max_terrain_angle),
                                     max_terrain_angle, -np.pi / 2)

        target_rise = target_elev[idx] - obs_elev - rng ** 2 / (2 * r_eff)
        target_angle = np.arctan2(target_rise, rng)
        grazing = np.degrees(target_angle - max_terrain_angle)

        for k, i in enumerate(idx):
            lm = landmarks[i]
            sightings.append(Sighting(
                index=int(i), name=lm.name, kind=lm.kind,
                range_km=float(rng[k] / 1000.0),
                bearing_deg=float(bearings[i]),
                elevation_angle_deg=float(math.degrees(target_angle[k])),
                grazing_deg=float(grazing[k]),
                visible=bool(grazing[k] > 0.0),
            ))
    return sightings


# --------------------------------------------------------------------------
# geometry scoring
# --------------------------------------------------------------------------

def axial_spread(bearings_deg) -> float:
    """How evenly bearings are spread, on the axis that matters for geometry.

    Doubled-angle circular variance: 1 - |mean(exp(2 i theta))|, in [0, 1].

    The doubling is not cosmetic. Bearing information enters as an outer product
    u u^T, and u and -u contribute identically -- so landmarks dead ahead and
    dead astern constrain the *same* direction and are geometrically redundant
    even though they are 180 degrees apart. An ordinary circular variance calls
    that pair maximally spread and scores it 1.0, which is exactly backwards.
    Here it scores 0.
    """
    if len(bearings_deg) == 0:
        return 0.0
    doubled = 2 * np.radians(np.asarray(bearings_deg, dtype=float))
    return float(1.0 - abs(np.exp(1j * doubled).mean()))


def azimuth_coverage(bearings_deg, n_bins: int = 36) -> float:
    """Fraction of azimuth bins holding at least one landmark."""
    if len(bearings_deg) == 0:
        return 0.0
    bins = (np.asarray(bearings_deg, dtype=float) % 360.0) / (360.0 / n_bins)
    return float(len(np.unique(bins.astype(int))) / n_bins)


def position_covariance(bearings_deg, ranges_km, sigma_deg: float = 1.0):
    """Position uncertainty axes implied by a set of bearing measurements.

    J = sum (1/sigma^2) u u^T / R^2 with u perpendicular to each line of sight,
    because moving the observer perpendicular to a sightline by d changes the
    measured bearing by d/R. Returns (sigma_major_m, sigma_minor_m, condition),
    which is the same content as det(J) but in units a reader can judge.

    The 1/R^2 is what makes this a *position* metric and not a far-field one:
    a landmark at 50 km contributes 10^4 times less than one at 500 m. Report it
    alongside the far-field counts, never instead of them.
    """
    bearings_deg = np.asarray(bearings_deg, dtype=float)
    ranges_m = np.asarray(ranges_km, dtype=float) * 1000.0
    if bearings_deg.size == 0:
        return float("inf"), float("inf"), float("inf")

    sigma_rad = math.radians(sigma_deg)
    theta = np.radians(bearings_deg)
    # Perpendicular to the line of sight, in (east, north).
    ux, uy = np.cos(theta), -np.sin(theta)
    w = 1.0 / (sigma_rad ** 2 * ranges_m ** 2)

    info = np.array([[np.sum(w * ux * ux), np.sum(w * ux * uy)],
                     [np.sum(w * ux * uy), np.sum(w * uy * uy)]])
    eigenvalues = np.linalg.eigvalsh(info)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    # Rank deficiency has to be judged *relatively*, not against zero. Bearings
    # exactly 180 apart are algebraically rank 1, but sin(pi) is 1.2e-16 rather
    # than 0, so the small eigenvalue comes back around 1e-37 and the naive test
    # reports a 2e18 m axis -- a number that is useless and, worse, finite, so
    # it survives every downstream filter that only guards against inf.
    if eigenvalues[1] <= 0 or eigenvalues[0] / eigenvalues[1] < 1e-12:
        minor = float("inf") if eigenvalues[1] <= 0 else 1.0 / math.sqrt(eigenvalues[1])
        return float("inf"), minor, float("inf")
    sigma_major = 1.0 / math.sqrt(eigenvalues[0])
    sigma_minor = 1.0 / math.sqrt(eigenvalues[1])
    return float(sigma_major), float(sigma_minor), float(eigenvalues[1] / eigenvalues[0])


def site_metrics(sightings: list[Sighting], far_km: float = 5.0,
                 min_grazing_deg: float = 0.05, sigma_deg: float = 1.0) -> dict:
    """Summarise one observer position.

    `min_grazing_deg` discards landmarks that clear the skyline by less than a
    marginal amount. At 40 km, 0.05 degrees is 35 m of clearance -- inside SRTM's
    own vertical error, and well inside what a tree line or a haze layer will
    take away. Counting those as visible is how a viewshed flatters a site.
    """
    seen = [s for s in sightings if s.visible and s.grazing_deg >= min_grazing_deg]
    far = [s for s in seen if s.range_km >= far_km]

    all_bearings = [s.bearing_deg for s in seen]
    all_ranges = [s.range_km for s in seen]
    far_bearings = [s.bearing_deg for s in far]

    sigma_major, sigma_minor, cond = position_covariance(all_bearings, all_ranges, sigma_deg)
    return {
        "n_visible": len(seen),
        "n_far": len(far),
        "far_km": far_km,
        "max_range_km": max((s.range_km for s in far), default=0.0),
        "median_far_range_km": float(np.median([s.range_km for s in far])) if far else 0.0,
        "axial_spread": axial_spread(far_bearings),
        "azimuth_coverage": azimuth_coverage(far_bearings),
        "median_grazing_deg": float(np.median([s.grazing_deg for s in far])) if far else 0.0,
        "pos_sigma_major_m": sigma_major,
        "pos_sigma_minor_m": sigma_minor,
        "pos_condition": cond,
        "kinds": _count_kinds(far),
    }


def _count_kinds(sightings: list[Sighting]) -> dict:
    counts: dict[str, int] = {}
    for s in sightings:
        counts[s.kind] = counts.get(s.kind, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def farfield_score(metrics: dict) -> float:
    """One number for ranking, deliberately simple enough to argue with.

    score = log1p(n_far) * axial_spread * clamp(median_grazing / 0.5)

    Each factor can veto: no far landmarks, all of them on one axis, or all of
    them scraping the skyline each drive it to zero. It is a screening rank, not
    a physical quantity -- report the components alongside it, and re-rank by
    whichever component the question actually cares about.
    """
    if metrics["n_far"] == 0:
        return 0.0
    grazing_factor = min(1.0, metrics["median_grazing_deg"] / 0.5)
    return float(math.log1p(metrics["n_far"]) * metrics["axial_spread"] * grazing_factor)


# --------------------------------------------------------------------------
# track scoring
# --------------------------------------------------------------------------

def sample_track(coords, n_samples: int = 12) -> list[tuple[float, float]]:
    """Evenly spaced observer positions along a (lon, lat) polyline.

    Spaced by *arc length*, not vertex index. Vector tiles vary vertex density
    with how much the track turns, so index sampling piles observers into
    corners and leaves long straight legs unrepresented -- which for a ferry
    means sampling the harbour manoeuvring and skipping the open crossing.
    """
    if len(coords) < 2:
        return [(coords[0][1], coords[0][0])] if coords else []

    cumulative = [0.0]
    for (lon0, lat0), (lon1, lat1) in zip(coords, coords[1:]):
        m_per_lon, m_per_lat = _local_scale((lat0 + lat1) / 2)
        cumulative.append(cumulative[-1] + math.hypot((lon1 - lon0) * m_per_lon,
                                                      (lat1 - lat0) * m_per_lat))
    total = cumulative[-1]
    if total <= 0:
        return [(coords[0][1], coords[0][0])]

    out = []
    for i in range(n_samples):
        target = total * (i + 0.5) / n_samples
        j = int(np.searchsorted(cumulative, target))
        j = max(1, min(j, len(coords) - 1))
        span = cumulative[j] - cumulative[j - 1]
        frac = 0.0 if span <= 0 else (target - cumulative[j - 1]) / span
        lon = coords[j - 1][0] + frac * (coords[j][0] - coords[j - 1][0])
        lat = coords[j - 1][1] + frac * (coords[j][1] - coords[j - 1][1])
        out.append((lat, lon))
    return out


def skyline(sightings: list[Sighting], far_km: float = 5.0,
            min_grazing_deg: float = 0.05, bin_deg: float = 5.0) -> list[dict]:
    """The farthest visible landmark in each azimuth bin.

    A viewer wants to *see* what the score summarises, but a single Alpine
    observer has ~1,600 visible far landmarks and shipping all of them to a
    browser is megabytes of overdraw that renders as a solid disc.

    Binning by azimuth and keeping the farthest in each bin is not merely a
    sample: at long range the farthest thing on a bearing is the one on the
    skyline, and the skyline is exactly what a far-field matcher works with.
    Everything dropped is a nearer object on the same bearing, which is
    occluded-in-practice by whatever is in front of it or is simply a closer
    member of the same ridge.
    """
    seen = [s for s in sightings
            if s.visible and s.grazing_deg >= min_grazing_deg and s.range_km >= far_km]
    best: dict[int, Sighting] = {}
    for s in seen:
        key = int((s.bearing_deg % 360.0) / bin_deg)
        if key not in best or s.range_km > best[key].range_km:
            best[key] = s
    return [{"name": s.name, "kind": s.kind, "bearing": round(s.bearing_deg, 1),
             "range_km": round(s.range_km, 2), "grazing": round(s.grazing_deg, 3),
             "elev": round(s.elevation_angle_deg, 2)}
            for s in sorted(best.values(), key=lambda s: s.bearing_deg)]


def score_track(dem: DemMosaic, coords, landmarks: list[Landmark],
                n_samples: int = 12, far_km: float = 5.0,
                min_grazing_deg: float = 0.05, sigma_deg: float = 1.0,
                **kwargs) -> dict:
    """Aggregate site metrics over sample points along one track.

    Aggregation is by *median*, not mean, because a single sample sitting on a
    hilltop overlook can carry the mean for a track that is otherwise in a
    cutting. The union of far landmarks seen anywhere is reported separately as
    `n_far_union`: a track whose views change along its length is more useful
    for localisation than one that stares at the same three peaks, and the
    median alone cannot tell those apart.

    `far_km` and `min_grazing_deg` are named parameters rather than passed
    through `**kwargs`, because those go to `visible_landmarks` and these go to
    `site_metrics`; folding them together silently drops them.
    """
    observers = sample_track(coords, n_samples)
    per_sample, union = [], set()
    for lat, lon in observers:
        sightings = visible_landmarks(dem, lat, lon, landmarks, **kwargs)
        metrics = site_metrics(sightings, far_km=far_km,
                               min_grazing_deg=min_grazing_deg, sigma_deg=sigma_deg)
        metrics["lat"], metrics["lon"] = lat, lon
        metrics["score"] = farfield_score(metrics)
        per_sample.append(metrics)
        # Same admission test as site_metrics, grazing cut included. Counting
        # the union on a looser rule than the per-sample counts makes
        # n_far_union > max(n_far) for reasons that are pure bookkeeping.
        union.update(s.index for s in sightings
                     if s.visible and s.grazing_deg >= min_grazing_deg
                     and s.range_km >= far_km)

    if not per_sample:
        return {"n_samples": 0, "score": 0.0}

    def med(key):
        return float(np.median([m[key] for m in per_sample]))

    finite_major = [m["pos_sigma_major_m"] for m in per_sample
                    if math.isfinite(m["pos_sigma_major_m"])]
    return {
        "n_samples": len(per_sample),
        "score": med("score"),
        "score_max": float(max(m["score"] for m in per_sample)),
        "n_far_median": med("n_far"),
        "n_far_union": len(union),
        "n_visible_median": med("n_visible"),
        "axial_spread_median": med("axial_spread"),
        "azimuth_coverage_median": med("azimuth_coverage"),
        "max_range_km": float(max(m["max_range_km"] for m in per_sample)),
        "median_far_range_km": med("median_far_range_km"),
        "median_grazing_deg": med("median_grazing_deg"),
        "pos_sigma_major_m": float(np.median(finite_major)) if finite_major else float("inf"),
        "pos_sigma_minor_m": med("pos_sigma_minor_m"),
        "samples": per_sample,
    }


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

def load_landmarks(path: Path) -> list[Landmark]:
    data = json.loads(Path(path).read_text())
    records = data["landmarks"] if isinstance(data, dict) else data
    return [Landmark(**{k: v for k, v in r.items() if k in Landmark.__annotations__})
            for r in records]


def landmarks_bbox(landmarks: list[Landmark], pad_deg: float = 0.1):
    lats = [lm.lat for lm in landmarks]
    lons = [lm.lon for lm in landmarks]
    return (min(lons) - pad_deg, min(lats) - pad_deg,
            max(lons) + pad_deg, max(lats) + pad_deg)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--landmarks", type=Path, required=True,
                        help="JSON from farfield_landmarks.py")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tracks", type=Path, help="JSON from discover_tracks.py")
    source.add_argument("--lat", type=float, help="single observer latitude")
    parser.add_argument("--lon", type=float, help="single observer longitude")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dem_cache", type=Path, default=DEFAULT_DEM_CACHE)
    parser.add_argument("--dem_stride", type=int, default=1,
                        help="DEM decimation for large regions; 1 = native 30 m "
                             "(default). Higher strides max-pool, so they "
                             "over-occlude rather than flatter a site.")
    parser.add_argument("--observer_height_m", type=float, default=2.0)
    parser.add_argument("--max_range_km", type=float, default=80.0)
    parser.add_argument("--far_km", type=float, default=5.0)
    parser.add_argument("--n_samples", type=int, default=12,
                        help="observer positions per track")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--max_tracks", type=int,
                        help="score only the N longest tracks. Scoring costs "
                             "~0.3 s per observer position, so a whole region "
                             "at 8 samples runs to tens of minutes and the tail "
                             "is mostly 3 km fragments.")
    args = parser.parse_args()

    landmarks = load_landmarks(args.landmarks)
    print(f"{len(landmarks)} landmarks loaded", file=sys.stderr)

    if args.lat is not None:
        if args.lon is None:
            parser.error("--lat requires --lon")
        pad = args.max_range_km / 100.0
        bbox = (args.lon - pad, args.lat - pad, args.lon + pad, args.lat + pad)
    else:
        bbox = landmarks_bbox(landmarks)

    dem = DemMosaic.for_bbox(bbox, args.dem_cache, stride=args.dem_stride)

    los_kwargs = dict(observer_height_agl_m=args.observer_height_m,
                      max_range_km=args.max_range_km)

    if args.lat is not None:
        sightings = visible_landmarks(dem, args.lat, args.lon, landmarks, **los_kwargs)
        metrics = site_metrics(sightings, far_km=args.far_km)
        metrics["score"] = farfield_score(metrics)
        print(json.dumps(metrics, indent=2))
        seen = sorted([s for s in sightings if s.visible],
                      key=lambda s: -s.range_km)[:args.top]
        print(f"\n{'range km':>9} {'bearing':>8} {'grazing':>8}  {'kind':<22} name",
              file=sys.stderr)
        for s in seen:
            print(f"{s.range_km:9.1f} {s.bearing_deg:8.1f} {s.grazing_deg:8.2f}  "
                  f"{s.kind:<22} {s.name}", file=sys.stderr)
        return

    tracks = json.loads(args.tracks.read_text())
    records = tracks["tracks"] if isinstance(tracks, dict) else tracks
    if args.max_tracks and len(records) > args.max_tracks:
        dropped = len(records) - args.max_tracks
        records = sorted(records, key=lambda r: -r.get("length_km", 0))[:args.max_tracks]
        # Said out loud, because a silently truncated candidate list reads
        # downstream as "these are all the tracks in the region", which is
        # exactly how a region gets written off on partial evidence.
        print(f"  scoring the {args.max_tracks} longest of "
              f"{args.max_tracks + dropped} tracks ({dropped} not scored)",
              file=sys.stderr, flush=True)
    scored = []
    for i, track in enumerate(records, 1):
        coords = [(c[0], c[1]) for c in track["coords"]]
        result = score_track(dem, coords, landmarks, n_samples=args.n_samples,
                             far_km=args.far_km, **los_kwargs)
        samples = result.pop("samples", [])
        result["n_sample_points"] = len(samples)
        scored.append({**{k: v for k, v in track.items() if k != "coords"}, **result})
        # flush=True because Python block-buffers a redirected stderr, and a
        # long scoring run then shows nothing for minutes -- the same trap the
        # collection README records for the download lanes.
        print(f"  [{i}/{len(records)}] {track.get('sequence_id', '?')[:24]:24s} "
              f"score {result['score']:6.2f}  n_far {result['n_far_union']:4d}",
              file=sys.stderr, flush=True)

    scored.sort(key=lambda r: -r["score"])
    payload = {"n_tracks": len(scored), "tracks": scored}
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
