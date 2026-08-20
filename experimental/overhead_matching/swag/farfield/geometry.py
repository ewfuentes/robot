"""The single owner of every farfield geometric convention.

Camera frame, mount offset, angle arithmetic, local ENU, and earth constants
are defined HERE and nowhere else. Import them; never restate them — not in
code, not in docstrings, not in docs. Convention *strings* (CAMERA_FRAME,
MOUNT_OFFSET_CONVENTION) exist so artifacts can embed the contract verbatim.

Conventions:
- Camera-frame azimuth is clockwise-positive from camera forward, and camera
  forward is the CENTRE column of the panorama:
  az_cw = (x / pano_w - 0.5) * 360. Elevation is up-positive:
  el_up = (0.5 - y / pano_h) * 180.
- Pinhole faces are rendered by scripts/panorama_to_pinhole.py with a
  CCW-positive azimuth az_ccw = face_yaw + atan((1 - 2u) tan(fov/2)) and pano
  column x = ((180 - az_ccw) / 360 mod 1) * W, so faces sit left-to-right in
  the panorama as 180 | 90 | 0 | 270. `direction_from_face_px` is the exact,
  empirically verified inverse of that render.
- Pinhole bbox coordinates are normalized 0-1000 per face, y down (the VLM
  extraction convention).
- Pano-space boxes may straddle the +-180 wrap; they are represented
  unwrapped: x_min in [0, W), x_max in (x_min, x_min + W], so x_max > W
  means the box wraps around the seam.
- Body-frame bearing = camera bearing minus the mount offset (see
  MOUNT_OFFSET_CONVENTION). World bearing = vehicle heading plus body
  bearing. Compass bearings are degrees clockwise from true north,
  atan2(east, north). Serialized bearings are stored in [0, 360); compare
  angles with the wrap helpers, never by subtraction.
- Local frame is metric ENU (x east, y north) anchored at a region point,
  equirectangular approximation with the longitude scale fixed at the anchor
  latitude (`RegionFrame`). lat/lon appears only at module boundaries.

This module is numpy-only (no torch, no repo deps) so viewers and tools can
import it freely.
"""

import math

import numpy as np

BBOX_NORM_MAX = 1000.0

# WGS84 equatorial radius; same value as common/gps/web_mercator.py. Used for
# degree<->meter scaling (which web-mercator tooling must agree with).
EARTH_RADIUS_M = 6378137.0
METERS_PER_DEG_LAT = 2.0 * math.pi * EARTH_RADIUS_M / 360.0
# Mean earth radius, the conventional haversine sphere. Kept distinct from
# EARTH_RADIUS_M on purpose: haversine distances and web-mercator scalings
# answer different questions and must each match their ecosystem.
MEAN_EARTH_RADIUS_M = 6371000.0

CAMERA_FRAME = (
    "Camera-frame azimuth is clockwise-positive from camera forward, and camera "
    "forward is the CENTRE column of the panorama: az_cw = (x/pano_w - 0.5)*360. "
    "Elevation is up-positive: el_up = (0.5 - y/pano_h)*180.")

MOUNT_OFFSET_CONVENTION = (
    "mount_offset_deg is the azimuth, IN THE CAMERA FRAME, of the vehicle's "
    "DIRECTION OF TRAVEL - not the bow. Applied as bearing_body_deg = "
    "(bearing_camera_deg - mount_offset_deg) mod 360. Camera-frame azimuth 0 is "
    "the CENTRE column of the panorama, not column 0; a prior reasoned in the "
    "column-0 convention is exactly 180 deg out.")


# ---------------------------------------------------------------------------
# Angle arithmetic
# ---------------------------------------------------------------------------

def wrap_deg(angle_deg):
    """Wrap angle(s) to [-180, 180). Works on scalars and numpy arrays."""
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


def wrap_rad(angle_rad):
    """Wrap angle(s) to [-pi, pi). Works on scalars and numpy arrays."""
    return (np.asarray(angle_rad) + np.pi) % (2.0 * np.pi) - np.pi


def circular_diff_deg(a_deg, b_deg):
    """Signed smallest difference a - b in degrees, in [-180, 180)."""
    return wrap_deg(np.asarray(a_deg) - np.asarray(b_deg))


def circular_mean_deg(angles_deg) -> float:
    angles_rad = np.radians(np.asarray(angles_deg, dtype=np.float64))
    mean_deg = float(np.degrees(np.arctan2(
        np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad))))) % 360.0
    # A tiny negative angle mod 360 can round to exactly 360.0.
    return 0.0 if mean_deg >= 360.0 else mean_deg


# ---------------------------------------------------------------------------
# Camera frame: pinhole faces <-> panorama pixels <-> directions
# ---------------------------------------------------------------------------

def direction_from_face_px(face_yaw_deg: float, x_norm: float, y_norm: float,
                           fov_deg: float = 90.0):
    """(az_cw_deg, el_up_deg) of a normalized pinhole-face pixel."""
    half_tan = math.tan(math.radians(fov_deg) / 2.0)
    # Render ray in face frame, z normalized to 1: [c, r, 1], c right-negative.
    c = (1.0 - 2.0 * x_norm / BBOX_NORM_MAX) * half_tan
    r = (2.0 * y_norm / BBOX_NORM_MAX - 1.0) * half_tan
    az_ccw_deg = face_yaw_deg + math.degrees(math.atan(c))
    el_down_deg = math.degrees(math.atan2(r, math.hypot(c, 1.0)))
    return (-az_ccw_deg) % 360.0, -el_down_deg


def bearing_camera_deg(face_yaw_deg: float, x_norm: float,
                       fov_deg: float = 90.0) -> float:
    """Camera-frame azimuth of a normalized x coordinate on a pinhole face."""
    return direction_from_face_px(face_yaw_deg, x_norm, 0.0, fov_deg)[0]


def pano_px_from_direction(az_cw_deg: float, el_up_deg: float,
                           pano_w: int, pano_h: int):
    """(x, y) pano pixel of a direction. x in [0, W), y clamped to [0, H)."""
    x = ((az_cw_deg / 360.0 + 0.5) % 1.0) * pano_w
    y = (0.5 - el_up_deg / 180.0) * pano_h
    return x, min(max(y, 0.0), pano_h - 1.0)


def direction_from_pano_px(x: float, y: float, pano_w: int, pano_h: int):
    """(az_cw_deg, el_up_deg) of a pano pixel; inverse of pano_px_from_direction."""
    az_cw_deg = (x / pano_w - 0.5) * 360.0 % 360.0
    el_up_deg = (0.5 - y / pano_h) * 180.0
    return az_cw_deg, el_up_deg


def azimuth_of_pano_column(x: float, pano_w: int) -> float:
    """Camera-frame azimuth of a pano column (the horizon row's direction).

    The named helper for the common "give me the azimuth of column x" case,
    so callers stop inventing sentinel pano heights.
    """
    return direction_from_pano_px(x % pano_w, 0.0, pano_w, 1)[0]


def bbox_angles(face_yaw_deg: float, xmin: float, ymin: float, xmax: float,
                ymax: float, fov_deg: float = 90.0):
    """Center azimuth, center elevation, and angular width of one face bbox.

    Elevation is evaluated at the bbox center (off-axis foreshortening
    included), not at the face-center column. Angular width is the circular
    difference of the edge bearings, so a full-width face box measures
    exactly fov_deg.
    """
    center_az, center_el = direction_from_face_px(
        face_yaw_deg, (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, fov_deg)
    left = bearing_camera_deg(face_yaw_deg, xmin, fov_deg)
    right = bearing_camera_deg(face_yaw_deg, xmax, fov_deg)
    width = abs(float(circular_diff_deg(right, left)))
    return center_az, center_el, width


def _edge_samples(xmin: float, ymin: float, xmax: float, ymax: float,
                  n_per_edge: int):
    """Points along the four edges of a bbox, in normalized face coords."""
    ts = np.linspace(0.0, 1.0, n_per_edge)
    xs = xmin + (xmax - xmin) * ts
    ys = ymin + (ymax - ymin) * ts
    top = np.stack([xs, np.full_like(xs, ymin)], axis=1)
    bottom = np.stack([xs, np.full_like(xs, ymax)], axis=1)
    left = np.stack([np.full_like(ys, xmin), ys], axis=1)
    right = np.stack([np.full_like(ys, xmax), ys], axis=1)
    return np.concatenate([top, bottom, left, right], axis=0)


def pano_bbox_from_face_bbox(face_yaw_deg: float, xmin: float, ymin: float,
                             xmax: float, ymax: float, pano_w: int,
                             pano_h: int, fov_deg: float = 90.0,
                             n_per_edge: int = 9):
    """Pano-pixel bbox (x_min, y_min, x_max, y_max) of one pinhole-face bbox.

    Edge points are sampled because straight pinhole edges curve in the
    equirectangular projection. x is unwrapped (see module docstring).
    """
    pts = _edge_samples(xmin, ymin, xmax, ymax, n_per_edge)
    center_az, _ = direction_from_face_px(
        face_yaw_deg, (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, fov_deg)
    xs, ys = [], []
    for x_norm, y_norm in pts:
        az, el = direction_from_face_px(face_yaw_deg, x_norm, y_norm, fov_deg)
        # Unwrap azimuth to within +-180 of the box center, keep x unwrapped.
        az_unwrapped = center_az + (az - center_az + 180.0) % 360.0 - 180.0
        xs.append((az_unwrapped / 360.0 + 0.5) * pano_w)
        ys.append(min(max((0.5 - el / 180.0) * pano_h, 0.0), pano_h - 1.0))
    x_min, x_max = min(xs), max(xs)
    # Normalize so x_min lands in [0, W).
    shift = math.floor(x_min / pano_w) * pano_w
    return x_min - shift, min(ys), x_max - shift, max(ys)


def pano_bbox_union(boxes_pano, pano_w: int):
    """Union of unwrapped pano bboxes (e.g. a seam-merged observation).

    Boxes are re-unwrapped around the first box's center so a group that
    straddles the seam stays contiguous.
    """
    ref = (boxes_pano[0][0] + boxes_pano[0][2]) / 2.0
    xs, ys = [], []
    for x_min, y_min, x_max, y_max in boxes_pano:
        center = (x_min + x_max) / 2.0
        shift = round((center - ref) / pano_w) * pano_w
        xs.extend([x_min - shift, x_max - shift])
        ys.extend([y_min, y_max])
    x_min, x_max = min(xs), max(xs)
    shift = math.floor(x_min / pano_w) * pano_w
    return x_min - shift, min(ys), x_max - shift, max(ys)


def pano_bbox_for_observation(obs_boxes, pano_w: int, pano_h: int,
                              fov_deg: float = 90.0):
    """Pano bbox of an observation's (possibly seam-merged) box group.

    obs_boxes: iterable of objects with face_yaw_deg/xmin/ymin/xmax/ymax
    attributes.
    """
    boxes = [
        pano_bbox_from_face_bbox(b.face_yaw_deg, b.xmin, b.ymin, b.xmax,
                                 b.ymax, pano_w, pano_h, fov_deg)
        for b in obs_boxes
    ]
    return pano_bbox_union(boxes, pano_w)


def signed_x_offset(x: float, window_x0: float, pano_w: int) -> float:
    """Signed horizontal offset of pano x from window_x0 in [-W/2, W/2)."""
    return (x - window_x0 + pano_w / 2.0) % pano_w - pano_w / 2.0


def extract_window(pano: np.ndarray, x0: float, y0: float, width: int,
                   height: int):
    """Crop a (H, W, C) pano with horizontal wrap; vertical range is clamped.

    x0/y0 are the requested top-left corner in pano pixels (x may be
    unwrapped). Returns (crop, y_start) where y_start is the actual top row
    after clamping the window inside the pano vertically.
    """
    pano_h, pano_w = pano.shape[:2]
    cols = (np.arange(int(round(x0)), int(round(x0)) + width) % pano_w)
    y_start = int(round(min(max(y0, 0), pano_h - height)))
    return pano[y_start:y_start + height][:, cols], y_start


# ---------------------------------------------------------------------------
# Frame chain: camera -> body -> world
# ---------------------------------------------------------------------------

def apply_mount_offset(bearing_camera_deg: float,
                       mount_offset_deg: float) -> float:
    """Camera-frame azimuth -> body-frame bearing, in [0, 360).

    See MOUNT_OFFSET_CONVENTION for what mount_offset_deg means and the
    column-0 trap.
    """
    return (bearing_camera_deg - mount_offset_deg) % 360.0


def body_to_world_bearing_deg(heading_deg, bearing_body_deg):
    """Body-frame bearing + vehicle heading -> compass bearing, in [0, 360).

    Works on scalars and numpy arrays.
    """
    return (np.asarray(heading_deg) + np.asarray(bearing_body_deg)) % 360.0


def world_to_body_bearing_deg(heading_deg, bearing_world_deg):
    """Compass bearing - vehicle heading -> body-frame bearing, in [0, 360)."""
    return (np.asarray(bearing_world_deg) - np.asarray(heading_deg)) % 360.0


# ---------------------------------------------------------------------------
# Local ENU frame and geodesy
# ---------------------------------------------------------------------------

def enu_from_latlon(lat_deg: float, lon_deg: float, anchor_lat_deg: float,
                    anchor_lon_deg: float):
    """(east_m, north_m) of a point relative to an anchor.

    Equirectangular approximation, longitude scale fixed at the anchor
    latitude — fine at the tens-of-km scale of one dataset.
    """
    east_m = ((lon_deg - anchor_lon_deg) * METERS_PER_DEG_LAT
              * math.cos(math.radians(anchor_lat_deg)))
    north_m = (lat_deg - anchor_lat_deg) * METERS_PER_DEG_LAT
    return east_m, north_m


def latlon_from_enu(east_m: float, north_m: float, anchor_lat_deg: float,
                    anchor_lon_deg: float):
    lat_deg = anchor_lat_deg + north_m / METERS_PER_DEG_LAT
    lon_deg = anchor_lon_deg + east_m / (
        METERS_PER_DEG_LAT * math.cos(math.radians(anchor_lat_deg)))
    return lat_deg, lon_deg


class RegionFrame:
    """Local tangent-plane ENU frame anchored at (anchor_lat, anchor_lon).

    Vectorized twin of enu_from_latlon/latlon_from_enu (same approximation,
    same anchor-latitude longitude scale); accepts scalars or numpy arrays.
    """

    def __init__(self, anchor_lat_deg: float, anchor_lon_deg: float):
        self.anchor_lat_deg = float(anchor_lat_deg)
        self.anchor_lon_deg = float(anchor_lon_deg)
        self._meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(
            math.radians(self.anchor_lat_deg))

    def enu_from_latlon(self, lat_deg, lon_deg):
        """lat/lon (scalars or arrays, degrees) -> (east_m, north_m)."""
        east_m = (np.asarray(lon_deg, dtype=np.float64)
                  - self.anchor_lon_deg) * self._meters_per_deg_lon
        north_m = (np.asarray(lat_deg, dtype=np.float64)
                   - self.anchor_lat_deg) * METERS_PER_DEG_LAT
        return east_m, north_m

    def latlon_from_enu(self, east_m, north_m):
        """(east_m, north_m) (scalars or arrays) -> (lat_deg, lon_deg)."""
        lat_deg = self.anchor_lat_deg + (
            np.asarray(north_m, dtype=np.float64) / METERS_PER_DEG_LAT)
        lon_deg = self.anchor_lon_deg + (
            np.asarray(east_m, dtype=np.float64) / self._meters_per_deg_lon)
        return lat_deg, lon_deg


def compass_bearing_deg(east_m: float, north_m: float) -> float:
    """Compass bearing (degrees clockwise from north) of an ENU vector."""
    return math.degrees(math.atan2(east_m, north_m)) % 360.0


def compass_bearing_rad(d_east_m, d_north_m):
    """Compass bearing (radians CW from north, in (-pi, pi]) of ENU deltas.

    Works on scalars and numpy arrays; the filter core consumes this.
    """
    return np.arctan2(d_east_m, d_north_m)


def bearing_unit_vector(bearing_deg: float):
    """ENU unit vector (east, north) pointing along a compass bearing."""
    bearing_rad = math.radians(bearing_deg)
    return math.sin(bearing_rad), math.cos(bearing_rad)


def haversine_m(lat1_deg: float, lon1_deg: float, lat2_deg: float,
                lon2_deg: float) -> float:
    """Great-circle distance in meters on the mean-radius sphere.

    The one haversine. Use this for distances beyond a RegionFrame's
    validity, or when no anchor exists; use ENU deltas inside a frame.
    """
    lat1, lon1, lat2, lon2 = map(
        math.radians, (lat1_deg, lon1_deg, lat2_deg, lon2_deg))
    a = (math.sin((lat2 - lat1) / 2.0) ** 2
         + math.cos(lat1) * math.cos(lat2)
         * math.sin((lon2 - lon1) / 2.0) ** 2)
    return 2.0 * MEAN_EARTH_RADIUS_M * math.asin(math.sqrt(a))
