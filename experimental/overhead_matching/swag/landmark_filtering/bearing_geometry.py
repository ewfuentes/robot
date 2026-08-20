"""Bearing and local-plane geometry for panorama landmark filtering.

Conventions:
- Bbox coordinates are normalized 0-1000 per pinhole face (Gemini convention).
- Camera-frame bearing: face yaw plus the in-face azimuth offset, wrapped to
  [0, 360). Matches the pinhole projection in
  scripts/point_debugging.py:azimuth_elevation_from_pinhole_pixel
  (azimuth = atan2(col_frac, 1/tan(fov/2)) with col_frac in [-1, 1]).
- Elevation: up is positive (image y grows downward, so the sign is flipped).
- Compass bearings in ENU: degrees clockwise from north, i.e.
  atan2(east, north).

RESOLVED 2026-08-19 (was a KNOWN ISSUE from 2026-08-05):
`bearing_camera_deg` and `bbox_angles` now delegate to
`object_tracking/pano_geometry.direction_from_face_px`, the empirically verified
mapping, instead of carrying their own copy of the maths. There is exactly one
definition of the camera frame in this project and it lives there; see
docs/conventions.md.

What the old copy got wrong, precisely: it returned
`face_yaw + atan((2u-1)tan(fov/2))` where the verified render convention is
`-(face_yaw + atan((1-2u)tan(fov/2)))`. The two differ by exactly
`-2*face_yaw mod 360`, i.e. **0 deg on faces 0/180 and exactly 180 deg on faces
90/270**. Both increase bearing image-right, so the older description of this as
a "mirror within each face" was itself slightly wrong -- it is a per-face
constant rotation, which is why a fitted per-dataset yaw offset could absorb part
of it (the 0/180 faces) and never the rest.

Consequences of the fix, for anyone reading older artifacts: any bearing this
module produced for a box on face 90 or 270 was 180 deg out. Faces 0 and 180 are
unchanged, exactly. `ingest._is_seam_pair` was matched to the old convention and
is corrected in the same commit.
"""

import math

import numpy as np

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    pano_geometry as pg,
)

# WGS84 Earth radius; same value as common/gps/web_mercator.py, inlined so this
# module (and the viewer that depends on it) doesn't import torch.
EARTH_RADIUS_M = 6378137.0
METERS_PER_DEG_LAT = 2.0 * math.pi * EARTH_RADIUS_M / 360.0
BBOX_NORM_MAX = 1000.0


def wrap_deg(angle_deg):
    """Wrap angle(s) to [-180, 180). Works on scalars and numpy arrays."""
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


def circular_diff_deg(a_deg, b_deg):
    """Signed smallest difference a - b in degrees, in [-180, 180)."""
    return wrap_deg(np.asarray(a_deg) - np.asarray(b_deg))


def circular_mean_deg(angles_deg) -> float:
    angles_rad = np.radians(np.asarray(angles_deg, dtype=np.float64))
    mean_deg = float(np.degrees(np.arctan2(
        np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad))))) % 360.0
    # A tiny negative angle mod 360 can round to exactly 360.0.
    return 0.0 if mean_deg >= 360.0 else mean_deg


def bearing_camera_deg(face_yaw_deg: float, x_norm: float,
                       fov_deg: float = 90.0) -> float:
    """Camera-frame bearing of a normalized x coordinate on a pinhole face.

    Thin wrapper over `pano_geometry.direction_from_face_px`, kept so existing
    callers keep working. Do not reimplement the maths here: one definition of
    the camera frame, in pano_geometry. See docs/conventions.md.
    """
    return pg.direction_from_face_px(face_yaw_deg, x_norm, 0.0, fov_deg)[0]


def elevation_deg(y_norm: float, fov_deg: float = 90.0) -> float:
    """Elevation (up positive) of a normalized y coordinate on a pinhole face."""
    row_frac = y_norm / BBOX_NORM_MAX * 2.0 - 1.0
    return -math.degrees(
        math.atan(row_frac * math.tan(math.radians(fov_deg) / 2.0)))


def bbox_angles(face_yaw_deg: float, xmin: float, ymin: float, xmax: float,
                ymax: float, fov_deg: float = 90.0):
    """Center bearing, center elevation, and angular width of one bbox.

    Angular width is the circular difference of the edge bearings, so a
    full-width face box measures exactly fov_deg.
    """
    center_bearing = bearing_camera_deg(
        face_yaw_deg, (xmin + xmax) / 2.0, fov_deg)
    left = bearing_camera_deg(face_yaw_deg, xmin, fov_deg)
    right = bearing_camera_deg(face_yaw_deg, xmax, fov_deg)
    width = abs(float(circular_diff_deg(right, left)))
    center_elevation = elevation_deg((ymin + ymax) / 2.0, fov_deg)
    return center_bearing, center_elevation, width


def enu_from_latlon(lat_deg: float, lon_deg: float, anchor_lat_deg: float,
                    anchor_lon_deg: float):
    """(east_m, north_m) of a point relative to an anchor.

    Equirectangular approximation - fine at the few-km scale of a walking
    trajectory.
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


def compass_bearing_deg(east_m: float, north_m: float) -> float:
    """Compass bearing (degrees clockwise from north) of an ENU vector."""
    return math.degrees(math.atan2(east_m, north_m)) % 360.0


def bearing_unit_vector(bearing_deg: float):
    """ENU unit vector (east, north) pointing along a compass bearing."""
    bearing_rad = math.radians(bearing_deg)
    return math.sin(bearing_rad), math.cos(bearing_rad)
