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

KNOWN ISSUE (found 2026-08-05 while building object_tracking/pano_geometry):
bearing_camera_deg does NOT return a physically consistent direction across
faces. The renderer (scripts/panorama_to_pinhole.py) builds rays with
col_frac = linspace(1, -1), i.e. yaw_090 faces 90 deg counter-clockwise-LEFT
of forward, and the pano is laid out left-to-right as faces 180|90|0|270.
This was verified empirically: regenerating the stored pinhole faces from the
panorama with that math matches to JPEG noise, and reprojected landmark boxes
land exactly on their objects (see object_tracking/m0_render_boxes.py).
Under the true convention, bearing_camera_deg is mirrored within each face
(bearing increases image-right here, but physical azimuth increases
image-left), which cancels on faces 0/180 only at the face center and leaves
faces 90/270 pointing ~180 deg away from the physical direction. Downstream
consumers (tracking gates, triangulation, yaw_offset) fit a per-dataset yaw
calibration with a sign parameter, which absorbs part but not all of this.
Do not use these bearings for anything that must land on panorama pixels;
use object_tracking/pano_geometry.py instead.
"""

import math

import numpy as np

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

    WARNING: mirrored relative to the true render convention; on faces
    90/270 the result is ~180 deg from the physical direction. See the
    KNOWN ISSUE note in the module docstring. The physically verified
    mapping is object_tracking/pano_geometry.direction_from_face_px.
    """
    col_frac = x_norm / BBOX_NORM_MAX * 2.0 - 1.0
    offset_rad = math.atan(col_frac * math.tan(math.radians(fov_deg) / 2.0))
    return (face_yaw_deg + math.degrees(offset_rad)) % 360.0


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
