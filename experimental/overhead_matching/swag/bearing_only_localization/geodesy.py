"""Region-anchored local ENU frame with vectorized conversions.

Working frame per docs/localization-design-doc.md §4: metric ENU
(x = east, y = north) anchored at a region centroid. lat/lon appears only at
the boundaries of this module. Compass bearings are degrees clockwise from
true north = atan2(east, north), matching
landmark_filtering/bearing_geometry.py; the filter core works in radians.
"""

import math

import numpy as np

from experimental.overhead_matching.swag.landmark_filtering import (
    bearing_geometry as bg,
)


class RegionFrame:
    """Local tangent-plane ENU frame anchored at (anchor_lat, anchor_lon).

    Equirectangular approximation, same convention as
    bearing_geometry.enu_from_latlon (longitude scale fixed at the anchor
    latitude); vectorized over numpy arrays.
    """

    def __init__(self, anchor_lat_deg: float, anchor_lon_deg: float):
        self.anchor_lat_deg = float(anchor_lat_deg)
        self.anchor_lon_deg = float(anchor_lon_deg)
        self._meters_per_deg_lon = bg.METERS_PER_DEG_LAT * math.cos(
            math.radians(self.anchor_lat_deg))

    def enu_from_latlon(self, lat_deg, lon_deg):
        """lat/lon (scalars or arrays, degrees) -> (east_m, north_m)."""
        east_m = (np.asarray(lon_deg, dtype=np.float64)
                  - self.anchor_lon_deg) * self._meters_per_deg_lon
        north_m = (np.asarray(lat_deg, dtype=np.float64)
                   - self.anchor_lat_deg) * bg.METERS_PER_DEG_LAT
        return east_m, north_m

    def latlon_from_enu(self, east_m, north_m):
        """(east_m, north_m) (scalars or arrays) -> (lat_deg, lon_deg)."""
        lat_deg = self.anchor_lat_deg + (
            np.asarray(north_m, dtype=np.float64) / bg.METERS_PER_DEG_LAT)
        lon_deg = self.anchor_lon_deg + (
            np.asarray(east_m, dtype=np.float64) / self._meters_per_deg_lon)
        return lat_deg, lon_deg


def compass_bearing_rad(d_east_m, d_north_m):
    """Compass bearing (radians CW from north, in (-pi, pi]) of ENU deltas."""
    return np.arctan2(d_east_m, d_north_m)


def wrap_rad(angle_rad):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(angle_rad) + np.pi) % (2.0 * np.pi) - np.pi
