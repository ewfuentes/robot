"""Causal crop course whose coordinate frame is fixed at the first fix.

The legacy ingest ENU anchor uses the full-trajectory mean latitude, which
changes longitude scaling even for earlier intervals. This adapter consumes
raw fixes and fixes its local scale once, without inspecting later fixes.
GPS remains a relative-rotation surrogate; this is not inertial odometry.
"""
import math
from causal_crop_course import CausalCropCourse
from experimental.overhead_matching.swag.farfield import geometry


class CausalGeodeticCourse:
    def __init__(self, window_seconds=10., min_displacement_m=3.):
        self.course = CausalCropCourse(window_seconds, min_displacement_m)
        self.anchor = None

    def append(self, time_s, lat_deg, lon_deg):
        if (not all(math.isfinite(v) for v in [time_s, lat_deg, lon_deg])
                or not -90 <= lat_deg <= 90 or not -180 <= lon_deg <= 180):
            raise ValueError('expected finite time and valid latitude/longitude')
        anchor = self.anchor or (lat_deg, lon_deg)
        east, north = geometry.enu_from_latlon(lat_deg, lon_deg, *anchor)
        snapshot = self.course.append(time_s, east, north)
        self.anchor = anchor
        return snapshot
