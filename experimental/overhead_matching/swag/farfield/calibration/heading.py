"""GPS course over time from positions.

GPS-derived course stands in for relative odometry available at deployment;
it is not a rigid camera calibration and is not true platform heading. Course
is measured in degrees clockwise from true north. It is undefined
while the vehicle is stationary (e.g. at the dock), so course samples are
computed between anchor points spaced a minimum displacement apart and
interpolated / held elsewhere.

Course here is always derived from POSITIONS. The intrinsics' `heading_deg`
column is never consulted: its meaning varies per dataset (compass vs camera
yaw, mount offset sometimes already folded in), while positions mean one
thing everywhere.
"""

import math

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo


class GpsCourseModel:
    """Piecewise-linear GPS course vs time from sparse moving fixes.

    Courses are stored unwrapped (continuous, may exceed [0, 360)) so
    interpolation never takes the long way around the circle.
    """

    def __init__(self, times_s: np.ndarray,
                 course_world_cw_unwrapped_deg: np.ndarray):
        self.times_s = times_s
        self.course_world_cw_unwrapped_deg = course_world_cw_unwrapped_deg

    def course_world_cw_deg_at(self, t_s):
        """Unwrapped course degrees CW from true north at time(s) ``t_s``."""
        return np.interp(
            t_s, self.times_s, self.course_world_cw_unwrapped_deg)

    def delta_course_cw_deg(self, t_s, t_ref_s) -> float:
        """Signed clockwise course change from ``t_ref_s`` to ``t_s``."""
        return float(self.course_world_cw_deg_at(t_s)
                     - self.course_world_cw_deg_at(t_ref_s))


def gps_course_model_from_positions(east_m, north_m, times_s,
                                    *, min_displacement_m: float,
                                    smooth_window_s: float) \
        -> GpsCourseModel | None:
    """Fit a course model, or abstain when displacement is inadequate.

    Anchor points are spaced at least min_displacement_m apart along the
    track; each inter-anchor segment contributes one course sample at its
    midpoint time. A moving average over ~smooth_window_s smooths GPS jitter.
    """
    east_m = np.asarray(east_m, dtype=np.float64)
    north_m = np.asarray(north_m, dtype=np.float64)
    times_s = np.asarray(times_s, dtype=np.float64)
    if (east_m.ndim != 1 or east_m.shape != north_m.shape
            or east_m.shape != times_s.shape or east_m.size < 2
            or not np.all(np.isfinite(east_m))
            or not np.all(np.isfinite(north_m))
            or not np.all(np.isfinite(times_s))):
        raise ValueError("positions/times must be matching finite 1-D arrays")
    if np.any(np.diff(times_s) <= 0.0):
        raise ValueError("times_s must be strictly increasing")
    for value, name, allow_zero in (
            (min_displacement_m, "min_displacement_m", False),
            (smooth_window_s, "smooth_window_s", True)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0.0
                or (not allow_zero and value == 0.0)):
            qualifier = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}")

    anchors = [0]
    for i in range(1, len(east_m)):
        d = math.hypot(east_m[i] - east_m[anchors[-1]],
                       north_m[i] - north_m[anchors[-1]])
        if d >= min_displacement_m:
            anchors.append(i)
    if len(anchors) < 2:
        return None

    seg_times, seg_headings = [], []
    for a, b in zip(anchors[:-1], anchors[1:]):
        de = east_m[b] - east_m[a]
        dn = north_m[b] - north_m[a]
        seg_times.append((times_s[a] + times_s[b]) / 2.0)
        # The one compass-bearing definition (geometry.py); unwrap below makes
        # the [0, 360) representative irrelevant.
        seg_headings.append(geo.compass_bearing_deg(de, dn))
    seg_times = np.asarray(seg_times)
    seg_headings = np.degrees(np.unwrap(np.radians(np.asarray(seg_headings))))

    if smooth_window_s > 0 and len(seg_headings) > 2:
        dt = np.median(np.diff(seg_times))
        n = max(1, int(round(smooth_window_s / max(dt, 1e-6))))
        if n > 1:
            kernel = np.ones(n) / n
            padded = np.concatenate([
                np.full(n // 2, seg_headings[0]), seg_headings,
                np.full(n - 1 - n // 2, seg_headings[-1])])
            seg_headings = np.convolve(padded, kernel, mode="valid")

    return GpsCourseModel(seg_times, seg_headings)
