"""Boat heading over time from positions.

GPS-derived course stands in for whatever odometry is available at
deployment (SLAM, compass); everything downstream consumes only
"heading in degrees, clockwise from north, at time t". Course is undefined
while the boat is stationary (e.g. at the dock), so headings are computed
between anchor points spaced a minimum displacement apart and interpolated /
held elsewhere.
"""

import math

import numpy as np


class HeadingModel:
    """Piecewise-linear heading vs time, built from sparse position fixes.

    Headings are stored unwrapped (continuous, may exceed [0, 360)) so
    interpolation never takes the long way around the circle.
    """

    def __init__(self, times_s: np.ndarray, headings_unwrapped_deg: np.ndarray):
        self.times_s = times_s
        self.headings_unwrapped_deg = headings_unwrapped_deg

    def at(self, t_s):
        """Heading (unwrapped degrees, CW from north) at time(s) t_s."""
        return np.interp(t_s, self.times_s, self.headings_unwrapped_deg)

    def delta(self, t_s, t_ref_s) -> float:
        """Signed heading change from t_ref_s to t_s in degrees."""
        return float(self.at(t_s) - self.at(t_ref_s))


def heading_model_from_positions(east_m, north_m, times_s,
                                 min_displacement_m: float = 3.0,
                                 smooth_window_s: float = 10.0) -> HeadingModel:
    """Fit a HeadingModel to a position track.

    Anchor points are spaced at least min_displacement_m apart along the
    track; each inter-anchor segment contributes one heading sample at its
    midpoint time. A moving average over ~smooth_window_s smooths GPS jitter.
    """
    east_m = np.asarray(east_m, dtype=np.float64)
    north_m = np.asarray(north_m, dtype=np.float64)
    times_s = np.asarray(times_s, dtype=np.float64)

    anchors = [0]
    for i in range(1, len(east_m)):
        d = math.hypot(east_m[i] - east_m[anchors[-1]],
                       north_m[i] - north_m[anchors[-1]])
        if d >= min_displacement_m:
            anchors.append(i)
    if len(anchors) < 2:
        # Never moved far enough: heading is arbitrary but defined.
        return HeadingModel(np.array([times_s[0], times_s[-1]]),
                            np.array([0.0, 0.0]))

    seg_times, seg_headings = [], []
    for a, b in zip(anchors[:-1], anchors[1:]):
        de = east_m[b] - east_m[a]
        dn = north_m[b] - north_m[a]
        seg_times.append((times_s[a] + times_s[b]) / 2.0)
        seg_headings.append(math.degrees(math.atan2(de, dn)))
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

    return HeadingModel(seg_times, seg_headings)
