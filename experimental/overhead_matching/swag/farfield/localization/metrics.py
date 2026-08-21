"""Belief statistics and run-history metrics for bearing-only localization.

Split out of filter.py: everything here only READS a belief or a
health/truth log, and the consumers (plots, evaluations, run viewers,
tests) should not have to import the filter's update machinery to score a
run. Beliefs are duck-typed — anything with `east_m` / `north_m` /
`heading_rad` arrays and a `normalized_weights()` method (filter's
ParticleBelief, or arrays reloaded from a run directory's checkpoints).

The error-series helpers (`position_errors_m`, `map_position_errors_m`,
`heading_errors_deg`) SKIP health keyframes with no matching truth pose
rather than raising: on real exports truth exists only where GPS was valid,
and every downstream consumer had re-implemented exactly that skip around
the old helper's KeyError — the canonical helper now owns the rule. The
returned arrays are therefore aligned to the health records that HAVE
truth; index-based slicing into them assumes full truth coverage (which
holds for synthetic scenarios).
"""

import math

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo


def mean_pose(belief):
    """Weighted mean position and circular-mean heading (rad)."""
    w = belief.normalized_weights()
    east = float(w @ belief.east_m)
    north = float(w @ belief.north_m)
    heading = math.atan2(float(w @ np.sin(belief.heading_rad)),
                         float(w @ np.cos(belief.heading_rad)))
    return east, north, heading


def map_pose(belief, cell_size_m: float = 50.0):
    """Highest-density position and its circular-mean heading.

    The weighted mean is meaningless for a multimodal belief — it sits
    between modes and describes no hypothesis the filter holds (§5.1). This
    bins particles and returns the densest cell's weighted centroid.
    """
    w = belief.normalized_weights()
    col = np.floor(belief.east_m / cell_size_m).astype(np.int64)
    row = np.floor(belief.north_m / cell_size_m).astype(np.int64)
    _, inverse = np.unique(np.stack([row, col]), axis=1, return_inverse=True)
    cell_mass = np.bincount(inverse, weights=w)
    in_cell = inverse == int(np.argmax(cell_mass))
    cell_w = w[in_cell] / w[in_cell].sum()
    heading = math.atan2(
        float(cell_w @ np.sin(belief.heading_rad[in_cell])),
        float(cell_w @ np.cos(belief.heading_rad[in_cell])))
    return (float(cell_w @ belief.east_m[in_cell]),
            float(cell_w @ belief.north_m[in_cell]), heading)


def position_covariance(belief) -> np.ndarray:
    w = belief.normalized_weights()
    d_east = belief.east_m - float(w @ belief.east_m)
    d_north = belief.north_m - float(w @ belief.north_m)
    off_diagonal = float(w @ (d_east * d_north))
    return np.array([[float(w @ (d_east * d_east)), off_diagonal],
                     [off_diagonal, float(w @ (d_north * d_north))]])


def position_std_m(belief) -> float:
    """Isotropic-equivalent position spread (RMS over both axes)."""
    return math.sqrt(max(0.5 * float(np.trace(position_covariance(belief))),
                         0.0))


def heading_std_deg(belief) -> float:
    """Circular standard deviation of heading, in degrees."""
    w = belief.normalized_weights()
    resultant = math.hypot(float(w @ np.sin(belief.heading_rad)),
                           float(w @ np.cos(belief.heading_rad)))
    resultant = min(max(resultant, 1e-15), 1.0 - 1e-15)
    return math.degrees(math.sqrt(-2.0 * math.log(resultant)))


def mass_within_radius(belief, east_m: float, north_m: float,
                       radius_m: float) -> float:
    """Posterior mass within `radius_m` of a point.

    The multimodality-safe accuracy metric: unlike mean error it stays
    meaningful when the belief holds several hypotheses (§5.1, A-9).
    """
    w = belief.normalized_weights()
    within = np.hypot(belief.east_m - east_m,
                      belief.north_m - north_m) <= radius_m
    return float(w[within].sum())


def position_nees(belief, east_m: float, north_m: float) -> float:
    """Normalized estimation error squared against a true position (T-F2).

    2 dof: mean ~2.0 for a consistent filter, 5.99 is the 95% single-sample
    bound. Large values mean the filter is overconfident.
    """
    w = belief.normalized_weights()
    error = np.array([float(w @ belief.east_m) - east_m,
                      float(w @ belief.north_m) - north_m])
    cov = position_covariance(belief) + 1e-9 * np.eye(2)
    return float(error @ np.linalg.solve(cov, error))


def _errors(health: list, truth: list, east_key: str, north_key: str):
    # Keyframes absent from `truth` are skipped, not KeyErrors: real exports
    # carry truth only where GPS was valid (see module docstring).
    truth_by_kf = {t.keyframe_idx: t for t in truth}
    return np.array([
        math.hypot(getattr(r, east_key) - truth_by_kf[r.keyframe_idx].east_m,
                   getattr(r, north_key) - truth_by_kf[r.keyframe_idx].north_m)
        for r in health if r.keyframe_idx in truth_by_kf])


def position_errors_m(health: list, truth: list) -> np.ndarray:
    """Weighted-mean position error per keyframe. Prefer
    `map_position_errors_m` when the belief may be multimodal (§5.1)."""
    return _errors(health, truth, "mean_east_m", "mean_north_m")


def map_position_errors_m(health: list, truth: list) -> np.ndarray:
    """Highest-density-mode position error per keyframe."""
    return _errors(health, truth, "map_east_m", "map_north_m")


def heading_errors_deg(health: list, truth: list) -> np.ndarray:
    truth_by_kf = {t.keyframe_idx: t for t in truth}
    return np.array([
        abs(math.degrees(float(geo.wrap_rad(
            math.radians(r.mean_heading_deg)
            - math.radians(truth_by_kf[r.keyframe_idx].heading_deg)))))
        for r in health if r.keyframe_idx in truth_by_kf])
