"""Belief statistics and run-history metrics for bearing-only localization.

Split out of filter.py: everything here only READS a belief or a
health/truth log, and the consumers (plots, evaluations, run viewers,
tests) should not have to import the filter's update machinery to score a
run. Beliefs are duck-typed — anything with `east_m` / `north_m` /
`heading_rad` arrays and a `normalized_weights()` method (filter's
ParticleBelief, or arrays reloaded from a run directory's checkpoints).

The error-series helpers (`position_errors_m`, `map_position_errors_m`,
`heading_errors_deg`) skip health keyframes with no matching truth pose.
Returned arrays are aligned to health records that have truth; their
``keyframe_idx`` attribute records that alignment. Index-based slicing assumes
full truth coverage, as provided by synthetic scenarios.
"""

import math

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.localization import structs


POSITION_MASS_METRIC_ID = (
    "posterior_position_probability_mass_within_true_position_radius")
POSITION_MASS_METRIC_VERSION = "1"
TIME_NORMALIZED_POSITION_MASS_METRIC_ID = (
    "time_normalized_posterior_position_probability_mass_within_"
    "true_position_radius")
TIME_NORMALIZED_POSITION_MASS_METRIC_VERSION = "1"
POSITION_MASS_SUMMARY_SCHEMA = "farfield_position_mass_summary/v1"
POSITION_MASS_SUMMARY_NAME = "metrics.json"
POSITION_MASS_TIME_NORMALIZATION = (
    "trapezoidal_area_divided_by_keyframe_span")
PRIMARY_POSITION_MASS_RADIUS_M = 500.0
DEFAULT_POSITION_MASS_RADII_M = (100.0, PRIMARY_POSITION_MASS_RADIUS_M)


class KeyedMetricSeries(np.ndarray):
    """A numeric series that retains the keyframe IDs of skipped-truth data."""

    def __new__(cls, values, keyframe_idx):
        result = np.asarray(values, dtype=np.float64).view(cls)
        result.keyframe_idx = np.asarray(keyframe_idx, dtype=np.int64)
        return result

    def __array_finalize__(self, source):
        if source is not None:
            self.keyframe_idx = getattr(
                source, "keyframe_idx", np.zeros(0, dtype=np.int64))


def position_mass_metric_config(
        radii_m=DEFAULT_POSITION_MASS_RADII_M
) -> structs.PositionMassMetricConfig:
    """Validate and canonicalize the resolved primary-metric configuration."""
    radii = [float(value) for value in radii_m]
    if (not radii or any(not math.isfinite(value) or value <= 0.0
                         for value in radii)
            or radii != sorted(set(radii))):
        raise ValueError(
            "position-mass radii must be finite, positive, unique, and sorted")
    if PRIMARY_POSITION_MASS_RADIUS_M not in radii:
        raise ValueError(
            "position-mass radii must include the primary 500 m radius")
    return structs.PositionMassMetricConfig(
        metric_id=POSITION_MASS_METRIC_ID,
        metric_version=POSITION_MASS_METRIC_VERSION,
        radii_m=radii)


def position_mass_metric_key(config: structs.PositionMassMetricConfig,
                             radius_m: float) -> str:
    """Stable identity for one configured radius within the metric family."""
    radius = float(radius_m)
    if radius not in config.radii_m:
        raise ValueError(f"radius {radius:g} m is not in the metric config")
    return (f"{config.metric_id}@{config.metric_version}:"
            f"radius_m={radius:g}")


def position_mass_radii_in_priority_order(
        config: structs.PositionMassMetricConfig) -> list[float]:
    """Put the 500 m headline radius first without changing stored config."""
    primary = PRIMARY_POSITION_MASS_RADIUS_M
    return ([primary] if primary in config.radii_m else []) + [
        radius for radius in config.radii_m if radius != primary]


def time_normalized_position_mass(
        health: list, config: structs.PositionMassMetricConfig,
        radius_m: float) -> float:
    """Normalized area under truth-centered posterior mass over keyframe time.

    Each health value was recorded against the true position at that keyframe
    by the evaluation-only observer in ``runner.py``.  Trapezoidal integration
    followed by division by the keyframe span makes runs of different duration
    comparable while retaining the whole trajectory.  The result is in [0, 1]
    and higher is better.  A one-keyframe run has the value at that keyframe.
    """
    key = position_mass_metric_key(config, radius_m)
    if not health:
        raise ValueError("position-mass summary requires at least one keyframe")
    keyframes = np.asarray(
        [record.keyframe_idx for record in health], dtype=np.float64)
    if keyframes.size > 1 and np.any(np.diff(keyframes) <= 0.0):
        raise ValueError(
            "position-mass summary keyframes must be strictly increasing")
    try:
        values = np.asarray([
            record.position_probability_mass[key] for record in health
        ], dtype=np.float64)
    except KeyError as error:
        raise ValueError(
            f"position-mass summary is missing configured metric {key!r}"
        ) from error
    if (not np.isfinite(values).all()
            or np.any(values < 0.0) or np.any(values > 1.0)):
        raise ValueError(
            "position-mass summary values must be finite probabilities")
    if values.size == 1:
        return float(values[0])
    intervals = np.diff(keyframes)
    area = float(np.sum(0.5 * (values[:-1] + values[1:]) * intervals))
    normalized = area / float(keyframes[-1] - keyframes[0])
    # The source values are probabilities, but protect the serialized contract
    # from floating-point accumulation a few ulps outside the closed interval.
    return min(1.0, max(0.0, normalized))


def position_mass_summary(
        health: list, config: structs.PositionMassMetricConfig) -> dict:
    """Machine-readable aggregate of the every-keyframe primary metric."""
    radii = {
        f"{float(radius_m):g}": {
            "radius_m": float(radius_m),
            "time_normalized_mass": time_normalized_position_mass(
                health, config, radius_m),
        }
        for radius_m in config.radii_m
    }
    primary = PRIMARY_POSITION_MASS_RADIUS_M
    return {
        "schema": POSITION_MASS_SUMMARY_SCHEMA,
        "metric_id": TIME_NORMALIZED_POSITION_MASS_METRIC_ID,
        "metric_version": TIME_NORMALIZED_POSITION_MASS_METRIC_VERSION,
        "source_metric_id": config.metric_id,
        "source_metric_version": config.metric_version,
        "reference_position": "truth",
        "normalization": POSITION_MASS_TIME_NORMALIZATION,
        "higher_is_better": True,
        "primary_radius_m": primary if primary in config.radii_m else None,
        "n_keyframes": len(health),
        "keyframe_span": (health[-1].keyframe_idx - health[0].keyframe_idx
                          if len(health) > 1 else 0),
        "radii": radii,
    }


def describe_position_mass_summary(summary: dict, run_kind: str) -> str:
    """Human-readable primary-first rendering of the aggregate contract."""
    primary_run = run_kind == "evaluation"
    classification = ("PRIMARY LOCALIZATION METRIC" if primary_run else
                      f"LOCALIZATION METRIC — {run_kind.replace('_', ' ').upper()}")
    lines = [f"--- {classification} ---"]
    primary_radius = summary["primary_radius_m"]
    ordered_radii = ([primary_radius] if primary_radius is not None else [])
    ordered_radii += [
        value["radius_m"] for value in summary["radii"].values()
        if value["radius_m"] != primary_radius
    ]
    for radius_m in ordered_radii:
        value = summary["radii"][f"{float(radius_m):g}"][
            "time_normalized_mass"]
        if radius_m == primary_radius:
            label = "PRIMARY" if primary_run else "headline (diagnostic)"
        else:
            label = "secondary"
        lines.append(
            f"{label}: normalized posterior mass within {radius_m:g} m of "
            f"the true position over time = {value:.4f} "
            f"({100.0 * value:.2f}%)")
    lines.append(
        "higher is better; truth is evaluation-only and never initializes "
        "or updates the filter")
    return "\n".join(lines)


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
    """Posterior mass within `radius_m` of a reference position.

    The run observer always supplies the true position as the reference.  This
    is an evaluation-only, multimodality-safe accuracy metric: unlike mean
    error it stays meaningful when the belief holds several hypotheses
    (§5.1, A-9).
    """
    w = belief.normalized_weights()
    within = np.hypot(belief.east_m - east_m,
                      belief.north_m - north_m) <= radius_m
    # Floating summation can land a few ulps outside the probability domain
    # even though normalized weights mathematically sum to one.
    return min(1.0, max(0.0, float(w[within].sum())))


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
    return KeyedMetricSeries([
        math.hypot(getattr(r, east_key) - truth_by_kf[r.keyframe_idx].east_m,
                   getattr(r, north_key) - truth_by_kf[r.keyframe_idx].north_m)
        for r in health if r.keyframe_idx in truth_by_kf],
        [r.keyframe_idx for r in health if r.keyframe_idx in truth_by_kf])


def position_errors_m(health: list, truth: list) -> np.ndarray:
    """Weighted-mean position error per keyframe. Prefer
    `map_position_errors_m` when the belief may be multimodal (§5.1)."""
    return _errors(health, truth, "mean_east_m", "mean_north_m")


def map_position_errors_m(health: list, truth: list) -> np.ndarray:
    """Highest-density-mode position error per keyframe."""
    return _errors(health, truth, "map_east_m", "map_north_m")


def heading_errors_deg(health: list, truth: list) -> np.ndarray:
    truth_by_kf = {t.keyframe_idx: t for t in truth}
    return KeyedMetricSeries([
        abs(math.degrees(float(geo.wrap_rad(
            math.radians(r.mean_heading_deg)
            - math.radians(
                truth_by_kf[r.keyframe_idx].course_world_cw_deg)))))
        for r in health if r.keyframe_idx in truth_by_kf],
        [r.keyframe_idx for r in health if r.keyframe_idx in truth_by_kf])


def bearing_residual_diagnostics(catalog, measurements: list, health: list) \
        -> list[structs.BearingResidualDiagnostic]:
    """Posterior-predictive residuals, never association-correctness scores.

    Each mode-specific association is evaluated at that exact mode's pose.
    null_dominated is true when null probability is at least as large as the
    most probable named landmark; those records remain visible but are
    stratified from landmark-dominated diagnostics.
    """
    measurement_by_key = {
        (measurement.anchor_keyframe_idx, measurement.tracklet_id): measurement
        for measurement in measurements
    }
    result = []
    for record in health:
        modes = {mode.mode_id: mode for mode in record.modes}
        for association in record.associations:
            key = (record.keyframe_idx, association.tracklet_id)
            measurement = measurement_by_key.get(key)
            if measurement is None:
                raise ValueError(
                    "association has no matching consumed measurement: "
                    f"keyframe={key[0]}, tracklet={key[1]!r}")
            if association.mode_id is None:
                east_m = record.mean_east_m
                north_m = record.mean_north_m
                heading_deg = record.mean_heading_deg
            else:
                mode = modes.get(association.mode_id)
                if mode is None:
                    raise ValueError(
                        "mode-specific association has no corresponding mode "
                        f"pose: keyframe={key[0]}, mode={association.mode_id}")
                east_m = mode.mean_east_m
                north_m = mode.mean_north_m
                heading_deg = mode.mean_heading_deg

            landmark_id = None
            probability = None
            signed_residual_deg = None
            if association.responsibilities:
                landmark_id = max(
                    association.responsibilities,
                    key=association.responsibilities.get)
                probability = float(
                    association.responsibilities[landmark_id])
                index = catalog.index_of(landmark_id)
                predicted_world = geo.compass_bearing_rad(
                    catalog.east_m[index] - east_m,
                    catalog.north_m[index] - north_m)
                measured_world = math.radians(
                    heading_deg + measurement.bearing_forward_cw_deg)
                signed_residual_deg = math.degrees(float(
                    geo.wrap_rad(measured_world - predicted_world)))
            null_dominated = (
                probability is None or association.null_share >= probability)
            result.append(structs.BearingResidualDiagnostic(
                keyframe_idx=record.keyframe_idx,
                tracklet_id=association.tracklet_id,
                mode_id=association.mode_id,
                pose_east_m=float(east_m),
                pose_north_m=float(north_m),
                pose_heading_cw_deg=float(heading_deg),
                null_share=float(association.null_share),
                null_dominated=null_dominated,
                landmark_id=landmark_id,
                association_probability=probability,
                signed_residual_deg=signed_residual_deg))
    return result
