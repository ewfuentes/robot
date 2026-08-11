"""Robust bearing-only triangulation of tracks in the local ENU plane.

Per kept track:
1. closed-form init minimizing squared perpendicular ray distances,
2. Huber-robust least squares on angular residuals,
3. negative-range check: each observation ray is the half-line from the
   camera along the measured bearing; a fit at negative range along that ray
   (i.e. in the bearing+180 direction) contradicts the measurement. With 360
   panoramas nothing is "behind the camera" - this is about the ray's sign,
   and it is also what disambiguates the yaw-offset mirror in the sweep.
4. covariance from the normalized-residual Jacobian,
5. observability classification: near / far / degenerate.

"far" tracks (self-consistent bearings, too little parallax or too much range
to pin a point) are the crossview-valuable bearing-only landmarks.
"""

import math

import numpy as np
from scipy.optimize import least_squares

from experimental.overhead_matching.swag.landmark_filtering import (
    artifact_schema as schema,
    bearing_geometry as bg,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    TriangulationConfig,
)


def _pairwise_parallax_deg(bearings_deg: np.ndarray) -> float:
    diffs = np.abs(bg.wrap_deg(bearings_deg[:, None] - bearings_deg[None, :]))
    return float(diffs.max()) if diffs.size else 0.0


def _bearing_spread_deg(bearings_deg: np.ndarray) -> float:
    mean = bg.circular_mean_deg(bearings_deg)
    return float(np.sqrt(np.mean(
        np.square(bg.wrap_deg(bearings_deg - mean)))))


def _closed_form_init(positions: np.ndarray,
                      directions: np.ndarray) -> np.ndarray | None:
    normals = np.stack([-directions[:, 1], directions[:, 0]], axis=1)
    a_mat = np.zeros((2, 2))
    b_vec = np.zeros(2)
    for p, n in zip(positions, normals):
        outer = np.outer(n, n)
        a_mat += outer
        b_vec += outer @ p
    if abs(np.linalg.det(a_mat)) < 1e-9:
        return None
    return np.linalg.solve(a_mat, b_vec)


def triangulate_rays(positions: np.ndarray, bearings_deg: np.ndarray,
                     config: TriangulationConfig) -> schema.TriangulationResult:
    """Triangulate one track from camera ENU positions + global bearings."""
    n_obs = len(positions)
    parallax = _pairwise_parallax_deg(bearings_deg)
    spread = _bearing_spread_deg(bearings_deg)

    if n_obs < 2:
        return schema.TriangulationResult(
            solved=False, observability="degenerate",
            degenerate_reason="too_few_obs", parallax_deg=parallax)

    if parallax < config.min_parallax_deg:
        # No unique intersection. Self-consistent constant bearing = a far,
        # bearing-only landmark; inconsistent = junk.
        if spread <= 2.0 * config.bearing_sigma_deg:
            return schema.TriangulationResult(
                solved=False, observability="far",
                residual_rms_deg=spread, parallax_deg=parallax,
                n_inliers=n_obs)
        return schema.TriangulationResult(
            solved=False, observability="degenerate",
            degenerate_reason="low_parallax", residual_rms_deg=spread,
            parallax_deg=parallax)

    bearings_rad = np.radians(bearings_deg)
    directions = np.stack(
        [np.sin(bearings_rad), np.cos(bearings_rad)], axis=1)
    init = _closed_form_init(positions, directions)
    if init is None:
        return schema.TriangulationResult(
            solved=False, observability="degenerate",
            degenerate_reason="low_parallax", parallax_deg=parallax)

    def residuals(x):
        delta = x[None, :] - positions
        predicted = np.degrees(np.arctan2(delta[:, 0], delta[:, 1]))
        return np.asarray(
            bg.wrap_deg(predicted - bearings_deg)) / config.bearing_sigma_deg

    fit = least_squares(residuals, init, loss="huber",
                        f_scale=config.huber_delta)
    if not fit.success:
        return schema.TriangulationResult(
            solved=False, observability="degenerate",
            degenerate_reason="no_convergence", parallax_deg=parallax)

    point = fit.x
    delta = point[None, :] - positions
    ranges = (delta * directions).sum(axis=1)
    n_negative = int((ranges < 0).sum())
    if n_negative > n_obs // 2:
        return schema.TriangulationResult(
            solved=False, observability="degenerate",
            degenerate_reason="negative_range", parallax_deg=parallax,
            n_outliers=n_negative)

    residual_deg = np.asarray(residuals(point)) * config.bearing_sigma_deg
    inlier = np.abs(residual_deg) <= (config.huber_delta
                                      * config.bearing_sigma_deg * 2.0)
    residual_rms = float(np.sqrt(np.mean(np.square(residual_deg))))
    mean_range = float(np.abs(ranges).mean())

    # Covariance of the fit: residuals are normalized by sigma, so
    # cov = (J^T J)^-1 directly, in m^2.
    jtj = fit.jac.T @ fit.jac
    try:
        cov = np.linalg.inv(jtj)
        eigvals = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
        sigma_minor, sigma_major = (math.sqrt(eigvals[0]),
                                    math.sqrt(eigvals[1]))
    except np.linalg.LinAlgError:
        cov, sigma_major, sigma_minor = None, None, None

    if residual_rms > 3.0 * config.bearing_sigma_deg:
        # The point model doesn't explain the bearings (e.g. a mirrored /
        # mis-associated track): don't trust the position.
        observability = "degenerate"
        reason = "high_residual"
    elif spread <= 1.5 * residual_rms:
        # The point fit is no better than a constant-bearing model, so the
        # apparent parallax is just noise (e.g. a landmark dead ahead along
        # the walking direction).
        if spread <= 2.0 * config.bearing_sigma_deg:
            observability = "far"
            reason = ""
        else:
            observability = "degenerate"
            reason = "low_parallax"
    elif mean_range > config.far_range_m:
        observability = "far"
        reason = ""
    elif (sigma_major is not None
          and sigma_major > config.max_sigma_major_m):
        observability = "degenerate"
        reason = "large_uncertainty"
    else:
        observability = "near"
        reason = ""

    return schema.TriangulationResult(
        solved=True,
        observability=observability,
        degenerate_reason=reason,
        x_m=float(point[0]),
        y_m=float(point[1]),
        mean_range_m=mean_range,
        residual_rms_deg=residual_rms,
        n_inliers=int(inlier.sum()),
        n_outliers=int((~inlier).sum()) + n_negative,
        cov_enu=cov.tolist() if cov is not None else None,
        sigma_major_m=sigma_major,
        sigma_minor_m=sigma_minor,
        parallax_deg=parallax,
    )


def run_triangulation(artifact: schema.RunArtifact,
                      config: TriangulationConfig) -> None:
    frames = artifact.frames
    obs_by_id = {o.obs_id: o for o in artifact.observations}
    stats_by_class: dict[str, int] = {}

    for track in artifact.tracks:
        if track.disposition != schema.KEPT:
            continue
        members = [obs_by_id[oid] for oid in track.obs_ids]
        positions = np.array(
            [[frames[o.frame_idx].x_m, frames[o.frame_idx].y_m]
             for o in members])
        bearings = np.array([o.bearing_global_deg for o in members])
        result = triangulate_rays(positions, bearings, config)
        if result.solved:
            result.lat, result.lon = bg.latlon_from_enu(
                result.x_m, result.y_m, artifact.anchor_lat,
                artifact.anchor_lon)
        track.triangulation = result
        stats_by_class[result.observability] = (
            stats_by_class.get(result.observability, 0) + 1)

    artifact.stats.tracks_by_observability = stats_by_class
