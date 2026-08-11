"""Global yaw-offset (+ slow drift) estimation.

The panoramas are rotation-stabilized relative to each other, but (a) the
absolute orientation of the yaw_000 face is unknown and (b) the stabilization
drifts slowly (measured ~0.2 deg/frame on walk_along_river), so the model is

    compass_bearing = sign * camera_bearing + offset0 + drift * frame_idx

with sign from YawOffsetConfig.bearing_sign (a mirror is an isometry that
triangulation consistency cannot detect - verify the sign on the map view).

triangulation_sweep: tracking is offset-invariant (it gates camera-frame
bearing differences), so tracks exist before the offset is known. A 2D grid
over (offset0, drift) scores each candidate by CONSENSUS: the number of
high-parallax tracks whose rays triangulate cleanly (small angular residual,
no negative ranges). Consensus counting is robust to junk tracks that a
summed-residual score lets dominate. At the mirrored offset (+180) the rays
of a high-parallax track diverge - the fit has large residuals or negative
ranges - so the mirror scores near zero consensus.

Scoring uses a fast closed-form triangulation (perpendicular least squares +
angular residuals at that point), not the full Huber refinement; the full
refinement runs once afterwards in the triangulation stage.
"""

import numpy as np

from experimental.overhead_matching.swag.landmark_filtering import (
    artifact_schema as schema,
    bearing_geometry as bg,
    triangulation,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    FilterPipelineConfig,
)

MAX_MEMBERS_PER_TRACK = 20
DRIFT_RANGE_DEG_PER_FRAME = 0.4
COARSE_DRIFT_STEP = 0.05
COARSE_THETA_STEP = 2.0
MAX_NEGATIVE_RANGE_FRACTION = 0.2


class _SweepTrack:
    def __init__(self, positions: np.ndarray, bearings_deg: np.ndarray,
                 frame_idxs: np.ndarray):
        self.positions = positions
        self.bearings_deg = bearings_deg
        self.frame_idxs = frame_idxs


def _sweep_tracks(artifact: schema.RunArtifact,
                  config: FilterPipelineConfig) -> list[_SweepTrack]:
    """High-parallax kept tracks; low-parallax (far) tracks are
    rotation-invariant in this scoring and are skipped."""
    frames = artifact.frames
    obs_by_id = {o.obs_id: o for o in artifact.observations}
    sign = config.yaw_offset.bearing_sign

    candidates = []
    for track in artifact.tracks:
        if track.disposition != schema.KEPT:
            continue
        members = [obs_by_id[oid] for oid in track.obs_ids]
        if len(members) > MAX_MEMBERS_PER_TRACK:
            step = len(members) / MAX_MEMBERS_PER_TRACK
            members = [members[int(i * step)]
                       for i in range(MAX_MEMBERS_PER_TRACK)]
        bearings = np.array(
            [(sign * o.bearing_camera_deg) % 360.0 for o in members])
        parallax = triangulation._pairwise_parallax_deg(bearings)
        if parallax < config.triangulation.min_parallax_deg:
            continue
        positions = np.array(
            [[frames[o.frame_idx].x_m, frames[o.frame_idx].y_m]
             for o in members])
        frame_idxs = np.array([o.frame_idx for o in members], dtype=float)
        candidates.append(
            (len(members), _SweepTrack(positions, bearings, frame_idxs)))

    candidates.sort(key=lambda c: -c[0])
    top_n = config.yaw_offset.sweep_top_n_tracks
    return [t for _, t in candidates[:top_n]]


def _fast_track_fit(positions: np.ndarray,
                    bearings_deg: np.ndarray) -> tuple[float, float] | None:
    """(residual_rms_deg, negative_range_fraction) of the closed-form
    triangulation, or None when the rays are near-parallel."""
    bearings_rad = np.radians(bearings_deg)
    directions = np.stack(
        [np.sin(bearings_rad), np.cos(bearings_rad)], axis=1)
    point = triangulation._closed_form_init(positions, directions)
    if point is None:
        return None
    delta = point[None, :] - positions
    predicted = np.degrees(np.arctan2(delta[:, 0], delta[:, 1]))
    residual = np.asarray(bg.wrap_deg(predicted - bearings_deg))
    ranges = (delta * directions).sum(axis=1)
    return (float(np.sqrt(np.mean(np.square(residual)))),
            float((ranges < 0).mean()))


def _consensus_score(tracks: list[_SweepTrack], offset_deg: float,
                     drift_deg_per_frame: float,
                     sigma_deg: float) -> tuple[int, float]:
    """(n_clean_tracks, summed residual of clean tracks)."""
    n_clean = 0
    residual_sum = 0.0
    for track in tracks:
        rotated = (track.bearings_deg + offset_deg
                   + drift_deg_per_frame * track.frame_idxs) % 360.0
        fit = _fast_track_fit(track.positions, rotated)
        if fit is None:
            continue
        rms, negative_fraction = fit
        if (rms < 2.0 * sigma_deg
                and negative_fraction <= MAX_NEGATIVE_RANGE_FRACTION):
            n_clean += 1
            residual_sum += rms
    return n_clean, residual_sum


def _grid_search(tracks: list[_SweepTrack], thetas: np.ndarray,
                 drifts: np.ndarray, sigma_deg: float):
    best = None
    for drift in drifts:
        for theta in thetas:
            n_clean, residual_sum = _consensus_score(
                tracks, theta, drift, sigma_deg)
            key = (-n_clean, residual_sum)
            if best is None or key < best[0]:
                best = (key, theta % 360.0, drift)
    return best


def estimate_yaw_offset(
        artifact: schema.RunArtifact, config: FilterPipelineConfig,
        override_deg: float | None
) -> tuple[float, float, str, dict[str, float]]:
    """Returns (offset0_deg, drift_deg_per_frame, method, details)."""
    if override_deg is not None:
        return (override_deg % 360.0,
                config.yaw_offset.fixed_drift_deg_per_frame, "fixed", {})
    if config.yaw_offset.method == "fixed":
        return (config.yaw_offset.fixed_offset_deg % 360.0,
                config.yaw_offset.fixed_drift_deg_per_frame, "fixed", {})
    if config.yaw_offset.method != "triangulation_sweep":
        raise ValueError(
            f"Unknown yaw offset method: {config.yaw_offset.method}")

    tracks = _sweep_tracks(artifact, config)
    if not tracks:
        print("yaw sweep: no parallax-informative tracks; falling back to "
              "fixed offset")
        return (config.yaw_offset.fixed_offset_deg % 360.0, 0.0, "fixed", {})

    sigma = config.triangulation.bearing_sigma_deg
    coarse = _grid_search(
        tracks,
        thetas=np.arange(0.0, 360.0, COARSE_THETA_STEP),
        drifts=np.arange(-DRIFT_RANGE_DEG_PER_FRAME,
                         DRIFT_RANGE_DEG_PER_FRAME + 1e-9, COARSE_DRIFT_STEP),
        sigma_deg=sigma)
    (coarse_key, coarse_theta, coarse_drift) = coarse

    fine = _grid_search(
        tracks,
        thetas=np.arange(coarse_theta - COARSE_THETA_STEP,
                         coarse_theta + COARSE_THETA_STEP + 1e-9, 0.25),
        drifts=np.arange(coarse_drift - COARSE_DRIFT_STEP,
                         coarse_drift + COARSE_DRIFT_STEP + 1e-9, 0.01),
        sigma_deg=sigma)
    (fine_key, offset, drift) = fine

    mirror_clean, _ = _consensus_score(
        tracks, (offset + 180.0) % 360.0, drift, sigma)
    consensus = float(-fine_key[0])
    details = {
        "consensus": consensus,
        "consensus_fraction": consensus / len(tracks),
        "consensus_residual_sum": float(fine_key[1]),
        "n_tracks_used": float(len(tracks)),
        "mirror_consensus_at_plus_180": float(mirror_clean),
        "drift_deg_per_frame": float(drift),
    }
    if consensus < 0.5 * len(tracks):
        print("=" * 70)
        print(
            f"YAW SWEEP IS WEAKLY CONSTRAINED: only {consensus:.0f} of "
            f"{len(tracks)} tracks triangulate cleanly at the optimum.\n"
            "The offset/drift below may be unreliable - most high-parallax\n"
            "tracks on this trajectory are near-field junk, and far tracks\n"
            "carry no rotation information. Prefer an external calibration\n"
            "(named-landmark anchoring against OSM, compass/IMU, or MASt3R\n"
            "relative rotations) via yaw_offset.method=fixed with\n"
            "fixed_offset_deg / fixed_drift_deg_per_frame.")
        print("=" * 70)
    return float(offset), float(drift), "triangulation_sweep", details
