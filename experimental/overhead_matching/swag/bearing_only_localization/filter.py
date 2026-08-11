"""SE(2) particle filter core for bearing-only localization.

Implements docs/localization-design-doc.md §5 on the synthetic-harness scale:
association-marginalized von Mises bearing likelihood with a mandatory null
hypothesis (§5.3), sparse information-epoch measurement events, systematic
resampling, and world-frame odometry composition (§5.2).

Pure numpy, CPU-only, one explicit np.random.Generator: the filter is a
deterministic function of (config, seed, ordered event log) — the replay
contract of §3.8. Positions are region-frame ENU metres; heading is radians
clockwise from north; measurement bearings are body-frame (§4).

Course over ground is consumed EXACTLY ONCE, as increments (d_k = c_k -
c_{k-1}) rotating the heading state — never also as an absolute heading
measurement. Using it both ways lets the propagation and the update confirm
the same noise sample, so the posterior sharpens without ever averaging the
noise down (see the §8.4 "course/heading evidence reuse" entry). Differencing
has a second payoff: a constant COG-vs-heading offset (leeway/crab, mount
misalignment) cancels, so no bias state is needed to survive it. The price is
that the absolute course reading carries no information here — absolute
heading is anchored solely by landmark bearings, which is what §5.2 intends.

Reported belief statistics and checkpoints are taken from the WEIGHTED
posterior, before resampling and roughening. Resampling is a representational
step and roughening deliberately adds variance the posterior does not have;
reporting after either would misstate the belief.
"""

import dataclasses
import hashlib
import math

import numpy as np
from scipy import special

from experimental.overhead_matching.swag.bearing_only_localization import (
    catalog as catalog_mod,
    geodesy,
    structs,
)

# Cap on any single measurement's concentration. §6 makes the matcher's LLRs
# clipped so semantics cannot steamroll geometry; kappa is the same hazard on
# the geometric side — 1e6 is ~0.06 deg, far below any real bearing accuracy.
MAX_KAPPA = 1.0e6
# Candidate-axis block size for the measurement update. Bounds the (N, M)
# temporaries; see the memory table in the Milestone 0 audit (A-8).
CANDIDATE_BLOCK = 256


@dataclasses.dataclass
class ParticleBelief:
    east_m: np.ndarray
    north_m: np.ndarray
    heading_rad: np.ndarray  # clockwise from north
    log_weight: np.ndarray

    @property
    def n(self) -> int:
        return self.east_m.shape[0]

    def copy(self) -> "ParticleBelief":
        return ParticleBelief(self.east_m.copy(), self.north_m.copy(),
                              self.heading_rad.copy(), self.log_weight.copy())

    def normalized_weights(self) -> np.ndarray:
        return np.exp(self.log_weight - special.logsumexp(self.log_weight))


@dataclasses.dataclass
class FilterHistory:
    health: list  # list[structs.HealthRecord]
    checkpoints: dict  # keyframe_idx -> ParticleBelief (weighted posterior)
    particle_history_sha256: str
    final_belief: ParticleBelief  # weighted posterior at the last keyframe


def von_mises_logpdf(delta_rad, kappa) -> np.ndarray:
    """Log density of a von Mises at angular residual `delta_rad` (mean 0).

    log I0(kappa) = log(i0e(kappa)) + kappa, stable for large kappa. `kappa`
    may be a scalar or broadcast against `delta_rad`.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    log_norm = math.log(2.0 * math.pi) + np.log(special.i0e(kappa)) + kappa
    return kappa * np.cos(delta_rad) - log_norm


def init_belief(config: structs.FilterConfig,
                rng: np.random.Generator) -> ParticleBelief:
    n = config.n_particles
    init = config.init
    if isinstance(init, structs.GaussianInit):
        east = rng.normal(init.mean_east_m, init.sigma_m, size=n)
        north = rng.normal(init.mean_north_m, init.sigma_m, size=n)
    elif isinstance(init, structs.UniformBoxInit):
        east = rng.uniform(init.east_min_m, init.east_max_m, size=n)
        north = rng.uniform(init.north_min_m, init.north_max_m, size=n)
    else:
        raise ValueError(f"Unknown init config: {init}")
    heading = rng.uniform(-np.pi, np.pi, size=n)
    return ParticleBelief(east, north, heading, np.zeros(n))


def course_delta_noise_rad(course_sigma_deg, heading_rw_rad: float) -> float:
    """Per-step heading proposal noise when rotating by a differenced course.

    d_k = c_k - c_{k-1} carries sqrt(2)*sigma_course of measurement noise on
    top of the true yaw change, so the proposal must cover both that and the
    unmodeled yaw drift. Consecutive d_k are anti-correlated (the errors
    telescope), so treating them as independent is conservative — the safe
    direction, and bearings pull the spread back in.
    """
    course_sigma_rad = math.radians(course_sigma_deg or 0.0)
    return math.sqrt(2.0 * course_sigma_rad ** 2 + heading_rw_rad ** 2)


def motion_update(belief: ParticleBelief, delta: structs.OdometryDelta,
                  course_delta_rad: float, heading_sigma_rad: float,
                  rng: np.random.Generator) -> None:
    """World-frame translation + heading rotation by differenced course.

    Position propagation never routes through heading (§5.2): the deltas are
    already world-frame.
    """
    n = belief.n
    belief.east_m += delta.dx_m + rng.normal(0.0, delta.sigma_m, size=n)
    belief.north_m += delta.dy_m + rng.normal(0.0, delta.sigma_m, size=n)
    belief.heading_rad = geodesy.wrap_rad(
        belief.heading_rad + course_delta_rad
        + rng.normal(0.0, heading_sigma_rad, size=n))


def _clipped_log_lr(table: structs.CompatibilityTable,
                    catalog: catalog_mod.LandmarkCatalog) -> np.ndarray:
    """Catalog-aligned, clipped LLR vector for one tracklet (§6)."""
    log_lr = np.full(catalog.n, table.default_log_lr)
    for entry in table.entries:
        try:
            log_lr[catalog.index_of(entry.landmark_id)] = entry.log_lr
        except ValueError:
            # A matcher may score landmarks outside this region's catalog.
            continue
    return np.clip(log_lr, table.clip_lo, table.clip_hi)


def measurement_update(
        belief: ParticleBelief,
        meas: structs.TrackletMeasurement,
        table: structs.CompatibilityTable,
        catalog: catalog_mod.LandmarkCatalog,
        pi0: float) -> structs.AssociationPosterior:
    """Association-marginalized bearing update (design doc §5.3).

    p(z|x) = pi0/(2 pi) + (1-pi0) * sum_j w_j * LR_j * vM(delta_j; kappa_eff)

    The candidate axis is processed in blocks so the (n_particles, n_cand)
    temporaries stay bounded; two passes are needed because responsibilities
    are normalized by the total likelihood.
    """
    if not 0.0 < pi0 < 1.0:
        raise ValueError(f"pi0 must be in (0, 1), got {pi0}")
    if not math.isfinite(meas.kappa) or meas.kappa <= 0.0:
        raise ValueError(f"kappa must be positive and finite, got "
                         f"{meas.kappa}")
    kappa_z = min(float(meas.kappa), MAX_KAPPA)
    log_lr = _clipped_log_lr(table, catalog)
    observed_rad = math.radians(meas.bearing_body_deg)
    log_null = math.log(pi0) - math.log(2.0 * math.pi)
    log_mix = math.log1p(-pi0)

    def block_log_terms(sl):
        """log[(1-pi0) * w_j * LR_j * vM(...)] for one candidate block."""
        bearing_world, range_m = catalog.bearings_from(
            belief.east_m, belief.north_m, sl)
        delta = geodesy.wrap_rad(
            bearing_world - belief.heading_rad[:, None] - observed_rad)
        kappa_eff = catalog.kappa_eff(kappa_z, range_m, sl)
        return (log_mix + catalog.log_prior[sl][None, :] + log_lr[sl][None, :]
                + von_mises_logpdf(delta, kappa_eff))

    blocks = [slice(start, min(start + CANDIDATE_BLOCK, catalog.n))
              for start in range(0, catalog.n, CANDIDATE_BLOCK)]

    # Pass 1: total log-likelihood per particle.
    per_block = np.empty((belief.n, len(blocks) + 1))
    per_block[:, 0] = log_null
    for i, sl in enumerate(blocks):
        per_block[:, i + 1] = special.logsumexp(block_log_terms(sl), axis=1)
    log_lik = special.logsumexp(per_block, axis=1)
    belief.log_weight += log_lik

    # Pass 2: responsibilities averaged under the updated particle weights
    # (§5.4). Per-mode reporting arrives with the mode tracker (§10.5).
    weights = belief.normalized_weights()
    responsibilities = {}
    for sl in blocks:
        avg = weights @ np.exp(block_log_terms(sl) - log_lik[:, None])
        for offset, value in enumerate(avg):
            responsibilities[catalog.landmark_ids[sl.start + offset]] = float(
                value)
    null_share = float(weights @ np.exp(log_null - log_lik))

    return structs.AssociationPosterior(
        tracklet_id=meas.tracklet_id,
        anchor_keyframe_idx=meas.anchor_keyframe_idx,
        null_share=null_share,
        responsibilities=responsibilities)


def ess(log_weight: np.ndarray) -> float:
    w = np.exp(log_weight - special.logsumexp(log_weight))
    return 1.0 / float(np.sum(np.square(w)))


def systematic_resample(belief: ParticleBelief, rng: np.random.Generator,
                        regularization: float = 1.0,
                        position_roughening_m: float = 0.0,
                        heading_roughening_rad: float = 0.0) -> None:
    """Low-variance (systematic) resampling with kernel regularization (T-U6).

    Plain resampling replaces the posterior with a set of Dirac atoms drawn
    only from locations already present, so repeated resampling collapses
    diversity and the filter grows confident about a spread it no longer
    represents. Measured on the harbor_loop scenario, final-state NEES ran 67
    at 4k particles and 12 at 20k against an ideal of 2.0 — the signature of
    impoverishment, not of a wrong model.

    So resample from a kernel-smoothed posterior instead (regularized
    particle filter, Musso/Oudjane/Le Gland): jitter each dimension by
    `regularization * sigma_dim * n^(-1/(d+4))`, d=2. The bandwidth is tied
    to the posterior's own spread, so it vanishes as the belief sharpens
    rather than injecting a fixed, arbitrary variance. `*_roughening_*` add
    a floor on top, for beliefs so collapsed that a proportional bandwidth
    cannot recover them (brute-force global init).
    """
    n = belief.n
    weights = belief.normalized_weights()
    # Bandwidths come from the pre-resample posterior — the distribution
    # actually being smoothed.
    bandwidth = regularization * n ** (-1.0 / 6.0) if regularization > 0 else 0.0
    east_scale = north_scale = heading_scale = 0.0
    if bandwidth > 0.0:
        cov = position_covariance(belief)
        east_scale = bandwidth * math.sqrt(max(cov[0, 0], 0.0))
        north_scale = bandwidth * math.sqrt(max(cov[1, 1], 0.0))
        heading_scale = bandwidth * math.radians(heading_std_deg(belief))

    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # guard against fp sum < 1
    positions = (rng.random() + np.arange(n)) / n
    idx = np.searchsorted(cumulative, positions, side="left")
    belief.east_m = belief.east_m[idx]
    belief.north_m = belief.north_m[idx]
    belief.heading_rad = belief.heading_rad[idx]

    east_sigma = math.hypot(east_scale, position_roughening_m)
    north_sigma = math.hypot(north_scale, position_roughening_m)
    heading_sigma = math.hypot(heading_scale, heading_roughening_rad)
    if east_sigma > 0.0:
        belief.east_m += rng.normal(0.0, east_sigma, size=n)
    if north_sigma > 0.0:
        belief.north_m += rng.normal(0.0, north_sigma, size=n)
    if heading_sigma > 0.0:
        belief.heading_rad = geodesy.wrap_rad(
            belief.heading_rad + rng.normal(0.0, heading_sigma, size=n))
    belief.log_weight = np.zeros(n)


def mean_pose(belief: ParticleBelief):
    """Weighted mean position and circular-mean heading (rad)."""
    w = belief.normalized_weights()
    east = float(w @ belief.east_m)
    north = float(w @ belief.north_m)
    heading = math.atan2(float(w @ np.sin(belief.heading_rad)),
                         float(w @ np.cos(belief.heading_rad)))
    return east, north, heading


def map_pose(belief: ParticleBelief, cell_size_m: float = 50.0):
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


def position_covariance(belief: ParticleBelief) -> np.ndarray:
    w = belief.normalized_weights()
    d_east = belief.east_m - float(w @ belief.east_m)
    d_north = belief.north_m - float(w @ belief.north_m)
    off_diagonal = float(w @ (d_east * d_north))
    return np.array([[float(w @ (d_east * d_east)), off_diagonal],
                     [off_diagonal, float(w @ (d_north * d_north))]])


def position_std_m(belief: ParticleBelief) -> float:
    """Isotropic-equivalent position spread (RMS over both axes)."""
    return math.sqrt(max(0.5 * float(np.trace(position_covariance(belief))),
                         0.0))


def heading_std_deg(belief: ParticleBelief) -> float:
    """Circular standard deviation of heading, in degrees."""
    w = belief.normalized_weights()
    resultant = math.hypot(float(w @ np.sin(belief.heading_rad)),
                           float(w @ np.cos(belief.heading_rad)))
    resultant = min(max(resultant, 1e-15), 1.0 - 1e-15)
    return math.degrees(math.sqrt(-2.0 * math.log(resultant)))


def mass_within_radius(belief: ParticleBelief, east_m: float, north_m: float,
                       radius_m: float) -> float:
    """Posterior mass within `radius_m` of a point.

    The multimodality-safe accuracy metric: unlike mean error it stays
    meaningful when the belief holds several hypotheses (§5.1, A-9).
    """
    w = belief.normalized_weights()
    within = np.hypot(belief.east_m - east_m,
                      belief.north_m - north_m) <= radius_m
    return float(w[within].sum())


def position_nees(belief: ParticleBelief, east_m: float,
                  north_m: float) -> float:
    """Normalized estimation error squared against a true position (T-F2).

    2 dof: mean ~2.0 for a consistent filter, 5.99 is the 95% single-sample
    bound. Large values mean the filter is overconfident.
    """
    w = belief.normalized_weights()
    error = np.array([float(w @ belief.east_m) - east_m,
                      float(w @ belief.north_m) - north_m])
    cov = position_covariance(belief) + 1e-9 * np.eye(2)
    return float(error @ np.linalg.solve(cov, error))


def _hash_belief(hasher, belief: ParticleBelief) -> None:
    for arr in (belief.east_m, belief.north_m, belief.heading_rad,
                belief.log_weight):
        hasher.update(np.ascontiguousarray(arr).tobytes())


def _validate(config: structs.FilterConfig, catalog, odometry,
              measurements, tables) -> None:
    if config.n_particles <= 0:
        raise ValueError(f"n_particles must be positive, got "
                         f"{config.n_particles}")
    if not 0.0 < config.pi0 < 1.0:
        raise ValueError(f"pi0 must be in (0, 1), got {config.pi0}")
    if not 0.0 <= config.ess_resample_frac <= 1.0:
        raise ValueError(f"ess_resample_frac must be in [0, 1], got "
                         f"{config.ess_resample_frac}")
    if config.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")

    for kf, delta in enumerate(odometry, start=1):
        if delta.keyframe_idx != kf:
            raise ValueError(f"odometry out of order: expected keyframe {kf},"
                             f" got {delta.keyframe_idx}")
        if delta.sigma_m < 0.0:
            raise ValueError("odometry sigma_m must be non-negative")

    # Information-epoch rule (§5.3 [CONTRACT], T-F1): a tracklet contributes
    # a new update only when it carries new information. Re-submitting the
    # same (tracklet, anchor) double-counts evidence and sharpens the
    # posterior without improving accuracy.
    seen = set()
    n_keyframes = len(odometry) + 1
    for meas in measurements:
        key = (meas.tracklet_id, meas.anchor_keyframe_idx)
        if key in seen:
            raise ValueError(
                f"duplicate measurement for tracklet {meas.tracklet_id!r} at "
                f"keyframe {meas.anchor_keyframe_idx}: each information "
                f"epoch may be submitted at most once (design doc §5.3)")
        seen.add(key)
        if not 0 <= meas.anchor_keyframe_idx < n_keyframes:
            raise ValueError(
                f"measurement anchored at keyframe "
                f"{meas.anchor_keyframe_idx}, outside [0, {n_keyframes})")
        if meas.tracklet_id not in tables:
            raise ValueError(
                f"no CompatibilityTable for tracklet {meas.tracklet_id!r}")

    for table in tables.values():
        if table.clip_lo > table.clip_hi:
            raise ValueError(
                f"table {table.tracklet_id!r} has clip_lo {table.clip_lo} > "
                f"clip_hi {table.clip_hi}")


def run_filter(
        config: structs.FilterConfig,
        catalog: catalog_mod.LandmarkCatalog,
        odometry: list,
        measurements: list,
        tables: dict) -> FilterHistory:
    """Pure function of (config, ordered event log) -> history (§3.8).

    Keyframes are the odometry timebase; tracklet measurements fire sparsely
    at their anchor keyframes. Keyframe k covers: motion from odometry[k-1],
    then all measurements anchored at k, then health/checkpoint from the
    weighted posterior, then ESS-triggered resampling.
    """
    _validate(config, catalog, odometry, measurements, tables)
    rng = np.random.default_rng(config.seed)
    belief = init_belief(config, rng)
    heading_rw_rad = math.radians(config.heading_random_walk_deg)
    heading_rough_rad = math.radians(config.heading_roughening_deg)

    meas_by_kf = {}
    for meas in measurements:
        meas_by_kf.setdefault(meas.anchor_keyframe_idx, []).append(meas)

    hasher = hashlib.sha256()
    health = []
    checkpoints = {}
    n_keyframes = len(odometry) + 1
    prev_course_rad = None

    for kf in range(n_keyframes):
        if kf > 0:
            delta = odometry[kf - 1]
            course_rad = (math.radians(delta.course_deg)
                          if delta.course_deg is not None else None)
            course_delta_rad = 0.0
            heading_sigma_rad = heading_rw_rad
            if course_rad is not None and prev_course_rad is not None:
                course_delta_rad = float(
                    geodesy.wrap_rad(course_rad - prev_course_rad))
                heading_sigma_rad = course_delta_noise_rad(
                    delta.course_sigma_deg, heading_rw_rad)
            motion_update(belief, delta, course_delta_rad, heading_sigma_rad,
                          rng)
            if course_rad is not None:
                prev_course_rad = course_rad

        associations = []
        for meas in meas_by_kf.get(kf, []):
            associations.append(measurement_update(
                belief, meas, tables[meas.tracklet_id], catalog, config.pi0))

        belief.log_weight -= special.logsumexp(belief.log_weight)
        current_ess = ess(belief.log_weight)

        mean_e, mean_n, mean_h = mean_pose(belief)
        map_e, map_n, map_h = map_pose(belief, config.map_cell_size_m)
        resampled = current_ess < config.ess_resample_frac * config.n_particles
        health.append(structs.HealthRecord(
            keyframe_idx=kf,
            ess=float(current_ess),
            resampled=resampled,
            mean_east_m=mean_e,
            mean_north_m=mean_n,
            mean_heading_deg=math.degrees(mean_h) % 360.0,
            map_east_m=map_e,
            map_north_m=map_n,
            map_heading_deg=math.degrees(map_h) % 360.0,
            position_std_m=position_std_m(belief),
            heading_std_deg=heading_std_deg(belief),
            n_measurements=len(meas_by_kf.get(kf, [])),
            associations=associations))
        _hash_belief(hasher, belief)
        # The last keyframe always checkpoints, so checkpoints[-1] is the
        # final weighted posterior — and it is a copy, unaffected by the
        # resampling below.
        if kf % config.checkpoint_every == 0 or kf == n_keyframes - 1:
            checkpoints[kf] = belief.copy()

        if resampled:
            systematic_resample(belief, rng, config.resample_regularization,
                                config.position_roughening_m,
                                heading_rough_rad)

    return FilterHistory(health=health, checkpoints=checkpoints,
                         particle_history_sha256=hasher.hexdigest(),
                         final_belief=checkpoints[n_keyframes - 1])


def _errors(health: list, truth: list, east_key: str, north_key: str):
    truth_by_kf = {t.keyframe_idx: t for t in truth}
    return np.array([
        math.hypot(getattr(r, east_key) - truth_by_kf[r.keyframe_idx].east_m,
                   getattr(r, north_key) - truth_by_kf[r.keyframe_idx].north_m)
        for r in health])


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
        abs(math.degrees(float(geodesy.wrap_rad(
            math.radians(r.mean_heading_deg)
            - math.radians(truth_by_kf[r.keyframe_idx].heading_deg)))))
        for r in health])
