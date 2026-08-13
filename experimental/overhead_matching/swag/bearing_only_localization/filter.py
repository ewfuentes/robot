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
    mode_tracker as mode_tracker_mod,
    proposal as proposal_mod,
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
    # Provenance (§5.5 [CONTRACT]): which proposal event and which of its
    # hypotheses produced each particle; -1 for motion-model descent. Two
    # ints per particle, and the single most valuable thing to have when
    # asking where a wrong mode came from.
    proposal_event_id: np.ndarray = None
    proposal_hypothesis: np.ndarray = None
    # Mode membership carried from the previous keyframe. Mode identity is
    # tracked by lineage (mode_tracker.py), so this has to survive
    # resampling and injection like any other per-particle state.
    mode_id: np.ndarray = None

    def __post_init__(self):
        if self.proposal_event_id is None:
            self.proposal_event_id = np.full(self.east_m.shape[0], -1,
                                             dtype=np.int64)
        if self.proposal_hypothesis is None:
            self.proposal_hypothesis = np.full(self.east_m.shape[0], -1,
                                               dtype=np.int64)
        if self.mode_id is None:
            self.mode_id = np.full(self.east_m.shape[0], -1, dtype=np.int64)

    @property
    def n(self) -> int:
        return self.east_m.shape[0]

    def copy(self) -> "ParticleBelief":
        return ParticleBelief(self.east_m.copy(), self.north_m.copy(),
                              self.heading_rad.copy(), self.log_weight.copy(),
                              self.proposal_event_id.copy(),
                              self.proposal_hypothesis.copy(),
                              self.mode_id.copy())

    def take(self, idx: np.ndarray) -> None:
        """Reindex every per-particle array in place (resample/inject)."""
        self.east_m = self.east_m[idx]
        self.north_m = self.north_m[idx]
        self.heading_rad = self.heading_rad[idx]
        self.proposal_event_id = self.proposal_event_id[idx]
        self.proposal_hypothesis = self.proposal_hypothesis[idx]
        self.mode_id = self.mode_id[idx]

    def normalized_weights(self) -> np.ndarray:
        return np.exp(self.log_weight - special.logsumexp(self.log_weight))


@dataclasses.dataclass
class FilterHistory:
    health: list  # list[structs.HealthRecord]
    checkpoints: dict  # keyframe_idx -> ParticleBelief (weighted posterior)
    particle_history_sha256: str
    final_belief: ParticleBelief  # weighted posterior at the last keyframe
    proposal_events: list = dataclasses.field(default_factory=list)
    mode_events: list = dataclasses.field(default_factory=list)


def _mode_records(belief, tracker) -> list:
    """Re-weight the tracked modes under the current posterior.

    The tracker clusters on the prior; weights and spreads are only
    meaningful after the measurement update, so they are recomputed here.
    """
    weights = belief.normalized_weights()
    records = []
    for record in tracker._previous.values():  # noqa: SLF001 - same module
        member = belief.mode_id == record.mode_id
        mass = float(weights[member].sum())
        if mass <= 0.0:
            continue
        member_weights = weights[member]
        records.append(structs.ModeRecord(
            mode_id=record.mode_id,
            weight=mass,
            n_particles=int(member.sum()),
            mean_east_m=float(member_weights @ belief.east_m[member] / mass),
            mean_north_m=float(member_weights @ belief.north_m[member] / mass),
            mean_heading_deg=mode_tracker_mod._circular_mean_deg(  # noqa: SLF001
                belief.heading_rad[member], member_weights),
            position_std_m=mode_tracker_mod.ModeTracker._position_std(  # noqa: SLF001
                belief, member, member_weights),
            heading_std_deg=mode_tracker_mod._circular_std_deg(  # noqa: SLF001
                belief.heading_rad[member], member_weights),
            birth_keyframe_idx=record.birth_keyframe_idx,
            parent_mode_ids=record.parent_mode_ids,
            provenance=record.provenance))
    records.sort(key=lambda r: -r.weight)
    return records


def _mode_entropy(modes) -> float:
    """Entropy of the mode weights in nats: the §5.1 multimodality flag as a
    number. 0 when one mode holds everything, log(k) when k modes tie."""
    if not modes:
        return 0.0
    weights = np.array([m.weight for m in modes])
    total = weights.sum()
    if total <= 0.0:
        return 0.0
    weights = weights / total
    weights = weights[weights > 0.0]
    return float(-(weights * np.log(weights)).sum())


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
        pi0: float,
        per_mode: bool = True) -> list:
    """Association-marginalized bearing update (design doc §5.3).

    p(z|x) = pi0/(2 pi) + (1-pi0) * sum_j w_j * LR_j * vM(delta_j; kappa_eff)

    The candidate axis is processed in blocks so the (n_particles, n_cand)
    temporaries stay bounded; two passes are needed because responsibilities
    are normalized by the total likelihood.

    Returns the whole-belief AssociationPosterior first, then one per mode
    (§5.4 `[CONTRACT]`): averaging responsibilities across a multimodal
    belief blends contradictory explanations into a number that describes
    neither, so the per-mode split is the reportable form.
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

    # Pass 2: responsibilities averaged under the updated particle weights,
    # for the whole belief and for each mode.
    weights = belief.normalized_weights()
    groups = [(None, np.ones(belief.n, dtype=bool), weights)]
    if per_mode:
        for mode_id in np.unique(belief.mode_id):
            if int(mode_id) < 0:
                continue
            member = belief.mode_id == mode_id
            mass = float(weights[member].sum())
            if mass <= 0.0:
                continue
            group_weights = np.zeros(belief.n)
            group_weights[member] = weights[member] / mass
            groups.append((int(mode_id), member, group_weights))

    responsibilities = [{} for _ in groups]
    null_shares = [0.0] * len(groups)
    for sl in blocks:
        resp = np.exp(block_log_terms(sl) - log_lik[:, None])
        for position, (_, _, group_weights) in enumerate(groups):
            avg = group_weights @ resp
            for offset, value in enumerate(avg):
                responsibilities[position][
                    catalog.landmark_ids[sl.start + offset]] = float(value)
    null_term = np.exp(log_null - log_lik)
    for position, (_, _, group_weights) in enumerate(groups):
        null_shares[position] = float(group_weights @ null_term)

    return [
        structs.AssociationPosterior(
            tracklet_id=meas.tracklet_id,
            anchor_keyframe_idx=meas.anchor_keyframe_idx,
            null_share=null_shares[position],
            responsibilities=responsibilities[position],
            mode_id=mode_id)
        for position, (mode_id, _, _) in enumerate(groups)]


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
    belief.take(np.searchsorted(cumulative, positions, side="left"))

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


def inject_proposal(belief: ParticleBelief, result, config: structs.FilterConfig,
                    rng: np.random.Generator) -> int:
    """Replace a fraction of the belief with proposal-drawn particles.

    Mixture-MCL restart: keep (1 - phi) of the mass resampled from the
    current belief and draw phi from the proposal, then let the caller apply
    this keyframe's measurement likelihood to everything.

    APPROXIMATION, stated plainly: a strict mixture proposal would weight
    injected particles by prior(x)/q(x), which needs a density estimate of
    both. We take q(x) as approximately proportional to the likelihood the
    proposal was built from — standard mixture-MCL practice — which biases
    the injected mass toward whatever the resected bearings support. That
    bias is exactly what post-recovery NEES would expose, so
    `proposal_test.RecoveryConsistencyTest` guards it rather than trusting
    the approximation.

    Returns the number of particles injected.
    """
    n_inject = int(round(config.proposal.inject_fraction * belief.n))
    if not result.hypotheses or n_inject <= 0:
        return 0
    n_inject = min(n_inject, belief.n)

    east, north, heading, hypothesis = proposal_mod.sample_particles(
        result, n_inject, config.proposal, rng)
    if east.size == 0:
        return 0

    n_keep = belief.n - n_inject
    if n_keep > 0:
        # Resample the retained mass so it is an unweighted sample too;
        # otherwise kept and injected particles are on different footings.
        weights = belief.normalized_weights()
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0
        positions = (rng.random() + np.arange(n_keep)) / n_keep
        belief.take(np.searchsorted(cumulative, positions, side="left"))
    else:
        belief.take(np.zeros(0, dtype=np.int64))

    belief.east_m = np.concatenate([belief.east_m, east])
    belief.north_m = np.concatenate([belief.north_m, north])
    belief.heading_rad = np.concatenate([belief.heading_rad,
                                         geodesy.wrap_rad(heading)])
    belief.proposal_event_id = np.concatenate([
        belief.proposal_event_id,
        np.full(east.size, result.event_id, dtype=np.int64)])
    belief.proposal_hypothesis = np.concatenate([
        belief.proposal_hypothesis, hypothesis.astype(np.int64)])
    # Injected particles have no ancestor: they will found new modes, and
    # the mode tracker reads their provenance to say where from.
    belief.mode_id = np.concatenate([
        belief.mode_id, np.full(east.size, -1, dtype=np.int64)])
    # Kept mass carries (1 - phi), injected carries phi, spread evenly
    # within each component.
    keep_share = 1.0 - config.proposal.inject_fraction
    belief.log_weight = np.concatenate([
        np.full(n_keep, math.log(keep_share / n_keep) if n_keep else 0.0),
        np.full(east.size,
                math.log(config.proposal.inject_fraction / east.size))])
    return int(east.size)


def _proposal_event_record(result, n_injected: int) -> structs.ProposalEvent:
    return structs.ProposalEvent(
        event_id=result.event_id,
        keyframe_idx=result.keyframe_idx,
        trigger=result.trigger,
        n_hypotheses=len(result.hypotheses),
        n_injected=n_injected,
        n_tracklets_considered=result.n_tracklets_considered,
        n_combinations_examined=result.n_combinations_examined,
        n_combinations_skipped=result.n_combinations_skipped,
        hypothesis_tracklet_ids=[list(h.tracklet_ids)
                                 for h in result.hypotheses],
        hypothesis_landmark_ids=[list(h.landmark_ids)
                                 for h in result.hypotheses])


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
    tracker = mode_tracker_mod.ModeTracker(config.modes)
    heading_rw_rad = math.radians(config.heading_random_walk_deg)
    heading_rough_rad = math.radians(config.heading_roughening_deg)

    meas_by_kf = {}
    for meas in measurements:
        meas_by_kf.setdefault(meas.anchor_keyframe_idx, []).append(meas)

    hasher = hashlib.sha256()
    health = []
    checkpoints = {}
    proposal_events = []
    n_keyframes = len(odometry) + 1
    prev_course_rad = None
    null_history = []
    all_mode_events = []
    low_ess_run = 0
    last_proposal_kf = None

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

        keyframe_measurements = meas_by_kf.get(kf, [])
        # Cluster BEFORE the measurement updates: modes are the hypotheses
        # entering the update, so "mode A believes tracklet 7 is X" refers to
        # a mode that existed before tracklet 7 was seen.
        mode_events = []
        if config.modes.enabled:
            assignment = tracker.update(belief, kf, proposal_events)
            belief.mode_id = assignment.mode_id
            mode_events = assignment.events

        associations = []
        for meas in keyframe_measurements:
            associations.extend(measurement_update(
                belief, meas, tables[meas.tracklet_id], catalog, config.pi0,
                per_mode=config.modes.enabled))

        belief.log_weight -= special.logsumexp(belief.log_weight)
        current_ess = ess(belief.log_weight)

        # --- mixture proposal: global init and recovery (§5.5) ---
        # Kidnapped detection runs off the same null-share the health stream
        # publishes: when the belief stops explaining the bearings, the
        # evidence lands on the null hypothesis rather than on any landmark.
        if associations:
            mean_null = float(np.mean([a.null_share for a in associations]))
            null_history.append(
                mean_null > config.proposal.null_share_threshold)
            del null_history[:-config.proposal.null_share_window]
        if current_ess < config.proposal.ess_floor_frac * config.n_particles:
            low_ess_run += 1
        else:
            low_ess_run = 0

        trigger = None
        if config.proposal.enabled and keyframe_measurements:
            refractory_ok = (
                last_proposal_kf is None
                or kf - last_proposal_kf >= config.proposal.refractory_keyframes)
            if config.proposal.on_init and not proposal_events:
                # The FIRST keyframe carrying bearings, not keyframe 0.
                # Real runs start observing several keyframes in (leg1's
                # first tracklet anchors at kf 3), and keying this to kf 0
                # meant the initial proposal silently never fired — the
                # uniform prior was then left to brute force.
                trigger = "init"
            elif refractory_ok and (
                    len(null_history) >= config.proposal.null_share_window
                    and (np.mean(null_history)
                         >= config.proposal.null_share_min_fraction)):
                trigger = "null_share"
            elif refractory_ok and (
                    low_ess_run >= config.proposal.ess_floor_keyframes):
                trigger = "ess_floor"

        event_id = None
        if trigger is not None:
            # Staggered epochs mean one keyframe usually carries a single
            # bearing, so gather a short window and treat it as simultaneous
            # (proposal.py documents the translation/range error this costs).
            window_start = kf - config.proposal.window_keyframes
            window = [m for m in measurements
                      if window_start <= m.anchor_keyframe_idx <= kf]
            result = proposal_mod.propose(
                window, tables, catalog, config.proposal,
                event_id=len(proposal_events), keyframe_idx=kf,
                trigger=trigger)
            n_injected = inject_proposal(belief, result, config, rng)
            proposal_events.append(_proposal_event_record(result, n_injected))
            if n_injected:
                event_id = result.event_id
                last_proposal_kf = kf
                null_history.clear()
                low_ess_run = 0
                # Injected particles have not seen this keyframe's bearings,
                # so re-apply them to the whole belief. Injected mass is
                # drawn from a subset of tracklets; scoring everything under
                # the full measurement model is what puts kept and injected
                # particles on the same footing.
                associations = []
                for meas in keyframe_measurements:
                    associations.extend(measurement_update(
                        belief, meas, tables[meas.tracklet_id], catalog,
                        config.pi0, per_mode=config.modes.enabled))
                belief.log_weight -= special.logsumexp(belief.log_weight)
                current_ess = ess(belief.log_weight)
                # Injection changed the belief, so the clusters that entered
                # the update no longer describe it; re-derive them.
                if config.modes.enabled:
                    assignment = tracker.update(belief, kf, proposal_events)
                    belief.mode_id = assignment.mode_id
                    mode_events = mode_events + assignment.events

        mean_e, mean_n, mean_h = mean_pose(belief)
        map_e, map_n, map_h = map_pose(belief, config.map_cell_size_m)
        modes = _mode_records(belief, tracker) if config.modes.enabled else []
        all_mode_events.extend(mode_events)
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
            n_measurements=len(keyframe_measurements),
            proposal_weight_share=float(
                belief.normalized_weights()[belief.proposal_event_id >= 0]
                .sum()),
            proposal_event_id=event_id,
            associations=associations,
            modes=modes,
            mode_entropy_nats=_mode_entropy(modes)))
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
                         final_belief=checkpoints[n_keyframes - 1],
                         proposal_events=proposal_events,
                         mode_events=all_mode_events)


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
