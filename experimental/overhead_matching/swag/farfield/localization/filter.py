"""SE(2) particle filter core for bearing-only localization.

Implements docs/localization-design-doc.md §5 on the synthetic-harness scale:
association-marginalized von Mises bearing likelihood with a mandatory null
hypothesis (§5.3), sparse information-epoch measurement events, systematic
resampling, and body-frame dead-reckoning propagation (§5.2).

Pure numpy, CPU-only, one explicit np.random.Generator: the filter is a
deterministic function of (config, seed, ordered event log) — the replay
contract of §3.8. Positions are region-frame ENU metres; heading is the
forward-axis world bearing in radians clockwise from north; measurement
bearings are forward-frame (§4).

Odometry is the §5.2 body-frame SE(2) increment, rotated through each
particle's own heading — position routes through heading by design, so
heading error grows the dead-reckoning wedge this filter exists to fight.
Absolute heading is anchored solely by landmark bearings; the increments
carry only yaw *changes*.

Reported belief statistics and checkpoints are taken from the WEIGHTED
posterior, before resampling and roughening. Resampling is a representational
step and roughening deliberately adds variance the posterior does not have;
reporting after either would misstate the belief.

Belief statistics and error-vs-truth series (mean/MAP pose, covariance,
NEES, error curves) live in metrics.py; this module only runs the filter.
"""

import dataclasses
import hashlib
import math

import numpy as np
from scipy import special

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.localization import (
    filter_catalog as catalog_mod,
    metrics,
    mode_tracker as mode_tracker_mod,
    proposal as proposal_mod,
    retrieval as retrieval_mod,
    structs,
)

# Cap on any single measurement's concentration. §6 makes the matcher's LLRs
# clipped so semantics cannot steamroll geometry; kappa is the same hazard on
# the geometric side — 1e6 is ~0.06 deg, far below any real bearing accuracy.
# The measurements behind this filter's design -- the whole-map campaign's
# five fixes, each with the number that justifies it -- are recorded in
# docs/farfield/decisions.md (2026-05 - 2026-08). Read that before
# simplifying persistence, the identity-posterior normalisation, the
# unendorsed-candidate rule, or the resampling scheme: every one of them is
# the scar of a run that drifted.
MAX_KAPPA = 1.0e6
# Candidate-axis block size for the measurement update, bounding the (N, M)
# temporary arrays.
CANDIDATE_BLOCK = 256
# Per-particle association state (§5.3 persistence): a particle's belief
# about WHICH physical object a tracklet is. Catalog index when committed.
ASSOC_UNCOMMITTED = -2  # tracklet not yet observed by this particle
ASSOC_NULL = -1  # committed to "this tracklet is clutter"


@dataclasses.dataclass
class ParticleBelief:
    east_m: np.ndarray
    north_m: np.ndarray
    heading_rad: np.ndarray  # clockwise from north
    log_weight: np.ndarray
    # Provenance (§5.5 [CONTRACT]): which proposal event and which of its
    # hypotheses produced each particle; -1 for motion-model descent. These
    # fields support mode attribution.
    proposal_event_id: np.ndarray = None
    proposal_hypothesis: np.ndarray = None
    # Mode membership carried from the previous keyframe. Mode identity is
    # tracked by lineage (mode_tracker.py), so this has to survive
    # resampling and injection like any other per-particle state.
    mode_id: np.ndarray = None
    # Per-tracklet association state (§5.3 persistence): tracklet_id ->
    # int32 array of catalog indices / ASSOC_NULL / ASSOC_UNCOMMITTED.
    # Survives resampling like any other per-particle state.
    associations: dict = None

    def __post_init__(self):
        if self.proposal_event_id is None:
            self.proposal_event_id = np.full(self.east_m.shape[0], -1,
                                             dtype=np.int64)
        if self.proposal_hypothesis is None:
            self.proposal_hypothesis = np.full(self.east_m.shape[0], -1,
                                               dtype=np.int64)
        if self.mode_id is None:
            self.mode_id = np.full(self.east_m.shape[0], -1, dtype=np.int64)
        if self.associations is None:
            self.associations = {}

    @property
    def n(self) -> int:
        return self.east_m.shape[0]

    def copy(self) -> "ParticleBelief":
        return ParticleBelief(self.east_m.copy(), self.north_m.copy(),
                              self.heading_rad.copy(), self.log_weight.copy(),
                              self.proposal_event_id.copy(),
                              self.proposal_hypothesis.copy(),
                              self.mode_id.copy(),
                              {tid: arr.copy()
                               for tid, arr in self.associations.items()})

    def take(self, idx: np.ndarray) -> None:
        """Reindex every per-particle array in place (resample/inject)."""
        self.east_m = self.east_m[idx]
        self.north_m = self.north_m[idx]
        self.heading_rad = self.heading_rad[idx]
        self.proposal_event_id = self.proposal_event_id[idx]
        self.proposal_hypothesis = self.proposal_hypothesis[idx]
        self.mode_id = self.mode_id[idx]
        for tid in self.associations:
            self.associations[tid] = self.associations[tid][idx]

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


class RunObserver:
    """The Tier-3 instrumentation seam (design doc §7.1, §7.5 [CONTRACT]).

    Every internal quantity the viewer wants — per-particle weight deltas,
    per-tracklet log-likelihood contributions, gating decisions — exists only
    transiently inside `run_filter`. §7.5 requires that the code which
    reconstructs them be the production filter in replay mode, not a second
    implementation, so that "viewer math disagrees with filter math" is
    structurally impossible rather than merely unlikely. This class is how:
    `run_filter` hands its internals out as it computes them, and consumers
    (attribution, what-if diffing) subclass rather than re-derive.

    Every hook is a no-op by default, and `run_filter` skips the bookkeeping
    that feeds them entirely when no observer is attached, so an
    uninstrumented run pays nothing — the determinism contract is unaffected
    either way, because observers only ever read.

    Arrays passed to hooks are the filter's live buffers. Copy before
    retaining anything; the filter mutates them in place.
    """

    def keyframe_start(self, keyframe_idx: int, belief: ParticleBelief) -> None:
        """After motion and mode clustering, before any measurement."""

    def measurement(self, keyframe_idx: int, meas, log_weight_before,
                    belief: ParticleBelief, pass_index: int) -> None:
        """After one measurement's weights land.

        `log_weight_before` is a copy taken before the update; the post-update
        weights are `belief.log_weight`, and `belief.mode_id` holds the
        grouping that entered the update. `pass_index` is 0 for the ordinary
        pass and 1 for the re-application that follows a proposal injection
        (§5.5), which scores kept and injected mass on the same footing.
        """

    def injection(self, keyframe_idx: int, event, n_injected: int) -> None:
        """After a proposal event resolves, injected or gate-rejected."""

    def resample(self, keyframe_idx: int, log_weight_before, mode_id_before,
                 belief: ParticleBelief) -> None:
        """After an ESS-triggered resample. The `_before` arrays describe the
        weighted posterior that was resampled, so a consumer can separate
        weight change caused by evidence from weight change caused by
        resampling — the "motion/resample effects" term of §7.2."""

    def keyframe_end(self, keyframe_idx: int, belief: ParticleBelief,
                     health) -> None:
        """After the keyframe's HealthRecord is built, before resampling."""


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


def motion_update(belief: ParticleBelief, delta: structs.OdometryDelta,
                  heading_rw_rad: float, rng: np.random.Generator) -> None:
    """Forward/left/CW-yaw increment under rotate-then-move semantics.

    The §5.2 order is load-bearing: heading noise is sampled BEFORE the
    rotation, the particle heading is updated, and translation is rotated by
    that updated heading. Sampling first is what makes heading spread flow into
    cross-track position spread; there is deliberately no cross-track sigma
    to double-count it. The config-level heading random walk is the filter's
    unmodeled-drift hedge, in quadrature with the increment's own
    sigma_yaw_rad.
    """
    n = belief.n
    sigma_h = math.hypot(delta.sigma_yaw_rad, heading_rw_rad)
    delta_yaw_cw = (delta.delta_yaw_cw_rad
                    + rng.normal(0.0, sigma_h, size=n))
    belief.heading_rad = geo.wrap_rad(
        belief.heading_rad + delta_yaw_cw)
    forward = delta.forward_m + rng.normal(0.0, delta.sigma_m, size=n)
    left = delta.left_m + rng.normal(0.0, delta.sigma_m, size=n)
    sin_heading = np.sin(belief.heading_rad)
    cos_heading = np.cos(belief.heading_rad)
    # Heading is compass (CW from north): forward -> (sin h, cos h),
    # left -> (-cos h, sin h). Pinned by the T-U8 golden fixture.
    belief.east_m += forward * sin_heading - left * cos_heading
    belief.north_m += forward * cos_heading + left * sin_heading


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


def _mixture_block_log_terms(east_m, north_m, heading_rad, observed_rad,
                             kappa_z, log_weight, catalog, sl):
    """log[p(j|appearance) * vM(delta_j; kappa_eff)] for one candidate
    block, from arbitrary pose arrays. `log_weight` is the proper identity
    posterior (`_identity_log_weights`); the (1-pi0) mixture constant is
    the caller's."""
    bearing_world, range_m = catalog.bearings_from(east_m, north_m, sl)
    delta = geo.wrap_rad(
        bearing_world - heading_rad[:, None] - observed_rad)
    kappa_eff = catalog.kappa_eff(kappa_z, range_m, sl)
    return log_weight[sl][None, :] + von_mises_logpdf(delta, kappa_eff)


def pose_log_likelihood(east_m, north_m, heading_rad,
                        meas: structs.TrackletMeasurement,
                        table: structs.CompatibilityTable,
                        catalog: catalog_mod.LandmarkCatalog,
                        pi0: float,
                        log_weight: np.ndarray = None,
                        matcher_recall: float = 0.5) -> np.ndarray:
    """p(z | pose) under the §5.3 mixture, for arbitrary pose arrays.

    Exactly the density the belief update applies — exposed so proposal
    hypotheses can be scored on the same footing as the belief (§5.5
    evidence gate)."""
    kappa_z = min(float(meas.kappa), MAX_KAPPA)
    if log_weight is None:
        log_weight = _identity_log_weights(table, catalog, matcher_recall)
    observed_rad = math.radians(meas.bearing_forward_cw_deg)
    log_null = math.log(pi0) - math.log(2.0 * math.pi)
    log_mix = math.log1p(-pi0)
    east_m = np.asarray(east_m, dtype=np.float64)
    north_m = np.asarray(north_m, dtype=np.float64)
    heading_rad = np.asarray(heading_rad, dtype=np.float64)
    blocks = [slice(start, min(start + CANDIDATE_BLOCK, catalog.n))
              for start in range(0, catalog.n, CANDIDATE_BLOCK)]
    per_block = np.empty((east_m.shape[0], len(blocks) + 1))
    per_block[:, 0] = log_null
    for i, sl in enumerate(blocks):
        per_block[:, i + 1] = special.logsumexp(
            log_mix + _mixture_block_log_terms(
                east_m, north_m, heading_rad, observed_rad, kappa_z,
                log_weight, catalog, sl), axis=1)
    return special.logsumexp(per_block, axis=1)


def _responsibility_groups(belief: ParticleBelief, per_mode: bool) -> list:
    """(mode_id, group_weights) rows: the whole belief first, then each mode.

    Group weights are the posterior weights renormalized within the group,
    zero outside it — the averaging weights for §5.4 per-mode association
    posteriors. Shared by the numpy and torch backends so the reportable
    form cannot drift between them.
    """
    weights = belief.normalized_weights()
    groups = [(None, weights)]
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
            groups.append((int(mode_id), group_weights))
    return groups


def measurement_update(
        belief: ParticleBelief,
        meas: structs.TrackletMeasurement,
        table: structs.CompatibilityTable,
        catalog: catalog_mod.LandmarkCatalog,
        pi0: float,
        per_mode: bool = True,
        log_weight: np.ndarray = None,
        resp_min: float = 0.0,
        assoc: np.ndarray = None,
        renewal_rate: float = 0.1,
        outlier_rate: float = 0.1,
        rng: np.random.Generator = None,
        surprise: np.ndarray = None,
        matcher_recall: float = 0.5) -> list:
    """Bearing update (design doc §5.3), in one of two association regimes.

    `assoc is None` — per-epoch marginalization:

        p(z|x) = pi0/(2 pi) + (1-pi0) * sum_j p(j|appearance) * vM(delta_j)

    where p(j|appearance) is the PROPER identity posterior built from the
    tracklet's table (`_identity_log_weights`; `matcher_recall` splits mass
    between endorsed entries and the rest of the catalog).

    `assoc` given — persistence: a tracklet is ONE physical object, so its
    identity is a latent variable that persists across its epochs under a
    renewal HMM, p(a_t | a_{t-1}) = (1-beta) delta + beta renewal:

        w *= (1-beta) * p(z | x, a) + beta * p_mixture(z | x)

    with p(z|x, a=j) = (1-eps) vM(delta_j) + eps/(2 pi) and p(z|x, null) =
    1/(2 pi); the association renews (sampled from the mixture components,
    Gumbel-max) exactly when the renewal branch wins, and uncommitted
    particles reduce to the pure mixture (beta_effective = 1). Marginalizing
    independently per epoch re-pays the 1/|catalog| candidate prior on every
    epoch of a tracklet and couples evidence to catalog size. Persistence pays
    the identity prior once per tracklet; subsequent epochs are pure geometry.
    `assoc` (int32: catalog index, ASSOC_NULL, or
    ASSOC_UNCOMMITTED) is updated IN PLACE; `rng` is required.

    The candidate axis is processed in blocks so the (n_particles, n_cand)
    temporaries stay bounded. `log_weight` (identity posterior) and
    `surprise` (endorsement mask) may be passed precomputed — run_filter
    caches one of each per table. `resp_min` drops responsibilities below
    the threshold from the returned posteriors — a reporting filter only,
    the likelihood itself is never truncated.

    Returns the whole-belief AssociationPosterior first, then one per mode
    (§5.4 `[CONTRACT]`): averaging responsibilities across a multimodal
    belief blends contradictory explanations into a number that describes
    neither, so the per-mode split is the reportable form. Under
    persistence the reported responsibilities are committed shares: the
    weighted fraction of the group committed to each landmark.
    """
    if not 0.0 < pi0 < 1.0:
        raise ValueError(f"pi0 must be in (0, 1), got {pi0}")
    if not math.isfinite(meas.kappa) or meas.kappa <= 0.0:
        raise ValueError(f"kappa must be positive and finite, got "
                         f"{meas.kappa}")
    kappa_z = min(float(meas.kappa), MAX_KAPPA)
    if log_weight is None:
        log_weight = _identity_log_weights(table, catalog, matcher_recall)
    if surprise is None:
        surprise = _surprise_mask(table, _clipped_log_lr(table, catalog))
    observed_rad = math.radians(meas.bearing_forward_cw_deg)
    log_null = math.log(pi0) - math.log(2.0 * math.pi)
    log_mix = math.log1p(-pi0)

    def block_log_terms(sl):
        """log[(1-pi0) * p(j|app) * vM(...)] for one candidate block."""
        return log_mix + _mixture_block_log_terms(
            belief.east_m, belief.north_m, belief.heading_rad,
            observed_rad, kappa_z, log_weight, catalog, sl)

    blocks = [slice(start, min(start + CANDIDATE_BLOCK, catalog.n))
              for start in range(0, catalog.n, CANDIDATE_BLOCK)]

    if assoc is not None:
        return _persistence_update(
            belief, meas, catalog, block_log_terms, blocks, log_null,
            observed_rad, kappa_z, per_mode, resp_min, assoc,
            renewal_rate, outlier_rate, rng, surprise)

    # Pass 1: total log-likelihood per particle.
    per_block = np.empty((belief.n, len(blocks) + 1))
    per_block[:, 0] = log_null
    for i, sl in enumerate(blocks):
        per_block[:, i + 1] = special.logsumexp(block_log_terms(sl), axis=1)
    log_lik = special.logsumexp(per_block, axis=1)
    belief.log_weight += log_lik

    # Pass 2: responsibilities averaged under the updated particle weights,
    # for the whole belief and for each mode.
    groups = _responsibility_groups(belief, per_mode)

    responsibilities = [{} for _ in groups]
    null_shares = [0.0] * len(groups)
    surprise_shares = [0.0] * len(groups)
    for sl in blocks:
        resp = np.exp(block_log_terms(sl) - log_lik[:, None])
        for position, (_, group_weights) in enumerate(groups):
            avg = group_weights @ resp
            surprise_shares[position] += float(avg @ surprise[sl])
            for offset in np.nonzero(avg >= resp_min)[0]:
                responsibilities[position][
                    catalog.landmark_ids[sl.start + offset]] = float(
                        avg[offset])
    null_term = np.exp(log_null - log_lik)
    for position, (_, group_weights) in enumerate(groups):
        null_shares[position] = float(group_weights @ null_term)

    return [
        structs.AssociationPosterior(
            tracklet_id=meas.tracklet_id,
            anchor_keyframe_idx=meas.anchor_keyframe_idx,
            null_share=null_shares[position],
            responsibilities=responsibilities[position],
            mode_id=mode_id,
            surprise_share=surprise_shares[position])
        for position, (mode_id, _) in enumerate(groups)]


def measurement_draw_seed(seed: int, meas) -> int:
    """Deterministic per-(tracklet, epoch) seed for persistence draws.

    Association renewal SAMPLES. If those draws came from the shared run
    rng, the outcome would depend on measurement order within a keyframe and
    on how many draws unrelated machinery consumed. Deriving
    each epoch's stream from (seed, tracklet, anchor) restores exact order
    invariance. (The post-injection re-apply is made single-count by
    restoring the kept mass's association state, not by draw identity —
    the internal resample permutes particles across the stream.)
    """
    digest = hashlib.sha256(
        f"{seed}:{meas.tracklet_id}:{meas.anchor_keyframe_idx}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _identity_log_weights(table: structs.CompatibilityTable,
                          catalog: catalog_mod.LandmarkCatalog,
                          matcher_recall: float) -> np.ndarray:
    """log p(j | tracklet appearance): a PROPER identity posterior over the
    catalog, replacing the unnormalized w_j * LR_j product in the §5.3
    mixture.

    Entries the matcher endorses (clipped LLR above the clipped default)
    share `matcher_recall` of the mass, softmax-weighted by their clipped
    LLRs; every other row — unlisted or rejected — shares the remainder
    uniformly. A table with no endorsed entries is uniform over the catalog
    (the matcher said nothing).

    Directly using w_j * LR_j would leave the landmark branch's integral
    table-dependent and therefore change the effective null probability as
    catalog size and table contents change. Normalizing the identity posterior
    preserves the configured mixture weights.
    `matcher_recall` is a property of the MATCHER (probability the true
    landmark appears among a table's endorsed entries), not of any map;
    the softmax preserves the table's relative evidence within entries.
    """
    if not 0.0 < matcher_recall < 1.0:
        raise ValueError(f"matcher_recall must be in (0, 1), got "
                         f"{matcher_recall}")
    log_lr = _clipped_log_lr(table, catalog)
    endorsed = ~_surprise_mask(table, log_lr)
    n_endorsed = int(endorsed.sum())
    weights = np.empty(catalog.n)
    if n_endorsed == 0:
        weights.fill(-math.log(catalog.n))
        return weights
    entry_logits = log_lr[endorsed]
    if n_endorsed == catalog.n:
        # No unendorsed remainder to hold (1 - recall): all mass goes to
        # the endorsed softmax (degenerate but common in small test worlds).
        weights[:] = log_lr - special.logsumexp(log_lr)
        return weights
    weights[endorsed] = (math.log(matcher_recall) + entry_logits
                         - special.logsumexp(entry_logits))
    weights[~endorsed] = (math.log1p(-matcher_recall)
                          - math.log(catalog.n - n_endorsed))
    return weights


def _surprise_mask(table: structs.CompatibilityTable,
                   log_lr: np.ndarray) -> np.ndarray:
    """Candidates the matcher does NOT vouch for: clipped LLR at or below
    the clipped default. Mass explaining a tracklet through these is
    'identity surprise' — geometry says yes, the matcher says no — the
    §8.4 kidnap signal that null share alone cannot see once a displaced
    belief re-explains bearings with wrong landmarks."""
    default_clipped = min(max(table.default_log_lr, table.clip_lo),
                          table.clip_hi)
    return log_lr <= default_clipped + 1e-12


def committed_log_density(east_m, north_m, heading_rad, landmark_idx,
                          observed_rad, kappa_z, outlier_rate, catalog):
    """log p(z | pose, committed to landmark_idx): the persistence-regime
    per-epoch geometry term, (1-eps) vM(delta; kappa_eff) + eps/(2 pi)."""
    d_east = catalog.east_m[landmark_idx] - east_m
    d_north = catalog.north_m[landmark_idx] - north_m
    range_m = np.hypot(d_east, d_north)
    bearing = geo.compass_bearing_rad(d_east, d_north)
    delta = geo.wrap_rad(bearing - heading_rad - observed_rad)
    kappa_eff = catalog.kappa_eff(kappa_z, range_m, landmark_idx)
    vm = np.exp(von_mises_logpdf(delta, kappa_eff))
    return np.log((1.0 - outlier_rate) * vm
                  + outlier_rate / (2.0 * math.pi))


def _persistence_update(belief, meas, catalog, block_log_terms, blocks,
                        log_null, observed_rad, kappa_z, per_mode, resp_min,
                        assoc, renewal_rate, outlier_rate, rng,
                        surprise_mask) -> list:
    """The persistence regime of `measurement_update` (see its docstring)."""
    if rng is None:
        raise ValueError("the persistence path samples associations and "
                         "needs an rng")
    if not 0.0 < renewal_rate <= 1.0:
        raise ValueError(f"renewal_rate must be in (0, 1], got "
                         f"{renewal_rate}")
    n = belief.n
    if assoc.shape != (n,):
        raise ValueError(f"assoc shape {assoc.shape} != ({n},)")

    # One blocked pass: mixture log-likelihood AND a Gumbel-max categorical
    # sample for the renewal draw, without holding the full (n, catalog.n)
    # matrix.
    #
    # Commitment requires matcher ENDORSEMENT: the sample space is
    # {endorsed candidates} + one background bucket holding the null AND
    # every default-LLR candidate. The mixture (and so the weight) keeps
    # unendorsed candidates exactly; they just cannot be committed to.
    # Committing to whichever unendorsed candidate aligns best would compound
    # incidental geometric agreement across epochs. The background commitment
    # instead scores 1/(2 pi), so it cannot sharpen through repetition.
    per_block = np.empty((n, len(blocks) + 1))
    per_block[:, 0] = log_null
    background = np.full(n, log_null)  # running logsumexp: null + defaults
    best_gumbel = np.full(n, -np.inf)
    sampled = np.full(n, ASSOC_NULL, dtype=np.int32)
    for i, sl in enumerate(blocks):
        terms = block_log_terms(sl)
        per_block[:, i + 1] = special.logsumexp(terms, axis=1)
        unendorsed = surprise_mask[sl]
        if unendorsed.any():
            background = np.logaddexp(
                background,
                special.logsumexp(terms[:, unendorsed], axis=1))
        endorsed = ~unendorsed
        if endorsed.any():
            gumbel = terms[:, endorsed] + rng.gumbel(
                size=(n, int(endorsed.sum())))
            arg = np.argmax(gumbel, axis=1)
            value = gumbel[np.arange(n), arg]
            better = value > best_gumbel
            best_gumbel = np.where(better, value, best_gumbel)
            block_idx = (sl.start + np.nonzero(endorsed)[0]).astype(np.int32)
            sampled = np.where(better, block_idx[arg], sampled)
    background_gumbel = background + rng.gumbel(size=n)
    sampled = np.where(background_gumbel > best_gumbel,
                       np.int32(ASSOC_NULL), sampled)
    log_mix_lik = special.logsumexp(per_block, axis=1)

    keep = np.zeros(n)
    committed = assoc >= 0
    if committed.any():
        keep[committed] = np.exp(committed_log_density(
            belief.east_m[committed], belief.north_m[committed],
            belief.heading_rad[committed], assoc[committed],
            observed_rad, kappa_z, outlier_rate, catalog))
    keep[assoc == ASSOC_NULL] = 1.0 / (2.0 * math.pi)

    uncommitted = assoc == ASSOC_UNCOMMITTED
    keep_scale = np.where(uncommitted, 0.0, 1.0 - renewal_rate)
    renew_scale = np.where(uncommitted, 1.0, renewal_rate)
    renew_term = renew_scale * np.exp(log_mix_lik)
    likelihood = keep_scale * keep + renew_term
    belief.log_weight += np.log(likelihood)

    renew = rng.random(n) < renew_term / likelihood
    assoc[renew] = sampled[renew]

    return _commit_share_posteriors(belief, meas, assoc,
                                    catalog.landmark_ids, per_mode, resp_min,
                                    surprise_mask)


def _commit_share_posteriors(belief, meas, assoc, landmark_ids, per_mode,
                             resp_min, surprise_mask=None) -> list:
    """AssociationPosterior from committed shares: the weighted fraction of
    each group committed to each landmark / to the null (§5.4 reportable
    form under persistence). Shared by the numpy and torch backends."""
    groups = _responsibility_groups(belief, per_mode)
    committed = assoc >= 0
    surprised = (committed & surprise_mask[np.clip(assoc, 0, None)]
                 if surprise_mask is not None
                 else np.zeros(assoc.shape[0], dtype=bool))
    posteriors = []
    for mode_id, group_weights in groups:
        responsibilities = {}
        if committed.any():
            mass = np.bincount(assoc[committed],
                               weights=group_weights[committed],
                               minlength=len(landmark_ids))
            for j in np.nonzero(mass > max(resp_min, 0.0))[0]:
                responsibilities[landmark_ids[j]] = float(mass[j])
        posteriors.append(structs.AssociationPosterior(
            tracklet_id=meas.tracklet_id,
            anchor_keyframe_idx=meas.anchor_keyframe_idx,
            null_share=float(group_weights[assoc == ASSOC_NULL].sum()),
            responsibilities=responsibilities,
            mode_id=mode_id,
            surprise_share=float(group_weights[surprised].sum())))
    return posteriors


def ess(log_weight: np.ndarray) -> float:
    w = np.exp(log_weight - special.logsumexp(log_weight))
    return 1.0 / float(np.sum(np.square(w)))


def _bandwidth_group_index(belief: ParticleBelief) -> np.ndarray:
    """Group id per particle for kernel-bandwidth estimation.

    Groups: each mode; then, for particles no mode has claimed, each
    proposal (event, hypothesis) cluster; then the diffuse remainder.
    Provenance matters because injected clusters are hypothesis-shaped long
    before they hold enough posterior mass to register as modes — smoothing
    them with the diffuse pool's bandwidth destroys them at birth.
    """
    kind = np.where(belief.mode_id >= 0, 0,
                    np.where(belief.proposal_event_id >= 0, 1, 2))
    a = np.where(kind == 0, belief.mode_id,
                 np.where(kind == 1, belief.proposal_event_id, 0))
    b = np.where(kind == 1, belief.proposal_hypothesis, 0)
    _, inverse = np.unique(np.stack([kind, a, b]), axis=1,
                           return_inverse=True)
    return inverse


def systematic_resample(belief: ParticleBelief, rng: np.random.Generator,
                        regularization: float = 1.0,
                        position_roughening_m: float = 0.0,
                        heading_roughening_rad: float = 0.0,
                        survival_floor: int = 0,
                        survival_min_mass: float = 1e-9) -> None:
    """Low-variance (systematic) resampling with kernel regularization (T-U6).

    Plain resampling replaces the posterior with a set of Dirac atoms drawn
    only from locations already present, so repeated resampling collapses
    diversity and the filter grows confident about a spread it no longer
    represents.

    So resample from a kernel-smoothed posterior instead (regularized
    particle filter, Musso/Oudjane/Le Gland): jitter each dimension by
    `regularization * sigma_dim * n^(-1/(d+4))`, d=2. The bandwidth is tied
    to the posterior's own spread, so it vanishes as the belief sharpens
    rather than injecting a fixed, arbitrary variance. `*_roughening_*` add
    a floor on top, for beliefs so collapsed that a proportional bandwidth
    cannot recover them (brute-force global init).

    Resampling and smoothing are both PER GROUP (`_bandwidth_group_index`),
    not global — the mixture-tracking construction (Vermaak et al.):

    - Stratified allocation: each group's offspring count is its posterior
      mass times n (largest-remainder rounding), and the low-variance draw
      runs within the group. Resampling is a representational step, so mass
      may move between hypotheses only through EVIDENCE (weight updates),
      never through sampling noise. Stratified allocation keeps symmetric
      modes balanced until evidence separates them.
    - Per-group bandwidth: one global bandwidth is only right for a
      unimodal belief. During global localization the belief is a set of
      tight hypotheses inside a region-scale cloud; using the diffuse pool's
      bandwidth would spread each tight hypothesis. Within a group the rule
      is unchanged, so a unimodal belief behaves identically. An arc-shaped
      proposal cluster receives an isotropic bandwidth from its own elongated
      spread and is re-scored by the next measurement.

    `survival_floor > 0` extends the mixture-tracking construction with
    guaranteed REPRESENTATION: every mode/proposal group at or above
    `survival_min_mass` keeps at least that many offspring, and offspring
    then carry their group's true mass (log(mass/count)) instead of the
    uniform reset, so the posterior is preserved exactly while no
    represented hypothesis can round to zero offspring. The diffuse
    remainder gets no floor — it is not a hypothesis. At 0 the historical
    uniform-weight behavior is reproduced bit-for-bit.
    """
    n = belief.n
    weights = belief.normalized_weights()
    group_index = _bandwidth_group_index(belief)
    n_groups = int(group_index.max()) + 1

    # Bandwidths come from the pre-resample posterior — the distribution
    # actually being smoothed — one scale triple per group.
    east_scale = np.zeros(n)
    north_scale = np.zeros(n)
    heading_scale = np.zeros(n)
    if regularization > 0.0:
        for group in range(n_groups):
            member = group_index == group
            n_group = int(member.sum())
            mass = float(weights[member].sum())
            if n_group < 2 or mass <= 0.0:
                continue
            w = weights[member] / mass
            bandwidth = regularization * n_group ** (-1.0 / 6.0)
            d_east = belief.east_m[member] - float(w @ belief.east_m[member])
            d_north = (belief.north_m[member]
                       - float(w @ belief.north_m[member]))
            east_scale[member] = bandwidth * math.sqrt(
                max(float(w @ (d_east * d_east)), 0.0))
            north_scale[member] = bandwidth * math.sqrt(
                max(float(w @ (d_north * d_north)), 0.0))
            resultant = math.hypot(
                float(w @ np.sin(belief.heading_rad[member])),
                float(w @ np.cos(belief.heading_rad[member])))
            resultant = min(max(resultant, 1e-15), 1.0 - 1e-15)
            heading_scale[member] = bandwidth * math.sqrt(
                -2.0 * math.log(resultant))

    # Offspring per group: mass * n, largest-remainder rounding among
    # groups that actually hold mass.
    group_mass = np.bincount(group_index, weights=weights,
                             minlength=n_groups)
    raw = group_mass * n
    counts = np.floor(raw).astype(np.int64)
    shortfall = n - int(counts.sum())
    if shortfall > 0:
        fraction = np.where(group_mass > 0.0, raw - np.floor(raw), -1.0)
        counts[np.argsort(-fraction)[:shortfall]] += 1

    minimum = np.zeros(n_groups, dtype=np.int64)
    if survival_floor > 0:
        # One representative member per group tells us whether the group is
        # a hypothesis (mode or proposal cluster) or the diffuse remainder.
        representative = np.zeros(n_groups, dtype=np.int64)
        representative[group_index[::-1]] = np.arange(n - 1, -1, -1)
        is_hypothesis = ((belief.mode_id[representative] >= 0)
                         | (belief.proposal_event_id[representative] >= 0))
        minimum[is_hypothesis & (group_mass >= survival_min_mass)] = (
            survival_floor)
        deficit = np.maximum(minimum - counts, 0)
        need = int(deficit.sum())
        if need > 0:
            counts += deficit
            available = np.maximum(counts - minimum, 0)
            total_available = int(available.sum())
            if need > total_available:
                raise ValueError(
                    f"survival_floor {survival_floor} needs {need} extra "
                    f"offspring but only {total_available} are above their "
                    "floors; the floor is too large for this particle "
                    "budget and group count")
            # Take the surplus back proportionally (largest-remainder), so
            # unfloored mass keeps its relative allocation.
            proportional = need * available / total_available
            reduction = np.floor(proportional).astype(np.int64)
            remainder = need - int(reduction.sum())
            if remainder > 0:
                fraction = np.where(available > 0,
                                    proportional - reduction, -1.0)
                reduction[np.argsort(-fraction)[:remainder]] += 1
            counts -= reduction

    idx_parts = []
    for group in range(n_groups):
        if counts[group] <= 0:
            continue
        members = np.nonzero(group_index == group)[0]
        w_group = weights[members]
        cumulative = np.cumsum(w_group / w_group.sum())
        cumulative[-1] = 1.0  # guard against fp sum < 1
        positions = (rng.random() + np.arange(counts[group])) / counts[group]
        idx_parts.append(
            members[np.searchsorted(cumulative, positions, side="left")])
    idx = np.concatenate(idx_parts)
    # Each offspring inherits its ancestor's group bandwidth.
    east_scale = east_scale[idx]
    north_scale = north_scale[idx]
    heading_scale = heading_scale[idx]
    offspring_group = group_index[idx]
    belief.take(idx)

    east_sigma = np.hypot(east_scale, position_roughening_m)
    north_sigma = np.hypot(north_scale, position_roughening_m)
    heading_sigma = np.hypot(heading_scale, heading_roughening_rad)
    belief.east_m += rng.normal(0.0, 1.0, size=n) * east_sigma
    belief.north_m += rng.normal(0.0, 1.0, size=n) * north_sigma
    belief.heading_rad = geo.wrap_rad(
        belief.heading_rad + rng.normal(0.0, 1.0, size=n) * heading_sigma)
    if survival_floor > 0:
        # Offspring carry their group's true mass, so a floored group holds
        # its (tiny) posterior probability across many guaranteed particles
        # rather than being rounded up to floor/n of the belief.
        with np.errstate(divide="ignore"):
            log_w = (np.log(group_mass[offspring_group])
                     - np.log(counts[offspring_group]))
        belief.log_weight = log_w - special.logsumexp(log_w)
    else:
        belief.log_weight = np.zeros(n)


def inject_proposal(belief: ParticleBelief, result, config: structs.FilterConfig,
                    rng: np.random.Generator) -> tuple[int, np.ndarray | None]:
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

    Returns (n_injected, kept_idx): kept_idx is the ancestor index of each
    retained particle, so per-tracklet association snapshots can be carried
    through the internal resample (§5.3 persistence).
    """
    original_count = belief.n
    requested_injection = int(round(
        config.proposal.inject_fraction * original_count))
    if not result.hypotheses or requested_injection <= 0:
        return 0, None
    requested_injection = min(requested_injection, original_count)

    east, north, heading, hypothesis = proposal_mod.sample_particles(
        result, requested_injection, config.proposal, rng)
    actual_injection = int(east.size)
    if actual_injection == 0:
        return 0, None
    return _splice_injected(belief, east, north, heading,
                            hypothesis.astype(np.int64), result.event_id, rng)


def _splice_injected(belief: ParticleBelief, east, north, heading,
                     hypothesis, event_id: int,
                     rng: np.random.Generator) -> tuple[int, np.ndarray]:
    """Replace `east.size` particles of the belief with the given poses,
    resampling the kept mass so both components are unweighted samples.
    Shared by the bearing resection proposal and the retrieval-seeded
    injection; returns (n_injected, kept ancestor indices)."""
    original_count = belief.n
    actual_injection = int(east.size)
    if actual_injection > original_count:
        raise RuntimeError(
            f"proposal returned {actual_injection} particles for a belief of "
            f"{original_count}")

    n_keep = original_count - actual_injection
    if n_keep > 0:
        # Resample the retained mass so it is an unweighted sample too;
        # otherwise kept and injected particles are on different footings.
        weights = belief.normalized_weights()
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0
        positions = (rng.random() + np.arange(n_keep)) / n_keep
        kept_idx = np.searchsorted(cumulative, positions, side="left")
    else:
        kept_idx = np.zeros(0, dtype=np.int64)
    belief.take(kept_idx)

    belief.east_m = np.concatenate([belief.east_m, east])
    belief.north_m = np.concatenate([belief.north_m, north])
    belief.heading_rad = np.concatenate([belief.heading_rad,
                                         geo.wrap_rad(heading)])
    belief.proposal_event_id = np.concatenate([
        belief.proposal_event_id,
        np.full(east.size, event_id, dtype=np.int64)])
    belief.proposal_hypothesis = np.concatenate([
        belief.proposal_hypothesis, hypothesis.astype(np.int64)])
    # Injected particles have no ancestor: they will found new modes, and
    # the mode tracker reads their provenance to say where from.
    belief.mode_id = np.concatenate([
        belief.mode_id, np.full(east.size, -1, dtype=np.int64)])
    # ...and no association history: they commit at each tracklet's next
    # epoch, paying the identity prior the kept mass has already paid.
    for tid in belief.associations:
        belief.associations[tid] = np.concatenate([
            belief.associations[tid],
            np.full(east.size, ASSOC_UNCOMMITTED, dtype=np.int32)])
    # Derive component mass from what was actually injected. This keeps the
    # particle and probability budgets consistent even if a future proposal
    # implementation legitimately returns less than requested.
    inject_share = actual_injection / original_count
    keep_share = 1.0 - inject_share
    belief.log_weight = np.concatenate([
        np.full(n_keep, math.log(keep_share / n_keep) if n_keep else 0.0),
        np.full(actual_injection,
                math.log(inject_share / actual_injection))])
    if belief.n != original_count:
        raise RuntimeError(
            f"proposal injection changed belief size from {original_count} "
            f"to {belief.n}")
    return actual_injection, kept_idx


def _proposal_event_record(result, n_injected: int, gate_passed: bool = True,
                           gate_best: float = None,
                           gate_ref: float = None) -> structs.ProposalEvent:
    return structs.ProposalEvent(
        event_id=result.event_id,
        keyframe_idx=result.keyframe_idx,
        trigger=result.trigger,
        n_hypotheses=len(result.hypotheses),
        n_injected=n_injected,
        n_tracklets_considered=result.n_tracklets_considered,
        n_combinations_examined=result.n_combinations_examined,
        n_combinations_skipped=result.n_combinations_skipped,
        particle_budget=result.particle_budget,
        n_combinations_total=result.n_combinations_total,
        n_combinations_enumerated=result.n_combinations_enumerated,
        n_combinations_sampled=result.n_combinations_sampled,
        n_combinations_geometry_pruned=(
            result.n_combinations_geometry_pruned),
        n_partially_represented_ties=(
            result.n_partially_represented_ties),
        n_solution_clusters_merged=result.n_solution_clusters_merged,
        represented_compatibility_mass=(
            result.represented_compatibility_mass),
        gate_passed=gate_passed,
        gate_best_hypothesis_nats=gate_best,
        gate_reference_nats=gate_ref,
        hypothesis_tracklet_ids=[list(h.tracklet_ids)
                                 for h in result.hypotheses],
        hypothesis_landmark_ids=[list(h.landmark_ids)
                                 for h in result.hypotheses])


def _belief_window_reference(belief, window_meas,
                             config: structs.FilterConfig, catalog,
                             score_fn, top_k: int = 512) -> float:
    """The belief's own best explanation of the window, scored EXACTLY as
    the filter scores it: committed geometry where a particle is committed,
    the mixture where it is not (§5.3 persistence), maximized over the
    top-weight particles.

    Scoring the incumbent through the plain mixture would understate a
    committed mode because persistence does not repay the identity prior at
    every epoch. The gate therefore uses the same association regime as the
    filter."""
    weights = belief.normalized_weights()
    idx = np.argsort(-weights)[:min(top_k, belief.n)]
    total = np.zeros(idx.size)
    beta = config.association_renewal_rate
    for meas in window_meas:
        mix = np.exp(score_fn(belief.east_m[idx], belief.north_m[idx],
                              belief.heading_rad[idx], meas))
        assoc_arr = (belief.associations.get(meas.tracklet_id)
                     if config.association_persistence else None)
        if assoc_arr is None:
            total += np.log(mix)
            continue
        assoc = assoc_arr[idx]
        keep = np.zeros(idx.size)
        committed = assoc >= 0
        if committed.any():
            keep[committed] = np.exp(committed_log_density(
                belief.east_m[idx][committed],
                belief.north_m[idx][committed],
                belief.heading_rad[idx][committed], assoc[committed],
                math.radians(meas.bearing_forward_cw_deg),
                min(float(meas.kappa), MAX_KAPPA),
                config.association_outlier_rate, catalog))
        keep[assoc == ASSOC_NULL] = 1.0 / (2.0 * math.pi)
        uncommitted = assoc == ASSOC_UNCOMMITTED
        keep_scale = np.where(uncommitted, 0.0, 1.0 - beta)
        renew_scale = np.where(uncommitted, 1.0, beta)
        total += np.log(keep_scale * keep + renew_scale * mix)
    return float(total.max())


def _init_window_is_observable(measurements, kf: int, first_bearing_kf,
                               config: structs.FilterConfig) -> bool:
    """Does this keyframe's proposal window hold enough distinct tracklets?

    See `ProposalConfig.min_tracklets_for_injection` for the counting argument.
    Returns True once the window has that many, and unconditionally once
    `init_max_wait_keyframes` have passed since the first bearing, so a sparse
    sequence still gets an initial proposal rather than being left to sample a
    region-sized uniform box.
    """
    window_start = kf - config.proposal.window_keyframes
    distinct = {m.tracklet_id for m in measurements
                if window_start <= m.anchor_keyframe_idx <= kf}
    if len(distinct) >= config.proposal.min_tracklets_for_injection:
        return True
    return (first_bearing_kf is not None
            and kf - first_bearing_kf >= config.proposal.init_max_wait_keyframes)


def _evidence_gate(tracker, result, window, config: structs.FilterConfig,
                   rng: np.random.Generator, score_fn, belief,
                   catalog) -> tuple:
    """May this proposal displace belief mass? (§5.5 evidence gate.)

    Injection is destructive — it hands `inject_fraction` of the posterior
    to the proposal — so it has to be justified by evidence, not by distress
    alone. A high null share says that the incumbent is struggling; it does not
    establish that any generated proposal explains the evidence better.

    The gate scores hypotheses and the incumbent belief against the same
    window of recent bearings, treated as simultaneous — the same
    approximation the proposal itself makes. Hypotheses are scored under
    the measurement mixture (`pose_log_likelihood`): that is exactly what
    an injected, uncommitted particle would experience. The incumbent is
    scored the way the filter actually scores it — committed geometry
    included (`_belief_window_reference`) — because the mixture understates
    a committed mode by orders of magnitude in a dense catalog. Injection
    proceeds only when the best hypothesis beats the incumbent by
    `evidence_gate_margin_nats`. With no modes to protect (global init, or
    modes disabled) the gate always passes.

    Returns (passed, best_hypothesis_nats, reference_nats).
    """
    latest = {}
    for meas in window:
        previous = latest.get(meas.tracklet_id)
        if (previous is None
                or meas.anchor_keyframe_idx > previous.anchor_keyframe_idx):
            latest[meas.tracklet_id] = meas
    window_meas = sorted(latest.values(), key=lambda m: m.tracklet_id)
    if not window_meas:
        return True, None, None

    def window_score(east, north, heading):
        total = np.zeros(np.asarray(east).shape[0])
        for meas in window_meas:
            total += score_fn(east, north, heading, meas)
        return total

    modes = list(tracker._previous.values())  # noqa: SLF001 - same module
    if not modes:
        return True, None, None
    ref = _belief_window_reference(belief, window_meas, config, catalog,
                                   score_fn)

    parts = []
    for hypothesis in result.hypotheses:
        east, north, heading = hypothesis.sample(
            config.proposal.evidence_gate_samples, config.proposal, rng)
        if east.size:
            parts.append((east, north, heading))
    if not parts:
        return True, None, ref
    scores = window_score(
        np.concatenate([p[0] for p in parts]),
        np.concatenate([p[1] for p in parts]),
        np.concatenate([p[2] for p in parts]))
    # Each hypothesis is scored by its MARGINAL window likelihood over its
    # own sampling density (logsumexp - log S: what injecting it would
    # actually contribute), never by its luckiest single pose. Maximizing over
    # every sampled pose would give larger hypothesis sets a multiple-comparison
    # advantage.
    best = -np.inf
    offset = 0
    n_scored = 0
    for east, _, _ in parts:
        segment = scores[offset:offset + east.size]
        offset += east.size
        n_scored += 1
        best = max(best, float(special.logsumexp(segment))
                   - math.log(segment.size))

    # When enabled, selection across hypotheses is charged log N for the same
    # multiple-comparisons reason.
    selection_penalty = (math.log(max(1, n_scored))
                         if config.proposal.evidence_gate_selection_charge
                         else 0.0)
    threshold = (ref + config.proposal.evidence_gate_margin_nats
                 + selection_penalty)
    return best >= threshold, best, ref


def _hash_belief(hasher, belief: ParticleBelief) -> None:
    for arr in (belief.east_m, belief.north_m, belief.heading_rad,
                belief.log_weight):
        hasher.update(np.ascontiguousarray(arr).tobytes())


def _validate(config: structs.FilterConfig, catalog, odometry,
              measurements, tables, retrieval_fields=None,
              retrieval_measurements=()) -> None:
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
        if delta.sigma_yaw_rad < 0.0:
            raise ValueError("odometry sigma_yaw_rad must be non-negative")
        if not all(math.isfinite(v) for v in (delta.forward_m, delta.left_m,
                                              delta.delta_yaw_cw_rad, delta.sigma_m,
                                              delta.sigma_yaw_rad)):
            raise ValueError(
                f"odometry increment at keyframe {kf} is not finite")

    # Information-epoch contract (§5.3): a tracklet contributes
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

    # Retrieval observation source (CLD-3). The checks are here rather than
    # in the engine so a misconfigured run dies before its first keyframe.
    if bool(retrieval_measurements) != (retrieval_fields is not None):
        raise ValueError(
            "retrieval_fields and retrieval_measurements come together: "
            "fields without events (or events without fields) is a "
            "misconfigured run, not an odometry-only control")
    if retrieval_measurements:
        if config.retrieval is None:
            raise ValueError(
                "retrieval measurements need config.retrieval (temperature "
                "and outlier_epsilon are modeling choices, never defaults)")
        if config.proposal.enabled:
            raise ValueError(
                "the resection proposal is a bearing mechanism and does not "
                "understand retrieval observations; run retrieval sources "
                "with proposal.enabled=False (the epsilon floor is what "
                "keeps deleted hypotheses recoverable)")
        n_fields = retrieval_fields.scores.shape[0]
        seen_kf = set()
        for meas in retrieval_measurements:
            if not 0 <= meas.keyframe_idx < n_keyframes:
                raise ValueError(
                    f"retrieval measurement at keyframe {meas.keyframe_idx},"
                    f" outside [0, {n_keyframes})")
            if not 0 <= meas.field_idx < n_fields:
                raise ValueError(
                    f"retrieval field_idx {meas.field_idx} outside the "
                    f"{n_fields} loaded fields")
            if meas.keyframe_idx in seen_kf:
                raise ValueError(
                    f"two retrieval measurements at keyframe "
                    f"{meas.keyframe_idx}: one panorama is one observation "
                    "(its crops are already pooled inside the field)")
            seen_kf.add(meas.keyframe_idx)


def run_filter(
        config: structs.FilterConfig,
        catalog: catalog_mod.LandmarkCatalog,
        odometry: list,
        measurements: list,
        tables: dict,
        observer: RunObserver | None = None,
        *,
        retrieval_fields=None,
        retrieval_measurements: list = ()) -> FilterHistory:
    """Pure function of (config, ordered event log) -> history (§3.8).

    Keyframes are the odometry timebase; tracklet measurements fire sparsely
    at their anchor keyframes. Keyframe k covers: motion from odometry[k-1],
    then all measurements anchored at k, then health/checkpoint from the
    weighted posterior, then ESS-triggered resampling.

    `observer` is the optional Tier-3 instrumentation seam (`RunObserver`).
    It only ever reads, so an instrumented run and a bare one produce
    identical histories and identical `particle_history_sha256` — which is
    what makes replay-with-instrumentation a faithful reconstruction of the
    original run rather than a different run that resembles it.

    `retrieval_fields` / `retrieval_measurements` are the second observation
    source (CLD-3): dense location-heading score fields from a retrieval
    baseline, one per scored keyframe (`retrieval.ScoreFields` plus
    `structs.RetrievalMeasurement` events). They share this loop's motion
    model, resampling, mode tracking, and health reporting; only the
    per-measurement likelihood differs. The resection proposal stays off for
    retrieval sources (validated), so hypothesis preservation rests on the
    calibration's epsilon floor.
    """
    _validate(config, catalog, odometry, measurements, tables,
              retrieval_fields, retrieval_measurements)
    rng = np.random.default_rng(config.seed)
    belief = init_belief(config, rng)
    tracker = mode_tracker_mod.ModeTracker(config.modes)
    heading_rw_rad = math.radians(config.heading_random_walk_deg)
    heading_rough_rad = math.radians(config.heading_roughening_deg)

    # Tables are static for a run, so their identity posteriors and
    # endorsement masks are computed once, not per measurement.
    weight_cache = {
        tid: _identity_log_weights(table, catalog, config.matcher_recall)
        for tid, table in tables.items()}
    surprise_cache = {
        tid: _surprise_mask(table, _clipped_log_lr(table, catalog))
        for tid, table in tables.items()}
    resp_min = config.min_reported_responsibility

    def _assoc_for(meas):
        """This tracklet's per-particle association state (§5.3), created
        uncommitted on its first epoch; None when persistence is off."""
        if not config.association_persistence:
            return None
        arr = belief.associations.get(meas.tracklet_id)
        if arr is None:
            arr = np.full(belief.n, ASSOC_UNCOMMITTED, dtype=np.int32)
            belief.associations[meas.tracklet_id] = arr
        return arr

    # The retrieval engine is numpy on the belief arrays either way, so it
    # composes with both bearing backends.
    retrieval_engine = (
        retrieval_mod.RetrievalEngine(retrieval_fields, config.retrieval)
        if retrieval_measurements else None)

    if config.measurement_backend == "torch":
        # Lazy import: torch is a heavyweight dependency that only the GPU
        # backend needs; the numpy path must stay importable without it.
        from experimental.overhead_matching.swag.farfield.localization import (  # noqa: E501
            torch_backend)
        engine = torch_backend.TorchMeasurementEngine(
            catalog, weight_cache, seed=config.seed,
            surprise_by_tracklet=surprise_cache)

        def apply_bearing(meas):
            return engine.update(
                belief, meas, config.pi0, per_mode=config.modes.enabled,
                resp_min=resp_min, assoc=_assoc_for(meas),
                renewal_rate=config.association_renewal_rate,
                outlier_rate=config.association_outlier_rate,
                draw_seed=measurement_draw_seed(config.seed, meas))
    elif config.measurement_backend == "numpy":
        def apply_bearing(meas):
            return measurement_update(
                belief, meas, tables[meas.tracklet_id], catalog, config.pi0,
                per_mode=config.modes.enabled,
                log_weight=weight_cache[meas.tracklet_id],
                resp_min=resp_min,
                assoc=_assoc_for(meas),
                renewal_rate=config.association_renewal_rate,
                outlier_rate=config.association_outlier_rate,
                rng=np.random.default_rng(
                    measurement_draw_seed(config.seed, meas)),
                surprise=surprise_cache[meas.tracklet_id])
    else:
        raise ValueError(
            f"unknown measurement_backend {config.measurement_backend!r}; "
            f"expected 'numpy' or 'torch'")

    def apply_measurement(meas):
        if isinstance(meas, structs.RetrievalMeasurement):
            return retrieval_engine.update(belief, meas)
        return apply_bearing(meas)

    if config.measurement_backend == "torch":
        def score_fn(east, north, heading, meas):
            return engine.pose_log_likelihood(east, north, heading, meas,
                                              config.pi0)
    else:
        def score_fn(east, north, heading, meas):
            return pose_log_likelihood(
                east, north, heading, meas, tables[meas.tracklet_id],
                catalog, config.pi0,
                log_weight=weight_cache[meas.tracklet_id])

    def apply_block(keyframe_measurements, kf, pass_index):
        """Apply one keyframe's measurement block in order.

        The pre-update log-weights are copied only when an observer is
        attached: they are exactly what §7.2 attribution needs (each
        tracklet's additive contribution is the difference across this call)
        and pure overhead otherwise.
        """
        block = []
        for meas in keyframe_measurements:
            before = (belief.log_weight.copy() if observer is not None
                      else None)
            block.extend(apply_measurement(meas))
            if observer is not None:
                observer.measurement(kf, meas, before, belief, pass_index)
        return block

    meas_by_kf = {}
    for meas in measurements:
        meas_by_kf.setdefault(meas.anchor_keyframe_idx, []).append(meas)
    # Retrieval fields apply after any bearings at the same keyframe; both
    # are multiplicative, so ordering only fixes the reported intermediate
    # attribution, not the posterior.
    for meas in retrieval_measurements:
        meas_by_kf.setdefault(meas.keyframe_idx, []).append(meas)

    hasher = hashlib.sha256()
    health = []
    checkpoints = {}
    proposal_events = []
    n_keyframes = len(odometry) + 1
    null_history = []
    all_mode_events = []
    low_ess_run = 0
    last_proposal_kf = None
    # When bearings first appear, so the init proposal's wait for an
    # over-determined window has a deadline rather than running forever.
    first_bearing_kf = None

    for kf in range(n_keyframes):
        if kf > 0:
            motion_update(belief, odometry[kf - 1], heading_rw_rad, rng)

        keyframe_measurements = meas_by_kf.get(kf, [])
        # Cluster BEFORE the measurement updates: modes are the hypotheses
        # entering the update, so "mode A believes tracklet 7 is X" refers to
        # a mode that existed before tracklet 7 was seen.
        mode_events = []
        if config.modes.enabled:
            assignment = tracker.update(belief, kf, proposal_events)
            belief.mode_id = assignment.mode_id
            mode_events = assignment.events

        # Snapshot this keyframe's association state: if a proposal fires
        # below, the kept mass must not consume these epochs twice
        # (information-epoch rule) — its state is restored before re-apply.
        assoc_snapshot = {
            m.tracklet_id: belief.associations[m.tracklet_id].copy()
            for m in keyframe_measurements
            if isinstance(m, structs.TrackletMeasurement)
            and m.tracklet_id in belief.associations}

        if observer is not None:
            observer.keyframe_start(kf, belief)

        associations = apply_block(keyframe_measurements, kf, 0)

        belief.log_weight -= special.logsumexp(belief.log_weight)
        current_ess = ess(belief.log_weight)

        # --- mixture proposal: global init and recovery (§5.5) ---
        # Kidnapped detection runs off the same null-share the health stream
        # publishes: when the belief stops explaining the bearings, the
        # evidence lands on the null hypothesis rather than on any landmark.
        if associations:
            # Distress = null share + identity-surprise share: the mass
            # whose explanation of the tracklet the matcher would reject.
            # Null alone misses a displaced belief that re-explains
            # bearings with wrong landmarks — under association
            # persistence those re-explainers ride committed vM weights
            # and null share never rises (§8.4 identity-surprise entry).
            mean_null = float(np.mean([a.null_share + a.surprise_share
                                       for a in associations]))
            null_history.append(
                mean_null > config.proposal.null_share_threshold)
            del null_history[:-config.proposal.null_share_window]
        if current_ess < config.proposal.ess_floor_frac * config.n_particles:
            low_ess_run += 1
        else:
            low_ess_run = 0

        trigger = None
        if keyframe_measurements and first_bearing_kf is None:
            first_bearing_kf = kf
        if config.proposal.enabled and keyframe_measurements:
            refractory_ok = (
                last_proposal_kf is None
                or kf - last_proposal_kf >= config.proposal.refractory_keyframes)
            if (config.proposal.on_init and not proposal_events
                    and isinstance(config.init, structs.UniformBoxInit)
                    and _init_window_is_observable(
                        measurements, kf, first_bearing_kf, config)):
                # Initialize only from a window with enough distinct tracklets
                # to constrain a pose, subject to the bounded-wait fallback.
                # This path applies only to an uninformative uniform prior;
                # informative local priors retain their declared authority.
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
            particle_budget = min(
                config.n_particles,
                int(round(config.proposal.inject_fraction
                          * config.n_particles)))
            result = proposal_mod.propose(
                window, tables, catalog, config.proposal,
                event_id=len(proposal_events), keyframe_idx=kf,
                trigger=trigger, particle_budget=particle_budget)
            gate_passed, gate_best, gate_ref = True, None, None
            # The init trigger bypasses the gate: it only fires for an
            # uninformative prior (§5.5), and the only thing standing is
            # prior mass — which the mode tracker can still report as a
            # "mode" in a small world (the uniform box's cells clear the
            # mass threshold), leaving the gate defending a prior that
            # says nothing.
            if (config.proposal.evidence_gate and result.hypotheses
                    and trigger != "init"):
                gate_passed, gate_best, gate_ref = _evidence_gate(
                    tracker, result, window, config, rng, score_fn, belief,
                    catalog)
            n_injected, kept_idx = ((0, None) if not gate_passed else
                                    inject_proposal(belief, result, config,
                                                    rng))
            proposal_events.append(_proposal_event_record(
                result, n_injected, gate_passed, gate_best, gate_ref))
            # Health links every proposal attempt, including an attempt that
            # the evidence gate rejects.  The event's n_injected/gate_passed
            # fields distinguish an accepted injection from a rejected one.
            event_id = result.event_id
            if observer is not None:
                observer.injection(kf, proposal_events[-1], n_injected)
            if not gate_passed:
                # A rejected event still consumes the refractory period: the
                # trigger condition that fired it persists legitimately in a
                # dense catalog (null share stays high while tracking), so
                # without this the gate would be re-evaluated every keyframe.
                last_proposal_kf = kf
            if n_injected:
                last_proposal_kf = kf
                null_history.clear()
                low_ess_run = 0
                # Restore the kept mass's pre-update association state for
                # this keyframe's tracklets (remapped through the internal
                # resample): the epochs are about to be re-applied to score
                # the injected particles, and consuming them twice would
                # double-count evidence (§5.3 information-epoch rule).
                n_kept = belief.n - n_injected
                for meas in keyframe_measurements:
                    tid = meas.tracklet_id
                    if tid not in belief.associations:
                        continue
                    previous = assoc_snapshot.get(tid)
                    if previous is None:
                        belief.associations[tid][:n_kept] = ASSOC_UNCOMMITTED
                    else:
                        belief.associations[tid][:n_kept] = (
                            previous[kept_idx])
                # Injected particles have not seen this keyframe's bearings,
                # so re-apply them to the whole belief. Injected mass is
                # drawn from a subset of tracklets; scoring everything under
                # the full measurement model is what puts kept and injected
                # particles on the same footing.
                associations = apply_block(keyframe_measurements, kf, 1)
                belief.log_weight -= special.logsumexp(belief.log_weight)
                current_ess = ess(belief.log_weight)
                # Injection changed the belief, so the clusters that entered
                # the update no longer describe it; re-derive them.
                if config.modes.enabled:
                    assignment = tracker.update(belief, kf, proposal_events)
                    belief.mode_id = assignment.mode_id
                    mode_events = mode_events + assignment.events

        # --- retrieval-seeded injection (CLD-3) -------------------------
        # The retrieval analogue of the resection proposal: the field is a
        # global posterior over the declared support, so init and recovery
        # injections sample it directly. Without this, the epsilon floor
        # preserves likelihood but nothing re-seeds PARTICLES in a basin the
        # early resamples emptied.
        retrieval_meas_here = [m for m in keyframe_measurements
                               if isinstance(m, structs.RetrievalMeasurement)]
        if retrieval_engine is not None and retrieval_meas_here:
            rcfg = config.retrieval
            r_trigger = None
            if (rcfg.inject_on_init and not proposal_events
                    and isinstance(config.init, structs.UniformBoxInit)):
                r_trigger = "retrieval_init"
            elif (rcfg.inject_fraction > 0.0
                  and (last_proposal_kf is None
                       or kf - last_proposal_kf
                       >= rcfg.recovery_refractory_keyframes)
                  and low_ess_run >= rcfg.recovery_ess_floor_keyframes):
                r_trigger = "retrieval_ess_floor"
            if r_trigger is not None:
                n_requested = int(round(rcfg.inject_fraction * belief.n))
                meas0 = retrieval_meas_here[0]
                east_i, north_i, heading_i = retrieval_engine.sample_poses(
                    meas0.field_idx, n_requested, rng)
                r_event_id = len(proposal_events)
                n_injected, kept_idx = _splice_injected(
                    belief, east_i, north_i, geo.wrap_rad(heading_i),
                    np.zeros(n_requested, dtype=np.int64), r_event_id, rng)
                proposal_events.append(structs.ProposalEvent(
                    event_id=r_event_id, keyframe_idx=kf, trigger=r_trigger,
                    n_hypotheses=1, n_injected=n_injected,
                    n_tracklets_considered=0, n_combinations_examined=0,
                    n_combinations_skipped=0))
                if observer is not None:
                    observer.injection(kf, proposal_events[-1], n_injected)
                event_id = r_event_id
                last_proposal_kf = kf
                low_ess_run = 0
                # Restore kept-mass association state and re-apply this
                # keyframe's measurements to score kept and injected
                # particles on the same footing (same discipline as the
                # bearing proposal path).
                n_kept = belief.n - n_injected
                for meas in keyframe_measurements:
                    if not isinstance(meas, structs.TrackletMeasurement):
                        continue
                    tid = meas.tracklet_id
                    if tid not in belief.associations:
                        continue
                    previous = assoc_snapshot.get(tid)
                    if previous is None:
                        belief.associations[tid][:n_kept] = ASSOC_UNCOMMITTED
                    else:
                        belief.associations[tid][:n_kept] = (
                            previous[kept_idx])
                associations = apply_block(keyframe_measurements, kf, 1)
                belief.log_weight -= special.logsumexp(belief.log_weight)
                current_ess = ess(belief.log_weight)
                if config.modes.enabled:
                    assignment = tracker.update(belief, kf, proposal_events)
                    belief.mode_id = assignment.mode_id
                    mode_events = mode_events + assignment.events

        mean_e, mean_n, mean_h = metrics.mean_pose(belief)
        map_e, map_n, map_h = metrics.map_pose(
            belief, config.map_cell_size_m)
        modes = _mode_records(belief, tracker) if config.modes.enabled else []
        if config.modes.enabled and associations:
            # An injection keyframe re-clusters AFTER the pass-1 update, so a
            # per-mode association posterior can reference a pre-injection
            # mode the re-clustering retired. The health record keeps only
            # entries its own `modes` list can anchor — the diagnostics join
            # them by mode_id and rightly refuse an orphan.
            live = {mode.mode_id for mode in modes}
            associations = [a for a in associations
                            if a.mode_id is None or a.mode_id in live]
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
            position_std_m=metrics.position_std_m(belief),
            heading_std_deg=metrics.heading_std_deg(belief),
            n_measurements=len(keyframe_measurements),
            proposal_weight_share=float(
                belief.normalized_weights()[belief.proposal_event_id >= 0]
                .sum()),
            proposal_event_id=event_id,
            associations=associations,
            modes=modes,
            mode_entropy_nats=_mode_entropy(modes)))
        if observer is not None:
            observer.keyframe_end(kf, belief, health[-1])
        _hash_belief(hasher, belief)
        # The last keyframe always checkpoints, so checkpoints[-1] is the
        # final weighted posterior — and it is a copy, unaffected by the
        # resampling below.
        if kf % config.checkpoint_every == 0 or kf == n_keyframes - 1:
            snapshot = belief.copy()
            # Association state is replayable from Tier 1 and would add
            # ~n_tracklets * n_particles ints per checkpoint.
            snapshot.associations = {}
            checkpoints[kf] = snapshot

        if resampled:
            resample_before = ((belief.log_weight.copy(), belief.mode_id.copy())
                               if observer is not None else None)
            systematic_resample(belief, rng, config.resample_regularization,
                                config.position_roughening_m,
                                heading_rough_rad,
                                survival_floor=config.resample_survival_floor,
                                survival_min_mass=(
                                    config.resample_survival_min_mass))
            if observer is not None:
                observer.resample(kf, resample_before[0], resample_before[1],
                                  belief)

    return FilterHistory(health=health, checkpoints=checkpoints,
                         particle_history_sha256=hasher.hexdigest(),
                         final_belief=checkpoints[n_keyframes - 1],
                         proposal_events=proposal_events,
                         mode_events=all_mode_events)
