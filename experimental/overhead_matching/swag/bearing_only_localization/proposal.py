"""Mixture proposal: sample poses from the bearings, not the motion model.

Design doc §5.5. Over a large area a motion-model-only proposal needs
hopeless particle counts, but bearings invert directly, and they do so at
three levels of completeness:

  landmarks | constraint set                          | pins
  ----------|-----------------------------------------|--------------------
      1     | 2-D: position free, heading determined  | heading
      2     | 1-D: circular arc, heading determined   | heading + 1 position
      3     | 0-D: discrete fixes                     | everything

Each is a subset of the one above, and all three are legitimate proposals —
a particle filter needs *samples*, not a solution, so a 1-D or 2-D
constraint set is as usable as a point fix. Even the single-landmark case
earns its keep: it collapses the heading axis exactly, which is worth roughly
2*pi/sigma in particle count against a uniform prior.

Candidate landmark identities come from the matcher's CompatibilityTable —
each tracklet's top-k entries by log_lr — rather than from a catalog type
index. That keeps the proposal matcher-agnostic and bounded by k, and means
the §5.5 type-pair index is a scaling optimization for when tables are
uninformative, not a prerequisite.

Identity candidates are enumerated in full rather than truncated: a real
matcher emits disjunctions ("one of these 41 storage tanks", all tied at the
same log_lr), and taking an arbitrary top-k from a tie means resecting from a
coin-flip identity. The cost is controlled by per-kind combination budgets
spent cheapest-first, and a tracklet combination that does not fit is skipped
whole and counted — never enumerated part-way.

Bearings from nearby keyframes are treated as simultaneous, because the
information-epoch design staggers tracklet anchors and a single keyframe
usually carries one bearing. The error is bounded by translation/range: over
one keyframe (~25 m) against a 2.5 km landmark it is ~0.6 deg, under the
bearing noise the resection already absorbs. It costs proposal *recall*, not
posterior accuracy — every injected particle is re-scored under the exact
measurement model.

`[CONTRACT]` Every proposed pose carries provenance: which proposal event
produced it, from which tracklets, under which landmark hypothesis. Modes
inherit it from their founding particles, and it is what makes "where did
this wrong mode come from" answerable in one click (§7.4).
"""

import dataclasses
import itertools
import math

import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    geodesy,
    resection,
    structs,
)

TRIPLE, PAIR, SINGLE = "triple", "pair", "single"


@dataclasses.dataclass(frozen=True)
class Hypothesis:
    """A constraint set on pose, plus the provenance that produced it.

    `sample` draws poses from it; the shape of the draw is what differs
    between the three kinds.
    """
    kind: str
    tracklet_ids: tuple
    landmark_ids: tuple
    # Ranking key within a kind: how well the pose explains its bearings.
    # Exact by construction for pairs and singles.
    residual_rad: float = 0.0

    def sample(self, n: int, config, rng):
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class PointHypothesis(Hypothesis):
    """Three-landmark fix, jittered by ITS OWN resection uncertainty.

    `position_sigma_m` is estimated at generation time as
    sigma_bearing * range: a bearing that is coarse, or landmarks that are
    far, both make the fix imprecise in exactly that proportion. Sampling a
    diffuse fix as a tight blob would assert a precision the geometry never
    had.
    """
    east_m: float = 0.0
    north_m: float = 0.0
    heading_rad: float = 0.0
    position_sigma_m: float = 0.0
    heading_sigma_deg: float = 0.0

    def sample(self, n, config, rng):
        spread = min(max(self.position_sigma_m, config.injection_sigma_m),
                     config.max_injection_sigma_m)
        heading_spread = math.radians(max(
            self.heading_sigma_deg, config.injection_heading_sigma_deg))
        return (self.east_m + rng.normal(0.0, spread, n),
                self.north_m + rng.normal(0.0, spread, n),
                self.heading_rad + rng.normal(0.0, heading_spread, n))


@dataclasses.dataclass(frozen=True)
class ArcHypothesis(Hypothesis):
    """Two-landmark arc: position on a circle, heading from a bearing."""
    arc: resection.ArcHypothesis = None
    landmark_a: tuple = (0.0, 0.0)
    landmark_b: tuple = (0.0, 0.0)
    bearing_a_rad: float = 0.0

    def sample(self, n, config, rng):
        east, north = resection.sample_arc(
            self.arc, self.landmark_a[0], self.landmark_a[1],
            self.landmark_b[0], self.landmark_b[1], n, rng)
        if east.size == 0:
            return east, north, east
        east = east + rng.normal(0.0, config.injection_sigma_m, east.size)
        north = north + rng.normal(0.0, config.injection_sigma_m, north.size)
        heading = _heading_array(east, north, self.landmark_a,
                                 self.bearing_a_rad)
        return east, north, heading + rng.normal(
            0.0, math.radians(config.injection_heading_sigma_deg), east.size)


@dataclasses.dataclass(frozen=True)
class VisibilityDiscHypothesis(Hypothesis):
    """One landmark: position anywhere it could be seen from, heading then
    determined by the bearing.

    Sampled uniformly in AREA (r = R sqrt(u)) rather than uniformly in
    range: the latter piles particles up near the landmark and silently
    biases the proposal density towards short ranges.
    """
    landmark: tuple = (0.0, 0.0)
    bearing_rad: float = 0.0
    max_range_m: float = 0.0

    def sample(self, n, config, rng):
        radius = self.max_range_m * np.sqrt(rng.uniform(0.0, 1.0, n))
        angle = rng.uniform(-math.pi, math.pi, n)
        east = self.landmark[0] + radius * np.sin(angle)
        north = self.landmark[1] + radius * np.cos(angle)
        heading = _heading_array(east, north, self.landmark, self.bearing_rad)
        return east, north, heading + rng.normal(
            0.0, math.radians(config.injection_heading_sigma_deg), n)


def _heading_array(east, north, landmark, bearing_rad):
    """Heading implied by seeing `landmark` at `bearing_rad` from each pose."""
    return geodesy.wrap_rad(
        geodesy.compass_bearing_rad(landmark[0] - east, landmark[1] - north)
        - bearing_rad)


@dataclasses.dataclass
class ProposalResult:
    event_id: int
    keyframe_idx: int
    trigger: str
    hypotheses: list
    # Observability, not decoration: a proposal that silently generated
    # nothing looks identical to one that never fired, and a capped
    # combination count looks identical to full coverage (§8.4 "no silent
    # caps").
    n_tracklets_considered: int
    n_combinations_examined: int
    n_combinations_skipped: int

    def counts_by_kind(self) -> dict:
        counts = {}
        for hypothesis in self.hypotheses:
            counts[hypothesis.kind] = counts.get(hypothesis.kind, 0) + 1
        return counts


def _top_k_landmarks(table: structs.CompatibilityTable, catalog, k: int):
    """The k landmarks this tracklet is most compatible with (§6 seam).

    Entries at or below `default_log_lr` are no better than an unlisted
    landmark, so they are never worth resecting against.
    """
    scored = [(min(max(e.log_lr, table.clip_lo), table.clip_hi),
               e.landmark_id)
              for e in table.entries
              if e.landmark_id in catalog
              and e.log_lr > table.default_log_lr]
    scored.sort(reverse=True)
    return [landmark_id for _, landmark_id in scored[:k]]


def _position(catalog, landmark_id):
    index = catalog.index_of(landmark_id)
    return (float(catalog.east_m[index]), float(catalog.north_m[index]))


def _collect_candidates(measurements, tables, catalog, config):
    """Most recent measurement per tracklet, with its candidate landmarks."""
    latest = {}
    for meas in measurements:
        if meas.tracklet_id not in tables:
            continue
        previous = latest.get(meas.tracklet_id)
        if previous is None or meas.anchor_keyframe_idx > previous.anchor_keyframe_idx:
            latest[meas.tracklet_id] = meas
    # Tie-break on tracklet_id, not on input order: equal-kappa measurements
    # are the common case (one tracker, one noise model), and letting their
    # arrival order pick which ones survive `max_tracklets` would make the
    # filter's output depend on measurement ordering — breaking both T-F7
    # and the replay contract.
    ordered = sorted(latest.values(),
                     key=lambda m: (-m.kappa, m.tracklet_id))
    ordered = ordered[:config.max_tracklets]

    candidates = []
    for meas in ordered:
        landmark_ids = _top_k_landmarks(tables[meas.tracklet_id], catalog,
                                        config.top_k_landmarks)
        if landmark_ids:
            candidates.append((meas, landmark_ids))
    return candidates


def propose(measurements, tables, catalog, config: structs.ProposalConfig,
            event_id: int, keyframe_idx: int, trigger: str) -> ProposalResult:
    """Build pose hypotheses from a window of recent tracklet bearings.

    `measurements` should span at most a few keyframes; bearings within the
    window are treated as simultaneous (see module docstring).
    """
    candidates = _collect_candidates(measurements, tables, catalog, config)
    hypotheses = []
    examined = {SINGLE: 0, PAIR: 0, TRIPLE: 0}
    skipped = {SINGLE: 0, PAIR: 0, TRIPLE: 0}
    budget = {SINGLE: config.max_combinations_single,
              PAIR: config.max_combinations_pair,
              TRIPLE: config.max_combinations_triple}

    def afford(kind, cost):
        """Take `cost` combinations from a kind's budget, all or nothing.

        Whole-combination accounting is the point: enumerating half of a
        41-way tie would resect from an arbitrary subset of identities,
        which is the failure the budget exists to prevent.
        """
        if examined[kind] + cost > budget[kind]:
            skipped[kind] += cost
            return False
        examined[kind] += cost
        return True

    def tolerance_for(group):
        sigma = 1.0 / math.sqrt(max(min(m.kappa for m in group), 1e-9))
        return min(config.residual_tolerance_sigma * sigma,
                   math.radians(config.max_residual_tolerance_deg))

    # --- one landmark: heading collapsed, position over its visibility disc.
    # Cheapest and always available, so it is funded first.
    for meas, landmark_ids in candidates:
        if not afford(SINGLE, len(landmark_ids)):
            continue
        for landmark_id in landmark_ids:
            hypotheses.append(VisibilityDiscHypothesis(
                kind=SINGLE,
                tracklet_ids=(meas.tracklet_id,),
                landmark_ids=(landmark_id,),
                landmark=_position(catalog, landmark_id),
                bearing_rad=math.radians(meas.bearing_body_deg),
                max_range_m=float(catalog.max_visible_range_m[
                    catalog.index_of(landmark_id)])))

    # --- two landmarks: inscribed-angle arcs ---
    for (meas_a, ids_a), (meas_b, ids_b) in itertools.combinations(
            candidates, 2):
        if not afford(PAIR, len(ids_a) * len(ids_b)):
            continue
        subtended = resection.subtended_angle_rad(
            math.radians(meas_a.bearing_body_deg),
            math.radians(meas_b.bearing_body_deg))
        for landmark_a, landmark_b in itertools.product(ids_a, ids_b):
            if landmark_a == landmark_b:
                continue
            position_a = _position(catalog, landmark_a)
            position_b = _position(catalog, landmark_b)
            # Signed: with both bearings identified, only one side of the
            # baseline is consistent, and injecting into the other puts half
            # the particles where the measurement flatly contradicts them.
            for arc in resection.arcs_for_signed_angle(
                    position_a[0], position_a[1], position_b[0],
                    position_b[1], subtended):
                hypotheses.append(ArcHypothesis(
                    kind=PAIR,
                    tracklet_ids=(meas_a.tracklet_id, meas_b.tracklet_id),
                    landmark_ids=(landmark_a, landmark_b),
                    arc=arc, landmark_a=position_a, landmark_b=position_b,
                    bearing_a_rad=math.radians(meas_a.bearing_body_deg)))

    # --- three landmarks: discrete fixes ---
    for trio in itertools.combinations(candidates, 3):
        trio_measurements = [c[0] for c in trio]
        cost = 1
        for _, ids in trio:
            cost *= len(ids)
        if not afford(TRIPLE, cost):
            continue
        tolerance = tolerance_for(trio_measurements)
        for landmark_trio in itertools.product(*[c[1] for c in trio]):
            if len(set(landmark_trio)) < 3:
                continue  # two tracklets cannot be the same landmark
            positions = [_position(catalog, lm) for lm in landmark_trio]
            bearings = [math.radians(m.bearing_body_deg)
                        for m in trio_measurements]
            for solution in resection.resect_three(positions, bearings,
                                                   tolerance):
                sigma_bearing = 1.0 / math.sqrt(
                    max(min(m.kappa for m in trio_measurements), 1e-9))
                ranges = [math.hypot(le - solution.east_m,
                                     ln - solution.north_m)
                          for le, ln in positions]
                hypotheses.append(PointHypothesis(
                    kind=TRIPLE,
                    tracklet_ids=tuple(m.tracklet_id
                                       for m in trio_measurements),
                    landmark_ids=tuple(landmark_trio),
                    residual_rad=solution.residual_rad,
                    east_m=solution.east_m, north_m=solution.north_m,
                    heading_rad=solution.heading_rad,
                    position_sigma_m=sigma_bearing * float(
                        np.median(ranges)),
                    heading_sigma_deg=math.degrees(sigma_bearing)))

    # Cap per kind so the many broad hypotheses cannot crowd out the few
    # sharp ones (or the reverse).
    kept = []
    for kind in (TRIPLE, PAIR, SINGLE):
        of_kind = sorted((h for h in hypotheses if h.kind == kind),
                         key=lambda h: (h.residual_rad, h.tracklet_ids,
                                        h.landmark_ids))
        kept.extend(of_kind[:config.max_hypotheses_per_kind])
    return ProposalResult(
        event_id=event_id, keyframe_idx=keyframe_idx, trigger=trigger,
        hypotheses=kept, n_tracklets_considered=len(candidates),
        n_combinations_examined=sum(examined.values()),
        n_combinations_skipped=sum(skipped.values()))


def sample_particles(result: ProposalResult, n_particles: int,
                     config: structs.ProposalConfig, rng: np.random.Generator):
    """Draw particles across the hypothesis set.

    Mass is split across *kinds* by configured share (renormalized over the
    kinds actually present), then evenly within a kind. Splitting by kind
    keeps a handful of sharp three-landmark fixes from being swamped by the
    many broad single-landmark discs, while still keeping mass on the broad
    ones in case the sharp fixes are all wrong. Within a kind, mass is even
    rather than residual-weighted: the residual measures geometric
    self-consistency, and letting it also set the prior would count the same
    bearings twice — deciding between hypotheses is the measurement update's
    job.

    Returns (east_m, north_m, heading_rad, hypothesis_index) arrays.
    """
    empty = (np.zeros(0), np.zeros(0), np.zeros(0),
             np.zeros(0, dtype=np.int64))
    if not result.hypotheses or n_particles <= 0:
        return empty

    shares = {TRIPLE: config.share_triple, PAIR: config.share_pair,
              SINGLE: config.share_single}
    present = {kind: shares[kind] for kind in shares
               if any(h.kind == kind for h in result.hypotheses)
               and shares[kind] > 0.0}
    if not present:
        return empty
    total_share = sum(present.values())

    east_parts, north_parts, heading_parts, index_parts = [], [], [], []
    allocated = 0
    kinds = sorted(present)
    for position, kind in enumerate(kinds):
        if position == len(kinds) - 1:
            n_kind = n_particles - allocated  # absorb rounding
        else:
            n_kind = int(round(n_particles * present[kind] / total_share))
        allocated += n_kind
        if n_kind <= 0:
            continue
        indices = [i for i, h in enumerate(result.hypotheses)
                   if h.kind == kind]
        chosen = rng.integers(0, len(indices), size=n_kind)
        for slot, count in zip(*np.unique(chosen, return_counts=True)):
            hypothesis = result.hypotheses[indices[int(slot)]]
            east, north, heading = hypothesis.sample(int(count), config, rng)
            if east.size == 0:
                continue
            east_parts.append(east)
            north_parts.append(north)
            heading_parts.append(heading)
            index_parts.append(np.full(east.size, indices[int(slot)],
                                       dtype=np.int64))
    if not east_parts:
        return empty
    return (np.concatenate(east_parts), np.concatenate(north_parts),
            geodesy.wrap_rad(np.concatenate(heading_parts)),
            np.concatenate(index_parts))
