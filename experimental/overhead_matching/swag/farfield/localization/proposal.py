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

Candidate identities come from every CompatibilityTable entry whose clipped
score is meaningfully better than the table default. Small correspondence
spaces are enumerated. Large spaces are sampled deterministically from the
complete compatibility distribution, with systematic strata preserving fair
coverage of tied identities. The injected particle count is the primary
budget: it determines how many distinct pose solutions can be populated at a
useful density and therefore how many correspondence tuples are examined.

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

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.localization import (
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
    # Normalized matcher evidence represented by this selected tuple.
    compatibility_mass: float = 1.0

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
    arc_length_m: float = 0.0

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
    return geo.wrap_rad(
        geo.compass_bearing_rad(landmark[0] - east, landmark[1] - north)
        - bearing_rad)


@dataclasses.dataclass
class ProposalResult:
    event_id: int
    keyframe_idx: int
    trigger: str
    hypotheses: list
    particle_budget: int
    n_tracklets_considered: int
    n_combinations_total: int
    n_combinations_enumerated: int
    n_combinations_sampled: int
    n_combinations_geometry_pruned: int
    n_partially_represented_ties: int
    n_solution_clusters_merged: int
    represented_compatibility_mass: float

    @property
    def n_combinations_examined(self) -> int:
        return self.n_combinations_enumerated + self.n_combinations_sampled

    @property
    def n_combinations_skipped(self) -> int:
        return self.n_combinations_total - self.n_combinations_examined

    def counts_by_kind(self) -> dict:
        counts = {}
        for hypothesis in self.hypotheses:
            counts[hypothesis.kind] = counts.get(hypothesis.kind, 0) + 1
        return counts


@dataclasses.dataclass(frozen=True)
class _Candidate:
    landmark_id: str
    score: float
    weight: float


@dataclasses.dataclass(frozen=True)
class _TupleSpace:
    kind: str
    measurements: tuple
    candidates: tuple

    @property
    def total_possible(self) -> int:
        return math.prod(len(group) for group in self.candidates)

    @property
    def total_mass(self) -> float:
        return math.prod(sum(item.weight for item in group)
                         for group in self.candidates)


def _candidate_landmarks(table: structs.CompatibilityTable, catalog):
    """All catalog candidates whose clipped score beats the clipped default."""
    default = min(max(table.default_log_lr, table.clip_lo), table.clip_hi)
    by_id = {}
    for entry in table.entries:
        if entry.landmark_id not in catalog:
            continue
        score = min(max(entry.log_lr, table.clip_lo), table.clip_hi)
        if score <= default:
            continue
        previous = by_id.get(entry.landmark_id)
        if previous is None or score > previous:
            by_id[entry.landmark_id] = score
    return tuple(
        _Candidate(landmark_id, score, math.exp(score - default))
        for landmark_id, score in sorted(
            by_id.items(), key=lambda item: (-item[1], item[0])))


def _position(catalog, landmark_id):
    index = catalog.index_of(landmark_id)
    return (float(catalog.east_m[index]), float(catalog.north_m[index]))


def _visibility_range(catalog, landmark_id) -> float:
    return float(catalog.max_visible_range_m[catalog.index_of(landmark_id)])


def _collect_candidates(measurements, tables, catalog, config):
    """Most recent measurement per tracklet, with every useful candidate."""
    latest = {}
    for measurement in measurements:
        if measurement.tracklet_id not in tables:
            continue
        previous = latest.get(measurement.tracklet_id)
        if (previous is None
                or measurement.anchor_keyframe_idx
                > previous.anchor_keyframe_idx):
            latest[measurement.tracklet_id] = measurement
    ordered = sorted(latest.values(),
                     key=lambda item: (-item.kappa, item.tracklet_id))
    ordered = ordered[:config.max_tracklets]

    candidates = []
    for measurement in ordered:
        identities = _candidate_landmarks(
            tables[measurement.tracklet_id], catalog)
        if identities:
            candidates.append((measurement, identities))
    return candidates


def _tuple_spaces(candidates) -> dict[str, list[_TupleSpace]]:
    spaces = {SINGLE: [], PAIR: [], TRIPLE: []}
    for size, kind in ((1, SINGLE), (2, PAIR), (3, TRIPLE)):
        for group in itertools.combinations(candidates, size):
            spaces[kind].append(_TupleSpace(
                kind=kind,
                measurements=tuple(item[0] for item in group),
                candidates=tuple(item[1] for item in group)))
    return spaces


_KIND_ORDER = (TRIPLE, PAIR, SINGLE)


def _systematic_counts(total: int, weights) -> np.ndarray:
    """Deterministic systematic allocation with an exact integer total."""
    weights = np.asarray(weights, dtype=np.float64)
    if total <= 0 or weights.size == 0:
        return np.zeros(weights.size, dtype=np.int64)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("allocation weights must be finite and nonnegative")
    if not np.any(weights > 0.0):
        weights = np.ones(weights.size, dtype=np.float64)
    cumulative = np.cumsum(weights / weights.sum())
    cumulative[-1] = 1.0
    positions = (np.arange(total, dtype=np.float64) + 0.5) / total
    selected = np.searchsorted(cumulative, positions, side="left")
    return np.bincount(selected, minlength=weights.size).astype(np.int64)


def _kind_particle_counts(total: int, kinds, config) -> dict[str, int]:
    kinds = [kind for kind in _KIND_ORDER if kind in kinds]
    if total <= 0 or not kinds:
        return {kind: 0 for kind in kinds}
    shares = {TRIPLE: config.share_triple, PAIR: config.share_pair,
              SINGLE: config.share_single}
    counts = _systematic_counts(total, [shares[kind] for kind in kinds])
    return {kind: int(count) for kind, count in zip(kinds, counts)}


def _particle_floor(kind: str, config) -> int:
    return {TRIPLE: config.min_particles_point_fix,
            PAIR: config.min_particles_arc,
            SINGLE: config.min_particles_single}[kind]


def _max_active_solutions(kind: str, particle_count: int, config) -> int:
    if particle_count <= 0:
        return 0
    return max(1, particle_count // _particle_floor(kind, config))


def _decode_flat_index(index: int, groups) -> tuple[_Candidate, ...]:
    selected = [None] * len(groups)
    for dimension in range(len(groups) - 1, -1, -1):
        selected[dimension] = groups[dimension][index % len(groups[dimension])]
        index //= len(groups[dimension])
    return tuple(selected)


def _coprime_stride(size: int, dimension: int) -> int:
    if size <= 1:
        return 1
    stride = 2 * dimension + 1
    while math.gcd(stride, size) != 1:
        stride += 1
    return stride


def _sample_product(space: _TupleSpace, count: int):
    """Systematic, deterministic tuples from a factorized distribution."""
    count = min(max(0, count), space.total_possible)
    if count == 0:
        return []
    dimensions = []
    for dimension, group in enumerate(space.candidates):
        weights = np.asarray([item.weight for item in group], dtype=np.float64)
        cumulative = np.cumsum(weights / weights.sum())
        cumulative[-1] = 1.0
        stride = _coprime_stride(count, dimension)
        slots = (np.arange(count, dtype=np.int64) * stride) % count
        positions = (slots.astype(np.float64) + 0.5) / count
        dimensions.append(np.searchsorted(cumulative, positions, side="left"))

    tuples = []
    seen = set()
    for row in range(count):
        candidate_tuple = tuple(
            space.candidates[dimension][int(dimensions[dimension][row])]
            for dimension in range(len(space.candidates)))
        key = tuple(item.landmark_id for item in candidate_tuple)
        if key not in seen:
            seen.add(key)
            tuples.append(candidate_tuple)

    # Strongly peaked factors can make multiple systematic strata land on the
    # same tuple. Fill the remaining strata at evenly spaced mixed-radix ranks;
    # this preserves bounded work while ensuring exact, reproducible coverage.
    for slot in range(count):
        if len(tuples) == count:
            break
        flat_index = min(
            space.total_possible - 1,
            int((slot + 0.5) * space.total_possible / count))
        candidate_tuple = _decode_flat_index(flat_index, space.candidates)
        key = tuple(item.landmark_id for item in candidate_tuple)
        if key not in seen:
            seen.add(key)
            tuples.append(candidate_tuple)
    if len(tuples) < count:
        for flat_index in range(space.total_possible):
            if len(tuples) == count:
                break
            candidate_tuple = _decode_flat_index(flat_index, space.candidates)
            key = tuple(item.landmark_id for item in candidate_tuple)
            if key not in seen:
                seen.add(key)
                tuples.append(candidate_tuple)
    return tuples


def _partially_represented_ties(space: _TupleSpace, tuples) -> int:
    partial = 0
    for dimension, group in enumerate(space.candidates):
        represented = {items[dimension].landmark_id for items in tuples}
        by_score = {}
        for item in group:
            by_score.setdefault(item.score, set()).add(item.landmark_id)
        for tied in by_score.values():
            covered = len(tied & represented)
            if len(tied) > 1 and 0 < covered < len(tied):
                partial += 1
    return partial


def _tuple_is_geometrically_possible(space: _TupleSpace, identity_tuple,
                                      catalog) -> bool:
    landmark_ids = tuple(item.landmark_id for item in identity_tuple)
    if len(set(landmark_ids)) != len(landmark_ids):
        return False
    positions = [_position(catalog, landmark_id) for landmark_id in landmark_ids]
    ranges = [_visibility_range(catalog, landmark_id)
              for landmark_id in landmark_ids]
    for first, second in itertools.combinations(range(len(positions)), 2):
        baseline = math.dist(positions[first], positions[second])
        if baseline < resection.MIN_BASELINE_M:
            return False
        if baseline > ranges[first] + ranges[second]:
            return False
    if space.kind in (PAIR, TRIPLE):
        usable_pairs = 0
        bearings = [math.radians(item.bearing_forward_cw_deg)
                    for item in space.measurements]
        for first, second in itertools.combinations(range(len(bearings)), 2):
            gamma = abs(resection.subtended_angle_rad(
                bearings[first], bearings[second]))
            if (resection.MIN_SUBTENDED_RAD < gamma
                    < math.pi - resection.MIN_SUBTENDED_RAD):
                usable_pairs += 1
        if usable_pairs < (1 if space.kind == PAIR else 2):
            return False
    return True


def _point_is_visible(solution, positions, landmark_ids, catalog) -> bool:
    if not all(math.isfinite(value) for value in (
            solution.east_m, solution.north_m, solution.heading_rad,
            solution.residual_rad)):
        return False
    return all(
        math.hypot(position[0] - solution.east_m,
                   position[1] - solution.north_m)
        <= _visibility_range(catalog, landmark_id)
        for position, landmark_id in zip(positions, landmark_ids))


def _same_solution(first, second, config) -> bool:
    position_tolerance = config.solution_cluster_position_m
    heading_tolerance = math.radians(config.solution_cluster_heading_deg)
    if isinstance(first, PointHypothesis) and isinstance(second, PointHypothesis):
        return (math.hypot(first.east_m - second.east_m,
                           first.north_m - second.north_m)
                <= position_tolerance
                and abs(float(geo.wrap_rad(
                    first.heading_rad - second.heading_rad)))
                <= heading_tolerance)
    if isinstance(first, ArcHypothesis) and isinstance(second, ArcHypothesis):
        return (first.arc.side == second.arc.side
                and math.hypot(first.arc.center_east_m
                               - second.arc.center_east_m,
                               first.arc.center_north_m
                               - second.arc.center_north_m)
                <= position_tolerance
                and abs(first.arc.radius_m - second.arc.radius_m)
                <= position_tolerance
                and abs(float(geo.wrap_rad(
                    first.bearing_a_rad - second.bearing_a_rad)))
                <= heading_tolerance)
    if (isinstance(first, VisibilityDiscHypothesis)
            and isinstance(second, VisibilityDiscHypothesis)):
        return (math.dist(first.landmark, second.landmark)
                <= position_tolerance
                and abs(float(geo.wrap_rad(
                    first.bearing_rad - second.bearing_rad)))
                <= heading_tolerance)
    return False


def _cluster_hypotheses(hypotheses, config):
    ordered = sorted(
        hypotheses,
        key=lambda item: (-item.compatibility_mass, item.residual_rad,
                          item.tracklet_ids, item.landmark_ids))
    clusters = []
    merged = 0
    for hypothesis in ordered:
        if any(_same_solution(hypothesis, existing, config)
               for existing in clusters):
            merged += 1
        else:
            clusters.append(hypothesis)
    return clusters, merged


def _solution_distance(first, second, config) -> float:
    position_scale = max(config.solution_cluster_position_m, 1e-9)
    heading_scale = max(math.radians(config.solution_cluster_heading_deg),
                        1e-9)
    if isinstance(first, PointHypothesis) and isinstance(second, PointHypothesis):
        return (math.hypot(first.east_m - second.east_m,
                           first.north_m - second.north_m) / position_scale
                + abs(float(geo.wrap_rad(
                    first.heading_rad - second.heading_rad))) / heading_scale)
    if isinstance(first, ArcHypothesis) and isinstance(second, ArcHypothesis):
        return (math.hypot(first.arc.center_east_m
                           - second.arc.center_east_m,
                           first.arc.center_north_m
                           - second.arc.center_north_m) / position_scale
                + abs(first.arc.radius_m - second.arc.radius_m) / position_scale)
    if (isinstance(first, VisibilityDiscHypothesis)
            and isinstance(second, VisibilityDiscHypothesis)):
        return (math.dist(first.landmark, second.landmark) / position_scale
                + abs(float(geo.wrap_rad(
                    first.bearing_rad - second.bearing_rad))) / heading_scale)
    return 1.0


def _select_diverse(hypotheses, limit: int, config):
    if limit <= 0 or not hypotheses:
        return []
    remaining = sorted(
        hypotheses,
        key=lambda item: (-item.compatibility_mass, item.residual_rad,
                          item.tracklet_ids, item.landmark_ids))
    selected = [remaining.pop(0)]
    while remaining and len(selected) < limit:
        max_mass = max(item.compatibility_mass for item in remaining) or 1.0
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                remaining[index].compatibility_mass / max_mass
                + config.pose_diversity_weight * min(
                    min(_solution_distance(remaining[index], chosen, config)
                        for chosen in selected), 10.0),
                -remaining[index].residual_rad,
                remaining[index].tracklet_ids,
                remaining[index].landmark_ids))
        selected.append(remaining.pop(best_index))
    return sorted(selected, key=lambda item: (
        item.residual_rad, item.tracklet_ids, item.landmark_ids))


def propose(measurements, tables, catalog, config: structs.ProposalConfig,
            event_id: int, keyframe_idx: int, trigger: str, *,
            particle_budget: int) -> ProposalResult:
    """Build a bounded, deterministic hypothesis set for an injection budget."""
    if (isinstance(particle_budget, (bool, np.bool_))
            or not isinstance(particle_budget, (int, np.integer))
            or particle_budget < 0):
        raise ValueError("particle_budget must be a nonnegative integer")
    particle_budget = int(particle_budget)
    candidates = _collect_candidates(measurements, tables, catalog, config)
    spaces_by_kind = _tuple_spaces(candidates)
    present_space_kinds = {
        kind for kind, spaces in spaces_by_kind.items() if spaces}
    particle_counts = _kind_particle_counts(
        particle_budget, present_space_kinds, config)
    max_active = {
        kind: _max_active_solutions(kind, particle_counts.get(kind, 0), config)
        for kind in _KIND_ORDER}

    total = sum(space.total_possible
                for spaces in spaces_by_kind.values() for space in spaces)
    enumerated = 0
    sampled = 0
    geometry_pruned = 0
    partial_ties = 0
    generated = []

    present_shares = {
        kind: share for kind, share in (
            (TRIPLE, config.share_triple), (PAIR, config.share_pair),
            (SINGLE, config.share_single))
        if spaces_by_kind[kind]}
    share_total = sum(present_shares.values())
    if share_total <= 0.0 and present_shares:
        present_shares = {kind: 1.0 for kind in present_shares}
        share_total = float(len(present_shares))

    def tolerance_for(group):
        sigma = 1.0 / math.sqrt(max(min(item.kappa for item in group), 1e-9))
        return min(config.residual_tolerance_sigma * sigma,
                   math.radians(config.max_residual_tolerance_deg))

    for kind in _KIND_ORDER:
        spaces = spaces_by_kind[kind]
        if not spaces:
            continue
        kind_total_mass = sum(space.total_mass for space in spaces)
        kind_mass_scale = ((present_shares[kind] / share_total)
                           / kind_total_mass)
        large_spaces = [space for space in spaces
                        if space.total_possible > config.exhaustive_tuple_limit]
        large_budget = (max_active[kind]
                        * config.tuple_samples_per_active_solution)
        large_counts = _systematic_counts(
            large_budget, [space.total_mass for space in large_spaces])
        large_count_by_id = {
            id(space): min(int(count), space.total_possible)
            for space, count in zip(large_spaces, large_counts)}

        for space in spaces:
            exhaustive = (
                space.total_possible <= config.exhaustive_tuple_limit)
            if exhaustive:
                tuples = list(itertools.product(*space.candidates))
                enumerated += len(tuples)
            else:
                tuples = _sample_product(
                    space, large_count_by_id.get(id(space), 0))
                sampled += len(tuples)
                partial_ties += _partially_represented_ties(space, tuples)

            for identity_tuple in tuples:
                if not _tuple_is_geometrically_possible(
                        space, identity_tuple, catalog):
                    geometry_pruned += 1
                    continue
                landmark_ids = tuple(item.landmark_id
                                     for item in identity_tuple)
                tuple_mass = (math.prod(item.weight for item in identity_tuple)
                              * kind_mass_scale)
                if kind == SINGLE:
                    measurement = space.measurements[0]
                    landmark_id = landmark_ids[0]
                    generated.append(VisibilityDiscHypothesis(
                        kind=SINGLE,
                        tracklet_ids=(measurement.tracklet_id,),
                        landmark_ids=landmark_ids,
                        compatibility_mass=tuple_mass,
                        landmark=_position(catalog, landmark_id),
                        bearing_rad=math.radians(
                            measurement.bearing_forward_cw_deg),
                        max_range_m=_visibility_range(catalog, landmark_id)))
                    continue

                positions = [_position(catalog, landmark_id)
                             for landmark_id in landmark_ids]
                bearings = [math.radians(item.bearing_forward_cw_deg)
                            for item in space.measurements]
                if kind == PAIR:
                    arcs = resection.arcs_for_signed_angle(
                        positions[0][0], positions[0][1],
                        positions[1][0], positions[1][1],
                        resection.subtended_angle_rad(*bearings))
                    if not arcs:
                        geometry_pruned += 1
                        continue
                    arc = arcs[0]
                    generated.append(ArcHypothesis(
                        kind=PAIR,
                        tracklet_ids=tuple(item.tracklet_id
                                           for item in space.measurements),
                        landmark_ids=landmark_ids,
                        compatibility_mass=tuple_mass,
                        arc=arc, landmark_a=positions[0],
                        landmark_b=positions[1], bearing_a_rad=bearings[0],
                        arc_length_m=resection.valid_arc_length_m(
                            arc, positions[0][0], positions[0][1],
                            positions[1][0], positions[1][1])))
                    continue

                solutions = resection.resect_three(
                    positions, bearings, tolerance_for(space.measurements))
                visible_solutions = [
                    solution for solution in solutions
                    if _point_is_visible(
                        solution, positions, landmark_ids, catalog)]
                if not visible_solutions:
                    geometry_pruned += 1
                    continue
                solution_mass = tuple_mass / len(visible_solutions)
                sigma_bearing = 1.0 / math.sqrt(max(
                    min(item.kappa for item in space.measurements), 1e-9))
                for solution in visible_solutions:
                    ranges = [math.hypot(east - solution.east_m,
                                         north - solution.north_m)
                              for east, north in positions]
                    generated.append(PointHypothesis(
                        kind=TRIPLE,
                        tracklet_ids=tuple(item.tracklet_id
                                           for item in space.measurements),
                        landmark_ids=landmark_ids,
                        residual_rad=solution.residual_rad,
                        compatibility_mass=solution_mass,
                        east_m=solution.east_m, north_m=solution.north_m,
                        heading_rad=solution.heading_rad,
                        position_sigma_m=sigma_bearing * float(
                            np.median(ranges)),
                        heading_sigma_deg=math.degrees(sigma_bearing)))

    selected = []
    clusters_merged = 0
    generated_kinds = {item.kind for item in generated}
    final_particle_counts = _kind_particle_counts(
        particle_budget, generated_kinds, config)
    for kind in _KIND_ORDER:
        clustered, merged = _cluster_hypotheses(
            [item for item in generated if item.kind == kind], config)
        clusters_merged += merged
        limit = _max_active_solutions(
            kind, final_particle_counts.get(kind, 0), config)
        selected.extend(_select_diverse(clustered, limit, config))

    represented_mass = min(
        1.0, sum(item.compatibility_mass for item in selected))
    return ProposalResult(
        event_id=event_id, keyframe_idx=keyframe_idx, trigger=trigger,
        hypotheses=selected, particle_budget=particle_budget,
        n_tracklets_considered=len(candidates),
        n_combinations_total=total,
        n_combinations_enumerated=enumerated,
        n_combinations_sampled=sampled,
        n_combinations_geometry_pruned=geometry_pruned,
        n_partially_represented_ties=partial_ties,
        n_solution_clusters_merged=clusters_merged,
        represented_compatibility_mass=represented_mass)


def _allocation_weight(hypothesis, config) -> float:
    weight = max(hypothesis.compatibility_mass, np.finfo(np.float64).tiny)
    if isinstance(hypothesis, ArcHypothesis):
        length_factor = max(
            1.0, hypothesis.arc_length_m / config.arc_length_reference_m)
        weight *= min(length_factor, config.arc_length_weight_cap)
    return weight


def sample_particles(result: ProposalResult, n_particles: int,
                     config: structs.ProposalConfig, rng: np.random.Generator):
    """Draw exactly ``n_particles`` whenever a valid hypothesis exists.

    Kind shares are retained. Within each kind, every activated solution first
    receives its configured density floor; remaining particles follow matcher
    evidence, with arc length increasing allocation to maintain linear density.
    """
    empty = (np.zeros(0), np.zeros(0), np.zeros(0),
             np.zeros(0, dtype=np.int64))
    if not result.hypotheses or n_particles <= 0:
        return empty

    present_kinds = {item.kind for item in result.hypotheses}
    kind_counts = _kind_particle_counts(n_particles, present_kinds, config)
    east_parts, north_parts, heading_parts, index_parts = [], [], [], []

    for kind in _KIND_ORDER:
        n_kind = kind_counts.get(kind, 0)
        if n_kind <= 0:
            continue
        all_indices = [index for index, item in enumerate(result.hypotheses)
                       if item.kind == kind]
        selected_hypotheses = _select_diverse(
            [result.hypotheses[index] for index in all_indices],
            _max_active_solutions(kind, n_kind, config), config)
        selected_ids = {id(item) for item in selected_hypotheses}
        indices = [index for index in all_indices
                   if id(result.hypotheses[index]) in selected_ids]
        indices.sort(key=lambda index: (
            result.hypotheses[index].residual_rad,
            result.hypotheses[index].tracklet_ids,
            result.hypotheses[index].landmark_ids))
        if not indices:
            raise RuntimeError(f"no {kind} hypothesis for allocated particles")

        floor = _particle_floor(kind, config)
        if n_kind >= floor * len(indices):
            counts = np.full(len(indices), floor, dtype=np.int64)
        else:
            counts = np.ones(len(indices), dtype=np.int64)
        remaining = n_kind - int(counts.sum())
        if remaining < 0:
            indices = indices[:n_kind]
            counts = np.ones(len(indices), dtype=np.int64)
            remaining = 0
        counts += _systematic_counts(
            remaining,
            [_allocation_weight(result.hypotheses[index], config)
             for index in indices])

        for index, count in zip(indices, counts):
            hypothesis = result.hypotheses[index]
            east, north, heading = hypothesis.sample(int(count), config, rng)
            expected_shape = (int(count),)
            if (east.shape != expected_shape or north.shape != expected_shape
                    or heading.shape != expected_shape):
                raise RuntimeError(
                    f"{hypothesis.kind} hypothesis returned "
                    f"{east.size}/{north.size}/{heading.size} particles, "
                    f"expected {count}")
            if not (np.all(np.isfinite(east))
                    and np.all(np.isfinite(north))
                    and np.all(np.isfinite(heading))):
                raise RuntimeError("proposal hypothesis returned non-finite poses")
            east_parts.append(east)
            north_parts.append(north)
            heading_parts.append(heading)
            index_parts.append(np.full(int(count), index, dtype=np.int64))

    if not east_parts:
        return empty
    east = np.concatenate(east_parts)
    north = np.concatenate(north_parts)
    heading = geo.wrap_rad(np.concatenate(heading_parts))
    hypothesis = np.concatenate(index_parts)
    if east.size != n_particles:
        raise RuntimeError(
            f"proposal returned {east.size} particles, expected {n_particles}")
    return east, north, heading, hypothesis
