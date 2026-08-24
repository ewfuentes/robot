"""Bearing-only resection: invert bearings into candidate poses (§5.5).

Two identified landmark-bearing correspondences constrain position to one
signed circular arc; three give a discrete fix. Two mirror arcs exist only
when the subtended angle is unsigned or the bearing assignment is unknown.
This is what lets the mixture proposal sample from the
likelihood rather than the motion model, so particle count scales with the
number of plausible hypotheses instead of with search area.

Everything here is forward-world-bearing-free by construction: the subtended
angle between two forward-frame bearings, gamma = beta_b - beta_a, is
independent of the platform's forward-axis world bearing. That bearing is recovered
afterwards from any one bearing (`heading_from_bearing`).

Conventions match the rest of the package: positions are region-frame ENU
metres, bearings are radians clockwise from north (compass convention), and
`geo.compass_bearing_rad` is the single source of truth for the angle of
a displacement.

Degeneracies are REJECTED rather than approximated. The important one is the
"danger circle" of the classic three-point resection (Snellius-Pothenot): an
observer standing on the circle through all three landmarks receives the same
pair of subtended angles from every point on that circle, so the fix is
genuinely indeterminate and any number returned would be fiction.
"""

import dataclasses
import math

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo

# A subtended angle this close to 0 or pi makes the arc radius blow up
# (R = baseline / (2 sin gamma)): the two landmarks are nearly in line with
# the observer and constrain position barely at all.
MIN_SUBTENDED_RAD = math.radians(8.0)
# Landmarks closer together than this cannot define a usable baseline.
MIN_BASELINE_M = 50.0
# Three-point fixes are rejected when the observer is within this fraction of
# the danger circle's radius of that circle.
DANGER_CIRCLE_FRAC = 0.06


@dataclasses.dataclass(frozen=True)
class PoseHypothesis:
    """A resected pose and how well it explains the bearings it came from.

    `residual_rad` is the largest per-bearing disagreement. It is the natural
    ranking key: spurious circle intersections survive the unsigned
    subtended-angle constraints but disagree with the actual directions by
    degrees, while the true pose disagrees only by measurement noise.
    """
    east_m: float
    north_m: float
    heading_rad: float
    residual_rad: float


@dataclasses.dataclass(frozen=True)
class ArcHypothesis:
    """Locus of poses seeing `subtended_rad` between two landmarks.

    A circle of radius `radius_m` about `center_east_m/north_m`; the valid
    locus is the arc on one side of the landmark baseline. ``side`` is the
    sign of the identified A-to-B subtended angle; the opposite side belongs
    to the opposite signed observation.
    """
    center_east_m: float
    center_north_m: float
    radius_m: float
    subtended_rad: float
    # +1 / -1: which side of the directed baseline a->b the observer is on.
    side: int


def subtended_angle_rad(bearing_a_rad: float, bearing_b_rad: float) -> float:
    """Signed angle from bearing a to bearing b, wrapped to (-pi, pi].

    Heading-independent: a common heading offset cancels in the difference.
    """
    return float(geo.wrap_rad(bearing_b_rad - bearing_a_rad))


def inscribed_angle_arcs(east_a: float, north_a: float, east_b: float,
                         north_b: float, subtended_rad: float):
    """Both arcs of poses subtending |subtended_rad| on segment AB.

    Inscribed-angle theorem: the locus is a circular arc of radius
    R = |AB| / (2 sin gamma), whose centre lies on the perpendicular
    bisector of AB at distance |AB| / (2 tan gamma) from the midpoint. The
    sign of that offset picks the side.

    Both sides see the same *unsigned* angle, so both are returned here. They
    are NOT interchangeable once the two bearings are identified: the signed
    angle from A to B is positive on one side and negative on the other, so
    knowing which bearing belongs to which landmark picks a side. Use
    `arcs_for_signed_angle` when the assignment is known; this primitive
    stays unsigned so the geometry is testable on its own terms.

    Returns [] for degenerate geometry.
    """
    gamma = abs(float(geo.wrap_rad(subtended_rad)))
    if not MIN_SUBTENDED_RAD < gamma < math.pi - MIN_SUBTENDED_RAD:
        return []
    d_east, d_north = east_b - east_a, north_b - north_a
    baseline = math.hypot(d_east, d_north)
    if baseline < MIN_BASELINE_M:
        return []

    radius = baseline / (2.0 * math.sin(gamma))
    offset = baseline / (2.0 * math.tan(gamma))
    mid_east, mid_north = 0.5 * (east_a + east_b), 0.5 * (north_a + north_b)
    # Unit normal to AB (rotate the baseline direction by -90 degrees).
    normal_east, normal_north = d_north / baseline, -d_east / baseline
    return [
        ArcHypothesis(center_east_m=mid_east + side * offset * normal_east,
                      center_north_m=mid_north + side * offset * normal_north,
                      radius_m=radius, subtended_rad=gamma, side=side)
        for side in (1, -1)]


def arcs_for_signed_angle(east_a: float, north_a: float, east_b: float,
                          north_b: float, signed_subtended_rad: float):
    """The one arc consistent with a signed subtended angle from A to B.

    With both bearings identified the sign is observable, so only one of the
    two inscribed-angle arcs is a real hypothesis — keeping both would put
    half the proposal's particles somewhere the measurement flatly
    contradicts. ``inscribed_angle_arcs`` constructs its ``side=+1`` arc for
    a positive A-to-B compass angle and ``side=-1`` for a negative one, so no
    sampling or random state is needed to select it.
    """
    arcs = inscribed_angle_arcs(east_a, north_a, east_b, north_b,
                               signed_subtended_rad)
    if not arcs:
        return []
    target = float(geo.wrap_rad(signed_subtended_rad))
    wanted_side = 1 if target > 0.0 else -1
    return [arc for arc in arcs if arc.side == wanted_side]


def valid_arc_interval_rad(arc: ArcHypothesis, east_a: float, north_a: float,
                           east_b: float, north_b: float) -> tuple[float, float]:
    """Return ``(start, positive_sweep)`` for the valid observer arc.

    The chord endpoints split the circle into exactly two angular intervals.
    Their interiors lie on opposite sides of directed baseline A->B. Under
    the compass-angle convention a positive signed A-to-B angle places the
    observer on the right side of that baseline, hence ``-arc.side`` in the
    ordinary Cartesian cross-product test below.
    """
    center_east = arc.center_east_m
    center_north = arc.center_north_m
    theta_a = math.atan2(north_a - center_north, east_a - center_east)
    theta_b = math.atan2(north_b - center_north, east_b - center_east)
    sweep_a_to_b = (theta_b - theta_a) % (2.0 * math.pi)

    midpoint = theta_a + 0.5 * sweep_a_to_b
    mid_east = center_east + arc.radius_m * math.cos(midpoint)
    mid_north = center_north + arc.radius_m * math.sin(midpoint)
    d_east = east_b - east_a
    d_north = north_b - north_a
    cross = d_east * (mid_north - north_a) - d_north * (mid_east - east_a)
    desired_cross_sign = -arc.side
    if cross * desired_cross_sign > 0.0:
        return theta_a, sweep_a_to_b
    return theta_b, 2.0 * math.pi - sweep_a_to_b


def valid_arc_length_m(arc: ArcHypothesis, east_a: float, north_a: float,
                       east_b: float, north_b: float) -> float:
    """Length of the valid signed observer arc in metres."""
    _, sweep = valid_arc_interval_rad(
        arc, east_a, north_a, east_b, north_b)
    return arc.radius_m * sweep


def sample_arc(arc: ArcHypothesis, east_a: float, north_a: float,
               east_b: float, north_b: float, n_samples: int,
               rng: np.random.Generator):
    """Sample poses uniformly along the *valid* arc.

    A circle carries the requested angle on only one interval between the
    chord endpoints. Resolve that interval analytically and sample it
    directly; no rejection or padding is needed, even for a very narrow arc.
    """
    if n_samples <= 0:
        return np.zeros(0), np.zeros(0)
    start, sweep = valid_arc_interval_rad(
        arc, east_a, north_a, east_b, north_b)
    # Avoid the chord endpoints themselves: there the observer coincides with
    # a landmark and its bearing is undefined.
    epsilon = np.finfo(np.float64).eps
    unit = epsilon + rng.random(n_samples) * (1.0 - 2.0 * epsilon)
    angles = start + sweep * unit
    east = arc.center_east_m + arc.radius_m * np.cos(angles)
    north = arc.center_north_m + arc.radius_m * np.sin(angles)
    return east, north


def _circle_intersections(arc_p: ArcHypothesis, arc_q: ArcHypothesis):
    """Intersection points of two circles; [] if they do not meet."""
    d_east = arc_q.center_east_m - arc_p.center_east_m
    d_north = arc_q.center_north_m - arc_p.center_north_m
    distance = math.hypot(d_east, d_north)
    if distance < 1e-9:
        return []  # concentric: coincident or disjoint, never a discrete fix
    if distance > arc_p.radius_m + arc_q.radius_m:
        return []
    if distance < abs(arc_p.radius_m - arc_q.radius_m):
        return []
    a = (arc_p.radius_m ** 2 - arc_q.radius_m ** 2 + distance ** 2) / (
        2.0 * distance)
    h_squared = arc_p.radius_m ** 2 - a ** 2
    if h_squared < 0.0:
        return []
    h = math.sqrt(h_squared)
    base_east = arc_p.center_east_m + a * d_east / distance
    base_north = arc_p.center_north_m + a * d_north / distance
    return [(base_east + sign * h * d_north / distance,
             base_north - sign * h * d_east / distance)
            for sign in (1.0, -1.0)]


def _on_danger_circle(east: float, north: float, landmarks) -> bool:
    """Is the observer on the circle through all three landmarks?

    There the three-point fix is indeterminate — every point of that circle
    yields the same pair of subtended angles.
    """
    (e1, n1), (e2, n2), (e3, n3) = landmarks
    # Circumcentre via the perpendicular-bisector determinant.
    d = 2.0 * (e1 * (n2 - n3) + e2 * (n3 - n1) + e3 * (n1 - n2))
    if abs(d) < 1e-6:
        return True  # collinear landmarks: no circumcircle, no fix
    sq1, sq2, sq3 = e1 ** 2 + n1 ** 2, e2 ** 2 + n2 ** 2, e3 ** 2 + n3 ** 2
    center_east = (sq1 * (n2 - n3) + sq2 * (n3 - n1) + sq3 * (n1 - n2)) / d
    center_north = (sq1 * (e3 - e2) + sq2 * (e1 - e3) + sq3 * (e2 - e1)) / d
    radius = math.hypot(e1 - center_east, n1 - center_north)
    observer = math.hypot(east - center_east, north - center_north)
    return abs(observer - radius) < DANGER_CIRCLE_FRAC * radius


def resect_three(landmark_positions, bearings_rad,
                 residual_tolerance_rad: float = math.radians(2.0)):
    """Discrete pose fixes from three landmarks and three body bearings.

    Tries every pairing of landmarks, not a fixed two: an observer nearly in
    line with one pair sees a subtended angle close to 180 degrees, which
    that pair's arc cannot constrain — but the other pairings usually still
    can, and fixing the pairing in advance would throw those poses away.

    `residual_tolerance_rad` rejects spurious intersections: the same circles
    also meet where the bearings point the wrong way. Those near-misses can
    sit only a couple of degrees off, so the tolerance must be matched to the
    measurement noise — callers with a known concentration should pass
    roughly `n_sigma / sqrt(kappa)`. Set it too loose and plausible-looking
    fiction survives; too tight and every noisy fix is thrown away.

    Returns PoseHypothesis list sorted by residual (best first), or [] when
    the geometry is degenerate (collinear landmarks, every pairing
    ill-conditioned, or an observer on the danger circle). More than one
    solution is a legitimate outcome, not a failure: this is a hypothesis
    generator feeding a mixture proposal, and the filter's measurement update
    is what adjudicates between them.
    """
    arcs_by_pair = {}
    for i, j in ((0, 1), (1, 2), (0, 2)):
        (ei, ni), (ej, nj) = landmark_positions[i], landmark_positions[j]
        arcs = arcs_for_signed_angle(
            ei, ni, ej, nj,
            subtended_angle_rad(bearings_rad[i], bearings_rad[j]))
        if arcs:
            arcs_by_pair[(i, j)] = arcs
    if len(arcs_by_pair) < 2:
        return []

    solutions = []
    pairs = sorted(arcs_by_pair)
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            for arc_p in arcs_by_pair[pairs[a]]:
                for arc_q in arcs_by_pair[pairs[b]]:
                    for east, north in _circle_intersections(arc_p, arc_q):
                        # Circles for pairs sharing a landmark always meet at
                        # that landmark; it is not an observer position.
                        shared = set(pairs[a]) & set(pairs[b])
                        if any(math.hypot(east - landmark_positions[k][0],
                                          north - landmark_positions[k][1])
                               < MIN_BASELINE_M for k in shared):
                            continue
                        if _on_danger_circle(east, north, landmark_positions):
                            continue
                        heading = heading_from_bearing(
                            east, north, landmark_positions[0][0],
                            landmark_positions[0][1], bearings_rad[0])
                        # Verify against all bearings because noisy pairwise
                        # signed arcs need not agree exactly at their circle
                        # intersections.
                        residual = max(
                            abs(float(geo.wrap_rad(
                                geo.compass_bearing_rad(le - east,
                                                            ln - north)
                                - heading - bearing)))
                            for (le, ln), bearing in zip(landmark_positions,
                                                         bearings_rad))
                        if residual > residual_tolerance_rad:
                            continue
                        if any(math.hypot(east - other.east_m,
                                          north - other.north_m) < 1.0
                               for other in solutions):
                            continue
                        solutions.append(PoseHypothesis(
                            east_m=east, north_m=north, heading_rad=heading,
                            residual_rad=residual))
    return sorted(solutions, key=lambda s: s.residual_rad)


def heading_from_bearing(east_m: float, north_m: float, landmark_east_m: float,
                         landmark_north_m: float,
                         body_bearing_rad: float) -> float:
    """Forward-axis world bearing implied by one forward-frame observation."""
    world = geo.compass_bearing_rad(landmark_east_m - east_m,
                                        landmark_north_m - north_m)
    return float(geo.wrap_rad(world - body_bearing_rad))
