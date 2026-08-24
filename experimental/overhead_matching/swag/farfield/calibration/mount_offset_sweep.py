"""Triangulation math for the alignment-diagnostics artifact.

These helpers evaluate camera-to-GPS-course candidates. They do not load
pipeline artifacts, write sidecars, or approve a candidate for localization;
`build_alignment_diagnostics` owns those contracts and records the result as
diagnostic evidence only.
"""

import math

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo

# A minimum this shallow relative to the rest of the curve is not a minimum.
MIN_CONTRAST = 1.5

# A camera-to-effective-course candidate cannot be supported by one tracklet:
# a single tracklet's residual is a smooth function of the candidate by
# construction, so the curve looks textbook-unimodal while saying nothing.
# Observed for real: a 1-tracklet sweep returned "SMOOTH UNIMODAL, 0.95 deg"
# while the same leg with less bearing fusion (7 tracklets) disagreed by
# nearly 180 deg. The other gates are all relative, so none of them can catch
# this. This is `assess`'s default floor; the typed diagnostic config supplies
# the value used for a particular build.
MIN_TRACKLETS = 5


def arc_deg(observations):
    """Angular spread of a tracklet's bearings -- the smallest arc holding them.

    Computed on `course + camera`, i.e. the world bearing at offset 0. The
    candidate offset is subtracted from every bearing alike, so it cannot
    change a spread: this number is the same at every offset, which is what
    lets it select a fixed tracklet set for the whole sweep.
    """
    angles = sorted((gps_course_world_cw_deg + camera) % 360.0
                    for _, _, camera, gps_course_world_cw_deg, _
                    in observations)
    if len(angles) < 2:
        return 0.0
    gaps = [b - a for a, b in zip(angles, angles[1:])]
    gaps.append(angles[0] + 360.0 - angles[-1])
    return 360.0 - max(gaps)


def triangulate(rays):
    """Least-squares intersection of bearing rays, in ENU metres.

    rays: [(east_m, north_m, bearing_world_deg)]. Returns
    (east_m, north_m, median_abs_residual_deg, condition_number), or None if
    under-determined.

    This is the honest consistency check for a tracklet, and it replaces
    "circular std of the world bearing", which was wrong: a static object at
    1 km sweeps tens of degrees of true bearing as the vessel passes it, so
    spread measures parallax at least as much as error. What we actually want
    to know is whether the bearings are consistent with *some* single static
    point -- which is exactly what the residual about the triangulated
    intersection reports.

    The condition number matters as much as the residual: bearings taken over
    a short arc intersect at a glancing angle, so a tiny residual can hide a
    position uncertain by kilometres along the line of sight. Callers must
    gate on both.

    Ported from bearing_matcher.triangulate on the checkpoint branch; when
    the matching package lands (REORG.md PR 09) its copy and this one should
    merge into a single owner.
    """
    if len(rays) < 2:
        return None
    a_mat = np.zeros((2, 2))
    b_vec = np.zeros(2)
    units, points = [], []
    for east_m, north_m, bearing_world_deg in rays:
        theta = math.radians(bearing_world_deg)
        unit = np.array([math.sin(theta), math.cos(theta)])   # (east, north)
        proj = np.eye(2) - np.outer(unit, unit)
        a_mat += proj
        b_vec += proj @ np.array([east_m, north_m])
        units.append(unit)
        points.append(np.array([east_m, north_m]))
    eigenvalues = np.linalg.eigvalsh(a_mat)
    if eigenvalues.min() <= 1e-9:
        return None
    condition = float(eigenvalues.max() / eigenvalues.min())
    point = np.linalg.solve(a_mat, b_vec)

    # The least-squares solve intersects LINES, not rays, so it can place the
    # object behind the observer -- which then reports a ~180 deg residual and
    # is meaningless. Seen for real: one tracklet came back at 179.7 deg.
    # Require the solution to lie ahead of most observations.
    ahead = sum(1 for unit, origin in zip(units, points)
                if float(np.dot(point - origin, unit)) > 0.0)
    if ahead * 2 < len(rays):
        return None

    residuals = []
    for (east_m, north_m, bearing_world_deg), origin in zip(rays, points):
        predicted = geo.compass_bearing_deg(point[0] - east_m,
                                            point[1] - north_m)
        residuals.append(abs(float(geo.circular_diff_deg(
            bearing_world_deg, predicted))))
    residuals.sort()
    return (float(point[0]), float(point[1]),
            float(residuals[len(residuals) // 2]), condition)


def camera_to_effective_gps_course_cw_deg(
        bearing_camera_cw_deg: float,
        camera_to_effective_gps_course_deg: float) -> float:
    """Rotate a camera bearing into an effective GPS-course frame.

    This transform exists only for alignment-candidate diagnostics. Its second
    argument is a swept hypothesis, not an approved nominal-forward
    calibration and not localization authority.
    """
    return (bearing_camera_cw_deg
            - camera_to_effective_gps_course_deg) % 360.0


def residual_at(candidate_deg: float, by_tracklet, max_condition: float):
    """Median triangulation residual over well-conditioned tracklets."""
    residuals = []
    for observations in by_tracklet.values():
        rays = [
            (east, north, float(geo.forward_to_world_bearing_cw_deg(
                forward_world_cw_deg=gps_course_world_cw_deg,
                bearing_forward_cw_deg=(
                    camera_to_effective_gps_course_cw_deg(
                        camera, candidate_deg)))))
            for east, north, camera, gps_course_world_cw_deg, _
            in observations
        ]
        result = triangulate(rays)
        if result is None:
            continue
        _, _, residual, condition = result
        if condition > max_condition:
            continue
        residuals.append(residual)
    if not residuals:
        return None, 0
    residuals.sort()
    return residuals[len(residuals) // 2], len(residuals)


def sweep(by_tracklet, max_condition, start, stop, step):
    """[(offset_deg, median_residual_deg, n_tracklets)] over a grid."""
    curve = []
    offset = start
    while offset < stop - 1e-9:
        residual, n = residual_at(offset % 360.0, by_tracklet, max_condition)
        if residual is not None:
            curve.append((offset % 360.0, residual, n))
        offset += step
    return curve


def eligible(curve, min_support_frac):
    """Candidates that keep enough tracklets to be compared on residual.

    The condition gate's survivor count varies with the offset, so an
    unrestricted argmin can be won by a candidate that discarded almost
    everything. Returns (eligible_curve, support_floor).
    """
    if not curve:
        return [], 0
    best_support = max(n for _, _, n in curve)
    floor = max(1, int(math.ceil(best_support * min_support_frac)))
    return [c for c in curve if c[2] >= floor], floor


def local_minima(curve, tolerance=1e-9):
    """Indices that are no worse than both neighbours (cyclic)."""
    found = []
    for i, (_, residual, _) in enumerate(curve):
        prev = curve[i - 1][1]
        nxt = curve[(i + 1) % len(curve)][1]
        if residual <= prev + tolerance and residual <= nxt + tolerance:
            found.append(i)
    return found


def assess(curve, best_residual, n_used, min_tracklets=MIN_TRACKLETS):
    """Assess internal support; this does not approve a calibration."""
    if n_used < min_tracklets:
        return ("UNDER-SUPPORTED",
                f"the winning candidate triangulates only {n_used} "
                f"well-conditioned tracklet(s) (want >={min_tracklets}); with "
                f"so few, the curve's shape is a property of those tracklets "
                f"rather than of the camera-to-course relationship", False)
    residuals = sorted(r for _, r, _ in curve)
    median = residuals[len(residuals) // 2]
    contrast = median / best_residual if best_residual > 0 else float("inf")

    # Count only minima that are actually competitive with the best; a coarse
    # grid on a smooth curve still produces tiny numerical wiggles.
    threshold = best_residual * 1.25
    basins = [i for i in local_minima(curve) if curve[i][1] <= threshold]
    # Adjacent indices are one basin sampled twice, not two basins.
    distinct = [i for j, i in enumerate(basins)
                if j == 0 or i - basins[j - 1] > 1]

    if contrast < MIN_CONTRAST:
        return ("FLAT",
                f"median/min residual is only {contrast:.2f}x (want "
                f">={MIN_CONTRAST}); the sweep barely prefers any offset, so "
                f"its argmin is noise", False)
    if len(distinct) > 1:
        angles = ", ".join(f"{curve[i][0]:.0f}" for i in distinct)
        return ("MULTIMODAL",
                f"{len(distinct)} competitive minima at {angles} deg; the "
                f"bearings or the poses are inconsistent", False)
    return ("SMOOTH UNIMODAL",
            f"single minimum, median/min residual {contrast:.1f}x", True)
