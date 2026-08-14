"""Geometry and seam helpers for landmark matching.

What remains here after the wedge-matching design was retired:

- `triangulate` / `estimate_mount_offset` - the map-free camera-to-body yaw
  calibration. A wrong mount offset rotates every bearing by a constant,
  which stops a static object's rays from intersecting, so minimising the
  triangulation residual over well-conditioned tracklets recovers the offset
  with no map and no assumed match. This is still load-bearing.
- `effective_candidates` - how many map rows a match is really spread over,
  as exp(entropy) of the scores. 1 for a unique identification, N for N
  indistinguishable candidates.
- `to_compatibility_table` - raw scores to the filter's CompatibilityTable
  shape, through the tuned affine transform + clip the design doc SS6
  specifies for an uncalibrated matcher.

**Removed: bearing-wedge candidate generation.** It selected map candidates
using the vessel's GPS position, which is circular - gating candidates by
where the boat was and then asking the filter to recover where the boat was
leaks the answer. Matching now runs against the whole map with no spatial
information (`m9_match_landmarks`). The rule-based `TagRuleScorer` went with
it: it only ever scored wedge candidates, and the matcher it was a baseline
for no longer produces a candidate shortlist to score.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    harbor_catalog as hc,
)

# Mirrors bearing_only_localization.structs.CompatibilityTable. Kept as a
# plain dataclass so this module does not depend on the filter package; the
# emitter below produces exactly the field names that struct expects.
DEFAULT_CLIP = 4.0
DEFAULT_LOG_LR = -2.0


@dataclass
class Observation:
    """One bearing of one tracklet, with the pose that produced it.

    Kept after the wedge removal because both surviving geometry routines
    need it: `triangulate` reads the world bearing and the pose, and
    `estimate_mount_offset` additionally needs the raw camera azimuth and the
    course, since the offset it solves for is exactly what relates them.
    """
    anchor_keyframe_idx: int
    east_m: float
    north_m: float
    bearing_world_deg: float
    half_width_deg: float = 0.0
    # Only needed by estimate_mount_offset, which works in the un-offset
    # frame: bearing_world = course + (bearing_camera - offset).
    bearing_camera_deg: float = 0.0
    course_deg: float = 0.0


def triangulate(observations):
    """Least-squares intersection of the bearing rays, in ENU metres.

    Returns (east_m, north_m, median_abs_residual_deg, condition_number) or
    None if under-determined.

    This is the honest consistency check for a tracklet, and it replaces
    "circular std of the world bearing", which was wrong: a static object at
    1 km sweeps tens of degrees of true bearing as the vessel passes it, so
    spread measures parallax at least as much as error. What we actually want
    to know is whether the bearings are consistent with *some* single static
    point - which is exactly what the residual about the triangulated
    intersection reports.

    The condition number matters as much as the residual: bearings taken over
    a short arc intersect at a glancing angle, so a tiny residual can hide a
    position uncertain by kilometres along the line of sight. Callers must
    gate on both.
    """
    if len(observations) < 2:
        return None
    a_mat = np.zeros((2, 2))
    b_vec = np.zeros(2)
    for obs in observations:
        theta = math.radians(obs.bearing_world_deg)
        unit = np.array([math.sin(theta), math.cos(theta)])   # (east, north)
        proj = np.eye(2) - np.outer(unit, unit)
        a_mat += proj
        b_vec += proj @ np.array([obs.east_m, obs.north_m])
    eigenvalues = np.linalg.eigvalsh(a_mat)
    if eigenvalues.min() <= 1e-9:
        return None
    condition = float(eigenvalues.max() / eigenvalues.min())
    point = np.linalg.solve(a_mat, b_vec)

    # The least-squares solve intersects LINES, not rays, so it can place the
    # object behind the observer - which then reports a ~180 deg residual and
    # is meaningless. Seen for real: one leg1 tracklet came back at 179.7 deg.
    # Require the solution to lie ahead of most observations.
    ahead = 0
    for obs in observations:
        theta = math.radians(obs.bearing_world_deg)
        unit = np.array([math.sin(theta), math.cos(theta)])
        if float(np.dot(point - np.array([obs.east_m, obs.north_m]),
                        unit)) > 0.0:
            ahead += 1
    if ahead * 2 < len(observations):
        return None

    residuals = []
    for obs in observations:
        predicted = math.degrees(math.atan2(point[0] - obs.east_m,
                                            point[1] - obs.north_m)) % 360.0
        residuals.append(abs(hc.angular_delta_deg(obs.bearing_world_deg,
                                                  predicted)))
    residuals.sort()
    return (float(point[0]), float(point[1]),
            float(residuals[len(residuals) // 2]), condition)


def effective_candidates(scores, scale=1.0):
    """How many map candidates a match is really spread over.

    exp(entropy) of the softmax-normalised scores: 1.0 when one candidate
    dominates, N when N are indistinguishable. This is the difference between
    "matched One International Place" and "matched a tower" - both are
    correct, but the first pins a position and the second selects a class
    whose members sit hundreds of metres apart.

    It is derived from the scores rather than asserted, and it is deliberately
    NOT folded into log_lr: the filter already handles a spread match
    correctly as a mixture, so down-weighting it here would penalise the same
    ambiguity twice. Report it so downstream can rank tracklets by how much
    they actually constrain the pose.
    """
    if not scores:
        return 0.0
    values = [scale * v for v in scores.values()]
    top = max(values)
    weights = [math.exp(v - top) for v in values]
    total = sum(weights)
    if total <= 0:
        return float(len(values))
    entropy = -sum((w / total) * math.log(w / total)
                   for w in weights if w > 0)
    return math.exp(entropy)


def estimate_mount_offset(observations_by_tracklet, hypotheses, catalog_by_id):
    """Camera->body yaw implied by hypothesised tracklet->landmark matches.

    For a rigidly mounted camera the offset is one constant, so every
    confident match should imply the same value:
        offset = course + camera_az - true_bearing(pose -> landmark)

    Returns {tracklet_id: (offset_deg, median_residual_deg, n)} plus a
    circular-mean consensus. A small per-tracklet residual with a LARGE
    spread across tracklets does not mean the camera moved - it means either
    a hypothesis is wrong or the heading reference (GPS course) differs from
    true heading by a slowly-varying amount, and the caller must not average
    the two cases together blindly.
    """
    per_tracklet, all_offsets = {}, []
    for tracklet_id, landmark_id in hypotheses.items():
        entry = catalog_by_id.get(landmark_id)
        observations = observations_by_tracklet.get(tracklet_id) or []
        values = []
        for obs in observations:
            centre, _ = hc.bearing_span_from(entry, obs.east_m, obs.north_m)
            values.append((obs.bearing_camera_deg + obs.course_deg - centre)
                          % 360.0)
        if len(values) < 3:
            continue
        radians = [math.radians(v) for v in values]
        mean = math.degrees(math.atan2(
            sum(math.sin(r) for r in radians) / len(radians),
            sum(math.cos(r) for r in radians) / len(radians))) % 360.0
        residuals = sorted(abs(hc.angular_delta_deg(v, mean)) for v in values)
        per_tracklet[tracklet_id] = (mean,
                                     residuals[len(residuals) // 2],
                                     len(values))
        all_offsets.append(mean)

    consensus = float("nan")
    if all_offsets:
        radians = [math.radians(v) for v in all_offsets]
        consensus = math.degrees(math.atan2(
            sum(math.sin(r) for r in radians) / len(radians),
            sum(math.cos(r) for r in radians) / len(radians))) % 360.0
    return per_tracklet, consensus


def to_compatibility_table(tracklet_id, scores, matcher_version,
                           scale=1.0, offset=0.0, clip=DEFAULT_CLIP,
                           default_log_lr=DEFAULT_LOG_LR, status="fast"):
    """Raw scores -> the filter's CompatibilityTable dict.

    Field names match bearing_only_localization.structs.CompatibilityTable
    exactly, so the consumer can construct the struct without translation.
    Only entries that differ from `default_log_lr` are emitted; everything
    absent scores the default, per that struct's contract.
    """
    entries = []
    for landmark_id, raw in scores.items():
        log_lr = max(-clip, min(clip, scale * raw + offset))
        if abs(log_lr - default_log_lr) > 1e-9:
            entries.append({"landmark_id": landmark_id,
                            "log_lr": float(log_lr)})
    entries.sort(key=lambda e: -e["log_lr"])
    return {
        "tracklet_id": tracklet_id,
        "matcher_version": matcher_version,
        "entries": entries,
        "default_log_lr": float(default_log_lr),
        "clip_lo": float(-clip),
        "clip_hi": float(clip),
        "status": status,
    }
