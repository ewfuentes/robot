"""Matching seam: merged landmarks -> per-tracklet CompatibilityTable.

This is the block between M6 (merged landmarks + fused bearings) and the
bearing-only localization filter. Its whole job is to turn "this tracklet
looks like a white banded standpipe" into "log-likelihood-ratio over map
candidates", in exactly the shape `bearing_only_localization.structs`
expects.

Candidate generation is a **bearing wedge**, the analogue of LOCI taking
Set 2 from the satellite tiles covering a panorama: from each observing
position, along the observed bearing, out to a visibility horizon. Since a
tracklet is observed from many positions, the *intersection* of its wedges is
far tighter than any single one - that intersection is triangulation, and it
is what makes a 156 k-row catalog affordable without filtering it.

Scoring is pluggable and deliberately uncalibrated (design doc SS6): a scorer
returns a raw per-candidate score, and `to_compatibility_table` maps scores to
log_lr through a tuned affine transform + clip. The clips and the filter's
null hypothesis carry the safety burden, not the scorer's calibration.

Scorers here:
  - `TagRuleScorer`   - interpretable baseline, no training (Method D)
Learned scorers (the released correspondence checkpoint, and the retrained
harbor model) plug in through the same `score_candidates` interface.
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
class WedgeConfig:
    # Angular slack added to the tracklet's own measured half-width. Covers
    # bearing noise plus the un-calibrated camera->body mount offset, which
    # is only known to a few degrees until it is re-estimated from confident
    # matches.
    bearing_slack_deg: float = 6.0
    max_range_m: float = 20000.0
    min_range_m: float = 50.0
    # An observation supports a candidate if the candidate falls in that
    # observation's wedge. Candidates seen in fewer than this fraction of a
    # tracklet's observations are dropped: a real landmark stays in the wedge
    # as the vessel moves, clutter drifts out.
    min_observation_support: float = 0.5


@dataclass
class Observation:
    """One fused bearing of one tracklet, with the pose that made it."""
    anchor_keyframe_idx: int
    east_m: float
    north_m: float
    bearing_world_deg: float   # heading + body bearing, already combined
    half_width_deg: float = 0.0


@dataclass
class CandidateEvidence:
    entry: hc.CatalogEntry
    n_observations: int
    n_in_wedge: int
    median_range_m: float
    median_abs_residual_deg: float

    @property
    def support_frac(self) -> float:
        return self.n_in_wedge / max(1, self.n_observations)


def gather_candidates(entries, observations, cfg: WedgeConfig):
    """Catalog entries consistent with a tracklet across its observations.

    Intersecting per-observation wedges is the cheap stand-in for
    triangulation: a candidate must sit in the wedge from most of the places
    the tracklet was seen from, not merely from one.
    """
    if not observations:
        return []

    seen = {}          # landmark_id -> CatalogEntry
    hits = {}          # landmark_id -> wedges containing it
    residuals = {}     # landmark_id -> |observed - predicted| per wedge
    ranges = {}        # landmark_id -> range per wedge

    for obs in observations:
        half = cfg.bearing_slack_deg + obs.half_width_deg
        for entry, range_m, centre, _ in hc.wedge_candidates(
                entries, obs.east_m, obs.north_m, obs.bearing_world_deg, half,
                max_range_m=cfg.max_range_m, min_range_m=cfg.min_range_m):
            key = entry.landmark_id
            seen[key] = entry
            hits[key] = hits.get(key, 0) + 1
            residuals.setdefault(key, []).append(
                abs(hc.angular_delta_deg(centre, obs.bearing_world_deg)))
            ranges.setdefault(key, []).append(range_m)

    def median(values):
        ordered = sorted(values)
        return ordered[len(ordered) // 2] if ordered else float("nan")

    out = []
    for key, count in hits.items():
        evidence = CandidateEvidence(
            entry=seen[key],
            n_observations=len(observations),
            n_in_wedge=count,
            median_range_m=median(ranges[key]),
            median_abs_residual_deg=median(residuals[key]))
        if evidence.support_frac >= cfg.min_observation_support:
            out.append(evidence)
    out.sort(key=lambda e: (-e.support_frac, e.median_abs_residual_deg))
    return out


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

@dataclass
class TagRuleScorer:
    """Interpretable baseline: tag agreement + name agreement (Method D).

    No training and no embeddings, so it is the control that learned scorers
    must beat. Scores are raw and unbounded-ish; the affine transform in
    `to_compatibility_table` maps them onto log_lr.
    """
    tag_weight: float = 1.0
    name_weight: float = 3.0
    enc_bonus: float = 0.5
    conspicuous_bonus: float = 1.0

    def score_candidates(self, landmark, candidates):
        weighted_tags = {t["tag"]: t.get("weight", 1.0)
                         for t in landmark.get("tags", [])}
        names = {c["name"].strip().lower()
                 for c in landmark.get("name_candidates", [])
                 if c.get("name")}
        scores = {}
        for cand in candidates:
            tags = cand.entry.tags
            cand_pairs = {f"{k}={v}" for k, v in tags.items()}
            score = sum(weight for tag, weight in weighted_tags.items()
                        if tag in cand_pairs) * self.tag_weight
            cand_name = tags.get("name", "").strip().lower()
            if cand_name and cand_name in names:
                score += self.name_weight
            if cand.entry.source == "enc":
                score += self.enc_bonus
                if "conspicuous" in tags.get("description", "").lower():
                    score += self.conspicuous_bonus
            scores[cand.entry.landmark_id] = score
        return scores


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
