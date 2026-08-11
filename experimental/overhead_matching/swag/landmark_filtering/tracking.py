"""Stage B association: kept observations -> tracks.

Frame-by-frame assignment of observations to active tracks with:
- a bearing gate on camera-frame bearings (offset-invariant, so tracking never
  depends on the global yaw offset),
- a semantic-similarity gate (pluggable backend),
- per-frame Hungarian assignment with dustbin columns (idiom borrowed from
  evaluation/correspondence_matching.match_and_aggregate): declining a
  marginal match and spawning a new track is part of the joint optimum.

Tracks shorter than min_track_length are marked filtered and their members
back-annotated with a track_length_gate FilterDecision.
"""

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from experimental.overhead_matching.swag.landmark_filtering import (
    artifact_schema as schema,
    bearing_geometry as bg,
    heuristic_filters,
    semantic_similarity,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    AssociationConfig,
    FilterPipelineConfig,
)

INFEASIBLE = 1.0e6
# Track-observation similarity is max over the most recent members; capped for
# cost.
MAX_REPRESENTATIVE_MEMBERS = 5


class _ActiveTrack:
    def __init__(self, track_id: int, obs: schema.Observation):
        self.track_id = track_id
        self.members: list[schema.Observation] = [obs]

    @property
    def last_obs(self) -> schema.Observation:
        return self.members[-1]

    def bearing_rate_deg_per_frame(self) -> float:
        if len(self.members) < 2:
            return 0.0
        a, b = self.members[-2], self.members[-1]
        gap = b.frame_idx - a.frame_idx
        return float(bg.circular_diff_deg(
            b.bearing_camera_deg, a.bearing_camera_deg)) / max(gap, 1)

    def predicted_bearing_deg(self, frame_idx: int) -> float:
        gap = frame_idx - self.last_obs.frame_idx
        return (self.last_obs.bearing_camera_deg
                + self.bearing_rate_deg_per_frame() * gap) % 360.0

    def recent_members(self) -> list[schema.Observation]:
        return self.members[-MAX_REPRESENTATIVE_MEMBERS:]

    def add(self, obs: schema.Observation) -> None:
        self.members.append(obs)


def _pair_cost(track: _ActiveTrack, obs: schema.Observation,
               similarity: float, baseline_m: float,
               config: AssociationConfig) -> float:
    if similarity < config.min_similarity:
        return INFEASIBLE
    residual = float(bg.circular_diff_deg(
        obs.bearing_camera_deg,
        track.predicted_bearing_deg(obs.frame_idx)))
    sigma_eff = (config.bearing_sigma_deg
                 + config.motion_gate_gain_deg_per_m * baseline_m)
    normalized_sq = (residual / sigma_eff) ** 2
    if normalized_sq > config.gate_chi2:
        return INFEASIBLE
    return (normalized_sq / config.gate_chi2
            + config.similarity_cost_weight * (1.0 - similarity))


def _finalize_track(track: _ActiveTrack, config: AssociationConfig,
                    backend) -> schema.Track:
    members = track.members
    for obs in members:
        obs.track_id = track.track_id
    representative = max(
        members,
        key=lambda o: (heuristic_filters.CONFIDENCE_RANK.get(o.confidence, 0),
                       o.angular_width_deg))

    mean_similarity = None
    if len(members) > 1:
        sample = members[:20]
        sim = backend.pairwise(sample, sample)
        off_diag = sim[~np.eye(len(sample), dtype=bool)]
        mean_similarity = float(off_diag.mean())

    result = schema.Track(
        track_id=track.track_id,
        obs_ids=[o.obs_id for o in members],
        first_frame_idx=members[0].frame_idx,
        last_frame_idx=members[-1].frame_idx,
        representative_obs_id=representative.obs_id,
        mean_pairwise_similarity=mean_similarity,
    )
    if len(members) < config.min_track_length:
        result.disposition = schema.FILTERED
        result.reason = "track_too_short"
        for obs in members:
            obs.decisions.append(schema.FilterDecision(
                filter_name="track_length_gate",
                disposition=schema.FILTERED,
                reason="track_too_short",
                details={"track_length": float(len(members))}))
    return result


def run_tracking(artifact: schema.RunArtifact, config: FilterPipelineConfig,
                 device: str = "cpu",
                 backend=None) -> None:
    assoc = config.association
    if backend is None:
        from pathlib import Path
        backend = semantic_similarity.make_backend(
            assoc.semantic_backend, Path(artifact.landmark_base),
            config.semantic_similarity, artifact.observations, device=device)

    frames = artifact.frames
    obs_by_frame: dict[int, list[schema.Observation]] = {}
    for obs in artifact.observations:
        if obs.final_disposition == schema.KEPT:
            obs_by_frame.setdefault(obs.frame_idx, []).append(obs)

    active: list[_ActiveTrack] = []
    finished: list[_ActiveTrack] = []
    next_track_id = 0

    for frame in frames:
        frame_idx = frame.frame_idx
        still_active = []
        for track in active:
            if frame_idx - track.last_obs.frame_idx > assoc.max_frame_gap:
                finished.append(track)
            else:
                still_active.append(track)
        active = still_active

        candidates = obs_by_frame.get(frame_idx, [])
        if not candidates:
            continue

        matched_obs_indices = set()
        if active:
            # Similarity of each candidate against each track's recent
            # members (max over members), one batched backend call.
            track_members = [t.recent_members() for t in active]
            flat_members = [m for members in track_members for m in members]
            sim_flat = backend.pairwise(candidates, flat_members)
            similarity = np.zeros((len(candidates), len(active)))
            col = 0
            for t_idx, members in enumerate(track_members):
                similarity[:, t_idx] = sim_flat[
                    :, col:col + len(members)].max(axis=1)
                col += len(members)

            cost = np.full(
                (len(candidates), len(active) + len(candidates)), INFEASIBLE)
            dustbin_cost = (1.0 + assoc.similarity_cost_weight
                            * (1.0 - assoc.min_similarity))
            for o_idx, obs in enumerate(candidates):
                cost[o_idx, len(active) + o_idx] = dustbin_cost
                for t_idx, track in enumerate(active):
                    last_frame = frames[track.last_obs.frame_idx]
                    baseline_m = math.hypot(frame.x_m - last_frame.x_m,
                                            frame.y_m - last_frame.y_m)
                    cost[o_idx, t_idx] = _pair_cost(
                        track, obs, similarity[o_idx, t_idx], baseline_m,
                        assoc)

            rows, cols = linear_sum_assignment(cost)
            for o_idx, t_idx in zip(rows, cols):
                if t_idx < len(active) and cost[o_idx, t_idx] < INFEASIBLE:
                    active[t_idx].add(candidates[o_idx])
                    matched_obs_indices.add(o_idx)

        for o_idx, obs in enumerate(candidates):
            if o_idx not in matched_obs_indices:
                active.append(_ActiveTrack(next_track_id, obs))
                next_track_id += 1

    finished.extend(active)
    finished.sort(key=lambda t: t.track_id)
    artifact.tracks = [
        _finalize_track(track, assoc, backend) for track in finished]

    heuristic_filters.finalize_dispositions(artifact)
    stats = artifact.stats
    stats.n_tracks = sum(
        1 for t in artifact.tracks if t.disposition == schema.KEPT)
    stats.n_singleton_obs = sum(
        1 for t in artifact.tracks if len(t.obs_ids) == 1)
