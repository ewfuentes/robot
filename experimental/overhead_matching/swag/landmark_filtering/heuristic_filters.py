"""Stage A heuristic filters.

Every filter annotates observations with a FilterDecision; nothing is ever
deleted. Filters run in registry order over ALL observations independently of
each other's outcomes, so the decision trail always shows every filter's
verdict. final_disposition is "filtered" iff any decision filtered, and
final_reason is the first filtering reason.

Adding a filter = one function + a config substruct + a registry entry; its
reason strings flow into SummaryStats.filtered_by_reason and the viewer
automatically.
"""

from experimental.overhead_matching.swag.landmark_filtering import (
    artifact_schema as schema,
    bearing_geometry as bg,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    HeuristicConfig,
)

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _kept(filter_name: str, **details) -> schema.FilterDecision:
    return schema.FilterDecision(
        filter_name=filter_name, disposition=schema.KEPT, reason="",
        details={k: float(v) for k, v in details.items()})


def _filtered(filter_name: str, reason: str,
              **details) -> schema.FilterDecision:
    return schema.FilterDecision(
        filter_name=filter_name, disposition=schema.FILTERED, reason=reason,
        details={k: float(v) for k, v in details.items()})


def confidence_gate(obs, frame, config):
    rank = CONFIDENCE_RANK.get(obs.confidence, 0)
    min_rank = CONFIDENCE_RANK.get(config.min_confidence, 1)
    if rank < min_rank:
        return _filtered("confidence_gate", "confidence_low",
                         confidence_rank=rank)
    return _kept("confidence_gate", confidence_rank=rank)


def angular_width_gate(obs, frame, config):
    if obs.angular_width_deg > config.max_angular_width_deg:
        return _filtered("angular_width_gate", "angular_width_excessive",
                         angular_width_deg=obs.angular_width_deg)
    return _kept("angular_width_gate", angular_width_deg=obs.angular_width_deg)


def tag_blocklist(obs, frame, config):
    if obs.primary_tag_key in config.blocked_primary_keys:
        return _filtered(
            "tag_blocklist", f"blocked_tag:{obs.primary_tag_key}")
    for key, value in config.blocked_primary_tags:
        if obs.primary_tag_key == key and obs.primary_tag_value == value:
            return _filtered("tag_blocklist", f"blocked_tag:{key}={value}")
    return _kept("tag_blocklist")


def elevation_gate(obs, frame, config):
    if obs.elevation_deg < config.min_center_elevation_deg:
        return _filtered("elevation_gate", "elevation_too_low",
                         elevation_deg=obs.elevation_deg)
    return _kept("elevation_gate", elevation_deg=obs.elevation_deg)


def edge_truncation(obs, frame, config):
    # Seam-merged groups found their continuation on the adjacent face; a
    # single box hitting a vertical face edge did not, so its bearing is
    # unreliable (the object extends past the edge).
    if not obs.seam_merged:
        box = obs.boxes[0]
        if (box.xmin <= config.edge_margin_norm
                or box.xmax >= bg.BBOX_NORM_MAX - config.edge_margin_norm):
            return _filtered("edge_truncation", "unmatched_edge_truncation",
                             xmin=box.xmin, xmax=box.xmax)
    return _kept("edge_truncation")


REGISTRY = [
    ("confidence_gate", "confidence_gate", confidence_gate),
    ("angular_width_gate", "angular_width_gate", angular_width_gate),
    ("tag_blocklist", "tag_blocklist", tag_blocklist),
    ("elevation_gate", "elevation_gate", elevation_gate),
    ("edge_truncation", "edge_truncation", edge_truncation),
]


def _dedup_rank(obs) -> tuple:
    """Higher is better: confidence, then angular width."""
    return (CONFIDENCE_RANK.get(obs.confidence, 0), obs.angular_width_deg)


def intra_frame_dedup(observations, config) -> dict[str, str]:
    """Returns {obs_id: duplicate_of_obs_id} for redundant same-frame obs.

    Within a frame, observations with the same primary tag whose bearings are
    within max_bearing_sep_deg are considered one detection; all but the
    highest-ranked are marked duplicates.
    """
    duplicates: dict[str, str] = {}
    by_frame: dict[int, list] = {}
    for obs in observations:
        by_frame.setdefault(obs.frame_idx, []).append(obs)
    for frame_obs in by_frame.values():
        by_tag: dict[tuple, list] = {}
        for obs in frame_obs:
            key = (obs.primary_tag_key, obs.primary_tag_value)
            by_tag.setdefault(key, []).append(obs)
        for group in by_tag.values():
            group = sorted(group, key=_dedup_rank, reverse=True)
            kept: list = []
            for obs in group:
                winner = next(
                    (k for k in kept if abs(float(bg.circular_diff_deg(
                        obs.bearing_camera_deg, k.bearing_camera_deg)))
                     < config.max_bearing_sep_deg),
                    None)
                if winner is not None:
                    duplicates[obs.obs_id] = winner.obs_id
                else:
                    kept.append(obs)
    return duplicates


def run_stage_a(artifact: schema.RunArtifact,
                config: HeuristicConfig) -> None:
    frames = {frame.frame_idx: frame for frame in artifact.frames}

    for name, config_attr, fn in REGISTRY:
        sub_config = getattr(config, config_attr)
        if not sub_config.enabled:
            continue
        for obs in artifact.observations:
            decision = fn(obs, frames[obs.frame_idx], sub_config)
            obs.decisions.append(decision)

    if config.intra_frame_dedup.enabled:
        duplicates = intra_frame_dedup(
            artifact.observations, config.intra_frame_dedup)
        for obs in artifact.observations:
            if obs.obs_id in duplicates:
                decision = _filtered(
                    "intra_frame_dedup", "intra_frame_duplicate")
                # Which observation this one duplicates, for the viewer.
                decision.details["duplicate_of_landmark_idx"] = float(
                    int(duplicates[obs.obs_id].split("__lm")[1]
                        .split("__")[0]))
                obs.decisions.append(decision)
            else:
                obs.decisions.append(_kept("intra_frame_dedup"))

    finalize_dispositions(artifact)


def finalize_dispositions(artifact: schema.RunArtifact) -> None:
    """Recompute final_disposition/final_reason and summary stats from the
    decision trails."""
    stats = artifact.stats
    stats.n_kept = 0
    stats.n_filtered = 0
    stats.filtered_by_reason = {}
    stats.filtered_by_filter = {}
    for obs in artifact.observations:
        filtering = [d for d in obs.decisions
                     if d.disposition == schema.FILTERED]
        if filtering:
            obs.final_disposition = schema.FILTERED
            obs.final_reason = filtering[0].reason
            stats.n_filtered += 1
            stats.filtered_by_reason[filtering[0].reason] = (
                stats.filtered_by_reason.get(filtering[0].reason, 0) + 1)
            for decision in filtering:
                stats.filtered_by_filter[decision.filter_name] = (
                    stats.filtered_by_filter.get(decision.filter_name, 0) + 1)
        else:
            obs.final_disposition = schema.KEPT
            obs.final_reason = ""
            stats.n_kept += 1
