"""Causal whole-track reranking of natural-release grid modes.

At each atomic track-release keyframe, this experiment places the effective
odometry prefix at every handed-off grid mode and reranks those placements
using all epoch-1 observations from tracks released so far.  A selected pose
is propagated by the same odometry until the next release.  Earlier emitted
poses are never revised.

This is a current-state MAP diagnostic.  Its point-error fractions are not
posterior mass and must not be reported as mass@500.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield import (
    artifact,
    geometry as geo,
)
from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    filter as filter_lib,
    metrics,
    odometry_profiles,
)


GRID_SCHEMA = "farfield_causal_grid/v1"
OUTPUT_SCHEMA = "farfield_causal_rigid_reranker/v1"
POSE_FRAME = "region_enu_heading_world_cw_from_north"
RANK_SEMANTICS = "one_based_probability_order_after_se2_nms"
PROBABILITY_SEMANTICS = "single_grid_state_mass_not_integrated_mode_mass"
OBSERVATION_POWER = 0.2
OUTLIER_RATE = 0.1
RADII_M = (100.0, 500.0, 1000.0)
_MODE_KEYS = frozenset({
    "source_rank", "source_probability", "east_m", "north_m",
    "heading_world_cw_deg", "heading_index", "north_index", "east_index",
})


def _number(value, name: str, *, minimum=None, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _read_json(path: Path) -> dict:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value!r}")))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read causal grid JSON {path}: {exc}") \
            from exc
    if not isinstance(payload, dict):
        raise ValueError("causal grid JSON must contain an object")
    return payload


def reconstruct_rigid_prefixes(current_poses, odometry,
                               end_keyframe_idx: int) -> np.ndarray:
    """Invert rotate-then-move odometry from candidate current poses."""
    poses = np.asarray(current_poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1:] != (3,) or not len(poses):
        raise ValueError("current_poses must have non-empty shape (modes, 3)")
    if not np.isfinite(poses).all():
        raise ValueError("current_poses must be finite")
    if (isinstance(end_keyframe_idx, bool)
            or not isinstance(end_keyframe_idx, int)
            or end_keyframe_idx < 0):
        raise ValueError("end_keyframe_idx must be a nonnegative integer")
    by_keyframe = {item.keyframe_idx: item for item in odometry}
    if len(by_keyframe) != len(odometry) or not set(
            range(1, end_keyframe_idx + 1)) <= set(by_keyframe):
        raise ValueError("odometry does not cover the requested prefix")

    paths = np.empty(
        (len(poses), end_keyframe_idx + 1, 3), dtype=np.float64)
    paths[:, -1] = poses
    paths[:, -1, 2] = geo.wrap_rad(paths[:, -1, 2])
    for keyframe in range(end_keyframe_idx, 0, -1):
        delta = by_keyframe[keyframe]
        current = paths[:, keyframe]
        previous = paths[:, keyframe - 1]
        heading = current[:, 2]
        previous[:, 0] = (
            current[:, 0] - delta.forward_m * np.sin(heading)
            + delta.left_m * np.cos(heading))
        previous[:, 1] = (
            current[:, 1] - delta.forward_m * np.cos(heading)
            - delta.left_m * np.sin(heading))
        previous[:, 2] = geo.wrap_rad(
            heading - delta.delta_yaw_cw_rad)
    return paths


def score_rigid_prefixes(paths, releases, catalog, *, pi0,
                         matcher_recall, range_softness, range_cap,
                         kappa_scale) -> np.ndarray:
    """Score paths with one marginalized, static identity per whole track."""
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 3 or paths.shape[2] != 3 \
            or not paths.shape[0] or not paths.shape[1] \
            or not np.isfinite(paths).all():
        raise ValueError("paths must have finite, non-empty shape (M, N, 3)")
    for value, name in ((pi0, "pi0"),
                        (matcher_recall, "matcher_recall"),
                        (OUTLIER_RATE, "outlier_rate")):
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be finite and in (0, 1)")
    for value, name in ((range_softness, "range_softness"),
                        (kappa_scale, "kappa_scale")):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    scores = np.zeros(paths.shape[0], dtype=np.float64)
    log_uniform = -math.log(2.0 * math.pi)
    for release in sorted(releases, key=lambda item: item.tracklet_id):
        observations = release.measurements
        if not observations:
            raise ValueError(f"track {release.tracklet_id!r} has no observations")
        if any(not 0 <= item.anchor_keyframe_idx < paths.shape[1]
               for item in observations):
            raise ValueError(
                f"track {release.tracklet_id!r} has an invalid observation")
        table = release.table
        log_lr = filter_lib._clipped_log_lr(  # noqa: SLF001
            table, catalog)
        endorsed = ~filter_lib._surprise_mask(table, log_lr)  # noqa: SLF001
        candidate_idx = np.nonzero(endorsed)[0]
        if not candidate_idx.size:
            scores += OBSERVATION_POWER * len(observations) * log_uniform
            continue

        log_identity = filter_lib._identity_log_weights(  # noqa: SLF001
            table, catalog, matcher_recall)[candidate_idx]
        log_identity -= np.logaddexp.reduce(log_identity)
        endorsed_mass = (
            1.0 if candidate_idx.size == catalog.n else matcher_recall)
        log_prior = (
            math.log1p(-pi0) + math.log(endorsed_mass) + log_identity)
        background = (
            math.log(pi0 + (1.0 - pi0) * (1.0 - endorsed_mass))
            + OBSERVATION_POWER * len(observations) * log_uniform)
        candidate_total = np.full(paths.shape[0], -np.inf)

        for start in range(0, candidate_idx.size,
                           filter_lib.CANDIDATE_BLOCK):
            block_idx = candidate_idx[
                start:start + filter_lib.CANDIDATE_BLOCK]
            accumulated = np.zeros(
                (paths.shape[0], len(block_idx)), dtype=np.float64)
            for observation in observations:
                pose = paths[:, observation.anchor_keyframe_idx]
                accumulated += OBSERVATION_POWER * (
                    filter_lib.committed_log_density(
                        pose[:, 0, None], pose[:, 1, None],
                        pose[:, 2, None], block_idx,
                        math.radians(observation.bearing_forward_cw_deg),
                        min(float(observation.kappa) * kappa_scale,
                            filter_lib.MAX_KAPPA),
                        OUTLIER_RATE, catalog,
                        range_max_m=(observation.range_max_m
                                     if range_cap else None),
                        range_softness=range_softness))
            block_total = np.logaddexp.reduce(
                accumulated + log_prior[start:start + len(block_idx)][None, :],
                axis=1)
            candidate_total = np.logaddexp(candidate_total, block_total)
        scores += np.logaddexp(background, candidate_total)
    return scores


def _advance_pose(pose: np.ndarray, delta) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float64).copy()
    result[2] = geo.wrap_rad(result[2] + delta.delta_yaw_cw_rad)
    result[0] += (
        delta.forward_m * math.sin(result[2])
        - delta.left_m * math.cos(result[2]))
    result[1] += (
        delta.forward_m * math.cos(result[2])
        + delta.left_m * math.sin(result[2]))
    return result


def _mode_poses(modes, label: str) -> np.ndarray:
    if not isinstance(modes, list) or not modes:
        raise ValueError(f"{label}.modes must be a non-empty list")
    previous_probability = math.inf
    result = []
    for offset, mode in enumerate(modes):
        if not isinstance(mode, dict) or set(mode) != _MODE_KEYS:
            raise ValueError(f"{label} mode {offset} fields differ")
        if mode["source_rank"] != offset + 1:
            raise ValueError(f"{label} source ranks must be contiguous")
        probability = _number(
            mode["source_probability"], f"{label} source_probability",
            minimum=0.0, maximum=1.0)
        if probability > previous_probability:
            raise ValueError(f"{label} probabilities must descend")
        previous_probability = probability
        east = _number(mode["east_m"], f"{label} east_m")
        north = _number(mode["north_m"], f"{label} north_m")
        heading = _number(
            mode["heading_world_cw_deg"], f"{label} heading",
            minimum=0.0, maximum=360.0)
        if heading == 360.0:
            raise ValueError(f"{label} heading must be in [0, 360)")
        for key in ("heading_index", "north_index", "east_index"):
            value = mode[key]
            if isinstance(value, bool) or not isinstance(value, int) \
                    or value < 0:
                raise ValueError(f"{label} {key} must be nonnegative integer")
        result.append((east, north, math.radians(heading)))
    return np.asarray(result, dtype=np.float64)


def _release_maps(snapshots, releases, n_keyframes):
    expected = {}
    for release in releases:
        if release.tracklet_id in expected:
            raise ValueError(f"duplicate release {release.tracklet_id!r}")
        expected[release.tracklet_id] = release.release_keyframe_idx
    actual = {}
    previous_keyframe = -1
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            raise ValueError("release snapshot must be an object")
        keyframe = snapshot.get("keyframe_idx")
        if isinstance(keyframe, bool) or not isinstance(keyframe, int) \
                or not previous_keyframe < keyframe < n_keyframes:
            raise ValueError("release snapshots must have increasing keyframes")
        previous_keyframe = keyframe
        ids = snapshot.get("released_tracklet_ids")
        if not isinstance(ids, list) or not ids or ids != sorted(set(ids)) \
                or any(not isinstance(value, str) or not value for value in ids):
            raise ValueError(
                "released_tracklet_ids must be non-empty, sorted, and unique")
        for tracklet_id in ids:
            if tracklet_id in actual:
                raise ValueError(f"grid repeats release {tracklet_id!r}")
            actual[tracklet_id] = keyframe
        modes = snapshot.get("modes")
        if snapshot.get("returned") != (
                len(modes) if isinstance(modes, list) else None):
            raise ValueError("release snapshot returned count disagrees")
        _mode_poses(modes, f"release snapshot {keyframe}")
    if actual != expected:
        raise ValueError("grid and epoch-1 release mappings differ")
    return expected


def causal_rerank(snapshots, map_states, releases, odometry, catalog, *,
                  pi0, matcher_recall, range_softness, range_cap,
                  kappa_scale):
    """Return immutable online estimates and one selection per release."""
    if not isinstance(map_states, list) or not map_states:
        raise ValueError("map_states must be a non-empty keyframe list")
    baseline = np.empty((len(map_states), 3), dtype=np.float64)
    for keyframe, state in enumerate(map_states):
        if not isinstance(state, dict) or set(state) != {
                "east_m", "north_m", "heading_world_cw_deg"}:
            raise ValueError(f"map state {keyframe} fields differ")
        baseline[keyframe] = (
            _number(state["east_m"], f"map state {keyframe} east"),
            _number(state["north_m"], f"map state {keyframe} north"),
            math.radians(_number(
                state["heading_world_cw_deg"],
                f"map state {keyframe} heading",
                minimum=0.0, maximum=360.0)))

    by_odometry = {item.keyframe_idx: item for item in odometry}
    if len(by_odometry) != len(odometry) or set(by_odometry) != set(
            range(1, len(map_states))):
        raise ValueError("odometry must cover map states 1..N-1 exactly")
    _release_maps(snapshots, releases, len(map_states))
    release_by_keyframe = {}
    for release in releases:
        release_by_keyframe.setdefault(
            release.release_keyframe_idx, []).append(release)

    snapshot_by_keyframe = {}
    for snapshot in snapshots:
        keyframe = snapshot.get("keyframe_idx")
        snapshot_by_keyframe[keyframe] = snapshot

    estimates = baseline.copy()
    selected_pose = None
    available = []
    reports = []
    for keyframe in range(len(map_states)):
        if selected_pose is not None:
            selected_pose = _advance_pose(
                selected_pose, by_odometry[keyframe])
            estimates[keyframe] = selected_pose
        snapshot = snapshot_by_keyframe.get(keyframe)
        if snapshot is None:
            continue
        arriving = sorted(
            release_by_keyframe[keyframe], key=lambda item: item.tracklet_id)
        available.extend(arriving)
        modes = snapshot["modes"]
        current_poses = _mode_poses(modes, f"release snapshot {keyframe}")
        paths = reconstruct_rigid_prefixes(
            current_poses, odometry, keyframe)
        # ponytail: recompute the growing score; cache only if target runtimes
        # make release-by-release evaluation impractical.
        scores = score_rigid_prefixes(
            paths, available, catalog, pi0=pi0,
            matcher_recall=matcher_recall,
            range_softness=range_softness, range_cap=range_cap,
            kappa_scale=kappa_scale)
        order = sorted(
            range(len(modes)),
            key=lambda index: (-float(scores[index]),
                               modes[index]["source_rank"]))
        selected_index = order[0]
        selected_pose = current_poses[selected_index].copy()
        estimates[keyframe] = selected_pose
        reports.append({
            "keyframe_idx": keyframe,
            "released_tracklet_ids": [item.tracklet_id for item in arriving],
            "cumulative_released_tracks": len(available),
            "cumulative_observations": sum(
                len(item.measurements) for item in available),
            "scored_modes": len(modes),
            "selected_source_rank": modes[selected_index]["source_rank"],
            "selected_raw_score_nats": float(scores[selected_index]),
            "runner_up_score_gap_nats": (
                float(scores[selected_index] - scores[order[1]])
                if len(order) > 1 else None),
            "selected_pose": {
                "east_m": float(selected_pose[0]),
                "north_m": float(selected_pose[1]),
                "heading_world_cw_deg": float(
                    math.degrees(selected_pose[2]) % 360.0),
            },
        })
    return estimates, reports


def point_metrics(estimates, truth) -> dict:
    """Posthoc point-error metrics, integrated over traveled distance."""
    estimates = np.asarray(estimates, dtype=np.float64)
    ordered_truth = sorted(truth, key=lambda item: item.keyframe_idx)
    if estimates.shape != (len(ordered_truth), 3) \
            or [item.keyframe_idx for item in ordered_truth] != list(
                range(len(ordered_truth))):
        raise ValueError("truth and estimates must cover keyframes 0..N-1")
    truth_xy = np.asarray([
        (item.east_m, item.north_m) for item in ordered_truth])
    error = np.hypot(
        estimates[:, 0] - truth_xy[:, 0],
        estimates[:, 1] - truth_xy[:, 1])
    cumulative = metrics.cumulative_distance_m(ordered_truth)
    distance = np.asarray([
        cumulative[keyframe] for keyframe in range(len(ordered_truth))])
    length = float(distance[-1])
    if not length > 0.0:
        raise ValueError("distance-normalized metrics need a moving trajectory")
    intervals = np.diff(distance)

    def integral(values):
        return float(np.sum(
            0.5 * (values[:-1] + values[1:]) * intervals) / length)

    return {
        "metric_kind": "online_current_state_point_estimate_not_posterior_mass",
        "trajectory_length_m": length,
        "error_m_by_keyframe": [float(value) for value in error],
        "summary": {
            "rms_error_m": float(np.sqrt(np.mean(error * error))),
            "median_error_m": float(np.median(error)),
            "final_error_m": float(error[-1]),
            "distance_normalized_mean_error_m": integral(error),
            "distance_normalized_rms_error_m": math.sqrt(integral(error * error)),
            "distance_normalized_fraction_within_m": {
                f"{radius:g}": integral((error <= radius).astype(np.float64))
                for radius in RADII_M
            },
        },
    }


def _find_grid_input(payload, consumer_dir: Path):
    document = payload.get("localization_inputs")
    try:
        expected = artifact.ArtifactRef.from_dict(document)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"grid localization_inputs is not an ArtifactRef: {exc}") from exc
    if expected.kind != "localization_inputs":
        raise ValueError("grid must reference localization_inputs")
    config = payload.get("config")
    config_path = config.get("input_dir") if isinstance(config, dict) else None
    candidates = [Path(expected.path)]
    if isinstance(config_path, str) and config_path:
        candidates.append(Path(config_path))
    candidates.append(consumer_dir.parent / expected.version)
    mismatch = False
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_dir():
            continue
        source = export_ingest.load(candidate)
        if artifact.records_same_artifact(document, source.artifact_ref):
            return source, candidate
        mismatch = True
    if mismatch:
        raise ValueError("grid localization ArtifactRef does not match source")
    raise ValueError("grid localization source artifact is unavailable")


def load_handoff(grid_json: Path, data, input_dir: Path, releases):
    """Bind an epoch-5 causal-grid handoff to the exact epoch-1 input."""
    payload = _read_json(grid_json)
    if payload.get("schema") != GRID_SCHEMA:
        raise ValueError(f"grid schema must be {GRID_SCHEMA!r}")
    config = payload.get("config")
    availability = payload.get("availability")
    if not isinstance(config, dict) or config.get("availability") != "natural" \
            or payload.get("episode") is not None:
        raise ValueError("reranking requires a natural whole-trajectory grid")
    if not isinstance(availability, dict) \
            or availability.get("policy") != \
            "natural_track_close_with_eof_flush" \
            or availability.get("post_closure_processing_delay_s") != 0.0 \
            or availability.get(
                "historical_measurements_keep_original_anchors") is not True \
            or availability.get("past_scores_revised_after_replay") is not False:
        raise ValueError("grid availability contract differs")
    if data.meta.reducer.get("epoch_keyframes") != 1:
        raise ValueError("reranking input must use epoch_keyframes=1")

    source, source_dir = _find_grid_input(payload, Path(input_dir))
    if source.meta.reducer.get("epoch_keyframes") != 5:
        raise ValueError("grid source must use epoch_keyframes=5")
    if source.artifact_ref.dataset != data.artifact_ref.dataset \
            or source.n_keyframes != data.n_keyframes:
        raise ValueError("grid and reranker dataset/keyframe counts differ")
    if source.manifest.upstreams != data.manifest.upstreams:
        raise ValueError("grid and reranker input lineages differ")
    if source.meta.nominal_forward != data.meta.nominal_forward:
        raise ValueError("grid and reranker nominal-forward calibration differs")
    for filename, label in (
            ("truth.jsonl", "truth"),
            ("landmarks.json", "catalog"),
            ("tier1_tables.json", "tables"),
            ("motion_source.csv", "motion")):
        if artifact.sha256_file(source_dir / filename) != artifact.sha256_file(
                Path(input_dir) / filename):
            raise ValueError(f"grid and reranker {label} differ")
    catalog_meta = (
        "anchor_lat_deg", "anchor_lon_deg", "max_visible_range_m",
        "landmark_position_sigma_m", "prior_region")
    if any(getattr(source.meta, key) != getattr(data.meta, key)
           for key in catalog_meta):
        raise ValueError("grid and reranker catalog metadata differ")

    mode_source = payload.get("release_top_modes")
    expected_mode_metadata = {
        "source": "causal_current_posterior_after_atomic_corelease_replay",
        "pose_frame": POSE_FRAME,
        "source_rank_semantics": RANK_SEMANTICS,
        "source_probability_semantics": PROBABILITY_SEMANTICS,
    }
    if not isinstance(mode_source, dict) or any(
            mode_source.get(key) != value
            for key, value in expected_mode_metadata.items()):
        raise ValueError("grid release-mode handoff contract differs")
    snapshots = mode_source.get("snapshots")
    map_source = payload.get("map_state_by_keyframe")
    if not isinstance(map_source, dict) \
            or map_source.get("source") != \
            "causal_current_joint_grid_state_argmax" \
            or map_source.get("keyframe_order") != \
            "list_index_equals_keyframe_idx":
        raise ValueError("grid current-state handoff contract differs")
    map_states = map_source.get("states")
    if not isinstance(map_states, list) or len(map_states) != data.n_keyframes:
        raise ValueError("grid current-state count differs from trajectory")

    profile_document = payload.get("odometry_profile")
    if not isinstance(profile_document, dict) \
            or profile_document.get("name") not in \
            odometry_profiles.PROFILE_CHOICES:
        raise ValueError("grid has no supported odometry profile")
    seed = config.get("odometry_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("grid odometry_seed must be nonnegative")
    profile_name = profile_document["name"]
    source_odometry, source_profile = odometry_profiles.derive(
        source_dir, source, profile_name, noise_seed=seed)
    odometry, profile = odometry_profiles.derive(
        Path(input_dir), data, profile_name, noise_seed=seed)
    if profile_document != source_profile or source_profile != profile \
            or source_odometry != odometry:
        raise ValueError("grid and reranker effective odometry/profile differ")

    range_cap = config.get("range_cap")
    if isinstance(range_cap, bool) or not isinstance(range_cap, int) \
            or range_cap not in (0, 1):
        raise ValueError("grid range_cap must be integer 0 or 1")
    scoring = {
        "pi0": _number(config.get("pi0"), "grid pi0"),
        "matcher_recall": _number(
            config.get("matcher_recall"), "grid matcher_recall"),
        "range_softness": _number(
            config.get("range_softness"), "grid range_softness"),
        "range_cap": bool(range_cap),
        "kappa_scale": _number(
            config.get("kappa_scale"), "grid kappa_scale"),
    }
    # Validate the exact release mapping before expensive scoring.
    _release_maps(snapshots, releases, data.n_keyframes)
    return payload, snapshots, map_states, odometry, profile, scoring, {
        "grid_localization_inputs": source.artifact_ref.to_dict(),
        "epoch5_reducer": source.meta.reducer,
        "epoch1_reducer": data.meta.reducer,
        "odometry_seed": seed,
    }


def main(argv=None):
    from experimental.overhead_matching.swag.farfield.localization import (
        release_schedule as release_schedule_lib,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--grid_json", type=Path, required=True)
    parser.add_argument("--release_schedule", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    started = time.monotonic()
    data = export_ingest.load(args.input_dir)
    if len(data.truth) != data.n_keyframes:
        raise ValueError("causal reranking requires truth at every keyframe")
    releases = release_schedule_lib.load_sidecar(
        args.release_schedule, data)
    (grid_payload, snapshots, map_states, odometry, profile, scoring,
     binding) = load_handoff(
         args.grid_json, data, args.input_dir, releases)
    estimates, selections = causal_rerank(
        snapshots, map_states, releases, odometry, data.catalog, **scoring)
    baseline = np.asarray([
        (state["east_m"], state["north_m"],
         math.radians(state["heading_world_cw_deg"]))
        for state in map_states], dtype=np.float64)
    reranked_metrics = point_metrics(estimates, data.truth)
    baseline_metrics = point_metrics(baseline, data.truth)

    output = {
        "schema": OUTPUT_SCHEMA,
        "localization_inputs": data.artifact_ref.to_dict(),
        "grid_json": str(args.grid_json),
        "release_schedule": str(args.release_schedule),
        "grid_handoff": binding,
        "odometry_profile": profile,
        "config": {
            **scoring,
            "observation_power": OBSERVATION_POWER,
            "association_outlier_rate": OUTLIER_RATE,
        },
        "scoring_contract": {
            "availability": "natural_atomic_whole_track_at_close_or_eof",
            "path": "effective_odometry_rigid_prefix_from_current_grid_mode",
            "identity": "one_endorsed_identity_marginalized_per_track",
            "identity_model_relation": (
                "static_whole_track_diagnostic_not_filter_renewal_hmm"),
            "background": "null_plus_unendorsed_uniform",
            "outlier": "per_observation_uniform",
            "historical_measurement_anchors_preserved": True,
            "source_probability_used": False,
            "truth_used_in_selection": False,
            "past_estimates_revised": False,
            "between_release_estimate": "selected_pose_odometry_propagation",
        },
        "counts": {
            "keyframes": data.n_keyframes,
            "release_keyframes": len(snapshots),
            "tracks": len(releases),
            "observations": sum(len(item.measurements) for item in releases),
        },
        "estimate_by_keyframe": [{
            "keyframe_idx": keyframe,
            "east_m": float(pose[0]),
            "north_m": float(pose[1]),
            "heading_world_cw_deg": float(math.degrees(pose[2]) % 360.0),
        } for keyframe, pose in enumerate(estimates)],
        "release_selections": selections,
        "point_metrics": reranked_metrics,
        "causal_grid_joint_map_baseline_point_metrics": baseline_metrics,
        "source_grid_primary_mass_summary": grid_payload.get("summary"),
        "runtime_seconds": time.monotonic() - started,
    }
    args.out.write_text(json.dumps(output, indent=1) + "\n")
    print(
        f"reranked {len(snapshots)} release prefixes; dn<=500 "
        f"{reranked_metrics['summary']['distance_normalized_fraction_within_m']['500']:.4f}; "
        f"final error {reranked_metrics['summary']['final_error_m']:.1f} m")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
