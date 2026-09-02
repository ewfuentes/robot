"""Build immutable farfield artifacts and then run localization.

``build_dir`` is mutable orchestration state.  It contains one immutable,
fully resolved ``build_config.json`` and may contain operational logs, but no
scientific outputs.  Every scientific stage publishes a typed artifact in its
own versioned lane.  ``run_dir`` is reserved for the completed localization
execution under ``runs/<experiment>/<localization_run>/``.

Create and execute a build with::

  bazel run //experimental/overhead_matching/swag/farfield:pipeline -- \\
      new-build --dataset boston_harbor_leg2 --build_name b001 \\
      --config /path/to/harbor_example.yaml
  bazel run //experimental/overhead_matching/swag/farfield:pipeline -- \\
      run --build_dir /data/farfield_matching/builds/boston_harbor_leg2/b001

There is no ``--force`` mode.  Published artifacts are immutable.  A changed
stage configuration or changed upstream identity requires a new downstream
artifact version; orchestration refuses to reuse a stale descendant.
"""

from __future__ import annotations

import argparse
import json
import copy
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from experimental.overhead_matching.swag.farfield import (
    artifact,
    artifact_identity,
    artifact_recipe,
    build_config,
    code_provenance,
    nominal_forward,
    paths as paths_lib,
    provenance,
)
from experimental.overhead_matching.swag.farfield.localization import run_identity
from experimental.overhead_matching.swag.farfield.matching import identity_review


FF = "//experimental/overhead_matching/swag/farfield"
WORKSPACE = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
LOCALIZATION_RUN_KIND = "localization_run"
VIEWER_TARGET = f"{FF}/localization:viewer"
VIEWER_GENERATOR = VIEWER_TARGET


def _text(*, choices: tuple[str, ...] | None = None
          ) -> build_config.ValueSpec:
    return build_config.ValueSpec((str,), choices=choices, nonempty=True)


def _integer(*, minimum: int = 0,
             maximum: int | None = None) -> build_config.ValueSpec:
    return build_config.ValueSpec((int,), minimum=minimum, maximum=maximum)


def _number(*, minimum: float = 0.0,
            maximum: float | None = None) -> build_config.ValueSpec:
    return build_config.ValueSpec(
        (int, float), minimum=minimum, maximum=maximum)


def _positive_number(*, maximum: float | None = None
                     ) -> build_config.ValueSpec:
    return build_config.ValueSpec(
        (int, float), exclusive_minimum=0.0, maximum=maximum)


def _open_probability() -> build_config.ValueSpec:
    return build_config.ValueSpec(
        (int, float), exclusive_minimum=0.0, exclusive_maximum=1.0)



def _boolean() -> build_config.ValueSpec:
    return build_config.ValueSpec((bool,))


# Exact leaf schema: an unknown key is an error, not an ignored pseudo-setting.
CONFIG_SCHEMA = {
    "experiment.name": _text(),
    "artifacts.frame_landmarks_version": _text(),
    "artifacts.pinhole_images_version": _text(),
    "artifacts.object_tracks_version": _text(),
    "artifacts.semantic_audits_version": _text(),
    "artifacts.bearing_observations_version": _text(),
    "artifacts.landmark_matches_version": _text(),
    "artifacts.alignment_diagnostics_version": _text(),
    "artifacts.localization_inputs_version": _text(),
    "artifacts.catalogs_version": _text(),
    "extraction.model": _text(),
    "extraction.prompt_type": _text(),
    "extraction.pinhole_resolution": _integer(minimum=1),
    "extraction.media_resolution": _text(),
    "extraction.thinking_level": _text(),
    "execution.llm_transport": _text(choices=("batch", "on_demand")),
    "execution.batch_gcs_prefix": build_config.ValueSpec(
        (str,), allow_none=True, nonempty=True),
    "execution.approve_cost": _boolean(),
    "cost.limit_usd": _number(),
    "ingest.fov_deg": _number(maximum=180.0),
    "ingest.seam_gap_norm": _number(maximum=1000.0),
    "ingest.seam_min_y_iou": _number(maximum=1.0),
    "tracking.sam2_checkpoint": _text(),
    "tracking.range.k_start": _integer(),
    "tracking.range.k_end": _integer(),
    "tracking.reference_pano_width": _integer(minimum=1),
    "tracking.window_px": _integer(minimum=1),
    "tracking.window_extent_factor": _number(),
    "tracking.window_quantum": _integer(minimum=1),
    "tracking.window_max_px": _integer(minimum=1),
    "tracking.clean_iou": _number(maximum=1.0),
    "tracking.superset_min_inter_over_box": _number(maximum=1.0),
    "tracking.reanchor_min_inter_over_mask": _number(maximum=1.0),
    "tracking.superset_inter_over_mask": _number(maximum=1.0),
    "tracking.superset_max_inter_over_box": _number(maximum=1.0),
    "tracking.child_inter_over_box": _number(maximum=1.0),
    "tracking.child_max_inter_over_mask": _number(maximum=1.0),
    "tracking.weak_min_iou": _number(maximum=1.0),
    "tracking.weak_min_containment": _number(maximum=1.0),
    "tracking.weak_min_complement": _number(maximum=1.0),
    "tracking.birth_min_coverage": _number(maximum=1.0),
    "tracking.birth_max_spill": _number(maximum=1.0),
    "tracking.birth_min_dominant_cc": _number(maximum=1.0),
    "tracking.patience_keyframes": _integer(minimum=1),
    "tracking.patience_unsupported_keyframes": _integer(minimum=1),
    "tracking.min_mask_area_px": _integer(minimum=1),
    "tracking.drift_gate_px": _number(),
    "tracking.drift_patience": _integer(minimum=1),
    "tracking.fragment_min_dominant_cc": _number(maximum=1.0),
    "tracking.fragment_patience": _integer(minimum=1),
    "audit.model": _text(),
    "audit.min_supports": _integer(minimum=1),
    "audit.thinking_level": _text(),
    "audit.max_support_chips": _integer(minimum=1),
    "audit.max_context_chips": _integer(),
    "audit.max_description_samples": _integer(minimum=1),
    "audit.chip_height_px": _integer(minimum=1),
    "matching.model": _text(),
    "matching.query_batch": _integer(minimum=1),
    "matching.chunk_size": _integer(minimum=1),
    "matching.thinking_level": _text(),
    "matching.confidence_floor": _number(maximum=1.0),
    "matching.instance_max_rows": _integer(minimum=1),
    "bearing_observations.bearing_sigma_deg": _positive_number(),
    "gps_course.min_displacement_m": _positive_number(),
    "gps_course.smooth_window_s": _number(),
    "alignment_diagnostics.sun.n_frames": _integer(minimum=1),
    "alignment_diagnostics.sun.min_speed_mps": _number(),
    "alignment_diagnostics.sun.elevation_tolerance_deg": _positive_number(
        maximum=90.0),
    "alignment_diagnostics.sun.work_width": _integer(minimum=2),
    "alignment_diagnostics.sweep.coarse_step_deg": _positive_number(),
    "alignment_diagnostics.sweep.fine_step_deg": _positive_number(),
    "alignment_diagnostics.sweep.fine_halfwidth_deg": _positive_number(),
    "alignment_diagnostics.sweep.min_observations": _integer(minimum=1),
    "alignment_diagnostics.sweep.min_arc_deg": _number(maximum=360.0),
    "alignment_diagnostics.sweep.max_condition": _positive_number(),
    "alignment_diagnostics.sweep.min_tracklets": _integer(minimum=1),
    "alignment_diagnostics.sweep.min_support_frac": _number(maximum=1.0),
    "localization_inputs.motion_source": _text(),
    "localization_inputs.nominal_forward_calibration": _text(),
    "localization_inputs.identity_review_dir": build_config.ValueSpec(
        (str,), allow_none=True, nonempty=True),
    "localization_inputs.use_uninformative_tables": _boolean(),
    "localization_inputs.default_log_compatibility": build_config.ValueSpec(
        (int, float)),
    "localization_inputs.compatibility_clip": _positive_number(),
    "localization_inputs.reducer_epoch_keyframes": _integer(minimum=1),
    "localization_inputs.odometry_sigma_pair_m": _positive_number(),
    "localization_inputs.displacement_gate_m": _positive_number(),
    "localization_inputs.stationary_sigma_m": _positive_number(),
    "localization_inputs.slow_yaw_sigma_deg": _positive_number(),
    "localization_inputs.course_yaw_drift_sigma_deg": _positive_number(),
    "localization_inputs.imu_translation_noise_frac": _positive_number(),
    "localization_inputs.imu_yaw_noise_frac": _positive_number(),
    "localization_inputs.reverse_keyframe_ranges": build_config.ValueSpec(
        (list,)),
    "localization_inputs.reverse_annotation_source": _text(),
    "localization_inputs.max_visible_range_m": _positive_number(),
    "localization_inputs.landmark_position_sigma_m": _positive_number(),
    "localization.run_name": _text(),
    "localization.init": _text(choices=("uniform", "truth_position")),
    "localization.ablation_tags": build_config.ValueSpec((list,)),
    "localization.position_mass_radii_m": build_config.ValueSpec((list,)),
    "localization.prior_sigma_m": build_config.ValueSpec(
        (int, float), allow_none=True, minimum=0.0),
    "localization.n_particles": _integer(minimum=1),
    "localization.seed": _integer(),
    "localization.margin_m": _number(),
    "localization.pi0": _open_probability(),
    "localization.ess_resample_frac": _number(maximum=1.0),
    "localization.heading_random_walk_deg": _number(),
    "localization.resample_regularization": _number(),
    "localization.position_roughening_m": _number(),
    "localization.heading_roughening_deg": _number(),
    "localization.map_cell_size_m": _positive_number(),
    "localization.checkpoint_every": _integer(minimum=1),
    "localization.measurement_backend": _text(choices=("numpy", "torch")),
    "localization.association_persistence": _boolean(),
    "localization.association_renewal_rate": _positive_number(maximum=1.0),
    "localization.association_outlier_rate": _number(maximum=1.0),
    "localization.matcher_recall": _open_probability(),
    "localization.min_reported_responsibility": _number(maximum=1.0),
    "localization.resample_survival_floor": _integer(minimum=0),
    "localization.resample_survival_min_mass": _number(maximum=1.0),
    "localization.bearings_enabled": _boolean(),
    "localization.proposal.enabled": _boolean(),
    "localization.proposal.on_init": _boolean(),
    "localization.proposal.null_share_threshold": _number(maximum=1.0),
    "localization.proposal.null_share_window": _integer(minimum=1),
    "localization.proposal.null_share_min_fraction": _number(maximum=1.0),
    "localization.proposal.ess_floor_frac": _number(maximum=1.0),
    "localization.proposal.ess_floor_keyframes": _integer(minimum=1),
    "localization.proposal.refractory_keyframes": _integer(),
    "localization.proposal.max_tracklets": _integer(minimum=1),
    "localization.proposal.exhaustive_tuple_limit": _integer(minimum=1),
    "localization.proposal.tuple_samples_per_active_solution": _integer(
        minimum=1),
    "localization.proposal.min_particles_point_fix": _integer(minimum=1),
    "localization.proposal.min_particles_arc": _integer(minimum=1),
    "localization.proposal.min_particles_single": _integer(minimum=1),
    "localization.proposal.solution_cluster_position_m": _number(),
    "localization.proposal.solution_cluster_heading_deg": _number(),
    "localization.proposal.pose_diversity_weight": _number(),
    "localization.proposal.arc_length_reference_m": _number(minimum=1e-9),
    "localization.proposal.arc_length_weight_cap": _number(minimum=1.0),
    "localization.proposal.residual_tolerance_sigma": _number(),
    "localization.proposal.max_residual_tolerance_deg": _number(),
    "localization.proposal.window_keyframes": _integer(minimum=1),
    "localization.proposal.evidence_gate": _boolean(),
    "localization.proposal.evidence_gate_margin_nats": build_config.ValueSpec(
        (int, float)),
    "localization.proposal.evidence_gate_selection_charge": _boolean(),
    "localization.proposal.min_tracklets_for_injection": _integer(minimum=1),
    "localization.proposal.init_max_wait_keyframes": _integer(minimum=1),
    "localization.proposal.evidence_gate_samples": _integer(minimum=1),
    "localization.proposal.share_triple": _number(maximum=1.0),
    "localization.proposal.share_pair": _number(maximum=1.0),
    "localization.proposal.share_single": _number(maximum=1.0),
    "localization.proposal.inject_fraction": _number(maximum=1.0),
    "localization.proposal.injection_sigma_m": _number(),
    "localization.proposal.injection_heading_sigma_deg": _number(),
    "localization.proposal.max_injection_sigma_m": _number(),
    "localization.modes.enabled": _boolean(),
    "localization.modes.cell_size_m": _positive_number(),
    "localization.modes.heading_cell_deg": _positive_number(),
    "localization.modes.min_cell_weight": _number(maximum=1.0),
    "localization.modes.min_mode_weight": _number(maximum=1.0),
}


_REQUIRED = object()


def _value(config: dict, key: str, default=_REQUIRED) -> Any:
    """A resolved config value. Absent is an error unless a default is given.

    The default path exists only for `PRESENTATION_SCHEMA` keys, which are
    optional by design; everything else must be present or the build recipe is
    not fully resolved.
    """
    try:
        return build_config.value({"config": config}, key)
    except build_config.MissingConfigValue:
        # Only ABSENCE falls back. A malformed value must still raise, or a
        # typo in a presentation key would silently become the fallback --
        # the exact failure the no-defaults rule exists to prevent.
        if default is _REQUIRED:
            raise
        return default


# Presentation settings. They shape the published page and cannot change any
# scientific artifact -- retuning them does not invalidate the localization run
# the page displays. They are therefore OPTIONAL rather than required: making
# them required made every build recipe recorded before they existed
# unreadable, which was measured against the real root (all 24 of them). They
# are still validated when present and still rejected when misspelled.
PRESENTATION_SCHEMA = {
    "viewer.max_particles": _integer(minimum=1),
    "viewer.basemap_detail": _positive_number(),
    "viewer.embed_source_chips": _boolean(),
}

# What the viewer uses when a recipe predates the keys. Named here rather than
# left to the viewer's own argparse, so the orchestrator and the viewer cannot
# hold two different sets -- which is what putting them in the config was for.
_REQUIRED_ABSENT = object()

PRESENTATION_FALLBACK = {
    "max_particles": 900,
    "basemap_detail": 1.0,
    "embed_source_chips": True,
}

# The two schemas always travel together: validating with the required set but
# not the optional one rejects a perfectly good `viewer:` block as "unknown".
# Passed as one mapping so a caller cannot supply half of it.
SCHEMA_ARGS = {"schema": CONFIG_SCHEMA, "optional": PRESENTATION_SCHEMA}


def validate_pipeline_config(config: dict) -> None:
    """Validate exact types, scalar domains, and cross-field invariants."""
    build_config.validate_resolved(config, **SCHEMA_ARGS)
    start = _value(config, "tracking.range.k_start")
    end = _value(config, "tracking.range.k_end")
    if start > end:
        raise build_config.InvalidConfigValue(
            "tracking.range.k_start must be <= tracking.range.k_end")
    if _value(config, "tracking.window_px") > _value(
            config, "tracking.window_max_px"):
        raise build_config.InvalidConfigValue(
            "tracking.window_px must be <= tracking.window_max_px")
    sigma_pair = _value(config, "localization_inputs.odometry_sigma_pair_m")
    stationary_sigma = _value(
        config, "localization_inputs.stationary_sigma_m")
    if stationary_sigma < sigma_pair:
        raise build_config.InvalidConfigValue(
            "localization_inputs.stationary_sigma_m must be >= "
            "localization_inputs.odometry_sigma_pair_m")
    if (_value(config, "localization_inputs.identity_review_dir") is not None
            and _value(
                config, "localization_inputs.use_uninformative_tables")):
        raise build_config.InvalidConfigValue(
            "localization_inputs.identity_review_dir cannot be combined with "
            "localization_inputs.use_uninformative_tables")
    reverse_ranges = _value(
        config, "localization_inputs.reverse_keyframe_ranges")
    previous_end = 0
    for index, interval in enumerate(reverse_ranges):
        if (not isinstance(interval, list) or len(interval) != 2
                or any(type(value) is not int for value in interval)):
            raise build_config.InvalidConfigValue(
                "localization_inputs.reverse_keyframe_ranges"
                f"[{index}] must be [start, end] integers")
        range_start, range_end = interval
        if range_start < 1 or range_end < range_start:
            raise build_config.InvalidConfigValue(
                "localization_inputs.reverse_keyframe_ranges"
                f"[{index}] is not a positive inclusive range")
        if range_start <= previous_end:
            raise build_config.InvalidConfigValue(
                "localization_inputs.reverse_keyframe_ranges must be sorted "
                "and non-overlapping")
        previous_end = range_end
    transport = _value(config, "execution.llm_transport")
    gcs_prefix = _value(config, "execution.batch_gcs_prefix")
    if transport == "batch" and (not isinstance(gcs_prefix, str)
                                  or not gcs_prefix.startswith("gs://")):
        raise build_config.InvalidConfigValue(
            "execution.batch_gcs_prefix must be a gs:// URI for batch mode")
    if transport == "on_demand" and gcs_prefix is not None:
        raise build_config.InvalidConfigValue(
            "execution.batch_gcs_prefix must be null for on_demand mode")
    init = _value(config, "localization.init")
    prior_sigma = _value(config, "localization.prior_sigma_m")
    if (init == "truth_position"
            and (prior_sigma is None or prior_sigma <= 0)):
        raise build_config.InvalidConfigValue(
            "localization.prior_sigma_m must be positive for truth_position init")
    if init == "uniform" and prior_sigma is not None:
        raise build_config.InvalidConfigValue(
            "localization.prior_sigma_m must be null for uniform init")
    radii = _value(config, "localization.position_mass_radii_m")
    if (not radii
            or any(type(radius) not in (int, float)
                   for radius in radii)):
        raise build_config.InvalidConfigValue(
            "localization.position_mass_radii_m must contain numbers")
    numeric_radii = [float(radius) for radius in radii]
    if (any(not math.isfinite(radius) or radius <= 0.0
            for radius in numeric_radii)
            or numeric_radii != sorted(set(numeric_radii))):
        raise build_config.InvalidConfigValue(
            "localization.position_mass_radii_m must be finite, positive, "
            "sorted, and unique")
    if 500.0 not in numeric_radii:
        raise build_config.InvalidConfigValue(
            "localization.position_mass_radii_m must include the primary "
            "500 m radius")
    tags = _value(config, "localization.ablation_tags")
    if (any(not isinstance(tag, str) or not tag for tag in tags)
            or tags != sorted(set(tags))):
        raise build_config.InvalidConfigValue(
            "localization.ablation_tags must be sorted unique identifiers")
    for tag in tags:
        try:
            paths_lib.require_identifier(tag, "localization.ablation_tags")
        except paths_lib.PathContractError as exc:
            raise build_config.InvalidConfigValue(str(exc)) from exc
    shares = sum(_value(config, f"localization.proposal.{name}")
                 for name in ("share_single", "share_pair", "share_triple"))
    if not math.isclose(shares, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise build_config.InvalidConfigValue(
            "localization proposal shares must sum to 1.0")
    identifier_keys = [
        "experiment.name",
        "localization.run_name",
        *(f"artifacts.{kind}_version" for kind in paths_lib.ARTIFACT_KINDS),
    ]
    for key in identifier_keys:
        try:
            paths_lib.require_identifier(_value(config, key), key)
        except paths_lib.PathContractError as exc:
            raise build_config.InvalidConfigValue(str(exc)) from exc


def load_pipeline_config(path: Path) -> dict:
    """Load YAML without permitting ambiguous duplicate mapping keys."""
    import yaml

    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_unique_mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    "found an unhashable mapping key", key_node.start_mark,
                ) from exc
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    f"found duplicate key {key!r}", key_node.start_mark,
                )
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_unique_mapping)
    try:
        raw = yaml.load(Path(path).read_text(), Loader=UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise build_config.InvalidConfigValue(
            f"invalid pipeline config YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise build_config.InvalidConfigValue(
            "pipeline config YAML must contain a top-level mapping")
    return raw


VERSION_KEYS = {
    kind: f"artifacts.{kind}_version"
    for kind in paths_lib.ARTIFACT_KINDS
}


@dataclass(frozen=True)
class StageSpec:
    outputs: tuple[str, ...]
    upstreams: tuple[str, ...]
    config_prefixes: tuple[str, ...]
    target: str
    # Recorded build inputs this stage does NOT read, and which therefore stay
    # out of its artifacts' identity. Everything else recorded is in: see
    # `identity_inputs` for why the list runs this way round. Forgetting an
    # entry here costs a rebuild that was not needed, which is loud; the
    # opposite default would cost a stale artifact used in silence.
    inputs_not_consumed: tuple[str, ...] = ()


STAGE_SPECS = {
    "extract": StageSpec(
        outputs=(paths_lib.PINHOLE_IMAGES, paths_lib.FRAME_LANDMARKS),
        upstreams=(),
        config_prefixes=("extraction", "execution", "cost"),
        target=f"{FF}/extraction:extract_landmarks",
        # Extraction reads panoramas and a prompt. It never opens the tracker
        # weights, the motion table, the mount calibration or the catalog, so
        # correcting any of those must not invalidate paid model calls.
        inputs_not_consumed=(
            "sam2_checkpoint_sha256", "motion_source_sha256",
            "nominal_forward_sha256", "video_sha256",
            "catalog_manifest_digest", "catalog_content_digest")),
    "track": StageSpec(
        outputs=(paths_lib.OBJECT_TRACKS,),
        upstreams=(paths_lib.PINHOLE_IMAGES, paths_lib.FRAME_LANDMARKS),
        config_prefixes=("ingest", "tracking", "gps_course"),
        target=f"{FF}/tracking:run_tracking",
        # Tracking runs SAM2 over the video and the GPS course, so the
        # checkpoint bytes, the video bytes and the motion table are all
        # determinants. It does not consult the mount calibration or the map.
        inputs_not_consumed=(
            "nominal_forward_sha256", "catalog_manifest_digest",
            "catalog_content_digest")),
    "audit": StageSpec(
        outputs=(paths_lib.SEMANTIC_AUDITS,),
        upstreams=(paths_lib.OBJECT_TRACKS, paths_lib.FRAME_LANDMARKS),
        config_prefixes=("ingest", "audit", "execution", "cost"),
        target=f"{FF}/tracking:audit_requests",
        # The audit shows a model chips cut from panoramas. No weights, no
        # motion, no calibration, no map.
        inputs_not_consumed=(
            "sam2_checkpoint_sha256", "motion_source_sha256",
            "nominal_forward_sha256", "video_sha256",
            "catalog_manifest_digest", "catalog_content_digest")),
    "bearings": StageSpec(
        outputs=(paths_lib.BEARING_OBSERVATIONS,),
        upstreams=(paths_lib.OBJECT_TRACKS, paths_lib.SEMANTIC_AUDITS),
        config_prefixes=("bearing_observations",
                         "tracking.reference_pano_width"),
        target=f"{FF}/tracking:build_bearing_observations",
        # Camera-frame bearings are geometry over audited tracks. The mount
        # calibration turns them into world bearings LATER, in localization
        # inputs, which is why it is not a determinant here.
        inputs_not_consumed=(
            "sam2_checkpoint_sha256", "motion_source_sha256",
            "nominal_forward_sha256", "video_sha256",
            "catalog_manifest_digest", "catalog_content_digest")),
    "match": StageSpec(
        outputs=(paths_lib.LANDMARK_MATCHES,),
        upstreams=(paths_lib.OBJECT_TRACKS, paths_lib.SEMANTIC_AUDITS,
                   paths_lib.CATALOGS),
        config_prefixes=("matching", "execution", "cost"),
        target=f"{FF}/matching:match_landmarks",
        # Matching reads the catalog (as a typed upstream AND as a recorded
        # digest) and the audit's names. Not the weights, motion or mount.
        inputs_not_consumed=(
            "sam2_checkpoint_sha256", "motion_source_sha256",
            "nominal_forward_sha256", "video_sha256")),
    "diagnostics": StageSpec(
        outputs=(paths_lib.ALIGNMENT_DIAGNOSTICS,),
        upstreams=(paths_lib.BEARING_OBSERVATIONS,),
        config_prefixes=("alignment_diagnostics",
                         "localization_inputs.nominal_forward_calibration"),
        target=f"{FF}/calibration:build_alignment_diagnostics",
        # Diagnostics compare bearings against the GPS course and the approved
        # nominal-forward record, so both are determinants.
        inputs_not_consumed=(
            "sam2_checkpoint_sha256", "video_sha256",
            "catalog_manifest_digest", "catalog_content_digest")),
    "localization_inputs": StageSpec(
        outputs=(paths_lib.LOCALIZATION_INPUTS,),
        upstreams=(paths_lib.BEARING_OBSERVATIONS,
                   paths_lib.LANDMARK_MATCHES, paths_lib.CATALOGS),
        config_prefixes=("localization_inputs", "gps_course"),
        target=f"{FF}/localization:build_export",
        # The export rotates camera bearings into the world through the
        # approved calibration and places landmarks from the catalog, using
        # the motion table for odometry. All three are determinants.
        inputs_not_consumed=("sam2_checkpoint_sha256", "video_sha256")),
    "localize": StageSpec(
        outputs=(LOCALIZATION_RUN_KIND,),
        upstreams=(paths_lib.LOCALIZATION_INPUTS,),
        config_prefixes=("localization",),
        target=f"{FF}/localization:run_export",
        # The filter consumes one typed localization_inputs artifact and
        # nothing else from the build's raw inputs.
        inputs_not_consumed=(
            "sam2_checkpoint_sha256", "motion_source_sha256",
            "nominal_forward_sha256", "video_sha256",
            "catalog_manifest_digest", "catalog_content_digest")),
}
STAGES = tuple(STAGE_SPECS)
PIPELINE_ARTIFACT_OWNER = {
    kind: stage
    for stage, spec in STAGE_SPECS.items()
    for kind in spec.outputs
    if kind in VERSION_KEYS and kind != paths_lib.PINHOLE_IMAGES
}


class StageContractError(ValueError):
    """A published stage output is partial, invalid, or stale."""


class StageDependencyError(ValueError):
    """A stage cannot start because a configured upstream is incomplete."""


def _binds_exact_pinhole_once(
        manifest, expected: artifact.ArtifactRef) -> bool:
    """Whether a build-scoped artifact binds only the configured pinhole ref."""
    return tuple(
        ref for ref in manifest.upstreams
        if ref.kind == paths_lib.PINHOLE_IMAGES) == (expected,)


def _prefixed_keys(prefixes: tuple[str, ...]) -> list[str]:
    return sorted(key for key in CONFIG_SCHEMA
                  if any(key == prefix or key.startswith(prefix + ".")
                         for prefix in prefixes))


def stage_config_selection(stage: str, config: dict) -> dict:
    """The resolved settings that shape one stage, as flat dotted keys.

    Public because this is the exact dict `config_digest` is computed over,
    and `artifact_recipe` has to store precisely it -- storing anything else
    would make the recorded config unable to reproduce the recorded digest.
    """
    spec = STAGE_SPECS[stage]
    selected = {key: _value(config, key)
                for key in _prefixed_keys(spec.config_prefixes)}
    for kind in spec.outputs:
        if kind in VERSION_KEYS:
            selected[VERSION_KEYS[kind]] = _value(config, VERSION_KEYS[kind])
    return selected


def stage_contract(stage: str, config: dict) -> dict:
    """Small manifest value proving which resolved settings shaped a stage."""
    return {
        "schema": "farfield_pipeline_stage/v1",
        "stage": stage,
        "config_digest": artifact.sha256_json(
            stage_config_selection(stage, config)),
    }


def versions_from_config(config: dict) -> dict[str, str]:
    return {kind: _value(config, key) for kind, key in VERSION_KEYS.items()}


def localization_run_dir(paths: paths_lib.FarfieldPaths, config: dict, *,
                         build_identity: str) -> Path:
    version = run_identity.localization_run_version(
        _value(config, "localization.run_name"),
        _value(config, "artifacts.object_tracks_version"), build_identity)
    return paths.experiment_dir(_value(config, "experiment.name")) / version


def _output_descriptors(paths: paths_lib.FarfieldPaths, config: dict,
                        stage: str, *,
                        build_identity: str) -> list[tuple[str, str, Path]]:
    output = []
    for kind in STAGE_SPECS[stage].outputs:
        if kind == LOCALIZATION_RUN_KIND:
            path = localization_run_dir(
                paths, config, build_identity=build_identity)
            output.append((kind, path.name, path))
        else:
            version = _value(config, VERSION_KEYS[kind])
            output.append((kind, version, paths.artifact(kind, version)))
    return output


def build_inputs_of(document: dict) -> dict[str, str]:
    """The inputs a build recorded, as the identity term."""
    inputs = document.get("inputs")
    return dict(inputs) if isinstance(inputs, dict) else {}


def expected_artifact_identity(paths: paths_lib.FarfieldPaths, config: dict,
                               kind: str, *,
                               build_inputs: Mapping[str, str]
                               ) -> str:
    """The identity an artifact of `kind` would have under this recipe."""
    owner = PIPELINE_ARTIFACT_OWNER[kind]
    upstreams = tuple(
        _configured_ref(paths, config, upstream_kind,
                        build_inputs=build_inputs)
        for upstream_kind in STAGE_SPECS[owner].upstreams)
    return artifact_identity.compute(
        kind=kind, dataset=paths.dataset,
        stage_config_digest=stage_contract(owner, config)["config_digest"],
        upstreams=upstreams, build_inputs=build_inputs,
        inputs_not_consumed=STAGE_SPECS[owner].inputs_not_consumed)


def _configured_ref(paths: paths_lib.FarfieldPaths, config: dict,
                    kind: str, *,
                    build_inputs: Mapping[str, str] | None = None,
                    ) -> artifact.ArtifactRef:
    version = _value(config, VERSION_KEYS[kind])
    path = paths.artifact(kind, version)
    if not path.exists():
        raise StageDependencyError(
            f"required {kind} artifact is not published: {path}")
    try:
        ref = artifact.reference_from_manifest(
            path, expected_kind=kind, expected_dataset=paths.dataset,
            expected_version=version)
    except artifact.ArtifactError as exc:
        raise StageDependencyError(
            f"required {kind} artifact is invalid: {exc}") from exc
    owner = PIPELINE_ARTIFACT_OWNER.get(kind)
    manifest = None
    if build_inputs is not None and owner is not None:
        manifest = artifact.load_manifest(path)
        if manifest.config.get("orchestration") != stage_contract(owner, config):
            raise StageDependencyError(
                f"required {kind} artifact has a different resolved "
                "configuration; publish the configured upstream stage first")
        # One rule for every stage. Extract and track used to be special:
        # they carried a whole-build check "because they bind the large
        # raw/model inputs that begin the chain", and `stage_reuse` existed
        # solely to grant human-attested exceptions to it. Those inputs are
        # now IN the artifact identity (see `identity_inputs`), so the
        # ordinary check covers what the special case reached for, and the
        # exception mechanism has nothing left to except.
        if build_inputs is not None:
            expected = expected_artifact_identity(
                paths, config, kind, build_inputs=build_inputs)
            found = artifact_identity.recorded(manifest)
            if found != expected:
                raise StageDependencyError(artifact_identity.explain(
                    expected=expected, manifest=manifest, kind=kind))
    if kind == paths_lib.FRAME_LANDMARKS:
        if manifest is None:
            manifest = artifact.load_manifest(path)
        pinhole_ref = _configured_ref(
            paths, config, paths_lib.PINHOLE_IMAGES,
            build_inputs=build_inputs)
        if not _binds_exact_pinhole_once(manifest, pinhole_ref):
            raise StageDependencyError(
                "required frame_landmarks artifact does not bind the exact "
                "configured pinhole_images artifact once")
    return ref


def expected_upstream_refs(paths: paths_lib.FarfieldPaths, config: dict,
                           stage: str, *,
                           build_inputs: Mapping[str, str] | None = None,
                           ) -> tuple[artifact.ArtifactRef, ...]:
    refs = tuple(_configured_ref(
        paths, config, kind,
        build_inputs=build_inputs)
                 for kind in STAGE_SPECS[stage].upstreams)
    if stage != "localization_inputs":
        return refs
    review_dir = _value(config, "localization_inputs.identity_review_dir")
    if review_dir is None:
        return refs
    matching_ref = next(
        ref for ref in refs if ref.kind == paths_lib.LANDMARK_MATCHES)
    try:
        review_ref, _ = identity_review.load(
            Path(review_dir), expected_matching_ref=matching_ref,
            matching_dir=matching_ref.path)
    except (artifact.ArtifactError, identity_review.IdentityReviewError,
            OSError) as error:
        raise StageDependencyError(
            "identity-review gate is not satisfied: first complete the match "
            "stage, review its identity_review_draft.json, and publish the "
            f"typed review to {review_dir}: {error}") from error
    return (*refs, review_ref)


def completed_stage_refs(paths: paths_lib.FarfieldPaths, config: dict,
                         stage: str, *,
                         build_identity: str,
                         build_inputs: Mapping[str, str] | None = None,
                         ) -> tuple[artifact.ArtifactRef, ...]:
    """Return validated outputs, ``()`` when absent, or reject stale output.

    `build_identity` still names the localization RUN DIRECTORY -- it is a
    label there, not a gate. What decides whether an artifact may be reused is
    `build_inputs`, through `artifact_identity`.
    """
    descriptors = _output_descriptors(
        paths, config, stage, build_identity=build_identity)
    present = [path.exists() or path.is_symlink()
               for _, _, path in descriptors]
    if not any(present):
        return ()
    upstreams = expected_upstream_refs(
        paths, config, stage,
        build_inputs=build_inputs)
    contract = stage_contract(stage, config)
    refs = []
    for (kind, version, path), exists in zip(descriptors, present):
        if not exists:
            continue
        try:
            ref = artifact.reference_from_manifest(
                path, expected_kind=kind, expected_dataset=paths.dataset,
                expected_version=version)
            manifest = artifact.load_manifest(path)
        except artifact.ArtifactError as exc:
            raise StageContractError(
                f"stage {stage!r} output {path} is not complete: {exc}") from exc
        missing_upstreams = [
            expected for expected in upstreams
            if manifest.upstreams.count(expected) != 1]
        if missing_upstreams:
            raise StageContractError(
                f"stage {stage!r} output {path} was built from different "
                "upstream artifact identities; choose a new output version")
        if kind in PIPELINE_ARTIFACT_OWNER:
            if manifest.config.get("orchestration") != contract:
                raise StageContractError(
                    f"stage {stage!r} output {path} has a different resolved "
                    "configuration; choose a new output version")
            if build_inputs is not None:
                expected = expected_artifact_identity(
                    paths, config, kind, build_inputs=build_inputs)
                found = artifact_identity.recorded(manifest)
                if found != expected:
                    raise StageContractError(
                        f"stage {stage!r} output: " + artifact_identity.explain(
                            expected=expected, manifest=manifest, kind=kind))
        if kind == paths_lib.FRAME_LANDMARKS:
            try:
                pinhole_ref = _configured_ref(
                    paths, config, paths_lib.PINHOLE_IMAGES,
                    build_inputs=build_inputs)
            except StageDependencyError as exc:
                raise StageContractError(
                    f"stage {stage!r} frame_landmarks dependency is invalid: "
                    f"{exc}") from exc
            if not _binds_exact_pinhole_once(manifest, pinhole_ref):
                raise StageContractError(
                    f"stage {stage!r} frame_landmarks output does not bind "
                    "the exact configured pinhole_images artifact once")
        refs.append(ref)
    # Extraction publishes pinhole images first and frame landmarks second.
    # A crash between the two leaves one valid immutable artifact that must be
    # reusable; it is pending, not corrupt.  Every present output was still
    # validated above, so a stale partial publication fails closed.
    if not all(present):
        return ()
    return tuple(refs)


def stage_done(stage: str, paths: paths_lib.FarfieldPaths,
               config: dict, *, build_identity: str,
               build_inputs: Mapping[str, str] | None = None) -> bool:
    """Completion is a validated typed manifest, never an existence marker."""
    return bool(completed_stage_refs(
        paths, config, stage, build_identity=build_identity,
        build_inputs=build_inputs))


def _resolved_path(value: str, base: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def resolve_config_paths(config: dict, paths: paths_lib.FarfieldPaths) -> dict:
    """Return a deep copy with every external filesystem input absolute."""
    result = copy.deepcopy(config)
    tracking = result.get("tracking")
    inputs = result.get("localization_inputs")
    if isinstance(tracking, dict) and isinstance(
            tracking.get("sam2_checkpoint"), str):
        tracking["sam2_checkpoint"] = _resolved_path(
            tracking["sam2_checkpoint"], paths.models_root)
    if isinstance(inputs, dict):
        for key in ("motion_source", "nominal_forward_calibration"):
            if isinstance(inputs.get(key), str):
                inputs[key] = _resolved_path(inputs[key], paths.dataset_base)
        if isinstance(inputs.get("identity_review_dir"), str):
            inputs["identity_review_dir"] = _resolved_path(
                inputs["identity_review_dir"], paths.root)
    return result

def _source_video_inputs(paths: paths_lib.FarfieldPaths) -> dict[str, str]:
    try:
        video = paths.video
    except paths_lib.MissingInput:
        return {}
    if video.is_symlink() or not video.is_file():
        raise FileNotFoundError(
            f"source video must be a regular, non-symlink file: {video}")
    return {
        "video": str(video.resolve()),
        "video_sha256": artifact.sha256_file(video),
    }


# Every key `_validate_build_inputs` can record. Declared rather than inferred
# so `identity_inputs`' exclusion lists have something stable to be audited
# against, and self-checked below so the declaration cannot drift from the
# function: a recorded key missing from here fails at build creation, which is
# the cheapest possible moment.
RECORDED_INPUT_KEYS = frozenset({
    "farfield_root", "dataset_base", "source_config", "source_config_sha256",
    "sam2_checkpoint", "sam2_checkpoint_sha256",
    "motion_source", "motion_source_sha256",
    "nominal_forward_calibration", "nominal_forward_sha256",
    "catalog_manifest_digest", "catalog_content_digest",
    "video", "video_sha256",
    "identity_review_output_dir", "identity_review_phase",
    *paths_lib.DATASET_SOURCE_DIGEST_KEYS,
})


def _validate_build_inputs(paths: paths_lib.FarfieldPaths, config: dict,
                           source_config: Path) -> dict[str, str]:
    if not paths.dataset_base.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: "
                                f"{paths.dataset_base}")
    try:
        dataset_digests = paths_lib.dataset_source_digests(paths.dataset_base)
    except paths_lib.MissingInput as exc:
        raise FileNotFoundError(str(exc)) from exc
    checkpoint = Path(_value(config, "tracking.sam2_checkpoint"))
    motion = Path(_value(config, "localization_inputs.motion_source"))
    calibration = Path(_value(
        config, "localization_inputs.nominal_forward_calibration"))
    for label, path in (("SAM2 checkpoint", checkpoint),
                        ("motion source", motion),
                        ("nominal-forward calibration", calibration)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    nominal_forward.load(calibration, expected_dataset=paths.dataset)
    catalog = _configured_ref(paths, config, paths_lib.CATALOGS)
    result = {
        "farfield_root": str(paths.root.resolve()),
        "dataset_base": str(paths.dataset_base.resolve()),
        "source_config": str(source_config.resolve()),
        "source_config_sha256": artifact.sha256_file(source_config),
        "sam2_checkpoint": str(checkpoint),
        "sam2_checkpoint_sha256": artifact.sha256_file(checkpoint),
        "motion_source": str(motion),
        "motion_source_sha256": artifact.sha256_file(motion),
        "nominal_forward_calibration": str(calibration),
        "nominal_forward_sha256": artifact.sha256_file(calibration),
        "catalog_manifest_digest": catalog.manifest_digest,
        "catalog_content_digest": catalog.content_digest,
        **dataset_digests,
        **_source_video_inputs(paths),
    }
    review_dir = _value(config, "localization_inputs.identity_review_dir")
    if review_dir is not None:
        review_path = Path(review_dir)
        if review_path.exists() or review_path.is_symlink():
            raise FileExistsError(
                "localization_inputs.identity_review_dir is a post-match "
                "output gate and must be unoccupied when the build is created: "
                f"{review_path}")
        result.update({
            "identity_review_output_dir": str(review_path.resolve()),
            "identity_review_phase": "post_match_gate",
        })
    undeclared = sorted(set(result) - RECORDED_INPUT_KEYS)
    if undeclared:
        raise StageContractError(
            f"build records inputs absent from RECORDED_INPUT_KEYS: "
            f"{undeclared}. Add them there so artifact identity's exclusion "
            "lists can be audited against the real key set.")
    return result


def cmd_new_build(args) -> None:
    try:
        paths_lib.require_identifier(args.dataset, "--dataset")
        paths_lib.require_identifier(args.build_name, "--build_name")
    except paths_lib.PathContractError as exc:
        raise SystemExit(str(exc)) from exc
    root = Path(args.farfield_root or paths_lib.default_root())
    overrides = ({"dataset_base": Path(args.dataset_base)}
                 if args.dataset_base else {})
    paths = paths_lib.FarfieldPaths(
        dataset=args.dataset, root=root, overrides=overrides)
    source_config = Path(args.config)
    try:
        raw = load_pipeline_config(source_config)
    except FileNotFoundError:
        raise SystemExit(f"config file {source_config} not found") from None
    except build_config.InvalidConfigValue as exc:
        raise SystemExit(str(exc)) from exc
    config = resolve_config_paths(raw, paths)
    validate_pipeline_config(config)
    paths.versions.update(versions_from_config(config))
    inputs = _validate_build_inputs(paths, config, source_config)
    build_dir = paths.build_dir(args.build_name)
    path = build_config.create(
        build_dir, dataset=paths.dataset, config=config, **SCHEMA_ARGS,
        generator="farfield.pipeline new-build", inputs=inputs,
        notes=args.notes)
    print(f"build created: {build_dir}")
    print(f"config recorded: {path}")
    print(f"next: bazel run {FF}:pipeline -- run --build_dir {build_dir}")


def resolve_build(build_dir: Path) -> tuple[paths_lib.FarfieldPaths, dict]:
    build_dir = Path(build_dir)
    document = build_config.load(build_dir)
    config = document["config"]
    validate_pipeline_config(config)
    root = Path(document["inputs"]["farfield_root"])
    dataset_base = Path(document["inputs"]["dataset_base"])
    paths = paths_lib.FarfieldPaths(
        dataset=document["dataset"], root=root,
        versions=versions_from_config(config),
        overrides={"dataset_base": dataset_base})
    expected = paths.build_dir(build_dir.name).resolve()
    if build_dir.resolve() != expected:
        raise ValueError(
            f"build config says dataset/root resolve to {expected}, not "
            f"{build_dir.resolve()}")
    return paths, document


def _execution_flags(config: dict) -> list[Any]:
    flags: list[Any] = ["--cost_limit", _value(config, "cost.limit_usd")]
    if _value(config, "execution.approve_cost"):
        flags.append("--approve_cost")
    if _value(config, "execution.llm_transport") == "on_demand":
        flags.append("--online")
    else:
        flags.extend(["--gcs_prefix",
                      _value(config, "execution.batch_gcs_prefix")])
    return flags


def _stage_base(build_dir: Path, config: dict, stage: str) -> list[Any]:
    return ["--build_config", build_dir / build_config.BUILD_CONFIG_NAME,
            "--orchestration_config_digest",
            stage_contract(stage, config)["config_digest"]]


def stage_recipe(paths: paths_lib.FarfieldPaths, config: dict, stage: str, *,
                 build_inputs: Mapping[str, str]) -> dict:
    """The self-describing block this stage's artifact records.

    Built here for the same reason the identity is: the two terms it holds are
    per-stage orchestrator knowledge (`config_prefixes` and
    `inputs_not_consumed`) that a producer does not have.
    """
    return artifact_recipe.build(
        stage=stage,
        stage_config=stage_config_selection(stage, config),
        build_inputs=build_inputs,
        identity_upstreams=tuple(
            _configured_ref(paths, config, kind, build_inputs=build_inputs)
            for kind in STAGE_SPECS[stage].upstreams),
        inputs_not_consumed=STAGE_SPECS[stage].inputs_not_consumed)


def write_stage_recipe(paths: paths_lib.FarfieldPaths, build_dir: Path,
                       config: dict, stage: str, *,
                       build_inputs: Mapping[str, str]) -> Path:
    """Hand the recipe over as a file, not a flag.

    A resolved stage config is far too large for a command line, and the
    producer's job is only to record what it was handed -- exactly as with
    `--orchestration_config_digest` and `--artifact_identity`. The file lives
    in the build directory, which is orchestration state; once the artifact is
    published the artifact carries the content and the file is disposable.
    """
    recipe = stage_recipe(paths, config, stage, build_inputs=build_inputs)
    path = Path(build_dir) / f"{stage}.recipe.json"
    artifact.atomic_write_json(path, recipe)
    return path


def stage_identity_flags(paths: paths_lib.FarfieldPaths, config: dict,
                         stage: str, *,
                         build_inputs: Mapping[str, str],
                         build_dir: Path | None = None) -> list[Any]:
    """The `--artifact_identity` flag for one stage, or none if it has no
    gated output.

    Computed here and passed down rather than derived by the producer,
    because the formula needs per-stage knowledge the producer does not have:
    which recorded build inputs this stage actually reads
    (`StageSpec.inputs_not_consumed`). Computed at run time rather than in
    `build_commands`, because it reads the upstream artifacts' manifest
    digests and those upstreams may be built earlier in this same run.
    """
    gated = [kind for kind in STAGE_SPECS[stage].outputs
             if kind in PIPELINE_ARTIFACT_OWNER]
    if not gated:
        return []
    if len(gated) > 1:
        raise StageContractError(
            f"stage {stage!r} has more than one gated output {gated}; one "
            "--artifact_identity flag can no longer describe it")
    flags = ["--artifact_identity", expected_artifact_identity(
        paths, config, gated[0], build_inputs=build_inputs)]
    if build_dir is not None:
        flags += ["--artifact_recipe", write_stage_recipe(
            paths, build_dir, config, stage, build_inputs=build_inputs)]
    return flags


def build_commands(paths: paths_lib.FarfieldPaths, build_dir: Path,
                   config: dict, *,
                   build_identity: str) -> dict[str, list[Any]]:
    """Construct explicit artifact-to-artifact stage commands.

    Result-shaping values live in ``build_config.json``.  Child tools receive
    that immutable file plus explicit input/output directories, so no stage
    can resolve a different root, version, range, or calibration by accident.
    """
    outputs = {
        kind: paths.artifact(kind, _value(config, VERSION_KEYS[kind]))
        for kind in paths_lib.ARTIFACT_KINDS
    }
    common = ["--dataset", paths.dataset,
              "--dataset_base", paths.dataset_base]
    llm = _execution_flags(config)
    range_flags = ["--k_start", _value(config, "tracking.range.k_start"),
                   "--k_end", _value(config, "tracking.range.k_end")]
    track = common + [
        "--frame_landmarks_dir", outputs[paths_lib.FRAME_LANDMARKS],
        "--pinhole_dir", outputs[paths_lib.PINHOLE_IMAGES],
        "--checkpoint", _value(config, "tracking.sam2_checkpoint"),
        "--output_dir", outputs[paths_lib.OBJECT_TRACKS],
    ] + range_flags
    try:
        video = paths.video
        if video.is_file() and not video.is_symlink():
            track += ["--video", video]
    except paths_lib.MissingInput:
        pass
    commands = {
        "extract": [
            "bazel", "run", STAGE_SPECS["extract"].target, "--",
        ] + common + [
            "--pinhole_output_dir", outputs[paths_lib.PINHOLE_IMAGES],
            "--output_dir", outputs[paths_lib.FRAME_LANDMARKS],
        ] + _stage_base(build_dir, config, "extract") + llm,
        "track": [
            "bazel", "run", STAGE_SPECS["track"].target, "--",
        ] + track + _stage_base(build_dir, config, "track"),
        "audit": [
            "bazel", "run", STAGE_SPECS["audit"].target, "--",
            "--tracks_dir", outputs[paths_lib.OBJECT_TRACKS],
            "--frame_landmarks_dir", outputs[paths_lib.FRAME_LANDMARKS],
            "--output_dir", outputs[paths_lib.SEMANTIC_AUDITS], "--submit",
        ] + common + _stage_base(build_dir, config, "audit") + llm,
        "bearings": [
            "bazel", "run", STAGE_SPECS["bearings"].target, "--",
            "--tracks_dir", outputs[paths_lib.OBJECT_TRACKS],
            "--audit_dir", outputs[paths_lib.SEMANTIC_AUDITS],
            "--output_dir", outputs[paths_lib.BEARING_OBSERVATIONS],
        ] + common + _stage_base(build_dir, config, "bearings"),
        "match": [
            "bazel", "run", STAGE_SPECS["match"].target, "--",
            "--tracks_dir", outputs[paths_lib.OBJECT_TRACKS],
            "--audit_dir", outputs[paths_lib.SEMANTIC_AUDITS],
            "--catalog_dir", outputs[paths_lib.CATALOGS],
            "--output_dir", outputs[paths_lib.LANDMARK_MATCHES], "--submit",
        ] + common + _stage_base(build_dir, config, "match") + llm,
        "diagnostics": [
            "bazel", "run", STAGE_SPECS["diagnostics"].target, "--",
            "--observations_dir", outputs[paths_lib.BEARING_OBSERVATIONS],
            "--nominal_forward_calibration",
            _value(config, "localization_inputs.nominal_forward_calibration"),
            "--output_dir", outputs[paths_lib.ALIGNMENT_DIAGNOSTICS],
        ] + common + _stage_base(build_dir, config, "diagnostics"),
        "localization_inputs": [
            "bazel", "run", STAGE_SPECS["localization_inputs"].target, "--",
            "--observations_dir", outputs[paths_lib.BEARING_OBSERVATIONS],
            "--matching_dir", outputs[paths_lib.LANDMARK_MATCHES],
            "--catalog_dir", outputs[paths_lib.CATALOGS],
            "--motion_source",
            _value(config, "localization_inputs.motion_source"),
            "--nominal_forward_calibration",
            _value(config, "localization_inputs.nominal_forward_calibration"),
            "--landmark_position_sigma_m",
            _value(config, "localization_inputs.landmark_position_sigma_m"),
            "--output_dir", outputs[paths_lib.LOCALIZATION_INPUTS],
        ] + common + _stage_base(build_dir, config, "localization_inputs"),
        "localize": [
            "bazel", "run", STAGE_SPECS["localize"].target, "--",
            "--input_dir", outputs[paths_lib.LOCALIZATION_INPUTS],
            "--run_dir", localization_run_dir(
                paths, config, build_identity=build_identity),
        ] + _stage_base(build_dir, config, "localize"),
    }
    review_dir = _value(config, "localization_inputs.identity_review_dir")
    if review_dir is not None:
        commands["localization_inputs"].extend(
            ["--identity_review_dir", review_dir])
    return commands


def localization_viewer_dir(paths: paths_lib.FarfieldPaths, config: dict, *,
                            build_identity: str) -> Path:
    run_dir = localization_run_dir(
        paths, config, build_identity=build_identity)
    return run_dir.with_name(run_dir.name + ".viewer")


def viewer_config(config: dict) -> dict:
    """The recorded viewer settings, in the shape the viewer records them.

    One owner for the values the orchestrator passes on the command line and
    the values it later expects to find in the published manifest, so the two
    cannot disagree.
    """
    def setting(name):
        """The recipe's value, or the named presentation fallback.

        A recipe recorded before these keys existed has none, and refusing to
        render its page over a presentation setting would be the wrong trade
        -- the run it displays is unaffected either way. `PRESENTATION_SCHEMA`
        keeps them validated when they ARE present.
        """
        value = _value(config, f"viewer.{name}", default=_REQUIRED_ABSENT)
        return PRESENTATION_FALLBACK[name] if value is _REQUIRED_ABSENT \
            else value

    return {
        "max_particles": setting("max_particles"),
        # `float` because the viewer parses this flag with `type=float` and
        # records what it parsed: a config written as `1` rather than `1.0`
        # would otherwise never compare equal to the manifest it produced.
        "basemap_detail": float(setting("basemap_detail")),
        # Fixed for the canonical page rather than configurable: `--body_only`
        # emits an embeddable fragment, and the index chain links documents.
        "body_only": False,
        "embed_source_chips": setting("embed_source_chips"),
    }


def build_viewer_command(paths: paths_lib.FarfieldPaths, config: dict, *,
                         build_identity: str) -> list[Any]:
    """Construct the canonical viewer from exact scientific artifact inputs."""
    run_dir = localization_run_dir(
        paths, config, build_identity=build_identity)
    tracks_dir = paths.artifact(
        paths_lib.OBJECT_TRACKS,
        _value(config, VERSION_KEYS[paths_lib.OBJECT_TRACKS]))
    audit_dir = paths.artifact(
        paths_lib.SEMANTIC_AUDITS,
        _value(config, VERSION_KEYS[paths_lib.SEMANTIC_AUDITS]))
    catalog_dir = paths.artifact(
        paths_lib.CATALOGS, _value(config, VERSION_KEYS[paths_lib.CATALOGS]))
    settings = viewer_config(config)
    command = [
        "bazel", "run", VIEWER_TARGET, "--",
        "--run_dir", run_dir,
        "--output_dir", localization_viewer_dir(
            paths, config, build_identity=build_identity),
        "--tracks_dir", tracks_dir,
        "--audit_dir", audit_dir,
        "--feather", catalog_dir / "catalog.feather",
        "--max_particles", settings["max_particles"],
        "--basemap_detail", settings["basemap_detail"],
    ]
    if not settings["embed_source_chips"]:
        command.append("--no_source_chips")
    return command


def viewer_completed(paths: paths_lib.FarfieldPaths, config: dict, *,
                     build_identity: str,
                     build_inputs: Mapping[str, str] | None = None) -> bool:
    """Validate the deterministic viewer and every recorded input identity."""
    output_dir = localization_viewer_dir(
        paths, config, build_identity=build_identity)
    if not (output_dir.exists() or output_dir.is_symlink()):
        return False
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise StageContractError(
            f"viewer output is not a regular directory: {output_dir}")
    viewer_file = output_dir / "viewer.html"
    if (viewer_file.is_symlink() or not viewer_file.is_file()
            or viewer_file.stat().st_size == 0):
        raise StageContractError(
            f"viewer output is incomplete: {viewer_file}")

    run_dir = localization_run_dir(
        paths, config, build_identity=build_identity)
    try:
        run_ref = artifact.reference_from_manifest(
            run_dir, expected_kind=LOCALIZATION_RUN_KIND,
            expected_dataset=paths.dataset, expected_version=run_dir.name)
        tracks_ref = _configured_ref(
            paths, config, paths_lib.OBJECT_TRACKS,
            build_inputs=build_inputs)
        audits_ref = _configured_ref(
            paths, config, paths_lib.SEMANTIC_AUDITS,
            build_inputs=build_inputs)
        catalog_ref = _configured_ref(paths, config, paths_lib.CATALOGS)
        feather = Path(catalog_ref.path) / "catalog.feather"
        manifest = provenance.read(output_dir)
        expected_inputs = {
            "run_dir": str(run_dir.resolve()),
            "run_manifest_digest": run_ref.manifest_digest,
            "tracks_dir": str(Path(tracks_ref.path).resolve()),
            "tracks_manifest_digest": tracks_ref.manifest_digest,
            "audit_dir": str(Path(audits_ref.path).resolve()),
            "audit_manifest_digest": audits_ref.manifest_digest,
            "feather": str(feather.resolve()),
            "feather_sha256": artifact.sha256_file(feather),
            # `build_viewer_command` passes neither `--ghost` nor
            # `--satellite` nor the review-workbench pages, so these are
            # the viewer's stringified empties. Overlays and workbench
            # pages are hand-driven investigations, not build products.
            "ghosts": str([]),
            "satellite": "",
            "matcher_page": "",
            "matcher_page_sha256": "",
            "audit_page": "",
            "audit_page_sha256": "",
        }
    except (artifact.ArtifactError, OSError, ValueError) as error:
        raise StageContractError(
            f"cannot validate viewer output {output_dir}: {error}") from error
    expected_config = viewer_config(config)
    if (manifest.get("schema") != provenance.SCHEMA
            or manifest.get("generator") != VIEWER_GENERATOR
            or manifest.get("inputs") != expected_inputs
            or manifest.get("config") != expected_config):
        raise StageContractError(
            "viewer output was built from different inputs or settings; "
            f"move the stale side output before rebuilding: {output_dir}")
    return True


def run(command: list[Any], description: str, *, dry_run: bool = False) -> None:
    print(f"\n{'=' * 72}\n{description}\n{'=' * 72}")
    print("  $ " + " ".join(str(value) for value in command), flush=True)
    if dry_run:
        print("  [DRY RUN] skipped")
        return
    if WORKSPACE is None:
        raise SystemExit(
            "BUILD_WORKSPACE_DIRECTORY is unset: invoke through `bazel run`")
    started = time.time()
    result = subprocess.run([str(value) for value in command], cwd=WORKSPACE)
    print(f"\n  {description}: exit {result.returncode} in "
          f"{time.time() - started:.0f}s", flush=True)
    if result.returncode:
        raise SystemExit(f"{description} failed; stopping")


def _selected_stages(args, parser: argparse.ArgumentParser) -> list[str]:
    if args.only:
        return [args.only]
    lo, hi = STAGES.index(args.from_stage), STAGES.index(args.to_stage)
    if lo > hi:
        parser.error(f"--from {args.from_stage} comes after --to "
                     f"{args.to_stage}")
    return [stage for stage in STAGES[lo:hi + 1] if stage not in args.skip]


def cmd_run(args, parser: argparse.ArgumentParser) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    try:
        paths, document = resolve_build(args.build_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    config = document["config"]
    build_inputs = build_inputs_of(document)
    commands = build_commands(
        paths, Path(args.build_dir), config,
        build_identity=document["build_identity"])
    selected = _selected_stages(args, parser)
    print(f"dataset:          {paths.dataset}")
    print(f"build_dir:        {args.build_dir}")
    print("localization run: " + str(localization_run_dir(
        paths, config, build_identity=document["build_identity"])))
    print(f"stages:           {' -> '.join(selected)}")
    for stage in selected:
        try:
            if stage_done(
                    stage, paths, config,
                    build_identity=document["build_identity"],
                    build_inputs=build_inputs):
                print(f"\n-- {stage}: validated complete")
                continue
            expected_upstream_refs(
                paths, config, stage,
                build_inputs=build_inputs)
        except (StageContractError, StageDependencyError) as exc:
            raise SystemExit(str(exc)) from exc
        run(commands[stage] + stage_identity_flags(
                paths, config, stage, build_inputs=build_inputs,
                build_dir=Path(args.build_dir)),
            stage, dry_run=args.dry_run)
        if not args.dry_run:
            try:
                if not stage_done(
                        stage, paths, config,
                        build_identity=document["build_identity"],
                        build_inputs=build_inputs):
                    raise StageContractError(
                        f"{stage} exited successfully without publishing its "
                        "complete artifact manifest")
            except (StageContractError, StageDependencyError) as exc:
                raise SystemExit(str(exc)) from exc

    if "localize" in selected:
        try:
            complete = viewer_completed(
                paths, config, build_identity=document["build_identity"],
                build_inputs=build_inputs)
        except (StageContractError, StageDependencyError) as exc:
            raise SystemExit(str(exc)) from exc
        if complete:
            print("\n-- viewer: validated complete")
        else:
            run(build_viewer_command(
                paths, config, build_identity=document["build_identity"]),
                "viewer", dry_run=args.dry_run)
            if not args.dry_run:
                if not viewer_completed(
                    paths, config,
                    build_identity=document["build_identity"],
                    build_inputs=build_inputs):
                    raise SystemExit(
                        "viewer exited successfully without publishing its "
                        "complete side output")


def code_lineage(paths: paths_lib.FarfieldPaths, config: dict, *,
                 build_identity: str) -> str:
    """One line on whether this build's artifacts share a code state.

    `code_provenance` records the commit and diff that produced every
    artifact and deliberately never gates on them -- code changes constantly
    in a research tree and the artifacts cost money to rebuild. But a record
    nothing reads is a record nobody can act on, and the failure it exists to
    surface is real and silent: an evaluation whose legs were built either
    side of a fix is a wrong conclusion with nothing visibly wrong. So it is
    reported here, where anyone asking after a build's state already looks.
    """
    blocks = []
    for stage in STAGES:
        for kind, _, path in _output_descriptors(
                paths, config, stage, build_identity=build_identity):
            if not path.exists():
                continue
            try:
                block = artifact.load_manifest(path).code_provenance
            except (artifact.ArtifactError, OSError, ValueError):
                continue
            if isinstance(block, Mapping):
                blocks.append(block)
    try:
        return code_provenance.describe(code_provenance.lineage_summary(blocks))
    except code_provenance.CodeProvenanceError as error:
        return f"unreadable code provenance: {error}"


def cmd_status(args, parser: argparse.ArgumentParser) -> None:
    try:
        paths, document = resolve_build(args.build_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    config = document["config"]
    build_inputs = build_inputs_of(document)
    print(f"build {args.build_dir}")
    print("localization run: " + str(localization_run_dir(
        paths, config, build_identity=document["build_identity"])))
    for stage in STAGES:
        try:
            state = ("done" if stage_done(
                stage, paths, config,
                build_identity=document["build_identity"],
                build_inputs=build_inputs) else "pending")
            detail = ""
        except (StageContractError, StageDependencyError) as exc:
            state, detail = "INVALID", f" ({exc})"
        print(f"  {stage:<20} {state}{detail}")

    try:
        state = ("done" if viewer_completed(
            paths, config, build_identity=document["build_identity"],
            build_inputs=build_inputs)
                 else "pending")
        detail = ""
    except (StageContractError, StageDependencyError) as exc:
        state, detail = "INVALID", f" ({exc})"
    print(f"  {'viewer':<20} {state}{detail}")
    print("code lineage: " + code_lineage(
        paths, config, build_identity=document["build_identity"]))


def cmd_recipe(args, parser: argparse.ArgumentParser) -> None:
    """Say how one artifact was made, reading only that artifact.

    No build directory, no data root, no config. If this can answer, the
    artifact is reproducible from itself; if it cannot, that is the honest
    finding and the reason is printed.
    """
    try:
        manifest = artifact.load_manifest(args.artifact_dir)
    except (artifact.ArtifactError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(artifact_recipe.describe(manifest))
    try:
        artifact_recipe.verify_self_describing(manifest)
    except artifact_recipe.ArtifactRecipeError as exc:
        print(f"\nNOT self-describing: {exc}")
        raise SystemExit(1)
    print("\nself-describing: the identity recomputed from this manifest "
          "alone matches the identity it records")


def cmd_verify(args, parser: argparse.ArgumentParser) -> None:
    """Re-hash every artifact this build names and compare with its manifest.

    The integrity question, asked deliberately. The reuse checks in `run` and
    `status` read identity from manifests and never touch content, because
    identity does not depend on it -- so nothing else in the pipeline would
    notice a file that changed under a published artifact. This is what
    notices.
    """
    try:
        paths, document = resolve_build(args.build_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    config = document["config"]
    checked = failed = 0
    for kind in sorted(PIPELINE_ARTIFACT_OWNER):
        version = _value(config, VERSION_KEYS[kind])
        path = paths.artifact(kind, version)
        if not path.exists():
            print(f"  {kind:<24} absent")
            continue
        checked += 1
        try:
            artifact.open_artifact(
                path, expected_kind=kind, expected_dataset=paths.dataset,
                expected_version=version)
        except artifact.ArtifactError as exc:
            failed += 1
            print(f"  {kind:<24} CORRUPT: {exc}")
        else:
            print(f"  {kind:<24} contents match their manifest")
    print(f"\n{checked} artifact(s) checked, {failed} corrupt")
    if failed:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    new = sub.add_parser("new-build", help="record one resolved build recipe")
    new.add_argument("--dataset", required=True)
    new.add_argument("--farfield_root", type=Path, default=None)
    new.add_argument("--dataset_base", type=Path, default=None)
    new.add_argument("--build_name", required=True)
    new.add_argument("--config", type=Path, required=True)
    new.add_argument("--notes", default="")

    execute = sub.add_parser("run", help="execute an existing build")
    execute.add_argument("--build_dir", type=Path, required=True)
    execute.add_argument("--from", dest="from_stage", choices=STAGES,
                         default=STAGES[0])
    execute.add_argument("--to", dest="to_stage", choices=STAGES,
                         default=STAGES[-1])
    execute.add_argument("--only", choices=STAGES, default=None)
    execute.add_argument("--skip", action="append", default=[], choices=STAGES)
    execute.add_argument("--dry_run", action="store_true")

    status = sub.add_parser("status", help="validate build stage manifests")
    status.add_argument("--build_dir", type=Path, required=True)

    recipe = sub.add_parser(
        "recipe", help="how one artifact was made, from its manifest alone")
    recipe.add_argument("--artifact_dir", type=Path, required=True)

    verify = sub.add_parser(
        "verify", help="re-hash this build's artifacts and check integrity")
    verify.add_argument("--build_dir", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "new-build":
        cmd_new_build(args)
    elif args.command == "run":
        cmd_run(args, parser)
    elif args.command == "status":
        cmd_status(args, parser)
    elif args.command == "recipe":
        cmd_recipe(args, parser)
    elif args.command == "verify":
        cmd_verify(args, parser)


if __name__ == "__main__":
    main()
