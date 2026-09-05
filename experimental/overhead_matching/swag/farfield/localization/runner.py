"""One execution path for real and synthetic localization runs.

Input acquisition differs, but filter execution, every-keyframe primary metrics,
posterior-predictive diagnostics, manifest finalization, and atomic publication
are deliberately shared.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import msgspec

from common.python.serialization import msgspec_enc_hook
from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.localization import (
    filter as pf,
    metrics,
    run_io,
    structs,
)

TRUTH_POSITION_SCHEMA = "farfield_truth_position/v1"
BEARING_DIAGNOSTIC_SCHEMA = "farfield_posterior_predictive_bearings/v1"
BEARING_DIAGNOSTIC_NAME = "posterior_predictive_bearings.json"


@dataclasses.dataclass(frozen=True)
class ExecutionResult:
    manifest: structs.RunManifest
    history: pf.FilterHistory
    artifact_ref: artifact.ArtifactRef
    bearing_diagnostics: tuple[structs.BearingResidualDiagnostic, ...]
    position_mass_summary: dict | None


class PositionMassRecorder(pf.RunObserver):
    """Evaluation-only truth scorer with no feedback into the filter.

    The observer reads the completed posterior at each keyframe and retains only
    scalar masses.  It cannot alter the belief, prior, measurements, or proposal
    path, so truth remains an evaluation reference rather than a filter input.
    """

    def __init__(self, truth: list, config: structs.PositionMassMetricConfig):
        self._truth = {record.keyframe_idx: record for record in truth}
        self._config = config
        self.by_keyframe: dict[int, dict[str, float]] = {}

    def keyframe_end(self, keyframe_idx, belief, unused_health):
        truth = self._truth[keyframe_idx]
        self.by_keyframe[keyframe_idx] = {
            metrics.position_mass_metric_key(self._config, radius_m):
                metrics.mass_within_radius(
                    belief, truth.east_m, truth.north_m, radius_m)
            for radius_m in self._config.radii_m
        }


def _validate_metadata(manifest: structs.RunManifest, truth: list) -> None:
    if manifest.ablation_tags != sorted(set(manifest.ablation_tags)):
        raise ValueError("ablation_tags must be sorted and unique")
    if any(not isinstance(tag, str) or not tag
           for tag in manifest.ablation_tags):
        raise ValueError("ablation_tags must contain non-empty strings")
    if manifest.run_kind == "evaluation":
        if (manifest.ablation_tags
                or manifest.initialization_kind != "uniform"
                or not manifest.bearings_consumed):
            raise ValueError(
                "evaluation runs require uniform initialization, bearings, "
                "and no ablation tags")
    elif manifest.run_kind == "diagnostic_control":
        if not manifest.ablation_tags:
            raise ValueError(
                "production diagnostic controls must identify an ablation tag")
    elif manifest.run_kind != "synthetic":
        raise ValueError(
            "run_kind must be evaluation, diagnostic_control, or synthetic")

    if truth:
        expected_keyframes = list(range(manifest.n_keyframes))
        if [record.keyframe_idx for record in truth] != expected_keyframes:
            raise ValueError(
                "truth positions must cover every keyframe to record primary "
                "posterior-mass metrics")
        if manifest.truth_position_schema != TRUTH_POSITION_SCHEMA:
            raise ValueError(
                f"truth_position_schema must be {TRUTH_POSITION_SCHEMA!r}")
        if manifest.position_mass_metric is None:
            raise ValueError(
                "truth-bearing runs require position_mass_metric config")
        canonical = metrics.position_mass_metric_config(
            manifest.position_mass_metric.radii_m)
        if manifest.position_mass_metric != canonical:
            raise ValueError(
                "position_mass_metric identity/version is not canonical")
    else:
        if manifest.truth_position_schema is not None:
            raise ValueError(
                "truth_position_schema is set but the truth artifact is empty")
        if manifest.truth_position_artifact is not None:
            raise ValueError(
                "truth_position_artifact is set but the truth artifact is empty")
        if manifest.position_mass_metric is not None:
            raise ValueError(
                "position_mass_metric is set but the truth artifact is empty")

    source = manifest.truth_position_artifact
    if source is not None and (
            not isinstance(source, dict)
            or not source
            or any(not isinstance(key, str) or not key
                   or not isinstance(value, str) or not value
                   for key, value in source.items())):
        raise ValueError(
            "truth_position_artifact must map non-empty strings to non-empty "
            "strings")


def execute_localization(
        run_dir: Path, manifest: structs.RunManifest, *,
        catalog, truth: list, odometry: list, measurements: list, tables: dict,
        dataset: str, version: str,
        upstreams: tuple[artifact.ArtifactRef, ...] = (),
        artifact_config: dict | None = None,
        generator: str = "farfield.localization.runner",
        arguments: tuple[str, ...] = (),
        extra_outputs: dict[str, bytes] | None = None) -> ExecutionResult:
    """Run, instrument, finalize, and atomically publish one localization run."""
    _validate_metadata(manifest, truth)
    recorder = (PositionMassRecorder(truth, manifest.position_mass_metric)
                if truth else None)
    history = pf.run_filter(
        manifest.filter_config, catalog, odometry, measurements, tables,
        observer=recorder)
    if recorder is not None:
        for record in history.health:
            record.position_probability_mass = recorder.by_keyframe[
                record.keyframe_idx]
    mass_summary = (
        metrics.position_mass_summary(
            history.health, truth, manifest.position_mass_metric)
        if recorder is not None else None)

    diagnostics = tuple(metrics.bearing_residual_diagnostics(
        catalog, measurements, history.health))
    diagnostic_payload = msgspec.json.encode({
        "schema": BEARING_DIAGNOSTIC_SCHEMA,
        "diagnostic_kind": "posterior_predictive_model_check",
        "evaluation_quality": False,
        "signed_residual_convention":
            "measured_world_bearing_minus_bearing_to_landmark_cw_deg",
        "null_dominated_policy":
            "null_share >= most_probable_named_landmark_share",
        "records": diagnostics,
    }, enc_hook=msgspec_enc_hook)

    manifest = msgspec.structs.replace(
        manifest, particle_history_sha256=history.particle_history_sha256)
    outputs = dict(extra_outputs or {})
    if metrics.POSITION_MASS_SUMMARY_NAME in outputs:
        raise ValueError(
            f"{metrics.POSITION_MASS_SUMMARY_NAME!r} is owned by the shared "
            "runner")
    if mass_summary is not None:
        outputs[metrics.POSITION_MASS_SUMMARY_NAME] = (
            msgspec.json.encode(mass_summary) + b"\n")
    if BEARING_DIAGNOSTIC_NAME in outputs:
        raise ValueError(
            f"{BEARING_DIAGNOSTIC_NAME!r} is owned by the shared runner")
    outputs[BEARING_DIAGNOSTIC_NAME] = diagnostic_payload
    reference = run_io.write_run(
        run_dir, manifest, truth, odometry, measurements, tables, history,
        dataset=dataset, version=version, upstreams=upstreams,
        artifact_config=artifact_config, generator=generator,
        arguments=arguments, extra_outputs=outputs)
    return ExecutionResult(
        manifest=manifest,
        history=history,
        artifact_ref=reference,
        bearing_diagnostics=diagnostics,
        position_mass_summary=mass_summary)
