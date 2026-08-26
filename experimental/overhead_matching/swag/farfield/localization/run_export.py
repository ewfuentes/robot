"""Run the filter on a real localization export and write a run directory.

The CLI accepts one completed ``localization_inputs`` artifact, one immutable
``build_config.json``, and the final ``run_dir``.  Every filter setting comes
from that build config; there are no per-run scientific overrides.

Reads the export's Tier-1 inputs, runs the filter, writes a run directory that
the plots and viewer consume unchanged, and reports the primary evaluation
metric: time-normalized posterior probability mass within 500 m of the true
position.  The same score is reported at 100 m by default.  Truth is read by an
evaluation-only observer after each posterior update and never changes the
uniform prior or any filter decision.

Posterior-predictive bearing diagnostics and association posteriors remain
model checks, not correctness labels. Final position error is printed as a
secondary diagnostic — a point estimate at one instant cannot score an entire,
possibly multimodal posterior trajectory.

Only uniform-prior runs that actually consume bearings are evaluations.
Truth initialization and odometry-only executions are diagnostic controls.

The run directory records what the filter actually consumed: an odometry-only
control writes empty measurement/table files, matching the run that happened.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield import (
    artifact,
    build_config,
    provenance,
)
from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    metrics,
    run_identity,
    runner,
    structs,
)


def _print_bearing_diagnostics(diagnostics) -> None:
    print("\n--- posterior-predictive bearing model diagnostics ---")
    print("(signed residuals; selected associations are not correctness labels)")
    for mode_specific in (False, True):
        scope = "mode-specific" if mode_specific else "whole-belief"
        scoped = [
            record for record in diagnostics
            if (record.mode_id is not None) == mode_specific
        ]
        for null_dominated in (False, True):
            label = "null-dominated" if null_dominated else "landmark-dominated"
            values = np.array([
                record.signed_residual_deg for record in scoped
                if record.null_dominated == null_dominated
                and record.signed_residual_deg is not None
            ], dtype=np.float64)
            count = sum(record.null_dominated == null_dominated
                        for record in scoped)
            if values.size:
                print(f"{scope}, {label}: n={count}, signed median "
                      f"{np.median(values):+.2f} deg, |residual| median "
                      f"{np.median(np.abs(values)):.2f} deg, p90 "
                      f"{np.percentile(np.abs(values), 90):.1f} deg")
            else:
                print(f"{scope}, {label}: n={count}, no named-landmark residual")


def _flatten(value: dict, prefix: str = "") -> dict:
    result = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, dict):
            result.update(_flatten(child, path))
        else:
            result[path] = child
    return result


def orchestration_contract(document: dict) -> dict:
    """Recompute the pipeline's exact localize-stage config selection."""
    localization = document.get("config", {}).get("localization")
    if not isinstance(localization, dict):
        raise ValueError("build config does not record localization")
    selected = {
        f"localization.{key}": value
        for key, value in _flatten(localization).items()
    }
    return {
        "schema": "farfield_pipeline_stage/v1",
        "stage": "localize",
        "config_digest": artifact.sha256_json(selected),
    }


def localization_inputs_contract(document: dict) -> dict:
    """Recompute the exact recipe that shaped the consumed input artifact.

    Localization is a downstream experiment over an immutable export.  A new
    filter backend or run name must not force that export to be republished
    when its own version and complete stage-scoped recipe are unchanged.
    """
    config = document.get("config")
    if not isinstance(config, dict):
        raise ValueError("build config has no config object")
    selected = {}
    for prefix in ("localization_inputs", "gps_course"):
        value = config.get(prefix)
        if not isinstance(value, dict):
            raise ValueError(f"build config does not record {prefix!r}")
        selected.update({
            f"{prefix}.{key}": child
            for key, child in _flatten(value).items()
        })
    selected["artifacts.localization_inputs_version"] = build_config.value(
        document, "artifacts.localization_inputs_version")
    return {
        "schema": "farfield_pipeline_stage/v1",
        "stage": "localization_inputs",
        "config_digest": artifact.sha256_json(selected),
    }


def _load_config(path: Path, data: export_ingest.ExportData,
                 expected_digest: str) -> tuple[dict, dict]:
    path = Path(path)
    if (path.name != build_config.BUILD_CONFIG_NAME or not path.is_file()
            or path.is_symlink()):
        raise ValueError(
            f"--build_config must name a regular, non-symlink "
            f"{build_config.BUILD_CONFIG_NAME}")
    document = build_config.load(path.parent)
    if document["dataset"] != data.meta.dataset:
        raise ValueError(
            "build config dataset disagrees with localization inputs")
    expected_version = build_config.value(
        document, "artifacts.localization_inputs_version")
    if data.artifact_ref.version != expected_version:
        raise ValueError(
            "localization input version disagrees with build config")
    if data.manifest is None:
        raise ValueError("localization inputs have no typed manifest")
    if (data.manifest.config.get("orchestration")
            != localization_inputs_contract(document)):
        raise ValueError(
            "localization inputs were produced by a different stage-scoped "
            "recipe")
    root = document.get("inputs", {}).get("farfield_root")
    if isinstance(root, str) and root:
        expected_path = (Path(root) / "artifacts" / "localization_inputs"
                         / document["dataset"] / expected_version)
        if (Path(data.artifact_ref.path) != expected_path
                or data.artifact_ref.path != str(expected_path.resolve())):
            raise ValueError(
                "localization inputs are not the exact configured artifact "
                "lane")
    orchestration = orchestration_contract(document)
    if expected_digest != orchestration["config_digest"]:
        raise ValueError(
            "--orchestration_config_digest does not match the immutable "
            "localization recipe")
    return document, orchestration


def _filter_config(localization: dict,
                   data: export_ingest.ExportData) -> structs.FilterConfig:
    init_kind = localization["init"]
    if init_kind == "uniform":
        init = export_ingest.region_box(data, localization["margin_m"])
    elif init_kind == "truth_position":
        if not data.truth:
            raise ValueError(
                "truth_position init requires diagnostic position truth")
        sigma = localization["prior_sigma_m"]
        if sigma is None or sigma <= 0.0:
            raise ValueError(
                "truth_position init requires a positive prior_sigma_m")
        start = data.truth[0]
        init = structs.GaussianInit(
            start.east_m, start.north_m, sigma)
    else:
        raise ValueError(
            "localization init must be 'uniform' or 'truth_position'")
    proposal = structs.ProposalConfig(**localization["proposal"])
    modes = structs.ModeConfig(**localization["modes"])
    return structs.FilterConfig(
        n_particles=localization["n_particles"],
        seed=localization["seed"],
        init=init,
        pi0=localization["pi0"],
        ess_resample_frac=localization["ess_resample_frac"],
        heading_random_walk_deg=localization["heading_random_walk_deg"],
        resample_regularization=localization["resample_regularization"],
        position_roughening_m=localization["position_roughening_m"],
        heading_roughening_deg=localization["heading_roughening_deg"],
        map_cell_size_m=localization["map_cell_size_m"],
        checkpoint_every=localization["checkpoint_every"],
        measurement_backend=localization["measurement_backend"],
        association_persistence=localization["association_persistence"],
        association_renewal_rate=localization[
            "association_renewal_rate"],
        association_outlier_rate=localization["association_outlier_rate"],
        matcher_recall=localization["matcher_recall"],
        min_reported_responsibility=localization[
            "min_reported_responsibility"],
        proposal=proposal,
        modes=modes)


def _classification(localization: dict) -> tuple[str, list[str]]:
    tags = set(localization.get("ablation_tags", []))
    if not localization["bearings_enabled"]:
        tags.add("no_bearings")
    if localization["init"] == "truth_position":
        tags.add("truth_position_initialization")
    if not localization["proposal"]["enabled"]:
        tags.add("proposal_disabled")
    run_kind = ("evaluation"
                if localization["init"] == "uniform"
                and localization["bearings_enabled"] and not tags
                else "diagnostic_control")
    return run_kind, sorted(tags)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--build_config", type=Path, required=True)
    parser.add_argument("--orchestration_config_digest", required=True)
    args = parser.parse_args()
    try:
        data = export_ingest.load(args.input_dir)
        document, orchestration = _load_config(
            args.build_config, data, args.orchestration_config_digest)
        localization = document["config"]["localization"]
        run_version = run_identity.from_build_document(document)
        if args.run_dir.name != run_version:
            raise ValueError(
                "--run_dir basename disagrees with the immutable localization "
                f"run identity {run_version!r}")
        filter_config = _filter_config(localization, data)
    except (artifact.ArtifactError, OSError, ValueError) as error:
        parser.error(str(error))
    print(export_ingest.describe(data))
    if localization["init"] == "uniform":
        init = filter_config.init
        print(f"prior       : uniform over "
              f"{(init.east_max_m - init.east_min_m) / 1000:.1f} x "
              f"{(init.north_max_m - init.north_min_m) / 1000:.1f} km, "
              f"uniform heading")
    else:
        print(f"prior       : Gaussian at the first truth pose, sigma "
              f"{localization['prior_sigma_m']:.0f} m "
              f"(DIAGNOSTIC CONTROL, not an "
              f"evaluation)")
    bearings_consumed = localization["bearings_enabled"]
    measurements = data.measurements if bearings_consumed else []
    tables = data.tables if bearings_consumed else {}
    run_kind, ablation_tags = _classification(localization)
    metric_config = (
        metrics.position_mass_metric_config(
            localization["position_mass_radii_m"])
        if data.truth else None)
    truth_source = data.artifact_ref.to_dict() if data.truth else None

    manifest = structs.RunManifest(
        schema_version=structs.SCHEMA_VERSION,
        dataset=data.meta.dataset,
        scenario_name=data.meta.scenario_name,
        run_kind=run_kind,
        initialization_kind=localization["init"],
        bearings_consumed=bearings_consumed,
        proposal_enabled=filter_config.proposal.enabled,
        localization_inputs_manifest_sha256=(
            data.artifact_ref.manifest_digest),
        anchor_lat_deg=data.meta.anchor_lat_deg,
        anchor_lon_deg=data.meta.anchor_lon_deg,
        n_keyframes=data.n_keyframes,
        filter_config=filter_config,
        landmarks=data.landmarks,
        matcher_version=data.meta.matcher_version,
        max_visible_range_m=data.meta.max_visible_range_m,
        export_dir=str(args.input_dir),
        git_commit=provenance.git_commit(),
        argv=list(sys.argv),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ablation_tags=ablation_tags,
        truth_position_artifact=truth_source,
        truth_position_schema=(
            runner.TRUTH_POSITION_SCHEMA if data.truth else None),
        position_mass_metric=metric_config)
    # The artifact records exactly what the filter consumed. An odometry-only
    # control therefore contains no measurements or compatibility tables.
    result = runner.execute_localization(
        args.run_dir, manifest, catalog=data.catalog, truth=data.truth,
        odometry=data.odometry, measurements=measurements, tables=tables,
        dataset=data.meta.dataset, version=run_version,
        upstreams=(data.artifact_ref,),
        artifact_config={
            "orchestration": orchestration,
            "build_identity": document["build_identity"],
            "object_tracks_version": document["config"]["artifacts"][
                "object_tracks_version"],
            "localization": localization,
            "run_identity": run_version,
            "localization_inputs_manifest_sha256": (
                data.artifact_ref.manifest_digest),
            "localization_inputs_build_identity": (
                data.manifest.config.get("build_identity")),
            "build_config_sha256": artifact.sha256_file(args.build_config),
        },
        generator="//experimental/overhead_matching/swag/farfield/"
                  "localization:run_export",
        arguments=tuple(sys.argv))
    history = result.history
    if result.position_mass_summary is not None:
        print("\n" + metrics.describe_position_mass_summary(
            result.position_mass_summary, result.manifest.run_kind))
    _print_bearing_diagnostics(result.bearing_diagnostics)
    print("\n--- association posteriors ---")
    nulls = [a.null_share for r in history.health for a in r.associations
             if a.mode_id is None]
    if nulls:
        print(f"null share: median {np.median(nulls):.2f}  "
              f"mean {np.mean(nulls):.2f}  "
              f">0.5 on {np.mean(np.array(nulls) > 0.5) * 100:.0f}% of "
              f"measurements")
    confident = [a for r in history.health for a in r.associations
                 if a.mode_id is None and a.responsibilities
                 and max(a.responsibilities.values()) > 0.8]
    print(f"measurements with a >80% single-landmark claim: "
          f"{len(confident)} / {len(nulls)}")

    print("\n--- secondary belief diagnostics ---")
    final = history.health[-1]
    print(f"modes at end: {len(final.modes)} "
          f"(entropy {final.mode_entropy_nats:.2f}); "
          f"reported sigma {final.position_std_m:.0f} m")
    for event in history.proposal_events:
        print(f"  proposal #{event.event_id} kf={event.keyframe_idx} "
              f"trigger={event.trigger} hypotheses={event.n_hypotheses} "
              f"injected={event.n_injected} "
              f"skipped_combinations={event.n_combinations_skipped}")
    if data.truth:
        errors = metrics.map_position_errors_m(history.health, data.truth)
        print(f"MAP position error vs GPS truth: final {errors[-1]:.0f} m, "
              f"median over last 50 kf {np.median(errors[-50:]):.0f} m")
        print("  (GPS odometry + GPS-selected candidates: read this as a "
              "sanity check, not as evidence)")
    print(f"\nRun written to {args.run_dir}")


if __name__ == "__main__":
    main()
