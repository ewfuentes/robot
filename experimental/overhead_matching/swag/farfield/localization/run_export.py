"""Run the filter on a real localization export and write a run directory.

The CLI accepts one completed ``localization_inputs`` artifact, one immutable
``build_config.json``, and the final ``run_dir``.  Every filter setting comes
from that build config; there are no per-run scientific overrides.

Reads the export's Tier-1 inputs, runs the filter, writes a run directory
that the plots and viewer consume unchanged, and reports the two things a
GPS-supervised export can honestly support: **bearing residuals** against the
filter's own pose, and **association posteriors**. Final position error is
printed but is not the figure of merit here — when odometry is GPS and the
candidates were selected using GPS, dead reckoning alone nearly solves the
leg, so a small position error demonstrates very little.

Only uniform-prior runs that actually consume bearings are evaluations.
Truth initialization and odometry-only executions are diagnostic controls.

The run directory records what the filter actually consumed: an odometry-only
control writes empty measurement/table files, matching the run that happened.
"""

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield import (
    artifact,
    build_config,
    geometry as geo,
    provenance,
)
from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    filter as pf,
    metrics,
    run_io,
    structs,
)


def bearing_residuals(data: export_ingest.ExportData, health: list):
    """Angle between each measured bearing and the direction to its
    best-associated landmark, under the filter's own pose estimate.

    This is the metric that survives GPS supervision: it asks whether the
    bearings, the mount offset, the catalog and the pose are mutually
    consistent, which no amount of good odometry can fake.
    """
    residuals, records = [], []
    for record in health:
        for assoc in record.associations:
            if assoc.mode_id is not None or not assoc.responsibilities:
                continue  # whole-belief posterior only
            landmark_id = max(assoc.responsibilities,
                              key=assoc.responsibilities.get)
            share = assoc.responsibilities[landmark_id]
            index = data.catalog.index_of(landmark_id)
            meas = next(m for m in data.measurements
                        if m.tracklet_id == assoc.tracklet_id
                        and m.anchor_keyframe_idx == record.keyframe_idx)
            predicted = geo.compass_bearing_rad(
                data.catalog.east_m[index] - record.mean_east_m,
                data.catalog.north_m[index] - record.mean_north_m)
            residual = abs(math.degrees(float(geo.wrap_rad(
                predicted - math.radians(record.mean_heading_deg)
                - math.radians(meas.bearing_forward_cw_deg)))))
            residuals.append(residual)
            records.append((record.keyframe_idx, assoc.tracklet_id,
                            landmark_id, share, residual))
    return np.array(residuals), records


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
    if data.manifest is None or data.manifest.config.get(
            "build_identity") != document["build_identity"]:
        raise ValueError(
            "localization inputs belong to a different immutable build")
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
    else:
        if not data.truth:
            raise ValueError(
                "truth initialization requires diagnostic GPS-course truth")
        sigma = localization["prior_sigma_m"]
        if sigma is None or sigma <= 0.0:
            raise ValueError(
                "truth initialization requires a positive prior_sigma_m")
        start = data.truth[0]
        init = structs.GaussianInit(
            start.east_m, start.north_m, sigma)
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
        if args.run_dir.name != localization["run_name"]:
            raise ValueError(
                "--run_dir basename disagrees with localization.run_name")
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
    history = pf.run_filter(filter_config, data.catalog, data.odometry,
                            measurements, tables)

    manifest = structs.RunManifest(
        schema_version=structs.SCHEMA_VERSION,
        dataset=data.meta.dataset,
        scenario_name=data.meta.scenario_name,
        run_kind=("evaluation" if localization["init"] == "uniform"
                  and bearings_consumed else "diagnostic_control"),
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
        matcher_version=(f"{data.meta.matcher_version} (bearings withheld)"
                         if not bearings_consumed
                         else data.meta.matcher_version),
        max_visible_range_m=data.meta.max_visible_range_m,
        export_dir=str(args.input_dir),
        git_commit=provenance.git_commit(),
        argv=list(sys.argv),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        particle_history_sha256=history.particle_history_sha256)
    # The artifact records exactly what the filter consumed.  In particular,
    # an odometry control contains no measurements or compatibility tables.
    run_io.write_run(
        args.run_dir, manifest, data.truth, data.odometry, measurements,
        tables, history, dataset=data.meta.dataset,
        version=localization["run_name"], upstreams=(data.artifact_ref,),
        artifact_config={
            "orchestration": orchestration,
            "build_identity": document["build_identity"],
            "localization": localization,
            "run_kind": manifest.run_kind,
            "localization_inputs_manifest_sha256": (
                data.artifact_ref.manifest_digest),
            "build_config_sha256": artifact.sha256_file(args.build_config),
        },
        generator="//experimental/overhead_matching/swag/farfield/"
                  "localization:run_export",
        arguments=tuple(sys.argv))

    print("\n--- bearing residuals (filter pose vs. best-associated "
          "landmark) ---")
    residuals, records = bearing_residuals(data, history.health)
    if residuals.size:
        print(f"n={residuals.size}  median {np.median(residuals):.2f} deg  "
              f"p90 {np.percentile(residuals, 90):.1f} deg  "
              f"<5 deg {np.mean(residuals < 5) * 100:.0f}%  "
              f"<15 deg {np.mean(residuals < 15) * 100:.0f}%  "
              f">60 deg {np.mean(residuals > 60) * 100:.0f}%")
        worst = sorted(records, key=lambda r: -r[4])[:5]
        print("worst:", ", ".join(
            f"kf{kf} {trk}->{lm} {res:.0f}deg(p={share:.2f})"
            for kf, trk, lm, share, res in worst))

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

    print("\n--- belief ---")
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
