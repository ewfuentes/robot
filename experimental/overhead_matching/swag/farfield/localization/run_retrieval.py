"""Run the filter with a retrieval observation source (CLD-3).

The retrieval counterpart of run_export: consumes one completed
``localization_inputs`` artifact (odometry, truth, anchor, catalog — the
bearings inside it are NOT consumed) plus one retrieval score-fields
directory, and publishes a run through the same shared runner as every other
localization run.

This driver stands outside the pipeline's build-config contract on purpose:
retrieval fields are produced by a baseline (CrossLocate-Depth) whose
database is not yet a pipeline stage. Its result-shaping settings are
explicit required flags, all echoed into the manifest. When the baseline
grows a pipeline stage, run_export's config-digest machinery is the model to
follow.

Classification: a uniform-prior run that consumes retrieval fields is an
evaluation IFF the calibration is frozen (--retrieval_calibration_frozen,
asserting temperature/epsilon were selected on declared validation regions)
and no ablation tags apply. Until then runs carry the
``retrieval_calibration_provisional`` tag and classify as diagnostics.

    bazel run //experimental/overhead_matching/swag/farfield/localization:run_retrieval -- \
        --input_dir <artifacts>/localization_inputs/<dataset>/<version> \
        --retrieval_dir <fields dir> \
        --run_dir <runs>/<experiment>/<run> \
        --init uniform --margin_m 1000 --n_particles 50000 \
        --retrieval_temperature 1.0 --retrieval_epsilon 0.05 \
        --position_roughening_m 25 --heading_roughening_deg 1
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield import provenance
from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    metrics,
    retrieval,
    runner,
    structs,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="completed localization_inputs artifact")
    parser.add_argument("--retrieval_dir", type=Path, required=True,
                        help="retrieval_meta.json + retrieval_fields.npz")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--init", required=True,
                        choices=["uniform", "truth_position"])
    parser.add_argument("--prior_region", default="retrieval_support",
                        choices=["retrieval_support", "catalog"],
                        help="what the uniform prior spans. The retrieval "
                             "support IS the declared candidate region "
                             "(plan section 5.1), so it is the default; "
                             "'catalog' spans the whole landmark extent "
                             "for parity with bearing runs")
    parser.add_argument("--margin_m", type=float, default=None,
                        help="uniform prior margin past the chosen region "
                             "(required with --init uniform)")
    parser.add_argument("--prior_sigma_m", type=float, default=None,
                        help="required with --init truth_position "
                             "(diagnostic control)")
    parser.add_argument("--n_particles", type=int, required=True)
    parser.add_argument("--retrieval_temperature", type=float, required=True,
                        help="softmax temperature of the score->likelihood "
                             "calibration (§5.5; a modeling choice)")
    parser.add_argument("--retrieval_epsilon", type=float, required=True,
                        help="uniform outlier mass of the calibration")
    parser.add_argument("--retrieval_calibration_frozen",
                        action="store_true",
                        help="assert tau/epsilon were frozen on declared "
                             "validation regions; without this the run is "
                             "tagged retrieval_calibration_provisional and "
                             "classifies as a diagnostic")
    parser.add_argument("--position_roughening_m", type=float, required=True)
    parser.add_argument("--heading_roughening_deg", type=float,
                        required=True)
    parser.add_argument("--resample_survival_floor", type=int, default=64,
                        help="minimum offspring per mode/proposal group at "
                             "resample (the adopted default from the "
                             "2026-08-27 survival-floor study; 0 restores "
                             "the historical behavior)")
    parser.add_argument("--resample_survival_min_mass", type=float,
                        default=1e-9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--backend", default="numpy",
                        choices=["numpy", "torch"])
    parser.add_argument("--position_mass_radii_m", type=float, nargs="+",
                        default=[100.0, 500.0])
    parser.add_argument("--ablation_tags", nargs="*", default=[])
    args = parser.parse_args()

    data = export_ingest.load(args.input_dir)
    print(export_ingest.describe(data))

    fields = retrieval.load_fields(args.retrieval_dir, data.frame)
    measurements = retrieval.measurements_from_fields(fields)
    print(retrieval.describe(fields))
    if fields.meta.dataset != data.meta.dataset:
        parser.error(f"retrieval fields are for {fields.meta.dataset!r}, "
                     f"inputs are {data.meta.dataset!r}")

    if args.init == "uniform":
        if args.margin_m is None:
            parser.error("--init uniform requires --margin_m")
        if args.prior_region == "retrieval_support":
            init = structs.UniformBoxInit(
                east_min_m=float(fields.east_m.min()) - args.margin_m,
                east_max_m=float(fields.east_m.max()) + args.margin_m,
                north_min_m=float(fields.north_m.min()) - args.margin_m,
                north_max_m=float(fields.north_m.max()) + args.margin_m)
        else:
            init = export_ingest.region_box(data, args.margin_m)
        print(f"prior       : uniform over "
              f"{(init.east_max_m - init.east_min_m) / 1000:.1f} x "
              f"{(init.north_max_m - init.north_min_m) / 1000:.1f} km, "
              f"uniform heading")
    else:
        if args.prior_sigma_m is None or not data.truth:
            parser.error("--init truth_position requires --prior_sigma_m "
                         "and diagnostic truth in the export")
        start = data.truth[0]
        init = structs.GaussianInit(start.east_m, start.north_m,
                                    args.prior_sigma_m)
        print(f"prior       : Gaussian at first truth pose, sigma "
              f"{args.prior_sigma_m:.0f} m (DIAGNOSTIC CONTROL)")

    forward_camera_cw_deg = float(
        data.meta.nominal_forward["bearing_camera_cw_deg"])
    print(f"heading frame: field bins are camera-forward; particles are "
          f"nominal-forward (camera {forward_camera_cw_deg:+.1f} deg CW)")
    retrieval_config = structs.RetrievalConfig(
        temperature=args.retrieval_temperature,
        outlier_epsilon=args.retrieval_epsilon,
        calibration_frozen=args.retrieval_calibration_frozen,
        forward_camera_cw_deg=forward_camera_cw_deg)
    filter_config = structs.FilterConfig(
        n_particles=args.n_particles, seed=args.seed, init=init,
        position_roughening_m=args.position_roughening_m,
        heading_roughening_deg=args.heading_roughening_deg,
        resample_survival_floor=args.resample_survival_floor,
        resample_survival_min_mass=args.resample_survival_min_mass,
        checkpoint_every=args.checkpoint_every,
        measurement_backend=args.backend,
        proposal=structs.ProposalConfig(enabled=False),
        retrieval=retrieval_config)

    tags = set(args.ablation_tags)
    if not args.retrieval_calibration_frozen:
        tags.add("retrieval_calibration_provisional")
        print("retrieval calibration: PROVISIONAL — this run classifies as "
              "a diagnostic, not an evaluation")
    if args.init == "truth_position":
        tags.add("truth_position_initialization")
    run_kind = ("evaluation" if args.init == "uniform" and not tags
                else "diagnostic_control")

    manifest = structs.RunManifest(
        schema_version=structs.SCHEMA_VERSION,
        dataset=data.meta.dataset,
        scenario_name=data.meta.scenario_name,
        run_kind=run_kind,
        initialization_kind=args.init,
        bearings_consumed=False,
        proposal_enabled=False,
        localization_inputs_manifest_sha256=data.artifact_ref.manifest_digest,
        anchor_lat_deg=data.meta.anchor_lat_deg,
        anchor_lon_deg=data.meta.anchor_lon_deg,
        n_keyframes=data.n_keyframes,
        filter_config=filter_config,
        landmarks=data.landmarks,
        matcher_version=f"retrieval:{fields.meta.scorer}",
        max_visible_range_m=data.meta.max_visible_range_m,
        export_dir=str(args.input_dir),
        git_commit=provenance.git_commit(),
        argv=list(sys.argv),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ablation_tags=sorted(tags),
        truth_position_artifact=(data.artifact_ref.to_dict()
                                 if data.truth else None),
        truth_position_schema=(runner.TRUTH_POSITION_SCHEMA
                               if data.truth else None),
        position_mass_metric=(
            metrics.position_mass_metric_config(args.position_mass_radii_m)
            if data.truth else None),
        retrieval_consumed=True,
        retrieval_dir=str(args.retrieval_dir))

    # Bearings deliberately unconsumed: this driver measures the retrieval
    # source alone, so the run artifact carries empty bearing tier-1 files.
    result = runner.execute_localization(
        args.run_dir, manifest, catalog=data.catalog, truth=data.truth,
        odometry=data.odometry, measurements=[], tables={},
        dataset=data.meta.dataset, version=args.run_dir.name,
        upstreams=(data.artifact_ref,),
        artifact_config={
            "observation_source": "retrieval",
            "retrieval": {
                "dir": str(args.retrieval_dir),
                "scorer": fields.meta.scorer,
                "db_dir": fields.meta.db_dir,
                "db_manifest_sha256": fields.meta.db_manifest_sha256,
                "node_spacing_m": fields.meta.node_spacing_m,
                "temperature": args.retrieval_temperature,
                "outlier_epsilon": args.retrieval_epsilon,
                "calibration_frozen": args.retrieval_calibration_frozen,
            },
        },
        generator="//experimental/overhead_matching/swag/farfield/"
                  "localization:run_retrieval",
        arguments=tuple(sys.argv),
        retrieval_fields=fields,
        retrieval_measurements=measurements,
        retrieval_source_dir=args.retrieval_dir)

    history = result.history
    if result.position_mass_summary is not None:
        print("\n" + metrics.describe_position_mass_summary(
            result.position_mass_summary, result.manifest.run_kind))
    print("\n--- belief ---")
    final = history.health[-1]
    print(f"modes at end: {len(final.modes)} "
          f"(entropy {final.mode_entropy_nats:.2f}); "
          f"reported sigma {final.position_std_m:.0f} m")
    if data.truth:
        errors = metrics.map_position_errors_m(history.health, data.truth)
        print(f"MAP position error vs GPS truth: final {errors[-1]:.0f} m, "
              f"median over last 50 kf {np.median(errors[-50:]):.0f} m")
    print(f"quantization floor: {fields.meta.node_spacing_m:.0f} m lattice, "
          f"{360.0 / fields.meta.n_heading_bins:.0f} deg heading bins")
    print(f"\nRun written to {args.run_dir}")


if __name__ == "__main__":
    main()
