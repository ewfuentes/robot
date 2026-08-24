"""Run the filter on a synthetic scenario and write a run directory.

Usage:
  bazel run //experimental/overhead_matching/swag/farfield/localization:run_localization -- \
    --scenario harbor_loop --output_dir /tmp/bol_demo --init global \
    --box_halfwidth_m 2500 --max_visible_range_m 10000

Synthetic scenarios exercise the filter under known truth; they are test and
demo instruments, never evaluations. Result-shaping knobs (prior geometry,
visibility radius) are required and echoed into the manifest.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield import provenance
from experimental.overhead_matching.swag.farfield.localization import (
    filter as pf,
    metrics,
    run_io,
    scenario,
    structs,
)

# Reporting threshold for the convergence printout only (not result-shaping).
CONVERGENCE_REPORT_M = 100.0


def build_filter_config(args, data) -> structs.FilterConfig:
    if args.init == "local":
        start = data.truth[0]
        init = structs.GaussianInit(
            mean_east_m=start.east_m + args.prior_offset_east_m,
            mean_north_m=start.north_m + args.prior_offset_north_m,
            sigma_m=args.prior_sigma_m)
        # A local Gaussian needs no jitter floor.
        roughening_m, roughening_deg = 0.0, 0.0
    else:
        half = args.box_halfwidth_m
        init = structs.UniformBoxInit(
            east_min_m=-half, east_max_m=half,
            north_min_m=-half, north_max_m=half)
        # A brute-force uniform prior can collapse faster than a bandwidth
        # proportional to the (still huge) spread can repair, so add a floor.
        roughening_m, roughening_deg = 15.0, 1.0
    return structs.FilterConfig(
        n_particles=args.n_particles,
        seed=args.seed,
        init=init,
        position_roughening_m=roughening_m,
        heading_roughening_deg=roughening_deg,
        checkpoint_every=args.checkpoint_every)


def _catalog_landmarks(data) -> list:
    """Manifest landmarks are what the FILTER was given, so the viewer draws
    the catalog the filter believed rather than ground truth."""
    lat, lon = data.frame.latlon_from_enu(data.catalog.east_m,
                                          data.catalog.north_m)
    by_id = {lm.landmark_id: lm for lm in data.config.landmarks}
    return [structs.LandmarkEntry(landmark_id=lm_id, lat_deg=float(la),
                                  lon_deg=float(lo),
                                  type_key=by_id[lm_id].type_key,
                                  position_sigma_m=float(sigma))
            for lm_id, la, lo, sigma in zip(
                data.catalog.landmark_ids, lat, lon,
                data.catalog.position_sigma_m)]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True,
                        choices=sorted(scenario.SCENARIO_BUILDERS.keys()))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--init", required=True, choices=["local", "global"])
    # Result-shaping: required (previous values quoted in help).
    parser.add_argument("--n_particles", type=int, required=True,
                        help="(previously 5000 local / 150000 global)")
    parser.add_argument("--max_visible_range_m", type=float, required=True,
                        help="catalog visibility radius (previously an "
                             "implicit 10000)")
    parser.add_argument("--prior_sigma_m", type=float, default=None,
                        help="--init local: prior sigma (previously 500)")
    parser.add_argument("--prior_offset_east_m", type=float, default=None,
                        help="--init local: deliberate prior offset "
                             "(previously +250)")
    parser.add_argument("--prior_offset_north_m", type=float, default=None,
                        help="--init local: (previously -250)")
    parser.add_argument("--box_halfwidth_m", type=float, default=None,
                        help="--init global: uniform box half-width "
                             "(previously 2500)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keyframe_period_s", type=float, required=True,
                        help="(previously 2.0)")
    parser.add_argument("--epoch_length", type=int, required=True,
                        help="(previously 5)")
    parser.add_argument("--bearing_sigma_deg", type=float, required=True,
                        help="(previously 1.0)")
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--kidnap_at", type=int, default=None,
                        help="teleport the vehicle at this keyframe")
    parser.add_argument("--kidnap_east_m", type=float, default=None)
    parser.add_argument("--kidnap_north_m", type=float, default=None)
    args = parser.parse_args()

    if args.init == "local":
        missing = [name for name in ("prior_sigma_m", "prior_offset_east_m",
                                     "prior_offset_north_m")
                   if getattr(args, name) is None]
        if missing:
            parser.error(f"--init local requires --{', --'.join(missing)}")
    else:
        if args.box_halfwidth_m is None:
            parser.error("--init global requires --box_halfwidth_m")
    if args.kidnap_at is not None and (args.kidnap_east_m is None
                                       or args.kidnap_north_m is None):
        parser.error("--kidnap_at requires --kidnap_east_m and "
                     "--kidnap_north_m (no default teleport)")

    scenario_config = scenario.get_scenario_config(
        args.scenario,
        keyframe_period_s=args.keyframe_period_s,
        epoch_length_keyframes=args.epoch_length,
        bearing_sigma_deg=args.bearing_sigma_deg,
        max_visible_range_m=args.max_visible_range_m)
    data = scenario.generate(scenario_config)
    if args.kidnap_at is not None:
        data = scenario.apply_kidnap(data, args.kidnap_at,
                                     args.kidnap_east_m, args.kidnap_north_m)
    filter_config = build_filter_config(args, data)

    print(f"Scenario '{scenario_config.name}': {data.n_keyframes} keyframes, "
          f"{len(data.measurements)} tracklet measurements, "
          f"{len(data.landmark_ids)} landmarks")
    print(f"Filter: {filter_config.n_particles} particles, "
          f"init={args.init}, seed={filter_config.seed}")

    history = pf.run_filter(filter_config, data.catalog, data.odometry,
                            data.measurements, data.tables)

    manifest = structs.RunManifest(
        schema_version=structs.SCHEMA_VERSION,
        dataset="synthetic",
        scenario_name=scenario_config.name,
        run_kind="synthetic",
        initialization_kind=args.init,
        bearings_consumed=True,
        proposal_enabled=filter_config.proposal.enabled,
        localization_inputs_manifest_sha256=None,
        anchor_lat_deg=scenario_config.anchor_lat_deg,
        anchor_lon_deg=scenario_config.anchor_lon_deg,
        n_keyframes=data.n_keyframes,
        filter_config=filter_config,
        landmarks=_catalog_landmarks(data),
        matcher_version=scenario.MATCHER_VERSION,
        max_visible_range_m=args.max_visible_range_m,
        export_dir=f"synthetic:{scenario_config.name}",
        git_commit=provenance.git_commit(),
        argv=list(sys.argv),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        particle_history_sha256=history.particle_history_sha256)
    run_io.write_run(args.output_dir, manifest, data.truth, data.odometry,
                     data.measurements, data.tables, history,
                     dataset="synthetic", version=args.output_dir.name,
                     artifact_config={
                         "run_kind": "synthetic",
                         "scenario": scenario_config.name,
                     },
                     generator=("//experimental/overhead_matching/swag/"
                                "farfield/localization:run_localization"),
                     arguments=tuple(sys.argv))

    errors = metrics.position_errors_m(history.health, data.truth)
    heading_errors = metrics.heading_errors_deg(history.health, data.truth)
    final_truth = data.truth[-1]
    nees = metrics.position_nees(history.final_belief, final_truth.east_m,
                                 final_truth.north_m)
    converged = np.nonzero(errors < CONVERGENCE_REPORT_M)[0]
    first_converged = int(converged[0]) if len(converged) else None
    print(f"Final position error: {errors[-1]:.1f} m; "
          f"final heading error: {heading_errors[-1]:.1f} deg")
    print(f"Final reported sigma: {history.health[-1].position_std_m:.1f} m; "
          f"NEES {nees:.1f} (2 dof, ideal ~2.0)")
    print(f"First keyframe with error < {CONVERGENCE_REPORT_M:.0f} m: "
          f"{first_converged}")
    for event in history.proposal_events:
        print(f"  proposal #{event.event_id} kf={event.keyframe_idx} "
              f"trigger={event.trigger} hypotheses={event.n_hypotheses} "
              f"injected={event.n_injected}")
    print(f"\nRun written to {args.output_dir}")


if __name__ == "__main__":
    main()
