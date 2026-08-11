"""Run the bearing-only localization filter on a synthetic scenario and
write a self-describing run directory.

Usage:
  bazel run //experimental/overhead_matching/swag/bearing_only_localization:run_localization -- \
    --scenario harbor_loop --output_dir /tmp/bol_demo --init local
"""

import argparse
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    filter as pf,
    run_log,
    scenario,
    structs,
)

LOCAL_DEFAULT_PARTICLES = 5000
GLOBAL_DEFAULT_PARTICLES = 150000


def build_filter_config(args, data: scenario.ScenarioData
                        ) -> structs.FilterConfig:
    if args.init == "local":
        start = data.truth[0]
        init = structs.GaussianInit(
            mean_east_m=start.east_m + 250.0,
            mean_north_m=start.north_m - 250.0,
            sigma_m=500.0)
        n_particles = args.n_particles or LOCAL_DEFAULT_PARTICLES
        roughening_m, roughening_deg = 0.0, 0.0
    else:
        init = structs.UniformBoxInit(
            east_min_m=-2500.0, east_max_m=2500.0,
            north_min_m=-2500.0, north_max_m=2500.0)
        n_particles = args.n_particles or GLOBAL_DEFAULT_PARTICLES
        # A brute-force uniform prior can collapse faster than a bandwidth
        # proportional to the (still huge) spread can repair, so add a floor.
        roughening_m, roughening_deg = 15.0, 1.0
    return structs.FilterConfig(
        n_particles=n_particles,
        seed=args.seed,
        init=init,
        position_roughening_m=roughening_m,
        heading_roughening_deg=roughening_deg,
        checkpoint_every=args.checkpoint_every)


def _catalog_landmarks(data: scenario.ScenarioData) -> list:
    """Manifest landmarks are what the FILTER was given, so the viewer draws
    the catalog the filter believed rather than ground truth."""
    lat, lon = data.frame.latlon_from_enu(data.catalog.east_m,
                                          data.catalog.north_m)
    by_id = {lm.landmark_id: lm for lm in data.config.landmarks}
    return [structs.LandmarkEntry(landmark_id=lm_id, lat_deg=float(la),
                                  lon_deg=float(lo),
                                  type_key=by_id[lm_id].type_key)
            for lm_id, la, lo in zip(data.catalog.landmark_ids, lat, lon)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="harbor_loop",
                        choices=sorted(scenario.SCENARIO_BUILDERS.keys()))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--init", default="local",
                        choices=["local", "global"])
    parser.add_argument("--n_particles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keyframe_period_s", type=float, default=2.0)
    parser.add_argument("--epoch_length", type=int, default=5)
    parser.add_argument("--bearing_sigma_deg", type=float, default=1.0)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    args = parser.parse_args()

    scenario_config = scenario.get_scenario_config(
        args.scenario,
        keyframe_period_s=args.keyframe_period_s,
        epoch_length_keyframes=args.epoch_length,
        bearing_sigma_deg=args.bearing_sigma_deg)
    data = scenario.generate(scenario_config)
    filter_config = build_filter_config(args, data)

    print(f"Scenario '{scenario_config.name}': {data.n_keyframes} keyframes, "
          f"{len(data.measurements)} tracklet measurements, "
          f"{len(data.landmark_ids)} landmarks")
    print(f"Filter: {filter_config.n_particles} particles, init={args.init}, "
          f"seed={filter_config.seed}")

    history = pf.run_filter(filter_config, data.catalog, data.odometry,
                            data.measurements, data.tables)

    manifest = structs.RunManifest(
        schema_version=structs.SCHEMA_VERSION,
        scenario_name=scenario_config.name,
        anchor_lat_deg=scenario_config.anchor_lat_deg,
        anchor_lon_deg=scenario_config.anchor_lon_deg,
        n_keyframes=data.n_keyframes,
        filter_config=filter_config,
        landmarks=_catalog_landmarks(data),
        matcher_version=scenario.MATCHER_VERSION,
        particle_history_sha256=history.particle_history_sha256)
    run_log.write_run(args.output_dir, manifest, data.truth, data.odometry,
                      data.measurements, data.tables, history)

    errors = pf.position_errors_m(history.health, data.truth)
    heading_errors = pf.heading_errors_deg(history.health, data.truth)
    final_truth = data.truth[-1]
    nees = pf.position_nees(history.final_belief, final_truth.east_m,
                            final_truth.north_m)
    converged = np.nonzero(errors < 100.0)[0]
    first_converged = int(converged[0]) if len(converged) else None
    print(f"Final position error: {errors[-1]:.1f} m; "
          f"final heading error: {heading_errors[-1]:.1f} deg")
    print(f"Final reported sigma: {history.health[-1].position_std_m:.1f} m; "
          f"NEES {nees:.1f} (2 dof, ideal ~2.0)")
    print(f"First keyframe with error < 100 m: {first_converged}")
    print(f"Particle history sha256: {history.particle_history_sha256}")
    print(f"Run written to {args.output_dir}")


if __name__ == "__main__":
    main()
