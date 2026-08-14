"""Generate run directories for the M4 finding demos (viewer inputs).

One run per setting discussed in the design doc's Milestone 4 notes, so the
findings can be inspected in the §7 viewer rather than taken from prose:

  adversarial_init_injected  T-F5 adversarial matcher (one tracklet scores
                             the WRONG landmark at +clip). A tight uniform
                             prior legitimately fires the init proposal, so
                             half the mass arrives heading-pinned at the
                             liar's fix and wins the weight war — the §8.4
                             "init-injection" pathology, reproduced.
  adversarial_gated          Same world, informative Gaussian prior: the
                             init proposal is gated (§5.5), and geometry
                             beats the poisoned LLR while tracking.
  symmetric_both             Twin lighthouses, matcher cannot tell them
                             apart: exact C2 symmetry, so the honest
                             posterior holds two balanced modes forever.
  symmetric_identity         Same world, identity matcher: evidence exists
                             and the belief collapses to the true mode.
  kidnap_recovery            1900 m teleport at kf 120; the null-share
                             trigger (min_fraction 0.25, §8.4) fires and
                             the proposal recovers the fix.

Usage:
  bazel run //experimental/overhead_matching/swag/bearing_only_localization:m4_demo_runs -- \
    --output_base /tmp/m4_demos
Then render each with the viewer:
  bazel run //...:viewer -- --run_dir /tmp/m4_demos/<name>
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

_PERIOD_S = 5.0


def _catalog_landmarks(data: scenario.ScenarioData) -> list:
    lat, lon = data.frame.latlon_from_enu(data.catalog.east_m,
                                          data.catalog.north_m)
    by_id = {lm.landmark_id: lm for lm in data.config.landmarks}
    return [structs.LandmarkEntry(landmark_id=lm_id, lat_deg=float(la),
                                  lon_deg=float(lo),
                                  type_key=by_id[lm_id].type_key)
            for lm_id, la, lo in zip(data.catalog.landmark_ids, lat, lon)]


def _write(output_base: Path, name: str, data: scenario.ScenarioData,
           tables: dict, config: structs.FilterConfig) -> None:
    history = pf.run_filter(config, data.catalog, data.odometry,
                            data.measurements, tables)
    manifest = structs.RunManifest(
        schema_version=structs.SCHEMA_VERSION,
        scenario_name=name,
        anchor_lat_deg=data.config.anchor_lat_deg,
        anchor_lon_deg=data.config.anchor_lon_deg,
        n_keyframes=data.n_keyframes,
        filter_config=config,
        landmarks=_catalog_landmarks(data),
        matcher_version=scenario.MATCHER_VERSION,
        particle_history_sha256=history.particle_history_sha256)
    run_dir = output_base / name
    run_log.write_run(run_dir, manifest, data.truth, data.odometry,
                      data.measurements, tables, history)

    errors = pf.position_errors_m(history.health, data.truth)
    final = data.truth[-1]
    nees = pf.position_nees(history.final_belief, final.east_m, final.north_m)
    modes = history.health[-1].modes
    events = ", ".join(f"#{e.event_id}@kf{e.keyframe_idx}:{e.trigger}"
                       f"(inj {e.n_injected})"
                       for e in history.proposal_events) or "none"
    print(f"{name}: final err {errors[-1]:.0f} m, "
          f"sigma {history.health[-1].position_std_m:.0f} m, "
          f"NEES {nees:.1f}, modes {len(modes)} "
          f"(entropy {history.health[-1].mode_entropy_nats:.2f}); "
          f"proposals: {events}")


def _adversarial_world():
    data = scenario.generate(scenario.harbor_loop(keyframe_period_s=_PERIOD_S))
    victim, liar = data.landmark_ids[0], data.landmark_ids[1]
    tables = dict(data.tables)
    original = tables[f"trk_{victim}"]
    tables[f"trk_{victim}"] = structs.CompatibilityTable(
        tracklet_id=original.tracklet_id,
        matcher_version="adversarial",
        entries=[structs.CompatibilityEntry(liar, original.clip_hi),
                 structs.CompatibilityEntry(victim, original.clip_lo)],
        default_log_lr=original.clip_lo, clip_lo=original.clip_lo,
        clip_hi=original.clip_hi, status="fast")
    return data, tables


def _symmetric_world():
    cfg = scenario.symmetric_pair(keyframe_period_s=_PERIOD_S)
    data = scenario.generate(cfg)
    both = [structs.CompatibilityEntry(lm_id, cfg.identity_clip)
            for lm_id in data.landmark_ids]
    both_tables = {tid: structs.CompatibilityTable(
        tracklet_id=t.tracklet_id, matcher_version=t.matcher_version,
        entries=both, default_log_lr=t.default_log_lr,
        clip_lo=t.clip_lo, clip_hi=t.clip_hi, status=t.status)
        for tid, t in data.tables.items()}
    return data, both_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_base", type=Path, required=True)
    args = parser.parse_args()

    data, tables = _adversarial_world()
    start = data.truth[0]
    # A tight uniform box is still an informative prior in spirit, but it is
    # the shape that fires the init proposal — used here deliberately to
    # reproduce the pathology the §5.5 gating exists to prevent.
    _write(args.output_base, "adversarial_init_injected", data, tables,
           structs.FilterConfig(
               n_particles=20000, seed=5,
               init=structs.UniformBoxInit(
                   start.east_m - 700.0, start.east_m + 1300.0,
                   start.north_m - 1200.0, start.north_m + 800.0),
               checkpoint_every=5))
    _write(args.output_base, "adversarial_gated", data, tables,
           structs.FilterConfig(
               n_particles=20000, seed=5,
               init=structs.GaussianInit(start.east_m + 300.0,
                                         start.north_m - 200.0, 500.0),
               checkpoint_every=5))

    data, both_tables = _symmetric_world()
    symmetric_config = structs.FilterConfig(
        n_particles=40000, seed=1,
        init=structs.UniformBoxInit(-1500.0, 1500.0, -1200.0, 1200.0),
        checkpoint_every=5)
    _write(args.output_base, "symmetric_both", data, both_tables,
           symmetric_config)
    _write(args.output_base, "symmetric_identity", data, data.tables,
           symmetric_config)

    data = scenario.generate(scenario.harbor_loop(keyframe_period_s=_PERIOD_S))
    kidnapped = scenario.apply_kidnap(data, 120, 1500.0, -1200.0)
    _write(args.output_base, "kidnap_recovery", kidnapped, kidnapped.tables,
           structs.FilterConfig(
               n_particles=20000, seed=5,
               init=structs.GaussianInit(kidnapped.truth[0].east_m,
                                         kidnapped.truth[0].north_m, 300.0),
               checkpoint_every=5))


if __name__ == "__main__":
    main()
