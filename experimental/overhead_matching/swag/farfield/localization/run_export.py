"""Run the filter on a real localization export and write a run directory.

Usage:
  bazel run //experimental/overhead_matching/swag/farfield/localization:run_export -- \
    --export_dir <run>/localization_export --output_dir <experiment>/<run> \
    --init uniform --n_particles 50000 --margin_m 1000 \
    --max_visible_range_m 15000 --position_roughening_m 25 \
    --heading_roughening_deg 1

Reads the export's Tier-1 inputs, runs the filter, writes a run directory
that the plots and viewer consume unchanged, and reports the two things a
GPS-supervised export can honestly support: **bearing residuals** against the
filter's own pose, and **association posteriors**. Final position error is
printed but is not the figure of merit here — when odometry is GPS and the
candidates were selected using GPS, dead reckoning alone nearly solves the
leg, so a small position error demonstrates very little.

ONLY uniform-prior runs are evaluations. --init truth is a diagnostic
instrument (basin-of-attraction control), never a result.

The run directory records what the filter actually consumed: an odometry-only
control (--no_bearings) writes EMPTY measurement/table files, matching the
run that happened. (The old driver wrote the full unconsumed inputs, so every
control run on disk described a run that never happened.)
"""

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield import (
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
                - math.radians(meas.bearing_body_deg)))))
            residuals.append(residual)
            records.append((record.keyframe_idx, assoc.tracklet_id,
                            landmark_id, share, residual))
    return np.array(residuals), records


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="run directory, inside a runs/<experiment>/")
    parser.add_argument("--init", required=True,
                        choices=["uniform", "truth"],
                        help="uniform: flat prior over the catalog extent — "
                             "the only init that counts as an evaluation. "
                             "truth: tight prior at the first truth pose, a "
                             "diagnostic control only.")
    # Result-shaping parameters: required, echoed into the manifest
    # (REORG.md rule 2; previous defaults quoted).
    parser.add_argument("--n_particles", type=int, required=True,
                        help="(previously 50000)")
    parser.add_argument("--margin_m", type=float, required=True,
                        help="how far past the catalog the uniform prior "
                             "extends (previously 1000)")
    parser.add_argument("--max_visible_range_m", type=float, required=True,
                        help="catalog visibility radius; recorded in the "
                             "manifest — the replay contract depends on the "
                             "recorded value being the used value "
                             "(previously an inconsistent 15000/10000 pair)")
    parser.add_argument("--position_roughening_m", type=float, required=True,
                        help="(previously hardcoded 25.0)")
    parser.add_argument("--heading_roughening_deg", type=float,
                        required=True, help="(previously hardcoded 1.0)")
    parser.add_argument("--prior_sigma_m", type=float, default=None,
                        help="with --init truth, the sigma of the Gaussian "
                             "prior at the first truth pose. Sweeping it "
                             "measures the basin of attraction. Required "
                             "when --init truth.")
    # Filter knobs that default in FilterConfig/ProposalConfig: passed only
    # when given, so the struct defaults (which the manifest records) stay
    # the single source (the old driver duplicated pi0 and the gate default
    # here, and the copies drifted).
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pi0", type=float, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--backend", default="numpy",
                        choices=["numpy", "torch"],
                        help="measurement-update engine; torch uses the GPU "
                             "when available (float32)")
    parser.add_argument("--association_renewal_rate", type=float,
                        default=None,
                        help="beta: per-epoch probability a tracklet's "
                             "identity switched (FilterConfig default)")
    parser.add_argument("--init_min_tracklets", type=int, default=None,
                        help="distinct tracklets the init proposal waits "
                             "for (ProposalConfig default)")
    parser.add_argument("--evidence_gate_selection_charge",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="charge log(n_hypotheses) in the evidence gate "
                             "(ProposalConfig default: on)")
    parser.add_argument("--no_proposal", action="store_true",
                        help="brute-force control: uniform prior with no "
                             "resection proposal")
    parser.add_argument("--no_bearings", action="store_true",
                        help="odometry-only control: how much of the answer "
                             "is dead reckoning alone")
    parser.add_argument("--sources_dir", type=Path, default=None,
                        help="optional pointer to the tracklet payload "
                             "(crops, prompts) for the viewer")
    args = parser.parse_args()

    data = export_ingest.load(args.export_dir,
                              max_visible_range_m=args.max_visible_range_m)
    print(export_ingest.describe(data))

    if args.init == "uniform":
        init = export_ingest.region_box(data, args.margin_m)
        print(f"prior       : uniform over "
              f"{(init.east_max_m - init.east_min_m) / 1000:.1f} x "
              f"{(init.north_max_m - init.north_min_m) / 1000:.1f} km, "
              f"uniform heading")
    else:
        if args.prior_sigma_m is None:
            parser.error("--init truth requires --prior_sigma_m")
        if not data.truth:
            parser.error("--init truth needs truth.jsonl in the export")
        start = data.truth[0]
        init = structs.GaussianInit(start.east_m, start.north_m,
                                    args.prior_sigma_m)
        print(f"prior       : Gaussian at the first truth pose, sigma "
              f"{args.prior_sigma_m:.0f} m (DIAGNOSTIC CONTROL, not an "
              f"evaluation)")

    optional = {}
    if args.pi0 is not None:
        optional["pi0"] = args.pi0
    if args.association_renewal_rate is not None:
        optional["association_renewal_rate"] = args.association_renewal_rate
    proposal_optional = {}
    if args.init_min_tracklets is not None:
        proposal_optional["min_tracklets_for_injection"] = \
            args.init_min_tracklets
    if args.evidence_gate_selection_charge is not None:
        proposal_optional["evidence_gate_selection_charge"] = \
            args.evidence_gate_selection_charge

    filter_config = structs.FilterConfig(
        n_particles=args.n_particles, seed=args.seed, init=init,
        position_roughening_m=args.position_roughening_m,
        heading_roughening_deg=args.heading_roughening_deg,
        checkpoint_every=args.checkpoint_every,
        measurement_backend=args.backend,
        proposal=structs.ProposalConfig(
            enabled=not args.no_proposal, **proposal_optional),
        **optional)

    measurements = [] if args.no_bearings else data.measurements
    tables = {} if args.no_bearings else data.tables
    history = pf.run_filter(filter_config, data.catalog, data.odometry,
                            measurements, tables)

    manifest = structs.RunManifest(
        schema_version=structs.SCHEMA_VERSION,
        scenario_name=data.meta.scenario_name,
        anchor_lat_deg=data.meta.anchor_lat_deg,
        anchor_lon_deg=data.meta.anchor_lon_deg,
        n_keyframes=data.n_keyframes,
        filter_config=filter_config,
        landmarks=data.landmarks,
        matcher_version=(f"{data.meta.matcher_version} (bearings withheld)"
                         if args.no_bearings else data.meta.matcher_version),
        max_visible_range_m=args.max_visible_range_m,
        export_dir=str(args.export_dir),
        git_commit=provenance.git_commit(),
        argv=list(sys.argv),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        particle_history_sha256=history.particle_history_sha256,
        sources_dir=str(args.sources_dir) if args.sources_dir else None)
    # The run records what the filter CONSUMED — empty under --no_bearings.
    run_io.write_run(args.output_dir, manifest, data.truth, data.odometry,
                     measurements, tables, history)

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
    print(f"\nRun written to {args.output_dir}")


if __name__ == "__main__":
    main()
