"""`runlog` — the viewer's primitives on the command line (design doc §7.5).

§7.5 asks for a CLI exposing the same primitives as the viewer, "for scripted
forensics and CI". Same modules, same answers, no browser:

  runlog check       is this run faithfully replayable, and what is missing
  runlog attribute   build the Tier-3 cache; print a mode's waterfall
  runlog replay      run a counterfactual and write it as a run directory
  runlog tracklet    one tracklet's dossier, including the truth triage
  runlog events      the §7.3 event index
  runlog triage      the per-tracklet culpability table for a whole run

Counterfactual output defaults under /tmp: a what-if is a question, not a
result, and questions should not accumulate next to the runs they were asked
about.

Examples
--------
  # Which tracklets are hurting, and why?
  runlog triage --run_dir RUN --sort worst

  # Build attribution once, then ask about a mode.
  runlog attribute --run_dir RUN
  runlog attribute --run_dir RUN --mode 0 --from_kf 290 --to_kf 310

  # Was tracklet LT267 actually the culprit?
  runlog replay --run_dir RUN --without_tracklet LT267

  # Would the run have converged if the matcher had got LT267 right?
  runlog replay --run_dir RUN --force_landmark LT267=enc:0226025A1C7615CA
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    attribution as attribution_mod,
    forensics,
    replay as replay_mod,
    run_log,
    sources as sources_mod,
)

DEFAULT_COUNTERFACTUAL_BASE = Path("/tmp/runlog_counterfactuals")


def _load(run_dir: Path):
    data = run_log.read_run(run_dir)
    inputs = replay_mod.load_inputs(run_dir, data=data)
    return data, inputs


def _map_errors(data) -> np.ndarray:
    truth_by_kf = {t.keyframe_idx: t for t in data.truth}
    return np.array([
        math.hypot(r.map_east_m - truth_by_kf[r.keyframe_idx].east_m,
                   r.map_north_m - truth_by_kf[r.keyframe_idx].north_m)
        for r in data.health if r.keyframe_idx in truth_by_kf])


def cmd_check(args) -> int:
    status = replay_mod.replayability(args.run_dir)
    print(status.report())
    if args.replay:
        result = replay_mod.replay(args.run_dir, verify=False,
                                  max_visible_range_m=args.max_visible_range_m)
        print(result.report())
        return 0 if result.hash_match else 1
    return 0 if status.replayable else 1


def cmd_attribute(args) -> int:
    cache = None
    if not args.force:
        try:
            data = run_log.read_run(args.run_dir)
            cache = attribution_mod.read_cache(
                args.run_dir,
                expected_sha256=data.manifest.particle_history_sha256 or None)
        except ValueError as exc:
            print(f"cached attribution rejected: {exc}", file=sys.stderr)
    if cache is None:
        print("replaying under instrumentation ...", file=sys.stderr)
        cache, result = attribution_mod.compute(
            args.run_dir, max_visible_range_m=args.max_visible_range_m,
            verify=not args.allow_divergence)
        print(result.report(), file=sys.stderr)
        path = attribution_mod.write_cache(args.run_dir, cache)
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB, "
              f"{len(cache.contributions)} rows)", file=sys.stderr)

    groups = sorted({w.group for w in cache.group_weights})
    if args.mode is not None:
        groups = [args.mode]
    span = None
    if args.from_kf is not None or args.to_kf is not None:
        span = (args.from_kf or 0,
                args.to_kf if args.to_kf is not None else cache.n_keyframes - 1)
    for group in groups:
        print()
        print(attribution_mod.attribute(cache, group, span,
                                        include_structural=not args.evidence_only
                                        ).report(top=args.top))
    return 0


def cmd_replay(args) -> int:
    log_lr: dict = {}
    for spec in args.set_log_lr or ():
        try:
            tracklet, rest = spec.split(":", 1)
            landmark, value = rest.rsplit("=", 1)
        except ValueError:
            print(f"--set_log_lr expects TRACKLET:LANDMARK=VALUE, got {spec!r}",
                  file=sys.stderr)
            return 2
        log_lr.setdefault(tracklet, {})[landmark] = float(value)

    force: dict = {}
    for spec in args.force_landmark or ():
        try:
            tracklet, landmark = spec.split("=", 1)
        except ValueError:
            print(f"--force_landmark expects TRACKLET=LANDMARK, got {spec!r}",
                  file=sys.stderr)
            return 2
        force[tracklet] = landmark

    if args.oracle_matcher:
        # [TRUTH-PRIVILEGED] Replace the matcher's claim with the landmark that
        # actually explains each tracklet's bearings, for every tracklet whose
        # matcher failed to endorse one. This is the experiment behind "is the
        # matcher the bottleneck": an oracle matcher is not a system we could
        # build, but the gap between it and the real run bounds how much is
        # left to win upstream of the filter.
        data, inputs = _load(args.run_dir)
        triage = forensics.triage_tracklets(data, inputs.catalog)
        if not triage:
            print("--oracle_matcher needs ground truth, which this run has no "
                  "record of", file=sys.stderr)
            return 2
        promoted, skipped = [], []
        for tracklet_id, verdict in sorted(triage.items()):
            if verdict.verdict == "consistent" or not verdict.geometry_explicable:
                continue
            if verdict.ambiguous and not args.include_ambiguous:
                skipped.append(tracklet_id)
                continue
            force.setdefault(tracklet_id, verdict.best_fit.landmark_id)
            promoted.append(tracklet_id)
        print(f"[TRUTH-PRIVILEGED] oracle matcher: endorsing the "
              f"best-fitting landmark for {len(promoted)} tracklet(s)",
              file=sys.stderr)
        if skipped:
            print(f"  skipped {len(skipped)} geometrically ambiguous "
                  f"tracklet(s), where 'the' right answer is not identifiable "
                  f"(--include_ambiguous to force them anyway): "
                  + ", ".join(skipped[:8])
                  + (", ..." if len(skipped) > 8 else ""), file=sys.stderr)

    edits = replay_mod.Edits(
        drop_tracklets=tuple(args.without_tracklet or ()),
        keep_only_tracklets=(tuple(args.only_tracklet)
                             if args.only_tracklet else None),
        log_lr=log_lr, force_landmark=force, pi0=args.pi0,
        matcher_recall=args.matcher_recall, seed=args.seed,
        n_particles=args.n_particles,
        measurement_backend=args.backend,
        disable_proposal=args.no_proposal,
        disable_persistence=args.no_persistence,
        disable_modes=args.no_modes)
    if edits.is_empty:
        print("no edits given: this would just re-run the baseline. Use "
              "`runlog check --replay` for that.", file=sys.stderr)
        return 2

    print(f"counterfactual: {edits.describe()}", file=sys.stderr)
    result = replay_mod.replay(args.run_dir, edits=edits,
                              max_visible_range_m=args.max_visible_range_m)
    print(result.report(), file=sys.stderr)

    output_dir = args.output_dir or (DEFAULT_COUNTERFACTUAL_BASE
                                    / Path(args.run_dir).name
                                    / edits.slug())
    replay_mod.write_counterfactual(output_dir, args.run_dir, result)

    baseline, _ = _load(args.run_dir)
    base_errors = _map_errors(baseline)
    ghost_errors = _map_errors(run_log.read_run(output_dir))
    print()
    print(f"wrote {output_dir}")
    if base_errors.size and ghost_errors.size:
        print(f"  MAP error   baseline -> counterfactual")
        print(f"    final     {base_errors[-1]:8.0f} m -> "
              f"{ghost_errors[-1]:8.0f} m")
        print(f"    median    {np.median(base_errors):8.0f} m -> "
              f"{np.median(ghost_errors):8.0f} m")
    print(f"  modes at end: {len(baseline.health[-1].modes)} -> "
          f"{len(run_log.read_run(output_dir).health[-1].modes)}")
    print(f"\nOverlay it:  viewer --run_dir {args.run_dir} "
          f"--ghost {output_dir}")
    return 0


def cmd_tracklet(args) -> int:
    data, inputs = _load(args.run_dir)
    triage = forensics.triage_tracklets(data, inputs.catalog)
    verdict = triage.get(args.tracklet)
    table = data.tables.get(args.tracklet)
    epochs = [m for m in data.measurements if m.tracklet_id == args.tracklet]
    if not epochs:
        print(f"no measurements for tracklet {args.tracklet!r}",
              file=sys.stderr)
        return 1

    print(f"tracklet {args.tracklet}: {len(epochs)} epochs, "
          f"kf {min(m.anchor_keyframe_idx for m in epochs)}"
          f"-{max(m.anchor_keyframe_idx for m in epochs)}")
    sigmas = [math.degrees(1.0 / math.sqrt(max(m.kappa, 1e-9))) for m in epochs]
    print(f"  bearing sigma {min(sigmas):.1f}-{max(sigmas):.1f} deg")

    if table is None:
        print("  matcher: NO TABLE — the filter had only geometry")
    else:
        default = min(max(table.default_log_lr, table.clip_lo), table.clip_hi)
        endorsed = [e for e in table.entries
                    if min(max(e.log_lr, table.clip_lo), table.clip_hi)
                    > default + 1e-12]
        print(f"  matcher: {table.status}, {len(table.entries)} entries, "
              f"{len(endorsed)} endorsed, default {default:.2f}, "
              f"clip [{table.clip_lo}, {table.clip_hi}]")
        for entry in sorted(endorsed, key=lambda e: -e.log_lr)[:args.top]:
            print(f"    {entry.log_lr:+6.2f}  {entry.landmark_id}")

    if args.sources_dir:
        bundle = sources_mod.load(args.sources_dir, [args.tracklet],
                                  embed_thumbnails=False)
        source = bundle.get(args.tracklet)
        if source is not None:
            print(f"  tracker: name {source.best_name!r}, "
                  f"{source.n_supports} supports, tracks {source.track_ids}")
            if source.description:
                print(f"    \"{source.description}\"")
            if source.unresolved:
                print(f"    flagged: {source.unresolved}")
            if source.no_match_rate is not None:
                print(f"    matcher's own verdict over "
                      f"{source.n_matcher_chunks} chunks: no-match "
                      f"{source.no_match_rate:.2f}, uniqueness "
                      f"{source.median_uniqueness}")
        for note in bundle.notes:
            print(f"    note: {note}")

    try:
        cache = attribution_mod.read_cache(
            args.run_dir,
            expected_sha256=data.manifest.particle_history_sha256 or None)
    except ValueError as exc:
        cache, _ = None, print(f"  attribution: {exc}")
    if cache is not None:
        series = attribution_mod.tracklet_series(cache, args.tracklet)
        total = sum(row.self_nats for row in series)
        print(f"  filter: contributed {total:+.2f} nats to the whole belief "
              f"over {len(series)} epochs")
        worst = sorted(series, key=lambda r: r.self_nats)[:3]
        if worst:
            print("    worst epochs: " + ", ".join(
                f"kf{row.keyframe_idx} {row.self_nats:+.2f}" for row in worst))
    else:
        print("  filter: no attribution cache (run `runlog attribute`)")

    if verdict is None:
        print("  triage: unavailable (no ground truth)")
        return 0
    print(f"\n  [TRUTH-PRIVILEGED] verdict: {verdict.verdict}"
          + ("  ANTI-EVIDENCE" if verdict.anti_evidence else "")
          + ("  GEOMETRICALLY-AMBIGUOUS" if verdict.ambiguous else ""))
    print(f"    tolerance {verdict.tolerance_deg:.1f} deg "
          f"(3x median sigma {verdict.median_sigma_deg:.2f}, clamped); "
          f"{verdict.n_consistent_catalog} catalog rows fit")

    def show(label, fit):
        if fit is None:
            print(f"    {label:<20} -")
            return
        verdicts = ("explains it" if fit.explains(verdict.tolerance_deg)
                    else "DOES NOT explain it")
        print(f"    {label:<20} {fit.landmark_id:<36.36} RMS "
              f"{fit.rms_deg:6.2f} deg  worst {fit.max_deg:6.1f}  "
              f"{fit.median_range_m / 1000:5.1f} km  "
              + (f"LLR {fit.log_lr:+.2f}  " if fit.log_lr is not None else "")
              + verdicts)

    show("best in catalog", verdict.best_fit)
    show("best endorsed", verdict.best_endorsed_fit)
    show("matcher's top claim", verdict.top_endorsed_fit)
    print(f"    max mass the filter put on a fitting endorsed entry: "
          f"{verdict.best_filter_share * 100:.0f}%")
    print(f"\n    {'kf':>5} {'bearing':>8} {'sigma':>6} {'best res':>9} "
          f"{'top res':>8} {'mass fit':>9} {'null':>6} {'surp':>6}  "
          f"filter believes")
    for epoch in verdict.epochs[:args.top]:
        print(f"    {epoch.keyframe_idx:>5} {epoch.bearing_body_deg:>8.1f} "
              f"{epoch.sigma_deg:>6.1f} "
              f"{(f'{epoch.best_fit_residual_deg:.2f}' if epoch.best_fit_residual_deg is not None else '-'):>9} "
              f"{(f'{epoch.top_endorsed_residual_deg:.1f}' if epoch.top_endorsed_residual_deg is not None else '-'):>8} "
              f"{epoch.best_fit_share * 100:>8.0f}% "
              f"{epoch.null_share * 100:>5.0f}% "
              f"{epoch.surprise_share * 100:>5.0f}%  "
              f"{epoch.filter_top_id or '-'}")
    if len(verdict.epochs) > args.top:
        print(f"    ... {len(verdict.epochs) - args.top} more epochs")
    return 0


def cmd_events(args) -> int:
    data = run_log.read_run(args.run_dir)
    events = forensics.derive_events(data)
    if args.kind:
        events = [e for e in events if e.kind == args.kind]
    print(f"{len(events)} events")
    for event in events:
        print(f"  kf {event.keyframe_idx:>5}  {event.severity:<5} "
              f"{event.kind:<17} {event.source:<7} {event.label} — "
              f"{event.detail}")
    return 0


def cmd_triage(args) -> int:
    data, inputs = _load(args.run_dir)
    triage = forensics.triage_tracklets(data, inputs.catalog)
    if not triage:
        print("no ground truth in this run: triage unavailable")
        return 1

    cache = None
    try:
        cache = attribution_mod.read_cache(
            args.run_dir,
            expected_sha256=data.manifest.particle_history_sha256 or None)
    except ValueError:
        pass
    nats = {}
    if cache is not None:
        for tracklet_id in triage:
            nats[tracklet_id] = sum(
                row.self_nats
                for row in attribution_mod.tracklet_series(cache, tracklet_id))

    rows = list(triage.values())
    if args.sort == "worst":
        # Most-negative contribution first when attribution is available, else
        # faults before consistent: either way the top of the list is where the
        # error budget lives.
        order = {"tracker-fault": 0, "matcher-fault": 1, "filter-fault": 2,
                 "no-evidence": 3, "consistent": 4}
        rows.sort(key=lambda r: (nats.get(r.tracklet_id, 0.0)
                                 if nats else order.get(r.verdict, 9)))
    else:
        rows.sort(key=lambda r: r.tracklet_id)

    print("[TRUTH-PRIVILEGED] per-tracklet culpability. A debugging "
          "instrument; not evidence of localization performance.")
    print(forensics.triage_summary(triage))
    print()
    print(f"{'tracklet':<20} {'verdict':<14} {'n':>3} {'tol':>5} {'bestRMS':>8} "
          f"{'endRMS':>7} {'topRMS':>7} {'fits':>6} {'nats':>8} {'end':>4} "
          f"{'mass':>5}  best-fitting catalog landmark")
    for row in rows:
        def rms(fit):
            if fit is None or not math.isfinite(fit.rms_deg):
                return "-"
            return f"{fit.rms_deg:.2f}"
        flags = ""
        if row.anti_evidence:
            flags += "  ANTI"
        if row.ambiguous:
            flags += "  AMBIG"
        print(f"{row.tracklet_id:<20.20} {row.verdict:<14} {row.n_epochs:>3} "
              f"{row.tolerance_deg:>5.1f} {rms(row.best_fit):>8} "
              f"{rms(row.best_endorsed_fit):>7} {rms(row.top_endorsed_fit):>7} "
              f"{row.n_consistent_catalog:>6} "
              f"{(f'{nats[row.tracklet_id]:+.1f}' if row.tracklet_id in nats else '-'):>8} "
              f"{row.n_endorsed:>4} "
              f"{row.best_filter_share * 100:>4.0f}%  "
              f"{(row.best_fit.landmark_id if row.best_fit else '-')}" + flags)
    print("\ntol = tolerance in degrees (3x the tracklet's median sigma, "
          "clamped to [5, 25]).")
    print("bestRMS = best angular fit anywhere in the catalog; endRMS = best "
          "among endorsed entries;")
    print("topRMS = the matcher's highest-LLR claim; fits = catalog rows "
          "within tolerance (AMBIG when")
    print("that is large enough that the geometry identifies nothing).")
    if not nats:
        print("\n(no attribution cache, so the nats column is empty; run "
              "`runlog attribute` to fill it)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="runlog", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub):
        sub.add_argument("--run_dir", type=Path, required=True)
        sub.add_argument("--max_visible_range_m", type=float, default=None,
                         help="override the catalog visibility radius for "
                              "runs whose manifest predates it")

    check = subparsers.add_parser("check", help="replayability of a run")
    add_common(check)
    check.add_argument("--replay", action="store_true",
                       help="actually replay and compare history hashes")
    check.set_defaults(func=cmd_check)

    attribute = subparsers.add_parser(
        "attribute", help="build/query the §7.2 attribution")
    add_common(attribute)
    attribute.add_argument("--mode", type=int, default=None)
    attribute.add_argument("--from_kf", type=int, default=None)
    attribute.add_argument("--to_kf", type=int, default=None)
    attribute.add_argument("--top", type=int, default=10)
    attribute.add_argument("--evidence_only", action="store_true",
                           help="omit the structural terms (re-clustering, "
                                "injection, resampling)")
    attribute.add_argument("--force", action="store_true",
                           help="recompute even if a valid cache exists")
    attribute.add_argument("--allow_divergence", action="store_true",
                           help="attribute a replay that does not reproduce "
                                "the run's hash (the numbers will describe a "
                                "different run)")
    attribute.set_defaults(func=cmd_attribute)

    rep = subparsers.add_parser("replay", help="run a counterfactual")
    add_common(rep)
    rep.add_argument("--without_tracklet", action="append",
                     help="silence a tracklet entirely (repeatable)")
    rep.add_argument("--only_tracklet", action="append",
                     help="keep only these tracklets (repeatable)")
    rep.add_argument("--force_landmark", action="append",
                     metavar="TRACKLET=LANDMARK",
                     help="rewrite a tracklet's table to endorse one landmark "
                          "at clip_hi: 'what if the matcher had got this "
                          "right?'")
    rep.add_argument("--set_log_lr", action="append",
                     metavar="TRACKLET:LANDMARK=VALUE")
    rep.add_argument("--oracle_matcher", action="store_true",
                     help="[TRUTH-PRIVILEGED] endorse the geometrically "
                          "best-fitting landmark for every tracklet the real "
                          "matcher got wrong: bounds how much is left to win "
                          "upstream of the filter")
    rep.add_argument("--include_ambiguous", action="store_true",
                     help="with --oracle_matcher, also force tracklets whose "
                          "geometry does not identify a unique landmark")
    rep.add_argument("--pi0", type=float, default=None)
    rep.add_argument("--matcher_recall", type=float, default=None)
    rep.add_argument("--seed", type=int, default=None)
    rep.add_argument("--n_particles", type=int, default=None)
    rep.add_argument("--backend", default=None, choices=["numpy", "torch"])
    rep.add_argument("--no_proposal", action="store_true")
    rep.add_argument("--no_persistence", action="store_true")
    rep.add_argument("--no_modes", action="store_true")
    rep.add_argument("--output_dir", type=Path, default=None,
                     help=f"defaults under {DEFAULT_COUNTERFACTUAL_BASE}")
    rep.set_defaults(func=cmd_replay)

    trk = subparsers.add_parser("tracklet", help="one tracklet's dossier")
    add_common(trk)
    trk.add_argument("tracklet")
    trk.add_argument("--sources_dir", type=Path, default=None)
    trk.add_argument("--top", type=int, default=12)
    trk.set_defaults(func=cmd_tracklet)

    events = subparsers.add_parser("events", help="the §7.3 event index")
    add_common(events)
    events.add_argument("--kind", default=None)
    events.set_defaults(func=cmd_events)

    triage = subparsers.add_parser(
        "triage", help="per-tracklet culpability (truth-privileged)")
    add_common(triage)
    triage.add_argument("--sort", default="worst",
                        choices=["worst", "name"])
    triage.set_defaults(func=cmd_triage)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
