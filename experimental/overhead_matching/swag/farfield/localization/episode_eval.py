"""Evaluate a sealed build's localization recipe on fixed-length episodes of
its export (see episodes.py): each episode is an independent uniform-prior
run over a segment, half of them driven in reverse.

  bazel run //experimental/overhead_matching/swag/farfield/localization:episode_eval -- \
      --build_dir /data/farfield_matching/builds/<dataset>/<build> \
      --out_dir /data/farfield_matching/runs/<experiment>/episodes/<dataset>__<build> \
      --segment_length_m 3000 --workers 2

Writes <out_dir>/<episode>/metrics.json (the shared position-mass summary)
and health.jsonl per episode, and <out_dir>/episodes_summary.json plus a
Markdown table. This is an evaluation tool: it does not publish run
artifacts (no run_io.write_run), and the filter config is exactly the
build's, with the init box unchanged (the whole region).
"""
import argparse
import concurrent.futures
import json
import math
import sys
from pathlib import Path

import msgspec

from experimental.overhead_matching.swag.farfield import build_config
from experimental.overhead_matching.swag.farfield.localization import (
    episodes as episodes_mod,
    export_ingest,
    filter as pf,
    metrics,
    run_export,
    runner,
    structs,
)


def _load(build_dir: Path, input_dir: Path | None = None):
    document = build_config.load(build_dir)
    version = build_config.value(document, "artifacts.localization_inputs_version")
    if input_dir is None:
        root = document["inputs"]["farfield_root"]
        input_dir = Path(root) / "artifacts" / "localization_inputs" / document["dataset"] / version
    elif input_dir.name != version:
        raise ValueError(f"--input_dir {input_dir} is not the build's export version {version}")
    data = export_ingest.load(input_dir)
    localization = document["config"]["localization"]
    config = run_export._filter_config(localization, data)  # noqa: SLF001
    return data, localization, config


def run_episode(build_dir: str, out_dir: str, index: int, segment_length_m: float,
                seed: int | None, n_particles: int | None = None,
                input_dir: str | None = None) -> dict:
    data, localization, config = _load(Path(build_dir), Path(input_dir) if input_dir else None)
    if seed is not None:
        config = msgspec.structs.replace(config, seed=seed)
    if n_particles is not None:
        config = msgspec.structs.replace(config, n_particles=n_particles)
    episode = episodes_mod.plan(data.truth, segment_length_m)[index]
    odometry, measurements, truth = episodes_mod.derive(
        data.odometry, data.measurements, data.truth, episode)
    if not localization["bearings_enabled"]:
        measurements, tables = [], {}
    else:
        tables = data.tables
    metric_config = metrics.position_mass_metric_config(localization["position_mass_radii_m"])
    recorder = runner.PositionMassRecorder(truth, metric_config)
    history = pf.run_filter(config, data.catalog, odometry, measurements, tables,
                            observer=recorder)
    for record in history.health:
        record.position_probability_mass = recorder.by_keyframe[record.keyframe_idx]
    summary = metrics.position_mass_summary(history.health, truth, metric_config)
    truth_by_kf = {p.keyframe_idx: p for p in truth}
    last = history.health[-1]
    final_err = math.hypot(last.map_east_m - truth_by_kf[last.keyframe_idx].east_m,
                           last.map_north_m - truth_by_kf[last.keyframe_idx].north_m)
    key_1km = metrics.position_mass_metric_key(metric_config, 1000.0)
    masses = [h.position_probability_mass[key_1km] for h in history.health]
    ep_dir = Path(out_dir) / episode.name
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / "metrics.json").write_text(json.dumps(summary, indent=1))
    with (ep_dir / "health.jsonl").open("w") as f:
        for h in history.health:
            t = truth_by_kf[h.keyframe_idx]
            f.write(json.dumps({
                "keyframe_idx": h.keyframe_idx,
                "mass_1km": h.position_probability_mass[key_1km],
                "map_error_m": math.hypot(h.map_east_m - t.east_m, h.map_north_m - t.north_m),
                "position_std_m": h.position_std_m}) + "\n")
    radii = summary["radii"]
    return {
        "episode": episode.name, "index": index, "reverse": episode.reverse,
        "start_keyframe": episode.start_keyframe, "end_keyframe": episode.end_keyframe,
        "length_m": round(episode.length_m), "n_keyframes": episode.n_keyframes,
        "dn100": radii["100"]["distance_normalized_mass"],
        "dn500": radii["500"]["distance_normalized_mass"],
        "dn1000": radii["1000"]["distance_normalized_mass"],
        "final_mass_1km": masses[-1], "final_map_error_m": final_err,
        "localized_at_end": bool(masses[-1] > 0.5),
        "n_proposal_events": len(history.proposal_events),
        "n_injections": sum(1 for e in history.proposal_events if e.n_injected > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--segment_length_m", type=float, default=3000.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--episodes", type=int, nargs="*", default=None,
                        help="subset of episode indices (default all)")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the build's filter seed")
    parser.add_argument("--n_particles", type=int, default=None,
                        help="override the particle count (smoke tests only)")
    parser.add_argument("--input_dir", type=Path, default=None,
                        help="localization_inputs export, when not under the build's farfield_root "
                             "(another machine); must be the build's version")
    args = parser.parse_args()
    data, localization, config = _load(args.build_dir, args.input_dir)
    plan = episodes_mod.plan(data.truth, args.segment_length_m)
    indices = args.episodes if args.episodes else list(range(len(plan)))
    print(f"{data.meta.dataset}: {len(plan)} episodes of ~{args.segment_length_m:.0f} m "
          f"over {metrics.cumulative_distance_m(data.truth)[data.truth[-1].keyframe_idx] / 1000:.1f} km; "
          f"running {len(indices)} with {args.workers} worker(s)")
    for ep in plan:
        print(f"  {ep.name}: {ep.n_keyframes} kf, {ep.length_m:.0f} m")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    if args.workers <= 1:
        for i in indices:
            results.append(run_episode(str(args.build_dir), str(args.out_dir), i,
                                       args.segment_length_m, args.seed, args.n_particles,
                                       str(args.input_dir) if args.input_dir else None))
            print(f"  done {results[-1]['episode']}: dn500 {results[-1]['dn500']:.3f}")
    else:
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = {pool.submit(run_episode, str(args.build_dir), str(args.out_dir), i,
                                   args.segment_length_m, args.seed, args.n_particles,
                                   str(args.input_dir) if args.input_dir else None): i
                       for i in indices}
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                print(f"  done {results[-1]['episode']}: dn500 {results[-1]['dn500']:.3f}")
    results.sort(key=lambda r: r["index"])
    n = len(results)
    aggregate = {
        "dataset": data.meta.dataset, "build_dir": str(args.build_dir),
        "segment_length_m": args.segment_length_m, "n_episodes": n,
        "mean_dn100": sum(r["dn100"] for r in results) / n,
        "mean_dn500": sum(r["dn500"] for r in results) / n,
        "mean_dn1000": sum(r["dn1000"] for r in results) / n,
        "localized_at_end": sum(r["localized_at_end"] for r in results),
        "median_final_map_error_m": sorted(r["final_map_error_m"] for r in results)[n // 2],
        "episodes": results,
    }
    (args.out_dir / "episodes_summary.json").write_text(json.dumps(aggregate, indent=1))
    lines = ["| episode | kf | m | dn100 | dn500 | dn1000 | final 1 km mass | final MAP err m | injections |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['episode']} | {r['n_keyframes']} | {r['length_m']} | {r['dn100']:.3f} | "
                     f"{r['dn500']:.3f} | {r['dn1000']:.3f} | {r['final_mass_1km']:.2f} | "
                     f"{r['final_map_error_m']:.0f} | {r['n_injections']} |")
    lines.append(f"| **mean / count** | | | {aggregate['mean_dn100']:.3f} | {aggregate['mean_dn500']:.3f} | "
                 f"{aggregate['mean_dn1000']:.3f} | localized {aggregate['localized_at_end']}/{n} | "
                 f"median {aggregate['median_final_map_error_m']:.0f} | |")
    table = "\n".join(lines)
    (args.out_dir / "episodes_table.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
