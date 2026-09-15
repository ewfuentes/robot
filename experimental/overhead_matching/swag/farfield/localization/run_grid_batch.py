"""Run a JSON list of grid_filter configs in one process, raw before hybrid."""

import argparse
import copy
import gc
import itertools
import json
from functools import lru_cache
from pathlib import Path

from experimental.overhead_matching.swag.farfield.localization import grid_filter


LOCI_COMPARISON_SETTINGS = {
    "availability": "immediate",
    "odometry_profile": "epson_mg570_calibrated_planar_v1",
    "odometry_seed": 0,
    "cell_m": 100.0,
    "n_heading": 36,
    "yaw_sigma_scale": 1.0,
    "heading_rw_deg": 1.0,
    "diffusion_m": 5.0,
    "margin_m": 0.0,
    "top_modes": 0,
}


def job_order(config):
    parent = str(Path(config.get("detection_input_dir") or config["input_dir"]).resolve())
    full, start = True, 0
    if config.get("episode_plan") is not None:
        windows = json.loads(Path(config["episode_plan"]).read_text())["windows"]
        start = windows[config["episode_index"]][0]
        full = len(windows) == 1
    # ponytail: fixed ordering, not a general cache scheduler. The cached tail
    # overlaps the latest window; pair each raw producer with its hybrid reader.
    return (parent, config.get("device", "cuda"), not full, -start,
            bool(config.get("detection_input_dir")))


def run(configs, cache_gib=8):
    if not isinstance(configs, list) or not configs:
        raise ValueError("jobs must be a nonempty JSON list of grid_filter configs")
    outputs = set()
    jobs = []
    for config in configs:
        config = dict(config)
        panorama = config.get("observation_source") in ("loci", "crosslocate")
        if panorama:
            mismatched = {
                key: (config.get(key), expected)
                for key, expected in LOCI_COMPARISON_SETTINGS.items()
                if key not in config or config[key] != expected
            }
            if (mismatched or config.get("episode_plan") is None
                    or config.get("episode_index") is None):
                raise ValueError(
                    "panorama comparison jobs require explicit paired episode, "
                    f"odometry, grid, and motion settings: {mismatched}")
        if ((not panorama and not config.get("track_joint"))
                or config.get("smoother", "none") != "none"
                or config.get("smooth_lag", 0)
                or config.get("smooth_lags", "").strip()):
            raise ValueError(
                "batch evaluation requires panorama or joint mode and causal-only scoring")
        if not config.get("out"):
            raise ValueError("every job needs a distinct output path")
        output = Path(config["out"]).resolve()
        if output.exists() or output in outputs:
            raise ValueError(f"refusing existing or repeated output: {output}")
        outputs.add(output)
        config.setdefault("top_modes", 0)
        config["likelihood_cache_gb"] = 0  # replay already retains its own factors
        jobs.append((job_order(config), config))

    for (parent, device), group in itertools.groupby(
            sorted(jobs, key=lambda job: job[0]), key=lambda job: job[0][:2]):
        cache = grid_filter.LikelihoodCache(cache_gib * 1024**3, device)
        # Inputs are frozen for this parent batch; validate on first load and
        # copy because per-run odometry and matching overrides are mutable.
        load = lru_cache(maxsize=2)(grid_filter.export_ingest.load)
        try:
            for _, config in group:
                print(f"batch: {parent} -> {config['out']}", flush=True)
                argv = [arg for key, value in config.items() if value is not None
                        for arg in ("--" + key, str(value))]
                grid_filter.main(argv, raw_cache=cache,
                                 load_input=lambda path: copy.deepcopy(load(path.resolve())))
                # Collect main's closure cycles before allocating the next replay.
                gc.collect()
                print("shared raw " + cache.describe(), flush=True)
        finally:
            load.cache_clear()
            del cache
            gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--cache_gib", type=float, default=8)
    args = parser.parse_args()
    run(json.loads(args.jobs.read_text()), args.cache_gib)


if __name__ == "__main__":
    main()
