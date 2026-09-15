"""Generate the far-field paper's LaTeX results table from pinned runs."""

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from experimental.overhead_matching.swag.farfield.localization import (
    metrics as metrics_lib,
)
from experimental.overhead_matching.swag.farfield.paper.table_common import (
    DATASET_GROUPS,
    DEFAULT_FARFIELD_ROOT,
    PAPER_LOCI_VERSION,
    PAPER_LOCI_STUDY,
    PAPER_TWOARM_STUDY,
    PAPER_NO_RANGE_STUDY,
    PAPER_CROSSLOCATE_STUDY,
    PAPER_CROSSLOCATE_FLEVOLAND_STUDY,
    PAPER_SEEDS,
    PAPER_SWEEPS,
    SEQUENCE_ARTIFACTS,
    DatasetGroup,
    emit_table,
    glob_runs,
    read_json_object,
)


DEFAULT_RADII_M = (100.0, 500.0)
SCORES = ("causal",)
METHODS = (
    ("crosslocate", "CrossLocate~\\cite{tomevsek2022crosslocate}"),
    ("loci", "LOCI~\\cite{fahnestockandfuentes2026loci}"),
    ("no_tracking", "\\shortstack{No\\\\tracking}"),
    ("no_range", "\\shortstack{No range\\\\bins}"),
    ("ours", "Ours"),
)
PAIRED_METHODS = (
    ("crosslocate_top6", "CrossLocate~\\cite{tomevsek2022crosslocate}"),
    *METHODS[1:],
)

SEQUENCE_DISPLAY_NAMES = {
    "mount_washington_20260815_leg1": "Mt. Washington, leg 1",
    "mount_washington_20260815_leg2": "Mt. Washington, leg 2",
    "mount_washington_20260815_leg3": "Mt. Washington, leg 3",
    "pohang_canal_04": "Pohang",
    "flevoland_polder": "Flevoland",
    "charles_river_20260727": "Charles River",
    "boston_harbor_leg1": "Boston Harbor, leg 1",
    "boston_harbor_leg2": "Boston Harbor, leg 2",
    "boston_harbor_leg3": "Boston Harbor, leg 3",
    "franconia_leg1": "Franconia",
    "portland_flight_20260906_leg1": "Portland, leg 1",
    "portland_flight_20260906_leg2": "Portland, leg 2",
    "portland_flight_20260906_leg3": "Portland, leg 3",
}

LOCI_VERSIONS = {
    sequence: PAPER_LOCI_VERSION
    for sequence in SEQUENCE_DISPLAY_NAMES
}

CURRENT_LOCALIZATION_VERSIONS = {
    sequence: artifacts["localization_inputs"]
    for sequence, artifacts in SEQUENCE_ARTIFACTS.items()
}

NO_TRACKING_VERSIONS = {
    sequence: (
        "notrack_20260913_nofull360_v1"
        if sequence in {"pohang_canal_04", "portland_flight_20260906_leg2"}
        else "notrack_20260913_v1"
    )
    for sequence in SEQUENCE_DISPLAY_NAMES
}


@dataclass(frozen=True)
class SequenceResult:
    dataset: str
    seed: int | None
    values: dict[float, float]
    trajectory_length_m: float
    run_dir: Path


@dataclass(frozen=True)
class Estimate:
    mean: float | None
    std: float | None
    count: int
    expected_count: int | None = None

    @property
    def incomplete(self) -> bool:
        return self.expected_count is not None and self.count < self.expected_count


def _localization_seed(manifest: dict, path: Path) -> int:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{path}: config must be an object")
    localization = config.get("localization")
    if isinstance(localization, dict):
        seed = localization.get("seed")
    else:
        contract = config.get("localization_run_contract")
        filter_config = contract.get("filter_config") if isinstance(contract, dict) else None
        seed = filter_config.get("seed") if isinstance(filter_config, dict) else None
    if type(seed) is not int:
        raise ValueError(f"{path}: localization seed must be an integer")
    return seed


def _finite_float(value: object, *, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path}: {field} must be finite")
    return result


def _legacy_metrics(run_dir: Path, radii_m: Sequence[float]) -> dict:
    """Re-score old time-normalized runs with the current distance metric."""
    def records(name: str) -> list[SimpleNamespace]:
        path = run_dir / name
        try:
            return [SimpleNamespace(**json.loads(line))
                    for line in path.read_text().splitlines() if line]
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Could not read legacy metric input {path}: {exc}") from exc

    return metrics_lib.position_mass_summary(
        records("tier0_health.jsonl"),
        records("truth.jsonl"),
        metrics_lib.position_mass_metric_config(radii_m),
    )


def _load_metrics(
    run_dir: Path, radii_m: Sequence[float]
) -> tuple[dict[float, float], float]:
    metrics_path = run_dir / "metrics.json"
    metrics = read_json_object(metrics_path)
    if metrics.get("schema") == "farfield_position_mass_summary/v1":
        metrics = _legacy_metrics(run_dir, radii_m)
    if metrics.get("schema") != metrics_lib.POSITION_MASS_SUMMARY_SCHEMA:
        raise ValueError(
            f"{metrics_path}: expected {metrics_lib.POSITION_MASS_SUMMARY_SCHEMA}, "
            f"got {metrics.get('schema')!r}")
    if metrics.get("higher_is_better") is not True:
        raise ValueError(f"{metrics_path}: expected a higher-is-better metric")
    if metrics.get("reference_position") != "truth":
        raise ValueError(f"{metrics_path}: expected truth-referenced position mass")
    if metrics.get("normalization") != metrics_lib.POSITION_MASS_DISTANCE_NORMALIZATION:
        raise ValueError(f"{metrics_path}: expected distance normalization")
    trajectory_length_m = _finite_float(
        metrics.get("trajectory_length_m"), field="trajectory_length_m",
        path=metrics_path)
    if trajectory_length_m <= 0.0:
        raise ValueError(f"{metrics_path}: trajectory length must be positive")
    radii = metrics.get("radii")
    if not isinstance(radii, dict):
        raise ValueError(f"{metrics_path}: radii must be an object")

    values = {}
    for radius_m in radii_m:
        radius_key = f"{radius_m:g}"
        radius_entry = radii.get(radius_key)
        if not isinstance(radius_entry, dict):
            raise ValueError(f"{metrics_path}: missing radius {radius_key} m")
        recorded_radius = _finite_float(
            radius_entry.get("radius_m"), field="radii[].radius_m", path=metrics_path)
        if recorded_radius != radius_m:
            raise ValueError(
                f"{metrics_path}: radius {radius_key} entry records "
                f"{recorded_radius:g} m")
        value = _finite_float(
            radius_entry.get("distance_normalized_mass"),
            field="radii[].distance_normalized_mass", path=metrics_path)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{metrics_path}: posterior mass must be in [0, 1]")
        values[radius_m] = value
    return values, trajectory_length_m


def _aggregate(
    results: Sequence[SequenceResult], radii_m: Sequence[float]
) -> dict[float, Estimate]:
    estimates = {}
    for radius_m in radii_m:
        values = [result.values[radius_m] for result in results]
        estimates[radius_m] = Estimate(
            mean=statistics.fmean(values),
            std=statistics.stdev(values) if len(values) > 1 else None,
            count=len(values),
        )
    return estimates


def _validate_method(manifest: dict, path: Path, method: str) -> None:
    config = manifest.get("config")
    contract = config.get("localization_run_contract") if isinstance(config, dict) else None
    if not isinstance(contract, dict):
        raise ValueError(f"{path}: missing localization run contract")
    tags = contract.get("ablation_tags")
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError(f"{path}: invalid ablation tags")
    expected = {
        "crosslocate": ("diagnostic_control", {"retrieval_calibration_provisional"}),
        "no_tracking": ("diagnostic_control", {"no_tracking"}),
        "no_range": ("evaluation", set()),
        "ours": ("evaluation", set()),
    }
    if method not in expected:
        raise ValueError(f"unknown localization method {method!r}")
    if (contract.get("run_kind"), set(tags)) != expected[method]:
        raise ValueError(f"{path}: run contract does not identify {method}")
    filter_config = contract.get("filter_config")
    if not isinstance(filter_config, dict):
        raise ValueError(f"{path}: missing filter configuration")
    range_enabled = filter_config.get("range_cap_enabled") is True
    if range_enabled != (method == "ours"):
        raise ValueError(f"{path}: range setting does not identify {method}")
    if method == "crosslocate":
        retrieval = config.get("retrieval")
        if (
            config.get("observation_source") != "retrieval"
            or not isinstance(retrieval, dict)
            or retrieval.get("calibration_frozen") is not False
        ):
            raise ValueError(f"{path}: unexpected CrossLocate retrieval configuration")


def load_localization_results(
    run_dirs: Sequence[Path],
    expected_datasets: set[str],
    expected_seeds: set[int],
    radii_m: Sequence[float] = DEFAULT_RADII_M,
    *,
    method: str | None = None,
) -> dict[str, dict[float, Estimate]]:
    """Load one complete localization result per expected dataset and seed."""
    found: dict[str, dict[int, SequenceResult]] = {}
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        manifest = read_json_object(manifest_path)
        if manifest.get("kind") != "localization_run" or manifest.get("complete") is not True:
            raise ValueError(f"{manifest_path}: expected a complete localization run")
        dataset = manifest.get("dataset")
        if dataset not in expected_datasets:
            raise ValueError(f"{manifest_path}: unexpected dataset {dataset!r}")
        seed = _localization_seed(manifest, manifest_path)
        if seed not in expected_seeds:
            raise ValueError(f"{manifest_path}: unexpected seed {seed}")
        if method is not None:
            _validate_method(manifest, manifest_path, method)
        values, trajectory_length_m = _load_metrics(run_dir, radii_m)
        result = SequenceResult(dataset, seed, values, trajectory_length_m, run_dir)
        previous = found.setdefault(dataset, {}).setdefault(seed, result)
        if previous is not result:
            raise ValueError(
                f"duplicate seed {seed} for {dataset}: "
                f"{previous.run_dir}, {run_dir}")

    missing = {
        (dataset, seed)
        for dataset in expected_datasets
        for seed in expected_seeds
        if seed not in found.get(dataset, {})
    }
    if missing:
        raise ValueError(f"missing localization results: {sorted(missing)}")
    return {
        dataset: _aggregate(
            [by_seed[seed] for seed in sorted(expected_seeds)], radii_m
        )
        for dataset, by_seed in found.items()
    }


def load_grid_results(
    result_paths: Sequence[Path],
    expected_datasets: set[str],
    radii_m: Sequence[float] = DEFAULT_RADII_M,
    *,
    method: str,
    score: str = "causal",
) -> dict[str, dict[float, Estimate]]:
    """Load the final seed-0 grid-filter outputs."""
    if method not in {"ours", "no_range", "no_tracking"}:
        raise ValueError(f"unknown grid-filter method {method!r}")
    if score not in ("causal", "smoothed"):
        raise ValueError(f"unknown score {score!r}")
    found = {}
    for path in result_paths:
        result = read_json_object(path)
        inputs = result.get("localization_inputs")
        config = result.get("config")
        if result.get("schema") != "farfield_causal_grid/v1":
            raise ValueError(f"{path}: unexpected grid-filter schema")
        if not isinstance(inputs, dict) or not isinstance(config, dict):
            raise ValueError(f"{path}: missing grid-filter identity")
        dataset = inputs.get("dataset")
        if dataset not in expected_datasets:
            raise ValueError(f"{path}: unexpected dataset {dataset!r}")
        expected_version = (NO_TRACKING_VERSIONS[dataset]
                            if method == "no_tracking"
                            else CURRENT_LOCALIZATION_VERSIONS[dataset])
        expected_variant = "no_tracking" if method == "no_tracking" else "current"
        if inputs.get("version") != expected_version:
            raise ValueError(f"{path}: unexpected localization input version")
        if config.get("odometry_seed") != 0:
            raise ValueError(f"{path}: expected odometry seed 0")
        if config.get("smoother") != "fixed_interval":
            raise ValueError(f"{path}: expected fixed-interval smoothing")
        if bool(config.get("range_cap")) != (method != "no_range"):
            raise ValueError(f"{path}: range setting does not identify {method}")
        tables_name = Path(str(config.get("tables_override", ""))).name
        if tables_name != f"{dataset}.{expected_variant}.divided.json":
            raise ValueError(f"{path}: result does not use {expected_variant} tracks")
        summary = result.get("summary")
        if not isinstance(summary, dict):
            raise ValueError(f"{path}: summary must be an object")
        estimates = {}
        for radius_m in radii_m:
            field = f"{'sm_' if score == 'smoothed' else ''}dn_mass_{radius_m:g}"
            value = _finite_float(summary.get(field), field=field, path=path)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{path}: posterior mass must be in [0, 1]")
            estimates[radius_m] = Estimate(value, None, 1)
        if dataset in found:
            raise ValueError(f"duplicate grid-filter result for {dataset}")
        found[dataset] = estimates

    missing = expected_datasets - found.keys()
    if missing:
        raise ValueError(f"missing {method} results: {sorted(missing)}")
    return found


def load_loci_results(
    farfield_root: Path,
    versions: dict[str, str] = LOCI_VERSIONS,
    radii_m: Sequence[float] = DEFAULT_RADII_M,
    *,
    score: str = "causal",
) -> dict[str, dict[float, Estimate]]:
    if score not in SCORES:
        raise ValueError(f"unknown score {score!r}")
    results = {}
    for dataset, version in versions.items():
        run_dir = farfield_root / "artifacts" / "loci_runs" / dataset / version
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            results[dataset] = {r: Estimate(None, None, 0, 1) for r in radii_m}
            continue
        manifest = read_json_object(manifest_path)
        if manifest.get("complete") is False:
            results[dataset] = {r: Estimate(None, None, 0, 1) for r in radii_m}
            continue
        if (
            manifest.get("kind") != "loci_runs"
            or manifest.get("complete") is not True
            or manifest.get("dataset") != dataset
            or manifest.get("version") != version
        ):
            raise ValueError(f"{manifest_path}: unexpected LOCI run identity")
        evaluation = manifest.get("config", {}).get("evaluation", {})
        if evaluation.get("directions", [None])[0] != "forward":
            raise ValueError(f"{manifest_path}: path 0000000 is not declared forward")
        expected = dict(known_heading=False, n_heading=36,
                        observation_streams=["landmark"], odometry_seed=0,
                        smoother="none", reported_estimate="causal_filter")
        if any(evaluation.get(k) != v for k, v in expected.items()):
            raise ValueError(f"{manifest_path}: expected full-area unknown-heading landmark-only LOCI")
        result_path = run_dir / "0000000" / "raw_result.json"
        raw_result = read_json_object(result_path)
        if raw_result.get("schema") != "farfield_causal_grid/v1":
            raise ValueError(f"{result_path}: unexpected LOCI result schema")
        config = raw_result.get("config", {})
        if (config.get("observation_source") != "loci"
                or raw_result.get("localization_inputs", {}).get("dataset") != dataset
                or config.get("loci_landmark_only") is not True
                or config.get("n_heading") != 36
                or config.get("odometry_seed") != 0
                or config.get("smoother") != "none"
                or config.get("smooth_lag", 0) != 0
                or config.get("smooth_lags", "") != ""
                or raw_result.get("episode") is not None
                or raw_result.get("loci", {}).get("image_matrix") is not None
                or raw_result.get("loci", {}).get("satellite") is not None):
            raise ValueError(f"{result_path}: unexpected LOCI evaluation protocol")
        summary = raw_result.get("summary")
        if not isinstance(summary, dict):
            raise ValueError(f"{result_path}: summary must be an object")
        estimates = {}
        for radius_m in radii_m:
            field = f"{'sm_' if score == 'smoothed' else ''}dn_mass_{radius_m:g}"
            value = _finite_float(summary.get(field), field=field, path=result_path)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{result_path}: posterior mass must be in [0, 1]")
            estimates[radius_m] = Estimate(value, None, 1, 1)
        results[dataset] = estimates
    return results


def _canonical_path(value):
    if isinstance(value, str):
        for marker in ("/farfield_matching/", "/farfield_tracking_batch2_20260913/"):
            if marker in value:
                return "farfield/" + value.split(marker, 1)[1]
    return value


def summarize_subsections(found, expected, radii_m=DEFAULT_RADII_M):
    """Equal weights for seeds within windows, then windows within parent legs."""
    windows, parents = {}, {}
    for (dataset, index), seeds in sorted(expected.items()):
        by_seed = found.get((dataset, index), {})
        if not by_seed.keys() <= seeds:
            raise ValueError(f"unexpected seeds for {dataset}, subsection {index}")
        windows[dataset, index] = {}
        for radius in radii_m:
            values = [v[radius] for v in by_seed.values()]
            windows[dataset, index][radius] = Estimate(
                statistics.fmean(values) if values else None,
                None, len(values), len(seeds))
    for dataset in sorted({d for d, i in expected}):
        parents[dataset] = {}
        for radius in radii_m:
            cells = [v[radius] for (d, i), v in windows.items() if d == dataset]
            means = [v.mean for v in cells if v.mean is not None]
            parents[dataset][radius] = Estimate(
                statistics.fmean(means) if means else None,
                None, sum(v.count for v in cells), sum(v.expected_count for v in cells))
    return parents, windows


def load_sweep_results(root, method, radii_m=DEFAULT_RADII_M):
    """Read only completed, plan-bound causal outputs; missing jobs stay missing."""
    experiment, arm = PAPER_SWEEPS[method]
    directory = root / "runs" / experiment
    plan_path = directory / "plan.json"
    plan = read_json_object(plan_path)
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if plan.get("seeds") != list(PAPER_SEEDS):
        raise ValueError(f"{plan_path}: expected seeds {PAPER_SEEDS}")
    jobs, expected = {}, {}
    for entry in plan.get("pairs", plan.get("jobs", [])):
        dataset, index, seed = entry["dataset"], entry["index"], entry["seed"]
        if dataset not in SEQUENCE_DISPLAY_NAMES or type(seed) is not int or seed not in PAPER_SEEDS:
            raise ValueError(f"{plan_path}: unexpected job identity")
        config = (entry["config"] if method == "no_range" else entry["configs"][arm])
        name = f"{dataset}.{arm}.episode{index}.seed{seed}.causal"
        if name in jobs:
            raise ValueError(f"{plan_path}: duplicate job {name}")
        jobs[name] = entry, config
        expected.setdefault((dataset, index), set()).add(seed)
    if not jobs or any(seeds != set(PAPER_SEEDS) for seeds in expected.values()):
        raise ValueError(f"{plan_path}: incomplete planned seed roster")

    found, seen, truth_cache = {}, set(), {}
    for path in sorted((directory / "results").glob(f"*/*.{arm}.episode*.seed*.causal.json")):
        name = path.stem
        if name not in jobs or name in seen:
            raise ValueError(f"{path}: unexpected or duplicate result")
        seen.add(name)
        entry, config = jobs[name]
        result = read_json_object(path)
        actual = result.get("config", {})
        if any(_canonical_path(actual.get(k)) != _canonical_path(v) for k, v in config.items()):
            raise ValueError(f"{path}: configuration differs from pinned plan")
        if (result.get("schema") != "farfield_causal_grid/v1"
                or actual.get("smoother") != "none"
                or actual.get("smooth_lag") != 0 or actual.get("smooth_lags") != ""
                or actual.get("output_end") is not None or "smoothing" in result
                or actual.get("range_cap") != (0 if method == "no_range" else 1)):
            raise ValueError(f"{path}: expected causal-only {method} configuration")
        hybrid = method != "no_tracking"
        if (bool(actual.get("detection_input_dir")) != hybrid
                or (hybrid and actual.get("detection_audit_policy") != "replace")):
            raise ValueError(f"{path}: expected {'hybrid replacement' if hybrid else 'independent detections'}")
        reused = plan.get("reused", {}).get(name)
        if reused:
            if hashlib.sha256(path.read_bytes()).hexdigest() != reused["sha256"]:
                raise ValueError(f"{path}: reused result checksum mismatch")
        else:
            validation = result.get("ablation_validation" if method == "no_range" else "sweep_validation", {})
            if validation.get("plan_sha256") != plan_sha:
                raise ValueError(f"{path}: result is not validated against this plan")
        dataset, index, seed = entry["dataset"], entry["index"], entry["seed"]
        start, end = entry["bounds"]
        episode = result.get("episode", {})
        inputs = result.get("localization_inputs", {})
        if (inputs.get("dataset") != dataset
                or inputs.get("version") != Path(config["input_dir"]).name
                or episode.get("parent_dataset") != dataset
                or episode.get("index") != index
                or episode.get("parent_keyframe_start") != start
                or episode.get("parent_keyframe_end_inclusive") != end
                or episode.get("direction") != "forward"
                or episode.get("initialization") != "fresh_uniform_parent_region_and_fresh_imu_error"
                or result.get("odometry_profile", {}).get("noise", {}).get("base_seed") != seed):
            raise ValueError(f"{path}: wrong input, episode, or odometry identity")
        relative = _canonical_path(config["input_dir"])
        truth_path = root / relative.removeprefix("farfield/") / "truth.jsonl"
        if truth_path not in truth_cache:
            truth_cache[truth_path] = {
                r["keyframe_idx"]: r for r in map(json.loads, truth_path.read_text().splitlines())}
        truth = truth_cache[truth_path]
        steps = [math.hypot(truth[i+1]["east_m"]-truth[i]["east_m"],
                            truth[i+1]["north_m"]-truth[i]["north_m"])
                 for i in range(start, end)]
        if not sum(steps) > 0:
            raise ValueError(f"{path}: nonpositive trajectory length")
        values = {}
        for radius in radii_m:
            series = result.get("mass_by_keyframe", {}).get(f"{radius:g}", [])
            if (len(series) != end-start+1
                    or any(not math.isfinite(x) or not -1e-6 <= x <= 1+1e-6 for x in series)):
                raise ValueError(f"{path}: invalid causal mass series")
            value = _finite_float(result.get("summary", {}).get(f"dn_mass_{radius:g}"),
                                  field=f"dn_mass_{radius:g}", path=path)
            integrated = sum(d*(a+b)/2 for d,a,b in zip(steps,series,series[1:]))/sum(steps)
            if not 0 <= value <= 1 or abs(value-integrated) > 2e-5:
                raise ValueError(f"{path}: causal distance-mass integration mismatch")
            values[radius] = value
        found.setdefault((dataset, index), {})[seed] = values
    return summarize_subsections(found, expected, radii_m)


def load_default_results(
    farfield_root: Path = DEFAULT_FARFIELD_ROOT,
    *, twoarm_study: Path = PAPER_TWOARM_STUDY,
    loci_study: Path = PAPER_LOCI_STUDY,
    no_range_study: Path | None = None,
    crosslocate_study: Path | None = None,
    crosslocate_flevoland_study: Path | None = None,
) -> dict:
    """Only the paired seed-0 studies; never backfill with legacy evaluations."""
    plans = [read_json_object(root / "plan.json") for root in (twoarm_study, loci_study)]
    plan_hashes = [hashlib.sha256((root / "plan.json").read_bytes()).hexdigest()
                   for root in (twoarm_study, loci_study)]
    if plans[1].get("reference_plan_sha256") != plan_hashes[0]:
        raise ValueError("LOCI and two-arm reference plans differ")
    jobs = {}
    for pair in plans[0]["pairs"]:
        for method, arm in (("ours", "hybrid"), ("no_tracking", "no_tracking")):
            jobs[method, pair["dataset"], pair["scope"]] = (
                pair, pair["configs"][arm], twoarm_study,
                f"{pair['dataset']}.{arm}.{pair['scope']}.seed0.causal.json", plan_hashes[0])
    for job in plans[1]["jobs"]:
        key = ("loci", job["dataset"], job["scope"])
        if key in jobs:
            raise ValueError(f"duplicate planned job {key}")
        reference = jobs["ours", job["dataset"], job["scope"]][0]
        if job["bounds"] != reference["bounds"] or job["grid"] != reference["grid"]:
            raise ValueError(f"LOCI bounds/grid differ for {key}")
        jobs[key] = (job, job["config"], loci_study,
                    f"{job['dataset']}.{job['scope']}.seed0.causal.json", plan_hashes[1])
    active_methods = ["ours", "no_tracking", "loci"]
    if no_range_study is not None:
        no_range_plan = read_json_object(no_range_study / "plan.json")
        digest = hashlib.sha256((no_range_study / "plan.json").read_bytes()).hexdigest()
        if no_range_plan.get("reference_plan_sha256") != plan_hashes[0]:
            raise ValueError("no-range and two-arm reference plans differ")
        for pair in no_range_plan["pairs"]:
            key = ("no_range", pair["dataset"], pair["scope"])
            original, original_config, *_ = jobs["ours", pair["dataset"], pair["scope"]]
            if (key in jobs or pair["bounds"] != original["bounds"]
                    or pair["grid"] != original["grid"]
                    or pair["configs"]["hybrid"] != dict(original_config, range_cap=0)):
                raise ValueError(f"no-range changes more than range_cap: {key}")
            jobs[key] = (pair, pair["configs"]["hybrid"], no_range_study,
                         f"{pair['dataset']}.hybrid_no_range.{pair['scope']}.seed0.causal.json", digest)
        active_methods.append("no_range")
    scopes = {"full", *(f"window{i}" for i in range(1, 6))}
    expected = {(m, d, s) for m in active_methods
                for d in SEQUENCE_DISPLAY_NAMES for s in scopes}
    if jobs.keys() != expected:
        raise ValueError("expected 13 datasets, three methods, full plus five windows")
    for cross_root in (crosslocate_study, crosslocate_flevoland_study):
        if cross_root is None:
            continue
        plan_path = cross_root / "plan.json"
        cross_plan = read_json_object(plan_path)
        digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        if cross_plan.get("reference_plan_sha256") != plan_hashes[0]:
            raise ValueError("CrossLocate and two-arm reference plans differ")
        for config in cross_plan["jobs"]:
            if (config["retrieval_temperature"], config["retrieval_outlier_epsilon"]) != (0.05, 0.05):
                continue
            filename = Path(config["out"]).name
            dataset, variant, scope = filename.split(".")[:3]
            if variant not in ("top6", "all12"):
                raise ValueError(f"unexpected CrossLocate variant {variant}")
            if variant != "top6":
                continue
            key = (f"crosslocate_{variant}", dataset, scope)
            if key in jobs:
                raise ValueError(f"duplicate CrossLocate job {key}")
            reference = jobs["ours", dataset, scope][0]
            config = dict(config, out=str(Path(config["out"]).with_suffix(".partial.json")),
                          likelihood_cache_gb=0)
            jobs[key] = (reference, config, cross_root, filename, digest)
        if "crosslocate_top6" not in active_methods:
            active_methods.append("crosslocate_top6")
    found, noise_by_episode, truth_cache = {}, {}, {}
    for (method, dataset, scope), (job, config, root, filename, plan_sha) in jobs.items():
        paths = list((root / "results").rglob(filename))
        if len(paths) > 1:
            raise ValueError(f"duplicate result {filename}")
        if not paths:
            continue
        path = paths[0]
        result = read_json_object(path)
        actual = result.get("config", {})
        if (result.get("study_validation", {}).get("plan_sha256") != plan_sha
                or any(_canonical_path(actual.get(k)) != _canonical_path(v)
                       for k, v in config.items())):
            raise ValueError(f"{path}: configuration/validation differs from pinned plan")
        if (result.get("schema") != "farfield_causal_grid/v1"
                or actual.get("odometry_seed") != 0
                or actual.get("smoother") != "none"
                or actual.get("smooth_lag") != 0 or actual.get("smooth_lags") != ""
                or actual.get("output_end") is not None or "smoothing" in result
                or actual.get("cell_m") != 100 or actual.get("n_heading") != 36
                or actual.get("odometry_profile") != "epson_mg570_calibrated_planar_v1"):
            raise ValueError(f"{path}: expected paired causal seed-0 protocol")
        if method.startswith("crosslocate_"):
            if (actual.get("observation_source") != "crosslocate"
                    or Path(actual.get("retrieval_dir", "")).name != method.removeprefix("crosslocate_")
                    or result.get("availability", {}).get("track_inputs_used") is not False):
                raise ValueError(f"{path}: wrong CrossLocate observation variant")
        elif method == "loci":
            if (actual.get("observation_source") != "loci"
                    or result.get("loci", {}).get("aggregation", {}).get("streams") != ["landmark"]):
                raise ValueError(f"{path}: expected landmark-only LOCI")
        elif (actual.get("range_cap") != (0 if method == "no_range" else 1)
              or bool(actual.get("detection_input_dir")) != (method in ("ours", "no_range"))
              or (method in ("ours", "no_range") and actual.get("detection_audit_policy") != "replace")):
            raise ValueError(f"{path}: wrong bearing arm")
        if any(result.get("grid", {}).get(k) != v for k, v in job["grid"].items()):
            raise ValueError(f"{path}: prior grid differs")
        start, end = job["bounds"]
        episode = result.get("episode", {})
        if (result.get("localization_inputs", {}).get("dataset") != dataset
                or episode.get("count") != (1 if scope == "full" else 5)
                or episode.get("index") != config["episode_index"]
                or [episode.get("parent_keyframe_start"), episode.get("parent_keyframe_end_inclusive")] != [start, end]):
            raise ValueError(f"{path}: episode identity differs")
        odometry = result.get("odometry_profile")
        if noise_by_episode.setdefault((dataset, scope), odometry) != odometry:
            raise ValueError(f"{path}: paired odometry differs")
        relative = _canonical_path(config["input_dir"])
        truth_path = farfield_root / relative.removeprefix("farfield/") / "truth.jsonl"
        if truth_path not in truth_cache:
            truth_cache[truth_path] = list(map(json.loads, truth_path.read_text().splitlines()))
        truth = truth_cache[truth_path][start:end+1]
        steps = [math.hypot(b["east_m"]-a["east_m"], b["north_m"]-a["north_m"])
                 for a, b in zip(truth, truth[1:])]
        values = {}
        for radius in DEFAULT_RADII_M:
            series = result.get("mass_by_keyframe", {}).get(f"{radius:g}", [])
            if (len(series) != end-start+1 or not sum(steps) > 0
                    or any(not math.isfinite(v) or not -1e-6 <= v <= 1+1e-6 for v in series)):
                raise ValueError(f"{path}: invalid causal mass series")
            value = _finite_float(result.get("summary", {}).get(f"dn_mass_{radius:g}"),
                                  field=f"dn_mass_{radius:g}", path=path)
            integrated = sum(d*(a+b)/2 for d, a, b in zip(steps, series, series[1:])) / sum(steps)
            if not 0 <= value <= 1 or abs(value-integrated) > 2e-5:
                raise ValueError(f"{path}: distance-mass integration mismatch")
            values[radius] = value
        found[method, dataset, scope] = values
    output = {}
    for section, selected in (("full", ["full"]), ("windows", [f"window{i}" for i in range(1, 6)])):
        methods = {m: {} for m, _ in (*METHODS, *PAIRED_METHODS)}
        for method in active_methods:
            for dataset in SEQUENCE_DISPLAY_NAMES:
                estimates = {}
                for radius in DEFAULT_RADII_M:
                    values = [found[method, dataset, s][radius] for s in selected
                              if (method, dataset, s) in found]
                    if not any((method, dataset, s) in jobs for s in selected):
                        continue
                    estimates[radius] = Estimate(
                        statistics.fmean(values) if values else None,
                        statistics.stdev(values) if len(values) > 1 else None,
                        len(values), len(selected))
                methods[method][dataset] = estimates
        output[section] = {"causal": methods}
    return output


def _format_value(estimate: Estimate | None, *, bold: bool = False,
                  show_std: bool = False, percent: bool = False) -> str:
    if estimate is None:
        return "--"
    marker = "\\textsuperscript{*}" if estimate.incomplete else ""
    if estimate.mean is None:
        return "--" + marker
    mean = f"{100 * estimate.mean:.0f}" if percent else f"{estimate.mean:.2f}"
    if bold:
        mean = f"\\mathbf{{{mean}}}"
    spread = ""
    if show_std and estimate.std is not None:
        std = f"{100 * estimate.std:.0f}" if percent else f"{estimate.std:.2f}"
        spread = f" \\pm {std}"
    return f"${mean}{spread}$" + marker


def render_results_table(
    results: dict[str, dict[str, dict[str, dict[float, Estimate]]]],
    groups: Sequence[DatasetGroup] = DATASET_GROUPS,
    radii_m: Sequence[float] = DEFAULT_RADII_M,
    *, scope: str | None = None,
) -> str:
    if scope not in (None, "full", "windows", "both"):
        raise ValueError(f"unknown scope {scope!r}")
    scores = ("windows", "full") if scope == "both" else SCORES
    methods = PAIRED_METHODS if scope else METHODS
    if scope == "both":
        results = {s: results[s]["causal"] for s in scores}
    compared = ("loci", "no_tracking", "ours") if scope else tuple(PAPER_SWEEPS)
    if scope and any(results[score].get("no_range") for score in scores):
        compared = (*compared, "no_range")
    if scope:
        compared += tuple(m for m, _ in PAIRED_METHODS if m.startswith("crosslocate_")
                          and any(results[s].get(m) for s in scores))
    headers = [
        "Dataset",
        *(f"${radius_m:g}$"
          for _score in scores for _method in methods for radius_m in radii_m),
    ]
    sequences = tuple(
        sequence for group in groups for sequence in group.sequences)

    def format_result_row(label, get_estimate) -> list[str]:
        winners = {}
        for score in scores:
            for radius_m in radii_m:
                displayed = [
                    None if method not in compared
                    or (estimate := get_estimate(score, method, radius_m)) is None
                    or estimate.mean is None
                    or (scope is not None and estimate.incomplete)
                    else (round(100 * estimate.mean) if scope is not None
                          else f"{estimate.mean:.2f}")
                    for method, _ in methods
                ]
                best = max((value for value in displayed if value is not None),
                           default=None)
                winners[score, radius_m] = (
                    {i for i, value in enumerate(displayed) if value == best}
                    if best is not None
                    and sum(value is not None for value in displayed) == len(compared)
                    else set())
        cells = [label]
        for score in scores:
            for method_index, (method, _) in enumerate(methods):
                for radius_m in radii_m:
                    cells.append(_format_value(
                        get_estimate(score, method, radius_m),
                        bold=(method_index in winners[score, radius_m]),
                        percent=scope is not None,
                    ))
        return cells

    body = [
        format_result_row(
            SEQUENCE_DISPLAY_NAMES[sequence],
            lambda score, method, radius_m, sequence=sequence: results[
                score].get(method, {}).get(sequence, {}).get(radius_m),
        )
        for sequence in sequences
    ]

    def average(score, method, radius_m) -> Estimate | None:
        cells = [results[score].get(method, {}).get(sequence, {}).get(radius_m)
                 for sequence in sequences]
        values = [cell.mean for cell in cells if cell is not None and cell.mean is not None]
        # Average parent-leg means, not all runs: longer/more subdivided legs
        # and faster-finishing seeds must not silently get extra weight.
        complete = sum(cell is not None and cell.mean is not None
                       and not cell.incomplete for cell in cells)
        return Estimate(statistics.fmean(values) if values else None,
                        None, complete, len(sequences))

    average_row = format_result_row("Average", average)
    widths = [
        max(len(row[i]) for row in [headers, *body, average_row])
        for i in range(len(headers))
    ]

    def format_row(cells: Sequence[str]) -> str:
        return " & ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " \\\\"

    score_headers = (
        f" & \\multicolumn{{{2 * len(radii_m)}}}{{c}}{{\\textit{{Full-leg baselines}}}}"
        f" & \\multicolumn{{{3 * len(radii_m)}}}{{c}}{{\\textit{{Fresh-window evaluations}}}} \\\\")
    score_cmidrules = (
        f"\\cmidrule(lr){{2-{1 + 2 * len(radii_m)}}} "
        f"\\cmidrule(lr){{{2 + 2 * len(radii_m)}-{1 + 5 * len(radii_m)}}}")
    if scope is not None:
        labels = (["Five-window mean", "Full trajectory"] if scope == "both" else
                  ["Full trajectory" if scope == "full" else "Five-window mean"])
        block = len(methods) * len(radii_m)
        score_headers = "".join(
            f" & \\multicolumn{{{block}}}{{c}}{{\\textit{{{label}}}}}" for label in labels) + " \\\\"
        score_cmidrules = " ".join(
            f"\\cmidrule(lr){{{2 + i * block}-{1 + (i + 1) * block}}}" for i in range(len(labels)))
    method_headers = " & ".join(
        [""] + [
            f"\\multicolumn{{{len(radii_m)}}}{{c}}{{{header}}}"
            for _score in scores for _, header in methods
        ]) + " \\\\"
    method_cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + i * len(radii_m)}-{1 + (i + 1) * len(radii_m)}}}"
        for i in range(len(scores) * len(methods)))
    lines = [
        "% Sources: CrossLocate 260901_dem_baseline_parity;",
        f"% LOCI {PAPER_LOCI_VERSION};",
        *(f"% {method}: {experiment}, arm={arm}, seeds=0..4;"
          for method, (experiment, arm) in PAPER_SWEEPS.items()),
        *(f"% Coverage {score} {method} {sequence}: "
          f"{estimate.count}/{estimate.expected_count if estimate.expected_count is not None else estimate.count} runs"
          for score in scores for method, _ in methods for sequence in sequences
          if (estimate := results[score].get(method, {}).get(sequence, {}).get(radii_m[0])) is not None),
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Distance-normalized posterior-mass score "
        "$\\overline P_R(\\tau)$ ($\\uparrow$) for "
        f"$R\\in\\{{{','.join(f'{radius_m:g}' for radius_m in radii_m)}\\}}$~m. "
        "All scores are causal, $p(x_t\\mid z_{0:t})$, and are fractions in $[0,1]$. "
        "For our hybrid method and its ablations, available seeds (0--4) are "
        "averaged within each subsection, then subsection means are averaged "
        "with equal weight within each parent leg. Subsections reset localization "
        "and IMU error; overlapping windows are not independent sequences. "
        "LOCI~\\cite{fahnestockandfuentes2026loci} uses full-area combined catalogs, "
        "landmark-only observations, unknown heading, and one seed-0 full-leg run. "
        "CrossLocate~\\cite{tomevsek2022crosslocate} is available "
        "only for Mount Washington (two full-leg seeds) and has provisional score calibration. "
        "The full-leg baselines are not reset-subsection evaluations and should "
        "not be interpreted as paired comparisons with the window columns. The "
        "no-tracking ablation treats detections independently; the no-range-bin "
        "ablation disables range gating. The average is the unweighted mean over "
        "available sequence rows. Bold marks the best window-method means "
        "in each row and radius, including ties at the displayed precision. "
        "Dashes indicate unavailable runs. An asterisk marks incomplete planned "
        "seed/subsection coverage, or incomplete sequence coverage in the average; "
        "missing runs are not replaced by zeros.}",
        "  \\label{tab:farfield-results}",
        "  \\scriptsize",
        "  \\setlength{\\tabcolsep}{1.0pt}",
        "  \\resizebox{\\textwidth}{!}{%",
        f"  \\begin{{tabular}}{{l{'c' * (len(scores) * len(methods) * len(radii_m))}}}",
        "  \\toprule",
        "  " + score_headers,
        "  " + score_cmidrules,
        "  " + method_headers,
        "  " + method_cmidrules,
        "  " + format_row(headers),
        "  \\midrule",
    ]
    lines.extend("  " + format_row(row) for row in body)
    lines.extend(["  \\midrule", "  " + format_row(average_row)])
    lines.extend([
        "  \\bottomrule", "  \\end{tabular}%", "  }", "\\end{table*}"
    ])
    if scope is not None:
        lines = [line for line in lines if not line.startswith("% Sources:")
                 and not line.startswith("% LOCI ")
                 and not any(line.startswith(f"% {m}:") for m in PAPER_SWEEPS)]
        description = ("Each cell is one full-trajectory seed-0 evaluation. "
                       if scope == "full" else
                       "Window columns report the equal-weight mean "
                       "across five approximately equal-length, evenly spaced, overlapping "
                       "windows within each leg, all at seed 0. Overlapping windows are "
                       "not independent trials. ")
        if scope == "both":
            description += "The right-hand columns report separate full-trajectory seed-0 runs. "
        for i, line in enumerate(lines):
            if line.startswith("  \\caption{"):
                lines[i] = (
                    "  \\caption{Distance-normalized posterior-mass scores "
                    "$\\overline P_R(\\tau)$ ($\\uparrow$), for $R\\in\\{100,500\\}$~m. "
                    "Scores are multiplied by 100 and rounded "
                    "to whole percentage points; percent signs are omitted. "
                    "All scores are causal, $p(x_t\\mid z_{0:t})$, on the same parent grid "
                    "with paired Epson IMU noise and unknown initial heading. " + description +
                    "LOCI~\\cite{fahnestockandfuentes2026loci} uses landmark-only observations. "
                    "Ours uses detection-first audited-track replacement; no tracking uses "
                    "independent detections. Both retain range bins. The no-range ablation "
                    "is identical to ours except that range-bin gating is disabled. "
                    "CrossLocate uses top6 at temperature 0.05 and outlier probability 0.05 "
                    "for every recording and scope. This configuration was chosen after a "
                    "nine-combination likelihood sweep on these evaluation recordings, not on held-out data. "
                    "Flevoland uses the same configuration with render CRS EPSG:28992. "
                    "Portland CrossLocate results are unavailable. "
                    "The average is an unweighted mean of available parent-leg means, not "
                    "a paired overall comparison while coverage differs. Bold marks the "
                    "best means, including ties at the displayed precision, only when all included methods are complete. "
                    "An asterisk marks incomplete coverage; missing results are never zero-filled.}")
            elif line.startswith("  \\label{"):
                lines[i] = f"  \\label{{tab:farfield-results-{scope}}}"
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--farfield-root", type=Path, default=DEFAULT_FARFIELD_ROOT,
        help=f"Far-field data root (default: {DEFAULT_FARFIELD_ROOT})")
    parser.add_argument("--output", type=Path,
                        help="Write LaTeX to this file instead of stdout")
    parser.add_argument("--twoarm-study", type=Path, default=PAPER_TWOARM_STUDY)
    parser.add_argument("--loci-study", type=Path, default=PAPER_LOCI_STUDY)
    parser.add_argument("--no-range-study", type=Path, default=PAPER_NO_RANGE_STUDY)
    parser.add_argument("--crosslocate-study", type=Path, default=PAPER_CROSSLOCATE_STUDY)
    parser.add_argument("--crosslocate-flevoland-study", type=Path, default=PAPER_CROSSLOCATE_FLEVOLAND_STUDY)
    parser.add_argument("--scope", choices=("full", "windows", "both"), default="both")
    args = parser.parse_args(argv)
    results = load_default_results(args.farfield_root, twoarm_study=args.twoarm_study,
                                   loci_study=args.loci_study, no_range_study=args.no_range_study,
                                   crosslocate_study=args.crosslocate_study,
                                   crosslocate_flevoland_study=args.crosslocate_flevoland_study)
    provenance = (f"% Two-arm study: {args.twoarm_study}\n"
                  f"% LOCI study: {args.loci_study}\n"
                  f"% No-range study: {args.no_range_study}\n"
                  f"% CrossLocate study: {args.crosslocate_study}; variant=top6, temperature=0.05, epsilon=0.05\n"
                  f"% CrossLocate Flevoland study: {args.crosslocate_flevoland_study}; EPSG:28992\n")
    selected = results if args.scope == "both" else results[args.scope]
    emit_table(provenance + render_results_table(selected, scope=args.scope), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
