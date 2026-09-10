"""Generate the far-field paper's LaTeX results table from pinned runs."""

import argparse
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
    FULL_METHOD_RUN_SPECS,
    DatasetGroup,
    emit_table,
    glob_runs,
    read_json_object,
)


DEFAULT_RADII_M = (100.0, 500.0)
METHODS = (
    ("crosslocate", "CrossLocate"),
    ("loci", "LOCI"),
    ("no_tracking", "\\shortstack{No\\\\tracking}"),
    ("no_range", "\\shortstack{No range\\\\bins}"),
    ("ours", "Ours"),
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
    "boston_snowy": "Boston Snowy",
}

# LOCI's corrected float64-truth runs for the original sequences are v3. The
# later datasets were produced after that correction and are v1.
LOCI_VERSIONS = {
    **{
        sequence: "paper_sigmas_full_leg_mass100_500_v3"
        for sequence in (
            "mount_washington_20260815_leg1",
            "mount_washington_20260815_leg2",
            "mount_washington_20260815_leg3",
            "charles_river_20260727",
            "boston_harbor_leg1",
            "boston_harbor_leg2",
            "boston_harbor_leg3",
        )
    },
    **{
        sequence: "paper_sigmas_full_leg_mass100_500_v1"
        for sequence in ("pohang_canal_04", "flevoland_polder", "boston_snowy")
    },
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
    mean: float
    std: float | None
    count: int


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


def load_loci_results(
    farfield_root: Path,
    versions: dict[str, str] = LOCI_VERSIONS,
    radii_m: Sequence[float] = DEFAULT_RADII_M,
) -> dict[str, dict[float, Estimate]]:
    results = {}
    for dataset, version in versions.items():
        run_dir = farfield_root / "artifacts" / "loci_runs" / dataset / version
        manifest_path = run_dir / "manifest.json"
        manifest = read_json_object(manifest_path)
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
        values, trajectory_length_m = _load_metrics(run_dir / "0000000", radii_m)
        results[dataset] = _aggregate(
            [
                SequenceResult(
                    dataset,
                    None,
                    values,
                    trajectory_length_m,
                    run_dir / "0000000",
                )
            ],
            radii_m,
        )
    return results


def load_default_results(
    farfield_root: Path = DEFAULT_FARFIELD_ROOT,
) -> dict[str, dict[str, dict[float, Estimate]]]:
    runs = farfield_root / "runs"
    original = {
        "mount_washington_20260815_leg1",
        "mount_washington_20260815_leg2",
        "mount_washington_20260815_leg3",
        "charles_river_20260727",
        "boston_harbor_leg1",
        "boston_harbor_leg2",
        "boston_harbor_leg3",
    }
    all_datasets = set(SEQUENCE_DISPLAY_NAMES)

    crosslocate = glob_runs((
        (runs / "260901_dem_baseline_parity", "leg[123]_baseline_parity_s[01]"),
    ))
    no_tracking = glob_runs((
        (runs / "260903_no_tracking_ablation", "*_notrack_seed[0-3]--tracks-*"),
    ))
    no_range = glob_runs((
        (runs / "260828_imu_baseline", "mount_washington_*_imu_seed[0-3]--tracks-*"),
        (runs / "260828_imu_baseline", "boston_harbor_*_imu_seed[0-3]--tracks-*"),
        (runs / "260828_imu_baseline", "charles_river_*_imu_seed[0-3]--tracks-*"),
        (runs / "260902_pohang_matching", "pohang_canal_04_baseline_seed[0-3]--tracks-*"),
        (runs / "260903_boston_flevoland_osmv2", "boston_snowy_*_seed[01]--tracks-*"),
        (runs / "260904_range_cap", "boston_snowy_osmv2_base_seed[23]--tracks-*"),
        (runs / "260903_boston_flevoland_osmv2", "flevoland_polder_*_pro_v1_seed[01]--tracks-*"),
        (runs / "260904_range_cap", "flevoland_polder_pro_base_seed[23]--tracks-*"),
    ))
    ours = glob_runs(tuple(
        (runs / directory, pattern)
        for directory, pattern in FULL_METHOD_RUN_SPECS
    ))
    return {
        "crosslocate": load_localization_results(
            crosslocate, {name for name in original if name.startswith("mount_")},
            {0, 1}, method="crosslocate"),
        "loci": load_loci_results(farfield_root),
        "no_tracking": load_localization_results(
            no_tracking, original, {0, 1, 2, 3}, method="no_tracking"),
        "no_range": load_localization_results(
            no_range, all_datasets, {0, 1, 2, 3}, method="no_range"),
        "ours": load_localization_results(
            ours, all_datasets, {0, 1, 2, 3}, method="ours"),
    }


def _format_value(estimate: Estimate | None, *, bold: bool = False) -> str:
    if estimate is None:
        return "--"
    mean = f"{estimate.mean:.2f}"
    if bold:
        mean = f"\\mathbf{{{mean}}}"
    return f"${mean} \\pm \\mathrm{{N/A}}$"


def render_results_table(
    results: dict[str, dict[str, dict[float, Estimate]]],
    groups: Sequence[DatasetGroup] = DATASET_GROUPS,
    radii_m: Sequence[float] = DEFAULT_RADII_M,
) -> str:
    method_headers = [header for _, header in METHODS]
    headers = ["Dataset", *(method_headers * len(radii_m))]
    body = []
    for group in groups:
        for sequence in group.sequences:
            cells = [SEQUENCE_DISPLAY_NAMES[sequence]]
            for radius_m in radii_m:
                estimates = [
                    results[method].get(sequence, {}).get(radius_m)
                    for method, _ in METHODS
                ]
                best = max(
                    (f"{estimate.mean:.2f}" for estimate in estimates
                     if estimate is not None),
                    default=None,
                )
                cells.extend(
                    _format_value(
                        estimate,
                        bold=(estimate is not None
                              and f"{estimate.mean:.2f}" == best),
                    )
                    for estimate in estimates)
            body.append(cells)
    widths = [max(len(row[i]) for row in [headers, *body]) for i in range(len(headers))]

    def format_row(cells: Sequence[str]) -> str:
        return " & ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " \\\\"

    metric_headers = " & ".join(
        [""] + [
            f"\\multicolumn{{{len(METHODS)}}}{{c}}"
            f"{{$\\overline P_{{{radius_m:g}}}(\\tau)$}}"
            for radius_m in radii_m
        ]) + " \\\\"
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + i * len(METHODS)}-{1 + (i + 1) * len(METHODS)}}}"
        for i in range(len(radii_m)))
    lines = [
        "% Sources: CrossLocate 260901_dem_baseline_parity; LOCI loci_runs/*/paper_sigmas_*;",
        "% no tracking 260903_no_tracking_ablation; no range 260828/260902/260903/260904;",
        "% ours 260902_pohang_matching and 260904_range_cap.",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Distance-normalized posterior-mass score "
        "$\\overline P_R(\\tau)$ ($\\uparrow$) for "
        f"$R\\in\\{{{','.join(f'{radius_m:g}' for radius_m in radii_m)}\\}}$~m. "
        "Our method and ablations average four filter seeds, CrossLocate "
        "averages two, and LOCI~\\cite{fahnestockandfuentes2026loci} is a single "
        "fixed-seed forward pass. These runs do not provide sufficient repeated "
        "trials for an uncertainty estimate, so all entries report $\\pm$ N/A. "
        "CrossLocate~\\cite{tomevsek2022crosslocate} is available "
        "only for Mount Washington and has provisional score calibration. The "
        "no-tracking ablation omits range-bin gating and is paired with the "
        "no-range-bin condition. Bold marks the best reported score in each "
        "dataset row and radius; ties at the displayed precision are all bold. "
        "Dashes indicate unavailable runs.}",
        "  \\label{tab:farfield-results}",
        "  \\scriptsize",
        "  \\setlength{\\tabcolsep}{1.2pt}",
        f"  \\begin{{tabular}}{{l{'c' * (len(METHODS) * len(radii_m))}}}",
        "  \\toprule",
        "  " + metric_headers,
        "  " + cmidrules,
        "  " + format_row(headers),
        "  \\midrule",
    ]
    lines.extend("  " + format_row(row) for row in body)
    lines.extend(["  \\bottomrule", "  \\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--farfield-root", type=Path, default=DEFAULT_FARFIELD_ROOT,
        help=f"Far-field data root (default: {DEFAULT_FARFIELD_ROOT})")
    parser.add_argument("--output", type=Path,
                        help="Write LaTeX to this file instead of stdout")
    args = parser.parse_args(argv)
    emit_table(render_results_table(load_default_results(args.farfield_root)), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
