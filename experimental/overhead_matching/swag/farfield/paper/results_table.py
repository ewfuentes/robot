"""Generate the far-field paper's mocked LaTeX results table.

The ``Ours`` column is loaded from one pinned experiment and one seed.
CrossLocate and LOCI are deliberate placeholders until their evaluation artifacts exist;
the uncertainty magnitude is likewise left blank until multiple seeds are
aggregated.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from experimental.overhead_matching.swag.farfield.localization import (
    metrics as metrics_lib,
)
from experimental.overhead_matching.swag.farfield.paper.table_common import (
    DATASET_GROUPS,
    DEFAULT_FARFIELD_ROOT,
    DatasetGroup,
    emit_table,
    read_json_object,
)


# Newest complete standard-run experiment with seed 0 for every paper sequence
# at the time this table was introduced. Keep this pinned for reproducibility;
# callers can point at a newer experiment explicitly with --experiment-dir.
DEFAULT_EXPERIMENT_DIR = DEFAULT_FARFIELD_ROOT / "runs" / "260828_imu_baseline"
DEFAULT_SEED = 0
DEFAULT_RADII_M = (100.0, 500.0)

SEQUENCE_DISPLAY_NAMES = {
    "mount_washington_20260815_leg1": "Mt. Washington, leg 1",
    "mount_washington_20260815_leg2": "Mt. Washington, leg 2",
    "mount_washington_20260815_leg3": "Mt. Washington, leg 3",
    "pohang_canal_04": "Pohang",
    "charles_river_20260727": "Charles River",
    "boston_harbor_leg1": "Boston Harbor, leg 1",
    "boston_harbor_leg2": "Boston Harbor, leg 2",
    "boston_harbor_leg3": "Boston Harbor, leg 3",
}


@dataclass(frozen=True)
class SequenceResult:
    dataset: str
    values: dict[float, float]
    trajectory_length_m: float
    run_dir: Path


def _localization_seed(manifest: dict, path: Path) -> int:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{path}: config must be an object")
    localization = config.get("localization")
    if not isinstance(localization, dict):
        raise ValueError(f"{path}: config.localization must be an object")
    seed = localization.get("seed")
    if type(seed) is not int:
        raise ValueError(f"{path}: config.localization.seed must be an integer")
    return seed


def _finite_float(value: object, *, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path}: {field} must be finite")
    return result


def _load_metrics(
    run_dir: Path, radii_m: Sequence[float]
) -> tuple[dict[float, float], float]:
    """Read the run's recorded distance-normalized posterior-mass scores.

    The metric is defined once, in ``localization/metrics.py``, and recorded
    by the run itself; this reads it rather than recomputing so the paper and
    the run artifacts cannot disagree.
    """
    metrics_path = run_dir / "metrics.json"
    metrics = read_json_object(metrics_path)
    if metrics.get("schema") != metrics_lib.POSITION_MASS_SUMMARY_SCHEMA:
        raise ValueError(
            f"{metrics_path}: expected "
            f"{metrics_lib.POSITION_MASS_SUMMARY_SCHEMA}, got "
            f"{metrics.get('schema')!r}. Runs recorded before the metric "
            "became distance-normalized must be re-scored.")
    if metrics.get("higher_is_better") is not True:
        raise ValueError(f"{metrics_path}: expected a higher-is-better metric")
    if metrics.get("reference_position") != "truth":
        raise ValueError(f"{metrics_path}: expected truth-referenced position mass")
    if metrics.get("normalization") != \
            metrics_lib.POSITION_MASS_DISTANCE_NORMALIZATION:
        raise ValueError(
            f"{metrics_path}: expected distance normalization, got "
            f"{metrics.get('normalization')!r}")
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
        if radius_key not in radii:
            raise ValueError(f"{metrics_path}: missing radius {radius_key} m")
        radius_entry = radii[radius_key]
        if not isinstance(radius_entry, dict):
            raise ValueError(
                f"{metrics_path}: radius {radius_key} entry must be an object"
            )
        recorded_radius = _finite_float(
            radius_entry.get("radius_m"), field="radii[].radius_m", path=metrics_path
        )
        if recorded_radius != radius_m:
            raise ValueError(
                f"{metrics_path}: radius {radius_key} entry records "
                f"{recorded_radius:g} m"
            )
        value = _finite_float(
            radius_entry.get("distance_normalized_mass"),
            field="radii[].distance_normalized_mass", path=metrics_path)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{metrics_path}: posterior mass must be in [0, 1]")
        values[radius_m] = value
    return values, trajectory_length_m


def load_sequence_results(
    experiment_dir: Path,
    seed: int = DEFAULT_SEED,
    radii_m: Sequence[float] = DEFAULT_RADII_M,
    groups: Sequence[DatasetGroup] = DATASET_GROUPS,
) -> dict[str, SequenceResult]:
    """Load exactly one complete run for every requested sequence."""
    expected = {sequence for group in groups for sequence in group.sequences}
    found = {}

    try:
        candidates = sorted(experiment_dir.iterdir())
    except OSError as exc:
        raise ValueError(f"Could not inspect experiment directory {experiment_dir}: {exc}") from exc

    for run_dir in candidates:
        if not run_dir.is_dir() or run_dir.name.endswith(".viewer"):
            continue
        manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "metrics.json"
        if not manifest_path.is_file() or not metrics_path.is_file():
            continue
        manifest = read_json_object(manifest_path)
        if manifest.get("kind") != "localization_run":
            continue
        dataset = manifest.get("dataset")
        if dataset not in expected:
            continue
        if (
            manifest.get("complete") is not True
            or _localization_seed(manifest, manifest_path) != seed
        ):
            continue
        if dataset in found:
            raise ValueError(
                f"{experiment_dir}: multiple complete seed-{seed} runs for {dataset}: "
                f"{found[dataset].run_dir.name}, {run_dir.name}"
            )
        values, trajectory_length_m = _load_metrics(run_dir, radii_m)
        found[dataset] = SequenceResult(
            dataset=dataset,
            values=values,
            trajectory_length_m=trajectory_length_m,
            run_dir=run_dir,
        )

    missing = sorted(expected - found.keys())
    if missing:
        raise ValueError(
            f"{experiment_dir}: missing complete seed-{seed} results for {missing}"
        )
    return found


def load_ours_results(
    experiment_dir: Path,
    seed: int = DEFAULT_SEED,
    radii_m: Sequence[float] = DEFAULT_RADII_M,
    groups: Sequence[DatasetGroup] = DATASET_GROUPS,
) -> list[SequenceResult]:
    sequence_results = load_sequence_results(
        experiment_dir, seed=seed, radii_m=radii_m, groups=groups
    )
    return [
        sequence_results[sequence]
        for group in groups
        for sequence in group.sequences
    ]


def _format_value(value: float) -> str:
    # The empty right-hand side is intentional scaffolding for multi-seed CI.
    return f"${value:.3f} \\pm {{}}$"


def render_results_table(
    rows: Sequence[SequenceResult],
    experiment_dir: Path,
    seed: int = DEFAULT_SEED,
    radii_m: Sequence[float] = DEFAULT_RADII_M,
) -> str:
    """Render results with deliberate baseline and uncertainty placeholders."""
    method_headers = [
        "CrossLocate~\\cite{tomevsek2022crosslocate}",
        "LOCI~\\cite{fahnestockandfuentes2026loci}",
        "\\shortstack{Ours without\\\\tracking}",
        "Ours",
    ]
    headers = ["Dataset", *(method_headers * len(radii_m))]
    body = []
    for row in rows:
        cells = [SEQUENCE_DISPLAY_NAMES[row.dataset]]
        for radius_m in radii_m:
            cells.extend(["--", "--", "--", _format_value(row.values[radius_m])])
        body.append(cells)
    widths = [
        max(len(row[i]) for row in [headers, *body])
        for i in range(len(headers))
    ]

    def format_row(cells: Sequence[str]) -> str:
        return " & ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " \\\\"

    metric_headers = " & ".join(
        [""]
        + [
            f"\\multicolumn{{{len(method_headers)}}}{{c}}"
            f"{{$\\overline P_{{{radius_m:g}}}(\\tau)$}}"
            for radius_m in radii_m
        ]
    ) + " \\\\"
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + i * len(method_headers)}-"
        f"{1 + (i + 1) * len(method_headers)}}}"
        for i in range(len(radii_m))
    )
    lines = [
        f"% Ours source: {experiment_dir}",
        f"% Ours seed: {seed}",
        "% Each row is one recorded sequence; legs are not aggregated.",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Distance-normalized posterior-mass score "
        "$\\overline P_R(\\tau)$ ($\\uparrow$) for "
        f"$R\\in\\{{{','.join(f'{radius_m:g}' for radius_m in radii_m)}\\}}$~m. "
        "CrossLocate, LOCI, and the no-tracking "
        "ablation are placeholders; uncertainty magnitudes are intentionally "
        "blank until multi-seed aggregation.}",
        "  \\label{tab:farfield-results}",
        "  \\small",
        f"  \\begin{{tabular}}{{l{'c' * (len(method_headers) * len(radii_m))}}}",
        "  \\toprule",
        "  " + metric_headers,
        "  " + cmidrules,
        "  " + format_row(headers),
        "  \\midrule",
    ]
    lines.extend("  " + format_row(row) for row in body)
    lines.extend(
        [
            "  \\bottomrule",
            "  \\end{tabular}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the far-field paper results table"
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help=f"Pinned experiment directory (default: {DEFAULT_EXPERIMENT_DIR})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write LaTeX to this file instead of stdout",
    )
    args = parser.parse_args(argv)

    rows = load_ours_results(args.experiment_dir, args.seed)
    emit_table(
        render_results_table(rows, args.experiment_dir, args.seed),
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
