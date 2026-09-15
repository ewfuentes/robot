"""Generate the far-field paper's LaTeX dataset-statistics table.

This is the far-field counterpart to the LOCI-era LaTeX emitter in
``dataset_statistics.py``. Numerical values come from frozen dataset, catalog,
and localization-run manifests; only short editorial descriptions live here.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from experimental.overhead_matching.swag.farfield.paper.table_common import (
    TABLE_GROUPS,
    DEFAULT_FARFIELD_ROOT,
    FULL_METHOD_RUN_SPECS,
    DatasetGroup,
    emit_table,
    glob_runs,
    read_json_object,
)


@dataclass(frozen=True)
class DatasetStatistics:
    group: DatasetGroup
    num_panoramas: int
    trajectory_km: float
    map_landmarks: int
    prior_areas_km2: tuple[float, ...]
    capture_date: str


def _required_positive_int(value: object, *, field: str, path: Path) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{path}: {field} must be a positive integer")
    return value


def _required_positive_float(value: object, *, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{path}: {field} must be a positive number")
    return float(value)


def _load_prior_areas(
    run_dirs: Sequence[Path],
    expected_datasets: set[str],
    expected_seeds: frozenset[int] = frozenset(range(4)),
) -> dict[str, float]:
    """Read the uniform position-prior support from the reported full runs."""
    found: dict[str, dict[int, float]] = {}
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        manifest = read_json_object(manifest_path)
        if manifest.get("kind") != "localization_run" or manifest.get("complete") is not True:
            raise ValueError(f"{manifest_path}: expected a complete localization run")
        dataset = manifest.get("dataset")
        if dataset not in expected_datasets:
            continue
        config = manifest.get("config")
        contract = config.get("localization_run_contract") if isinstance(config, dict) else None
        if (
            not isinstance(contract, dict)
            or contract.get("run_kind") != "evaluation"
            or contract.get("ablation_tags") != []
        ):
            raise ValueError(f"{manifest_path}: expected an evaluation run contract")
        filter_config = contract.get("filter_config")
        if (
            not isinstance(filter_config, dict)
            or filter_config.get("range_cap_enabled") is not True
        ):
            raise ValueError(f"{manifest_path}: expected the full-method filter")
        seed = filter_config.get("seed")
        if type(seed) is not int or seed not in expected_seeds:
            raise ValueError(f"{manifest_path}: unexpected filter seed {seed!r}")
        init = filter_config.get("init")
        if not isinstance(init, dict) or init.get("kind") != "UniformBoxInit":
            raise ValueError(f"{manifest_path}: expected a uniform box prior")
        bounds = []
        for field in ("east_min_m", "east_max_m", "north_min_m", "north_max_m"):
            value = init.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{manifest_path}: init.{field} must be numeric")
            bounds.append(float(value))
        east_min, east_max, north_min, north_max = bounds
        area_km2 = (east_max - east_min) * (north_max - north_min) / 1e6
        if not math.isfinite(area_km2) or area_km2 <= 0.0:
            raise ValueError(f"{manifest_path}: uniform prior has invalid bounds")
        by_seed = found.setdefault(dataset, {})
        if seed in by_seed:
            raise ValueError(f"{manifest_path}: duplicate seed {seed} for {dataset}")
        by_seed[seed] = area_km2

    missing = {
        (dataset, seed)
        for dataset in expected_datasets
        for seed in expected_seeds
        if seed not in found.get(dataset, {})
    }
    if missing:
        raise ValueError(f"missing full-method runs: {sorted(missing)}")

    areas = {}
    for dataset, by_seed in found.items():
        unique = set(by_seed.values())
        if len(unique) != 1:
            raise ValueError(f"{dataset}: full-method seeds use different priors")
        areas[dataset] = next(iter(unique))
    return areas


def _capture_date(metadata: dict, metadata_path: Path) -> str:
    capture_date = metadata.get("capture_date")
    if isinstance(capture_date, str) and capture_date:
        return capture_date
    video = metadata.get("video")
    sync = video.get("sync") if isinstance(video, dict) else None
    timestamp = sync.get("source_video_start_utc") if isinstance(sync, dict) else None
    if isinstance(timestamp, str) and len(timestamp) >= 10:
        return timestamp[:10]
    raise ValueError(f"{metadata_path}: capture date is not recorded")


def collect_dataset_statistics(
    farfield_root: Path,
    catalog_version: str | None = None,
    groups: Sequence[DatasetGroup] = TABLE_GROUPS,
    localization_run_dirs: Sequence[Path] | None = None,
) -> list[DatasetStatistics]:
    """Load and aggregate the paper datasets.

    Multi-sequence locations sum panorama counts and trajectory lengths. Their
    catalog manifests must identify the same content, so a shared map catalog
    is counted once rather than once per leg.
    """
    expected_datasets = {
        sequence for group in groups for sequence in group.sequences
    }
    if localization_run_dirs is None:
        runs_root = farfield_root / "runs"
        localization_run_dirs = glob_runs(tuple(
            (runs_root / directory, pattern)
            for directory, pattern in FULL_METHOD_RUN_SPECS
        ))
    prior_areas = _load_prior_areas(localization_run_dirs, expected_datasets)

    rows = []
    for group in groups:
        num_panoramas = 0
        trajectory_km = 0.0
        capture_dates = set()
        resolutions = set()
        catalog_digests = set()
        catalog_counts = set()
        group_prior_areas = []

        for sequence in group.sequences:
            metadata_path = (
                farfield_root / "datasets" / sequence / "pipeline_metadata.json"
            )
            metadata = read_json_object(metadata_path)
            if metadata.get("dataset_name") != sequence:
                raise ValueError(
                    f"{metadata_path}: dataset_name must be {sequence!r}, got "
                    f"{metadata.get('dataset_name')!r}"
                )
            num_panoramas += _required_positive_int(
                metadata.get("num_images"), field="num_images", path=metadata_path
            )
            trajectory_km += _required_positive_float(
                metadata.get("trajectory_km"),
                field="trajectory_km",
                path=metadata_path,
            )
            capture_dates.add(_capture_date(metadata, metadata_path))
            resolution = metadata.get("resolution")
            if not isinstance(resolution, str) or not resolution:
                raise ValueError(f"{metadata_path}: resolution must be a string")
            resolutions.add(resolution)

            catalog_path = (
                farfield_root
                / "artifacts"
                / "catalogs"
                / sequence
                / (catalog_version or group.catalog_version)
                / "manifest.json"
            )
            manifest = read_json_object(catalog_path)
            if manifest.get("schema") != "farfield.artifact.v1":
                raise ValueError(f"{catalog_path}: unexpected artifact schema")
            if manifest.get("kind") != "catalogs" or manifest.get("complete") is not True:
                raise ValueError(f"{catalog_path}: expected a complete catalogs artifact")
            if manifest.get("dataset") != sequence:
                raise ValueError(f"{catalog_path}: dataset does not match {sequence!r}")
            digest = manifest.get("content_digest")
            if not isinstance(digest, str) or not digest:
                raise ValueError(f"{catalog_path}: missing content_digest")
            catalog_digests.add(digest)
            config = manifest.get("config")
            if not isinstance(config, dict):
                raise ValueError(f"{catalog_path}: config must be an object")
            catalog_counts.add(
                _required_positive_int(
                    config.get("rows_out"), field="config.rows_out", path=catalog_path
                )
            )
            group_prior_areas.append(prior_areas[sequence])

        if len(capture_dates) != 1:
            raise ValueError(
                f"{group.display_name}: sequence capture dates disagree: "
                f"{sorted(capture_dates)}"
            )
        if len(resolutions) != 1:
            raise ValueError(
                f"{group.display_name}: sequence resolutions disagree: "
                f"{sorted(resolutions)}"
            )
        if len(catalog_digests) != 1 or len(catalog_counts) != 1:
            raise ValueError(
                f"{group.display_name}: sequences do not share one catalog artifact"
            )

        rows.append(
            DatasetStatistics(
                group=group,
                num_panoramas=num_panoramas,
                trajectory_km=trajectory_km,
                map_landmarks=next(iter(catalog_counts)),
                prior_areas_km2=tuple(group_prior_areas),
                capture_date=next(iter(capture_dates)),
            )
        )
    return rows


def render_dataset_table(rows: Sequence[DatasetStatistics]) -> str:
    """Render dataset statistics as a booktabs-compatible LaTeX table."""
    def format_area(areas: Sequence[float]) -> str:
        return f"{max(areas):,.0f}"

    headers = [
        "Dataset",
        "Conditions",
        "\\# Seq. / Panos",
        "Traj. (km)",
        "\\# landmarks",
        "Area (km$^2$)",
        "MSM Sources",
        "Capture",
    ]
    body = [
        [
            row.group.display_name,
            row.group.conditions,
            f"{len(row.group.sequences)} / {row.num_panoramas:,}",
            f"{row.trajectory_km:.1f}",
            f"{row.map_landmarks:,}",
            format_area(row.prior_areas_km2),
            row.group.map_source,
            row.capture_date,
        ]
        for row in rows
    ]
    widths = [max(len(row[i]) for row in [headers, *body]) for i in range(len(headers))]

    def format_row(cells: Sequence[str]) -> str:
        return " & ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " \\\\"

    lines = [
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Far-field dataset statistics. Multi-sequence locations "
        "report totals across sequences; landmarks are counted once for a "
        "shared MSM. Area reports the largest rectangular support used for "
        "uniform particle initialization among grouped sequences, rounded to "
        "the nearest km$^2$. Baseline supports differ as described in the text.}",
        "  \\label{tab:farfield-datasets}",
        "  \\small",
        "  \\setlength{\\tabcolsep}{3.5pt}",
        "  \\begin{tabular}{llrrrrcc}",
        "  \\toprule",
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
        description="Generate the far-field paper dataset table"
    )
    parser.add_argument(
        "--farfield-root",
        type=Path,
        default=DEFAULT_FARFIELD_ROOT,
        help=f"Far-field data root (default: {DEFAULT_FARFIELD_ROOT})",
    )
    parser.add_argument(
        "--catalog-version",
        help="Override the pinned catalog version for every dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write LaTeX to this file instead of stdout",
    )
    args = parser.parse_args(argv)

    rows = collect_dataset_statistics(args.farfield_root, args.catalog_version)
    emit_table(render_dataset_table(rows), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
