"""Generate the far-field paper's LaTeX dataset-statistics table.

This is the far-field counterpart to the LOCI-era LaTeX emitter in
``dataset_statistics.py``. Numerical values come from frozen dataset metadata
and a pinned catalog version; only short editorial descriptions live here.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from experimental.overhead_matching.swag.farfield.paper.table_common import (
    DATASET_GROUPS,
    DEFAULT_FARFIELD_ROOT,
    DatasetGroup,
    emit_table,
    read_json_object,
)


DEFAULT_CATALOG_VERSION = "stage3_7b88e81_trim_v1"


@dataclass(frozen=True)
class DatasetStatistics:
    group: DatasetGroup
    num_panoramas: int
    trajectory_km: float
    map_landmarks: int
    osm_bbox_area_km2: float
    capture_date: str


def _required_positive_int(value: object, *, field: str, path: Path) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{path}: {field} must be a positive integer")
    return value


def _required_positive_float(value: object, *, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{path}: {field} must be a positive number")
    return float(value)


def _osm_bbox_area_km2(
    manifest: dict, *, sequence: str, catalog_path: Path
) -> float:
    """Read the mapped OSM area from a trimmed catalog's exact full parent."""
    upstreams = manifest.get("upstreams")
    if not isinstance(upstreams, list):
        raise ValueError(f"{catalog_path}: upstreams must be a list")
    catalog_upstreams = []
    for upstream in upstreams:
        if not isinstance(upstream, dict):
            raise ValueError(f"{catalog_path}: each upstream must be an object")
        if upstream.get("kind") == "catalogs":
            catalog_upstreams.append(upstream)
    if len(catalog_upstreams) != 1:
        raise ValueError(
            f"{catalog_path}: expected exactly one catalogs upstream, got "
            f"{len(catalog_upstreams)}"
        )

    upstream = catalog_upstreams[0]
    if upstream.get("dataset") != sequence:
        raise ValueError(f"{catalog_path}: catalogs upstream crosses datasets")
    upstream_dir = upstream.get("path")
    if not isinstance(upstream_dir, str) or not upstream_dir:
        raise ValueError(f"{catalog_path}: catalogs upstream path must be a string")
    full_path = Path(upstream_dir) / "manifest.json"
    full_manifest = read_json_object(full_path)
    if (
        full_manifest.get("schema") != "farfield.artifact.v1"
        or full_manifest.get("kind") != "catalogs"
        or full_manifest.get("complete") is not True
    ):
        raise ValueError(f"{full_path}: expected a complete catalogs artifact")
    if full_manifest.get("dataset") != sequence:
        raise ValueError(f"{full_path}: dataset does not match {sequence!r}")
    if full_manifest.get("version") != upstream.get("version"):
        raise ValueError(f"{full_path}: version does not match its upstream reference")
    if full_manifest.get("content_digest") != upstream.get("content_digest"):
        raise ValueError(
            f"{full_path}: content digest does not match its upstream reference"
        )

    config = full_manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{full_path}: config must be an object")
    coverage = config.get("source_coverage")
    if not isinstance(coverage, dict):
        raise ValueError(f"{full_path}: config.source_coverage must be an object")
    if coverage.get("schema") != "farfield_catalog_source_coverage/v2":
        raise ValueError(f"{full_path}: unexpected source-coverage schema")
    if coverage.get("status") != "passed":
        raise ValueError(f"{full_path}: source coverage did not pass")
    details = coverage.get("details")
    if not isinstance(details, list):
        raise ValueError(f"{full_path}: source-coverage details must be a list")
    areas = []
    for detail in details:
        if not isinstance(detail, dict):
            raise ValueError(
                f"{full_path}: each source-coverage detail must be an object"
            )
        if "mapped_area_km2" in detail:
            areas.append(
                _required_positive_float(
                    detail["mapped_area_km2"],
                    field="config.source_coverage.details[].mapped_area_km2",
                    path=full_path,
                )
            )
    if len(areas) != 1:
        raise ValueError(
            f"{full_path}: expected exactly one mapped_area_km2 value, got "
            f"{len(areas)}"
        )
    return areas[0]


def collect_dataset_statistics(
    farfield_root: Path,
    catalog_version: str = DEFAULT_CATALOG_VERSION,
    groups: Sequence[DatasetGroup] = DATASET_GROUPS,
) -> list[DatasetStatistics]:
    """Load and aggregate the four paper datasets.

    Multi-sequence locations sum panorama counts and trajectory lengths. Their
    catalog manifests must identify the same content, so a shared map catalog
    is counted once rather than once per leg.
    """
    rows = []
    for group in groups:
        num_panoramas = 0
        trajectory_km = 0.0
        capture_dates = set()
        resolutions = set()
        catalog_digests = set()
        catalog_counts = set()
        catalog_areas = set()

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
            capture_date = metadata.get("capture_date")
            if not isinstance(capture_date, str) or not capture_date:
                raise ValueError(f"{metadata_path}: capture_date must be a string")
            capture_dates.add(capture_date)
            resolution = metadata.get("resolution")
            if not isinstance(resolution, str) or not resolution:
                raise ValueError(f"{metadata_path}: resolution must be a string")
            resolutions.add(resolution)

            catalog_path = (
                farfield_root
                / "artifacts"
                / "catalogs"
                / sequence
                / catalog_version
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
            catalog_areas.add(
                _osm_bbox_area_km2(
                    manifest, sequence=sequence, catalog_path=catalog_path
                )
            )

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
        if (
            len(catalog_digests) != 1
            or len(catalog_counts) != 1
            or len(catalog_areas) != 1
        ):
            raise ValueError(
                f"{group.display_name}: sequences do not share one catalog artifact"
            )

        rows.append(
            DatasetStatistics(
                group=group,
                num_panoramas=num_panoramas,
                trajectory_km=trajectory_km,
                map_landmarks=next(iter(catalog_counts)),
                osm_bbox_area_km2=next(iter(catalog_areas)),
                capture_date=next(iter(capture_dates)),
            )
        )
    return rows


def render_dataset_table(rows: Sequence[DatasetStatistics]) -> str:
    """Render dataset statistics as a booktabs-compatible LaTeX table."""
    headers = [
        "Dataset",
        "Conditions",
        "\\# Seq. / Panos",
        "Traj. (km)",
        "Map landmarks",
        "OSM bbox area (km$^2$)",
        "Map",
        "Capture",
    ]
    body = [
        [
            row.group.display_name,
            row.group.conditions,
            f"{len(row.group.sequences)} / {row.num_panoramas:,}",
            f"{row.trajectory_km:.1f}",
            f"{row.map_landmarks:,}",
            f"{row.osm_bbox_area_km2:,.1f}",
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
        "report totals across sequences; map landmarks are counted once for "
        "the shared catalog. OSM bounding-box area is the manifest-reported "
        "mappable area within the catalog bounding box.}",
        "  \\label{tab:farfield-datasets}",
        "  \\small",
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
        default=DEFAULT_CATALOG_VERSION,
        help=f"Pinned catalog artifact version (default: {DEFAULT_CATALOG_VERSION})",
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
