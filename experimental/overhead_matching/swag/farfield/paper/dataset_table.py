"""Generate the far-field paper's LaTeX dataset-statistics table.

This is the far-field counterpart to the LOCI-era LaTeX emitter in
``dataset_statistics.py``. Numerical values come from frozen dataset and
catalog manifests; only short editorial descriptions live here.
"""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from experimental.overhead_matching.swag.farfield.paper.table_common import (
    DATASET_GROUPS,
    DEFAULT_FARFIELD_ROOT,
    SEQUENCE_ARTIFACTS,
    DatasetGroup,
    emit_table,
    read_json_object,
    region_area_km2,
)


@dataclass(frozen=True)
class DatasetStatistics:
    group: DatasetGroup
    panoramas_per_leg: tuple[int, ...]
    video_minutes_per_leg: tuple[float, ...]
    trajectory_km_per_leg: tuple[float, ...]
    accepted_tracks_per_leg: tuple[int, ...]
    map_landmarks: int
    area_km2: float
    capture_date: str


def _required_positive_int(value: object, *, field: str, path: Path) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{path}: {field} must be a positive integer")
    return value


def _required_positive_float(value: object, *, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{path}: {field} must be a positive number")
    return float(value)


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


def _artifact_config(path: Path, kind: str, sequence: str) -> tuple[dict, dict]:
    manifest = read_json_object(path)
    if manifest.get("schema") != "farfield.artifact.v1":
        raise ValueError(f"{path}: unexpected artifact schema")
    if manifest.get("kind") != kind or manifest.get("complete") is not True:
        raise ValueError(f"{path}: expected a complete {kind} artifact")
    if manifest.get("dataset") != sequence:
        raise ValueError(f"{path}: dataset does not match {sequence!r}")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{path}: config must be an object")
    return manifest, config


def _video_minutes(dataset_path: Path, metadata: dict) -> float:
    path = dataset_path / "frames_gps.csv"
    try:
        with path.open(newline="") as stream:
            timestamps = [float(row["video_t_s"]) for row in csv.DictReader(stream)]
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: video_t_s must be numeric") from exc
    if (len(timestamps) < 2 or any(not math.isfinite(value) for value in timestamps)
            or any(end <= start for start, end in zip(timestamps, timestamps[1:]))):
        raise ValueError(f"{path}: retained frame timestamps must increase")

    trims = metadata.get("trims") or []
    if not any(trim.get("trim_kind") == "range" for trim in trims):
        return (timestamps[-1] - timestamps[0]) / 60.0

    log_path = dataset_path / "extraction_log.csv"
    try:
        with log_path.open(newline="") as stream:
            positions = [
                int(row["sequence_position"]) for row in csv.DictReader(stream)
            ]
    except OSError as exc:
        raise ValueError(f"Could not read {log_path}: {exc}") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{log_path}: sequence_position must be an integer") from exc
    if len(positions) != len(timestamps) or any(
            end <= start for start, end in zip(positions, positions[1:])):
        raise ValueError(f"{log_path}: positions must align and increase")
    duration_s = sum(
        end_t - start_t
        for start_t, end_t, start_pos, end_pos in zip(
            timestamps, timestamps[1:], positions, positions[1:])
        if end_pos == start_pos + 1
    )
    if duration_s <= 0:
        raise ValueError(f"{dataset_path}: retained video duration is empty")
    return duration_s / 60.0


def collect_dataset_statistics(
    farfield_root: Path,
    catalog_version: str | None = None,
    groups: Sequence[DatasetGroup] = DATASET_GROUPS,
) -> list[DatasetStatistics]:
    """Load and aggregate the paper datasets.

    Multi-sequence locations retain per-leg panorama, trajectory, and accepted
    track counts. Their catalog manifests must identify the same content.
    """
    rows = []
    for group in groups:
        panoramas_per_leg = []
        video_minutes_per_leg = []
        trajectory_km_per_leg = []
        accepted_tracks_per_leg = []
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
            panoramas_per_leg.append(_required_positive_int(
                metadata.get("num_images"), field="num_images", path=metadata_path))
            video_minutes_per_leg.append(_video_minutes(metadata_path.parent, metadata))
            trajectory_km_per_leg.append(_required_positive_float(
                metadata.get("trajectory_km"), field="trajectory_km", path=metadata_path))
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
            manifest, config = _artifact_config(catalog_path, "catalogs", sequence)
            digest = manifest.get("content_digest")
            if not isinstance(digest, str) or not digest:
                raise ValueError(f"{catalog_path}: missing content_digest")
            catalog_digests.add(digest)
            catalog_counts.add(
                _required_positive_int(
                    config.get("rows_out"), field="config.rows_out", path=catalog_path
                )
            )
            area = region_area_km2(config, group.region_policy)
            if area is None or not math.isfinite(area) or area <= 0.0:
                raise ValueError(f"{catalog_path}: catalog region has invalid area")
            catalog_areas.add(area)

            bearings_version = SEQUENCE_ARTIFACTS[sequence]["bearing_observations"]
            if bearings_version is None:
                raise ValueError(f"{sequence}: bearing observations are not pinned")
            bearings_path = (
                farfield_root / "artifacts" / "bearing_observations"
                / sequence / bearings_version / "manifest.json"
            )
            _, bearings_config = _artifact_config(
                bearings_path, "bearing_observations", sequence)
            accepted_tracks_per_leg.append(_required_positive_int(
                bearings_config.get("n_accepted_tracklets"),
                field="config.n_accepted_tracklets",
                path=bearings_path,
            ))

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
        if (len(catalog_digests) != 1 or len(catalog_counts) != 1
                or len(catalog_areas) != 1):
            raise ValueError(
                f"{group.display_name}: sequences do not share one catalog artifact"
            )

        rows.append(
            DatasetStatistics(
                group=group,
                panoramas_per_leg=tuple(panoramas_per_leg),
                video_minutes_per_leg=tuple(video_minutes_per_leg),
                trajectory_km_per_leg=tuple(trajectory_km_per_leg),
                accepted_tracks_per_leg=tuple(accepted_tracks_per_leg),
                map_landmarks=next(iter(catalog_counts)),
                area_km2=next(iter(catalog_areas)),
                capture_date=next(iter(capture_dates)),
            )
        )
    return rows


def render_dataset_table(rows: Sequence[DatasetStatistics]) -> str:
    """Render dataset statistics as a booktabs-compatible LaTeX table."""
    markers = {
        "pohang": "\\textsuperscript{*}",
        "flevoland": "\\textsuperscript{\\textdagger}",
    }
    headers = [
        "Dataset",
        "Setting",
        "Panos",
        "\\shortstack{Video\\\\(min)}",
        "Traj. (km)",
        "\\shortstack{Accepted\\\\tracks}",
        "\\shortstack{MSM\\\\landmarks}",
        "\\shortstack{Overhead\\\\Area (km$^2$)}",
        "\\shortstack{MSM\\\\sources}",
        "\\shortstack{Capture\\\\Date}",
    ]
    body = [
        [
            row.group.display_name + markers.get(row.group.key, ""),
            row.group.conditions,
            "/".join(f"{value:,}" for value in row.panoramas_per_leg),
            "/".join(f"{value:.0f}" for value in row.video_minutes_per_leg),
            "/".join(f"{value:.1f}" for value in row.trajectory_km_per_leg),
            "/".join(f"{value:,}" for value in row.accepted_tracks_per_leg),
            f"{row.map_landmarks:,}",
            f"{row.area_km2:,.0f}",
            row.group.map_source.replace(" / ", "/"),
            row.capture_date[2:7],
        ]
        for row in rows
    ]
    widths = [max(len(row[i]) for row in [headers, *body]) for i in range(len(headers))]

    def format_row(cells: Sequence[str]) -> str:
        return " & ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " \\\\"

    lines = [
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Released evaluation dataset statistics. Slash-separated "
        "values report individual legs. Accepted tracks are landmark tracks "
        "retained after semantic audit, and MSM landmark counts are from the "
        "trimmed maps. \\textsuperscript{*} identifies "
        "data from the Pohang Canal "
        "Dataset~\\cite{chung2023pohang}; \\textsuperscript{$\\dagger$} identifies "
        "imagery from the Mapillary platform~\\cite{MapillaryPlatform}.}",
        "  \\label{tab:farfield-datasets}",
        "  \\scriptsize",
        "  \\setlength{\\tabcolsep}{0.5pt}",
        "  \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrrrrrcc@{}}",
        "  \\toprule",
        "  " + format_row(headers),
        "  \\midrule",
    ]
    lines.extend("  " + format_row(row) for row in body)
    lines.extend(
        [
            "  \\bottomrule",
            "  \\end{tabular*}",
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
