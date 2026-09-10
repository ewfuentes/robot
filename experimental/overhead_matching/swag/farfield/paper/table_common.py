"""Shared roster and small helpers for the far-field paper tables."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_FARFIELD_ROOT = Path("/data/farfield_matching")

# Complete four-seed runs used for the paper's full-method column. Keep the
# dataset table and results table on the same pinned evaluation artifacts.
FULL_METHOD_RUN_SPECS = (
    (
        "260904_range_cap",
        "mount_washington_*_rangecap_seed[0-3]--tracks-*",
    ),
    ("260904_range_cap", "boston_harbor_*_rangecap_seed[0-3]--tracks-*"),
    ("260904_range_cap", "charles_river_*_rangecap_seed[0-3]--tracks-*"),
    ("260904_range_cap", "boston_snowy_rangecap_seed[0-3]--tracks-*"),
    (
        "260904_range_cap",
        "flevoland_polder_pro_rangecap_seed[0-3]--tracks-*",
    ),
    (
        "260902_pohang_matching",
        "pohang_canal_04_baseline_rangecap_seed[0-3]--tracks-*",
    ),
)


@dataclass(frozen=True)
class DatasetGroup:
    """One paper row, possibly backed by several recorded sequences."""

    key: str
    display_name: str
    conditions: str
    map_source: str
    catalog_version: str
    sequences: tuple[str, ...]


# Editorial fields are intentionally kept next to the paper roster. The
# numerical fields in both tables are loaded from the data root.
DATASET_GROUPS = (
    DatasetGroup(
        key="washington",
        display_name="Mt. Washington",
        conditions="Mountain trail",
        map_source="OSM",
        catalog_version="stage3_7b88e81_trim_v1",
        sequences=(
            "mount_washington_20260815_leg1",
            "mount_washington_20260815_leg2",
            "mount_washington_20260815_leg3",
        ),
    ),
    DatasetGroup(
        key="pohang",
        display_name="Pohang",
        conditions="Urban canal",
        map_source="OSM",
        catalog_version="stage3_b847f55_trim_v1",
        sequences=("pohang_canal_04",),
    ),
    DatasetGroup(
        key="flevoland",
        display_name="Flevoland",
        conditions="Rural polder",
        map_source="OSM",
        catalog_version="osm_20260903_trim625_v1",
        sequences=("flevoland_polder",),
    ),
    DatasetGroup(
        key="charles",
        display_name="Charles River",
        conditions="Urban river",
        map_source="OSM + ENC",
        catalog_version="stage3_7b88e81_trim_v1",
        sequences=("charles_river_20260727",),
    ),
    DatasetGroup(
        key="boston_harbor",
        display_name="Boston Harbor",
        conditions="Coastal harbor",
        map_source="OSM + ENC",
        catalog_version="stage3_a6e45b9_trim625_v1",
        sequences=(
            "boston_harbor_leg1",
            "boston_harbor_leg2",
            "boston_harbor_leg3",
        ),
    ),
    DatasetGroup(
        key="boston_snowy",
        display_name="Boston Snowy",
        conditions="Snowy urban road",
        map_source="OSM",
        catalog_version="osm_20260903_trim625_v1",
        sequences=("boston_snowy",),
    ),
)


def read_json_object(path: Path) -> dict:
    """Read a JSON object with a path-bearing error for malformed inputs."""
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(value).__name__}")
    return value


def glob_runs(specs: Sequence[tuple[Path, str]]) -> list[Path]:
    """Resolve pinned run globs, requiring the files consumed by the tables."""
    runs = []
    for directory, pattern in specs:
        matches = sorted(
            path
            for path in directory.glob(pattern)
            if (path / "manifest.json").is_file()
            and (path / "metrics.json").is_file()
        )
        if not matches:
            raise ValueError(f"no complete-looking runs match {directory / pattern}")
        runs.extend(matches)
    return runs


def emit_table(table: str, output: Path | None) -> None:
    """Print a table or write it to an explicitly requested path."""
    if output is None:
        print(table)
    else:
        output.write_text(table + "\n")
