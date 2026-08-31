"""Shared roster and small helpers for the far-field paper tables."""

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_FARFIELD_ROOT = Path("/data/farfield_matching")


@dataclass(frozen=True)
class DatasetGroup:
    """One paper row, possibly backed by several recorded sequences."""

    key: str
    display_name: str
    conditions: str
    map_source: str
    sequences: tuple[str, ...]


# Editorial fields are intentionally kept next to the paper roster. The
# numerical fields in both tables are loaded from the data root.
DATASET_GROUPS = (
    DatasetGroup(
        key="washington",
        display_name="Mt. Washington",
        conditions="Mountain trail",
        map_source="OSM",
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
        sequences=("pohang_canal_04",),
    ),
    DatasetGroup(
        key="charles",
        display_name="Charles River",
        conditions="Urban river",
        map_source="OSM + ENC",
        sequences=("charles_river_20260727",),
    ),
    DatasetGroup(
        key="boston_harbor",
        display_name="Boston Harbor",
        conditions="Coastal harbor",
        map_source="OSM + ENC",
        sequences=(
            "boston_harbor_leg1",
            "boston_harbor_leg2",
            "boston_harbor_leg3",
        ),
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


def emit_table(table: str, output: Path | None) -> None:
    """Print a table or write it to an explicitly requested path."""
    if output is None:
        print(table)
    else:
        output.write_text(table + "\n")
