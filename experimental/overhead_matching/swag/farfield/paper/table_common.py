"""The far-field paper roster: which sequences, regions, artifacts and runs.

This file is the single source of truth for every input the paper reports.
`check_roster` walks the data root and reports where the artifacts on disk
disagree with it; the dataset and results tables read their pins from here.
Change a pin here first, then regenerate; never the other way round.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_FARFIELD_ROOT = Path("/data/farfield_matching")

# Region policy for a dataset group. The uniform prior, the catalog trim, and
# every baseline must use the one box this names.
#   area625     trim_catalog clip plan: trajectory-union bbox padded
#               symmetrically to at least 625 km^2 (the reviewed policy)
#   fetch_bbox  no clip; the region is the full catalog's fetch bbox
REGION_POLICIES = ("area625", "fetch_bbox")

# Per-sequence artifact lanes, in pipeline order. `catalogs` is pinned on the
# group because legs share one catalog.
SEQUENCE_LANES = (
    "pinhole_images",
    "frame_landmarks",
    "object_tracks",
    "semantic_audits",
    "bearing_observations",
    "landmark_matches",
    "alignment_diagnostics",
    "localization_inputs",
)


@dataclass(frozen=True)
class DatasetGroup:
    """One paper row, possibly backed by several recorded sequences."""

    key: str
    display_name: str
    conditions: str
    map_source: str
    # `collection/active_catalogs.py` scope that fetched the full catalog.
    catalog_scope: str
    region_policy: str
    catalog_version: str | None
    sequences: tuple[str, ...]
    # (runs/<experiment>, glob) of the complete four-seed full-method runs the
    # tables report, or None while the group has no reportable runs yet.
    run_spec: tuple[str, str] | None

    def __post_init__(self) -> None:
        if self.region_policy not in REGION_POLICIES:
            raise ValueError(
                f"{self.key}: unknown region policy {self.region_policy!r}")


# Editorial fields are intentionally kept next to the paper roster. The
# numerical fields in both tables are loaded from the data root.
DATASET_GROUPS = (
    DatasetGroup(
        key="washington",
        display_name="Mt. Washington",
        conditions="Mountain trail",
        map_source="OSM",
        catalog_scope="mount_washington_20260815",
        region_policy="fetch_bbox",
        catalog_version="trim_20260910_v1",
        sequences=(
            "mount_washington_20260815_leg1",
            "mount_washington_20260815_leg2",
            "mount_washington_20260815_leg3",
        ),
        run_spec=("260904_range_cap", "mount_washington_*_rangecap_seed[0-3]--tracks-*"),
    ),
    DatasetGroup(
        key="pohang",
        display_name="Pohang",
        conditions="Urban canal",
        map_source="OSM",
        catalog_scope="pohang_canal_04",
        region_policy="fetch_bbox",
        catalog_version="trim_20260910_v1",
        sequences=("pohang_canal_04",),
        run_spec=("260902_pohang_matching", "pohang_canal_04_baseline_rangecap_seed[0-3]--tracks-*"),
    ),
    DatasetGroup(
        key="flevoland",
        display_name="Flevoland",
        conditions="Rural polder",
        map_source="OSM",
        catalog_scope="flevoland_polder_20250111",
        region_policy="area625",
        catalog_version="trim625_20260910_v1",
        sequences=("flevoland_polder",),
        run_spec=("260904_range_cap", "flevoland_polder_pro_rangecap_seed[0-3]--tracks-*"),
    ),
    DatasetGroup(
        key="charles",
        display_name="Charles River",
        conditions="Urban river",
        map_source="OSM + ENC",
        catalog_scope="charles_river_20260727",
        region_policy="area625",
        catalog_version="trim625_20260910_v1",
        sequences=("charles_river_20260727",),
        run_spec=("260904_range_cap", "charles_river_*_rangecap_seed[0-3]--tracks-*"),
    ),
    DatasetGroup(
        key="boston_harbor",
        display_name="Boston Harbor",
        conditions="Coastal harbor",
        map_source="OSM + ENC",
        catalog_scope="boston_harbor_20260712",
        region_policy="area625",
        catalog_version="stage3_a6e45b9_trim625_v1",
        sequences=(
            "boston_harbor_leg1",
            "boston_harbor_leg2",
            "boston_harbor_leg3",
        ),
        run_spec=("260904_range_cap", "boston_harbor_*_rangecap_seed[0-3]--tracks-*"),
    ),
    DatasetGroup(
        key="franconia",
        display_name="Franconia",
        conditions="Mountain road",
        map_source="OSM",
        catalog_scope="franconia_20260829",
        region_policy="area625",
        catalog_version="trim625_20260910_v1",
        sequences=("franconia_leg1",),
        # Four range-cap seeds exist, but no baseline runs do yet.
        run_spec=None,
    ),
    DatasetGroup(
        key="portland",
        display_name="Portland",
        conditions="Light aircraft",
        map_source="OSM + ENC + FAA",
        catalog_scope="portland_flight_20260906",
        region_policy="fetch_bbox",
        catalog_version="trim_osmfaa_20260910_v1",
        sequences=(
            "portland_flight_20260906_leg1",
            "portland_flight_20260906_leg2",
            "portland_flight_20260906_leg3",
        ),
        run_spec=None,
    ),
)

# Groups the tables can report today: every seed of every method exists.
TABLE_GROUPS = tuple(group for group in DATASET_GROUPS if group.run_spec)

# Complete four-seed runs used for the paper's full-method column. Keep the
# dataset table and results table on the same pinned evaluation artifacts.
FULL_METHOD_RUN_SPECS = tuple(group.run_spec for group in TABLE_GROUPS)


def _chain(pinhole, landmarks, tracks, audits, bearings, matches, inputs):
    return dict(zip(SEQUENCE_LANES, (
        pinhole, landmarks, tracks, audits, bearings, matches, bearings, inputs)))


# The lane versions each sequence's reported runs were built from, read off
# the pinned runs' manifests on 2026-09-10. Alignment diagnostics share the
# bearings version by convention. None = not produced yet.
_STAGE3_HARBOR = _chain(
    "stage3_7b88e81_adopted_v2", "stage3_7b88e81_adopted_v2",
    "stage3_7b88e81_regen_v2", "stage3_17c8031_regen_v8_machine",
    "stage3_c5e2220_rangecap_v1", "stage3_17c8031_regen_v8_machine",
    "stage3_c5e2220_imu_rangecap_v1")
_UNBUILT = dict.fromkeys(SEQUENCE_LANES)

SEQUENCE_ARTIFACTS: dict[str, dict[str, str | None]] = {
    "mount_washington_20260815_leg1": _STAGE3_HARBOR,
    "mount_washington_20260815_leg2": _STAGE3_HARBOR,
    "mount_washington_20260815_leg3": _STAGE3_HARBOR,
    "boston_harbor_leg1": _STAGE3_HARBOR,
    "boston_harbor_leg2": _STAGE3_HARBOR,
    "boston_harbor_leg3": _STAGE3_HARBOR,
    "charles_river_20260727": _chain(
        "stage3_7b88e81_adopted_v2", "stage3_7b88e81_adopted_v2",
        "stage3_7b88e81_regen_v2", "stage3_17c8031_regen_v8_machine",
        "stage3_acd4216_rangecap_v1", "stage3_17c8031_regen_v8_machine",
        "stage3_acd4216_imu_rangecap_v1"),
    "pohang_canal_04": _chain(
        "stage3_7b88e81_adopted_v2", "stage3_7b88e81_adopted_v2",
        "stage3_7b88e81_regen_v2", "stage3_17c8031_regen_v8_machine",
        "stage3_c5e2220_rangecap_v1", "stage3_b847f55_baseline_v1",
        "stage3_c5e2220_baseline_rangecap_v1"),
    "flevoland_polder": _chain(
        "stage3_b847f55_osmv2_v1", "stage3_b847f55_osmv2_v1",
        "stage3_b847f55_osmv2_v1", "stage3_b847f55_osmv2_v1",
        "stage3_acd4216_rangecap_v1", "stage3_b847f55_osmv2_pro_v1",
        "stage3_acd4216_osmv2_pro_rangecap_v1"),
    "franconia_leg1": _chain(
        "stage3_b847f55_v1", "stage3_b847f55_v1",
        "stage3_b847f55_v1", "stage3_b847f55_v1",
        "stage3_acd4216_rangecap_v1", "stage3_b847f55_v1",
        "stage3_acd4216_imu_rangecap_v1"),
    "portland_flight_20260906_leg1": _UNBUILT,
    "portland_flight_20260906_leg2": _UNBUILT,
    "portland_flight_20260906_leg3": _UNBUILT,
}

# The one model/prompt every dataset must use per LLM stage. None means the
# standard is not decided yet; the checker then only reports disagreement.
LLM_STANDARD: dict[str, str | None] = {
    "extraction.model": None,
    "extraction.prompt_type": None,
    "audit.model": None,
    "matching.model": None,
}


def all_sequences(groups: Sequence[DatasetGroup] = DATASET_GROUPS) -> tuple[str, ...]:
    return tuple(sequence for group in groups for sequence in group.sequences)


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
