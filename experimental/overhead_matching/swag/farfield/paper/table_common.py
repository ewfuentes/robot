"""The far-field paper roster: which sequences, regions, artifacts and runs.

This file is the single source of truth for every input the paper reports.
`check_roster` walks the data root and reports where the artifacts on disk
disagree with it; the dataset and results tables read their pins from here.
Change a pin here first, then regenerate; never the other way round.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_FARFIELD_ROOT = Path("/data/farfield_matching")

# Frozen seed-0 studies: five distance windows plus a separate full trajectory.
PAPER_EVALUATIONS_ROOT = DEFAULT_FARFIELD_ROOT / "runs/paper_evaluations_20260915"
PAPER_TWOARM_STUDY = PAPER_EVALUATIONS_ROOT / "farfield_twoarms_20260914"
PAPER_LOCI_STUDY = PAPER_EVALUATIONS_ROOT / "farfield_loci_20260915"
PAPER_NO_RANGE_STUDY = PAPER_EVALUATIONS_ROOT / "farfield_norange_20260915"
PAPER_CROSSLOCATE_STUDY = PAPER_EVALUATIONS_ROOT / "farfield_crosslocate_20260915"
PAPER_CROSSLOCATE_FLEVOLAND_STUDY = PAPER_EVALUATIONS_ROOT / "farfield_crosslocate_flevoland_20260915"

# Current causal evaluations. The older DatasetGroup.run_spec pins below are
# retained for historical artifact/prior audits, not the new results table.
PAPER_SEEDS = (0, 1, 2, 3, 4)
PAPER_SWEEPS = {
    "ours": ("260914_hybrid_notracking_5seeds", "hybrid"),
    "no_tracking": ("260914_hybrid_notracking_5seeds", "no_tracking"),
    "no_range": ("260914_hybrid_no_range_bins", "hybrid_no_range"),
}
PAPER_LOCI_VERSION = "paper_full_multisource_landmark_only_unknown_heading_20260914_v1"

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


def bbox_area_km2(wsen) -> float:
    west, south, east, north = wsen
    mid = math.radians((south + north) / 2.0)
    return (east - west) * 111.32 * math.cos(mid) * (north - south) * 111.32


def region_area_km2(config: dict, region_policy: str) -> float | None:
    """Return the catalog region selected by a dataset group's policy."""
    plan = config.get("clip_plan")
    if region_policy == "area625" and isinstance(plan, dict):
        return float((plan.get("policy") or {}).get("resolved_area_km2", 0)) or None
    if region_policy == "fetch_bbox":
        bbox = config.get("region_bbox_wsen") or config.get("bbox_wsen")
        return bbox_area_km2(bbox) if bbox else None
    return None


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
        conditions="Rural",
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
        map_source="OSM / ENC",
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
        map_source="OSM / ENC",
        catalog_scope="boston_harbor_20260712",
        region_policy="area625",
        catalog_version="trim625_20260911_v1",
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
        conditions="Mountain ridge",
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
        map_source="OSM / ENC / FAA",
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

# Groups the results table can report today: every seed of every method exists.
TABLE_GROUPS = tuple(group for group in DATASET_GROUPS if group.run_spec)

# Complete four-seed runs used for the results table's full-method column.
FULL_METHOD_RUN_SPECS = tuple(group.run_spec for group in TABLE_GROUPS)


def _chain(pinhole, landmarks, tracks, audits, bearings, matches, diagnostics, inputs):
    return dict(zip(SEQUENCE_LANES, (
        pinhole, landmarks, tracks, audits, bearings, matches, diagnostics, inputs)))


# Regeneration inputs, September 2026. All extractions are complete. The
# current tracker is the comparison control, not a selected winner over #722.
# Batch 2 is queued through matching; None means a downstream version is not selected
# yet. DATASET_GROUPS.run_spec still names historical evaluations, which must
# be replaced after the regenerated inputs have been evaluated.
BATCH1_SEQUENCES = (
    "boston_harbor_leg1", "mount_washington_20260815_leg2",
    "flevoland_polder", "portland_flight_20260906_leg1",
)

SEQUENCE_ARTIFACTS: dict[str, dict[str, str | None]] = {
    sequence: _chain(
        "pin2048_down30_20260911_v1" if group.key == "portland"
        else "pin2048_20260911_v1",
        "v3pro_20260911_v1", "v3pro_20260911_v1",
        "v3pro_20260911_v1", "v3pro_20260911_v1",
        "promatch_20260911_v1" if sequence in BATCH1_SEQUENCES else "v3pro_20260911_v1",
        "v3pro_20260911_v1" if sequence in BATCH1_SEQUENCES else None,
        "promatch_20260911_epson_v1" if sequence in BATCH1_SEQUENCES else "v3pro_20260911_v1")
    for group in DATASET_GROUPS for sequence in group.sequences
}

# Batch 1 dedup tracks were produced at b53a0514. Batch 2 uses PR #722's
# c46b91ac, including its follow-up that counts rebirth/duplicate as support.
TRACKING_COMPARISON = {
    sequence: {
        "current": "v3pro_20260911_v1",
        "dedup_pr722": ("v3dedup_20260912_v1" if sequence in BATCH1_SEQUENCES
                        else "v3dedup_20260913_v1"),
    }
    for sequence in SEQUENCE_ARTIFACTS
}

# Portland's prompt explicitly declares the -30 degree pinhole pitch.
LLM_STANDARD: dict[str, str | None] = {
    "extraction.model": "gemini-3.1-pro-preview",
    "extraction.prompt_type": "osm_tags_farfield_v3",
    "audit.model": "gemini-3-flash-preview",
    "matching.model": "gemini-3.1-pro-preview",
}
LLM_SEQUENCE_OVERRIDES = {
    sequence: {"extraction.prompt_type": "osm_tags_farfield_v3_down30"}
    for group in DATASET_GROUPS if group.key == "portland"
    for sequence in group.sequences
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
