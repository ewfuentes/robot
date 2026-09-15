"""Resolve or create satellite imagery for localization viewers.

Viewer imagery is presentation evidence, not a scientific run input.  It still
needs one deterministic owner: without one, every run either has no imagery or
downloads an identical multi-megabyte mosaic by hand.  Resolution follows this
order:

1. an exact ``<run>.satellite`` sibling;
2. a compatible shared entry under
   ``<root>/reviews/satellite/<dataset>/<version>``;
3. a compatible legacy ``*.satellite`` sibling belonging to any other run of
   the same dataset.

If none exists, the dataset's recorded capture date drives an automatic ESRI
Wayback fetch. Plans of at most ``RUN_LOCAL_TILE_LIMIT`` tiles stay beside the
run; larger plans publish once in the shared review cache. Concurrent viewer
builds serialize per dataset and re-check discovery after taking the lock, so
only the first one downloads.

Compatibility is stricter than a matching filename: dataset, capture month,
ENU anchor, exact bounded-prior coverage, and fine trajectory coverage must all
agree. Every raster must also stay inside that prior. This is what makes
cross-run reuse safe even when old run names are inconsistent.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Iterator

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.localization import (
    run_io,
    satellite_underlay,
    side_outputs,
)


SATELLITE_MANIFEST = "satellite.json"
WIDE_EXTENT_KIND = "localization_prior"
SHARED_RELATIVE_ROOT = Path("reviews") / "satellite"
# 64 x 256 px tiles are about 4.2 megapixels across both mosaics. The
# encoded result is ordinarily small enough that keeping the source next to one
# run is cheaper and simpler than creating shared-cache structure for it.
RUN_LOCAL_TILE_LIMIT = 64
_CAPTURE_DATE = re.compile(r"^\d{4}-\d{2}(?:-\d{2})?$")
_ANCHOR_TOLERANCE_DEG = 1e-9
_COVERAGE_TOLERANCE_M = 1.0


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    root: Path
    dataset: str
    capture_date: str | None
    anchor_lat_deg: float
    anchor_lon_deg: float
    prior_bounds: tuple[float, float, float, float] | None
    trajectory_bounds: tuple[float, float, float, float] | None


def _bounds(east: list[float], north: list[float]
            ) -> tuple[float, float, float, float] | None:
    if not east or not north or len(east) != len(north):
        return None
    if not all(math.isfinite(value) for value in (*east, *north)):
        return None
    return min(east), max(east), min(north), max(north)


def _prior_bounds(document: dict) -> tuple[float, float, float, float] | None:
    filter_config = document.get("filter_config")
    if not isinstance(filter_config, dict):
        return None
    init = filter_config.get("init")
    if not isinstance(init, dict) or init.get("kind") != "UniformBoxInit":
        return None
    try:
        bounds = (float(init["east_min_m"]), float(init["east_max_m"]),
                  float(init["north_min_m"]), float(init["north_max_m"]))
    except (KeyError, TypeError, ValueError):
        return None
    if (not all(math.isfinite(value) for value in bounds)
            or not bounds[0] < bounds[1] or not bounds[2] < bounds[3]):
        return None
    return bounds


def _trajectory_bounds(run_dir: Path
                       ) -> tuple[float, float, float, float] | None:
    for name, east_key, north_key in (
            ("truth.jsonl", "east_m", "north_m"),
            ("tier0_health.jsonl", "mean_east_m", "mean_north_m")):
        path = run_dir / name
        if not side_outputs.regular_file(path):
            continue
        try:
            records = [json.loads(line)
                       for line in path.read_text().splitlines()
                       if line.strip()]
            east = [float(record[east_key]) for record in records]
            north = [float(record[north_key]) for record in records]
        except (OSError, UnicodeError, ValueError, KeyError, TypeError):
            continue
        result = _bounds(east, north)
        if result is not None:
            return result
    return None


def _run_local_destination(summary: RunSummary) -> Path:
    """The run's own ``<run>.satellite`` sibling."""
    return summary.run_dir.with_name(summary.run_dir.name + ".satellite")


def summarize_run(run_dir: Path) -> RunSummary | None:
    """Read only the small identity/geometry records needed for discovery."""
    run_dir = side_outputs.absolute(run_dir)
    if not side_outputs.regular_directory(run_dir):
        return None
    # Canonical layout: <root>/runs/<experiment>/<run>.
    if len(run_dir.parents) < 3 or run_dir.parents[1].name != "runs":
        return None
    root = run_dir.parents[2]
    artifact_document = side_outputs.read_json_dict(
        run_dir / artifact.MANIFEST_NAME)
    run_document = side_outputs.read_json_dict(
        run_dir / run_io.RUN_MANIFEST_NAME)
    if artifact_document is None or run_document is None:
        return None
    if artifact_document.get("kind") != run_io.RUN_KIND:
        return None
    dataset = artifact_document.get("dataset")
    if (not isinstance(dataset, str) or not dataset
            or run_document.get("dataset") != dataset):
        return None
    try:
        artifact.require_identifier(dataset, "satellite dataset")
        anchor_lat = float(run_document["anchor_lat_deg"])
        anchor_lon = float(run_document["anchor_lon_deg"])
    except (artifact.ArtifactValidationError, KeyError, TypeError, ValueError):
        return None
    if (not math.isfinite(anchor_lat) or not math.isfinite(anchor_lon)
            or not -90.0 <= anchor_lat <= 90.0
            or not -180.0 <= anchor_lon <= 180.0):
        return None

    capture_date = None
    dataset_metadata = side_outputs.read_json_dict(
        root / "datasets" / dataset / "pipeline_metadata.json")
    if dataset_metadata is not None:
        candidate = dataset_metadata.get("capture_date")
        if isinstance(candidate, str) and _CAPTURE_DATE.fullmatch(candidate):
            capture_date = candidate
    return RunSummary(
        run_dir=run_dir,
        root=root,
        dataset=dataset,
        capture_date=capture_date,
        anchor_lat_deg=anchor_lat,
        anchor_lon_deg=anchor_lon,
        prior_bounds=_prior_bounds(run_document),
        trajectory_bounds=_trajectory_bounds(run_dir),
    )


def _source_dataset(provenance_document: dict) -> str | None:
    inputs = provenance_document.get("inputs")
    run_dir = inputs.get("run_dir") if isinstance(inputs, dict) else None
    if not isinstance(run_dir, str) or not run_dir:
        return None
    source_manifest = side_outputs.read_json_dict(
        Path(run_dir) / artifact.MANIFEST_NAME)
    if source_manifest is None:
        return None
    dataset = source_manifest.get("dataset")
    return dataset if isinstance(dataset, str) else None


def _entry_bounds(entry: dict
                  ) -> tuple[float, float, float, float] | None:
    try:
        return (float(entry["east_min"]), float(entry["east_max"]),
                float(entry["north_min"]), float(entry["north_max"]))
    except (KeyError, TypeError, ValueError):
        return None


def _contains(outer: tuple[float, float, float, float],
              inner: tuple[float, float, float, float]) -> bool:
    """Whether ``outer`` spans all of ``inner`` within tolerance."""
    return (outer[0] <= inner[0] + _COVERAGE_TOLERANCE_M
            and outer[1] >= inner[1] - _COVERAGE_TOLERANCE_M
            and outer[2] <= inner[2] + _COVERAGE_TOLERANCE_M
            and outer[3] >= inner[3] - _COVERAGE_TOLERANCE_M)


def _same_extent(actual: tuple[float, float, float, float],
                 wanted: tuple[float, float, float, float]) -> bool:
    return all(abs(value - target) <= _COVERAGE_TOLERANCE_M
               for value, target in zip(actual, wanted))


def compatible(candidate: Path, summary: RunSummary) -> bool:
    """Whether an underlay is safe to reuse for ``summary``."""
    candidate = Path(candidate)
    if not side_outputs.regular_directory(candidate):
        return False
    spec = side_outputs.read_json_dict(candidate / SATELLITE_MANIFEST)
    provenance_document = side_outputs.read_json_dict(
        candidate / artifact.MANIFEST_NAME)
    if spec is None or provenance_document is None:
        return False
    if (spec.get("wide_extent_kind") != WIDE_EXTENT_KIND
            or summary.prior_bounds is None):
        return False

    dataset = spec.get("dataset")
    if dataset is None:
        dataset = _source_dataset(provenance_document)
    if dataset != summary.dataset:
        return False
    config = provenance_document.get("config")
    recorded_date = spec.get("capture_date")
    if recorded_date is None and isinstance(config, dict):
        recorded_date = config.get("date")
    if summary.capture_date is not None:
        if (not isinstance(recorded_date, str)
                or recorded_date[:7] != summary.capture_date[:7]):
            return False
    try:
        if (abs(float(spec["anchor_lat_deg"]) - summary.anchor_lat_deg)
                > _ANCHOR_TOLERANCE_DEG
                or abs(float(spec["anchor_lon_deg"])
                       - summary.anchor_lon_deg) > _ANCHOR_TOLERANCE_DEG):
            return False
    except (KeyError, TypeError, ValueError):
        return False

    layers = spec.get("layers")
    if not isinstance(layers, list) or not layers:
        return False
    usable = []
    for entry in layers:
        if not isinstance(entry, dict):
            return False
        image_name = entry.get("image")
        if (not isinstance(image_name, str) or not image_name
                or Path(image_name).name != image_name):
            return False
        image = candidate / image_name
        if (not side_outputs.regular_file(image)
                or image.stat().st_size <= 0):
            return False
        try:
            n_tiles = int(entry["n_tiles"])
            n_failed = int(entry["n_failed"])
        except (KeyError, TypeError, ValueError):
            return False
        if n_tiles <= 0 or not 0 <= n_failed < n_tiles:
            return False
        bounds = _entry_bounds(entry)
        if bounds is None:
            return False
        usable.append((Path(image_name).stem.casefold(), bounds))

    prior = summary.prior_bounds
    if not all(_contains(prior, bounds) for _, bounds in usable):
        return False
    if not any(_same_extent(bounds, prior)
               for stem, bounds in usable if stem.startswith("wide")):
        return False
    wanted_trajectory = (
        None if summary.trajectory_bounds is None
        else satellite_underlay.intersection(
            summary.trajectory_bounds, prior))
    return wanted_trajectory is None or any(
        _contains(bounds, wanted_trajectory)
        for stem, bounds in usable if stem.startswith("fine"))


def _candidate_directories(summary: RunSummary) -> Iterator[Path]:
    return side_outputs.discovery_candidates(
        _run_local_destination(summary),
        (summary.root / SHARED_RELATIVE_ROOT / summary.dataset, "*"),
        (summary.root / "runs", "*/*.satellite"))


def discover(run_dir: Path) -> Path | None:
    """Find the first deterministic compatible underlay without writing."""
    summary = summarize_run(run_dir)
    if summary is None:
        return None
    for candidate in _candidate_directories(summary):
        if compatible(candidate, summary):
            return candidate
    return None


def _shared_destination(summary: RunSummary, plans: list[dict]) -> Path:
    recipe = {
        "dataset": summary.dataset,
        "capture_date": summary.capture_date,
        "anchor_lat_deg": summary.anchor_lat_deg,
        "anchor_lon_deg": summary.anchor_lon_deg,
        "layers": [
            {"name": plan["name"], "zoom": plan["zoom"],
             "tiles": list(plan["tiles"]),
             "bounds_enu": list(plan.get("bounds_enu", ())),
             "output_px": list(plan.get("output_px", ()))}
            for plan in plans
        ],
    }
    digest = artifact.sha256_json(recipe)[:12]
    date = (summary.capture_date or "unknown-date").replace("-", "")
    return (summary.root / SHARED_RELATIVE_ROOT / summary.dataset
            / f"wayback-{date}-{digest}")


def automatic_destination(summary: RunSummary, plans: list[dict]) -> Path:
    """Run-local for compact plans; content-keyed shared cache otherwise."""
    total = sum(int(plan["n_tiles"]) for plan in plans)
    if total <= RUN_LOCAL_TILE_LIMIT:
        return _run_local_destination(summary)
    return _shared_destination(summary, plans)


@contextmanager
def _generation_lock(summary: RunSummary) -> Iterator[None]:
    directory = summary.root / SHARED_RELATIVE_ROOT / summary.dataset
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".generate.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"satellite generation lock is not regular: "
                             f"{lock_path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def find_or_generate(run_dir: Path) -> Path | None:
    """Resolve an underlay, fetching one from the recorded date if needed."""
    existing = discover(run_dir)
    if existing is not None:
        return existing
    summary = summarize_run(run_dir)
    if (summary is None or summary.capture_date is None
            or summary.prior_bounds is None):
        return None

    data = run_io.read_run(summary.run_dir)
    plans = satellite_underlay.plan_underlay(data)
    destination = automatic_destination(summary, plans)
    with _generation_lock(summary):
        # Another seed may have completed the same download while we waited.
        existing = discover(summary.run_dir)
        if existing is not None:
            return existing
        if destination.exists() or destination.is_symlink():
            raise ValueError(
                "automatic satellite destination already exists but is not "
                f"compatible: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        total = sum(int(plan["n_tiles"]) for plan in plans)
        placement = ("run-local" if total <= RUN_LOCAL_TILE_LIMIT
                     else "shared dataset")
        print(f"  satellite: no compatible underlay; generating {placement} "
              f"imagery from capture date {summary.capture_date}")
        satellite_underlay.describe_plan(
            plans, anchor_lat_deg=summary.anchor_lat_deg,
            max_tiles=satellite_underlay.DEFAULT_MAX_TILES)
        return satellite_underlay.generate_underlay(
            summary.run_dir, date=summary.capture_date,
            output_dir=destination, data=data, plans=plans)
