"""Find review pages that exactly match a localization viewer's inputs.

The viewer's matcher/audit links are presentation context, but linking a page
from a different matching artifact would be actively misleading.  A matcher
review records all of its scientific inputs in its provenance manifest.  This
module follows the localization run to its exact ``landmark_matches`` ancestor
and only reuses a review whose matching, tracks, audits, and catalog paths all
agree.

Review pages are side outputs rather than canonical artifacts, so they may live
beside another seed or experiment using those same immutable inputs.  Discovery
checks the current run's sibling first, then other runs under the same data
root.  The audit page is taken from the compatible matcher review's own
recorded input, preserving the pair that was generated together.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
from typing import Iterator

from experimental.overhead_matching.swag.farfield import artifact, paths
from experimental.overhead_matching.swag.farfield.localization import run_io


MATCHER_SUFFIX = ".matcher-review"
AUDIT_SUFFIX = ".audit-review"
MATCHER_GENERATOR = ("//experimental/overhead_matching/swag/farfield/"
                     "matching:match_viewer")


@dataclass(frozen=True)
class ReviewPages:
    matcher: Path | None = None
    audit: Path | None = None


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _regular_directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def _regular_json(path: Path) -> dict | None:
    if not _regular_file(path):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _input_path(value) -> Path | None:
    return _absolute(value) if isinstance(value, str) and value else None


def _matching_ancestor(run_dir: Path) -> Path | None:
    """Exact landmark-matches ancestor recorded by a typed run."""
    run_manifest = _regular_json(run_dir / artifact.MANIFEST_NAME)
    if (run_manifest is None
            or run_manifest.get("kind") != run_io.RUN_KIND):
        return None
    upstreams = run_manifest.get("upstreams")
    if not isinstance(upstreams, list):
        return None
    localization_inputs = [
        entry for entry in upstreams
        if isinstance(entry, dict)
        and entry.get("kind") == paths.LOCALIZATION_INPUTS
        and isinstance(entry.get("path"), str)
    ]
    if len(localization_inputs) != 1:
        return None
    inputs_dir = _absolute(localization_inputs[0]["path"])
    inputs_manifest = _regular_json(inputs_dir / artifact.MANIFEST_NAME)
    if (inputs_manifest is None
            or inputs_manifest.get("kind") != paths.LOCALIZATION_INPUTS
            or inputs_manifest.get("dataset") != run_manifest.get("dataset")):
        return None
    ancestors = inputs_manifest.get("upstreams")
    if not isinstance(ancestors, list):
        return None
    matching = [
        _input_path(entry.get("path")) for entry in ancestors
        if isinstance(entry, dict)
        and entry.get("kind") == paths.LANDMARK_MATCHES
    ]
    return matching[0] if len(matching) == 1 else None


def _candidate_directories(run_dir: Path) -> Iterator[Path]:
    seen: set[Path] = set()

    def emit(candidate: Path) -> Path | None:
        candidate = _absolute(candidate)
        if candidate in seen:
            return None
        seen.add(candidate)
        return candidate

    sibling = emit(run_dir.with_name(run_dir.name + MATCHER_SUFFIX))
    if sibling is not None:
        yield sibling

    # Canonical layout: <root>/runs/<experiment>/<run>.  Outside it, the exact
    # sibling is still useful but there is no bounded tree that is safe to scan.
    if len(run_dir.parents) < 3 or run_dir.parents[1].name != "runs":
        return
    runs = run_dir.parents[1]
    if not _regular_directory(runs):
        return
    for candidate in sorted(runs.glob(f"*/*{MATCHER_SUFFIX}"),
                            key=lambda value: str(value)):
        candidate = emit(candidate)
        if candidate is not None:
            yield candidate


def _compatible(candidate: Path, *, matching_dir: Path, tracks_dir: Path,
                audit_dir: Path, catalog_dir: Path) -> ReviewPages | None:
    if not _regular_directory(candidate):
        return None
    matcher_page = candidate / "index.html"
    provenance = _regular_json(candidate / artifact.MANIFEST_NAME)
    if (not _regular_file(matcher_page) or provenance is None
            or provenance.get("generator") != MATCHER_GENERATOR):
        return None
    inputs = provenance.get("inputs")
    if not isinstance(inputs, dict):
        return None
    wanted = {
        "matching": matching_dir,
        "tracks": tracks_dir,
        "semantic_audits": audit_dir,
        "catalog": catalog_dir,
    }
    if any(_input_path(inputs.get(name)) != expected
           for name, expected in wanted.items()):
        return None
    audit_page = _input_path(inputs.get("semantic_audit_review"))
    if audit_page is not None and not _regular_file(audit_page):
        audit_page = None
    return ReviewPages(matcher=matcher_page, audit=audit_page)


def discover(run_dir: Path, *, tracks_dir: Path, audit_dir: Path,
             catalog_dir: Path) -> ReviewPages:
    """Find a deterministic exact-input matcher/audit review pair."""
    run_dir = _absolute(run_dir)
    matching_dir = _matching_ancestor(run_dir)
    if matching_dir is None:
        return ReviewPages()
    expected = {
        "matching_dir": matching_dir,
        "tracks_dir": _absolute(tracks_dir),
        "audit_dir": _absolute(audit_dir),
        "catalog_dir": _absolute(catalog_dir),
    }
    for candidate in _candidate_directories(run_dir):
        pages = _compatible(candidate, **expected)
        if pages is not None:
            return pages
    return ReviewPages()
