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
from pathlib import Path
from typing import Iterator

from experimental.overhead_matching.swag.farfield import artifact, paths
from experimental.overhead_matching.swag.farfield.localization import (
    run_io,
    side_outputs,
)


MATCHER_SUFFIX = ".matcher-review"
AUDIT_SUFFIX = ".audit-review"
MATCHER_GENERATOR = ("//experimental/overhead_matching/swag/farfield/"
                     "matching:match_viewer")


@dataclass(frozen=True)
class ReviewPages:
    matcher: Path | None = None
    audit: Path | None = None


def _input_path(value) -> Path | None:
    return (side_outputs.absolute(value)
            if isinstance(value, str) and value else None)


def _matching_ancestor(run_dir: Path) -> Path | None:
    """Exact landmark-matches ancestor recorded by a typed run."""
    run_manifest = side_outputs.read_json_dict(
        run_dir / artifact.MANIFEST_NAME)
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
    inputs_dir = side_outputs.absolute(localization_inputs[0]["path"])
    inputs_manifest = side_outputs.read_json_dict(
        inputs_dir / artifact.MANIFEST_NAME)
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
    sibling = run_dir.with_name(run_dir.name + MATCHER_SUFFIX)
    # Canonical layout: <root>/runs/<experiment>/<run>.  Outside it, the exact
    # sibling is still useful but there is no bounded tree that is safe to scan.
    if len(run_dir.parents) < 3 or run_dir.parents[1].name != "runs":
        return side_outputs.discovery_candidates(sibling)
    return side_outputs.discovery_candidates(
        sibling, (run_dir.parents[1], f"*/*{MATCHER_SUFFIX}"))


def _compatible(candidate: Path, *, matching_dir: Path, tracks_dir: Path,
                audit_dir: Path, catalog_dir: Path) -> ReviewPages | None:
    if not side_outputs.regular_directory(candidate):
        return None
    matcher_page = candidate / "index.html"
    provenance = side_outputs.read_json_dict(
        candidate / artifact.MANIFEST_NAME)
    if (not side_outputs.regular_file(matcher_page) or provenance is None
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
    if audit_page is not None and not side_outputs.regular_file(audit_page):
        audit_page = None
    return ReviewPages(matcher=matcher_page, audit=audit_page)


def discover(run_dir: Path, *, tracks_dir: Path, audit_dir: Path,
             catalog_dir: Path) -> ReviewPages:
    """Find a deterministic exact-input matcher/audit review pair."""
    run_dir = side_outputs.absolute(run_dir)
    matching_dir = _matching_ancestor(run_dir)
    if matching_dir is None:
        return ReviewPages()
    expected = {
        "matching_dir": matching_dir,
        "tracks_dir": side_outputs.absolute(tracks_dir),
        "audit_dir": side_outputs.absolute(audit_dir),
        "catalog_dir": side_outputs.absolute(catalog_dir),
    }
    for candidate in _candidate_directories(run_dir):
        pages = _compatible(candidate, **expected)
        if pages is not None:
            return pages
    return ReviewPages()
