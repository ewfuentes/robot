"""Safe publication for diagnostics derived from immutable localization runs.

Viewer pages and plots are reproducible views of a completed run, not part of
the run artifact itself. They therefore publish as sibling directories and
never write through the run directory. A sibling ``.incomplete`` directory is
renamed only after the producer exits successfully, matching the visibility
rule used by typed farfield artifacts without pretending these diagnostics
are pipeline inputs.

Discovering an existing side output is the mirror image of publishing one, so
the small symlink-safe filesystem predicates both jobs need live here too.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
from typing import Iterator

from experimental.overhead_matching.swag.farfield import artifact


class SideOutputError(ValueError):
    """A diagnostics destination would violate the immutable-run boundary."""


@dataclass(frozen=True)
class SideOutputDirectory:
    """The unpublished working directory and its eventual public path."""

    staging_dir: Path
    destination: Path


def absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def regular_file(path: Path) -> bool:
    """A real file, never a symlink to one."""
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def regular_directory(path: Path) -> bool:
    """A real directory, never a symlink to one."""
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def read_json_dict(path: Path) -> dict | None:
    """A JSON object read from a regular file; None for anything else.

    Discovery walks directories nobody promised are well formed, so an
    unreadable or wrongly shaped candidate is "not a match", not an error.
    """
    if not regular_file(path):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def discovery_candidates(sibling: Path,
                         *globs: tuple[Path, str]) -> Iterator[Path]:
    """The exact sibling first, then each root's sorted glob, de-duplicated.

    Deterministic order is the point: two runs asking the same question must
    reuse the same asset rather than whichever one the filesystem listed first.
    """
    seen: set[Path] = set()

    def fresh(candidate: Path) -> Path | None:
        candidate = absolute(candidate)
        if candidate in seen:
            return None
        seen.add(candidate)
        return candidate

    first = fresh(sibling)
    if first is not None:
        yield first
    for root, pattern in globs:
        if not regular_directory(root):
            continue
        for candidate in sorted(root.glob(pattern), key=str):
            found = fresh(candidate)
            if found is not None:
                yield found


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _reject_symlink_components(path: Path) -> None:
    """Reject every existing symlink on an absolute path."""
    path = absolute(path)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            # Descendants cannot exist until this missing ancestor does.
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise SideOutputError(
                f"side-output paths cannot contain symlinks: {current}")


def _destination(run_dir: Path, output_dir: Path | None,
                 suffix: str) -> tuple[Path, Path]:
    run_dir = absolute(Path(run_dir))
    if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
        raise SideOutputError(
            f"side-output suffix must be one path-free extension: {suffix!r}")
    _reject_symlink_components(run_dir)
    if not run_dir.is_dir():
        raise SideOutputError(
            f"run directory must be an existing regular directory: {run_dir}")

    destination = (absolute(Path(output_dir)) if output_dir is not None
                   else run_dir.with_name(run_dir.name + suffix))
    _reject_symlink_components(destination)
    if not destination.parent.is_dir():
        raise SideOutputError(
            "side-output parent must be an existing regular directory: "
            f"{destination.parent}")

    # Check both the spelling and resolved targets. The first catches ``..``
    # aliases; the second catches an existing ancestor redirected into the run.
    if (_is_within(destination, run_dir)
            or _is_within(destination.resolve(strict=False),
                          run_dir.resolve(strict=True))):
        raise SideOutputError(
            f"side-output destination cannot be inside immutable run "
            f"directory {run_dir}: {destination}")
    return run_dir, destination


def default_directory(run_dir: Path, suffix: str) -> Path:
    """Return the validated deterministic sibling destination."""
    _, destination = _destination(Path(run_dir), None, suffix)
    return destination


def _validate_staging_tree(staging_dir: Path) -> None:
    entries = list(staging_dir.rglob("*"))
    if not entries:
        raise SideOutputError(
            f"refusing to publish an empty side-output directory: {staging_dir}")
    for entry in entries:
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise SideOutputError(
                f"side-output content cannot contain symlinks: {entry}")
        if not (stat.S_ISREG(metadata.st_mode)
                or stat.S_ISDIR(metadata.st_mode)):
            raise SideOutputError(
                f"side-output content must be regular files: {entry}")


@contextmanager
def publish_directory(run_dir: Path, *, output_dir: Path | None,
                      suffix: str) -> Iterator[SideOutputDirectory]:
    """Build and atomically publish one no-clobber diagnostics directory.

    An exception deliberately leaves ``<destination>.incomplete`` in place so
    operators can inspect the failed attempt and must acknowledge it before a
    retry. Completed output is never replaced or merged with a prior run.
    """
    _, destination = _destination(Path(run_dir), output_dir, suffix)
    staging_dir = destination.with_name(
        destination.name + artifact.INCOMPLETE_SUFFIX)
    _reject_symlink_components(staging_dir)
    if destination.exists() or destination.is_symlink():
        raise SideOutputError(
            f"completed side-output directory already exists: {destination}")
    if staging_dir.exists() or staging_dir.is_symlink():
        raise SideOutputError(
            f"incomplete side-output directory already exists: {staging_dir}")
    try:
        staging_dir.mkdir()
    except FileExistsError as exc:
        raise SideOutputError(
            f"incomplete side-output directory already exists: "
            f"{staging_dir}") from exc

    publication = SideOutputDirectory(
        staging_dir=staging_dir, destination=destination)
    yield publication
    _validate_staging_tree(staging_dir)
    artifact.publish_directory_no_clobber(staging_dir, destination)
