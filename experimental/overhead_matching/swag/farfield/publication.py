"""Validated artifact publication with best-effort index refresh.

Direct producers use :func:`published_artifact` in place of constructing an
``ArtifactDirectoryBuilder`` themselves.  The yielded value is the builder,
so staging and output declarations retain the ordinary artifact API.  Once
publication succeeds, this module re-opens the immutable artifact and, only
for a canonical data-root lane, refreshes the surrounding navigation pages.

Index pages are derived conveniences, not part of artifact validity.  A
refresh failure is therefore reported prominently but never rolls back or
invalidates an already-published artifact.
"""

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.viewers import indexes


class PublicationValidationError(RuntimeError):
    """A directory was published but did not re-open as its exact ref."""


def _canonical_data_root(reference: artifact.ArtifactRef) -> Path | None:
    """Return the owning canonical data root, or ``None`` for work outputs."""
    published = Path(reference.path)

    # <root>/artifacts/<kind>/<dataset>/<version>
    if (len(published.parents) >= 4
            and published.parents[2].name == "artifacts"
            and published.parent.name == reference.dataset
            and published.parent.parent.name == reference.kind
            and published.name == reference.version):
        return published.parents[3]

    # <root>/runs/<experiment>/<run>.  Runs are the one artifact kind whose
    # canonical path has an experiment component rather than a kind/dataset
    # pair.  Validate every identity-bearing component before opting in.
    if (reference.kind == "localization_run"
            and len(published.parents) >= 3
            and published.parents[1].name == "runs"
            and published.name == reference.version):
        try:
            artifact.require_identifier(
                published.parent.name, "run experiment path component")
        except artifact.ArtifactValidationError:
            return None
        return published.parents[2]

    return None


@contextmanager
def published_artifact(
        destination: Path | str, *, kind: str, dataset: str, version: str,
        generator: str, git_commit: str = "unknown",
        arguments: Iterable[str] | None = None,
        upstreams: Iterable[artifact.ArtifactRef] = (),
        config: Mapping[str, Any] | None = None,
        declared_outputs: Iterable[str | Path],
) -> Iterator[artifact.ArtifactDirectoryBuilder]:
    """Publish, validate, and refresh indexes for one canonical artifact.

    The yielded object is the underlying ``ArtifactDirectoryBuilder``.  No
    refresh is attempted when the destination is outside the two canonical
    data-root layouts.  Once publication succeeds, index-refresh errors are
    warnings because the immutable artifact remains valid and discoverable by
    its explicit path and ref.
    """
    builder = artifact.ArtifactDirectoryBuilder(
        destination,
        kind=kind,
        dataset=dataset,
        version=version,
        generator=generator,
        git_commit=git_commit,
        arguments=arguments,
        upstreams=upstreams,
        config=config,
        declared_outputs=declared_outputs,
    )
    with builder:
        yield builder

    reference = builder.artifact_ref
    if reference is None:
        raise PublicationValidationError(
            f"artifact publication returned no immutable ref: {destination}")
    validated = artifact.open_artifact(
        destination,
        expected_kind=reference.kind,
        expected_dataset=reference.dataset,
        expected_version=reference.version,
    )
    if validated != reference:
        raise PublicationValidationError(
            "published artifact identity changed during validation: "
            f"expected {reference}, found {validated}")

    data_root = _canonical_data_root(validated)
    if data_root is not None:
        try:
            indexes.refresh(data_root)
        except Exception as error:  # Derived navigation must not undo science.
            print(
                "WARNING: artifact published successfully, but data-root "
                f"index refresh failed for {data_root}: {error}",
                file=sys.stderr,
            )
