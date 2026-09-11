"""Strict immutable source-coverage lineage for catalog consumers.

Matching may consume either the full catalog selected by collection or a
derived (for example, semantically trimmed) catalog. A derived catalog is
usable only when its single exact CATALOGS parent chain terminates at the
stage-5 full catalog, whose source-coverage attestation passed. Every recorded
ArtifactRef is reopened and compared by digest; paths alone are never identity.
"""

from __future__ import annotations

from typing import Any

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.catalog import schema


SOURCE_COVERAGE_SCHEMA = "farfield_catalog_source_coverage/v2"
_SOURCE_COVERAGE_KEYS = frozenset({
    "schema",
    "status",
    "message",
    "details",
})


class CatalogLineageError(artifact.ArtifactValidationError):
    """A catalog cannot prove complete source coverage through exact lineage."""


def _exact_keys(value: Any, expected: frozenset[str], where: str) -> dict:
    if not isinstance(value, dict):
        raise CatalogLineageError(f"{where} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise CatalogLineageError(
            f"{where} has missing={missing}, unknown={unknown}")
    return value


def _open_exact_catalog(
        reference: artifact.ArtifactRef,
        *,
        dataset: str,
) -> tuple[artifact.ArtifactRef, artifact.ArtifactManifest]:
    if not isinstance(reference, artifact.ArtifactRef):
        raise CatalogLineageError(
            "catalog lineage must contain only ArtifactRef values")
    if reference.kind != paths_lib.CATALOGS:
        raise CatalogLineageError(
            "catalog lineage contains non-CATALOGS artifact "
            f"{reference.kind!r}")
    if reference.dataset != dataset:
        raise CatalogLineageError(
            "catalog lineage crosses datasets: expected "
            f"{dataset!r}, found {reference.dataset!r}")
    try:
        opened = artifact.open_artifact(
            reference.path,
            expected_kind=paths_lib.CATALOGS,
            expected_dataset=dataset,
            expected_version=reference.version,
        )
    except artifact.ArtifactError as error:
        raise CatalogLineageError(
            f"cannot open recorded catalog lineage artifact {reference.path}: "
            f"{error}") from error
    if opened != reference:
        raise CatalogLineageError(
            "catalog lineage ArtifactRef does not match the artifact at its "
            f"recorded path: {reference.path}")
    try:
        manifest = artifact.load_manifest(opened.path)
    except artifact.ArtifactError as error:
        raise CatalogLineageError(
            f"cannot load catalog lineage manifest {opened.path}: {error}"
        ) from error
    if manifest.declared_outputs != ("catalog.feather",):
        raise CatalogLineageError(
            "each catalog lineage artifact must declare exactly "
            "catalog.feather")
    return opened, manifest


def _validate_auxiliary_reference(
        reference: artifact.ArtifactRef, *, dataset: str) -> None:
    """Reopen a non-catalog provenance edge without traversing its graph."""
    if not isinstance(reference, artifact.ArtifactRef):
        raise CatalogLineageError(
            "catalog lineage must contain only ArtifactRef values")
    if reference.dataset != dataset:
        raise CatalogLineageError(
            "catalog provenance crosses datasets: expected "
            f"{dataset!r}, found {reference.dataset!r}")
    try:
        opened = artifact.open_artifact(
            reference.path,
            expected_kind=reference.kind,
            expected_dataset=dataset,
            expected_version=reference.version,
        )
    except artifact.ArtifactError as error:
        raise CatalogLineageError(
            f"cannot open recorded catalog provenance artifact "
            f"{reference.path}: {error}") from error
    if opened != reference:
        raise CatalogLineageError(
            "catalog provenance ArtifactRef does not match the artifact at "
            f"its recorded path: {reference.path}")


def _validate_passed_coverage(config: Any) -> None:
    coverage = _exact_keys(
        config, _SOURCE_COVERAGE_KEYS, "full catalog source_coverage")
    if coverage["schema"] != SOURCE_COVERAGE_SCHEMA:
        raise CatalogLineageError(
            "unsupported full catalog source_coverage schema")
    if coverage["status"] != "passed":
        raise CatalogLineageError(
            "full catalog source_coverage must attest status='passed'")
    if (not isinstance(coverage["message"], str)
            or not coverage["message"].strip()):
        raise CatalogLineageError(
            "full catalog source_coverage message must be non-empty")
    if not isinstance(coverage["details"], list):
        raise CatalogLineageError(
            "full catalog source_coverage details must be a list")


def require_passed_source_coverage(
        catalog_ref: artifact.ArtifactRef,
) -> artifact.ArtifactRef:
    """Return the exact terminal full catalog or reject ``catalog_ref``.

    The selected catalog itself and every parent are validated by immutable
    identity. A full catalog is terminal. Any non-full catalog must have
    exactly one CATALOGS parent, which permits deliberate chains of derived
    catalogs without accepting ambiguous branches or untyped inputs.
    """
    if not isinstance(catalog_ref, artifact.ArtifactRef):
        raise CatalogLineageError("catalog_ref must be an ArtifactRef")
    if catalog_ref.kind != paths_lib.CATALOGS:
        raise CatalogLineageError("catalog_ref must identify CATALOGS")

    dataset = catalog_ref.dataset
    current = catalog_ref
    visited: set[artifact.ArtifactRef] = set()
    while True:
        if current in visited:
            raise CatalogLineageError("catalog lineage contains a cycle")
        visited.add(current)
        current, manifest = _open_exact_catalog(current, dataset=dataset)

        artifact_schema = manifest.config.get("schema")
        if artifact_schema == schema.FULL_ARTIFACT_SCHEMA:
            if manifest.upstreams:
                raise CatalogLineageError(
                    "the stage-5 full catalog must terminate catalog lineage")
            _validate_passed_coverage(
                manifest.config.get("source_coverage"))
            return current
        if artifact_schema is not None:
            raise CatalogLineageError(
                f"unsupported catalog artifact schema {artifact_schema!r}")
        catalog_parents = tuple(
            reference for reference in manifest.upstreams
            if reference.kind == paths_lib.CATALOGS)
        auxiliary = tuple(
            reference for reference in manifest.upstreams
            if reference.kind != paths_lib.CATALOGS)
        if len(catalog_parents) != 1:
            raise CatalogLineageError(
                "a derived catalog must have exactly one CATALOGS upstream")
        for reference in auxiliary:
            _validate_auxiliary_reference(reference, dataset=dataset)
        current = catalog_parents[0]
