import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.catalog import lineage
from experimental.overhead_matching.swag.farfield.catalog import schema


DATASET = "lineage_test"


def coverage(status="passed"):
    return {
        "schema": lineage.SOURCE_COVERAGE_SCHEMA,
        "status": status,
        "message": "all requested source area covered",
        "details": [],
    }


def publish_catalog(path: Path, *, upstreams=(), config=None, dataset=DATASET):
    with artifact.ArtifactDirectoryBuilder(
            path,
            kind=paths_lib.CATALOGS,
            dataset=dataset,
            version=path.name,
            generator="test:catalog",
            git_commit="test",
            arguments=(),
            upstreams=upstreams,
            config=config,
            declared_outputs=("catalog.feather",)) as builder:
        artifact.atomic_write_file(
            builder.output_path("catalog.feather"), b"test catalog")
    return artifact.open_artifact(path)


def publish_matching(path: Path, *, dataset=DATASET):
    with artifact.ArtifactDirectoryBuilder(
            path,
            kind=paths_lib.LANDMARK_MATCHES,
            dataset=dataset,
            version=path.name,
            generator="test:matching",
            git_commit="test",
            arguments=(),
            upstreams=(),
            config={},
            declared_outputs=("matches.json",)) as builder:
        artifact.atomic_write_file(builder.output_path("matches.json"), b"{}")
    return artifact.open_artifact(path)


class CatalogLineageTest(unittest.TestCase):
    def test_full_and_derived_catalogs_resolve_to_exact_full_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = publish_catalog(
                root / "full-v1",
                config={
                    "schema": schema.FULL_ARTIFACT_SCHEMA,
                    "source_coverage": coverage(),
                },
            )
            trimmed = publish_catalog(
                root / "trimmed-v1", upstreams=(full,), config={"rows": 1})
            retrimmed = publish_catalog(
                root / "trimmed-v2", upstreams=(trimmed,), config={"rows": 1})

            self.assertEqual(
                lineage.require_passed_source_coverage(full), full)
            self.assertEqual(
                lineage.require_passed_source_coverage(retrimmed), full)

    def test_failed_source_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = publish_catalog(
                Path(tmp) / "full-v1",
                config={
                    "schema": schema.FULL_ARTIFACT_SCHEMA,
                    "source_coverage": coverage("failed"),
                },
            )
            with self.assertRaisesRegex(
                    lineage.CatalogLineageError, "status='passed'"):
                lineage.require_passed_source_coverage(full)

    def test_derived_catalog_may_record_exact_auxiliary_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = publish_catalog(
                root / "full-v1",
                config={
                    "schema": schema.FULL_ARTIFACT_SCHEMA,
                    "source_coverage": coverage(),
                },
            )
            matching = publish_matching(root / "matching-v1")
            trimmed = publish_catalog(
                root / "trimmed-v1", upstreams=(full, matching), config={})

            self.assertEqual(
                lineage.require_passed_source_coverage(trimmed), full)

    def test_auxiliary_provenance_requires_exact_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = publish_catalog(
                root / "full-v1",
                config={
                    "schema": schema.FULL_ARTIFACT_SCHEMA,
                    "source_coverage": coverage(),
                },
            )
            matching = publish_matching(root / "matching-v1")
            forged = artifact.ArtifactRef(
                path=matching.path,
                kind=matching.kind,
                dataset=matching.dataset,
                version=matching.version,
                manifest_digest="0" * 64,
                content_digest=matching.content_digest,
            )
            trimmed = publish_catalog(
                root / "trimmed-v1", upstreams=(full, forged), config={})

            with self.assertRaisesRegex(
                    lineage.CatalogLineageError,
                    "catalog provenance ArtifactRef does not match"):
                lineage.require_passed_source_coverage(trimmed)

    def test_recorded_parent_must_match_exact_artifact_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = publish_catalog(
                root / "full-v1",
                config={
                    "schema": schema.FULL_ARTIFACT_SCHEMA,
                    "source_coverage": coverage(),
                },
            )
            forged = artifact.ArtifactRef(
                path=full.path,
                kind=full.kind,
                dataset=full.dataset,
                version=full.version,
                manifest_digest="0" * 64,
                content_digest=full.content_digest,
            )
            derived = publish_catalog(
                root / "trimmed-v1", upstreams=(forged,), config={})
            with self.assertRaisesRegex(
                    lineage.CatalogLineageError, "ArtifactRef does not match"):
                lineage.require_passed_source_coverage(derived)

    def test_cross_dataset_and_ambiguous_lineage_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = publish_catalog(
                root / "other-v1",
                dataset="other_dataset",
                config={
                    "schema": schema.FULL_ARTIFACT_SCHEMA,
                    "source_coverage": coverage(),
                },
            )
            cross_dataset = publish_catalog(
                root / "trimmed-v1", upstreams=(other,), config={})
            with self.assertRaisesRegex(
                    lineage.CatalogLineageError, "crosses datasets"):
                lineage.require_passed_source_coverage(cross_dataset)

            full = publish_catalog(
                root / "full-v1",
                config={
                    "schema": schema.FULL_ARTIFACT_SCHEMA,
                    "source_coverage": coverage(),
                },
            )
            branched = publish_catalog(
                root / "trimmed-v2", upstreams=(full, other), config={})
            with self.assertRaisesRegex(
                    lineage.CatalogLineageError, "exactly one"):
                lineage.require_passed_source_coverage(branched)

    def test_untyped_terminal_catalog_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog_ref = publish_catalog(Path(tmp) / "catalog-v1", config={})
            with self.assertRaisesRegex(
                    lineage.CatalogLineageError, "exactly one"):
                lineage.require_passed_source_coverage(catalog_ref)

    def test_cycle_is_rejected_before_revisiting_artifact(self):
        digest_a = "a" * 64
        digest_b = "b" * 64
        ref_a = artifact.ArtifactRef(
            path="catalog-a", kind=paths_lib.CATALOGS, dataset=DATASET,
            version="a", manifest_digest=digest_a, content_digest=digest_a)
        ref_b = artifact.ArtifactRef(
            path="catalog-b", kind=paths_lib.CATALOGS, dataset=DATASET,
            version="b", manifest_digest=digest_b, content_digest=digest_b)
        manifests = {
            "catalog-a": artifact.ArtifactManifest(
                kind=paths_lib.CATALOGS, dataset=DATASET, version="a",
                generator="test", git_commit="test", created="now",
                arguments=(), content_digest=digest_a, upstreams=(ref_b,),
                config={}, declared_outputs=("catalog.feather",)),
            "catalog-b": artifact.ArtifactManifest(
                kind=paths_lib.CATALOGS, dataset=DATASET, version="b",
                generator="test", git_commit="test", created="now",
                arguments=(), content_digest=digest_b, upstreams=(ref_a,),
                config={}, declared_outputs=("catalog.feather",)),
        }
        references = {"catalog-a": ref_a, "catalog-b": ref_b}
        with mock.patch.object(
                lineage.artifact, "open_artifact",
                side_effect=lambda path, **unused: references[path]), mock.patch.object(
                    lineage.artifact, "load_manifest",
                    side_effect=lambda path: manifests[path]):
            with self.assertRaisesRegex(
                    lineage.CatalogLineageError, "cycle"):
                lineage.require_passed_source_coverage(ref_a)


if __name__ == "__main__":
    unittest.main()
