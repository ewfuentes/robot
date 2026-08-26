import concurrent.futures as cf
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import code_provenance


class DigestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_canonical_json_is_order_independent(self):
        left = {"b": [2, 1], "a": {"value": "café"}}
        right = {"a": {"value": "café"}, "b": [2, 1]}
        self.assertEqual(artifact.canonical_json_bytes(left),
                         artifact.canonical_json_bytes(right))
        self.assertEqual(artifact.sha256_json(left), artifact.sha256_json(right))

    def test_canonical_json_rejects_non_json_and_non_finite_values(self):
        with self.assertRaises(artifact.ArtifactValidationError):
            artifact.canonical_json_bytes({"bad": Path("x")})
        with self.assertRaises(artifact.ArtifactValidationError):
            artifact.canonical_json_bytes({"bad": float("nan")})

    def test_directory_digest_covers_names_and_content_but_not_manifest(self):
        (self.root / "nested").mkdir()
        (self.root / "nested" / "result.jsonl").write_bytes(b"one\n")
        before = artifact.sha256_directory(self.root)
        (self.root / artifact.MANIFEST_NAME).write_text("ignored")
        self.assertEqual(artifact.sha256_directory(self.root), before)
        (self.root / "nested" / "result.jsonl").write_bytes(b"two\n")
        self.assertNotEqual(artifact.sha256_directory(self.root), before)

    def test_directory_digest_rejects_symlinks(self):
        source = self.root / "source"
        source.write_bytes(b"payload")
        (self.root / "link").symlink_to(source)
        with self.assertRaises(artifact.ArtifactValidationError):
            artifact.sha256_directory(self.root)


class AtomicWriteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_json_write_replaces_file_and_leaves_no_temporary_sibling(self):
        output = self.root / "state.json"
        output.write_text("old")
        artifact.atomic_write_json(output, {"b": 2, "a": 1})
        self.assertEqual(json.loads(output.read_text()), {"a": 1, "b": 2})
        self.assertEqual(list(self.root.glob(".state.json.*.tmp")), [])

    def test_json_create_never_replaces_an_existing_file_or_symlink(self):
        output = self.root / "recipe.json"
        artifact.atomic_create_json(output, {"owner": "first"})
        with self.assertRaises(FileExistsError):
            artifact.atomic_create_json(output, {"owner": "second"})
        self.assertEqual(json.loads(output.read_text()), {"owner": "first"})
        self.assertEqual(list(self.root.glob(".recipe.json.*.tmp")), [])

        output.unlink()
        target = self.root / "target.json"
        target.write_text("do not replace")
        output.symlink_to(target)
        with self.assertRaises(FileExistsError):
            artifact.atomic_create_json(output, {"owner": "third"})
        self.assertEqual(target.read_text(), "do not replace")


class ManifestValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root / "artifact"

    def tearDown(self):
        self.tmp.cleanup()

    def publish(self, *, version="v1", upstreams=()):
        with artifact.ArtifactDirectoryBuilder(
                self.output,
                kind="matches",
                dataset="harbor",
                version=version,
                generator="artifact_test.publish",
                git_commit="abc123",
                arguments=("--threshold", "0.7"),
                upstreams=upstreams,
                config={"threshold": 0.7, "nested": {"enabled": True}},
                declared_outputs=["reports/coverage.json", "matches.jsonl"],
        ) as builder:
            coverage = builder.output_path("reports/coverage.json")
            artifact.atomic_write_json(coverage, {"complete": True})
            builder.output_path("matches.jsonl").write_bytes(b"{}\n")
        return builder

    def test_context_publishes_valid_typed_manifest(self):
        builder = self.publish()
        self.assertFalse(builder.staging_dir.exists())
        self.assertTrue(self.output.is_dir())
        reference = artifact.validate_artifact(
            self.output,
            expected_kind="matches",
            expected_dataset="harbor",
            expected_version="v1",
        )
        self.assertEqual(reference, builder.artifact_ref)
        manifest = artifact.load_manifest(self.output)
        self.assertEqual(manifest.schema, artifact.SCHEMA)
        self.assertEqual(manifest.generator, "artifact_test.publish")
        self.assertEqual(manifest.git_commit, "abc123")
        self.assertEqual(manifest.arguments, ("--threshold", "0.7"))
        self.assertTrue(manifest.created)
        self.assertEqual(manifest.declared_outputs,
                         ("matches.jsonl", "reports/coverage.json"))
        self.assertTrue(manifest.complete)

    def test_upstream_path_is_informational_not_identity(self):
        digest = "a" * 64
        first = artifact.ArtifactRef(
            kind="tracks", dataset="harbor", version="v3",
            manifest_digest="b" * 64, content_digest=digest,
            path="/first/location")
        moved = artifact.ArtifactRef(
            kind="tracks", dataset="harbor", version="v3",
            manifest_digest="b" * 64, content_digest=digest,
            path="/moved/location")
        self.assertEqual(first, moved)

    def test_upstream_identity_round_trips(self):
        upstream = artifact.ArtifactRef(
            kind="tracks", dataset="harbor", version="v3",
            manifest_digest="b" * 64, content_digest="a" * 64,
            path="/artifacts/tracks/harbor/v3")
        self.publish(upstreams=[upstream])
        manifest = artifact.load_manifest(self.output)
        self.assertEqual(manifest.upstreams, (upstream,))

    def test_duplicate_upstream_identity_is_rejected(self):
        upstream = artifact.ArtifactRef(
            kind="tracks", dataset="harbor", version="v3",
            manifest_digest="b" * 64, content_digest="a" * 64,
            path="/artifacts/tracks/harbor/v3")
        with self.assertRaisesRegex(
                artifact.ArtifactValidationError, "identities must be unique"):
            self.publish(upstreams=[upstream, upstream])

    def test_expected_identity_mismatch_is_rejected(self):
        self.publish()
        with self.assertRaisesRegex(
                artifact.ArtifactValidationError, "kind mismatch"):
            artifact.validate_artifact(self.output, expected_kind="audits")

    def test_content_tampering_is_rejected(self):
        self.publish()
        (self.output / "matches.jsonl").write_bytes(b"changed\n")
        with self.assertRaisesRegex(
                artifact.ArtifactValidationError, "content digest mismatch"):
            artifact.validate_artifact(self.output)

    def test_missing_and_undeclared_outputs_are_rejected(self):
        self.publish()
        (self.output / "matches.jsonl").unlink()
        with self.assertRaisesRegex(
                artifact.ArtifactValidationError, "missing declared outputs"):
            artifact.validate_artifact(self.output)

        (self.output / "matches.jsonl").write_bytes(b"{}\n")
        (self.output / "surprise.txt").write_text("extra")
        with self.assertRaisesRegex(
                artifact.ArtifactValidationError, "undeclared outputs"):
            artifact.validate_artifact(self.output)

    def test_manifest_requires_exact_schema(self):
        self.publish()
        manifest_path = self.output / artifact.MANIFEST_NAME
        document = json.loads(manifest_path.read_text())
        document["unexpected"] = 1
        artifact.atomic_write_json(manifest_path, document)
        with self.assertRaisesRegex(
                artifact.ArtifactValidationError, "unknown"):
            artifact.load_manifest(self.output)

    def test_manifest_rejects_false_completion_and_bad_digest(self):
        self.publish()
        manifest_path = self.output / artifact.MANIFEST_NAME
        document = json.loads(manifest_path.read_text())
        document["complete"] = False
        artifact.atomic_write_json(manifest_path, document)
        with self.assertRaisesRegex(
                artifact.ArtifactValidationError, "complete=true"):
            artifact.load_manifest(self.output)

        document["complete"] = True
        document["content_digest"] = "not-a-digest"
        artifact.atomic_write_json(manifest_path, document)
        with self.assertRaisesRegex(
                artifact.ArtifactValidationError, "SHA-256"):
            artifact.load_manifest(self.output)

    def test_duplicate_json_keys_are_rejected(self):
        self.output.mkdir()
        (self.output / artifact.MANIFEST_NAME).write_text(
            '{"schema":"x","schema":"y"}')
        with self.assertRaisesRegex(
                artifact.ArtifactValidationError, "duplicate JSON"):
            artifact.load_manifest(self.output)


class TransactionFailureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root / "artifact"

    def tearDown(self):
        self.tmp.cleanup()

    def builder(self):
        return artifact.ArtifactDirectoryBuilder(
            self.output,
            kind="tracks",
            dataset="harbor",
            version="v1",
            generator="artifact_test.builder",
            git_commit="abc123",
            arguments=(),
            config={},
            declared_outputs=["tracks.json"],
        )

    def test_body_failure_preserves_incomplete_directory(self):
        with self.assertRaisesRegex(RuntimeError, "simulated crash"):
            with self.builder() as builder:
                builder.output_path("tracks.json").write_text("partial")
                raise RuntimeError("simulated crash")
        self.assertFalse(self.output.exists())
        self.assertTrue(Path(str(self.output) + ".incomplete").is_dir())

    def test_missing_output_preserves_incomplete_and_has_no_manifest(self):
        with self.assertRaisesRegex(
                artifact.ArtifactValidationError, "missing declared outputs"):
            with self.builder():
                pass
        staging = Path(str(self.output) + ".incomplete")
        self.assertTrue(staging.is_dir())
        self.assertFalse((staging / artifact.MANIFEST_NAME).exists())

    def test_incomplete_artifact_cannot_be_loaded_or_validated(self):
        with self.assertRaises(RuntimeError):
            with self.builder() as builder:
                builder.output_path("tracks.json").write_text("partial")
                raise RuntimeError("stop")
        staging = Path(str(self.output) + ".incomplete")
        digest = artifact.sha256_directory(staging)
        manifest = artifact.ArtifactManifest(
            kind="tracks", dataset="harbor", version="v1",
            generator="artifact_test.manual", git_commit="abc123",
            created="2026-08-23T00:00:00+00:00", arguments=(),
            content_digest=digest, upstreams=(), config={},
            declared_outputs=("tracks.json",))
        artifact.atomic_write_json(staging / artifact.MANIFEST_NAME,
                                   manifest.to_dict())
        with self.assertRaisesRegex(
                artifact.ArtifactValidationError, "incomplete artifact"):
            artifact.load_manifest(staging)
        with self.assertRaisesRegex(
                artifact.ArtifactValidationError, "incomplete artifact"):
            artifact.validate_artifact(staging)

    def test_existing_final_or_incomplete_is_never_reused(self):
        self.output.mkdir()
        with self.assertRaisesRegex(artifact.ArtifactExistsError, "completed"):
            with self.builder():
                pass
        self.output.rmdir()
        Path(str(self.output) + ".incomplete").mkdir()
        with self.assertRaisesRegex(artifact.ArtifactExistsError, "incomplete"):
            with self.builder():
                pass

    def test_concurrent_directory_publication_never_clobbers_winner(self):
        left = self.root / "left.incomplete"
        right = self.root / "right.incomplete"
        left.mkdir()
        right.mkdir()
        (left / "owner.txt").write_text("left")
        (right / "owner.txt").write_text("right")
        barrier = threading.Barrier(2)

        def publish(staging):
            barrier.wait()
            try:
                artifact.publish_directory_no_clobber(staging, self.output)
                return "published"
            except artifact.ArtifactExistsError:
                return "exists"

        with cf.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(publish, (left, right)))
        self.assertCountEqual(results, ["published", "exists"])
        self.assertIn(
            (self.output / "owner.txt").read_text(), ("left", "right"))
        remaining = [path for path in (left, right) if path.exists()]
        self.assertEqual(len(remaining), 1)

    def test_generic_directory_publication_flushes_tree_before_rename(self):
        staging = self.root / "diagnostics.incomplete"
        staging.mkdir()
        (staging / "viewer.html").write_text("complete")
        events = []
        real_rename = artifact.os.rename

        def flush_tree(path):
            self.assertEqual(path, staging)
            self.assertTrue((path / "viewer.html").is_file())
            events.append("tree-flushed")

        def rename(source, destination):
            self.assertEqual(events, ["tree-flushed"])
            events.append("renamed")
            real_rename(source, destination)

        with mock.patch.object(
                artifact, "_fsync_artifact_tree",
                side_effect=flush_tree) as flush_mock, \
                mock.patch.object(artifact.os, "rename",
                                  side_effect=rename):
            artifact.publish_directory_no_clobber(staging, self.output)

        self.assertEqual(events, ["tree-flushed", "renamed"])
        flush_mock.assert_called_once_with(staging)
        self.assertEqual((self.output / "viewer.html").read_text(),
                         "complete")

    def test_output_path_must_have_been_declared(self):
        with self.builder() as builder:
            with self.assertRaisesRegex(
                    artifact.ArtifactValidationError, "not declared"):
                builder.output_path("other.json")
            builder.output_path("tracks.json").write_text("[]")

    def test_declared_outputs_must_be_normal_relative_paths(self):
        for bad_path in ("/absolute.json", "../escape.json", ".", "a//b"):
            with self.subTest(bad_path=bad_path):
                with self.assertRaises(artifact.ArtifactValidationError):
                    artifact.ArtifactDirectoryBuilder(
                        self.output,
                        kind="tracks",
                        dataset="harbor",
                        version="v1",
                        generator="artifact_test.invalid_output",
                        git_commit="abc123",
                        arguments=(),
                        config={},
                        declared_outputs=[bad_path],
                    )


class ManifestDigestExcludesIdentityTest(unittest.TestCase):
    """`manifest_digest` must not move when an identity is added.

    This is what let the 56 legacy artifacts be signed in place. Their
    manifest digests are recorded by downstream refs AND baked into
    downstream artifacts' immutable content (frozen request sets and work
    snapshots, covered by `content_digest`), so a digest that moved could not
    be chased -- 32 of the 56 were unreachable that way. If this test ever
    fails, signing an artifact has become a breaking change to everything
    that points at it.
    """

    def _manifest(self, **extra):
        base = {
            "schema": artifact.SCHEMA,
            "kind": "object_tracks", "dataset": "ds", "version": "v1",
            "generator": "test", "git_commit": "deadbeef",
            "created": "2026-08-26", "arguments": [],
            "content_digest": "c" * 64, "upstreams": [], "config": {},
            "declared_outputs": [], "complete": True,
        }
        base.update(extra)
        return base

    def test_adding_an_identity_does_not_move_the_digest(self):
        without = self._manifest()
        with_identity = self._manifest(artifact_identity="a" * 64)
        self.assertEqual(
            artifact.manifest_digest_of_document(without),
            artifact.manifest_digest_of_document(with_identity))

    def test_changing_the_identity_does_not_move_the_digest(self):
        self.assertEqual(
            artifact.manifest_digest_of_document(
                self._manifest(artifact_identity="a" * 64)),
            artifact.manifest_digest_of_document(
                self._manifest(artifact_identity="b" * 64)))

    def test_changing_anything_else_does_move_the_digest(self):
        self.assertNotEqual(
            artifact.manifest_digest_of_document(self._manifest()),
            artifact.manifest_digest_of_document(
                self._manifest(content_digest="d" * 64)))

    def test_it_matches_the_file_bytes_a_manifest_is_written_with(self):
        """Backward compatibility with every digest already recorded.

        Manifests are written by `atomic_write_json`, so a manifest with no
        identity hashes to exactly what a digest over the file's bytes gave
        before this changed.
        """
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / artifact.MANIFEST_NAME
            document = self._manifest()
            artifact.atomic_write_json(path, document)
            self.assertEqual(artifact.manifest_digest(path),
                             artifact.sha256_file(path))
            artifact.atomic_write_json(
                path, self._manifest(artifact_identity="a" * 64))
            self.assertEqual(artifact.manifest_digest(path),
                             artifact.manifest_digest_of_document(document))
            self.assertNotEqual(artifact.manifest_digest(path),
                                artifact.sha256_file(path))


class CodeProvenanceStampTest(unittest.TestCase):
    """Stamped by the builder, so no producer can forget it.

    There are nine producers. A producer that forgot would be
    indistinguishable from one whose code was genuinely never recorded, which
    is the distinction the record exists to make.
    """

    def test_a_published_artifact_records_its_code_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "v1"
            with artifact.ArtifactDirectoryBuilder(
                    target, kind="object_tracks", dataset="ds", version="v1",
                    generator="test", arguments=(), upstreams=(), config={},
                    declared_outputs=("payload.txt",)) as builder:
                builder.output_path("payload.txt").write_text("x")
            manifest = artifact.load_manifest(target)
            self.assertIsNotNone(manifest.code_provenance)
            block = code_provenance.validate(manifest.code_provenance)
            self.assertEqual(block["schema"], code_provenance.SCHEMA)

    def test_a_manifest_without_it_still_reads(self):
        """Every artifact on disk predates this field. Refusing to read one
        for lacking it would strand the corpus for no gain -- nothing depends
        on it being there."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "v1"
            with artifact.ArtifactDirectoryBuilder(
                    target, kind="object_tracks", dataset="ds", version="v1",
                    generator="test", arguments=(), upstreams=(), config={},
                    declared_outputs=("payload.txt",)) as builder:
                builder.output_path("payload.txt").write_text("x")
            path = target / artifact.MANIFEST_NAME
            document = json.loads(path.read_text())
            del document["code_provenance"]
            path.write_text(json.dumps(document))
            manifest = artifact.load_manifest(target)
            self.assertIsNone(manifest.code_provenance)

    def test_a_non_object_code_block_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "v1"
            with artifact.ArtifactDirectoryBuilder(
                    target, kind="object_tracks", dataset="ds", version="v1",
                    generator="test", arguments=(), upstreams=(), config={},
                    declared_outputs=("payload.txt",)) as builder:
                builder.output_path("payload.txt").write_text("x")
            path = target / artifact.MANIFEST_NAME
            document = json.loads(path.read_text())
            document["code_provenance"] = "not an object"
            path.write_text(json.dumps(document))
            with self.assertRaises(artifact.ArtifactValidationError):
                artifact.load_manifest(target)

    def test_a_manifest_without_it_round_trips_to_its_own_bytes(self):
        """Omitted, not written as null: an older artifact re-serialized must
        equal what it came from, or a re-read would look like a change."""
        manifest = artifact.ArtifactManifest(
            kind="object_tracks", dataset="ds", version="v1",
            generator="test", git_commit="c0ffee", created="2026-08-25",
            arguments=(), content_digest="a" * 64, upstreams=(), config={},
            declared_outputs=())
        self.assertNotIn("code_provenance", manifest.to_dict())


if __name__ == "__main__":
    unittest.main()
