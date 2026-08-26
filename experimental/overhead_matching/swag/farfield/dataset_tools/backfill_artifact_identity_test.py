"""The backfill must compute, never assume, and never touch an artifact."""

import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    artifact,
    artifact_identity,
    provenance,
)
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    backfill_artifact_identity as backfill,
)

BUILD_IDENTITY = "b" * 64
STAGE_DIGEST = "c" * 64


def write_build(root: Path, identity=BUILD_IDENTITY) -> None:
    target = root / "builds" / "ds" / "b001"
    target.mkdir(parents=True, exist_ok=True)
    (target / "build_config.json").write_text(json.dumps({
        "build_identity": identity,
        "inputs": {"sam2_checkpoint_sha256": "d" * 64,
                   "dataset_panorama_sha256": "e" * 64},
    }))


def publish(root: Path, kind="object_tracks", *, config=None) -> Path:
    target = root / "artifacts" / kind / "ds" / "v1"
    with artifact.ArtifactDirectoryBuilder(
            target, kind=kind, dataset="ds", version="v1",
            generator="backfill_test", git_commit="c0ffee", arguments=(),
            upstreams=(), declared_outputs=("payload.txt",),
            config=config if config is not None else {
                "build_identity": BUILD_IDENTITY,
                "orchestration": {"schema": "farfield_pipeline_stage/v1",
                                  "stage": "track",
                                  "config_digest": STAGE_DIGEST},
            }) as builder:
        builder.output_path("payload.txt").write_text("x")
    return target


class PlanTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_an_identity_is_computed_from_the_manifest_and_the_build(self):
        write_build(self.root)
        publish(self.root)
        plan = backfill.build_plan(self.root)
        self.assertEqual(plan["by_status"], {"computed": 1})
        record = plan["artifacts"][0]
        # Not merely present -- equal to what `compute` gives for the same
        # terms. A backfill that invented a value would still be "present".
        self.assertEqual(record["identity"], artifact_identity.compute(
            kind="object_tracks", dataset="ds",
            stage_config_digest=STAGE_DIGEST, upstreams=(),
            build_inputs={"sam2_checkpoint_sha256": "d" * 64,
                          "dataset_panorama_sha256": "e" * 64},
            inputs_not_consumed=(
                "nominal_forward_sha256", "catalog_manifest_digest",
                "catalog_content_digest")))

    def test_without_a_surviving_build_nothing_is_invented(self):
        """The honest failure. An identity that cannot be computed must be
        reported as such, never filled in with a plausible value."""
        publish(self.root)
        plan = backfill.build_plan(self.root)
        self.assertEqual(plan["by_status"], {"no_surviving_build": 1})
        self.assertNotIn("identity", plan["artifacts"][0])

    def test_an_artifact_that_already_has_one_is_left_alone(self):
        write_build(self.root)
        publish(self.root, config={
            "build_identity": BUILD_IDENTITY,
            "artifact_identity": "f" * 64,
            "orchestration": {"schema": "farfield_pipeline_stage/v1",
                              "stage": "track",
                              "config_digest": STAGE_DIGEST}})
        plan = backfill.build_plan(self.root)
        self.assertEqual(plan["by_status"], {"already_attributed": 1})

    def test_a_non_stage_artifact_is_not_given_an_identity(self):
        """Catalogs and viewers are not produced by a pipeline stage, so no
        stage config or exclusion list describes them. Better to leave them
        than to mint an identity nothing would check."""
        write_build(self.root)
        publish(self.root, kind="catalogs")
        plan = backfill.build_plan(self.root)
        self.assertEqual(plan["by_status"], {"not_a_pipeline_stage_output": 1})

    def test_a_side_output_is_classified_as_one(self):
        """The survey and this tool must agree; the survey learned this the
        hard way against the real root."""
        viewer = self.root / "runs" / "260826_experiment" / "run_a.viewer"
        viewer.mkdir(parents=True)
        provenance.write(viewer, generator="viewer_test",
                         inputs={"run_dir": "/x"}, config={})
        plan = backfill.build_plan(self.root)
        self.assertEqual(plan["by_status"], {"side_output": 1})

    def test_the_plan_digest_changes_with_the_plan(self):
        write_build(self.root)
        publish(self.root)
        first = backfill.build_plan(self.root)["plan_digest"]
        publish(self.root, kind="semantic_audits")
        self.assertNotEqual(backfill.build_plan(self.root)["plan_digest"],
                            first)


class ApplyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_build(self.root)
        self.target = publish(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_no_published_artifact_is_modified(self):
        """The constraint that forced a sidecar. `manifest_digest` is the
        sha256 of `manifest.json`, and every downstream ArtifactRef records
        it, so editing a published manifest silently invalidates every
        reference to it."""
        manifest_path = self.target / artifact.MANIFEST_NAME
        before = manifest_path.read_bytes()
        backfill.apply_plan(self.root, backfill.build_plan(self.root))
        self.assertEqual(manifest_path.read_bytes(), before)

    def test_the_index_records_how_the_value_was_obtained(self):
        backfill.apply_plan(self.root, backfill.build_plan(self.root))
        document = json.loads(
            backfill.index_path(self.root).read_text())
        entry = next(iter(document["entries"].values()))
        self.assertEqual(entry["basis"],
                         "computed_from_manifest_and_surviving_build_recipe")

    def test_the_index_reads_back_as_a_path_to_identity_map(self):
        plan = backfill.build_plan(self.root)
        backfill.apply_plan(self.root, plan)
        index = backfill.load_index(self.root)
        self.assertEqual(index[str(self.target)],
                         plan["artifacts"][0]["identity"])

    def test_applying_twice_is_refused_rather_than_silently_merged(self):
        backfill.apply_plan(self.root, backfill.build_plan(self.root))
        with self.assertRaises(backfill.BackfillError):
            backfill.apply_plan(self.root, backfill.build_plan(self.root))

    def test_a_missing_index_is_empty_not_an_error(self):
        self.assertEqual(backfill.load_index(self.root), {})

    def test_a_corrupt_index_is_empty_not_an_error(self):
        backfill.index_path(self.root).write_text("{not json")
        self.assertEqual(backfill.load_index(self.root), {})


if __name__ == "__main__":
    unittest.main()
