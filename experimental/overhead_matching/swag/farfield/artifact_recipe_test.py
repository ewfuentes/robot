import unittest

from experimental.overhead_matching.swag.farfield import (
    artifact,
    artifact_identity,
    artifact_recipe as subject,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64

STAGE_CONFIG = {
    "artifacts.object_tracks_version": "v1",
    "tracking.drift_gate_px": 150.0,
    "tracking.patience": 3,
}
BUILD_INPUTS = {
    "dataset_panorama_sha256": DIGEST_A,
    "sam2_checkpoint_sha256": DIGEST_B,
}


def ref(kind, *, manifest_digest=DIGEST_A):
    return artifact.ArtifactRef(
        kind=kind, dataset="ds", version="v1",
        manifest_digest=manifest_digest, content_digest=DIGEST_C,
        path=f"/root/artifacts/{kind}/ds/v1")


def manifest(*, recipe=None, identity=None, upstreams=()):
    return artifact.ArtifactManifest(
        kind="object_tracks", dataset="ds", version="v1",
        generator="test", git_commit="deadbeef", created="2026-08-26",
        arguments=(), content_digest=DIGEST_C, upstreams=tuple(upstreams),
        config={}, declared_outputs=(),
        artifact_identity=identity, recipe=recipe)


def recipe(stage="track", stage_config=None, build_inputs=None,
           identity_upstreams=None, inputs_not_consumed=()):
    return subject.build(
        stage=stage,
        stage_config=STAGE_CONFIG if stage_config is None else stage_config,
        build_inputs=BUILD_INPUTS if build_inputs is None else build_inputs,
        identity_upstreams=((ref("frame_landmarks"),)
                            if identity_upstreams is None
                            else identity_upstreams),
        inputs_not_consumed=inputs_not_consumed)


class SelfDescribingTest(unittest.TestCase):
    """The whole point: a manifest must carry every term its identity needs.

    This is stronger than checking that the recipe round-trips, because it
    cannot be satisfied by a manifest that is missing something -- a dropped
    term changes the recomputed digest.
    """

    def _signed(self, **kwargs):
        upstreams = kwargs.pop("upstreams", (ref("frame_landmarks"),))
        block = kwargs.pop("recipe", recipe())
        identity = artifact_identity.compute(
            kind="object_tracks", dataset="ds",
            stage_config_digest=subject.stage_config_digest(block),
            upstreams=upstreams,
            build_inputs=block["build_inputs"])
        return manifest(recipe=block, identity=identity, upstreams=upstreams)

    def test_a_manifest_with_a_recipe_verifies(self):
        subject.verify_self_describing(self._signed())

    def test_the_recomputed_identity_is_the_recorded_one(self):
        document = self._signed()
        self.assertEqual(subject.identity_from_manifest(document),
                         artifact_identity.recorded(document))

    def test_a_manifest_with_no_recipe_cannot_be_verified(self):
        document = manifest(identity=DIGEST_A)
        with self.assertRaisesRegex(subject.ArtifactRecipeError,
                                    "records no recipe"):
            subject.verify_self_describing(document)

    def test_an_unattributed_manifest_has_nothing_to_verify_against(self):
        with self.assertRaisesRegex(subject.ArtifactRecipeError,
                                    "records no identity"):
            subject.verify_self_describing(manifest(recipe=recipe()))

    def test_a_dropped_stage_config_key_is_caught(self):
        """The failure this exists to catch: a producer records less than the
        identity depends on."""
        document = self._signed()
        thinner = dict(document.recipe)
        thinner["stage_config"] = {
            key: value for key, value in STAGE_CONFIG.items()
            if key != "tracking.patience"}
        with self.assertRaisesRegex(subject.ArtifactRecipeError,
                                    "not self-describing"):
            subject.verify_self_describing(
                manifest(recipe=thinner,
                         identity=document.artifact_identity,
                         upstreams=document.upstreams))

    def test_a_dropped_build_input_is_caught(self):
        document = self._signed()
        thinner = dict(document.recipe)
        thinner["build_inputs"] = {
            key: value for key, value in BUILD_INPUTS.items()
            if key != "sam2_checkpoint_sha256"}
        with self.assertRaisesRegex(subject.ArtifactRecipeError,
                                    "not self-describing"):
            subject.verify_self_describing(
                manifest(recipe=thinner,
                         identity=document.artifact_identity,
                         upstreams=document.upstreams))

    def test_an_upstream_the_manifest_no_longer_shows_is_caught(self):
        """The recipe names an identity upstream; the manifest must show it.

        Repointing the manifest's lineage without repointing the recipe is the
        shape of a hand-edited artifact, and it fails on the subset rule
        before the digests are even compared."""
        document = self._signed()
        with self.assertRaisesRegex(subject.ArtifactRecipeError,
                                    "manifest does not record"):
            subject.verify_self_describing(
                manifest(recipe=document.recipe,
                         identity=document.artifact_identity,
                         upstreams=(ref("frame_landmarks",
                                        manifest_digest=DIGEST_B),)))


class RecipeShapeTest(unittest.TestCase):
    def test_build_inputs_are_stored_already_reduced(self):
        """So a reader needs no exclusion list to recompute the identity."""
        block = recipe(inputs_not_consumed=("sam2_checkpoint_sha256",))
        self.assertEqual(set(block["build_inputs"]),
                         {"dataset_panorama_sha256"})

    def test_path_valued_inputs_never_enter(self):
        """Globally excluded, because a path is not a value."""
        block = recipe(build_inputs=dict(
            BUILD_INPUTS, farfield_root="/data/farfield_matching"))
        self.assertNotIn("farfield_root", block["build_inputs"])

    def test_stage_config_is_stored_sorted_and_verbatim(self):
        block = recipe()
        self.assertEqual(block["stage_config"], STAGE_CONFIG)
        self.assertEqual(list(block["stage_config"]), sorted(STAGE_CONFIG))

    def test_the_digest_matches_what_the_stage_contract_would_compute(self):
        self.assertEqual(subject.stage_config_digest(recipe()),
                         artifact.sha256_json(STAGE_CONFIG))

    def test_a_malformed_recipe_is_refused(self):
        for bad in ({}, {"schema": "wrong", "stage": "track",
                         "stage_config": {}, "build_inputs": {},
                         "identity_upstreams": []},
                    {"schema": subject.SCHEMA, "stage": "track",
                     "stage_config": "not-an-object", "build_inputs": {},
                     "identity_upstreams": []},
                    {"schema": subject.SCHEMA, "stage": "track",
                     "stage_config": {}, "build_inputs": {},
                     "identity_upstreams": "not-a-list"}):
            with self.assertRaises(subject.ArtifactRecipeError):
                subject.validate(bad)

    def test_no_recipe_file_means_no_recipe_not_an_error(self):
        self.assertIsNone(subject.load(None))

    def test_an_unreadable_recipe_file_is_an_error(self):
        with self.assertRaisesRegex(subject.ArtifactRecipeError,
                                    "cannot read artifact recipe"):
            subject.load("/nonexistent/recipe.json")


class IdentityUpstreamsTest(unittest.TestCase):
    """Identity uses the stage's CONFIGURED upstreams, which is a subset of
    the manifest's lineage.

    `frame_landmarks` records its pinhole artifact and the canonical LLM
    result artifact, but `extract` declares no artifact upstreams, and the
    orchestrator could not name the result artifact anyway -- it does not
    exist until the stage has run. So the recipe records which upstreams
    entered the identity, and that set must be a subset of what the manifest
    shows.
    """

    def test_the_recorded_subset_is_what_recomputes(self):
        block = recipe(identity_upstreams=())
        identity = artifact_identity.compute(
            kind="object_tracks", dataset="ds",
            stage_config_digest=subject.stage_config_digest(block),
            upstreams=(), build_inputs=block["build_inputs"])
        document = manifest(recipe=block, identity=identity,
                            upstreams=(ref("frame_landmarks"),))
        # The manifest records lineage the identity does not use, exactly as a
        # real frame_landmarks artifact does. That must still verify.
        subject.verify_self_describing(document)

    def test_a_recipe_cannot_invent_lineage(self):
        block = recipe(identity_upstreams=(ref("frame_landmarks",
                                              manifest_digest=DIGEST_B),))
        document = manifest(recipe=block, identity=DIGEST_A,
                            upstreams=(ref("frame_landmarks"),))
        with self.assertRaisesRegex(subject.ArtifactRecipeError,
                                    "manifest does not record"):
            subject.verify_self_describing(document)


    def test_a_manifest_cannot_list_an_upstream_twice(self):
        """Why `identity_from_manifest` can select by set rather than
        consuming one ref per recorded digest: a duplicate cannot exist, so a
        digest cannot be counted twice."""
        with self.assertRaisesRegex(artifact.ArtifactValidationError,
                                    "upstream identities must be unique"):
            manifest(upstreams=(ref("catalogs"), ref("catalogs")))


class DescribeTest(unittest.TestCase):
    def test_it_names_the_settings_and_the_inputs(self):
        text = subject.describe(
            manifest(recipe=recipe(), identity=DIGEST_A,
                     upstreams=(ref("frame_landmarks"),)))
        self.assertIn("tracking.drift_gate_px", text)
        self.assertIn("dataset_panorama_sha256", text)
        self.assertIn("frame_landmarks/ds/v1", text)

    def test_it_says_so_when_there_is_no_recipe(self):
        text = subject.describe(manifest(identity=DIGEST_A))
        self.assertIn("NONE RECORDED", text)


if __name__ == "__main__":
    unittest.main()
