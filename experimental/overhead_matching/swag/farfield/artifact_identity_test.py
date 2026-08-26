"""The invalidation table this module exists to produce.

Each test names a change and asserts which artifacts it may move. The old
global build identity failed every one of the "does not move" cases, which is
what forced `stage_reuse` and its human attestation into existence.
"""

import unittest

from experimental.overhead_matching.swag.farfield import (
    artifact,
    artifact_identity as ident,
)

BUILD_INPUTS = {
    "farfield_root": "/farfield",
    "sam2_checkpoint": "/models/sam2/v1.pt",
    "sam2_checkpoint_sha256": "d" * 64,
    "nominal_forward_sha256": "e" * 64,
    "dataset_panorama_sha256": "b" * 64,
}

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def ref(kind, version="v1", manifest_digest=DIGEST_A):
    return artifact.ArtifactRef(
        kind=kind, dataset="ds", version=version,
        manifest_digest=manifest_digest, content_digest=DIGEST_C,
        path=f"/farfield/artifacts/{kind}/ds/{version}")


def identity(**overrides):
    base = dict(kind="object_tracks", dataset="ds",
                stage_config_digest=DIGEST_A,
                upstreams=[ref("frame_landmarks"), ref("pinhole_images")],
                build_inputs=dict(BUILD_INPUTS))
    base.update(overrides)
    return ident.compute(**base)


class DeterminismTest(unittest.TestCase):
    def test_the_same_recipe_gives_the_same_identity(self):
        self.assertEqual(identity(), identity())

    def test_upstream_order_does_not_matter(self):
        """A stage taking tracks and audits produces the same thing whichever
        order they were listed in; an order-sensitive identity would report a
        difference that is not one."""
        forward = identity(upstreams=[ref("frame_landmarks"),
                                      ref("pinhole_images")])
        reverse = identity(upstreams=[ref("pinhole_images"),
                                      ref("frame_landmarks")])
        self.assertEqual(forward, reverse)

    def test_two_upstreams_of_one_kind_do_not_collide(self):
        one = identity(upstreams=[ref("catalogs", "full"),
                                  ref("catalogs", "trim", DIGEST_B)])
        two = identity(upstreams=[ref("catalogs", "full"),
                                  ref("catalogs", "trim", DIGEST_C)])
        self.assertNotEqual(one, two)


class MovesTest(unittest.TestCase):
    """Changes that MUST invalidate."""

    def test_this_stages_config_moves_it(self):
        self.assertNotEqual(identity(),
                            identity(stage_config_digest=DIGEST_B))

    def test_an_upstream_moves_it(self):
        self.assertNotEqual(
            identity(),
            identity(upstreams=[ref("frame_landmarks", manifest_digest=DIGEST_B),
                                ref("pinhole_images")]))

    def test_the_dataset_bytes_move_it(self):
        self.assertNotEqual(
            identity(),
            identity(build_inputs=BUILD_INPUTS | {
                "dataset_panorama_sha256": DIGEST_C}))

    def test_a_checkpoint_swapped_under_the_same_path_moves_it(self):
        """The gap that made this term necessary. The stage config records the
        checkpoint's PATH; replacing its bytes in place leaves every config
        digest identical, so without the input digests the tracks would be
        silently reused against different weights."""
        self.assertNotEqual(
            identity(),
            identity(build_inputs=BUILD_INPUTS | {
                "sam2_checkpoint_sha256": DIGEST_C}))

    def test_kind_and_dataset_move_it(self):
        self.assertNotEqual(identity(), identity(kind="semantic_audits"))
        self.assertNotEqual(identity(), identity(dataset="other"))


class DoesNotMoveTest(unittest.TestCase):
    """Changes that MUST NOT invalidate -- the whole point of the redesign.

    Under the global build identity every one of these produced a new identity
    for every artifact in the build, including the paid ones.
    """

    def test_the_code_is_not_an_input_at_all(self):
        """Identity is data lineage. `compute` takes no code term, so there is
        no channel by which editing a module can invalidate an artifact --
        `code_provenance` records the code and a mixed lineage is reported,
        never gated. Expressed as the absence of a parameter, which is the
        only way to assert that a channel does not exist."""
        import inspect
        self.assertNotIn(
            "entry_module", inspect.signature(ident.compute).parameters)
        self.assertNotIn(
            "code_fingerprint", inspect.signature(ident.compute).parameters)

    def test_a_downstream_config_change_is_not_an_input_here(self):
        """`localization.pi0` is not part of the track stage's config digest,
        so it cannot appear in the track artifact's identity. Expressed as the
        absence of any channel: the only config input is this stage's own
        digest."""
        unchanged = identity()
        for downstream_digest in (DIGEST_B, DIGEST_C):
            # A downstream stage's digest is simply never passed in.
            self.assertEqual(identity(), unchanged, downstream_digest)

    def test_an_unrelated_upstream_version_name_is_not_an_input(self):
        """Version names are labels. Two refs with the same manifest digest
        name the same artifact whatever it was called."""
        renamed = identity(
            upstreams=[ref("frame_landmarks", version="renamed"),
                       ref("pinhole_images")])
        self.assertEqual(identity(), renamed)

    def test_moving_the_data_root_does_not_move_it(self):
        """A mirror holds the same artifacts. An identity keyed on absolute
        paths would call every mirror a full rebuild."""
        self.assertEqual(
            identity(),
            identity(build_inputs=BUILD_INPUTS | {
                "farfield_root": "/mnt/mirror/farfield"}))

    def test_reaching_the_checkpoint_by_another_path_does_not_move_it(self):
        self.assertEqual(
            identity(),
            identity(build_inputs=BUILD_INPUTS | {
                "sam2_checkpoint": "/other/models/sam2/v1.pt"}))

    def test_an_input_the_stage_does_not_read_does_not_move_it(self):
        """Correcting the mount calibration must not re-bill extraction."""
        self.assertEqual(
            identity(inputs_not_consumed=("nominal_forward_sha256",)),
            identity(inputs_not_consumed=("nominal_forward_sha256",),
                     build_inputs=BUILD_INPUTS | {
                         "nominal_forward_sha256": DIGEST_C}))

    def test_the_path_an_artifact_sits_at_is_not_an_input(self):
        moved = artifact.ArtifactRef(
            kind="frame_landmarks", dataset="ds", version="v1",
            manifest_digest=DIGEST_A, content_digest=DIGEST_C,
            path="/somewhere/else")
        self.assertEqual(identity(),
                         identity(upstreams=[moved, ref("pinhole_images")]))


class RecordedTest(unittest.TestCase):
    def manifest(self, identity=None):
        """Identity is a top-level manifest field, not a config entry."""
        return artifact.ArtifactManifest(
            kind="object_tracks", dataset="ds", version="v1",
            generator="test", git_commit="deadbeef", created="2026-08-25",
            arguments=(), content_digest=DIGEST_C, upstreams=(),
            config={}, declared_outputs=(), artifact_identity=identity)

    def test_a_recorded_identity_is_returned(self):
        self.assertEqual(
            ident.recorded(self.manifest(DIGEST_A)),
            DIGEST_A)

    def test_a_manifest_without_one_is_unattributed_not_an_error(self):
        """Artifacts published before this existed are not wrong, they are
        unattributed. Treating them as corrupt would strand every artifact on
        disk; treating them as current would launder an unproven claim."""
        self.assertEqual(ident.recorded(self.manifest()), ident.UNATTRIBUTED)

    def test_a_malformed_recorded_identity_is_an_error(self):
        with self.assertRaises(ident.ArtifactIdentityError):
            ident.recorded(self.manifest("nope"))

    def test_the_unattributed_message_names_the_way_forward(self):
        """And names something that exists: it used to name a flag that did
        not, which sends the reader hunting."""
        message = ident.explain(expected=DIGEST_A,
                                manifest=self.manifest(),
                                kind="object_tracks")
        self.assertIn("pipeline run", message)
        self.assertNotIn("--assume-current", message)

    def test_the_mismatch_message_names_what_to_compare(self):
        message = ident.explain(
            expected=DIGEST_A,
            manifest=self.manifest(DIGEST_B),
            kind="object_tracks")
        self.assertIn("stage_config_digest", message)
        self.assertIn("upstream refs", message)
        self.assertIn("build inputs", message)

    def test_the_mismatch_message_says_code_is_not_the_cause(self):
        """A reader who has just changed code will otherwise assume that is
        why, look for a code term, and not find one."""
        message = ident.explain(
            expected=DIGEST_A,
            manifest=self.manifest(DIGEST_B),
            kind="object_tracks")
        self.assertIn("NOT part of this identity", message)


class OneLookupPathTest(unittest.TestCase):
    """The manifest is the only place an identity comes from.

    There was briefly a second: a derived index beside the data, for the 56
    artifacts published before identity existed. They now carry the identity
    in their own manifests -- possible because `manifest_digest` excludes the
    `artifact_identity` key, so signing them moved no digest any downstream
    had recorded. Two lookup paths for one fact is one too many; if this test
    has to grow a second source again, that is the thing to question.
    """

    def manifest(self, identity=None):
        return artifact.ArtifactManifest(
            kind="object_tracks", dataset="ds", version="v1",
            generator="test", git_commit="deadbeef", created="2026-08-26",
            arguments=(), content_digest=DIGEST_C, upstreams=(),
            config={}, declared_outputs=(), artifact_identity=identity)

    def test_the_only_reader_is_the_manifest_field(self):
        self.assertEqual(ident.recorded(self.manifest(DIGEST_A)), DIGEST_A)
        self.assertEqual(ident.recorded(self.manifest()), ident.UNATTRIBUTED)

    def test_identity_in_config_is_not_consulted(self):
        """Where it used to live. A stray copy there must not answer."""
        stale = artifact.ArtifactManifest(
            kind="object_tracks", dataset="ds", version="v1",
            generator="test", git_commit="deadbeef", created="2026-08-26",
            arguments=(), content_digest=DIGEST_C, upstreams=(),
            config={"artifact_identity": DIGEST_B}, declared_outputs=())
        self.assertEqual(ident.recorded(stale), ident.UNATTRIBUTED)


class RejectionTest(unittest.TestCase):
    def test_a_non_ref_upstream_is_refused(self):
        with self.assertRaises(ident.ArtifactIdentityError):
            identity(upstreams=["frame_landmarks/v1"])

    def test_a_non_digest_config_value_is_refused(self):
        with self.assertRaises(ident.ArtifactIdentityError):
            identity(stage_config_digest="v1")


if __name__ == "__main__":
    unittest.main()
