"""Guards on the two hand-written exclusion lists.

Both lists are maintained by hand, so each needs something that reads the real
key set back and complains. The asymmetry they encode -- forget an exclusion
and you over-invalidate, forget an inclusion and you use a stale artifact --
is the reason they run default-include, and these tests are what keep the
default from being quietly inverted later.
"""

import unittest

from experimental.overhead_matching.swag.farfield import (
    identity_inputs,
    pipeline,
)

# Read from `pipeline`, never copied: a second list here could go stale
# against the first, and a stale key set is exactly what makes an audit
# reassuring and wrong. `_validate_build_inputs` self-checks against this
# constant, so it cannot drift from what a build actually records either.
RECORDED_INPUTS = pipeline.RECORDED_INPUT_KEYS


def sample_inputs():
    return {key: ("f" * 64 if key.endswith(("sha256", "digest")) else "/x")
            for key in sorted(RECORDED_INPUTS)}


class ExclusionAuditTest(unittest.TestCase):
    def test_every_recorded_input_is_accounted_for(self):
        """The guard that matters: add an input to a build and forget to place
        it, and this fails rather than an artifact silently outliving it."""
        consumed = frozenset(
            key for key in RECORDED_INPUTS
            if any(key not in spec.inputs_not_consumed
                   for spec in pipeline.STAGE_SPECS.values()))
        findings = identity_inputs.audit(RECORDED_INPUTS, consumed)
        self.assertEqual(findings, [], "\n".join(findings))

    def test_no_stage_excludes_an_input_that_is_not_recorded(self):
        """A stale exclusion is harmless today and wrong the moment the key
        returns under a different meaning."""
        for stage, spec in pipeline.STAGE_SPECS.items():
            with self.subTest(stage=stage):
                self.assertEqual(
                    sorted(set(spec.inputs_not_consumed) - RECORDED_INPUTS),
                    [], f"{stage} excludes unrecorded inputs")

    def test_every_global_exclusion_states_a_reason(self):
        for key, reason in identity_inputs.GLOBAL_EXCLUSIONS.items():
            with self.subTest(key=key):
                self.assertTrue(reason.strip(), key)

    def test_every_excluded_path_has_an_included_digest_twin(self):
        """Excluding a path is only safe because its bytes still count."""
        for key in ("sam2_checkpoint", "motion_source", "video"):
            twin = f"{key}_sha256"
            self.assertIn(twin, RECORDED_INPUTS)
            self.assertNotIn(twin, identity_inputs.GLOBAL_EXCLUSIONS)
        self.assertIn("nominal_forward_sha256", RECORDED_INPUTS)
        self.assertNotIn("nominal_forward_sha256",
                         identity_inputs.GLOBAL_EXCLUSIONS)


class ContributingTest(unittest.TestCase):
    def test_an_unfamiliar_key_is_included_by_default(self):
        """The whole design in one assertion. A build that starts recording
        something new is covered before anyone edits this file."""
        inputs = sample_inputs() | {"brand_new_thing_sha256": "a" * 64}
        self.assertIn("brand_new_thing_sha256",
                      identity_inputs.contributing(inputs))

    def test_global_exclusions_are_dropped(self):
        contributed = identity_inputs.contributing(sample_inputs())
        for key in identity_inputs.GLOBAL_EXCLUSIONS:
            self.assertNotIn(key, contributed)

    def test_stage_exclusions_are_dropped_on_top(self):
        contributed = identity_inputs.contributing(
            sample_inputs(), ("sam2_checkpoint_sha256",))
        self.assertNotIn("sam2_checkpoint_sha256", contributed)
        self.assertIn("motion_source_sha256", contributed)

    def test_excluding_an_unrecorded_key_is_refused(self):
        with self.assertRaises(identity_inputs.IdentityInputError):
            identity_inputs.contributing(sample_inputs(), ("no_such_input",))

    def test_the_result_is_ordered_so_the_digest_is_stable(self):
        contributed = identity_inputs.contributing(sample_inputs())
        self.assertEqual(list(contributed), sorted(contributed))


class StageCoverageTest(unittest.TestCase):
    """What each stage's exclusions mean, asserted in the terms of the system.

    These read as documentation on purpose: they are the claims a reviewer has
    to check, and a claim nobody can find is a claim nobody checks.
    """

    def consumed(self, stage):
        return set(RECORDED_INPUTS) - set(
            pipeline.STAGE_SPECS[stage].inputs_not_consumed)

    def test_extraction_does_not_depend_on_the_mount_calibration(self):
        """Correcting a 180-degree mount error must not re-bill extraction."""
        self.assertNotIn("nominal_forward_sha256", self.consumed("extract"))

    def test_extraction_does_not_depend_on_the_tracker_weights(self):
        self.assertNotIn("sam2_checkpoint_sha256", self.consumed("extract"))

    def test_tracking_does_depend_on_the_weights_and_the_video(self):
        """Swap the checkpoint bytes under the same path and tracks MUST be
        rebuilt. This is the case the old per-stage config digest missed: the
        config records the checkpoint's PATH, never its bytes."""
        self.assertIn("sam2_checkpoint_sha256", self.consumed("track"))
        self.assertIn("video_sha256", self.consumed("track"))

    def test_localization_inputs_depends_on_the_calibration(self):
        """It is the stage that rotates camera bearings into the world."""
        self.assertIn("nominal_forward_sha256",
                      self.consumed("localization_inputs"))
        self.assertIn("motion_source_sha256",
                      self.consumed("localization_inputs"))

    def test_matching_depends_on_the_catalog(self):
        self.assertIn("catalog_content_digest", self.consumed("match"))

    def test_the_filter_reads_only_its_typed_upstream(self):
        for key in ("sam2_checkpoint_sha256", "motion_source_sha256",
                    "nominal_forward_sha256", "video_sha256",
                    "catalog_content_digest"):
            self.assertNotIn(key, self.consumed("localize"), key)

    def test_the_dataset_bytes_are_a_determinant_for_every_stage(self):
        """No stage may opt out of the frozen dataset it was built from."""
        for stage in pipeline.STAGE_SPECS:
            with self.subTest(stage=stage):
                self.assertIn("dataset_panorama_sha256", self.consumed(stage))


if __name__ == "__main__":
    unittest.main()
