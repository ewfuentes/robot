import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experimental.overhead_matching.swag.farfield.paper import check_roster
from experimental.overhead_matching.swag.farfield.paper.table_common import (
    BATCH1_SEQUENCES, DATASET_GROUPS, SEQUENCE_ARTIFACTS,
)


class RosterTest(unittest.TestCase):
    def test_batch2_has_inputs_without_optional_diagnostics(self):
        for sequence, lanes in SEQUENCE_ARTIFACTS.items():
            if sequence in BATCH1_SEQUENCES:
                continue
            for lane in ("semantic_audits", "bearing_observations", "landmark_matches",
                         "localization_inputs"):
                self.assertEqual(lanes[lane], "v3pro_20260911_v1")
            self.assertIsNone(lanes["alignment_diagnostics"])

    def test_platform_prompt_and_tracker_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key, correct_prompt in (
                    ("washington", "osm_tags_farfield_v3"),
                    ("portland", "osm_tags_farfield_v3_down30")):
                group = replace(next(g for g in DATASET_GROUPS if g.key == key),
                                catalog_version=None, run_spec=None)
                sequence = group.sequences[0]
                for prompt in ("osm_tags_farfield_v3", "osm_tags_farfield_v3_down30"):
                    manifest = SimpleNamespace(upstreams=[], config={}, recipe={
                        "stage_config": {
                            "extraction.model": "gemini-3.1-pro-preview",
                            "extraction.prompt_type": prompt}})
                    with patch.object(check_roster.artifact, "load_manifest",
                                      return_value=manifest):
                        rows = check_roster.check_sequence(root, group, sequence, {})
                    row = next(row for row in rows if row[0] == "frame_landmarks")
                    self.assertEqual(row[2], "OK" if prompt == correct_prompt
                                     else "MISMATCH")

            sequence = "portland_flight_20260906_leg2"
            manifest = SimpleNamespace(git_commit="c46b91ac", upstreams=[
                SimpleNamespace(kind="frame_landmarks", version="retired-v1")])
            with patch.object(check_roster, "TRACKING_COMPARISON", {
                    sequence: {"dedup_pr722": "v3dedup_20260913_v1"}}), \
                    patch.object(check_roster.artifact, "load_manifest",
                                 return_value=manifest):
                self.assertEqual(check_roster.tracking_comparison_rows(root)[0][3],
                                 "MISMATCH")
                manifest.upstreams[0].version = "v3pro_20260911_v1"
                self.assertEqual(check_roster.tracking_comparison_rows(root)[0][3],
                                 "OK")


if __name__ == "__main__":
    unittest.main()
