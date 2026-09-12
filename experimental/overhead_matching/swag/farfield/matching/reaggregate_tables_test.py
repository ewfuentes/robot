import math
import unittest

from experimental.overhead_matching.swag.farfield.matching import (
    match_landmarks as ml,
    reaggregate_tables as rt,
)


def _sig(tags, *ids):
    return ml.signature(tags), {
        "canonical_tags": dict(sorted(tags.items())),
        "display_label": ml.signature_display(tags),
        "landmark_ids": list(ids)}


class ReaggregateTablesTest(unittest.TestCase):
    def setUp(self):
        self.signatures = dict([
            _sig({"amenity": "school", "name": "North"}, "n"),
            _sig({"amenity": "school", "name": "South"}, "s1", "s2"),
            _sig({"man_made": "mast"}, "m"),
            _sig({"brand": "Ford"}, "u"),  # no recognised kind
        ])
        sid = {v["landmark_ids"][0]: k for k, v in self.signatures.items()}
        self.matches = {"t": {
            "matches": [
                {"landmark_id": "n", "signature_id": sid["n"],
                 "aggregate_confidence": 1.0, "match_type": "instance"},
                {"landmark_id": "s1", "signature_id": sid["s1"],
                 "aggregate_confidence": 0.9, "match_type": "category"},
                {"landmark_id": "s2", "signature_id": sid["s1"],
                 "aggregate_confidence": 0.9, "match_type": "category"},
                {"landmark_id": "u", "signature_id": sid["u"],
                 "aggregate_confidence": 0.3, "match_type": "category"},
            ],
            "aggregate_no_match_confidence": 0.5}}
        self.shipped = [ml.to_compatibility_table(
            "t", {lid: ml.to_log_lr(c) for lid, c in
                  (("n", 1.0), ("s1", 0.9), ("s2", 0.9), ("u", 0.3))},
            matcher_version="v", default_log_lr=ml.to_log_lr(1.0 / 4),
            clip_lo=-ml.DEFAULT_CLIP)]

    def test_baseline_reproduces_shipped_table(self):
        (table,) = rt.reaggregate(
            self.matches, self.signatures, self.shipped, "baseline")
        self.assertEqual(table, self.shipped[0])

    def test_divided_expands_kind_at_c_over_n_and_keeps_instances(self):
        (table,) = rt.reaggregate(
            self.matches, self.signatures, self.shipped, "catexpand_divided")
        log_lr = {e.landmark_id: e.log_lr for e in table.entries}
        # The school kind has 3 rows; the instance row keeps 1.0, the other
        # two carry 0.9/3 each. The unrecognised-kind row keeps its 0.3.
        self.assertEqual(log_lr["n"], ml.DEFAULT_CLIP)
        self.assertAlmostEqual(log_lr["s1"], math.log(0.3 / 0.7), places=6)
        self.assertAlmostEqual(log_lr["s2"], math.log(0.3 / 0.7), places=6)
        self.assertAlmostEqual(log_lr["u"], math.log(0.3 / 0.7), places=6)
        self.assertNotIn("m", log_lr)
        self.assertEqual(table.clip_lo, ml.CATEGORY_CLIP_LO)
        self.assertLess(table.default_log_lr, min(log_lr.values()))
        self.assertEqual(table.matcher_version, "v+reagg_catexpand_divided")

    def test_kind_only_track_stays_endorsed_under_the_category_floor(self):
        self.matches["t"]["matches"] = [
            self.matches["t"]["matches"][1]]  # one category row, c=0.9
        (table,) = rt.reaggregate(
            self.matches, self.signatures, self.shipped, "catexpand_divided")
        log_lr = {e.landmark_id: e.log_lr for e in table.entries}
        self.assertEqual(sorted(log_lr), ["n", "s1", "s2"])
        self.assertTrue(all(v > table.default_log_lr for v in log_lr.values()))


if __name__ == "__main__":
    unittest.main()
