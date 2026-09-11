import json
import unittest

from shapely.geometry import Point

from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    feather_utils as subject,
)


def frame(order=("osm", "enc")):
    records = {
        "osm": ("('node', 7)", Point(-71.0001, 42.0)),
        "enc": ("('enc', 'ABC')", Point(-71.0, 42.0)),
    }
    return schema.build_frame(
        ids=[records[source][0] for source in order],
        geometries=[records[source][1] for source in order],
        landmark_types=list(order),
        tags=[{"man_made": "lighthouse"} for _ in order],
    )


class CrossSourceDedupeTest(unittest.TestCase):

    def test_enc_wins_and_both_source_records_survive(self):
        result = subject.dedupe_exact_duplicates(
            frame(), tolerance_m=20.0, verbose=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["id"], "('enc', 'ABC')")
        self.assertEqual(result.iloc[0]["landmark_type"], "enc")
        self.assertTrue(result.geometry.iloc[0].equals(Point(-71.0, 42.0)))
        tags = schema.tag_dicts(result)[0]
        records = json.loads(tags[subject.SOURCE_RECORDS_TAG])
        self.assertEqual(records, [
            {"id": "('enc', 'ABC')", "landmark_type": "enc"},
            {"id": "('node', 7)", "landmark_type": "osm"},
        ])

    def test_input_order_does_not_change_the_winner_or_provenance(self):
        left = subject.dedupe_exact_duplicates(
            frame(("osm", "enc")), tolerance_m=20.0, verbose=False)
        right = subject.dedupe_exact_duplicates(
            frame(("enc", "osm")), tolerance_m=20.0, verbose=False)
        self.assertEqual(left.iloc[0]["id"], right.iloc[0]["id"])
        self.assertEqual(schema.tag_dicts(left), schema.tag_dicts(right))
        self.assertTrue(left.geometry.iloc[0].equals(right.geometry.iloc[0]))


if __name__ == "__main__":
    unittest.main()
