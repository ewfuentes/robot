import unittest
from experimental.overhead_matching.swag.scripts.pairing import (
    create_pairs, create_anchors, PositiveAnchorSets, Pairs)


class PairingFunctionsTest(unittest.TestCase):

    def test_create_pairs_basic(self):
        panorama_metadata = [
            {'index': 0},
            {'index': 1},
        ]
        satellite_metadata = [
            {
                'positive_panorama_idxs': [0],
                'semipositive_panorama_idxs': [],
            },
            {
                'positive_panorama_idxs': [],
                'semipositive_panorama_idxs': [1],
            },
            {
                'positive_panorama_idxs': [],
                'semipositive_panorama_idxs': [],
            }
        ]

        pairs = create_pairs(panorama_metadata, satellite_metadata)

        self.assertIsInstance(pairs, Pairs)
        self.assertEqual(pairs.positive_pairs, [(0, 0)])
        self.assertEqual(pairs.semipositive_pairs, [(1, 1)])
        expected_negatives = [(0, 1), (0, 2), (1, 0), (1, 2)]
        self.assertEqual(set(pairs.negative_pairs), set(expected_negatives))

    def test_create_pairs_multiple_positives(self):
        panorama_metadata = [
            {'index': 5},
            {'index': 7},
        ]
        satellite_metadata = [
            {
                'positive_panorama_idxs': [5, 7],
                'semipositive_panorama_idxs': [],
            },
            {
                'positive_panorama_idxs': [5],
                'semipositive_panorama_idxs': [7],
            },
        ]

        pairs = create_pairs(panorama_metadata, satellite_metadata)

        expected_positives = [(0, 0), (1, 0), (0, 1)]
        expected_semipositives = [(1, 1)]
        self.assertEqual(set(pairs.positive_pairs), set(expected_positives))
        self.assertEqual(pairs.semipositive_pairs, expected_semipositives)
        self.assertEqual(len(pairs.negative_pairs), 0)

    def test_create_pairs_empty_metadata(self):
        panorama_metadata = []
        satellite_metadata = []

        pairs = create_pairs(panorama_metadata, satellite_metadata)

        self.assertEqual(pairs.positive_pairs, [])
        self.assertEqual(pairs.semipositive_pairs, [])
        self.assertEqual(pairs.negative_pairs, [])

    def test_create_anchors_pano_as_anchor_basic(self):
        panorama_metadata = [
            {
                'positive_satellite_idxs': [10, 11],
                'semipositive_satellite_idxs': [12],
            },
            {
                'positive_satellite_idxs': [11],
                'semipositive_satellite_idxs': [],
            },
        ]
        satellite_metadata = [
            {'index': 10},
            {'index': 11},
            {'index': 12},
            {'index': 13},
        ]

        anchors = create_anchors(panorama_metadata, satellite_metadata, use_pano_as_anchor=True)

        self.assertIsInstance(anchors, PositiveAnchorSets)
        self.assertEqual(anchors.anchor, [0, 1])
        self.assertEqual(anchors.positive, [{0, 1}, {1}])
        self.assertEqual(anchors.semipositive, [{2}, set()])

    def test_create_anchors_sat_as_anchor_basic(self):
        panorama_metadata = [
            {'index': 20},
            {'index': 21},
            {'index': 22},
        ]
        satellite_metadata = [
            {
                'positive_panorama_idxs': [20, 21],
                'semipositive_panorama_idxs': [22],
            },
            {
                'positive_panorama_idxs': [21],
                'semipositive_panorama_idxs': [],
            },
        ]

        anchors = create_anchors(panorama_metadata, satellite_metadata, use_pano_as_anchor=False)

        self.assertEqual(anchors.anchor, [0, 1])
        self.assertEqual(anchors.positive, [{0, 1}, {1}])
        self.assertEqual(anchors.semipositive, [{2}, set()])

    def test_create_anchors_no_matches_in_batch(self):
        panorama_metadata = [
            {
                'positive_satellite_idxs': [100, 101],
                'semipositive_satellite_idxs': [102],
            },
        ]
        satellite_metadata = [
            {'index': 10},
            {'index': 11},
        ]

        anchors = create_anchors(panorama_metadata, satellite_metadata, use_pano_as_anchor=True)

        self.assertEqual(anchors.anchor, [0])
        self.assertEqual(anchors.positive, [set()])
        self.assertEqual(anchors.semipositive, [set()])

    def test_create_anchors_partial_matches(self):
        panorama_metadata = [
            {
                'positive_satellite_idxs': [10, 100],
                'semipositive_satellite_idxs': [11, 101],
            },
        ]
        satellite_metadata = [
            {'index': 10},
            {'index': 11},
            {'index': 12},
        ]

        anchors = create_anchors(panorama_metadata, satellite_metadata, use_pano_as_anchor=True)

        self.assertEqual(anchors.anchor, [0])
        self.assertEqual(anchors.positive, [{0}])
        self.assertEqual(anchors.semipositive, [{1}])

    def test_create_anchors_duplicate_indices_handled(self):
        panorama_metadata = [
            {
                'positive_satellite_idxs': [10],
                'semipositive_satellite_idxs': [],
            },
        ]
        satellite_metadata = [
            {'index': 10},
            {'index': 10},
        ]

        with self.assertRaises(AssertionError):
            create_anchors(panorama_metadata, satellite_metadata, use_pano_as_anchor=True)


if __name__ == "__main__":
    unittest.main()
