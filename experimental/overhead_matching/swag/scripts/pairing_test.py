import unittest
from experimental.overhead_matching.swag.scripts.pairing import create_pairs, Pairs


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


if __name__ == "__main__":
    unittest.main()
