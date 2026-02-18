import numpy as np

from experimental.overhead_matching.swag.evaluation import proper_noun_matcher_python as pnm


class TestComputeKeyedSubstringMatchesDetailed:
    def test_single_match(self):
        query_tags = [[("name", "starbucks")]]
        target_tags = [[("name", "Starbucks Coffee")]]

        q, t, k = pnm.compute_keyed_substring_matches_detailed(query_tags, target_tags)

        assert len(q) == 1
        assert q[0] == 0
        assert t[0] == 0
        assert k[0] == 0

    def test_no_match_different_keys(self):
        query_tags = [[("name", "starbucks")]]
        target_tags = [[("amenity", "starbucks")]]

        q, t, k = pnm.compute_keyed_substring_matches_detailed(query_tags, target_tags)

        assert len(q) == 0

    def test_no_match_substring_not_found(self):
        query_tags = [[("name", "starbucks")]]
        target_tags = [[("name", "dunkin donuts")]]

        q, t, k = pnm.compute_keyed_substring_matches_detailed(query_tags, target_tags)

        assert len(q) == 0

    def test_multiple_keys_match_independently(self):
        query_tags = [[("name", "central"), ("amenity", "cafe")]]
        target_tags = [[("name", "Central Perk"), ("amenity", "cafe")]]

        q, t, k = pnm.compute_keyed_substring_matches_detailed(query_tags, target_tags)

        assert len(q) == 2
        np.testing.assert_array_equal(q, [0, 0])
        np.testing.assert_array_equal(t, [0, 0])
        np.testing.assert_array_equal(k, [0, 1])

    def test_multiple_queries_and_targets(self):
        query_tags = [
            [("name", "alpha")],
            [("name", "beta")],
        ]
        target_tags = [
            [("name", "Alpha Bravo")],
            [("name", "Beta Gamma")],
            [("name", "Alpha Beta")],
        ]

        q, t, k = pnm.compute_keyed_substring_matches_detailed(query_tags, target_tags)

        matches = set(zip(q.tolist(), t.tolist(), k.tolist()))
        assert matches == {(0, 0, 0), (0, 2, 0), (1, 1, 0), (1, 2, 0)}

    def test_case_insensitive(self):
        query_tags = [[("name", "HELLO")]]
        target_tags = [[("name", "say hello world")]]

        q, t, k = pnm.compute_keyed_substring_matches_detailed(query_tags, target_tags)

        assert len(q) == 1

    def test_empty_inputs(self):
        q, t, k = pnm.compute_keyed_substring_matches_detailed([], [])
        assert len(q) == 0
        assert len(t) == 0
        assert len(k) == 0

    def test_empty_query_tags(self):
        q, t, k = pnm.compute_keyed_substring_matches_detailed(
            [[]], [[("name", "foo")]]
        )
        assert len(q) == 0

    def test_empty_target_tags(self):
        q, t, k = pnm.compute_keyed_substring_matches_detailed(
            [[("name", "foo")]], [[]]
        )
        assert len(q) == 0

    def test_key_idx_reflects_query_position(self):
        query_tags = [[("addr", "main"), ("cuisine", "pizza"), ("name", "joe")]]
        target_tags = [[("name", "Joe's Pizza")]]

        q, t, k = pnm.compute_keyed_substring_matches_detailed(query_tags, target_tags)

        assert len(q) == 1
        assert k[0] == 2  # "name" is at index 2 in the query

    def test_returns_int64_arrays(self):
        query_tags = [[("name", "test")]]
        target_tags = [[("name", "test")]]

        q, t, k = pnm.compute_keyed_substring_matches_detailed(query_tags, target_tags)

        assert q.dtype == np.int64
        assert t.dtype == np.int64
        assert k.dtype == np.int64

    def test_agrees_with_binary_version(self):
        """The set of (query, target) pairs with any match should equal the
        nonzero entries in the binary matrix from compute_keyed_substring_matches."""
        query_tags = [
            [("name", "alpha"), ("amenity", "cafe")],
            [("name", "beta")],
            [("cuisine", "sushi")],
        ]
        target_tags = [
            [("name", "Alpha Cafe"), ("amenity", "restaurant")],
            [("name", "Beta Lounge"), ("cuisine", "sushi bar")],
            [("amenity", "cafe"), ("cuisine", "italian")],
        ]

        binary = pnm.compute_keyed_substring_matches(query_tags, target_tags)
        q, t, k = pnm.compute_keyed_substring_matches_detailed(query_tags, target_tags)

        # Reconstruct binary matrix from detailed results
        reconstructed = np.zeros_like(binary)
        for qi, ti in zip(q, t):
            reconstructed[qi, ti] = 1.0

        np.testing.assert_array_equal(binary, reconstructed)
