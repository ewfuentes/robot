#pragma once

#include <string>
#include <utility>
#include <vector>

namespace robot::experimental::overhead_matching::swag::evaluation {

// Computes a binary similarity matrix between proper nouns and OSM texts.
// For each pair (i, j), result[i * num_osm + j] = 1 if any string in
// proper_nouns[i] is a case-insensitive substring of any string in osm_texts[j].
//
// Args:
//   proper_nouns: List of lists of proper noun strings (from panorama landmarks)
//   osm_texts: List of lists of OSM text fields (from OSM landmarks)
//   num_threads: Number of threads to use (0 = auto-detect)
//
// Returns:
//   Flat vector of size (proper_nouns.size() * osm_texts.size()) containing
//   1.0f for matches and 0.0f for non-matches. Row-major order.
std::vector<float> compute_proper_noun_matches(
    const std::vector<std::vector<std::string>>& proper_nouns,
    const std::vector<std::vector<std::string>>& osm_texts, int num_threads = 0);

// Computes a binary similarity matrix between keyed tags using substring matching.
// For each pair (i, j), result[i * num_targets + j] = 1 if any (key, value) in
// query_tags[i] has a matching key in target_tags[j] where query_value is a
// case-insensitive substring of target_value.
//
// Args:
//   query_tags: List of lists of (key, value) pairs (from extracted pano tags)
//   target_tags: List of lists of (key, value) pairs (from OSM tags)
//   num_threads: Number of threads to use (0 = auto-detect)
//
// Returns:
//   Flat vector of size (query_tags.size() * target_tags.size()) containing
//   1.0f for matches and 0.0f for non-matches. Row-major order.
std::vector<float> compute_keyed_substring_matches(
    const std::vector<std::vector<std::pair<std::string, std::string>>>& query_tags,
    const std::vector<std::vector<std::pair<std::string, std::string>>>& target_tags,
    int num_threads = 0);

}  // namespace robot::experimental::overhead_matching::swag::evaluation
