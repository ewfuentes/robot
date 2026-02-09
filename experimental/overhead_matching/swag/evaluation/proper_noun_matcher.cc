#include "experimental/overhead_matching/swag/evaluation/proper_noun_matcher.hh"

#include <algorithm>
#include <cctype>
#include <thread>

#include "BS_thread_pool.hpp"

namespace robot::experimental::overhead_matching::swag::evaluation {

namespace {
// Convert string to lowercase
std::string to_lower(const std::string& s) {
    std::string result;
    result.reserve(s.size());
    for (char c : s) {
        result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return result;
}

// Check if needle is a substring of haystack
bool contains_substring(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}
}  // namespace

std::vector<float> compute_proper_noun_matches(
    const std::vector<std::vector<std::string>>& proper_nouns,
    const std::vector<std::vector<std::string>>& osm_texts, int num_threads) {
    const size_t num_proper = proper_nouns.size();
    const size_t num_osm = osm_texts.size();
    std::vector<float> result(num_proper * num_osm, 0.0f);

    if (num_proper == 0 || num_osm == 0) {
        return result;
    }

    // Pre-lowercase all OSM texts once
    std::vector<std::vector<std::string>> osm_texts_lower(num_osm);
    for (size_t j = 0; j < num_osm; ++j) {
        osm_texts_lower[j].reserve(osm_texts[j].size());
        for (const auto& text : osm_texts[j]) {
            osm_texts_lower[j].push_back(to_lower(text));
        }
    }

    // Pre-lowercase all proper nouns once
    std::vector<std::vector<std::string>> proper_nouns_lower(num_proper);
    for (size_t i = 0; i < num_proper; ++i) {
        proper_nouns_lower[i].reserve(proper_nouns[i].size());
        for (const auto& pn : proper_nouns[i]) {
            proper_nouns_lower[i].push_back(to_lower(pn));
        }
    }

    // Determine number of threads
    const int actual_threads =
        (num_threads <= 0) ? static_cast<int>(std::thread::hardware_concurrency()) : num_threads;

    BS::thread_pool pool(actual_threads);

    // Parallelize by proper noun rows
    pool.detach_blocks<size_t>(0, num_proper, [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            const auto& pn_list = proper_nouns_lower[i];
            if (pn_list.empty()) {
                continue;
            }

            for (size_t j = 0; j < num_osm; ++j) {
                const auto& osm_list = osm_texts_lower[j];
                if (osm_list.empty()) {
                    continue;
                }

                // Check if any proper noun is a substring of any OSM text
                bool found = false;
                for (const auto& pn : pn_list) {
                    for (const auto& osm_text : osm_list) {
                        if (contains_substring(osm_text, pn)) {
                            found = true;
                            break;
                        }
                    }
                    if (found) {
                        break;
                    }
                }

                if (found) {
                    result[i * num_osm + j] = 1.0f;
                }
            }
        }
    });

    pool.wait();

    return result;
}

std::vector<float> compute_keyed_substring_matches(
    const std::vector<std::vector<std::pair<std::string, std::string>>>& query_tags,
    const std::vector<std::vector<std::pair<std::string, std::string>>>& target_tags,
    int num_threads) {
    const size_t num_queries = query_tags.size();
    const size_t num_targets = target_tags.size();
    std::vector<float> result(num_queries * num_targets, 0.0f);

    if (num_queries == 0 || num_targets == 0) {
        return result;
    }

    // Pre-lowercase all target tags once, grouped by key for faster lookup
    // target_tags_lower[j] = vector of (lowercase_key, lowercase_value)
    std::vector<std::vector<std::pair<std::string, std::string>>> target_tags_lower(num_targets);
    for (size_t j = 0; j < num_targets; ++j) {
        target_tags_lower[j].reserve(target_tags[j].size());
        for (const auto& [key, value] : target_tags[j]) {
            target_tags_lower[j].emplace_back(to_lower(key), to_lower(value));
        }
    }

    // Pre-lowercase all query tags
    std::vector<std::vector<std::pair<std::string, std::string>>> query_tags_lower(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
        query_tags_lower[i].reserve(query_tags[i].size());
        for (const auto& [key, value] : query_tags[i]) {
            query_tags_lower[i].emplace_back(to_lower(key), to_lower(value));
        }
    }

    // Determine number of threads
    const int actual_threads =
        (num_threads <= 0) ? static_cast<int>(std::thread::hardware_concurrency()) : num_threads;

    BS::thread_pool pool(actual_threads);

    // Parallelize by query rows
    pool.detach_blocks<size_t>(0, num_queries, [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            const auto& query_list = query_tags_lower[i];
            if (query_list.empty()) {
                continue;
            }

            for (size_t j = 0; j < num_targets; ++j) {
                const auto& target_list = target_tags_lower[j];
                if (target_list.empty()) {
                    continue;
                }

                // Check if any query (key, value) matches a target (key, value)
                // where keys match exactly and query_value is substring of target_value
                bool found = false;
                for (const auto& [query_key, query_value] : query_list) {
                    for (const auto& [target_key, target_value] : target_list) {
                        if (query_key == target_key &&
                            contains_substring(target_value, query_value)) {
                            found = true;
                            break;
                        }
                    }
                    if (found) {
                        break;
                    }
                }

                if (found) {
                    result[i * num_targets + j] = 1.0f;
                }
            }
        }
    });

    pool.wait();

    return result;
}

KeyedSubstringMatchResults compute_keyed_substring_matches_detailed(
    const std::vector<std::vector<std::pair<std::string, std::string>>>& query_tags,
    const std::vector<std::vector<std::pair<std::string, std::string>>>& target_tags,
    int num_threads) {
    const size_t num_queries = query_tags.size();
    const size_t num_targets = target_tags.size();

    KeyedSubstringMatchResults result;

    if (num_queries == 0 || num_targets == 0) {
        return result;
    }

    // Pre-lowercase all target tags once
    std::vector<std::vector<std::pair<std::string, std::string>>> target_tags_lower(num_targets);
    for (size_t j = 0; j < num_targets; ++j) {
        target_tags_lower[j].reserve(target_tags[j].size());
        for (const auto& [key, value] : target_tags[j]) {
            target_tags_lower[j].emplace_back(to_lower(key), to_lower(value));
        }
    }

    // Pre-lowercase all query tags
    std::vector<std::vector<std::pair<std::string, std::string>>> query_tags_lower(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
        query_tags_lower[i].reserve(query_tags[i].size());
        for (const auto& [key, value] : query_tags[i]) {
            query_tags_lower[i].emplace_back(to_lower(key), to_lower(value));
        }
    }

    // Per-query result vectors (disjoint writes, no synchronization needed)
    struct PerQueryResults {
        std::vector<size_t> query_idxs;
        std::vector<size_t> target_idxs;
        std::vector<size_t> key_idxs;
    };
    std::vector<PerQueryResults> per_query_results(num_queries);

    const int actual_threads =
        (num_threads <= 0) ? static_cast<int>(std::thread::hardware_concurrency()) : num_threads;

    BS::thread_pool pool(actual_threads);

    pool.detach_blocks<size_t>(0, num_queries, [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            const auto& query_list = query_tags_lower[i];
            if (query_list.empty()) {
                continue;
            }

            for (size_t j = 0; j < num_targets; ++j) {
                const auto& target_list = target_tags_lower[j];
                if (target_list.empty()) {
                    continue;
                }

                for (size_t k = 0; k < query_list.size(); ++k) {
                    const auto& [query_key, query_value] = query_list[k];
                    for (const auto& [target_key, target_value] : target_list) {
                        if (query_key == target_key) {
                            // OSM tag keys are unique per landmark, so this is the only
                            // target tag with this key. Check if the value substring-matches.
                            if (contains_substring(target_value, query_value)) {
                                per_query_results[i].query_idxs.push_back(i);
                                per_query_results[i].target_idxs.push_back(j);
                                per_query_results[i].key_idxs.push_back(k);
                            }
                            break;  // no other target tag can have this key
                        }
                    }
                }
            }
        }
    });

    pool.wait();

    // Compute total size for reservation
    size_t total = 0;
    for (const auto& pq : per_query_results) {
        total += pq.query_idxs.size();
    }

    result.query_idxs.reserve(total);
    result.target_idxs.reserve(total);
    result.key_idxs.reserve(total);

    for (auto& pq : per_query_results) {
        result.query_idxs.insert(result.query_idxs.end(), pq.query_idxs.begin(),
                                 pq.query_idxs.end());
        result.target_idxs.insert(result.target_idxs.end(), pq.target_idxs.begin(),
                                  pq.target_idxs.end());
        result.key_idxs.insert(result.key_idxs.end(), pq.key_idxs.begin(), pq.key_idxs.end());
    }

    return result;
}

}  // namespace robot::experimental::overhead_matching::swag::evaluation
