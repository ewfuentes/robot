#include "experimental/overhead_matching/swag/evaluation/proper_noun_matcher.hh"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace robot::experimental::overhead_matching::swag::evaluation {

PYBIND11_MODULE(proper_noun_matcher_python, m) {
    m.doc() = "Fast substring matching for panorama-OSM comparison";

    m.def(
        "compute_proper_noun_matches",
        [](const std::vector<std::vector<std::string>>& proper_nouns,
           const std::vector<std::vector<std::string>>& osm_texts,
           int num_threads) -> py::array_t<float> {
            // Call C++ implementation
            std::vector<float> result =
                compute_proper_noun_matches(proper_nouns, osm_texts, num_threads);

            const size_t num_proper = proper_nouns.size();
            const size_t num_osm = osm_texts.size();

            // Create numpy array with proper shape
            py::array_t<float> arr({num_proper, num_osm});
            auto buf = arr.request();
            float* ptr = static_cast<float*>(buf.ptr);

            std::copy(result.begin(), result.end(), ptr);

            return arr;
        },
        py::arg("proper_nouns"), py::arg("osm_texts"), py::arg("num_threads") = 0,
        R"pbdoc(
        Compute binary similarity matrix between proper nouns and OSM texts.

        For each pair (i, j), result[i, j] = 1.0 if any string in proper_nouns[i]
        is a case-insensitive substring of any string in osm_texts[j].

        Parameters
        ----------
        proper_nouns : list[list[str]]
            List of lists of proper noun strings (from panorama landmarks)
        osm_texts : list[list[str]]
            List of lists of OSM text fields (from OSM landmarks)
        num_threads : int, optional
            Number of threads to use (0 = auto-detect, default)

        Returns
        -------
        numpy.ndarray
            Float32 array of shape (len(proper_nouns), len(osm_texts)) containing
            1.0 for matches and 0.0 for non-matches.
    )pbdoc");

    m.def(
        "compute_keyed_substring_matches",
        [](const std::vector<std::vector<std::pair<std::string, std::string>>>& query_tags,
           const std::vector<std::vector<std::pair<std::string, std::string>>>& target_tags,
           int num_threads) -> py::array_t<float> {
            // Call C++ implementation
            std::vector<float> result =
                compute_keyed_substring_matches(query_tags, target_tags, num_threads);

            const size_t num_queries = query_tags.size();
            const size_t num_targets = target_tags.size();

            // Create numpy array with proper shape
            py::array_t<float> arr({num_queries, num_targets});
            auto buf = arr.request();
            float* ptr = static_cast<float*>(buf.ptr);

            std::copy(result.begin(), result.end(), ptr);

            return arr;
        },
        py::arg("query_tags"), py::arg("target_tags"), py::arg("num_threads") = 0,
        R"pbdoc(
        Compute binary similarity matrix between keyed tags using substring matching.

        For each pair (i, j), result[i, j] = 1.0 if any (key, value) in query_tags[i]
        has a matching key in target_tags[j] where query_value is a case-insensitive
        substring of target_value.

        Parameters
        ----------
        query_tags : list[list[tuple[str, str]]]
            List of lists of (key, value) pairs (from extracted pano tags)
        target_tags : list[list[tuple[str, str]]]
            List of lists of (key, value) pairs (from OSM tags)
        num_threads : int, optional
            Number of threads to use (0 = auto-detect, default)

        Returns
        -------
        numpy.ndarray
            Float32 array of shape (len(query_tags), len(target_tags)) containing
            1.0 for matches and 0.0 for non-matches.
    )pbdoc");
}

}  // namespace robot::experimental::overhead_matching::swag::evaluation
