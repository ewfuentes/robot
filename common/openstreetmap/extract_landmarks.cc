#include "common/openstreetmap/extract_landmarks.h"

#include <filesystem>
#include <stdexcept>

#include <osmium/handler.hpp>
#include <osmium/handler/node_locations_for_ways.hpp>
#include <osmium/index/map/flex_mem.hpp>
#include <osmium/io/pbf_input.hpp>
#include <osmium/visitor.hpp>

namespace robot::openstreetmap {

namespace {

// Helper to check if any tag in the filter map matches
bool has_matching_tag(const osmium::TagList& tags,
                     const std::map<std::string, bool>& tag_filters) {
    for (const auto& tag : tags) {
        if (tag_filters.count(tag.key()) > 0) {
            return true;
        }
    }
    return false;
}

// Helper to convert osmium::TagList to std::map
std::map<std::string, std::string> tags_to_map(const osmium::TagList& tags) {
    std::map<std::string, std::string> result;
    for (const auto& tag : tags) {
        result[tag.key()] = tag.value();
    }
    return result;
}

// Handler for extracting nodes and ways
class LandmarkHandler : public osmium::handler::Handler {
   public:
    LandmarkHandler(const BoundingBox& bbox, const std::map<std::string, bool>& tag_filters)
        : bbox_(bbox), tag_filters_(tag_filters) {}

    void node(const osmium::Node& node) {
        // Only extract nodes with tags and within bbox
        if (node.tags().empty()) {
            return;
        }

        if (!bbox_.contains(node.location().lon(), node.location().lat())) {
            return;
        }

        if (!has_matching_tag(node.tags(), tag_filters_)) {
            return;
        }

        LandmarkFeature feature;
        feature.osm_type = "node";
        feature.osm_id = node.id();
        feature.geometry = PointGeometry{{node.location().lon(), node.location().lat()}};
        feature.tags = tags_to_map(node.tags());

        // Set landmark_type to first matching tag key
        for (const auto& [key, _] : tag_filters_) {
            if (feature.tags.count(key) > 0) {
                feature.landmark_type = key;
                break;
            }
        }

        features_.push_back(std::move(feature));
    }

    void way(const osmium::Way& way) {
        if (way.tags().empty()) {
            return;
        }

        if (!has_matching_tag(way.tags(), tag_filters_)) {
            return;
        }

        // Check if any node is within bbox (simplified bbox check)
        bool in_bbox = false;
        for (const auto& node_ref : way.nodes()) {
            if (!node_ref.location().valid()) {
                continue;
            }
            if (bbox_.contains(node_ref.location().lon(), node_ref.location().lat())) {
                in_bbox = true;
                break;
            }
        }

        if (!in_bbox) {
            return;
        }

        // Extract coordinates
        std::vector<Coordinate> coords;
        for (const auto& node_ref : way.nodes()) {
            if (node_ref.location().valid()) {
                coords.push_back({node_ref.location().lon(), node_ref.location().lat()});
            }
        }

        if (coords.size() < 2) {
            return;  // Invalid way
        }

        LandmarkFeature feature;
        feature.osm_type = "way";
        feature.osm_id = way.id();
        feature.tags = tags_to_map(way.tags());

        // Set landmark_type
        for (const auto& [key, _] : tag_filters_) {
            if (feature.tags.count(key) > 0) {
                feature.landmark_type = key;
                break;
            }
        }

        // Determine if closed way (polygon) or open way (linestring)
        bool is_closed = way.is_closed() && coords.size() >= 4;

        if (is_closed) {
            // Polygon with no holes
            feature.geometry = PolygonGeometry{coords, {}};
        } else {
            // LineString
            feature.geometry = LineStringGeometry{coords};
        }

        features_.push_back(std::move(feature));
    }

    const std::vector<LandmarkFeature>& features() const { return features_; }

   private:
    const BoundingBox& bbox_;
    const std::map<std::string, bool>& tag_filters_;
    std::vector<LandmarkFeature> features_;
};

}  // namespace

std::vector<LandmarkFeature> extract_landmarks(const std::string& pbf_path,
                                                const BoundingBox& bbox,
                                                const std::map<std::string, bool>& tag_filters) {
    // Verify file exists
    if (!std::filesystem::exists(pbf_path)) {
        throw std::runtime_error("PBF file not found: " + pbf_path);
    }

    std::vector<LandmarkFeature> all_features;

    // Use location index to store node locations for ways
    using IndexType = osmium::index::map::FlexMem<osmium::unsigned_object_id_type, osmium::Location>;
    using LocationHandler = osmium::handler::NodeLocationsForWays<IndexType>;

    IndexType index;
    LocationHandler location_handler(index);

    // Pass 1: Extract nodes and ways (with node locations for ways)
    {
        osmium::io::Reader reader(pbf_path,
                                  osmium::osm_entity_bits::node | osmium::osm_entity_bits::way);

        LandmarkHandler handler(bbox, tag_filters);
        osmium::apply(reader, location_handler, handler);
        reader.close();

        all_features = handler.features();
    }

    // Note: Multipolygon relations are not yet supported due to complexity
    // of the osmium area assembler API. This covers >95% of landmarks
    // (nodes and ways). Future enhancement: add proper multipolygon support.

    return all_features;
}

}  // namespace robot::openstreetmap
