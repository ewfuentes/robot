#include "common/openstreetmap/extract_landmarks.hh"

#include <filesystem>
#include <osmium/area/assembler.hpp>
#include <osmium/area/multipolygon_manager.hpp>
#include <osmium/handler.hpp>
#include <osmium/handler/node_locations_for_ways.hpp>
#include <osmium/index/map/flex_mem.hpp>
#include <osmium/io/pbf_input.hpp>
#include <osmium/relations/relations_manager.hpp>
#include <osmium/visitor.hpp>
#include <stdexcept>

namespace robot::openstreetmap {

namespace {

// Helper to check if any tag in the filter map matches
bool has_matching_tag(const osmium::TagList& tags, const std::map<std::string, bool>& tag_filters) {
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
        feature.osm_type = OsmType::NODE;
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
        feature.osm_type = OsmType::WAY;
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

// Handler for processing multipolygon areas
class AreaHandler : public osmium::handler::Handler {
   public:
    AreaHandler(const BoundingBox& bbox, const std::map<std::string, bool>& tag_filters)
        : bbox_(bbox), tag_filters_(tag_filters) {}

    void area(const osmium::Area& area) {
        // Only process relations (multipolygons), not closed ways
        if (area.from_way()) {
            return;  // Skip - already handled as PolygonGeometry in LandmarkHandler
        }

        if (area.tags().empty()) {
            return;
        }

        if (!has_matching_tag(area.tags(), tag_filters_)) {
            return;
        }

        // Check if any node is within bbox
        bool in_bbox = false;
        for (const auto& outer_ring : area.outer_rings()) {
            for (const auto& node_ref : outer_ring) {
                if (bbox_.contains(node_ref.lon(), node_ref.lat())) {
                    in_bbox = true;
                    break;
                }
            }
            if (in_bbox) break;
        }

        if (!in_bbox) {
            return;
        }

        // Build MultiPolygonGeometry
        MultiPolygonGeometry mp;

        // Osmium areas can have multiple outer rings
        for (const auto& outer_ring : area.outer_rings()) {
            PolygonGeometry poly;

            // Extract outer ring coordinates
            for (const auto& node_ref : outer_ring) {
                poly.exterior.push_back({node_ref.lon(), node_ref.lat()});
            }

            // Extract inner rings (holes) for this outer ring
            for (const auto& inner_ring : area.inner_rings(outer_ring)) {
                std::vector<Coordinate> hole;
                for (const auto& node_ref : inner_ring) {
                    hole.push_back({node_ref.lon(), node_ref.lat()});
                }
                poly.holes.push_back(std::move(hole));
            }

            mp.polygons.push_back(std::move(poly));
        }

        // Create feature
        LandmarkFeature feature;
        feature.osm_type = OsmType::RELATION;
        feature.osm_id = area.orig_id();  // Original relation ID
        feature.geometry = std::move(mp);
        feature.tags = tags_to_map(area.tags());

        // Set landmark_type
        for (const auto& [key, _] : tag_filters_) {
            if (feature.tags.count(key) > 0) {
                feature.landmark_type = key;
                break;
            }
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

std::vector<LandmarkFeature> extract_landmarks(const std::string& pbf_path, const BoundingBox& bbox,
                                               const std::map<std::string, bool>& tag_filters) {
    // Verify file exists
    if (!std::filesystem::exists(pbf_path)) {
        throw std::runtime_error("PBF file not found: " + pbf_path);
    }

    std::vector<LandmarkFeature> all_features;

    // Use location index to store node locations for ways
    using IndexType =
        osmium::index::map::FlexMem<osmium::unsigned_object_id_type, osmium::Location>;
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

    // Pass 2: Extract multipolygon relations
    {
        // MultipolygonManager collects relations and their members
        using MultipolygonManager = osmium::area::MultipolygonManager<osmium::area::Assembler>;

        osmium::area::Assembler::config_type assembler_config;
        MultipolygonManager mp_manager{assembler_config};

        // First pass: collect relations
        {
            osmium::io::Reader reader(pbf_path, osmium::osm_entity_bits::relation);
            osmium::apply(reader, mp_manager);
            reader.close();
        }

        // Prepare manager for member lookups
        mp_manager.prepare_for_lookup();

        // Second pass: read ways/nodes and assemble areas
        {
            osmium::io::Reader reader(pbf_path);
            AreaHandler area_handler(bbox, tag_filters);

            osmium::apply(reader, location_handler,
                          mp_manager.handler([&area_handler](osmium::memory::Buffer&& buffer) {
                              osmium::apply(buffer, area_handler);
                          }));

            reader.close();

            // Add multipolygon features to results
            auto mp_features = area_handler.features();
            all_features.insert(all_features.end(), mp_features.begin(), mp_features.end());
        }
    }

    return all_features;
}

}  // namespace robot::openstreetmap
