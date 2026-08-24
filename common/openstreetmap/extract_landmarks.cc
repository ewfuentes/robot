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
#include <unordered_map>

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
    LandmarkHandler(const std::unordered_map<std::string, BoundingBox>& bboxes,
                    const std::map<std::string, bool>& tag_filters)
        : bboxes_(bboxes), tag_filters_(tag_filters) {}

    void node(const osmium::Node& node) {
        // Only extract nodes with tags
        if (node.tags().empty()) {
            return;
        }

        if (!has_matching_tag(node.tags(), tag_filters_)) {
            return;
        }

        double lon = node.location().lon();
        double lat = node.location().lat();

        // Check all bboxes, assign to first match
        for (const auto& [region_id, bbox] : bboxes_) {
            if (bbox.contains(lon, lat)) {
                LandmarkFeature feature;
                feature.osm_type = OsmType::NODE;
                feature.osm_id = node.id();
                feature.geometry = PointGeometry{{lon, lat}};
                feature.tags = tags_to_map(node.tags());
                features_.emplace_back(region_id, std::move(feature));
                break;  // Assign to first matching region only
            }
        }
    }

    void way(const osmium::Way& way) {
        if (way.tags().empty()) {
            return;
        }

        if (!has_matching_tag(way.tags(), tag_filters_)) {
            return;
        }

        // Extract coordinates first
        std::vector<Coordinate> coords;
        for (const auto& node_ref : way.nodes()) {
            if (node_ref.location().valid()) {
                coords.push_back({node_ref.location().lon(), node_ref.location().lat()});
            }
        }

        if (coords.size() < 2) {
            return;  // Invalid way
        }

        // Check all bboxes using first valid coordinate
        std::string matched_region;
        for (const auto& [region_id, bbox] : bboxes_) {
            for (const auto& coord : coords) {
                if (bbox.contains(coord.lon, coord.lat)) {
                    matched_region = region_id;
                    break;
                }
            }
            if (!matched_region.empty()) {
                break;
            }
        }

        if (matched_region.empty()) {
            return;  // Not in any bbox
        }

        LandmarkFeature feature;
        feature.osm_type = OsmType::WAY;
        feature.osm_id = way.id();
        feature.tags = tags_to_map(way.tags());

        // Determine if closed way (polygon) or open way (linestring)
        bool is_closed = way.is_closed() && coords.size() >= 4;

        if (is_closed) {
            // Polygon with no holes
            feature.geometry = PolygonGeometry{coords, {}};
        } else {
            // LineString
            feature.geometry = LineStringGeometry{coords};
        }

        features_.emplace_back(matched_region, std::move(feature));
    }

    const std::vector<std::pair<std::string, LandmarkFeature>>& features() const {
        return features_;
    }

   private:
    const std::unordered_map<std::string, BoundingBox>& bboxes_;
    const std::map<std::string, bool>& tag_filters_;
    std::vector<std::pair<std::string, LandmarkFeature>> features_;
};

// Handler for processing multipolygon areas
class AreaHandler : public osmium::handler::Handler {
   public:
    AreaHandler(const std::unordered_map<std::string, BoundingBox>& bboxes,
                const std::map<std::string, bool>& tag_filters)
        : bboxes_(bboxes), tag_filters_(tag_filters) {}

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

        // Check all bboxes to find first match
        std::string matched_region;
        for (const auto& [region_id, bbox] : bboxes_) {
            bool in_bbox = false;
            for (const auto& outer_ring : area.outer_rings()) {
                for (const auto& node_ref : outer_ring) {
                    if (bbox.contains(node_ref.lon(), node_ref.lat())) {
                        in_bbox = true;
                        break;
                    }
                }
                if (in_bbox) break;
            }
            if (in_bbox) {
                matched_region = region_id;
                break;
            }
        }

        if (matched_region.empty()) {
            return;  // Not in any bbox
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

        features_.emplace_back(matched_region, std::move(feature));
    }

    const std::vector<std::pair<std::string, LandmarkFeature>>& features() const {
        return features_;
    }

   private:
    const std::unordered_map<std::string, BoundingBox>& bboxes_;
    const std::map<std::string, bool>& tag_filters_;
    std::vector<std::pair<std::string, LandmarkFeature>> features_;
};

// Feeds the way-geometry index only with nodes near the requested bboxes.
//
// osmium's NodeLocationsForWays stores every node it is handed, which is why
// peak memory tracks the size of the PBF rather than the size of the request. It
// is not a requirement of the problem: a way here is selected iff one of its own
// vertices falls in a bbox, and that question is answered just as well by an
// index holding only the nodes near those bboxes.
//
// Wrapping rather than subclassing because the inner handler's way() mutates the
// node refs it is given, and we want that behaviour untouched.
template <typename TIndex>
class BoundedNodeLocations : public osmium::handler::Handler {
   public:
    BoundedNodeLocations(TIndex& index, std::vector<BoundingBox> keep)
        : inner_(index), keep_(std::move(keep)) {
        // Nodes outside the margin are deliberately absent, so a way that
        // references them must not be treated as a corrupt file. The existing
        // way handler already skips refs whose location is invalid.
        inner_.ignore_errors();
    }

    void node(const osmium::Node& node) {
        const auto& loc = node.location();
        if (!loc.valid()) {
            return;
        }
        for (const auto& bbox : keep_) {
            if (bbox.contains(loc.lon(), loc.lat())) {
                inner_.node(node);
                ++kept_;
                return;
            }
        }
        ++dropped_;
    }

    void way(osmium::Way& way) { inner_.way(way); }

    std::size_t kept() const { return kept_; }
    std::size_t dropped() const { return dropped_; }

   private:
    osmium::handler::NodeLocationsForWays<TIndex> inner_;
    std::vector<BoundingBox> keep_;
    std::size_t kept_ = 0;
    std::size_t dropped_ = 0;
};

std::vector<BoundingBox> expand_bboxes(const std::unordered_map<std::string, BoundingBox>& bboxes,
                                       double margin_deg) {
    std::vector<BoundingBox> out;
    out.reserve(bboxes.size());
    for (const auto& [region_id, bbox] : bboxes) {
        out.push_back(BoundingBox{bbox.left_deg - margin_deg, bbox.bottom_deg - margin_deg,
                                  bbox.right_deg + margin_deg, bbox.top_deg + margin_deg});
    }
    return out;
}

}  // namespace

std::vector<std::pair<std::string, LandmarkFeature>> extract_landmarks(
    const std::string& pbf_path, const std::unordered_map<std::string, BoundingBox>& bboxes,
    const std::map<std::string, bool>& tag_filters, double node_margin_deg) {
    // Verify file exists
    if (!std::filesystem::exists(pbf_path)) {
        throw std::runtime_error("PBF file not found: " + pbf_path);
    }

    std::vector<std::pair<std::string, LandmarkFeature>> all_features;

    // Use location index to store node locations for ways
    using IndexType =
        osmium::index::map::FlexMem<osmium::unsigned_object_id_type, osmium::Location>;
    using LocationHandler = osmium::handler::NodeLocationsForWays<IndexType>;

    IndexType index;

    // Both passes need the same location handler, and the two handler types are
    // distinct, so the pass sequence is templated on it rather than duplicated.
    auto run_passes = [&](auto& location_handler) {
        // Pass 1: Extract nodes and ways (with node locations for ways)
        {
            osmium::io::Reader reader(pbf_path,
                                      osmium::osm_entity_bits::node | osmium::osm_entity_bits::way);

            LandmarkHandler handler(bboxes, tag_filters);
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
                AreaHandler area_handler(bboxes, tag_filters);

                osmium::apply(reader, location_handler,
                              mp_manager.handler([&area_handler](osmium::memory::Buffer&& buffer) {
                                  osmium::apply(buffer, area_handler);
                              }));

                reader.close();

                // Add multipolygon features to results
                const auto& mp_features = area_handler.features();
                all_features.insert(all_features.end(), mp_features.begin(), mp_features.end());
            }
        }
    };

    if (node_margin_deg >= 0.0) {
        BoundedNodeLocations<IndexType> location_handler(index,
                                                         expand_bboxes(bboxes, node_margin_deg));
        run_passes(location_handler);
    } else {
        LocationHandler location_handler(index);
        run_passes(location_handler);
    }

    return all_features;
}

}  // namespace robot::openstreetmap
