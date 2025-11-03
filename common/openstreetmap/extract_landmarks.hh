#pragma once

#include <array>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace robot::openstreetmap {

struct Coordinate {
    double lon;  // Longitude in degrees (WGS84)
    double lat;  // Latitude in degrees (WGS84)
};

enum class OsmType { NODE, WAY, RELATION };

struct PointGeometry {
    Coordinate coord;
};

struct LineStringGeometry {
    std::vector<Coordinate> coords;
};

struct PolygonGeometry {
    std::vector<Coordinate> exterior;
    std::vector<std::vector<Coordinate>> holes;
};

struct MultiPolygonGeometry {
    std::vector<PolygonGeometry> polygons;
};

using Geometry =
    std::variant<PointGeometry, LineStringGeometry, PolygonGeometry, MultiPolygonGeometry>;

struct LandmarkFeature {
    OsmType osm_type;                         // Type of OSM element (NODE, WAY, or RELATION)
    int64_t osm_id;                           // OSM element ID
    Geometry geometry;                        // Geometric representation
    std::map<std::string, std::string> tags;  // All OSM tags (key-value pairs)
    std::string landmark_type;  // The tag key that matched the filter (e.g., "amenity", "building")
};

struct BoundingBox {
    double left_deg;
    double bottom_deg;
    double right_deg;
    double top_deg;

    bool contains(double lon, double lat) const {
        return lon >= left_deg && lon <= right_deg && lat >= bottom_deg && lat <= top_deg;
    }
};

// Extract landmarks from a PBF file
// - pbf_path: Path to the OSM PBF file
// - bbox: Bounding box to filter features
// - tag_filters: Map of OSM tags to filter by (e.g., {"amenity": true, "building": true})
// Returns: Vector of extracted landmark features
std::vector<LandmarkFeature> extract_landmarks(const std::string& pbf_path, const BoundingBox& bbox,
                                               const std::map<std::string, bool>& tag_filters);

}  // namespace robot::openstreetmap
