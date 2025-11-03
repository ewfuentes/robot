#pragma once

#include <array>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace robot::openstreetmap {

struct Coordinate {
    double lon;
    double lat;
};

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

using Geometry = std::variant<PointGeometry, LineStringGeometry, PolygonGeometry,
                               MultiPolygonGeometry>;

struct LandmarkFeature {
    std::string osm_type;  // "node", "way", "relation"
    int64_t osm_id;
    Geometry geometry;
    std::map<std::string, std::string> tags;
    std::string landmark_type;
};

struct BoundingBox {
    double left;
    double bottom;
    double right;
    double top;

    bool contains(double lon, double lat) const {
        return lon >= left && lon <= right && lat >= bottom && lat <= top;
    }
};

// Extract landmarks from a PBF file
// - pbf_path: Path to the OSM PBF file
// - bbox: Bounding box to filter features
// - tag_filters: Map of OSM tags to filter by (e.g., {"amenity": true, "building": true})
// Returns: Vector of extracted landmark features
std::vector<LandmarkFeature> extract_landmarks(const std::string& pbf_path,
                                                const BoundingBox& bbox,
                                                const std::map<std::string, bool>& tag_filters);

}  // namespace robot::openstreetmap
