#include "common/openstreetmap/extract_landmarks.h"

#include <chrono>
#include <filesystem>

#include "gtest/gtest.h"

namespace robot::openstreetmap {

class ExtractLandmarksTest : public ::testing::Test {
   protected:
    const std::filesystem::path test_pbf_ =
        "external/openstreetmap_snippet/us-virgin-islands-latest.osm.pbf";

    const BoundingBox full_bbox_{-65.0, 17.5, -64.5, 18.5};  // Covers all USVI
};

// === Geometry Type Tests ===

TEST_F(ExtractLandmarksTest, ExtractsPointGeometry) {
    auto features = extract_landmarks(test_pbf_.string(), full_bbox_, {{"amenity", true}});

    // Find at least one point
    auto point_it =
        std::find_if(features.begin(), features.end(), [](const auto& f) {
            return std::holds_alternative<PointGeometry>(f.geometry);
        });

    ASSERT_NE(point_it, features.end()) << "Should have at least one point feature";
    EXPECT_EQ(point_it->osm_type, OsmType::NODE);

    auto& point = std::get<PointGeometry>(point_it->geometry);
    EXPECT_GE(point.coord.lat, 17.5);
    EXPECT_LE(point.coord.lat, 18.5);
    EXPECT_GE(point.coord.lon, -65.0);
    EXPECT_LE(point.coord.lon, -64.5);
}

TEST_F(ExtractLandmarksTest, ExtractsLineStringGeometry) {
    auto features = extract_landmarks(test_pbf_.string(), full_bbox_, {{"highway", true}});

    // Find a linestring (open way)
    auto line_it =
        std::find_if(features.begin(), features.end(), [](const auto& f) {
            return std::holds_alternative<LineStringGeometry>(f.geometry);
        });

    ASSERT_NE(line_it, features.end()) << "Should have at least one linestring";
    EXPECT_EQ(line_it->osm_type, OsmType::WAY);

    auto& line = std::get<LineStringGeometry>(line_it->geometry);
    EXPECT_GE(line.coords.size(), 2) << "LineString must have at least 2 points";
}

TEST_F(ExtractLandmarksTest, ExtractsPolygonGeometry) {
    auto features = extract_landmarks(test_pbf_.string(), full_bbox_, {{"building", true}});

    // Find a polygon
    auto poly_it =
        std::find_if(features.begin(), features.end(), [](const auto& f) {
            return std::holds_alternative<PolygonGeometry>(f.geometry);
        });

    ASSERT_NE(poly_it, features.end()) << "Should have at least one polygon";

    auto& poly = std::get<PolygonGeometry>(poly_it->geometry);
    EXPECT_GE(poly.exterior.size(), 4) << "Polygon must have at least 4 points";

    // First and last coordinates should match (closed)
    EXPECT_NEAR(poly.exterior.front().lat, poly.exterior.back().lat, 1e-7);
    EXPECT_NEAR(poly.exterior.front().lon, poly.exterior.back().lon, 1e-7);
}

TEST_F(ExtractLandmarksTest, ExtractsMultiPolygonGeometry) {
    auto features = extract_landmarks(test_pbf_.string(), full_bbox_, {{"landuse", true}});

    // Check if any multipolygons exist
    auto mp_it =
        std::find_if(features.begin(), features.end(), [](const auto& f) {
            return std::holds_alternative<MultiPolygonGeometry>(f.geometry);
        });

    if (mp_it != features.end()) {
        EXPECT_EQ(mp_it->osm_type, OsmType::RELATION);
        auto& mp = std::get<MultiPolygonGeometry>(mp_it->geometry);
        EXPECT_GT(mp.polygons.size(), 0);
    }
}

// === Filtering Tests ===

TEST_F(ExtractLandmarksTest, BoundingBoxFilterWorks) {
    // Tight bbox (0.02 degrees ≈ 2km)
    BoundingBox tight_bbox{-64.95, 18.33, -64.93, 18.35};

    auto features_tight =
        extract_landmarks(test_pbf_.string(), tight_bbox, {{"amenity", true}});
    auto features_full =
        extract_landmarks(test_pbf_.string(), full_bbox_, {{"amenity", true}});

    EXPECT_LT(features_tight.size(), features_full.size())
        << "Tight bbox should return fewer features";

    // Verify all features are within bbox (check points only for simplicity)
    for (const auto& f : features_tight) {
        std::visit(
            [&](auto&& geom) {
                using T = std::decay_t<decltype(geom)>;
                if constexpr (std::is_same_v<T, PointGeometry>) {
                    EXPECT_GE(geom.coord.lat, tight_bbox.bottom_deg);
                    EXPECT_LE(geom.coord.lat, tight_bbox.top_deg);
                    EXPECT_GE(geom.coord.lon, tight_bbox.left_deg);
                    EXPECT_LE(geom.coord.lon, tight_bbox.right_deg);
                }
            },
            f.geometry);
    }
}

TEST_F(ExtractLandmarksTest, TagFilterWorks) {
    auto amenities = extract_landmarks(test_pbf_.string(), full_bbox_, {{"amenity", true}});
    auto buildings = extract_landmarks(test_pbf_.string(), full_bbox_, {{"building", true}});

    EXPECT_GT(amenities.size(), 0) << "Should find some amenities";
    EXPECT_GT(buildings.size(), 0) << "Should find some buildings";

    // Verify all have the requested tag
    for (const auto& f : amenities) {
        EXPECT_TRUE(f.tags.count("amenity") > 0);
    }
    for (const auto& f : buildings) {
        EXPECT_TRUE(f.tags.count("building") > 0);
    }
}

TEST_F(ExtractLandmarksTest, EmptyTagFilterReturnsEmpty) {
    auto features =
        extract_landmarks(test_pbf_.string(), full_bbox_, {{"nonexistent_tag_xyz", true}});
    EXPECT_EQ(features.size(), 0);
}

TEST_F(ExtractLandmarksTest, MultipleTagFilters) {
    auto features = extract_landmarks(test_pbf_.string(), full_bbox_,
                                      {{"amenity", true}, {"building", true}, {"highway", true}});

    EXPECT_GT(features.size(), 0) << "Should find features with any of the tags";

    // Each feature should have at least one of the requested tags
    for (const auto& f : features) {
        bool has_tag = f.tags.count("amenity") > 0 || f.tags.count("building") > 0 ||
                       f.tags.count("highway") > 0;
        EXPECT_TRUE(has_tag) << "Feature should have at least one requested tag";
    }
}

// === Data Integrity Tests ===

TEST_F(ExtractLandmarksTest, PreservesOsmMetadata) {
    auto features = extract_landmarks(test_pbf_.string(), full_bbox_, {{"amenity", true}});

    ASSERT_GT(features.size(), 0);

    for (const auto& f : features) {
        EXPECT_TRUE(f.osm_type == OsmType::NODE || f.osm_type == OsmType::WAY ||
                   f.osm_type == OsmType::RELATION);
        EXPECT_GT(f.osm_id, 0);
    }
}

TEST_F(ExtractLandmarksTest, PreservesTags) {
    auto features = extract_landmarks(test_pbf_.string(), full_bbox_, {{"amenity", true}});

    ASSERT_GT(features.size(), 0);

    // At least some features should have multiple tags
    auto multi_tag_count = std::count_if(
        features.begin(), features.end(), [](const auto& f) { return f.tags.size() > 1; });

    EXPECT_GT(multi_tag_count, 0) << "Some features should have multiple tags";
}

TEST_F(ExtractLandmarksTest, SetsLandmarkType) {
    auto features = extract_landmarks(test_pbf_.string(), full_bbox_, {{"amenity", true}});

    ASSERT_GT(features.size(), 0);

    for (const auto& f : features) {
        EXPECT_FALSE(f.landmark_type.empty()) << "landmark_type should be set";
        EXPECT_EQ(f.landmark_type, "amenity") << "landmark_type should match filter key";
    }
}

// === Edge Cases ===

TEST_F(ExtractLandmarksTest, HandlesEmptyBoundingBox) {
    BoundingBox empty_bbox{0.0, 0.0, 0.0, 0.0};  // Not in USVI
    auto features = extract_landmarks(test_pbf_.string(), empty_bbox, {{"amenity", true}});
    EXPECT_EQ(features.size(), 0);
}

TEST_F(ExtractLandmarksTest, HandlesInvalidPbfPath) {
    EXPECT_THROW(extract_landmarks("/nonexistent/file.osm.pbf", full_bbox_, {{"amenity", true}}),
                 std::runtime_error);
}

// === Performance Test ===

TEST_F(ExtractLandmarksTest, ExtractsFullIslandReasonablyFast) {
    auto start = std::chrono::steady_clock::now();

    auto features =
        extract_landmarks(test_pbf_.string(), full_bbox_,
                          {{"building", true}, {"highway", true}, {"amenity", true}});

    auto duration = std::chrono::steady_clock::now() - start;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();

    EXPECT_LT(seconds, 10) << "Should extract in under 10 seconds";
    EXPECT_GT(features.size(), 100) << "Should find many features";
}

// === Geometry Validation Tests ===

TEST_F(ExtractLandmarksTest, PolygonExteriorIsClosed) {
    auto features = extract_landmarks(test_pbf_.string(), full_bbox_, {{"building", true}});

    for (const auto& f : features) {
        if (std::holds_alternative<PolygonGeometry>(f.geometry)) {
            const auto& poly = std::get<PolygonGeometry>(f.geometry);
            ASSERT_GE(poly.exterior.size(), 4);

            // Verify closed ring
            EXPECT_NEAR(poly.exterior.front().lat, poly.exterior.back().lat, 1e-7)
                << "Polygon exterior should be closed";
            EXPECT_NEAR(poly.exterior.front().lon, poly.exterior.back().lon, 1e-7)
                << "Polygon exterior should be closed";
        }
    }
}

TEST_F(ExtractLandmarksTest, LineStringNotClosed) {
    auto features = extract_landmarks(test_pbf_.string(), full_bbox_, {{"highway", true}});

    // Find some linestrings
    int linestring_count = 0;
    for (const auto& f : features) {
        if (std::holds_alternative<LineStringGeometry>(f.geometry)) {
            const auto& line = std::get<LineStringGeometry>(f.geometry);
            ASSERT_GE(line.coords.size(), 2);
            linestring_count++;

            if (linestring_count > 10) break;  // Check first 10
        }
    }

    EXPECT_GT(linestring_count, 0) << "Should find some linestrings";
}

}  // namespace robot::openstreetmap
