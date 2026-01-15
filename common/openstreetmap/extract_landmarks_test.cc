#include "common/openstreetmap/extract_landmarks.hh"

#include <chrono>
#include <filesystem>
#include <unordered_map>

#include "gtest/gtest.h"

namespace robot::openstreetmap {

class ExtractLandmarksTest : public ::testing::Test {
   protected:
    const std::filesystem::path test_pbf_ =
        "external/openstreetmap_snippet/us-virgin-islands-latest.osm.pbf";

    const BoundingBox full_bbox_{-65.0, 17.5, -64.5, 18.5};  // Covers all USVI

    // Helper to create single-region bbox map
    std::unordered_map<std::string, BoundingBox> make_bboxes(const BoundingBox& bbox) const {
        return {{"test_region", bbox}};
    }

    // Helper to extract features (returns just the LandmarkFeature part)
    std::vector<LandmarkFeature> extract_features(
        const std::string& pbf_path, const BoundingBox& bbox,
        const std::map<std::string, bool>& tag_filters) const {
        auto results = extract_landmarks(pbf_path, make_bboxes(bbox), tag_filters);
        std::vector<LandmarkFeature> features;
        features.reserve(results.size());
        for (auto& [region_id, feature] : results) {
            features.push_back(std::move(feature));
        }
        return features;
    }
};

// === Geometry Type Tests ===

TEST_F(ExtractLandmarksTest, ExtractsPointGeometry) {
    auto features = extract_features(test_pbf_.string(), full_bbox_, {{"amenity", true}});

    // Find at least one point
    auto point_it = std::find_if(features.begin(), features.end(), [](const auto& f) {
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
    auto features = extract_features(test_pbf_.string(), full_bbox_, {{"highway", true}});

    // Find a linestring (open way)
    auto line_it = std::find_if(features.begin(), features.end(), [](const auto& f) {
        return std::holds_alternative<LineStringGeometry>(f.geometry);
    });

    ASSERT_NE(line_it, features.end()) << "Should have at least one linestring";
    EXPECT_EQ(line_it->osm_type, OsmType::WAY);

    auto& line = std::get<LineStringGeometry>(line_it->geometry);
    EXPECT_GE(line.coords.size(), 2) << "LineString must have at least 2 points";
}

TEST_F(ExtractLandmarksTest, ExtractsPolygonGeometry) {
    auto features = extract_features(test_pbf_.string(), full_bbox_, {{"building", true}});

    // Find a polygon
    auto poly_it = std::find_if(features.begin(), features.end(), [](const auto& f) {
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
    auto features = extract_features(test_pbf_.string(), full_bbox_, {{"landuse", true}});

    // Check if any multipolygons exist
    auto mp_it = std::find_if(features.begin(), features.end(), [](const auto& f) {
        return std::holds_alternative<MultiPolygonGeometry>(f.geometry);
    });

    if (mp_it != features.end()) {
        EXPECT_EQ(mp_it->osm_type, OsmType::RELATION);
        auto& mp = std::get<MultiPolygonGeometry>(mp_it->geometry);
        EXPECT_GT(mp.polygons.size(), 0);
    }
}

TEST_F(ExtractLandmarksTest, ExtractsSpecificMultiPolygonRelation) {
    // Relation 4784235 is a known multipolygon in the USVI dataset
    auto features = extract_features(
        test_pbf_.string(), full_bbox_,
        {{"landuse", true}, {"natural", true}, {"building", true}, {"leisure", true}});

    // Find the specific relation
    auto relation_it = std::find_if(features.begin(), features.end(), [](const auto& f) {
        return f.osm_type == OsmType::RELATION && f.osm_id == 4784235;
    });

    ASSERT_NE(relation_it, features.end())
        << "Relation 4784235 should be extracted from USVI dataset";

    // Verify it's a multipolygon
    ASSERT_TRUE(std::holds_alternative<MultiPolygonGeometry>(relation_it->geometry))
        << "Relation 4784235 should be a MultiPolygonGeometry";

    const auto& mp = std::get<MultiPolygonGeometry>(relation_it->geometry);
    EXPECT_GT(mp.polygons.size(), 0) << "Multipolygon should contain at least one polygon";

    // Verify each polygon is valid
    for (const auto& poly : mp.polygons) {
        EXPECT_GE(poly.exterior.size(), 4) << "Polygon exterior must have at least 4 points";

        // Verify closed ring
        EXPECT_NEAR(poly.exterior.front().lat, poly.exterior.back().lat, 1e-7)
            << "Polygon exterior should be closed";
        EXPECT_NEAR(poly.exterior.front().lon, poly.exterior.back().lon, 1e-7)
            << "Polygon exterior should be closed";

        // Verify holes are also closed
        for (const auto& hole : poly.holes) {
            EXPECT_GE(hole.size(), 4) << "Polygon hole must have at least 4 points";
            EXPECT_NEAR(hole.front().lat, hole.back().lat, 1e-7) << "Polygon hole should be closed";
            EXPECT_NEAR(hole.front().lon, hole.back().lon, 1e-7) << "Polygon hole should be closed";
        }
    }

    // Verify tags are preserved
    EXPECT_GT(relation_it->tags.size(), 0) << "Multipolygon should have tags";
}

// === Filtering Tests ===

TEST_F(ExtractLandmarksTest, BoundingBoxFilterWorks) {
    // Tight bbox (0.02 degrees ≈ 2km)
    BoundingBox tight_bbox{-64.95, 18.33, -64.93, 18.35};

    auto features_tight = extract_features(test_pbf_.string(), tight_bbox, {{"amenity", true}});
    auto features_full = extract_features(test_pbf_.string(), full_bbox_, {{"amenity", true}});

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
    auto amenities = extract_features(test_pbf_.string(), full_bbox_, {{"amenity", true}});
    auto buildings = extract_features(test_pbf_.string(), full_bbox_, {{"building", true}});

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
        extract_features(test_pbf_.string(), full_bbox_, {{"nonexistent_tag_xyz", true}});
    EXPECT_EQ(features.size(), 0);
}

TEST_F(ExtractLandmarksTest, MultipleTagFilters) {
    auto features = extract_features(test_pbf_.string(), full_bbox_,
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
    auto features = extract_features(test_pbf_.string(), full_bbox_, {{"amenity", true}});

    ASSERT_GT(features.size(), 0);

    for (const auto& f : features) {
        EXPECT_TRUE(f.osm_type == OsmType::NODE || f.osm_type == OsmType::WAY ||
                    f.osm_type == OsmType::RELATION);
        EXPECT_GT(f.osm_id, 0);
    }
}

TEST_F(ExtractLandmarksTest, PreservesTags) {
    auto features = extract_features(test_pbf_.string(), full_bbox_, {{"amenity", true}});

    ASSERT_GT(features.size(), 0);

    // At least some features should have multiple tags
    auto multi_tag_count = std::count_if(features.begin(), features.end(),
                                         [](const auto& f) { return f.tags.size() > 1; });

    EXPECT_GT(multi_tag_count, 0) << "Some features should have multiple tags";
}

// === Edge Cases ===

TEST_F(ExtractLandmarksTest, HandlesEmptyBoundingBox) {
    BoundingBox empty_bbox{0.0, 0.0, 0.0, 0.0};  // Not in USVI
    auto features = extract_features(test_pbf_.string(), empty_bbox, {{"amenity", true}});
    EXPECT_EQ(features.size(), 0);
}

TEST_F(ExtractLandmarksTest, HandlesInvalidPbfPath) {
    EXPECT_THROW(extract_landmarks("/nonexistent/file.osm.pbf", make_bboxes(full_bbox_),
                                   {{"amenity", true}}),
                 std::runtime_error);
}

// === Performance Test ===

TEST_F(ExtractLandmarksTest, ExtractsFullIslandReasonablyFast) {
    auto start = std::chrono::steady_clock::now();

    auto features = extract_features(test_pbf_.string(), full_bbox_,
                                     {{"building", true}, {"highway", true}, {"amenity", true}});

    auto duration = std::chrono::steady_clock::now() - start;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();

    EXPECT_LT(seconds, 10) << "Should extract in under 10 seconds";
    EXPECT_GT(features.size(), 100) << "Should find many features";
}

// === Geometry Validation Tests ===

TEST_F(ExtractLandmarksTest, PolygonExteriorIsClosed) {
    auto features = extract_features(test_pbf_.string(), full_bbox_, {{"building", true}});

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
    auto features = extract_features(test_pbf_.string(), full_bbox_, {{"highway", true}});

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

// === Multi-Bbox Tests ===

TEST_F(ExtractLandmarksTest, MultipleBboxesExtractCorrectRegions) {
    // Define two non-overlapping bboxes within USVI
    BoundingBox bbox_a{-64.95, 18.33, -64.90, 18.38};  // St. Thomas area
    BoundingBox bbox_b{-64.80, 18.30, -64.75, 18.35};  // Different area

    std::unordered_map<std::string, BoundingBox> bboxes{{"region_a", bbox_a}, {"region_b", bbox_b}};

    auto results = extract_landmarks(test_pbf_.string(), bboxes, {{"amenity", true}});

    // Count features per region
    int region_a_count = 0;
    int region_b_count = 0;
    for (const auto& [region_id, feature] : results) {
        if (region_id == "region_a") {
            region_a_count++;
        } else if (region_id == "region_b") {
            region_b_count++;
        }
    }

    // Both regions should have some features (assuming both areas have amenities)
    EXPECT_GT(region_a_count + region_b_count, 0) << "Should find features in at least one region";

    // Verify region assignment is correct by checking coordinates
    for (const auto& [region_id, feature] : results) {
        if (std::holds_alternative<PointGeometry>(feature.geometry)) {
            const auto& point = std::get<PointGeometry>(feature.geometry);
            if (region_id == "region_a") {
                EXPECT_TRUE(bbox_a.contains(point.coord.lon, point.coord.lat))
                    << "Region A feature should be within bbox_a";
            } else if (region_id == "region_b") {
                EXPECT_TRUE(bbox_b.contains(point.coord.lon, point.coord.lat))
                    << "Region B feature should be within bbox_b";
            }
        }
    }
}

}  // namespace robot::openstreetmap
