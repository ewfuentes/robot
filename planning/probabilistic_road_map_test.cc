
#include "planning/probabilistic_road_map.hh"

#include "gtest/gtest.h"

namespace robot::planning {

// Represents a map as an sdf.
// This is an SDF of a moon shape as illustrated here:
// https://iquilezles.org/articles/distfunctions2d/
struct SdfMap {
    // Computes the signed distance to a moon shaped obstacle
    double compute_dist(const Eigen::Vector2d &query_pt) const {
        constexpr double outer_diameter = 1.0;
        constexpr double inner_diameter = 0.8;
        constexpr double offset = 0.25;

        // Symmetric about the y axis:
        const double query_x = query_pt.x();
        const double query_y = std::abs(query_pt.y());

        const double a =
            (outer_diameter * outer_diameter - inner_diameter * inner_diameter + offset * offset) /
            (2.0 * offset);
        const double b = std::sqrt(std::max(outer_diameter * outer_diameter - a * a, 0.0));

        if (offset * (query_x * b - query_y * a) > offset * offset * std::max(b - query_y, 0.0)) {
            return std::hypot(query_x - a, query_y - b);
        }
        return std::max(std::hypot(query_x, query_y) - outer_diameter,
                        -(std::hypot(query_x - offset, query_y) - inner_diameter));
    }

    bool in_free_space(const Eigen::Vector2d &pt) const { return compute_dist(pt) > 0.0; }

    bool in_free_space(const Eigen::Vector2d &start, const Eigen::Vector2d &end) const {
        constexpr double TOL = 1e-3;
        const double total_dist = (end - start).norm();
        double dist = 0;
        while (dist < total_dist) {
            const Eigen::Vector2d query_pt = start + dist / total_dist * (end - start);
            const double dist_to_obstacle = compute_dist(query_pt);
            if (dist_to_obstacle < TOL) {
                return false;
            }
            dist += dist_to_obstacle;
        }
        return true;
    }

    MapBounds map_bounds() const {
        return MapBounds{.bottom_left = Eigen::Vector2d{-2.0, -1.5},
                         .top_right = Eigen::Vector2d{2.0, 1.5}};
    }
};

TEST(ProbabilisticRoadMapTest, sample_points) {
    // Setup
    const SdfMap sdf_map;
    const RoadmapCreationConfig config = {
        .seed = 0,
        .num_valid_points = 25,
        .max_node_degree = 4,
    };

    // Action
    RoadMap road_map = create_road_map(sdf_map, config);

    // Verification
    // Check that the appropriate number of points are in the map and that the max degree is
    // respected
    EXPECT_EQ(road_map.points.size(), config.num_valid_points);
    EXPECT_EQ(road_map.adj.rows(), road_map.adj.cols());
    EXPECT_EQ(road_map.adj.rows(), config.num_valid_points);

    EXPECT_LE(road_map.adj.colwise().sum().maxCoeff(), config.max_node_degree);

    // Check that it's symmetric and that all edges are free
    for (int i = 0; i < road_map.adj.rows(); i++) {
        // Self edges are not allowed
        EXPECT_EQ(road_map.adj(i, i), 0);
        for (int j = i + 1; j < road_map.adj.cols(); j++) {
            // Check that edges are undirected
            EXPECT_EQ(road_map.adj(i, j), road_map.adj(j, i));

            // If the edge exists, enure that it is in free space
            if (road_map.adj(i, j) > 0) {
                EXPECT_TRUE(sdf_map.in_free_space(road_map.points.at(i), road_map.points.at(j)));
            }
        }
    }
}
}  // namespace robot::planning
