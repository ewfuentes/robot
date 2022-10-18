
#pragma once

#include <random>

#include "Eigen/Core"

namespace robot::planning {

struct RoadmapCreationConfig {
    // Initial seed for the PRNG used in sampling the graph
    std::mt19937::result_type seed;

    // The number of required points in free space
    int num_valid_points;

    // The maximum number of connections between nodes
    int max_node_degree;
};

struct RoadMap {
    // The location of each roadmap point
    std::vector<Eigen::Vector2d> points;

    // An adjacency matrix where the i'th column contains the connectivity to other nodes
    Eigen::MatrixXd adj;
};

struct MapBounds {
    // These bounds are used to define the limits for sampling
    Eigen::Vector2d bottom_left;
    Eigen::Vector2d top_right;
};

// Create a probabilistic road map from a map.
// The Map type must support the following interface:
// MapBounds
// bool in_free_space(const Eigen::Vector2d &point);
//  - Returns true if a point is in free space. Returns false otherwise.
// bool in_free_space(const Eigen::Vector2d &point_a, const Eigen::Vector2d &point_b);
//  - Returns true if a line segment lies entirely in free space, Returns false otherwise.
// MapBounds map_bounds()
//  - Returns the limits of where points should be sampled from
//
// initial_sample_points contains sample points that the caller would like to include as part of
// the road map. It may be empty.
template <typename Map>
RoadMap create_road_map(const Map &map, const RoadmapCreationConfig &config,
                        const std::vector<Eigen::Vector2d> &initial_sample_points = {}) {
    // Copy valid points into point list
    std::vector<Eigen::Vector2d> sample_points;
    sample_points.reserve(config.num_valid_points);

    std::copy_if(initial_sample_points.begin(), initial_sample_points.end(),
                 std::back_inserter(sample_points),
                 [&map](const Eigen::Vector2d &pt) { return map.in_free_space(pt); });

    // Sample up to sample point limit
    std::mt19937 gen(config.seed);
    const auto bounds = map.map_bounds();
    std::uniform_real_distribution<double> x_gen(bounds.bottom_left.x(), bounds.top_right.x());
    std::uniform_real_distribution<double> y_gen(bounds.bottom_left.y(), bounds.top_right.y());
    while (static_cast<int>(sample_points.size()) < config.num_valid_points) {
        Eigen::Vector2d pt{x_gen(gen), y_gen(gen)};
        if (map.in_free_space(pt)) {
            sample_points.push_back(pt);
        }
    }

    // Create connections up to appropriate degree
    Eigen::MatrixXd adjacency =
        Eigen::MatrixXd::Zero(config.num_valid_points, config.num_valid_points);
    for (int i = 0; i < static_cast<int>(sample_points.size()); i++) {
        int edge_count = adjacency.col(i).sum();
        for (int j = i + 1;
             j < static_cast<int>(sample_points.size()) && edge_count < config.max_node_degree;
             j++) {
            if (map.in_free_space(sample_points.at(i), sample_points.at(j))) {
                adjacency(i, j) = 1;
                adjacency(j, i) = 1;
                edge_count += 1;
            }
        }
    }
    return RoadMap{.points = std::move(sample_points), .adj = adjacency};
}

}  // namespace robot::planning
