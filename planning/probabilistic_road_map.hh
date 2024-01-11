
#pragma once

#include <concepts>
#include <random>

#include "Eigen/Core"
#include "planning/road_map.hh"

namespace robot::planning {

struct RoadmapCreationConfig {
    // Initial seed for the PRNG used in sampling the graph
    std::mt19937::result_type seed;

    // The number of required points in free space
    int num_valid_points;

    // The desired number of connections between nodes
    // When adding edges, try to have at least this many edges per node
    int desired_node_degree;
};

struct MapBounds {
    // These bounds are used to define the limits for sampling
    Eigen::Vector2d bottom_left;
    Eigen::Vector2d top_right;
};

template <typename T>
concept Map = requires(T m, Eigen::Vector2d a, Eigen::Vector2d b) {
                  { m.in_free_space(a) } -> std::same_as<bool>;
                  { m.in_free_space(a, b) } -> std::same_as<bool>;
                  { m.map_bounds() } -> std::same_as<MapBounds>;
              };

// Create a probabilistic road map from a map.
// initial_sample_points contains sample points that the caller would like to include as part of
// the road map. It may be empty.
template <Map M>
RoadMap create_road_map(const M &map, const RoadmapCreationConfig &config,
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
    std::vector<int> sample_idxs(sample_points.size());
    std::iota(sample_idxs.begin(), sample_idxs.end(), 0);
    for (int i = 0; i < static_cast<int>(sample_points.size()); i++) {
        // Sort the points by distance and ensure that at least
        const auto &pt = sample_points.at(i);
        const auto dist_sq_to_point = [&pt](const Eigen::Vector2d &other) {
            const double dx = pt.x() - other.x();
            const double dy = pt.y() - other.y();
            return dx * dx + dy * dy;
        };
        std::sort(sample_idxs.begin(), sample_idxs.end(),
                  [&dist_sq_to_point, &sample_points](const int a, const int b) {
                      return dist_sq_to_point(sample_points.at(a)) <
                             dist_sq_to_point(sample_points.at(b));
                  });
        int edges_added = 0;
        for (const int other_idx : sample_idxs) {
            const auto &other_pt = sample_points.at(other_idx);
            if (pt == other_pt) {
                continue;
            }
            if (map.in_free_space(pt, other_pt)) {
                adjacency(i, other_idx) = 1;
                adjacency(other_idx, i) = 1;
                edges_added++;
            }
            if (edges_added >= config.desired_node_degree) {
                // Enough edges added for this node, continue to the next node
                break;
            }
        }
    }
    return RoadMap(std::move(sample_points), std::move(adjacency));
}

}  // namespace robot::planning
