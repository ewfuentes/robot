
#pragma once

#include <optional>
#include <span>
#include <vector>

#include "Eigen/Core"

namespace robot::domain {

class CanadianTravelerGraph {
   public:
    using NodeId = int;
    struct Node {
        Eigen::Vector2d location;
    };

    struct Edge {
        NodeId node_a;
        NodeId node_b;

        double cost;
        std::optional<double> traversal_prob;

        bool operator==(const Edge &other) const;
    };

    CanadianTravelerGraph(std::vector<Node> nodes, std::vector<Edge> edges);

    const std::vector<Node> &nodes() const { return nodes_; }

    const std::vector<Edge> &edges() const { return edges_; }

    const std::span<Edge> &neighbors(const NodeId id) const { return neighbors_.at(id); }

   private:
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    std::vector<std::span<Edge>> neighbors_;
};
}  // namespace robot::domain
