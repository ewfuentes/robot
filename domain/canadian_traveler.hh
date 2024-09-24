
#pragma once

#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "Eigen/Core"

namespace robot::domain {

class Weather;

class CanadianTravelerGraph : public std::enable_shared_from_this<CanadianTravelerGraph> {
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

    enum class EdgeState { Unknown, Untraversable, Traversable };
    struct EdgeBelief {
        CanadianTravelerGraph::NodeId id_a;
        CanadianTravelerGraph::NodeId id_b;
        EdgeState traversability;
    };

    // A weather represents a (possibly incomplete) traversability assignment to the unknown
    // edges of a CanadianTravelerGraph
    class Weather {
       public:
        const std::span<Edge> &neighbors(const NodeId id) const { return neighbors_.at(id); }

       private:
        friend CanadianTravelerGraph;
        Weather(std::weak_ptr<const CanadianTravelerGraph> graph,
                const std::vector<EdgeBelief> &edge_beliefs);
        std::vector<Edge> edges_;
        std::vector<std::span<Edge>> neighbors_;
        std::weak_ptr<const CanadianTravelerGraph> graph_;
    };

    static std::shared_ptr<CanadianTravelerGraph> create(std::vector<Node> nodes,
                                                         std::vector<Edge> edges);

    const std::vector<Node> &nodes() const { return nodes_; }

    const std::span<Edge> &neighbors(const NodeId id) const { return neighbors_.at(id); }

    const std::vector<Edge> &edges() const { return edges_; }

    Weather create_weather(const std::vector<EdgeBelief> &edge_beliefs = {}) const;

   private:
    CanadianTravelerGraph(std::vector<Node> nodes, std::vector<Edge> edges);
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    std::vector<std::span<Edge>> neighbors_;
};

}  // namespace robot::domain
