
#include "domain/canadian_traveler.hh"

#include <algorithm>

namespace robot::domain {

CanadianTravelerGraph::CanadianTravelerGraph(std::vector<CanadianTravelerGraph::Node> nodes,
                                             std::vector<CanadianTravelerGraph::Edge> edges)
    : nodes_{std::move(nodes)}, edges_{} {
    // Push back a copy of each edge with the directions reversed
    edges_.reserve(2 * edges.size());
    for (auto &edge : edges) {
        edges_.push_back(edge);
        std::swap(edge.node_a, edge.node_b);
        edges_.push_back(edge);
    }
    // Sort the edges lexographically by node_a then by node_b
    std::sort(edges_.begin(), edges_.end(), [](const auto &a, const auto &b) {
        const bool is_node_a_less = a.node_a < b.node_a;
        const bool is_node_a_equal_and_node_b_less = a.node_a == b.node_a && a.node_b < b.node_b;
        return is_node_a_less || is_node_a_equal_and_node_b_less;
    });

    // Enable fast lookups of neighbors
    neighbors_.reserve(nodes.size());
    auto range_start = edges_.begin();
    for (auto iter = edges_.begin(); iter != edges_.end(); iter++) {
        if (iter->node_a != range_start->node_a) {
            while (static_cast<int>(neighbors_.size()) != range_start->node_a) {
                neighbors_.push_back(std::span<Edge>());
            }
            neighbors_.push_back(std::span(range_start, iter));
            range_start = iter;
        }
    }
    while (static_cast<int>(neighbors_.size()) != range_start->node_a) {
        neighbors_.push_back(std::span<Edge>());
    }
    neighbors_.push_back(std::span(range_start, edges_.end()));

    while (neighbors_.size() != nodes_.size()) {
        neighbors_.push_back(std::span<Edge>());
    }
}

bool CanadianTravelerGraph::Edge::operator==(const CanadianTravelerGraph::Edge &other) const {
    return node_a == other.node_a && node_b == other.node_b && cost == other.cost &&
           traversal_prob == other.traversal_prob;
}
}  // namespace robot::domain
