
#include "domain/canadian_traveler.hh"

#include <algorithm>
#include <memory>

namespace robot::domain {
namespace {
using CTG = CanadianTravelerGraph;
}

std::shared_ptr<CTG> CTG::create(std::vector<CTG::Node> nodes, std::vector<CTG::Edge> edges) {
    return std::shared_ptr<CTG>(new CTG(std::move(nodes), std::move(edges)));
}

CTG::CanadianTravelerGraph(std::vector<CTG::Node> nodes, std::vector<CTG::Edge> edges)
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

CTG::Weather CTG::create_weather(const std::vector<EdgeBelief> &edge_beliefs) const {
    return CTG::Weather(weak_from_this(), edge_beliefs);
}

CTG::Weather::Weather(std::weak_ptr<const CTG> graph,
                      const std::vector<CTG::EdgeBelief> &edge_beliefs)
    : graph_(std::move(graph)) {
    const auto graph_ptr = graph_.lock();
    edges_.reserve(graph_ptr->edges().size());

    for (const auto &edge : graph_ptr->edges()) {
        if (!edge.traversal_prob.has_value()) {
            edges_.push_back(edge);
        } else {
            // This is a probabilistic edge, check to see if we have an observation for it
            const auto iter = std::find_if(
                edge_beliefs.begin(), edge_beliefs.end(), [&edge](const auto &edge_belief) -> bool {
                    const bool is_match =
                        (edge_belief.id_a == edge.node_a && edge_belief.id_b == edge.node_b) ||
                        (edge_belief.id_a == edge.node_b && edge_belief.id_b == edge.node_a);
                    return is_match;
                });
            if (iter == edge_beliefs.end()) {
                // No observation for this edge, copy the edge over
                edges_.push_back(edge);
            }

            if (iter->traversability == CTG::EdgeState::Unknown) {
                edges_.push_back(edge);
            } else if (iter->traversability == CTG::EdgeState::Traversable) {
                edges_.push_back(CTG::Edge{
                    .node_a = edge.node_a,
                    .node_b = edge.node_b,
                    .cost = edge.cost,
                    .traversal_prob = std::nullopt,
                });
            }
        }
    }

    neighbors_.reserve(graph_ptr->nodes().size());
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

    while (neighbors_.size() != graph_ptr->nodes().size()) {
        neighbors_.push_back(std::span<Edge>());
    }
}

bool CTG::Edge::operator==(const CTG::Edge &other) const {
    return node_a == other.node_a && node_b == other.node_b && cost == other.cost &&
           traversal_prob == other.traversal_prob;
}
}  // namespace robot::domain
