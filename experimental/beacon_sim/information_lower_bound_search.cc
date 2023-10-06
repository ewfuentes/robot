
#include "experimental/beacon_sim/information_lower_bound_search.hh"

#include <algorithm>
#include <iostream>
#include <queue>

namespace robot::experimental::beacon_sim {
namespace {
std::ostream &operator<<(std::ostream &out, const detail::InProgressPath &path) {
    out << "[";
    for (const auto id : path.path_to_goal) {
        out << id << ", ";
    }
    out << "] cost: " << path.cost_to_go << " info_lb: " << path.info_lower_bound;
    return out;
}
}
namespace detail {
MergeResult should_merge(const std::vector<InProgressPath> &existing,
                         const InProgressPath &new_path) {
    if (existing.empty()) {
        return {.should_merge = true, .to_boot = {}};
    }

    std::vector<int> to_boot;
    bool should_merge = false;
    for (int i = 0; i < static_cast<int>(existing.size()); i++) {
        const InProgressPath &existing_path = existing.at(i);
        if (existing_path.info_lower_bound >= new_path.info_lower_bound &&
            existing_path.cost_to_go >= new_path.cost_to_go) {
            to_boot.push_back(i);
            should_merge = true;
        } else if (existing_path.info_lower_bound <= new_path.info_lower_bound &&
                   existing_path.cost_to_go <= new_path.cost_to_go) {
            should_merge = false;
            break;
        } else {
            // Either one of the info_lower_bound or the cost to go is lower than the existing
            // path, but the other is higher, so we keep both.
            should_merge = true;
        }
    }
    return {.should_merge = should_merge, .to_boot = std::move(to_boot)};
}
}  // namespace detail

InformationLowerBoundResult information_lower_bound_search(
    const planning::RoadMap &road_map, const int start_idx, const int end_idx,
    const double start_information, const double end_information_lower_bound,
    const LowerBoundReversePropagator &rev_propagator) {
    std::unordered_map<int, std::vector<detail::InProgressPath>> paths_from_node_id;
    std::priority_queue<detail::InProgressPath, std::vector<detail::InProgressPath>,
                        std::greater<detail::InProgressPath>>
        open_list;
    open_list.push(detail::InProgressPath{.info_lower_bound = end_information_lower_bound,
                                          .cost_to_go = 0.0,
                                          .path_to_goal{end_idx}});

    while (!open_list.empty()) {
        std::cout << "------------------------ Open List Size: " << open_list.size() << std::endl;
        // Note that this creates a copy. It is non-const so we can move it later.
        detail::InProgressPath in_progress = open_list.top();
        open_list.pop();

        std::cout << "Popping: " << in_progress << std::endl;

        const int current_node_id = in_progress.path_to_goal.back();
        std::vector<detail::InProgressPath> &best_paths_at_node =
            paths_from_node_id[current_node_id];

        if (in_progress.path_to_goal.back() == start_idx &&
            start_information >= in_progress.info_lower_bound) {
            std::cout << "Found valid path! Terminating" << std::endl;
            std::reverse(in_progress.path_to_goal.begin(), in_progress.path_to_goal.end());
            return {
                .info_lower_bound = in_progress.info_lower_bound,
                .cost_to_go = in_progress.cost_to_go,
                .path_to_goal = in_progress.path_to_goal,
            };
        }

        std::cout << "Current Node Id: " << current_node_id << std::endl;
        std::cout << "Candidates: " << std::endl;
        for (const auto &path : best_paths_at_node) {
            std::cout << "\t" << path << std::endl;
        }

        // Try to merge the path into the per node queues
        const detail::MergeResult merge_result = should_merge(best_paths_at_node, in_progress);
        std::cout << "Merge Result: should merge? " << merge_result.should_merge << " to boot: [";
        for (const auto to_boot : merge_result.to_boot) {
            std::cout << to_boot << ", ";
        }
        std::cout << "]" << std::endl;

        // If the path wasn't added, then we continue to the next path
        if (!merge_result.should_merge) {
            continue;
        }

        // remove any paths that have been dominated
        if (!merge_result.to_boot.empty()) {
            auto insert_iter = best_paths_at_node.begin();
            auto should_skip_iter = merge_result.to_boot.begin();
            for (auto select_iter = best_paths_at_node.begin();
                 select_iter != best_paths_at_node.end(); select_iter++) {
                if (should_skip_iter != merge_result.to_boot.end()) {
                    if (*should_skip_iter ==
                        std::distance(best_paths_at_node.begin(), select_iter)) {
                        should_skip_iter++;
                        continue;
                    }
                }
                *insert_iter = std::move(*select_iter);
                insert_iter++;
            }
            best_paths_at_node.resize(best_paths_at_node.size() - merge_result.to_boot.size());
        }


        // Create new paths from neighbors and queue.
        for (int other_node_id = 0; other_node_id < static_cast<int>(road_map.points.size());
             other_node_id++) {
            if (road_map.adj(current_node_id, other_node_id)) {
                std::vector<int> new_path(in_progress.path_to_goal.begin(),
                                          in_progress.path_to_goal.end());
                new_path.push_back(other_node_id);
                const auto prop_result =
                    rev_propagator(other_node_id, current_node_id, in_progress.info_lower_bound);

                std::cout << "Considering neighbor: " << other_node_id << " info_lb: " << prop_result.info_lower_bound << " edge_cost " << prop_result.edge_cost << std::endl;
                if (!std::isfinite(prop_result.info_lower_bound)) {
                    continue;
                }
                open_list.emplace(detail::InProgressPath{
                    .info_lower_bound = prop_result.info_lower_bound,
                   .cost_to_go = prop_result.edge_cost + in_progress.cost_to_go,
                    .path_to_goal = std::move(new_path),
                });
            }
        }

        best_paths_at_node.emplace_back(std::move(in_progress));
    }

    return {};
}
}  // namespace robot::experimental::beacon_sim
