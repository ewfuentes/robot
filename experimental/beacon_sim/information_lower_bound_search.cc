
#include "experimental/beacon_sim/information_lower_bound_search.hh"

#include <algorithm>
#include <queue>

namespace robot::experimental::beacon_sim {
namespace detail {
MergeResult should_merge(const std::vector<InProgressPath> &existing,
                         const InProgressPath &new_path) {
    if (existing.empty()) {
        return {.should_merge = true, .dominated_paths_idxs = {}};
    }

    // Path A is dominated by Path B if both the info. lower bound and the cost of B is lower than
    // A. It is assumed that among paths in the existing list, no path is dominated by any other
    // path. This function then determines if new_path is dominated by any path in `existing`,
    // and if not, whether any paths in `existing` are dominated by the new path.
    //
    // If `new_path` is not dominated, then it should be merged into the existing list. The
    // indices of the dominated paths are returned in `dominated_paths_idxs` in increasing order.

    std::vector<int> dominated_paths_idxs;
    bool should_merge = false;
    for (int i = 0; i < static_cast<int>(existing.size()); i++) {
        const InProgressPath &existing_path = existing.at(i);
        if (existing_path.info_lower_bound >= new_path.info_lower_bound &&
            existing_path.cost_to_go >= new_path.cost_to_go) {
            // The existing path is dominated by the new path, add it to the list of dominated
            // paths and mark the new path for inclusion.
            dominated_paths_idxs.push_back(i);
            should_merge = true;
        } else if (existing_path.info_lower_bound <= new_path.info_lower_bound &&
                   existing_path.cost_to_go <= new_path.cost_to_go) {
            // The new path is dominated by an existing path. Since the paths in `existing` are not
            // mutually dominating, `dominated_paths_idxs` will be empty. Mark the new path for
            // exclusion.
            should_merge = false;
            break;
        } else {
            // Either one of the info_lower_bound or the cost to go is lower than the existing
            // path, but the other is higher, so we keep both.
            should_merge = true;
        }
    }
    return {.should_merge = should_merge, .dominated_paths_idxs = std::move(dominated_paths_idxs)};
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
        // Note that this creates a copy. It is non-const so we can move it later.
        detail::InProgressPath in_progress = open_list.top();
        open_list.pop();

        const int current_node_id = in_progress.path_to_goal.back();
        std::vector<detail::InProgressPath> &best_paths_at_node =
            paths_from_node_id[current_node_id];

        if (in_progress.path_to_goal.back() == start_idx &&
            start_information >= in_progress.info_lower_bound) {
            std::reverse(in_progress.path_to_goal.begin(), in_progress.path_to_goal.end());
            return {
                .info_lower_bound = in_progress.info_lower_bound,
                .cost_to_go = in_progress.cost_to_go,
                .path_to_goal = in_progress.path_to_goal,
            };
        }

        // Try to merge the path into the per node queues
        const detail::MergeResult merge_result = should_merge(best_paths_at_node, in_progress);

        // If the path wasn't added, then we continue to the next path
        if (!merge_result.should_merge) {
            continue;
        }

        // remove any paths that have been dominated
        if (!merge_result.dominated_paths_idxs.empty()) {
            // Shift any elements that should be kept to the front of the list, overwriting those
            // that will be removed.
            auto insert_iter = best_paths_at_node.begin();
            auto should_skip_iter = merge_result.dominated_paths_idxs.begin();
            for (auto select_iter = best_paths_at_node.begin();
                 select_iter != best_paths_at_node.end(); select_iter++) {
                if (should_skip_iter != merge_result.dominated_paths_idxs.end()) {
                    if (*should_skip_iter ==
                        std::distance(best_paths_at_node.begin(), select_iter)) {
                        should_skip_iter++;
                        continue;
                    }
                }
                *insert_iter = std::move(*select_iter);
                insert_iter++;
            }
            // Resize the vector to the number of expected paths.
            best_paths_at_node.resize(best_paths_at_node.size() -
                                      merge_result.dominated_paths_idxs.size());
        }

        // Create new paths from neighbors and queue.
        for (int other_node_id = 0; other_node_id < static_cast<int>(road_map.points.size());
             other_node_id++) {
            if (road_map.adj(current_node_id, other_node_id)) {
                // For each adjacent node to the current node, create a new in progress path and
                // propagate the information lower bound.
                std::vector<int> new_path(in_progress.path_to_goal.begin(),
                                          in_progress.path_to_goal.end());
                new_path.push_back(other_node_id);
                const auto prop_result =
                    rev_propagator(other_node_id, current_node_id, in_progress.info_lower_bound);

                if (!std::isfinite(prop_result.info_lower_bound)) {
                    // If infinite information is required, it isn't possible to start at the node
                    // in question and follow the rest of the current path and satisfy the
                    // information lower bound at the goal. We choose not to consider this path.
                    continue;
                }
                open_list.emplace(detail::InProgressPath{
                    .info_lower_bound = prop_result.info_lower_bound,
                    .cost_to_go = prop_result.edge_cost + in_progress.cost_to_go,
                    .path_to_goal = std::move(new_path),
                });
            }
        }

        // Include the new path to the list of paths at the current node.
        best_paths_at_node.emplace_back(std::move(in_progress));
    }

    return {};
}
}  // namespace robot::experimental::beacon_sim
