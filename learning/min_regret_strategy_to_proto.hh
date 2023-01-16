
#include <type_traits>

#include "cfr.hh"
#include "learning/min_regret_strategy.pb.h"

namespace robot::learning::proto {

template <typename T>
void pack_into(const learning::CountsFromInfoSetId<T> &counts_from_infoset_id,
               MinRegretStrategy *out) {
    for (const auto &[action, name] : Range<typename T::Actions>::value) {
        out->add_actions(std::string(name));
    }

    for (const auto &[id, counts] : counts_from_infoset_id) {
        InfoSetCounts &proto_counts = *(out->add_infoset_counts());
        if constexpr (std::is_same_v<typename T::InfoSetId, std::string>) {
            proto_counts.set_id_str(id);
        } else {
            proto_counts.set_id_num(id);
        }

        for (const auto &[action, item] : counts.regret_sum) {
            proto_counts.add_regret_sum(item);
        }
        for (const auto &[action, item] : counts.strategy_sum) {
            proto_counts.add_strategy_sum(item);
        }
        proto_counts.set_iter_count(counts.iter_count);
    }
}

template <typename T>
learning::CountsFromInfoSetId<T> unpack_from(const MinRegretStrategy &in) {
    learning::CountsFromInfoSetId<T> out;
    for (const auto &counts : in.infoset_counts()) {
        const typename T::InfoSetId id = [&]() {
            if constexpr (std::is_same_v<typename T::InfoSetId, std::string>) {
                return counts.id_str();
            } else {
                return counts.id_num();
            }
        }();
        learning::InfoSetCounts<T> out_counts;
        int i = 0;
        for (const auto &[action, _] : Range<typename T::Actions>::value) {
            // Ideally we would get the actions from the strings so that we would support adding new
            // actions Punt on this for now.
            out_counts.regret_sum[action] = counts.regret_sum(i);
            out_counts.strategy_sum[action] = counts.strategy_sum(i);
            i++;
        }
        out_counts.iter_count = counts.iter_count();
        out[id] = out_counts;
    }
    return out;
}
}  // namespace robot::learning::proto
