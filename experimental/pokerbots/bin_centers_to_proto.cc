
#include "experimental/pokerbots/bin_centers_to_proto.hh"


namespace robot::experimental::pokerbots::proto {
pokerbots::PerTurnBinCenters unpack_from(const PerTurnBinCenters &in) {
    const auto unpack_vector = [](const auto &list) {
        std::vector<pokerbots::BinCenter> out;
        out.reserve(list.size());

        for (const auto &center : list) {
            out.push_back(unpack_from(center));
        }
        return out;
    };

    return {
        .preflop_centers = unpack_vector(in.preflop_centers()),
        .flop_centers = unpack_vector(in.flop_centers()),
        .turn_centers = unpack_vector(in.turn_centers()),
        .river_centers = unpack_vector(in.river_centers()),
    };
}
pokerbots::BinCenter unpack_from(const BinCenter &in) {
    return {
        .strength = in.strength(),
        .negative_potential = in.negative_potential(),
        .positive_potential = in.positive_potential(),
    };
}
}  // namespace robot::experimental::pokerbots::proto
