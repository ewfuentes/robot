
#include "experimental/beacon_sim/beacon_observation_to_proto.hh"

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {

TEST(BeaconObservationToProtoTest, empty_observation_pack_unpack) {
    // Setup
    const BeaconObservation in;

    // Action
    proto::BeaconObservation proto;
    pack_into(in, &proto);
    const BeaconObservation out = unpack_from(proto);

    // Verification
    EXPECT_FALSE(out.maybe_id.has_value());
    EXPECT_FALSE(out.maybe_range_m.has_value());
    EXPECT_FALSE(out.maybe_bearing_rad.has_value());
}

TEST(BeaconObservationToProtoTest, full_observation_pack_unpack) {
    // Setup
    const BeaconObservation in = {
        .maybe_id = 123,
        .maybe_range_m = -456.0,
        .maybe_bearing_rad = 135.79,
    };

    // Action
    proto::BeaconObservation proto;
    pack_into(in, &proto);
    const BeaconObservation out = unpack_from(proto);

    // Verification
    EXPECT_EQ(out.maybe_id, in.maybe_id);
    EXPECT_EQ(out.maybe_range_m, in.maybe_range_m);
    EXPECT_EQ(out.maybe_bearing_rad, in.maybe_bearing_rad);
}

TEST(BeaconObservationToProtoTest, partial_observation_pack_unpack) {
    // Setup
    const std::vector<BeaconObservation> in = {
        {
            .maybe_id = std::nullopt,
            .maybe_range_m = -456.0,
            .maybe_bearing_rad = 135.79,
        },
        {
            .maybe_id = 123,
            .maybe_range_m = std::nullopt,
            .maybe_bearing_rad = 135.79,
        },
        {
            .maybe_id = 123,
            .maybe_range_m = -456.0,
            .maybe_bearing_rad = std::nullopt,
        },
    };

    // Action
    proto::BeaconObservations proto;
    pack_into(in, &proto);
    const std::vector<BeaconObservation> out = unpack_from(proto);

    // Verification
    EXPECT_EQ(out.size(), in.size());
    for (int i = 0; i < static_cast<int>(out.size()); i++) {
        EXPECT_EQ(out.at(i).maybe_id, in.at(i).maybe_id);
        EXPECT_EQ(out.at(i).maybe_range_m, in.at(i).maybe_range_m);
        EXPECT_EQ(out.at(i).maybe_bearing_rad, in.at(i).maybe_bearing_rad);
    }
}

}  // namespace robot::experimental::beacon_sim
