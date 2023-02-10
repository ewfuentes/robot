
#include "planning/road_map_to_proto.hh"

#include "gtest/gtest.h"

namespace robot::planning {

TEST(RoadMapToProtoTest, pack_unpack) {
    // Setup
    const RoadMap road_map = {
        .points = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}},
        .adj = (Eigen::MatrixXd(2, 3) << 10.0, 11.0, 12.0, 13.0, 14.0, 15.0).finished(),
    };

    // Action
    proto::RoadMap proto;
    pack_into(road_map, &proto);
    const RoadMap unpacked = unpack_from(proto);

    // Verification
    EXPECT_EQ(road_map.points.size(), unpacked.points.size());
    EXPECT_EQ(road_map.adj.rows(), unpacked.adj.rows());
    EXPECT_EQ(road_map.adj.cols(), unpacked.adj.cols());

    for (int i = 0; i < static_cast<int>(road_map.points.size()); i++) {
        EXPECT_EQ(road_map.points.at(i).x(), unpacked.points.at(i).x());
        EXPECT_EQ(road_map.points.at(i).y(), unpacked.points.at(i).y());
    }

    for (int r = 0; r < static_cast<int>(road_map.adj.rows()); r++) {
        for (int c = 0; c < static_cast<int>(road_map.adj.rows()); c++) {
            EXPECT_EQ(road_map.adj(r, c), unpacked.adj(r, c));
        }
    }
}
}  // namespace robot::planning
