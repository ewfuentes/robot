
#include "common/math/matrix_to_proto.hh"

#include "gtest/gtest.h"

namespace robot::math {
TEST(MatrixToProtoTest, fixed_vector_pack_unpack) {
    // Setup
    const Eigen::Vector3d in(1.0, 2.0, 3.0);
    // Action

    proto::Matrix proto;
    pack_into(in, &proto);
    const Eigen::Vector3d out = unpack_from<Eigen::Vector3d>(proto);

    // Verification
    EXPECT_EQ(in.rows(), out.rows());
    EXPECT_EQ(in.cols(), out.cols());
    for (int i = 0; i < in.rows(); i++) {
        EXPECT_EQ(in(i), out(i));
    }
}

TEST(MatrixToProtoTest, dynamic_vector_pack_unpack) {
    // Setup
    const Eigen::VectorXd in{{1.0, 2.0, 3.0}};

    // Action
    proto::Matrix proto;
    pack_into(in, &proto);
    const Eigen::VectorXd out = unpack_from<Eigen::VectorXd>(proto);

    // Verification
    EXPECT_EQ(in.rows(), out.rows());
    EXPECT_EQ(in.cols(), out.cols());
    for (int i = 0; i < in.rows(); i++) {
        EXPECT_EQ(in(i), out(i));
    }
}

TEST(MatrixToProtoTest, dynamic_row_vector_pack_unpack) {
    // Setup
    const Eigen::RowVectorXd in{{1.0, 2.0, 3.0}};

    // Action
    proto::Matrix proto;
    pack_into(in, &proto);
    const Eigen::RowVectorXd out = unpack_from<Eigen::RowVectorXd>(proto);

    // Verification
    EXPECT_EQ(in.rows(), out.rows());
    EXPECT_EQ(in.cols(), out.cols());
    for (int i = 0; i < in.rows(); i++) {
        EXPECT_EQ(in(i), out(i));
    }
}

TEST(MatrixToProtoTest, fixed_matrix_pack_unpack) {
    // Setup
    const Eigen::Matrix3d in{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    // Action
    proto::Matrix proto;
    pack_into(in, &proto);
    const Eigen::Matrix3d out = unpack_from<Eigen::Matrix3d>(proto);

    // Verification
    EXPECT_EQ(in.rows(), out.rows());
    EXPECT_EQ(in.cols(), out.cols());
    for (int i = 0; i < in.rows(); i++) {
        EXPECT_EQ(in(i), out(i));
    }
}

TEST(MatrixToProtoTest, dynamic_matrix_pack_unpack) {
    // Setup
    const Eigen::MatrixXd in{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    // Action
    proto::Matrix proto;
    pack_into(in, &proto);
    const Eigen::MatrixXd out = unpack_from<Eigen::MatrixXd>(proto);

    // Verification
    EXPECT_EQ(in.rows(), out.rows());
    EXPECT_EQ(in.cols(), out.cols());
    for (int i = 0; i < in.rows(); i++) {
        EXPECT_EQ(in(i), out(i));
    }
}
}  // namespace robot::math
