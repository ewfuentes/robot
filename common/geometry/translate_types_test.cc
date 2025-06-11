#include "common/geometry/translate_types.hh"

#include "gtest/gtest.h"

namespace robot::geometry {
TEST(TranslateTypesTest, eigen_mat_and_cv) {
    int rows = 5, cols = 5;
    Eigen::MatrixXd mat_eig = Eigen::MatrixXd::Random(rows, cols);
    cv::Mat mat_cv_from_eig = eigen_mat_to_cv(mat_eig);
    Eigen::MatrixXd mat_eig_from_cv = cv_to_eigen_mat(mat_cv_from_eig);
    EXPECT_TRUE(static_cast<int>(mat_eig.rows()) == mat_cv_from_eig.rows);
    EXPECT_TRUE(static_cast<int>(mat_eig.cols()) == mat_cv_from_eig.cols);
    EXPECT_TRUE(static_cast<int>(mat_eig.rows()) == static_cast<int>(mat_eig_from_cv.rows()));
    EXPECT_TRUE(static_cast<int>(mat_eig.cols()) == static_cast<int>(mat_eig_from_cv.cols()));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            EXPECT_DOUBLE_EQ(mat_eig(i, j), mat_cv_from_eig.at<double>(i, j));
            EXPECT_DOUBLE_EQ(mat_eig(i, j), mat_eig_from_cv(i, j));
        }
    }
}

TEST(TranslateTypesTest, eigen_vec_and_cv) {
    int cols = 5;
    Eigen::VectorXd vec_eig = Eigen::VectorXd::Random(cols);
    cv::Mat mat_cv_from_eig = eigen_vec_to_cv(vec_eig);
    Eigen::VectorXd vec_eig_from_cv = Eigen::VectorXd(cv_to_eigen_mat(mat_cv_from_eig));
    EXPECT_TRUE(static_cast<int>(vec_eig.rows()) == mat_cv_from_eig.rows);
    EXPECT_TRUE(static_cast<int>(vec_eig.cols()) == mat_cv_from_eig.cols);
    EXPECT_TRUE(static_cast<int>(vec_eig.rows()) == static_cast<int>(vec_eig_from_cv.rows()));
    EXPECT_TRUE(static_cast<int>(vec_eig.cols()) == static_cast<int>(vec_eig_from_cv.cols()));
    for (int i = 0; i < cols; i++) {
        EXPECT_DOUBLE_EQ(vec_eig(i), mat_cv_from_eig.at<double>(i, 0));
        EXPECT_DOUBLE_EQ(vec_eig(i), vec_eig_from_cv(i, 0));
    }
}
}  // namespace robot::geometry