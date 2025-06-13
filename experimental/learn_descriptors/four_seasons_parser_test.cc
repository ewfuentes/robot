#include "experimental/learn_descriptors/four_seasons_parser.hh"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "common/liegroups/se3.hh"
#include "gtest/gtest.h"

class FourSeasonsParserTestHelper {
   public:
    static bool images_equal(cv::Mat img1, cv::Mat img2) {
        if (img1.size() != img2.size() || img1.type() != img2.type()) {
            return false;
        }
        cv::Mat diff;
        cv::absdiff(img1, img2, diff);
        diff = diff.reshape(1);
        return cv::countNonZero(diff) == 0;
    }
};
namespace robot::experimental::learn_descriptors {
TEST(FourSeasonsParserTest, parser_test) {
    const std::filesystem::path dir_snippet =
        "external/four_seasons_snippet/recording_2020-04-07_11-33-45";
    const std::filesystem::path dir_calibration = "external/four_seasons_snippet/calibration";

    std::cout << "dir_snippet: " << dir_snippet
              << "\texists: " << std::filesystem::exists(dir_snippet) << "\n"
              << "dir_calibration: " << dir_calibration << "\texists"
              << std::filesystem::exists(dir_calibration) << std::endl;

    FourSeasonsParser parser(dir_snippet, dir_calibration);

    // target transformations test
    const liegroups::SE3 T_S_AS(Eigen::Quaterniond(0.999992, 0.000869, 0.003288, -0.002016),
                                Eigen::Vector3d::Zero());
    const liegroups::SE3 T_cam_imu(Eigen::Quaterniond(-0.002350, -0.007202, 0.708623, -0.705546),
                                   Eigen::Vector3d(0.175412, 0.003689, -0.058106));
    const liegroups::SE3 T_w_gpsw(Eigen::Quaterniond(0.802501, 0.000344, -0.000541, 0.596650),
                                  Eigen::Vector3d(0.256671, 0.021142, 0.073751));
    const liegroups::SE3 T_gps_imu(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
    const liegroups::SE3 T_e_gpsw(Eigen::Quaterniond(0.590438, 0.224931, 0.275937, 0.724326),
                                  Eigen::Vector3d(4164702.580389, 857109.771387, 4738828.771006));
    const double gnss_scale = 0.911501;
    EXPECT_TRUE(T_S_AS.matrix().isApprox(parser.get_T_S_AS().matrix()));
    EXPECT_TRUE(T_cam_imu.matrix().isApprox(parser.get_T_cam_imu().matrix()));
    EXPECT_TRUE(T_w_gpsw.matrix().isApprox(parser.get_T_w_gpsw().matrix()));
    EXPECT_TRUE(T_gps_imu.matrix().isApprox(parser.get_T_gps_imu().matrix()));
    EXPECT_TRUE(T_e_gpsw.matrix().isApprox(parser.get_T_e_gpsw().matrix()));
    EXPECT_DOUBLE_EQ(gnss_scale, parser.get_gnss_scale());

    std::cout << "T_S_AS: \n"
              << parser.get_T_S_AS().matrix() << "\nT_cam_imu : \n"
              << parser.get_T_cam_imu().matrix() << "\nT_gps_imu: \n"
              << parser.get_T_gps_imu().matrix() << "\nT_e_gpsw: \n"
              << parser.get_T_e_gpsw().matrix() << "\ngnss scale: \n"
              << parser.get_gnss_scale() << std::endl;

    EXPECT_NE(parser.num_images(), 0);

    cv::Mat img_first_and_last;
    cv::hconcat(parser.load_image(0), parser.load_image(parser.num_images() - 1),
                img_first_and_last);

    const std::filesystem::path dir_img = dir_snippet / "distorted_images/cam0";
    const std::filesystem::path path_img_time = dir_img / "times.txt";
    const std::filesystem::path path_gps = dir_img / "GNSSPoses.txt";
    const std::filesystem::path path_result = dir_img / "result.txt";
    std::ifstream file_img_times(path_img_time);
    std::ifstream file_gps(path_gps);
    std::string line;
    std::getline(file_gps, line);
    std::ifstream file_result(path_result);
    for (size_t i = 0; i < parser.num_images(); i++) {
        const ImagePoint img_pt = parser.get_image_point(i);
        const cv::Mat img = parser.load_image(i);

        // target image test
        const std::filesystem::path path_img = dir_img / (std::to_string(img_pt.seq) + ".png");
        const cv::Mat img_target = cv::imread(path_img);
        EXPECT_TRUE(FourSeasonsParserTestHelper::images_equal(img_target, img));

        // seq (time in nanoseconds) test

        // gps test
        line.clear();
        std::getline(file_gps, line);
        const std::vector<std::string>(absl::StrSplit(line, ','));
        const liegroups::SE3 transform_target(
            Eigen::Quaterniond(std::stod(std::string(parsed_line_gps[7])),
                               std::stod(parsed_line_gps[4]), std::stod(parsed_line_gps[5]),
                               std::stod(parsed_line_gps[6])),
            Eigen::Vector3d(std::stod(parsed_line_gps[1]), std::stod(parsed_line_gps[2]),
                            std::stod(parsed_line_gps[3])));

        std::cout << img_pt.to_string() << "\n";
    }
}
}  // namespace robot::experimental::learn_descriptors