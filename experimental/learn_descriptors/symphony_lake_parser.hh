#pragma once
#include <coroutine>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "symphony_lake_dataset/ImagePoint.h"
#include "symphony_lake_dataset/SurveyVector.h"

namespace robot::experimental::learn_descriptors {
class SymphonyLakeCamParams {
   public:
    constexpr static double fx = 759.308, fy = 690.44;
    constexpr static double px = 370.915, py = 250.909;

    constexpr static double k1 = -0.302805, k2 = 0.171088, k3 = 0.0;
    constexpr static double p1 = 0.001151, p2 = -0.00038;
};
class DataParser {
   public:
    static const Eigen::Vector3d t_boat_cam;
    static const Eigen::Isometry3d T_boat_gps;
    static const Eigen::Isometry3d T_boat_imu;

    DataParser(const std::filesystem::path &image_root_dir,
               const std::vector<std::string> &survey_list);
    ~DataParser();

    // Eigen::Affine3d get_T_world_camera(size_t survey_idx, size_t image_idx, bool use_gps = false,
    //                                    bool use_compass = false);
    static const Eigen::Isometry3d get_boat_from_camera(
        const symphony_lake_dataset::ImagePoint &img_pt);
    static const Eigen::Isometry3d get_boat_from_camera(double theta_pan, double theta_tilt);
    static const Eigen::Isometry3d get_world_from_gps(
        const symphony_lake_dataset::ImagePoint &img_pt);
    /// @brief get_world_from_boat assuming z_axis_boat dot z_axis_world ~ -1
    /// @param img_pt
    /// @return world_from_boat
    static const Eigen::Isometry3d get_world_from_boat(
        const symphony_lake_dataset::ImagePoint &img_pt);

    const symphony_lake_dataset::SurveyVector &get_surveys() const { return surveys_; };

   private:
    std::filesystem::path image_root_dir_;
    std::vector<std::string> survey_list_;
    symphony_lake_dataset::SurveyVector surveys_;
};
class SymphonyLakeDatasetTestHelper {
   public:
    static constexpr const char *test_image_root_dir =
        "external/symphony_lake_snippet/symphony_lake";
    static constexpr std::array<const char *, 1> test_survey_list = {"140106_snippet"};
    static bool images_equal(cv::Mat img1, cv::Mat img2) {
        if (img1.size() != img2.size() || img1.type() != img2.type()) {
            return false;
        }
        cv::Mat diff;
        cv::absdiff(img1, img2, diff);
        diff = diff.reshape(1);
        return cv::countNonZero(diff) == 0;
    };
    static DataParser get_test_parser() {
        return DataParser(get_test_iamge_root_dir(), get_test_survey_list());
    };
    static std::filesystem::path get_test_iamge_root_dir() {
        return std::filesystem::path(test_image_root_dir);
    };
    static std::vector<std::string> get_test_survey_list() {
        return std::vector<std::string>(test_survey_list.begin(), test_survey_list.end());
    };
};
}  // namespace robot::experimental::learn_descriptors