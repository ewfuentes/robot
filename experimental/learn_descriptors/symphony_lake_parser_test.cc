#include "experimental/learn_descriptors/symphony_lake_parser.hh"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common/check.hh"
#include "common/geometry/opencv_viz.hh"
#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

namespace robot::experimental::learn_descriptors {
namespace {
bool is_test() { return std::getenv("BAZEL_TEST") != nullptr; }
}  // namespace
TEST(SymphonyLakeParserTest, snippet_140106) {
    DataParser data_parser = SymphonyLakeDatasetTestHelper::get_test_parser();

    const symphony_lake_dataset::SurveyVector &survey_vector = data_parser.get_surveys();

    cv::Mat image;
    cv::Mat target_img;
    if (!is_test()) {
        cv::namedWindow("Symphony Dataset Image", cv::WINDOW_AUTOSIZE);
    }
    printf("Press 'q' in graphic window to quit\n");
    for (int i = 0; i < static_cast<int>(survey_vector.getNumSurveys()); i++) {
        const symphony_lake_dataset::Survey &survey = survey_vector.get(i);
        for (int j = 0; j < static_cast<int>(survey.getNumImages()); j++) {
            // img_point contains all the csv data entries
            const symphony_lake_dataset::ImagePoint img_point = survey.getImagePoint(j);
            (void)img_point;  // suppress unused variable
            image = survey.loadImageByImageIndex(j);

            // get the target image
            std::stringstream target_img_name;
            target_img_name << "0000" << j << ".jpg";
            const size_t target_img_name_length = 8;
            std::string target_img_name_str = target_img_name.str();
            target_img_name_str.replace(0, target_img_name_str.size() - target_img_name_length, "");
            std::filesystem::path target_img_dir =
                SymphonyLakeDatasetTestHelper::get_test_iamge_root_dir() /
                SymphonyLakeDatasetTestHelper::get_test_survey_list()[i] / "0027" /
                target_img_name_str;
            target_img = cv::imread(target_img_dir.string());

            EXPECT_TRUE(SymphonyLakeDatasetTestHelper::images_equal(image, target_img));
            // if (!is_test()) {
            //     cv::imshow("Symphony Dataset Image", image);
            //     cv::waitKey(2);
            // }
            // cv::imshow("Symphony Dataset Image", image);
            // cv::waitKey(200);
        }
    }
}

TEST(SymphonyLakeParserTest, test_cam_frames) {
    const std::vector<int> indices = []() {
        std::vector<int> tmp;
        for (int i = 0; i < 200; i += 10) {
            tmp.push_back(i);
        }
        return tmp;
    }();
    DataParser data_parser = SymphonyLakeDatasetTestHelper::get_test_parser();
    const symphony_lake_dataset::SurveyVector &survey_vector = data_parser.get_surveys();
    const symphony_lake_dataset::Survey &survey = survey_vector.get(0);
    const symphony_lake_dataset::ImagePoint image_point_first =
        survey.getImagePoint(indices.front());

    std::vector<Eigen::Isometry3d> cam_frames;

    // NOTE: the world in these images is east, north, up centered at boat0 translation
    Eigen::Vector3d t_world_boat0 = DataParser::get_T_world_boat(image_point_first).translation();

    for (size_t i = 0; i < indices.size(); i++) {
        const symphony_lake_dataset::ImagePoint img_pt = survey.getImagePoint(indices[i]);
        Eigen::Isometry3d T_world_boatidx = DataParser::get_T_world_boat(img_pt);
        Eigen::Isometry3d T_boatidx_camidx =
            DataParser::get_T_boat_camera(img_pt);  // current boat to current camera

        Eigen::Isometry3d T_world_camidx = T_world_boatidx * T_boatidx_camidx;
        T_world_camidx.translation() -= t_world_boat0;

        cam_frames.push_back(T_world_camidx);
    }

    geometry::viz_scene(cam_frames, std::vector<Eigen::Vector3d>(), true, true, "test_cam_frames");
}

TEST(SymphonyLakeParserTest, test_gps_frames) {
    const std::vector<int> indices = []() {
        std::vector<int> tmp;
        for (int i = 0; i < 200; i += 10) {
            tmp.push_back(i);
        }
        return tmp;
    }();
    DataParser data_parser = SymphonyLakeDatasetTestHelper::get_test_parser();
    const symphony_lake_dataset::SurveyVector &survey_vector = data_parser.get_surveys();
    const symphony_lake_dataset::Survey &survey = survey_vector.get(0);
    const symphony_lake_dataset::ImagePoint image_point_first =
        survey.getImagePoint(indices.front());

    std::vector<Eigen::Isometry3d> gps_frames;

    // NOTE: the world in these images is east, north, up centered at boat0 translation
    Eigen::Vector3d t_world_boat0 = DataParser::get_T_world_boat(image_point_first).translation();
    // Eigen::Vector3d t_world_gps0(image_point_first.x, image_point_first.y, 0);

    for (size_t i = 0; i < indices.size(); i++) {
        const symphony_lake_dataset::ImagePoint img_pt = survey.getImagePoint(indices[i]);
        Eigen::Isometry3d T_world_gpsidx = DataParser::get_T_world_gps(img_pt);
        T_world_gpsidx.translation() -= t_world_boat0;
        gps_frames.push_back(T_world_gpsidx);
    }

    geometry::viz_scene(gps_frames, std::vector<Eigen::Vector3d>(), true, true, "test_gps_frames");
}
}  // namespace robot::experimental::learn_descriptors
