#include "experimental/learn_descriptors/structure_from_motion.hh"

#include <iostream>
#include <sstream>
#include <thread>

#include "Eigen/Geometry"
#include "common/geometry/translate_types.hh"
#include "experimental/learn_descriptors/frame.hh"
#include "experimental/learn_descriptors/symphony_lake_parser.hh"
#include "gtest/gtest.h"
#include "visualization/opencv/opencv_viz.hh"

class GtsamTestHelper {
   public:
    static bool pixel_in_range(Eigen::Vector2d pixel, size_t img_width, size_t img_height) {
        return pixel[0] > 0 && pixel[0] < img_width && pixel[1] > 0 && pixel[1] < img_height;
    }
};

namespace robot::experimental::learn_descriptors {
TEST(StructureFromMotiontest, sfm_snippet_small) {
    const std::vector<int> indices{120, 130};  // 0-199
    DataParser data_parser = SymphonyLakeDatasetTestHelper::get_test_parser();
    const symphony_lake_dataset::SurveyVector &survey_vector = data_parser.get_surveys();
    const symphony_lake_dataset::Survey &survey = survey_vector.get(0);
    const symphony_lake_dataset::ImagePoint img_pt_first = survey.getImagePoint(indices.front());

    // const size_t img_width = img_pt_first.width, img_height = img_pt_first.height;
    const double fx = img_pt_first.fx, fy = img_pt_first.fy;
    const double cx = img_pt_first.cx, cy = img_pt_first.cy;
    gtsam::Cal3_S2 K(fx, fy, 0, cx, cy);
    Eigen::Matrix<double, 5, 1> D =
        (Eigen::Matrix<double, 5, 1>() << SymphonyLakeCamParams::k1, SymphonyLakeCamParams::k2,
         SymphonyLakeCamParams::p1, SymphonyLakeCamParams::p2, SymphonyLakeCamParams::k3)
            .finished();

    // let world be the first boat base recorded. T_world_camera0 = T_earth_boat0 * T_boat_camera
    // T_earth_boat0 =
    Eigen::Isometry3d T_earth_world = DataParser::get_world_from_boat(img_pt_first);
    Eigen::Isometry3d T_world_camera0 = DataParser::get_boat_from_camera(img_pt_first);
    StructureFromMotion sfm(Frontend::ExtractorType::SIFT, K, D,
                            gtsam::Pose3(T_world_camera0.matrix()));

    std::vector<cv::Mat> img_vector;
    for (const int &idx : indices) {
        const cv::Mat img = survey.loadImageByImageIndex(idx);
        img_vector.push_back(img);
        const symphony_lake_dataset::ImagePoint img_pt = survey.getImagePoint(idx);
        Eigen::Isometry3d T_earth_boat = DataParser::get_world_from_boat(img_pt);
        Eigen::Isometry3d T_world_boat = T_earth_world.inverse() * T_earth_boat;
        Eigen::Isometry3d T_world_cam = T_world_boat * DataParser::get_boat_from_camera(img_pt);

        sfm.add_image(img, gtsam::Pose3(T_world_cam.matrix()));
    }

    const gtsam::Values initial_values = sfm.get_backend().get_current_initial_values();

    sfm.graph_values(initial_values, "initial_values");

    std::cout << "Solving for structure!" << std::endl;

    sfm.solve_structure();

    std::cout << "Solution complete." << std::endl;

    const gtsam::Values result_values = sfm.get_structure_result();
    sfm.graph_values(result_values, "result_values");
}

TEST(StructureFromMotiontest, sfm_building) {
    // indices 0-199
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
    const symphony_lake_dataset::ImagePoint img_pt_first = survey.getImagePoint(indices.front());

    // const size_t img_width = img_pt_first.width, img_height = img_pt_first.height;
    const double fx = img_pt_first.fx, fy = img_pt_first.fy;
    const double cx = img_pt_first.cx, cy = img_pt_first.cy;
    gtsam::Cal3_S2 K(fx, fy, 0, cx, cy);
    Eigen::Matrix<double, 5, 1> D =
        (Eigen::Matrix<double, 5, 1>() << SymphonyLakeCamParams::k1, SymphonyLakeCamParams::k2,
         SymphonyLakeCamParams::p1, SymphonyLakeCamParams::p2, SymphonyLakeCamParams::k3)
            .finished();

    // let world be the first boat base recorded. T_world_camera0 = T_earth_boat0 * T_boat_camera
    Eigen::Isometry3d T_earth_boat0 = DataParser::get_world_from_boat(img_pt_first);
    Eigen::Isometry3d T_world_boat0;
    T_world_boat0.linear() = T_earth_boat0.linear();
    Eigen::Isometry3d T_world_camera0 =
        T_world_boat0 * DataParser::get_boat_from_camera(img_pt_first);
    StructureFromMotion sfm(Frontend::ExtractorType::SIFT, K, D,
                            gtsam::Pose3(T_world_camera0.matrix()));

    for (const int &idx : indices) {
        const cv::Mat img = survey.loadImageByImageIndex(idx);
        const symphony_lake_dataset::ImagePoint img_pt = survey.getImagePoint(idx);

        Eigen::Isometry3d T_world_boat = DataParser::get_world_from_boat(img_pt);
        T_world_boat.translation() -= T_earth_boat0.translation();

        Eigen::Isometry3d T_world_cam = T_world_boat * DataParser::get_boat_from_camera(img_pt);

        sfm.add_image(img, gtsam::Pose3(T_world_cam.matrix()));
    }

    const gtsam::Values initial_values = sfm.get_backend().get_current_initial_values();
    sfm.graph_values(initial_values, "initial values");

    std::cout << "Solving for structure!" << std::endl;

    Backend::graph_step_debug_func solve_iter_debug_func = [&sfm](const gtsam::Values &vals,
                                                                  const Backend::epoch iter) {
        std::cout << "iteration " << iter << " complete!";
        std::string window_name = "Iteration_" + std::to_string(iter);
        sfm.graph_values(vals, window_name);
    };
    sfm.solve_structure(5, solve_iter_debug_func);

    std::cout << "Solution complete." << std::endl;

    const gtsam::Values result_values = sfm.get_structure_result();
    sfm.graph_values(result_values, "optimized values");
}

TEST(StructureFromMotiontest, random_test) {
    std::cout << DataParser::T_boat_gps.matrix() << std::endl;
    geometry::viz_scene(
        std::vector<Eigen::Isometry3d>{DataParser::T_boat_imu, DataParser::T_boat_gps},
        std::vector<Eigen::Vector3d>{DataParser::t_boat_cam});
}
}  // namespace robot::experimental::learn_descriptors
