#include "experimental/learn_descriptors/structure_from_motion.hh"

#include <iostream>
#include <sstream>

#include "Eigen/Geometry"
#include "common/geometry/opencv_viz.hh"
#include "common/geometry/translate_types.hh"
#include "experimental/learn_descriptors/symphony_lake_parser.hh"
#include "gtest/gtest.h"

class GtsamTestHelper {
   public:
    static bool pixel_in_range(Eigen::Vector2d pixel, size_t img_width, size_t img_height) {
        return pixel[0] > 0 && pixel[0] < img_width && pixel[1] > 0 && pixel[1] < img_height;
    }
};

namespace robot::experimental::learn_descriptors {
TEST(SFM_TEST, frontend_pipeline_sweep) {
    const size_t width = 640;
    const size_t height = 480;

    const size_t pixel_shift_x = 20;
    const size_t PIXEL_COMP_TOL = 20;

    cv::Mat image_1 = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat image_2;

    cv::Mat translation_mat = (cv::Mat_<double>(2, 3) << 1, 0, pixel_shift_x, 0, 1, 0);
    cv::Point rect_points[4] = {{150, 200}, {350, 200}, {350, 300}, {150, 300}};
    float com_x = 0.0f, com_y = 0.0f;
    for (const cv::Point &rect_point : rect_points) {
        com_x += rect_point.x;
        com_y += rect_point.y;
    }
    com_x *= 0.25;
    com_y *= 0.25;
    cv::Point rotation_center(com_x, com_y);
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(rotation_center, 45, 1.0);

    const size_t line_spacing = 100;
    for (size_t i = 0; i <= width / line_spacing; i++) {
        size_t x = i * line_spacing + (width % line_spacing) / 2;
        size_t b = i * 255.0 / (width / line_spacing);
        size_t g = 255.0 - i * 255.0 / (width / line_spacing);
        cv::line(image_1, cv::Point(x, 0), cv::Point(x, height - 1), cv::Scalar(b, g, 0), 2);
    }
    for (size_t i = 0; i <= height / line_spacing; i++) {
        size_t y = i * line_spacing + (height % line_spacing) / 2;
        size_t b = i * 255.0 / (width / line_spacing);
        size_t g = 255.0 - i * 255.0 / (width / line_spacing);
        cv::line(image_1, cv::Point(0, y), cv::Point(width - 1, y), cv::Scalar(b, g, 0), 2);
    }

    cv::warpAffine(image_1, image_1, rotation_matrix, image_1.size());
    cv::warpAffine(image_1, image_2, translation_mat, image_1.size());

    // cv::Mat img_test_disp;
    // cv::hconcat(image_1, image_2, img_test_disp);
    // cv::imshow("Test", img_test_disp);
    // cv::waitKey(1000);

    Frontend::ExtractorType extractor_types[2] = {Frontend::ExtractorType::SIFT,
                                                  Frontend::ExtractorType::ORB};
    Frontend::MatcherType matcher_types[3] = {Frontend::MatcherType::BRUTE_FORCE,
                                              Frontend::MatcherType::FLANN,
                                              Frontend::MatcherType::KNN};

    Frontend frontend;
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> keypoints_descriptors_pair_1;
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> keypoints_descriptors_pair_2;
    std::vector<cv::DMatch> matches;
    cv::Mat img_keypoints_out_1(height, width, CV_8UC3),
        img_keypoints_out_2(height, width, CV_8UC3), img_matches_out(height, 2 * width, CV_8UC3);
    // cv::Mat img_display_test;
    for (Frontend::ExtractorType extractor_type : extractor_types) {
        for (Frontend::MatcherType matcher_type : matcher_types) {
            // printf("started frontend combination: (%d, %d)\n", static_cast<int>(extractor_type),
            //        static_cast<int>(matcher_type));
            try {
                frontend = Frontend(extractor_type, matcher_type);
            } catch (const std::invalid_argument &e) {
                assert(std::string(e.what()) == "FLANN can not be used with ORB.");  // very jank...
                continue;
            }
            keypoints_descriptors_pair_1 = frontend.get_keypoints_and_descriptors(image_1);
            keypoints_descriptors_pair_2 = frontend.get_keypoints_and_descriptors(image_2);
            matches = frontend.get_matches(keypoints_descriptors_pair_1.second,
                                           keypoints_descriptors_pair_2.second);
            frontend.draw_keypoints(image_1, keypoints_descriptors_pair_1.first,
                                    img_keypoints_out_1);
            frontend.draw_keypoints(image_2, keypoints_descriptors_pair_2.first,
                                    img_keypoints_out_2);
            frontend.draw_matches(image_1, keypoints_descriptors_pair_1.first, image_2,
                                  keypoints_descriptors_pair_2.first, matches, img_matches_out);
            // cv::hconcat(img_keypoints_out_1, img_keypoints_out_2, img_display_test);
            // cv::vconcat(img_display_test, img_matches_out, img_display_test);
            // std::stringstream text;
            // text << "Extractor " << static_cast<int>(extractor_type) << ", matcher "
            //      << static_cast<int>(matcher_type);
            // cv::putText(img_display_test, text.str(), cv::Point(20, height - 50),
            //             cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            // cv::imshow("Keypoints and Matches Output.", img_display_test);
            // std::cout << "Press spacebar to pause." << std::endl;
            // while (cv::waitKey(1000) == 32) {
            // }
            // printf("completed frontend combination: (%d, %d)\n",
            // static_cast<int>(extractor_type),
            //        static_cast<int>(matcher_type));
            if (extractor_type != Frontend::ExtractorType::ORB) {  // don't check ORB for now
                for (const cv::DMatch match : matches) {
                    EXPECT_NEAR(keypoints_descriptors_pair_1.first[match.queryIdx].pt.x -
                                    keypoints_descriptors_pair_2.first[match.trainIdx].pt.x,
                                pixel_shift_x, pixel_shift_x + PIXEL_COMP_TOL);
                    EXPECT_NEAR(keypoints_descriptors_pair_2.first[match.trainIdx].pt.y -
                                    keypoints_descriptors_pair_1.first[match.queryIdx].pt.y,
                                0, PIXEL_COMP_TOL);
                }
            }
        }
    }
}

// TEST(SFM_TEST, boat_and_cam_frames) {    
//     // world is boat frame. +x is boat "forward", +y is the right side, +z is "down" into the water/hull
//     Eigen::Isometry3d T_boat_cam = StructureFromMotion::T_symlake_boat_cam;
//     T_boat_cam.translation() = Eigen::Vector3d::Ones();
//     geometry::viz_scene(std::vector<Eigen::Isometry3d>{T_boat_cam}, std::vector<Eigen::Vector3d>());
// }

TEST(SFM_TEST, sfm_snippet_small) {
    const std::vector<int> indices {120, 130}; // 0-199
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
    Eigen::Isometry3d T_earth_world = DataParser::get_T_world_boat(img_pt_first);
    Eigen::Isometry3d T_world_camera0 = DataParser::get_T_boat_camera(img_pt_first);
    StructureFromMotion sfm(Frontend::ExtractorType::SIFT, K, D, gtsam::Pose3(T_world_camera0.matrix()));  

    std::vector<cv::Mat> img_vector;
    for (const int &idx : indices) {    
        const cv::Mat img = survey.loadImageByImageIndex(idx);
        img_vector.push_back(img);
        const symphony_lake_dataset::ImagePoint img_pt = survey.getImagePoint(idx);        
        Eigen::Isometry3d T_earth_boat = DataParser::get_T_world_boat(img_pt);
        Eigen::Isometry3d T_world_boat = T_earth_world.inverse() * T_earth_boat;
        Eigen::Isometry3d T_world_cam = T_world_boat * DataParser::get_T_boat_camera(img_pt);

        // T_world_cam.linear() = T_world_camera0.linear().matrix();
        
        sfm.add_image(img, gtsam::Pose3(T_world_cam.matrix()));
    }
    // for (const cv::Mat &image : images) {
    //     sfm.add_image(image);
    // }

    const gtsam::Values initial_values = sfm.get_backend().get_current_initial_values();
    std::vector<Eigen::Isometry3d> poses_world;
    for (size_t i = 0; i < indices.size(); i++) {
        gtsam::Pose3 pose =
            initial_values.at<gtsam::Pose3>(gtsam::Symbol(sfm.get_backend().pose_symbol_char, i));
        poses_world.emplace_back(pose.matrix());
    }
    std::vector<Eigen::Vector3d> points_world;
    for (int i = 0; i < static_cast<int>(sfm.get_matches().size()); i++) {
        // NOTE: this j for lmk_symbol is wrong.
        for (int j = 0; j < static_cast<int>(sfm.get_matches()[i].size()); j++) {
            gtsam::Symbol lmk_symbol = gtsam::Symbol(Backend::landmark_symbol_char, j);
            if (initial_values.exists(lmk_symbol)) {
                points_world.emplace_back(initial_values.at<gtsam::Point3>(lmk_symbol));
            } else {
                std::cout << "lmk symbol doesn't exist in initial_values!" << std::endl;
            }
        }
    }
    // std::cout << poses_world.front() << std::endl;

    geometry::viz_scene(poses_world, points_world, true, true);

    std::cout << "Solving for structure!" << std::endl;

    sfm.solve_structure();

    std::cout << "Solution complete." << std::endl;

    const gtsam::Values result_values = sfm.get_structure_result();
    std::vector<Eigen::Isometry3d> final_poses;
    std::vector<Eigen::Vector3d> final_lmks;
    for (size_t i = 0; i < indices.size(); i++) {
        final_poses.emplace_back(result_values.at<gtsam::Pose3>(gtsam::Symbol(sfm.get_backend().pose_symbol_char, i)).matrix());
    }
    for (int i = 0; i < static_cast<int>(sfm.get_matches().size()); i++) {
        // NOTE: this j for lmk_symbol is wrong.
        for (int j = 0; j < static_cast<int>(sfm.get_matches()[i].size()); j++) {
            gtsam::Symbol lmk_symbol = gtsam::Symbol(Backend::landmark_symbol_char, j);
            if (initial_values.exists(lmk_symbol)) {
                final_lmks.emplace_back(initial_values.at<gtsam::Point3>(lmk_symbol));
            } else {
                std::cout << "lmk symbol doesn't exist in initial_values!" << std::endl;
            }
        }
    }
    geometry::viz_scene(final_poses, final_lmks, true, true);
}

TEST(SFM_TEST, sfm_building) {
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
    Eigen::Isometry3d T_earth_world = DataParser::get_T_world_boat(img_pt_first);
    Eigen::Isometry3d T_world_camera0 = DataParser::get_T_boat_camera(img_pt_first);
    StructureFromMotion sfm(Frontend::ExtractorType::SIFT, K, D, gtsam::Pose3(T_world_camera0.matrix()));  

    for (const int &idx : indices) {    
        const cv::Mat img = survey.loadImageByImageIndex(idx);
        const symphony_lake_dataset::ImagePoint img_pt = survey.getImagePoint(idx);  

        Eigen::Isometry3d T_earth_boat = DataParser::get_T_world_boat(img_pt);
        Eigen::Isometry3d T_world_boat = T_earth_world.inverse() * T_earth_boat;
        Eigen::Isometry3d T_world_cam = T_world_boat * DataParser::get_T_boat_camera(img_pt);
        
        sfm.add_image(img, gtsam::Pose3(T_world_cam.matrix()));
    }
    // for (const cv::Mat &image : images) {
    //     sfm.add_image(image);
    // }

    const gtsam::Values initial_values = sfm.get_backend().get_current_initial_values();
    std::vector<Eigen::Isometry3d> poses_world;
    for (size_t i = 0; i < indices.size(); i++) {
        gtsam::Pose3 pose =
            initial_values.at<gtsam::Pose3>(gtsam::Symbol(sfm.get_backend().pose_symbol_char, i));
        poses_world.emplace_back(pose.matrix());
    }
    std::vector<Eigen::Vector3d> points_world;
    for (int i = 0; i < static_cast<int>(sfm.get_matches().size()); i++) {
        // NOTE: this j for lmk_symbol is wrong.
        for (int j = 0; j < static_cast<int>(sfm.get_matches()[i].size()); j++) {
            gtsam::Symbol lmk_symbol = gtsam::Symbol(Backend::landmark_symbol_char, j);
            if (initial_values.exists(lmk_symbol)) {
                points_world.emplace_back(initial_values.at<gtsam::Point3>(lmk_symbol));
            } else {
                std::cout << "lmk symbol doesn't exist in initial_values!" << std::endl;
            }
        }
    }
    // std::cout << poses_world.front() << std::endl;

    geometry::viz_scene(poses_world, points_world, true, true);

    std::cout << "Solving for structure!" << std::endl;

    sfm.solve_structure();

    std::cout << "Solution complete." << std::endl;

    const gtsam::Values result_values = sfm.get_structure_result();
    std::vector<Eigen::Isometry3d> final_poses;
    std::vector<Eigen::Vector3d> final_lmks;
    for (size_t i = 0; i < indices.size(); i++) {
        final_poses.emplace_back(result_values.at<gtsam::Pose3>(gtsam::Symbol(sfm.get_backend().pose_symbol_char, i)).matrix());
    }
    for (int i = 0; i < static_cast<int>(sfm.get_matches().size()); i++) {
        // NOTE: this j for lmk_symbol is wrong.
        for (int j = 0; j < static_cast<int>(sfm.get_matches()[i].size()); j++) {
            gtsam::Symbol lmk_symbol = gtsam::Symbol(Backend::landmark_symbol_char, j);
            if (initial_values.exists(lmk_symbol)) {
                final_lmks.emplace_back(initial_values.at<gtsam::Point3>(lmk_symbol));
            } else {
                std::cout << "lmk symbol doesn't exist in initial_values!" << std::endl;
            }
        }
    }
    geometry::viz_scene(final_poses, final_lmks, true, true);
}
}  // namespace robot::experimental::learn_descriptors
