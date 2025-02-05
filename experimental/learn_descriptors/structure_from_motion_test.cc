#include "experimental/learn_descriptors/structure_from_motion.hh"

#include <iostream>
#include <sstream>

#include "Eigen/Geometry"
#include "common/geometry/opencv_viz.hh"
#include "common/math/matrix_to_proto.hh"
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
            printf("started frontend combination: (%d, %d)\n", static_cast<int>(extractor_type),
                   static_cast<int>(matcher_type));
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
            printf("completed frontend combination: (%d, %d)\n", static_cast<int>(extractor_type),
                   static_cast<int>(matcher_type));
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

TEST(GtsamTesting, sfm_test_estimate_pose) {
    std::vector<gtsam::Point3> cube_W;
    float cube_size = 1.0f;
    cube_W.push_back(gtsam::Point3(0, 0, 0));
    cube_W.push_back(gtsam::Point3(cube_size, 0, 0));
    cube_W.push_back(gtsam::Point3(cube_size, cube_size, 0));
    cube_W.push_back(gtsam::Point3(0, cube_size, 0));
    cube_W.push_back(gtsam::Point3(0, 0, cube_size));
    cube_W.push_back(gtsam::Point3(cube_size, 0, cube_size));
    cube_W.push_back(gtsam::Point3(cube_size, cube_size, cube_size));
    cube_W.push_back(gtsam::Point3(0, cube_size, cube_size));

    Eigen::Matrix3d R_new_points(
        Eigen::AngleAxis(M_PI / 4, Eigen::Vector3d(0, 0, 1)).toRotationMatrix() *
        Eigen::AngleAxis(M_PI / 4, Eigen::Vector3d(1, 0, 0)).toRotationMatrix());

    const int initial_size = cube_W.size();
    const gtsam::Point3 p_world_cube_center(cube_size / 2, cube_size / 2, cube_size / 2);
    for (int i = 0; i < initial_size; i++) {
        cube_W.emplace_back(
            R_new_points * (cube_W[i] - p_world_cube_center) + p_world_cube_center);
    }

    const size_t img_width = 640;
    const size_t img_height = 480;
    const double fx = 500.0;
    const double fy = fx;
    const double cx = img_width / 2.0;
    const double cy = img_height / 2.0;

    gtsam::Cal3_S2::shared_ptr K(new gtsam::Cal3_S2(fx, fy, 0, cx, cy));
    StructureFromMotion sfm(Frontend::ExtractorType::SIFT, *K);

    std::vector<gtsam::Pose3> poses;
    std::vector<gtsam::PinholeCamera<gtsam::Cal3_S2>> cameras;

    Eigen::Matrix3d R_world_pose0(
        Eigen::AngleAxis(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix() *
        Eigen::AngleAxis(-M_PI / 2, Eigen::Vector3d(1, 0, 0)).toRotationMatrix());
    gtsam::Pose3 T_world_pose0(gtsam::Rot3(R_world_pose0),
                               gtsam::Point3(4, cube_size / 2, cube_size / 2));
    gtsam::PinholeCamera<gtsam::Cal3_S2> camera0(T_world_pose0, *K);
    poses.push_back(T_world_pose0);
    cameras.push_back(camera0);

    Eigen::Matrix3d R_world_pose0_to_pose1(
        Eigen::AngleAxis(M_PI / 4, Eigen::Vector3d(0, 0, 1)).toRotationMatrix());
    gtsam::Pose3 T_world_pose1(
        gtsam::Rot3(R_world_pose0_to_pose1) * T_world_pose0.rotation(),
        (R_world_pose0_to_pose1 * (T_world_pose0.translation() - p_world_cube_center) +
         p_world_cube_center));
    gtsam::PinholeCamera<gtsam::Cal3_S2> camera1(T_world_pose1, *K);
    poses.push_back(T_world_pose1);
    cameras.push_back(camera1);

    std::vector<Eigen::Isometry3d> isometries;
    for (const gtsam::Pose3 &pose : poses) {
        isometries.emplace_back(pose.matrix());
    }

    std::vector<cv::KeyPoint> kpts1;
    std::vector<cv::KeyPoint> kpts2;
    std::vector<cv::DMatch> matches;

    for (size_t i = 0; i < cube_W.size(); i++) {
        gtsam::Point2 pixel_p_cube_C0 = cameras[0].project(cube_W[i]);
        gtsam::Point2 pixel_p_cube_C1 = cameras[1].project(cube_W[i]);
        if (GtsamTestHelper::pixel_in_range(pixel_p_cube_C0, img_width, img_height) &&
            GtsamTestHelper::pixel_in_range(pixel_p_cube_C1, img_width, img_height)) {
            kpts1.push_back(cv::KeyPoint(pixel_p_cube_C0[0], pixel_p_cube_C0[1], 3));
            kpts2.push_back(cv::KeyPoint(pixel_p_cube_C1[0], pixel_p_cube_C1[1], 3));
            matches.emplace_back(cv::DMatch(kpts1.size() - 1, kpts2.size() - 1, 0));
        }
    }

    std::cout << "kpts1.size(): " << kpts1.size() << std::endl;
    std::cout << "kpts2.size(): " << kpts2.size() << std::endl;

    gtsam::Pose3 pose_diff_estimate = sfm.get_backend().estimate_pose(kpts1, kpts2, matches, *K);
    gtsam::Pose3 pose_diff_groundtruth = T_world_pose0.between(T_world_pose1);
    std::vector<gtsam::Pose3> result_poses = {T_world_pose0, T_world_pose0 * pose_diff_estimate};
    std::cout << "pose_diff_estimate: " << pose_diff_estimate << std::endl;
    std::cout << "pose_diff_ground_truth: " << pose_diff_groundtruth << std::endl;

    isometries.emplace_back(result_poses.back().matrix());
    robot::geometry::opencv_viz::viz_scene(isometries, cube_W);

    // std::cout << result_poses[1] << std::endl;
    // std::cout << pose1 << std::endl;
    // std::cout << pose0 << std::endl;

    constexpr double TOL = 1e-3;
    for (size_t i = 0; i < poses.size(); i++) {
        gtsam::Pose3 result_pose = result_poses[i];
        EXPECT_NEAR(poses[i].translation()[0], result_pose.translation()[0], TOL);
        EXPECT_NEAR(poses[i].translation()[1], result_pose.translation()[1], TOL);
        EXPECT_NEAR(poses[i].translation()[2], result_pose.translation()[2], TOL);
        EXPECT_TRUE(poses[i].rotation().matrix().isApprox(result_pose.rotation().matrix(), TOL));
    }
}

TEST(SFM_TEST, sfm_snippet_two) {
    DataParser data_parser = SymphonyLakeDatasetTestHelper::get_test_parser();
    const symphony_lake_dataset::SurveyVector &survey_vector = data_parser.get_surveys();
    const symphony_lake_dataset::Survey &survey = survey_vector.get(0);
    const symphony_lake_dataset::ImagePoint image_point = survey.getImagePoint(180);
    const std::vector<cv::Mat> images = {survey.loadImageByImageIndex(180),
                                         survey.loadImageByImageIndex(185)};

    // const size_t img_width = image_point.width, img_height = image_point.height;
    const double fx = image_point.fx, fy = image_point.fy;
    const double cx = image_point.cx, cy = image_point.cy;
    gtsam::Cal3_S2 K(fx, fy, 0, cx, cy);

    StructureFromMotion sfm(Frontend::ExtractorType::SIFT, K);

    std::cout << "sfm.landmark_count_: " << sfm.get_landmark_count() << std::endl;

    for (const cv::Mat &image : images) {
        sfm.add_image(
            image);  //, gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0.05, 0, 0)));
    }

    gtsam::Values initial_values = sfm.get_backend().get_current_initial_values();
    std::vector<Eigen::MatrixXd> poses;
    std::vector<robot::math::proto::Matrix> protos;
    for (size_t i = 0; i < images.size(); i++) {
        gtsam::Pose3 pose =
            initial_values.at<gtsam::Pose3>(gtsam::Symbol(sfm.get_backend().pose_symbol_char, i));
        Eigen::MatrixXd eigen_affine(4, 4);
        gtsam::Matrix3 rot_mat = pose.rotation().matrix();
        gtsam::Point3 translation = pose.translation();
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                eigen_affine(j, k) = rot_mat(j, k);
            }
            eigen_affine(3, j) = translation(j);
        }
        eigen_affine(3, 3) = 1;
        poses.push_back(eigen_affine);
        protos.emplace_back();
        pack_into(eigen_affine, &protos.back());
    }
    std::cout << poses.front() << std::endl;

    // const Eigen::Vector3d in(1.0, 2.0, 3.0);
    // // Action

    // robot::math::proto::Matrix proto;
    // pack_into(in, &proto);
    // const Eigen::Vector3d out = unpack_from<Eigen::Vector3d>(proto);
    // std::cout << "vector out " << out << std::endl;

    // const Eigen::VectorXd in{{1.0, 2.0, 3.0}};

    // // Action
    // robot::math::proto::Matrix proto;
    // pack_into(in, &proto);
    // const Eigen::VectorXd out = unpack_from<Eigen::VectorXd>(proto);

    // sfm.solve_structure();

    // cv::hconcat(img_keypoints_out_1, img_keypoints_out_2, img_display_test);
    // cv::vconcat(img_display_test, img_matches_out, img_display_test);
    // cv::Mat og_images;
    // cv::hconcat(image_1, image_2, og_images);
    // cv::vconcat(og_images, img_display_test, img_display_test);
    // std::cout << "og_images.cols: " << og_images.cols << ". img_display_test.cols" <<
    // img_display_test.cols << std::endl; cv::imshow("Keypoints and Matches Output.",
    // img_display_test); std::cout << "Press spacebar to pause." << std::endl; while
    // (cv::waitKey(1000) != 32) {}
}

// TEST(SFM_TEST, structure_from_motion) {
//     const size_t img_width = 640;
//     const size_t img_height = 480;
//     const double fx = 500.0;
//     const double fy = fx;
//     const double cx = img_width / 2.0;
//     const double cy = img_height / 2.0;

//     gtsam::Cal3_S2 K(fx, fy, 0, cx, cy);

//     StructureFromMotion sfm(Frontend::ExtractorType::SIFT, K);
//     DataParser data_parser = SymphonyLakeDatasetTestHelper::get_test_parser();
//     // DataParser::Generator<cv::Mat> generator = data_parser.create_img_generator();
//     const symphony_lake_dataset::SurveyVector &survey_vector = data_parser.get_surveys();

//     cv::Mat image;
//     for (int i = 0; i < static_cast<int>(survey_vector.getNumSurveys()); i++) {
//         const symphony_lake_dataset::Survey &survey = survey_vector.get(i);
//         // for (int j = 0; j < static_cast<int>(survey.getNumImages()); j++) {
//         //     image = survey.loadImageByImageIndex(j);
//             // sfm.add_image(image);
//         // }
//         for (int j = 0; j < 2; j++) {
//             image = survey.loadImageByImageIndex(j);
//             sfm.add_image(image);
//         }
//     }
//     // DataParser::Generator<cv::Mat>::iterator img_itr = generator.begin();
//     // while (img_itr != generator.end()) {
//     //     if ((*img_itr).empty()) {
//     //         continue;
//     //     }
//     //     std::cout << "ooga booga" << std::endl;
//     //     std::cout << "image is empty: " << (*img_itr).empty() << std::endl;
//     //     cv::imshow("ooga", *img_itr);
//     //     cv::waitKey(1000);
//     //     sfm.add_image(*img_itr);
//     //     ++img_itr;
//     // }
//     sfm.solve_structure();
//     // gtsam::Values output = sfm.get_structure_result();
//     // output.print("result values: ");
//     // for (size_t i = 0; i < sfm.get_num_images_added(); i++) {

//     // }
// }
}  // namespace robot::experimental::learn_descriptors
