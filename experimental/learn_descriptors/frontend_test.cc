#include "experimental/learn_descriptors/frontend.hh"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "experimental/learn_descriptors/image_point.hh"
#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

namespace robot::experimental::learn_descriptors {
TEST(FrontendTest, pipeline_sweep) {
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

    cv::Mat img_test_disp;
    cv::hconcat(image_1, image_2, img_test_disp);
    cv::imshow("Test", img_test_disp);
    cv::waitKey(1000);

    FrontendParams::ExtractorType extractor_types[2] = {FrontendParams::ExtractorType::SIFT,
                                                        FrontendParams::ExtractorType::ORB};
    FrontendParams::MatcherType matcher_types[3] = {FrontendParams::MatcherType::BRUTE_FORCE,
                                                    FrontendParams::MatcherType::FLANN,
                                                    FrontendParams::MatcherType::KNN};

    FrontendParams params{FrontendParams::ExtractorType::SIFT, FrontendParams::MatcherType::KNN,
                          true, false};
    Frontend frontend(params);
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> keypoints_descriptors_pair_1;
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> keypoints_descriptors_pair_2;
    std::vector<cv::DMatch> matches;
    cv::Mat img_keypoints_out_1(height, width, CV_8UC3),
        img_keypoints_out_2(height, width, CV_8UC3), img_matches_out(height, 2 * width, CV_8UC3);
    cv::Mat img_display_test;
    for (FrontendParams::ExtractorType extractor_type : extractor_types) {
        for (FrontendParams::MatcherType matcher_type : matcher_types) {
            printf("started frontend combination: (%d, %d)\n", static_cast<int>(extractor_type),
                   static_cast<int>(matcher_type));
            FrontendParams params{extractor_type, matcher_type, true, false};
            try {
                frontend = Frontend(params);
            } catch (const std::invalid_argument &e) {
                ROBOT_CHECK(std::string(e.what()) ==
                            "FLANN can not be used with ORB.");  // very jank...
                continue;
            }
            keypoints_descriptors_pair_1 = frontend.extract_features(image_1);
            keypoints_descriptors_pair_2 = frontend.extract_features(image_2);
            matches = frontend.compute_matches(keypoints_descriptors_pair_1.second,
                                               keypoints_descriptors_pair_2.second);
            frontend.draw_keypoints(image_1, keypoints_descriptors_pair_1.first,
                                    img_keypoints_out_1);
            frontend.draw_keypoints(image_2, keypoints_descriptors_pair_2.first,
                                    img_keypoints_out_2);
            frontend.draw_matches(image_1, keypoints_descriptors_pair_1.first, image_2,
                                  keypoints_descriptors_pair_2.first, matches, img_matches_out);
            cv::hconcat(img_keypoints_out_1, img_keypoints_out_2, img_display_test);
            cv::vconcat(img_display_test, img_matches_out, img_display_test);
            std::stringstream text;
            text << "Extractor " << static_cast<int>(extractor_type) << ", matcher "
                 << static_cast<int>(matcher_type);
            cv::putText(img_display_test, text.str(), cv::Point(20, height - 50),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::imshow("Keypoints and Matches Output.", img_display_test);
            std::cout << "Hold spacebar to pause." << std::endl;
            while (cv::waitKey(1000) == 32) {
            }
            printf("completed frontend combination: (%d, %d)\n", static_cast<int>(extractor_type),
                   static_cast<int>(matcher_type));
            if (extractor_type != FrontendParams::ExtractorType::ORB) {  // don't check ORB for now
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

TEST(FrontendTest, interpolate_frames) {
    constexpr size_t width = 800;
    constexpr size_t height = 400;
    constexpr double fx = 500.0;
    constexpr double fy = fx;
    constexpr double cx = static_cast<double>(width) / 2.0;
    constexpr double cy = static_cast<double>(height) / 2.0;
    // gtsam::Cal3_S2::shared_ptr K(new gtsam::Cal3_S2(fx, fy, 0, cx, cy));
    std::shared_ptr<CameraCalibrationFisheye> shared_K =
        std::make_shared<CameraCalibrationFisheye>(fx, fy, cx, cy, 1, 1, 1, 1);
    cv::Mat white_image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    FrontendParams params{FrontendParams::ExtractorType::SIFT, FrontendParams::MatcherType::KNN,
                          true, false};
    Frontend frontend(params);

    struct VelocityPoint {
        double time;               // time in seconds
        Eigen::Vector3d velocity;  // m/s
    };
    std::vector<VelocityPoint> velocity_points{
        // basically a piecewise constant function
        VelocityPoint{1.0, Eigen::Vector3d(1, 0, 0)},  // go in x dir from seconds 0 - 1
        VelocityPoint{2.0, Eigen::Vector3d(0, 1, 0)},  // go in y dir from seconds 1 - 2
        VelocityPoint{
            3.0, Eigen::Vector3d(1, 1, 0)},  // go diagonal in first quadrant from seconds 2 - 3
    };
    size_t idx_vel_pt = 0;
    const Eigen::Vector3d pt0_in_world = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> pts_in_world{pt0_in_world};
    double time = 0.0;
    std::vector<std::shared_ptr<ImagePoint>> img_pts;
    ImagePoint img_pt_first;
    img_pt_first.id = 0;
    img_pt_first.seq = static_cast<size_t>(time * 1e9);
    img_pt_first.K = shared_K;
    img_pt_first.set_cam_in_world(pt0_in_world);
    img_pts.push_back(std::make_shared<ImagePoint>(img_pt_first));
    frontend.add_image(ImageAndPoint{white_image, img_pts.back()});
    constexpr double sample_hz = 6.0;
    constexpr size_t num_samples_skip = 2;
    ROBOT_CHECK(static_cast<size_t>(sample_hz) % (num_samples_skip + 1) ==
                0);  // needed so that the interpolation isn't lossy
    constexpr double dt = 1 / sample_hz;
    time += dt;
    while (time < velocity_points.back().time) {
        while (idx_vel_pt < velocity_points.size() && time > velocity_points[idx_vel_pt].time) {
            idx_vel_pt++;
        }
        pts_in_world.emplace_back(pts_in_world.back() + dt * velocity_points[idx_vel_pt].velocity);
        ImagePoint img_pt;
        img_pt.id = img_pts.size();
        img_pt.seq = static_cast<size_t>(time * 1e9);
        img_pt.K = shared_K;
        if (img_pts.size() % (num_samples_skip + 1) == 0) {
            img_pt.set_cam_in_world(pts_in_world.back());
        }
        img_pts.emplace_back(std::make_shared<ImagePoint>(img_pt));
        frontend.add_image(ImageAndPoint{white_image, img_pts.back()});
        time += dt;
    }
    frontend.populate_frames();
    std::optional<std::vector<Eigen::Vector3d>> interpolated_pts =
        frontend.interpolated_initial_translations();
    ROBOT_CHECK(interpolated_pts);
    ROBOT_CHECK(interpolated_pts->size() == img_pts.size());
    constexpr double TOL = 1e-5;
    for (size_t i = 0; i < pts_in_world.size(); i++) {
        ROBOT_CHECK(pts_in_world[i].isApprox((*interpolated_pts)[i], TOL), i, pts_in_world[i],
                    (*interpolated_pts)[i]);
    }
}
}  // namespace robot::experimental::learn_descriptors