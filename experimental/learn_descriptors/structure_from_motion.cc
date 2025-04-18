#include "experimental/learn_descriptors/structure_from_motion.hh"

#include <filesystem>
#include <sstream>
#include <stdexcept>

#include "common/geometry/camera.hh"
#include "common/geometry/opencv_viz.hh"
#include "common/geometry/translate_types.hh"
#include "gtsam/geometry/Point2.h"

namespace fs = std::filesystem;

namespace geom = robot::geometry;

namespace robot::experimental::learn_descriptors {

const Eigen::Isometry3d StructureFromMotion::T_symlake_boat_cam = []() {
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.linear() = (Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d(0, 1, 0)) *
                          Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d(0, 0, 1)))
                             .toRotationMatrix();
    return transform;
}();

std::string pose_to_string(gtsam::Pose3 pose) {
    std::stringstream ss;
    ss << pose;
    return ss.str();
};

const gtsam::Pose3 StructureFromMotion::default_initial_pose(
    StructureFromMotion::T_symlake_boat_cam.matrix());

StructureFromMotion::StructureFromMotion(Frontend::ExtractorType frontend_extractor,
                                         gtsam::Cal3_S2 K, Eigen::Matrix<double, 5, 1> D,
                                         gtsam::Pose3 initial_pose,
                                         Frontend::MatcherType frontend_matcher) {
    // : feature_manager_(std::make_shared<FeatureManager>()), initial_pose_(initial_pose) {
    frontend_ = Frontend(frontend_extractor, frontend_matcher);
    backend_ = Backend(K);
    // backend_ = Backend(feature_manager_, K);

    K_ = (cv::Mat_<double>(3, 3) << K.fx(), 0, K.px(), 0, K.fy(), K.py(), 0, 0, 1);
    D_ = (cv::Mat_<double>(5, 1) << D(0, 0), D(1, 0), D(2, 0), D(3, 0), D(4, 0));

    set_initial_pose(initial_pose);
}

void StructureFromMotion::set_initial_pose(gtsam::Pose3 initial_pose) {
    backend_.add_prior_factor(gtsam::Symbol(Backend::pose_symbol_char, 0), initial_pose,
                              gtsam::noiseModel::Isotropic::Sigma(6, 0));
}

void StructureFromMotion::add_image(const cv::Mat &img, const gtsam::Pose3 &T_world_cam) {
    cv::Mat img_undistorted;
    cv::undistort(img, img_undistorted, K_, D_);
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> keypoints_and_descriptors =
        frontend_.get_keypoints_and_descriptors(img);
    // feature_manager_->append_img_data(keypoints_and_descriptors.first,
    //                                   keypoints_and_descriptors.second);
    // keypoint_to_landmarks_.push_back(std::unordered_map<cv::KeyPoint, Backend::Landmark>());
    // const size_t idx_img_current = get_num_images_added();
    // const size_t idx_img_current = feature_manager_->get_num_images_added() - 1;
    std::cout << "current index " << idx_img_current << std::endl;
    if (idx_img_current > 0) {
        std::vector<cv::DMatch> matches =
            get_matches(img_keypoints_and_descriptors_.back().second,
                        keypoints_and_descriptors.second, Frontend::enforce_bijective_matches);

        const gtsam::Symbol sym_T_w_c0(Backend::pose_symbol_char, idx_img_current - 1);
        const gtsam::Symbol sym_T_w_c1(Backend::pose_symbol_char, idx_img_current);

        backend_.add_factor_GPS(sym_T_w_c1, T_world_cam.translation(), backend_.get_gps_noise(),
                                T_world_cam.rotation());

        for (const cv::DMatch match : matches) {
            const cv::KeyPoint kpt_cam0 =
                img_keypoints_and_descriptors_.back().first[match.queryIdx];
            const cv::KeyPoint kpt_cam1 = keypoints_and_descriptors.first[match.trainIdx];

            size_t idx;
            unsigned char chr;

            // auto maybe_symbol0 = feature_manager_->get_symbol(idx_img_current - 1, kpt_cam0);
            if (maybe_symbol0) {
                idx = (*maybe_symbol0).index();
                chr = (*maybe_symbol0).chr();
            } else {
                gtsam::Symbol symbol_temp =
                    gtsam::Symbol(Backend::landmark_symbol_char, landmark_count_);
                idx = symbol_temp.index();
                chr = symbol_temp.chr();
                landmark_count_++;
                // feature_manager_->insert_symbol(idx_img_current - 1, kpt_cam0, symbol_temp);
            }
            gtsam::Symbol symbol_lmk = gtsam::Symbol(chr, idx);
            std::cout << "value: char: " << chr << " , idx: " << idx << ". " << symbol_lmk
                      << std::endl;
            // feature_manager_->insert_symbol(idx_img_current, kpt_cam1, symbol_lmk);

            const Backend::Landmark landmark_cam_0(
                symbol_lmk, sym_T_w_c0,
                gtsam::Point2(
                    static_cast<double>(
                        img_keypoints_and_descriptors_.back().first[match.queryIdx].pt.x),
                    static_cast<double>(
                        img_keypoints_and_descriptors_.back().first[match.queryIdx].pt.y)),
                backend_.get_K(), 3.0);

            const Backend::Landmark landmark_cam_1(
                symbol_lmk, sym_T_w_c1,
                gtsam::Point2(
                    static_cast<double>(keypoints_and_descriptors.first[match.trainIdx].pt.x),
                    static_cast<double>(keypoints_and_descriptors.first[match.trainIdx].pt.y)),
                backend_.get_K(), 3.0);

            backend_.add_landmark(landmark_cam_0);
            backend_.add_landmark(landmark_cam_1);
        }
    }
    img_keypoints_and_descriptors_.push_back(keypoints_and_descriptors);
}

std::vector<cv::DMatch> StructureFromMotion::get_matches(
    const cv::Mat &descriptors_1, const cv::Mat &descriptors_2,
    std::optional<StructureFromMotion::match_function> post_process_func) {
    std::vector<cv::DMatch> matches = frontend_.get_matches(descriptors_1, descriptors_2);
    if (post_process_func) {
        (*post_process_func)(matches);
    }
    matches_.push_back(matches);
    return matches;
}

void StructureFromMotion::graph_values(const gtsam::Values &values,
                                       const std::string &window_name) {
    std::vector<Eigen::Isometry3d> final_poses;
    std::vector<Eigen::Vector3d> final_lmks;
    // for (size_t i = 0; i < feature_manager_->get_num_images_added(); i++) {
    //     final_poses.emplace_back(
    //         values.at<gtsam::Pose3>(gtsam::Symbol(get_backend().pose_symbol_char, i)).matrix());
    // }
    // for (const gtsam::Symbol &lmk_symbol : feature_manager_->get_added_symbols()) {
    //     if (!values.exists(lmk_symbol)) {
    //         std::cout << "WTF " << lmk_symbol << std::endl;
    //     }
    //     final_lmks.emplace_back(values.at<gtsam::Point3>(lmk_symbol));
    // }
    geometry::viz_scene(final_poses, final_lmks, true, true, window_name);
}

void StructureFromMotion::solve_structure(
    const int num_steps, std::optional<Backend::graph_step_debug_func> inter_debug_func) {
    backend_.solve_graph(num_steps, inter_debug_func);
}
}  // namespace robot::experimental::learn_descriptors