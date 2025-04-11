#include "experimental/learn_descriptors/backend.hh"

#include "gtsam/geometry/triangulation.h"
#include "gtsam/navigation/GPSFactor.h"
#include "gtsam/nonlinear/LevenbergMarquardtOptimizer.h"
#include "gtsam/slam/BetweenFactor.h"
#include "gtsam/slam/PriorFactor.h"
#include "gtsam/slam/ProjectionFactor.h"

namespace robot::experimental::learn_descriptors {

Backend::Backend(std::shared_ptr<FeatureManager> feature_manager)
    : feature_manager_(feature_manager) {
    const size_t img_width = 640;
    const size_t img_height = 480;
    const double fx = 500.0;
    const double fy = fx;
    const double cx = img_width / 2.0;
    const double cy = img_height / 2.0;

    gtsam::Cal3_S2 K(fx, fy, 0, cx, cy);
    K_ = boost::make_shared<gtsam::Cal3_S2>(K);
    // initial_estimate_.insert(gtsam::Symbol(camera_symbol_char, 0), K);
}

Backend::Backend(std::shared_ptr<FeatureManager> feature_manager, gtsam::Cal3_S2 K)
    : feature_manager_(feature_manager) {
    K_ = boost::make_shared<gtsam::Cal3_S2>(K);
    // initial_estimate_.insert(gtsam::Symbol(camera_symbol_char, 0), K);
}

template <>
void Backend::add_prior_factor(const gtsam::Symbol &symbol, const gtsam::Pose3 &value,
                               const gtsam::SharedNoiseModel &noise) {
    graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(symbol, value, noise);
    initial_estimate_.insert_or_assign(symbol, value);
    // std::cout << "adding a prior factor! with symbol: " << symbol << std::endl;
    // initial_estimate_.print("values after adding prior: ");
}

template <>
void Backend::add_prior_factor(const gtsam::Symbol &symbol, const gtsam::Point3 &value,
                               const gtsam::SharedNoiseModel &noise) {
    graph_.emplace_shared<gtsam::PriorFactor<gtsam::Point3>>(symbol, value, noise);
    gtsam::Rot3 R;
    if (initial_estimate_.exists(symbol)) {
        R = initial_estimate_.at<gtsam::Pose3>(symbol).rotation();
    }
    initial_estimate_.insert_or_assign(symbol, gtsam::Pose3(R, value));
}

template <>
void Backend::add_between_factor<gtsam::Pose3>(const gtsam::Symbol &symbol_1,
                                               const gtsam::Symbol &symbol_2,
                                               const gtsam::Pose3 &value,
                                               const gtsam::SharedNoiseModel &model) {
    graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(symbol_1, symbol_2, value, model);
    // std::cout << "adding between factor. symbol_1: " << symbol_1 << ". symbol_2: " << symbol_2 <<
    // std::endl; initial_estimate_.print("values when adding between factor: ");
    initial_estimate_.insert_or_assign(symbol_2,
                                       initial_estimate_.at<gtsam::Pose3>(symbol_1).compose(value));
}

template <>
void Backend::add_between_factor<gtsam::Rot3>(const gtsam::Symbol &symbol_1,
                                              const gtsam::Symbol &symbol_2,
                                              const gtsam::Rot3 &value,
                                              const gtsam::SharedNoiseModel &model) {
    graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Rot3>>(symbol_1, symbol_2, value, model);
    initial_estimate_.insert_or_assign(symbol_2,
                                       initial_estimate_.at<gtsam::Pose3>(symbol_1).compose(
                                           gtsam::Pose3(value, gtsam::Point3::Zero())));
}

void Backend::add_factor_GPS(const gtsam::Symbol &symbol, const gtsam::Point3 &t_world_cam,
                             const gtsam::SharedNoiseModel &model, const gtsam::Rot3 &R_world_cam) {
    graph_.emplace_shared<gtsam::GPSFactor>(symbol, t_world_cam, model);
    initial_estimate_.insert_or_assign(symbol, gtsam::Pose3(R_world_cam, t_world_cam));
}

std::pair<std::vector<gtsam::Pose3>, std::vector<gtsam::Point2>> Backend::get_obs_for_lmk(
    const gtsam::Symbol &lmk_symbol) {
    std::vector<gtsam::Pose3> cam_poses;
    std::vector<gtsam::Point2> observations;

    // Iterate over all factors in the graph
    for (const auto &factor : graph_) {
        auto projFactor = boost::dynamic_pointer_cast<
            gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3>>(factor);

        if (projFactor && projFactor->keys().at(1) == lmk_symbol) {
            // Get the camera pose symbol
            gtsam::Symbol cameraSymbol = projFactor->keys().at(0);

            // Retrieve the camera pose from values
            if (!initial_estimate_.exists(cameraSymbol)) continue;
            gtsam::Pose3 cameraPose = initial_estimate_.at<gtsam::Pose3>(cameraSymbol);

            // Get the 2D observation (keypoint measurement)
            gtsam::Point2 observation = projFactor->measured();

            // Store the pose and corresponding observation
            observations.push_back(observation);
            cam_poses.push_back(cameraPose);
        }
    }
    return std::pair<std::vector<gtsam::Pose3>, std::vector<gtsam::Point2>>(cam_poses,
                                                                            observations);
}

void Backend::add_landmarks(const std::vector<Landmark> &landmarks) {
    for (const Landmark &landmark : landmarks) {
        add_landmark(landmark);
    }
}

void Backend::add_landmark(const Landmark &landmark) {
    graph_.emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3>>(
        landmark.projection, landmark_noise_, landmark.cam_pose_symbol, landmark.lmk_factor_symbol,
        K_);
    if (!initial_estimate_.exists(landmark.cam_pose_symbol)) {
        throw std::runtime_error(
            "landmark.cam_pose_symbol must already exist in Backend.initial_estimate_ before "
            "add_landmark is called.");
    }
    gtsam::Point3 p_world_lmk_estimate =
        initial_estimate_.at<gtsam::Pose3>(landmark.cam_pose_symbol) * landmark.p_cam_lmk_guess;
    initial_estimate_.insert_or_assign(landmark.lmk_factor_symbol, p_world_lmk_estimate);

    std::pair<std::vector<gtsam::Pose3>, std::vector<gtsam::Point2>> lmk_obs =
        get_obs_for_lmk(landmark.lmk_factor_symbol);
    if (lmk_obs.first.size() >= 2) {
        try {
            // Attempt triangulation using DLT (or the GTSAM provided method)
            p_world_lmk_estimate = gtsam::triangulatePoint3(
                lmk_obs.first, K_,
                gtsam::Point2Vector(lmk_obs.second.begin(), lmk_obs.second.end()));

            // Optional: perform an explicit cheirality check
            bool valid = true;
            for (const auto &pose : lmk_obs.first) {
                // Transform point to the camera coordinate system.
                // transformTo() converts a world point to the camera frame.
                gtsam::Point3 p_cam = pose.transformTo(p_world_lmk_estimate);
                if (p_cam.z() <= 0) {  // Check that the depth is positive
                    valid = false;
                    break;
                }
            }

            if (valid) {
                initial_estimate_.update(landmark.lmk_factor_symbol, p_world_lmk_estimate);
            } else {
                std::cerr << "Triangulated landmark failed cheirality check; keeping initial guess."
                          << std::endl;
            }
        } catch (const gtsam::TriangulationCheiralityException &e) {
            // Handle the exception gracefully by logging and retaining the previous estimate.
            std::cerr << "Triangulation Cheirality Exception: " << e.what()
                      << ". Keeping initial landmark estimate." << std::endl;
        }
    }
}

void Backend::solve_graph() {
    gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_estimate_);
    result_ = optimizer.optimize();
}

void Backend::solve_graph(const int num_steps,
                          std::optional<Backend::graph_step_debug_func> inter_debug_func) {
    gtsam::LevenbergMarquardtParams params;
    params.setVerbosityLM("SUMMARY");  // or "TERMINATION", "TRYLAMBDA", etc.
    params.maxIterations = 1;          // We'll manually step it
    gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_estimate_, params);

    double prev_error = optimizer.error();
    for (int i = 0; i < num_steps; i++) {
        optimizer.iterate();
        double curr_error = optimizer.error();

        if (inter_debug_func) {
            (*inter_debug_func)(optimizer.values(), i);
        }
        if (std::abs(prev_error - curr_error) < 1e-6) {
            std::cout << "Converged at iteration " << i << "\n";
            break;
        }
    }
    result_ = optimizer.values();
}
}  // namespace robot::experimental::learn_descriptors