#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "common/geometry/camera.hh"
#include "experimental/learn_descriptors/feature_manager.hh"
#include "gtsam/geometry/Cal3_S2.h"
#include "gtsam/geometry/Point2.h"
#include "gtsam/geometry/Point3.h"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/geometry/Rot3.h"
#include "gtsam/inference/Symbol.h"
#include "gtsam/linear/NoiseModel.h"
#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include "gtsam/nonlinear/Values.h"

namespace robot::experimental::learn_descriptors {
class Backend {
   public:
    static constexpr char pose_symbol_char = 'x';
    static constexpr char pose_rot_symbol_char = 'r';
    static constexpr char pose_translation_symbol_char = 't';
    static constexpr char pose_bearing_symbol_char = 'b';
    static constexpr char landmark_symbol_char = 'l';
    static constexpr char camera_symbol_char = 'k';

    struct Landmark {
        Landmark(const gtsam::Symbol &lmk_factor_symbol, const gtsam::Symbol &cam_pose_symbol,
                 const gtsam::Point2 &projection, const gtsam::Cal3_S2 K,
                 float initial_depth_guess = 5.0)
            : lmk_factor_symbol(lmk_factor_symbol),
              cam_pose_symbol(cam_pose_symbol),
              projection(projection),
              p_cam_lmk_guess(robot::geometry::deproject(robot::geometry::get_intrinsic_matrix(K),
                                                         projection, initial_depth_guess)){};
        const gtsam::Symbol lmk_factor_symbol;
        const gtsam::Symbol cam_pose_symbol;
        const gtsam::Point2 projection;
        const gtsam::Point3 p_cam_lmk_guess;
    };

    Backend(std::shared_ptr<FeatureManager> feature_manager = nullptr);
    Backend(std::shared_ptr<FeatureManager> feature_manager, gtsam::Cal3_S2 K);
    Backend(gtsam::Cal3_S2::shared_ptr K) : K_(K){};
    ~Backend(){};

    template <typename T>
    void add_prior_factor(const gtsam::Symbol &symbol, const T &value,
                          const gtsam::SharedNoiseModel &model);

    template <typename T>
    void add_between_factor(const gtsam::Symbol &symbol_1, const gtsam::Symbol &symbol_2,
                            const T &value, const gtsam::SharedNoiseModel &model);

    void add_factor_GPS(const gtsam::Symbol &symbol, const gtsam::Point3 &p_world_gps,
                        const gtsam::SharedNoiseModel &model,
                        const gtsam::Rot3 &R_world_cam = gtsam::Rot3::Identity());

    std::pair<std::vector<gtsam::Pose3>, std::vector<gtsam::Point2>> get_obs_for_lmk(
        const gtsam::Symbol &lmk_symbol);
    void add_landmarks(const std::vector<Landmark> &landmarks);
    void add_landmark(const Landmark &landmark);

    void solve_graph();
    typedef int epoch;
    using graph_step_debug_func = std::function<void(const gtsam::Values &, const epoch)>;
    void solve_graph(const int num_steps,
                     std::optional<graph_step_debug_func> inter_debug_func = std::nullopt);

    const gtsam::Values &get_current_initial_values() const { return initial_estimate_; };
    const gtsam::Values &get_result() const { return result_; };
    const gtsam::Cal3_S2 &get_K() const { return *K_; };

    const gtsam::SharedNoiseModel get_lmk_noise() { return landmark_noise_; };
    const gtsam::SharedNoiseModel get_pose_noise() { return pose_noise_; };
    const gtsam::SharedNoiseModel get_translation_noise() { return translation_noise_; };
    const gtsam::SharedNoiseModel get_gps_noise() { return gps_noise_; };

   private:
    std::shared_ptr<FeatureManager> feature_manager_;
    gtsam::Cal3_S2::shared_ptr K_;

    gtsam::Values initial_estimate_;
    gtsam::Values result_;
    gtsam::NonlinearFactorGraph graph_;

    gtsam::noiseModel::Isotropic::shared_ptr landmark_noise_ =
        gtsam::noiseModel::Isotropic::Sigma(2, 1.0);
    gtsam::noiseModel::Diagonal::shared_ptr pose_noise_ =
        gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6(0.1, 0.1, 0.1, 0.01, 0.01, 0.01));
    gtsam::noiseModel::Isotropic::shared_ptr translation_noise_ =
        gtsam::noiseModel::Isotropic::Sigma(2, 0.1);
    gtsam::noiseModel::Isotropic::shared_ptr gps_noise_ =
        gtsam::noiseModel::Isotropic::Sigma(3, 2.);
};

template <>
void Backend::add_prior_factor<gtsam::Pose3>(const gtsam::Symbol &, const gtsam::Pose3 &,
                                             const gtsam::SharedNoiseModel &);

template <>
void Backend::add_prior_factor<gtsam::Point3>(const gtsam::Symbol &, const gtsam::Point3 &,
                                              const gtsam::SharedNoiseModel &);

template <>
void Backend::add_between_factor<gtsam::Pose3>(const gtsam::Symbol &, const gtsam::Symbol &,
                                               const gtsam::Pose3 &,
                                               const gtsam::SharedNoiseModel &);
template <>
void Backend::add_between_factor<gtsam::Rot3>(const gtsam::Symbol &, const gtsam::Symbol &,
                                              const gtsam::Rot3 &, const gtsam::SharedNoiseModel &);
}  // namespace robot::experimental::learn_descriptors