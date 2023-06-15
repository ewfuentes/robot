
#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "Eigen/Core"
#include "common/argument_wrapper.hh"
#include "drake/math/rigid_transform.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/sensors/image.h"

namespace robot::experimental::beacon_dist {

struct SceneData {
    std::unique_ptr<drake::systems::Diagram<double>> diagram;
    std::vector<std::string> object_list;
};

struct MovingCamera {
    Eigen::Vector3d start_in_world;
    Eigen::Vector3d end_in_world;
};

using CameraStrategy = std::variant<MovingCamera>;

struct CameraParams {
    int width_px;
    int height_px;
    double fov_y_rad;
    int num_views;
    CameraStrategy camera_strategy;
};

using Descriptor = std::array<std::uint8_t, 32>;

struct KeyPoint {
    double angle;
    int class_id;
    int octave;
    double x;
    double y;
    double response;
    double size;
};

struct ViewResult {
    drake::math::RigidTransformd world_from_camera;
    std::vector<KeyPoint> keypoints;
    std::vector<Descriptor> descriptors;
    std::vector<std::unordered_set<int>> labels;
};

struct SceneResult {
    std::unordered_map<std::string, drake::math::RigidTransformd> world_from_objects;
    std::vector<ViewResult> view_results;
};

SceneData load_ycb_objects(
    const std::filesystem::path &ycb_path,
    const std::optional<std::unordered_set<std::string>> &allow_list = std::nullopt);

SceneData load_ycb_objects(
    const std::unordered_map<std::string, std::filesystem::path> &names_and_paths);

drake::systems::sensors::ImageRgba8U render_scene(
    const SceneData &scene_data, const CameraParams &camera_params,
    const std::unordered_map<std::string, drake::math::RigidTransformd> &world_from_objects,
    const drake::math::RigidTransformd &world_from_camera,
    InOut<drake::systems::Context<double>> root_context);

SceneResult compute_scene_result(const SceneData &scene_data, const CameraParams &params,
                                 const int64_t scene_id,
                                 InOut<drake::systems::Context<double>> root_context);

std::vector<SceneResult> build_dataset(const SceneData &scene_data, const CameraParams &params,
                                       const int64_t num_scenes, const int num_workers,
                                       const std::function<bool(int)> &progress_callback = {});

Eigen::MatrixX<uint64_t> convert_class_labels_to_matrix(
    const std::vector<std::unordered_set<int>> &labels, const int num_longs);
}  // namespace robot::experimental::beacon_dist
