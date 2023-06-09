
#include <filesystem>
#include <memory>
#include <optional>
#include <unordered_set>

#include "common/argument_wrapper.hh"
#include "drake/math/rigid_transform.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/sensors/image.h"

namespace robot::experimental::beacon_dist {

struct SceneData {
    std::unique_ptr<drake::systems::Diagram<double>> diagram;
    std::vector<std::string> object_list;
};

struct CameraParams {
    int width_px;
    int height_px;
    double fov_y_rad;
};

SceneData load_ycb_objects(
    const std::filesystem::path &ycb_path, const int num_renderers,
    const std::optional<std::unordered_set<std::string>> &allow_list = std::nullopt);

SceneData load_ycb_objects(const std::unordered_map<std::string, std::filesystem::path> &names_and_paths,
                           const int num_renderers);

drake::systems::sensors::ImageRgba8U render_scene(
    const SceneData &scene_data, const CameraParams &camera_params,
    const std::unordered_map<std::string, drake::math::RigidTransformd> &world_from_objects,
    const drake::math::RigidTransformd &world_from_camera, const int renderer_id,
    InOut<drake::systems::Context<double>> root_context);

}  // namespace robot::experimental::beacon_dist
