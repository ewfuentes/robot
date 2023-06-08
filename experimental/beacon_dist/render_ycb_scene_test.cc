
#include "experimental/beacon_dist/render_ycb_scene.hh"

#include <numbers>
#include <thread>

#include "drake/geometry/scene_graph.h"
#include "gtest/gtest.h"

namespace robot::experimental::beacon_dist {

TEST(RenderYcbTest, load_ycb_dataset_and_render) {
    // Setup
    const std::filesystem::path path = "/home/erick/scratch/ycb_objects";
    constexpr int num_renderers = 3;
    constexpr int RENDERER_ID = 1;
    constexpr CameraParams camera_params = {
        .width_px = 1280,
        .height_px = 1024,
        .fov_y_rad = std::numbers::pi / 2.0,
    };
    const std::unordered_map<std::string, drake::math::RigidTransformd> world_from_objects = {
        {"071_nine_hole_peg_test", drake::math::RigidTransformd()},{"006_mustard_bottle", drake::math::RigidTransformd(Eigen::Vector3d{0.25, 0.0, 0.0})}
    };

    const drake::math::RigidTransformd world_from_camera{Eigen::Vector3d{0.0, 0.0, -1.0}};

    // Action
    const auto &scene_data =
        load_ycb_objects(path, num_renderers, {{"071_nine_hole_peg_test", "006_mustard_bottle"}});
    auto root_context = scene_data.diagram->CreateDefaultContext();

    render_scene(scene_data, camera_params, world_from_objects, world_from_camera, RENDERER_ID, make_in_out(*root_context));

    // Verification
}
}  // namespace robot::experimental::beacon_dist
