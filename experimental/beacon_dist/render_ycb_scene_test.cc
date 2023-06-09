
#include "experimental/beacon_dist/render_ycb_scene.hh"

#include <numbers>
#include <thread>

#include "drake/geometry/scene_graph.h"
#include "drake/systems/sensors/image_writer.h"
#include "gtest/gtest.h"

namespace robot::experimental::beacon_dist {

TEST(RenderYcbTest, load_ycb_dataset_and_render) {
    // Setup
    const std::unordered_map<std::string, std::filesystem::path> objects_and_paths = {
        {"sugar_box", "external/drake_models/ycb/meshes/004_sugar_box_textured.obj"},
        {"mustard_bottle", "external/drake_models/ycb/meshes/006_mustard_bottle_textured.obj"},
    };

    constexpr int num_renderers = 3;
    constexpr int RENDERER_ID = 1;
    constexpr CameraParams CAMERA_PARAMS = {
        .width_px = 1280,
        .height_px = 1024,
        .fov_y_rad = std::numbers::pi / 2.0,
    };
    const std::unordered_map<std::string, drake::math::RigidTransformd> world_from_objects = {
        {"sugar_box", drake::math::RigidTransformd()},
        {"mustard_bottle", drake::math::RigidTransformd(Eigen::Vector3d{0.25, 0.0, 0.0})}};

    const drake::math::RigidTransformd world_from_camera{Eigen::Vector3d{0.0, 0.0, -1.0}};

    // Action
    const auto &scene_data = load_ycb_objects(objects_and_paths, num_renderers);
    auto root_context = scene_data.diagram->CreateDefaultContext();

    const auto image = render_scene(scene_data, CAMERA_PARAMS, world_from_objects,
                                    world_from_camera, RENDERER_ID, make_in_out(*root_context));

    // Verification
    EXPECT_EQ(image.width(), CAMERA_PARAMS.width_px);
    EXPECT_EQ(image.height(), CAMERA_PARAMS.height_px);
}
}  // namespace robot::experimental::beacon_dist
