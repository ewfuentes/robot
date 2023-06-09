
#include "experimental/beacon_dist/render_ycb_scene.hh"

#include <numbers>
#include <thread>

#include "drake/geometry/scene_graph.h"
#include "drake/systems/sensors/image_writer.h"
#include "gtest/gtest.h"

namespace robot::experimental::beacon_dist {
namespace {
SceneData get_default_scene_data() {
    const std::unordered_map<std::string, std::filesystem::path> objects_and_paths = {
        {"sugar_box", "external/drake_models/ycb/meshes/004_sugar_box_textured.obj"},
        {"mustard_bottle", "external/drake_models/ycb/meshes/006_mustard_bottle_textured.obj"},
    };

    constexpr int num_renderers = 3;
    return load_ycb_objects(objects_and_paths, num_renderers);
}
}  // namespace

TEST(RenderYcbSceneTest, load_ycb_dataset_and_render) {
    // Setup
    constexpr int RENDERER_ID = 1;
    const CameraParams CAMERA_PARAMS = {
        .width_px = 1280,
        .height_px = 1024,
        .fov_y_rad = std::numbers::pi / 2.0,
        .num_views = 4,
        .camera_strategy =
            MovingCamera{.start_in_world = {0.0, 0.0, -1.0}, .end_in_world = {0.0, 0.0, -1.0}}};

    const std::unordered_map<std::string, drake::math::RigidTransformd> world_from_objects = {
        {"sugar_box", drake::math::RigidTransformd()},
        {"mustard_bottle", drake::math::RigidTransformd(Eigen::Vector3d{0.25, 0.0, 0.0})}};

    const drake::math::RigidTransformd world_from_camera{Eigen::Vector3d{0.0, 0.0, -1.0}};
    const auto &scene_data = get_default_scene_data();

    // Action
    auto root_context = scene_data.diagram->CreateDefaultContext();

    const auto image = render_scene(scene_data, CAMERA_PARAMS, world_from_objects,
                                    world_from_camera, RENDERER_ID, make_in_out(*root_context));

    // Verification
    EXPECT_EQ(image.width(), CAMERA_PARAMS.width_px);
    EXPECT_EQ(image.height(), CAMERA_PARAMS.height_px);
}

TEST(RenderYcbSceneTest, compute_scene_result) {
    // Setup
    constexpr int NUM_VIEWS = 4;
    const auto &scene_data = get_default_scene_data();
    const CameraParams CAMERA_PARAMS = {
        .width_px = 1280,
        .height_px = 1024,
        .fov_y_rad = std::numbers::pi / 2.0,
        .num_views = NUM_VIEWS,
        .camera_strategy =
            MovingCamera{.start_in_world = {0.0, 0.0, -1.0}, .end_in_world = {0.0, 0.0, -1.0}},
    };
    constexpr int RENDERER_ID = 1;
    constexpr int SCENE_ID = 1024;
    auto root_context = scene_data.diagram->CreateDefaultContext();

    // Action
    const auto &scene_result = compute_scene_result(scene_data, CAMERA_PARAMS, RENDERER_ID,
                                                    SCENE_ID, make_in_out(*root_context));
    // Verification
    EXPECT_EQ(scene_result.view_results.size(), NUM_VIEWS);
    EXPECT_EQ(scene_result.view_results[0].keypoints.size(), 300);
}
}  // namespace robot::experimental::beacon_dist
