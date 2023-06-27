
#include "experimental/beacon_dist/render_ycb_scene.hh"

#include <numbers>
#include <thread>

#include "gtest/gtest.h"

namespace robot::experimental::beacon_dist {
namespace {
SceneData get_default_scene_data() {
    const std::unordered_map<std::string, std::filesystem::path> objects_and_paths = {
        {"sugar_box", "external/drake_models/ycb/meshes/004_sugar_box_textured.obj"},
        {"mustard_bottle", "external/drake_models/ycb/meshes/006_mustard_bottle_textured.obj"},
    };

    return load_ycb_objects(objects_and_paths);
}
}  // namespace

TEST(RenderYcbSceneTest, load_ycb_dataset_and_render) {
    // Setup
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
                                    world_from_camera, make_in_out(*root_context));

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
            MovingCamera{.start_in_world = {-1.0, 0.0, 0.0}, .end_in_world = {-1.0, 0.0, 0.0}},
    };
    constexpr int SCENE_ID = 1024;
    auto root_context = scene_data.diagram->CreateDefaultContext();

    // Action
    const auto &scene_result =
        compute_scene_result(scene_data, CAMERA_PARAMS, SCENE_ID, make_in_out(*root_context));
    // Verification
    EXPECT_EQ(scene_result.view_results.size(), NUM_VIEWS);
    EXPECT_GE(scene_result.view_results[0].keypoints.size(), 300);
}

TEST(RenderYcbSceneTest, compute_scene_result_spherical_camera) {
    // Setup
    constexpr int NUM_VIEWS = 4;
    const auto &scene_data = get_default_scene_data();
    const CameraParams CAMERA_PARAMS = {
        .width_px = 1280,
        .height_px = 1024,
        .fov_y_rad = std::numbers::pi / 2.0,
        .num_views = NUM_VIEWS,
        .camera_strategy =
            SphericalCamera{.radial_distance_m = {1.0, 2.0},
                            .azimuth_range_rad = {-1.0, 1.0},
                            .inclination_range_rad = {std::numbers::pi / 2.0, std::numbers::pi}},
    };
    constexpr int SCENE_ID = 1024;
    auto root_context = scene_data.diagram->CreateDefaultContext();

    // Action
    const auto &scene_result =
        compute_scene_result(scene_data, CAMERA_PARAMS, SCENE_ID, make_in_out(*root_context));
    // Verification
    EXPECT_EQ(scene_result.view_results.size(), NUM_VIEWS);
    EXPECT_GE(scene_result.view_results[0].keypoints.size(), 100);
}

TEST(RenderYcbSceneTest, build_dataset) {
    // Setup
    constexpr int NUM_VIEWS = 1;
    constexpr int NUM_WORKERS = 4;
    const auto &scene_data = get_default_scene_data();
    const CameraParams CAMERA_PARAMS = {
        .width_px = 1280,
        .height_px = 1024,
        .fov_y_rad = std::numbers::pi / 2.0,
        .num_views = NUM_VIEWS,
        .camera_strategy =
            MovingCamera{.start_in_world = {-1.0, 0.0, 0.0}, .end_in_world = {-1.0, 0.0, 0.0}},
    };
    constexpr int NUM_SCENES = 10;
    constexpr int START_SCENE_ID = 0;

    // Action
    const auto &dataset =
        build_dataset(scene_data, CAMERA_PARAMS, START_SCENE_ID, NUM_SCENES, NUM_WORKERS);

    // Verification
    EXPECT_EQ(dataset.size(), NUM_SCENES);
}

TEST(RenderYcbSceneTest, convert_class_labels) {
    // Setup
    constexpr int NUM_LONGS = 4;
    const std::vector<std::unordered_set<int>> labels = {
        {0, 64, 128, 192},
        {1, 65, 129, 193},
        {2, 66, 130, 194},
    };

    // Action
    const Eigen::MatrixX<uint64_t> out = convert_class_labels_to_matrix(labels, NUM_LONGS);

    // Verification
    for (int i = 0; i < out.rows(); i++) {
        for (int j = 0; j < out.cols(); j++) {
            EXPECT_EQ(out(i, j), 1 << i);
        }
    }
}
}  // namespace robot::experimental::beacon_dist
