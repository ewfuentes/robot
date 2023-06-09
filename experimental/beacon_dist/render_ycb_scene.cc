
#include "experimental/beacon_dist/render_ycb_scene.hh"

#include <filesystem>
#include <random>

#include "drake/geometry/render_vtk/factory.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/diagram_builder.h"

namespace robot::experimental::beacon_dist {
namespace {
struct SampleObjectPosesParams {
    int max_objects;
    Eigen::Vector3d min;
    Eigen::Vector3d max;
};

std::string renderer_name_from_id(const int renderer_id) {
    return "my_renderer_" + std::to_string(renderer_id);
}

drake::geometry::render::ColorRenderCamera get_color_camera(const CameraParams &camera_params,
                                                            const int renderer_id) {
    const auto camera_core = drake::geometry::render::RenderCameraCore(
        renderer_name_from_id(renderer_id),
        drake::systems::sensors::CameraInfo(camera_params.width_px, camera_params.height_px,
                                            camera_params.fov_y_rad),
        drake::geometry::render::ClippingRange(0.1, 10.0), drake::math::RigidTransformd());

    constexpr bool DONT_SHOW_WINDOW = false;
    return drake::geometry::render::ColorRenderCamera(camera_core, DONT_SHOW_WINDOW);
}

std::unordered_map<std::string, drake::math::RigidTransformd> sample_object_poses(
    const std::vector<std::string> &objects, const SampleObjectPosesParams &params,
    InOut<std::mt19937> gen) {
    // Choose how many objects to display
    std::uniform_int_distribution<> num_objects_dist(2, params.max_objects);
    const size_t num_objects = num_objects_dist(*gen);

    // Choose which objects to include
    std::uniform_int_distribution<> object_dist(0, objects.size() - 1);
    std::unordered_set<int> selected_object_idx;
    while (selected_object_idx.size() < num_objects &&
           selected_object_idx.size() != objects.size()) {
        selected_object_idx.insert(static_cast<int>(object_dist(*gen)));
    }

    // Sample a pose for each object
    std::uniform_real_distribution<> translation_dist(0.0, 1.0);
    std::normal_distribution<> rot_dist(0.0, 10.0);
    std::unordered_map<std::string, drake::math::RigidTransformd> out;
    for (const int object_idx : selected_object_idx) {
        const auto &object_name = objects[object_idx];
        const Eigen::Vector3d translation{
            (params.max.x() - params.min.x()) * translation_dist(*gen) + params.min.x(),
            (params.max.y() - params.min.y()) * translation_dist(*gen) + params.min.y(),
            (params.max.z() - params.min.z()) * translation_dist(*gen) + params.min.z(),
        };
        const Eigen::Vector3d rot{
            (params.max.x() - params.min.x()) * translation_dist(*gen) + params.min.x(),
            (params.max.y() - params.min.y()) * translation_dist(*gen) + params.min.y(),
            (params.max.z() - params.min.z()) * translation_dist(*gen) + params.min.z(),
        };
        const double angle_rad = rot.norm();
        const Eigen::Vector3d axis = rot.normalized();
        out[object_name] = drake::math::RigidTransformd({angle_rad, axis}, translation);
    }
    return out;
}

Eigen::Matrix3d compute_look_at_origin_rotation(
    [[maybe_unused]] const Eigen::Vector3d &center_in_world) {
    return Eigen::Matrix3d::Identity();
}

template <typename T>
std::vector<drake::math::RigidTransformd> sample_world_from_camera(const CameraParams camera_params,
                                                                   InOut<std::mt19937> gen);

template <>
std::vector<drake::math::RigidTransformd> sample_world_from_camera<MovingCamera>(
    const CameraParams camera_params, [[maybe_unused]] InOut<std::mt19937> gen) {
    const auto strategy = std::get<MovingCamera>(camera_params.camera_strategy);

    std::vector<drake::math::RigidTransformd> out;
    for (int i = 0; i < camera_params.num_views; i++) {
        const double frac = static_cast<double>(i) /
                            (camera_params.num_views > 1 ? camera_params.num_views - 1 : 1);
        const Eigen::Vector3d center_in_world =
            frac * strategy.end_in_world + (1 - frac) * strategy.start_in_world;
        const Eigen::Matrix3d world_from_camera_rot =
            compute_look_at_origin_rotation(center_in_world);
        out.push_back(drake::math::RigidTransformd(
            drake::math::RotationMatrix(world_from_camera_rot), center_in_world));
    }
    return out;
}

std::vector<drake::math::RigidTransformd> sample_world_from_camera(const CameraParams camera_params,
                                                                   InOut<std::mt19937> gen) {
    return std::visit(
        [&](auto &&arg) {
            using T = std::decay_t<decltype(arg)>;
            return sample_world_from_camera<T>(camera_params, gen);
        },
        camera_params.camera_strategy);
}

std::tuple<std::vector<KeyPoint>, std::vector<Descriptor>, std::vector<std::unordered_set<int>>>
compute_keypoints_descriptors_labels(
    [[maybe_unused]] const drake::systems::sensors::ImageRgba8U &all_objects,
    [[maybe_unused]] const std::unordered_map<std::string, drake::systems::sensors::ImageRgba8U>
        &image_by_object) {
    return {{}, {}, {}};
}
}  // namespace

SceneData load_ycb_objects(const std::filesystem::path &ycb_path, const int num_renderers,
                           const std::optional<std::unordered_set<std::string>> &allow_list) {
    // Load all objects from the given folder
    std::unordered_map<std::string, std::filesystem::path> names_and_paths;
    const std::filesystem::path model_path = "google_16k/textured.obj";
    for (const auto &item : std::filesystem::directory_iterator(ycb_path)) {
        const auto maybe_object_path = item.path() / model_path;
        if (item.is_directory() && std::filesystem::exists(maybe_object_path)) {
            const std::string object_name = item.path().filename();
            if (allow_list.has_value() && !allow_list->contains(object_name)) {
                continue;
            }
            names_and_paths[object_name] = maybe_object_path;
        }
    }
    return load_ycb_objects(names_and_paths, num_renderers);
}

SceneData load_ycb_objects(
    const std::unordered_map<std::string, std::filesystem::path> &names_and_paths,
    const int num_renderers) {
    drake::systems::DiagramBuilder<double> builder{};
    auto plant_and_scene_graph =
        drake::multibody::AddMultibodyPlantSceneGraph<double>(&builder, 0.0);
    auto &[plant, scene_graph] = plant_and_scene_graph;

    // Add the desired number of renderers
    for (int i = 0; i < num_renderers; i++) {
        scene_graph.AddRenderer(renderer_name_from_id(i), drake::geometry::MakeRenderEngineVtk({}));
    }

    std::vector<std::string> object_list;
    drake::multibody::Parser parser(&plant, &scene_graph);

    for (const auto &[name, path] : names_and_paths) {
        parser.AddModelFromFile(path, name);
        object_list.push_back(name);
    }

    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object");
    plant.Finalize();
    return {
        .diagram = builder.Build(),
        .object_list = std::move(object_list),
    };
}

drake::systems::sensors::ImageRgba8U render_scene(
    const SceneData &scene_data, const CameraParams &camera_params,
    const std::unordered_map<std::string, drake::math::RigidTransformd> &world_from_objects,
    const drake::math::RigidTransformd &world_from_camera, const int renderer_id,
    InOut<drake::systems::Context<double>> root_context) {
    auto &plant =
        scene_data.diagram->GetDowncastSubsystemByName<drake::multibody::MultibodyPlant>("plant");
    auto &scene_graph =
        scene_data.diagram->GetDowncastSubsystemByName<drake::geometry::SceneGraph>("scene_graph");
    auto &context = plant.GetMyMutableContextFromRoot(&*root_context);
    for (const auto &object_name : scene_data.object_list) {
        const auto iter = world_from_objects.find(object_name);
        const auto &body = plant.GetBodyByName(object_name);
        const auto &world_from_object =
            iter == world_from_objects.end()
                ? drake::math::RigidTransformd(Eigen::Vector3d{0.0, 0.0, 100.0})
                : iter->second;

        plant.SetFreeBodyPose(&context, body, world_from_object);
    }

    const auto camera = get_color_camera(camera_params, renderer_id);
    const auto &query_object = scene_data.diagram->GetOutputPort("query_object")
                                   .Eval<drake::geometry::QueryObject<double>>(*root_context);
    drake::systems::sensors::ImageRgba8U out(camera_params.width_px, camera_params.height_px);
    query_object.RenderColorImage(camera, scene_graph.world_frame_id(), world_from_camera, &out);
    return out;
}

SceneResult compute_scene_result(const SceneData &scene_data, const CameraParams &camera_params,
                                 const int renderer_id, const int64_t scene_id,
                                 InOut<drake::systems::Context<double>> root_context) {
    // Create RNG
    std::mt19937 gen(scene_id + 0x8e072e3c);

    // Sample Object Poses
    const auto &world_from_objects = sample_object_poses(scene_data.object_list,
                                                         {
                                                             .max_objects = 10,
                                                             .min = Eigen::Vector3d::Ones() * -0.5,
                                                             .max = Eigen::Vector3d::Ones() * 0.5,
                                                         },
                                                         make_in_out(gen));

    // Compute Camera Poses
    const auto &world_from_cameras = sample_world_from_camera(camera_params, make_in_out(gen));

    std::vector<ViewResult> view_results;
    for (const auto &world_from_camera : world_from_cameras) {
        std::unordered_map<std::string, drake::systems::sensors::ImageRgba8U> images_by_object;

        // Render Scenes
        const auto all_objects_scene = render_scene(scene_data, camera_params, world_from_objects,
                                                    world_from_camera, renderer_id, root_context);
        for (const auto &[object, world_from_object] : world_from_objects) {
            images_by_object[object] =
                render_scene(scene_data, camera_params, {{object, world_from_object}},
                             world_from_camera, renderer_id, root_context);
        }

        const auto [keypoints, descriptors, labels] =
            compute_keypoints_descriptors_labels(all_objects_scene, images_by_object);

        view_results.push_back(ViewResult{
            .world_from_camera = world_from_camera,
            .keypoints = std::move(keypoints),
            .descriptors = std::move(descriptors),
            .labels = std::move(labels),
        });
    }

    // Compute Keypoints, Descriptors and Labels
    return SceneResult{
        .world_from_objects = std::move(world_from_objects),
        .view_results = std::move(view_results),
    };
}
}  // namespace robot::experimental::beacon_dist
