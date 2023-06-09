
#include "experimental/beacon_dist/render_ycb_scene.hh"

#include <filesystem>

#include "drake/geometry/render_vtk/factory.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/diagram_builder.h"

namespace robot::experimental::beacon_dist {
namespace {
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
}  // namespace robot::experimental::beacon_dist
