
#include "experimental/beacon_dist/render_ycb_scene.hh"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iterator>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

#include "common/argument_wrapper.hh"
#include "drake/geometry/render_gl/factory.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/diagram_builder.h"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"

namespace robot::experimental::beacon_dist {
namespace {

struct WorkerState {
    std::jthread thread;
    std::mutex mutex;
    bool is_result_ready;
    bool is_request_ready;
    bool shutdown_requested;
    int requested_scene_id;
    std::optional<SceneResult> maybe_scene_result;
};

struct SampleObjectPosesParams {
    int max_objects;
    Eigen::Vector3d min;
    Eigen::Vector3d max;
};

const std::string RENDERER_ID = "my_renderer";

drake::geometry::render::ColorRenderCamera get_color_camera(const CameraParams &camera_params) {
    const auto camera_core = drake::geometry::render::RenderCameraCore(
        RENDERER_ID,
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

Eigen::Matrix3d compute_look_at_origin_rotation(const Eigen::Vector3d &center_in_world) {
    // The +Z axis points toward the center
    const Eigen::Vector3d camera_z_axis = -center_in_world.normalized();
    // The +X axis points right in the camera frame
    const Eigen::Vector3d camera_x_axis = camera_z_axis.cross(Eigen::Vector3d::UnitZ());
    // The +Y axis points down in the camera frame
    const Eigen::Vector3d camera_y_axis = camera_z_axis.cross(camera_x_axis);

    return (Eigen::Matrix3d() << camera_x_axis, camera_y_axis, camera_z_axis).finished();
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

cv::Mat opencv_image_from_drake_image(const drake::systems::sensors::ImageRgba8U &drake_image) {
    cv::Mat out(drake_image.height(), drake_image.width(), CV_8UC3);

    for (int row = 0; row < drake_image.height(); row++) {
        for (int col = 0; col < drake_image.width(); col++) {
            const auto src_pixel_data = drake_image.at(col, row);
            auto &dst_pixel_data = out.at<cv::Vec3b>(row, col);
            // Convert RGBA to BGR
            dst_pixel_data[0] = src_pixel_data[2];
            dst_pixel_data[1] = src_pixel_data[1];
            dst_pixel_data[2] = src_pixel_data[0];
        }
    }
    return out;
}

std::tuple<std::vector<KeyPoint>, std::vector<Descriptor>,
           std::vector<std::unordered_set<std::string>>>
compute_keypoints_descriptors_labels(
    const drake::systems::sensors::ImageRgba8U &all_objects,
    const std::unordered_map<std::string, drake::systems::sensors::ImageRgba8U> &image_by_object) {
    const auto all_objects_image = opencv_image_from_drake_image(all_objects);
    constexpr int NUM_FEATURES = 300;
    const auto orb_features = cv::ORB::create(NUM_FEATURES);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_features->detect(all_objects_image, keypoints);
    orb_features->compute(all_objects_image, keypoints, descriptors);

    std::vector<std::unordered_set<std::string>> class_labels(keypoints.size());

    for (const auto &[name, image] : image_by_object) {
        const auto object_image = opencv_image_from_drake_image(image);
        cv::Mat object_descriptors;
        orb_features->compute(object_image, keypoints, object_descriptors);

        cv::Mat descriptor_sum;
        constexpr int SUM_OVER_ROWS = 1;
        cv::reduce(object_descriptors, descriptor_sum, SUM_OVER_ROWS, cv::REDUCE_AVG);

        for (int i = 0; i < static_cast<int>(class_labels.size()); i++) {
            if (descriptor_sum.at<uchar>(i) > 0) {
                class_labels.at(i).insert(name);
            }
        }
    }

    // Transform the keypoints
    std::vector<KeyPoint> out_keypoints;
    std::transform(keypoints.begin(), keypoints.end(), std::back_inserter(out_keypoints),
                   [](const cv::KeyPoint &kp) {
                       return KeyPoint{.angle = kp.angle,
                                       .class_id = kp.class_id,
                                       .octave = kp.octave,
                                       .x = kp.pt.x,
                                       .y = kp.pt.y,
                                       .response = kp.response,
                                       .size = kp.size};
                   });

    std::vector<Descriptor> out_descriptor(keypoints.size());
    for (int i = 0; i < descriptors.rows; i++) {
        const auto row = descriptors.row(i);
        std::copy(row.begin<uchar>(), row.end<uchar>(), out_descriptor.at(i).begin());
    }

    return {out_keypoints, out_descriptor, class_labels};
}
}  // namespace

SceneData load_ycb_objects(const std::filesystem::path &ycb_path,
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
    return load_ycb_objects(names_and_paths);
}

SceneData load_ycb_objects(
    const std::unordered_map<std::string, std::filesystem::path> &names_and_paths) {
    drake::systems::DiagramBuilder<double> builder{};
    auto plant_and_scene_graph =
        drake::multibody::AddMultibodyPlantSceneGraph<double>(&builder, 0.0);
    auto &[plant, scene_graph] = plant_and_scene_graph;

    scene_graph.AddRenderer(RENDERER_ID, drake::geometry::MakeRenderEngineGl({}));

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
    const drake::math::RigidTransformd &world_from_camera,
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

    const auto camera = get_color_camera(camera_params);
    const auto &query_object = scene_data.diagram->GetOutputPort("query_object")
                                   .Eval<drake::geometry::QueryObject<double>>(*root_context);
    drake::systems::sensors::ImageRgba8U out(camera_params.width_px, camera_params.height_px);
    query_object.RenderColorImage(camera, scene_graph.world_frame_id(), world_from_camera, &out);
    return out;
}

SceneResult compute_scene_result(const SceneData &scene_data, const CameraParams &camera_params,
                                 const int64_t scene_id,
                                 InOut<drake::systems::Context<double>> root_context) {
    // Create RNG
    std::mt19937 gen(scene_id + 0x8e072e3c);

    // Sample Object Poses
    const auto &world_from_objects = sample_object_poses(scene_data.object_list,
                                                         {
                                                             .max_objects = 10,
                                                             .min = Eigen::Vector3d::Ones() * -0.25,
                                                             .max = Eigen::Vector3d::Ones() * 0.25,
                                                         },
                                                         make_in_out(gen));

    // Compute Camera Poses
    const auto &world_from_cameras = sample_world_from_camera(camera_params, make_in_out(gen));

    std::vector<ViewResult> view_results;
    for (const auto &world_from_camera : world_from_cameras) {
        std::unordered_map<std::string, drake::systems::sensors::ImageRgba8U> images_by_object;
        // Render Scenes
        const auto all_objects_scene = render_scene(scene_data, camera_params, world_from_objects,
                                                    world_from_camera, root_context);

        for (const auto &[object, world_from_object] : world_from_objects) {
            images_by_object[object] =
                render_scene(scene_data, camera_params, {{object, world_from_object}},
                             world_from_camera, root_context);
        }

        const auto [keypoints, descriptors, labels] =
            compute_keypoints_descriptors_labels(all_objects_scene, images_by_object);

        std::unordered_map<std::string, int> object_idx_from_name;
        for (int i = 0; i < static_cast<int>(scene_data.object_list.size()); i++) {
            object_idx_from_name[scene_data.object_list.at(i)] = i;
        }
        std::vector<std::unordered_set<int>> out_labels;
        std::transform(labels.begin(), labels.end(), std::back_inserter(out_labels),
                       [&object_idx_from_name](const std::unordered_set<std::string> &kp_objects) {
                           std::unordered_set<int> object_idxs;
                           std::transform(kp_objects.begin(), kp_objects.end(),
                                          std::inserter(object_idxs, object_idxs.begin()),
                                          [&object_idx_from_name](const std::string &object_name) {
                                              return object_idx_from_name[object_name];
                                          });
                           return object_idxs;
                       });

        view_results.push_back(ViewResult{
            .world_from_camera = world_from_camera,
            .keypoints = std::move(keypoints),
            .descriptors = std::move(descriptors),
            .labels = std::move(out_labels),
        });
    }

    // Compute Keypoints, Descriptors and Labels
    return SceneResult{
        .world_from_objects = std::move(world_from_objects),
        .view_results = std::move(view_results),
    };
}

std::vector<SceneResult> build_dataset(const SceneData &scene_data,
                                       const CameraParams &camera_params, const int64_t num_scenes,
                                       const int num_workers,
                                       const std::function<bool(int)> &progress_callback) {
    std::vector<WorkerState> workers(num_workers);

    for (int i = 0; i < num_workers; i++) {
        auto worker_func = [&state = workers.at(i), &scene_data, &camera_params]() {
            auto root_context = scene_data.diagram->CreateDefaultContext();
            render_scene(scene_data, camera_params, {}, {}, make_in_out(*root_context));
            bool run = true;
            bool is_request_ready = false;
            int requested_scene_id = 0;
            while (run) {
                {
                    std::lock_guard<std::mutex> guard(state.mutex);
                    run = !state.shutdown_requested;
                    is_request_ready = state.is_request_ready;
                    requested_scene_id = state.requested_scene_id;
                }

                if (is_request_ready) {
                    auto result = compute_scene_result(
                        scene_data, camera_params, requested_scene_id, make_in_out(*root_context));

                    std::lock_guard<std::mutex> guard(state.mutex);
                    state.maybe_scene_result = std::move(result);
                    state.is_request_ready = false;
                    state.is_result_ready = true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
        };
        workers.at(i).thread = std::jthread(worker_func);
    }

    std::vector<SceneResult> out(num_scenes);
    for (int64_t current_scene_id = 0; current_scene_id < num_scenes; current_scene_id++) {
        bool run = true;

        while (run) {
            for (auto &worker_state : workers) {
                std::lock_guard<std::mutex> guard(worker_state.mutex);
                if (worker_state.is_result_ready) {
                    out.at(worker_state.requested_scene_id) =
                        worker_state.maybe_scene_result.value();
                    run = progress_callback(worker_state.requested_scene_id);
                    worker_state.is_result_ready = false;
                }

                if (!worker_state.is_request_ready) {
                    worker_state.requested_scene_id = current_scene_id;
                    worker_state.is_request_ready = true;
                    run = false;
                    break;
                }
            }
        }
    }

    for (auto &worker : workers) {
        {
            std::lock_guard<std::mutex> guard(worker.mutex);
            worker.shutdown_requested = true;
        }
        worker.thread.join();
        if (worker.is_result_ready) {
            out.at(worker.requested_scene_id) = worker.maybe_scene_result.value();
            progress_callback(worker.requested_scene_id);
        }
    }

    return out;
}
}  // namespace robot::experimental::beacon_dist
