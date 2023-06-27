
#include <algorithm>

#include "experimental/beacon_dist/render_ycb_scene.hh"
#include "pybind11/eigen.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl/filesystem.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace robot::experimental::beacon_dist {
namespace {
using TransformMat = Eigen::Matrix<double, 3, 4>;
std::unordered_map<std::string, TransformMat> convert_world_from_objects(
    const std::unordered_map<std::string, drake::math::RigidTransformd> &world_from_objects) {
    std::unordered_map<std::string, TransformMat> out;
    for (const auto &[name, transform] : world_from_objects) {
        out[name] = transform.GetAsMatrix34();
    }
    return out;
}
}  // namespace
PYBIND11_MODULE(render_ycb_scene_python, m) {
    m.doc() = "Render YCB Scene";

    py::class_<SceneResult>(m, "SceneResult")
        .def_property_readonly("world_from_objects",
                               [](const SceneResult &self) {
                                   return convert_world_from_objects(self.world_from_objects);
                               })
        .def_readwrite("view_results", &SceneResult::view_results)
        .def(py::pickle(
            [](const SceneResult &self) {
                return py::make_tuple(convert_world_from_objects(self.world_from_objects),
                                      self.view_results);
            },
            [](py::tuple t) {
                SceneResult out;
                const std::unordered_map<std::string, TransformMat> &transform_mat_by_name =
                    t[0].cast<std::unordered_map<std::string, TransformMat>>();
                std::transform(
                    transform_mat_by_name.begin(), transform_mat_by_name.end(),
                    std::inserter(out.world_from_objects, out.world_from_objects.begin()),
                    [](const auto &name_and_transform) {
                        return std::make_pair(
                            name_and_transform.first,
                            drake::math::RigidTransformd(name_and_transform.second));
                    });
                out.view_results = t[1].cast<std::vector<ViewResult>>();

                return out;
            }));
    py::class_<ViewResult>(m, "ViewResult")
        .def_property_readonly(
            "world_from_camera",
            [](const ViewResult &self) { return self.world_from_camera.GetAsMatrix34(); })
        .def_readwrite("keypoints", &ViewResult::keypoints)
        .def_readwrite("descriptors", &ViewResult::descriptors)
        .def_readwrite("labels", &ViewResult::labels)
        .def(py::pickle(
            [](const ViewResult &self) {
                return py::make_tuple(self.world_from_camera.GetAsMatrix34(), self.keypoints,
                                      self.descriptors, self.labels);
            },
            [](py::tuple t) {
                return ViewResult{
                    .world_from_camera = drake::math::RigidTransformd(t[0].cast<TransformMat>()),
                    .keypoints = t[1].cast<std::vector<KeyPoint>>(),
                    .descriptors = t[2].cast<std::vector<Descriptor>>(),
                    .labels = t[3].cast<std::vector<std::unordered_set<int>>>()};
            }));

    py::class_<KeyPoint>(m, "KeyPoint")
        .def_readwrite("angle", &KeyPoint::angle)
        .def_readwrite("class_id", &KeyPoint::class_id)
        .def_readwrite("octave", &KeyPoint::octave)
        .def_readwrite("x", &KeyPoint::x)
        .def_readwrite("y", &KeyPoint::y)
        .def_readwrite("response", &KeyPoint::response)
        .def_readwrite("size", &KeyPoint::size)
        .def(py::pickle(
            [](const KeyPoint &self) {
                return py::make_tuple(self.angle, self.class_id, self.octave, self.x, self.y,
                                      self.response, self.size);
            },
            [](py::tuple t) {
                return KeyPoint{.angle = t[0].cast<double>(),
                                .class_id = t[1].cast<int>(),
                                .octave = t[2].cast<int>(),
                                .x = t[3].cast<double>(),
                                .y = t[4].cast<double>(),
                                .response = t[5].cast<double>(),
                                .size = t[6].cast<double>()};
            }));
    py::class_<CameraParams>(m, "CameraParams")
        .def(py::init<int, int, double, int, CameraStrategy>(), "width_px"_a, "height_px"_a,
             "fov_y_rad"_a, "num_views"_a, "camera_strategy"_a)
        .def_readwrite("width_px", &CameraParams::width_px)
        .def_readwrite("height_px", &CameraParams::height_px)
        .def_readwrite("fov_y_rad", &CameraParams::fov_y_rad)
        .def_readwrite("num_views", &CameraParams::num_views)
        .def_readwrite("camera_strategy", &CameraParams::camera_strategy);

    py::class_<Range>(m, "Range")
        .def(py::init<double, double>(), "min"_a, "max"_a)
        .def_readwrite("min", &Range::min)
        .def_readwrite("max", &Range::max);

    py::class_<SphericalCamera>(m, "SphericalCamera")
        .def(py::init<Range, Range, Range>(), "radial_distance_m"_a, "azimuth_range_rad"_a,
             "inclination_range_rad"_a)
        .def_readwrite("radial_distance_m", &SphericalCamera::radial_distance_m)
        .def_readwrite("azimuth_range_rad", &SphericalCamera::azimuth_range_rad)
        .def_readwrite("inclination_range_rad", &SphericalCamera::inclination_range_rad);

    py::class_<MovingCamera>(m, "MovingCamera")
        .def(py::init<Eigen::Vector3d, Eigen::Vector3d>(), "start_in_world"_a, "end_in_world"_a)
        .def_readwrite("start_in_world", &MovingCamera::start_in_world)
        .def_readwrite("end_in_world", &MovingCamera::end_in_world);

    py::class_<SceneData>(m, "SceneData")
        // Don't expose the diagram until ownership is figured out
        // .def_readwrite("diagram", &SceneData::diagram)
        .def_readwrite("object_list", &SceneData::object_list);

    m.def("build_dataset", &build_dataset);
    m.def("load_ycb_objects",
          py::overload_cast<const std::filesystem::path &,
                            const std::optional<std::unordered_set<std::string>> &>(
              &load_ycb_objects));
    m.def("load_ycb_objects",
          py::overload_cast<const std::unordered_map<std::string, std::filesystem::path> &>(
              &load_ycb_objects));

    m.def("convert_class_labels_to_matrix", &convert_class_labels_to_matrix);
}
}  // namespace robot::experimental::beacon_dist
