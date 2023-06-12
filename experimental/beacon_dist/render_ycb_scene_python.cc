
#include <algorithm>

#include "experimental/beacon_dist/render_ycb_scene.hh"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl/filesystem.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace robot::experimental::beacon_dist {
PYBIND11_MODULE(render_ycb_scene_python, m) {
    m.doc() = "Render YCB Scene";

    py::class_<SceneResult>(m, "SceneResult")
        .def_property_readonly("world_from_objects",
                               [](const SceneResult &self) {
                                   std::unordered_map<std::string, Eigen::Matrix<double, 3, 4>> out;
                                   for (const auto &[name, transform] : self.world_from_objects) {
                                       out[name] = transform.GetAsMatrix34();
                                   }
                                   return out;
                               })
        .def_readwrite("view_results", &SceneResult::view_results);
    py::class_<ViewResult>(m, "ViewResult")
        .def_property_readonly(
            "world_from_camera",
            [](const ViewResult &self) { return self.world_from_camera.GetAsMatrix34(); })
        .def_readwrite("keypoints", &ViewResult::keypoints)
        .def_readwrite("descriptors", &ViewResult::descriptors)
        .def_readwrite("labels", &ViewResult::labels);
    py::class_<KeyPoint>(m, "KeyPoint")
        .def_readwrite("angle", &KeyPoint::angle)
        .def_readwrite("class_id", &KeyPoint::class_id)
        .def_readwrite("octave", &KeyPoint::octave)
        .def_readwrite("x", &KeyPoint::x)
        .def_readwrite("y", &KeyPoint::y)
        .def_readwrite("response", &KeyPoint::response)
        .def_readwrite("size", &KeyPoint::size);
    py::class_<CameraParams>(m, "CameraParams")
        .def(py::init<int, int, double, int, CameraStrategy>(), "width_px"_a, "height_px"_a,
             "fov_y_rad"_a, "num_views"_a, "camera_strategy"_a)
        .def_readwrite("width_px", &CameraParams::width_px)
        .def_readwrite("height_px", &CameraParams::height_px)
        .def_readwrite("fov_y_rad", &CameraParams::fov_y_rad)
        .def_readwrite("num_views", &CameraParams::num_views)
        .def_readwrite("camera_strategy", &CameraParams::camera_strategy);
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
}
}  // namespace robot::experimental::beacon_dist
