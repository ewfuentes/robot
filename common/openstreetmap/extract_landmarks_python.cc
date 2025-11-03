#include "common/openstreetmap/extract_landmarks.hh"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace robot::openstreetmap {

PYBIND11_MODULE(extract_landmarks_python, m) {
    m.doc() = "Python bindings for OSM landmark extraction from PBF files";

    // OsmType enum
    py::enum_<OsmType>(m, "OsmType")
        .value("NODE", OsmType::NODE)
        .value("WAY", OsmType::WAY)
        .value("RELATION", OsmType::RELATION)
        .export_values();

    // Coordinate struct
    py::class_<Coordinate>(m, "Coordinate")
        .def(py::init<>())
        .def(py::init<double, double>())
        .def_readwrite("lon", &Coordinate::lon)
        .def_readwrite("lat", &Coordinate::lat)
        .def("__repr__", [](const Coordinate& c) {
            return "Coordinate(lon=" + std::to_string(c.lon) + ", lat=" + std::to_string(c.lat) +
                   ")";
        });

    // PointGeometry
    py::class_<PointGeometry>(m, "PointGeometry")
        .def(py::init<>())
        .def_readwrite("coord", &PointGeometry::coord)
        .def("__repr__", [](const PointGeometry& g) {
            return "PointGeometry(coord=Coordinate(lon=" + std::to_string(g.coord.lon) +
                   ", lat=" + std::to_string(g.coord.lat) + "))";
        });

    // LineStringGeometry
    py::class_<LineStringGeometry>(m, "LineStringGeometry")
        .def(py::init<>())
        .def_readwrite("coords", &LineStringGeometry::coords)
        .def("__repr__", [](const LineStringGeometry& g) {
            return "LineStringGeometry(coords=" + std::to_string(g.coords.size()) + " points)";
        });

    // PolygonGeometry
    py::class_<PolygonGeometry>(m, "PolygonGeometry")
        .def(py::init<>())
        .def_readwrite("exterior", &PolygonGeometry::exterior)
        .def_readwrite("holes", &PolygonGeometry::holes)
        .def("__repr__", [](const PolygonGeometry& g) {
            return "PolygonGeometry(exterior=" + std::to_string(g.exterior.size()) +
                   " points, holes=" + std::to_string(g.holes.size()) + ")";
        });

    // MultiPolygonGeometry
    py::class_<MultiPolygonGeometry>(m, "MultiPolygonGeometry")
        .def(py::init<>())
        .def_readwrite("polygons", &MultiPolygonGeometry::polygons)
        .def("__repr__", [](const MultiPolygonGeometry& g) {
            return "MultiPolygonGeometry(polygons=" + std::to_string(g.polygons.size()) + ")";
        });

    // LandmarkFeature
    py::class_<LandmarkFeature>(m, "LandmarkFeature")
        .def(py::init<>())
        .def_readwrite("osm_type", &LandmarkFeature::osm_type)
        .def_readwrite("osm_id", &LandmarkFeature::osm_id)
        .def_readwrite("geometry", &LandmarkFeature::geometry)
        .def_readwrite("tags", &LandmarkFeature::tags)
        .def_readwrite("landmark_type", &LandmarkFeature::landmark_type)
        .def("__repr__", [](const LandmarkFeature& f) {
            std::string type_str;
            switch (f.osm_type) {
                case OsmType::NODE:
                    type_str = "NODE";
                    break;
                case OsmType::WAY:
                    type_str = "WAY";
                    break;
                case OsmType::RELATION:
                    type_str = "RELATION";
                    break;
            }
            return "LandmarkFeature(osm_type=" + type_str + ", osm_id=" + std::to_string(f.osm_id) +
                   ", landmark_type='" + f.landmark_type + "')";
        });

    // BoundingBox
    py::class_<BoundingBox>(m, "BoundingBox")
        .def(py::init<>())
        .def(py::init<double, double, double, double>(), py::arg("left_deg"), py::arg("bottom_deg"),
             py::arg("right_deg"), py::arg("top_deg"))
        .def_readwrite("left_deg", &BoundingBox::left_deg)
        .def_readwrite("bottom_deg", &BoundingBox::bottom_deg)
        .def_readwrite("right_deg", &BoundingBox::right_deg)
        .def_readwrite("top_deg", &BoundingBox::top_deg)
        .def("contains", &BoundingBox::contains, py::arg("lon"), py::arg("lat"))
        .def("__repr__", [](const BoundingBox& b) {
            return "BoundingBox(left_deg=" + std::to_string(b.left_deg) +
                   ", bottom_deg=" + std::to_string(b.bottom_deg) +
                   ", right_deg=" + std::to_string(b.right_deg) +
                   ", top_deg=" + std::to_string(b.top_deg) + ")";
        });

    // Main extraction function
    m.def("extract_landmarks", &extract_landmarks, py::arg("pbf_path"), py::arg("bbox"),
          py::arg("tag_filters"),
          R"pbdoc(
        Extract landmarks from an OSM PBF file.

        Parameters
        ----------
        pbf_path : str
            Path to the OSM PBF file
        bbox : BoundingBox
            Bounding box to filter features
        tag_filters : dict
            Dictionary of OSM tags to filter by (e.g., {"amenity": True, "building": True})

        Returns
        -------
        list of LandmarkFeature
            List of extracted landmark features with geometry and tags
    )pbdoc");
}

}  // namespace robot::openstreetmap
