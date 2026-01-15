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
        })
        .def(py::pickle(
            [](const Coordinate& c) { return py::make_tuple(c.lon, c.lat); },
            [](py::tuple t) {
                return Coordinate{t[0].cast<double>(), t[1].cast<double>()};
            }));

    // PointGeometry
    py::class_<PointGeometry>(m, "PointGeometry")
        .def(py::init<>())
        .def_readwrite("coord", &PointGeometry::coord)
        .def("__repr__", [](const PointGeometry& g) {
            return "PointGeometry(coord=Coordinate(lon=" + std::to_string(g.coord.lon) +
                   ", lat=" + std::to_string(g.coord.lat) + "))";
        })
        .def(py::pickle(
            [](const PointGeometry& g) { return py::make_tuple(g.coord); },
            [](py::tuple t) { return PointGeometry{t[0].cast<Coordinate>()}; }));

    // LineStringGeometry
    py::class_<LineStringGeometry>(m, "LineStringGeometry")
        .def(py::init<>())
        .def_readwrite("coords", &LineStringGeometry::coords)
        .def("__repr__", [](const LineStringGeometry& g) {
            return "LineStringGeometry(coords=" + std::to_string(g.coords.size()) + " points)";
        })
        .def(py::pickle(
            [](const LineStringGeometry& g) { return py::make_tuple(g.coords); },
            [](py::tuple t) {
                return LineStringGeometry{t[0].cast<std::vector<Coordinate>>()};
            }));

    // PolygonGeometry
    py::class_<PolygonGeometry>(m, "PolygonGeometry")
        .def(py::init<>())
        .def_readwrite("exterior", &PolygonGeometry::exterior)
        .def_readwrite("holes", &PolygonGeometry::holes)
        .def("__repr__", [](const PolygonGeometry& g) {
            return "PolygonGeometry(exterior=" + std::to_string(g.exterior.size()) +
                   " points, holes=" + std::to_string(g.holes.size()) + ")";
        })
        .def(py::pickle(
            [](const PolygonGeometry& g) { return py::make_tuple(g.exterior, g.holes); },
            [](py::tuple t) {
                return PolygonGeometry{t[0].cast<std::vector<Coordinate>>(),
                                       t[1].cast<std::vector<std::vector<Coordinate>>>()};
            }));

    // MultiPolygonGeometry
    py::class_<MultiPolygonGeometry>(m, "MultiPolygonGeometry")
        .def(py::init<>())
        .def_readwrite("polygons", &MultiPolygonGeometry::polygons)
        .def("__repr__", [](const MultiPolygonGeometry& g) {
            return "MultiPolygonGeometry(polygons=" + std::to_string(g.polygons.size()) + ")";
        })
        .def(py::pickle(
            [](const MultiPolygonGeometry& g) { return py::make_tuple(g.polygons); },
            [](py::tuple t) {
                return MultiPolygonGeometry{t[0].cast<std::vector<PolygonGeometry>>()};
            }));

    // LandmarkFeature
    py::class_<LandmarkFeature>(m, "LandmarkFeature")
        .def(py::init<>())
        .def_readwrite("osm_type", &LandmarkFeature::osm_type)
        .def_readwrite("osm_id", &LandmarkFeature::osm_id)
        .def_readwrite("geometry", &LandmarkFeature::geometry)
        .def_readwrite("tags", &LandmarkFeature::tags)
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
                   ", tags=" + std::to_string(f.tags.size()) + ")";
        })
        .def(py::pickle(
            [](const LandmarkFeature& f) {
                // Serialize geometry variant with type index
                py::object geom_obj;
                int geom_type = static_cast<int>(f.geometry.index());
                std::visit([&geom_obj](auto&& g) { geom_obj = py::cast(g); }, f.geometry);
                return py::make_tuple(f.osm_type, f.osm_id, geom_type, geom_obj, f.tags);
            },
            [](py::tuple t) {
                LandmarkFeature f;
                f.osm_type = t[0].cast<OsmType>();
                f.osm_id = t[1].cast<int64_t>();
                int geom_type = t[2].cast<int>();
                switch (geom_type) {
                    case 0:
                        f.geometry = t[3].cast<PointGeometry>();
                        break;
                    case 1:
                        f.geometry = t[3].cast<LineStringGeometry>();
                        break;
                    case 2:
                        f.geometry = t[3].cast<PolygonGeometry>();
                        break;
                    case 3:
                        f.geometry = t[3].cast<MultiPolygonGeometry>();
                        break;
                }
                f.tags = t[4].cast<std::map<std::string, std::string>>();
                return f;
            }));

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
    m.def("extract_landmarks", &extract_landmarks, py::arg("pbf_path"), py::arg("bboxes"),
          py::arg("tag_filters"),
          R"pbdoc(
        Extract landmarks from an OSM PBF file for multiple bounding boxes in one pass.

        Parameters
        ----------
        pbf_path : str
            Path to the OSM PBF file
        bboxes : dict[str, BoundingBox]
            Dictionary mapping region_id to BoundingBox to extract landmarks for
        tag_filters : dict
            Dictionary of OSM tags to filter by (e.g., {"amenity": True, "building": True})

        Returns
        -------
        list of tuple[str, LandmarkFeature]
            List of (region_id, LandmarkFeature) pairs with geometry and tags
    )pbdoc");
}

}  // namespace robot::openstreetmap
