import unittest
import tempfile
from pathlib import Path
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import geopandas as gpd

from common.openstreetmap import extract_landmarks_python as elm


class ExtractLandmarksPythonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_pbf = Path("external/openstreetmap_snippet/us-virgin-islands-latest.osm.pbf")
        cls.full_bbox = elm.BoundingBox(-65.0, 17.5, -64.5, 18.5)

    # === Geometry Type Tests ===

    def test_extracts_point_geometry(self):
        """Python wrapper correctly exposes PointGeometry"""
        features = elm.extract_landmarks(str(self.test_pbf), self.full_bbox, {"amenity": True})

        point_features = [f for f in features if isinstance(f.geometry, elm.PointGeometry)]

        self.assertGreater(len(point_features), 0, "Should have point features")

        point = point_features[0]
        self.assertEqual(point.osm_type, elm.OsmType.NODE)
        self.assertIsInstance(point.geometry.coord, elm.Coordinate)
        self.assertIsInstance(point.geometry.coord.lat, float)
        self.assertIsInstance(point.geometry.coord.lon, float)

    def test_extracts_linestring_geometry(self):
        """Python wrapper correctly exposes LineStringGeometry"""
        features = elm.extract_landmarks(str(self.test_pbf), self.full_bbox, {"highway": True})

        line_features = [f for f in features if isinstance(f.geometry, elm.LineStringGeometry)]

        self.assertGreater(len(line_features), 0, "Should have linestring features")

        line = line_features[0]
        self.assertEqual(line.osm_type, elm.OsmType.WAY)
        self.assertGreaterEqual(len(line.geometry.coords), 2)

    def test_extracts_polygon_geometry(self):
        """Python wrapper correctly exposes PolygonGeometry"""
        features = elm.extract_landmarks(str(self.test_pbf), self.full_bbox, {"building": True})

        poly_features = [f for f in features if isinstance(f.geometry, elm.PolygonGeometry)]

        self.assertGreater(len(poly_features), 0, "Should have polygon features")

        poly = poly_features[0]
        self.assertGreaterEqual(len(poly.geometry.exterior), 4)
        self.assertIsInstance(poly.geometry.holes, list)

    def test_extracts_multipolygon_geometry(self):
        """Python wrapper correctly exposes MultiPolygonGeometry"""
        features = elm.extract_landmarks(str(self.test_pbf), self.full_bbox, {"landuse": True})

        mp_features = [f for f in features if isinstance(f.geometry, elm.MultiPolygonGeometry)]

        if len(mp_features) > 0:
            mp = mp_features[0]
            self.assertEqual(mp.osm_type, elm.OsmType.RELATION)
            self.assertGreater(len(mp.geometry.polygons), 0)

    # === Conversion to Shapely ===

    def test_converts_to_shapely_point(self):
        """Can convert PointGeometry to Shapely Point"""
        features = elm.extract_landmarks(str(self.test_pbf), self.full_bbox, {"amenity": True})

        point_features = [f for f in features if isinstance(f.geometry, elm.PointGeometry)]

        self.assertGreater(len(point_features), 0)

        geom = point_features[0].geometry
        shapely_point = Point(geom.coord.lon, geom.coord.lat)

        self.assertIsInstance(shapely_point, Point)
        self.assertTrue(shapely_point.is_valid)

    def test_converts_to_shapely_linestring(self):
        """Can convert LineStringGeometry to Shapely LineString"""
        features = elm.extract_landmarks(str(self.test_pbf), self.full_bbox, {"highway": True})

        line_features = [f for f in features if isinstance(f.geometry, elm.LineStringGeometry)]

        self.assertGreater(len(line_features), 0)

        geom = line_features[0].geometry
        shapely_line = LineString([(c.lon, c.lat) for c in geom.coords])

        self.assertIsInstance(shapely_line, LineString)
        self.assertTrue(shapely_line.is_valid)

    def test_converts_to_shapely_polygon(self):
        """Can convert PolygonGeometry to Shapely Polygon"""
        features = elm.extract_landmarks(str(self.test_pbf), self.full_bbox, {"building": True})

        poly_features = [f for f in features if isinstance(f.geometry, elm.PolygonGeometry)]

        self.assertGreater(len(poly_features), 0)

        geom = poly_features[0].geometry
        exterior = [(c.lon, c.lat) for c in geom.exterior]
        holes = [[(c.lon, c.lat) for c in hole] for hole in geom.holes]

        shapely_poly = Polygon(exterior, holes if holes else None)

        self.assertIsInstance(shapely_poly, Polygon)
        self.assertTrue(shapely_poly.is_valid)

    # === GeoDataFrame Integration ===

    def test_creates_geopandas_dataframe(self):
        """Can create GeoDataFrame from extracted features"""
        features = elm.extract_landmarks(str(self.test_pbf), self.full_bbox, {"amenity": True})

        self.assertGreater(len(features), 0)

        # Convert to GeoDataFrame (simplified conversion)
        data = {
            "osm_id": [f.osm_id for f in features],
            "osm_type": [f.osm_type for f in features],
            "geometry": [self._to_shapely(f.geometry) for f in features],
        }

        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertEqual(len(gdf), len(features))
        self.assertEqual(gdf.crs.to_string(), "EPSG:4326")

    def test_writes_and_reads_feather(self):
        """Can write to Feather and read back with geopandas"""
        features = elm.extract_landmarks(
            str(self.test_pbf),
            elm.BoundingBox(-64.95, 18.33, -64.93, 18.35),  # Small area
            {"building": True},
        )

        self.assertGreater(len(features), 0)

        data = {
            "osm_id": [f.osm_id for f in features],
            "geometry": [self._to_shapely(f.geometry) for f in features],
        }
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

        with tempfile.NamedTemporaryFile(suffix=".feather", delete=False) as f:
            temp_path = Path(f.name)

        try:
            gdf.to_feather(temp_path)
            gdf_read = gpd.read_feather(temp_path)

            self.assertEqual(len(gdf), len(gdf_read))
            self.assertEqual(gdf_read.crs.to_string(), "EPSG:4326")
        finally:
            temp_path.unlink()

    # === Tag Handling ===

    def test_preserves_tags_as_dict(self):
        """Tags are exposed as Python dict"""
        features = elm.extract_landmarks(str(self.test_pbf), self.full_bbox, {"amenity": True})

        self.assertGreater(len(features), 0)

        feature = features[0]
        self.assertIsInstance(feature.tags, dict)
        self.assertIn("amenity", feature.tags)

    # === BoundingBox Tests ===

    def test_bounding_box_constructor(self):
        """BoundingBox can be constructed from Python"""
        bbox = elm.BoundingBox(-65.0, 17.5, -64.5, 18.5)

        self.assertEqual(bbox.left_deg, -65.0)
        self.assertEqual(bbox.bottom_deg, 17.5)
        self.assertEqual(bbox.right_deg, -64.5)
        self.assertEqual(bbox.top_deg, 18.5)

    def test_bounding_box_contains(self):
        """BoundingBox.contains works from Python"""
        bbox = elm.BoundingBox(-65.0, 17.5, -64.5, 18.5)

        self.assertTrue(bbox.contains(-64.7, 18.0))
        self.assertFalse(bbox.contains(-66.0, 18.0))
        self.assertFalse(bbox.contains(-64.7, 19.0))

    # === Filtering Tests ===

    def test_tag_filter_works(self):
        """Tag filtering works from Python"""
        amenities = elm.extract_landmarks(str(self.test_pbf), self.full_bbox, {"amenity": True})
        buildings = elm.extract_landmarks(str(self.test_pbf), self.full_bbox, {"building": True})

        self.assertGreater(len(amenities), 0)
        self.assertGreater(len(buildings), 0)

        # Verify tags
        for f in amenities:
            self.assertIn("amenity", f.tags)

        for f in buildings:
            self.assertIn("building", f.tags)

    def test_multiple_tag_filters(self):
        """Multiple tag filters work"""
        features = elm.extract_landmarks(
            str(self.test_pbf),
            self.full_bbox,
            {"amenity": True, "building": True, "highway": True},
        )

        self.assertGreater(len(features), 0)

        # Each feature should have at least one requested tag
        for f in features:
            has_tag = "amenity" in f.tags or "building" in f.tags or "highway" in f.tags
            self.assertTrue(has_tag)

    # === Error Handling ===

    def test_handles_invalid_path(self):
        """Raises error for invalid PBF path"""
        with self.assertRaises(RuntimeError):
            elm.extract_landmarks("/nonexistent/file.osm.pbf", self.full_bbox, {"amenity": True})

    # === Repr Tests ===

    def test_coordinate_repr(self):
        """Coordinate has useful repr"""
        coord = elm.Coordinate(10.5, 20.3)
        repr_str = repr(coord)
        self.assertIn("10.5", repr_str)
        self.assertIn("20.3", repr_str)

    def test_bounding_box_repr(self):
        """BoundingBox has useful repr"""
        bbox = elm.BoundingBox(-65.0, 17.5, -64.5, 18.5)
        repr_str = repr(bbox)
        self.assertIn("-65", repr_str)
        self.assertIn("17.5", repr_str)

    def test_landmark_feature_repr(self):
        """LandmarkFeature has useful repr"""
        features = elm.extract_landmarks(str(self.test_pbf), self.full_bbox, {"amenity": True})
        self.assertGreater(len(features), 0)

        repr_str = repr(features[0])
        self.assertIn("LandmarkFeature", repr_str)
        self.assertIn("osm_type", repr_str)

    # Helper method
    def _to_shapely(self, geom):
        """Convert C++ geometry to Shapely"""
        if isinstance(geom, elm.PointGeometry):
            return Point(geom.coord.lon, geom.coord.lat)
        elif isinstance(geom, elm.LineStringGeometry):
            return LineString([(c.lon, c.lat) for c in geom.coords])
        elif isinstance(geom, elm.PolygonGeometry):
            exterior = [(c.lon, c.lat) for c in geom.exterior]
            holes = [[(c.lon, c.lat) for c in hole] for hole in geom.holes]
            return Polygon(exterior, holes if holes else None)
        elif isinstance(geom, elm.MultiPolygonGeometry):
            polygons = []
            for poly in geom.polygons:
                exterior = [(c.lon, c.lat) for c in poly.exterior]
                holes = [[(c.lon, c.lat) for c in hole] for hole in poly.holes]
                polygons.append(Polygon(exterior, holes if holes else None))
            return MultiPolygon(polygons)


if __name__ == "__main__":
    unittest.main()
