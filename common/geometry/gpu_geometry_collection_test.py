"""Tests for GPU geometry collection."""

import time
import unittest
from pathlib import Path

import common.torch.load_torch_deps  # Must be imported before torch  # noqa: F401
import geopandas as gpd
import shapely
import torch

from common.geometry.gpu_geometry_collection import (
    GPUGeometryCollection,
    GeometryType,
)


class TestGPUGeometryCollection(unittest.TestCase):
    def test_from_shapely_points(self):
        """Test conversion of Point geometries."""
        points = [shapely.Point(0, 0), shapely.Point(1, 2), shapely.Point(3, 4)]

        collection = GPUGeometryCollection.from_shapely(points)

        self.assertEqual(collection.num_geometries, 3)
        self.assertEqual(collection.point_coords.shape, (3, 2))
        self.assertEqual(collection.segment_starts.shape[0], 0)
        torch.testing.assert_close(
            collection.point_coords,
            torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]),
        )

    def test_from_shapely_linestrings(self):
        """Test conversion of LineString geometries."""
        lines = [
            shapely.LineString([(0, 0), (1, 0), (1, 1)]),
            shapely.LineString([(2, 2), (3, 3)]),
        ]

        collection = GPUGeometryCollection.from_shapely(lines)

        self.assertEqual(collection.num_geometries, 2)
        # First line has 2 segments, second has 1
        self.assertEqual(collection.segment_starts.shape[0], 3)
        self.assertEqual(collection.point_coords.shape[0], 0)

    def test_from_shapely_polygons(self):
        """Test conversion of Polygon geometries."""
        polygons = [
            shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            shapely.Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ]

        collection = GPUGeometryCollection.from_shapely(polygons)

        self.assertEqual(collection.num_geometries, 2)
        # Each polygon has 4 segments
        self.assertEqual(collection.segment_starts.shape[0], 8)
        # Each polygon has 4 vertices
        self.assertEqual(collection.polygon_vertices.shape[0], 8)
        self.assertEqual(collection.polygon_ranges.shape, (2, 2))

    def test_from_shapely_mixed(self):
        """Test conversion of mixed geometry types."""
        geometries = [
            shapely.Point(0, 0),
            shapely.LineString([(1, 1), (2, 2)]),
            shapely.Polygon([(3, 3), (4, 3), (4, 4), (3, 4)]),
        ]

        collection = GPUGeometryCollection.from_shapely(geometries)

        self.assertEqual(collection.num_geometries, 3)
        self.assertEqual(
            collection.geometry_types.tolist(),
            [GeometryType.POINT, GeometryType.LINESTRING, GeometryType.POLYGON],
        )

    def test_distance_to_points_single_point(self):
        """Test distance computation to point geometries."""
        points = [shapely.Point(0, 0), shapely.Point(10, 0)]
        collection = GPUGeometryCollection.from_shapely(points)

        query = torch.tensor([[5.0, 0.0]])
        distances = collection.distance_to_points(query)

        self.assertEqual(distances.shape, (1, 2))
        torch.testing.assert_close(distances[0, 0], torch.tensor(5.0))
        torch.testing.assert_close(distances[0, 1], torch.tensor(5.0))

    def test_distance_to_linestring(self):
        """Test distance computation to linestring geometries."""
        lines = [shapely.LineString([(0, 0), (10, 0)])]
        collection = GPUGeometryCollection.from_shapely(lines)

        # Point directly above the middle of the line
        query = torch.tensor([[5.0, 3.0]])
        distances = collection.distance_to_points(query)

        torch.testing.assert_close(distances[0, 0], torch.tensor(3.0))

    def test_distance_to_polygon_outside(self):
        """Test distance to polygon for points outside."""
        polygons = [shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        collection = GPUGeometryCollection.from_shapely(polygons)

        # Point outside, 1 unit away from right edge
        query = torch.tensor([[2.0, 0.5]])
        distances = collection.distance_to_points(query)

        # Distance should be positive (outside)
        self.assertGreater(distances[0, 0].item(), 0)
        torch.testing.assert_close(
            distances[0, 0], torch.tensor(1.0), rtol=1e-5, atol=1e-5
        )

    def test_distance_to_polygon_inside(self):
        """Test distance to polygon for points inside."""
        polygons = [shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        collection = GPUGeometryCollection.from_shapely(polygons)

        # Point inside at center
        query = torch.tensor([[0.5, 0.5]])
        distances = collection.distance_to_points(query)

        # Distance should be negative (inside)
        self.assertLess(distances[0, 0].item(), 0)
        # Magnitude should be 0.5 (distance to nearest edge)
        torch.testing.assert_close(
            distances[0, 0].abs(), torch.tensor(0.5), rtol=1e-5, atol=1e-5
        )

    def test_distance_matches_shapely(self):
        """Compare distance computation to shapely for mixed geometries."""
        geometries = [
            shapely.Point(0, 0),
            shapely.LineString([(5, 0), (5, 10)]),
            shapely.Polygon([(10, 0), (15, 0), (15, 5), (10, 5)]),
        ]
        collection = GPUGeometryCollection.from_shapely(geometries)

        # Random query points
        torch.manual_seed(42)
        queries = torch.rand(20, 2) * 20 - 2

        gpu_distances = collection.distance_to_points(queries)

        for i in range(queries.shape[0]):
            point = shapely.Point(queries[i].tolist())

            # Point geometry
            shapely_dist_0 = point.distance(geometries[0])
            torch.testing.assert_close(
                gpu_distances[i, 0].abs(),
                torch.tensor(shapely_dist_0, dtype=torch.float32),
                rtol=1e-4,
                atol=1e-5,
            )

            # LineString geometry
            shapely_dist_1 = point.distance(geometries[1])
            torch.testing.assert_close(
                gpu_distances[i, 1].abs(),
                torch.tensor(shapely_dist_1, dtype=torch.float32),
                rtol=1e-4,
                atol=1e-5,
            )

            # Polygon geometry - check magnitude and sign
            shapely_inside = geometries[2].contains(point)
            shapely_dist_2 = point.distance(geometries[2].boundary)

            gpu_inside = gpu_distances[i, 2].item() < 0
            self.assertEqual(
                gpu_inside,
                shapely_inside,
                f"Inside/outside mismatch at point {i}",
            )
            torch.testing.assert_close(
                gpu_distances[i, 2].abs(),
                torch.tensor(shapely_dist_2, dtype=torch.float32),
                rtol=1e-4,
                atol=1e-5,
            )


class TestWithVigorSnippet(unittest.TestCase):
    """Tests using the vigor_snippet dataset."""

    @classmethod
    def setUpClass(cls):
        """Load vigor_snippet landmarks."""
        landmark_path = Path(
            "external/vigor_snippet/vigor_snippet/landmarks/v1_test.feather"
        )
        if not landmark_path.exists():
            raise unittest.SkipTest(f"Landmark file not found: {landmark_path}")

        cls.landmarks_df = gpd.read_feather(landmark_path)
        cls.geometries = cls.landmarks_df.geometry.values

    def test_load_all_geometries(self):
        """Test that all geometries can be loaded."""
        collection = GPUGeometryCollection.from_shapely(self.geometries)

        self.assertEqual(collection.num_geometries, len(self.geometries))

        # Check geometry type counts
        type_counts = collection.geometry_types.bincount()
        self.assertGreater(type_counts[GeometryType.POINT], 0)
        self.assertGreater(type_counts[GeometryType.LINESTRING], 0)
        self.assertGreater(type_counts[GeometryType.POLYGON], 0)

    def test_distance_sample_matches_shapely(self):
        """Compare GPU distances to shapely for sample of vigor_snippet."""
        # Use a smaller subset for comparison test
        sample_size = 100
        sample_geoms = self.geometries[:sample_size]
        collection = GPUGeometryCollection.from_shapely(sample_geoms)

        # Generate random query points within the bounding box
        bounds = shapely.bounds(shapely.GeometryCollection(list(sample_geoms)))
        min_x, min_y, max_x, max_y = bounds

        torch.manual_seed(42)
        num_queries = 50
        queries = torch.zeros(num_queries, 2)
        queries[:, 0] = torch.rand(num_queries) * (max_x - min_x) + min_x
        queries[:, 1] = torch.rand(num_queries) * (max_y - min_y) + min_y

        gpu_distances = collection.distance_to_points(queries)

        # Compare sample of results with shapely
        for i in range(min(10, num_queries)):
            point = shapely.Point(queries[i].tolist())
            for j in range(min(10, sample_size)):
                geom = sample_geoms[j]
                shapely_dist = point.distance(geom)

                # For polygons, check signed distance
                if geom.geom_type == "Polygon":
                    shapely_inside = geom.contains(point)
                    gpu_inside = gpu_distances[i, j].item() < 0
                    self.assertEqual(
                        gpu_inside,
                        shapely_inside,
                        f"Inside/outside mismatch at point {i}, geom {j}",
                    )
                    shapely_dist = point.distance(geom.boundary)

                torch.testing.assert_close(
                    gpu_distances[i, j].abs(),
                    torch.tensor(shapely_dist, dtype=torch.float32),
                    rtol=1e-3,
                    atol=1e-4,
                )


class TestBenchmark(unittest.TestCase):
    """Benchmark tests comparing GPU vs CPU performance."""

    @classmethod
    def setUpClass(cls):
        """Load vigor_snippet landmarks for benchmarking."""
        landmark_path = Path(
            "external/vigor_snippet/vigor_snippet/landmarks/v1_test.feather"
        )
        if not landmark_path.exists():
            raise unittest.SkipTest(f"Landmark file not found: {landmark_path}")

        cls.landmarks_df = gpd.read_feather(landmark_path)
        cls.geometries = cls.landmarks_df.geometry.values

    def test_benchmark_distance_computation(self):
        """Benchmark distance computation: GPU collection vs shapely."""
        # Use subset for benchmarking
        num_geoms = 1000
        num_queries = 1000
        sample_geoms = self.geometries[:num_geoms]

        # Build GPU collection
        collection = GPUGeometryCollection.from_shapely(sample_geoms)

        # Generate random query points
        bounds = shapely.bounds(shapely.GeometryCollection(list(sample_geoms)))
        min_x, min_y, max_x, max_y = bounds

        torch.manual_seed(42)
        queries = torch.zeros(num_queries, 2)
        queries[:, 0] = torch.rand(num_queries) * (max_x - min_x) + min_x
        queries[:, 1] = torch.rand(num_queries) * (max_y - min_y) + min_y

        # Benchmark GPU
        start = time.time()
        _ = collection.distance_to_points(queries)
        gpu_time = time.time() - start

        # Benchmark shapely (sample due to O(n*m) complexity)
        shapely_sample_queries = 10
        shapely_sample_geoms = 100
        start = time.time()
        for i in range(shapely_sample_queries):
            point = shapely.Point(queries[i].tolist())
            for j in range(shapely_sample_geoms):
                _ = point.distance(sample_geoms[j])
        shapely_time = time.time() - start

        # Extrapolate shapely time
        shapely_extrapolated = shapely_time * (
            num_queries / shapely_sample_queries
        ) * (num_geoms / shapely_sample_geoms)

        print(f"\nBenchmark: {num_queries} queries x {num_geoms} geometries")
        print(f"  GPU time: {gpu_time:.3f}s")
        print(f"  Shapely time (extrapolated): {shapely_extrapolated:.3f}s")
        print(f"  Speedup: {shapely_extrapolated / gpu_time:.1f}x")


if __name__ == "__main__":
    unittest.main()
