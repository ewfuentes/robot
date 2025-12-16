"""Tests for GPU geometry collection and spatial index."""

import time
import unittest
from pathlib import Path

import common.torch.load_torch_deps  # Must be imported before torch  # noqa: F401
import geopandas as gpd
import numpy as np
import shapely
import torch

from common.geometry.gpu_geometry_collection import (
    GPUGeometryCollection,
    GPUSpatialIndex,
    GeometryType,
    sample_geometry_boundary,
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


class TestSampleGeometryBoundary(unittest.TestCase):
    def test_sample_point(self):
        """Sampling a point returns just that point."""
        point = shapely.Point(1, 2)
        samples = sample_geometry_boundary(point, spacing=10.0)

        np.testing.assert_array_equal(samples, [[1.0, 2.0]])

    def test_sample_linestring(self):
        """Sampling a linestring returns evenly spaced points."""
        line = shapely.LineString([(0, 0), (10, 0)])
        samples = sample_geometry_boundary(line, spacing=2.5)

        # Should have 5 points: 0, 2.5, 5, 7.5, 10
        self.assertEqual(samples.shape[0], 5)
        np.testing.assert_array_almost_equal(samples[0], [0, 0])
        np.testing.assert_array_almost_equal(samples[-1], [10, 0])

    def test_sample_polygon(self):
        """Sampling a polygon samples its boundary."""
        polygon = shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        samples = sample_geometry_boundary(polygon, spacing=5.0)

        # Perimeter is 40, so roughly 9 samples
        self.assertGreater(samples.shape[0], 5)


class TestGPUSpatialIndex(unittest.TestCase):
    def test_build_index(self):
        """Test building spatial index."""
        geometries = [
            shapely.Point(0, 0),
            shapely.LineString([(5, 0), (5, 10)]),
            shapely.Polygon([(10, 0), (15, 0), (15, 5), (10, 5)]),
        ]

        index = GPUSpatialIndex.build(geometries, sample_spacing=2.0)

        self.assertEqual(index.collection.num_geometries, 3)
        self.assertGreater(index.sample_points.shape[0], 3)

    def test_query_nearest(self):
        """Test nearest neighbor query."""
        geometries = [
            shapely.Point(0, 0),
            shapely.Point(10, 0),
            shapely.Point(20, 0),
        ]

        index = GPUSpatialIndex.build(geometries, sample_spacing=1.0)

        query = torch.tensor([[5.0, 0.0]])
        geom_idxs, distances = index.query_nearest(query, k=2)

        self.assertEqual(geom_idxs.shape, (1, 2))
        # Nearest should be geometry 0 or 1 (both at distance 5)
        self.assertIn(geom_idxs[0, 0].item(), [0, 1])

    def test_query_within_distance(self):
        """Test range query."""
        geometries = [
            shapely.Point(0, 0),
            shapely.Point(10, 0),
            shapely.Point(100, 0),
        ]

        index = GPUSpatialIndex.build(geometries, sample_spacing=1.0)

        query = torch.tensor([[5.0, 0.0]])
        query_idxs, geom_idxs, distances = index.query_within_distance(
            query, distance=6.0
        )

        # Should find geometries 0 and 1 (both within distance 6)
        found_geoms = set(geom_idxs.tolist())
        self.assertIn(0, found_geoms)
        self.assertIn(1, found_geoms)
        self.assertNotIn(2, found_geoms)


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

    def test_build_spatial_index(self):
        """Test building spatial index on full dataset."""
        index = GPUSpatialIndex.build(self.geometries, sample_spacing=0.001)

        self.assertEqual(index.collection.num_geometries, len(self.geometries))
        self.assertGreater(index.sample_points.shape[0], len(self.geometries))

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

    def test_benchmark_spatial_index(self):
        """Benchmark spatial index query vs shapely STRtree."""
        num_geoms = min(10000, len(self.geometries))
        num_queries = 1000
        search_distance = 0.001  # degrees

        sample_geoms = list(self.geometries[:num_geoms])

        # Build GPU spatial index
        start = time.time()
        index = GPUSpatialIndex.build(sample_geoms, sample_spacing=0.0001)
        gpu_build_time = time.time() - start

        # Build shapely STRtree
        start = time.time()
        strtree = shapely.STRtree(sample_geoms)
        shapely_build_time = time.time() - start

        # Generate random query points
        bounds = shapely.bounds(shapely.GeometryCollection(sample_geoms))
        min_x, min_y, max_x, max_y = bounds

        torch.manual_seed(42)
        queries = torch.zeros(num_queries, 2)
        queries[:, 0] = torch.rand(num_queries) * (max_x - min_x) + min_x
        queries[:, 1] = torch.rand(num_queries) * (max_y - min_y) + min_y

        query_points = shapely.points(queries.numpy())

        # Benchmark GPU spatial index
        start = time.time()
        _, _, _ = index.query_within_distance(queries, distance=search_distance)
        gpu_query_time = time.time() - start

        # Benchmark shapely STRtree
        start = time.time()
        for i, pt in enumerate(query_points):
            candidates_idx = strtree.query(pt.buffer(search_distance))
            for j in candidates_idx:
                _ = pt.distance(sample_geoms[j])
        shapely_query_time = time.time() - start

        print(f"\nSpatial Index Benchmark: {num_queries} queries, {num_geoms} geometries")
        print(f"  Build time - GPU: {gpu_build_time:.3f}s, STRtree: {shapely_build_time:.3f}s")
        print(f"  Query time - GPU: {gpu_query_time:.3f}s, STRtree: {shapely_query_time:.3f}s")
        if shapely_query_time > 0:
            print(f"  Query speedup: {shapely_query_time / gpu_query_time:.1f}x")


if __name__ == "__main__":
    unittest.main()
