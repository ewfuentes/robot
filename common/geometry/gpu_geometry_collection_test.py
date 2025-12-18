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

    def test_build_spatial_index_simple(self):
        """Test building spatial index on simple geometries."""
        # Create simple geometry collection
        geometries = [
            shapely.Point(0, 0),
            shapely.Point(10, 10),
            shapely.LineString([(5, 0), (5, 10)]),
        ]
        collection = GPUGeometryCollection.from_shapely(geometries)

        # Build spatial index
        cell_size = 5.0
        expansion_distance = 2.5
        collection.build_spatial_index(cell_size, expansion_distance)

        # Verify index was created
        self.assertIsNotNone(collection.spatial_index)
        self.assertEqual(collection.spatial_index.cell_size, cell_size)
        self.assertEqual(collection.spatial_index.expansion_distance, expansion_distance)

        # Verify grid bounds make sense
        self.assertEqual(collection.spatial_index.grid_origin.shape, (2,))
        self.assertEqual(collection.spatial_index.grid_dims.shape, (2,))
        self.assertGreater(collection.spatial_index.grid_dims[0].item(), 0)
        self.assertGreater(collection.spatial_index.grid_dims[1].item(), 0)

    def test_spatial_index_csr_structure(self):
        """Test that CSR structure is correctly built."""
        # Create a small grid with known geometry
        geometries = [
            shapely.LineString([(0, 0), (10, 0)]),  # Horizontal line
            shapely.Point(20, 20),  # Far away point
        ]
        collection = GPUGeometryCollection.from_shapely(geometries)

        cell_size = 5.0
        expansion_distance = 2.5
        collection.build_spatial_index(cell_size, expansion_distance)

        idx = collection.spatial_index

        # Check CSR format
        num_cells = idx.grid_dims[0].item() * idx.grid_dims[1].item()
        self.assertEqual(idx.cell_offsets.shape[0], num_cells + 1)

        # First offset should be 0
        self.assertEqual(idx.cell_offsets[0].item(), 0)

        # Last offset should equal length of segment indices
        self.assertEqual(
            idx.cell_offsets[-1].item(), idx.cell_segment_indices.shape[0]
        )

        # Offsets should be monotonically increasing
        for i in range(len(idx.cell_offsets) - 1):
            self.assertLessEqual(idx.cell_offsets[i].item(), idx.cell_offsets[i + 1].item())

    def test_spatial_index_segment_assignment(self):
        """Test that segments are assigned to correct cells."""
        # Create a single horizontal segment at y=0
        geometries = [shapely.LineString([(0, 0), (10, 0)])]
        collection = GPUGeometryCollection.from_shapely(geometries)

        cell_size = 5.0
        expansion_distance = 1.0  # Small expansion
        collection.build_spatial_index(cell_size, expansion_distance)

        idx = collection.spatial_index

        # Segment should appear in cells along the line
        # With expansion_distance=1.0, segment from (0,0) to (10,0) should be in cells
        # that cover y in [-1, 1] and x in [-1, 11]

        # Check that at least some cells have segments
        non_empty_cells = (idx.cell_offsets[1:] - idx.cell_offsets[:-1]) > 0
        self.assertGreater(non_empty_cells.sum().item(), 0)

        # All segment indices should be valid (0 since we have 1 segment)
        self.assertTrue((idx.cell_segment_indices == 0).all())

    def test_spatial_index_empty_collection(self):
        """Test building spatial index on empty collection."""
        geometries = []
        # Need to create manually since from_shapely expects non-empty
        device = torch.device("cpu")
        collection = GPUGeometryCollection(
            device=device,
            num_geometries=0,
            geometry_types=torch.empty(0, dtype=torch.long),
            segment_starts=torch.empty((0, 2), dtype=torch.float32),
            segment_ends=torch.empty((0, 2), dtype=torch.float32),
            segment_to_geom=torch.empty(0, dtype=torch.long),
            point_coords=torch.empty((0, 2), dtype=torch.float32),
            point_to_geom=torch.empty(0, dtype=torch.long),
            polygon_vertices=torch.empty((0, 2), dtype=torch.float32),
            polygon_ranges=torch.empty((0, 2), dtype=torch.long),
            polygon_geom_indices=torch.empty(0, dtype=torch.long),
        )

        # Should not crash on empty collection
        collection.build_spatial_index(cell_size=5.0, expansion_distance=2.5)

        self.assertIsNotNone(collection.spatial_index)
        self.assertEqual(collection.spatial_index.cell_segment_indices.shape[0], 0)
        self.assertEqual(collection.spatial_index.cell_point_indices.shape[0], 0)

    def test_spatial_index_point_assignment(self):
        """Test that points are assigned to cells correctly."""
        geometries = [shapely.Point(5, 5), shapely.Point(15, 15)]
        collection = GPUGeometryCollection.from_shapely(geometries)

        cell_size = 10.0
        expansion_distance = 3.0
        collection.build_spatial_index(cell_size, expansion_distance)

        idx = collection.spatial_index

        # Points should be in the index
        self.assertGreater(idx.cell_point_indices.shape[0], 0)

        # All point indices should be valid (0 or 1)
        self.assertTrue((idx.cell_point_indices >= 0).all())
        self.assertTrue((idx.cell_point_indices < 2).all())

    def test_spatial_index_mixed_geometries(self):
        """Test spatial index with mixed geometry types."""
        geometries = [
            shapely.Point(5, 5),
            shapely.LineString([(0, 0), (10, 10)]),
            shapely.Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),
        ]
        collection = GPUGeometryCollection.from_shapely(geometries)

        cell_size = 5.0
        expansion_distance = 2.0
        collection.build_spatial_index(cell_size, expansion_distance)

        idx = collection.spatial_index

        # Should have both segments (from LineString and Polygon) and points
        self.assertGreater(idx.cell_segment_indices.shape[0], 0)
        self.assertGreater(idx.cell_point_indices.shape[0], 0)

        # Verify CSR structure is valid
        num_cells = idx.grid_dims[0].item() * idx.grid_dims[1].item()
        self.assertEqual(idx.cell_offsets.shape[0], num_cells + 1)
        self.assertEqual(idx.cell_point_offsets.shape[0], num_cells + 1)

    def test_spatial_index_large_expansion(self):
        """Test spatial index with large expansion distance."""
        geometries = [shapely.Point(50, 50)]
        collection = GPUGeometryCollection.from_shapely(geometries)

        cell_size = 10.0
        expansion_distance = 50.0  # Very large expansion
        collection.build_spatial_index(cell_size, expansion_distance)

        idx = collection.spatial_index

        # Point should appear in many cells due to large expansion
        # Each cell that contains this point should have it in the index
        total_point_assignments = idx.cell_point_indices.shape[0]
        self.assertGreater(total_point_assignments, 1)

        # All assignments should point to the same point (index 0)
        self.assertTrue((idx.cell_point_indices == 0).all())

    def test_spatial_index_correctness(self):
        """Test that spatial index correctly identifies geometries near query points."""
        # Create a grid of known geometries
        geometries = [
            shapely.Point(10, 10),
            shapely.Point(100, 100),
            shapely.LineString([(50, 50), (60, 50)]),
        ]
        collection = GPUGeometryCollection.from_shapely(geometries)

        cell_size = 20.0
        expansion_distance = 15.0
        collection.build_spatial_index(cell_size, expansion_distance)

        idx = collection.spatial_index

        # Test query: point at (10, 10) should find geometry 0 (the point at 10,10)
        query_point = torch.tensor([[10.0, 10.0]], device=collection.device)

        # Hash query to cell
        cell_coords = torch.floor((query_point - idx.grid_origin) / idx.cell_size).long()
        zeros = torch.zeros_like(idx.grid_dims)
        grid_max = idx.grid_dims - 1
        cell_coords = torch.maximum(cell_coords, zeros)
        cell_coords = torch.minimum(cell_coords, grid_max)
        cell_id = cell_coords[0, 1] * idx.grid_dims[0] + cell_coords[0, 0]

        # Get geometries in this cell
        start = idx.cell_point_offsets[cell_id].item()
        end = idx.cell_point_offsets[cell_id + 1].item()
        point_indices_in_cell = idx.cell_point_indices[start:end]

        # Geometry 0 (point at 10,10) should be in this cell
        self.assertIn(0, point_indices_in_cell.tolist())

        # Test query: point at (100, 100) should find geometry 1
        query_point2 = torch.tensor([[100.0, 100.0]], device=collection.device)
        cell_coords2 = torch.floor((query_point2 - idx.grid_origin) / idx.cell_size).long()
        cell_coords2 = torch.maximum(cell_coords2, zeros)
        cell_coords2 = torch.minimum(cell_coords2, grid_max)
        cell_id2 = cell_coords2[0, 1] * idx.grid_dims[0] + cell_coords2[0, 0]

        start2 = idx.cell_point_offsets[cell_id2].item()
        end2 = idx.cell_point_offsets[cell_id2 + 1].item()
        point_indices_in_cell2 = idx.cell_point_indices[start2:end2]

        self.assertIn(1, point_indices_in_cell2.tolist())

    def test_spatial_index_boundary_cases(self):
        """Test spatial index with geometries at grid boundaries."""
        # Create geometry at origin and far corner
        geometries = [
            shapely.Point(0, 0),
            shapely.Point(1000, 1000),
        ]
        collection = GPUGeometryCollection.from_shapely(geometries)

        cell_size = 100.0
        expansion_distance = 50.0
        collection.build_spatial_index(cell_size, expansion_distance)

        idx = collection.spatial_index

        # Both points should be in the index
        self.assertEqual(collection.num_geometries, 2)
        self.assertGreater(idx.cell_point_indices.shape[0], 0)

        # Verify grid covers both points
        self.assertLessEqual(idx.grid_origin[0].item(), 0 - expansion_distance)
        self.assertLessEqual(idx.grid_origin[1].item(), 0 - expansion_distance)

    def test_spatial_index_custom_bounds(self):
        """Test spatial index with custom grid bounds."""
        geometries = [shapely.Point(50, 50)]
        collection = GPUGeometryCollection.from_shapely(geometries)

        cell_size = 10.0
        expansion_distance = 5.0

        # Provide custom bounds larger than needed
        bbox_min = torch.tensor([0.0, 0.0], device=collection.device)
        bbox_max = torch.tensor([200.0, 200.0], device=collection.device)
        collection.build_spatial_index(
            cell_size,
            expansion_distance,
            grid_bounds=(bbox_min, bbox_max)
        )

        idx = collection.spatial_index

        # Grid should use custom bounds
        self.assertAlmostEqual(idx.grid_origin[0].item(), 0.0, places=1)
        self.assertAlmostEqual(idx.grid_origin[1].item(), 0.0, places=1)

        expected_dims_x = int(torch.ceil(torch.tensor(200.0 / cell_size)).item())
        expected_dims_y = int(torch.ceil(torch.tensor(200.0 / cell_size)).item())

        self.assertEqual(idx.grid_dims[0].item(), expected_dims_x)
        self.assertEqual(idx.grid_dims[1].item(), expected_dims_y)

    def test_spatial_index_multipolygon(self):
        """Test spatial index with MultiPolygon geometry."""
        # Create a MultiPolygon with two separated polygons
        poly1 = shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = shapely.Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])
        multipolygon = shapely.MultiPolygon([poly1, poly2])

        geometries = [multipolygon]
        collection = GPUGeometryCollection.from_shapely(geometries)

        cell_size = 20.0
        expansion_distance = 5.0
        collection.build_spatial_index(cell_size, expansion_distance)

        idx = collection.spatial_index

        # MultiPolygon should generate multiple segments (8 segments: 4 per polygon)
        self.assertGreater(idx.cell_segment_indices.shape[0], 0)

        # All segment indices should reference geometry 0 (the multipolygon)
        unique_geom_indices = set()
        for seg_idx in idx.cell_segment_indices:
            geom_idx = collection.segment_to_geom[seg_idx].item()
            unique_geom_indices.add(geom_idx)

        self.assertEqual(unique_geom_indices, {0})


if __name__ == "__main__":
    unittest.main()
