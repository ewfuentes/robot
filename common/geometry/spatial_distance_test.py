"""Tests for GPU-accelerated spatial distance queries."""

import unittest

import common.torch.load_torch_deps  # Must be before torch  # noqa: F401
import numpy as np
import shapely
import torch

from common.geometry.gpu_geometry_collection import GPUGeometryCollection


class SpatialDistanceTest(unittest.TestCase):
    """Test cases for spatial distance queries."""

    def test_query_distances_requires_spatial_index(self):
        """Test that query_distances requires spatial index to be built."""
        # Create simple collection without spatial index
        geometries = [shapely.Point(0, 0), shapely.Point(1, 1)]
        collection = GPUGeometryCollection.from_shapely(geometries, device=torch.device("cpu"))

        query_points = torch.tensor([[0.5, 0.5]], dtype=torch.float32).cuda()

        with self.assertRaises(ValueError) as context:
            collection.query_distances_cuda(query_points)
        self.assertIn("Spatial index must be built", str(context.exception))

    def test_query_distances_empty_cells(self):
        """Test query_distances with particles in empty cells."""
        # Create geometries clustered in one area
        geometries = [
            shapely.Point(0, 0),
            shapely.Point(0.1, 0.1),
        ]
        collection = GPUGeometryCollection.from_shapely(
            geometries, device=torch.device("cuda")
        )

        # Build spatial index
        collection.build_spatial_index(cell_size=1.0, expansion_distance=0.5)

        # Query points far from geometries (should be in empty cells)
        query_points = torch.tensor(
            [[10.0, 10.0], [20.0, 20.0]], dtype=torch.float32
        ).cuda()

        result = collection.query_distances_cuda(query_points)

        # Should return empty result (no geometries in those cells)
        self.assertEqual(result.shape[0], 0)
        self.assertEqual(result.shape[1], 3)

    def test_query_distances_single_point(self):
        """Test query_distances with a single point geometry."""
        # Create single point
        geometries = [shapely.Point(5.0, 5.0)]
        collection = GPUGeometryCollection.from_shapely(
            geometries, device=torch.device("cuda")
        )

        # Build spatial index
        collection.build_spatial_index(cell_size=2.0, expansion_distance=1.0)

        # Query from nearby location
        query_points = torch.tensor([[5.5, 5.5]], dtype=torch.float32).cuda()

        result = collection.query_distances_cuda(query_points)

        # Should have one result
        self.assertEqual(result.shape, (1, 3))
        particle_idx, geom_idx, distance = result[0]

        self.assertEqual(particle_idx, 0)
        self.assertEqual(geom_idx, 0)
        # Distance should be sqrt(0.5^2 + 0.5^2) ≈ 0.707
        expected_dist = np.sqrt(0.5**2 + 0.5**2)
        self.assertAlmostEqual(float(distance), expected_dist, places=5)

    def test_query_distances_line_segment(self):
        """Test query_distances with a line segment."""
        # Create horizontal line from (0,0) to (10,0)
        geometries = [shapely.LineString([(0, 0), (10, 0)])]
        collection = GPUGeometryCollection.from_shapely(
            geometries, device=torch.device("cuda")
        )

        # Build spatial index
        collection.build_spatial_index(cell_size=5.0, expansion_distance=2.0)

        # Query from point above the line
        query_points = torch.tensor([[5.0, 1.0]], dtype=torch.float32).cuda()

        result = collection.query_distances_cuda(query_points)

        # Should have one result
        self.assertEqual(result.shape, (1, 3))
        particle_idx, geom_idx, distance = result[0]

        self.assertEqual(particle_idx, 0)
        self.assertEqual(geom_idx, 0)
        # Distance from (5, 1) to line should be 1.0 (perpendicular distance)
        self.assertAlmostEqual(float(distance), 1.0, places=5)

    def test_query_distances_polygon_minimum(self):
        """Test that distance to polygon is minimum across all segments."""
        # Create square polygon
        geometries = [shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]
        collection = GPUGeometryCollection.from_shapely(
            geometries, device=torch.device("cuda")
        )

        # Build spatial index
        collection.build_spatial_index(cell_size=5.0, expansion_distance=2.0)

        # Query from point outside polygon, closest to one edge
        query_points = torch.tensor([[5.0, -1.0]], dtype=torch.float32).cuda()

        result = collection.query_distances_cuda(query_points)

        # Should have one result (one geometry)
        self.assertEqual(result.shape, (1, 3))
        particle_idx, geom_idx, distance = result[0]

        self.assertEqual(particle_idx, 0)
        self.assertEqual(geom_idx, 0)
        # Distance should be 1.0 (to bottom edge)
        self.assertAlmostEqual(float(distance), 1.0, places=5)

    def test_query_distances_multiple_particles(self):
        """Test query_distances with multiple query points."""
        # Create two points
        geometries = [shapely.Point(0, 0), shapely.Point(10, 10)]
        collection = GPUGeometryCollection.from_shapely(
            geometries, device=torch.device("cuda")
        )

        # Build spatial index
        collection.build_spatial_index(cell_size=5.0, expansion_distance=3.0)

        # Query from two locations
        query_points = torch.tensor(
            [[1.0, 1.0], [9.0, 9.0]], dtype=torch.float32
        ).cuda()

        result = collection.query_distances_cuda(query_points)

        # Each particle should find one geometry
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 3)

        # Check particle 0 (closest to geom 0)
        particle_0_results = result[result[:, 0] == 0]
        self.assertEqual(len(particle_0_results), 1)
        self.assertEqual(particle_0_results[0, 1], 0)  # geom_idx
        expected_dist_0 = np.sqrt(1.0**2 + 1.0**2)
        self.assertAlmostEqual(float(particle_0_results[0, 2]), expected_dist_0, places=5)

        # Check particle 1 (closest to geom 1)
        particle_1_results = result[result[:, 0] == 1]
        self.assertEqual(len(particle_1_results), 1)
        self.assertEqual(particle_1_results[0, 1], 1)  # geom_idx
        expected_dist_1 = np.sqrt(1.0**2 + 1.0**2)
        self.assertAlmostEqual(float(particle_1_results[0, 2]), expected_dist_1, places=5)

    def test_query_distances_vs_shapely(self):
        """Test correctness vs shapely on small dataset."""
        # Create mixed geometries
        geometries = [
            shapely.Point(0, 0),
            shapely.LineString([(5, 5), (5, 10)]),
            shapely.Polygon([(10, 0), (15, 0), (15, 5), (10, 5)]),
        ]
        collection = GPUGeometryCollection.from_shapely(
            geometries, device=torch.device("cuda")
        )

        # Build spatial index with large expansion to ensure all geometries visible
        collection.build_spatial_index(cell_size=100.0, expansion_distance=20.0)

        # Query points
        query_points = torch.tensor(
            [[1.0, 1.0], [6.0, 7.0], [9.0, 2.0]], dtype=torch.float32
        ).cuda()

        result = collection.query_distances_cuda(query_points)

        # Convert to dict for easy lookup
        result_dict = {}
        for particle_idx, geom_idx, distance in result:
            particle_idx = int(particle_idx)
            geom_idx = int(geom_idx)
            if particle_idx not in result_dict:
                result_dict[particle_idx] = {}
            result_dict[particle_idx][geom_idx] = float(distance)

        # Verify against shapely
        for particle_idx, query_pt in enumerate(query_points):
            query_shapely = shapely.Point(query_pt[0].item(), query_pt[1].item())

            for geom_idx, geom in enumerate(geometries):
                expected_dist = query_shapely.distance(geom)
                # Check if we found this pair
                if particle_idx in result_dict and geom_idx in result_dict[particle_idx]:
                    actual_dist = result_dict[particle_idx][geom_idx]
                    self.assertAlmostEqual(
                        actual_dist, expected_dist, places=4,
                        msg=f"Mismatch for particle {particle_idx}, geom {geom_idx}: "
                            f"expected {expected_dist}, got {actual_dist}"
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_query_distances_cuda(self):
        """Test query_distances on CUDA device if available."""
        device = torch.device("cuda")

        # Create geometries
        geometries = [
            shapely.Point(0, 0),
            shapely.LineString([(5, 5), (5, 10)]),
            shapely.Polygon([(10, 0), (15, 0), (15, 5), (10, 5)]),
        ]
        collection = GPUGeometryCollection.from_shapely(geometries, device=device)

        # Build spatial index
        collection.build_spatial_index(cell_size=10.0, expansion_distance=20.0)

        # Query points
        query_points = torch.tensor(
            [[1.0, 1.0], [6.0, 7.0]], dtype=torch.float32, device=device
        ).cuda()

        result = collection.query_distances_cuda(query_points)

        # Check result is on CUDA
        self.assertEqual(result.device.type, "cuda")
        self.assertGreater(result.shape[0], 0)  # Should find some distances
        self.assertEqual(result.shape[1], 3)


    def test_polygon_inside_distance_zero(self):
        """Test that points inside polygon have distance 0."""
        # Create square polygon
        geometries = [shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]
        collection = GPUGeometryCollection.from_shapely(
            geometries, device=torch.device("cuda")
        )

        # Build spatial index
        collection.build_spatial_index(cell_size=5.0, expansion_distance=2.0)

        # Query from point inside polygon
        query_points = torch.tensor([[5.0, 5.0]], dtype=torch.float32).cuda()

        result = collection.query_distances_cuda(query_points)

        # Should have one result
        self.assertEqual(result.shape, (1, 3))
        particle_idx, geom_idx, distance = result[0]

        self.assertEqual(particle_idx, 0)
        self.assertEqual(geom_idx, 0)
        # Distance should be 0 for point inside polygon
        self.assertAlmostEqual(float(distance), 0.0, places=5)

    def test_polygon_with_hole(self):
        """Test polygon with a hole - point inside hole should have positive distance."""
        # Create square polygon with a square hole in the center
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]  # CW orientation for hole
        polygon_with_hole = shapely.Polygon(exterior, [hole])

        geometries = [polygon_with_hole]
        collection = GPUGeometryCollection.from_shapely(
            geometries, device=torch.device("cuda")
        )

        # Build spatial index
        collection.build_spatial_index(cell_size=5.0, expansion_distance=2.0)

        # Test point inside the hole (should NOT be inside polygon)
        query_inside_hole = torch.tensor([[5.0, 5.0]], dtype=torch.float32).cuda()
        result_hole = collection.query_distances_cuda(query_inside_hole)

        self.assertEqual(result_hole.shape, (1, 3))
        distance_in_hole = float(result_hole[0, 2])
        # Point is inside hole, so it's outside the polygon - distance should be > 0
        # Distance from (5, 5) to nearest edge of hole is min(5-3, 7-5, 5-3, 7-5) = 2
        self.assertAlmostEqual(distance_in_hole, 2.0, places=4)

        # Test point between exterior and hole (inside polygon)
        query_inside_polygon = torch.tensor([[1.0, 5.0]], dtype=torch.float32).cuda()
        result_inside = collection.query_distances_cuda(query_inside_polygon)

        self.assertEqual(result_inside.shape, (1, 3))
        distance_inside = float(result_inside[0, 2])
        # Point is inside polygon (between exterior and hole) - distance should be 0
        self.assertAlmostEqual(distance_inside, 0.0, places=4)

        # Test point outside exterior (outside polygon)
        query_outside = torch.tensor([[-1.0, 5.0]], dtype=torch.float32).cuda()
        result_outside = collection.query_distances_cuda(query_outside)

        self.assertEqual(result_outside.shape, (1, 3))
        distance_outside = float(result_outside[0, 2])
        # Point is outside polygon - distance should be 1.0 (to left edge)
        self.assertAlmostEqual(distance_outside, 1.0, places=4)

    def test_polygon_with_multiple_holes(self):
        """Test polygon with multiple holes."""
        # Create rectangle with two holes
        exterior = [(0, 0), (20, 0), (20, 10), (0, 10)]
        hole1 = [(2, 2), (8, 2), (8, 8), (2, 8)]
        hole2 = [(12, 2), (18, 2), (18, 8), (12, 8)]
        polygon_with_holes = shapely.Polygon(exterior, [hole1, hole2])

        geometries = [polygon_with_holes]
        collection = GPUGeometryCollection.from_shapely(
            geometries, device=torch.device("cuda")
        )

        # Build spatial index
        collection.build_spatial_index(cell_size=5.0, expansion_distance=2.0)

        # Test points at various locations
        query_points = torch.tensor([
            [5.0, 5.0],   # Inside hole 1
            [15.0, 5.0],  # Inside hole 2
            [10.0, 5.0],  # Between holes (inside polygon)
            [1.0, 5.0],   # Left of hole 1 (inside polygon)
        ], dtype=torch.float32).cuda()

        result = collection.query_distances_cuda(query_points)

        # Convert to dict for easy lookup
        distances = {}
        for particle_idx, geom_idx, distance in result:
            distances[int(particle_idx)] = float(distance)

        # Inside hole 1 - should have positive distance
        self.assertGreater(distances[0], 0, "Point in hole 1 should have positive distance")

        # Inside hole 2 - should have positive distance
        self.assertGreater(distances[1], 0, "Point in hole 2 should have positive distance")

        # Between holes - inside polygon, distance should be 0
        self.assertAlmostEqual(distances[2], 0.0, places=4,
                               msg="Point between holes should be inside polygon")

        # Left of hole 1 - inside polygon, distance should be 0
        self.assertAlmostEqual(distances[3], 0.0, places=4,
                               msg="Point left of hole should be inside polygon")

    def test_multipolygon_with_holes(self):
        """Test MultiPolygon where some polygons have holes."""
        # Create a multipolygon: one simple polygon, one with a hole
        simple_poly = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        exterior = [(10, 0), (20, 0), (20, 10), (10, 10)]
        hole = [(13, 3), (17, 3), (17, 7), (13, 7)]
        poly_with_hole = shapely.Polygon(exterior, [hole])
        multi = shapely.MultiPolygon([simple_poly, poly_with_hole])

        geometries = [multi]
        collection = GPUGeometryCollection.from_shapely(
            geometries, device=torch.device("cuda")
        )

        # Build spatial index
        collection.build_spatial_index(cell_size=5.0, expansion_distance=2.0)

        # Test points
        query_points = torch.tensor([
            [2.5, 2.5],   # Inside simple polygon
            [15.0, 5.0],  # Inside hole of second polygon
            [11.0, 5.0],  # Inside second polygon (not in hole)
        ], dtype=torch.float32).cuda()

        result = collection.query_distances_cuda(query_points)

        distances = {}
        for particle_idx, geom_idx, distance in result:
            distances[int(particle_idx)] = float(distance)

        # Inside simple polygon
        self.assertAlmostEqual(distances[0], 0.0, places=4)

        # Inside hole - should have positive distance
        self.assertGreater(distances[1], 0, "Point in hole should have positive distance")

        # Inside second polygon (not in hole)
        self.assertAlmostEqual(distances[2], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
