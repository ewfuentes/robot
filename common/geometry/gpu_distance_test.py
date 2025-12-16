"""Tests for GPU distance computation functions."""

import unittest
import math

import common.torch.load_torch_deps  # Must be imported before torch  # noqa: F401
import torch
import shapely
import numpy as np

from common.geometry import gpu_distance


class TestPointToPointDistance(unittest.TestCase):
    def test_simple_distance(self):
        query = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        target = torch.tensor([[3.0, 4.0], [1.0, 1.0]])

        result = gpu_distance.point_to_point_distance(query, target)

        expected = torch.tensor([5.0, 1.0])
        torch.testing.assert_close(result, expected)

    def test_broadcasting(self):
        # Single query point, multiple targets
        query = torch.tensor([0.0, 0.0])
        target = torch.tensor([[3.0, 4.0], [0.0, 1.0], [1.0, 0.0]])

        result = gpu_distance.point_to_point_distance(query, target)

        expected = torch.tensor([5.0, 1.0, 1.0])
        torch.testing.assert_close(result, expected)

    def test_zero_distance(self):
        query = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[1.0, 2.0]])

        result = gpu_distance.point_to_point_distance(query, target)

        torch.testing.assert_close(result, torch.tensor([0.0]))


class TestPointToSegmentDistance(unittest.TestCase):
    def test_point_projects_onto_segment(self):
        # Point directly above middle of horizontal segment
        query = torch.tensor([[0.5, 1.0]])
        starts = torch.tensor([[0.0, 0.0]])
        ends = torch.tensor([[1.0, 0.0]])

        result = gpu_distance.point_to_segment_distance(query, starts, ends)

        expected = torch.tensor([[1.0]])
        torch.testing.assert_close(result, expected)

    def test_point_closest_to_start(self):
        # Point to the left of segment start
        query = torch.tensor([[-1.0, 0.0]])
        starts = torch.tensor([[0.0, 0.0]])
        ends = torch.tensor([[1.0, 0.0]])

        result = gpu_distance.point_to_segment_distance(query, starts, ends)

        expected = torch.tensor([[1.0]])
        torch.testing.assert_close(result, expected)

    def test_point_closest_to_end(self):
        # Point to the right of segment end
        query = torch.tensor([[2.0, 0.0]])
        starts = torch.tensor([[0.0, 0.0]])
        ends = torch.tensor([[1.0, 0.0]])

        result = gpu_distance.point_to_segment_distance(query, starts, ends)

        expected = torch.tensor([[1.0]])
        torch.testing.assert_close(result, expected)

    def test_degenerate_segment(self):
        # Segment with start == end (degenerate to a point)
        query = torch.tensor([[1.0, 1.0]])
        starts = torch.tensor([[0.0, 0.0]])
        ends = torch.tensor([[0.0, 0.0]])

        result = gpu_distance.point_to_segment_distance(query, starts, ends)

        expected = torch.tensor([[math.sqrt(2)]])
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_multiple_queries_multiple_segments(self):
        # 2 query points, 3 segments
        queries = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
        starts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        ends = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

        result = gpu_distance.point_to_segment_distance(queries, starts, ends)

        self.assertEqual(result.shape, (2, 3))
        # Query 0 (0, 1) to segment 0 (horizontal at y=0): distance 1
        # Query 0 (0, 1) to segment 1 (vertical at x=1): distance 1
        # Query 0 (0, 1) to segment 2 (vertical at x=0): distance 0
        torch.testing.assert_close(result[0, 0], torch.tensor(1.0))
        torch.testing.assert_close(result[0, 1], torch.tensor(1.0))
        torch.testing.assert_close(result[0, 2], torch.tensor(0.0))

    def test_matches_shapely(self):
        """Compare GPU results to shapely.distance() for random inputs."""
        torch.manual_seed(42)
        Q, S = 50, 30
        queries = torch.rand(Q, 2) * 100
        starts = torch.rand(S, 2) * 100
        ends = torch.rand(S, 2) * 100

        gpu_result = gpu_distance.point_to_segment_distance(queries, starts, ends)

        # Compare with shapely
        for i in range(Q):
            point = shapely.Point(queries[i].tolist())
            for j in range(S):
                line = shapely.LineString(
                    [starts[j].tolist(), ends[j].tolist()]
                )
                shapely_dist = point.distance(line)
                torch.testing.assert_close(
                    gpu_result[i, j],
                    torch.tensor(shapely_dist, dtype=torch.float32),
                    rtol=1e-4,
                    atol=1e-5,
                )


class TestPointToSegmentsMinDistance(unittest.TestCase):
    def test_simple_groups(self):
        queries = torch.tensor([[0.0, 0.5]])
        # Two segments in group 0, one segment in group 1
        starts = torch.tensor([[1.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
        ends = torch.tensor([[1.0, 1.0], [2.0, 1.0], [5.0, 1.0]])
        segment_to_group = torch.tensor([0, 0, 1])

        result = gpu_distance.point_to_segments_min_distance(
            queries, starts, ends, segment_to_group, num_groups=2
        )

        self.assertEqual(result.shape, (1, 2))
        # Group 0: min distance to segments at x=1 or x=2 -> distance 1.0
        # Group 1: distance to segment at x=5 -> distance 5.0
        torch.testing.assert_close(result[0, 0], torch.tensor(1.0))
        torch.testing.assert_close(result[0, 1], torch.tensor(5.0))


class TestPointInPolygonWinding(unittest.TestCase):
    def test_point_inside_square(self):
        # Unit square with vertices at (0,0), (1,0), (1,1), (0,1)
        vertices = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        ranges = torch.tensor([[0, 4]])

        # Point at center of square
        query = torch.tensor([[0.5, 0.5]])

        result = gpu_distance.point_in_polygon_winding(query, vertices, ranges)

        self.assertEqual(result.shape, (1, 1))
        self.assertTrue(result[0, 0].item())

    def test_point_outside_square(self):
        vertices = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        ranges = torch.tensor([[0, 4]])

        # Point outside square
        query = torch.tensor([[2.0, 0.5]])

        result = gpu_distance.point_in_polygon_winding(query, vertices, ranges)

        self.assertFalse(result[0, 0].item())

    def test_multiple_polygons(self):
        # Two squares: one at origin, one shifted by (3, 0)
        vertices = torch.tensor([
            # Polygon 0
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
            # Polygon 1
            [3.0, 0.0], [4.0, 0.0], [4.0, 1.0], [3.0, 1.0],
        ])
        ranges = torch.tensor([[0, 4], [4, 8]])

        # Point inside polygon 0 only
        query = torch.tensor([[0.5, 0.5], [3.5, 0.5], [5.0, 0.5]])

        result = gpu_distance.point_in_polygon_winding(query, vertices, ranges)

        self.assertEqual(result.shape, (3, 2))
        # Point 0: inside poly 0, outside poly 1
        self.assertTrue(result[0, 0].item())
        self.assertFalse(result[0, 1].item())
        # Point 1: outside poly 0, inside poly 1
        self.assertFalse(result[1, 0].item())
        self.assertTrue(result[1, 1].item())
        # Point 2: outside both
        self.assertFalse(result[2, 0].item())
        self.assertFalse(result[2, 1].item())

    def test_matches_shapely(self):
        """Compare GPU results to shapely.contains() for random polygons."""
        torch.manual_seed(42)

        # Create a few random convex polygons
        num_polygons = 5
        vertices_list = []
        ranges_list = []
        shapely_polygons = []

        offset = 0
        for _ in range(num_polygons):
            # Generate random convex polygon using convex hull of random points
            points = torch.rand(10, 2) * 10
            shapely_points = [shapely.Point(p.tolist()) for p in points]
            hull = shapely.convex_hull(shapely.MultiPoint(shapely_points))

            if hull.geom_type != "Polygon":
                continue

            coords = list(hull.exterior.coords)[:-1]  # Remove duplicate closing point
            poly_verts = torch.tensor(coords, dtype=torch.float32)

            vertices_list.append(poly_verts)
            ranges_list.append([offset, offset + len(poly_verts)])
            offset += len(poly_verts)
            shapely_polygons.append(hull)

        if not vertices_list:
            self.skipTest("No valid polygons generated")

        vertices = torch.cat(vertices_list, dim=0)
        ranges = torch.tensor(ranges_list)

        # Generate random query points
        queries = torch.rand(100, 2) * 10

        gpu_result = gpu_distance.point_in_polygon_winding(queries, vertices, ranges)

        # Compare with shapely
        for i in range(queries.shape[0]):
            point = shapely.Point(queries[i].tolist())
            for j, poly in enumerate(shapely_polygons):
                shapely_inside = poly.contains(point)
                gpu_inside = gpu_result[i, j].item()
                self.assertEqual(
                    gpu_inside,
                    shapely_inside,
                    f"Mismatch at point {i}, polygon {j}: "
                    f"GPU={gpu_inside}, Shapely={shapely_inside}",
                )


class TestSignedDistanceToPolygons(unittest.TestCase):
    def test_inside_negative_outside_positive(self):
        # Unit square
        vertices = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        ranges = torch.tensor([[0, 4]])

        # Points inside and outside
        queries = torch.tensor([[0.5, 0.5], [2.0, 0.5]])

        result = gpu_distance.signed_distance_to_polygons(queries, vertices, ranges)

        self.assertEqual(result.shape, (2, 1))
        # Inside point: negative distance
        self.assertLess(result[0, 0].item(), 0)
        # Outside point: positive distance
        self.assertGreater(result[1, 0].item(), 0)

    def test_distance_magnitude(self):
        # Unit square
        vertices = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        ranges = torch.tensor([[0, 4]])

        # Point at center (distance to nearest edge is 0.5)
        query_center = torch.tensor([[0.5, 0.5]])
        result_center = gpu_distance.signed_distance_to_polygons(
            query_center, vertices, ranges
        )
        torch.testing.assert_close(
            result_center[0, 0].abs(), torch.tensor(0.5), rtol=1e-5, atol=1e-5
        )

        # Point outside, 1 unit away from right edge
        query_outside = torch.tensor([[2.0, 0.5]])
        result_outside = gpu_distance.signed_distance_to_polygons(
            query_outside, vertices, ranges
        )
        torch.testing.assert_close(
            result_outside[0, 0], torch.tensor(1.0), rtol=1e-5, atol=1e-5
        )

    def test_matches_shapely_magnitude(self):
        """Verify distance magnitude matches shapely for points outside."""
        # Triangle polygon
        vertices = torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
        ranges = torch.tensor([[0, 3]])

        shapely_poly = shapely.Polygon(vertices.tolist())

        # Test points outside the polygon
        torch.manual_seed(42)
        queries = torch.rand(20, 2) * 4 - 0.5  # Some will be outside

        result = gpu_distance.signed_distance_to_polygons(queries, vertices, ranges)

        for i in range(queries.shape[0]):
            point = shapely.Point(queries[i].tolist())
            shapely_dist = point.distance(shapely_poly.boundary)
            shapely_inside = shapely_poly.contains(point)

            gpu_dist = result[i, 0].item()
            gpu_inside = gpu_dist < 0

            # Check inside/outside agreement
            self.assertEqual(
                gpu_inside,
                shapely_inside,
                f"Inside/outside mismatch at point {i}: "
                f"GPU inside={gpu_inside}, Shapely inside={shapely_inside}",
            )

            # Check distance magnitude
            torch.testing.assert_close(
                torch.tensor(abs(gpu_dist)),
                torch.tensor(shapely_dist, dtype=torch.float32),
                rtol=1e-4,
                atol=1e-5,
            )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestGPUDevice(unittest.TestCase):
    """Test that all functions work correctly on GPU."""

    def test_point_to_segment_on_gpu(self):
        device = torch.device("cuda")
        queries = torch.rand(100, 2, device=device)
        starts = torch.rand(50, 2, device=device)
        ends = torch.rand(50, 2, device=device)

        result = gpu_distance.point_to_segment_distance(queries, starts, ends)

        self.assertEqual(result.device.type, "cuda")
        self.assertEqual(result.shape, (100, 50))

    def test_point_in_polygon_on_gpu(self):
        device = torch.device("cuda")
        vertices = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], device=device
        )
        ranges = torch.tensor([[0, 4]], device=device)
        queries = torch.rand(100, 2, device=device)

        result = gpu_distance.point_in_polygon_winding(queries, vertices, ranges)

        self.assertEqual(result.device.type, "cuda")
        self.assertEqual(result.shape, (100, 1))

    def test_signed_distance_on_gpu(self):
        device = torch.device("cuda")
        vertices = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], device=device
        )
        ranges = torch.tensor([[0, 4]], device=device)
        queries = torch.rand(100, 2, device=device)

        result = gpu_distance.signed_distance_to_polygons(queries, vertices, ranges)

        self.assertEqual(result.device.type, "cuda")
        self.assertEqual(result.shape, (100, 1))


if __name__ == "__main__":
    unittest.main()
