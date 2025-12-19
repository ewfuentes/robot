"""Tests for GPU-accelerated spatial distance queries."""

import common.torch.load_torch_deps  # Must be before torch  # noqa: F401
import numpy as np
import shapely
import torch

from common.geometry.gpu_geometry_collection import GPUGeometryCollection


def test_query_distances_requires_spatial_index():
    """Test that query_distances requires spatial index to be built."""
    # Create simple collection without spatial index
    geometries = [shapely.Point(0, 0), shapely.Point(1, 1)]
    collection = GPUGeometryCollection.from_shapely(geometries, device=torch.device("cpu"))

    query_points = torch.tensor([[0.5, 0.5]], dtype=torch.float32)

    try:
        collection.query_distances(query_points)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Spatial index must be built" in str(e)


def test_query_distances_empty_cells():
    """Test query_distances with particles in empty cells."""
    # Create geometries clustered in one area
    geometries = [
        shapely.Point(0, 0),
        shapely.Point(0.1, 0.1),
    ]
    collection = GPUGeometryCollection.from_shapely(
        geometries, device=torch.device("cpu")
    )

    # Build spatial index
    collection.build_spatial_index(cell_size=1.0, expansion_distance=0.5)

    # Query points far from geometries (should be in empty cells)
    query_points = torch.tensor(
        [[10.0, 10.0], [20.0, 20.0]], dtype=torch.float32
    )

    result = collection.query_distances(query_points)

    # Should return empty result (no geometries in those cells)
    assert result.shape[0] == 0
    assert result.shape[1] == 3


def test_query_distances_single_point():
    """Test query_distances with a single point geometry."""
    # Create single point
    geometries = [shapely.Point(5.0, 5.0)]
    collection = GPUGeometryCollection.from_shapely(
        geometries, device=torch.device("cpu")
    )

    # Build spatial index
    collection.build_spatial_index(cell_size=2.0, expansion_distance=1.0)

    # Query from nearby location
    query_points = torch.tensor([[5.5, 5.5]], dtype=torch.float32)

    result = collection.query_distances(query_points)

    # Should have one result
    assert result.shape == (1, 3)
    particle_idx, geom_idx, distance = result[0]

    assert particle_idx == 0
    assert geom_idx == 0
    # Distance should be sqrt(0.5^2 + 0.5^2) ≈ 0.707
    expected_dist = np.sqrt(0.5**2 + 0.5**2)
    assert abs(distance - expected_dist) < 1e-5


def test_query_distances_line_segment():
    """Test query_distances with a line segment."""
    # Create horizontal line from (0,0) to (10,0)
    geometries = [shapely.LineString([(0, 0), (10, 0)])]
    collection = GPUGeometryCollection.from_shapely(
        geometries, device=torch.device("cpu")
    )

    # Build spatial index
    collection.build_spatial_index(cell_size=5.0, expansion_distance=2.0)

    # Query from point above the line
    query_points = torch.tensor([[5.0, 1.0]], dtype=torch.float32)

    result = collection.query_distances(query_points)

    # Should have one result
    assert result.shape == (1, 3)
    particle_idx, geom_idx, distance = result[0]

    assert particle_idx == 0
    assert geom_idx == 0
    # Distance from (5, 1) to line should be 1.0 (perpendicular distance)
    assert abs(distance - 1.0) < 1e-5


def test_query_distances_polygon_minimum():
    """Test that distance to polygon is minimum across all segments."""
    # Create square polygon
    geometries = [shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]
    collection = GPUGeometryCollection.from_shapely(
        geometries, device=torch.device("cpu")
    )

    # Build spatial index
    collection.build_spatial_index(cell_size=5.0, expansion_distance=2.0)

    # Query from point outside polygon, closest to one edge
    query_points = torch.tensor([[5.0, -1.0]], dtype=torch.float32)

    result = collection.query_distances(query_points)

    # Should have one result (one geometry)
    assert result.shape == (1, 3)
    particle_idx, geom_idx, distance = result[0]

    assert particle_idx == 0
    assert geom_idx == 0
    # Distance should be 1.0 (to bottom edge)
    assert abs(distance - 1.0) < 1e-5


def test_query_distances_multiple_particles():
    """Test query_distances with multiple query points."""
    # Create two points
    geometries = [shapely.Point(0, 0), shapely.Point(10, 10)]
    collection = GPUGeometryCollection.from_shapely(
        geometries, device=torch.device("cpu")
    )

    # Build spatial index
    collection.build_spatial_index(cell_size=5.0, expansion_distance=3.0)

    # Query from two locations
    query_points = torch.tensor(
        [[1.0, 1.0], [9.0, 9.0]], dtype=torch.float32
    )

    result = collection.query_distances(query_points)

    # Each particle should find one geometry
    assert result.shape[0] == 2
    assert result.shape[1] == 3

    # Check particle 0 (closest to geom 0)
    particle_0_results = result[result[:, 0] == 0]
    assert len(particle_0_results) == 1
    assert particle_0_results[0, 1] == 0  # geom_idx
    expected_dist_0 = np.sqrt(1.0**2 + 1.0**2)
    assert abs(particle_0_results[0, 2] - expected_dist_0) < 1e-5

    # Check particle 1 (closest to geom 1)
    particle_1_results = result[result[:, 0] == 1]
    assert len(particle_1_results) == 1
    assert particle_1_results[0, 1] == 1  # geom_idx
    expected_dist_1 = np.sqrt(1.0**2 + 1.0**2)
    assert abs(particle_1_results[0, 2] - expected_dist_1) < 1e-5


def test_query_distances_vs_shapely():
    """Test correctness vs shapely on small dataset."""
    # Create mixed geometries
    geometries = [
        shapely.Point(0, 0),
        shapely.LineString([(5, 5), (5, 10)]),
        shapely.Polygon([(10, 0), (15, 0), (15, 5), (10, 5)]),
    ]
    collection = GPUGeometryCollection.from_shapely(
        geometries, device=torch.device("cpu")
    )

    # Build spatial index with large expansion to ensure all geometries visible
    collection.build_spatial_index(cell_size=10.0, expansion_distance=20.0)

    # Query points
    query_points = torch.tensor(
        [[1.0, 1.0], [6.0, 7.0], [12.0, 2.0]], dtype=torch.float32
    )

    result = collection.query_distances(query_points)

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
                assert abs(actual_dist - expected_dist) < 1e-4, (
                    f"Mismatch for particle {particle_idx}, geom {geom_idx}: "
                    f"expected {expected_dist}, got {actual_dist}"
                )


def test_query_distances_cuda():
    """Test query_distances on CUDA device if available."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA test")
        return

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
    )

    result = collection.query_distances(query_points)

    # Check result is on CUDA
    assert result.device.type == "cuda"
    assert result.shape[0] > 0  # Should find some distances
    assert result.shape[1] == 3


if __name__ == "__main__":
    test_query_distances_requires_spatial_index()
    test_query_distances_empty_cells()
    test_query_distances_single_point()
    test_query_distances_line_segment()
    test_query_distances_polygon_minimum()
    test_query_distances_multiple_particles()
    test_query_distances_vs_shapely()
    test_query_distances_cuda()
    print("All tests passed!")
