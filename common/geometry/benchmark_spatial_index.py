"""Benchmark spatial index with real VIGOR landmark data.

This script loads VIGOR Chicago landmarks and benchmarks:
1. Conversion from shapely to GPUGeometryCollection
2. Spatial index construction
3. Memory usage
"""

import time
from pathlib import Path

import common.torch.load_torch_deps  # Must be before torch  # noqa: F401
import geopandas as gpd
import numpy as np
import shapely
import torch

from common.geometry.gpu_geometry_collection import GPUGeometryCollection
from common.gps import web_mercator


def convert_geometry_to_pixels(geometry, zoom_level: int):
    """Convert lat/lon geometry to web mercator pixels (same as VIGOR dataset)."""

    def coord_transform(lon, lat):
        x_px, y_px = web_mercator.latlon_to_pixel_coords(lat, lon, zoom_level)
        return x_px, y_px

    return shapely.ops.transform(coord_transform, geometry)


def load_vigor_landmarks(feather_path: Path, zoom_level: int = 20) -> gpd.GeoDataFrame:
    """Load VIGOR landmarks and convert to pixel coordinates."""
    print(f"Loading landmarks from {feather_path}")
    df = gpd.read_feather(feather_path)

    print(f"  Loaded {len(df)} landmarks")

    # Convert to pixels (same as vigor_dataset.py)
    df["geometry_px"] = df["geometry"].apply(
        lambda g: convert_geometry_to_pixels(g, zoom_level)
    )

    return df


def analyze_landmark_types(df: gpd.GeoDataFrame) -> None:
    """Print statistics about landmark geometry types."""
    print("\nLandmark geometry types:")
    type_counts = df.geometry_px.apply(lambda g: g.geom_type).value_counts()
    for geom_type, count in type_counts.items():
        print(f"  {geom_type}: {count}")

    # Compute total segments
    total_segments = 0
    for geom in df.geometry_px:
        if geom.geom_type == "Point":
            continue
        elif geom.geom_type == "LineString":
            total_segments += len(geom.coords) - 1
        elif geom.geom_type == "Polygon":
            total_segments += len(geom.exterior.coords) - 1
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                total_segments += len(poly.exterior.coords) - 1

    print(f"\nTotal line segments: {total_segments:,}")


def benchmark_conversion(df: gpd.GeoDataFrame, device: torch.device) -> GPUGeometryCollection:
    """Benchmark conversion from shapely to GPUGeometryCollection."""
    print("\n" + "=" * 60)
    print("BENCHMARKING: Shapely → GPUGeometryCollection")
    print("=" * 60)

    start = time.time()
    collection = GPUGeometryCollection.from_shapely(
        df.geometry_px.values, device=device
    )
    elapsed = time.time() - start

    print(f"Conversion time: {elapsed:.3f}s")
    print(f"  Segments: {collection.segment_starts.shape[0]:,}")
    print(f"  Points: {collection.point_coords.shape[0]:,}")
    print(f"  Polygons: {collection.polygon_ranges.shape[0]:,}")

    return collection


def benchmark_spatial_index(
    collection: GPUGeometryCollection,
    cell_size: float,
    expansion_distance: float,
) -> None:
    """Benchmark spatial index construction."""
    print("\n" + "=" * 60)
    print("BENCHMARKING: Spatial Index Construction")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  cell_size: {cell_size:.1f} pixels")
    print(f"  expansion_distance: {expansion_distance:.1f} pixels")

    start = time.time()
    profile = {}
    collection.build_spatial_index(cell_size, expansion_distance, profile=profile)
    elapsed = time.time() - start

    idx = collection.spatial_index
    print(f"\nBuild time: {elapsed:.3f}s")

    # Display detailed profiling breakdown
    if profile and 'segments' in profile:
        seg_prof = profile['segments']
        print(f"\n  Segment indexing breakdown:")
        if 'bbox_compute' in seg_prof:
            print(f"    Bbox computation: {seg_prof['bbox_compute']*1000:.1f}ms")
        if 'cell_coords' in seg_prof:
            print(f"    Cell coord conversion: {seg_prof['cell_coords']*1000:.1f}ms")
        if 'vectorized_expansion' in seg_prof:
            print(f"    Vectorized expansion: {seg_prof['vectorized_expansion']*1000:.1f}ms")
            if 'num_pairs' in seg_prof:
                print(f"      ({seg_prof['num_pairs']:,} pairs generated)")
        if 'sorting' in seg_prof:
            print(f"    Sorting: {seg_prof['sorting']*1000:.1f}ms")
        if 'csr_build' in seg_prof:
            print(f"    CSR construction: {seg_prof['csr_build']*1000:.1f}ms")

    if profile and 'points' in profile:
        pt_prof = profile['points']
        print(f"\n  Point indexing breakdown:")
        if 'bbox_compute' in pt_prof:
            print(f"    Bbox computation: {pt_prof['bbox_compute']*1000:.1f}ms")
        if 'cell_coords' in pt_prof:
            print(f"    Cell coord conversion: {pt_prof['cell_coords']*1000:.1f}ms")
        if 'vectorized_expansion' in pt_prof:
            print(f"    Vectorized expansion: {pt_prof['vectorized_expansion']*1000:.1f}ms")
            if 'num_pairs' in pt_prof:
                print(f"      ({pt_prof['num_pairs']:,} pairs generated)")
        if 'sorting' in pt_prof:
            print(f"    Sorting: {pt_prof['sorting']*1000:.1f}ms")
        if 'csr_build' in pt_prof:
            print(f"    CSR construction: {pt_prof['csr_build']*1000:.1f}ms")
    print(f"\nGrid info:")
    print(f"  Grid origin: ({idx.grid_origin[0]:.1f}, {idx.grid_origin[1]:.1f})")
    print(f"  Grid dimensions: {idx.grid_dims[0]} × {idx.grid_dims[1]}")
    print(f"  Total cells: {idx.grid_dims[0] * idx.grid_dims[1]:,}")

    print(f"\nIndex size:")
    print(f"  Segment assignments: {idx.cell_segment_indices.shape[0]:,}")
    print(f"  Point assignments: {idx.cell_point_indices.shape[0]:,}")

    # Compute average segments per cell
    num_cells = idx.grid_dims[0].item() * idx.grid_dims[1].item()
    non_empty_segment_cells = (
        (idx.cell_offsets[1:] - idx.cell_offsets[:-1]) > 0
    ).sum().item()
    non_empty_point_cells = (
        (idx.cell_point_offsets[1:] - idx.cell_point_offsets[:-1]) > 0
    ).sum().item()

    print(f"\nOccupancy:")
    print(
        f"  Non-empty cells (segments): {non_empty_segment_cells:,} "
        f"({100 * non_empty_segment_cells / num_cells:.1f}%)"
    )
    print(
        f"  Non-empty cells (points): {non_empty_point_cells:,} "
        f"({100 * non_empty_point_cells / num_cells:.1f}%)"
    )

    if idx.cell_segment_indices.shape[0] > 0:
        avg_segments = idx.cell_segment_indices.shape[0] / max(non_empty_segment_cells, 1)
        print(f"  Avg segments per non-empty cell: {avg_segments:.1f}")

    if idx.cell_point_indices.shape[0] > 0:
        avg_points = idx.cell_point_indices.shape[0] / max(non_empty_point_cells, 1)
        print(f"  Avg points per non-empty cell: {avg_points:.1f}")

    # Memory usage
    segment_mem = (
        idx.cell_segment_indices.numel() + idx.cell_offsets.numel()
    ) * 8  # int64 = 8 bytes
    point_mem = (idx.cell_point_indices.numel() + idx.cell_point_offsets.numel()) * 8
    total_mem = segment_mem + point_mem

    print(f"\nMemory usage:")
    print(f"  Segment index: {segment_mem / 1024**2:.1f} MB")
    print(f"  Point index: {point_mem / 1024**2:.1f} MB")
    print(f"  Total: {total_mem / 1024**2:.1f} MB")


def simulate_query(
    collection: GPUGeometryCollection,
    num_queries: int = 100000,
) -> None:
    """Simulate looking up cells for query points (for profiling)."""
    print("\n" + "=" * 60)
    print(f"SIMULATING: {num_queries:,} query lookups")
    print("=" * 60)

    if collection.spatial_index is None:
        print("No spatial index built!")
        return

    idx = collection.spatial_index

    # Generate random query points within grid bounds
    grid_min = idx.grid_origin
    grid_max = idx.grid_origin + idx.grid_dims.float() * idx.cell_size

    query_points = torch.rand(num_queries, 2, device=collection.device)
    query_points = query_points * (grid_max - grid_min) + grid_min

    # Hash to cells
    start = time.time()
    cell_coords = torch.floor((query_points - idx.grid_origin) / idx.cell_size).long()
    cell_coords = torch.clamp(cell_coords, torch.tensor([0, 0], device=collection.device), idx.grid_dims - 1)
    cell_ids = cell_coords[:, 1] * idx.grid_dims[0] + cell_coords[:, 0]
    elapsed = time.time() - start

    print(f"Cell hash time: {elapsed * 1000:.2f}ms")

    # Gather segment counts per query (CPU operation for now)
    start = time.time()
    segment_counts = idx.cell_offsets[cell_ids + 1] - idx.cell_offsets[cell_ids]
    point_counts = idx.cell_point_offsets[cell_ids + 1] - idx.cell_point_offsets[cell_ids]
    elapsed = time.time() - start

    print(f"Count lookup time: {elapsed * 1000:.2f}ms")

    print(f"\nQuery statistics:")
    print(f"  Avg segments per query: {segment_counts.float().mean():.1f}")
    print(f"  Max segments per query: {segment_counts.max()}")
    print(f"  Avg points per query: {point_counts.float().mean():.1f}")
    print(f"  Max points per query: {point_counts.max()}")

    # Estimate computation reduction
    total_geometries = collection.num_geometries
    avg_geoms_per_query = (segment_counts + point_counts).float().mean()
    reduction_factor = total_geometries / avg_geoms_per_query

    print(f"\nSpatial culling efficiency:")
    print(f"  Total geometries: {total_geometries:,}")
    print(f"  Avg checked per query: {avg_geoms_per_query:.1f}")
    print(f"  Reduction factor: {reduction_factor:.1f}x")


def benchmark_query_distances(
    collection: GPUGeometryCollection,
    num_queries: int = 1000,
) -> None:
    """Benchmark query_distances method with both PyTorch and CUDA."""
    if collection.spatial_index is None:
        print("No spatial index built!")
        return

    idx = collection.spatial_index

    # Generate random query points within grid bounds
    grid_min = idx.grid_origin
    grid_max = idx.grid_origin + idx.grid_dims.float() * idx.cell_size

    query_points = torch.rand(num_queries, 2, device=collection.device)
    query_points = query_points * (grid_max - grid_min) + grid_min

    # Benchmark PyTorch implementation
    print("\n" + "=" * 60)
    print(f"BENCHMARKING: PyTorch Implementation ({num_queries:,} queries)")
    print("=" * 60)

    # Warm-up
    for _ in range(3):
        _ = collection.query_distances(query_points[:10], use_cuda_kernel=False)

    # Benchmark with multiple iterations
    num_iterations = 1
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        result_pytorch = collection.query_distances(query_points, use_cuda_kernel=False)
    torch.cuda.synchronize()
    elapsed_pytorch = time.time() - start
    avg_time = elapsed_pytorch / num_iterations

    print(f"Total time ({num_iterations} iterations): {elapsed_pytorch:.3f}s")
    print(f"Average time per iteration: {avg_time * 1000:.3f}ms")
    print(f"Time per query: {avg_time * 1000000 / num_queries:.3f}µs")
    print(f"Queries per second: {num_queries / avg_time:.1f}")

    print(f"\nResult statistics:")
    print(f"  Total distance pairs: {result_pytorch.shape[0]:,}")
    print(f"  Avg pairs per query: {result_pytorch.shape[0] / num_queries:.1f}")

    # Benchmark CUDA implementation
    if collection.device.type == "cuda":
        print("\n" + "=" * 60)
        print(f"BENCHMARKING: CUDA Kernel ({num_queries:,} queries)")
        print("=" * 60)

        # Warm-up
        for _ in range(3):
            _ = collection.query_distances(query_points[:10], use_cuda_kernel=True)

        # Benchmark with multiple iterations
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            result_cuda = collection.query_distances(query_points, use_cuda_kernel=True)
        torch.cuda.synchronize()
        elapsed_cuda = time.time() - start
        avg_time_cuda = elapsed_cuda / num_iterations

        print(f"Total time ({num_iterations} iterations): {elapsed_cuda:.3f}s")
        print(f"Average time per iteration: {avg_time_cuda * 1000:.3f}ms")
        print(f"Time per query: {avg_time_cuda * 1000000 / num_queries:.3f}µs")
        print(f"Queries per second: {num_queries / avg_time_cuda:.1f}")

        print(f"\nResult statistics:")
        print(f"  Total distance pairs: {result_cuda.shape[0]:,}")
        print(f"  Avg pairs per query: {result_cuda.shape[0] / num_queries:.1f}")

        print(f"\n" + "=" * 60)
        print("SPEEDUP COMPARISON")
        print("=" * 60)
        speedup = avg_time / avg_time_cuda
        print(f"CUDA vs PyTorch: {speedup:.2f}x faster")
        print(f"PyTorch: {avg_time * 1000:.3f}ms per iteration")
        print(f"CUDA:    {avg_time_cuda * 1000:.3f}ms per iteration")

    # Analyze result distribution
    result = result_pytorch
    if result.shape[0] > 0:
        distances = result[:, 2]
        print(f"\nDistance statistics:")
        print(f"  Min distance: {distances.min():.2f}")
        print(f"  Max distance: {distances.max():.2f}")
        print(f"  Mean distance: {distances.mean():.2f}")
        print(f"  Median distance: {distances.median():.2f}")


def main():
    # Configuration
    feather_path = Path("/data/overhead_matching/datasets/VIGOR/Chicago/landmarks/v4_202001.feather")
    zoom_level = 20  # VIGOR uses zoom 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Typical parameters from observation likelihood
    sigma_px = 200.0  # Typical sigma value
    cell_size = sigma_px  # Grid cell size
    expansion_distance = 5.0 * sigma_px  # 5-sigma threshold

    print("=" * 60)
    print("VIGOR Landmark Spatial Index Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Landmark file: {feather_path}")
    print(f"Zoom level: {zoom_level}")

    # Check if file exists
    if not feather_path.exists():
        print(f"\nERROR: Landmark file not found: {feather_path}")
        print("Please ensure VIGOR dataset is downloaded.")
        return

    # Load landmarks
    df = load_vigor_landmarks(feather_path, zoom_level)
    analyze_landmark_types(df)

    # Benchmark conversion
    collection = benchmark_conversion(df, device)

    # Benchmark spatial index construction
    benchmark_spatial_index(collection, cell_size, expansion_distance)

    # Simulate queries
    simulate_query(collection, num_queries=100000)

    # Benchmark query_distances
    benchmark_query_distances(collection, num_queries=10000)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
