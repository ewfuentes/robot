"""Profile CUDA kernel and PyTorch implementation using PyTorch profiler."""

import time
from pathlib import Path

import common.torch.load_torch_deps  # Must be before torch  # noqa: F401
import geopandas as gpd
import numpy as np
import shapely
import torch
from torch.profiler import profile, ProfilerActivity, record_function

from common.geometry.gpu_geometry_collection import GPUGeometryCollection
from common.gps import web_mercator


def convert_geometry_to_pixels(geometry, zoom_level: int):
    """Convert lat/lon geometry to web mercator pixels."""
    def coord_transform(lon, lat):
        x_px, y_px = web_mercator.latlon_to_pixel_coords(lat, lon, zoom_level)
        return x_px, y_px
    return shapely.ops.transform(coord_transform, geometry)


def verify_with_shapely(
    query_points: torch.Tensor,
    cuda_results: torch.Tensor,
    geometries: list,
    tolerance: float = 1e-3,
) -> dict:
    """Verify all CUDA results against shapely.

    Args:
        query_points: (N, 2) tensor of query positions
        cuda_results: (K, 3) tensor of [particle_idx, geometry_idx, distance]
        geometries: List of shapely geometries
        tolerance: Relative tolerance for distance comparison

    Returns:
        Dict with verification results
    """
    query_np = query_points.cpu().numpy()
    results_np = cuda_results.cpu().numpy()

    errors = []
    checked = 0
    matched = 0

    for i in range(len(results_np)):
        p_idx = int(results_np[i, 0])
        g_idx = int(results_np[i, 1])
        cuda_dist = results_np[i, 2]

        if g_idx >= len(geometries):
            continue

        point = shapely.Point(query_np[p_idx])
        geom = geometries[g_idx]
        shapely_dist = point.distance(geom)

        # For points inside polygons, CUDA returns 0
        if cuda_dist == 0 and isinstance(geom, (shapely.Polygon, shapely.MultiPolygon)):
            # Check if point is inside
            if geom.contains(point) or point.within(geom):
                matched += 1
                checked += 1
                continue

        # Compare distances
        checked += 1
        if abs(cuda_dist - shapely_dist) <= tolerance * max(1.0, shapely_dist):
            matched += 1
        else:
            errors.append({
                "point_idx": p_idx,
                "geom_idx": g_idx,
                "cuda_dist": cuda_dist,
                "shapely_dist": shapely_dist,
                "diff": abs(cuda_dist - shapely_dist),
            })

    # Sort errors by diff (largest first) and take top 10
    errors.sort(key=lambda e: e["diff"], reverse=True)

    return {
        "checked": checked,
        "matched": matched,
        "match_rate": matched / checked if checked > 0 else 1.0,
        "errors": errors[:10],  # Largest 10 errors
    }


def main():
    # Configuration
    feather_path = Path("/data/overhead_matching/datasets/VIGOR/Chicago/landmarks/v4_202001.feather")
    zoom_level = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sigma_px = 200.0
    cell_size = sigma_px
    expansion_distance = 5.0 * sigma_px

    print("Loading landmarks...")
    df = gpd.read_feather(feather_path)
    df["geometry_px"] = df["geometry"].apply(
        lambda g: convert_geometry_to_pixels(g, zoom_level)
    )

    print("Building geometry collection...")
    collection = GPUGeometryCollection.from_shapely(
        df.geometry_px.values, device=device
    )

    print("Building spatial index...")
    collection.build_spatial_index(cell_size, expansion_distance)

    # Generate query points
    idx = collection.spatial_index
    grid_min = idx.grid_origin
    grid_max = idx.grid_origin + idx.grid_dims.float() * idx.cell_size

    num_queries = 100000
    query_points = torch.rand(num_queries, 2, device=device)
    query_points = query_points * (grid_max - grid_min) + grid_min

    print(f"\nProfiling with {num_queries:,} queries...")
    print("=" * 80)

    # Warm-up
    print("Warming up...")
    _ = collection.query_distances_cuda(query_points[:100])
    torch.cuda.synchronize()

    # Profile CUDA kernel implementation
    print("\n" + "=" * 80)
    print("PROFILING: CUDA Kernel Implementation")
    print("=" * 80)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("cuda_query_distances"):
            for _ in range(10):
                result_cuda = collection.query_distances_cuda(query_points)

    torch.cuda.synchronize()

    # Export chrome trace
    trace_file = "/tmp/cuda_trace.json"
    prof.export_chrome_trace(trace_file)
    print(f"\nExported chrome trace to: {trace_file}")

    print("\nTop operations by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("\nTop operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"CUDA result pairs: {result_cuda.shape[0]:,}")

    # Verify correctness against shapely
    print("\n" + "=" * 80)
    print("CORRECTNESS VERIFICATION (vs Shapely)")
    print("=" * 80)
    print(f"Verifying all {result_cuda.shape[0]:,} result pairs against shapely...")

    geometries_px = list(df["geometry_px"].values)
    verification = verify_with_shapely(
        query_points, result_cuda, geometries_px, tolerance=1e-2
    )

    print(f"Checked: {verification['checked']:,} (point, geometry) pairs")
    print(f"Matched: {verification['matched']:,}")
    print(f"Match rate: {verification['match_rate']:.2%}")

    if verification["errors"]:
        print(f"\nLargest {len(verification['errors'])} errors:")
        for err in verification["errors"]:
            print(f"  Point {err['point_idx']}, Geom {err['geom_idx']}: "
                  f"CUDA={err['cuda_dist']:.6f}, Shapely={err['shapely_dist']:.6f}, "
                  f"diff={err['diff']:.6f}")


if __name__ == "__main__":
    main()
