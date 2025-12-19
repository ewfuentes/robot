"""Profile CUDA kernel and PyTorch implementation using PyTorch profiler."""

import time
from pathlib import Path

import common.torch.load_torch_deps  # Must be before torch  # noqa: F401
import geopandas as gpd
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

    num_queries = 10000
    query_points = torch.rand(num_queries, 2, device=device)
    query_points = query_points * (grid_max - grid_min) + grid_min

    print(f"\nProfiling with {num_queries:,} queries...")
    print("=" * 80)

    # Warm-up both implementations
    print("Warming up...")
    _ = collection.query_distances(query_points[:100])
    _ = collection.query_distances_cuda(query_points[:100])
    torch.cuda.synchronize()

    # Profile PyTorch implementation
    print("\n" + "=" * 80)
    print("PROFILING: PyTorch Implementation")
    print("=" * 80)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("pytorch_query_distances"):
            result_pytorch = collection.query_distances(query_points)

    torch.cuda.synchronize()

    # Export chrome trace
    trace_file = "/tmp/pytorch_trace.json"
    prof.export_chrome_trace(trace_file)
    print(f"\nExported chrome trace to: {trace_file}")

    print("\nTop operations by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("\nTop operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

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
            for i in range(10):
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
    print(f"PyTorch result pairs: {result_pytorch.shape[0]:,}")
    print(f"CUDA result pairs: {result_cuda.shape[0]:,}")
    print(f"Results match: {torch.allclose(result_pytorch, result_cuda, rtol=1e-4)}")


if __name__ == "__main__":
    main()
