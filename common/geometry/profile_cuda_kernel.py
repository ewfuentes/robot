"""Minimal script to profile just the CUDA kernel."""

import time
from pathlib import Path

import common.torch.load_torch_deps  # Must be before torch  # noqa: F401
import geopandas as gpd
import shapely
import torch

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

    print(f"\nRunning CUDA kernel with {num_queries:,} queries...")
    print("=" * 60)

    # Warm-up
    _ = collection.query_distances(query_points[:100], use_cuda_kernel=True)

    # Profile this section
    torch.cuda.synchronize()
    start = time.time()

    result = collection.query_distances(query_points, use_cuda_kernel=True)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"Total time: {elapsed:.3f}s")
    print(f"Time per query: {elapsed * 1000 / num_queries:.3f}ms")
    print(f"Result pairs: {result.shape[0]:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
