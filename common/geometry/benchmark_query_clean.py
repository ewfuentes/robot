"""Clean benchmark for query_distances with proper CUDA synchronization."""

import time
from pathlib import Path
import sys

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


def benchmark_implementation(collection, query_points, use_cuda_kernel, num_iterations=100):
    """Benchmark a single implementation with proper synchronization."""

    # Warm-up: run a few times to ensure kernels are compiled
    print(f"  Warming up ({use_cuda_kernel=})...")
    for _ in range(5):
        if use_cuda_kernel:
            _ = collection.query_distances_cuda(query_points[:100])
        else:
            _ = collection.query_distances(query_points[:100])
        torch.cuda.synchronize()

    # Clear any pending operations
    torch.cuda.synchronize()

    # Benchmark
    print(f"  Running {num_iterations} iterations...")
    times = []
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        if use_cuda_kernel:
            result = collection.query_distances_cuda(query_points)
        else:
            result = collection.query_distances(query_points)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if (i + 1) % 20 == 0:
            print(f"    Completed {i + 1}/{num_iterations} iterations")

    # Statistics
    times_ms = [t * 1000 for t in times]
    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    # Remove outliers (top/bottom 10%) and recalculate
    times_ms_sorted = sorted(times_ms)
    trimmed = times_ms_sorted[len(times_ms_sorted)//10 : -len(times_ms_sorted)//10]
    trimmed_avg_ms = sum(trimmed) / len(trimmed) if trimmed else avg_ms

    return {
        'result': result,
        'avg_ms': avg_ms,
        'min_ms': min_ms,
        'max_ms': max_ms,
        'trimmed_avg_ms': trimmed_avg_ms,
        'times_ms': times_ms
    }


def analyze_distribution(collection):
    """Analyze geometry and segment distribution per cell."""
    idx = collection.spatial_index

    print("=" * 80)
    print("GEOMETRY DISTRIBUTION PER CELL")
    print("=" * 80)

    geom_counts = idx.cell_geom_offsets[1:] - idx.cell_geom_offsets[:-1]
    non_empty = geom_counts > 0

    print(f"\nGeometries per cell (non-empty cells only):")
    print(f"  Total non-empty cells: {non_empty.sum().item():,}")
    print(f"  Max geoms/cell: {geom_counts.max().item()}")
    print(f"  Mean geoms/cell: {geom_counts[non_empty].float().mean().item():.1f}")
    print(f"  Median geoms/cell: {geom_counts[non_empty].float().median().item():.1f}")
    print(f"  95th percentile: {geom_counts[non_empty].float().quantile(0.95).item():.1f}")
    print(f"  99th percentile: {geom_counts[non_empty].float().quantile(0.99).item():.1f}")
    print(f"  Cells with >64 geoms: {(geom_counts > 64).sum().item():,}")
    print(f"  Cells with >128 geoms: {(geom_counts > 128).sum().item():,}")
    print(f"  Cells with >256 geoms: {(geom_counts > 256).sum().item():,}")

    print("\n" + "=" * 80)
    print("SEGMENT DISTRIBUTION PER CELL")
    print("=" * 80)

    seg_counts = idx.cell_offsets[1:] - idx.cell_offsets[:-1]
    non_empty_seg = seg_counts > 0

    print(f"\nSegments per cell (non-empty cells only):")
    print(f"  Total non-empty cells: {non_empty_seg.sum().item():,}")
    print(f"  Max segs/cell: {seg_counts.max().item()}")
    print(f"  Mean segs/cell: {seg_counts[non_empty_seg].float().mean().item():.1f}")
    print(f"  Median segs/cell: {seg_counts[non_empty_seg].float().median().item():.1f}")
    print(f"  95th percentile: {seg_counts[non_empty_seg].float().quantile(0.95).item():.1f}")
    print(f"  99th percentile: {seg_counts[non_empty_seg].float().quantile(0.99).item():.1f}")
    print(f"  Cells with >256 segs: {(seg_counts > 256).sum().item():,}")
    print(f"  Cells with >512 segs: {(seg_counts > 512).sum().item():,}")
    print(f"  Cells with >1024 segs: {(seg_counts > 1024).sum().item():,}")

    print("\n" + "=" * 80)
    print("POINT GEOMETRY DISTRIBUTION PER CELL")
    print("=" * 80)

    pt_geom_counts = idx.cell_point_geom_offsets[1:] - idx.cell_point_geom_offsets[:-1]
    non_empty_pt = pt_geom_counts > 0

    print(f"\nPoint geometries per cell (non-empty cells only):")
    print(f"  Total non-empty cells: {non_empty_pt.sum().item():,}")
    print(f"  Max pt_geoms/cell: {pt_geom_counts.max().item()}")
    print(f"  Cells with >128 pt_geoms: {(pt_geom_counts > 128).sum().item():,}")


def main():
    # Parse command line argument for which implementation to test
    if len(sys.argv) < 2:
        print("Usage: benchmark_query_clean.py [pytorch|cuda|both|analyze]")
        sys.exit(1)

    mode = sys.argv[1]

    # Configuration
    feather_path = Path("/data/overhead_matching/datasets/VIGOR/Chicago/landmarks/v4_202001.feather")
    zoom_level = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sigma_px = 200.0
    cell_size = sigma_px
    expansion_distance = 5.0 * sigma_px

    num_queries = 100000
    num_iterations = 1

    print("=" * 80)
    print(f"CLEAN BENCHMARK - Mode: {mode}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Queries: {num_queries:,}")
    print(f"Iterations: {num_iterations}")
    print()

    # Load data
    print("Loading landmarks...")
    if not feather_path.exists():
        print(f"ERROR: Landmark file not found: {feather_path}")
        sys.exit(1)

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

    # If analyze mode, just print distribution and exit
    if mode == 'analyze':
        analyze_distribution(collection)
        return

    # Generate query points
    idx = collection.spatial_index
    grid_min = idx.grid_origin
    grid_max = idx.grid_origin + idx.grid_dims.float() * idx.cell_size

    query_points = torch.rand(num_queries, 2, device=device)
    query_points = query_points * (grid_max - grid_min) + grid_min

    print()

    # Benchmark requested implementation(s)
    results = {}

    if mode in ['pytorch', 'both']:
        print("=" * 80)
        print("BENCHMARKING: PyTorch Implementation")
        print("=" * 80)
        results['pytorch'] = benchmark_implementation(
            collection, query_points, use_cuda_kernel=False, num_iterations=num_iterations
        )
        print()
        print(f"Results:")
        print(f"  Average time: {results['pytorch']['avg_ms']:.3f}ms")
        print(f"  Trimmed average (remove outliers): {results['pytorch']['trimmed_avg_ms']:.3f}ms")
        print(f"  Min time: {results['pytorch']['min_ms']:.3f}ms")
        print(f"  Max time: {results['pytorch']['max_ms']:.3f}ms")
        print(f"  Time per query: {results['pytorch']['trimmed_avg_ms'] * 1000 / num_queries:.3f}µs")
        print(f"  Result pairs: {results['pytorch']['result'].shape[0]:,}")
        print()

    if mode in ['cuda', 'both']:
        print("=" * 80)
        print("BENCHMARKING: CUDA Kernel Implementation")
        print("=" * 80)
        results['cuda'] = benchmark_implementation(
            collection, query_points, use_cuda_kernel=True, num_iterations=num_iterations
        )
        print()
        print(f"Results:")
        print(f"  Average time: {results['cuda']['avg_ms']:.3f}ms")
        print(f"  Trimmed average (remove outliers): {results['cuda']['trimmed_avg_ms']:.3f}ms")
        print(f"  Min time: {results['cuda']['min_ms']:.3f}ms")
        print(f"  Max time: {results['cuda']['max_ms']:.3f}ms")
        print(f"  Time per query: {results['cuda']['trimmed_avg_ms'] * 1000 / num_queries:.3f}µs")
        print(f"  Result pairs: {results['cuda']['result'].shape[0]:,}")
        print()

    # Comparison
    if mode == 'both':
        print("=" * 80)
        print("COMPARISON")
        print("=" * 80)
        speedup = results['pytorch']['trimmed_avg_ms'] / results['cuda']['trimmed_avg_ms']
        print(f"CUDA vs PyTorch speedup: {speedup:.3f}x")
        print(f"PyTorch: {results['pytorch']['trimmed_avg_ms']:.3f}ms")
        print(f"CUDA:    {results['cuda']['trimmed_avg_ms']:.3f}ms")

        # Check if results match
        if results['pytorch']['result'].shape == results['cuda']['result'].shape:
            print(f"\nResult shapes match: {results['pytorch']['result'].shape}")
        else:
            print(f"\nWARNING: Result shapes differ!")
            print(f"  PyTorch: {results['pytorch']['result'].shape}")
            print(f"  CUDA:    {results['cuda']['result'].shape}")

    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
