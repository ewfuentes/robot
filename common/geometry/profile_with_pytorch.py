"""Profile CUDA kernel or verify correctness against shapely.

Usage:
    bazel run //common/geometry:profile_with_pytorch -- --mode profile
    bazel run //common/geometry:profile_with_pytorch -- --mode correctness
"""

import argparse
from pathlib import Path

import common.torch.load_torch_deps  # Must be before torch  # noqa: F401
import geopandas as gpd
import matplotlib.pyplot as plt
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

        point_coords = query_np[p_idx]
        point = shapely.Point(point_coords)
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
                "point_coords": point_coords,
                "geometry": geom,
            })

    # Sort errors by diff (largest first) and take top 10
    errors.sort(key=lambda e: e["diff"], reverse=True)

    return {
        "checked": checked,
        "matched": matched,
        "match_rate": matched / checked if checked > 0 else 1.0,
        "errors": errors[:10],  # Largest 10 errors
    }


def visualize_errors(errors: list, cell_size: float, expansion_distance: float, output_path: str):
    """Visualize top 3 errors showing query point, geometry, and cell bounds.

    Args:
        errors: List of error dicts with point_coords, geometry, cuda_dist, shapely_dist
        cell_size: Grid cell size in pixels
        expansion_distance: Expansion distance for spatial index
        output_path: Path to save the output image
    """
    num_errors = min(3, len(errors))
    if num_errors == 0:
        print("No errors to visualize")
        return

    fig, axes = plt.subplots(1, num_errors, figsize=(6 * num_errors, 6))
    if num_errors == 1:
        axes = [axes]

    for i, (ax, err) in enumerate(zip(axes, errors[:num_errors])):
        point_coords = err["point_coords"]
        geom = err["geometry"]
        cuda_dist = err["cuda_dist"]
        shapely_dist = err["shapely_dist"]

        # Compute cell bounds for the query point
        cell_x = int(point_coords[0] / cell_size) * cell_size
        cell_y = int(point_coords[1] / cell_size) * cell_size

        # Plot cell boundary (blue dashed)
        cell_rect = plt.Rectangle(
            (cell_x, cell_y), cell_size, cell_size,
            fill=False, edgecolor='blue', linestyle='--', linewidth=1.5, label='Cell'
        )
        ax.add_patch(cell_rect)

        # Plot expanded cell boundary (cyan dotted)
        expanded_rect = plt.Rectangle(
            (cell_x - expansion_distance, cell_y - expansion_distance),
            cell_size + 2 * expansion_distance, cell_size + 2 * expansion_distance,
            fill=False, edgecolor='cyan', linestyle=':', linewidth=1.5, label='Cell + expansion'
        )
        ax.add_patch(expanded_rect)

        # Plot geometry
        if isinstance(geom, shapely.Point):
            ax.plot(geom.x, geom.y, 'go', markersize=10, label='Geometry (Point)')
        elif isinstance(geom, shapely.LineString):
            coords = np.array(geom.coords)
            ax.plot(coords[:, 0], coords[:, 1], 'g-', linewidth=2, label='Geometry (LineString)')
        elif isinstance(geom, shapely.Polygon):
            # Plot exterior
            coords = np.array(geom.exterior.coords)
            ax.fill(coords[:, 0], coords[:, 1], alpha=0.3, color='green')
            ax.plot(coords[:, 0], coords[:, 1], 'g-', linewidth=2, label='Geometry (Polygon)')
            # Plot holes
            for interior in geom.interiors:
                coords = np.array(interior.coords)
                ax.fill(coords[:, 0], coords[:, 1], alpha=1.0, color='white')
                ax.plot(coords[:, 0], coords[:, 1], 'g--', linewidth=1)
        elif isinstance(geom, shapely.MultiPolygon):
            for poly in geom.geoms:
                coords = np.array(poly.exterior.coords)
                ax.fill(coords[:, 0], coords[:, 1], alpha=0.3, color='green')
                ax.plot(coords[:, 0], coords[:, 1], 'g-', linewidth=2)
                for interior in poly.interiors:
                    coords = np.array(interior.coords)
                    ax.fill(coords[:, 0], coords[:, 1], alpha=1.0, color='white')
                    ax.plot(coords[:, 0], coords[:, 1], 'g--', linewidth=1)

        # Plot query point
        ax.plot(point_coords[0], point_coords[1], 'ro', markersize=10, label='Query point')

        # Draw circles showing CUDA and shapely distances
        if cuda_dist > 0:
            cuda_circle = plt.Circle(
                point_coords, cuda_dist, fill=False, edgecolor='red',
                linestyle='-', linewidth=1.5, label=f'CUDA dist: {cuda_dist:.2f}'
            )
            ax.add_patch(cuda_circle)
        if shapely_dist > 0:
            shapely_circle = plt.Circle(
                point_coords, shapely_dist, fill=False, edgecolor='orange',
                linestyle='--', linewidth=1.5, label=f'Shapely dist: {shapely_dist:.2f}'
            )
            ax.add_patch(shapely_circle)

        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f"Error #{i+1}: diff={err['diff']:.2f}\n"
                     f"Point {err['point_idx']}, Geom {err['geom_idx']}")

        # Auto-scale to show relevant area
        margin = max(expansion_distance, cuda_dist, shapely_dist) * 1.2
        ax.set_xlim(point_coords[0] - margin, point_coords[0] + margin)
        ax.set_ylim(point_coords[1] - margin, point_coords[1] + margin)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved error visualization to: {output_path}")
    plt.close()


def load_data_and_build_collection(feather_path: Path, zoom_level: int, device: torch.device):
    """Load landmarks and build GPU geometry collection.

    Returns:
        Tuple of (dataframe, collection, cell_size, expansion_distance)
    """
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

    return df, collection, cell_size, expansion_distance


def generate_query_points(collection: GPUGeometryCollection, num_queries: int, device: torch.device, seed: int = 42):
    """Generate random query points within the spatial index bounds (world coordinates)."""
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    idx = collection.spatial_index

    # Grid bounds are in local coordinates (relative to coordinate_origin).
    # Convert to world coordinates by adding coordinate_origin.
    grid_min = idx.grid_origin
    grid_max = idx.grid_origin + idx.grid_dims.float() * idx.cell_size

    if collection.coordinate_origin is not None:
        origin = collection.coordinate_origin.to(device=device, dtype=grid_min.dtype)
        grid_min = grid_min + origin
        grid_max = grid_max + origin

    query_points = torch.rand(num_queries, 2, device=device)
    query_points = query_points * (grid_max - grid_min) + grid_min
    return query_points


def run_profiling(collection: GPUGeometryCollection, query_points: torch.Tensor):
    """Run profiling mode with PyTorch profiler."""
    print(f"\nProfiling with {len(query_points):,} queries...")
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


def run_correctness(
    collection: GPUGeometryCollection,
    query_points: torch.Tensor,
    geometries: list,
    cell_size: float,
    expansion_distance: float,
):
    """Run correctness mode comparing CUDA results to shapely."""
    print(f"\nRunning correctness check with {len(query_points):,} queries...")
    print("=" * 80)

    # Warm-up
    _ = collection.query_distances_cuda(query_points[:100])
    torch.cuda.synchronize()

    # Run CUDA
    result_cuda = collection.query_distances_cuda(query_points)
    torch.cuda.synchronize()

    print(f"CUDA result pairs: {result_cuda.shape[0]:,}")
    print(f"  (This is {len(query_points):,} queries x ~{result_cuda.shape[0] / len(query_points):.1f} "
          f"geometries/query on average)")

    # Verify correctness against shapely
    print("\n" + "=" * 80)
    print("CORRECTNESS VERIFICATION (vs Shapely)")
    print("=" * 80)
    print(f"Verifying all {result_cuda.shape[0]:,} result pairs against shapely...")

    verification = verify_with_shapely(
        query_points, result_cuda, geometries, tolerance=1e-2
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

        # Visualize top 3 errors
        visualize_errors(
            verification["errors"],
            cell_size,
            expansion_distance,
            "/tmp/correctness_errors.png"
        )
    else:
        print("\nNo errors found!")


def main():
    parser = argparse.ArgumentParser(description="Profile CUDA kernel or verify correctness")
    parser.add_argument(
        "--mode",
        choices=["profile", "correctness"],
        default="profile",
        help="Mode to run: 'profile' for PyTorch profiler, 'correctness' for shapely comparison"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100000,
        help="Number of query points to generate"
    )
    args = parser.parse_args()

    # Configuration
    feather_path = Path("/data/overhead_matching/datasets/VIGOR/Chicago/landmarks/v4_202001.feather")
    zoom_level = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, collection, cell_size, expansion_distance = load_data_and_build_collection(
        feather_path, zoom_level, device
    )
    query_points = generate_query_points(collection, args.num_queries, device)

    if args.mode == "profile":
        run_profiling(collection, query_points)
    else:
        geometries_px = list(df["geometry_px"].values)
        run_correctness(collection, query_points, geometries_px, cell_size, expansion_distance)


if __name__ == "__main__":
    main()
