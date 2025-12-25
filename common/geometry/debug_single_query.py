"""Debug script for testing query_distances_cuda with a single geometry and query point.

Usage: # Load geometry by index from dataset and specify query point:
    bazel run //common/geometry:debug_single_query -- \
        --geom-idx 57446 --query-x 9975200 --query-y 6889500

    # Or use the top error from a correctness run:
    bazel run //common/geometry:debug_single_query -- --from-error 0

    # Show all segments in the geometry:
    bazel run //common/geometry:debug_single_query -- --geom-idx 57446 --show-segments
"""

import argparse
from pathlib import Path

import common.torch.load_torch_deps  # Must be before torch  # noqa: F401
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
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


def load_dataset(feather_path: Path, zoom_level: int):
    """Load landmarks dataset and convert to pixels."""
    print("Loading landmarks...")
    df = gpd.read_feather(feather_path)
    df["geometry_px"] = df["geometry"].apply(
        lambda g: convert_geometry_to_pixels(g, zoom_level)
    )
    return df


def debug_single_geometry(
    geom: shapely.Geometry,
    query_point: np.ndarray,
    device: torch.device,
    show_segments: bool = False,
    cuda_debug: bool = False,
):
    """Debug distance computation for a single geometry and query point."""
    print("\n" + "=" * 80)
    print("GEOMETRY INFO")
    print("=" * 80)
    print(f"Type: {geom.geom_type}")
    print(f"Bounds: {geom.bounds}")

    if isinstance(geom, shapely.Polygon):
        print(f"Exterior points: {len(geom.exterior.coords)}")
        print(f"Holes: {len(geom.interiors)}")
        for i, interior in enumerate(geom.interiors):
            print(f"  Hole {i}: {len(interior.coords)} points")
    elif isinstance(geom, shapely.LineString):
        print(f"Points: {len(geom.coords)}")
    elif isinstance(geom, shapely.MultiPolygon):
        print(f"Polygons: {len(geom.geoms)}")
        for i, g in enumerate(geom.geoms):
            print(f"Polygon {i}:")
            print(f"\tExterior points: {len(g.exterior.coords)}")
            print(f"\tHoles: {len(g.interiors)}")
            for i, interior in enumerate(g.interiors):
                print(f"\t  Hole {i}: {len(interior.coords)} points")


    print("\n" + "=" * 80)
    print("QUERY POINT")
    print("=" * 80)
    print(f"Coordinates: ({query_point[0]:.2f}, {query_point[1]:.2f})")

    # Check point-in-polygon
    point = shapely.Point(query_point)
    if isinstance(geom, (shapely.Polygon, shapely.MultiPolygon)):
        inside = geom.contains(point)
        print(f"Inside polygon: {inside}")

    # Shapely distance
    shapely_dist = point.distance(geom)
    print(f"Shapely distance: {shapely_dist:.6f}")

    # Build GPU collection with single geometry
    print("\n" + "=" * 80)
    print("CUDA COMPUTATION")
    print("=" * 80)

    collection = GPUGeometryCollection.from_shapely([geom], device=device)

    # Print segment info
    print(f"Segments: {collection.segment_starts.shape[0]}")
    print(f"Points: {collection.point_coords.shape[0]}")

    if show_segments and collection.segment_starts.shape[0] > 0:
        print("\nSegment details:")
        starts = collection.segment_starts.cpu().numpy()
        ends = collection.segment_ends.cpu().numpy()
        for i in range(min(20, len(starts))):
            print(f"  Seg {i}: ({starts[i, 0]:.2f}, {starts[i, 1]:.2f}) -> "
                  f"({ends[i, 0]:.2f}, {ends[i, 1]:.2f})")
        if len(starts) > 20:
            print(f"  ... and {len(starts) - 20} more segments")

    # Build spatial index
    cell_size = 200.0
    expansion_distance = 1000.0
    collection.build_spatial_index(cell_size, expansion_distance)

    # Print spatial index info
    idx = collection.spatial_index
    print(f"\nSpatial index info:")
    print(f"  Grid origin: ({idx.grid_origin[0].item():.2f}, {idx.grid_origin[1].item():.2f})")
    print(f"  Grid dims: ({idx.grid_dims[0].item()}, {idx.grid_dims[1].item()})")
    print(f"  Cell size: {idx.cell_size}")
    print(f"  Total segments indexed: {idx.cell_segment_indices.shape[0]}")
    print(f"  Total cells with segments: {(idx.cell_offsets[1:] - idx.cell_offsets[:-1] > 0).sum().item()}")

    # Compute which cell the query point falls into
    query_cell = ((torch.tensor(query_point, device=device) - idx.grid_origin) / cell_size).floor().long()
    cell_id = query_cell[1] * idx.grid_dims[0] + query_cell[0]
    print(f"\nQuery point cell:")
    print(f"  Cell coords: ({query_cell[0].item()}, {query_cell[1].item()})")
    print(f"  Cell ID: {cell_id.item()}")

    # Check segments in this cell
    cell_start = idx.cell_offsets[cell_id].item()
    cell_end = idx.cell_offsets[cell_id + 1].item()
    print(f"  Segments in cell: {cell_end - cell_start}")

    if cell_end > cell_start:
        print(f"\n  Segments indexed to this cell:")
        seg_indices = idx.cell_segment_indices[cell_start:cell_end]
        starts = collection.segment_starts.cpu().numpy()
        ends = collection.segment_ends.cpu().numpy()
        for i, seg_idx in enumerate(seg_indices.cpu().numpy()):
            sx1, sy1 = starts[seg_idx]
            sx2, sy2 = ends[seg_idx]
            # Compute distance from query point to this segment
            seg_dist = shapely.Point(query_point).distance(
                shapely.LineString([(sx1, sy1), (sx2, sy2)])
            )
            print(f"    Seg {seg_idx}: ({sx1:.1f},{sy1:.1f})->({sx2:.1f},{sy2:.1f}), dist={seg_dist:.1f}")

            # Check if this segment should be in this cell
            seg_bbox_min = (min(sx1, sx2) - expansion_distance, min(sy1, sy2) - expansion_distance)
            seg_bbox_max = (max(sx1, sx2) + expansion_distance, max(sy1, sy2) + expansion_distance)
            cell_min = (idx.grid_origin[0].item() + query_cell[0].item() * cell_size,
                       idx.grid_origin[1].item() + query_cell[1].item() * cell_size)
            cell_max = (cell_min[0] + cell_size, cell_min[1] + cell_size)
            overlaps = (seg_bbox_max[0] >= cell_min[0] and seg_bbox_min[0] <= cell_max[0] and
                       seg_bbox_max[1] >= cell_min[1] and seg_bbox_min[1] <= cell_max[1])
            print(f"      Seg bbox: ({seg_bbox_min[0]:.1f},{seg_bbox_min[1]:.1f})->({seg_bbox_max[0]:.1f},{seg_bbox_max[1]:.1f})")
            print(f"      Cell bbox: ({cell_min[0]:.1f},{cell_min[1]:.1f})->({cell_max[0]:.1f},{cell_max[1]:.1f})")
            print(f"      Overlaps: {overlaps}")

    # Find the segment that shapely says is closest
    print(f"\n  Finding closest segment via brute force:")
    min_dist = float('inf')
    closest_seg_idx = -1
    query_pt = shapely.Point(query_point)
    for seg_idx in range(len(starts)):
        sx1, sy1 = starts[seg_idx]
        sx2, sy2 = ends[seg_idx]
        seg_line = shapely.LineString([(sx1, sy1), (sx2, sy2)])
        dist = query_pt.distance(seg_line)
        if dist < min_dist:
            min_dist = dist
            closest_seg_idx = seg_idx

    if closest_seg_idx >= 0:
        sx1, sy1 = starts[closest_seg_idx]
        sx2, sy2 = ends[closest_seg_idx]
        print(f"    Closest seg {closest_seg_idx}: ({sx1:.1f},{sy1:.1f})->({sx2:.1f},{sy2:.1f}), dist={min_dist:.1f}")
        seg_bbox_min = (min(sx1, sx2) - expansion_distance, min(sy1, sy2) - expansion_distance)
        seg_bbox_max = (max(sx1, sx2) + expansion_distance, max(sy1, sy2) + expansion_distance)
        cell_min = (idx.grid_origin[0].item() + query_cell[0].item() * cell_size,
                   idx.grid_origin[1].item() + query_cell[1].item() * cell_size)
        cell_max = (cell_min[0] + cell_size, cell_min[1] + cell_size)
        overlaps = (seg_bbox_max[0] >= cell_min[0] and seg_bbox_min[0] <= cell_max[0] and
                   seg_bbox_max[1] >= cell_min[1] and seg_bbox_min[1] <= cell_max[1])
        print(f"      Seg bbox: ({seg_bbox_min[0]:.1f},{seg_bbox_min[1]:.1f})->({seg_bbox_max[0]:.1f},{seg_bbox_max[1]:.1f})")
        print(f"      Cell bbox: ({cell_min[0]:.1f},{cell_min[1]:.1f})->({cell_max[0]:.1f},{cell_max[1]:.1f})")
        print(f"      Overlaps: {overlaps}")

    # Run CUDA query
    query_tensor = torch.tensor([query_point], dtype=torch.float32, device=device)
    result = collection.query_distances_cuda(query_tensor, debug=cuda_debug)

    print(f"\nCUDA result shape: {result.shape}")
    if result.shape[0] > 0:
        result_np = result.cpu().numpy()
        for i in range(result.shape[0]):
            p_idx = int(result_np[i, 0])
            g_idx = int(result_np[i, 1])
            cuda_dist = result_np[i, 2]
            print(f"  Result {i}: point_idx={p_idx}, geom_idx={g_idx}, distance={cuda_dist:.6f}")

        cuda_dist = result_np[0, 2]
        diff = abs(cuda_dist - shapely_dist)
        print(f"\nComparison:")
        print(f"  CUDA distance:   {cuda_dist:.6f}")
        print(f"  Shapely distance: {shapely_dist:.6f}")
        print(f"  Difference:       {diff:.6f}")
        if shapely_dist > 0:
            print(f"  Relative error:   {diff / shapely_dist * 100:.2f}%")
    else:
        print("  No results returned from CUDA!")
        print("  (Query point may be outside spatial index bounds)")

    return collection, geom, query_point, shapely_dist, result[0, 2].item()


def visualize_debug(
    geom: shapely.Geometry,
    query_point: np.ndarray,
    collection: GPUGeometryCollection,
    shapely_dist: float,
    cuda_dist: float,
    output_path: str,
):
    """Visualize the geometry and query point."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot geometry
    if isinstance(geom, shapely.Point):
        ax.plot(geom.x, geom.y, 'go', markersize=10, label='Geometry (Point)')
    elif isinstance(geom, shapely.LineString):
        coords = np.array(geom.coords)
        ax.plot(coords[:, 0], coords[:, 1], 'g-', linewidth=2, label='Geometry (LineString)')
        ax.plot(coords[:, 0], coords[:, 1], 'g.', markersize=4)
    elif isinstance(geom, shapely.Polygon):
        coords = np.array(geom.exterior.coords)
        ax.fill(coords[:, 0], coords[:, 1], alpha=0.3, color='green')
        ax.plot(coords[:, 0], coords[:, 1], 'g-', linewidth=2, label='Geometry (Polygon)')
        ax.plot(coords[:, 0], coords[:, 1], 'g.', markersize=4)
        for interior in geom.interiors:
            coords = np.array(interior.coords)
            ax.fill(coords[:, 0], coords[:, 1], alpha=1.0, color='white')
            ax.plot(coords[:, 0], coords[:, 1], 'g--', linewidth=1)
            ax.plot(coords[:, 0], coords[:, 1], 'g.', markersize=4)
    elif isinstance(geom, shapely.MultiPolygon):
        for poly in geom.geoms:
            coords = np.array(poly.exterior.coords)
            ax.fill(coords[:, 0], coords[:, 1], alpha=0.3, color='green')
            ax.plot(coords[:, 0], coords[:, 1], 'g-', linewidth=2)
            ax.plot(coords[:, 0], coords[:, 1], 'g.', markersize=4)

    # Plot segments from collection
    if collection.segment_starts.shape[0] > 0:
        starts = collection.segment_starts.cpu().numpy()
        ends = collection.segment_ends.cpu().numpy()
        for i in range(len(starts)):
            ax.plot([starts[i, 0], ends[i, 0]], [starts[i, 1], ends[i, 1]],
                    'b-', linewidth=0.5, alpha=0.5)

    # Plot query point
    ax.plot(query_point[0], query_point[1], 'ro', markersize=12, label='Query point')

    # Draw shapely distance circle
    if shapely_dist > 0:
        circle = plt.Circle(
            query_point, shapely_dist, fill=False, edgecolor='orange',
            linestyle='--', linewidth=2, label=f'Shapely dist: {shapely_dist:.2f}'
        )
        ax.add_patch(circle)

    if cuda_dist > 0:
        circle = plt.Circle(
            query_point, cuda_dist, fill=False, edgecolor='blue',
            linestyle='--', linewidth=2, label=f'Cuda Dist dist: {cuda_dist:.2f}'
        )
        ax.add_patch(circle)

    # Draw grid cells near query point
    cell_size = 200.0
    expansion_distance = 1000.0
    cell_x = int(query_point[0] / cell_size) * cell_size
    cell_y = int(query_point[1] / cell_size) * cell_size

    cell_rect = plt.Rectangle(
        (cell_x, cell_y), cell_size, cell_size,
        fill=False, edgecolor='blue', linestyle='--', linewidth=1.5, label='Cell'
    )
    ax.add_patch(cell_rect)

    expanded_rect = plt.Rectangle(
        (cell_x - expansion_distance, cell_y - expansion_distance),
        cell_size + 2 * expansion_distance, cell_size + 2 * expansion_distance,
        fill=False, edgecolor='cyan', linestyle=':', linewidth=1.5, label='Cell + expansion'
    )
    ax.add_patch(expanded_rect)

    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(f"Debug: {geom.geom_type}")
    ax.grid(True, alpha=0.3)

    # Auto-scale
    bounds = geom.bounds
    margin = max(expansion_distance, shapely_dist, 500) * 1.2
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    half_size = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 2 + margin
    ax.set_xlim(center_x - half_size, center_x + half_size)
    ax.set_ylim(center_y - half_size, center_y + half_size)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved visualization to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Debug single geometry distance query")
    parser.add_argument("--geom-idx", type=int, help="Geometry index from dataset")
    parser.add_argument("--query-x", type=float, help="Query point X coordinate")
    parser.add_argument("--query-y", type=float, help="Query point Y coordinate")
    parser.add_argument("--from-error", type=int, help="Load from error index (runs correctness first)")
    parser.add_argument("--show-segments", action="store_true", help="Print segment coordinates")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--debug", action="store_true", help="Enable CUDA kernel debug output")
    args = parser.parse_args()

    feather_path = Path("/data/overhead_matching/datasets/VIGOR/Chicago/landmarks/v4_202001.feather")
    zoom_level = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_dataset(feather_path, zoom_level)
    geometries_px = list(df["geometry_px"].values)

    if args.from_error is not None:
        # Run a quick correctness check to find error cases
        print("Running correctness check to find error cases...")

        torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed(42)

        full_collection = GPUGeometryCollection.from_shapely(geometries_px, device=device)
        full_collection.build_spatial_index(200.0, 1000.0)

        idx = full_collection.spatial_index
        grid_min = idx.grid_origin
        grid_max = idx.grid_origin + idx.grid_dims.float() * idx.cell_size
        query_points = torch.rand(10000, 2, device=device)
        query_points = query_points * (grid_max - grid_min) + grid_min

        result_cuda = full_collection.query_distances_cuda(query_points)
        query_np = query_points.cpu().numpy()
        results_np = result_cuda.cpu().numpy()

        # Find errors
        errors = []
        for i in range(len(results_np)):
            p_idx = int(results_np[i, 0])
            g_idx = int(results_np[i, 1])
            cuda_dist = results_np[i, 2]

            point = shapely.Point(query_np[p_idx])
            geom = geometries_px[g_idx]
            shapely_dist = point.distance(geom)

            diff = abs(cuda_dist - shapely_dist)
            if diff > 1e-5 * max(1.0, shapely_dist):
                errors.append({
                    "point_idx": p_idx,
                    "geom_idx": g_idx,
                    "cuda_dist": cuda_dist,
                    "shapely_dist": shapely_dist,
                    "diff": diff,
                    "point_coords": query_np[p_idx],
                })

        errors.sort(key=lambda e: e["diff"], reverse=True)

        if args.from_error >= len(errors):
            print(f"Error index {args.from_error} out of range (only {len(errors)} errors)")
            return

        err = errors[args.from_error]
        geom_idx = err["geom_idx"]
        query_point = err["point_coords"]
        print(f"\nUsing error #{args.from_error}:")
        print(f"  Geometry index: {geom_idx}")
        print(f"  Query point: ({query_point[0]:.2f}, {query_point[1]:.2f})")
        print(f"  Original diff: {err['diff']:.6f}")

    elif args.geom_idx is not None and args.query_x is not None and args.query_y is not None:
        geom_idx = args.geom_idx
        query_point = np.array([args.query_x, args.query_y], dtype=np.float32)
    else:
        parser.error("Must specify either --from-error OR (--geom-idx, --query-x, --query-y)")

    if geom_idx >= len(geometries_px):
        print(f"Geometry index {geom_idx} out of range (only {len(geometries_px)} geometries)")
        return

    geom = geometries_px[geom_idx]

    collection, geom, query_point, shapely_dist, result = debug_single_geometry(
        geom, query_point, device, show_segments=args.show_segments, cuda_debug=args.debug
    )

    if not args.no_viz:
        visualize_debug(geom, query_point, collection, shapely_dist, result, "/tmp/debug_single_query.png")


if __name__ == "__main__":
    main()
