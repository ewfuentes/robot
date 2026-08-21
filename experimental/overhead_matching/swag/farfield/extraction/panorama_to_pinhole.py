"""
Process panoramas to pinhole projections.

Takes a directory of panoramas and projects them to pinhole images with
configurable resolution and FOV. Saves images from 90-degree yaw rotations.
"""

import argparse
import numpy as np
import scipy.ndimage
from pathlib import Path
from PIL import Image
import tqdm
from multiprocessing import Pool
from functools import partial


def spherical_pixel_from_azimuth_elevation(azimuth_rad, elev_rad, image_shape):
    """Map azimuth/elevation to pixel coordinates in equirectangular image."""
    col_frac = (np.pi - azimuth_rad) / (2 * np.pi)
    row_frac = elev_rad / np.pi + 0.5

    row_px = row_frac * image_shape[0]
    col_px = col_frac * image_shape[1]

    return np.stack([row_px, col_px], axis=0)  # shape (2, H_out, W_out)


def reproject_pinhole(panorama, output_shape, fov, yaw=0.0, pitch=0.0):
    """
    Reproject equirectangular panorama to pinhole image.

    Args:
        panorama: (H_pano, W_pano, C) — equirectangular panorama
        output_shape: (H_out, W_out)
        fov: (fov_x, fov_y) in radians
        yaw: camera yaw orientation in radians
        pitch: camera pitch orientation in radians

    Returns:
        Pinhole image of shape (H_out, W_out, C)
    """
    H_out, W_out = output_shape
    fx = 1 / np.tan(fov[0] / 2.0)
    fy = 1 / np.tan(fov[1] / 2.0)

    # Create normalized pixel grid
    row_frac = np.linspace(-1, 1, H_out)
    col_frac = np.linspace(1, -1, W_out)
    row_frac, col_frac = np.meshgrid(row_frac, col_frac, indexing='ij')  # (H_out, W_out)

    # Convert to 3D ray directions (camera coordinates)
    x = col_frac
    y = row_frac

    dirs = np.stack([
        x,
        y * (fx / fy),  # correct for aspect ratio
        fx * np.ones_like(x)
    ], axis=-1)  # shape (H_out, W_out, 3)

    # Normalize directions
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    # --- Apply yaw and pitch rotations ---
    # Yaw (around y-axis)
    Ry = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [          0, 1,          0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    # Pitch (around x-axis)
    Rx = np.array([
        [1,           0,            0],
        [0,  np.cos(pitch), -np.sin(pitch)],
        [0,  np.sin(pitch),  np.cos(pitch)]
    ])
    R = Ry @ Rx  # Combined rotation

    dirs_rot = dirs @ R.T  # shape (H_out, W_out, 3)

    # Convert rotated dirs to azimuth and elevation
    x_rot = dirs_rot[..., 0]
    y_rot = dirs_rot[..., 1]
    z_rot = dirs_rot[..., 2]

    azimuth = np.arctan2(x_rot, z_rot)
    elevation = np.arcsin(y_rot)

    # Map to panorama pixel coordinates
    pano_coords = spherical_pixel_from_azimuth_elevation(azimuth, elevation, panorama.shape[:2])
    pano_coords[1] %= panorama.shape[1]  # wrap
    pano_coords[0] = np.clip(pano_coords[0], 0, panorama.shape[0] - 1)  # clamp

    # Interpolate per channel
    output = np.zeros((H_out, W_out, panorama.shape[2]), dtype=panorama.dtype)
    for c in range(panorama.shape[2]):
        output[..., c] = scipy.ndimage.map_coordinates(
            panorama[..., c],
            pano_coords,
            order=1,
            mode='wrap' if c == 1 else 'nearest'
        )
    return output


def process_single_panorama(pano_file, output_path, fov_x, fov_y, res_x, res_y, yaw_angles):
    """
    Process a single panorama file.

    This function is designed to be called by multiprocessing workers.
    """
    # Load panorama
    pano_img = Image.open(pano_file)
    pano_array = np.array(pano_img).astype(np.float32) / 255.0

    # Create output directory for this panorama
    pano_output_dir = output_path / pano_file.stem
    pano_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save pinhole images for each yaw
    for yaw in yaw_angles:
        projected = reproject_pinhole(
            pano_array,
            (res_y, res_x),
            (fov_x, fov_y),
            yaw=yaw,
            pitch=0.0
        )

        # Convert back to uint8
        projected_uint8 = np.clip(projected * 255.0, 0, 255).astype(np.uint8)

        # Save image with yaw angle in filename
        output_filename = f"yaw_{np.degrees(yaw):03.0f}.jpg"
        output_file = pano_output_dir / output_filename
        Image.fromarray(projected_uint8).save(output_file)


def process_panoramas(input_dir, output_dir, fov_x, fov_y, res_x, res_y, num_workers=1):
    """
    Process all panoramas in input_dir and save pinhole projections.

    Args:
        input_dir: Path to directory containing panorama images
        output_dir: Path to output directory
        fov_x: Horizontal field of view in radians
        fov_y: Vertical field of view in radians
        res_x: Horizontal resolution in pixels
        res_y: Vertical resolution in pixels
        num_workers: Number of parallel workers (default: 1 for sequential processing)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    panorama_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not panorama_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(panorama_files)} panorama files")
    print(f"Output resolution: {res_x}x{res_y}")
    print(f"FOV: {np.degrees(fov_x):.1f}° x {np.degrees(fov_y):.1f}°")
    print(f"Using {num_workers} worker(s)")

    # Generate 4 yaw angles at 90-degree intervals
    yaw_angles = np.arange(0, 2*np.pi, np.pi / 2.0)

    # Create partial function with fixed parameters
    worker_func = partial(
        process_single_panorama,
        output_path=output_path,
        fov_x=fov_x,
        fov_y=fov_y,
        res_x=res_x,
        res_y=res_y,
        yaw_angles=yaw_angles
    )

    if num_workers > 1:
        # Parallel processing
        with Pool(num_workers) as pool:
            list(tqdm.tqdm(
                pool.imap(worker_func, panorama_files),
                total=len(panorama_files),
                desc="Processing panoramas"
            ))
    else:
        # Sequential processing
        for pano_file in tqdm.tqdm(panorama_files, desc="Processing panoramas"):
            worker_func(pano_file)

    print(f"Processing complete. Images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert panoramas to pinhole images with 90-degree yaw rotations"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing panorama images"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save output images"
    )
    parser.add_argument(
        "--fov_x",
        type=float,
        default=np.pi/2.0,
        help="Horizontal field of view in radians (default: π/2)"
    )
    parser.add_argument(
        "--fov_y",
        type=float,
        default=np.pi/2.0,
        help="Vertical field of view in radians (default: π/2)"
    )
    parser.add_argument(
        "--res_x",
        type=int,
        default=1024,
        help="Horizontal resolution in pixels (default: 1024)"
    )
    parser.add_argument(
        "--res_y",
        type=int,
        default=None,
        help="Vertical resolution in pixels (default: computed from res_x and fov ratio)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 for sequential processing)"
    )

    args = parser.parse_args()

    # Compute res_y if not provided
    if args.res_y is None:
        args.res_y = int(np.round(args.res_x * args.fov_y / args.fov_x))

    process_panoramas(
        args.input_dir,
        args.output_dir,
        args.fov_x,
        args.fov_y,
        args.res_x,
        args.res_y,
        args.num_workers
    )


if __name__ == "__main__":
    main()
