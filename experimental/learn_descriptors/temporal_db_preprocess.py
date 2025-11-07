import argparse

import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="MegaDepth preprocessing script")

    parser.add_argument(
        "--dir_images", type=str, required=True, help="path to undistorted images"
    )

    parser.add_argument(
        "--dir_reconstruction",
        type=str,
        required=True,
        help="reconstruction of all images in dir_images",
    )

    parser.add_argument(
        "--output_path", type=str, required=True, help="path to the output directory"
    )

    args = parser.parse_args()
    return args


def process_reconstruction_data(dir_reconstruction: Path):
    assert dir_reconstruction.exists()

    # Process cameras.txt

    with open(dir_reconstruction / "cameras.txt", "r") as f:
        raw = f.readlines()[3:]  # skip the header

    camera_intrinsics = {}
    for camera in raw:
        camera = camera.split(" ")
        camera_intrinsics[int(camera[0])] = [float(elem) for elem in camera[2:]]

    # Process points3D.txt

    with open(dir_reconstruction / "points3D.txt", "r") as f:
        raw = f.readlines()[3:]  # skip the header

    points3D = {}
    for point3D in raw:
        point3D = point3D.split(" ")
        points3D[int(point3D[0])] = np.array(
            [float(point3D[1]), float(point3D[2]), float(point3D[3])]
        )

    # Process images.txt

    with open(dir_reconstruction, "images.txt", "r") as f:
        raw = f.readlines()[4:]  # skip the header
    image_id_to_idx = {}

    image_names = []
    raw_pose = []
    camera = []
    points3D_id_to_2D = []
    n_points3D = []
    for idx, (image, points) in enumerate(zip(raw[::2], raw[1::2])):
        image = image.split(" ")
        points = points.split(" ")

        image_id_to_idx[int(image[0])] = idx

        image_name = image[-1].strip("\n")
        image_names.append(image_name)

        raw_pose.append([float(elem) for elem in image[1:-2]])
        camera.append(int(image[-2]))
        current_points3D_id_to_2D = {}
        for x, y, point3D_id in zip(points[::3], points[1::3], points[2::3]):
            if int(point3D_id) == -1:
                continue
            current_points3D_id_to_2D[int(point3D_id)] = [float(x), float(y)]
        points3D_id_to_2D.append(current_points3D_id_to_2D)
        n_points3D.append(len(current_points3D_id_to_2D))


if __name__ == "__main__":
    args = parse_args()
    # get undistorted images
    dir_images = Path(args.dir_images)
    dir_reconstruction = Path(args.dir_reconstruction)

    assert dir_images.exists()
    assert dir_reconstruction.exists()


# get reconstruction of images (per scene)

# localize across seasons, then create overlap matrix (number of correspondences over total)

# save results
