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


# get undistorted images

# get reconstruction of images (per scene)

# localize across seasons, then create overlap matrix (number of correspondences over total)

# save results
