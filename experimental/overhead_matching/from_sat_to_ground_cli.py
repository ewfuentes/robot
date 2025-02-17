from experimental.overhead_matching.from_sat_to_ground import from_sat_to_ground
import numpy as np 
if __name__ == "__main__":
    import argparse
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    parser = argparse.ArgumentParser()
    parser.add_argument("semantic_pointcloud_path", type=str)
    parser.add_argument("semantic_satellite_path", type=str)

    args = parser.parse_args()
    semantic_pointcloud = np.load(args.semantic_pointcloud_path)
    semantic_satellite = Image.open(args.semantic_satellite_path)
    semantic_satellite = np.array(semantic_satellite)

    prob_dist = from_sat_to_ground(
        semantic_pointcloud,
        semantic_satellite,
        sigma=1.0,
        num_orientation_bins=64,
    )


