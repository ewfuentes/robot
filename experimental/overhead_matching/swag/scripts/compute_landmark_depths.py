"""Compute depth estimates for all landmarks using monocular depth estimation.

Runs DepthAnythingV2 on pinhole yaw images, extracts median depth within each
landmark's bounding box, and saves per-landmark depth values to a pickle file.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:compute_landmark_depths -- \
        --output /tmp/landmark_depths.pkl \
        --model-size large \
        --batch-size 8
"""

import common.torch.load_torch_deps

import argparse
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from experimental.learn_descriptors.depth_models import DepthAnythingV2, UniDepthV2


# Datasets to process: (relative pickle path from embeddings base, pinhole city name)
DATASETS = [
    ("mapillary/Framingham", "Framingham"),
    ("mapillary/MiamiBeach", "MiamiBeach"),
    ("mapillary/Gap", "Gap"),
    ("mapillary/Norway", "Norway"),
    ("mapillary/Middletown", "Middletown"),
    ("mapillary/SanFrancisco_mapillary", "SanFrancisco_mapillary"),
    ("mapillary/post_hurricane_ian", "post_hurricane_ian"),
    ("nightdrive", "nightdrive"),
    ("boston_snowy", "boston_snowy"),
]

DEFAULT_EMBEDDINGS_BASE = "/data/overhead_matching/datasets/semantic_landmark_embeddings"
DEFAULT_PINHOLE_BASE = "/data/overhead_matching/datasets/pinhole_images"


def load_v2_pickle(pickle_path: Path) -> dict | None:
    if not pickle_path.exists():
        return None
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and data.get("version") != "2.0":
        raise RuntimeError(f"Embedding pickle was not version 2.0: {pickle_path}")
    return data


def build_work_list(embeddings_base: Path, pinhole_base: Path):
    """Build a mapping from yaw image path -> list of (city, pano_key, landmark_idx, bbox) tuples.

    Groups bounding boxes by the yaw image they reference so we only run depth
    estimation once per image.
    """
    # yaw_image_path -> [(city, pano_key, landmark_idx, bbox_dict)]
    image_to_bboxes = defaultdict(list)
    # (city, pano_key, landmark_idx) -> landmark metadata for output
    landmark_meta = {}

    for rel_path, pinhole_city in DATASETS:
        pkl_path = embeddings_base / rel_path / "embeddings" / "embeddings.pkl"
        data = load_v2_pickle(pkl_path)
        if data is None:
            print(f"Warning: pickle not found at {pkl_path}, skipping")
            continue

        pinhole_dir = pinhole_base / pinhole_city
        if not pinhole_dir.exists():
            print(f"Warning: pinhole dir not found at {pinhole_dir}, skipping")
            continue

        panos = data.get("panoramas", {})
        for pano_key, pano_data in panos.items():
            for landmark in pano_data.get("landmarks", []):
                landmark_idx = landmark["landmark_idx"]
                key = (pinhole_city, pano_key, landmark_idx)

                landmark_meta[key] = {
                    "city": pinhole_city,
                    "description": landmark.get("description", ""),
                    "primary_tag": landmark.get("primary_tag", {}),
                    "confidence": landmark.get("confidence", ""),
                }

                for bbox in landmark.get("bounding_boxes", []):
                    yaw_str = bbox.get("yaw_angle", "")
                    if yaw_str not in {"0", "90", "180", "270"}:
                        continue

                    img_path = pinhole_dir / pano_key / f"yaw_{int(yaw_str):03d}.jpg"
                    image_to_bboxes[img_path].append((pinhole_city, pano_key, landmark_idx, bbox))

    return image_to_bboxes, landmark_meta


def extract_bbox_depth(depth_map: np.ndarray, bbox: dict) -> float:
    """Extract 10th percentile depth within a bounding box.

    Uses the 10th percentile (closest 10% of pixels) rather than the median
    to better represent the distance to the nearest part of the landmark,
    avoiding bias from background pixels at the edges of the box.

    Bbox coords are normalized 0-1000. We scale to depth map resolution directly.
    """
    dh, dw = depth_map.shape[:2]
    ymin = int(bbox["ymin"] * dh / 1000)
    xmin = int(bbox["xmin"] * dw / 1000)
    ymax = int(bbox["ymax"] * dh / 1000)
    xmax = int(bbox["xmax"] * dw / 1000)

    # Clamp and ensure at least 1 pixel
    ymin = max(0, min(ymin, dh - 1))
    xmin = max(0, min(xmin, dw - 1))
    ymax = max(ymin + 1, min(ymax, dh))
    xmax = max(xmin + 1, min(xmax, dw))

    region = depth_map[ymin:ymax, xmin:xmax]
    return float(np.percentile(region, 10))


def run_depth_extraction(args):
    embeddings_base = Path(args.embeddings_base)
    pinhole_base = Path(args.pinhole_base)
    output_path = Path(args.output)
    batch_size = args.batch_size

    # Load existing results for resumption
    existing_results = {}
    if output_path.exists():
        with open(output_path, "rb") as f:
            existing_results = pickle.load(f)
        print(f"Loaded {len(existing_results)} existing depth results from {output_path}")

    print("Building work list...")
    image_to_bboxes, landmark_meta = build_work_list(embeddings_base, pinhole_base)

    # Figure out which images we actually need to process
    # A landmark needs processing if we don't have a result for it yet
    needed_keys = set(landmark_meta.keys()) - set(existing_results.keys())
    needed_images = set()
    for img_path, bbox_list in image_to_bboxes.items():
        for city, pano_key, landmark_idx, bbox in bbox_list:
            if (city, pano_key, landmark_idx) in needed_keys:
                needed_images.add(img_path)
                break

    # Filter to images that actually exist
    needed_images = sorted([p for p in needed_images if p.exists()])
    skipped = len(set(image_to_bboxes.keys()) - set(needed_images) - {p for p in image_to_bboxes if p not in needed_images or (p.exists() and (p not in needed_images))})

    print(f"Total unique yaw images: {len(image_to_bboxes)}")
    print(f"Already computed landmarks: {len(existing_results)}")
    print(f"Remaining landmarks: {len(needed_keys)}")
    print(f"Yaw images to process: {len(needed_images)}")

    if not needed_images:
        print("Nothing to do!")
        return

    # Initialize depth model
    # Pinhole images are 2048x2048, 90° FoV -> f_px = 1024
    PINHOLE_INTRINSICS = np.array([
        [1024.0,    0.0, 1024.0],
        [   0.0, 1024.0, 1024.0],
        [   0.0,    0.0,    1.0],
    ])

    if args.model == "unidepth":
        print(f"Loading UniDepthV2 {args.model_size} (with known intrinsics: f=1024, c=1024)...")
        model = UniDepthV2(args.model_size, device=args.device, intrinsics=PINHOLE_INTRINSICS)
    elif args.model == "depthanything":
        print(f"Loading DepthAnythingV2 outdoor/{args.model_size}...")
        model = DepthAnythingV2("outdoor", args.model_size, device=args.device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    print("Model loaded.")

    # Per-landmark: collect all bbox depths, then take min median
    # (city, pano_key, landmark_idx) -> list of median depths from each bbox
    landmark_bbox_depths = defaultdict(list)

    # Process in batches
    results = dict(existing_results)
    save_interval = 500  # save every N images

    for batch_start in tqdm(range(0, len(needed_images), batch_size), desc="Depth batches"):
        batch_paths = needed_images[batch_start : batch_start + batch_size]

        # Run depth model
        depth_maps = model.infer_batch(batch_paths)

        # Extract bbox depths
        for img_path, depth_map in zip(batch_paths, depth_maps):
            depth_map = np.asarray(depth_map, dtype=np.float32).squeeze()
            for city, pano_key, landmark_idx, bbox in image_to_bboxes[img_path]:
                key = (city, pano_key, landmark_idx)
                if key not in needed_keys:
                    continue
                median_depth = extract_bbox_depth(depth_map, bbox)
                landmark_bbox_depths[key].append(median_depth)

        # Periodic save
        if (batch_start // batch_size + 1) % (save_interval // batch_size + 1) == 0:
            _finalize_batch_results(results, landmark_bbox_depths, landmark_meta)
            _save_results(results, output_path)

    # Final save
    _finalize_batch_results(results, landmark_bbox_depths, landmark_meta)
    _save_results(results, output_path)
    print(f"\nDone! {len(results)} total landmark depths saved to {output_path}")

    # Print summary stats
    depths = [v["depth_m"] for v in results.values()]
    if depths:
        print(f"Depth stats: min={min(depths):.1f}m, median={np.median(depths):.1f}m, "
              f"max={max(depths):.1f}m, mean={np.mean(depths):.1f}m")


def _finalize_batch_results(results, landmark_bbox_depths, landmark_meta):
    """Convert accumulated bbox depths into final per-landmark results."""
    for key, depths_list in landmark_bbox_depths.items():
        if not depths_list:
            continue
        results[key] = {
            "depth_m": min(depths_list),  # min median across bboxes = closest view
            **landmark_meta.get(key, {}),
        }
    landmark_bbox_depths.clear()


def _save_results(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)


def main():
    parser = argparse.ArgumentParser(description="Compute landmark depths using monocular depth estimation")
    parser.add_argument("--embeddings-base", default=DEFAULT_EMBEDDINGS_BASE,
                        help="Base path to semantic landmark embeddings")
    parser.add_argument("--pinhole-base", default=DEFAULT_PINHOLE_BASE,
                        help="Base path to pinhole images")
    parser.add_argument("--output", default="/tmp/landmark_depths.pkl",
                        help="Output pickle path for depth results")
    parser.add_argument("--model", default="unidepth", choices=["unidepth", "depthanything"],
                        help="Depth model to use")
    parser.add_argument("--model-size", default="large", choices=["small", "base", "large"],
                        help="Model size variant")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Inference batch size")
    parser.add_argument("--device", default="cuda:0",
                        help="Device for depth model")
    args = parser.parse_args()
    run_depth_extraction(args)


if __name__ == "__main__":
    main()
