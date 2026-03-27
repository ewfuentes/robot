"""Save RGB + depth visualizations for all landmarks above a depth threshold.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:save_discarded_landmarks -- \
        --depths-pkl /tmp/landmark_depths_unidepth_calibrated.pkl \
        --threshold 100 \
        --output-dir /tmp/discarded_landmarks
"""

import common.torch.load_torch_deps

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

from experimental.learn_descriptors.depth_models import DepthAnythingV2, UniDepthV2

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


def load_v2_pickle(pkl_path):
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and data.get("version") != "2.0":
        raise RuntimeError(f"Not v2.0: {pkl_path}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths-pkl", required=True)
    parser.add_argument("--threshold", type=float, default=100.0)
    parser.add_argument("--output-dir", default="/tmp/discarded_landmarks")
    parser.add_argument("--embeddings-base", default=DEFAULT_EMBEDDINGS_BASE)
    parser.add_argument("--pinhole-base", default=DEFAULT_PINHOLE_BASE)
    parser.add_argument("--model", default="depthanything", choices=["unidepth", "depthanything"],
                        help="Depth model for visualization (should match the one used to generate depths-pkl)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    embeddings_base = Path(args.embeddings_base)
    pinhole_base = Path(args.pinhole_base)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load depths
    with open(args.depths_pkl, "rb") as f:
        landmark_depths = pickle.load(f)

    # Filter to discarded landmarks
    discarded = {k: v for k, v in landmark_depths.items() if v["depth_m"] > args.threshold}
    print(f"Total landmarks: {len(landmark_depths)}, discarded (>{args.threshold}m): {len(discarded)}")

    # Load all embedding pickles to get bounding boxes
    city_pano_data = {}  # city -> {pano_key -> pano_data}
    for rel_path, pinhole_city in DATASETS:
        pkl_path = embeddings_base / rel_path / "embeddings" / "embeddings.pkl"
        data = load_v2_pickle(pkl_path)
        if data is None:
            continue
        city_pano_data[pinhole_city] = data.get("panoramas", {})

    # Load depth model for on-the-fly depth maps
    PINHOLE_INTRINSICS = np.array([
        [1024.0, 0.0, 1024.0],
        [0.0, 1024.0, 1024.0],
        [0.0, 0.0, 1.0],
    ])
    if args.model == "unidepth":
        print("Loading UniDepthV2...")
        model = UniDepthV2("large", device=args.device, intrinsics=PINHOLE_INTRINSICS)
    else:
        print("Loading DepthAnythingV2 outdoor/large...")
        model = DepthAnythingV2("outdoor", "large", device=args.device)
    print("Model loaded.")

    # Group discarded by yaw image to avoid running depth model multiple times
    # yaw_image_path -> [(city, pano_key, lm_idx, depth_m, desc, bboxes_for_this_yaw)]
    yaw_image_jobs = defaultdict(list)

    for (city, pano_key, lm_idx), val in discarded.items():
        panos = city_pano_data.get(city, {})
        pano_data = panos.get(pano_key)
        if pano_data is None:
            continue

        landmark = None
        for lm in pano_data.get("landmarks", []):
            if lm["landmark_idx"] == lm_idx:
                landmark = lm
                break
        if landmark is None:
            continue

        for bbox in landmark.get("bounding_boxes", []):
            yaw_str = bbox.get("yaw_angle", "")
            if yaw_str not in {"0", "90", "180", "270"}:
                continue
            img_path = pinhole_base / city / pano_key / f"yaw_{int(yaw_str):03d}.jpg"
            tag = landmark.get("primary_tag", {})
            tag_str = f"{tag.get('key', '')}={tag.get('value', '')}" if isinstance(tag, dict) else ""
            yaw_image_jobs[img_path].append({
                "city": city,
                "pano_key": pano_key,
                "lm_idx": lm_idx,
                "depth_m": val["depth_m"],
                "desc": val.get("description", ""),
                "tag": tag_str,
                "bbox": bbox,
            })

    print(f"Unique yaw images to process: {len(yaw_image_jobs)}")

    DISPLAY_SIZE = 512
    saved = 0

    for img_path in tqdm(sorted(yaw_image_jobs.keys()), desc="Rendering"):
        if not img_path.exists():
            continue

        jobs = yaw_image_jobs[img_path]
        city = jobs[0]["city"]

        # Load RGB
        full_img = Image.open(img_path)
        rgb_small = full_img.resize((DISPLAY_SIZE, DISPLAY_SIZE))

        # Run depth model
        depth_map = model.infer(img_path)
        depth_map = np.asarray(depth_map, dtype=np.float32).squeeze()
        depth_small = np.array(Image.fromarray(depth_map).resize((DISPLAY_SIZE, DISPLAY_SIZE)))

        # Render one figure per landmark
        for job in jobs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

            ax1.imshow(rgb_small)
            ax1.set_title("RGB")

            im = ax2.imshow(depth_small, cmap="turbo")
            ax2.set_title(f"Depth ({depth_map.min():.1f}-{depth_map.max():.1f}m)")
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

            bbox = job["bbox"]
            xmin = bbox["xmin"] * DISPLAY_SIZE / 1000
            ymin = bbox["ymin"] * DISPLAY_SIZE / 1000
            xmax = bbox["xmax"] * DISPLAY_SIZE / 1000
            ymax = bbox["ymax"] * DISPLAY_SIZE / 1000

            for ax in [ax1, ax2]:
                rect = mpatches.FancyBboxPatch(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    linewidth=3, edgecolor="red", facecolor="none",
                    boxstyle="square,pad=0",
                )
                ax.add_patch(rect)
                ax.axis("off")

            pano_id = job["pano_key"].split(",")[0]
            fig.suptitle(
                f"[DISCARD] {city} / {pano_id} / #{job['lm_idx']} ({job['tag']})\n"
                f"{job['desc'][:100]}\n"
                f"Estimated depth: {job['depth_m']:.1f}m",
                fontsize=10, color="red",
            )
            plt.tight_layout()

            city_dir = output_dir / city
            city_dir.mkdir(parents=True, exist_ok=True)
            out_path = city_dir / f"{pano_id}_lm{job['lm_idx']}_yaw{bbox['yaw_angle']}.png"
            plt.savefig(out_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved += 1

    print(f"\nSaved {saved} images to {output_dir}")


if __name__ == "__main__":
    main()
