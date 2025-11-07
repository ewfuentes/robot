import common.torch.load_torch_deps
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import numpy as np
from PIL import Image
from experimental.learn_descriptors.depth_models import (
    DepthModel,
    DepthAnythingV2,
    DepthPro,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_depth_pro_checkpt", type=str, default=0, required=True)
    parser.add_argument("--dir_output", type=str, default="~/depth_models_test")
    parser.add_argument(
        "--depth_anythingv2_model_type",
        choices=["indoor", "outdoor"],
        default="outdoor",
    )
    parser.add_argument(
        "--depth_anythingv2_model_size",
        choices=["small", "base", "large"],
        default="large",
    )
    parser.add_argument(
        "--no_output",
        action="store_true",
        help="Use if you don't want to write the visualizations to disk",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dir_input", type=str, default="")
    args = parser.parse_args()
    return args


def colorize_depth(depth_m, dmin_m=0.2, dmax_m=5.0, invert=True):
    """
    depth_m: depth in meters (or anything if you like)
    path_in:  depth image (e.g., 16-bit PNG), ScanNet is mm
    dmin_m/dmax_m: visible depth range in meters
    invert: True makes near = warm colors, far = cool (common convention)
    """

    # Build a valid mask (0 or very large values are invalid)
    valid = (depth_m > 0) & np.isfinite(depth_m)

    # Clip to visualization range
    depth_vis = depth_m.copy()
    depth_vis[~valid] = np.nan
    depth_vis = np.clip(depth_vis, dmin_m, dmax_m)

    # Normalize to 0–255 for colormap
    norm = (depth_vis - dmin_m) / max(1e-6, (dmax_m - dmin_m))
    if invert:
        norm = 1.0 - norm
    norm[np.isnan(depth_vis)] = 0  # send invalid to 0
    norm = (norm * 255.0).astype(np.uint8)

    # Apply a pleasant colormap (COLORMAP_TURBO or JET)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)

    # Optional: make invalid pixels transparent instead of black
    alpha = np.where(valid & (depth_m >= dmin_m) & (depth_m <= dmax_m), 255, 0).astype(
        np.uint8
    )
    bgra = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
    bgra[..., 3] = alpha

    return bgra


if __name__ == "__main__":
    dir_images = Path("external/sacre_coeur_snippet/sacre_coeur/query")
    files_images = [f for f in dir_images.iterdir() if f.is_file()]
    file_image = files_images[0]
    image = Image.open(files_images[0])
    args = parse_args()
    path_depth_pro_checkpt = Path(args.path_depth_pro_checkpt)
    dir_out = Path(args.dir_output).expanduser()
    dir_out.mkdir(parents=True, exist_ok=True)

    assert path_depth_pro_checkpt.exists()
    assert dir_out.exists()
    depth_pro = DepthPro(
        path_depth_pro_checkpt, default_focal_px=528
    )  # default px is for 4Seasons here
    depth_anythingv2 = DepthAnythingV2(
        args.depth_anythingv2_model_type, args.depth_anythingv2_model_size
    )

    depth1 = depth_pro.infer(file_image)
    depth2 = depth_anythingv2.infer(file_image)
    depth_viz = np.hstack((depth1, depth2))
    depth_viz = colorize_depth(depth_viz, dmin_m=0, dmax_m=np.max(depth_viz))
    depth1 = np.array(depth1)

    plt.imshow(depth1, cmap="turbo")
    plt.colorbar(label="Depth (m)")
    plt.title("Depth map (hover mouse to see values)")

    depth2 = np.array(depth2)
    depth1 = colorize_depth(
        np.array(depth1), dmin_m=0, dmax_m=np.max(depth1), invert=False
    )
    depth2 = colorize_depth(np.array(depth2), dmin_m=0, dmax_m=np.max(depth2))
    if not args.no_output:
        if args.dir_input != "" and Path(args.dir_input).exists():
            depth_pro.infer_dir(
                Path(args.dir_input),
                batch_size=args.batch_size,
                recursive=True,
                dir_out_relative="depth/depth_pro",
            )
            depth_anythingv2.infer_dir(
                Path(args.dir_input),
                batch_size=args.batch_size,
                recursive=True,
                dir_out_relative="depth/depth_anythingv2",
            )
        path_output = dir_out / "depth_pro" / f"plt_{file_image.name}"
        path_output.parent.mkdir(exist_ok=True, parents=True)
        path_out_og = dir_out / "original" / file_image.name
        path_out1 = dir_out / "depth_pro" / file_image.name
        path_out2 = dir_out / "depth_anythingv2" / file_image.name
        path_out_og.parent.mkdir(parents=True, exist_ok=True)
        path_out1.parent.mkdir(parents=True, exist_ok=True)
        path_out2.parent.mkdir(parents=True, exist_ok=True)
        path_out = dir_out / file_image.name
        print(f"depth viz written to: {path_out}")
        cv2.imwrite(str(path_out_og), np.array(image))
        cv2.imwrite(str(path_out1), depth1)
        cv2.imwrite(str(path_out2), depth2)
        cv2.imwrite(str(path_out), depth_viz)
        plt.savefig(path_output)
    plt.show()
