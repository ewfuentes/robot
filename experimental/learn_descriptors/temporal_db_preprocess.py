import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np


@dataclass(slots=True)
class PathsImage:
    paths_image: List[Optional[Path]] = field(default_factory=list)
    paths_depth: List[Optional[Path]] = field(default_factory=list)


@dataclass(slots=True)
class ReconstructionData:
    paths_image: PathsImage
    points3d_id_to_2d: List[Dict[int, np.ndarray]] = field(default_factory=list)
    n_points3d: List[int] = field(default_factory=list)
    intrinsics: List[Optional[np.ndarray]] = field(default_factory=list)
    poses: List[Optional[np.ndarray]] = field(default_factory=list)
    points3d_id_to_ndepth: List[Dict[int, float]] = field(default_factory=list)


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MegaDepth preprocessing script")

    parser.add_argument("--dir_images", type=str, required=True)
    parser.add_argument("--dir_depths", type=str, required=True)
    parser.add_argument(
        "--depth_extension",
        choices=[".npy", ".h5"],
        required=True,
        help="Extension of depth files.",
    )
    parser.add_argument(
        "--dir_reconstruction",
        type=str,
        required=True,
        help="Directory containing COLMAP cameras.txt / images.txt / points3D.txt",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        help="Scene identifier; default is dir_images name.",
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        required=True,
        help="Output directory for saved scene_info",
    )
    return parser.parse_args()


# ---------- Core helpers (pure functions) ----------

def build_image_paths(
    dir_images: Path,
    image_names: List[str],
    dir_depths: Path,
    depth_extension: str,
) -> PathsImage:
    paths_image: List[Optional[Path]] = []
    paths_depth: List[Optional[Path]] = []

    for image_name in image_names:
        img_path = dir_images / image_name
        depth_path = dir_depths / f"{img_path.stem}{depth_extension}"

        if not depth_path.exists():
            paths_image.append(None)
            paths_depth.append(None)
            continue

        file_size = depth_path.stat().st_size
        if file_size < 100 * 1024:
            # Tiny mask / invalid file → skip
            paths_image.append(None)
            paths_depth.append(None)
        else:
            paths_image.append(img_path)
            paths_depth.append(depth_path)

    return PathsImage(paths_image=paths_image, paths_depth=paths_depth)


def cam_configuration(
    image_names: List[str],
    dir_reconstruction: Path,
    raw_pose: List[List[float]],
    points3d_id_to_2d: List[Dict[int, np.ndarray]],
    camera_ids: List[int],
) -> Tuple[List[Optional[np.ndarray]],
           List[Optional[np.ndarray]],
           np.ndarray,
           List[Dict[int, float]]]:

    # cameras.txt
    with open(dir_reconstruction / "cameras.txt", "r") as f:
        raw = f.readlines()[3:]
    camera_intrinsics: Dict[int, List[float]] = {}
    for line in raw:
        parts = line.split()
        cam_id = int(parts[0])
        camera_intrinsics[cam_id] = [float(x) for x in parts[2:]]

    # points3D.txt
    with open(dir_reconstruction / "points3D.txt", "r") as f:
        raw = f.readlines()[3:]
    points3D: Dict[int, np.ndarray] = {}
    for line in raw:
        parts = line.split()
        pid = int(parts[0])
        points3D[pid] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

    intrinsics: List[Optional[np.ndarray]] = []
    poses: List[Optional[np.ndarray]] = []
    principal_axis: List[np.ndarray] = []
    points3d_id_to_ndepth: List[Dict[int, float]] = []

    for idx, image_name in enumerate(image_names):
        # If this image was invalid / missing, mirror that here
        if raw_pose[idx] is None:
            intrinsics.append(None)
            poses.append(None)
            principal_axis.append(np.zeros(3))
            points3d_id_to_ndepth.append({})
            continue

        cam_params = camera_intrinsics[camera_ids[idx]]
        # Assumes PINHOLE: fx, fy, cx, cy at indices 2-5
        K = np.zeros((3, 3), dtype=float)
        K[0, 0] = cam_params[2]
        K[1, 1] = cam_params[3]
        K[0, 2] = cam_params[4]
        K[1, 2] = cam_params[5]
        K[2, 2] = 1.0
        intrinsics.append(K)

        pose = raw_pose[idx]
        qvec = np.array(pose[:4], dtype=float)
        qvec /= np.linalg.norm(qvec)
        w, x, y, z = qvec

        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
        ])
        t = np.array(pose[4:7], dtype=float)

        T_wc = np.eye(4, dtype=float)
        T_wc[:3, :3] = R
        T_wc[:3, 3] = t
        poses.append(T_wc)

        principal_axis.append(R[2, :])

        nd = {}
        for pid in points3d_id_to_2d[idx].keys():
            p3d = points3D[pid]
            nd[pid] = (R[2, :] @ p3d + t[2]) / (0.5 * (K[0, 0] + K[1, 1]))
        points3d_id_to_ndepth.append(nd)

    return intrinsics, poses, np.array(principal_axis), points3d_id_to_ndepth


def process_reconstruction_data(
    dir_reconstruction: Path,
    dir_images: Path,
    dir_depths: Path,
    depth_extension: str,
) -> ReconstructionData:
    assert (dir_reconstruction / "images.txt").exists()

    with open(dir_reconstruction / "images.txt", "r") as f:
        raw = f.readlines()[4:]

    image_names: List[str] = []
    raw_pose: List[List[float]] = []
    camera_ids: List[int] = []
    points3d_id_to_2d: List[Dict[int, np.ndarray]] = []
    n_points3d: List[int] = []

    for image_line, points_line in zip(raw[::2], raw[1::2]):
        img_parts = image_line.split()
        pts_parts = points_line.split()

        image_name = img_parts[-1].strip()
        image_names.append(image_name)

        pose_vals = [float(x) for x in img_parts[1:-2]]
        raw_pose.append(pose_vals)

        camera_ids.append(int(img_parts[-2]))

        pmap: Dict[int, np.ndarray] = {}
        for x, y, pid in zip(pts_parts[0::3], pts_parts[1::3], pts_parts[2::3]):
            pid = int(pid)
            if pid == -1:
                continue
            pmap[pid] = np.array([float(x), float(y)], dtype=float)
        points3d_id_to_2d.append(pmap)
        n_points3d.append(len(pmap))

    paths = build_image_paths(dir_images, image_names, dir_depths, depth_extension)
    intrinsics, poses, _, points3d_id_to_ndepth = cam_configuration(
        image_names, dir_reconstruction, raw_pose, points3d_id_to_2d, camera_ids
    )

    return ReconstructionData(
        paths_image=paths,
        points3d_id_to_2d=points3d_id_to_2d,
        n_points3d=n_points3d,
        intrinsics=intrinsics,
        poses=poses,
        points3d_id_to_ndepth=points3d_id_to_ndepth,
    )


def compute_overlap_and_scale(
    reconstruction_data: ReconstructionData,
) -> Tuple[np.ndarray, np.ndarray]:
    n_images = len(reconstruction_data.paths_image.paths_image)
    overlap = np.full((n_images, n_images), -1.0, dtype=float)
    scale = np.full((n_images, n_images), -1.0, dtype=float)

    for i in range(n_images):
        if reconstruction_data.paths_image.paths_depth[i] is None:
            continue
        for j in range(i + 1, n_images):
            if reconstruction_data.paths_image.paths_depth[j] is None:
                continue

            pts1 = reconstruction_data.points3d_id_to_2d[i]
            pts2 = reconstruction_data.points3d_id_to_2d[j]
            matches = pts1.keys() & pts2.keys()
            if not matches:
                continue

            overlap[i, j] = len(matches) / max(1, len(pts1))
            overlap[j, i] = len(matches) / max(1, len(pts2))

            nd1 = np.array(
                [reconstruction_data.points3d_id_to_ndepth[i][m] for m in matches]
            )
            nd2 = np.array(
                [reconstruction_data.points3d_id_to_ndepth[j][m] for m in matches]
            )
            min_scale = float(np.min(np.maximum(nd1 / nd2, nd2 / nd1)))
            scale[i, j] = scale[j, i] = min_scale

    return overlap, scale


def make_loftr_scene_info(
    reconstruction_data: ReconstructionData,
    overlap_matrix: np.ndarray,
    scene_id: Union[str, int],
) -> dict:
    n_images = len(reconstruction_data.paths_image.paths_image)

    pair_infos = []
    for i in range(n_images):
        for j in range(n_images):
            if i == j:
                continue
            pair_infos.append(
                ((i, j), float(overlap_matrix[i, j]), [])
            )

    return {
        "image_paths": reconstruction_data.paths_image.paths_image,
        "depth_paths": reconstruction_data.paths_image.paths_depth,
        "scene_id": scene_id,
        "pair_infos": pair_infos,
        "intrinsics": reconstruction_data.intrinsics,
        "poses": reconstruction_data.poses,
    }


# ---------- Public API ----------

def build_scene_info(
    dir_images: Path,
    dir_depths: Path,
    depth_extension: str,
    dir_reconstruction: Path,
    scene_id: Optional[Union[str, int]] = None,
) -> dict:
    """
    Pure Python entrypoint:
    - Reads COLMAP reconstruction
    - Builds ReconstructionData
    - Computes overlap
    - Returns a LoFTR-style scene_info dict
    """
    dir_images = dir_images.expanduser().resolve()
    dir_depths = dir_depths.expanduser().resolve()
    dir_reconstruction = dir_reconstruction.expanduser().resolve()
    assert dir_images.exists()
    assert dir_depths.exists()
    assert dir_reconstruction.exists()

    if scene_id is None:
        scene_id = dir_images.name

    reconstruction_data = process_reconstruction_data(
        dir_reconstruction, dir_images, dir_depths, depth_extension
    )
    overlap, _ = compute_overlap_and_scale(reconstruction_data)
    return make_loftr_scene_info(reconstruction_data, overlap, scene_id)


def build_scene_info_and_save(
    dir_images: Path,
    dir_depths: Path,
    depth_extension: str,
    dir_reconstruction: Path,
    dir_out: Path,
    scene_id: Optional[Union[str, int]] = None,
) -> Path:
    scene_info = build_scene_info(
        dir_images=dir_images,
        dir_depths=dir_depths,
        depth_extension=depth_extension,
        dir_reconstruction=dir_reconstruction,
        scene_id=scene_id,
    )

    dir_out = dir_out.expanduser().resolve()
    dir_out.mkdir(parents=True, exist_ok=True)
    out_path = dir_out / f"{scene_info['scene_id']}.npz"
    # Save keys explicitly so it's nice to load
    np.savez(
        out_path,
        image_paths=np.array(scene_info["image_paths"], dtype=object),
        depth_paths=np.array(scene_info["depth_paths"], dtype=object),
        scene_id=scene_info["scene_id"],
        pair_infos=np.array(scene_info["pair_infos"], dtype=object),
        intrinsics=np.array(scene_info["intrinsics"], dtype=object),
        poses=np.array(scene_info["poses"], dtype=object),
    )
    return out_path


# ---------- main for CLI ----------

def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    build_scene_info_and_save(
        dir_images=Path(args.dir_images),
        dir_depths=Path(args.dir_depths),
        depth_extension=args.depth_extension,
        dir_reconstruction=Path(args.dir_reconstruction),
        dir_out=Path(args.dir_out),
        scene_id=args.scene_id,
    )


if __name__ == "__main__":
    main()
