import argparse
from typing import NamedTuple
import numpy as np
import os

from pydrake.all import RigidTransform


class Range(NamedTuple):
    min: float
    max: float


class SceneParams(NamedTuple):
    max_num_objects: int
    centroid_bounding_box: Range
    ...


class CameraParams(NamedTuple):
    num_views: int
    ...


class Scene(NamedTuple):
    world_from_objects: dict[str, RigidTransform]
    ...


def get_ycb_objects_list(ycb_path: str) -> list[str]:
    return sorted(filter(lambda x: '.tgz' not in x, os.listdir(ycb_path)))
    ...


def generate_scene(ycb_path: str, scene_params: SceneParams, rng: np.random.Generator) -> Scene:
    ycb_objects_list = get_ycb_objects_list(ycb_path)
    print(ycb_objects_list)
    print(len(ycb_objects_list))
    ...


def generate_world_from_camera_poses(
    camera_params: CameraParams, rng: np.random.Generator
) -> list[RigidTransform]:
    ...


def generate_keypoints_and_labels(
    scene: Scene, local_from_cameras: list[RigidTransform]
):
    ...


def main(output_path: str, ycb_objects: str, num_scenes: int, num_views_per_scene: int):
    # Create paths to output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for scene_idx in range(num_scenes):
        # Generate a scene
        rng = np.random.default_rng(seed=scene_idx)
        scene = generate_scene(
            ycb_objects,
            SceneParams(
                max_num_objects=5,
                centroid_bounding_box=Range(
                    min=np.array([-0.5, -0.5, 0.0]), max=np.array([0.5, 0.5, 1.0])
                )
            ),
            rng,
        )
        # Sample camera positions
        world_from_cameras = generate_world_from_camera_poses(
            CameraParams(num_views=num_views_per_scene), rng
        )

        # Generate keypoints and class labels
        keypoints, labels = generate_keypoints_and_labels(scene, world_from_cameras)

        ...
    ...


if __name__ == "__main__":
    NUM_SCENES = 10
    NUM_VIEWS_PER_SCENE = 5
    parser = argparse.ArgumentParser("YCB scene data generator")

    parser.add_argument("--output", required=True, help="Path to write output")
    parser.add_argument("--ycb_objects", required=True, help="Path to ycb objects")
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=NUM_SCENES,
        help=f"Number of scenes to generate. Default: ({NUM_SCENES})",
    )
    parser.add_argument(
        "--num_views_per_scene",
        type=int,
        default=NUM_VIEWS_PER_SCENE,
        help=f"Number of views per generated scene. Default: ({NUM_VIEWS_PER_SCENE})",
    )

    args = parser.parse_args()
    main(args.output, args.ycb_objects, args.num_scenes, args.num_views_per_scene)
