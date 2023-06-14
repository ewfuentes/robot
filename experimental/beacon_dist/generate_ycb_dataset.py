import argparse
import numpy as np
import os
import time
from experimental.beacon_dist.utils import KeypointDescriptorDtype, CLASS_SIZE
import experimental.beacon_dist.render_ycb_scene_python as rys

RENDERER_NAME = "my_renderer"

CameraSamplingStrategy = rys.MovingCamera

SAMPLING_STRATEGIES: dict[str, CameraSamplingStrategy] = {
    "fixed": rys.MovingCamera(
        start_in_world=np.array([1.0, 0.0, 0.0]),
        end_in_world=np.array([1.0, 0.0, 0.0]),
    ),
    "move_away": rys.MovingCamera(
        start_in_world=np.array([0.5, 0.0, 0.0]),
        end_in_world=np.array([2.0, 0.0, 0.0]),
    ),
    "strafe": rys.MovingCamera(
        start_in_world=np.array([1.0, -0.25, 0.0]),
        end_in_world=np.array([1.0, 0.25, 0.0]),
    ),
    # "rotate_xy": SphericalCamera(
    #     radial_distance_m=Range(min=2.0, max=2.0),
    #     azimuth_range_rad=Range(min=0.0, max=2 * np.pi),
    #     inclination_range_rad=Range(min=np.pi / 2.0, max=np.pi / 2.0),
    # ),
    # "sphere": SphericalCamera(
    #     radial_distance_m=Range(min=2.0, max=2.0),
    #     azimuth_range_rad=Range(min=0.0, max=2 * np.pi),
    #     inclination_range_rad=Range(min=0.0, max=np.pi),
    # ),
}


def class_label_from_label_set(labels: set[int], ycb_objects: list[str]) -> np.ndarray:
    out = np.zeros((CLASS_SIZE), dtype=np.uint64)
    for object_idx in labels:
        idx = object_idx // 64
        bit_idx = object_idx % 64
        flag = np.left_shift(np.uint64(1), np.uint64(bit_idx))
        out[idx] = np.bitwise_or(out[idx], flag)

    return out


def serialize_results(
    scene_results: list[rys.SceneResult], ycb_objects: list[str], output_path: str
):
    max_name_size = max([len(x) for x in ycb_objects])
    image_id = 0
    keypoint_data = []
    scene_info = []
    image_info = []
    for scene_id, scene_result in enumerate(scene_results):
        scene_info.append(
            np.array(
                [
                    (scene_id, name, transform)
                    for name, transform in scene_result.world_from_objects.items()
                ],
                dtype=[
                    ("scene_id", np.uint64),
                    ("object_name", (np.unicode_, max_name_size)),
                    ("world_from_object", np.float64, (3, 4)),
                ],
            )
        )

        for view_result in scene_result.view_results:
            image_info.append(
                np.array(
                    [
                        (
                            image_id,
                            scene_id,
                            view_result.world_from_camera,
                        )
                    ],
                    dtype=[
                        ("image_id", np.uint64),
                        ("scene_id", np.uint64),
                        ("world_from_camera", np.float64, (3, 4)),
                    ],
                )
            )
            for i in range(len(view_result.keypoints)):
                kp = view_result.keypoints[i]
                keypoint_data.append(
                    np.array(
                        [
                            (
                                image_id,
                                kp.angle,
                                kp.class_id,
                                kp.octave,
                                kp.x,
                                kp.y,
                                kp.response,
                                kp.size,
                                view_result.descriptors[i],
                                class_label_from_label_set(
                                    view_result.labels[i], ycb_objects
                                ),
                            )
                        ],
                        dtype=KeypointDescriptorDtype,
                    )
                )
            image_id += 1

    np.savez(
        output_path,
        data=np.concatenate(keypoint_data),
        scene_info=np.concatenate(scene_info),
        image_info=np.concatenate(image_info),
        objects=np.array(ycb_objects),
    )


def main(
    output_path: str,
    ycb_path: str,
    num_scenes: int,
    num_views_per_scene: int,
    camera_strategy: str,
    num_workers: int,
):
    # Create paths to output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    camera_params = rys.CameraParams(
        num_views=num_views_per_scene,
        camera_strategy=SAMPLING_STRATEGIES[camera_strategy],
        height_px=1024,
        width_px=1280,
        fov_y_rad=np.pi / 4.0,
    )

    print('Loading YCB Objects')
    start_time = time.time()
    scene_data = rys.load_ycb_objects(ycb_path, None)
    end_load_time = time.time()
    print('Load time:', end_load_time - start_time, 'Num objects:', len(scene_data.object_list))
    print('Building dataset')
    scene_results = rys.build_dataset(scene_data, camera_params, num_scenes, num_workers)
    end_dataset_time = time.time()
    print('Dataset time:', end_dataset_time - end_load_time)

    # Write out generated data
    print('Serializing Results')
    serialize_results(
        scene_results,
        scene_data.object_list,
        output_path,
    )
    end_serialize_time = time.time()
    print('Serialize time:', end_serialize_time - end_dataset_time)


if __name__ == "__main__":
    NUM_SCENES = 10
    NUM_VIEWS_PER_SCENE = 5
    DEFAULT_CAMERA_STRATEGY = "fixed"
    NUM_WORKERS = 4
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
    parser.add_argument(
        "--camera_strategy",
        choices=list(SAMPLING_STRATEGIES.keys()),
        default=DEFAULT_CAMERA_STRATEGY,
        help=f"Camera Pose sampling strategy. Default ({DEFAULT_CAMERA_STRATEGY})",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of threads to spawn. Default ({NUM_WORKERS})",
    )

    args = parser.parse_args()
    main(
        args.output,
        args.ycb_objects,
        args.num_scenes,
        args.num_views_per_scene,
        args.camera_strategy,
        args.num_workers,
    )
