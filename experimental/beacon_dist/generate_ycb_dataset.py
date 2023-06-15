import argparse
import numpy as np
import os
import time
import tqdm
from experimental.beacon_dist.utils import KeypointDescriptorDtype, CLASS_SIZE
import experimental.beacon_dist.render_ycb_scene_python as rys
import multiprocessing

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


def class_label_from_label_set(labels: set[int]) -> np.ndarray:
    out = np.zeros((CLASS_SIZE), dtype=np.uint64)
    for object_idx in labels:
        idx = object_idx // 64
        bit_idx = object_idx % 64
        flag = np.left_shift(np.uint64(1), np.uint64(bit_idx))
        out[idx] = np.bitwise_or(out[idx], flag)

    return out


def serialize_result(
    scene_result: rys.SceneResult,
    image_id_start: int,
    scene_id: int,
    max_name_size: int,
):
    scene_info = np.array(
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

    image_info = []
    keypoint_data = []
    for i, view_result in enumerate(scene_result.view_results):
        class_labels = rys.convert_class_labels_to_matrix(view_result.labels, 4)
        image_info.append(
            np.array(
                [
                    (
                        image_id_start + i,
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
                            image_id_start + i,
                            kp.angle,
                            kp.class_id,
                            kp.octave,
                            kp.x,
                            kp.y,
                            kp.response,
                            kp.size,
                            view_result.descriptors[i],
                            class_labels[i],
                        )
                    ],
                    dtype=KeypointDescriptorDtype,
                )
            )
    return scene_info, image_info, keypoint_data


def serialize_results(
    scene_results: list[rys.SceneResult],
    ycb_objects: list[str],
    output_path: str,
):

    max_name_size = max([len(x) for x in ycb_objects])
    num_images_for_scene = [len(x.view_results) for x in scene_results]
    image_id_starts = [0] + list(np.cumsum(num_images_for_scene)[:-1])
    num_scenes = len(scene_results)

    with multiprocessing.Pool() as pool:
        results = pool.starmap(
            serialize_result,
            zip(
                scene_results,
                image_id_starts,
                range(num_scenes),
                [max_name_size] * num_scenes,
            ),
        )
        scene_info = [x[0] for x in results]
        image_info = [x[1] for x in results]
        keypoint_data = [x[2] for x in results]

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

    print("Loading YCB Objects")
    start_time = time.time()
    scene_data = rys.load_ycb_objects(ycb_path, None)
    end_load_time = time.time()
    print(
        "Load time:",
        end_load_time - start_time,
        "Num objects:",
        len(scene_data.object_list),
    )
    print("Building dataset")
    progress_bar = tqdm.tqdm(total=num_scenes)

    def progress_update(scene_id):
        progress_bar.update()
        progress_bar.refresh()
        return True

    scene_results = rys.build_dataset(
        scene_data, camera_params, num_scenes, num_workers, progress_update
    )
    progress_bar.close()
    end_dataset_time = time.time()
    print("Dataset time:", end_dataset_time - end_load_time)

    # Write out generated data
    print("Serializing Results")
    serialize_results(
        scene_results,
        scene_data.object_list,
        output_path,
    )
    end_serialize_time = time.time()
    print("Serialize time:", end_serialize_time - end_dataset_time)


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
