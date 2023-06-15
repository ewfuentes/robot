import argparse
import numpy as np
import os
import time
import tqdm
import tqdm.contrib.concurrent
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
    for view_idx, view_result in enumerate(scene_result.view_results):
        class_labels = rys.convert_class_labels_to_matrix(view_result.labels, CLASS_SIZE)
        image_info.append(
            np.array(
                [
                    (
                        image_id_start + view_idx,
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
        for kp_idx in range(len(view_result.keypoints)):
            kp = view_result.keypoints[kp_idx]
            keypoint_data.append(
                np.array(
                    [
                        (
                            image_id_start + view_idx,
                            kp.angle,
                            kp.class_id,
                            kp.octave,
                            kp.x,
                            kp.y,
                            kp.response,
                            kp.size,
                            view_result.descriptors[kp_idx],
                            class_labels[kp_idx],
                        )
                    ],
                    dtype=KeypointDescriptorDtype,
                )
            )
    return scene_info, image_info, keypoint_data


def serialize_results(
    scene_results: list[rys.SceneResult],
    ycb_objects: list[str],
    initial_image_id: int,
    initial_scene_id: int,
    output_path: str,
):

    max_name_size = max([len(x) for x in ycb_objects])
    num_images_for_scene = [len(x.view_results) for x in scene_results]
    image_id_starts = [initial_image_id] + list(
        initial_image_id + np.cumsum(num_images_for_scene)[:-1]
    )
    num_scenes = len(scene_results)

    results = tqdm.contrib.concurrent.process_map(
        serialize_result,
        scene_results,
        image_id_starts,
        range(initial_scene_id, initial_scene_id + num_scenes),
        [max_name_size] * num_scenes,
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

    generated_scenes = 0
    start_image_id = 0
    MAX_NUM_SCENES_PER_ROUND = 5000
    print(
        "Splitting up into",
        num_scenes // MAX_NUM_SCENES_PER_ROUND
        + (1 if (num_scenes % MAX_NUM_SCENES_PER_ROUND) > 0 else 0),
        "chunks.",
    )
    while generated_scenes != num_scenes:
        scenes_to_generate = min(
            MAX_NUM_SCENES_PER_ROUND, num_scenes - generated_scenes
        )
        print("Building dataset")

        progress_bar = tqdm.tqdm(total=scenes_to_generate)

        def progress_update(scene_id):
            progress_bar.update()
            progress_bar.refresh()
            return True

        start_dataset_time = time.time()
        scene_results = rys.build_dataset(
            scene_data,
            camera_params,
            scenes_to_generate,
            generated_scenes,
            num_workers,
            progress_update,
        )
        progress_bar.close()
        end_dataset_time = time.time()
        print("Dataset time:", end_dataset_time - start_dataset_time)

        # Write out generated data
        print("Serializing Results")

        if num_scenes > MAX_NUM_SCENES_PER_ROUND:
            file, ext = os.path.splitext(output_path)
            file_path = (
                    f"{file}.part_{generated_scenes // MAX_NUM_SCENES_PER_ROUND:05d}{ext}"
            )
        else:
            file_path = output_path

        serialize_results(
            scene_results=scene_results,
            ycb_objects=scene_data.object_list,
            initial_image_id=start_image_id,
            initial_scene_id=generated_scenes,
            output_path=file_path,
        )
        end_serialize_time = time.time()
        print("Serialize time:", end_serialize_time - end_dataset_time)

        start_image_id += sum([len(x.view_results) for x in scene_results])
        generated_scenes += scenes_to_generate


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
