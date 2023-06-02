import argparse
from typing import NamedTuple
import numpy as np
import os
import tqdm
import cv2 as cv

from experimental.beacon_dist.utils import KeypointDescriptorDtype, CLASS_SIZE

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    AngleAxis,
    Diagram,
    DiagramBuilder,
    Image,
    MultibodyPlant,
    Parser,
    RigidTransform,
    RotationMatrix,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    RenderCameraCore,
    CameraInfo,
    ColorRenderCamera,
    ClippingRange,
)


class Range(NamedTuple):
    min: float
    max: float


class MovingCamera(NamedTuple):
    start_camera_in_world: np.ndarray
    end_camera_in_world: np.ndarray


class SphericalCamera(NamedTuple):
    radial_distance_m: Range
    azimuth_range_rad: Range
    inclination_range_rad: Range


CameraSamplingStrategy = MovingCamera | SphericalCamera


class SceneParams(NamedTuple):
    max_num_objects: int
    centroid_bounding_box: Range


class CameraParams(NamedTuple):
    num_views: int
    strategy: CameraSamplingStrategy
    height_px: int
    width_px: int
    fov_y_rad: float


class CameraViewResult(NamedTuple):
    world_from_camera: RigidTransform
    keypoints: list[cv.KeyPoint]
    descriptors: np.ndarray
    labels: list[set[str]]


class SceneResult(NamedTuple):
    camera_view_results: list[CameraViewResult]
    world_from_objects: dict[str, RigidTransform]


RENDERER_NAME = "my_renderer"

SAMPLING_STRATEGIES: dict[str, CameraSamplingStrategy] = {
    "fixed": MovingCamera(
        start_camera_in_world=np.array([2.0, 0.0, 0.0]),
        end_camera_in_world=np.array([2.0, 0.0, 0.0]),
    ),
    "move_away": MovingCamera(
        start_camera_in_world=np.array([2.0, 0.0, 0.0]),
        end_camera_in_world=np.array([5.0, 0.0, 0.0]),
    ),
    "strafe": MovingCamera(
        start_camera_in_world=np.array([2.0, -0.5, 0.0]),
        end_camera_in_world=np.array([2.0, 0.5, 0.0]),
    ),
    "rotate_xy": SphericalCamera(
        radial_distance_m=Range(min=2.0, max=2.0),
        azimuth_range_rad=Range(min=0.0, max=2 * np.pi),
        inclination_range_rad=Range(min=np.pi / 2.0, max=np.pi / 2.0),
    ),
    "sphere": SphericalCamera(
        radial_distance_m=Range(min=2.0, max=2.0),
        azimuth_range_rad=Range(min=0.0, max=2 * np.pi),
        inclination_range_rad=Range(min=0.0, max=np.pi),
    ),
}


def get_ycb_objects_list(ycb_path: str) -> list[str]:
    # TODO: remove before landing
    return sorted(filter(lambda x: ".tgz" not in x, os.listdir(ycb_path)))[:10]


def load_object(
    ycb_path: str,
    plant: MultibodyPlant,
    object_name: str,
) -> None:
    parser = Parser(plant, object_name)
    parser.AddModelFromFile(
        os.path.join(ycb_path, object_name, "google_16k/textured.obj"),
        model_name=object_name,
    )


def generate_diagram_from_poses(
    world_from_objects: dict[str, RigidTransform], ycb_path: str
) -> Diagram:
    TIME_STEP_S = 0.0
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, TIME_STEP_S)
    renderer = MakeRenderEngineVtk(RenderEngineVtkParams())
    scene_graph.AddRenderer(RENDERER_NAME, renderer)

    for object_name, world_from_object in world_from_objects.items():
        load_object(ycb_path, plant, object_name)

    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    plant.Finalize()
    diagram = builder.Build()

    for body_idx in plant.GetFloatingBaseBodies():
        body = plant.get_body(body_idx)
        plant.SetDefaultFreeBodyPose(body, world_from_objects[body.name()])

    return diagram


def sample_objects_and_poses(
    ycb_objects: list[str], scene_params: SceneParams, rng: np.random.Generator
) -> dict[str, RigidTransform]:
    num_objects = rng.integers(2, scene_params.max_num_objects, endpoint=True)
    object_idxs = rng.choice(len(ycb_objects), size=num_objects, replace=False)
    object_names = [ycb_objects[x] for x in object_idxs]

    centroid_range = (
        scene_params.centroid_bounding_box.max - scene_params.centroid_bounding_box.min
    )

    # Sample poses
    world_from_objects = {}
    for name in object_names:
        # sample a centroid position
        centroid_in_world = (
            centroid_range * rng.random((3,)) + scene_params.centroid_bounding_box.min
        )
        rot_vector = rng.normal(size=3, scale=np.pi)
        angle = np.linalg.norm(rot_vector)
        axis = rot_vector / angle
        angle_axis = AngleAxis(angle, axis)

        world_from_objects[name] = RigidTransform(angle_axis, centroid_in_world)

    return world_from_objects


def create_origin_facing_world_from_pt(pt_in_world: np.ndarray) -> RotationMatrix:
    # The camera frame has the +X axis to the right, the +y axis down and the +z axis into the image
    z_axis = -pt_in_world / np.linalg.norm(pt_in_world)
    x_axis = np.cross(z_axis, np.array([0, 0, 1.0]))
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    # This should be unit norm by construction, but normalize anyway
    y_axis = y_axis / np.linalg.norm(y_axis)
    # Right multiplying a vector in the camera frame by the rotation matrix below yields a vector
    # in the world frame
    return RotationMatrix(np.column_stack([x_axis, y_axis, z_axis]))


def generate_moving_world_from_camera_poses(
    camera_params: CameraParams,
) -> list[RigidTransform]:
    start_camera_in_world = camera_params.strategy.start_camera_in_world
    end_camera_in_world = camera_params.strategy.end_camera_in_world

    camera_movement_m = np.linalg.norm(end_camera_in_world - start_camera_in_world)
    assert camera_params.num_views > 0
    assert (
        camera_movement_m > 1e-3 or camera_params.num_views == 1
    ), f"Asking for {camera_params.num_views} but camera only moving {camera_movement_m} m."

    # compute the camera rotation matrix
    center_camera_in_world = (end_camera_in_world + start_camera_in_world) / 2.0
    world_from_camera_rot = create_origin_facing_world_from_pt(center_camera_in_world)

    out = []
    for i in range(camera_params.num_views):
        frac = i / max(camera_params.num_views - 1.0, 1.0)
        camera_in_world = (
            end_camera_in_world - start_camera_in_world
        ) * frac + start_camera_in_world
        out.append(RigidTransform(world_from_camera_rot, camera_in_world))
    return out


def generate_spherical_world_from_camera_poses(
    camera_params: CameraParams,
) -> list[RigidTransform]:
    ...


def generate_world_from_camera_poses(
    camera_params: CameraParams,
) -> list[RigidTransform]:
    if isinstance(camera_params.strategy, MovingCamera):
        return generate_moving_world_from_camera_poses(camera_params)

    assert f"Unknown Camera Strategy: {camera_params.strategy}"


def render_scene(
    diagram: Diagram,
    world_from_objects: dict[str, RigidTransform],
    world_from_camera: RigidTransform,
    camera_params: CameraParams,
) -> Image:
    # Create the camera
    camera_intrinsics = CameraInfo(
        width=camera_params.width_px,
        height=camera_params.height_px,
        fov_y=camera_params.fov_y_rad,
    )

    body_from_sensor = RigidTransform()
    camera_core = RenderCameraCore(
        renderer_name=RENDERER_NAME,
        intrinsics=camera_intrinsics,
        clipping=ClippingRange(near=0.10, far=10.0),
        X_BS=body_from_sensor,
    )
    color_camera = ColorRenderCamera(camera_core, show_window=False)

    # Set the object poses
    root_context = diagram.CreateDefaultContext()
    plant = diagram.GetSubsystemByName("plant")
    context = plant.GetMyContextFromRoot(root_context)
    for body_idx in plant.GetFloatingBaseBodies():
        body = plant.get_body(body_idx)
        plant.SetFreeBodyPose(
            context,
            body,
            world_from_objects.get(
                body.name(), RigidTransform(np.array([0, 0, 100.0]))
            ),
        )

    world_frame_id = diagram.GetSubsystemByName("scene_graph").world_frame_id()
    query_object = diagram.GetOutputPort("query_object").Eval(root_context)

    rgb_image = query_object.RenderColorImage(
        color_camera, world_frame_id, world_from_camera
    )

    return rgb_image


def opencv_image_from_drake_image(img: Image):
    return np.stack([img.data[:, :, 2], img.data[:, :, 1], img.data[:, :, 0]], axis=-1)


def generate_keypoints_and_labels_from_images(
    all_object_image: Image, image_from_object_name: dict[str, Image]
) -> tuple[list[cv.KeyPoint], np.ndarray, list[set[str]]]:
    orb = cv.ORB_create(nfeatures=300)
    img = opencv_image_from_drake_image(all_object_image)
    kp = orb.detect(img, None)
    kp, descriptors = orb.compute(img, kp)

    labels = [set() for i in range(len(kp))]
    for object_name, object_img in image_from_object_name.items():
        object_img = opencv_image_from_drake_image(object_img)
        object_kp, object_descriptors = orb.compute(object_img, kp)

        # The nonzero object descriptors point to the keypoints associated with this object
        # Note that the same key point maybe associated with multiple objects
        kp_idxs = np.nonzero(np.sum(object_descriptors, axis=1))[0]
        if len(kp_idxs) == 0:
            print(f"No keypoints associated with {object_name}")
            continue

        for kp_idx in kp_idxs:
            labels[kp_idx].add(object_name)

    return kp, descriptors, labels


def generate_keypoints_and_labels(
    diagram: Diagram,
    world_from_objects: dict[str, RigidTransform],
    world_from_camera: RigidTransform,
    camera_params: CameraParams,
) -> tuple[list[cv.KeyPoint], np.ndarray, list[set[str]]]:
    # Render an RGB image and a label image
    all_object_image = render_scene(
        diagram, world_from_objects, world_from_camera, camera_params
    )
    image_from_object_name = {
        object_name: render_scene(
            diagram,
            {object_name: world_from_objects[object_name]},
            world_from_camera,
            camera_params,
        )
        for object_name in world_from_objects
    }

    return generate_keypoints_and_labels_from_images(
        all_object_image, image_from_object_name
    )


def generate_scene(ycb_path: str) -> tuple[Diagram, list[str]]:
    TIME_STEP_S = 0.0
    ycb_objects_list = get_ycb_objects_list(ycb_path)
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, TIME_STEP_S)
    renderer = MakeRenderEngineVtk(RenderEngineVtkParams())
    scene_graph.AddRenderer(RENDERER_NAME, renderer)

    for object_name in ycb_objects_list:
        load_object(ycb_path, plant, object_name)

    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    plant.Finalize()
    return builder.Build(), ycb_objects_list


def class_label_from_label_set(labels: set[str], ycb_objects: list[str]) -> np.ndarray:
    out = np.zeros((CLASS_SIZE), dtype=np.uint64)
    for label in labels:
        object_idx = ycb_objects.index(label)
        idx = object_idx // 64
        bit_idx = object_idx % 64
        flag = np.left_shift(np.uint64(1), np.uint64(bit_idx))
        out[idx] = np.bitwise_or(out[idx], flag)

    return out


def serialize_results(
    scene_results: list[SceneResult], ycb_objects: list[str], output_path: str
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
                    (scene_id, name, transform.GetAsMatrix34())
                    for name, transform in scene_result.world_from_objects.items()
                ],
                dtype=[
                    ("scene_id", np.uint64),
                    ("object_name", (np.unicode_, max_name_size)),
                    ("world_from_object", np.float64, (3, 4)),
                ],
            )
        )

        for camera_view_result in scene_result.camera_view_results:
            image_info.append(
                np.array([
                    (
                        image_id,
                        scene_id,
                        camera_view_result.world_from_camera.GetAsMatrix34(),
                    )],
                    dtype=[
                        ("image_id", np.uint64),
                        ("scene_id", np.uint64),
                        ("world_from_camera", np.float64, (3, 4)),
                    ],
                )
            )
            for i in range(len(camera_view_result.keypoints)):
                kp = camera_view_result.keypoints[i]
                keypoint_data.append(
                    np.array(
                        [
                            (
                                image_id,
                                kp.angle,
                                kp.class_id,
                                kp.octave,
                                kp.pt[0],
                                kp.pt[1],
                                kp.response,
                                kp.size,
                                camera_view_result.descriptors[i, :],
                                class_label_from_label_set(
                                    camera_view_result.labels[i], ycb_objects
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
):
    # Create paths to output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    camera_params = CameraParams(
        num_views=num_views_per_scene,
        strategy=SAMPLING_STRATEGIES[camera_strategy],
        height_px=1024,
        width_px=1280,
        fov_y_rad=np.pi / 4.0,
    )

    diagram, ycb_objects = generate_scene(ycb_path)

    scene_results = []
    for scene_idx in tqdm.tqdm(range(num_scenes)):
        # Generate a scene
        rng = np.random.default_rng(seed=scene_idx)
        world_from_objects = sample_objects_and_poses(
            ycb_objects,
            SceneParams(
                max_num_objects=10,
                centroid_bounding_box=Range(
                    min=np.array([-0.5, -0.5, -0.5]), max=np.array([0.5, 0.5, 0.5])
                ),
            ),
            rng,
        )

        # Sample camera positions
        world_from_cameras = generate_world_from_camera_poses(camera_params)

        # Generate keypoints and class labels
        camera_view_results = []
        for world_from_camera in world_from_cameras:
            keypoints, descriptors, labels = generate_keypoints_and_labels(
                diagram, world_from_objects, world_from_camera, camera_params
            )
            camera_view_results.append(
                CameraViewResult(
                    world_from_camera=world_from_camera,
                    keypoints=keypoints,
                    descriptors=descriptors,
                    labels=labels,
                )
            )

        scene_results.append(
            SceneResult(
                camera_view_results=camera_view_results,
                world_from_objects=world_from_objects,
            )
        )

    # Write out generated data
    serialize_results(
        scene_results,
        ycb_objects,
        output_path,
    )


if __name__ == "__main__":
    NUM_SCENES = 10
    NUM_VIEWS_PER_SCENE = 5
    DEFAULT_CAMERA_STRATEGY = "fixed"
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

    args = parser.parse_args()
    main(
        args.output,
        args.ycb_objects,
        args.num_scenes,
        args.num_views_per_scene,
        args.camera_strategy,
    )
