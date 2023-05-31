import argparse
from typing import NamedTuple
import numpy as np
import os
import time

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


class Scene(NamedTuple):
    world_from_objects: dict[str, RigidTransform]
    diagram: Diagram


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
    return sorted(filter(lambda x: ".tgz" not in x, os.listdir(ycb_path)))


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


def generate_scene(
    ycb_path: str, scene_params: SceneParams, rng: np.random.Generator
) -> Scene:
    ycb_objects_list = get_ycb_objects_list(ycb_path)
    num_objects = rng.integers(2, scene_params.max_num_objects, endpoint=True)
    object_idxs = rng.choice(len(ycb_objects_list), size=num_objects, replace=False)
    object_names = [ycb_objects_list[x] for x in object_idxs]

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

    # Create a scene from the sampled poses
    diagram = generate_diagram_from_poses(world_from_objects, ycb_path)
    return Scene(
        world_from_objects=world_from_objects,
        diagram=diagram,
    )


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
    scene: Scene, camera_params: CameraParams, world_from_cameras: list[RigidTransform]
) -> tuple[Image, Image]:
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
    color_camera = ColorRenderCamera(camera_core, show_window=True)

    root_context = scene.diagram.CreateDefaultContext()

    rgb_images = []
    label_images = []
    world_frame_id = scene.diagram.GetSubsystemByName('scene_graph').world_frame_id()
    for world_from_camera in world_from_cameras:
        query_object = scene.diagram.GetOutputPort("query_object").Eval(root_context)

        rgb_images.append(
            query_object.RenderColorImage(
                color_camera, world_frame_id, world_from_camera
            )
        )
        label_images.append(
            query_object.RenderLabelImage(
                color_camera, world_frame_id, world_from_camera
            )
        )
        time.sleep(2.0)

    return rgb_images, label_images


def generate_keypoints_and_labels(
    scene: Scene, camera_params: CameraParams, world_from_cameras: list[RigidTransform]
):
    # Render an RGB image and a label image
    rgb_images, label_images = render_scene(scene, camera_params, world_from_cameras)
    ...
    return None, None


def main(
    output_path: str,
    ycb_objects: str,
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

    for scene_idx in range(num_scenes):
        # Generate a scene
        rng = np.random.default_rng(seed=scene_idx)
        scene = generate_scene(
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
        keypoints, labels = generate_keypoints_and_labels(
            scene, camera_params, world_from_cameras
        )

        ...
    ...


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
