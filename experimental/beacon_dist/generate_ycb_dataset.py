import argparse
from typing import NamedTuple
import numpy as np
import os

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    AngleAxis,
    Diagram,
    DiagramBuilder,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MultibodyPlant,
    Parser,
    RigidTransform,
    JointSliders,
)


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
    diagram: Diagram
    ...


def get_ycb_objects_list(ycb_path: str) -> list[str]:
    return sorted(filter(lambda x: ".tgz" not in x, os.listdir(ycb_path)))
    ...


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


def generate_world_from_camera_poses(
    camera_params: CameraParams, rng: np.random.Generator
) -> list[RigidTransform]:
    ...


def generate_keypoints_and_labels(
    scene: Scene, local_from_cameras: list[RigidTransform]
):
    ...


def visualize_scene(scene: Scene, meshcat: Meshcat):
    builder = DiagramBuilder()
    sub_diagram = builder.AddNamedSystem("diagram", scene.diagram)

    plant = sub_diagram.GetSubsystemByName("plant")
    MeshcatVisualizer.AddToBuilder(
        builder,
        sub_diagram.GetOutputPort("query_object"),
        meshcat,
        MeshcatVisualizerParams(),
    )
    joint_sliders = JointSliders(meshcat, plant)
    diagram = builder.Build()
    print(diagram.GetGraphvizString())
    joint_sliders.Run(diagram)


def main(output_path: str, ycb_objects: str, num_scenes: int, num_views_per_scene: int):
    # Create paths to output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    meshcat = Meshcat()

    for scene_idx in range(num_scenes):
        # Generate a scene
        rng = np.random.default_rng(seed=scene_idx)
        scene = generate_scene(
            ycb_objects,
            SceneParams(
                max_num_objects=5,
                centroid_bounding_box=Range(
                    min=np.array([-0.5, -0.5, 0.0]), max=np.array([0.5, 0.5, 1.0])
                ),
            ),
            rng,
        )

        visualize_scene(scene, meshcat)
        # # Sample camera positions
        # world_from_cameras = generate_world_from_camera_poses(
        #     CameraParams(num_views=num_views_per_scene), rng
        # )

        # # Generate keypoints and class labels
        # keypoints, labels = generate_keypoints_and_labels(scene, world_from_cameras)

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
