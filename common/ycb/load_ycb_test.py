import unittest

import numpy as np

from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    MultibodyPlant,
    Parser,
    Meshcat,
    ModelVisualizer,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    RenderCameraCore,
    CameraInfo,
    RigidTransform,
    RgbdSensor,
    DepthRenderCamera,
    DepthRange,
    ClippingRange,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    JointSliders,
)


def file_name_from_object_name(name: str):
    return f"/home/erick/scratch/ycb_objects/{name}/google_16k/textured.obj"


def load_object(plant: MultibodyPlant, name: str):
    parser = Parser(plant, name)
    parser.AddModelFromFile(
        file_name_from_object_name(name),
        model_name=name,
    )


class LoadYcbTest(unittest.TestCase):
    def test_load_ycb_object(self):
        # Setup
        TIME_STEP_S = 0.0
        RENDERER_NAME = 'my_renderer'
        meshcat = Meshcat()
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, TIME_STEP_S)
        load_object(plant, "053_mini_soccer_ball")
        load_object(plant, "048_hammer")
        load_object(plant, "065-j_cups")
        renderer = MakeRenderEngineVtk(RenderEngineVtkParams())
        scene_graph.AddRenderer(RENDERER_NAME, renderer)
        camera_intrinsics = CameraInfo(
            width=1280,
            height=1024,
            fov_y=np.pi / 6.0,
        )

        camera_core = RenderCameraCore(
            renderer_name=RENDERER_NAME,
            intrinsics=camera_intrinsics,
            clipping=ClippingRange(near=0.1, far=3.0),
            X_BS=RigidTransform(),
        )

        depth_camera = DepthRenderCamera(camera_core, DepthRange(min_in=0.1, min_out=3.0))
        world_from_camera = RigidTransform()
        rgbd_camera = builder.AddSystem(
            RgbdSensor(
                scene_graph.world_frame_id(),
                world_from_camera,
                depth_camera,
                show_window=True,
            )
        )

        plant.Finalize()

        joint_sliders = JointSliders(meshcat, plant)
        builder.AddSystem(joint_sliders)

        builder.ExportOutput(rgbd_camera.color_image_output_port(), 'rgbd_image')
        builder.ExportOutput(rgbd_camera.depth_image_32F_output_port(), 'depth_image')

        builder.Connect(scene_graph.get_query_output_port(),
                        rgbd_camera.query_object_input_port())

        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, MeshcatVisualizerParams())

        diagram = builder.Build()

        print(diagram.GetGraphvizString(), flush=True)

        # Action
        joint_sliders.Run(diagram)

        # Verification

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
