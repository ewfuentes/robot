import unittest

import numpy as np

from pydrake.all import (
    AbstractValue,
    AddMultibodyPlantSceneGraph,
    CameraInfo,
    ClippingRange,
    Context,
    DepthRange,
    DepthRenderCamera,
    DiagramBuilder,
    JointSliders,
    LeafSystem,
    Image,
    MakeRenderEngineVtk,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MultibodyPlant,
    Parser,
    PixelType,
    RenderCameraCore,
    RenderEngineVtkParams,
    RgbdSensor,
    RigidTransform,
)


def file_name_from_object_name(name: str):
    return f"/home/erick/scratch/ycb_objects/{name}/google_16k/textured.obj"


def load_object(plant: MultibodyPlant, name: str):
    parser = Parser(plant, name)
    parser.AddModelFromFile(
        file_name_from_object_name(name),
        model_name=name,
    )


class ImageEvaluator(LeafSystem):
    def __init__(self):
        super().__init__()
        self._image_in = self.DeclareAbstractInputPort(
            name="image_in", model_value=AbstractValue.Make(Image[PixelType.kRgba8U]())
        )
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=0.1, offset_sec=0.0, update=self.eval_image
        )
        self.DeclareForcedPublishEvent(self.eval_image)

    def eval_image(self, context: Context):
        self._image_in.Eval(context)

    def image_input_port(self):
        return self._image_in


class LoadYcbTest(unittest.TestCase):
    def test_load_ycb_object(self):
        # Setup
        TIME_STEP_S = 0.0
        RENDERER_NAME = "my_renderer"
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

        depth_camera = DepthRenderCamera(
            camera_core, DepthRange(min_in=0.1, min_out=3.0)
        )
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

        builder.ExportOutput(rgbd_camera.color_image_output_port(), "rgbd_image")
        builder.ExportOutput(rgbd_camera.depth_image_32F_output_port(), "depth_image")

        image_evaluator = ImageEvaluator()
        builder.AddSystem(image_evaluator)

        builder.Connect(
            scene_graph.get_query_output_port(), rgbd_camera.query_object_input_port()
        )
        builder.Connect(
            rgbd_camera.color_image_output_port(), image_evaluator.image_input_port()
        )

        MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat, MeshcatVisualizerParams()
        )

        diagram = builder.Build()

        print(diagram.GetGraphvizString(), flush=True)

        # Action
        joint_sliders.Run(diagram)

        # Verification

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
