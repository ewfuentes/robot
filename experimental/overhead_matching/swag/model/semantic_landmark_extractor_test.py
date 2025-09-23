
import unittest

import common.torch.load_torch_deps
import torch
from pathlib import Path
import math
import shapely

import experimental.overhead_matching.swag.model.semantic_landmark_extractor as sle
import experimental.overhead_matching.swag.data.vigor_dataset as vd


def compute_column_from_pano_landmark(pano_loc, landmark_loc, pano_width):
    dx = landmark_loc[1] - pano_loc[1]
    dy = landmark_loc[0] - pano_loc[0]
    theta = math.atan2(dx, -dy)
    frac = (theta + math.pi) / (2 * math.pi)
    return frac * pano_width


class SemanticLandmarkExtractorWithDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dataset = vd.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet/"),
            vd.VigorDatasetConfig(
                satellite_tensor_cache_info=None, panorama_tensor_cache_info=None))

    def test_satellite_landmark_extractor_with_dataset(self):
        # Setup
        BATCH_SIZE = 13
        config = sle.SemanticLandmarkExtractorConfig()
        model = sle.SemanticLandmarkExtractor(config)

        dataloader = vd.get_dataloader(self._dataset, batch_size=BATCH_SIZE)
        batch = next(iter(dataloader))
        model_input = sle.ModelInput(
            image=batch.satellite,
            metadata=batch.satellite_metadata,
        )

        # Action
        extractor_output = model(model_input)

        # Verification
        max_num_landmarks = extractor_output.features.shape[1]
        self.assertEqual(extractor_output.mask.shape, (BATCH_SIZE, max_num_landmarks))
        self.assertEqual(extractor_output.features.shape,
                         (BATCH_SIZE, max_num_landmarks, model.output_dim))
        self.assertEqual(extractor_output.positions.shape, (BATCH_SIZE, max_num_landmarks, 2, 2))

    def test_panorama_landmark_extractor_with_dataset(self):
        # Setup
        BATCH_SIZE = 7
        config = sle.SemanticLandmarkExtractorConfig()
        model = sle.SemanticLandmarkExtractor(config)

        dataloader = vd.get_dataloader(self._dataset, batch_size=BATCH_SIZE)
        batch = next(iter(dataloader))
        model_input = sle.ModelInput(
            image=batch.panorama,
            metadata=batch.panorama_metadata,
        )

        # Action
        extractor_output = model(model_input)

        # Verification
        max_num_landmarks = extractor_output.features.shape[1]
        self.assertEqual(extractor_output.mask.shape, (BATCH_SIZE, max_num_landmarks))
        self.assertEqual(extractor_output.features.shape,
                         (BATCH_SIZE, max_num_landmarks, model.output_dim))
        self.assertEqual(extractor_output.positions.shape, (BATCH_SIZE, max_num_landmarks, 2, 2))


class SemanticLandmarkExtractorTest(unittest.TestCase):
    def test_satellite_landmark_extractor(self):
        # Setup
        BATCH_SIZE = 2
        MAX_NUM_LANDMARKS = 2
        loc_1 = (123.0, 456.0)
        loc_2 = (987.0, 654.0)
        landmark_1 = (1000.0, 2000.0)
        landmark_2 = (125.0, 458.0)
        model_input = sle.ModelInput(
            image=torch.zeros((BATCH_SIZE, 0, 256, 256)),
            metadata=[
                dict(
                    web_mercator_y=loc_1[0],
                    web_mercator_x=loc_1[1],
                    landmarks=[
                        dict(
                            geometry_px=shapely.Point(*landmark_1[::-1]),
                            landmark_type="bus_stop"),
                        dict(
                            geometry_px=shapely.Point(*landmark_2[::-1]),
                            landmark_type="restaurants")]),
                dict(
                    web_mercator_y=loc_2[0],
                    web_mercator_x=loc_2[1],
                    landmarks=[
                        dict(
                            geometry_px=shapely.Point(*landmark_1[::-1]),
                            landmark_type="places_of_worship")]),
            ],
        )
        config = sle.SemanticLandmarkExtractorConfig()
        model = sle.SemanticLandmarkExtractor(config)

        # Action
        extractor_output = model(model_input)

        # Verification
        self.assertEqual(extractor_output.mask.shape, (BATCH_SIZE, MAX_NUM_LANDMARKS))
        self.assertEqual(extractor_output.features.shape,
                         (BATCH_SIZE, MAX_NUM_LANDMARKS, model.output_dim))
        self.assertEqual(extractor_output.positions.shape, (BATCH_SIZE, MAX_NUM_LANDMARKS, 2, 2))

        self.assertTrue((extractor_output.mask ==
                         torch.tensor([[False, False], [False, True]])).all())
        self.assertTrue((extractor_output.positions[0, 0] ==
                         torch.tensor([landmark_1[0] - loc_1[0], landmark_1[1] - loc_1[1]])).all())
        self.assertTrue((extractor_output.positions[0, 1] ==
                         torch.tensor([landmark_2[0] - loc_1[0], landmark_2[1] - loc_1[1]])).all())
        self.assertTrue((extractor_output.positions[1, 0] ==
                         torch.tensor([landmark_1[0] - loc_2[0], landmark_1[1] - loc_2[1]])).all())

    def test_panorama_landmark_extractor(self):
        # Setup
        BATCH_SIZE = 2
        MAX_NUM_LANDMARKS = 2
        PANO_WIDTH = 512
        loc_1 = (123.0, 456.0)
        loc_2 = (987.0, 654.0)
        landmark_1 = (1000.0, 2000.0)
        landmark_2 = (125.0, 458.0)
        model_input = sle.ModelInput(
            image=torch.zeros((BATCH_SIZE, 0, 256, PANO_WIDTH)),
            metadata=[
                dict(
                    web_mercator_y=loc_1[0],
                    web_mercator_x=loc_1[1],
                    pano_id="pano_id",
                    landmarks=[
                        dict(
                            geometry_px=shapely.Point(landmark_1[::-1]),
                            landmark_type="bus_stop"),
                        dict(
                            geometry_px=shapely.Point(landmark_2[::-1]),
                            landmark_type="restaurants")]),
                dict(
                    web_mercator_y=loc_2[0],
                    web_mercator_x=loc_2[1],
                    landmarks=[
                        dict(
                            geometry_px=shapely.Point(*landmark_1[::-1]),
                            landmark_type="places_of_worship")]),
            ],
        )
        config = sle.SemanticLandmarkExtractorConfig()
        model = sle.SemanticLandmarkExtractor(config)

        # Action
        extractor_output = model(model_input)

        # Verification
        self.assertEqual(extractor_output.mask.shape, (BATCH_SIZE, MAX_NUM_LANDMARKS))
        self.assertEqual(extractor_output.features.shape,
                         (BATCH_SIZE, MAX_NUM_LANDMARKS, model.output_dim))
        self.assertEqual(extractor_output.positions.shape, (BATCH_SIZE, MAX_NUM_LANDMARKS, 2, 2))

        self.assertTrue((extractor_output.mask ==
                         torch.tensor([[False, False], [False, True]])).all())

        self.assertTrue((extractor_output.positions[0, 0, :, 1] ==
                         compute_column_from_pano_landmark(loc_1, landmark_1, PANO_WIDTH)).all())
        self.assertTrue((extractor_output.positions[0, 1, :, 1] ==
                         compute_column_from_pano_landmark(loc_1, landmark_2, PANO_WIDTH)).all())
        self.assertTrue((extractor_output.positions[1, 0, :, 1] ==
                         compute_column_from_pano_landmark(loc_2, landmark_1, PANO_WIDTH)).all())
 
    def test_panorama_landmark_extractor_line_string_feature(self):
        # Setup
        BATCH_SIZE = 1
        MAX_NUM_LANDMARKS = 2
        PANO_WIDTH = 512
        # Note that this is row, column
        loc = (0.0, 0.0)
        # Shapely expects column, row
        landmarks = [
            # This is a line that goes from northeast to north
            [(10.0, -10.0), (0.0, -10.0)],
            # This is a line that goes from southwest to south east
            [(-20.0, 20.0), (20.0, 20.0)],
        ]
        model_input = sle.ModelInput(
            image=torch.zeros((BATCH_SIZE, 0, 256, PANO_WIDTH)),
            metadata=[
                dict(
                    web_mercator_y=loc[0],
                    web_mercator_x=loc[1],
                    pano_id="pano_id",
                    landmarks=[
                        dict(
                            geometry_px=shapely.LineString(x),
                            landmark_type="bus_stop") for x in landmarks])])

        config = sle.SemanticLandmarkExtractorConfig()
        model = sle.SemanticLandmarkExtractor(config)

        # Action
        extractor_output = model(model_input)

        # Verification
        self.assertEqual(extractor_output.mask.shape, (BATCH_SIZE, MAX_NUM_LANDMARKS))
        self.assertEqual(extractor_output.features.shape,
                         (BATCH_SIZE, MAX_NUM_LANDMARKS, model.output_dim))
        self.assertEqual(extractor_output.positions.shape, (BATCH_SIZE, MAX_NUM_LANDMARKS, 2, 2))

        self.assertTrue((extractor_output.mask ==
                         torch.tensor([[False, False]])).all())

        self.assertAlmostEqual(extractor_output.positions[0, 0, 0, 1].item(), PANO_WIDTH * 0.5, places=3)
        self.assertAlmostEqual(extractor_output.positions[0, 0, 1, 1].item(), PANO_WIDTH * 5.0 / 8.0, places=3)

        self.assertAlmostEqual(extractor_output.positions[0, 1, 0, 1].item(), PANO_WIDTH * 7.0 / 8.0, places=3)
        self.assertAlmostEqual(extractor_output.positions[0, 1, 1, 1].item(), PANO_WIDTH * 1.0 / 8.0, places=3)

    def test_panorama_landmark_extractor_polygon_feature(self):
        # Setup
        BATCH_SIZE = 1
        MAX_NUM_LANDMARKS = 6
        PANO_WIDTH = 512
        # Note that this is row, column
        loc = (0.0, 0.0)
        # Shapely expects column, row
        landmarks = [
            # This is a line that goes from north east to north
            shapely.buffer(
                geometry=shapely.LineString([(10.0, -10.0), (0.0, -10.0)]),
                distance=3, cap_style='flat'),
            # This is a line that goes from south west to south east
            shapely.buffer(
                geometry=shapely.LineString([(-20.0, 20.0), (20.0, 20.0)]),
                distance=3, cap_style='flat'),
            # A bug trap with an exit south
            shapely.buffer(
                geometry=shapely.LineString([(-100.0, 100.0),
                                             (-100.0, -100.0),
                                             (100.0, -100.0),
                                             (100.0, 100.0)]),
                distance=3, cap_style='flat'),
            # A bug trap with an exit north
            shapely.buffer(
                geometry=shapely.LineString([(-100.0, -100.0),
                                             (-100.0, 100.0),
                                             (100.0, 100.0),
                                             (100.0, -100.0)]),
                distance=3, cap_style='flat'),
            # A bug trap that encircles the robot
            shapely.buffer(
                geometry=shapely.LineString([
                    (100, -200),
                    (-100.0, -200.0),
                    (-100.0, 100.0),
                    (100.0, 100.0),
                    (100.0, -100.0),
                    (50.0, -100.0)]),
                distance=3, cap_style='flat'),
            # A bug trap that encircles the robot, crossing the +/- pi boundary multiple times
            shapely.buffer(
                geometry=shapely.LineString([
                    (100, 200),
                    (-100.0, 200.0),
                    (-100.0, -100.0),
                    (100.0, -100.0),
                    (100.0, 100.0),
                    (50.0, 100.0)]),
                distance=3, cap_style='flat'),
        ]
        model_input = sle.ModelInput(
            image=torch.zeros((BATCH_SIZE, 0, 256, PANO_WIDTH)),
            metadata=[
                dict(
                    web_mercator_y=loc[0],
                    web_mercator_x=loc[1],
                    pano_id="pano_id",
                    landmarks=[dict(geometry_px=x, landmark_type="bus_stop") for x in landmarks])])

        config = sle.SemanticLandmarkExtractorConfig()
        model = sle.SemanticLandmarkExtractor(config)

        # Action
        extractor_output = model(model_input)

        # Verification
        self.assertEqual(extractor_output.mask.shape, (BATCH_SIZE, MAX_NUM_LANDMARKS))
        self.assertEqual(extractor_output.positions.shape, (BATCH_SIZE, MAX_NUM_LANDMARKS, 2, 2))

        # Check line from north east to north
        self.assertAlmostEqual(extractor_output.positions[0, 0, 0, 1].item(), PANO_WIDTH / 2.0, places=-2)
        self.assertAlmostEqual(extractor_output.positions[0, 0, 1, 1].item(), PANO_WIDTH * 5.0 / 8.0, places=-2)

        # Check line from south west to south east
        self.assertAlmostEqual(extractor_output.positions[0, 1, 0, 1].item(), PANO_WIDTH * 7.0 / 8.0, places=-2)
        self.assertAlmostEqual(extractor_output.positions[0, 1, 1, 1].item(), PANO_WIDTH * 1.0 / 8.0, places=-2)

        # Check bug trap with exit south
        self.assertAlmostEqual(extractor_output.positions[0, 2, 0, 1].item(), PANO_WIDTH * 1.0 / 8.0, places=-2)
        self.assertAlmostEqual(extractor_output.positions[0, 2, 1, 1].item(), PANO_WIDTH * 7.0 / 8.0, places=-2)

        # Check bug trap with exit north
        self.assertAlmostEqual(extractor_output.positions[0, 3, 0, 1].item(), PANO_WIDTH * 5.0 / 8.0, places=-2)
        self.assertAlmostEqual(extractor_output.positions[0, 3, 1, 1].item(), PANO_WIDTH * 3.0 / 8.0, places=-2)

        # Check encircling bug trap
        self.assertAlmostEqual((
            extractor_output.positions[0, 4, 1, 1] - extractor_output.positions[0, 4, 0, 1]).item(),
            PANO_WIDTH, places=3)
        self.assertAlmostEqual((
            extractor_output.positions[0, 5, 1, 1] - extractor_output.positions[0, 5, 0, 1]).item(),
            PANO_WIDTH, places=3)


if __name__ == "__main__":
    unittest.main()
