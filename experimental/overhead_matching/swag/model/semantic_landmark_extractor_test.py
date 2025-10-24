
import unittest
import tempfile
import json

import common.torch.load_torch_deps
import torch
from pathlib import Path
import math
import shapely

import experimental.overhead_matching.swag.model.semantic_landmark_extractor as sle
import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.model.swag_config_types import LandmarkType
from experimental.overhead_matching.swag.model.semantic_landmark_extractor import (
    prune_landmark, _custom_id_from_props as custom_id_from_props
)


def compute_column_from_pano_landmark(pano_loc, landmark_loc, pano_width):
    dx = landmark_loc[1] - pano_loc[1]
    dy = landmark_loc[0] - pano_loc[0]
    theta = math.atan2(dx, -dy)
    frac = (theta + math.pi) / (2 * math.pi)
    return frac * pano_width


def create_embedding_response(custom_id, embedding):
    """Create an OpenAI batch API response in the expected format"""
    return {
        "id": f"batch_req_{custom_id[:16]}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": "test_request",
            "body": {
                "object": "list",
                "data": [{
                    "object": "embedding",
                    "index": 0,
                    "embedding": embedding
                }],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 1, "total_tokens": 1}
            }
        },
        "error": None
    }


class SemanticLandmarkExtractorWithDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dataset = vd.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet/"),
            vd.VigorDatasetConfig(
                satellite_tensor_cache_info=None, panorama_tensor_cache_info=None))

        # Create test embedding directory
        cls._temp_dir = tempfile.TemporaryDirectory()
        embedding_dir = Path(cls._temp_dir.name) / "test_version" / "embeddings"
        embedding_dir.mkdir(parents=True, exist_ok=True)

        # Create test embedding file with actual format from OpenAI batch API
        # The test landmarks only have geometry_px and landmark_type keys, which both get
        # dropped by prune_landmark(), so all landmarks have empty props and the same custom_id
        test_custom_id = custom_id_from_props(prune_landmark({}))
        with open(embedding_dir / "test.jsonl", "w") as f:
            f.write(json.dumps(create_embedding_response(test_custom_id, [0.1] * 1536)) + "\n")

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    def test_satellite_landmark_extractor_with_dataset(self):
        # Setup
        BATCH_SIZE = 13
        config = sle.SemanticLandmarkExtractorConfig(
            landmark_type=LandmarkType.POINT,
            openai_embedding_size=1536,
            embedding_version="test_version",
            auxiliary_info_key="test_key")
        model = sle.SemanticLandmarkExtractor(config, Path(self._temp_dir.name))

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
        config = sle.SemanticLandmarkExtractorConfig(
            landmark_type=LandmarkType.POINT,
            openai_embedding_size=1536,
            embedding_version="test_version",
            auxiliary_info_key="test_key")
        model = sle.SemanticLandmarkExtractor(config, Path(self._temp_dir.name))

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
    @classmethod
    def setUpClass(cls):
        # Create test embedding directory
        cls._temp_dir = tempfile.TemporaryDirectory()
        embedding_dir = Path(cls._temp_dir.name) / "test_version" / "embeddings"
        embedding_dir.mkdir(parents=True, exist_ok=True)

        # Define test landmarks with diverse properties that won't be pruned
        cls.test_landmarks = {
            'restaurant': {'name': 'Pizza Palace', 'amenity': 'restaurant', 'cuisine': 'italian'},
            'bus_stop': {'name': 'Main St & 5th Ave', 'highway': 'bus_stop'},
            'bank': {'name': 'First National Bank', 'amenity': 'bank', 'building': 'yes'},
            'church': {'name': 'St. Mary Church', 'amenity': 'place_of_worship', 'religion': 'christian'},
            'empty': {},  # Empty landmark (all props get pruned)
        }

        # Create embeddings for each landmark with distinct values
        cls.landmark_embeddings = {}
        for idx, (key, props) in enumerate(cls.test_landmarks.items()):
            pruned_props = prune_landmark(props)
            custom_id = custom_id_from_props(pruned_props)
            # Create a unique embedding - use idx to make them different
            embedding = [0.1 * (idx + 1)] * 1536
            cls.landmark_embeddings[key] = {
                'custom_id': custom_id,
                'pruned_props': pruned_props,
                'embedding': embedding
            }

        # Write all embeddings to a single JSONL file
        with open(embedding_dir / "test.jsonl", "w") as f:
            for key, info in cls.landmark_embeddings.items():
                response = create_embedding_response(info['custom_id'], info['embedding'])
                f.write(json.dumps(response) + "\n")

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    def test_embedding_loading(self):
        """Test that embeddings are loaded correctly from files"""
        config = sle.SemanticLandmarkExtractorConfig(
            landmark_type=LandmarkType.POINT,
            openai_embedding_size=1536,
            embedding_version="test_version",
            auxiliary_info_key="test_key")
        model = sle.SemanticLandmarkExtractor(config, Path(self._temp_dir.name))
        model.load_files()

        # Verify all embeddings are loaded
        self.assertEqual(model.all_embeddings_tensor.shape[0], len(self.landmark_embeddings))

        # Verify each embedding matches what we created
        for key, info in self.landmark_embeddings.items():
            self.assertIn(info['custom_id'], model.landmark_id_to_idx)
            loaded_embedding = model.all_embeddings_tensor[
                    model.landmark_id_to_idx[info['custom_id']]]
            expected_embedding = torch.tensor(info['embedding'])
            expected_embedding = expected_embedding / torch.linalg.norm(expected_embedding)
            self.assertAlmostEqual(torch.linalg.norm(loaded_embedding - expected_embedding).item(),
                                   0.0, places=4)

    def test_correct_embeddings_fetched_for_landmarks(self):
        """Test that the correct embeddings are fetched for specific landmark properties"""
        BATCH_SIZE = 2
        loc_1 = (123.0, 456.0)
        loc_2 = (987.0, 654.0)
        landmark_1 = (1000.0, 2000.0)
        landmark_2 = (125.0, 458.0)

        # Create metadata with landmarks that match our test data
        restaurant_props = self.test_landmarks['restaurant'].copy()
        restaurant_geom = shapely.Point(*landmark_1[::-1])
        restaurant_props['geometry'] = restaurant_geom
        restaurant_props['geometry_px'] = restaurant_geom
        restaurant_props["pruned_props"] = prune_landmark(restaurant_props)

        bank_props = self.test_landmarks['bank'].copy()
        bank_geom = shapely.Point(*landmark_2[::-1])
        bank_props['geometry'] = bank_geom
        bank_props['geometry_px'] = bank_geom
        bank_props["pruned_props"] = prune_landmark(bank_props)

        church_props = self.test_landmarks['church'].copy()
        church_geom = shapely.Point(*landmark_1[::-1])
        church_props['geometry'] = church_geom
        church_props['geometry_px'] = church_geom
        church_props["pruned_props"] = prune_landmark(church_props)

        model_input = sle.ModelInput(
            image=torch.zeros((BATCH_SIZE, 0, 256, 256)),
            metadata=[
                dict(
                    web_mercator_y=loc_1[0],
                    web_mercator_x=loc_1[1],
                    landmarks=[restaurant_props, bank_props]),
                dict(
                    web_mercator_y=loc_2[0],
                    web_mercator_x=loc_2[1],
                    landmarks=[church_props]),
            ],
        )

        config = sle.SemanticLandmarkExtractorConfig(
            landmark_type=LandmarkType.POINT,
            openai_embedding_size=1536,
            embedding_version="test_version",
            auxiliary_info_key="test_key")
        model = sle.SemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # Action
        extractor_output = model(model_input)

        # Verification - check that correct embeddings were fetched
        # First item has restaurant and bank
        restaurant_embedding = torch.tensor(self.landmark_embeddings['restaurant']['embedding'])
        restaurant_embedding = restaurant_embedding / torch.norm(restaurant_embedding)
        bank_embedding = torch.tensor(self.landmark_embeddings['bank']['embedding'])
        bank_embedding = bank_embedding / torch.norm(bank_embedding)

        # Second item has church
        church_embedding = torch.tensor(self.landmark_embeddings['church']['embedding'])
        church_embedding = church_embedding / torch.norm(church_embedding)

        # Check embeddings match (they should be normalized)
        self.assertTrue(torch.allclose(extractor_output.features[0, 0], restaurant_embedding, atol=1e-6))
        self.assertTrue(torch.allclose(extractor_output.features[0, 1], bank_embedding, atol=1e-6))
        self.assertTrue(torch.allclose(extractor_output.features[1, 0], church_embedding, atol=1e-6))

    def test_satellite_landmark_extractor(self):
        # Setup
        BATCH_SIZE = 2
        MAX_NUM_LANDMARKS = 2
        loc_1 = (123.0, 456.0)
        loc_2 = (987.0, 654.0)
        landmark_1 = (1000.0, 2000.0)
        landmark_2 = (125.0, 458.0)

        # Use landmarks with empty props (gets pruned to empty dict)
        geom_1_1 = shapely.Point(*landmark_1[::-1])
        geom_1_2 = shapely.Point(*landmark_2[::-1])
        geom_2_1 = shapely.Point(*landmark_1[::-1])

        model_input = sle.ModelInput(
            image=torch.zeros((BATCH_SIZE, 0, 256, 256)),
            metadata=[
                dict(
                    web_mercator_y=loc_1[0],
                    web_mercator_x=loc_1[1],
                    landmarks=[
                        dict(
                            geometry=geom_1_1,
                            geometry_px=geom_1_1,
                            landmark_type="bus_stop",
                            pruned_props=frozenset()),
                        dict(
                            geometry=geom_1_2,
                            geometry_px=geom_1_2,
                            landmark_type="restaurants",
                            pruned_props=frozenset())]),
                dict(
                    web_mercator_y=loc_2[0],
                    web_mercator_x=loc_2[1],
                    landmarks=[
                        dict(
                            geometry=geom_2_1,
                            geometry_px=geom_2_1,
                            landmark_type="places_of_worship",
                            pruned_props={})]),
            ],
        )
        config = sle.SemanticLandmarkExtractorConfig(
            landmark_type=LandmarkType.POINT,
            openai_embedding_size=1536,
            embedding_version="test_version",
            auxiliary_info_key="test_key")
        model = sle.SemanticLandmarkExtractor(config, Path(self._temp_dir.name))

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

        geom_1_1 = shapely.Point(landmark_1[::-1])
        geom_1_2 = shapely.Point(landmark_2[::-1])
        geom_2_1 = shapely.Point(*landmark_1[::-1])

        model_input = sle.ModelInput(
            image=torch.zeros((BATCH_SIZE, 0, 256, PANO_WIDTH)),
            metadata=[
                dict(
                    web_mercator_y=loc_1[0],
                    web_mercator_x=loc_1[1],
                    pano_id="pano_id",
                    landmarks=[
                        dict(
                            geometry=geom_1_1,
                            geometry_px=geom_1_1,
                            landmark_type="bus_stop",
                            pruned_props={}),
                        dict(
                            geometry=geom_1_2,
                            geometry_px=geom_1_2,
                            landmark_type="restaurants",
                            pruned_props={})]),
                dict(
                    web_mercator_y=loc_2[0],
                    web_mercator_x=loc_2[1],
                    landmarks=[
                        dict(
                            geometry=geom_2_1,
                            geometry_px=geom_2_1,
                            landmark_type="places_of_worship",
                            pruned_props={})]),
            ],
        )
        config = sle.SemanticLandmarkExtractorConfig(
            landmark_type=LandmarkType.POINT,
            openai_embedding_size=1536,
            embedding_version="test_version",
            auxiliary_info_key="test_key")
        model = sle.SemanticLandmarkExtractor(config, Path(self._temp_dir.name))

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
        landmark_geoms = [shapely.LineString(x) for x in landmarks]

        model_input = sle.ModelInput(
            image=torch.zeros((BATCH_SIZE, 0, 256, PANO_WIDTH)),
            metadata=[
                dict(
                    web_mercator_y=loc[0],
                    web_mercator_x=loc[1],
                    pano_id="pano_id",
                    landmarks=[
                        dict(
                            geometry=geom,
                            geometry_px=geom,
                            landmark_type="bus_stop",
                            pruned_props={}) for geom in landmark_geoms])])

        config = sle.SemanticLandmarkExtractorConfig(
            landmark_type=LandmarkType.LINESTRING,
            openai_embedding_size=1536,
            embedding_version="test_version",
            auxiliary_info_key="test_key")
        model = sle.SemanticLandmarkExtractor(config, Path(self._temp_dir.name))

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
                    landmarks=[dict(geometry=x, geometry_px=x, landmark_type="bus_stop", pruned_props={}) for x in landmarks])])

        config = sle.SemanticLandmarkExtractorConfig(
            landmark_type=LandmarkType.POLYGON,
            openai_embedding_size=1536,
            embedding_version="test_version",
            auxiliary_info_key="test_key")
        model = sle.SemanticLandmarkExtractor(config, Path(self._temp_dir.name))

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

    def test_embedding_size_cropping(self):
        """Test that embeddings can be cropped to a smaller size"""
        SMALLER_SIZE = 512
        config = sle.SemanticLandmarkExtractorConfig(
            landmark_type=LandmarkType.POINT,
            openai_embedding_size=SMALLER_SIZE,
            embedding_version="test_version",
            auxiliary_info_key="test_key")
        model = sle.SemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # Create a simple input with one landmark
        restaurant_props = self.test_landmarks['restaurant'].copy()
        geom = shapely.Point(100.0, 200.0)
        restaurant_props['geometry'] = geom
        restaurant_props['geometry_px'] = geom
        restaurant_props['pruned_props'] = prune_landmark(restaurant_props)

        model_input = sle.ModelInput(
            image=torch.zeros((1, 0, 256, 256)),
            metadata=[dict(
                web_mercator_y=0.0,
                web_mercator_x=0.0,
                landmarks=[restaurant_props])],
        )

        extractor_output = model(model_input)

        # Verify the output has the cropped size
        self.assertEqual(extractor_output.features.shape[2], SMALLER_SIZE)

        # Verify the embedding is normalized
        embedding_norm = torch.norm(extractor_output.features[0, 0])
        self.assertAlmostEqual(embedding_norm.item(), 1.0, places=5)

    def test_missing_embedding_warning(self):
        """Test that missing embeddings produce a warning but don't crash"""
        config = sle.SemanticLandmarkExtractorConfig(
            landmark_type=LandmarkType.POINT,
            openai_embedding_size=1536,
            embedding_version="test_version",
            auxiliary_info_key="test_key")
        model = sle.SemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # Create a landmark with properties that don't have an embedding
        geom = shapely.Point(100.0, 200.0)
        missing_props = {
            'name': 'Unknown Landmark',
            'amenity': 'unknown_type',
            'geometry': geom,
            'geometry_px': geom,
            'pruned_props': frozenset({("Unknown", "Prop")})
        }

        model_input = sle.ModelInput(
            image=torch.zeros((1, 0, 256, 256)),
            metadata=[dict(
                web_mercator_y=0.0,
                web_mercator_x=0.0,
                landmarks=[missing_props])],
        )

        # Should not crash, but landmark should be masked out
        extractor_output = model(model_input)

        # The landmark should be masked (True = invalid/masked)
        self.assertTrue(extractor_output.mask[0, 0].item())


if __name__ == "__main__":
    unittest.main()
