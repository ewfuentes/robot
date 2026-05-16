import unittest
from types import SimpleNamespace

import common.torch.load_torch_deps  # noqa: F401
import torch
from shapely.geometry import Point

from experimental.overhead_matching.swag.model.swag_config_types import (
    LandmarkType,
    OSMTagBundleExtractorConfig,
    PanoramaTagBundleExtractorConfig,
    TagBundleEncoderConfigStruct,
)
from experimental.overhead_matching.swag.model.swag_model_input_output import ModelInput
from experimental.overhead_matching.swag.model.tag_bundle_extractor import (
    OSMTagBundleExtractor,
    PanoramaTagBundleExtractor,
)


def _make_encoder_cfg() -> TagBundleEncoderConfigStruct:
    return TagBundleEncoderConfigStruct(
        key_dim=4, text_input_dim=8, text_proj_dim=4, per_tag_dim=6,
    )


class OSMTagBundleExtractorTest(unittest.TestCase):
    def test_forward_produces_per_landmark_tokens(self):
        cfg = OSMTagBundleExtractorConfig(
            landmark_type=LandmarkType.POINT,
            encoder=_make_encoder_cfg(),
            tag_text_embedding_path="/tmp/ignored.pkl",
        )
        extractor = OSMTagBundleExtractor(cfg)
        # Inject a stub embeddings dict instead of loading from disk.
        extractor._text_embeddings = {
            "Starbucks": torch.ones(8) * 0.1,
            "park": torch.ones(8) * 0.2,
        }
        extractor._files_loaded = True

        landmarks = [
            {"geometry": Point(0, 0), "geometry_px": Point(2.0, 3.0),
             "pruned_props": {"name": "Starbucks", "amenity": "park"}},
            {"geometry": Point(0, 0), "geometry_px": Point(4.0, 5.0),
             "pruned_props": {"name": "Starbucks"}},  # one tag has no embedding match
        ]
        empty_landmarks = [
            {"geometry": Point(0, 0), "geometry_px": Point(7.0, 8.0),
             "pruned_props": {}},
        ]
        metadata = [
            {"web_mercator_y": 1.0, "web_mercator_x": 1.0, "landmarks": landmarks},
            {"web_mercator_y": 0.0, "web_mercator_x": 0.0, "landmarks": empty_landmarks},
        ]
        image = torch.zeros((2, 3, 8, 8))
        out = extractor(ModelInput(image=image, metadata=metadata))

        self.assertEqual(out.features.shape, (2, 2, extractor.output_dim))
        self.assertEqual(out.positions.shape, (2, 2, 2, 2))
        # First batch: both landmarks valid → mask all False.
        self.assertFalse(out.mask[0, 0].item())
        self.assertFalse(out.mask[0, 1].item())
        # Second batch: only one (empty-tags) landmark → it's still emitted but the
        # second slot is padded.
        self.assertTrue(out.mask[1, 1].item())
        # First-batch positions should be (geometry_px.y - sat_y, geometry_px.x - sat_x).
        # Landmark 0 has Point(x=2, y=3), sat at (web_mercator_y=1, web_mercator_x=1)
        # → [[3-1, 2-1], [3-1, 2-1]] = [[2, 1], [2, 1]]
        torch.testing.assert_close(
            out.positions[0, 0], torch.tensor([[2.0, 1.0], [2.0, 1.0]]))


class PanoramaTagBundleExtractorTest(unittest.TestCase):
    def test_forward_uses_panov2_pickle_lookup(self):
        cfg = PanoramaTagBundleExtractorConfig(
            encoder=_make_encoder_cfg(),
            tag_text_embedding_path="/tmp/ignored.pkl",
            panov2_root="/tmp/panov2_root",
        )
        extractor = PanoramaTagBundleExtractor(cfg)
        extractor._text_embeddings = {
            "Starbucks": torch.ones(8) * 0.5,
            "shop": torch.ones(8) * 0.25,
        }
        extractor._panorama_landmarks = {
            "panoA": [
                {"primary_tag": {"key": "shop", "value": "shop"},
                 "additional_tags": [{"key": "name", "value": "Starbucks"}],
                 "bounding_boxes": [{"yaw_angle": "0", "ymin": 0, "xmin": 0,
                                     "ymax": 100, "xmax": 100}],
                 "landmark_idx": 0},
                {"primary_tag": {"key": "name", "value": "Starbucks"},
                 "additional_tags": [],
                 "bounding_boxes": [{"yaw_angle": "180", "ymin": 0, "xmin": 0,
                                     "ymax": 50, "xmax": 50}],
                 "landmark_idx": 1},
            ],
        }
        extractor._files_loaded = True

        metadata = [
            {"pano_id": "panoA"},
            {"pano_id": "panoMissing"},
        ]
        image = torch.zeros((2, 3, 16, 32))
        out = extractor(ModelInput(image=image, metadata=metadata))

        self.assertEqual(out.features.shape, (2, 2, extractor.output_dim))
        self.assertEqual(out.positions.shape, (2, 2, 2, 2))
        # Batch 0 has 2 landmarks, both valid (mask=False).
        self.assertFalse(out.mask[0, 0].item())
        self.assertFalse(out.mask[0, 1].item())
        # Batch 1 has no landmarks → both slots padded.
        self.assertTrue(out.mask[1, 0].item())
        self.assertTrue(out.mask[1, 1].item())

        # Yaw 0 → position (1, 0, 0, 0); yaw 180 → (0, 0, 1, 0).
        torch.testing.assert_close(
            out.positions[0, 0],
            torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
        torch.testing.assert_close(
            out.positions[0, 1],
            torch.tensor([[0.0, 0.0], [1.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
