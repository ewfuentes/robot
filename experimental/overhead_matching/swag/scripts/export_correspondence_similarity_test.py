import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import common.torch.load_torch_deps  # noqa: F401
import numpy as np
import pandas as pd
import torch

from experimental.overhead_matching.swag.evaluation import correspondence_matching as cm
from experimental.overhead_matching.swag.scripts import (
    export_correspondence_similarity,
)


class ExportCorrespondenceSimilarityTest(unittest.TestCase):
    @staticmethod
    def _raw(cost_matrix, cost_matrix_path=None):
        return cm.RawCorrespondenceData(
            cost_matrix=cost_matrix,
            cost_matrix_path=cost_matrix_path,
            pano_id_to_lm_rows={"p0": [0]},
            pano_lm_tags=[["amenity=cafe"]],
            osm_lm_indices=[0, 1],
            osm_lm_tags=[{"amenity": "cafe"}, {"amenity": "school"}],
        )

    @staticmethod
    def _dataset():
        return SimpleNamespace(
            _panorama_metadata=pd.DataFrame({"pano_id": ["f0000"]}),
            _satellite_metadata=pd.DataFrame({
                "path": [Path("satellite_1_2.jpg")],
                "landmark_idxs": [[0, 1]],
            }),
            _config=export_correspondence_similarity.vd.VigorDatasetConfig(
                satellite_tensor_cache_info=None,
                panorama_tensor_cache_info=None,
                landmark_correspondence_inflation_factor=1.0,
            ),
        )

    def test_external_paths_are_passed_to_vigor_config(self):
        dataset_path = Path("/datasets/leg1")
        satellite_dir = Path("/artifacts/satellite")
        landmark_path = Path("/artifacts/landmarks.feather")
        sentinel = object()

        with mock.patch.object(
                export_correspondence_similarity,
                "auto_detect_landmark_version") as auto_detect, mock.patch.object(
                    export_correspondence_similarity.vd, "VigorDataset",
                    return_value=sentinel) as constructor:
            result = export_correspondence_similarity.load_vigor_dataset(
                dataset_path, None, 1.25, satellite_dir, landmark_path)

        self.assertIs(result, sentinel)
        auto_detect.assert_not_called()
        config = constructor.call_args.args[1]
        self.assertEqual(config.satellite_dir, satellite_dir)
        self.assertEqual(config.landmark_path, landmark_path)
        self.assertEqual(config.landmark_correspondence_inflation_factor, 1.25)
        self.assertIsNone(config.satellite_tensor_cache_info)
        self.assertIsNone(config.panorama_tensor_cache_info)

    def test_legacy_landmark_auto_detection_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset_path = Path(temporary) / "LegacyCity"
            landmarks_dir = dataset_path / "landmarks"
            landmarks_dir.mkdir(parents=True)
            (landmarks_dir / "legacy_v3.feather").touch()

            with mock.patch.object(
                    export_correspondence_similarity.vd,
                    "VigorDataset") as constructor:
                export_correspondence_similarity.load_vigor_dataset(
                    dataset_path, None, 1.0)

            config = constructor.call_args.args[1]
            self.assertEqual(config.landmark_version, "legacy_v3")
            self.assertIsNone(config.satellite_dir)
            self.assertIsNone(config.landmark_path)

    def test_multiple_vlm_roots_reject_duplicate_panorama_ids(self):
        with mock.patch.object(
                export_correspondence_similarity,
                "extract_panorama_data_across_cities",
                side_effect=[{"f0000": []}, {"f0000": []}]):
            with self.assertRaisesRegex(
                    ValueError, "duplicates panorama ID 'f0000'"):
                export_correspondence_similarity.load_panorama_tags([
                    Path("/artifacts/leg1"), Path("/artifacts/leg2")])

    def test_streamed_raw_path_is_relative_and_survives_publication_rename(self):
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary) / "artifact.incomplete"
            staging.mkdir()
            output_path = staging / "raw.pt"
            cost_path = staging / "raw_cost_matrix.npy"
            expected = np.array([[0.25, 0.75]], dtype=np.float32)
            np.save(cost_path, expected)

            export_correspondence_similarity.save_raw_cost_data(
                self._raw(np.load(cost_path, mmap_mode="r"), cost_path),
                output_path,
                Path("/models/classifier.pt"),
                Path("/embeddings/text.pkl"),
            )
            metadata = torch.load(output_path, weights_only=False)
            self.assertEqual(metadata["cost_matrix_path"], "raw_cost_matrix.npy")

            published = staging.with_name("artifact")
            staging.rename(published)
            loaded = export_correspondence_similarity.load_raw_cost_data(
                published / "raw.pt")
            self.assertTrue(np.array_equal(loaded.cost_matrix, expected))
            self.assertEqual(
                loaded.cost_matrix_path, published / "raw_cost_matrix.npy")

    def test_loader_falls_back_for_existing_absolute_staging_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            published = Path(temporary) / "artifact"
            published.mkdir()
            raw_path = published / "raw.pt"
            expected = np.array([[0.1, 0.9]], dtype=np.float32)
            np.save(published / "raw_cost_matrix.npy", expected)
            metadata = {
                "cost_matrix_path": str(
                    published.with_name("artifact.incomplete")
                    / "raw_cost_matrix.npy"),
                "pano_id_to_lm_rows": {"p0": [0]},
                "pano_lm_tags": [["amenity=cafe"]],
                "osm_lm_indices": [0, 1],
                "osm_lm_tags": [
                    {"amenity": "cafe"}, {"amenity": "school"}],
            }
            torch.save(metadata, raw_path)

            loaded = export_correspondence_similarity.load_raw_cost_data(raw_path)
            self.assertTrue(np.array_equal(loaded.cost_matrix, expected))

    def test_streamed_matrix_outside_artifact_keeps_absolute_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "artifact"
            output_dir.mkdir()
            output_path = output_dir / "raw.pt"
            cost_path = root / "matrix-cache" / "scores.npy"
            cost_path.parent.mkdir()
            expected = np.array([[0.2, 0.8]], dtype=np.float32)
            np.save(cost_path, expected)

            export_correspondence_similarity.save_raw_cost_data(
                self._raw(np.load(cost_path, mmap_mode="r"), cost_path),
                output_path,
                Path("/models/classifier.pt"),
                Path("/embeddings/text.pkl"),
            )

            metadata = torch.load(output_path, weights_only=False)
            self.assertEqual(metadata["cost_matrix_path"], str(cost_path))
            loaded = export_correspondence_similarity.load_raw_cost_data(
                output_path)
            self.assertTrue(np.array_equal(loaded.cost_matrix, expected))

    def test_raw_identity_survives_save_and_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "raw.pt"
            raw = self._raw(np.array([[0.2, 0.8]], dtype=np.float32))
            raw.identity = {
                "schema": export_correspondence_similarity.RAW_CORRESPONDENCE_IDENTITY_SCHEMA,
                "aggregation_inputs": {"mapping": "digest"},
                "raw_payload": {"metadata": "digest"},
                "inference_inputs": {"model": "digest"},
            }
            export_correspondence_similarity.save_raw_cost_data(
                raw, output_path, Path("/model.pt"), Path("/text.pkl"))

            loaded = export_correspondence_similarity.load_raw_cost_data(
                output_path)

            self.assertEqual(loaded.identity, raw.identity)

    def test_raw_identity_rejects_wrong_leg_with_same_panorama_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            live = root / "live_leg"
            wrong = root / "wrong_leg"
            live.mkdir()
            wrong.mkdir()
            (live / "pano_id_mapping.csv").write_text(
                "pano_id,lat,lon\nf0000,42.0,-71.0\n")
            (wrong / "pano_id_mapping.csv").write_text(
                "pano_id,lat,lon\nf0000,44.0,-71.0\n")
            landmark_path = root / "landmarks.feather"
            landmark_path.write_bytes(b"same OSM source")
            dataset = self._dataset()
            raw = self._raw(np.array([[0.2, 0.8]], dtype=np.float32))
            raw.identity = {
                "schema": export_correspondence_similarity.RAW_CORRESPONDENCE_IDENTITY_SCHEMA,
                "aggregation_inputs": (
                    export_correspondence_similarity._raw_aggregation_identity(
                        dataset, wrong, landmark_path)),
                "raw_payload": (
                    export_correspondence_similarity._raw_payload_identity(raw)),
                "inference_inputs": {},
            }

            with self.assertRaisesRegex(
                    ValueError, "different dataset/OSM/satellite"):
                export_correspondence_similarity.validate_raw_identity(
                    raw, dataset, live, landmark_path,
                    allow_legacy=True, require_identity=True)

    def test_legacy_raw_identity_requires_explicit_opt_in(self):
        raw = self._raw(np.array([[0.2, 0.8]], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "allow_legacy_raw_identity"):
            export_correspondence_similarity.validate_raw_identity(
                raw, None, Path("unused"), Path("unused"),
                require_identity=True)
        with self.assertWarnsRegex(RuntimeWarning, "no source/alignment"):
            export_correspondence_similarity.validate_raw_identity(
                raw, None, Path("unused"), Path("unused"),
                allow_legacy=True, require_identity=True)

    def test_raw_identity_rejects_same_shape_value_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset_path = Path(temporary)
            (dataset_path / "pano_id_mapping.csv").write_text(
                "pano_id,lat,lon\nf0000,42.0,-71.0\n")
            landmark_path = dataset_path / "landmarks.feather"
            landmark_path.write_bytes(b"OSM source")
            dataset = self._dataset()
            raw = self._raw(np.array([[0.2, 0.8]], dtype=np.float32))
            raw.identity = {
                "schema": export_correspondence_similarity.RAW_CORRESPONDENCE_IDENTITY_SCHEMA,
                "aggregation_inputs": (
                    export_correspondence_similarity._raw_aggregation_identity(
                        dataset, dataset_path, landmark_path)),
                "raw_payload": (
                    export_correspondence_similarity._raw_payload_identity(raw)),
                "inference_inputs": {},
            }
            raw.cost_matrix[0, 0] = 0.9

            with self.assertRaisesRegex(
                    ValueError, "values or metadata"):
                export_correspondence_similarity.validate_raw_identity(
                    raw, dataset, dataset_path, landmark_path,
                    require_identity=True)


if __name__ == "__main__":
    unittest.main()
