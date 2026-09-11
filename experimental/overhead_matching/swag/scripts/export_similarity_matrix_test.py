import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest import mock

import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.scripts import export_similarity_matrix


class ExportSimilarityMatrixTest(unittest.TestCase):
    def test_published_artifact_path_removes_staging_suffix(self):
        staged = Path("/tmp/scope/version.incomplete/satellite_embeddings.pt")
        self.assertEqual(
            export_similarity_matrix._published_artifact_path(staged),
            Path("/tmp/scope/version/satellite_embeddings.pt"),
        )
        final = Path("/tmp/scope/version/satellite_embeddings.pt")
        self.assertEqual(
            export_similarity_matrix._published_artifact_path(final),
            final,
        )

    def test_file_hash_binds_content_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.jpg"
            second = Path(temporary) / "second.jpg"
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            digest = export_similarity_matrix._hash_ordered_files(
                [first, second])
            self.assertNotEqual(
                digest,
                export_similarity_matrix._hash_ordered_files(
                    [second, first]))
            first.write_bytes(b"FIRST")
            self.assertNotEqual(
                digest,
                export_similarity_matrix._hash_ordered_files(
                    [first, second]))

    def test_panorama_identity_binds_images_model_and_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.jpg"
            second = root / "second.jpg"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            model = torch.nn.Linear(2, 1)
            model_path = root / "best_panorama"
            config = {"kind": "test", "normalize": True}

            identity = export_similarity_matrix._panorama_identity(
                [first, second], model, model_path, config)
            self.assertEqual(
                identity["panorama_filenames"], ["first.jpg", "second.jpg"])

            first.write_bytes(b"changed")
            changed_data = export_similarity_matrix._panorama_identity(
                [first, second], model, model_path, config)
            self.assertNotEqual(
                identity["panorama_files_sha256"],
                changed_data["panorama_files_sha256"],
            )

            with torch.no_grad():
                model.bias.add_(1)
            changed_model = export_similarity_matrix._panorama_identity(
                [first, second], model, model_path, config)
            self.assertNotEqual(
                changed_data["panorama_model_sha256"],
                changed_model["panorama_model_sha256"],
            )

            changed_config = export_similarity_matrix._panorama_identity(
                [first, second], model, model_path,
                {"kind": "test", "normalize": False})
            self.assertNotEqual(
                changed_model["panorama_model_config_sha256"],
                changed_config["panorama_model_config_sha256"],
            )

    def test_satellite_embedding_cache_validates_filename_order_and_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "satellite_embeddings.pt"
            filenames = ["satellite_a.jpg", "satellite_b.jpg"]
            identity = {
                "satellite_filenames": filenames,
                "satellite_files_sha256": "files-sha256",
                "satellite_model_path": "/models/best_satellite",
                "satellite_model_sha256": "model-sha256",
                "satellite_behavior": {
                    "model_config_sha256": "config-sha256",
                    "dataset_preprocessing": {"resize_shape": [640, 640]},
                    "tag_text_embeddings_override_sha256": "tags-sha256",
                },
            }
            embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            torch.save({
                "schema": export_similarity_matrix.SATELLITE_EMBEDDINGS_SCHEMA,
                **identity,
                "embeddings": embeddings,
            }, path)

            loaded = export_similarity_matrix.load_satellite_embeddings(
                path, identity)
            self.assertTrue(torch.equal(loaded, embeddings))
            self.assertEqual(loaded.device.type, "cpu")

            # Model location is provenance only; relocating identical model
            # behavior and weights must not invalidate an expensive cache.
            relocated = {
                **identity,
                "satellite_model_path": "/relocated/best_satellite",
            }
            loaded = export_similarity_matrix.load_satellite_embeddings(
                path, relocated)
            self.assertTrue(torch.equal(loaded, embeddings))

            for key, replacement in (
                    ("satellite_filenames", list(reversed(filenames))),
                    ("satellite_files_sha256", "different-files"),
                    ("satellite_model_sha256", "different-model"),
                    ("satellite_behavior", {
                        **identity["satellite_behavior"],
                        "tag_text_embeddings_override_sha256": "different-tags",
                    })):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    expected = {**identity, key: replacement}
                    export_similarity_matrix.load_satellite_embeddings(
                        path, expected)

            torch.save({
                "schema": export_similarity_matrix.SATELLITE_EMBEDDINGS_SCHEMA,
                **identity,
                "embeddings": torch.tensor([[float("nan"), 0.0], [0.0, 1.0]]),
            }, path)
            with self.assertRaisesRegex(ValueError, "finite floating-point"):
                export_similarity_matrix.load_satellite_embeddings(path, identity)

    def test_legacy_cache_requires_explicit_opt_in_and_warns(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "satellite_embeddings.pt"
            identity = {
                "satellite_filenames": ["satellite_a.jpg"],
                "satellite_files_sha256": "files-sha256",
                "satellite_model_path": "/new/location/best_satellite",
                "satellite_model_sha256": "model-sha256",
                "satellite_behavior": {"schema": "behavior/v1"},
            }
            torch.save({
                "schema": export_similarity_matrix.LEGACY_SATELLITE_EMBEDDINGS_SCHEMA,
                **{**identity, "satellite_model_path": "/old/location/best_satellite"},
                "satellite_behavior": None,
                "embeddings": torch.ones(1, 2),
            }, path)

            with self.assertRaisesRegex(
                    ValueError, "allow_legacy_satellite_embeddings"):
                export_similarity_matrix.load_satellite_embeddings(path, identity)
            with self.assertWarnsRegex(RuntimeWarning, "unverified"):
                loaded = export_similarity_matrix.load_satellite_embeddings(
                    path, identity, allow_legacy=True)
            self.assertEqual(tuple(loaded.shape), (1, 2))

    def test_atomic_cache_save_does_not_leave_partial_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "satellite_embeddings.pt"
            with mock.patch.object(
                    export_similarity_matrix.torch, "save",
                    side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    export_similarity_matrix._atomic_torch_save({}, path)

            self.assertFalse(path.exists())
            self.assertEqual(list(directory.iterdir()), [])

    def test_unreadable_cache_has_clear_rebuild_instruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "satellite_embeddings.pt"
            path.write_bytes(b"not a torch archive")
            with self.assertRaisesRegex(ValueError, "remove it and rerun"):
                export_similarity_matrix.load_satellite_embeddings(path, {})

    def test_behavior_identity_binds_config_preprocessing_and_tag_content(self):
        model = SimpleNamespace(
            _config={"kind": "test", "normalize": True},
            patch_dims=(640, 640),
        )
        base_config = export_similarity_matrix.vd.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            should_load_images=True,
            should_load_landmarks=False,
        )
        dataset = SimpleNamespace(
            _config=base_config,
            _satellite_patch_size=(640, 640),
        )
        with tempfile.TemporaryDirectory() as temporary:
            tag_embeddings = Path(temporary) / "tags.pkl"
            tag_embeddings.write_bytes(b"first")
            with mock.patch.object(
                    export_similarity_matrix,
                    "_satellite_forward_code_sha256",
                    return_value="forward-code-a") as forward_code:
                identity = export_similarity_matrix._satellite_behavior_identity(
                    model, dataset, set(), [], tag_embeddings)

                forward_code.return_value = "forward-code-b"
                self.assertNotEqual(
                    identity,
                    export_similarity_matrix._satellite_behavior_identity(
                        model, dataset, set(), [], tag_embeddings),
                )
                forward_code.return_value = "forward-code-a"

                model._config = {"kind": "test", "normalize": False}
                self.assertNotEqual(
                    identity,
                    export_similarity_matrix._satellite_behavior_identity(
                        model, dataset, set(), [], tag_embeddings),
                )
                model._config = {"kind": "test", "normalize": True}
                dataset._config = base_config._replace(should_load_landmarks=True)
                self.assertNotEqual(
                    identity,
                    export_similarity_matrix._satellite_behavior_identity(
                        model, dataset, set(), [], tag_embeddings),
                )
                dataset._config = base_config
                tag_embeddings.write_bytes(b"second")
                self.assertNotEqual(
                    identity,
                    export_similarity_matrix._satellite_behavior_identity(
                        model, dataset, set(), [], tag_embeddings),
                )

    def test_real_v2_behavior_identity_round_trips_with_weights_only(self):
        model = SimpleNamespace(
            patch_dims=(640, 640),
        )
        dataset = SimpleNamespace(
            _config=export_similarity_matrix.vd.VigorDatasetConfig(
                satellite_tensor_cache_info=None,
                panorama_tensor_cache_info=None,
                should_load_images=True,
                should_load_landmarks=False,
            ),
            _satellite_patch_size=(640, 640),
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                export_similarity_matrix,
                "_satellite_forward_code_sha256",
                return_value="forward-code"):
            training_output = Path(temporary)
            (training_output / "train_config.yaml").write_text(
                "sat_model_config:\n"
                "  kind: WagPatchEmbeddingConfig\n"
                "  patch_dims: [640, 640]\n"
                "  num_aggregation_heads: 4\n")
            model_config = (
                export_similarity_matrix._load_training_model_config(
                    training_output, "sat_model_config"))
            path = training_output / "satellite_embeddings.pt"
            behavior = export_similarity_matrix._satellite_behavior_identity(
                model, dataset, set(), [], None, model_config)
            identity = {
                "satellite_filenames": ["satellite_a.jpg"],
                "satellite_files_sha256": "files-sha256",
                "satellite_model_path": "/models/best_satellite",
                "satellite_model_sha256": "model-sha256",
                "satellite_behavior": behavior,
            }
            export_similarity_matrix._atomic_torch_save({
                "schema": export_similarity_matrix.SATELLITE_EMBEDDINGS_SCHEMA,
                **identity,
                "embeddings": torch.ones(1, 2),
            }, path)

            loaded = export_similarity_matrix.load_satellite_embeddings(
                path, identity)

            self.assertEqual(tuple(loaded.shape), (1, 2))
            self.assertIs(type(behavior["runtime_versions"]["torch"]), str)
            self.assertIs(
                type(behavior["runtime_versions"]["torchvision"]), str)

    def test_behavior_identity_requires_config_for_legacy_model(self):
        model = SimpleNamespace(patch_dims=(640, 640))
        dataset = SimpleNamespace(
            _config=export_similarity_matrix.vd.VigorDatasetConfig(
                satellite_tensor_cache_info=None,
                panorama_tensor_cache_info=None,
            ),
            _satellite_patch_size=(640, 640),
        )
        with self.assertRaisesRegex(ValueError, "no embedded config"):
            export_similarity_matrix._satellite_behavior_identity(
                model, dataset, set(), [], None)


if __name__ == "__main__":
    unittest.main()
