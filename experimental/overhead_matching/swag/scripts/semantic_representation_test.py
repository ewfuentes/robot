"""Regression coverage for checkpoint compatibility and the portable ablation workflow."""

import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import common.torch.load_torch_deps  # noqa: F401
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    FixedEmbeddingClassifier,
)
from experimental.overhead_matching.swag.scripts import (
    semantic_paper_eval as evaluation,
)
from experimental.overhead_matching.swag.scripts import (
    train_representation_ablation as training,
)
from experimental.overhead_matching.swag.scripts.train_landmark_correspondence import (
    train_epoch,
    evaluate,
)


class SemanticRepresentationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def legacy_checkpoint(self):
        # Independent definition from the archived trainer, including non-default BN state.
        legacy = nn.Sequential(
            nn.Linear(2304, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        legacy(torch.randn(8, 2304) + 2)
        checkpoint = self.root / "best_model.pt"
        torch.save(
            {"classifier." + k: v for k, v in legacy.state_dict().items()}, checkpoint
        )
        return legacy.eval(), checkpoint

    def test_archived_checkpoint_logits_and_batchnorm_match(self):
        legacy, checkpoint = self.legacy_checkpoint()
        model = FixedEmbeddingClassifier.load_checkpoint(checkpoint)
        pano, osm = torch.randn(5, 768), torch.randn(5, 768)
        with torch.no_grad():
            expected = legacy(torch.cat([pano, osm, pano * osm], dim=-1))
            actual = model(pano, osm)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(model.training)
        self.assertEqual(model.classifier[1].num_batches_tracked.item(), 1)
        with self.assertRaisesRegex(ValueError, "cross features"):
            model(pano, osm, cross_features=torch.ones(5, 4))

    def test_shared_training_loop_and_supplemental_rows(self):
        osm_dir = self.root / "osm"
        osm_dir.mkdir()
        np.save(
            self.root / "pano_embeddings.npy",
            np.random.default_rng(1).normal(size=(4, 768)).astype("float32"),
        )
        np.save(osm_dir / "descriptions.npy", np.ones((1, 768), dtype="float32"))
        np.save(
            self.root / "supplement_descriptions.npy",
            np.full((1, 768), 2, dtype="float32"),
        )
        np.savez(
            self.root / "Train_pairs.npz",
            pano_idx=[0, 1, 2, 3],
            osm_idx=[0, 1, 0, 1],
            labels=[0.0, 1.0, 1.0, 0.0],
        )
        dataset = training.FixedEmbeddingPairs(
            self.root, osm_dir, "Train", "descriptions"
        )
        self.assertEqual(dataset[0][1][0], 1)
        self.assertEqual(dataset[1][1][0], 2)
        loader = DataLoader(dataset, batch_size=4, collate_fn=training.collate_fixed)
        model = FixedEmbeddingClassifier()
        before = model.classifier[0].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        loss, labels, probabilities = train_epoch(
            model, loader, optimizer, scheduler, torch.device("cpu"), None, 1.0
        )
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(len(probabilities), 4)
        self.assertFalse(torch.equal(before, model.classifier[0].weight))
        val_loss, metrics = evaluate(model, loader, torch.device("cpu"))
        self.assertTrue(np.isfinite(val_loss))
        self.assertTrue(np.isfinite(metrics["auc_roc"]))

    def test_export_matches_archived_head_and_resumes(self):
        legacy, checkpoint = self.legacy_checkpoint()
        rng = np.random.default_rng(2)
        pano = rng.normal(size=(20, 768)).astype("float32")
        osm = rng.normal(size=(3, 768)).astype("float32")
        np.save(self.root / "pano.npy", pano)
        np.save(self.root / "osm.npy", osm)
        # Nontrivial row mappings exercise the prepared input contract.
        pi, oi = np.arange(19, -1, -1), np.array([2, 0, 1])
        np.savez(self.root / "indices.npz", pano=pi, osm=oi)
        with torch.no_grad():
            expected = np.array(
                [
                    [
                        legacy(torch.from_numpy(np.concatenate([p, o, p * o]))[None])
                        .sigmoid()
                        .item()
                        for o in osm[oi]
                    ]
                    for p in pano[pi]
                ],
                dtype="float32",
            )
        partial = np.lib.format.open_memmap(
            self.root / "cost_matrix.partial.npy",
            mode="w+",
            dtype="float32",
            shape=expected.shape,
        )
        partial[:16] = expected[:16]
        partial.flush()
        (self.root / "inference_progress.json").write_text(json.dumps({"rows": 16}))
        args = argparse.Namespace(
            indices=self.root / "indices.npz",
            pano_embeddings=self.root / "pano.npy",
            osm_embeddings=self.root / "osm.npy",
            checkpoint=checkpoint,
            device="cpu",
            dataset_path=self.root / "City",
            variant="descriptions",
        )
        actual = np.load(evaluation.inference(args, self.root))
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
        np.testing.assert_array_equal(actual[:16], expected[:16])
        self.assertFalse((self.root / "cost_matrix.partial.npy").exists())

    def test_spawn_workers_match_serial_without_shared_globals(self):
        path = self.root / "cost.npy"
        np.save(path, np.array([[0.95, 0.1, 0.85], [0.2, 0.98, 0.1]], dtype="float32"))
        rows, columns = [[0], [1], [], [0, 1]], [[0], [1], [], [0, 1]]
        settings = {"uniqueness_weighted": True, "use_dustbin": True}
        serial = evaluation.score_matrix(path, rows, columns, settings, workers=1)
        parallel = evaluation.score_matrix(path, rows, columns, settings, workers=2)
        np.testing.assert_array_equal(parallel, serial)
        self.assertTrue(np.any(serial > 0))
        np.testing.assert_array_equal(serial[2], 0)
        np.testing.assert_array_equal(serial[:, 2], 0)
        # A second call must not retain the previous job's weights/settings.
        np.save(path, np.zeros((2, 3), dtype="float32"))
        np.testing.assert_array_equal(
            evaluation.score_matrix(path, rows, columns, settings, 1), 0
        )

    def test_embedding_without_supplement_uses_repository_provider(self):
        (self.root / "pano_texts.json").write_text(json.dumps(["first", "second"]))
        (self.root / "prepared.json").write_text(
            json.dumps({"supplemental_osm_rows": 0})
        )
        args = argparse.Namespace(root=self.root, supplement_dir=self.root / "absent")
        with patch(
            "experimental.overhead_matching.swag.scripts.precompute_value_embeddings.embed_texts_vertex",
            return_value=np.ones((2, 768), dtype="float32"),
        ) as embed:
            training.embed_inputs(args)
        self.assertFalse(embed.call_args.kwargs["auto_truncate"])
        self.assertTrue((self.root / "embedding_complete.json").exists())
        self.assertEqual(
            np.load(self.root / "supplement_descriptions.npy").shape, (0, 768)
        )

    def test_audit_and_score_cli_with_relocated_inputs(self):
        _, checkpoint = self.legacy_checkpoint()
        raw = {
            "osm_lm_indices": [0, 1, 2],
            "osm_lm_tags": [{}, {}, {}],
            "pano_lm_tags": [{}, {}],
            "pano_id_to_lm_rows": {"a": [0], "b": [1]},
            "cost_matrix_path": "missing-archived-location.npy",
        }
        raw_path = self.root / "metadata.pt"
        torch.save(raw, raw_path)
        cost_path = self.root / "reference.npy"
        np.save(
            cost_path, np.array([[0.95, 0.1, 0.85], [0.2, 0.98, 0.1]], dtype="float32")
        )
        reference = evaluation.score_matrix(
            cost_path,
            [[0], [1]],
            [[0], [1]],
            {"uniqueness_weighted": True, "use_dustbin": True},
            1,
        )
        reference_path = self.root / "reference.pt"
        torch.save(torch.from_numpy(reference), reference_path)
        dataset = SimpleNamespace(
            _landmark_metadata=SimpleNamespace(iloc=[{"pruned_props": {}}] * 3),
            _satellite_metadata=SimpleNamespace(
                landmark_idxs=[[0], [1]], path=["tile0", "tile1"]
            ),
            _panorama_metadata=SimpleNamespace(pano_id=["a", "b"]),
        )
        audit_dir = self.root / "audit"
        common = [
            "--dataset-path",
            str(self.root / "RelocatedCity"),
            "--raw-metadata",
            str(raw_path),
            "--audit-dir",
            str(audit_dir),
        ]
        pano_path, osm_path, indices_path = [
            self.root / name for name in ("pano.npy", "osm.npy", "indices.npz")
        ]
        np.save(pano_path, np.ones((2, 768), dtype="float32"))
        np.save(osm_path, np.ones((3, 768), dtype="float32"))
        np.savez(indices_path, pano=[0, 1], osm=[0, 1, 2])
        output = self.root / "scores"
        with patch.object(evaluation, "load_vigor_dataset", return_value=dataset):
            evaluation.main(
                [
                    "audit",
                    *common,
                    "--reference-cost-matrix",
                    str(cost_path),
                    "--reference-similarity",
                    str(reference_path),
                ]
            )
            evaluation.main(
                [
                    "score",
                    *common,
                    "--checkpoint",
                    str(checkpoint),
                    "--variant",
                    "descriptions",
                    "--pano-embeddings",
                    str(pano_path),
                    "--osm-embeddings",
                    str(osm_path),
                    "--indices",
                    str(indices_path),
                    "--output-dir",
                    str(output),
                    "--device",
                    "cpu",
                    "--workers",
                    "1",
                ]
            )
        self.assertEqual(
            torch.load(output / "similarity.pt", weights_only=True).shape, (2, 2)
        )
        self.assertTrue((output / "complete.json").exists())
        self.assertTrue((audit_dir / "identity.json").exists())

    def test_embedding_batches_preserve_byte_limit(self):
        self.assertEqual(list(training.batches(["x"] * 251, 0)), [(0, 250), (250, 251)])
        self.assertEqual(list(training.batches(["é" * 5000] * 2, 0)), [(0, 1), (1, 2)])
        with self.assertRaises(ValueError):
            list(training.batches(["x" * 18001], 0))


if __name__ == "__main__":
    unittest.main()
