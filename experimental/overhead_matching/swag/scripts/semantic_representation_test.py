"""Regression coverage for checkpoint compatibility and the portable ablation workflow."""

import argparse
import json
from pathlib import Path
import pickle
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
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    custom_id_from_props,
)
from experimental.overhead_matching.swag.scripts.correspondence_configs import (
    CorrespondenceClassifierMlpConfig,
    CorrespondenceEncoderConfig,
    CorrespondenceTrainConfig,
    save_config,
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

    def test_shared_training_loop(self):
        osm_dir = self.root / "osm"
        osm_dir.mkdir()
        np.save(
            self.root / "pano_embeddings.npy",
            np.random.default_rng(1).normal(size=(4, 768)).astype("float32"),
        )
        np.save(
            osm_dir / "descriptions.npy",
            np.stack([np.ones(768), np.full(768, 2)]).astype("float32"),
        )
        (self.root / "prepared.json").write_text(json.dumps({"base_osm_rows": 2}))
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
        # An OSM array that is not the prepared population is refused.
        np.save(osm_dir / "descriptions.npy", np.ones((3, 768), dtype="float32"))
        with self.assertRaises(AssertionError):
            training.FixedEmbeddingPairs(self.root, osm_dir, "Train", "descriptions")

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

    def test_embedding_uses_repository_provider(self):
        (self.root / "pano_texts.json").write_text(json.dumps(["first", "second"]))
        args = argparse.Namespace(root=self.root)
        with patch(
            "experimental.overhead_matching.swag.scripts.precompute_value_embeddings.embed_texts_vertex",
            return_value=np.ones((2, 768), dtype="float32"),
        ) as embed:
            training.embed_inputs(args)
        self.assertFalse(embed.call_args.kwargs["auto_truncate"])
        self.assertTrue((self.root / "embedding_complete.json").exists())
        self.assertEqual(np.load(self.root / "pano_embeddings.npy").shape, (2, 768))

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

    def test_tags_text_matches_osm_serialization(self):
        self.assertEqual(training.tags_text({"b": "2", "a": "1"}), "a=1, b=2")
        # Repeated keys collapse to the last value, as in the parser and exporter.
        self.assertEqual(
            training.tags_text([("name", "A"), ("b", "2"), ("name", "B")]), "b=2, name=B"
        )

    def prepare_fixture(self):
        """One labelled panorama per city; one OSM bundle whose map value has trailing whitespace."""
        data, pano, osm, source = [
            self.root / name for name in ("data", "pano", "osm", "source")
        ]
        prompt = (
            "Set 1 (street-level observations):\n 0. amenity=cafe; name=Joe's\n"
            " 1. building=yes; addr:housenumber=5\n\nSet 2 (map database):\n"
            " 0. amenity=cafe; name=Joe's\n 1. building=yes; addr:housenumber=7\n"
        )
        response = {
            "matches": [
                {
                    "set_1_id": 0,
                    "set_2_matches": [0],
                    "negatives": [{"set_2_id": 1, "difficulty": "hard"}],
                },
                {
                    "set_1_id": 1,
                    "set_2_matches": [],
                    "negatives": [{"set_2_id": 0, "difficulty": "easy"}],
                },
            ]
        }
        row = {
            "key": "p1",
            "request": {"contents": [{"parts": [{"text": prompt}]}]},
            "response": {
                "candidates": [{"content": {"parts": [{"text": json.dumps(response)}]}}]
            },
        }
        tag = lambda key, value: {"key": key, "value": value}  # noqa: E731
        landmarks = [
            dict(
                primary_tag=tag("amenity", "cafe"),
                additional_tags=[tag("name", "Joe's")],
                description="A cafe named Joe's.",
            ),
            dict(primary_tag=tag("bogus", "x"), additional_tags=[], description="x"),
            dict(
                primary_tag=tag("building", "yes"),
                additional_tags=[tag("addr:housenumber", "5")],
                description="A building at 5.",
            ),
        ]
        for city in ("Train", "Val"):
            (data / city).mkdir(parents=True)
            (data / city / "predictions.jsonl").write_text(json.dumps(row) + "\n")
            (pano / city / "embeddings").mkdir(parents=True)
            with (pano / city / "embeddings/embeddings.pkl").open("wb") as f:
                pickle.dump({"panoramas": {"p1,1.0,2.0": {"landmarks": landmarks}}}, f)
        population = [
            {"amenity": "cafe", "name": "Joe's"},
            {"building": "yes", "addr:housenumber": "7 "},  # parser strips the space
        ]
        osm.mkdir()
        (osm / "keys.json").write_text(json.dumps([custom_id_from_props(t) for t in population]))
        for variant in ("descriptions", "tag_strings"):
            np.save(osm / f"{variant}.npy", np.ones((2, 768), dtype="float32"))
        source.mkdir()
        (source / "manifest.json").write_text(
            json.dumps({"sample": {custom_id_from_props(t): {"tags": t} for t in population}})
        )
        config = CorrespondenceTrainConfig(
            data_dir=data,
            text_embeddings_path=self.root / "unused.pkl",
            output_dir=self.root / "baseline",
            train_city="Train",
            val_city="Val",
            include_difficulties=["positive", "easy", "hard"],
            batch_size=2,
            num_epochs=1,
            lr=1e-3,
            weight_decay=0.0,
            warmup_fraction=0.0,
            gradient_clip_norm=1.0,
            use_amp=False,
            num_workers=0,
            seed=0,
            encoder=CorrespondenceEncoderConfig(text_input_dim=768, text_proj_dim=8),
            classifier=CorrespondenceClassifierMlpConfig(mlp_hidden_dim=8, dropout=0.0),
        )
        save_config(config, self.root / "config.yaml")
        return lambda pano_text: argparse.Namespace(
            root=self.root / pano_text,
            config=self.root / "config.yaml",
            osm_embeddings_dir=osm,
            osm_source_dir=source,
            pano_source_dir=pano,
            pano_text=pano_text,
        )

    def test_prepare_pano_text_variants_and_whitespace_lookup(self):
        make_args = self.prepare_fixture()
        expected = {
            "descriptions": ["A cafe named Joe's.", "A building at 5."],
            "tag_strings": ["amenity=cafe, name=Joe's", "addr:housenumber=5, building=yes"],
        }
        for pano_text, texts in expected.items():
            args = make_args(pano_text)
            training.prepare(args)
            self.assertEqual(json.loads((args.root / "pano_texts.json").read_text()), texts)
            prepared = json.loads((args.root / "prepared.json").read_text())
            self.assertEqual(prepared["pano_text"], pano_text)
            self.assertEqual(prepared["base_osm_rows"], 2)
            for city in ("Train", "Val"):
                pairs = np.load(args.root / f"{city}_pairs.npz")
                self.assertEqual(pairs["pano_idx"].tolist(), [0, 0, 1])
                self.assertEqual(pairs["osm_idx"].tolist(), [0, 1, 0])
                self.assertEqual(pairs["labels"].tolist(), [1.0, 0.0, 0.0])
                self.assertEqual(prepared["cities"][city]["whitespace_resolved_osm_pairs"], 1)

    def test_prepare_refuses_unresolvable_osm_bundle(self):
        make_args = self.prepare_fixture()
        # Simulate a parser that truncates "addr:housenumber=7" out of the prompt bundle.
        path = self.root / "data/Train/predictions.jsonl"
        path.write_text(path.read_text().replace("building=yes; addr:housenumber=7", "building=yes"))
        with self.assertRaisesRegex(AssertionError, "OSM entities match"):
            training.prepare(make_args("descriptions"))

    def test_embedding_batches_preserve_byte_limit(self):
        self.assertEqual(list(training.batches(["x"] * 251, 0)), [(0, 250), (250, 251)])
        self.assertEqual(list(training.batches(["é" * 5000] * 2, 0)), [(0, 1), (1, 2)])
        with self.assertRaises(ValueError):
            list(training.batches(["x" * 18001], 0))


if __name__ == "__main__":
    unittest.main()
