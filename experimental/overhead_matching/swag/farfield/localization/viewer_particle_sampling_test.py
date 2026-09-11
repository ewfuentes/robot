import tempfile
import unittest
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.localization import (
    run_io,
    viewer_particle_sampling as particle_sampling,
)


def arrays(n=100):
    return {
        "east_m": np.arange(n, dtype=np.float64),
        "north_m": -np.arange(n, dtype=np.float64),
        "log_weight": np.zeros(n, dtype=np.float64),
        "mode_id": np.arange(n, dtype=np.int64) % 3,
    }


def publish_run(root: Path, values: dict[str, np.ndarray]) -> Path:
    run_dir = root / "runs" / "experiment" / "run"
    relative = "checkpoints/kf_00000.npz"
    with artifact.ArtifactDirectoryBuilder(
            run_dir, kind=run_io.RUN_KIND, dataset="dataset", version="run",
            generator="viewer_particle_sampling_test", git_commit="deadbeef",
            arguments=(), config={}, declared_outputs=(relative,)) as builder:
        checkpoint = builder.output_path(relative)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint, **values)
    return run_dir


class ParticleSamplingTest(unittest.TestCase):
    def test_percentages_are_sized_against_the_full_population(self):
        values = arrays()
        for percent, expected in ((10, 10), (20, 20), (30, 30),
                                  (50, 50), (100, 100)):
            with self.subTest(percent=percent):
                payload = particle_sampling.payload_from_arrays(
                    values, keyframe_idx=0, percent=percent)
                self.assertEqual(payload["percent"], percent)
                self.assertEqual(payload["n"], expected)
                self.assertEqual(payload["total"], 100)
                for key in ("e", "n_m", "mode"):
                    self.assertEqual(len(payload[key]), expected)

    def test_subsets_are_deterministic_distinct_and_nested(self):
        values = arrays()
        del values["log_weight"]

        ten = particle_sampling.payload_from_arrays(
            values, keyframe_idx=7, percent=10)
        repeated = particle_sampling.payload_from_arrays(
            values, keyframe_idx=7, percent=10)
        twenty = particle_sampling.payload_from_arrays(
            values, keyframe_idx=7, percent=20)

        self.assertEqual(ten, repeated)
        self.assertEqual(len(set(ten["e"])), 10)
        self.assertTrue(set(ten["e"]).issubset(twenty["e"]))
        self.assertEqual(ten["sampling"], "without_replacement")

    def test_100_percent_visits_every_particle_once_regardless_of_weights(self):
        values = arrays()
        values["log_weight"][:] = -np.inf
        payload = particle_sampling.payload_from_arrays(
            values, keyframe_idx=3, percent=100)
        self.assertEqual(payload["e"], list(range(100)))
        self.assertEqual(payload["sampling"], "all")

    def test_checkpoint_reader_reads_only_a_declared_regular_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = publish_run(Path(tmp), arrays())

            payload = particle_sampling.checkpoint_payload(
                run_dir, keyframe_idx=0, percent=20)

            self.assertEqual(payload["n"], 20)
            with self.assertRaises(FileNotFoundError):
                particle_sampling.checkpoint_payload(
                    run_dir, keyframe_idx=1, percent=20)

    def test_invalid_percentage_and_array_shapes_are_rejected(self):
        with self.assertRaisesRegex(
                particle_sampling.ParticleSamplingError, "one of"):
            particle_sampling.payload_from_arrays(
                arrays(), keyframe_idx=0, percent=25)
        malformed = arrays()
        malformed["mode_id"] = malformed["mode_id"][:-1]
        with self.assertRaisesRegex(
                particle_sampling.ParticleSamplingError, "shape"):
            particle_sampling.payload_from_arrays(
                malformed, keyframe_idx=0, percent=10)


if __name__ == "__main__":
    unittest.main()
