import json
import tempfile
import unittest
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.farfield.localization import metrics
from experimental.overhead_matching.swag.scripts import summarize_loci_position_mass


class SummarizeLociPositionMassTest(unittest.TestCase):
    def _path_dir(self, root: Path) -> Path:
        path_dir = root / "0000000"
        path_dir.mkdir()
        torch.save(torch.tensor([0.0, 10.0, 30.0]),
                   path_dir / "distance_traveled_m.pt")
        return path_dir

    def test_writes_canonical_summary_from_post_observation_masses(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            path_dir = self._path_dir(run_dir)
            torch.save({
                100: torch.tensor([0.0, 0.1, 0.3, 0.5]),
                500: torch.tensor([1.0, 0.2, 0.6, 1.0]),
            }, path_dir / "prob_mass_by_radius.pt")

            written = summarize_loci_position_mass.summarize_run(run_dir)

            self.assertEqual(written, [path_dir])
            summary = json.loads((path_dir / "metrics.json").read_text())
            self.assertEqual(summary["schema"],
                             metrics.POSITION_MASS_SUMMARY_SCHEMA)
            self.assertEqual(summary["n_keyframes"], 3)
            self.assertEqual(summary["trajectory_length_m"], 30.0)
            self.assertAlmostEqual(
                summary["radii"]["100"]["distance_normalized_mass"], 1.0 / 3.0)
            self.assertAlmostEqual(
                summary["radii"]["500"]["distance_normalized_mass"], 2.0 / 3.0)

    def test_requires_both_canonical_radii(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path_dir = self._path_dir(Path(temporary_dir))
            torch.save({100: torch.ones(4)}, path_dir / "prob_mass_by_radius.pt")
            with self.assertRaisesRegex(ValueError, "500 m"):
                summarize_loci_position_mass.summarize_path(path_dir)

    def test_rejects_mass_length_without_initial_prior(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path_dir = self._path_dir(Path(temporary_dir))
            torch.save({100: torch.ones(3), 500: torch.ones(3)},
                       path_dir / "prob_mass_by_radius.pt")
            with self.assertRaisesRegex(ValueError, "initial prior plus keyframes"):
                summarize_loci_position_mass.summarize_path(path_dir)

    def test_rejects_nonmonotonic_cumulative_radii(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path_dir = self._path_dir(Path(temporary_dir))
            torch.save({
                100: torch.tensor([0.0, 0.1, 0.4, 0.2]),
                500: torch.tensor([0.0, 0.2, 0.3, 0.5]),
            }, path_dir / "prob_mass_by_radius.pt")
            with self.assertRaisesRegex(ValueError, "500 m mass below 100 m"):
                summarize_loci_position_mass.summarize_path(path_dir)

    def test_rejects_invalid_initial_prior_mass(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path_dir = self._path_dir(Path(temporary_dir))
            torch.save({
                100: torch.tensor([float("nan"), 0.1, 0.2, 0.3]),
                500: torch.tensor([0.5, 0.2, 0.3, 0.4]),
            }, path_dir / "prob_mass_by_radius.pt")
            with self.assertRaisesRegex(ValueError, "finite probabilities"):
                summarize_loci_position_mass.summarize_path(path_dir)

    def test_refuses_to_modify_published_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            (run_dir / "manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "published artifact"):
                summarize_loci_position_mass.summarize_run(run_dir)


if __name__ == "__main__":
    unittest.main()
