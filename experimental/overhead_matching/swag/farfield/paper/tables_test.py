import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield.paper import dataset_table
from experimental.overhead_matching.swag.farfield.paper import results_table
from experimental.overhead_matching.swag.farfield.paper.table_common import (
    TABLE_GROUPS,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


class DatasetTableTest(unittest.TestCase):

    def test_aggregates_sequences_and_counts_shared_catalog_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dirs = []
            for group_index, group in enumerate(TABLE_GROUPS, start=1):
                for sequence_index, sequence in enumerate(group.sequences):
                    _write_json(
                        root / "datasets" / sequence / "pipeline_metadata.json",
                        {
                            "dataset_name": sequence,
                            "num_images": 10,
                            "trajectory_km": 1.5,
                            **(
                                {"video": {"sync": {"source_video_start_utc":
                                    f"2026-08-{group_index:02d}T12:00:00Z"}}}
                                if group.key == "pohang"
                                else {"capture_date": f"2026-08-{group_index:02d}"}
                            ),
                            "resolution": "100x50",
                        },
                    )
                    area_km2 = group_index * 1000.0 + sequence_index
                    for seed in range(4):
                        run_dir = root / "runs" / f"{sequence}_seed{seed}"
                        run_dirs.append(run_dir)
                        _write_json(
                            run_dir / "manifest.json",
                            {
                                "kind": "localization_run",
                                "complete": True,
                                "dataset": sequence,
                                "config": {"localization_run_contract": {
                                    "run_kind": "evaluation",
                                    "ablation_tags": [],
                                    "filter_config": {
                                        "seed": seed,
                                        "range_cap_enabled": True,
                                        "init": {
                                            "kind": "UniformBoxInit",
                                            "east_min_m": 0.0,
                                            "east_max_m": 100000.0,
                                            "north_min_m": 0.0,
                                            "north_max_m": area_km2 * 10.0,
                                        },
                                    },
                                }},
                            },
                        )
                    _write_json(
                        root
                        / "artifacts"
                        / "catalogs"
                        / sequence
                        / "catalog-v1"
                        / "manifest.json",
                        {
                            "schema": "farfield.artifact.v1",
                            "kind": "catalogs",
                            "complete": True,
                            "dataset": sequence,
                            "content_digest": f"digest-{group.key}",
                            "config": {"rows_out": group_index * 100},
                        },
                    )

            rows = dataset_table.collect_dataset_statistics(
                root, "catalog-v1", localization_run_dirs=run_dirs)

            self.assertEqual(rows[0].num_panoramas, 30)
            self.assertAlmostEqual(rows[0].trajectory_km, 4.5)
            self.assertEqual(rows[0].map_landmarks, 100)
            self.assertEqual(rows[0].prior_areas_km2, (1000.0, 1001.0, 1002.0))
            self.assertEqual(rows[-1].capture_date, "2026-08-05")
            self.assertEqual(rows[-1].prior_areas_km2, (5000.0, 5001.0, 5002.0))
            rendered = dataset_table.render_dataset_table(rows)
            self.assertIn("Mt. Washington", rendered)
            self.assertIn("3 / 30", rendered)
            self.assertIn("\\# landmarks", rendered)
            self.assertIn("Area (km$^2$)", rendered)
            self.assertIn("MSM Sources", rendered)
            self.assertIn("1,002", rendered)
            self.assertEqual(rendered.count(" \\\\"), 6)


class ResultsTableTest(unittest.TestCase):

    def _make_run(
        self,
        experiment_dir: Path,
        dataset: str,
        value: float,
        trajectory_length_m: int,
        seed: int = 0,
        suffix: str = "",
    ) -> Path:
        run_dir = experiment_dir / f"{dataset}_seed{seed}{suffix}"
        metric_id = (
            "posterior_position_probability_mass_within_true_position_radius"
        )
        metric_version = "1"
        _write_json(
            run_dir / "manifest.json",
            {
                "kind": "localization_run",
                "complete": True,
                "dataset": dataset,
                "config": {
                    "localization": {"seed": seed},
                    "localization_run_contract": {
                        "run_kind": "evaluation",
                        "ablation_tags": [],
                        "filter_config": {"range_cap_enabled": True},
                    },
                },
            },
        )
        _write_json(
            run_dir / "metrics.json",
            {
                "schema": "farfield_position_mass_summary/v2",
                "higher_is_better": True,
                "reference_position": "truth",
                "normalization": (
                    "trapezoidal_area_over_distance_divided_by_"
                    "trajectory_length"
                ),
                "source_metric_id": metric_id,
                "source_metric_version": metric_version,
                "trajectory_length_m": float(trajectory_length_m),
                "radii": {
                    "100": {
                        "radius_m": 100.0,
                        "distance_normalized_mass": value / 2,
                    },
                    "500": {
                        "radius_m": 500.0,
                        "distance_normalized_mass": value,
                    },
                },
            },
        )
        _write_jsonl(
            run_dir / "truth.jsonl",
            [
                {"keyframe_idx": 0, "east_m": 0.0, "north_m": 0.0},
                {
                    "keyframe_idx": 1,
                    "east_m": trajectory_length_m / 4,
                    "north_m": 0.0,
                },
                {
                    "keyframe_idx": 2,
                    "east_m": float(trajectory_length_m),
                    "north_m": 0.0,
                },
            ],
        )
        _write_jsonl(
            run_dir / "tier0_health.jsonl",
            [
                {
                    "keyframe_idx": keyframe_idx,
                    "position_probability_mass": {
                        f"{metric_id}@{metric_version}:radius_m=100": value / 2,
                        f"{metric_id}@{metric_version}:radius_m=500": value,
                    },
                }
                for keyframe_idx in range(3)
            ],
        )
        return run_dir

    def test_loads_and_aggregates_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            dataset = "mount_washington_20260815_leg1"
            runs = [
                self._make_run(experiment_dir, dataset, 0.2, 10, seed=0),
                self._make_run(experiment_dir, dataset, 0.4, 10, seed=1),
            ]

            loaded = results_table.load_localization_results(
                runs, {dataset}, {0, 1})

            self.assertAlmostEqual(loaded[dataset][100.0].mean, 0.15)
            self.assertAlmostEqual(loaded[dataset][500.0].mean, 0.3)
            self.assertAlmostEqual(
                loaded[dataset][500.0].std, 2 ** 0.5 / 10)
            with self.assertRaisesRegex(ValueError, "range setting"):
                results_table.load_localization_results(
                    [runs[0]], {dataset}, {0}, method="no_range")

            group = TABLE_GROUPS[0]
            method_values = {
                "crosslocate": (0.1, 0.6),
                "loci": (0.2, 0.4),
                "no_tracking": (0.3, 0.3),
                "no_range": (0.504, 0.2),
                "ours": (0.501, 0.1),
            }
            rendered = results_table.render_results_table(
                {
                    method: {
                        sequence: {
                            radius: results_table.Estimate(
                                value,
                                None if method == "loci" else 0.01,
                                1 if method == "loci" else 4,
                            )
                            for radius, value in zip(
                                results_table.DEFAULT_RADII_M,
                                method_values[method],
                            )
                        }
                        for sequence in group.sequences
                    }
                    for method, _ in results_table.METHODS
                },
                groups=[group],
            )
            self.assertIn("CrossLocate~\\cite{tomevsek2022crosslocate}", rendered)
            self.assertIn("LOCI~\\cite{fahnestockandfuentes2026loci}", rendered)
            self.assertIn("No\\\\tracking", rendered)
            self.assertIn("No range\\\\bins", rendered)
            self.assertIn(
                "\\multicolumn{5}{c}{$\\overline P_{100}(\\tau)$}", rendered
            )
            self.assertIn(
                "\\multicolumn{5}{c}{$\\overline P_{500}(\\tau)$}", rendered
            )
            self.assertIn("Distance-normalized posterior-mass score", rendered)
            self.assertIn("$R\\in\\{100,500\\}$~m", rendered)
            self.assertIn("Mt. Washington, leg 1", rendered)
            self.assertIn("Mt. Washington, leg 3", rendered)
            self.assertIn("$0.10 \\pm \\mathrm{N/A}$", rendered)
            self.assertIn("$0.20 \\pm \\mathrm{N/A}$", rendered)
            self.assertEqual(
                rendered.count("$\\mathbf{0.50} \\pm \\mathrm{N/A}$"),
                2 * len(group.sequences),
            )
            self.assertIn("$\\mathbf{0.60} \\pm \\mathrm{N/A}$", rendered)
            self.assertNotIn("\\pm 0.01", rendered)

    def test_rescores_time_normalized_metrics_from_run_history(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            run_dir = self._make_run(experiment_dir, "example", 0.4, 10)
            superseded = json.loads((run_dir / "metrics.json").read_text())
            superseded["schema"] = "farfield_position_mass_summary/v1"
            _write_json(run_dir / "metrics.json", superseded)

            values, trajectory_length_m = results_table._load_metrics(
                run_dir, (100.0, 500.0))

            self.assertAlmostEqual(values[100.0], 0.2)
            self.assertAlmostEqual(values[500.0], 0.4)
            self.assertEqual(trajectory_length_m, 10.0)

    def test_rejects_duplicate_results_for_a_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            dataset = TABLE_GROUPS[0].sequences[0]
            first = self._make_run(experiment_dir, dataset, 0.4, 1)
            duplicate = self._make_run(
                experiment_dir, dataset, 0.5, 1, suffix="_duplicate")

            with self.assertRaisesRegex(ValueError, "duplicate seed 0"):
                results_table.load_localization_results(
                    [first, duplicate], {dataset}, {0})


if __name__ == "__main__":
    unittest.main()
