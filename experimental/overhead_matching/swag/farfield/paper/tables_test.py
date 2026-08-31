import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield.paper import dataset_table
from experimental.overhead_matching.swag.farfield.paper import results_table
from experimental.overhead_matching.swag.farfield.paper.table_common import (
    DATASET_GROUPS,
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
            for group_index, group in enumerate(DATASET_GROUPS, start=1):
                for sequence in group.sequences:
                    full_dir = (
                        root
                        / "artifacts"
                        / "catalogs"
                        / sequence
                        / "catalog-full-v1"
                    )
                    full_digest = f"full-digest-{group.key}"
                    _write_json(
                        root / "datasets" / sequence / "pipeline_metadata.json",
                        {
                            "dataset_name": sequence,
                            "num_images": 10,
                            "trajectory_km": 1.5,
                            "capture_date": f"2026-08-{group_index:02d}",
                            "resolution": "100x50",
                        },
                    )
                    _write_json(
                        full_dir / "manifest.json",
                        {
                            "schema": "farfield.artifact.v1",
                            "kind": "catalogs",
                            "complete": True,
                            "dataset": sequence,
                            "version": "catalog-full-v1",
                            "content_digest": full_digest,
                            "config": {
                                "source_coverage": {
                                    "schema": "farfield_catalog_source_coverage/v2",
                                    "status": "passed",
                                    "details": [
                                        {
                                            "mapped_area_km2": group_index
                                            * 1000.0
                                        }
                                    ],
                                }
                            },
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
                            "upstreams": [
                                {
                                    "kind": "catalogs",
                                    "dataset": sequence,
                                    "version": "catalog-full-v1",
                                    "path": str(full_dir),
                                    "content_digest": full_digest,
                                }
                            ],
                        },
                    )

            rows = dataset_table.collect_dataset_statistics(root, "catalog-v1")

            self.assertEqual(rows[0].num_panoramas, 30)
            self.assertAlmostEqual(rows[0].trajectory_km, 4.5)
            self.assertEqual(rows[0].map_landmarks, 100)
            self.assertEqual(rows[0].osm_bbox_area_km2, 1000.0)
            rendered = dataset_table.render_dataset_table(rows)
            self.assertIn("Mt. Washington", rendered)
            self.assertIn("3 / 30", rendered)
            self.assertIn("OSM bbox area (km$^2$)", rendered)
            self.assertIn("1,000.0", rendered)
            self.assertEqual(rendered.count(" \\\\"), 5)


class ResultsTableTest(unittest.TestCase):

    def _make_run(
        self,
        experiment_dir: Path,
        dataset: str,
        value: float,
        trajectory_length_m: int,
        suffix: str = "",
    ) -> Path:
        run_dir = experiment_dir / f"{dataset}_seed0{suffix}"
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
                "config": {"localization": {"seed": 0}},
            },
        )
        _write_json(
            run_dir / "metrics.json",
            {
                "schema": "farfield_position_mass_summary/v1",
                "higher_is_better": True,
                "reference_position": "truth",
                "source_metric_id": metric_id,
                "source_metric_version": metric_version,
                "keyframe_span": 2,
                "radii": {
                    "100": {
                        "radius_m": 100.0,
                        "time_normalized_mass": value / 2,
                    },
                    "500": {
                        "radius_m": 500.0,
                        "time_normalized_mass": value,
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

    def test_loads_seed_zero_and_keeps_separate_leg_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            washington_values = iter(((0.1, 1), (0.2, 2), (0.7, 1)))
            for group in DATASET_GROUPS:
                for sequence in group.sequences:
                    if group.key == "washington":
                        value, span = next(washington_values)
                    else:
                        value, span = 0.4, 1
                    self._make_run(experiment_dir, sequence, value, span)

            rows = results_table.load_ours_results(experiment_dir)

            self.assertEqual(len(rows), 8)
            self.assertAlmostEqual(rows[0].values[100.0], 0.05)
            self.assertAlmostEqual(rows[0].values[500.0], 0.1)
            self.assertAlmostEqual(rows[1].values[500.0], 0.2)
            self.assertAlmostEqual(rows[2].values[500.0], 0.7)
            self.assertEqual(rows[1].trajectory_length_m, 2.0)
            rendered = results_table.render_results_table(rows, experiment_dir)
            self.assertIn("CrossLocate~\\cite{tomevsek2022crosslocate}", rendered)
            self.assertIn("LOCI~\\cite{fahnestockandfuentes2026loci}", rendered)
            self.assertIn("Ours without\\\\tracking", rendered)
            self.assertIn(
                "\\multicolumn{4}{c}{$\\overline P_{100}(\\tau)$}", rendered
            )
            self.assertIn(
                "\\multicolumn{4}{c}{$\\overline P_{500}(\\tau)$}", rendered
            )
            self.assertIn("Distance-normalized posterior-mass score", rendered)
            self.assertIn("$R\\in\\{100,500\\}$~m", rendered)
            self.assertIn("Mt. Washington, leg 1", rendered)
            self.assertIn("Mt. Washington, leg 3", rendered)
            self.assertIn("Boston Harbor, leg 3", rendered)
            self.assertIn("$0.050 \\pm {}$", rendered)
            self.assertIn("$0.100 \\pm {}$", rendered)

    def test_distance_weights_posterior_mass_by_truth_arc_length(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            run_dir = self._make_run(experiment_dir, "example", 0.4, 10)
            metric_id = (
                "posterior_position_probability_mass_within_true_position_radius"
            )
            _write_jsonl(
                run_dir / "truth.jsonl",
                [
                    {"keyframe_idx": 0, "east_m": 0.0, "north_m": 0.0},
                    {"keyframe_idx": 1, "east_m": 1.0, "north_m": 0.0},
                    {"keyframe_idx": 2, "east_m": 10.0, "north_m": 0.0},
                ],
            )
            _write_jsonl(
                run_dir / "tier0_health.jsonl",
                [
                    {
                        "keyframe_idx": keyframe_idx,
                        "position_probability_mass": {
                            f"{metric_id}@1:radius_m=100": mass,
                            f"{metric_id}@1:radius_m=500": mass,
                        },
                    }
                    for keyframe_idx, mass in enumerate((0.0, 0.0, 1.0))
                ],
            )

            values, trajectory_length_m = results_table._load_metrics(
                run_dir, (100.0,)
            )

            self.assertEqual(trajectory_length_m, 10.0)
            self.assertAlmostEqual(values[100.0], 0.45)

    def test_rejects_duplicate_results_for_a_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            for group in DATASET_GROUPS:
                for sequence in group.sequences:
                    self._make_run(experiment_dir, sequence, 0.4, 1)
            duplicate = DATASET_GROUPS[0].sequences[0]
            self._make_run(experiment_dir, duplicate, 0.5, 1, suffix="_duplicate")

            with self.assertRaisesRegex(ValueError, "multiple complete seed-0 runs"):
                results_table.load_ours_results(experiment_dir)


if __name__ == "__main__":
    unittest.main()
