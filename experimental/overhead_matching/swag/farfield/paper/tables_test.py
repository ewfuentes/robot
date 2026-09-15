import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield.paper import dataset_table
from experimental.overhead_matching.swag.farfield.paper import results_table
from experimental.overhead_matching.swag.farfield.paper.table_common import (
    DATASET_GROUPS,
    SEQUENCE_ARTIFACTS,
    TABLE_GROUPS,
    bbox_area_km2,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


class DatasetTableTest(unittest.TestCase):

    def test_video_duration_excludes_internal_range_cuts(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            (dataset / "frames_gps.csv").write_text(
                "idx,video_t_s\n0,10\n1,20\n2,80\n3,90\n")
            (dataset / "extraction_log.csv").write_text(
                "frame_idx,sequence_position\n0,0\n1,1\n2,3\n3,4\n")

            minutes = dataset_table._video_minutes(
                dataset, {"trims": [{"trim_kind": "range"}]})

            self.assertAlmostEqual(minutes, 1 / 3)

    def test_aggregates_sequences_and_counts_shared_catalog_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for group_index, group in enumerate(DATASET_GROUPS, start=1):
                for sequence_index, sequence in enumerate(group.sequences):
                    _write_json(
                        root / "datasets" / sequence / "pipeline_metadata.json",
                        {
                            "dataset_name": sequence,
                            "num_images": 10 + sequence_index,
                            "trajectory_km": 1.5 + sequence_index,
                            **(
                                {"video": {"sync": {"source_video_start_utc":
                                    f"2026-08-{group_index:02d}T12:00:00Z"}}}
                                if group.key == "pohang"
                                else {"capture_date": f"2026-08-{group_index:02d}"}
                            ),
                            "resolution": "100x50",
                        },
                    )
                    (root / "datasets" / sequence / "frames_gps.csv").write_text(
                        "idx,video_t_s\n"
                        "0,30.0\n"
                        f"1,{894.0 + 60.0 * sequence_index}\n"
                    )
                    bearings_version = SEQUENCE_ARTIFACTS[sequence][
                        "bearing_observations"]
                    _write_json(
                        root / "artifacts" / "bearing_observations" / sequence
                        / bearings_version / "manifest.json",
                        {
                            "schema": "farfield.artifact.v1",
                            "kind": "bearing_observations",
                            "complete": True,
                            "dataset": sequence,
                            "config": {"n_accepted_tracklets": 20 + sequence_index},
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
                            "config": {
                                "rows_out": group_index * 100,
                                "region_bbox_wsen": [0.0, 0.0, 1.0, 1.0],
                                "clip_plan": {"policy": {
                                    "resolved_area_km2": group_index * 1000.0,
                                }},
                            },
                        },
                    )

            rows = dataset_table.collect_dataset_statistics(root, "catalog-v1")

            self.assertEqual(rows[0].panoramas_per_leg, (10, 11, 12))
            self.assertEqual(rows[0].video_minutes_per_leg, (14.4, 15.4, 16.4))
            self.assertEqual(rows[0].trajectory_km_per_leg, (1.5, 2.5, 3.5))
            self.assertEqual(rows[0].accepted_tracks_per_leg, (20, 21, 22))
            self.assertEqual(rows[0].map_landmarks, 100)
            self.assertAlmostEqual(
                rows[0].area_km2, bbox_area_km2([0.0, 0.0, 1.0, 1.0]))
            self.assertEqual(rows[-1].capture_date, "2026-08-07")
            self.assertAlmostEqual(
                rows[-1].area_km2, bbox_area_km2([0.0, 0.0, 1.0, 1.0]))
            rendered = dataset_table.render_dataset_table(rows)
            self.assertIn("Mt. Washington", rendered)
            self.assertIn("10/11/12", rendered)
            self.assertIn("14/15/16", rendered)
            self.assertIn("1.5/2.5/3.5", rendered)
            self.assertIn("20/21/22", rendered)
            self.assertIn("\\shortstack{MSM\\\\landmarks}", rendered)
            self.assertIn("\\shortstack{Overhead\\\\Area (km$^2$)}", rendered)
            self.assertIn("\\shortstack{MSM\\\\sources}", rendered)
            self.assertIn("12,392", rendered)
            self.assertIn("Pohang\\textsuperscript{*}", rendered)
            self.assertIn("Flevoland\\textsuperscript{\\textdagger}", rendered)
            self.assertIn("OSM/ENC/FAA", rendered)
            self.assertIn("\\begin{tabular*}{\\textwidth}", rendered)
            self.assertIn("\\cite{MapillaryPlatform}", rendered)
            self.assertNotIn("retained timestamp ranges", rendered)
            self.assertEqual(rendered.count(" \\\\"), 8)


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
            method_results = {
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
            }
            rendered = results_table.render_results_table(
                {score: method_results for score in results_table.SCORES},
                groups=[group],
            )
            self.assertIn("CrossLocate~\\cite{tomevsek2022crosslocate}", rendered)
            self.assertIn(
                "LOCI~\\cite{fahnestockandfuentes2026loci}",
                rendered)
            self.assertIn("No\\\\tracking", rendered)
            self.assertIn("No range\\\\bins", rendered)
            self.assertIn("\\multicolumn{4}{c}{\\textit{Full-leg baselines}}", rendered)
            self.assertIn("\\multicolumn{6}{c}{\\textit{Fresh-window evaluations}}", rendered)
            self.assertEqual(
                rendered.count(
                    "\\multicolumn{2}{c}{LOCI~\\cite{fahnestockandfuentes2026loci}}"
                ),
                len(results_table.SCORES),
            )
            self.assertIn("Distance-normalized posterior-mass score", rendered)
            self.assertIn("$R\\in\\{100,500\\}$~m", rendered)
            self.assertIn("All scores are causal", rendered)
            self.assertNotIn("Full-trajectory smoothing", rendered)
            self.assertIn("p(x_t\\mid z_{0:t})", rendered)
            self.assertNotIn("p(x_t\\mid z_{0:T})", rendered)
            self.assertIn("Mt. Washington, leg 1", rendered)
            self.assertIn("Mt. Washington, leg 3", rendered)
            self.assertIn("$0.10$", rendered)
            self.assertIn("$0.20$", rendered)
            self.assertGreaterEqual(rendered.count("$\\mathbf{0.50}$"), 2)
            self.assertNotIn("$\\mathbf{0.60}$", rendered)  # no cross-protocol winner
            self.assertIn("$\\mathbf{0.30}$", rendered)
            self.assertEqual(rendered.count("Average"), 1)
            self.assertIn("unweighted mean over available sequence rows", rendered)
            self.assertIn("including ties at the displayed precision", rendered)
            percent_table = results_table.render_results_table(
                {"causal": method_results}, groups=[group], scope="windows")
            self.assertGreaterEqual(percent_table.count("$\\mathbf{50}$"), 2)
            self.assertNotIn("\\pm", rendered)
            self.assertEqual(rendered.count("Mt. Washington, leg 1"), 1)
            self.assertIn("\\resizebox{\\textwidth}{!}", rendered)
            n_columns = (
                len(results_table.SCORES)
                * len(results_table.METHODS)
                * len(results_table.DEFAULT_RADII_M)
            )
            self.assertIn(
                f"\\begin{{tabular}}{{l{'c' * n_columns}}}", rendered)

    def test_subsections_have_equal_weight_and_missing_runs_are_starred(self):
        dataset = TABLE_GROUPS[0].sequences[0]
        expected = {(dataset, i): set(range(5)) for i in range(3)}
        values = lambda v: {100.0: v, 500.0: v}
        # Unequal numbers of finished seeds must not weight subsection 0 fivefold.
        found = {(dataset, 0): {s: values(0.2) for s in range(5)},
                 (dataset, 1): {0: values(0.8)}}
        parents, windows = results_table.summarize_subsections(found, expected)
        cell = parents[dataset][500.0]
        self.assertEqual(cell.mean, 0.5)
        self.assertEqual((cell.count, cell.expected_count), (6, 15))
        self.assertTrue(cell.incomplete)
        self.assertIsNone(windows[dataset, 2][500.0].mean)
        self.assertEqual(results_table._format_value(cell), "$0.50$\\textsuperscript{*}")
        self.assertEqual(results_table._format_value(windows[dataset, 2][500.0]),
                         "--\\textsuperscript{*}")
        rendered = results_table.render_results_table(
            {"causal": {"ours": parents}}, groups=[TABLE_GROUPS[0]])
        self.assertIn("$0.50$\\textsuperscript{*}", rendered)
        self.assertIn("Average", rendered)
        self.assertNotIn("\\pm", rendered)

    def test_loads_plan_bound_sweep_and_rejects_wrong_or_duplicate_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = "mount_washington_20260815_leg1"
            experiment, arm = results_table.PAPER_SWEEPS["ours"]
            run = root / "runs" / experiment
            relative = f"artifacts/localization_inputs/{dataset}/input-v1"
            pairs = []
            for seed in range(5):
                config = dict(input_dir=f"/data/farfield_matching/{relative}",
                              odometry_seed=seed, smoother="none", smooth_lag=0,
                              smooth_lags="", range_cap=1,
                              detection_input_dir="/data/farfield_matching/artifacts/raw",
                              detection_audit_policy="replace")
                pairs.append(dict(dataset=dataset, index=0, seed=seed,
                                  bounds=[0, 1], configs={"hybrid": config}))
            _write_json(run / "plan.json", dict(seeds=list(range(5)), pairs=pairs))
            _write_jsonl(root / relative / "truth.jsonl", [
                dict(keyframe_idx=0, east_m=0.0, north_m=0.0),
                dict(keyframe_idx=1, east_m=10.0, north_m=0.0)])
            name = f"{dataset}.{arm}.episode0.seed0.causal.json"
            path = run / "results" / "worker" / name
            result = dict(
                schema="farfield_causal_grid/v1", config=pairs[0]["configs"]["hybrid"],
                localization_inputs=dict(dataset=dataset, version="input-v1"),
                episode=dict(parent_dataset=dataset, index=0, parent_keyframe_start=0,
                             parent_keyframe_end_inclusive=1, direction="forward",
                             initialization="fresh_uniform_parent_region_and_fresh_imu_error"),
                odometry_profile=dict(noise=dict(base_seed=0)),
                summary=dict(dn_mass_100=0.2, dn_mass_500=0.4),
                mass_by_keyframe={"100": [0.1, 0.3], "500": [0.3, 0.5]},
                sweep_validation=dict(plan_sha256=hashlib.sha256(
                    (run / "plan.json").read_bytes()).hexdigest()))
            _write_json(path, result)
            _write_json(path.with_suffix(".partial.json"), {"ignored": True})
            parent, windows = results_table.load_sweep_results(root, "ours")
            self.assertEqual(parent[dataset][500.0].mean, 0.4)
            self.assertEqual(parent[dataset][500.0].expected_count, 5)
            self.assertTrue(parent[dataset][500.0].incomplete)
            result["summary"]["dn_mass_500"] = 0.9
            _write_json(path, result)
            with self.assertRaisesRegex(ValueError, "integration mismatch"):
                results_table.load_sweep_results(root, "ours")
            result["summary"]["dn_mass_500"] = 0.4
            result["config"]["smoother"] = "fixed_interval"
            _write_json(path, result)
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                results_table.load_sweep_results(root, "ours")
            result["config"]["smoother"] = "none"
            _write_json(path, result)
            _write_json(run / "results" / "duplicate" / name, result)
            with self.assertRaisesRegex(ValueError, "duplicate result"):
                results_table.load_sweep_results(root, "ours")

    def test_loci_pin_and_incomplete_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = "mount_washington_20260815_leg1"
            versions = {dataset: results_table.PAPER_LOCI_VERSION}
            loaded = results_table.load_loci_results(root, versions)
            self.assertIsNone(loaded[dataset][500.0].mean)
            self.assertTrue(loaded[dataset][500.0].incomplete)
            self.assertIn("full_multisource_landmark_only_unknown_heading", versions[dataset])
            run = root / "artifacts/loci_runs" / dataset / versions[dataset]
            manifest = dict(kind="loci_runs", complete=True, dataset=dataset,
                            version=versions[dataset], config=dict(evaluation=dict(
                                directions=["forward"], known_heading=False, n_heading=36,
                                observation_streams=["landmark"], odometry_seed=0,
                                smoother="none", reported_estimate="causal_filter")))
            _write_json(run / "manifest.json", manifest)
            result = dict(schema="farfield_causal_grid/v1",
                          localization_inputs=dict(dataset=dataset),
                          config=dict(observation_source="loci", loci_landmark_only=True,
                                      n_heading=36, odometry_seed=0, smoother="none"),
                          summary=dict(dn_mass_100=0.1, dn_mass_500=0.6,
                                       sm_dn_mass_100=0.9, sm_dn_mass_500=0.99))
            _write_json(run / "0000000/raw_result.json", result)
            loaded = results_table.load_loci_results(root, versions)
            self.assertEqual(loaded[dataset][500.0].mean, 0.6)
            self.assertFalse(loaded[dataset][500.0].incomplete)
            manifest["config"]["evaluation"]["known_heading"] = True
            _write_json(run / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "unknown-heading"):
                results_table.load_loci_results(root, versions)

    def test_loads_causal_and_smoothed_current_results(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = "mount_washington_20260815_leg1"
            path = Path(directory) / f"{dataset}.current.seed0.both.json"
            result = {
                "schema": "farfield_causal_grid/v1",
                "localization_inputs": {
                    "dataset": dataset,
                    "version": results_table.CURRENT_LOCALIZATION_VERSIONS[dataset],
                },
                "config": {
                    "odometry_seed": 0,
                    "smoother": "fixed_interval",
                    "range_cap": 1,
                    "tables_override": f"{dataset}.current.divided.json",
                },
                "summary": {
                    "dn_mass_100": 0.05,
                    "dn_mass_500": 0.10,
                    "sm_dn_mass_100": 0.25,
                    "sm_dn_mass_500": 0.75,
                },
            }
            _write_json(path, result)

            causal = results_table.load_grid_results(
                [path], {dataset}, method="ours")
            smoothed = results_table.load_grid_results(
                [path], {dataset}, method="ours", score="smoothed")

            self.assertEqual(causal[dataset][100.0].mean, 0.05)
            self.assertEqual(causal[dataset][500.0].mean, 0.10)
            self.assertEqual(smoothed[dataset][100.0].mean, 0.25)
            self.assertEqual(smoothed[dataset][500.0].mean, 0.75)

            result["config"]["range_cap"] = 0
            no_range_path = Path(directory) / f"{dataset}.current.seed0.no_range.json"
            _write_json(no_range_path, result)
            no_range = results_table.load_grid_results(
                [no_range_path], {dataset}, method="no_range")
            self.assertEqual(no_range[dataset][500.0].mean, 0.10)

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


class LatestStudyTableTest(unittest.TestCase):
    def test_full_and_windows_stay_separate_and_missing_results_stay_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            twoarm, loci = root / "twoarm", root / "loci"
            pairs, jobs = [], []
            dataset = next(iter(results_table.SEQUENCE_DISPLAY_NAMES))
            for name in results_table.SEQUENCE_DISPLAY_NAMES:
                for index in range(6):
                    scope = "full" if index == 0 else f"window{index}"
                    config = dict(input_dir=f"/data/farfield_matching/inputs/{name}",
                                  episode_index=max(index-1, 0), odometry_seed=0,
                                  odometry_profile="epson_mg570_calibrated_planar_v1",
                                  cell_m=100, n_heading=36, smoother="none",
                                  smooth_lag=0, smooth_lags="", range_cap=1)
                    hybrid = dict(config, detection_input_dir="raw", detection_audit_policy="replace")
                    pair = dict(dataset=name, scope=scope, bounds=[0, 1], grid={"cell_m": 100},
                                configs={"hybrid": hybrid, "no_tracking": config})
                    pairs.append(pair)
                    jobs.append(dict(dataset=name, scope=scope, bounds=[0, 1], grid=pair["grid"],
                                     config=dict(config, observation_source="loci")))
            _write_json(twoarm / "plan.json", {"pairs": pairs})
            plan_sha = hashlib.sha256((twoarm / "plan.json").read_bytes()).hexdigest()
            _write_json(loci / "plan.json", {"reference_plan_sha256": plan_sha, "jobs": jobs})
            _write_jsonl(root / "inputs" / dataset / "truth.jsonl", [
                {"east_m": 0, "north_m": 0}, {"east_m": 10, "north_m": 0}])
            for index, pair in enumerate(pairs[:6]):
                value = 0.9 if index == 0 else index / 10
                result = dict(schema="farfield_causal_grid/v1",
                              config=pair["configs"]["hybrid"],
                              localization_inputs={"dataset": dataset}, grid=pair["grid"],
                              study_validation={"plan_sha256": plan_sha},
                              odometry_profile={"noise": {"base_seed": 0}},
                              episode=dict(count=1 if index == 0 else 5,
                                           index=max(index-1, 0), parent_keyframe_start=0,
                                           parent_keyframe_end_inclusive=1),
                              summary={"dn_mass_100": value, "dn_mass_500": value},
                              mass_by_keyframe={"100": [value, value], "500": [value, value]})
                path = twoarm / "results" / "worker" / f"{dataset}.hybrid.{pair['scope']}.seed0.causal.json"
                _write_json(path, result)
            loaded = results_table.load_default_results(root, twoarm_study=twoarm, loci_study=loci)
            full = loaded["full"]["causal"]["ours"][dataset][100.0]
            windows = loaded["windows"]["causal"]["ours"][dataset][100.0]
            self.assertEqual((full.mean, full.count, full.std), (0.9, 1, None))
            self.assertAlmostEqual(windows.mean, 0.3)
            self.assertAlmostEqual(windows.std, 0.025 ** 0.5)
            self.assertEqual(windows.count, 5)
            self.assertEqual(loaded["full"]["causal"]["crosslocate"], {})
            self.assertEqual(loaded["full"]["causal"]["no_range"], {})
            text = results_table.render_results_table(loaded["windows"], scope="windows")
            self.assertIn("$30$", text)
            self.assertNotIn("\\pm", text)
            self.assertIn("multiplied by 100", text)
            self.assertEqual(results_table._format_value(
                results_table.Estimate(0.9, None, 1), percent=True), "$90$")
            self.assertNotIn("Full-leg baselines", text)
            self.assertNotIn("seeds (0--4)", text)
            self.assertIn("not independent trials", text)
            self.assertNotIn("\\mathbf", text)  # no wins over unfinished methods
            combined = results_table.render_results_table(loaded, scope="both")
            self.assertEqual(combined.count("\\begin{table*}"), 1)
            self.assertLess(combined.index("Five-window mean"), combined.index("Full trajectory"))
            row = next(line for line in combined.splitlines()
                       if line.strip().startswith("Mt. Washington, leg 1 &"))
            cells = [cell.strip() for cell in row.split(" & ")]
            self.assertEqual(len(cells), 21)
            self.assertEqual(cells[9:11], ["$30$", "$30$"])
            self.assertEqual(cells[19], "$90$")
            self.assertTrue(cells[20].startswith("$90$"))
            self.assertNotIn("\\pm", combined)
            cross = root / "cross"
            cross_jobs = []
            for variant in ("top6", "all12"):
                filename = f"{dataset}.{variant}.window5.t0.05.e0.05.json"
                cross_jobs.append(dict(pairs[5]["configs"]["no_tracking"],
                    observation_source="crosslocate", retrieval_dir=f"/retrieval/{variant}",
                    retrieval_temperature=0.05, retrieval_outlier_epsilon=0.05,
                    out=str(cross / "results" / filename)))
            _write_json(cross / "plan.json", dict(reference_plan_sha256=plan_sha,
                jobs=[*cross_jobs, dict(cross_jobs[0], retrieval_temperature=0.1),
                      dict(cross_jobs[0], retrieval_temperature=0.2)]))
            cross_sha = hashlib.sha256((cross / "plan.json").read_bytes()).hexdigest()
            for index, config in enumerate(cross_jobs):
                cross_path = Path(config["out"])
                value = 0.2 + index * 0.4
                cross_result = dict(result,
                    config=dict(config, out=str(cross_path.with_suffix(".partial.json")), likelihood_cache_gb=0),
                    availability={"track_inputs_used": False},
                    study_validation={"plan_sha256": cross_sha},
                    summary={"dn_mass_100": value, "dn_mass_500": value},
                    mass_by_keyframe={"100": [value, value], "500": [value, value]})
                _write_json(cross_path, cross_result)
            with_cross = results_table.load_default_results(
                root, twoarm_study=twoarm, loci_study=loci, crosslocate_study=cross)
            self.assertAlmostEqual(with_cross["windows"]["causal"]["crosslocate_top6"][dataset][100.0].mean, 0.2)
            self.assertNotIn("crosslocate_all12", with_cross["windows"]["causal"])
            supplemental = results_table.load_default_results(
                root, twoarm_study=twoarm, loci_study=loci, crosslocate_flevoland_study=cross)
            self.assertEqual(supplemental, with_cross)
            with self.assertRaisesRegex(ValueError, "duplicate CrossLocate job"):
                results_table.load_default_results(
                    root, twoarm_study=twoarm, loci_study=loci,
                    crosslocate_study=cross, crosslocate_flevoland_study=cross)
            self.assertEqual(with_cross["full"]["causal"]["crosslocate_top6"][dataset], {})
            rendered = results_table.render_results_table(with_cross, scope="both")
            self.assertIn("temperature 0.05 and outlier probability 0.05", rendered)
            self.assertIn("not on held-out data", rendered)
            self.assertNotIn("all12", rendered)
            self.assertNotIn("CrossLocate top6", rendered)
            self.assertIn(r"{c}{CrossLocate~\cite{tomevsek2022crosslocate}}", rendered)
            self.assertIn(r"{c}{LOCI~\cite{fahnestockandfuentes2026loci}}", rendered)
            cross_path = Path(cross_jobs[0]["out"])
            cross_result = json.loads(cross_path.read_text())
            cross_result["odometry_profile"] = {"noise": {"base_seed": 1}}
            _write_json(cross_path, cross_result)
            with self.assertRaisesRegex(ValueError, "paired odometry differs"):
                results_table.load_default_results(
                    root, twoarm_study=twoarm, loci_study=loci, crosslocate_study=cross)
            no_range = root / "no_range"
            no_range_pairs = [dict(p, configs={"hybrid": dict(p["configs"]["hybrid"], range_cap=0)})
                              for p in pairs]
            _write_json(no_range / "plan.json", {"reference_plan_sha256": plan_sha,
                                                 "pairs": no_range_pairs})
            no_range_sha = hashlib.sha256((no_range / "plan.json").read_bytes()).hexdigest()
            no_range_result = dict(result, config=dict(result["config"], range_cap=0),
                                   study_validation={"plan_sha256": no_range_sha})
            _write_json(no_range / "results" / "worker" / f"{dataset}.hybrid_no_range.window5.seed0.causal.json",
                        no_range_result)
            with_ablation = results_table.load_default_results(
                root, twoarm_study=twoarm, loci_study=loci, no_range_study=no_range)
            estimate = with_ablation["windows"]["causal"]["no_range"][dataset][100.0]
            self.assertEqual((estimate.mean, estimate.count, estimate.expected_count), (0.5, 1, 5))
            self.assertTrue(estimate.incomplete)
            no_range_pairs[0]["configs"]["hybrid"]["odometry_seed"] = 1
            _write_json(no_range / "plan.json", {"reference_plan_sha256": plan_sha,
                                                 "pairs": no_range_pairs})
            with self.assertRaisesRegex(ValueError, "more than range_cap"):
                results_table.load_default_results(
                    root, twoarm_study=twoarm, loci_study=loci, no_range_study=no_range)
            path.unlink()
            loaded = results_table.load_default_results(root, twoarm_study=twoarm, loci_study=loci)
            self.assertTrue(loaded["windows"]["causal"]["ours"][dataset][100.0].incomplete)
            result["study_validation"]["plan_sha256"] = "wrong"
            _write_json(path, result)
            with self.assertRaisesRegex(ValueError, "pinned plan"):
                results_table.load_default_results(root, twoarm_study=twoarm, loci_study=loci)


if __name__ == "__main__":
    unittest.main()
