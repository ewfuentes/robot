import json
import hashlib
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.scripts import evaluate_histogram_on_paths as subject


class _Belief:
    def get_mean_latlon(self):
        return torch.zeros(2)

    def get_mode_latlon(self):
        return torch.zeros(2)

    def get_variance_deg_sq(self):
        return torch.zeros(2)

    def apply_observation(self, _log_likelihood, _mapping):
        pass

    def apply_motion(self, _delta, _noise):
        pass


class EvaluateHistogramOnPathsTest(unittest.TestCase):
    def test_default_convergence_radii_match_farfield_metrics(self):
        self.assertEqual(subject.DEFAULT_CONVERGENCE_RADII, (100, 500))

    def test_loads_exact_applied_motion_deltas(self):
        with tempfile.TemporaryDirectory() as temporary:
            path_dir = Path(temporary)
            expected = torch.tensor([[1e-5, -2e-5], [3e-5, 4e-5]])
            torch.save(
                expected,
                path_dir / subject.APPLIED_MOTION_DELTAS_FILENAME,
            )

            actual = subject.load_applied_motion_deltas(path_dir, path_len=3)

            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_applied_motion_deltas_are_required_and_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            path_dir = Path(temporary)
            with self.assertRaisesRegex(
                    FileNotFoundError, "regenerate.*rather than reconstructing"):
                subject.load_applied_motion_deltas(path_dir, path_len=2)

            torch.save(
                torch.zeros(2, 2),
                path_dir / subject.APPLIED_MOTION_DELTAS_FILENAME,
            )
            with self.assertRaisesRegex(ValueError, r"shape \(1, 2\)"):
                subject.load_applied_motion_deltas(path_dir, path_len=2)

    def test_converts_sharp_turn_and_reverse_from_truth_chords(self):
        meters_per_degree = subject.web_mercator.METERS_PER_DEG_LAT
        anchor_lat = 42.0
        anchor_lon = -71.0
        meters_per_degree_lon = meters_per_degree * math.cos(
            math.radians(anchor_lat))
        truth = [
            subject.farfield_structs.TruthPose(0, 0.0, 0.0, 45.0),
            subject.farfield_structs.TruthPose(1, 0.0, 10.0, 45.0),
            subject.farfield_structs.TruthPose(2, 10.0, 10.0, 45.0),
        ]
        odometry = [
            subject.farfield_structs.OdometryDelta(
                1, 10.0, 2.0, 0.75, 1.0, 0.1),
            subject.farfield_structs.OdometryDelta(
                2, -5.0, 3.0, -0.5, 1.0, 0.1),
        ]
        path_latlons = torch.tensor([
            [anchor_lat, anchor_lon],
            [anchor_lat + 10.0 / meters_per_degree, anchor_lon],
            [anchor_lat + 10.0 / meters_per_degree,
             anchor_lon + 10.0 / meters_per_degree_lon],
        ], dtype=torch.float64)

        actual = subject.farfield_odometry_to_latlon_deltas(
            odometry,
            truth,
            path_latlons,
            reverse_keyframe_ranges=[[2, 2]],
            displacement_gate_m=2.0,
        )

        expected = torch.tensor([
            [10.0 / meters_per_degree, -2.0 / meters_per_degree_lon],
            [-3.0 / meters_per_degree, 5.0 / meters_per_degree_lon],
        ], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-18)

    def test_truth_binding_allows_filename_rounding_not_wrong_trajectory(self):
        meters_per_degree = subject.web_mercator.METERS_PER_DEG_LAT
        truth = [
            subject.farfield_structs.TruthPose(0, 0.0, 0.0, 0.0),
            subject.farfield_structs.TruthPose(1, 0.0, 10.0, 0.0),
        ]
        odometry = [subject.farfield_structs.OdometryDelta(
            1, 10.0, 0.0, 0.0, 1.0, 0.1)]
        mapping_positions = torch.tensor([
            [42.00000049, -70.99999951],
            [42.00000049 + 10.0 / meters_per_degree + 9e-7,
             -70.99999951 - 9e-7],
        ], dtype=torch.float64)

        subject.farfield_odometry_to_latlon_deltas(
            odometry,
            truth,
            mapping_positions,
            reverse_keyframe_ranges=[],
            displacement_gate_m=2.0,
        )

        wrong_positions = mapping_positions.clone()
        wrong_positions[1, 0] += 1e-4
        with self.assertRaisesRegex(ValueError, "do not match"):
            subject.farfield_odometry_to_latlon_deltas(
                odometry,
                truth,
                wrong_positions,
                reverse_keyframe_ranges=[],
                displacement_gate_m=2.0,
            )

    def test_farfield_odometry_requires_valid_contiguous_records(self):
        truth = [
            subject.farfield_structs.TruthPose(0, 0.0, 0.0, 0.0),
            subject.farfield_structs.TruthPose(1, 0.0, 1.0, 0.0),
        ]
        positions = torch.tensor(
            [[42.0, -71.0],
             [42.0 + 1.0 / subject.web_mercator.METERS_PER_DEG_LAT, -71.0]],
            dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "odometry keyframes.*1..N-1"):
            subject.farfield_odometry_to_latlon_deltas([
                subject.farfield_structs.OdometryDelta(
                    2, 1.0, 0.0, 0.0, 1.0, 0.1),
            ], truth, positions,
                reverse_keyframe_ranges=[], displacement_gate_m=0.5)
        with self.assertRaisesRegex(ValueError, "odometry at keyframe 1 is invalid"):
            subject.farfield_odometry_to_latlon_deltas([
                subject.farfield_structs.OdometryDelta(
                    1, float("nan"), 0.0, 0.0, 1.0, 0.1),
            ], truth, positions,
                reverse_keyframe_ranges=[], displacement_gate_m=0.5)
        with self.assertRaisesRegex(ValueError, "stationary odometry.*zero"):
            subject.farfield_odometry_to_latlon_deltas([
                subject.farfield_structs.OdometryDelta(
                    1, 0.1, 0.0, 0.0, 1.0, 0.1),
            ], truth, positions,
                reverse_keyframe_ranges=[], displacement_gate_m=2.0)
        with self.assertRaisesRegex(ValueError, "sorted, non-overlapping"):
            subject.farfield_odometry_to_latlon_deltas([
                subject.farfield_structs.OdometryDelta(
                    1, 1.0, 0.0, 0.0, 1.0, 0.1),
            ], truth, positions,
                reverse_keyframe_ranges=[[1, 1], [1, 1]],
                displacement_gate_m=0.5)

    def test_loads_published_farfield_odometry_for_one_forward_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset_a"
            dataset.mkdir()
            meters_per_degree = subject.web_mercator.METERS_PER_DEG_LAT
            mapping = (
                "pano_id,lat,lon\n"
                "f0000,42.0,-71.0\n"
                f"f0001,{42.0 + 10.0 / meters_per_degree},-71.0\n")
            (dataset / "pano_id_mapping.csv").write_text(mapping)
            positions = subject.load_authoritative_panorama_positions(
                dataset / "pano_id_mapping.csv")

            artifact_dir = root / "odometry"
            artifact_dir.mkdir()
            (artifact_dir / "truth.jsonl").write_text(
                '{"keyframe_idx":0,"east_m":0.0,"north_m":0.0,'
                '"course_world_cw_deg":0.0}\n'
                '{"keyframe_idx":1,"east_m":0.0,"north_m":10.0,'
                '"course_world_cw_deg":0.0}\n')
            (artifact_dir / "tier1_odometry.jsonl").write_text(
                '{"kind":"OdometryDelta","keyframe_idx":1,'
                '"forward_m":9.5,"left_m":0.0,"delta_yaw_cw_rad":0.1,'
                '"sigma_m":1.0,"sigma_yaw_rad":0.2}\n')
            manifest = {
                "schema": subject.farfield_artifact.SCHEMA,
                "kind": subject.farfield_paths.LOCALIZATION_INPUTS,
                "dataset": "dataset_a",
                "version": "v1",
                "generator": "test",
                "git_commit": "test",
                "created": "test",
                "arguments": [],
                "content_digest": subject.farfield_artifact.sha256_directory(
                    artifact_dir),
                "upstreams": [],
                "config": {"localization_inputs": {
                    "displacement_gate_m": 2.0,
                    "reverse_keyframe_ranges": [],
                }},
                "declared_outputs": [
                    "tier1_odometry.jsonl", "truth.jsonl"],
                "complete": True,
            }
            (artifact_dir / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                + "\n")

            motion, source = subject.load_farfield_motion_deltas(
                artifact_dir,
                expected_dataset=dataset.name,
                paths=[["f0000", "f0001"]],
                positions_by_id=positions,
            )

            self.assertEqual(source["artifact"]["dataset"], "dataset_a")
            self.assertEqual(source["delta_yaw"], "ignored_known_heading")
            self.assertEqual(
                source["translation"],
                "forward_left_rotated_by_truth_chord_body_heading")
            self.assertEqual(motion.dtype, torch.float64)
            self.assertAlmostEqual(
                motion[0, 0].item(), 9.5 / meters_per_degree)
            self.assertEqual(motion[0, 1].item(), 0.0)
            with self.assertRaisesRegex(ValueError, "exactly one forward path"):
                subject.load_farfield_motion_deltas(
                    artifact_dir,
                    expected_dataset=dataset.name,
                    paths=[["f0001", "f0000"]],
                    positions_by_id=positions,
                )
            with self.assertRaisesRegex(ValueError, "artifact dataset mismatch"):
                subject.load_farfield_motion_deltas(
                    artifact_dir,
                    expected_dataset="other_dataset",
                    paths=[["f0000", "f0001"]],
                    positions_by_id=positions,
                )

    def test_shared_farfield_odometry_cannot_be_renoised(self):
        config = subject.HistogramFilterConfig(
            odometry_noise=subject.OdometryNoiseConfig(0.02, 0))
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            subject.evaluate_histogram_on_paths(
                vigor_dataset=None,
                log_likelihood_aggregator=None,
                paths=[["f0000", "f0001"]],
                config=config,
                seed=0,
                output_path=Path("unused"),
                shared_motion_deltas=torch.zeros(1, 2),
            )

    def test_loads_authoritative_truth_at_csv_precision_and_path_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            mapping = Path(temporary) / "pano_id_mapping.csv"
            mapping.write_text(
                "pano_id,lat,lon,filename\n"
                "f0000,44.266883236,-71.308299125,a.jpg\n"
                "f0001,44.266900789,-71.308250456,b.jpg\n")

            positions_by_id = subject.load_authoritative_panorama_positions(mapping)
            positions = subject.authoritative_positions_for_path(
                positions_by_id, ["f0001", "f0000"])

            self.assertEqual(positions.dtype, torch.float64)
            self.assertEqual(
                positions.tolist(),
                [[44.266900789, -71.308250456],
                 [44.266883236, -71.308299125]],
            )

    def test_authoritative_truth_is_independent_of_dataset_positions(self):
        class Dataset:
            def get_panorama_positions(self, _path):
                raise AssertionError("rounded filename truth must not be read")

        truth = torch.tensor(
            [[44.266883236, -71.308299125]], dtype=torch.float64)
        history = torch.cat((torch.zeros(1, 2, dtype=torch.float64), truth))

        error = subject.get_distance_error_from_estimate_history(
            Dataset(), ["f0000"], history, truth)

        self.assertEqual(error.item(), 0.0)

    def test_authoritative_truth_rejects_missing_path_id(self):
        with self.assertRaisesRegex(ValueError, "f0001.*pano_id_mapping.csv"):
            subject.authoritative_positions_for_path(
                {"f0000": (42.0, -71.0)}, ["f0000", "f0001"])

    def test_authoritative_truth_rejects_duplicate_and_invalid_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            mapping = Path(temporary) / "pano_id_mapping.csv"
            mapping.write_text(
                "pano_id,lat,lon\n"
                "f0000,42.0,-71.0\n"
                "f0000,42.1,-71.1\n")
            with self.assertRaisesRegex(ValueError, "duplicates pano_id"):
                subject.load_authoritative_panorama_positions(mapping)

            mapping.write_text("pano_id,lat,lon\nf0000,nan,-71.0\n")
            with self.assertRaisesRegex(ValueError, "out-of-range lat/lon"):
                subject.load_authoritative_panorama_positions(mapping)

    def test_mapping_distance_is_float64_and_reverse_symmetric(self):
        positions = torch.tensor([
            [44.266883236, -71.308299125],
            [44.266900789, -71.308250456],
            [44.267000123, -71.308100987],
        ], dtype=torch.float64)

        forward = subject.compute_distance_traveled_from_positions(positions)
        reverse = subject.compute_distance_traveled_from_positions(
            torch.flip(positions, dims=(0,)))

        self.assertEqual(forward.dtype, torch.float64)
        self.assertAlmostEqual(forward[-1].item(), reverse[-1].item(), places=9)
        torch.testing.assert_close(
            forward[1:] - forward[:-1],
            torch.flip(reverse[1:] - reverse[:-1], dims=(0,)),
        )

    def test_convergence_uses_truth_at_observation_step(self):
        truth = torch.tensor([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]])
        seen = []

        def record_truth(_belief, true_latlon, _radius):
            seen.append(true_latlon.tolist())
            return 0.5

        with mock.patch.object(
                subject, "compute_probability_mass_within_radius",
                side_effect=record_truth):
            subject.run_histogram_filter_on_path(
                belief=_Belief(),
                motion_deltas=torch.zeros(2, 2),
                path_pano_ids=["p0", "p1", "p2"],
                log_likelihood_aggregator=lambda _pano_id: torch.zeros(1),
                mapping=object(),
                config=subject.HistogramFilterConfig(),
                true_latlons=truth,
                convergence_radii=[25],
            )

        self.assertEqual(seen, [
            [10.0, 20.0],  # initial belief
            [10.0, 20.0],  # observation p0
            [11.0, 21.0],  # observation p1
            [12.0, 22.0],  # observation p2
        ])

    def test_reads_nested_farfield_source_px(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact"
            satellite = artifact / "satellite"
            satellite.mkdir(parents=True)
            (artifact / "satellite_bbox.json").write_text(json.dumps({
                "grid": {"source_px": 512, "zoom": 20, "patch_px": 640},
            }))

            self.assertEqual(
                subject.read_satellite_source_px(Path("unused"), satellite),
                512,
            )

    def test_external_satellite_bbox_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            satellite = Path(temporary) / "artifact" / "satellite"
            satellite.mkdir(parents=True)

            with self.assertRaisesRegex(
                    FileNotFoundError, "external satellite directory requires"):
                subject.read_satellite_source_px(Path("unused"), satellite)

    def test_external_satellite_bbox_validates_grid_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact"
            satellite = artifact / "satellite"
            satellite.mkdir(parents=True)
            bbox = artifact / "satellite_bbox.json"

            bbox.write_text(json.dumps({
                "grid": {"source_px": 512, "patch_px": 640},
            }))
            with self.assertRaisesRegex(ValueError, "zoom must be 20"):
                subject.read_satellite_source_px(
                    Path("unused"), satellite)

            for field, value, message in (
                    ("source_px", 0, "source_px must be a positive number"),
                    ("zoom", 19, "zoom must be 20"),
                    ("patch_px", 512, "patch_px must be 640")):
                with self.subTest(field=field):
                    grid = {"source_px": 512, "zoom": 20, "patch_px": 640}
                    grid[field] = value
                    bbox.write_text(json.dumps({"grid": grid}))
                    with self.assertRaisesRegex(ValueError, message):
                        subject.read_satellite_source_px(
                            Path("unused"), satellite)

    def test_config_already_in_output_is_not_copied_onto_itself(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            config = output / "aggregator_config.yaml"
            config.write_text("kind: test\n")

            subject.copy_aggregator_config(config, output)

            self.assertEqual(config.read_text(), "kind: test\n")

    def test_documented_loci_aggregator_config_decodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "aggregator_config.yaml"
            config.write_text(
                "kind: SafaPlusNormalizedLandmarkAggregatorConfig\n"
                "image_similarity_matrix_path: /tmp/wag.pt\n"
                "landmark_similarity_matrix_path: /tmp/correspondence.pt\n"
                "image_sigma: 0.1809\n"
                "landmark_sigma: 0.4673\n"
                "landmark_use_raw_residual: false\n"
                "allow_legacy_similarity_identity: true\n"
            )

            loaded = subject.load_aggregator_config(config)

            self.assertEqual(loaded.image_sigma, 0.1809)
            self.assertEqual(loaded.landmark_sigma, 0.4673)
            self.assertFalse(loaded.landmark_use_raw_residual)
            self.assertTrue(loaded.allow_legacy_similarity_identity)

    def test_path_identity_rejects_wrong_leg_with_same_panorama_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            live = root / "live_leg"
            wrong = root / "wrong_leg"
            live.mkdir()
            wrong.mkdir()
            (live / "pano_id_mapping.csv").write_text(
                "pano_id,lat,lon\nf0000,42.0,-71.0\nf0001,42.1,-71.1\n")
            wrong_mapping = (
                "pano_id,lat,lon\nf0000,44.0,-71.0\nf0001,44.1,-71.1\n")
            (wrong / "pano_id_mapping.csv").write_text(wrong_mapping)
            wrong_hash = hashlib.sha256(wrong_mapping.encode()).hexdigest()

            with self.assertRaisesRegex(ValueError, "different dataset revision or leg"):
                subject.validate_path_dataset_identity(
                    {"dataset_hash": wrong_hash}, live,
                    require_identity=True)

    def test_path_identity_accepts_exact_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            mapping = "pano_id,lat,lon\nf0000,42.0,-71.0\n"
            (dataset / "pano_id_mapping.csv").write_text(mapping)

            subject.validate_path_dataset_identity(
                {"dataset_hash": hashlib.sha256(mapping.encode()).hexdigest()},
                dataset,
                require_identity=True)

    def test_legacy_path_identity_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            (dataset / "pano_id_mapping.csv").write_text(
                "pano_id,lat,lon\nf0000,42.0,-71.0\n")
            legacy = {"dataset_hash": "new"}

            with self.assertRaisesRegex(ValueError, "allow-legacy-path-identity"):
                subject.validate_path_dataset_identity(
                    legacy, dataset, require_identity=True)
            with self.assertWarnsRegex(RuntimeWarning, "no cryptographic"):
                subject.validate_path_dataset_identity(
                    legacy, dataset, allow_legacy=True,
                    require_identity=True)


if __name__ == "__main__":
    unittest.main()
