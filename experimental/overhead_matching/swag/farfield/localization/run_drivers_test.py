import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import msgspec
import numpy as np

from experimental.overhead_matching.swag.farfield import (
    artifact,
    build_config,
    geometry as geo,
    paths as paths_lib,
)
from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    filter_catalog,
    metrics,
    run_export,
    run_io,
    run_localization,
    run_identity,
    runner,
    structs,
)


DATASET = "tiny_harbor"
INPUT_DIGEST = "a" * 64


def tiny_export(input_dir: Path, build_identity: str, input_contract: dict) \
        -> export_ingest.ExportData:
    frame = geo.RegionFrame(42.35, -71.05)
    landmarks = [
        structs.LandmarkEntry("osm:node:1", 42.36, -71.05, "x", 25.0),
        structs.LandmarkEntry("osm:node:2", 42.35, -71.03, "y", 25.0),
    ]
    global_id = ("object_tracks:tiny_harbor:v1@sha256:" + "b" * 64
                 + "#T1")
    table = structs.CompatibilityTable(
        global_id, "m", [structs.CompatibilityEntry("osm:node:1", 1.0)],
        0.0, -4.0, 4.0, "fast")
    odometry = [
        structs.OdometryDelta(k, 40.0, 0.0, 0.0, 1.0, 0.05)
        for k in range(1, 6)]
    truth = [
        structs.TruthPose(k, 40.0 * k, 0.0, 90.0) for k in range(6)]
    meta = export_ingest.ExportMeta(
        schema_version=export_ingest.EXPORT_SCHEMA,
        message_schema_version=structs.SCHEMA_VERSION,
        dataset=DATASET,
        scenario_name="tiny",
        anchor_lat_deg=42.35,
        anchor_lon_deg=-71.05,
        n_keyframes=6,
        matcher_version="m",
        matching_coverage="complete",
        max_visible_range_m=10000.0,
        landmark_position_sigma_m=25.0,
        nominal_forward={
            "bearing_camera_cw_deg": 0.0,
            "mounting_id": "test-rig",
            "approved_at": "2026-08-23T00:00:00Z",
        },
        motion={"course_heading_status": "gps_course_diagnostic_only"},
        reducer={"name": "epoch_fused_compat_v1"})
    reference = artifact.ArtifactRef(
        kind=paths_lib.LOCALIZATION_INPUTS,
        dataset=DATASET,
        version="v1",
        manifest_digest=INPUT_DIGEST,
        content_digest="c" * 64,
        path=str(input_dir))
    return export_ingest.ExportData(
        artifact_ref=reference,
        manifest=SimpleNamespace(
            config={"build_identity": build_identity,
                    "orchestration": input_contract}),
        meta=meta,
        frame=frame,
        catalog=filter_catalog.LandmarkCatalog(
            [item.landmark_id for item in landmarks],
            [0.0, 1000.0], [1000.0, 0.0],
            max_visible_range_m=10000.0, position_sigma_m=25.0),
        landmarks=landmarks,
        odometry=odometry,
        measurements=[structs.TrackletMeasurement(
            global_id, 2, 10.0, 100.0)],
        tables={global_id: table},
        truth=truth)


def localization_config(*, init="uniform", prior_sigma_m=None,
                        bearings_enabled=True) -> dict:
    template = structs.FilterConfig(
        n_particles=500,
        seed=3,
        init=structs.UniformBoxInit(-1.0, 1.0, -1.0, 1.0),
        position_roughening_m=25.0,
        heading_roughening_deg=1.0,
        checkpoint_every=2)
    config = msgspec.to_builtins(template)
    del config["init"]
    config.pop("kind", None)
    config["proposal"].pop("kind", None)
    config["modes"].pop("kind", None)
    config.update({
        "run_name": "run",
        "init": init,
        "prior_sigma_m": prior_sigma_m,
        "margin_m": 500.0,
        "bearings_enabled": bearings_enabled,
        "ablation_tags": [],
        "position_mass_radii_m": [100.0, 500.0],
        # The build config spells the cap as enabled + softness; FilterConfig
        # holds a RangeCap or None.
        "range_cap": {"enabled": template.range_cap is not None,
                      "softness_frac": (template.range_cap
                                        or structs.RangeCap()).softness_frac},
    })
    return config


def write_build_config(root: Path, localization: dict) \
        -> tuple[Path, str, str, dict]:
    path = build_config.create(
        root / "build",
        dataset=DATASET,
        config={
            "artifacts": {"localization_inputs_version": "v1",
                          "object_tracks_version": "tracks-v1"},
            "localization_inputs": {},
            "gps_course": {},
            "localization": localization,
        },
        generator="test",
        inputs={})
    document = build_config.load(path.parent)
    return (path,
            run_export.orchestration_contract(document)["config_digest"],
            document["build_identity"],
            run_export.localization_inputs_contract(document))


def export_argv(input_dir: Path, out_dir: Path, config_path: Path,
                digest: str):
    return [
        "run_export",
        "--input_dir", str(input_dir),
        "--run_dir", str(out_dir),
        "--build_config", str(config_path),
        "--orchestration_config_digest", digest,
    ]


def expected_run_dir(root: Path, build_identity: str) -> Path:
    return root / run_identity.localization_run_version(
        "run", "tracks-v1", build_identity)


class RunExportTest(unittest.TestCase):
    def test_rejects_run_dir_that_conflicts_with_stable_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "localization_inputs"
            (config_path, digest, build_identity,
             input_contract) = write_build_config(root, localization_config())
            data = tiny_export(input_dir, build_identity, input_contract)
            argv = export_argv(
                input_dir, root / "conflicting-run-name", config_path, digest)
            with mock.patch.object(run_export.export_ingest, "load",
                                   return_value=data), \
                    mock.patch("sys.argv", argv), \
                    mock.patch("sys.stderr", new_callable=io.StringIO) \
                    as stderr, self.assertRaises(SystemExit):
                run_export.main()
            self.assertIn("immutable localization run identity",
                          stderr.getvalue())

    def test_stage_scoped_cross_build_inputs_and_symlinked_build_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (config_path, digest, build_identity,
             input_contract) = write_build_config(root, localization_config())
            data = tiny_export(
                root / "localization_inputs", build_identity, input_contract)
            data.manifest.config["build_identity"] = "0" * 64
            run_export._load_config(config_path, data, digest)

            changed_contract = dict(input_contract)
            changed_contract["config_digest"] = "0" * 64
            data.manifest.config["orchestration"] = changed_contract
            with self.assertRaisesRegex(ValueError, "stage-scoped recipe"):
                run_export._load_config(config_path, data, digest)

            data.manifest.config["build_identity"] = build_identity
            data.manifest.config["orchestration"] = input_contract
            linked_dir = root / "linked"
            linked_dir.mkdir()
            linked_config = linked_dir / build_config.BUILD_CONFIG_NAME
            linked_config.symlink_to(config_path)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                run_export._load_config(linked_config, data, digest)

    def test_uniform_run_records_provenance_and_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "localization_inputs"
            (config_path, digest, build_identity,
             input_contract) = write_build_config(root, localization_config())
            out = expected_run_dir(root, build_identity)
            data = tiny_export(input_dir, build_identity, input_contract)
            with mock.patch.object(run_export.export_ingest, "load",
                                   return_value=data), mock.patch(
                    "sys.argv", export_argv(
                        input_dir, out, config_path, digest)), mock.patch(
                    "sys.stdout", new_callable=io.StringIO) as stdout:
                run_export.main()
            console = stdout.getvalue()
            summary_path = out / metrics.POSITION_MASS_SUMMARY_NAME
            summary = json.loads(summary_path.read_text())
            data = run_io.read_run(out)
            self.assertTrue(
                (out / "posterior_predictive_bearings.json").is_file())
        manifest = data.manifest
        self.assertEqual(manifest.export_dir, str(input_dir))
        self.assertEqual(manifest.max_visible_range_m, 10000.0)
        self.assertTrue(manifest.git_commit)
        self.assertIn("--build_config", manifest.argv)
        self.assertEqual(manifest.run_kind, "evaluation")
        self.assertEqual(manifest.localization_inputs_manifest_sha256,
                         INPUT_DIGEST)
        self.assertEqual(manifest.ablation_tags, [])
        self.assertEqual(manifest.truth_position_schema,
                         "farfield_truth_position/v1")
        self.assertEqual(manifest.position_mass_metric.radii_m,
                         [100.0, 500.0])
        self.assertIsInstance(manifest.filter_config.init,
                              structs.UniformBoxInit)
        self.assertEqual(len(data.measurements), 1)
        self.assertEqual(len(data.health), 6)
        expected_metrics = {
            "posterior_position_probability_mass_within_true_position_radius"
            "@1:radius_m=100",
            "posterior_position_probability_mass_within_true_position_radius"
            "@1:radius_m=500",
        }
        self.assertTrue(all(
            set(record.position_probability_mass) == expected_metrics
            for record in data.health))
        key100 = metrics.position_mass_metric_key(
            manifest.position_mass_metric, 100.0)
        key500 = metrics.position_mass_metric_key(
            manifest.position_mass_metric, 500.0)
        self.assertTrue(all(
            0.0 <= record.position_probability_mass[key100]
            <= record.position_probability_mass[key500] <= 1.0
            for record in data.health))
        self.assertEqual(summary["reference_position"], "truth")
        self.assertEqual(summary["primary_radius_m"], 500.0)
        self.assertEqual(set(summary["radii"]), {"100", "500"})
        self.assertIn("--- PRIMARY LOCALIZATION METRIC ---", console)
        self.assertIn("within 500 m of the true position over distance travelled", console)
        self.assertLess(console.index("PRIMARY: normalized posterior mass"),
                        console.index("MAP position error"))

    def test_mass_recorder_centers_both_radii_on_true_position(self):
        config = metrics.position_mass_metric_config()
        recorder = runner.PositionMassRecorder(
            [structs.TruthPose(7, 1000.0, -500.0, 0.0)], config)
        belief = SimpleNamespace(
            east_m=np.array([1000.0, 1100.0, 1500.0, 1500.01]),
            north_m=np.full(4, -500.0),
            normalized_weights=lambda: np.array([0.1, 0.2, 0.3, 0.4]))

        recorder.keyframe_end(7, belief, None)

        values = recorder.by_keyframe[7]
        self.assertAlmostEqual(
            values[metrics.position_mass_metric_key(config, 100.0)], 0.3)
        self.assertAlmostEqual(
            values[metrics.position_mass_metric_key(config, 500.0)], 0.6)

    def test_no_bearings_writes_the_run_that_happened(self):
        # The old driver wrote the FULL unconsumed measurements under
        # --no_bearings, so every odometry-only control on disk described
        # a run that never happened.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "localization_inputs"
            (config_path, digest, build_identity,
             input_contract) = write_build_config(
                 root, localization_config(bearings_enabled=False))
            out = expected_run_dir(root, build_identity)
            data = tiny_export(input_dir, build_identity, input_contract)
            with mock.patch.object(run_export.export_ingest, "load",
                                   return_value=data), mock.patch(
                    "sys.argv", export_argv(
                        input_dir, out, config_path, digest)):
                run_export.main()
            data = run_io.read_run(out)
        self.assertEqual(data.measurements, [])
        self.assertEqual(data.tables, {})
        self.assertEqual(data.manifest.matcher_version, "m")
        self.assertEqual(data.manifest.run_kind, "diagnostic_control")
        self.assertEqual(data.manifest.ablation_tags, ["no_bearings"])

    def test_truth_init_requires_prior_sigma(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "localization_inputs"
            (config_path, digest, build_identity,
             input_contract) = write_build_config(
                 root, localization_config(init="truth_position"))
            data = tiny_export(input_dir, build_identity, input_contract)
            argv = export_argv(
                input_dir, expected_run_dir(root, build_identity), config_path,
                digest)
            with mock.patch.object(run_export.export_ingest, "load",
                                   return_value=data), \
                    mock.patch("sys.argv", argv), \
                    self.assertRaises(SystemExit):
                run_export.main()


class RunLocalizationTest(unittest.TestCase):
    def test_synthetic_run_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            argv = ["run_localization",
                    "--scenario", "harbor_loop",
                    "--run_dir", str(out),
                    "--init", "local",
                    "--n_particles", "400",
                    "--max_visible_range_m", "10000",
                    "--prior_sigma_m", "300",
                    "--prior_offset_east_m", "100",
                    "--prior_offset_north_m", "-100",
                    "--keyframe_period_s", "2.0",
                    "--epoch_length", "5",
                    "--bearing_sigma_deg", "1.0"]
            with mock.patch("sys.argv", argv):
                run_localization.main()
            data = run_io.read_run(out)
        self.assertEqual(data.manifest.export_dir, "synthetic:harbor_loop")
        self.assertEqual(data.manifest.max_visible_range_m, 10000.0)
        self.assertTrue(data.manifest.git_commit)
        self.assertEqual(data.manifest.run_kind, "synthetic")
        self.assertTrue(all(record.position_probability_mass
                            for record in data.health))
        self.assertGreater(len(data.health), 10)

    def test_kidnap_requires_explicit_teleport(self):
        argv = ["run_localization", "--scenario", "harbor_loop",
                "--run_dir", "/tmp/x", "--init", "global",
                "--n_particles", "400", "--max_visible_range_m", "10000",
                "--box_halfwidth_m", "2500",
                "--keyframe_period_s", "2.0", "--epoch_length", "5",
                "--bearing_sigma_deg", "1.0", "--kidnap_at", "20"]
        with mock.patch("sys.argv", argv), self.assertRaises(SystemExit):
            run_localization.main()


if __name__ == "__main__":
    unittest.main()
