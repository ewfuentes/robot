import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml
from python.runfiles import runfiles

from experimental.overhead_matching.swag.farfield import artifact


_PIPELINE_RUNFILE = (
    "robot/experimental/overhead_matching/swag/farfield/pipeline")
_CONFIG_RUNFILE = (
    "robot/experimental/overhead_matching/swag/farfield/"
    "configs/harbor_example.yaml")


class PipelineCliTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temporary.name)
        self.runfiles = runfiles.Create()
        self.assertIsNotNone(self.runfiles)
        self.pipeline = self._runfile(_PIPELINE_RUNFILE)
        self.example_config = self._runfile(_CONFIG_RUNFILE)

    def tearDown(self):
        self.temporary.cleanup()

    def _runfile(self, logical_path: str) -> Path:
        resolved = self.runfiles.Rlocation(logical_path)
        self.assertIsNotNone(resolved, logical_path)
        path = Path(resolved)
        self.assertTrue(path.exists(), path)
        return path

    def _run_pipeline(self, *arguments) -> str:
        environment = os.environ.copy()
        environment.update(self.runfiles.EnvVars())
        environment.pop("BUILD_WORKSPACE_DIRECTORY", None)
        result = subprocess.run(
            [str(self.pipeline), *(str(value) for value in arguments)],
            cwd=self.work_dir,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            "packaged pipeline command failed:\n"
            f"argv: {result.args!r}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
        )
        return result.stdout

    def _fixture(self) -> tuple[Path, Path]:
        root = self.work_dir / "farfield-root"
        dataset_base = root / "datasets" / "cli_ds"
        panorama = dataset_base / "panorama"
        panorama.mkdir(parents=True)
        (dataset_base / "pipeline_metadata.json").write_text(
            "{}\n", encoding="utf-8")
        (dataset_base / "frames_gps.csv").write_text(
            "keyframe_idx,lat,lon\n0,42.0,-71.0\n", encoding="utf-8")
        (panorama / "000000.jpg").write_bytes(b"jpeg fixture")

        checkpoint = root / "models" / "sam2" / "sam2.1_hiera_large.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint fixture")
        calibration = {
            "schema": "farfield_nominal_forward/v1",
            "frame": "camera_centre_column_nominal_forward_axis_v1",
            "dataset": "cli_ds",
            "version": "v1",
            "mounting_id": "cli-mount",
            "panorama_column": 2.0,
            "panorama_width": 4,
            "bearing_camera_cw_deg": 0.0,
            "uncertainty_deg": 1.0,
            "evidence_frame_ids": ["0"],
            "operator": "pipeline-cli-test",
            "approved_at": "2026-08-24T12:00:00+00:00",
            "approved": True,
            "notes": "packaged CLI fixture",
        }
        (dataset_base / "nominal_forward.json").write_text(
            json.dumps(calibration), encoding="utf-8")

        catalog_dir = (
            root / "artifacts" / "catalogs" / "cli_ds" / "v3_trimmed")
        with artifact.ArtifactDirectoryBuilder(
                catalog_dir,
                kind="catalogs",
                dataset="cli_ds",
                version="v3_trimmed",
                generator="pipeline_cli_test",
                git_commit="test",
                arguments=(),
                upstreams=(),
                config={},
                declared_outputs=("catalog.feather",)) as builder:
            builder.output_path("catalog.feather").write_bytes(
                b"catalog fixture")

        config = yaml.safe_load(self.example_config.read_text(encoding="utf-8"))
        config["tracking"]["range"] = {"k_start": 0, "k_end": 0}
        config["execution"]["batch_gcs_prefix"] = (
            "gs://fixture/farfield/cli_ds/b001")
        config["localization"]["run_name"] = "cli_ds_v1"
        source_config = self.work_dir / "pipeline.yaml"
        source_config.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return root, source_config

    def test_packaged_new_build_and_status_use_custom_root(self):
        root, source_config = self._fixture()
        build_dir = root / "builds" / "cli_ds" / "b001"

        created = self._run_pipeline(
            "new-build",
            "--dataset", "cli_ds",
            "--farfield_root", root,
            "--build_name", "b001",
            "--config", source_config,
            "--notes", "packaged CLI test",
        )
        config_path = build_dir / "build_config.json"
        self.assertIn(f"build created: {build_dir}", created)
        self.assertIn(f"config recorded: {config_path}", created)

        document = json.loads(config_path.read_text(encoding="utf-8"))
        inputs = document["inputs"]
        dataset_base = root / "datasets" / "cli_ds"
        self.assertEqual(inputs["farfield_root"], str(root.resolve()))
        self.assertEqual(inputs["dataset_base"], str(dataset_base.resolve()))
        self.assertEqual(
            inputs["sam2_checkpoint"],
            str((root / "models/sam2/sam2.1_hiera_large.pt").resolve()))
        self.assertEqual(
            inputs["motion_source"],
            str((dataset_base / "frames_gps.csv").resolve()))
        self.assertEqual(
            inputs["nominal_forward_calibration"],
            str((dataset_base / "nominal_forward.json").resolve()))

        status = self._run_pipeline("status", "--build_dir", build_dir)
        self.assertIn(f"build {build_dir}", status)
        for stage in (
                "extract", "track", "audit", "bearings", "match",
                "diagnostics", "localization_inputs", "localize", "viewer"):
            self.assertIn(f"  {stage:<20} pending", status)
        self.assertNotIn("INVALID", status)


if __name__ == "__main__":
    unittest.main()
