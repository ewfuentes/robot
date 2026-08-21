import argparse
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from experimental.overhead_matching.swag.farfield import (
    provenance,
    testing,
)
from experimental.overhead_matching.swag.farfield.extraction import (
    extract_landmarks as ex,
    prompts,
)

RES = 24  # tiny faces keep the request-building tests fast


def make_pinhole_render(pinhole_dir: Path, stems, res=RES):
    for stem in stems:
        face_dir = Path(pinhole_dir) / stem
        face_dir.mkdir(parents=True, exist_ok=True)
        for face in prompts.PINHOLE_FACES:
            Image.new("RGB", (res, res), (10, 120, 200)).save(
                face_dir / f"{face}.jpg")


def ok_record(stem, landmarks):
    return {
        "key": stem,
        "response": {"candidates": [{"content": {"parts": [{
            "text": json.dumps({"location_type": "harbor",
                                "landmarks": landmarks}),
        }]}}]},
    }


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class ExtractionFixture(unittest.TestCase):
    """A synthetic dataset + pinhole render inside a fake farfield root."""

    N_FRAMES = 4

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.dataset_base = testing.make_dataset(
            self.tmp / "datasets" / "testset", n_frames=self.N_FRAMES)
        self.panorama_dir = self.dataset_base / "panorama"
        self.stems = sorted(p.stem for p in self.panorama_dir.glob("*.jpg"))
        self.config = ex.Config(
            dataset="testset",
            root=self.tmp,
            version="v1",
            pinhole_version="v1",
            dataset_base=self.dataset_base,
            panorama_dir=self.panorama_dir,
            pinhole_dir=self.tmp / "artifacts" / "pinhole_images" / "testset"
            / "v1",
            artifact_dir=self.tmp / "artifacts" / "frame_landmarks" /
            "testset" / "v1",
            prompt_type="osm_tags_farfield_v2",
            pinhole_resolution=RES,
            media_resolution="MEDIA_RESOLUTION_HIGH",
            thinking_level="HIGH",
            num_workers=1,
            allow_incomplete=False,
            force=False,
            start_stage=1,
            end_stage=3,
            execution=argparse.Namespace(
                model="gemini-3.7-flash", online=False, gcs_prefix=None,
                parallel=1, poll_interval=1, cost_limit=50.0,
                approve_cost=False),
        )

    def render(self):
        make_pinhole_render(self.config.pinhole_dir, self.stems)

    def build_requests(self):
        return prompts.write_requests(
            self.config.pinhole_dir, self.panorama_dir,
            self.config.requests_dir,
            prompt_type=self.config.prompt_type,
            media_resolution=self.config.media_resolution,
            thinking_level=self.config.thinking_level,
            num_workers=1, disable_tqdm=True)


class RequestBuildingTest(ExtractionFixture):
    def test_one_request_per_stem_in_stem_order(self):
        self.render()
        written = self.build_requests()
        self.assertEqual(len(written), 1)
        records = [json.loads(line) for line in
                   written[0].read_text().splitlines()]
        self.assertEqual([r["key"] for r in records], self.stems)
        for record in records:
            parts = record["request"]["contents"][0]["parts"]
            self.assertEqual(len(parts), 5)
            self.assertEqual(sum("inline_data" in p for p in parts), 4)
            self.assertEqual(parts[4]["text"], prompts.USER_PROMPT)
            self.assertEqual(
                record["request"]["systemInstruction"]["parts"][0]["text"],
                prompts.SYSTEM_PROMPTS["osm_tags_farfield_v2"])

    def test_stale_rendered_stems_are_excluded(self):
        self.render()
        # A stem the dataset no longer contains (post-trim leftover).
        make_pinhole_render(self.config.pinhole_dir,
                            ["f9999,0.0000000,0.0000000,"])
        written = self.build_requests()
        keys = [json.loads(line)["key"] for line in
                written[0].read_text().splitlines()]
        self.assertEqual(keys, self.stems)

    def test_missing_render_is_an_error_not_a_gap(self):
        self.render()
        shutil.rmtree(self.config.pinhole_dir / self.stems[1])
        with self.assertRaisesRegex(RuntimeError, "no pinhole render"):
            self.build_requests()

    def test_request_fingerprints_pin_prompt_and_request(self):
        self.render()
        self.build_requests()
        prompt_fp = ex.prompt_fingerprint(self.config.requests_dir)
        # What went out is byte-identical to the registry text.
        self.assertEqual(prompt_fp["prompt_sha256"],
                         prompts.prompt_sha256("osm_tags_farfield_v2"))
        request_fp = ex.request_fingerprint(self.config.requests_dir)
        self.assertEqual(request_fp["n_requests"], self.N_FRAMES)
        self.assertTrue(request_fp["request_sha256"])


class PinholeReuseTest(ExtractionFixture):
    def test_reuse_accepts_a_complete_matching_render(self):
        self.render()
        self.assertTrue(ex.check_pinhole_reuse(self.config))

    def test_reuse_rejects_wrong_resolution(self):
        self.render()
        self.config.pinhole_resolution = RES * 2
        self.assertFalse(ex.check_pinhole_reuse(self.config))

    def test_reuse_rejects_missing_face_or_stem(self):
        self.render()
        (self.config.pinhole_dir / self.stems[0] / "yaw_090.jpg").unlink()
        self.assertFalse(ex.check_pinhole_reuse(self.config))
        shutil.rmtree(self.config.pinhole_dir / self.stems[0])
        self.assertFalse(ex.check_pinhole_reuse(self.config))

    def test_reuse_rejects_absent_dir(self):
        self.assertFalse(ex.check_pinhole_reuse(self.config))

    def test_pinhole_manifest_records_kind_and_observed_geometry(self):
        self.render()
        ex.write_pinhole_manifest(self.config)
        manifest = provenance.read(self.config.pinhole_dir)
        self.assertEqual(manifest["kind"], "pinhole_images")
        self.assertEqual(manifest["dataset"], "testset")
        self.assertEqual(manifest["version"], "v1")
        self.assertEqual(manifest["config"]["res_x"], RES)
        self.assertEqual(manifest["config"]["n_panoramas"], self.N_FRAMES)
        self.assertEqual(manifest["inputs"]["panorama_dir"],
                         "datasets/testset/panorama")


class CompletenessGateTest(ExtractionFixture):
    def write_main_results(self):
        """stem0 ok, stem1 legitimately empty, stem2 failed, stem3 missing."""
        landmark = testing.landmark("Fort Point Light",
                                    [(0, 100, 100, 200, 200)])
        write_jsonl(self.config.main_predictions, [
            ok_record(self.stems[0], [landmark]),
            ok_record(self.stems[1], []),
            {"key": self.stems[2], "response": None,
             "error": "TPU device returned error"},
        ])

    def test_missing_responses_are_detected_and_classified(self):
        self.write_main_results()
        report = ex.validate_predictions(self.config.sentences_dir,
                                         self.panorama_dir)
        self.assertEqual(report["ok"], [self.stems[0]])
        self.assertEqual(report["empty"], [self.stems[1]])
        self.assertEqual(report["failed"], [self.stems[2]])
        self.assertEqual(report["missing"], [self.stems[3]])
        self.assertEqual(report["n_panoramas"], self.N_FRAMES)
        # The retry list is exactly failed + missing.
        self.assertEqual(report["failed"] + report["missing"],
                         [self.stems[2], self.stems[3]])
        self.assertFalse(ex.print_validation(report, repair_hint="<hint>"))

    def test_complete_when_every_stem_has_a_usable_response(self):
        self.write_main_results()
        # Repair the failed and missing stems via a retry-shaped directory
        # that sorts after 000_main and therefore supersedes it key by key.
        retry = (self.config.sentences_dir / "results" / "zz_retry_1" /
                 "prediction-retry-1" / "predictions.jsonl")
        write_jsonl(retry, [ok_record(self.stems[2], []),
                            ok_record(self.stems[3], [])])
        report = ex.validate_predictions(self.config.sentences_dir,
                                         self.panorama_dir)
        self.assertEqual(report["failed"], [])
        self.assertEqual(report["missing"], [])
        self.assertTrue(ex.print_validation(report, repair_hint="<hint>"))

    def test_retry_request_records_selects_exactly_the_broken_keys(self):
        self.render()
        self.build_requests()
        self.write_main_results()
        report = ex.validate_predictions(self.config.sentences_dir,
                                         self.panorama_dir)
        wanted = set(report["failed"] + report["missing"])
        subset = ex.retry_request_records(self.config.requests_dir, wanted)
        self.assertEqual({r["key"] for r in subset}, wanted)
        # And the stored request is complete enough to re-run: it carries the
        # prompt and the images verbatim.
        for record in subset:
            self.assertIn("systemInstruction", record["request"])

    def test_coverage_summary_records_the_gap(self):
        self.write_main_results()
        report = ex.validate_predictions(self.config.sentences_dir,
                                         self.panorama_dir)
        summary = ex.coverage_summary(report)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["missing_keys"],
                         sorted([self.stems[2], self.stems[3]]))
        self.assertEqual(summary["n_with_landmarks"], 1)
        self.assertEqual(summary["n_empty_responses"], 1)
        self.assertEqual(summary["n_no_usable_response"], 2)


class ManifestTest(ExtractionFixture):
    def complete_extraction(self):
        self.render()
        self.build_requests()
        landmark = testing.landmark("Boston Light", [(90, 10, 10, 40, 40)])
        write_jsonl(self.config.main_predictions,
                    [ok_record(stem, [landmark]) for stem in self.stems])

    def test_manifest_content(self):
        self.complete_extraction()
        report = ex.validate_predictions(self.config.sentences_dir,
                                         self.panorama_dir)
        ex.write_frame_landmarks_manifest(self.config, report)

        manifest = provenance.read(self.config.artifact_dir)
        self.assertEqual(manifest["kind"], "frame_landmarks")
        self.assertEqual(manifest["dataset"], "testset")
        self.assertEqual(manifest["version"], "v1")
        self.assertEqual(manifest["generator"], ex.GENERATOR)
        self.assertEqual(manifest["inputs"]["dataset_base"],
                         "datasets/testset")
        self.assertEqual(manifest["inputs"]["pinhole_images"],
                         "artifacts/pinhole_images/testset/v1")

        config = manifest["config"]
        self.assertEqual(config["prompt_type"], "osm_tags_farfield_v2")
        self.assertEqual(config["prompt_sha256"],
                         prompts.prompt_sha256("osm_tags_farfield_v2"))
        self.assertTrue(config["request_sha256"])
        self.assertEqual(config["model"], "gemini-3.7-flash")
        self.assertEqual(config["pinhole_resolution"], RES)
        self.assertEqual(config["media_resolution"],
                         "MEDIA_RESOLUTION_HIGH")
        self.assertEqual(config["thinking_level"], "HIGH")
        self.assertEqual(config["execution"], "batch")
        self.assertTrue(config["complete"])
        self.assertEqual(config["n_panoramas"], self.N_FRAMES)
        self.assertEqual(config["n_with_landmarks"], self.N_FRAMES)
        self.assertEqual(config["n_prediction_lines"], self.N_FRAMES)

    def test_manifest_records_an_accepted_gap(self):
        self.render()
        self.build_requests()
        write_jsonl(self.config.main_predictions,
                    [ok_record(self.stems[0], [])])
        report = ex.validate_predictions(self.config.sentences_dir,
                                         self.panorama_dir)
        ex.write_frame_landmarks_manifest(self.config, report)
        config = provenance.read(self.config.artifact_dir)["config"]
        self.assertFalse(config["complete"])
        self.assertEqual(config["missing_keys"], self.stems[1:])
        self.assertIn("retry_failed", config["warning"])


if __name__ == "__main__":
    unittest.main()
