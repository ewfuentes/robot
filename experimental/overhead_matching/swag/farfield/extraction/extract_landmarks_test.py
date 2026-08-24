import argparse
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from experimental.overhead_matching.swag.farfield import (
    artifact,
    build_config,
    llm_lifecycle,
    paths as paths_lib,
    testing,
)
from experimental.overhead_matching.swag.farfield.collection import (
    run_farfield_collection as collection_runner,
)
from experimental.overhead_matching.swag.farfield.extraction import (
    extract_landmarks as ex,
    prompts,
)


RESOLUTION = 24
REAL_PROCESS_PANORAMAS = ex.panorama_to_pinhole.process_panoramas


def provider_prediction(*, landmarks=None):
    return {
        "location_type": "harbor",
        "landmarks": [] if landmarks is None else landmarks,
    }


def provider_landmark():
    return {
        "primary_tag": {"key": "man_made", "value": "tower"},
        "additional_tags": [{"key": "name", "value": "Boston Light"}],
        "confidence": "high",
        "bounding_boxes": [{
            "yaw_angle": "90",
            "ymin": 100,
            "xmin": 200,
            "ymax": 400,
            "xmax": 500,
        }],
        "description": "Boston Light",
    }


def response(prediction):
    return {
        "candidates": [{
            "content": {"parts": [{"text": json.dumps(prediction)}]},
        }],
        "usageMetadata": {"totalTokenCount": 10},
    }


class ExtractionFixture(unittest.TestCase):
    N_FRAMES = 3

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.dataset_base = testing.make_dataset(
            self.tmp / "testset", n_frames=self.N_FRAMES,
            pano_size=(32, 16))
        self.stems = sorted(
            path.stem for path in (self.dataset_base / "panorama").glob("*.jpg"))
        self.config = {
            "artifacts": {
                "frame_landmarks_version": "frame-v7",
                "pinhole_images_version": "pinhole-v3",
            },
            "extraction": {
                "model": "gemini-test-model",
                "prompt_type": "osm_tags_farfield_v2",
                "pinhole_resolution": RESOLUTION,
                "media_resolution": "MEDIA_RESOLUTION_HIGH",
                "thinking_level": "HIGH",
            },
            "execution": {
                "llm_transport": "batch",
                "batch_gcs_prefix": "gs://unit-test/extraction",
                "approve_cost": False,
            },
            "cost": {"limit_usd": 1000.0},
        }
        self.build_dir = self.tmp / "build"
        self.args = self.make_recipe(self.build_dir, self.config)
        worker_patch = mock.patch.object(ex, "NUM_WORKERS", 1)
        worker_patch.start()
        self.addCleanup(worker_patch.stop)

    def make_recipe(self, build_dir, config):
        source_digests = paths_lib.dataset_source_digests(self.dataset_base)
        build_config.create(
            build_dir,
            dataset="testset",
            config=copy.deepcopy(config),
            generator="extract_landmarks_test",
            inputs={
                "dataset_base": str(self.dataset_base.resolve()),
                **source_digests,
            },
        )
        document = build_config.load(build_dir)
        selected = {
            key: build_config.value(document, key)
            for key in ex.EXTRACTION_CONFIG_KEYS
        }
        return argparse.Namespace(
            dataset="testset",
            dataset_base=self.dataset_base,
            pinhole_output_dir=self.tmp / "pinhole",
            output_dir=self.tmp / "frames",
            build_config=build_dir / build_config.BUILD_CONFIG_NAME,
            orchestration_config_digest=artifact.sha256_json(selected),
            online=False,
            gcs_prefix="gs://unit-test/extraction",
            parallel=2,
            poll_interval=1,
            cost_limit=1000.0,
            approve_cost=False,
        )

    def fake_render(self, input_dir, output_dir, fov_x, fov_y, res_x,
                    res_y, num_workers):
        self.assertEqual((res_x, res_y), (RESOLUTION, RESOLUTION))
        REAL_PROCESS_PANORAMAS(
            input_dir, output_dir, fov_x, fov_y, res_x, res_y,
            num_workers=num_workers)

    def fake_success_transport(self, execution, input_path, output_path, *, tag):
        del execution, tag
        records = [json.loads(line) for line in
                   Path(input_path).read_text().splitlines()]
        with Path(output_path).open("w") as stream:
            for record in records:
                stream.write(json.dumps({
                    "key": record["key"],
                    "response": response(provider_prediction()),
                }) + "\n")

    def run_successfully(self):
        with mock.patch.object(
                ex.panorama_to_pinhole, "process_panoramas",
                side_effect=self.fake_render), mock.patch.object(
                    ex.vbm, "run_requests",
                    side_effect=self.fake_success_transport):
            return ex.run(self.args, arguments=("extract_landmarks", "--test"))


class CliAndConfigTest(ExtractionFixture):
    def test_cli_is_the_current_pipeline_boundary_without_legacy_knobs(self):
        parser = ex.make_parser()
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertTrue({
            "--dataset", "--dataset_base", "--pinhole_output_dir",
            "--output_dir", "--build_config",
            "--orchestration_config_digest", "--online", "--gcs_prefix",
            "--parallel", "--poll_interval", "--cost_limit",
            "--approve_cost", "--prepare_adoption",
        }.issubset(options))
        self.assertTrue({
            "--run_dir", "--allow_incomplete", "--retry_failed", "--force",
            "--start_stage", "--end_stage", "--validate_only", "--model",
            "--prompt_type", "--pinhole_resolution", "--media_resolution",
            "--thinking_level", "--frame_landmarks_version",
            "--pinhole_version",
        }.isdisjoint(options))

    def test_recipe_owns_model_versions_and_transport(self):
        context = ex.load_context(self.args)
        self.assertEqual(context.frame_version, "frame-v7")
        self.assertEqual(context.pinhole_version, "pinhole-v3")
        self.assertEqual(self.args.model, "gemini-test-model")
        self.assertEqual(context.orchestration, {
            "schema": "farfield_pipeline_stage/v1",
            "stage": "extract",
            "config_digest": self.args.orchestration_config_digest,
        })

    def test_transport_flags_must_agree_with_recipe(self):
        self.args.online = True
        with self.assertRaisesRegex(ValueError, "--online selection disagrees"):
            ex.load_context(self.args)
        self.args.online = False
        self.args.cost_limit = 999.0
        with self.assertRaisesRegex(ValueError, "--cost_limit disagrees"):
            ex.load_context(self.args)

    def test_config_digest_is_exact(self):
        self.args.orchestration_config_digest = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match the immutable"):
            ex.load_context(self.args)

    def test_dataset_mutation_is_rejected_against_frozen_build_identity(self):
        panorama = next(self.dataset_base.glob("panorama/*.jpg"))
        panorama.write_bytes(panorama.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "dataset source bytes differ"):
            ex.load_context(self.args)


class ResponseValidationTest(unittest.TestCase):
    def test_schema_valid_response_is_canonicalized_for_ingest(self):
        result = ex.validate_response(
            "frame", response(provider_prediction(
                landmarks=[provider_landmark()])))
        self.assertEqual(
            result["landmarks"][0]["bounding_boxes"][0]["yaw_angle"], 90)
        # The canonical payload is accepted by the final-artifact validator.
        self.assertEqual(ex._validate_canonical_prediction(result), result)

    def test_legitimately_empty_prediction_is_successful(self):
        self.assertEqual(
            ex.validate_response("frame", response(provider_prediction())),
            provider_prediction())

    def test_unknown_fields_and_fenced_json_are_not_compatibility_paths(self):
        extra = provider_prediction()
        extra["legacy"] = True
        with self.assertRaisesRegex(ValueError, "exact keys"):
            ex.validate_response("frame", response(extra))
        fenced = response(provider_prediction())
        fenced["candidates"][0]["content"]["parts"][0]["text"] = (
            "```json\n{}\n```")
        with self.assertRaisesRegex(ValueError, "invalid strict JSON"):
            ex.validate_response("frame", fenced)

    def test_invalid_yaw_or_zero_width_box_is_not_successful(self):
        landmark = provider_landmark()
        landmark["bounding_boxes"][0]["yaw_angle"] = "45"
        with self.assertRaisesRegex(ValueError, "yaw_angle"):
            ex.validate_response(
                "frame", response(provider_prediction(landmarks=[landmark])))
        landmark = provider_landmark()
        landmark["bounding_boxes"][0]["xmax"] = 200
        with self.assertRaisesRegex(ValueError, "positive width"):
            ex.validate_response(
                "frame", response(provider_prediction(landmarks=[landmark])))

    def test_multiple_candidates_are_ambiguous(self):
        wrapped = response(provider_prediction())
        wrapped["candidates"].append(wrapped["candidates"][0])
        with self.assertRaisesRegex(ValueError, "exactly one candidate"):
            ex.validate_response("frame", wrapped)


class TypedPublicationTest(ExtractionFixture):
    def test_prepare_adoption_succeeds_before_cost_or_provider_boundary(self):
        with mock.patch.object(
                ex.panorama_to_pinhole, "process_panoramas",
                side_effect=self.fake_render), mock.patch.object(
                    ex.llm_cost, "enforce_limit") as cost_gate, \
                mock.patch.object(
                    ex.vbm, "run_requests") as provider:
            first = ex.prepare_adoption(
                self.args,
                arguments=("extract_landmarks", "--prepare_adoption"))

        cost_gate.assert_not_called()
        provider.assert_not_called()
        pinhole_ref, request_ref = first
        self.assertFalse(self.args.output_dir.exists())
        self.assertEqual(
            artifact.load_manifest(Path(request_ref.path)).upstreams,
            (pinhole_ref,))
        request_set = llm_lifecycle.load_request_set(
            Path(request_ref.path) / llm_lifecycle.REQUEST_SET_NAME)
        context = ex.load_context(self.args)
        self.assertTrue(ex._request_set_matches(
            request_set, context, pinhole_ref))
        work_dir = Path(request_ref.path).parent
        self.assertFalse((work_dir / ex.ATTEMPTS_DIR_NAME).exists())
        self.assertFalse((work_dir / "transport").exists())

        with mock.patch.object(
                ex.panorama_to_pinhole, "process_panoramas",
                side_effect=AssertionError("published pinhole must be reused")), \
                mock.patch.object(
                    ex.llm_cost, "enforce_limit",
                    side_effect=AssertionError("cost gate must not run")), \
                mock.patch.object(
                    ex.vbm, "run_requests",
                    side_effect=AssertionError("provider must not run")):
            second = ex.prepare_adoption(
                self.args,
                arguments=("extract_landmarks", "--prepare_adoption"))
        self.assertEqual(second, first)

    def test_two_artifacts_publish_with_exact_files_and_provenance(self):
        pinhole_ref, frame_ref = self.run_successfully()
        self.assertEqual(pinhole_ref.kind, paths_lib.PINHOLE_IMAGES)
        self.assertEqual(frame_ref.kind, paths_lib.FRAME_LANDMARKS)

        pinhole_manifest = artifact.load_manifest(self.args.pinhole_output_dir)
        self.assertEqual(pinhole_manifest.upstreams, ())
        self.assertEqual(len(pinhole_manifest.declared_outputs),
                         self.N_FRAMES * len(prompts.PINHOLE_FACES))
        context = ex.load_context(self.args)
        self.assertEqual(
            dict(pinhole_manifest.config), ex._pinhole_config(context))
        self.assertNotIn("orchestration", pinhole_manifest.config)
        self.assertNotIn("build_identity", pinhole_manifest.config)

        frame_manifest = artifact.load_manifest(self.args.output_dir)
        self.assertEqual(frame_manifest.declared_outputs,
                         ("predictions.jsonl",))
        self.assertEqual(frame_manifest.upstreams[0], pinhole_ref)
        self.assertEqual(frame_manifest.upstreams.count(pinhole_ref), 1)
        self.assertEqual(frame_manifest.upstreams[1].kind,
                         llm_lifecycle.RESULT_ARTIFACT_KIND)
        self.assertEqual(frame_manifest.config["coverage"], "complete")
        self.assertEqual(frame_manifest.config["n_expected"], self.N_FRAMES)
        self.assertEqual(frame_manifest.config["n_successful"], self.N_FRAMES)

        files = sorted(path.relative_to(self.args.output_dir).as_posix()
                       for path in self.args.output_dir.rglob("*")
                       if path.is_file())
        self.assertEqual(files, ["manifest.json", "predictions.jsonl"])
        predictions = (
            self.args.output_dir / "predictions.jsonl").read_text()
        records = [json.loads(line) for line in predictions.splitlines()]
        self.assertEqual([record["key"] for record in records], self.stems)
        self.assertTrue(all(set(record) == {"key", "prediction"}
                            for record in records))

        work_root = self.args.output_dir.with_name(
            self.args.output_dir.name + ex.WORK_SUFFIX)
        work_dirs = [path for path in work_root.iterdir()
                     if path.is_dir() and ex._DIGEST_RE.fullmatch(path.name)]
        self.assertEqual(len(work_dirs), 1)
        work_dir = work_dirs[0]
        artifact.open_artifact(
            work_dir / ex.REQUEST_ARTIFACT_DIR,
            expected_kind=llm_lifecycle.REQUEST_ARTIFACT_KIND)
        artifact.open_artifact(
            work_dir / ex.RESULT_ARTIFACT_DIR,
            expected_kind=llm_lifecycle.RESULT_ARTIFACT_KIND)
        attempts = llm_lifecycle.load_attempts(
            work_dir / ex.ATTEMPTS_DIR_NAME)
        self.assertEqual(len(attempts), self.N_FRAMES)

    def test_completed_outputs_are_validated_and_reused(self):
        first = self.run_successfully()
        with mock.patch.object(
                ex.panorama_to_pinhole, "process_panoramas",
                side_effect=AssertionError("must not render")), mock.patch.object(
                    ex.vbm, "run_requests",
                    side_effect=AssertionError("must not execute")):
            second = ex.run(self.args)
        self.assertEqual(first, second)

    def test_adoption_marker_never_weakens_normal_frame_validation(self):
        self.run_successfully()
        manifest_path = self.args.output_dir / artifact.MANIFEST_NAME
        document = json.loads(manifest_path.read_text())
        document["config"]["legacy_adoption_schema"] = (
            ex.LEGACY_ADOPTION_CONFIG_SCHEMA)
        artifact.atomic_write_json(manifest_path, document)

        with mock.patch.object(
                ex.panorama_to_pinhole, "process_panoramas",
                side_effect=AssertionError("pinhole must be reused")), \
                mock.patch.object(
                    ex.vbm, "run_requests",
                    side_effect=AssertionError("provider must not run")):
            with self.assertRaisesRegex(
                    ValueError, "invalid adopted-frame config shape"):
                ex.run(self.args)

    def test_valid_pinhole_is_reused_after_crash_before_frame_publish(self):
        context = ex.load_context(self.args)
        with mock.patch.object(
                ex.panorama_to_pinhole, "process_panoramas",
                side_effect=self.fake_render):
            pinhole_ref = ex.ensure_pinhole_artifact(self.args, context)
        self.assertTrue(self.args.pinhole_output_dir.exists())
        self.assertFalse(self.args.output_dir.exists())

        with mock.patch.object(
                ex.panorama_to_pinhole, "process_panoramas",
                side_effect=AssertionError("published pinhole must be reused")), \
                mock.patch.object(
                    ex.vbm, "run_requests",
                    side_effect=self.fake_success_transport):
            reused, frame_ref = ex.run(self.args)
        self.assertEqual(reused, pinhole_ref)
        self.assertEqual(frame_ref.kind, paths_lib.FRAME_LANDMARKS)

    def test_collection_published_pinhole_is_reused_without_rendering(self):
        collection_args = argparse.Namespace(
            output_base=self.dataset_base.parent,
            pinhole_base=self.tmp / "artifacts" / paths_lib.PINHOLE_IMAGES,
            pinhole_version="pinhole-v3",
            pinhole_res=RESOLUTION,
            convert_workers=1,
            dry_run=False,
        )

        def collection_render(target, argv, description, args):
            del target, description, args
            REAL_PROCESS_PANORAMAS(
                Path(argv[0]), Path(argv[1]),
                ex.FACE_FOV_RAD, ex.FACE_FOV_RAD,
                RESOLUTION, RESOLUTION, num_workers=1)
            return True

        with mock.patch.object(
                collection_runner, "run_bazel",
                side_effect=collection_render), mock.patch.object(
                    collection_runner.publication.indexes, "refresh"):
            self.assertTrue(collection_runner.stage_pinhole(
                "testset", {"pano": True}, collection_args))

        self.args.pinhole_output_dir = (
            collection_args.pinhole_base / "testset" / "pinhole-v3")
        context = ex.load_context(self.args)
        with mock.patch.object(
                ex.panorama_to_pinhole, "process_panoramas",
                side_effect=AssertionError(
                    "collection-published pinholes must be reused")):
            pinhole_ref = ex.ensure_pinhole_artifact(self.args, context)

        manifest = artifact.load_manifest(self.args.pinhole_output_dir)
        self.assertEqual(dict(manifest.config), ex._pinhole_config(context))
        self.assertEqual(
            manifest.declared_outputs, ex._pinhole_outputs(context))
        self.assertEqual(pinhole_ref.kind, paths_lib.PINHOLE_IMAGES)

    def test_corrupt_jpeg_is_rejected_before_publication(self):
        context = ex.load_context(self.args)

        def corrupt_render(input_dir, output_dir, fov_x, fov_y, res_x,
                           res_y, num_workers):
            self.fake_render(
                input_dir, output_dir, fov_x, fov_y, res_x, res_y,
                num_workers)
            target = (
                Path(output_dir) / self.stems[1] / "yaw_090.jpg")
            target.write_bytes(b"not a jpeg")

        with mock.patch.object(
                ex.panorama_to_pinhole, "process_panoramas",
                side_effect=corrupt_render):
            with self.assertRaisesRegex(ValueError, "not a decodable JPEG"):
                ex.ensure_pinhole_artifact(self.args, context)
        self.assertFalse(self.args.pinhole_output_dir.exists())

    def test_wrong_dimensions_on_last_face_are_rejected(self):
        context = ex.load_context(self.args)

        def wrong_size_render(input_dir, output_dir, fov_x, fov_y, res_x,
                              res_y, num_workers):
            self.fake_render(
                input_dir, output_dir, fov_x, fov_y, res_x, res_y,
                num_workers)
            target = (
                Path(output_dir) / self.stems[-1] / "yaw_270.jpg")
            Image.new("RGB", (res_x, res_y - 1)).save(target)

        with mock.patch.object(
                ex.panorama_to_pinhole, "process_panoramas",
                side_effect=wrong_size_render):
            with self.assertRaisesRegex(ValueError, "expected 24x24"):
                ex.ensure_pinhole_artifact(self.args, context)
        self.assertFalse(self.args.pinhole_output_dir.exists())

    def test_reuse_reproduces_sampled_faces_from_current_source_bytes(self):
        context = ex.load_context(self.args)
        with mock.patch.object(
                ex.panorama_to_pinhole, "process_panoramas",
                side_effect=self.fake_render):
            ex.ensure_pinhole_artifact(self.args, context)

        source = self.dataset_base / "panorama" / f"{self.stems[0]}.jpg"
        with Image.open(source) as original:
            size = original.size
        Image.new("RGB", size, (255, 0, 255)).save(source)
        with self.assertRaisesRegex(
                ValueError, "does not reproduce its source panorama"):
            ex.ensure_pinhole_artifact(self.args, context)


class CompleteCoverageResumeTest(ExtractionFixture):
    def test_pre_output_failure_reserves_a_fresh_raw_path(self):
        raw_paths = []

        def fail_before_output(execution, input_path, output_path, *, tag):
            del execution, input_path, tag
            raw_paths.append(Path(output_path))
            raise RuntimeError("provider stopped before local output")

        with mock.patch.object(
                ex.panorama_to_pinhole, "process_panoramas",
                side_effect=self.fake_render), mock.patch.object(
                    ex.vbm, "run_requests", side_effect=fail_before_output):
            with self.assertRaisesRegex(RuntimeError, "before local output"):
                ex.run(self.args)

        def repair(execution, input_path, output_path, *, tag):
            del execution, tag
            raw_paths.append(Path(output_path))
            records = [json.loads(line) for line in
                       Path(input_path).read_text().splitlines()]
            with Path(output_path).open("x") as stream:
                for record in records:
                    stream.write(json.dumps({
                        "key": record["key"],
                        "response": response(provider_prediction()),
                    }) + "\n")

        with mock.patch.object(
                ex.panorama_to_pinhole, "process_panoramas",
                side_effect=AssertionError("pinhole must be reused")), \
                mock.patch.object(ex.vbm, "run_requests", side_effect=repair):
            ex.run(self.args)

        self.assertEqual([path.name for path in raw_paths],
                         ["raw_0000.jsonl", "raw_0001.jsonl"])

    def test_partial_attempts_do_not_publish_and_rerun_only_missing_keys(self):
        first_keys = []

        def partial(execution, input_path, output_path, *, tag):
            del execution, tag
            records = [json.loads(line) for line in
                       Path(input_path).read_text().splitlines()]
            first_keys.extend(record["key"] for record in records)
            with Path(output_path).open("w") as stream:
                stream.write(json.dumps({
                    "key": records[0]["key"],
                    "response": response(provider_prediction()),
                }) + "\n")
                stream.write(json.dumps({
                    "key": records[1]["key"],
                    "error": "transient provider failure",
                }) + "\n")

        with mock.patch.object(
                ex.panorama_to_pinhole, "process_panoramas",
                side_effect=self.fake_render), mock.patch.object(
                    ex.vbm, "run_requests", side_effect=partial):
            with self.assertRaises(llm_lifecycle.IncompleteCoverageError):
                ex.run(self.args)
        self.assertEqual(first_keys, self.stems)
        self.assertTrue(self.args.pinhole_output_dir.exists())
        self.assertFalse(self.args.output_dir.exists())

        retried_keys = []

        def repair(execution, input_path, output_path, *, tag):
            del execution, tag
            records = [json.loads(line) for line in
                       Path(input_path).read_text().splitlines()]
            retried_keys.extend(record["key"] for record in records)
            with Path(output_path).open("w") as stream:
                for record in records:
                    stream.write(json.dumps({
                        "key": record["key"],
                        "response": response(provider_prediction()),
                    }) + "\n")

        with mock.patch.object(
                ex.panorama_to_pinhole, "process_panoramas",
                side_effect=AssertionError("pinhole must be reused")), \
                mock.patch.object(ex.vbm, "run_requests", side_effect=repair):
            ex.run(self.args)
        self.assertEqual(retried_keys, self.stems[1:])
        self.assertTrue(self.args.output_dir.exists())

    def test_two_valid_responses_for_one_key_are_rejected(self):
        request_set = llm_lifecycle.RequestSet.create(
            stage="frame_landmark_extraction",
            model="model",
            system_prompt="prompt",
            response_schema={},
            media_settings={},
            input_digests={"input": "a" * 64},
            upstreams=(),
            units=(llm_lifecycle.RequestUnit(
                key="frame", request={"contents": []}, metadata={}),),
        )
        attempts = tuple(
            llm_lifecycle.Attempt(
                request_set_fingerprint=request_set.fingerprint,
                key="frame",
                attempt_id=f"attempt-{index}",
                response=response(provider_prediction()),
                error=None,
                metadata={},
            )
            for index in range(2))
        with self.assertRaisesRegex(
                llm_lifecycle.IncompleteCoverageError, "duplicate valid"):
            ex.pending_units(request_set, attempts)


class TransportNormalizationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_online_success_and_error_sidecar_become_exact_boundary_records(self):
        raw = self.tmp / "raw_0000.jsonl"
        raw.write_text(json.dumps({
            "key": "a", "request": {"ignored": True},
            "response": response(provider_prediction()), "error": None,
        }) + "\n")
        raw.with_suffix(".errors.jsonl").write_text(json.dumps({
            "key": "b", "request": {"ignored": True},
            "response": None, "error": "quota",
        }) + "\n")
        normalized = ex.normalize_transport_shard(raw)
        records = [json.loads(line) for line in
                   normalized.read_text().splitlines()]
        self.assertEqual(set(records[0]), {"key", "response"})
        self.assertEqual(records[1], {"key": "b", "error": "quota"})


if __name__ == "__main__":
    unittest.main()
