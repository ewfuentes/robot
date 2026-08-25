import argparse
import base64
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from experimental.overhead_matching.swag.farfield import (
    artifact,
    llm_lifecycle,
    paths as paths_lib,
)
from experimental.overhead_matching.swag.farfield.extraction import (
    legacy_extraction_adoption as adoption,
    prompts,
)


def _jpeg_bytes(color):
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(stream, format="JPEG")
    return stream.getvalue()


def _landmark(*boxes):
    return {
        "primary_tag": {"key": "man_made", "value": "lighthouse"},
        "additional_tags": [{"key": "name", "value": "Test Light"}],
        "confidence": "high",
        "bounding_boxes": list(boxes),
        "description": "white cylindrical lighthouse",
    }


def _box(*, xmin=100, ymin=200, xmax=300, ymax=500, yaw="90"):
    return {
        "yaw_angle": yaw,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
    }


def _prediction(*landmarks):
    return {"location_type": "harbor", "landmarks": list(landmarks)}


def _response(prediction, *, fenced=False):
    text = json.dumps(prediction)
    if fenced:
        text = f"```json\n{text}\n```"
    return {
        "candidates": [{
            "content": {"parts": [{"text": text}], "role": "model"},
        }],
        "usageMetadata": {"totalTokenCount": 10},
    }


def _vertex_batch_provider_echo(request):
    value = json.loads(artifact.canonical_json_bytes(request))
    parts = value["contents"][0]["parts"]
    for part in parts[:4]:
        part["text"] = None
    parts[4]["inline_data"] = None
    parts[4]["media_resolution"] = None
    return value


def _retry_provider_echo(request):
    value = json.loads(artifact.canonical_json_bytes(request))
    landmark = value["generationConfig"]["responseSchema"]["properties"][
        "landmarks"]["items"]
    landmark["property_ordering"] = [
        "primary_tag", "additional_tags", "confidence", "bounding_boxes",
        "description"]
    landmark["properties"]["primary_tag"]["property_ordering"] = [
        "key", "value"]
    landmark["properties"]["additional_tags"]["items"][
        "property_ordering"] = ["key", "value"]
    landmark["properties"]["bounding_boxes"]["items"][
        "property_ordering"] = [
            "yaw_angle", "ymin", "xmin", "ymax", "xmax"]
    return value


class LegacyAdoptionFixture(unittest.TestCase):
    KEYS = ("frame000", "frame001")

    def setUp(self):
        self._temporary = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._temporary)
        self.root = Path(self._temporary)
        self.build_identity = "a" * 64
        self.pinhole = self.root / "pinhole"
        self.requests = {}
        face_payloads = {}
        colors = ((120, 80, 30), (30, 80, 120))
        for key, color in zip(self.KEYS, colors):
            images = []
            for offset, face in enumerate(prompts.PINHOLE_FACES):
                data = _jpeg_bytes(tuple(min(255, item + offset)
                                         for item in color))
                face_payloads[f"{key}/{face}.jpg"] = data
                images.append(("image/jpeg", base64.b64encode(data).decode()))
            self.requests[key] = prompts.build_request(
                key,
                images,
                prompt_type="osm_tags_farfield",
                media_resolution="MEDIA_RESOLUTION_HIGH",
                thinking_level="HIGH",
            )["request"]
        with artifact.ArtifactDirectoryBuilder(
                self.pinhole,
                kind=paths_lib.PINHOLE_IMAGES,
                dataset="testset",
                version="pinhole-v4",
                generator="legacy-extraction-adoption-test",
                git_commit="test",
                arguments=(),
                declared_outputs=tuple(sorted(face_payloads))) as builder:
            for relative, data in face_payloads.items():
                output = builder.output_path(relative)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(data)
        assert builder.artifact_ref is not None
        self.pinhole_ref = builder.artifact_ref
        self.primary_requests = {
            key: json.loads(artifact.canonical_json_bytes(request))
            for key, request in self.requests.items()
        }
        self.request_set = self._build_request_set(self.pinhole_ref)

        self.primary_path = self.root / "requests_000.jsonl"
        self._write_jsonl(self.primary_path, [
            {"key": key, "request": self.primary_requests[key]}
            for key in self.KEYS
        ])
        self.retry_request_path = self.root / "retry_requests.jsonl"
        self._write_jsonl(self.retry_request_path, [{
            "key": self.KEYS[1],
            "request": self.requests[self.KEYS[1]],
        }])

        self.batch_path = self.root / "predictions.jsonl"
        self._write_jsonl(self.batch_path, [
            self._batch_record(
                self.KEYS[0],
                _response(_prediction(_landmark(_box())))),
            # This historical attempt is preserved but is not consumer-valid.
            self._batch_record(
                self.KEYS[1],
                _response(_prediction(), fenced=True)),
        ])
        self.retry_path = self.root / "retry_predictions.jsonl"
        self._write_jsonl(self.retry_path, [self._retry_record(
            self.KEYS[1],
            _response(_prediction(_landmark(
                _box(yaw="180"),
                _box(xmin=400, xmax=400, yaw="270"))))),
        ])
        self.error_sidecar = self.root / "retry_predictions.errors.jsonl"
        self.error_sidecar.write_bytes(b"")

    def _build_request_set(self, pinhole_ref, requests=None):
        requests = self.requests if requests is None else requests
        return llm_lifecycle.RequestSet.create(
            stage="frame_landmark_extraction",
            model="gemini-paid-model",
            system_prompt=prompts.SYSTEM_PROMPTS["osm_tags_farfield"],
            response_schema=prompts.response_schema(),
            media_settings={
                "prompt_type": "osm_tags_farfield",
                "pinhole_resolution": 8,
                "media_resolution": "MEDIA_RESOLUTION_HIGH",
                "thinking_level": "HIGH",
                "face_order": list(prompts.PINHOLE_FACES),
            },
            input_digests={
                "build_identity": self.build_identity,
                "dataset_source": "b" * 64,
                "orchestration_config": "d" * 64,
            },
            upstreams=(pinhole_ref,),
            units=tuple(llm_lifecycle.RequestUnit(
                key=key,
                request=requests[key],
                metadata={"panorama_stem": key},
            ) for key in self.KEYS),
        )

    @staticmethod
    def _write_jsonl(path, records):
        path.write_text("".join(json.dumps(record) + "\n"
                                for record in records))

    def _batch_record(self, key, response, *, request=None, status=""):
        return {
            "key": key,
            "processed_time": "2026-08-01T12:00:00Z",
            "request": _vertex_batch_provider_echo(self.requests[key])
            if request is None else request,
            "response": response,
            "status": status,
        }

    def _retry_record(self, key, response, *, request=None, error=None):
        return {
            "error": error,
            "key": key,
            "request": _retry_provider_echo(self.requests[key])
            if request is None else request,
            "response": response,
        }

    def _verify(self, *, request_set=None, request_sources=None,
                result_sources=None, sidecars=None):
        if request_sources is None:
            request_sources = (
                adoption.LegacyRequestSource(
                    "primary", self.primary_path,
                    adoption.REQUEST_ROLE_PRIMARY),
                adoption.LegacyRequestSource(
                    "retry-requests", self.retry_request_path,
                    adoption.REQUEST_ROLE_RETRY),
            )
        if result_sources is None:
            result_sources = (
                adoption.LegacyResultSource(
                    "batch", self.batch_path,
                    adoption.RESULT_FORMAT_VERTEX_BATCH),
                adoption.LegacyResultSource(
                    "retry", self.retry_path,
                    adoption.RESULT_FORMAT_ONLINE_RETRY),
            )
        if sidecars is None:
            sidecars = (adoption.EmptyErrorSidecar(
                "retry-errors", self.error_sidecar),)
        return adoption.verify_adoption(
            dataset="testset",
            request_set=self.request_set if request_set is None else request_set,
            pinhole_dir=self.pinhole,
            request_sources=request_sources,
            result_sources=result_sources,
            empty_error_sidecars=sidecars,
        )

    def _publish(self, *, plan=None, final_build_identity=None,
                 request_dir=None, result_dir=None, frame_dir=None):
        plan = self._verify() if plan is None else plan
        publication_root = self.root / "publication"
        request_dir = (publication_root / "requests"
                       if request_dir is None else Path(request_dir))
        result_dir = (publication_root / "results"
                      if result_dir is None else Path(result_dir))
        frame_dir = (publication_root / "frame-landmarks"
                     if frame_dir is None else Path(frame_dir))
        published = adoption.publish_verified_adoption(
            plan=plan,
            request_set=self.request_set,
            dataset="testset",
            final_build_identity=(
                self.build_identity if final_build_identity is None
                else final_build_identity),
            frame_version="frame-v7",
            request_output_dir=request_dir,
            result_output_dir=result_dir,
            frame_output_dir=frame_dir,
            arguments=("--publish",),
            git_commit="test")
        return published, (request_dir, result_dir, frame_dir)


class CompleteAdoptionTest(LegacyAdoptionFixture):
    def test_complete_history_plans_zero_call_typed_publication(self):
        before = {path.relative_to(self.root): path.read_bytes()
                  for path in self.root.rglob("*") if path.is_file()}

        plan = self._verify()

        after = {path.relative_to(self.root): path.read_bytes()
                 for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(after, before)
        self.assertEqual(plan.report["status"],
                         "ready_for_explicit_publication")
        self.assertEqual(plan.report["provider_calls"], 0)
        self.assertEqual(plan.report["request_set"]["n_expected"], 2)
        self.assertEqual(plan.report["attempt_summary"], {
            "n_total": 3,
            "n_valid": 2,
            "n_failed_or_invalid": 1,
            "raw_provenance": "complete_by_source_and_line_digest",
        })
        self.assertEqual(
            [result.key for result in plan.canonical_results],
            list(self.KEYS))
        self.assertEqual(len(plan.attempts), 3)
        self.assertTrue(all(
            attempt.metadata["legacy_source"]["source_sha256"]
            for attempt in plan.attempts))
        actions = [event["action"] for event in
                   plan.report["normalization_ledger"]]
        self.assertEqual(
            actions.count("remove_exact_vertex_batch_null"), 12)
        self.assertNotIn("remove_exact_primary_null", actions)
        self.assertEqual(
            actions.count("remove_exact_retry_property_ordering"), 4)
        self.assertEqual(
            actions.count("preserve_transport_metadata_digest"), 3)
        source = plan.attempts[0].metadata["legacy_source"]
        self.assertEqual(source["echo_binding"]["request_role"], "primary")
        self.assertNotEqual(source["raw_echoed_request_sha256"],
                            source["echo_binding"]["current_request_sha256"])
        self.assertEqual(
            source["transport_metadata"]["sha256"],
            artifact.sha256_json({
                "processed_time": "2026-08-01T12:00:00Z", "status": ""}))
        self.assertEqual(
            [event["action"] for event in
             plan.report["sanitation_ledger"]],
            ["drop_bbox"])
        prediction_records = [json.loads(line) for line in
                              plan.predictions_bytes.splitlines()]
        boxes = prediction_records[1]["prediction"]["landmarks"][0][
            "bounding_boxes"]
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["yaw_angle"], 180)
        self.assertEqual(
            plan.report["pinhole_images"]["content_digest"],
            self.pinhole_ref.content_digest)
        self.assertEqual(
            plan.report["pinhole_images"]["artifact_ref"],
            self.pinhole_ref.to_dict())
        self.assertTrue(plan.report["publication_plan"]
                        ["requires_explicit_write_authorization"])
        self.assertFalse(plan.report["publication_plan"]
                         ["normal_reader_compatibility_fallback"])

    def test_all_invalid_boxes_drop_landmark_with_explicit_ledger(self):
        self._write_jsonl(self.retry_path, [self._retry_record(
            self.KEYS[1],
            _response(_prediction(_landmark(
                _box(xmin=500, xmax=500))))),
        ])

        plan = self._verify()

        self.assertEqual(
            [event["action"] for event in
             plan.report["sanitation_ledger"]],
            ["drop_bbox", "drop_landmark_no_valid_boxes"])
        self.assertEqual(plan.canonical_results[1].result["landmarks"], [])

    def test_originally_empty_boxes_drop_landmark_with_distinct_reason(self):
        self._write_jsonl(self.retry_path, [self._retry_record(
            self.KEYS[1], _response(_prediction(_landmark())))])

        plan = self._verify()

        event = plan.report["sanitation_ledger"][0]
        self.assertEqual(event["action"], "drop_landmark_empty_boxes")
        self.assertEqual(event["reason"],
                         "model_authored_empty_bounding_boxes")
        self.assertEqual(plan.canonical_results[1].result["landmarks"], [])


class TransactionalPublicationTest(LegacyAdoptionFixture):
    def test_publishes_exact_typed_lineage_and_authenticated_proof(self):
        sources = (
            self.primary_path, self.retry_request_path, self.batch_path,
            self.retry_path, self.error_sidecar)
        source_bytes = {path: path.read_bytes() for path in sources}
        plan = self._verify()

        with mock.patch.object(
                adoption.extract_landmarks.vbm, "run_requests") as provider:
            published, (request_dir, result_dir, frame_dir) = self._publish(
                plan=plan)

        provider.assert_not_called()
        self.assertEqual(
            {path: path.read_bytes() for path in sources}, source_bytes)
        self.assertEqual(
            artifact.open_artifact(request_dir).to_dict(),
            published.request_artifact.to_dict())
        self.assertEqual(
            artifact.open_artifact(result_dir).to_dict(),
            published.canonical_results_artifact.to_dict())
        self.assertEqual(
            artifact.open_artifact(frame_dir).to_dict(),
            published.frame_landmarks_artifact.to_dict())
        self.assertEqual(
            published.verification_report_sha256, plan.report_sha256)

        request_manifest = artifact.load_manifest(request_dir)
        self.assertEqual(request_manifest.upstreams, (self.pinhole_ref,))
        self.assertEqual(
            request_manifest.declared_outputs,
            (llm_lifecycle.REQUEST_SET_NAME, llm_lifecycle.REQUESTS_NAME))
        self.assertEqual(
            request_manifest.config["build_identity"], self.build_identity)
        orchestration = {
            "schema": "farfield_pipeline_stage/v1",
            "stage": "extract",
            "config_digest": "d" * 64,
        }
        self.assertEqual(
            request_manifest.config["orchestration"], orchestration)
        self.assertEqual(
            request_manifest.config["legacy_adoption_schema"],
            adoption.extract_landmarks.LEGACY_ADOPTION_CONFIG_SCHEMA)
        proof = request_manifest.config["legacy_adoption"]
        self.assertEqual(proof["schema"], adoption.PUBLICATION_PROOF_SCHEMA)
        self.assertEqual(
            proof["verification_report_sha256"], plan.report_sha256)
        self.assertEqual(proof["verification_report"], plan.report)
        self.assertEqual(
            artifact.sha256_json(proof["verification_report"]),
            plan.report_sha256)
        self.assertEqual(
            llm_lifecycle.load_request_set(
                request_dir / llm_lifecycle.REQUEST_SET_NAME).to_dict(),
            self.request_set.to_dict())
        self.assertEqual(
            (request_dir / llm_lifecycle.REQUESTS_NAME).read_bytes(),
            llm_lifecycle.transport_requests_bytes(self.request_set))

        result_manifest = artifact.load_manifest(result_dir)
        self.assertEqual(
            result_manifest.upstreams, (published.request_artifact,))
        self.assertEqual(
            result_manifest.config["legacy_adoption_report_sha256"],
            plan.report_sha256)
        self.assertEqual(
            result_manifest.config["orchestration"], orchestration)
        self.assertEqual(
            result_manifest.config["legacy_adoption_schema"],
            adoption.extract_landmarks.LEGACY_ADOPTION_CONFIG_SCHEMA)
        self.assertEqual(
            (result_dir / llm_lifecycle.CANONICAL_RESULTS_NAME).read_bytes(),
            plan.canonical_results_bytes)
        self.assertEqual(
            llm_lifecycle.load_canonical_results(
                result_dir / llm_lifecycle.CANONICAL_RESULTS_NAME,
                self.request_set),
            plan.canonical_results)

        frame_manifest = artifact.load_manifest(frame_dir)
        self.assertEqual(
            frame_manifest.upstreams,
            (self.pinhole_ref, published.canonical_results_artifact))
        self.assertNotIn(published.request_artifact, frame_manifest.upstreams)
        self.assertEqual(
            frame_manifest.config["build_identity"], self.build_identity)
        self.assertEqual(
            frame_manifest.config["orchestration"], orchestration)
        self.assertEqual(
            frame_manifest.config["legacy_adoption_report_sha256"],
            plan.report_sha256)
        self.assertEqual(
            frame_manifest.config["legacy_adoption_schema"],
            adoption.extract_landmarks.LEGACY_ADOPTION_CONFIG_SCHEMA)
        self.assertEqual(
            frame_manifest.config["request_artifact"]["manifest_digest"],
            published.request_artifact.manifest_digest)
        self.assertEqual(
            frame_manifest.config["canonical_results_artifact"]
            ["manifest_digest"],
            published.canonical_results_artifact.manifest_digest)
        self.assertEqual(
            (frame_dir / adoption.extract_landmarks.PREDICTIONS_NAME)
            .read_bytes(),
            plan.predictions_bytes)
        for path in (request_dir, result_dir, frame_dir):
            self.assertFalse(path.with_name(
                path.name + artifact.INCOMPLETE_SUFFIX).exists())

    def test_published_adoption_is_accepted_by_exact_extraction_reuse(self):
        plan = self._verify()
        published, (_, _, frame_dir) = self._publish(plan=plan)
        selected = {
            "artifacts.frame_landmarks_version": "frame-v7",
            "artifacts.pinhole_images_version": "pinhole-v4",
            "extraction.model": "gemini-paid-model",
            "extraction.prompt_type": "osm_tags_farfield",
            "extraction.pinhole_resolution": 8,
            "extraction.media_resolution": "MEDIA_RESOLUTION_HIGH",
            "extraction.thinking_level": "HIGH",
        }
        context = adoption.extract_landmarks.ExtractionContext(
            document={"build_identity": self.build_identity},
            selected=selected,
            orchestration={
                "schema": "farfield_pipeline_stage/v1",
                "stage": "extract",
                "config_digest": "d" * 64,
            },
            metadata={},
            frames=tuple(
                adoption.extract_landmarks.dataset_lib.Frame(
                    frame_idx=index, pano_id=f"f{index:04d}",
                    pano_stem=key, lat=0.0, lon=0.0)
                for index, key in enumerate(self.KEYS)),
            dataset_base=self.root,
            panorama_dir=self.root,
            input_digests=dict(self.request_set.input_digests),
        )
        args = argparse.Namespace(dataset="testset", output_dir=frame_dir)
        with mock.patch.object(
                adoption.extract_landmarks, "load_context",
                return_value=context), mock.patch.object(
                    adoption.extract_landmarks, "ensure_pinhole_artifact",
                    return_value=self.pinhole_ref), mock.patch.object(
                    adoption.extract_landmarks.llm_cost,
                    "enforce_limit") as cost_gate, mock.patch.object(
                    adoption.extract_landmarks.vbm,
                    "run_requests") as provider:
            reused = adoption.extract_landmarks.run(args)

        cost_gate.assert_not_called()
        provider.assert_not_called()
        self.assertEqual(
            reused, (self.pinhole_ref, published.frame_landmarks_artifact))

    def test_retry_validates_and_reuses_each_completed_prefix(self):
        plan = self._verify()
        original_reopen = adoption._reopen_exact_artifact
        for prefix_length in (1, 2):
            with self.subTest(prefix_length=prefix_length):
                root = self.root / f"resume-{prefix_length}"
                paths = (root / "requests", root / "results", root / "frames")
                reopened = 0

                def fail_after_reopen(*args, **kwargs):
                    nonlocal reopened
                    ref = original_reopen(*args, **kwargs)
                    reopened += 1
                    if reopened == prefix_length:
                        raise adoption.AdoptionError(
                            "injected failure after completed publication")
                    return ref

                with mock.patch.object(
                        adoption, "_reopen_exact_artifact",
                        side_effect=fail_after_reopen), mock.patch.object(
                            adoption.extract_landmarks.vbm,
                            "run_requests") as failed_provider:
                    with self.assertRaisesRegex(
                            adoption.AdoptionError, "injected failure"):
                        self._publish(
                            plan=plan,
                            request_dir=paths[0], result_dir=paths[1],
                            frame_dir=paths[2])
                failed_provider.assert_not_called()

                self.assertEqual(
                    [path.exists() for path in paths],
                    [index < prefix_length for index in range(3)])
                prefix_refs = [
                    artifact.open_artifact(path).to_dict()
                    for path in paths[:prefix_length]
                ]
                with mock.patch.object(
                        adoption.extract_landmarks.vbm,
                        "run_requests") as provider:
                    published, _ = self._publish(
                        plan=plan,
                        request_dir=paths[0], result_dir=paths[1],
                        frame_dir=paths[2])
                provider.assert_not_called()
                self.assertEqual(
                    [artifact.open_artifact(path).to_dict()
                     for path in paths[:prefix_length]],
                    prefix_refs)
                self.assertTrue(all(path.exists() for path in paths))
                self.assertEqual(
                    artifact.load_manifest(paths[1]).upstreams,
                    (published.request_artifact,))
                self.assertEqual(
                    artifact.load_manifest(paths[2]).upstreams,
                    (self.pinhole_ref,
                     published.canonical_results_artifact))

                with mock.patch.object(
                        adoption.publication, "published_artifact",
                        side_effect=AssertionError(
                            "an exact complete prefix must only be reused")):
                    repeated, _ = self._publish(
                        plan=plan,
                        request_dir=paths[0], result_dir=paths[1],
                        frame_dir=paths[2])
                self.assertEqual(repeated, published)

    def test_retry_rejects_mismatched_completed_prefix_before_new_writes(self):
        plan = self._verify()
        root = self.root / "mismatched-prefix"
        request_dir, result_dir, frame_dir = (
            root / "requests", root / "results", root / "frames")
        original_reopen = adoption._reopen_exact_artifact

        def fail_after_request(*args, **kwargs):
            ref = original_reopen(*args, **kwargs)
            raise adoption.AdoptionError(
                "injected failure after completed request")

        with mock.patch.object(
                adoption, "_reopen_exact_artifact",
                side_effect=fail_after_request):
            with self.assertRaisesRegex(adoption.AdoptionError, "injected"):
                self._publish(
                    plan=plan,
                    request_dir=request_dir, result_dir=result_dir,
                    frame_dir=frame_dir)

        manifest_path = request_dir / artifact.MANIFEST_NAME
        document = json.loads(manifest_path.read_text())
        document["config"]["n_expected"] += 1
        artifact.atomic_write_json(manifest_path, document)
        with self.assertRaisesRegex(
                adoption.AdoptionError, "different immutable config"):
            self._publish(
                plan=plan,
                request_dir=request_dir, result_dir=result_dir,
                frame_dir=frame_dir)
        self.assertFalse(result_dir.exists())
        self.assertFalse(frame_dir.exists())

    def test_preflight_refuses_completed_or_incomplete_target_before_writes(self):
        plan = self._verify()
        for label in ("completed", "incomplete"):
            with self.subTest(label=label):
                root = self.root / f"occupied-{label}"
                request_dir = root / "requests"
                result_dir = root / "results"
                frame_dir = root / "frames"
                occupied = (
                    result_dir if label == "completed"
                    else frame_dir.with_name(
                        frame_dir.name + artifact.INCOMPLETE_SUFFIX))
                occupied.mkdir(parents=True)
                sentinel = occupied / "sentinel"
                sentinel.write_bytes(b"keep")

                with self.assertRaisesRegex(
                        artifact.ArtifactExistsError,
                        "artifact already exists"):
                    self._publish(
                        plan=plan,
                        request_dir=request_dir,
                        result_dir=result_dir,
                        frame_dir=frame_dir)

                self.assertEqual(sentinel.read_bytes(), b"keep")
                self.assertFalse(request_dir.exists())
                if label == "incomplete":
                    self.assertFalse(result_dir.exists())
                    self.assertFalse(frame_dir.exists())
                else:
                    self.assertFalse(frame_dir.exists())

    def test_final_build_identity_must_match_verified_request_set(self):
        plan = self._verify()
        with self.assertRaisesRegex(
                adoption.AdoptionError,
                "final build identity does not match"):
            self._publish(
                plan=plan, final_build_identity="c" * 64)
        self.assertFalse((self.root / "publication").exists())

    def test_mutated_verification_report_is_not_publishable(self):
        plan = self._verify()
        plan.report["sanitation_ledger"].append({"action": "tampered"})

        with self.assertRaisesRegex(
                adoption.AdoptionError,
                "report changed after verification"):
            self._publish(plan=plan)

        self.assertFalse((self.root / "publication").exists())


class FailClosedCoverageTest(LegacyAdoptionFixture):
    def test_missing_valid_response_is_rejected(self):
        self._write_jsonl(self.retry_path, [self._retry_record(
            self.KEYS[1], _response(_prediction(), fenced=True))])

        with self.assertRaisesRegex(
                adoption.AdoptionError, "no valid response"):
            self._verify()

    def test_duplicate_valid_response_is_rejected_even_when_identical(self):
        records = [json.loads(line) for line in
                   self.batch_path.read_text().splitlines()]
        records.append(self._batch_record(
            self.KEYS[0], _response(_prediction(_landmark(_box())))))
        self._write_jsonl(self.batch_path, records)

        with self.assertRaisesRegex(
                adoption.AdoptionError, "duplicate valid responses"):
            self._verify()

    def test_unknown_result_key_is_rejected(self):
        records = [json.loads(line) for line in
                   self.batch_path.read_text().splitlines()]
        unknown = records[0].copy()
        unknown["key"] = "not-in-request-set"
        records.append(unknown)
        self._write_jsonl(self.batch_path, records)

        with self.assertRaisesRegex(adoption.AdoptionError,
                                    "unknown result key"):
            self._verify()

    def test_conflicting_provider_echo_is_rejected(self):
        records = [json.loads(line) for line in
                   self.batch_path.read_text().splitlines()]
        records[0]["request"] = json.loads(
            artifact.canonical_json_bytes(records[0]["request"]))
        records[0]["request"]["generationConfig"][
            "thinkingConfig"]["thinkingLevel"] = "LOW"
        self._write_jsonl(self.batch_path, records)

        with self.assertRaisesRegex(adoption.AdoptionError,
                                    "provider echo.*does not bind"):
            self._verify()

    def test_vertex_batch_echo_requires_all_observed_null_decorations(self):
        records = [json.loads(line) for line in
                   self.batch_path.read_text().splitlines()]
        records[1]["request"] = self.requests[self.KEYS[1]]
        self._write_jsonl(self.batch_path, records)

        with self.assertRaisesRegex(
                adoption.AdoptionError,
                "vertex batch provider echo lacks the exact observed null decoration"):
            self._verify()

    def test_vertex_batch_echo_rejects_partial_null_decorations(self):
        records = [json.loads(line) for line in
                   self.batch_path.read_text().splitlines()]
        del records[0]["request"]["contents"][0]["parts"][0]["text"]
        self._write_jsonl(self.batch_path, records)

        with self.assertRaisesRegex(
                adoption.AdoptionError,
                "vertex batch provider echo lacks the exact observed null decoration"):
            self._verify()

    def test_vertex_batch_echo_rejects_wrong_null_decoration(self):
        records = [json.loads(line) for line in
                   self.batch_path.read_text().splitlines()]
        records[0]["request"]["contents"][0]["parts"][0]["text"] = ""
        self._write_jsonl(self.batch_path, records)

        with self.assertRaisesRegex(
                adoption.AdoptionError,
                "vertex batch provider echo lacks the exact observed null decoration"):
            self._verify()

    def test_retry_echo_requires_preserved_retry_snapshot(self):
        primary_only = (adoption.LegacyRequestSource(
            "primary", self.primary_path, adoption.REQUEST_ROLE_PRIMARY),)

        with self.assertRaisesRegex(
                adoption.AdoptionError,
                "provider echo.*preserved retry raw request"):
            self._verify(request_sources=primary_only)

    def test_wrong_retry_property_ordering_is_rejected(self):
        records = [json.loads(line) for line in
                   self.retry_path.read_text().splitlines()]
        landmark = records[0]["request"]["generationConfig"][
            "responseSchema"]["properties"]["landmarks"]["items"]
        landmark["property_ordering"] = ["description"]
        self._write_jsonl(self.retry_path, records)

        with self.assertRaisesRegex(
                adoption.AdoptionError,
                "exact observed property_ordering decoration"):
            self._verify()

    def test_preserved_primary_request_must_match_current_exactly(self):
        records = [json.loads(line) for line in
                   self.primary_path.read_text().splitlines()]
        records[0]["request"]["contents"][0]["parts"][0]["text"] = None
        self._write_jsonl(self.primary_path, records)

        with self.assertRaisesRegex(
                adoption.AdoptionError,
                "conflicts with the exact current request set"):
            self._verify()

    def test_preserved_retry_request_must_match_current_exactly(self):
        records = [json.loads(line) for line in
                   self.retry_request_path.read_text().splitlines()]
        records[0]["request"] = _retry_provider_echo(records[0]["request"])
        self._write_jsonl(self.retry_request_path, records)

        with self.assertRaisesRegex(
                adoption.AdoptionError,
                "conflicts with the exact current request set"):
            self._verify()

    def test_retry_echo_requires_all_observed_ordering_decorations(self):
        records = [json.loads(line) for line in
                   self.retry_path.read_text().splitlines()]
        records[0]["request"] = self.requests[self.KEYS[0]]
        self._write_jsonl(self.retry_path, records)

        with self.assertRaisesRegex(
                adoption.AdoptionError,
                "online retry provider echo lacks the exact observed property_ordering decoration"):
            self._verify()

    def test_retry_echo_requires_every_observed_ordering_decoration(self):
        records = [json.loads(line) for line in
                   self.retry_path.read_text().splitlines()]
        landmark = records[0]["request"]["generationConfig"][
            "responseSchema"]["properties"]["landmarks"]["items"]
        del landmark["property_ordering"]
        self._write_jsonl(self.retry_path, records)

        with self.assertRaisesRegex(
                adoption.AdoptionError,
                "exact observed property_ordering decoration"):
            self._verify()

    def test_primary_requests_must_cover_exact_order_once(self):
        self._write_jsonl(self.primary_path, [{
            "key": self.KEYS[0],
            "request": self.primary_requests[self.KEYS[0]],
        }])

        with self.assertRaisesRegex(adoption.AdoptionError,
                                    "exactly cover request-set order"):
            self._verify()

    def test_request_media_must_equal_retained_face_bytes(self):
        requests = json.loads(artifact.canonical_json_bytes(self.requests))
        requests[self.KEYS[0]]["contents"][0]["parts"][0][
            "inline_data"]["data"] = base64.b64encode(
                _jpeg_bytes((1, 2, 3))).decode()
        request_set = self._build_request_set(self.pinhole_ref, requests)

        with self.assertRaisesRegex(adoption.AdoptionError,
                                    "request media differs"):
            self._verify(request_set=request_set)

    def test_pinhole_manifest_digest_must_match_upstream(self):
        manifest_path = self.pinhole / artifact.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest["generator"] = "tampered-but-schema-valid-generator"
        manifest_path.write_text(json.dumps(manifest) + "\n")

        with self.assertRaisesRegex(
                adoption.AdoptionError,
                "does not exactly match the request-set upstream identity"):
            self._verify()

    def test_pinhole_path_must_match_upstream(self):
        wrong_path_ref = artifact.ArtifactRef(
            path=str(self.root / "different-pinhole-path"),
            kind=self.pinhole_ref.kind,
            dataset=self.pinhole_ref.dataset,
            version=self.pinhole_ref.version,
            manifest_digest=self.pinhole_ref.manifest_digest,
            content_digest=self.pinhole_ref.content_digest,
        )
        request_set = self._build_request_set(wrong_path_ref)

        with self.assertRaisesRegex(
                adoption.AdoptionError,
                "does not exactly match the request-set upstream identity"):
            self._verify(request_set=request_set)

    def test_nonempty_error_sidecar_is_rejected_as_unknown_shape(self):
        self.error_sidecar.write_text(json.dumps({"error": "quota"}) + "\n")

        with self.assertRaisesRegex(adoption.AdoptionError,
                                    "unobserved record shape"):
            self._verify()

    def test_unknown_result_field_is_not_a_compatibility_path(self):
        records = [json.loads(line) for line in
                   self.batch_path.read_text().splitlines()]
        records[0]["new_provider_field"] = True
        self._write_jsonl(self.batch_path, records)

        with self.assertRaisesRegex(adoption.AdoptionError,
                                    "unknown.*new_provider_field"):
            self._verify()

    def test_retry_request_snapshot_may_repeat_only_the_exact_request(self):
        record = {
            "key": self.KEYS[1],
            "request": json.loads(artifact.canonical_json_bytes(
                self.requests[self.KEYS[1]])),
        }
        record["request"]["systemInstruction"]["parts"][0]["text"] += "x"
        self._write_jsonl(self.retry_request_path, [record])

        with self.assertRaisesRegex(adoption.AdoptionError,
                                    "conflicts with the exact current request set"):
            self._verify()


class SpecTest(LegacyAdoptionFixture):
    def test_strict_spec_loads_relative_paths_and_cli_requires_write_gate(self):
        request_set_path = self.root / "request_set.json"
        request_set_path.write_text(json.dumps(self.request_set.to_dict()))
        spec_path = self.root / "adoption_spec.json"
        spec = {
            "schema": adoption.SPEC_SCHEMA,
            "dataset": "testset",
            "request_set": request_set_path.name,
            "pinhole_dir": self.pinhole.name,
            "request_sources": [
                {"id": "primary", "path": self.primary_path.name,
                 "role": adoption.REQUEST_ROLE_PRIMARY},
                {"id": "retry-requests",
                 "path": self.retry_request_path.name,
                 "role": adoption.REQUEST_ROLE_RETRY},
            ],
            "result_sources": [
                {"id": "batch", "path": self.batch_path.name,
                 "format": adoption.RESULT_FORMAT_VERTEX_BATCH},
                {"id": "retry", "path": self.retry_path.name,
                 "format": adoption.RESULT_FORMAT_ONLINE_RETRY},
            ],
            "empty_error_sidecars": [{
                "id": "retry-errors", "path": self.error_sidecar.name,
            }],
        }
        spec_path.write_text(json.dumps(spec))

        loaded = adoption.load_spec(spec_path)
        plan = adoption.verify_spec(loaded)

        self.assertEqual(plan.report["spec_sha256"],
                         artifact.sha256_file(spec_path))
        self.assertEqual(plan.report["provider_calls"], 0)

        cli_root = self.root / "cli-publication"
        unauthorized = [
            "--spec", str(spec_path),
            "--frame-version", "frame-v7",
        ]
        with mock.patch("builtins.print"):
            self.assertEqual(adoption.main(unauthorized), 1)
        self.assertFalse(cli_root.exists())

        authorized = [
            "--spec", str(spec_path),
            "--publish",
            "--final-build-identity", self.build_identity,
            "--frame-version", "frame-v7",
            "--request-output-dir", str(cli_root / "requests"),
            "--result-output-dir", str(cli_root / "results"),
            "--frame-output-dir", str(cli_root / "frames"),
        ]
        with mock.patch.object(
                adoption.extract_landmarks.vbm, "run_requests") as provider, \
             mock.patch("builtins.print"):
            self.assertEqual(adoption.main(authorized), 0)
        provider.assert_not_called()
        self.assertEqual(
            artifact.load_manifest(cli_root / "results").upstreams,
            (artifact.open_artifact(cli_root / "requests"),))
        self.assertEqual(
            artifact.load_manifest(cli_root / "frames").upstreams,
            (self.pinhole_ref, artifact.open_artifact(cli_root / "results")))


if __name__ == "__main__":
    unittest.main()
