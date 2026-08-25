"""Tests for strict request identity, attempt retention, and full coverage."""

import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import llm_lifecycle as llm


_A = "a" * 64
_B = "b" * 64


def upstream(path="/old/location"):
    return artifact.ArtifactRef(
        path=path,
        kind="tracks",
        dataset="harbor",
        version="v3",
        manifest_digest=_A,
        content_digest=_B,
    )


def request_set(*, model="test-model", keys=("u1", "u2"),
                upstream_ref=None):
    return llm.RequestSet.create(
        stage="matching",
        model=model,
        system_prompt="match without spatial information",
        response_schema={"type": "object"},
        media_settings={"thinking_level": "HIGH", "mime_type": "image/jpeg"},
        input_digests={"catalog": "c" * 64},
        upstreams=(upstream_ref or upstream(),),
        units=(llm.RequestUnit(key, {"contents": [{"text": key}]},
                               {"ordinal": index})
               for index, key in enumerate(keys)),
    )


def success(snapshot, key, attempt_id, value):
    return llm.Attempt(
        request_set_fingerprint=snapshot.fingerprint,
        key=key,
        attempt_id=attempt_id,
        response={"value": value},
        error=None,
        metadata={"transport": "test"},
    )


def failure(snapshot, key, attempt_id):
    return llm.Attempt(
        request_set_fingerprint=snapshot.fingerprint,
        key=key,
        attempt_id=attempt_id,
        response=None,
        error={"code": "FAILED"},
        metadata={},
    )


def validate(_key, response):
    if set(response) != {"value"} or type(response["value"]) is not int:
        raise ValueError("response must contain one integer value")
    return response


class RequestSetTest(unittest.TestCase):

    def test_round_trip_and_fingerprint(self):
        snapshot = request_set()
        self.assertEqual(llm.RequestSet.from_dict(snapshot.to_dict()), snapshot)
        self.assertEqual(len(snapshot.fingerprint), 64)

    def test_informational_upstream_path_does_not_change_identity(self):
        first = request_set(upstream_ref=upstream("/first"))
        moved = request_set(upstream_ref=upstream("/moved"))
        self.assertEqual(first.fingerprint, moved.fingerprint)

    def test_model_and_order_are_identity(self):
        baseline = request_set()
        self.assertNotEqual(
            baseline.fingerprint, request_set(model="other-model").fingerprint)
        self.assertNotEqual(
            baseline.fingerprint,
            request_set(keys=("u2", "u1")).fingerprint)

    def test_snapshot_detaches_and_freezes_nested_inputs(self):
        request = {"contents": [{"text": "before"}]}
        unit = llm.RequestUnit("u", request, {"batch": ["u"]})
        request["contents"][0]["text"] = "after"
        self.assertEqual(unit.to_dict()["request"]["contents"][0]["text"],
                         "before")
        with self.assertRaises(TypeError):
            unit.request["new"] = 1

    def test_duplicate_unit_key_is_rejected(self):
        with self.assertRaisesRegex(llm.LlmLifecycleError, "unique"):
            request_set(keys=("same", "same"))

    def test_unbound_request_set_is_rejected(self):
        with self.assertRaisesRegex(llm.LlmLifecycleError, "bind at least"):
            llm.RequestSet.create(
                stage="unbound",
                model="test-model",
                system_prompt="prompt",
                response_schema={"type": "object"},
                media_settings={},
                input_digests={},
                upstreams=(),
                units=(llm.RequestUnit("u", {"contents": []}, {}),),
            )

    def test_tampered_snapshot_fingerprint_is_rejected(self):
        value = request_set().to_dict()
        value["model"] = "tampered"
        with self.assertRaisesRegex(llm.LlmLifecycleError, "fingerprint"):
            llm.RequestSet.from_dict(value)

    def test_request_artifact_is_transactional_and_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "requests-v1"
            snapshot = request_set()
            ref = llm.publish_request_set(
                destination,
                request_set=snapshot,
                dataset="harbor",
                version="v1",
                generator="unit-test",
                git_commit="deadbeef",
            )
            self.assertEqual(ref, artifact.open_artifact(
                destination,
                expected_kind=llm.REQUEST_ARTIFACT_KIND,
                expected_dataset="harbor",
                expected_version="v1"))
            loaded = llm.load_request_set(destination / llm.REQUEST_SET_NAME)
            self.assertEqual(loaded, snapshot)
            records = [json.loads(line) for line in
                       (destination / llm.REQUESTS_NAME).read_text().splitlines()]
            self.assertEqual([record["key"] for record in records], ["u1", "u2"])
            subset = [json.loads(line) for line in
                      llm.transport_requests_bytes(
                          snapshot, ("u2",)).decode().splitlines()]
            self.assertEqual([record["key"] for record in subset], ["u2"])
            self.assertFalse(destination.with_name(
                destination.name + artifact.INCOMPLETE_SUFFIX).exists())


class AttemptShardTest(unittest.TestCase):

    def test_immutable_shard_round_trips_and_refuses_duplicate_attempt_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempts_dir = Path(temporary) / llm.ATTEMPTS_DIR_NAME
            snapshot = request_set()
            item = success(snapshot, "u1", "try-1", 7)
            shard = llm.publish_attempt(attempts_dir, item)
            self.assertEqual(llm.load_attempts(attempts_dir), (item,))
            self.assertEqual(tuple(attempts_dir.iterdir()), (shard,))
            original = shard.read_bytes()
            with self.assertRaisesRegex(llm.LlmLifecycleError, "duplicate"):
                llm.publish_attempt(
                    attempts_dir, success(snapshot, "u1", "try-1", 8))
            self.assertEqual(shard.read_bytes(), original)

    def test_malformed_or_unexpected_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempts_dir = Path(temporary) / llm.ATTEMPTS_DIR_NAME
            attempts_dir.mkdir()
            malformed = attempts_dir / f"attempt-{'0' * 64}.json"
            malformed.write_text("{}\n\n")
            with self.assertRaises(llm.LlmLifecycleError):
                llm.load_attempts(attempts_dir)
            malformed.unlink()
            (attempts_dir / "attempts.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(llm.LlmLifecycleError, "unexpected"):
                llm.load_attempts(attempts_dir)

    def test_legacy_attempt_log_is_not_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            legacy = Path(temporary) / "attempts.jsonl"
            legacy.write_text("{}\n")
            with self.assertRaisesRegex(llm.LlmLifecycleError, "directory"):
                llm.load_attempts(legacy)

    def test_attempt_requires_exactly_response_or_error(self):
        snapshot = request_set()
        with self.assertRaisesRegex(llm.LlmLifecycleError, "exactly one"):
            llm.Attempt(snapshot.fingerprint, "u1", "x", {}, "bad", {})

    def test_transport_import_is_strict_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transport = root / "transport.jsonl"
            attempts = root / llm.ATTEMPTS_DIR_NAME
            snapshot = request_set()
            transport.write_text(
                json.dumps({"key": "u1", "response": {"value": 1}}) + "\n" +
                json.dumps({"key": "u2", "error": "failed"}) + "\n")
            self.assertEqual(
                llm.import_transport_results(transport, attempts, snapshot), 2)
            self.assertEqual(
                llm.import_transport_results(transport, attempts, snapshot), 0)
            self.assertEqual(len(llm.load_attempts(attempts)), 2)
            transport.write_text(json.dumps(
                {"key": "u1", "response": {}, "extra": True}) + "\n")
            with self.assertRaisesRegex(llm.LlmLifecycleError, "exact keys"):
                llm.import_transport_results(transport, attempts, snapshot)

    def test_identical_records_from_two_retry_rounds_are_distinct_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "transport_submit_0001.jsonl"
            second = root / "transport_submit_0002.jsonl"
            attempts = root / llm.ATTEMPTS_DIR_NAME
            snapshot = request_set(keys=("u1",))
            line = json.dumps(
                {"key": "u1", "response": {"value": 1}}) + "\n"
            first.write_text(line)
            second.write_text(line)
            llm.import_transport_results(first, attempts, snapshot)
            llm.import_transport_results(second, attempts, snapshot)
            loaded = llm.load_attempts(attempts)
            self.assertEqual(len(loaded), 2)
            self.assertNotEqual(loaded[0].attempt_id, loaded[1].attempt_id)


class CompleteCoverageTest(unittest.TestCase):

    def test_failed_attempt_can_be_retained_then_retried(self):
        snapshot = request_set()
        results = llm.compile_canonical_results(
            snapshot,
            (failure(snapshot, "u1", "u1-failed"),
             success(snapshot, "u1", "u1-good", 1),
             success(snapshot, "u2", "u2-good", 2)),
            validate,
        )
        self.assertEqual([item.key for item in results], ["u1", "u2"])
        self.assertEqual([item.result["value"] for item in results], [1, 2])

    def test_missing_or_failed_unit_is_rejected_without_threshold(self):
        snapshot = request_set()
        with self.assertRaisesRegex(llm.IncompleteCoverageError, "u2"):
            llm.compile_canonical_results(
                snapshot,
                (success(snapshot, "u1", "u1-good", 1),
                 failure(snapshot, "u2", "u2-failed")),
                validate,
            )

    def test_two_valid_responses_for_one_unit_are_rejected(self):
        snapshot = request_set()
        with self.assertRaisesRegex(llm.IncompleteCoverageError, "duplicate"):
            llm.compile_canonical_results(
                snapshot,
                (success(snapshot, "u1", "u1-a", 1),
                 success(snapshot, "u1", "u1-b", 2),
                 success(snapshot, "u2", "u2", 3)),
                validate,
            )

    def test_unknown_key_or_wrong_fingerprint_is_rejected(self):
        snapshot = request_set()
        with self.assertRaisesRegex(llm.LlmLifecycleError, "unknown key"):
            llm.compile_canonical_results(
                snapshot,
                (success(snapshot, "unknown", "x", 1),),
                validate,
            )
        wrong = success(request_set(model="other"), "u1", "wrong", 1)
        with self.assertRaisesRegex(llm.LlmLifecycleError, "different request"):
            llm.compile_canonical_results(snapshot, (wrong,), validate)

    def test_malformed_success_can_be_retried_but_cannot_stand_alone(self):
        snapshot = request_set(keys=("u1",))
        malformed = success(snapshot, "u1", "bad", "not-an-int")
        with self.assertRaisesRegex(llm.IncompleteCoverageError, "no valid"):
            llm.compile_canonical_results(snapshot, (malformed,), validate)
        results = llm.compile_canonical_results(
            snapshot,
            (malformed, success(snapshot, "u1", "good", 4)),
            validate,
        )
        self.assertEqual(results[0].attempt_id, "good")

    def test_pending_keys_include_only_units_without_a_valid_success(self):
        snapshot = request_set()
        malformed = success(snapshot, "u1", "bad", "not-an-int")
        attempts = (malformed, success(snapshot, "u2", "good", 2))
        self.assertEqual(
            llm.pending_request_keys(snapshot, attempts, validate), ("u1",))

    def test_canonical_artifact_requires_and_preserves_complete_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = request_set()
            request_ref = llm.publish_request_set(
                root / "requests",
                request_set=snapshot,
                dataset="harbor",
                version="v1",
                generator="unit-test",
                git_commit="deadbeef",
            )
            results = llm.compile_canonical_results(
                snapshot,
                (success(snapshot, "u2", "u2", 2),
                 success(snapshot, "u1", "u1", 1)),
                validate,
            )
            result_ref = llm.publish_canonical_results(
                root / "results",
                request_set=snapshot,
                request_artifact=request_ref,
                results=results,
                dataset="harbor",
                version="v1",
                generator="unit-test",
                git_commit="deadbeef",
            )
            artifact.open_artifact(
                result_ref.path,
                expected_kind=llm.RESULT_ARTIFACT_KIND,
                expected_dataset="harbor")
            loaded = llm.load_canonical_results(
                root / "results" / llm.CANONICAL_RESULTS_NAME, snapshot)
            self.assertEqual([item.key for item in loaded], ["u1", "u2"])
            with self.assertRaises(llm.IncompleteCoverageError):
                llm.publish_canonical_results(
                    root / "incomplete-results",
                    request_set=snapshot,
                    request_artifact=request_ref,
                    results=results[:1],
                    dataset="harbor",
                    version="v2",
                    generator="unit-test",
                    git_commit="deadbeef",
                )


if __name__ == "__main__":
    unittest.main()
