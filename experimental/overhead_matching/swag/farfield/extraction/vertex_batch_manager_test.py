"""Tests for vertex_batch_manager's pure helpers.

Nothing here touches the network: argument parsing, GCS URI arithmetic,
record normalization, retry-path allocation, and the cost guard's placement in
`run_requests` are all exercised without a client ever being constructed.
"""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experimental.overhead_matching.swag.farfield.extraction import llm_cost
from experimental.overhead_matching.swag.farfield.extraction import prompts
from experimental.overhead_matching.swag.farfield.extraction import (
    vertex_batch_manager as vbm,
)


def write_requests(path: Path, requests):
    with open(path, "w") as handle:
        for i, request in enumerate(requests):
            handle.write(json.dumps({"key": f"k{i}", "request": request}) + "\n")


def text_request(chars: int) -> dict:
    return {"contents": [{"parts": [{"text": "x" * chars}], "role": "user"}]}


def execution_args(**overrides) -> argparse.Namespace:
    """A parsed-args namespace matching add_execution_arguments' contract."""
    base = dict(model="test-model", online=False, gcs_prefix=None, parallel=1,
                poll_interval=1, cost_limit=50.0, approve_cost=False)
    base.update(overrides)
    return argparse.Namespace(**base)


class ExecutionArgumentsTest(unittest.TestCase):

    def make_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        vbm.add_execution_arguments(parser)
        return parser

    def test_model_is_required(self):
        # The model is a modeling choice recorded in the run config; the flag
        # block must not supply one.
        parser = self.make_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([])

    def test_model_has_no_default(self):
        parser = self.make_parser()
        (model_action,) = [a for a in parser._actions if a.dest == "model"]
        self.assertTrue(model_action.required)
        self.assertIsNone(model_action.default)

    def test_no_default_gcs_bucket_or_prefix(self):
        # The hardcoded staging bucket is gone: no --gcs_bucket flag at all,
        # and --gcs_prefix carries no default.
        parser = self.make_parser()
        args = parser.parse_args(["--model", "test-model"])
        self.assertIsNone(args.gcs_prefix)
        self.assertFalse(hasattr(args, "gcs_bucket"))

    def test_mechanical_knobs_still_default(self):
        args = self.make_parser().parse_args(["--model", "test-model"])
        self.assertFalse(args.online)
        self.assertEqual(args.parallel, 8)
        self.assertEqual(args.poll_interval, 120)
        self.assertEqual(args.cost_limit, 50.0)
        self.assertFalse(args.approve_cost)


class StandaloneExecutionRemovedTest(unittest.TestCase):
    """Raw execution CLIs cannot bypass lifecycle and cost ownership."""

    def assert_execution_command_is_absent(self, command):
        err = io.StringIO()
        with mock.patch.object(
                vbm.sys, "argv", ["vertex_batch_manager", command]):
            with contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as ctx:
                    vbm.main()
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("invalid choice", err.getvalue())

    def test_raw_execution_subcommands_are_absent(self):
        for command in ("submit-all", "run-online", "run-batch"):
            with self.subTest(command=command):
                self.assert_execution_command_is_absent(command)


class GcsUriHelpersTest(unittest.TestCase):

    def test_parse_gcs_uri(self):
        self.assertEqual(vbm.parse_gcs_uri("gs://bucket/path/to/files/"),
                         ("bucket", "path/to/files/"))
        self.assertEqual(vbm.parse_gcs_uri("gs://bucket"), ("bucket", ""))

    def test_parse_gcs_uri_rejects_non_gcs(self):
        with self.assertRaises(ValueError):
            vbm.parse_gcs_uri("s3://bucket/path")

    def test_batch_stage_uri(self):
        requests_uri, results_prefix = vbm._batch_stage_uri(
            "gs://b/stage/", "matching_123", "submission-1")
        self.assertEqual(
            requests_uri,
            "gs://b/stage/matching_123/submissions/submission-1/requests.jsonl")
        self.assertEqual(
            results_prefix,
            "gs://b/stage/matching_123/submissions/submission-1/results/")

    def test_identical_tags_get_distinct_submission_prefixes(self):
        first = vbm._batch_stage_uri("gs://b/stage", "matching_123")
        second = vbm._batch_stage_uri("gs://b/stage", "matching_123")
        self.assertNotEqual(first, second)
        self.assertTrue(first[0].startswith(
            "gs://b/stage/matching_123/submissions/"))
        self.assertTrue(second[1].startswith(
            "gs://b/stage/matching_123/submissions/"))

    def test_request_shard_reserves_a_fresh_local_raw_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            first_index, first_request, first_raw = (
                vbm.next_submission_paths(work_dir))
            self.assertEqual(first_index, 1)
            first_request.write_text("reserved before provider execution\n")

            second_index, second_request, second_raw = (
                vbm.next_submission_paths(work_dir))
            self.assertEqual(second_index, 2)
            self.assertNotEqual(first_request, second_request)
            self.assertNotEqual(first_raw, second_raw)

    def test_completed_results_include_error_sidecar_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            _, request_path, raw_path = vbm.next_submission_paths(work_dir)
            request_path.write_text("reserved\n")
            raw_path.write_text('{"key":"a","response":{}}\n')
            errors_path = raw_path.with_suffix(".errors.jsonl")
            errors_path.write_text('{"key":"b","error":"quota"}\n')
            self.assertEqual(
                vbm.completed_submission_results(work_dir),
                (raw_path, errors_path))


class NormalizeBatchRecordTest(unittest.TestCase):

    def test_failure_status_becomes_error(self):
        out = vbm._normalize_batch_record(
            {"key": "a", "status": "INVALID_ARGUMENT", "response": "{}"})
        self.assertEqual(out["key"], "a")
        self.assertEqual(out["error"], "INVALID_ARGUMENT")
        self.assertNotIn("response", out)

    def test_missing_response_object_becomes_error(self):
        out = vbm._normalize_batch_record({"key": "b", "response": "{}"})
        self.assertIn("error", out)

    def test_success_passes_the_response_through(self):
        response = {"candidates": [], "usageMetadata": {"promptTokenCount": 1}}
        out = vbm._normalize_batch_record(
            {"key": "c", "status": "", "response": response})
        self.assertEqual(out, {"key": "c", "response": response})


class RunRequestsGuardTest(unittest.TestCase):
    """The cost guard fires inside run_requests, before either transport."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.requests = self.root / "requests.jsonl"
        self.results = self.root / "results.jsonl"

    def test_over_limit_raises_before_any_transport_runs(self):
        write_requests(self.requests, [text_request(4000)])
        args = execution_args(online=True, cost_limit=0.0)
        with mock.patch.object(vbm, "cmd_run_online") as run_online, \
                mock.patch.object(vbm, "cmd_run_batch") as run_batch, \
                mock.patch.object(llm_cost.sys, "stdin") as stdin, \
                contextlib.redirect_stdout(io.StringIO()):
            stdin.isatty.return_value = False
            with self.assertRaises(llm_cost.CostLimitExceeded):
                vbm.run_requests(args, self.requests, self.results, tag="t")
        run_online.assert_not_called()
        run_batch.assert_not_called()

    def test_existing_raw_output_is_refused_before_transport(self):
        write_requests(self.requests, [text_request(100)])
        self.results.write_text("sentinel\n")
        args = execution_args(online=True)
        with mock.patch.object(vbm, "cmd_run_online") as run_online, \
                mock.patch.object(vbm, "cmd_run_batch") as run_batch:
            with self.assertRaises(FileExistsError):
                vbm.run_requests(args, self.requests, self.results, tag="t")
        self.assertEqual(self.results.read_text(), "sentinel\n")
        run_online.assert_not_called()
        run_batch.assert_not_called()

    def test_estimate_is_priced_at_the_runs_model(self):
        # The guard must compare against the model actually being run, not the
        # Pro-rate fallback, which over-reports Flash work ~5x.
        write_requests(self.requests, [text_request(4000)])
        args = execution_args(model="gemini-3.7-flash", online=True)
        out = io.StringIO()
        with mock.patch.object(vbm, "cmd_run_online") as run_online, \
                contextlib.redirect_stdout(out):
            vbm.run_requests(args, self.requests, self.results, tag="t")
        run_online.assert_called_once()
        self.assertIn("gemini-3.7-flash", out.getvalue())
        self.assertNotIn("no table entry", out.getvalue())

    def test_batch_without_gcs_prefix_is_a_clear_error(self):
        write_requests(self.requests, [text_request(100)])
        args = execution_args(online=False, gcs_prefix=None)
        with mock.patch.object(vbm, "cmd_run_batch") as run_batch, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                vbm.run_requests(args, self.requests, self.results, tag="t")
        run_batch.assert_not_called()
        self.assertIn("--gcs_prefix", str(ctx.exception.code))
        self.assertIn("--online", str(ctx.exception.code))

    def test_batch_with_prefix_dispatches_to_run_batch(self):
        write_requests(self.requests, [text_request(100)])
        args = execution_args(online=False, gcs_prefix="gs://bucket/stage")
        with mock.patch.object(vbm, "cmd_run_batch") as run_batch, \
                contextlib.redirect_stdout(io.StringIO()):
            vbm.run_requests(args, self.requests, self.results, tag="t")
        run_batch.assert_called_once()
        batch_args = run_batch.call_args.args[0]
        self.assertEqual(batch_args.gcs_prefix, "gs://bucket/stage")
        self.assertEqual(batch_args.model, "test-model")
        self.assertEqual(batch_args.tag, "t")


class OnlineAdapterExecutionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_all_media_resolutions_reach_sdk_in_exact_location(self):
        images = [
            ("image/jpeg", "AAAA"),
            ("image/jpeg", "BBBB"),
            ("image/jpeg", "CCCC"),
            ("image/jpeg", "DDDD"),
        ]
        for index, resolution in enumerate(prompts.MEDIA_RESOLUTIONS):
            with self.subTest(resolution=resolution):
                record = prompts.build_request(
                    "frame", images,
                    prompt_type="osm_tags_farfield_v2",
                    media_resolution=resolution,
                    thinking_level="HIGH")
                request_path = self.root / f"requests_{index}.jsonl"
                output_path = self.root / f"results_{index}.jsonl"
                request_path.write_text(json.dumps(record) + "\n")

                response = SimpleNamespace(
                    text="{}",
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=1,
                        candidates_token_count=2,
                        thoughts_token_count=3,
                        total_token_count=6,
                    ),
                )
                client = mock.Mock()
                client.models.generate_content.return_value = response
                args = argparse.Namespace(
                    input=str(request_path),
                    output=str(output_path),
                    model="test-model",
                    parallel=1,
                )
                with mock.patch.object(vbm, "check_environment"), \
                        mock.patch.object(
                            vbm.genai, "Client", return_value=client), \
                        contextlib.redirect_stdout(io.StringIO()):
                    vbm.cmd_run_online(args)

                boundary = json.loads(output_path.read_text())
                self.assertEqual(set(boundary), {"key", "response"})
                self.assertEqual(
                    boundary,
                    vbm._normalize_batch_record({
                        "key": boundary["key"],
                        "status": "",
                        "response": boundary["response"],
                    }))
                self.assertEqual(
                    output_path.with_suffix(".errors.jsonl").read_text(), "")

                call = client.models.generate_content.call_args
                config = call.kwargs["config"]
                image_parts = call.kwargs["contents"][0]["parts"][:4]
                self.assertEqual(
                    config["thinking_config"], {
                        "thinking_level": "HIGH",
                    })
                if resolution == "MEDIA_RESOLUTION_ULTRA_HIGH":
                    self.assertNotIn("media_resolution", config)
                    self.assertTrue(all(
                        part["media_resolution"] == {"level": resolution}
                        for part in image_parts))
                else:
                    self.assertEqual(config["media_resolution"], resolution)
                    self.assertTrue(all(
                        "media_resolution" not in part
                        for part in image_parts))

    def test_online_error_sidecar_uses_exact_lifecycle_boundary(self):
        record = prompts.build_request(
            "frame", [("image/jpeg", "AAAA")],
            prompt_type="osm_tags_farfield_v2",
            media_resolution="MEDIA_RESOLUTION_HIGH",
            thinking_level="HIGH")
        request_path = self.root / "requests_error.jsonl"
        output_path = self.root / "results_error.jsonl"
        request_path.write_text(json.dumps(record) + "\n")
        client = mock.Mock()
        client.models.generate_content.side_effect = RuntimeError("quota")
        args = argparse.Namespace(
            input=str(request_path), output=str(output_path),
            model="test-model", parallel=1)
        with mock.patch.object(vbm, "check_environment"), mock.patch.object(
                vbm.genai, "Client", return_value=client), \
                contextlib.redirect_stdout(io.StringIO()):
            vbm.cmd_run_online(args)

        self.assertEqual(output_path.read_text(), "")
        error = json.loads(
            output_path.with_suffix(".errors.jsonl").read_text())
        self.assertEqual(set(error), {"key", "error"})
        self.assertEqual(error["key"], "frame")
        self.assertIn("RuntimeError: quota", error["error"])


if __name__ == "__main__":
    unittest.main()
