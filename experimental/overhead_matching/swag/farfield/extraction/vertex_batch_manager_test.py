"""Tests for vertex_batch_manager's pure helpers.

Nothing here touches the network: argument parsing, GCS URI arithmetic,
record normalization, resume-key scanning, and the cost guard's placement in
`run_requests` are all exercised without a client ever being constructed.
"""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experimental.overhead_matching.swag.farfield.extraction import llm_cost
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


class SubcommandModelRequiredTest(unittest.TestCase):
    """Each standalone subcommand refuses to parse without --model."""

    def assert_exits_wanting_model(self, argv):
        err = io.StringIO()
        with mock.patch.object(vbm.sys, "argv", ["vertex_batch_manager"] + argv):
            with contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as ctx:
                    vbm.main()
        self.assertEqual(ctx.exception.code, 2)  # argparse usage error
        self.assertIn("--model", err.getvalue())

    def test_run_online_requires_model(self):
        self.assert_exits_wanting_model(
            ["run-online", "--input", "in.jsonl", "--output", "out.jsonl"])

    def test_run_batch_requires_model(self):
        self.assert_exits_wanting_model(
            ["run-batch", "--input", "in.jsonl", "--output", "out.jsonl",
             "--gcs_prefix", "gs://bucket/stage"])

    def test_submit_all_requires_model(self):
        self.assert_exits_wanting_model(
            ["submit-all", "--input_prefix", "gs://bucket/in/",
             "--output_prefix", "gs://bucket/out/"])


class GcsUriHelpersTest(unittest.TestCase):

    def test_parse_gcs_uri(self):
        self.assertEqual(vbm.parse_gcs_uri("gs://bucket/path/to/files/"),
                         ("bucket", "path/to/files/"))
        self.assertEqual(vbm.parse_gcs_uri("gs://bucket"), ("bucket", ""))

    def test_parse_gcs_uri_rejects_non_gcs(self):
        with self.assertRaises(ValueError):
            vbm.parse_gcs_uri("s3://bucket/path")

    def test_get_output_uri(self):
        self.assertEqual(
            vbm.get_output_uri("gs://b/in/requests_0.jsonl", "gs://b/out"),
            "gs://b/out/requests_0/")
        self.assertEqual(
            vbm.get_output_uri("gs://b/in/requests_0.jsonl", "gs://b/out/"),
            "gs://b/out/requests_0/")

    def test_batch_stage_uri(self):
        requests_uri, results_prefix = vbm._batch_stage_uri(
            "gs://b/stage/", "matching_123")
        self.assertEqual(requests_uri, "gs://b/stage/matching_123/requests.jsonl")
        self.assertEqual(results_prefix, "gs://b/stage/matching_123/results/")


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


class UsableKeysTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_missing_file_has_no_usable_keys(self):
        self.assertEqual(vbm._usable_keys(self.root / "absent.jsonl"), set())

    def test_only_records_with_a_usable_response_count(self):
        path = self.root / "results.jsonl"
        with open(path, "w") as handle:
            handle.write(json.dumps({"key": "good", "response": {"x": 1}}) + "\n")
            handle.write(json.dumps({"key": "bad", "error": "boom"}) + "\n")
            handle.write(json.dumps({"key": "empty"}) + "\n")
            handle.write("not json\n")
        self.assertEqual(vbm._usable_keys(path), {"good"})


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


if __name__ == "__main__":
    unittest.main()
