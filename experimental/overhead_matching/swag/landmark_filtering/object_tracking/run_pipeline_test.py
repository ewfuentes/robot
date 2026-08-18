import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.data import farfield_paths
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    run_pipeline as rp,
)


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def usage_record(key, prompt, output, thinking=0):
    return {"key": key, "response": {
        "candidates": [{"content": {"parts": [{"text": "{}"}]}}],
        "usageMetadata": {"promptTokenCount": prompt,
                          "candidatesTokenCount": output,
                          "thoughtsTokenCount": thinking,
                          "totalTokenCount": prompt + output + thinking}}}


class TokenCostTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_sums_usage_across_files(self):
        a = self.root / "a.jsonl"
        b = self.root / "b.jsonl"
        write_jsonl(a, [usage_record("k1", 1000, 100, 50)])
        write_jsonl(b, [usage_record("k2", 2000, 200)])
        cost = rp.token_cost([a, b])
        self.assertEqual(cost["calls"], 2)
        self.assertEqual(cost["prompt_tokens"], 3000)
        # output + thinking are billed the same, so they are summed together.
        self.assertEqual(cost["output_tokens"], 350)
        self.assertEqual(cost["total_tokens"], 3350)
        expected = 3000 * 1.00 / 1e6 + 350 * 6.00 / 1e6
        self.assertAlmostEqual(cost["usd_on_demand"], round(expected, 2))

    def test_ignores_missing_files_and_error_records(self):
        path = self.root / "r.jsonl"
        write_jsonl(path, [
            usage_record("ok", 100, 10),
            {"key": "failed", "error": "TPU device returned error"},
            {"key": "batch_style_failure", "response": "{}"},
        ])
        cost = rp.token_cost([path, self.root / "does_not_exist.jsonl"])
        self.assertEqual(cost["calls"], 1)
        self.assertEqual(cost["prompt_tokens"], 100)

    def test_tolerates_unparseable_lines(self):
        path = self.root / "r.jsonl"
        path.write_text('{"key": "ok"\nnot json at all\n')
        self.assertEqual(rp.token_cost([path])["calls"], 0)


class GateTest(unittest.TestCase):
    """The two conditions that must stop a run rather than warn."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.paths = farfield_paths.FarfieldPaths(dataset="leg9",
                                                  root=self.root)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()

    def write_manifest(self, config):
        target = self.paths.frame_landmarks
        target.mkdir(parents=True, exist_ok=True)
        (target / "manifest.json").write_text(json.dumps({"config": config}))

    def test_incomplete_extraction_stops(self):
        self.write_manifest({"complete": False, "n_no_usable_response": 23})
        with self.assertRaises(SystemExit) as ctx:
            rp.check_extraction(self.paths, dry_run=False)
        message = str(ctx.exception)
        self.assertIn("23", message)
        # The message must carry the repair command, not just the complaint.
        self.assertIn("--retry_failed", message)

    def test_complete_extraction_passes(self):
        self.write_manifest({"complete": True, "n_no_usable_response": 0})
        rp.check_extraction(self.paths, dry_run=False)

    def test_absent_manifest_does_not_stop(self):
        # Nothing to judge yet; the extract stage itself will produce it.
        rp.check_extraction(self.paths, dry_run=False)

    def test_dry_run_never_stops(self):
        self.write_manifest({"complete": False, "n_no_usable_response": 5})
        rp.check_extraction(self.paths, dry_run=True)

    def test_unusable_offset_curve_stops_before_matching(self):
        (self.run_dir / "mount_offset_sweep.json").write_text(json.dumps({
            "usable": False, "verdict": "MULTIMODAL",
            "detail": "3 competitive minima at 65, 85, 210 deg"}))
        with self.assertRaises(SystemExit) as ctx:
            rp.check_offset(self.run_dir, dry_run=False)
        self.assertIn("MULTIMODAL", str(ctx.exception))

    def test_usable_offset_curve_passes(self):
        (self.run_dir / "mount_offset_sweep.json").write_text(json.dumps({
            "usable": True, "verdict": "SMOOTH UNIMODAL", "detail": "ok"}))
        rp.check_offset(self.run_dir, dry_run=False)

    def test_absent_sweep_does_not_stop(self):
        rp.check_offset(self.run_dir, dry_run=False)


class StageOrderTest(unittest.TestCase):

    def test_offset_precedes_match(self):
        # Matching consumes the offset, so the sweep must run first.
        self.assertLess(rp.STAGES.index("offset"), rp.STAGES.index("match"))

    def test_merge_precedes_offset(self):
        # The sweep reads merged/measurements.json.
        self.assertLess(rp.STAGES.index("merge"), rp.STAGES.index("offset"))

    def test_every_stage_has_a_completion_marker(self):
        paths = farfield_paths.FarfieldPaths(dataset="leg9",
                                            root=Path("/tmp/nonexistent"))
        outputs = rp.stage_outputs(paths, Path("/tmp/nonexistent/run"))
        self.assertEqual(set(outputs), set(rp.STAGES))


if __name__ == "__main__":
    unittest.main()
