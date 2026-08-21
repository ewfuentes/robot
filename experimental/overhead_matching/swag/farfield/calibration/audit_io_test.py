import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield.calibration import audit_io


def result_line(key, payload):
    """One results.jsonl line the way the batch runner writes it."""
    return json.dumps({
        "key": key,
        "response": {"candidates": [{"content": {"parts": [
            {"text": json.dumps(payload)}]}}]},
    })


def write_run(run_dir: Path, meta: dict, lines: list):
    audit_dir = run_dir / "semantic_audit"
    audit_dir.mkdir(parents=True)
    (audit_dir / "audit_meta.json").write_text(json.dumps(meta))
    (audit_dir / "results.jsonl").write_text("\n".join(lines) + "\n")


class LoadAuditsTest(unittest.TestCase):
    def test_maps_keys_through_meta_to_track_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {"verdict": "keep",
                       "valid_segments": [{"start_t": 0, "end_t": 3}]}
            write_run(Path(tmp), {"k1": {"track_id": 7}},
                      [result_line("k1", payload)])
            audits = audit_io.load_audits(Path(tmp))
            self.assertEqual(set(audits), {7})
            self.assertEqual(audits[7]["valid_segments"],
                             [{"start_t": 0, "end_t": 3}])

    def test_error_lines_and_unknown_keys_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(Path(tmp), {"k1": {"track_id": 1}}, [
                result_line("k1", {"verdict": "keep"}),
                json.dumps({"key": "k1", "error": "quota"}),
                result_line("k_unknown", {"verdict": "keep"}),
            ])
            self.assertEqual(set(audit_io.load_audits(Path(tmp))), {1})

    def test_unparseable_payload_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken = json.dumps({
                "key": "k2",
                "response": {"candidates": [{"content": {"parts": [
                    {"text": "not json {"}]}}]},
            })
            write_run(Path(tmp), {"k1": {"track_id": 1},
                                  "k2": {"track_id": 2}},
                      [result_line("k1", {"verdict": "keep"}), broken, ""])
            self.assertEqual(set(audit_io.load_audits(Path(tmp))), {1})

    def test_missing_audit_stage_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(audit_io.load_audits(Path(tmp)), {})

    def test_later_line_for_same_key_wins(self):
        # A rerun appends a fresh result for the same key; the reader keeps
        # the last one, matching "latest result is the result".
        with tempfile.TemporaryDirectory() as tmp:
            write_run(Path(tmp), {"k1": {"track_id": 5}}, [
                result_line("k1", {"verdict": "drop"}),
                result_line("k1", {"verdict": "keep"}),
            ])
            self.assertEqual(audit_io.load_audits(Path(tmp))[5]["verdict"],
                             "keep")


if __name__ == "__main__":
    unittest.main()
