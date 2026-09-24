import csv
import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.loci import vlm_responses


MODEL = "gemini-3-flash-preview"
SCHEMA = {
    "type": "object",
    "properties": {
        "location_type": {"type": "string"},
        "landmarks": {
            "type": "array",
            "items": {"type": "object", "properties": {}, "required": []},
        },
    },
    "required": ["location_type", "landmarks"],
}


def _jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(
        artifact.canonical_json_bytes(record) + b"\n" for record in records))


def _response(text: str, model: str = MODEL) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "modelVersion": model,
    }


class VlmResponsesTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        self.dataset = base / "test_leg"
        self.root = base / "artifacts" / "test_leg" / "v1"
        self.dataset.mkdir(parents=True)
        self.root.mkdir(parents=True)
        self.keys = ("f0001,1,1,", "f0000,0,0,")
        with (self.dataset / "pano_id_mapping.csv").open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=("pano_id", "filename"))
            writer.writeheader()
            for key in self.keys:
                writer.writerow({
                    "pano_id": key.split(",", 1)[0],
                    "filename": f"{key}.jpg",
                })

        self.requests = {
            key: {
                "contents": [{"parts": [{"text": key}]}],
                "generationConfig": {"responseSchema": SCHEMA},
            }
            for key in self.keys
        }
        request_path = (
            self.root / "sentence_requests" / "panorama_sentence_requests" /
            "panorama_request_000.jsonl")
        _jsonl(request_path, [
            {"key": key, "request": self.requests[key]}
            for key in reversed(self.keys)
        ])
        shard = {
            "path": request_path.relative_to(self.root).as_posix(),
            "requests": 2,
            "bytes": request_path.stat().st_size,
            "sha256": artifact.sha256_file(request_path),
        }
        mapping = self.dataset / "pano_id_mapping.csv"
        manifest = {
            "complete": True,
            "counts": {
                "pano_id_mapping_rows": 2,
                "request_jsonl_bytes": shard["bytes"],
                "request_keys": 2,
                "shards": 1,
                "unique_request_keys": 2,
            },
            "dataset": self.dataset.name,
            "intended_inference": {"model": MODEL},
            "prompt_contract": {
                "response_schema_canonical_sha256": artifact.sha256_json(SCHEMA),
            },
            "request_set_sha256": artifact.sha256_json([shard]),
            "schema": vlm_responses.REQUEST_MANIFEST_SCHEMA,
            "shards": [shard],
            "upstream": {
                "pano_id_mapping_sha256": artifact.sha256_file(mapping),
            },
            "validation": {"status": "PASS"},
            "version": "v1",
        }
        artifact.atomic_write_json(self.root / "request_manifest.json", manifest)

    def tearDown(self):
        self.temporary.cleanup()

    def _raw(self, key: str, response: dict, *, request: dict | None = None,
             status: str = "") -> dict:
        echo = json.loads(json.dumps(request or self.requests[key]))
        for part in echo["contents"][0]["parts"]:
            part["inline_data"] = None
            part["media_resolution"] = None
        echo["cachedContent"] = None
        return {
            "key": key,
            "processed_time": "2026-09-13T00:00:00+00:00",
            "request": echo,
            "response": response,
            "status": status,
        }

    def _write_original(self, records: list[dict]) -> Path:
        path = (
            self.root / "sentences" / "results" / "panorama_request_000" /
            "prediction-model-1" / "predictions.jsonl")
        _jsonl(path, records)
        return path

    def _write_retry(self, key: str, record: dict) -> None:
        retry = self.root / "retries" / "retry_001"
        _jsonl(retry / "requests" / "panorama_request_000.jsonl", [{
            "key": key, "request": self.requests[key]}])
        _jsonl(
            retry / "sentences" / "results" / "panorama_request_000" /
            "prediction-model-2" / "predictions.jsonl", [record])

    def test_invalid_original_is_replaced_and_ordered(self):
        good = json.dumps({"location_type": "test", "landmarks": []})
        self._write_original([
            self._raw(self.keys[1], _response(good)),
            self._raw(self.keys[0], _response("{")),
        ])
        self._write_retry(self.keys[0], self._raw(
            self.keys[0], _response(good)))

        manifest = vlm_responses.finalize(self.dataset, self.root)

        canonical = [json.loads(line) for line in (
            self.root / "responses" / "canonical.jsonl").read_text().splitlines()]
        self.assertEqual([record["key"] for record in canonical], list(self.keys))
        self.assertEqual(manifest["counts"]["selected_retry"], 1)
        self.assertEqual(manifest["counts"]["invalid_attempts"], 1)
        self.assertEqual(
            manifest["selected_retry_lineage"][0]["key"], self.keys[0])
        self.assertEqual(manifest["validation"]["status"], "PASS")

    def test_echo_mismatch_and_duplicate_valid_response_fail_closed(self):
        good = json.dumps({"location_type": "test", "landmarks": []})
        changed = json.loads(json.dumps(self.requests[self.keys[0]]))
        changed["contents"][0]["parts"][0]["text"] = "changed"
        path = self._write_original([
            self._raw(self.keys[1], _response(good)),
            self._raw(self.keys[0], _response(good), request=changed),
        ])
        with self.assertRaisesRegex(ValueError, "echoed request differs"):
            vlm_responses.finalize(self.dataset, self.root)

        _jsonl(path, [
            self._raw(self.keys[1], _response(good)),
            self._raw(self.keys[0], _response(good)),
        ])
        self._write_retry(self.keys[0], self._raw(
            self.keys[0], _response(good)))
        with self.assertRaisesRegex(ValueError, "duplicate valid responses"):
            vlm_responses.finalize(self.dataset, self.root)


if __name__ == "__main__":
    unittest.main()
