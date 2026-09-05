import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.localization import run_io
from experimental.overhead_matching.swag.farfield.viewers import (
    match_notes,
    server,
)


DIGEST = "c" * 64
MATCHING = {
    "kind": "landmark_matches",
    "dataset": "pohang_canal_04",
    "version": "matcher-v1",
    "content_digest": DIGEST,
}


def publish_localization_run(root: Path, n_particles: int = 50) -> Path:
    run_dir = root / "runs" / "experiment" / "run"
    relative = "checkpoints/kf_00000.npz"
    with artifact.ArtifactDirectoryBuilder(
            run_dir, kind=run_io.RUN_KIND, dataset="dataset", version="run",
            generator="viewer_server_test", git_commit="deadbeef",
            arguments=(), config={}, declared_outputs=(relative,)) as builder:
        checkpoint = builder.output_path(relative)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            checkpoint,
            east_m=np.arange(n_particles, dtype=np.float64),
            north_m=-np.arange(n_particles, dtype=np.float64),
            log_weight=np.zeros(n_particles, dtype=np.float64),
            mode_id=np.arange(n_particles, dtype=np.int64) % 2)
    return run_dir


class ViewerServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "index.html").write_text("root index")
        (self.root / "browse").mkdir()
        (self.root / "browse" / "result.txt").write_text("result")
        self.app = server.create_app(self.root)
        self.client = self.app.test_client()
        self.origin = "http://localhost"
        self.headers = {
            "Origin": self.origin,
            server.WRITE_HEADER: server.WRITE_HEADER_VALUE,
            "Sec-Fetch-Site": "same-origin",
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_serves_indexes_and_safe_directory_walks(self):
        self.assertEqual(self.client.get("/").data, b"root index")
        listing = self.client.get("/browse/").data.decode()
        self.assertIn("result.txt", listing)
        self.assertEqual(
            self.client.get("/browse/result.txt").data, b"result")
        self.assertEqual(self.client.get("/_annotations/match_notes.json")
                         .status_code, 404)

    def test_health_names_read_only_viewer_features(self):
        health = self.client.get("/api/health").get_json()
        self.assertEqual(
            health["features"], ["localization_particles", "match_notes"])
        self.assertNotIn("replay", json.dumps(health))

    def test_localization_particle_percentage_reads_one_run_checkpoint(self):
        run_dir = publish_localization_run(self.root)

        response = self.client.get(
            "/api/localization-particles/0",
            query_string={"run": str(run_dir), "percent": 10})

        self.assertEqual(response.status_code, 200, response.data)
        body = response.get_json()
        self.assertEqual(body["percent"], 10)
        self.assertEqual(body["n"], 5)
        self.assertEqual(body["total"], 50)
        self.assertEqual(len(body["e"]), 5)

    def test_particle_requests_are_confined_and_percentages_are_bounded(self):
        run_dir = publish_localization_run(self.root)
        outside = Path(self.tmp.name).parent
        self.assertEqual(self.client.get(
            "/api/localization-particles/0",
            query_string={"run": str(outside), "percent": 10}).status_code,
                         404)
        response = self.client.get(
            "/api/localization-particles/0",
            query_string={"run": str(run_dir), "percent": 25})
        self.assertEqual(response.status_code, 400)
        self.assertIn("one of", response.get_json()["error"])

    def test_same_origin_put_round_trips(self):
        body = {
            "matching": MATCHING,
            "tracklet_id": "tracks:pohang#T1",
            "text": "human says this match is odd",
        }
        response = self.client.put(
            "/api/match-notes", json=body, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.data)
        loaded = self.client.get(
            "/api/match-notes", query_string={
                "matching_digest": DIGEST,
            }).get_json()
        self.assertEqual(
            loaded["tracks"]["tracks:pohang#T1"]["text"], body["text"])
        document = json.loads(
            (self.root / match_notes.ANNOTATIONS_DIR_NAME
             / match_notes.NOTES_NAME).read_text())
        self.assertEqual(document["runs"][DIGEST]["matching"], MATCHING)

    def test_cross_origin_and_unmarked_writes_are_rejected(self):
        body = {
            "matching": MATCHING,
            "tracklet_id": "tracks:pohang#T1",
            "text": "attack",
        }
        self.assertEqual(
            self.client.put("/api/match-notes", json=body,
                            headers={"Origin": self.origin}).status_code,
            403)
        hostile = dict(self.headers)
        hostile["Origin"] = "http://attacker.example"
        hostile["Sec-Fetch-Site"] = "cross-site"
        self.assertEqual(
            self.client.put("/api/match-notes", json=body,
                            headers=hostile).status_code,
            403)

    def test_read_only_mode_serves_files_without_creating_notes(self):
        root = self.root / "read-only"
        root.mkdir()
        (root / "index.html").write_text("read-only index")
        annotations = root / match_notes.ANNOTATIONS_DIR_NAME
        app = server.create_app(root, read_only=True)
        client = app.test_client()

        self.assertEqual(client.get("/").data, b"read-only index")
        self.assertEqual(
            client.get("/api/health").get_json()["features"],
            ["localization_particles"])
        self.assertEqual(client.get("/api/match-notes").status_code, 404)
        self.assertEqual(client.put("/api/match-notes", json={}).status_code, 405)
        self.assertFalse(annotations.exists())


if __name__ == "__main__":
    unittest.main()
