import json
import tempfile
import unittest
from pathlib import Path

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

    def test_health_names_only_match_notes(self):
        health = self.client.get("/api/health").get_json()
        self.assertEqual(health["features"], ["match_notes"])
        self.assertNotIn("replay", json.dumps(health))

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


if __name__ == "__main__":
    unittest.main()
