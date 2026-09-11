import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield.viewers import match_notes


DIGEST = "a" * 64
MATCHING = {
    "kind": "landmark_matches",
    "dataset": "pohang_canal_04",
    "version": "matcher-v1",
    "content_digest": DIGEST,
}
TRACKLET = "object_tracks:pohang_canal_04:tracks-v1@sha256:" + "b" * 64 + "#T7"


class MatchNotesStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = match_notes.MatchNotesStore(self.root)
        self.store.initialize()

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_update_and_delete(self):
        self.assertEqual(
            self.store.get(DIGEST), {"matching": None, "tracks": {}})
        note = self.store.put(
            matching=MATCHING, tracklet_id=TRACKLET, text="wrong buoy")
        self.assertEqual(note["text"], "wrong buoy")
        loaded = self.store.get(DIGEST)
        self.assertEqual(loaded["matching"], MATCHING)
        self.assertEqual(loaded["tracks"][TRACKLET]["text"], "wrong buoy")

        revised = self.store.put(
            matching=MATCHING, tracklet_id=TRACKLET,
            text="two objects merged")
        self.assertEqual(revised["text"], "two objects merged")
        self.assertIsNone(self.store.put(
            matching=MATCHING, tracklet_id=TRACKLET, text=""))
        self.assertEqual(
            self.store.get(DIGEST), {"matching": None, "tracks": {}})

    def test_file_is_one_central_validated_document(self):
        self.store.put(
            matching=MATCHING, tracklet_id=TRACKLET, text="check geometry")
        path = self.root / "_annotations" / "match_notes.json"
        document = json.loads(path.read_text())
        self.assertEqual(document["schema"], match_notes.SCHEMA)
        self.assertEqual(list(document["runs"]), [DIGEST])
        match_notes.validate_document(document)

    def test_conflicting_identity_and_malformed_file_fail_closed(self):
        self.store.put(
            matching=MATCHING, tracklet_id=TRACKLET, text="first")
        conflicting = dict(MATCHING, version="other-v1")
        with self.assertRaisesRegex(
                match_notes.MatchNotesError, "metadata conflicts"):
            self.store.put(
                matching=conflicting, tracklet_id=TRACKLET, text="second")
        self.store.notes_path.write_text('{"schema":"wrong","runs":{}}')
        with self.assertRaisesRegex(
                match_notes.MatchNotesError, "unsupported"):
            self.store.get(DIGEST)

    def test_limits_and_identity_are_enforced(self):
        with self.assertRaisesRegex(match_notes.MatchNotesError, "SHA-256"):
            self.store.get("not-a-digest")
        with self.assertRaisesRegex(match_notes.MatchNotesError, "at most"):
            self.store.put(
                matching=MATCHING, tracklet_id=TRACKLET,
                text="x" * (match_notes.MAX_NOTE_LENGTH + 1))


if __name__ == "__main__":
    unittest.main()
