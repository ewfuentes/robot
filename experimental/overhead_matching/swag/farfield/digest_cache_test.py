"""The cache must be fast, correct on change, and honest about its one gap."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experimental.overhead_matching.swag.farfield import (
    artifact,
    digest_cache,
)


class CacheTest(unittest.TestCase):
    def setUp(self):
        digest_cache.clear_process_memo()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.file = self.root / "input.bin"
        self.file.write_bytes(b"original")

    def tearDown(self):
        digest_cache.clear_process_memo()
        self._tmp.cleanup()

    def digest(self):
        return digest_cache.sha256_file(self.file, root=self.root)

    def test_matches_an_uncached_hash(self):
        self.assertEqual(self.digest(), artifact.sha256_file(self.file))

    def test_the_second_read_does_not_touch_the_file(self):
        """The point of the cache: the bytes are read once."""
        first = self.digest()
        digest_cache.flush()
        digest_cache.clear_process_memo()
        with mock.patch.object(artifact, "sha256_file",
                               side_effect=AssertionError(
                                   "file was rehashed")) as spy:
            self.assertEqual(
                digest_cache.sha256_file(self.file, root=self.root), first)
            spy.assert_not_called()

    def test_a_changed_file_is_rehashed(self):
        before = self.digest()
        digest_cache.flush()
        digest_cache.clear_process_memo()
        self.file.write_bytes(b"different length content")
        self.assertNotEqual(self.digest(), before)
        self.assertEqual(self.digest(), artifact.sha256_file(self.file))

    def test_a_same_length_rewrite_is_still_caught_by_mtime(self):
        before = self.digest()
        digest_cache.flush()
        digest_cache.clear_process_memo()
        stat = self.file.stat()
        self.file.write_bytes(b"chan9ed!")          # same 8 bytes
        self.assertEqual(self.file.stat().st_size, stat.st_size)
        self.assertNotEqual(self.digest(), before)

    def test_the_documented_gap_is_real_and_narrow(self):
        """A same-length rewrite that also restores mtime returns the stale
        digest. Asserted rather than left implicit: this is the trade the
        module makes, and a reader deserves to see its exact shape."""
        before = self.digest()
        digest_cache.flush()
        digest_cache.clear_process_memo()
        stat = self.file.stat()
        self.file.write_bytes(b"chan9ed!")
        os.utime(self.file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(self.digest(), before)
        # And the full check still sees it, which is why the trade is bounded.
        self.assertNotEqual(artifact.sha256_file(self.file), before)


class StoreTest(unittest.TestCase):
    def setUp(self):
        digest_cache.clear_process_memo()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        digest_cache.clear_process_memo()
        self._tmp.cleanup()

    def test_nothing_is_written_before_a_flush(self):
        target = self.root / "a.bin"
        target.write_bytes(b"a")
        digest_cache.sha256_file(target, root=self.root)
        self.assertFalse(digest_cache.cache_path(self.root).exists())
        digest_cache.flush()
        self.assertTrue(digest_cache.cache_path(self.root).exists())

    def test_a_flush_merges_rather_than_replaces(self):
        """Stages run concurrently against one root; a last-writer-wins
        replace would discard digests another process just paid for."""
        store = digest_cache.cache_path(self.root)
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps({
            "schema": digest_cache.SCHEMA,
            "entries": {"[\"other\",1,2,3,4]": "f" * 64},
        }))
        target = self.root / "b.bin"
        target.write_bytes(b"b")
        digest_cache.sha256_file(target, root=self.root)
        digest_cache.flush()
        entries = json.loads(store.read_text())["entries"]
        self.assertIn("[\"other\",1,2,3,4]", entries)
        self.assertEqual(len(entries), 2)

    def test_a_corrupt_cache_is_ignored_not_fatal(self):
        store = digest_cache.cache_path(self.root)
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text("{not json")
        target = self.root / "c.bin"
        target.write_bytes(b"c")
        self.assertEqual(digest_cache.sha256_file(target, root=self.root),
                         artifact.sha256_file(target))

    def test_a_foreign_schema_is_ignored_not_fatal(self):
        store = digest_cache.cache_path(self.root)
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps({"schema": "something/v9", "entries": {}}))
        target = self.root / "d.bin"
        target.write_bytes(b"d")
        self.assertEqual(digest_cache.sha256_file(target, root=self.root),
                         artifact.sha256_file(target))

    def test_an_unwritable_root_costs_speed_not_the_run(self):
        target = self.root / "e.bin"
        target.write_bytes(b"e")
        expected = artifact.sha256_file(target)
        digest_cache.sha256_file(target, root=self.root)
        with mock.patch.object(digest_cache, "_flush",
                               side_effect=OSError("read-only")):
            digest_cache.flush()
        digest_cache.clear_process_memo()
        self.assertEqual(digest_cache.sha256_file(target, root=self.root),
                         expected)

    def test_without_a_root_it_memoizes_in_process_only(self):
        target = self.root / "f.bin"
        target.write_bytes(b"f")
        self.assertEqual(digest_cache.sha256_file(target),
                         artifact.sha256_file(target))
        self.assertFalse(digest_cache.cache_path(self.root).exists())


if __name__ == "__main__":
    unittest.main()
