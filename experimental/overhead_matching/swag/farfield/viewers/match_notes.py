"""One mutable, centralized note file for human matcher observations.

Notes are deliberately not scientific artifacts.  They are workbench comments
bound to an exact immutable ``landmark_matches`` content digest and one full
tracklet id.  The store serializes writers with a persistent file lock and
atomically replaces the JSON document, so two viewer tabs cannot corrupt it.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact


SCHEMA = "farfield.match_notes/v1"
ANNOTATIONS_DIR_NAME = "_annotations"
NOTES_NAME = "match_notes.json"
LOCK_NAME = ".match_notes.lock"
MAX_TRACKLET_ID_LENGTH = 2048
MAX_NOTE_LENGTH = 20_000

_ROOT_KEYS = frozenset({"schema", "runs"})
_RUN_KEYS = frozenset({"matching", "tracks"})
_MATCHING_KEYS = frozenset(
    {"kind", "dataset", "version", "content_digest"})
_NOTE_KEYS = frozenset({"text", "updated_at"})
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


class MatchNotesError(ValueError):
    """The notes store or a requested update is malformed."""


def _exact_keys(value, expected, what):
    actual = set(value) if isinstance(value, dict) else set()
    if not isinstance(value, dict) or actual != set(expected):
        raise MatchNotesError(
            f"{what} must have exact keys {sorted(expected)}; "
            f"missing={sorted(set(expected) - actual)}, "
            f"unknown={sorted(actual - set(expected))}")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise MatchNotesError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _empty_document():
    return {"schema": SCHEMA, "runs": {}}


def validate_matching(value) -> dict:
    """Return the small, portable identity recorded beside a note."""
    _exact_keys(value, _MATCHING_KEYS, "matching identity")
    if value["kind"] != "landmark_matches":
        raise MatchNotesError("matching identity kind must be landmark_matches")
    for key in ("dataset", "version"):
        try:
            artifact.require_identifier(value[key], f"matching.{key}")
        except artifact.ArtifactError as error:
            raise MatchNotesError(str(error)) from error
    digest = value["content_digest"]
    if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
        raise MatchNotesError(
            "matching.content_digest must be a lowercase SHA-256 digest")
    return dict(value)


def validate_tracklet_id(value) -> str:
    if (not isinstance(value, str) or not value
            or len(value) > MAX_TRACKLET_ID_LENGTH or "\x00" in value):
        raise MatchNotesError(
            f"tracklet_id must be 1..{MAX_TRACKLET_ID_LENGTH} characters")
    return value


def validate_text(value) -> str:
    if not isinstance(value, str):
        raise MatchNotesError("text must be a string")
    if len(value) > MAX_NOTE_LENGTH or "\x00" in value:
        raise MatchNotesError(
            f"text must be at most {MAX_NOTE_LENGTH} characters and contain "
            "no NUL bytes")
    return value


def _validate_timestamp(value, what):
    if not isinstance(value, str) or not value:
        raise MatchNotesError(f"{what} must be a non-empty timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise MatchNotesError(f"{what} is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MatchNotesError(f"{what} must include a timezone")


def validate_document(value) -> dict:
    _exact_keys(value, _ROOT_KEYS, "match notes document")
    if value["schema"] != SCHEMA:
        raise MatchNotesError(
            f"unsupported match notes schema {value['schema']!r}")
    runs = value["runs"]
    if not isinstance(runs, dict):
        raise MatchNotesError("match notes runs must be an object")
    for digest, run in runs.items():
        if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
            raise MatchNotesError(f"invalid run digest {digest!r}")
        _exact_keys(run, _RUN_KEYS, f"runs[{digest!r}]")
        matching = validate_matching(run["matching"])
        if matching["content_digest"] != digest:
            raise MatchNotesError(
                f"runs[{digest!r}] disagrees with its matching identity")
        tracks = run["tracks"]
        if not isinstance(tracks, dict):
            raise MatchNotesError(f"runs[{digest!r}].tracks must be an object")
        for tracklet_id, note in tracks.items():
            validate_tracklet_id(tracklet_id)
            _exact_keys(note, _NOTE_KEYS, f"note for {tracklet_id!r}")
            text = validate_text(note["text"])
            if not text.strip():
                raise MatchNotesError(
                    f"stored note for {tracklet_id!r} is empty")
            _validate_timestamp(
                note["updated_at"], f"note for {tracklet_id!r}.updated_at")
    return value


class MatchNotesStore:
    """Locked access to ``<data-root>/_annotations/match_notes.json``."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.annotations_dir = self.data_root / ANNOTATIONS_DIR_NAME
        self.notes_path = self.annotations_dir / NOTES_NAME
        self.lock_path = self.annotations_dir / LOCK_NAME

    def initialize(self) -> None:
        self._require_data_root()
        try:
            self.annotations_dir.mkdir(mode=0o700)
        except FileExistsError:
            pass
        self._require_annotations_dir()
        with self._locked():
            if not self.notes_path.exists():
                artifact.atomic_write_json(self.notes_path, _empty_document())
            self._load_unlocked()

    def _require_data_root(self):
        try:
            metadata = self.data_root.lstat()
        except FileNotFoundError as error:
            raise MatchNotesError(
                f"data root does not exist: {self.data_root}") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise MatchNotesError(
                f"data root must be a real directory: {self.data_root}")

    def _require_annotations_dir(self):
        try:
            metadata = self.annotations_dir.lstat()
        except FileNotFoundError as error:
            raise MatchNotesError(
                f"annotations directory is missing: {self.annotations_dir}") \
                from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise MatchNotesError(
                "annotations path must be a real directory, not a symlink")

    @contextmanager
    def _locked(self):
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.lock_path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise MatchNotesError("match notes lock is not a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _load_unlocked(self):
        try:
            metadata = self.notes_path.lstat()
        except FileNotFoundError:
            return _empty_document()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise MatchNotesError("match notes path is not a regular file")
        try:
            with self.notes_path.open(encoding="utf-8") as stream:
                document = json.load(
                    stream, object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        MatchNotesError(
                            f"non-finite JSON constant {value!r}")))
        except MatchNotesError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise MatchNotesError(
                f"cannot read {self.notes_path}: {error}") from error
        return validate_document(document)

    def get(self, matching_digest: str) -> dict:
        if (not isinstance(matching_digest, str)
                or not _DIGEST_RE.fullmatch(matching_digest)):
            raise MatchNotesError(
                "matching_digest must be a lowercase SHA-256 digest")
        with self._locked():
            document = self._load_unlocked()
            run = document["runs"].get(matching_digest)
            return ({"matching": None, "tracks": {}} if run is None else {
                "matching": dict(run["matching"]),
                "tracks": {
                    key: dict(value) for key, value in run["tracks"].items()
                },
            })

    def put(self, *, matching, tracklet_id, text) -> dict | None:
        matching = validate_matching(matching)
        tracklet_id = validate_tracklet_id(tracklet_id)
        text = validate_text(text)
        digest = matching["content_digest"]
        with self._locked():
            document = self._load_unlocked()
            run = document["runs"].get(digest)
            if run is not None and run["matching"] != matching:
                raise MatchNotesError(
                    "matching metadata conflicts with the existing digest")
            if not text.strip():
                if run is None:
                    return None
                run["tracks"].pop(tracklet_id, None)
                if not run["tracks"]:
                    del document["runs"][digest]
                artifact.atomic_write_json(self.notes_path, document)
                return None
            if run is None:
                run = {"matching": matching, "tracks": {}}
                document["runs"][digest] = run
            note = {
                "text": text,
                "updated_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds").replace("+00:00", "Z"),
            }
            run["tracks"][tracklet_id] = note
            artifact.atomic_write_json(self.notes_path, document)
            return dict(note)
