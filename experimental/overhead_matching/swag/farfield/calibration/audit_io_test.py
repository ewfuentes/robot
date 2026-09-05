"""Tests for the explicit, source-bound semantic-audit reader."""

import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    artifact,
    paths as paths_lib,
)
from experimental.overhead_matching.swag.farfield.calibration import audit_io


DATASET = "example"


def source_track(track_id, birth=0, end=3):
    return {
        "track_id": track_id,
        "birth_keyframe": birth,
        "end_keyframe": end,
        "records": [
            {"keyframe": keyframe}
            for keyframe in range(birth, end + 1)
        ],
    }


def valid_audit(verdict="keep", segments=None):
    if segments is None:
        segments = [{"start_t": 0, "end_t": 3}]
    return {
        "landmark_kind": "fixed_structure",
        "single_object": verdict == "keep",
        "valid_segments": segments,
        "verdict": verdict,
        "drop_reason": "dynamic_object" if verdict == "drop" else "none",
        "primary_object": {
            "tags": [{"tag": "man_made=lighthouse", "weight": 0.9}],
            "name_candidates": [{
                "name": "Example Light",
                "weight": 0.8,
                "basis": "reported_by_detections",
            }],
            "name_aliases": [],
            "description": "A fixed lighthouse.",
            "distinctive_features": ["white tower"],
            "extent": "point_like",
        },
        "strike_votes": [],
        "secondary_objects": [],
        "confidence": "high",
        "unresolved": "",
    }


def result_line(key, payload):
    return json.dumps({
        "key": key,
        "response": {"candidates": [{"content": {"parts": [
            {"text": json.dumps(payload)},
        ]}}]},
    })


def tracks_document(tracks):
    return {
        "range": {"name": "full", "k_start": 0, "k_end": 3},
        "tracks": tracks,
    }


def write_artifacts(root: Path, *, tracks=None, lines=None,
                    track_files=None, mutate_meta=None,
                    audit_upstreams=None, audit_dataset=DATASET,
                    audit_kind=paths_lib.SEMANTIC_AUDITS,
                    include_results=True, audit_config=None):
    tracks = tracks if tracks is not None else [source_track(1)]
    tracks_dir = root / "tracks"
    payloads = track_files or {"tracks_full.json": tracks_document(tracks)}
    with artifact.ArtifactDirectoryBuilder(
            tracks_dir,
            kind=paths_lib.OBJECT_TRACKS,
            dataset=DATASET,
            version="v2",
            generator="audit_io_test",
            git_commit="test",
            declared_outputs=sorted(payloads)) as builder:
        for name, payload in payloads.items():
            artifact.atomic_write_json(builder.output_path(name), payload)
    tracks_ref = artifact.open_artifact(tracks_dir)
    tracks_path = tracks_dir / sorted(payloads)[0]
    requests = {
        f"T{track['track_id']}": {
            "track_id": track["track_id"],
            "range": "full",
            "birth_keyframe": track["birth_keyframe"],
            "source_track_sha256": audit_io.canonical_sha256(track),
        }
        for track in tracks
    }
    meta = {
        "schema": audit_io.META_SCHEMA,
        "source_tracks": {
            "artifact_id": audit_io.source_artifact_id(tracks_ref),
            "file": tracks_path.name,
            "sha256": audit_io.file_sha256(tracks_path),
        },
        "requests": requests,
    }
    if mutate_meta is not None:
        mutate_meta(meta)
    if lines is None:
        lines = [
            result_line(f"T{track['track_id']}", valid_audit())
            for track in tracks
        ]
    outputs = ["audit_meta.json"]
    if include_results:
        outputs.append("results.jsonl")
    semantic_audits_dir = root / "semantic_audits"
    config = audit_config or {
        "phase": "canonical_results",
        "coverage": "complete",
        "n_expected": len(requests),
        "n_successful": len(requests),
    }
    upstreams = ((tracks_ref,) if audit_upstreams is None
                 else audit_upstreams)
    with artifact.ArtifactDirectoryBuilder(
            semantic_audits_dir,
            kind=audit_kind,
            dataset=audit_dataset,
            version="v1",
            generator="audit_io_test",
            git_commit="test",
            upstreams=upstreams,
            config=config,
            declared_outputs=sorted(outputs)) as builder:
        artifact.atomic_write_json(
            builder.output_path("audit_meta.json"), meta)
        if include_results:
            artifact.atomic_write_file(
                builder.output_path("results.jsonl"),
                (("\n".join(lines) + "\n") if lines else "").encode())
    return tracks_dir, semantic_audits_dir, tracks_ref, meta


class LoadAuditsTest(unittest.TestCase):
    def test_maps_results_and_retains_both_artifact_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            track = source_track(7)
            tracks_dir, audits_dir, tracks_ref, _ = write_artifacts(
                root,
                tracks=[track],
                lines=[result_line("T7", valid_audit())])
            audits = audit_io.load_audits(tracks_dir, audits_dir)
            self.assertEqual(set(audits), {7})
            self.assertEqual(
                audits[7]["valid_segments"],
                [{"start_t": 0, "end_t": 3}])
            self.assertEqual(audits.tracks_ref, tracks_ref)
            self.assertEqual(
                audits.semantic_audits_ref,
                artifact.open_artifact(audits_dir))
            provenance = audits.provenance_by_track[7]
            self.assertEqual(
                provenance["source_tracks_artifact_id"],
                audit_io.source_artifact_id(tracks_ref))
            self.assertEqual(
                provenance["source_track_sha256"],
                audit_io.canonical_sha256(track))
            self.assertEqual(provenance["audit_key"], "T7")
            self.assertEqual(provenance["result_attempts"], 1)

    def test_transport_error_is_not_a_canonical_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracks_dir, audits_dir, _, _ = write_artifacts(
                Path(temporary), lines=[
                    json.dumps({"key": "T1", "error": "quota"}),
                ])
            with self.assertRaisesRegex(
                    audit_io.AuditArtifactError, "invalid fields"):
                audit_io.load_audits(tracks_dir, audits_dir)

    def test_missing_success_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracks_dir, audits_dir, _, _ = write_artifacts(
                Path(temporary), lines=[])
            with self.assertRaisesRegex(
                    audit_io.AuditArtifactError, "lacks canonical"):
                audit_io.load_audits(tracks_dir, audits_dir)

    def test_malformed_or_schema_invalid_payload_is_rejected(self):
        for line in (
                json.dumps({
                    "key": "T1",
                    "response": {"candidates": [{"content": {"parts": [
                        {"text": "not JSON {"},
                    ]}}]},
                }),
                result_line("T1", {"verdict": "keep"})):
            with self.subTest(line=line):
                with tempfile.TemporaryDirectory() as temporary:
                    tracks_dir, audits_dir, _, _ = write_artifacts(
                        Path(temporary), lines=[line])
                    with self.assertRaisesRegex(
                            audit_io.AuditArtifactError,
                            "invalid canonical audit result"):
                        audit_io.load_audits(tracks_dir, audits_dir)

    def test_duplicate_or_unexpected_result_key_is_rejected(self):
        line = result_line("T1", valid_audit())
        cases = (
            ([line, line], "duplicate canonical"),
            ([result_line("T999", valid_audit())], "unexpected result key"),
        )
        for lines, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary:
                    tracks_dir, audits_dir, _, _ = write_artifacts(
                        Path(temporary), lines=lines)
                    with self.assertRaisesRegex(
                            audit_io.AuditArtifactError, message):
                        audit_io.load_audits(tracks_dir, audits_dir)

    def test_exact_expected_key_coverage_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracks = [source_track(1), source_track(2)]
            tracks_dir, audits_dir, _, _ = write_artifacts(
                Path(temporary), tracks=tracks,
                lines=[result_line("T1", valid_audit())])
            with self.assertRaisesRegex(
                    audit_io.AuditArtifactError, "T2"):
                audit_io.load_audits(tracks_dir, audits_dir)

    def test_relationally_invalid_audit_and_weight_are_rejected(self):
        invalid_partial = valid_audit("keep_partial", [])
        invalid_weight = valid_audit()
        invalid_weight["primary_object"]["tags"][0]["weight"] = 1.1
        cases = (
            (invalid_partial, "keep_partial requires"),
            (invalid_weight, r"within \[0, 1\]"),
        )
        for payload, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary:
                    tracks_dir, audits_dir, _, _ = write_artifacts(
                        Path(temporary),
                        lines=[result_line("T1", payload)])
                    with self.assertRaisesRegex(
                            audit_io.AuditArtifactError, message):
                        audit_io.load_audits(tracks_dir, audits_dir)

    def test_legacy_or_unknown_audit_fields_are_rejected(self):
        legacy = valid_audit()
        legacy["primary_object"]["name"] = "legacy name"
        with tempfile.TemporaryDirectory() as temporary:
            tracks_dir, audits_dir, _, _ = write_artifacts(
                Path(temporary), lines=[result_line("T1", legacy)])
            with self.assertRaisesRegex(
                    audit_io.AuditArtifactError, "fields must be exactly"):
                audit_io.load_audits(tracks_dir, audits_dir)

    def test_old_or_extended_metadata_shapes_are_rejected(self):
        mutations = (
            lambda meta: meta.update({
                "schema": "farfield_semantic_audit_meta/v1"}),
            lambda meta: meta.update({"legacy": True}),
            lambda meta: meta["source_tracks"].update({"path": "old"}),
            lambda meta: meta["requests"]["T1"].update({"legacy": True}),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as temporary:
                    tracks_dir, audits_dir, _, _ = write_artifacts(
                        Path(temporary), mutate_meta=mutation)
                    with self.assertRaises(audit_io.AuditArtifactError):
                        audit_io.load_audits(tracks_dir, audits_dir)

    def test_file_track_and_artifact_identity_mismatches_are_rejected(self):
        mutations = (
            (lambda meta: meta["source_tracks"].update({"sha256": "0" * 64}),
             "file digest mismatch"),
            (lambda meta: meta["requests"]["T1"].update(
                {"source_track_sha256": "0" * 64}),
             "source-track digest mismatch"),
            (lambda meta: meta["source_tracks"].update(
                {"artifact_id": "object_tracks:wrong:v1@sha256:" + "0" * 64}),
             "artifact identity mismatch"),
        )
        for mutation, message in mutations:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary:
                    tracks_dir, audits_dir, _, _ = write_artifacts(
                        Path(temporary), mutate_meta=mutation)
                    with self.assertRaisesRegex(
                            audit_io.AuditArtifactError, message):
                        audit_io.load_audits(tracks_dir, audits_dir)

    def test_exactly_one_tracks_file_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            track = source_track(1)
            payload = tracks_document([track])
            tracks_dir, audits_dir, _, _ = write_artifacts(
                Path(temporary),
                tracks=[track],
                track_files={
                    "tracks_a.json": payload,
                    "tracks_b.json": payload,
                })
            with self.assertRaisesRegex(
                    audit_io.AuditArtifactError, "exactly one tracks_"):
                audit_io.load_audits(tracks_dir, audits_dir)

    def test_semantic_audit_must_bind_exact_tracks_ref(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracks_dir, audits_dir, _, _ = write_artifacts(
                Path(temporary), audit_upstreams=())
            with self.assertRaisesRegex(
                    audit_io.AuditArtifactError, "not bound"):
                audit_io.load_audits(tracks_dir, audits_dir)

    def test_both_inputs_must_be_completed_typed_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incomplete_tracks = root / "tracks"
            incomplete_tracks.mkdir()
            audits = root / "audits"
            audits.mkdir()
            with self.assertRaisesRegex(
                    audit_io.AuditArtifactError, "completed object_tracks"):
                audit_io.load_audits(incomplete_tracks, audits)

    def test_dataset_and_manifest_coverage_must_agree(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracks_dir, audits_dir, _, _ = write_artifacts(
                Path(temporary), audit_dataset="other")
            with self.assertRaisesRegex(
                    audit_io.AuditArtifactError, "dataset mismatch"):
                audit_io.load_audits(tracks_dir, audits_dir)
        with tempfile.TemporaryDirectory() as temporary:
            tracks_dir, audits_dir, _, _ = write_artifacts(
                Path(temporary), audit_config={
                    "phase": "canonical_results",
                    "coverage": "partial",
                    "n_expected": 1,
                    "n_successful": 1,
                })
            with self.assertRaisesRegex(
                    audit_io.AuditArtifactError, "complete canonical"):
                audit_io.load_audits(tracks_dir, audits_dir)

    def test_missing_results_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracks_dir, audits_dir, _, _ = write_artifacts(
                Path(temporary), include_results=False)
            with self.assertRaisesRegex(
                    audit_io.AuditArtifactError, "cannot open canonical"):
                audit_io.load_audits(tracks_dir, audits_dir)


if __name__ == "__main__":
    unittest.main()
