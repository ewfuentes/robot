import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    release_schedule,
    structs,
)
from experimental.overhead_matching.swag.farfield.tracking import tracklets


def accepted(tracklet_id, status, last_keyframe):
    return tracklets.AcceptedTracklet(
        tracklet_id=tracklet_id,
        local_id=tracklet_id,
        source_track={
            "status": status,
            "last_keyframe": last_keyframe,
        },
        audit={},
        valid_segments=(),
        provenance={},
        quality={})


def measurement(tracklet_id, anchor):
    return structs.TrackletMeasurement(
        tracklet_id=tracklet_id,
        anchor_keyframe_idx=anchor,
        bearing_forward_cw_deg=10.0,
        kappa=4.0)


def table(tracklet_id):
    return structs.CompatibilityTable(
        tracklet_id=tracklet_id,
        matcher_version="test",
        entries=[],
        default_log_lr=-2.0,
        clip_lo=-4.0,
        clip_hi=4.0,
        status="fast")


def reference(kind, digest, path):
    return artifact.ArtifactRef(
        kind=kind,
        dataset="dataset",
        version="v1",
        manifest_digest=digest * 64,
        content_digest=digest * 64,
        path=path)


def manifest(kind, upstreams=()):
    return artifact.ArtifactManifest(
        kind=kind,
        dataset="dataset",
        version="v1",
        generator="test",
        git_commit="test",
        created="now",
        arguments=(),
        content_digest="f" * 64,
        upstreams=tuple(upstreams),
        config={},
        declared_outputs=())


def export_data(upstreams, measurements=(), tables=None, *,
                input_digest="e", input_path="/artifacts/inputs"):
    inputs_ref = reference(
        paths_lib.LOCALIZATION_INPUTS, input_digest, input_path)
    return export_ingest.ExportData(
        artifact_ref=inputs_ref,
        manifest=manifest(paths_lib.LOCALIZATION_INPUTS, upstreams),
        meta=SimpleNamespace(n_keyframes=13),
        frame=None,
        catalog=None,
        landmarks=[],
        odometry=[],
        measurements=list(measurements),
        tables={} if tables is None else tables,
        truth=[])


class NaturalClosureScheduleTest(unittest.TestCase):

    def test_closed_and_eof_tracks_release_atomically(self):
        closed_early = measurement("closed", 2)
        closed_late = measurement("closed", 5)
        alive_early = measurement("alive", 1)
        alive_late = measurement("alive", 10)
        closed_table = table("closed")
        alive_table = table("alive")

        schedule = release_schedule.natural_closure_schedule(
            [accepted("alive", "alive", 10),
             accepted("closed", "closed", 8)],
            [alive_late, closed_late, alive_early, closed_early],
            {"alive": alive_table, "closed": closed_table},
            final_keyframe_idx=12)

        self.assertEqual(
            [(item.release_keyframe_idx, item.tracklet_id,
              [m.anchor_keyframe_idx for m in item.measurements])
             for item in schedule],
            [(8, "closed", [2, 5]), (12, "alive", [1, 10])])
        self.assertIs(schedule[0].measurements[0], closed_early)
        self.assertIs(schedule[0].table, closed_table)

    def test_requires_exact_matched_and_measurement_coverage(self):
        item = accepted("T1", "closed", 4)
        with self.assertRaisesRegex(ValueError, "do not exactly cover"):
            release_schedule.natural_closure_schedule(
                [item], [measurement("T1", 2)], {}, 5)
        with self.assertRaisesRegex(ValueError, "no localization measurements"):
            release_schedule.natural_closure_schedule(
                [item], [], {"T1": table("T1")}, 5)

    def test_rejects_measurement_after_track_release(self):
        with self.assertRaisesRegex(ValueError, "after release 4"):
            release_schedule.natural_closure_schedule(
                [accepted("T1", "closed", 4)],
                [measurement("T1", 5)],
                {"T1": table("T1")},
                final_keyframe_idx=8)


class ExportAncestryTest(unittest.TestCase):

    def test_loads_exact_bearing_track_and_audit_ancestry(self):
        bearing_ref = reference(
            paths_lib.BEARING_OBSERVATIONS, "a", "/artifacts/bearings")
        tracks_ref = reference(
            paths_lib.OBJECT_TRACKS, "b", "/artifacts/tracks")
        audits_ref = reference(
            paths_lib.SEMANTIC_AUDITS, "c", "/artifacts/audits")
        accepted_item = accepted("T1", "closed", 8)
        data = export_data(
            [bearing_ref], [measurement("T1", 3)], {"T1": table("T1")})
        loaded_audits = SimpleNamespace(
            tracks_ref=tracks_ref,
            semantic_audits_ref=audits_ref,
            source_tracks={1: accepted_item.source_track})

        with mock.patch.object(
                release_schedule.artifact, "open_artifact",
                return_value=bearing_ref) as open_artifact, \
             mock.patch.object(
                 release_schedule.artifact, "load_manifest",
                 return_value=manifest(
                     paths_lib.BEARING_OBSERVATIONS,
                     [tracks_ref, audits_ref])), \
             mock.patch.object(
                 release_schedule.audit_io, "load_audits",
                 return_value=loaded_audits) as load_audits, \
             mock.patch.object(
                 release_schedule.tracklets, "build_accepted_tracklets",
                 return_value=[accepted_item]):
            schedule = release_schedule.natural_closure_schedule_from_export(
                data)

        self.assertEqual(schedule[0].release_keyframe_idx, 8)
        self.assertEqual(schedule[0].measurements[0].anchor_keyframe_idx, 3)
        open_artifact.assert_called_once_with(
            bearing_ref.path,
            expected_kind=paths_lib.BEARING_OBSERVATIONS,
            expected_dataset="dataset",
            expected_version="v1")
        load_audits.assert_called_once_with(
            Path(tracks_ref.path), Path(audits_ref.path))

    def test_rejects_missing_bearing_ancestor(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            release_schedule.natural_closure_schedule_from_export(
                export_data([]))

    def test_rejects_wrong_bearing_ancestry(self):
        bearing_ref = reference(
            paths_lib.BEARING_OBSERVATIONS, "a", "/artifacts/bearings")
        tracks_ref = reference(
            paths_lib.OBJECT_TRACKS, "b", "/artifacts/tracks")
        data = export_data([bearing_ref])
        with mock.patch.object(
                release_schedule.artifact, "open_artifact",
                return_value=bearing_ref), \
             mock.patch.object(
                 release_schedule.artifact, "load_manifest",
                 return_value=manifest(
                     paths_lib.BEARING_OBSERVATIONS, [tracks_ref])):
            with self.assertRaisesRegex(ValueError, "upstreams must be exactly"):
                release_schedule.natural_closure_schedule_from_export(data)


class ReleaseSidecarTest(unittest.TestCase):

    def test_round_trip_rebuilds_bundles_after_artifact_relocation(self):
        original_measurement = measurement("T1", 3)
        original_table = table("T1")
        data = export_data(
            [], [original_measurement], {"T1": original_table})
        schedule = release_schedule.natural_closure_schedule(
            [accepted("T1", "closed", 8)], data.measurements, data.tables,
            data.n_keyframes - 1)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "track_releases.json"
            release_schedule.write_sidecar(path, data, schedule)
            document = json.loads(path.read_text())
            relocated = export_data(
                [], [original_measurement], {"T1": original_table},
                input_path="/pika/localization_inputs")
            loaded = release_schedule.load_sidecar(path, relocated)
            document["release_keyframe_by_tracklet"]["T1"] = 2
            artifact.atomic_write_json(path, document)
            with self.assertRaisesRegex(ValueError, "after release 2"):
                release_schedule.load_sidecar(path, relocated)

        self.assertEqual(loaded[0].release_keyframe_idx, 8)
        self.assertNotIn("measurements", document)
        self.assertNotIn("tables", document)
        self.assertEqual(
            [(item.tracklet_id, item.release_keyframe_idx) for item in loaded],
            [("T1", 8)])
        self.assertIs(loaded[0].measurements[0], original_measurement)
        self.assertIs(loaded[0].table, original_table)

    def test_rejects_stale_localization_inputs_reference(self):
        original_measurement = measurement("T1", 3)
        data = export_data(
            [], [original_measurement], {"T1": table("T1")})
        schedule = release_schedule.natural_closure_schedule(
            [accepted("T1", "closed", 8)], data.measurements, data.tables,
            data.n_keyframes - 1)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "track_releases.json"
            release_schedule.write_sidecar(path, data, schedule)
            stale_data = export_data(
                [], [original_measurement], {"T1": table("T1")},
                input_digest="d")
            with self.assertRaisesRegex(
                    ValueError, "different localization_inputs"):
                release_schedule.load_sidecar(path, stale_data)


if __name__ == "__main__":
    unittest.main()
