import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    checksums,
    make_dataset_timelapse as timelapse,
    trim_dataset,
)


def gps(dists):
    return [{"dist_m": str(d)} for d in dists]


class KeepBySpacingTest(unittest.TestCase):
    def test_uniform_source_lands_on_the_target_spacing(self):
        # A 3.3 m source cannot hit 10 m exactly; 9.9 is the nearest it can
        # do. The point is that it undershoots rather than overshooting to
        # 13.2.
        dists = [3.3 * i for i in range(200)]
        keep = trim_dataset.keep_by_spacing(gps(dists), 10)
        gaps = [dists[b] - dists[a] for a, b in zip(keep, keep[1:])]
        self.assertTrue(all(abs(g - 9.9) < 1e-6 for g in gaps), gaps[:5])

    def test_first_past_target_would_overshoot(self):
        # Guards the choice of "nearest" over "first at or past": the naive
        # rule gives 13.2 m for a 10 m request on a 3.3 m collect, a 32% error
        # that compounds into a third fewer frames than asked for.
        dists = [3.3 * i for i in range(200)]
        keep = trim_dataset.keep_by_spacing(gps(dists), 10)
        realized = (dists[keep[-1]] - dists[keep[0]]) / (len(keep) - 1)
        self.assertLess(abs(realized - 10.0), abs(13.2 - 10.0))

    def test_stationary_run_collapses_to_one_frame(self):
        # 50 frames at a dead stop: distance-based selection must not keep
        # them all just because they are 50 separate rows.
        dists = [0.0] * 50 + [3.0 * i for i in range(1, 60)]
        keep = trim_dataset.keep_by_spacing(gps(dists), 10)
        self.assertEqual([i for i in keep if i < 50], [0])

    def test_short_tail_dropped_long_tail_kept(self):
        base = [10.0 * i for i in range(10)]
        self.assertEqual(
            trim_dataset.keep_by_spacing(gps(base + [92.0]), 10)[-1], 9)
        self.assertEqual(
            trim_dataset.keep_by_spacing(gps(base + [97.0]), 10)[-1], 10)

    def test_endpoint_never_creates_a_sub_half_spacing_gap(self):
        dists = [3.3 * i for i in range(200)]
        keep = trim_dataset.keep_by_spacing(gps(dists), 10)
        gaps = [dists[b] - dists[a] for a, b in zip(keep, keep[1:])]
        self.assertGreaterEqual(min(gaps), 5.0)

    def test_track_shorter_than_one_spacing_raises(self):
        with self.assertRaises(ValueError):
            trim_dataset.keep_by_spacing(gps([0.0, 1.0]), 100)

    def test_non_monotonic_distance_raises(self):
        with self.assertRaises(ValueError):
            trim_dataset.keep_by_spacing(gps([0.0, 5.0, 3.0]), 1)

    def test_keeps_every_frame_when_spacing_is_below_the_source(self):
        dists = [10.0 * i for i in range(20)]
        keep = trim_dataset.keep_by_spacing(gps(dists), 1)
        self.assertEqual(keep, list(range(20)))


DEG_PER_M_LAT = 1.0 / 111195.0   # 1 deg latitude on the haversine sphere


def build_dataset(root: Path, n=12, step=3.0, with_checksums=True,
                  metadata_extra=None):
    """Minimum dataset satisfying the tables trim_dataset rewrites.

    Positions and `dist_m` are kept consistent with each other: selection
    reads `dist_m` but `rebuild_gps_rows` re-derives it by haversine over the
    kept positions, so a fixture where the two disagree measures the
    disagreement rather than the code.
    """
    frames = root / "frames"
    frames.mkdir(parents=True)
    (root / "panorama").symlink_to("frames")
    gps_rows, log_rows, intr_rows, map_rows = [], [], [], []
    for i in range(n):
        lat, lon = 42.0 + i * step * DEG_PER_M_LAT, -71.0
        name = f"f{i:04d},{lat:.7f},{lon:.7f},.jpg"
        (frames / name).write_bytes(f"image {i}".encode())
        gps_rows.append({"idx": i, "video_t_s": f"{2.0 * i:.3f}",
                         "sensor_elapsed_s": f"{2.0 * i:.3f}",
                         "dist_m": f"{100.0 + step * i:.1f}",
                         "latitude": f"{lat:.7f}", "longitude": f"{lon:.7f}",
                         "altitude_m": "1.0", "speed_mps": "1.5",
                         "frame_file": name})
        log_rows.append({"frame_idx": i, "pano_id": f"f{i:04d}",
                         "sequence_id": "seq", "sequence_position": i,
                         "camera_type": "equirectangular",
                         "geometry_source": "fix", "lat": f"{lat:.7f}",
                         "lng": f"{lon:.7f}", "heading_used": "0",
                         "captured_at": str(1_700_000_000_000 + i * 2000),
                         "original_path": f"raw_{i}.jpg",
                         "output_filename": name, "frame_id": f"p{i:05d}",
                         "gps_quality": "fix", "gps_valid": "1"})
        intr_rows.append({"idx": i, "pano_id": f"f{i:04d}",
                          "projection": "equirectangular", "width": "7680",
                          "height": "3840", "heading_deg": "0",
                          "future_source_column": f"raw-{i}"})
        map_rows.append({"pano_id": f"f{i:04d}", "lat": f"{lat:.7f}",
                         "lon": f"{lon:.7f}", "filename": name})
    for name, rows in (("frames_gps.csv", gps_rows),
                       ("extraction_log.csv", log_rows),
                       ("intrinsics.csv", intr_rows),
                       ("pano_id_mapping.csv", map_rows)):
        trim_dataset.write_csv(root / name, rows, list(rows[0]))
    meta = {
        "num_images": n, "projection": "equirectangular",
        "trajectory_km": round(step * (n - 1) / 1000.0, 3),
    }
    meta.update(metadata_extra or {})
    (root / "pipeline_metadata.json").write_text(json.dumps(meta, indent=2))
    if with_checksums:
        (root / checksums.CHECKSUM_FILE).write_text("")
        checksums.regenerate(root)
    return root


def read_rows(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def snapshot_tree(root: Path):
    records = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            records.append((relative, "symlink", path.readlink().as_posix()))
        elif path.is_file():
            records.append((relative, "file", path.read_bytes()))
        elif path.is_dir():
            records.append((relative, "dir", None))
    return records


class ApplyTrimTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ds = build_dataset(Path(self.tmp.name) / "ds")
        self.addCleanup(self.tmp.cleanup)

    def density_trim(self, spacing=9.0, **kwargs):
        rows, _ = trim_dataset.read_csv(self.ds / "frames_gps.csv")
        keep = trim_dataset.keep_by_spacing(rows, spacing)
        info = trim_dataset.apply_trim(
            self.ds, keep, kwargs.pop("reason", "too dense"), None,
            trim_dir=kwargs.pop("trim_dir", "trimmed_frames_for_density"),
            kind=kwargs.pop("kind", "density"))
        return keep, info

    def publish_timelapse(self, *, color=(20, 40, 60)):
        def plot(dataset, lats, lons, times, dists, out):
            del dataset, lats, lons, times, dists
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color).save(out, format="PNG")

        def video(paths, lats, lons, out, width, fps, max_frames):
            del paths, lats, lons, width, fps, max_frames
            out.write_bytes(b"\x00\x00\x00\x18ftypmp42video")

        with mock.patch.object(timelapse, "stage_plot", side_effect=plot), \
             mock.patch.object(timelapse, "stage_video", side_effect=video):
            return timelapse.render(
                self.ds, width=640, fps=12,
                max_frames=100, skip_video=False)

    def test_tables_stay_consistent_after_a_density_trim(self):
        keep, _ = self.density_trim()
        n = len(keep)
        for name in trim_dataset.CSV_NAMES:
            self.assertEqual(len(read_rows(self.ds / name)), n, name)
        images = sorted(p.name for p in (self.ds / "frames").glob("*.jpg"))
        self.assertEqual(len(images), n)
        gps_rows = read_rows(self.ds / "frames_gps.csv")
        self.assertEqual([int(r["idx"]) for r in gps_rows], list(range(n)))
        # The join key the farfield ingest relies on.
        self.assertEqual([r["frame_file"] for r in gps_rows], images)
        for i, row in enumerate(gps_rows):
            self.assertEqual(int(row["frame_file"].split(",")[0][1:]), i)

    def test_dropped_images_move_to_the_named_trim_dir(self):
        keep, info = self.density_trim()
        trim_dir = self.ds / "trimmed_frames_for_density"
        moved = list(trim_dir.glob("*.jpg"))
        self.assertEqual(len(moved), 12 - len(keep))
        self.assertEqual(info["n_dropped"], 12 - len(keep))
        self.assertFalse((self.ds / "trimmed_frames").exists())

    def test_dropped_csv_records_reason_and_original_index(self):
        keep, _ = self.density_trim(reason="3 m is denser than we need")
        rows = read_rows(self.ds / "trimmed_frames_for_density"
                         / "dropped_frames.csv")
        self.assertEqual(len(rows), 12 - len(keep))
        self.assertTrue(all(r["trim_kind"] == "density" for r in rows))
        self.assertTrue(all(r["reason"] == "3 m is denser than we need"
                            for r in rows))
        dropped_idx = sorted(int(r["original_idx"]) for r in rows)
        self.assertEqual(dropped_idx,
                         [i for i in range(12) if i not in set(keep)])

    def test_existing_trim_output_is_a_no_clobber_error(self):
        self.density_trim(spacing=6.0)
        before = snapshot_tree(self.ds)
        rows, _ = trim_dataset.read_csv(self.ds / "frames_gps.csv")
        keep2 = trim_dataset.keep_by_spacing(rows, 15.0)
        with self.assertRaises(FileExistsError):
            trim_dataset.apply_trim(
                self.ds, keep2, "thin again", None,
                trim_dir="trimmed_frames_for_density", kind="density")
        self.assertEqual(snapshot_tree(self.ds), before)

    def test_short_intrinsics_fails_preflight_without_mutation(self):
        rows, fields = trim_dataset.read_csv(self.ds / "intrinsics.csv")
        trim_dataset.write_csv(self.ds / "intrinsics.csv", rows[:-1], fields)
        before = snapshot_tree(self.ds)
        with self.assertRaisesRegex(ValueError, "intrinsics.csv has 11 rows"):
            trim_dataset.apply_trim(
                self.ds, [0, 1, 2], "tail", None,
                trim_dir="short-table", kind="range")
        self.assertEqual(snapshot_tree(self.ds), before)
        self.assertFalse((self.ds / "short-table").exists())
        self.assertFalse(
            (self.ds.parent / f".{self.ds.name}.trim_dataset.incomplete")
            .exists())

    def test_commit_failure_rolls_back_every_dataset_byte(self):
        before = snapshot_tree(self.ds)

        def fail_checksum(dataset):
            (dataset / checksums.CHECKSUM_FILE).write_text("partial\n")
            raise RuntimeError("injected checksum failure")

        with mock.patch.object(trim_dataset.checksums, "regenerate",
                               side_effect=fail_checksum):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                trim_dataset.apply_trim(
                    self.ds, [0, 1, 2], "tail", None,
                    trim_dir="rollback-test", kind="range")
        self.assertEqual(snapshot_tree(self.ds), before)
        self.assertFalse((self.ds / "rollback-test").exists())
        self.assertTrue(
            (self.ds.parent / f".{self.ds.name}.trim_dataset.incomplete")
            .is_dir())

    def test_trim_archives_old_timelapse_and_publishes_new_pair(self):
        old_reference = self.publish_timelapse(color=(10, 20, 30))

        def regenerate():
            self.publish_timelapse(color=(30, 20, 10))

        trim_dataset.apply_trim(
            self.ds, [0, 1, 2], "tail", None,
            trim_dir="timelapse-swap", kind="range",
            regenerate=regenerate)

        new_reference = timelapse.validate_completed(self.ds)
        archived = self.ds / "timelapse-swap" / "pre_trim_timelapse"
        archived_reference = artifact.open_artifact(
            archived,
            expected_kind=timelapse.REVIEW_KIND,
            expected_dataset=self.ds.name,
            expected_version=timelapse.REVIEW_VERSION)
        self.assertEqual(archived_reference, old_reference)
        self.assertNotEqual(new_reference, old_reference)
        self.assertFalse(
            timelapse.view_output_dir(self.ds).with_name(
                "timelapse.incomplete").exists())

    def test_timelapse_regeneration_failure_restores_every_dataset_byte(self):
        old_reference = self.publish_timelapse()
        before = snapshot_tree(self.ds)

        def fail_regeneration():
            def plot(dataset, lats, lons, times, dists, out):
                del dataset, lats, lons, times, dists
                out.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8)).save(out, format="PNG")

            with mock.patch.object(
                    timelapse, "stage_plot", side_effect=plot), \
                 mock.patch.object(
                     timelapse, "stage_video",
                     side_effect=RuntimeError("injected encoder failure")):
                timelapse.render(
                    self.ds, width=640, fps=12,
                    max_frames=100, skip_video=False)

        with self.assertRaisesRegex(RuntimeError, "injected encoder failure"):
            trim_dataset.apply_trim(
                self.ds, [0, 1, 2], "tail", None,
                trim_dir="timelapse-rollback", kind="range",
                regenerate=fail_regeneration)

        self.assertEqual(snapshot_tree(self.ds), before)
        self.assertEqual(timelapse.validate_completed(self.ds), old_reference)
        self.assertFalse((self.ds / "timelapse-rollback").exists())
        transaction = (
            self.ds.parent / f".{self.ds.name}.trim_dataset.incomplete")
        self.assertTrue(
            (transaction / "failed_timelapse" / "incomplete").is_dir())

    def test_arbitrary_intrinsics_columns_survive(self):
        keep, _ = self.density_trim()
        rows = read_rows(self.ds / "intrinsics.csv")
        self.assertEqual([row["future_source_column"] for row in rows],
                         [f"raw-{old}" for old in keep])

    def test_metadata_records_the_trim_and_both_trajectory_lengths(self):
        keep, _ = self.density_trim()
        meta = json.loads((self.ds / "pipeline_metadata.json").read_text())
        self.assertEqual(meta["num_images"], len(keep))
        self.assertEqual(meta["trajectory_km_before_trim"], 0.033)
        self.assertLessEqual(meta["trajectory_km"], 0.033)
        record = meta["trims"][-1]
        self.assertEqual(record["trim_kind"], "density")
        self.assertEqual(record["n_before"], 12)
        self.assertEqual(record["n_after"], len(keep))
        self.assertEqual(record["trim_dir"], "trimmed_frames_for_density")

    def test_trim_record_carries_provenance(self):
        self.density_trim()
        meta = json.loads((self.ds / "pipeline_metadata.json").read_text())
        record = meta["trims"][-1]
        self.assertIn("git_commit", record)
        self.assertIn("argv", record)
        note = json.loads((self.ds / "trimmed_frames_for_density"
                           / "trim_note.json").read_text())
        self.assertIn("git_commit", note)
        self.assertIn("argv", note)

    def test_many_runs_are_summarized_not_inlined(self):
        # A density trim on a dense collect produces hundreds of one-frame
        # runs; inlining them would bury the rest of pipeline_metadata.json.
        big = build_dataset(Path(self.tmp.name) / "big",
                            n=2 * trim_dataset.MAX_RECORDED_RANGES + 4)
        keep = list(range(0, 2 * trim_dataset.MAX_RECORDED_RANGES + 4, 2))
        trim_dataset.apply_trim(big, keep, "alternate", None,
                                trim_dir="d", kind="density")
        meta = json.loads((big / "pipeline_metadata.json").read_text())
        recorded = meta["trims"][-1]["kept_original_ranges"]
        self.assertIsInstance(recorded, str)
        self.assertIn("dropped_frames.csv", recorded)

    def test_few_runs_are_still_inlined(self):
        trim_dataset.apply_trim(self.ds, list(range(0, 6)), "tail", None,
                                trim_dir="trimmed_frames", kind="range")
        meta = json.loads((self.ds / "pipeline_metadata.json").read_text())
        self.assertEqual(meta["trims"][-1]["kept_original_ranges"], [[0, 6]])

    def test_sequence_position_still_carries_the_original_index(self):
        keep, _ = self.density_trim()
        rows = read_rows(self.ds / "extraction_log.csv")
        self.assertEqual([int(r["sequence_position"]) for r in rows], keep)

    def test_density_trim_preserves_the_original_along_track_distance(self):
        # dist_m normally rides a smoothed track. Re-deriving it from raw
        # positions would let a trim that removes no track report a
        # *different* trajectory length, so a density trim carries the column
        # forward.
        before, _ = trim_dataset.read_csv(self.ds / "frames_gps.csv")
        keep, _ = self.density_trim()
        after = read_rows(self.ds / "frames_gps.csv")
        base = float(before[keep[0]]["dist_m"])
        for new_idx, old in enumerate(keep):
            self.assertAlmostEqual(float(after[new_idx]["dist_m"]),
                                   float(before[old]["dist_m"]) - base,
                                   places=1)

    def test_range_trim_still_rederives_distance_from_positions(self):
        before, _ = trim_dataset.read_csv(self.ds / "frames_gps.csv")
        trim_dataset.apply_trim(self.ds, [0, 1, 2], "tail", None,
                                trim_dir="trimmed_frames", kind="range")
        after = read_rows(self.ds / "frames_gps.csv")
        expected = geo.haversine_m(
            float(before[0]["latitude"]), float(before[0]["longitude"]),
            float(before[2]["latitude"]), float(before[2]["longitude"]))
        self.assertAlmostEqual(float(after[-1]["dist_m"]), expected, places=0)

    def test_head_cut_keeps_video_t_s_addressing_the_untrimmed_video(self):
        # The regression that produced charles_river_20260727's bad tracking:
        # video_t_s was rebased to zero at the new first frame, but the source
        # video is not trimmed, so every kept frame then addressed content
        # earlier than itself by the length of the head cut.
        before, _ = trim_dataset.read_csv(self.ds / "frames_gps.csv")
        keep = [3, 4, 5]
        trim_dataset.apply_trim(self.ds, keep, "head cut", None,
                                trim_dir="trimmed_frames", kind="range")
        after = read_rows(self.ds / "frames_gps.csv")
        for new_idx, old in enumerate(keep):
            for column in ("video_t_s", "sensor_elapsed_s"):
                self.assertEqual(float(after[new_idx][column]),
                                 float(before[old][column]),
                                 f"{column} must survive the trim verbatim")
        self.assertGreater(float(after[0]["video_t_s"]), 0.0)

    def test_pre_trim_csvs_are_preserved_for_restore(self):
        self.density_trim()
        backups = sorted((self.ds / "trimmed_frames_for_density")
                         .glob("*.frames_gps.csv"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(len(read_rows(backups[0])), 12)


SEAMS_RECORD = {
    "median_dt_s": 2.0,
    "gap_multiple": 10.0,
    "min_gap_s": 20.0,
    "n_seams": 3,
    "seams": [
        {"after_idx": 1, "kind": "sequence", "dt_s": 60.0, "step_m": 5.0,
         "implied_speed_mps": 0.08},
        {"after_idx": 5, "kind": "gap", "dt_s": 120.0, "step_m": 300.0,
         "implied_speed_mps": 2.5},
        {"after_idx": 9, "kind": "sequence", "dt_s": 45.0, "step_m": 12.0,
         "implied_speed_mps": 0.27},
    ],
}


class SeamRebaseTest(unittest.TestCase):
    """recording_seams are keyed on frame indices; a renumbering trim that
    leaves them alone produces a record that is well-formed and wrong — the
    silent staleness this port fixes."""

    def test_pure_rebase_when_flanking_frames_survive(self):
        # Drop frames 2:4; seams after 1 (flanks 1,2... frame 2 dropped),
        # after 5 and after 9 survive with shifted indices.
        keep = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]
        out = trim_dataset.rebase_seams(SEAMS_RECORD, keep, "range")
        by_original = {s["original_after_idx"]: s for s in out["seams"]}
        self.assertEqual(by_original[1]["after_idx"], 1)
        self.assertEqual(by_original[5]["after_idx"], 3)
        self.assertEqual(by_original[9]["after_idx"], 7)
        self.assertEqual(out["n_seams"], 3)
        # Seam after original 5: flanks 5 and 6 both kept -> metrics valid.
        self.assertNotIn("metrics_stale", by_original[5])
        # Seam after original 1: original flank 2 was dropped -> the boundary
        # now spans farther than the seam measured.
        self.assertTrue(by_original[1]["metrics_stale"])

    def test_seams_off_the_kept_ends_are_dropped(self):
        keep = [3, 4, 5, 6, 7, 8]      # head cut past seam@1, tail cut past 9
        out = trim_dataset.rebase_seams(SEAMS_RECORD, keep, "range")
        self.assertEqual([s["original_after_idx"] for s in out["seams"]], [5])
        self.assertEqual(out["seams"][0]["after_idx"], 2)
        self.assertEqual(out["rebased_by_trim"]["n_seams_dropped"], 2)

    def test_seams_collapsing_onto_one_boundary_are_deduped(self):
        # Dropping 2..9 leaves one boundary (between original 1 and 10);
        # both surviving seams map onto it, and one seam is enough to mark a
        # break.
        keep = [0, 1, 10, 11]
        out = trim_dataset.rebase_seams(SEAMS_RECORD, keep, "range")
        self.assertEqual(len(out["seams"]), 1)
        self.assertEqual(out["seams"][0]["after_idx"], 1)

    def test_apply_trim_rebases_metadata_block_and_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = build_dataset(Path(tmp) / "ds",
                               metadata_extra={
                                   "recording_seams": dict(SEAMS_RECORD)})
            sidecar = ds / trim_dataset.SEAMS_SIDECAR
            sidecar.parent.mkdir()
            sidecar.write_text(json.dumps(SEAMS_RECORD))
            keep = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]
            info = trim_dataset.apply_trim(ds, keep, "cut 2:4", None,
                                           trim_dir="trimmed_frames",
                                           kind="range")
            self.assertEqual(info["n_seam_records_rebased"], 2)
            meta = json.loads((ds / "pipeline_metadata.json").read_text())
            for record in (meta["recording_seams"],
                           json.loads(sidecar.read_text())):
                self.assertIn("rebased_by_trim", record)
                self.assertEqual(
                    [s["after_idx"] for s in record["seams"]], [1, 3, 7])
                self.assertEqual(record["n_seams"], 3)

    def test_density_trim_also_rebases_seams(self):
        # A density trim keeps the geometry but still renumbers frames, so
        # the index-keyed seams record must move with it (unlike the mount
        # offset, which is index-free and correctly left alone).
        with tempfile.TemporaryDirectory() as tmp:
            ds = build_dataset(Path(tmp) / "ds",
                               metadata_extra={
                                   "recording_seams": dict(SEAMS_RECORD)})
            rows, _ = trim_dataset.read_csv(ds / "frames_gps.csv")
            keep = trim_dataset.keep_by_spacing(rows, 9.0)
            trim_dataset.apply_trim(ds, keep, "dense", None,
                                    trim_dir="d", kind="density")
            meta = json.loads((ds / "pipeline_metadata.json").read_text())
            self.assertIn("rebased_by_trim", meta["recording_seams"])
            for seam in meta["recording_seams"]["seams"]:
                self.assertLess(seam["after_idx"], len(keep) - 1)


class ChecksumTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ds = build_dataset(Path(self.tmp.name) / "ds")
        self.addCleanup(self.tmp.cleanup)

    def test_manifest_matches_sha256sum_format(self):
        lines = (self.ds / checksums.CHECKSUM_FILE).read_text().splitlines()
        digest, _, rel = lines[0].partition("  ")
        self.assertEqual(len(digest), 64)
        self.assertTrue(rel.startswith("./"))
        on_disk = hashlib.sha256((self.ds / rel[2:]).read_bytes()).hexdigest()
        self.assertEqual(digest, on_disk)

    def test_manifest_excludes_the_symlink_and_itself(self):
        text = (self.ds / checksums.CHECKSUM_FILE).read_text()
        self.assertNotIn("./panorama/", text)
        self.assertNotIn(checksums.CHECKSUM_FILE, text)
        self.assertIn("./frames/", text)

    def test_manifest_excludes_derived_caches_and_manifests_dir(self):
        cache = self.ds / "landmarks" / "catalog_cache"
        cache.mkdir(parents=True)
        (cache / "catalog_abc.pkl").write_bytes(b"derived")
        manifests = self.ds / "_manifests"
        manifests.mkdir()
        (manifests / "recording_seams.json").write_text("{}")
        checksums.regenerate(self.ds)
        text = (self.ds / checksums.CHECKSUM_FILE).read_text()
        self.assertNotIn("catalog_cache", text)
        # _manifests is the derived triage lane, rewritten by every tool run;
        # covering it would make each run look like corruption.
        self.assertNotIn("_manifests", text)

    def test_trim_refreshes_the_manifest_over_renamed_frames(self):
        stale = (self.ds / checksums.CHECKSUM_FILE).read_text()
        rows, _ = trim_dataset.read_csv(self.ds / "frames_gps.csv")
        keep = trim_dataset.keep_by_spacing(rows, 9.0)
        trim_dataset.apply_trim(self.ds, keep, "dense", None,
                                trim_dir="d", kind="density")
        checksums.regenerate(self.ds)
        fresh = (self.ds / checksums.CHECKSUM_FILE).read_text()
        self.assertNotEqual(stale, fresh)
        for line in fresh.splitlines():
            digest, _, rel = line.partition("  ")
            path = self.ds / rel[2:]
            self.assertTrue(path.exists(), rel)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             digest, rel)

    def test_absent_manifest_is_left_absent(self):
        root = build_dataset(Path(self.tmp.name) / "nosums",
                             with_checksums=False)
        self.assertIsNone(checksums.regenerate(root))
        self.assertFalse((root / checksums.CHECKSUM_FILE).exists())


if __name__ == "__main__":
    unittest.main()
