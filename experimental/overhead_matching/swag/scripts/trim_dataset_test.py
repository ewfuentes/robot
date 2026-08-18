import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.scripts import trim_dataset


def gps(dists):
    return [{"dist_m": str(d)} for d in dists]


class KeepBySpacingTest(unittest.TestCase):
    def test_uniform_source_lands_on_the_target_spacing(self):
        # A 3.3 m source cannot hit 10 m exactly; 9.9 is the nearest it can do.
        # The point is that it undershoots rather than overshooting to 13.2.
        dists = [3.3 * i for i in range(200)]
        keep = trim_dataset.keep_by_spacing(gps(dists), 10)
        gaps = [dists[b] - dists[a] for a, b in zip(keep, keep[1:])]
        self.assertTrue(all(abs(g - 9.9) < 1e-6 for g in gaps), gaps[:5])

    def test_first_past_target_would_overshoot(self):
        # Guards the choice of "nearest" over "first at or past": the naive rule
        # gives 13.2 m for a 10 m request on a 3.3 m collect, a 32% error that
        # compounds into a third fewer frames than asked for.
        dists = [3.3 * i for i in range(200)]
        keep = trim_dataset.keep_by_spacing(gps(dists), 10)
        realized = (dists[keep[-1]] - dists[keep[0]]) / (len(keep) - 1)
        self.assertLess(abs(realized - 10.0), abs(13.2 - 10.0))

    def test_stationary_run_collapses_to_one_frame(self):
        # 50 frames at a dead stop: distance-based selection must not keep them
        # all just because they are 50 separate rows.
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


DEG_PER_M_LAT = 1.0 / 111195.0   # 1 deg latitude on the R_EARTH_M sphere


def build_dataset(root: Path, n=12, step=3.0, with_checksums=True):
    """Minimum dataset satisfying the tables trim_dataset rewrites.

    Positions and `dist_m` are kept consistent with each other: selection reads
    `dist_m` but `rebuild_gps_rows` re-derives it by haversine over the kept
    positions, so a fixture where the two disagree measures the disagreement
    rather than the code.
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
                          "height": "3840", "heading_deg": "0"})
        map_rows.append({"pano_id": f"f{i:04d}", "lat": f"{lat:.7f}",
                         "lon": f"{lon:.7f}", "filename": name})
    for name, rows in (("frames_gps.csv", gps_rows),
                       ("extraction_log.csv", log_rows),
                       ("intrinsics.csv", intr_rows),
                       ("pano_id_mapping.csv", map_rows)):
        trim_dataset.write_csv(root / name, rows, list(rows[0]))
    (root / "pipeline_metadata.json").write_text(json.dumps({
        "num_images": n, "projection": "equirectangular",
        "trajectory_km": round(step * (n - 1) / 1000.0, 3),
        "mount_offset": {"mount_offset_deg": 214.0, "status": "prior"},
    }, indent=2))
    if with_checksums:
        (root / trim_dataset.CHECKSUM_FILE).write_text("")
        trim_dataset.regenerate_checksums(root)
    return root


def read_rows(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


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

    def test_tables_stay_consistent_after_a_density_trim(self):
        keep, _ = self.density_trim()
        n = len(keep)
        for name in trim_dataset.CSV_NAMES:
            self.assertEqual(len(read_rows(self.ds / name)), n, name)
        images = sorted(p.name for p in (self.ds / "frames").glob("*.jpg"))
        self.assertEqual(len(images), n)
        gps_rows = read_rows(self.ds / "frames_gps.csv")
        self.assertEqual([int(r["idx"]) for r in gps_rows], list(range(n)))
        # The join key the landmark_filtering ingest relies on.
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

    def test_dropped_csv_appends_across_successive_trims(self):
        self.density_trim(spacing=6.0)
        first = read_rows(self.ds / "trimmed_frames_for_density"
                          / "dropped_frames.csv")
        rows, _ = trim_dataset.read_csv(self.ds / "frames_gps.csv")
        keep2 = trim_dataset.keep_by_spacing(rows, 15.0)
        trim_dataset.apply_trim(self.ds, keep2, "thin again", None,
                                trim_dir="trimmed_frames_for_density",
                                kind="density")
        second = read_rows(self.ds / "trimmed_frames_for_density"
                           / "dropped_frames.csv")
        self.assertGreater(len(second), len(first))
        # Earlier rows survive verbatim, so the file stays a full history rather
        # than a record of only the most recent trim.
        self.assertEqual(second[:len(first)], first)
        self.assertEqual({r["reason"] for r in second[len(first):]},
                         {"thin again"})

    def test_density_trim_leaves_mount_offset_alone(self):
        self.density_trim()
        meta = json.loads((self.ds / "pipeline_metadata.json").read_text())
        self.assertNotIn("stale_after_trim", meta["mount_offset"])
        self.assertEqual(meta["mount_offset"]["mount_offset_deg"], 214.0)

    def test_range_trim_still_marks_mount_offset_stale(self):
        trim_dataset.apply_trim(self.ds, list(range(0, 6)), "bad tail", None,
                                trim_dir="trimmed_frames", kind="range")
        meta = json.loads((self.ds / "pipeline_metadata.json").read_text())
        self.assertTrue(meta["mount_offset"]["stale_after_trim"])
        self.assertFalse(meta["mount_offset"]["self_consistent"])

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

    def test_many_runs_are_summarized_not_inlined(self):
        # A density trim on a dense collect produces hundreds of one-frame runs;
        # inlining them would bury the rest of pipeline_metadata.json.
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
        # positions would let a trim that removes no track report a *different*
        # trajectory length, so a density trim carries the column forward.
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
        expected = trim_dataset.haversine_m(
            float(before[0]["latitude"]), float(before[0]["longitude"]),
            float(before[2]["latitude"]), float(before[2]["longitude"]))
        self.assertAlmostEqual(float(after[-1]["dist_m"]), expected, places=0)

    def test_pre_trim_csvs_are_preserved_for_restore(self):
        self.density_trim()
        backups = sorted((self.ds / "trimmed_frames_for_density")
                         .glob("*.frames_gps.csv"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(len(read_rows(backups[0])), 12)


class ChecksumTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ds = build_dataset(Path(self.tmp.name) / "ds")
        self.addCleanup(self.tmp.cleanup)

    def test_manifest_matches_sha256sum_format(self):
        lines = (self.ds / trim_dataset.CHECKSUM_FILE).read_text().splitlines()
        digest, _, rel = lines[0].partition("  ")
        self.assertEqual(len(digest), 64)
        self.assertTrue(rel.startswith("./"))
        on_disk = hashlib.sha256((self.ds / rel[2:]).read_bytes()).hexdigest()
        self.assertEqual(digest, on_disk)

    def test_manifest_excludes_the_symlink_and_itself(self):
        text = (self.ds / trim_dataset.CHECKSUM_FILE).read_text()
        self.assertNotIn("./panorama/", text)
        self.assertNotIn(trim_dataset.CHECKSUM_FILE, text)
        self.assertIn("./frames/", text)

    def test_manifest_excludes_derived_caches(self):
        cache = self.ds / "landmarks" / "catalog_cache"
        cache.mkdir(parents=True)
        (cache / "catalog_abc.pkl").write_bytes(b"derived")
        trim_dataset.regenerate_checksums(self.ds)
        self.assertNotIn(
            "catalog_cache",
            (self.ds / trim_dataset.CHECKSUM_FILE).read_text())

    def test_trim_refreshes_the_manifest_over_renamed_frames(self):
        stale = (self.ds / trim_dataset.CHECKSUM_FILE).read_text()
        rows, _ = trim_dataset.read_csv(self.ds / "frames_gps.csv")
        keep = trim_dataset.keep_by_spacing(rows, 9.0)
        trim_dataset.apply_trim(self.ds, keep, "dense", None,
                                trim_dir="d", kind="density")
        trim_dataset.regenerate_checksums(self.ds)
        fresh = (self.ds / trim_dataset.CHECKSUM_FILE).read_text()
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
        self.assertIsNone(trim_dataset.regenerate_checksums(root))
        self.assertFalse((root / trim_dataset.CHECKSUM_FILE).exists())


if __name__ == "__main__":
    unittest.main()
