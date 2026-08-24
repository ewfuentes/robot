import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experimental.overhead_matching.swag.farfield import geometry
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    migrate_active_datasets as migration,
)


def _metadata(name: str) -> dict:
    metadata = {
        "dataset_name": name,
        "source": migration.EXPECTED_SOURCE_BY_DATASET[name],
        "raw_material": f"raw_material/{name}",
        "is_equirectangular": True,
        "north_aligned": False,
        "azimuth_convention": {
            "images_rotated": False,
            "frame": "camera (as captured)",
            "bearing_increases": "left_to_right",
            "heading_deg_is_bearing_of": "column_0",
            "formula": "azimuth_deg = heading_deg + col / width * 360",
            "heading_per_frame": "intrinsics.csv:heading_deg",
            "mount_offset_frame": "legacy-camera-course-frame",
        },
        "intrinsics_csv": "intrinsics.csv",
        "image_dir": "frames",
        "mount_offset": {
            "mount_offset_deg": 214.0,
            "status": "prior",
            "source": "legacy fit",
        },
        "heading_source": "gps_course_minus_mount_prior",
        "heading_reliable": False,
        "heading_note": "course-derived placeholder",
        "bbox": {"south": 42.0, "north": 42.1,
                 "west": -71.1, "east": -71.0},
        "unrelated_provenance": {"keep": "verbatim"},
    }
    if name in migration.BOSTON_SOURCE_VIDEOS:
        metadata["video"] = {
            "source_video": migration.BOSTON_SOURCE_VIDEOS[name],
        }
    else:
        metadata["video"] = {
            "source_video": f"raw_material/{name}/source.mp4",
            "retained": True,
        }
    if name == "pohang_canal_04":
        metadata["azimuth_convention"]["frame"] = (
            migration.POHANG_AZIMUTH_FRAME)
        metadata["pending"] = list(migration.POHANG_PENDING)
        metadata["post_ingest_fixups"] = [
            dict(migration.POHANG_POST_INGEST_FIXUPS[0])]
        metadata["elevation"] = {"source": "DEM", "meters": 3.0}
        metadata["note"] = "retain Pohang provenance note"
    return metadata


def _write_intrinsics(path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "idx", "pano_id", "projection", "width", "height",
            "hfov_deg", "vfov_deg", "heading_deg", "heading_reference",
            "heading_source",
        ], lineterminator="\n")
        writer.writeheader()
        for index in range(2):
            writer.writerow({
                "idx": index,
                "pano_id": f"f{index:04d}",
                "projection": "equirectangular",
                "width": 64,
                "height": 32,
                "hfov_deg": 360,
                "vfov_deg": 180,
                "heading_deg": 10 + index,
                "heading_reference": "column_0",
                "heading_source": "gps_course_placeholder",
            })


def _tree_snapshot(root: Path) -> list[tuple]:
    records = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            records.append((rel, "symlink", os.readlink(path)))
        elif path.is_dir():
            records.append((rel, "directory"))
        else:
            records.append((rel, "file", path.read_bytes()))
    return records


def _build_root(parent: Path) -> Path:
    root = parent / "farfield_matching"
    (root / "datasets").mkdir(parents=True)
    (root / "archive").mkdir()
    (root / "artifacts" / "catalogs").mkdir(parents=True)
    for index, name in enumerate(migration.ACTIVE_DATASETS):
        dataset = root / "datasets" / name
        frames = dataset / "frames"
        frames.mkdir(parents=True)
        (frames / "f0000,42.000000,-71.000000,.jpg").write_bytes(
            b"pixels-" + name.encode())
        (dataset / "panorama").symlink_to("frames")
        metadata = _metadata(name)
        (dataset / migration.METADATA_FILE).write_text(
            json.dumps(metadata, indent=2) + "\n")
        video_path = metadata["video"]["source_video"].split(" (")[0]
        source_video = root / video_path
        source_video.parent.mkdir(parents=True, exist_ok=True)
        source_video.write_bytes(b"video:" + name.encode())
        _write_intrinsics(dataset / migration.INTRINSICS_FILE)
        (dataset / "frames_gps.csv").write_text(
            "idx,latitude,longitude,dist_m,video_t_s\n"
            "0,42,-71,0,0\n")
        landmarks = dataset / migration.LANDMARKS_DIR
        (landmarks / "nested").mkdir(parents=True)
        (landmarks / "v1.feather").symlink_to(
            f"../../../artifacts/catalogs/{name}/v1.feather")
        (landmarks / "nested" / "v1_trimmed.feather").symlink_to(
            f"../../../../artifacts/catalogs/{name}/v1_trimmed.feather")
        catalog = root / "artifacts" / "catalogs" / name
        catalog.mkdir(parents=True)
        (catalog / "v1.feather").write_bytes(b"catalog:" + name.encode())
        (catalog / "v1_trimmed.feather").write_bytes(
            b"trimmed:" + name.encode())
        if index % 2 == 0:
            (dataset / migration.CHECKSUM_FILE).write_text(
                "stale  ./pipeline_metadata.json\n")
        if name in {
                "boston_harbor_leg2", "boston_harbor_leg3",
                "charles_river_20260727"}:
            (dataset / "checksums.sha256.pre_mount_offset_refresh").write_text(
                f"legacy checksum backup for {name}\n")

    # Explicitly out of scope.  Its bytes and directory placement must remain
    # unchanged across a successful migration.
    london = root / "datasets" / "unvetted" / "london_thames"
    london.mkdir(parents=True)
    (london / "sentinel").write_text("untouched")
    retired = root / "datasets" / "mt_washington_auto_road"
    retired.mkdir()
    (retired / "sentinel").write_text("untouched")
    return root


class PlanningTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = _build_root(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_default_cli_is_report_only_and_targets_exact_active_set(self):
        before = _tree_snapshot(self.root)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            self.assertEqual(migration.main([
                "--data_root", str(self.root),
            ]), 0)
        plan = json.loads(stdout.getvalue())
        self.assertEqual(plan["active_datasets"],
                         list(migration.ACTIVE_DATASETS))
        self.assertEqual([item["dataset"] for item in plan["datasets"]],
                         list(migration.ACTIVE_DATASETS))
        self.assertIn(plan["plan_digest"], stderr.getvalue())
        self.assertEqual(_tree_snapshot(self.root), before)

    def test_plan_reports_only_explicit_heading_and_metadata_changes(self):
        plan = migration.build_plan(self.root)
        item = plan["datasets"][0]
        changes = item["changes"]
        self.assertEqual(
            set(changes["metadata"]["removed_authority_paths"]), {
                "mount_offset", "heading_source", "heading_reliable",
                "heading_note", "azimuth_convention.heading_deg_is_bearing_of",
                "azimuth_convention.formula",
                "azimuth_convention.heading_per_frame",
                "azimuth_convention.mount_offset_frame",
            })
        self.assertEqual(
            changes["metadata"]["set"][
                "azimuth_convention.camera_frame"], geometry.CAMERA_FRAME)
        self.assertNotIn(
            "azimuth_convention.frame", changes["metadata"]["set"])
        self.assertNotIn(
            "azimuth_convention.bearing_increases",
            changes["metadata"]["set"])
        self.assertEqual(changes["metadata"]["preserved"], {
            "azimuth_convention.frame": migration.SELF_COLLECT_AZIMUTH_FRAME,
            "azimuth_convention.bearing_increases":
                migration.EXPECTED_BEARING_INCREASES,
        })
        self.assertEqual(
            changes["intrinsics"]["heading_shape_columns"],
            list(migration.HEADING_SHAPE_FIELDS))
        self.assertEqual(
            changes["intrinsics"][
                "nonempty_heading_values_archived_then_cleared"][
                    "heading_deg"], 2)

    def test_exact_boston_video_correction_is_content_bound(self):
        name = "boston_harbor_leg1"
        source = json.dumps(_metadata(name)).encode()
        migrated, changes = migration._migrate_metadata(source, name)
        metadata = json.loads(migrated)
        expected = migration.BOSTON_SOURCE_VIDEOS[name].removesuffix(
            migration.BOSTON_FALSE_NOT_RETAINED_SUFFIX)
        self.assertEqual(metadata["video"], {
            "source_video": expected,
            "retained": True,
        })
        self.assertEqual(changes["set"]["video.retained"], True)

        unexpected = _metadata(name)
        unexpected["video"]["source_video"] = expected
        with self.assertRaisesRegex(migration.MigrationError,
                                    "unexpected video.source_video"):
            migration._migrate_metadata(
                json.dumps(unexpected).encode(), name)

    def test_pohang_removes_only_exact_stale_status_and_preserves_context(self):
        name = "pohang_canal_04"
        source = _metadata(name)
        migrated, changes = migration._migrate_metadata(
            json.dumps(source).encode(), name)
        metadata = json.loads(migrated)
        self.assertNotIn("pending", metadata)
        self.assertNotIn("post_ingest_fixups", metadata)
        self.assertEqual(
            metadata["azimuth_convention"]["frame"],
            migration.POHANG_AZIMUTH_FRAME)
        self.assertEqual(metadata["azimuth_convention"]["bearing_increases"],
                         "left_to_right")
        self.assertEqual(metadata["elevation"], source["elevation"])
        self.assertEqual(metadata["note"], source["note"])
        self.assertEqual(changes["preserved"]["azimuth_convention.frame"],
                         source["azimuth_convention"]["frame"])

        changed = _metadata(name)
        changed["pending"] = changed["pending"][:-1]
        with self.assertRaisesRegex(migration.MigrationError,
                                    "pending differs"):
            migration._migrate_metadata(json.dumps(changed).encode(), name)

    def test_wrong_pre_migration_frame_or_bearing_is_never_relabelled(self):
        cases = (
            ("charles_river_20260727", "frame", "vehicle forward",
             "azimuth_convention.frame"),
            ("charles_river_20260727", "bearing_increases", "right_to_left",
             "azimuth_convention.bearing_increases"),
            ("pohang_canal_04", "frame", "camera (as captured)",
             "azimuth_convention.frame"),
            ("pohang_canal_04", "bearing_increases", "right_to_left",
             "azimuth_convention.bearing_increases"),
        )
        for name, field, value, pattern in cases:
            with self.subTest(dataset=name, field=field):
                metadata = _metadata(name)
                metadata["azimuth_convention"][field] = value
                with self.assertRaisesRegex(
                        migration.MigrationError, pattern):
                    migration._migrate_metadata(
                        json.dumps(metadata).encode(), name)

    def test_source_video_and_landmark_targets_are_bound_by_content(self):
        plan = migration.build_plan(self.root)
        item = plan["datasets"][0]
        video = item["source"]["source_video"]
        self.assertEqual(video["path"],
                         migration.BOSTON_SOURCE_VIDEOS[item["dataset"]]
                         .removesuffix(
                             migration.BOSTON_FALSE_NOT_RETAINED_SUFFIX))
        self.assertEqual(len(video["sha256"]), 64)
        link = next(record for record in item["source"][
            migration.LANDMARKS_DIR]["records"]
                    if record["path"] == "v1.feather")
        self.assertEqual(
            link["original_target"],
            f"../../../artifacts/catalogs/{item['dataset']}/v1.feather")
        self.assertEqual(
            link["archive_target"],
            f"../../../../../../artifacts/catalogs/{item['dataset']}/v1.feather")
        self.assertEqual(len(link["target_identity"]["sha256"]), 64)

    def test_missing_or_symlinked_source_video_is_refused(self):
        name = migration.ACTIVE_DATASETS[3]
        path = self.root / f"raw_material/{name}/source.mp4"
        path.unlink()
        with self.assertRaisesRegex(migration.MigrationError,
                                    "source video is missing"):
            migration.build_plan(self.root)
        path.symlink_to(self.root / "datasets" / name / "frames_gps.csv")
        with self.assertRaisesRegex(migration.MigrationError,
                                    "regular non-symlink"):
            migration.build_plan(self.root)

    def test_metadata_rejects_duplicate_keys_and_nonfinite_numbers(self):
        name = migration.ACTIVE_DATASETS[0]
        for payload, pattern in (
                (f'{{"dataset_name":"{name}","dataset_name":"x"}}',
                 "duplicate"),
                (f'{{"dataset_name":"{name}","value":NaN}}',
                 "non-finite")):
            with self.assertRaisesRegex(migration.MigrationError, pattern):
                migration._parse_metadata(payload.encode(), name)

    def test_missing_active_dataset_is_not_silently_skipped(self):
        missing = self.root / "datasets" / migration.ACTIVE_DATASETS[-1]
        renamed = missing.with_name(missing.name + ".missing")
        missing.rename(renamed)
        with self.assertRaisesRegex(migration.MigrationError,
                                    "active dataset"):
            migration.build_plan(self.root)

    def test_landmarks_real_content_is_refused(self):
        name = migration.ACTIVE_DATASETS[0]
        (self.root / "datasets" / name / migration.LANDMARKS_DIR
         / "not-a-link.feather").write_bytes(b"valuable")
        with self.assertRaisesRegex(migration.MigrationError,
                                    "refusing real entry"):
            migration.build_plan(self.root)

    def test_only_exact_regular_checksum_backup_prefix_is_discovered(self):
        plan = migration.build_plan(self.root)
        by_name = {item["dataset"]: item for item in plan["datasets"]}
        backup = "checksums.sha256.pre_mount_offset_refresh"
        self.assertEqual(
            list(by_name["boston_harbor_leg2"]["source"][
                "legacy_checksum_backups"]), [backup])
        self.assertEqual(
            by_name["boston_harbor_leg1"]["source"][
                "legacy_checksum_backups"], {})

        # Similar names outside the exact direct-child prefix remain ordinary
        # data and are not migration targets.
        dataset = self.root / "datasets" / "boston_harbor_leg1"
        (dataset / "checksums.sha25.not-the-prefix").write_text("keep")
        nested = dataset / "nested"
        nested.mkdir()
        (nested / backup).write_text("keep nested")
        plan = migration.build_plan(self.root)
        item = plan["datasets"][0]
        self.assertEqual(item["source"]["legacy_checksum_backups"], {})

    def test_nonregular_exact_prefix_backup_is_refused(self):
        dataset = self.root / "datasets" / "boston_harbor_leg1"
        (dataset / "checksums.sha256.unexpected").symlink_to("missing")
        with self.assertRaisesRegex(migration.MigrationError,
                                    "must be a regular file"):
            migration.build_plan(self.root)

    def test_unexpected_dataset_identity_or_source_is_refused(self):
        path = (self.root / "datasets" / migration.ACTIVE_DATASETS[0]
                / migration.METADATA_FILE)
        metadata = json.loads(path.read_text())
        metadata["source"] = "mapillary"
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(migration.MigrationError,
                                    "source must be exactly"):
            migration.build_plan(self.root)

    def test_pohang_requires_public_third_party_provenance(self):
        name = "pohang_canal_04"
        path = self.root / "datasets" / name / migration.METADATA_FILE
        metadata = json.loads(path.read_text())
        self.assertEqual(metadata["source"], "third_party_public")
        metadata["source"] = "self_collect"
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(
                migration.MigrationError, "third_party_public"):
            migration.build_plan(self.root)

    def test_plan_output_is_no_overwrite_and_outside_data_root(self):
        output = Path(self.temp.name) / "reviewed.json"
        plan = migration.build_plan(self.root)
        migration._write_plan_output(output, plan)
        with self.assertRaises(FileExistsError):
            migration._write_plan_output(output, plan)
        with self.assertRaisesRegex(migration.MigrationError,
                                    "outside the data root"):
            migration._write_plan_output(self.root / "plan.json", plan)

        linked_parent = Path(self.temp.name) / "linked-into-data"
        linked_parent.symlink_to(self.root / "datasets", target_is_directory=True)
        with self.assertRaisesRegex(migration.MigrationError,
                                    "outside the data root"):
            migration._write_plan_output(
                linked_parent / "reviewed.json", plan)


class ApplyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = _build_root(Path(self.temp.name))
        self.plan = migration.build_plan(self.root)
        self.originals = {
            name: {
                filename: ((self.root / "datasets" / name / filename).read_bytes()
                           if (self.root / "datasets" / name / filename).exists()
                           else None)
                for filename in (migration.METADATA_FILE,
                                 migration.INTRINSICS_FILE,
                                 migration.CHECKSUM_FILE)
            }
            for name in migration.ACTIVE_DATASETS
        }
        self.untouched_before = _tree_snapshot(
            self.root / "datasets" / "unvetted") + _tree_snapshot(
                self.root / "datasets" / "mt_washington_auto_road")

    def tearDown(self):
        self.temp.cleanup()

    def test_apply_archives_originals_and_publishes_contract(self):
        archive = migration.apply_reviewed_plan(self.plan)
        self.assertEqual(archive.name, self.plan["plan_digest"])
        self.assertFalse(archive.name.startswith("."))
        for name in migration.ACTIVE_DATASETS:
            dataset = self.root / "datasets" / name
            archived = archive / "datasets" / name
            self.assertFalse((dataset / migration.LANDMARKS_DIR).exists())
            self.assertTrue((archived / migration.LANDMARKS_DIR).is_dir())
            self.assertTrue((archived / migration.LANDMARKS_DIR
                             / "v1.feather").is_symlink())
            archived_link = (archived / migration.LANDMARKS_DIR
                             / "v1.feather")
            self.assertEqual(
                os.readlink(archived_link),
                f"../../../../../../artifacts/catalogs/{name}/v1.feather")
            self.assertEqual(
                archived_link.read_bytes(), b"catalog:" + name.encode())
            for filename in (migration.METADATA_FILE,
                             migration.INTRINSICS_FILE):
                self.assertEqual((archived / filename).read_bytes(),
                                 self.originals[name][filename])
            if self.originals[name][migration.CHECKSUM_FILE] is None:
                self.assertFalse(
                    (archived / migration.CHECKSUM_FILE).exists())
            else:
                self.assertEqual(
                    (archived / migration.CHECKSUM_FILE).read_bytes(),
                    self.originals[name][migration.CHECKSUM_FILE])

            metadata = json.loads(
                (dataset / migration.METADATA_FILE).read_text())
            self.assertEqual(
                metadata["azimuth_convention"]["camera_frame"],
                geometry.CAMERA_FRAME)
            self.assertEqual(metadata["intrinsics_csv"], "intrinsics.csv")
            self.assertEqual(metadata["unrelated_provenance"],
                             {"keep": "verbatim"})
            if name in migration.BOSTON_SOURCE_VIDEOS:
                self.assertTrue(metadata["video"]["retained"])
                self.assertNotIn("not retained",
                                 metadata["video"]["source_video"])
            if name == "pohang_canal_04":
                self.assertEqual(
                    metadata["azimuth_convention"]["frame"],
                    migration.POHANG_AZIMUTH_FRAME)
                self.assertNotIn("pending", metadata)
                self.assertNotIn("post_ingest_fixups", metadata)
                self.assertEqual(
                    metadata["note"], "retain Pohang provenance note")
            for field in migration.LEGACY_METADATA_AUTHORITY_FIELDS:
                self.assertNotIn(field, metadata)

            with (dataset / migration.INTRINSICS_FILE).open(newline="") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                for field in migration.HEADING_SHAPE_FIELDS:
                    self.assertEqual(row[field], "")
                for field in migration.LEGACY_INTRINSICS_HEADING_FIELDS:
                    self.assertNotIn(field, row)
            expected_checksum = migration._checksum_manifest(dataset, {
                migration.METADATA_FILE:
                    (dataset / migration.METADATA_FILE).read_bytes(),
                migration.INTRINSICS_FILE:
                    (dataset / migration.INTRINSICS_FILE).read_bytes(),
            })
            self.assertEqual((dataset / migration.CHECKSUM_FILE).read_bytes(),
                             expected_checksum)
            backup = "checksums.sha256.pre_mount_offset_refresh"
            had_backup = name in {
                "boston_harbor_leg2", "boston_harbor_leg3",
                "charles_river_20260727"}
            self.assertFalse((dataset / backup).exists())
            self.assertEqual((archived / backup).exists(), had_backup)
            if had_backup:
                self.assertEqual(
                    (archived / backup).read_text(),
                    f"legacy checksum backup for {name}\n")
            self.assertNotIn(backup, expected_checksum.decode("utf-8"))

        untouched_after = _tree_snapshot(
            self.root / "datasets" / "unvetted") + _tree_snapshot(
                self.root / "datasets" / "mt_washington_auto_road")
        self.assertEqual(untouched_after, self.untouched_before)

    def test_checksums_are_the_final_dataset_renames(self):
        renames = []
        original = migration._rename_no_overwrite

        def record(source, destination, journal, journal_path, fail_after):
            renames.append((source, destination))
            return original(source, destination, journal, journal_path,
                            fail_after)

        with mock.patch.object(migration, "_rename_no_overwrite",
                               side_effect=record):
            migration.apply_reviewed_plan(self.plan)
        final = renames[-len(migration.ACTIVE_DATASETS):]
        self.assertEqual(
            [destination.name for _, destination in final],
            [migration.CHECKSUM_FILE] * len(migration.ACTIVE_DATASETS))
        self.assertTrue(all("/.staged/" in str(source) for source, _ in final))

    def test_changed_source_after_review_is_refused_before_transaction(self):
        path = (self.root / "datasets" / migration.ACTIVE_DATASETS[0]
                / "frames_gps.csv")
        path.write_text(path.read_text() + "1,42,-71,1,1\n")
        with self.assertRaisesRegex(migration.MigrationError,
                                    "changed after this plan"):
            migration.apply_reviewed_plan(self.plan)
        parent = self.root / migration.ARCHIVE_PARENT
        self.assertFalse(parent.exists())

    def test_same_size_source_video_change_invalidates_reviewed_plan(self):
        item = self.plan["datasets"][0]
        video = self.root / item["source"]["source_video"]["path"]
        original = video.read_bytes()
        replacement = bytes(byte ^ 0x01 for byte in original)
        self.assertEqual(len(replacement), len(original))
        video.write_bytes(replacement)
        with self.assertRaisesRegex(migration.MigrationError,
                                    "changed after this plan"):
            migration.apply_reviewed_plan(self.plan)
        self.assertFalse((self.root / migration.ARCHIVE_PARENT).exists())

    def test_existing_archive_is_never_overwritten(self):
        final = (self.root / migration.ARCHIVE_PARENT
                 / self.plan["plan_digest"])
        final.mkdir(parents=True)
        (final / "sentinel").write_text("keep")
        with self.assertRaisesRegex(migration.MigrationError,
                                    "refusing to overwrite"):
            migration.apply_reviewed_plan(self.plan)
        self.assertEqual((final / "sentinel").read_text(), "keep")

    def test_failure_rolls_back_every_live_dataset_without_deleting_evidence(self):
        before = {
            name: _tree_snapshot(self.root / "datasets" / name)
            for name in migration.ACTIVE_DATASETS
        }
        with self.assertRaisesRegex(OSError, "injected"):
            migration.apply_reviewed_plan(
                self.plan, fail_after_renames=9)
        for name in migration.ACTIVE_DATASETS:
            self.assertEqual(
                _tree_snapshot(self.root / "datasets" / name), before[name])
        transaction = (self.root / migration.ARCHIVE_PARENT
                       / f".{self.plan['plan_digest']}.incomplete")
        self.assertTrue(transaction.is_dir())
        journal = json.loads((transaction / "transaction.json").read_text())
        self.assertEqual(journal["status"], "rolled_back")
        self.assertTrue((transaction / ".failed_new").is_dir())

    def test_failure_mid_landmark_rebase_restores_original_links(self):
        before = {
            name: _tree_snapshot(self.root / "datasets" / name)
            for name in migration.ACTIVE_DATASETS
        }
        with self.assertRaisesRegex(OSError, "landmark rebase"):
            migration.apply_reviewed_plan(
                self.plan, fail_after_rebases=1)
        for name in migration.ACTIVE_DATASETS:
            self.assertEqual(
                _tree_snapshot(self.root / "datasets" / name), before[name])
            for link in (self.root / "datasets" / name
                         / migration.LANDMARKS_DIR).rglob("*.feather"):
                self.assertTrue(link.is_symlink())
                self.assertTrue(link.resolve(strict=True).is_file())

    def test_explicit_rollback_recovers_an_interrupted_commit(self):
        before = {
            name: _tree_snapshot(self.root / "datasets" / name)
            for name in migration.ACTIVE_DATASETS
        }
        transaction, _ = migration._transaction_paths(
            self.root, self.plan["plan_digest"])
        transaction.parent.mkdir(parents=True)
        journal = migration._stage_transaction(self.plan, transaction)
        journal_path = transaction / "transaction.json"
        name = migration.ACTIVE_DATASETS[0]
        dataset = self.root / "datasets" / name
        archived = transaction / "datasets" / name
        staged = transaction / ".staged" / name
        migration._rename_no_overwrite(
            dataset / migration.METADATA_FILE,
            archived / migration.METADATA_FILE,
            journal, journal_path, None)
        migration._rename_no_overwrite(
            staged / migration.METADATA_FILE,
            dataset / migration.METADATA_FILE,
            journal, journal_path, None)

        recovered = migration.rollback_reviewed_plan(self.plan)
        self.assertEqual(recovered, transaction)
        for dataset_name in migration.ACTIVE_DATASETS:
            self.assertEqual(
                _tree_snapshot(self.root / "datasets" / dataset_name),
                before[dataset_name])
        self.assertEqual(
            json.loads((transaction / "transaction.json").read_text())[
                "status"], "rolled_back")

    def test_rollback_prevalidates_archive_before_live_overwrite(self):
        transaction, _ = migration._transaction_paths(
            self.root, self.plan["plan_digest"])
        transaction.parent.mkdir(parents=True)
        journal = migration._stage_transaction(self.plan, transaction)
        journal_path = transaction / "transaction.json"
        name = migration.ACTIVE_DATASETS[0]
        dataset = self.root / "datasets" / name
        archived = transaction / "datasets" / name
        staged = transaction / ".staged" / name
        migration._rename_no_overwrite(
            dataset / migration.METADATA_FILE,
            archived / migration.METADATA_FILE,
            journal, journal_path, None)
        migration._rename_no_overwrite(
            staged / migration.METADATA_FILE,
            dataset / migration.METADATA_FILE,
            journal, journal_path, None)
        live_before = (dataset / migration.METADATA_FILE).read_bytes()
        (archived / migration.METADATA_FILE).write_bytes(b"tampered archive")

        with self.assertRaisesRegex(
                migration.MigrationError, "prevalidation failed"):
            migration.rollback_reviewed_plan(self.plan)
        self.assertEqual(
            (dataset / migration.METADATA_FILE).read_bytes(), live_before)
        self.assertEqual(
            json.loads((transaction / "transaction.json").read_text())[
                "status"], "rollback_failed")

    def test_kernel_rename_never_overwrites_a_destination(self):
        source = Path(self.temp.name) / "rename-source"
        destination = Path(self.temp.name) / "rename-destination"
        source.write_bytes(b"source")
        destination.write_bytes(b"destination")
        with self.assertRaisesRegex(migration.MigrationError,
                                    "refusing to overwrite"):
            migration._rename_noreplace(source, destination)
        self.assertEqual(source.read_bytes(), b"source")
        self.assertEqual(destination.read_bytes(), b"destination")

    def test_tampered_or_unconfirmed_saved_plan_is_refused(self):
        saved = Path(self.temp.name) / "reviewed.json"
        saved.write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(migration.MigrationError,
                                    "confirm_plan_digest"):
            migration.load_reviewed_plan(saved, "0" * 64)
        tampered = dict(self.plan)
        tampered["archive_parent"] = "somewhere_else"
        saved.write_text(json.dumps(tampered))
        with self.assertRaisesRegex(migration.MigrationError,
                                    "digest mismatch"):
            migration.load_reviewed_plan(saved, self.plan["plan_digest"])

        for payload, pattern in (
                ('{"schema":"one","schema":"two"}', "duplicate"),
                ('{"value":Infinity}', "non-finite")):
            saved.write_text(payload)
            with self.assertRaisesRegex(migration.MigrationError, pattern):
                migration.load_reviewed_plan(saved, self.plan["plan_digest"])


if __name__ == "__main__":
    unittest.main()
