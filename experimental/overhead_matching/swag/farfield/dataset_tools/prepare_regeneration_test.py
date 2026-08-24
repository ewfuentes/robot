import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield.dataset_tools import (
    prepare_regeneration as prep,
)


def _dataset(root: Path, name: str) -> Path:
    dataset = root / "datasets" / name
    frames = dataset / "frames"
    frames.mkdir(parents=True)
    (frames / "f0000,42.0,-71.0,.jpg").write_bytes(
        f"image:{name}".encode())
    (dataset / "panorama").symlink_to("frames", target_is_directory=True)
    (dataset / "pipeline_metadata.json").write_text(
        json.dumps({"dataset": name}))
    (dataset / "frames_gps.csv").write_text("frame_idx\n0\n")
    return dataset


def _root(base: Path) -> Path:
    root = base / "farfield_matching"
    (root / "datasets").mkdir(parents=True)
    (root / "artifacts" / "catalogs").mkdir(parents=True)
    for name in (*prep.ACTIVE_DATASETS,
                 *prep.RETIRED_MAPILLARY_DATASETS):
        _dataset(root, name)
    london = root / "datasets" / "unvetted" / "london_thames"
    (london / "frames").mkdir(parents=True)
    (london / "frames" / "london.jpg").write_bytes(b"london")
    (london / "panorama").symlink_to("frames", target_is_directory=True)

    # One external compatibility link pins the extra-depth rewrite. The real
    # tree contains 28 links of this form; all panorama links are internal.
    catalog = (root / "artifacts" / "catalogs" / "folkestone_dover"
               / "catalog.feather")
    catalog.parent.mkdir(parents=True)
    catalog.write_bytes(b"catalog")
    landmarks = root / "datasets" / "folkestone_dover" / "landmarks"
    landmarks.mkdir()
    (landmarks / "v1.feather").symlink_to(
        "../../../artifacts/catalogs/folkestone_dover/catalog.feather")
    version = (root / "artifacts" / "frame_landmarks"
               / "folkestone_dover" / "v4")
    version.mkdir(parents=True)
    (version / "manifest.json").write_text('{"complete": true}\n')
    (version / "request_manifest.json").write_text('{"requests": 1}\n')
    (version / "payload.jsonl").write_text("large payload is not hashed\n")
    source_manifest = (root / "raw_material" / "mapillary_manifests"
                       / "folkestone_dover.json")
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text('{"sequence": "folkestone"}\n')
    return root


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = _root(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def plan(self):
        return prep.build_plan(
            self.root, created="2026-08-24T12:00:00+00:00",
            git_commit="a" * 40)

    def test_exact_reviewed_scope(self):
        plan = self.plan()
        self.assertEqual(
            [item["name"] for item in plan["retirements"]],
            list(prep.RETIRED_MAPILLARY_DATASETS))
        self.assertIn("mt_washington_auto_road",
                      prep.RETIRED_MAPILLARY_DATASETS)
        self.assertEqual(
            [item["name"] for item in plan["active_datasets"]],
            list(prep.ACTIVE_DATASETS))
        self.assertEqual(
            [item["name"] for item in plan["untouched_collections"]],
            ["unvetted"])
        self.assertNotIn(
            "london_thames",
            [item["name"] for item in plan["retirements"]])
        self.assertEqual(
            plan["retirement_collection"],
            "datasets/out_of_date_but_usable_mapillary_datasets")

    def test_plan_is_read_only(self):
        before = prep._tree_summary(prep._walk_tree(self.root))
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = prep.main(["--data_root", str(self.root)])
        after = prep._tree_summary(prep._walk_tree(self.root))
        self.assertEqual(result, 0)
        self.assertEqual(after, before)
        self.assertIn('"plan_digest"', stdout.getvalue())
        self.assertIn("nothing moved", stderr.getvalue())

    def test_external_link_is_planned_for_one_more_parent(self):
        item = next(item for item in self.plan()["retirements"]
                    if item["name"] == "folkestone_dover")
        link = next(link for link in item["links"]
                    if link["path"] == "landmarks/v1.feather")
        self.assertEqual(
            link["original_target"],
            "../../../artifacts/catalogs/folkestone_dover/catalog.feather")
        self.assertEqual(
            link["retired_target"],
            "../../../../artifacts/catalogs/folkestone_dover/catalog.feather")
        panorama = next(link for link in item["links"]
                        if link["path"] == "panorama")
        self.assertEqual(panorama["original_target"], "frames")
        self.assertEqual(panorama["retired_target"], "frames")

    def test_records_only_artifact_identity_and_source_manifest_hashes(self):
        item = next(item for item in self.plan()["retirements"]
                    if item["name"] == "folkestone_dover")
        lane = next(lane for lane in item["artifact_lanes"]
                    if lane["kind"] == "frame_landmarks")
        version = next(version for version in lane["versions"]
                       if version["version"] == "v4")
        paths = {record["path"]
                 for record in version["manifest_provenance_files"]}
        self.assertEqual(paths, {"manifest.json", "request_manifest.json"})
        self.assertNotIn("payload.jsonl", paths)
        self.assertEqual(len(version["manifest_provenance_digest"]), 64)
        source = item["mapillary_manifest"]
        self.assertEqual(
            source["path"],
            "raw_material/mapillary_manifests/folkestone_dover.json")
        self.assertEqual(len(source["sha256"]), 64)

    def test_missing_or_unexpected_dataset_fails_closed(self):
        missing = self.root / "datasets" / "seattle"
        missing.rename(self.root / "seattle-away")
        with self.assertRaisesRegex(prep.PreparationError, "missing=.*seattle"):
            self.plan()
        (self.root / "seattle-away").rename(missing)
        (self.root / "datasets" / "new_collect").mkdir()
        with self.assertRaisesRegex(prep.PreparationError,
                                    "unexpected=.*new_collect"):
            self.plan()

    def test_existing_destination_or_symlinked_source_fails_closed(self):
        destination = self.root / "datasets" / prep.RETIREMENT_DIRNAME
        destination.mkdir()
        with self.assertRaisesRegex(prep.PreparationError,
                                    "destination already exists"):
            self.plan()
        destination.rmdir()

        source = self.root / "datasets" / "seattle"
        held = self.root / "seattle-held"
        source.rename(held)
        source.symlink_to(held, target_is_directory=True)
        with self.assertRaisesRegex(prep.PreparationError,
                                    "top-level dataset entry.*symlink"):
            self.plan()

    def test_dangling_absolute_and_escaping_links_are_rejected(self):
        dataset = self.root / "datasets" / "seattle"
        for target, pattern in (("missing", "dangling"),
                                (str(dataset / "frames"), "absolute"),
                                ("../../../../etc", "escapes data root")):
            link = dataset / "bad-link"
            link.symlink_to(target)
            with self.assertRaisesRegex(prep.PreparationError, pattern):
                self.plan()
            link.unlink()

    def test_plan_digest_binds_every_field(self):
        plan = self.plan()
        prep._validate_plan(plan)
        plan["retirements"][0]["regular_file_bytes"] += 1
        with self.assertRaisesRegex(prep.PreparationError,
                                    "plan_digest does not match"):
            prep._validate_plan(plan)

    def test_saved_plan_rejects_duplicate_keys_and_nonfinite_numbers(self):
        saved = Path(self.temp.name) / "reviewed.json"
        for payload, pattern in (
                ('{"schema":"first","schema":"second"}', "duplicate"),
                ('{"value":NaN}', "non-finite")):
            saved.write_text(payload)
            with self.assertRaisesRegex(prep.PreparationError, pattern):
                prep._load_plan(saved)


class ApplyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = _root(Path(self.temp.name))
        self.plan = prep.build_plan(
            self.root, created="2026-08-24T12:00:00+00:00",
            git_commit="a" * 40)
        self.artifacts_before = prep._tree_summary(
            prep._walk_tree(self.root / "artifacts"))
        self.unvetted_before = prep._tree_summary(
            prep._walk_tree(self.root / "datasets" / "unvetted"))

    def tearDown(self):
        self.temp.cleanup()

    def apply(self, hook=None):
        return prep.apply_plan(
            self.plan, confirm_plan_digest=self.plan["plan_digest"],
            hook=hook)

    def test_apply_moves_only_allowlist_and_preserves_links(self):
        destination = self.apply()
        self.assertEqual(destination.name, prep.RETIREMENT_DIRNAME)
        for name in prep.RETIRED_MAPILLARY_DATASETS:
            self.assertFalse((self.root / "datasets" / name).exists())
            self.assertTrue((destination / name).is_dir())
        for name in prep.ACTIVE_DATASETS:
            self.assertTrue((self.root / "datasets" / name).is_dir())
        self.assertTrue(
            (self.root / "datasets" / "unvetted" / "london_thames").is_dir())

        retired = destination / "folkestone_dover"
        panorama = retired / "panorama"
        catalog = retired / "landmarks" / "v1.feather"
        self.assertEqual(os.readlink(panorama), "frames")
        self.assertTrue((panorama / "f0000,42.0,-71.0,.jpg").is_file())
        self.assertEqual(
            os.readlink(catalog),
            "../../../../artifacts/catalogs/folkestone_dover/catalog.feather")
        self.assertEqual(catalog.read_bytes(), b"catalog")

        manifest = json.loads(
            (destination / "retirement_manifest.json").read_text())
        self.assertEqual(manifest["schema"], prep.RETIREMENT_SCHEMA)
        self.assertEqual(manifest["plan_digest"], self.plan["plan_digest"])
        self.assertFalse(manifest["artifacts_moved"])
        note = (destination / "UNRETIRE.md").read_text()
        self.assertIn("unvetted/london_thames", note)
        self.assertIn("mt_washington_auto_road", note)
        self.assertTrue((destination / "transaction_journal.json").is_file())

        self.assertEqual(
            prep._tree_summary(prep._walk_tree(self.root / "artifacts")),
            self.artifacts_before)
        self.assertEqual(
            prep._tree_summary(
                prep._walk_tree(self.root / "datasets" / "unvetted")),
            self.unvetted_before)

    def test_stale_plan_is_rejected_before_staging(self):
        payload = (self.root / "datasets" / "seattle" / "frames"
                   / "f0000,42.0,-71.0,.jpg")
        payload.write_bytes(b"changed")
        with self.assertRaisesRegex(prep.PreparationError,
                                    "source changed after plan"):
            self.apply()
        self.assertFalse((self.root / "datasets" / prep.RETIREMENT_DIRNAME).exists())
        self.assertFalse(list((self.root / "datasets").glob(".out_of_date*")))

    def test_wrong_confirmation_is_rejected(self):
        with self.assertRaisesRegex(prep.PreparationError,
                                    "does not match the reviewed plan"):
            prep.apply_plan(self.plan, confirm_plan_digest="0" * 64)

    def test_mid_transaction_failure_rolls_everything_back(self):
        calls = 0

        def fail_after_three_moves(event, _name):
            nonlocal calls
            if event == "before_move":
                calls += 1
                if calls == 4:
                    raise OSError("simulated interruption")

        with self.assertRaisesRegex(OSError, "simulated interruption"):
            self.apply(hook=fail_after_three_moves)
        for name in prep.RETIRED_MAPILLARY_DATASETS:
            self.assertTrue((self.root / "datasets" / name).is_dir(), name)
        self.assertFalse((self.root / "datasets" / prep.RETIREMENT_DIRNAME).exists())
        self.assertFalse(list((self.root / "datasets").glob(
            f".{prep.RETIREMENT_DIRNAME}.incomplete-*")))
        link = (self.root / "datasets" / "folkestone_dover" / "landmarks"
                / "v1.feather")
        self.assertEqual(
            os.readlink(link),
            "../../../artifacts/catalogs/folkestone_dover/catalog.feather")
        journals = list((self.root / "datasets").glob(
            f".{prep.RETIREMENT_DIRNAME}.journal-*.json"))
        self.assertEqual(len(journals), 1)
        self.assertEqual(json.loads(journals[0].read_text())["status"],
                         "rolled_back")

    def test_failure_during_rebase_restores_original_link_text(self):
        def fail_before_publish(event, _name):
            if event == "before_publish":
                raise OSError("publish failed")

        with self.assertRaisesRegex(OSError, "publish failed"):
            self.apply(hook=fail_before_publish)
        link = (self.root / "datasets" / "folkestone_dover" / "landmarks"
                / "v1.feather")
        self.assertEqual(
            os.readlink(link),
            "../../../artifacts/catalogs/folkestone_dover/catalog.feather")
        self.assertEqual(link.read_bytes(), b"catalog")

    def test_kernel_rename_never_overwrites_a_racing_directory(self):
        source = Path(self.temp.name) / "rename-source"
        destination = Path(self.temp.name) / "rename-destination"
        source.mkdir()
        destination.mkdir()
        (source / "source").write_text("source")
        (destination / "destination").write_text("destination")
        with self.assertRaisesRegex(prep.PreparationError,
                                    "refusing to overwrite"):
            prep._rename_noreplace(source, destination)
        self.assertEqual((source / "source").read_text(), "source")
        self.assertEqual(
            (destination / "destination").read_text(), "destination")

    def test_link_restore_failure_returns_no_dataset_to_live_lane(self):
        prefix = self.plan["plan_digest"][:12]
        staging = (self.root / "datasets"
                   / f".{prep.RETIREMENT_DIRNAME}.incomplete-{prefix}")

        def corrupt_first_external_link(event, label):
            if event == "before_rebase" and label == (
                    "folkestone_dover/landmarks/v1.feather"):
                link = (staging / "folkestone_dover" / "landmarks"
                        / "v1.feather")
                link.unlink()
                link.symlink_to("unexpected-target")

        with self.assertRaisesRegex(
                prep.PreparationError, "no dataset was returned"):
            self.apply(hook=corrupt_first_external_link)
        for name in prep.RETIRED_MAPILLARY_DATASETS:
            self.assertFalse((self.root / "datasets" / name).exists(), name)
            self.assertTrue((staging / name).is_dir(), name)
        journals = list((self.root / "datasets").glob(
            f".{prep.RETIREMENT_DIRNAME}.journal-*.json"))
        self.assertEqual(len(journals), 1)
        self.assertEqual(
            json.loads(journals[0].read_text())["status"],
            "recovery_required")


if __name__ == "__main__":
    unittest.main()
