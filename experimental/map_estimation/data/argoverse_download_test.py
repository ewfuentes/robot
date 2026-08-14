import datetime
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experimental.map_estimation.data import argoverse_catalog as ac
from experimental.map_estimation.data import argoverse_download as ad
from experimental.map_estimation.data import argoverse_layout as al
from experimental.map_estimation.data import s5cmd


def stat(num_bytes: int, num_objects: int) -> ac.ItemStat:
    return ac.ItemStat(num_bytes=num_bytes, num_objects=num_objects)


LOG_A = "02678d04-cc9f-3148-9f95-1ba66347dff9"
LOG_B = "0b86f508-5df9-4a46-bc59-5b9536dbde9f"


def make_entry(log_id: str, *, annotations: bool = True, sweeps: int = 4) -> ac.LogEntry:
    items = {
        "map": stat(1_700_000, 3),
        "calibration": stat(9_916, 2),
        "poses": stat(164_650, 1),
        "lidar": stat(sweeps * 900_000, sweeps),
        "ring_front_center": stat(3 * 285_000, 3),
    }
    if annotations:
        items["annotations"] = stat(285_474, 1)
    return ac.LogEntry(log_id=log_id, city="PIT", items=items)


def make_catalog(entries, *, split="val") -> ac.Catalog:
    return ac.Catalog(
        schema_version=ac.SCHEMA_VERSION,
        dataset=al.Dataset.SENSOR,
        split=split,
        built_at=datetime.datetime(2026, 8, 13, tzinfo=datetime.timezone.utc),
        s5cmd_version="v2.3.0-test",
        strategy=ac.BuildStrategy.PER_LOG_DETAIL,
        logs=tuple(entries),
    )


def write_item(root: Path, request: al.Request, log_id: str, item: al._Item, count: int,
               size: int = 100) -> None:
    """Materialize `count` files for an item, so local_status has something to find."""
    path = al.local_path(request, log_id, item, root)
    if item.is_dir:
        path.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (path / f"{index}.bin").write_bytes(b"x" * size)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)


class TmpRootTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)


class LocalStatusTest(TmpRootTest):
    def setUp(self):
        super().setUp()
        self.request = al.SensorRequest(
            split=al.SensorSplit.VAL,
            items=(al.SensorItem.MAP, al.SensorItem.POSES, al.SensorItem.LIDAR),
        )
        self.entry = make_entry(LOG_A, sweeps=4)

    def _status(self, **kwargs):
        return ad.local_status(self.request, [self.entry], root=self.root, **kwargs)[0]

    def test_nothing_local_is_missing(self):
        status = self._status()
        self.assertEqual(status.state, ad.State.MISSING)
        self.assertEqual(len(status.missing()), 3)

    def test_complete_when_object_counts_match(self):
        write_item(self.root, self.request, LOG_A, al.SensorItem.MAP, 3)
        write_item(self.root, self.request, LOG_A, al.SensorItem.POSES, 1)
        write_item(self.root, self.request, LOG_A, al.SensorItem.LIDAR, 4)
        self.assertEqual(self._status().state, ad.State.COMPLETE)

    def test_partial_when_some_objects_are_absent(self):
        write_item(self.root, self.request, LOG_A, al.SensorItem.MAP, 3)
        write_item(self.root, self.request, LOG_A, al.SensorItem.LIDAR, 2)  # of 4
        status = self._status()
        self.assertEqual(status.state, ad.State.PARTIAL)
        lidar = next(item for item in status.items if item.item == "lidar")
        self.assertEqual(lidar.state, ad.State.PARTIAL)
        self.assertEqual(lidar.local.num_objects, 2)

    def test_verify_bytes_catches_a_truncated_file(self):
        """Object counts match but bytes do not -- only --verify_bytes should notice."""
        write_item(self.root, self.request, LOG_A, al.SensorItem.MAP, 3, size=1)
        write_item(self.root, self.request, LOG_A, al.SensorItem.POSES, 1, size=1)
        write_item(self.root, self.request, LOG_A, al.SensorItem.LIDAR, 4, size=1)
        self.assertEqual(self._status().state, ad.State.COMPLETE)
        self.assertEqual(self._status(verify_bytes=True).state, ad.State.PARTIAL)

    def test_an_unreadable_entry_does_not_abandon_the_rest_of_the_directory(self):
        """One bad file must skip that file, not under-count the whole item."""
        write_item(self.root, self.request, LOG_A, al.SensorItem.LIDAR, 4)
        real_scandir = ad.os.scandir

        class Exploding:
            """A DirEntry whose stat() fails, as with a permissions error or a live write."""

            def __init__(self, entry):
                self._entry = entry
                self.path = entry.path

            def is_dir(self, follow_symlinks=True):
                return self._entry.is_dir(follow_symlinks=follow_symlinks)

            def is_file(self, follow_symlinks=True):
                return self._entry.is_file(follow_symlinks=follow_symlinks)

            def stat(self, follow_symlinks=True):
                raise PermissionError("denied")

        def flaky_scandir(path):
            entries = list(real_scandir(path))
            return [Exploding(entries[0])] + entries[1:] if entries else entries

        with mock.patch.object(ad.os, "scandir", side_effect=flaky_scandir):
            status = self._status()
        lidar = next(item for item in status.items if item.item == "lidar")
        self.assertEqual(lidar.local.num_objects, 3, "the other 3 files must still be counted")

    def test_items_absent_remotely_are_not_present_not_missing(self):
        """Annotations on a log that has none must not read as an incomplete download."""
        request = al.SensorRequest(
            split=al.SensorSplit.VAL, items=(al.SensorItem.ANNOTATIONS,)
        )
        entry = make_entry(LOG_A, annotations=False)
        status = ad.local_status(request, [entry], root=self.root)[0]
        self.assertEqual(status.items[0].state, ad.State.NOT_PRESENT)
        self.assertEqual(status.state, ad.State.COMPLETE)
        self.assertEqual(status.missing(), ())


class PlanTest(TmpRootTest):
    def setUp(self):
        super().setUp()
        self.request = al.SensorRequest(
            split=al.SensorSplit.VAL,
            items=(al.SensorItem.MAP, al.SensorItem.LIDAR),
        )
        self.entries = [make_entry(LOG_A, sweeps=4), make_entry(LOG_B, sweeps=4)]

    def test_plan_covers_every_log_and_item(self):
        download_plan = ad.plan(self.request, self.entries, root=self.root)
        self.assertEqual(len(download_plan.transfers), 4)
        self.assertEqual(download_plan.num_logs, 2)
        self.assertEqual(
            download_plan.total_bytes, 2 * (1_700_000 + 4 * 900_000)
        )
        self.assertEqual(download_plan.total_objects, 2 * (3 + 4))

    def test_plan_is_offline(self):
        """plan() must never touch s5cmd; that is what makes --dry_run trustworthy."""
        with mock.patch.object(ad.s5cmd, "run_commands") as run, \
             mock.patch.object(ad.s5cmd, "list_objects") as lister:
            ad.plan(self.request, self.entries, root=self.root)
        run.assert_not_called()
        lister.assert_not_called()

    def test_complete_items_are_skipped_not_dropped(self):
        write_item(self.root, self.request, LOG_A, al.SensorItem.MAP, 3)
        download_plan = ad.plan(self.request, self.entries, root=self.root)
        self.assertEqual(len(download_plan.transfers), 3)
        self.assertEqual(download_plan.num_skipped(ad.SkipReason.ALREADY_COMPLETE), 1)

    def test_partial_items_are_re_planned(self):
        write_item(self.root, self.request, LOG_A, al.SensorItem.LIDAR, 2)  # of 4
        download_plan = ad.plan(self.request, self.entries, root=self.root)
        self.assertEqual(len(download_plan.transfers), 4)
        self.assertEqual(download_plan.num_skipped(), 0)

    def test_overwrite_replans_complete_items(self):
        write_item(self.root, self.request, LOG_A, al.SensorItem.MAP, 3)
        download_plan = ad.plan(self.request, self.entries, root=self.root, overwrite=True)
        self.assertEqual(len(download_plan.transfers), 4)
        self.assertEqual(download_plan.num_skipped(), 0)

    def test_items_absent_remotely_are_skipped(self):
        request = al.SensorRequest(
            split=al.SensorSplit.VAL,
            items=(al.SensorItem.MAP, al.SensorItem.ANNOTATIONS),
        )
        download_plan = ad.plan(request, [make_entry(LOG_A, annotations=False)], root=self.root)
        self.assertEqual(len(download_plan.transfers), 1)
        self.assertEqual(download_plan.num_skipped(ad.SkipReason.NOT_PRESENT), 1)

    def test_everything_local_yields_an_empty_plan(self):
        for log_id in (LOG_A, LOG_B):
            write_item(self.root, self.request, log_id, al.SensorItem.MAP, 3)
            write_item(self.root, self.request, log_id, al.SensorItem.LIDAR, 4)
        download_plan = ad.plan(self.request, self.entries, root=self.root)
        self.assertTrue(download_plan.is_empty)
        self.assertEqual(download_plan.num_skipped(ad.SkipReason.ALREADY_COMPLETE), 4)

    def test_directory_destinations_end_in_a_slash_in_the_command(self):
        """s5cmd needs a trailing slash to treat a local destination as a directory.

        pathlib strips it, so the slash has to be re-added when the command is formatted --
        this pins that behaviour down.
        """
        download_plan = ad.plan(self.request, self.entries, root=self.root)
        by_item = {t.item: t for t in download_plan.transfers}
        lidar = by_item["lidar"]
        self.assertTrue(lidar.is_dir)
        self.assertTrue(lidar.src.endswith("/*"))
        self.assertTrue(lidar.to_command().endswith("/sensors/lidar/'"))
        self.assertEqual(lidar.dst_dir, lidar.dst)

    def test_file_destinations_are_exact(self):
        request = al.SensorRequest(split=al.SensorSplit.VAL, items=(al.SensorItem.POSES,))
        transfer = ad.plan(request, self.entries, root=self.root).transfers[0]
        self.assertFalse(transfer.is_dir)
        self.assertTrue(str(transfer.dst).endswith("city_SE3_egovehicle.feather"))
        self.assertFalse(transfer.src.endswith("*"))
        self.assertTrue(transfer.to_command().endswith("city_SE3_egovehicle.feather'"))
        self.assertEqual(transfer.dst_dir, transfer.dst.parent)

    def test_commands_use_no_clobber_unless_overwriting(self):
        download_plan = ad.plan(self.request, self.entries, root=self.root)
        self.assertTrue(all(c.startswith("cp -n ") for c in download_plan.to_commands()))
        self.assertTrue(
            all(c.startswith("cp '") for c in download_plan.to_commands(overwrite=True))
        )

    def test_wrong_sized_files_drop_no_clobber_so_they_can_be_repaired(self):
        """`cp -n` would decline to replace a truncated file, making it unrepairable."""
        request = al.SensorRequest(split=al.SensorSplit.VAL, items=(al.SensorItem.POSES,))
        write_item(self.root, request, LOG_A, al.SensorItem.POSES, 1, size=1)  # truncated

        # Without --verify_bytes the object count matches, so nothing is flagged at all.
        relaxed = ad.plan(request, [self.entries[0]], root=self.root)
        self.assertTrue(relaxed.is_empty)

        strict = ad.plan(request, [self.entries[0]], root=self.root, verify_bytes=True)
        self.assertEqual(len(strict.transfers), 1)
        transfer = strict.transfers[0]
        self.assertTrue(transfer.force_overwrite)
        self.assertNotIn(" -n ", transfer.to_command())

    def test_merely_missing_files_keep_no_clobber_for_cheap_resume(self):
        request = al.SensorRequest(split=al.SensorSplit.VAL, items=(al.SensorItem.LIDAR,))
        write_item(self.root, request, LOG_A, al.SensorItem.LIDAR, 2)  # of 4, all intact
        download_plan = ad.plan(request, [self.entries[0]], root=self.root)
        transfer = download_plan.transfers[0]
        self.assertFalse(transfer.force_overwrite)
        self.assertIn(" -n ", transfer.to_command())

    def test_commands_quote_wildcards(self):
        download_plan = ad.plan(self.request, self.entries, root=self.root)
        lidar = next(c for c in download_plan.to_commands() if "lidar" in c)
        self.assertIn("'s3://", lidar)
        self.assertIn("/*'", lidar)


class FreeSpaceGuardTest(TmpRootTest):
    def setUp(self):
        super().setUp()
        self.request = al.SensorRequest(
            split=al.SensorSplit.VAL, items=(al.SensorItem.LIDAR,)
        )
        self.entries = [make_entry(LOG_A, sweeps=4)]

    def test_refuses_a_plan_that_would_not_fit(self):
        usage = mock.Mock(free=1000)
        with mock.patch.object(ad.shutil, "disk_usage", return_value=usage):
            with self.assertRaises(ad.InsufficientSpaceError) as ctx:
                ad.plan(self.request, self.entries, root=self.root)
        message = str(ctx.exception)
        self.assertIn("short by", message)
        self.assertIn("ignore_free_space", message)

    def test_ignore_free_space_proceeds(self):
        usage = mock.Mock(free=1000)
        with mock.patch.object(ad.shutil, "disk_usage", return_value=usage):
            download_plan = ad.plan(self.request, self.entries, root=self.root,
                                    ignore_free_space=True)
        self.assertEqual(len(download_plan.transfers), 1)

    def test_ample_space_is_fine_and_recorded(self):
        usage = mock.Mock(free=10**12)
        with mock.patch.object(ad.shutil, "disk_usage", return_value=usage):
            download_plan = ad.plan(self.request, self.entries, root=self.root)
        self.assertEqual(download_plan.free_bytes, 10**12)

    def test_nonexistent_root_walks_up_to_an_existing_parent(self):
        """The destination usually does not exist yet; that must not defeat the check."""
        deep = self.root / "a" / "b" / "c"
        with mock.patch.object(ad.shutil, "disk_usage",
                               return_value=mock.Mock(free=10**12)) as usage:
            ad.plan(self.request, self.entries, root=deep)
        self.assertEqual(usage.call_args.args[0], self.root)


class ExecuteTest(TmpRootTest):
    def setUp(self):
        super().setUp()
        self.request = al.SensorRequest(
            split=al.SensorSplit.VAL, items=(al.SensorItem.MAP, al.SensorItem.LIDAR)
        )
        self.entries = [make_entry(LOG_A, sweeps=4)]

    def test_empty_plan_does_not_invoke_s5cmd(self):
        empty = ad.DownloadPlan(spec="sensor/val", root=self.root, items=("map",),
                                transfers=())
        with mock.patch.object(ad.s5cmd, "run_commands") as run:
            result = ad.execute(empty)
        run.assert_not_called()
        self.assertTrue(result.ok)

    def test_destination_directories_are_created(self):
        download_plan = ad.plan(self.request, self.entries, root=self.root)
        with mock.patch.object(ad.s5cmd, "run_commands",
                               return_value=s5cmd.Result(returncode=0, num_commands=2,
                                                         elapsed_s=1.0)):
            ad.execute(download_plan)
        self.assertTrue(al.local_path(self.request, LOG_A, al.SensorItem.LIDAR,
                                      self.root).is_dir())

    def test_dry_run_creates_nothing(self):
        download_plan = ad.plan(self.request, self.entries, root=self.root)
        with mock.patch.object(ad.s5cmd, "run_commands",
                               return_value=s5cmd.Result(returncode=0, num_commands=2,
                                                         elapsed_s=0.0)):
            ad.execute(download_plan, dry_run=True)
        self.assertFalse(al.log_dir(self.request, LOG_A, self.root).exists())

    def test_batches_all_transfers_into_one_call(self):
        """One `s5cmd run` so --numworkers parallelizes across logs, not one process per copy."""
        download_plan = ad.plan(self.request, self.entries, root=self.root)
        with mock.patch.object(ad.s5cmd, "run_commands",
                               return_value=s5cmd.Result(returncode=0, num_commands=2,
                                                         elapsed_s=1.0)) as run:
            ad.execute(download_plan)
        run.assert_called_once()
        self.assertEqual(len(run.call_args.args[0]), 2)


class EnsureLogsTest(TmpRootTest):
    def setUp(self):
        super().setUp()
        self.catalog = make_catalog([make_entry(LOG_A, sweeps=4), make_entry(LOG_B, sweeps=4)])

    def _ensure(self, request, **kwargs):
        with mock.patch.object(ad.ac, "load_or_build", return_value=self.catalog):
            return ad.ensure_logs(request, root=self.root, **kwargs)

    def test_download_false_raises_naming_the_fix(self):
        request = al.SensorRequest(split=al.SensorSplit.VAL, items=(al.SensorItem.MAP,))
        with self.assertRaises(ad.MissingDataError) as ctx:
            self._ensure(request, download=False)
        message = str(ctx.exception)
        self.assertIn("map", message)
        self.assertIn("download sensor/val", message)

    def test_download_false_succeeds_when_everything_is_present(self):
        request = al.SensorRequest(split=al.SensorSplit.VAL, items=(al.SensorItem.MAP,))
        for log_id in (LOG_A, LOG_B):
            write_item(self.root, request, log_id, al.SensorItem.MAP, 3)
        logs = self._ensure(request, download=False)
        self.assertEqual([log.log_id for log in logs], sorted([LOG_A, LOG_B]))

    def test_no_network_when_nothing_is_missing(self):
        """The idempotence guarantee: a second call must not transfer anything."""
        request = al.SensorRequest(split=al.SensorSplit.VAL, items=(al.SensorItem.MAP,))
        for log_id in (LOG_A, LOG_B):
            write_item(self.root, request, log_id, al.SensorItem.MAP, 3)
        with mock.patch.object(ad.s5cmd, "run_commands") as run:
            self._ensure(request)
        run.assert_not_called()

    def test_downloads_then_verifies(self):
        request = al.SensorRequest(split=al.SensorSplit.VAL, items=(al.SensorItem.MAP,))

        def fake_run(commands, opts=None, dry_run=False):
            # Simulate s5cmd actually fetching the files.
            for log_id in (LOG_A, LOG_B):
                write_item(self.root, request, log_id, al.SensorItem.MAP, 3)
            return s5cmd.Result(returncode=0, num_commands=len(commands), elapsed_s=1.0)

        with mock.patch.object(ad.s5cmd, "run_commands", side_effect=fake_run):
            logs = self._ensure(request)
        self.assertEqual(len(logs), 2)
        self.assertTrue(logs[0].path.is_dir())

    def test_verification_catches_a_silent_short_download(self):
        """A zero exit code is not proof the data landed, so ensure_logs re-checks."""
        request = al.SensorRequest(split=al.SensorSplit.VAL, items=(al.SensorItem.MAP,))
        with mock.patch.object(
            ad.s5cmd, "run_commands",
            return_value=s5cmd.Result(returncode=0, num_commands=2, elapsed_s=1.0),
        ):
            with self.assertRaises(ad.DownloadFailedError) as ctx:
                self._ensure(request)
        self.assertIn("incomplete", str(ctx.exception))

    def test_s5cmd_failures_are_raised(self):
        request = al.SensorRequest(split=al.SensorSplit.VAL, items=(al.SensorItem.MAP,))
        with mock.patch.object(
            ad.s5cmd, "run_commands",
            return_value=s5cmd.Result(returncode=1, num_commands=2, elapsed_s=1.0,
                                      failures=('ERROR "cp ...": denied',)),
        ):
            with self.assertRaises(ad.DownloadFailedError) as ctx:
                self._ensure(request)
        self.assertIn("denied", str(ctx.exception))

    def test_log_ids_restrict_the_selection(self):
        request = al.SensorRequest(
            split=al.SensorSplit.VAL, items=(al.SensorItem.MAP,), log_ids=(LOG_A,)
        )
        write_item(self.root, request, LOG_A, al.SensorItem.MAP, 3)
        logs = self._ensure(request, download=False)
        self.assertEqual([log.log_id for log in logs], [LOG_A])

    def test_unknown_log_id_raises_before_any_transfer(self):
        request = al.SensorRequest(
            split=al.SensorSplit.VAL, items=(al.SensorItem.MAP,), log_ids=("nope",)
        )
        with mock.patch.object(ad.s5cmd, "run_commands") as run:
            with self.assertRaises(KeyError):
                self._ensure(request)
        run.assert_not_called()

    def test_local_log_item_path(self):
        request = al.SensorRequest(split=al.SensorSplit.VAL, items=(al.SensorItem.MAP,))
        for log_id in (LOG_A, LOG_B):
            write_item(self.root, request, log_id, al.SensorItem.MAP, 3)
        log = self._ensure(request, download=False)[0]
        self.assertEqual(log.item_path(al.SensorItem.MAP).name, "map")
        self.assertTrue(log.item_path(al.SensorItem.MAP).is_dir())


class ParseSizeTest(unittest.TestCase):
    def test_suffixes(self):
        self.assertEqual(ad.parse_size("10GB"), 10 * 1024**3)
        self.assertEqual(ad.parse_size("500mb"), 500 * 1024**2)
        self.assertEqual(ad.parse_size("2TB"), 2 * 1024**4)
        self.assertEqual(ad.parse_size("1.5GB"), int(1.5 * 1024**3))
        self.assertEqual(ad.parse_size("2048"), 2048)

    def test_invalid(self):
        for text in ("banana", "", "GB"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    ad.parse_size(text)


if __name__ == "__main__":
    unittest.main()
