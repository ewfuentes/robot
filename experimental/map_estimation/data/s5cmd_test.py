import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from experimental.map_estimation.data import s5cmd

# A real `s5cmd ls --json` capture from the Argoverse bucket, trimmed to a few lines.
LS_RECURSIVE_JSONL = (
    '{"key":"s3://argoverse/datasets/av2/sensor/val/L/annotations.feather",'
    '"etag":"c2b80c","last_modified":"2023-03-24T21:31:58Z","type":"file",'
    '"size":285474,"storage_class":"INTELLIGENT_TIERING"}\n'
    '{"key":"s3://argoverse/datasets/av2/sensor/val/L/calibration/intrinsics.feather",'
    '"etag":"b92a82","last_modified":"2023-03-24T21:31:58Z","type":"file",'
    '"size":5330,"storage_class":"STANDARD"}\n'
    "\n"  # s5cmd sometimes emits a trailing blank line
)

LS_PREFIXES_JSONL = (
    '{"key":"s3://argoverse/datasets/av2/sensor/val/02678d04-cc9f/","type":"directory"}\n'
    '{"key":"s3://argoverse/datasets/av2/sensor/val/02a00399-3857/","type":"directory"}\n'
)


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout,
                                       stderr=stderr)


class BinaryLookupTest(unittest.TestCase):
    def test_explicit_missing_path_raises(self):
        with self.assertRaises(s5cmd.S5cmdNotFoundError):
            s5cmd.binary_path(Path("/nonexistent/s5cmd"))

    def test_missing_from_path_names_the_remedy(self):
        with mock.patch.object(s5cmd.shutil, "which", return_value=None), \
             mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(s5cmd.S5cmdNotFoundError) as ctx:
                s5cmd.binary_path()
        self.assertIn("setup.sh", str(ctx.exception))

    def test_env_override_is_consulted_before_path(self):
        with mock.patch.dict(os.environ, {"S5CMD_BINARY": "/nope/s5cmd"}):
            with self.assertRaises(s5cmd.S5cmdNotFoundError) as ctx:
                s5cmd.binary_path()
        self.assertIn("S5CMD_BINARY", str(ctx.exception))


class CommandConstructionTest(unittest.TestCase):
    """s5cmd requires global flags before the subcommand, and the region must be forced."""

    def setUp(self):
        self.which = mock.patch.object(s5cmd.shutil, "which", return_value="/usr/bin/s5cmd")
        self.which.start()
        self.addCleanup(self.which.stop)

    def test_region_is_injected_into_the_child_env(self):
        """Without this, the us-east-1 bucket fails with BucketRegionError on this machine."""
        with mock.patch.object(s5cmd.subprocess, "run",
                               return_value=_completed(LS_PREFIXES_JSONL)) as run:
            s5cmd.list_prefixes("s3://argoverse/datasets/av2/sensor/val/")
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["AWS_REGION"], "us-east-1")
        self.assertEqual(env["AWS_DEFAULT_REGION"], "us-east-1")

    def test_ambient_region_is_overridden(self):
        with mock.patch.dict(os.environ, {"AWS_REGION": "us-east-2"}), \
             mock.patch.object(s5cmd.subprocess, "run",
                               return_value=_completed(LS_PREFIXES_JSONL)) as run:
            s5cmd.list_prefixes("s3://argoverse/datasets/av2/sensor/val/")
        self.assertEqual(run.call_args.kwargs["env"]["AWS_REGION"], "us-east-1")

    def test_global_flags_precede_the_subcommand(self):
        with mock.patch.object(s5cmd.subprocess, "run",
                               return_value=_completed(LS_PREFIXES_JSONL)) as run:
            s5cmd.list_prefixes("s3://argoverse/datasets/av2/sensor/val/")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[0], "/usr/bin/s5cmd")
        subcommand_index = cmd.index("ls")
        for flag in ["--json", "--no-sign-request", "--numworkers", "--retry-count"]:
            self.assertLess(cmd.index(flag), subcommand_index, f"{flag} must precede 'ls'")

    def test_options_are_threaded_through(self):
        opts = s5cmd.Options(num_workers=4, retry_count=1, region="eu-west-1")
        with mock.patch.object(s5cmd.subprocess, "run",
                               return_value=_completed(LS_PREFIXES_JSONL)) as run:
            s5cmd.list_prefixes("s3://b/p/", opts=opts)
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--numworkers") + 1], "4")
        self.assertEqual(cmd[cmd.index("--retry-count") + 1], "1")
        self.assertEqual(run.call_args.kwargs["env"]["AWS_REGION"], "eu-west-1")

    def test_no_sign_request_can_be_disabled(self):
        with mock.patch.object(s5cmd.subprocess, "run",
                               return_value=_completed(LS_PREFIXES_JSONL)) as run:
            s5cmd.list_prefixes("s3://b/p/", opts=s5cmd.Options(no_sign_request=False))
        self.assertNotIn("--no-sign-request", run.call_args.args[0])

    def test_nonzero_exit_raises_with_stderr(self):
        with mock.patch.object(s5cmd.subprocess, "run",
                               return_value=_completed(stderr="ERROR boom", returncode=1)):
            with self.assertRaises(s5cmd.S5cmdError) as ctx:
                s5cmd.version()
        self.assertIn("boom", str(ctx.exception))


class ListingTest(unittest.TestCase):
    def setUp(self):
        self.which = mock.patch.object(s5cmd.shutil, "which", return_value="/usr/bin/s5cmd")
        self.which.start()
        self.addCleanup(self.which.stop)

    def test_list_prefixes_returns_bare_names(self):
        with mock.patch.object(s5cmd.subprocess, "run",
                               return_value=_completed(LS_PREFIXES_JSONL)):
            names = s5cmd.list_prefixes("s3://argoverse/datasets/av2/sensor/val/")
        self.assertEqual(names, ["02678d04-cc9f", "02a00399-3857"])

    def test_list_objects_appends_a_wildcard_and_keeps_only_files(self):
        with mock.patch.object(s5cmd.subprocess, "run",
                               return_value=_completed(LS_RECURSIVE_JSONL)) as run:
            objects = s5cmd.list_objects("s3://argoverse/datasets/av2/sensor/val/L/")
        self.assertTrue(run.call_args.args[0][-1].endswith("/L/*"))
        self.assertEqual([o.size for o in objects], [285474, 5330])
        self.assertTrue(all(o.is_file for o in objects))

    def test_blank_lines_are_skipped(self):
        """The capture above ends in a blank line, which must not become a decode error."""
        with mock.patch.object(s5cmd.subprocess, "run",
                               return_value=_completed(LS_RECURSIVE_JSONL)):
            self.assertEqual(len(s5cmd.list_objects("s3://b/p/")), 2)

    def test_trailing_slash_is_required(self):
        for func in (s5cmd.list_prefixes, s5cmd.list_objects):
            with self.subTest(func=func.__name__):
                with self.assertRaises(ValueError):
                    func("s3://argoverse/datasets/av2/sensor/val")


class RunCommandsTest(unittest.TestCase):
    def setUp(self):
        self.which = mock.patch.object(s5cmd.shutil, "which", return_value="/usr/bin/s5cmd")
        self.which.start()
        self.addCleanup(self.which.stop)

    def test_empty_batch_is_a_noop(self):
        """A plan that needs nothing must not spawn a process."""
        with mock.patch.object(s5cmd.subprocess, "run") as run:
            result = s5cmd.run_commands([])
        run.assert_not_called()
        self.assertTrue(result.ok)
        self.assertEqual(result.num_commands, 0)

    def test_commands_are_written_to_a_batch_file_that_is_cleaned_up(self):
        seen = {}

        def fake_run(cmd, **kwargs):
            batch_file = Path(cmd[-1])
            seen["path"] = batch_file
            seen["contents"] = batch_file.read_text()
            return _completed()

        with mock.patch.object(s5cmd.subprocess, "run", side_effect=fake_run):
            result = s5cmd.run_commands(["cp 'a' 'b'", "cp 'c' 'd'"])

        self.assertEqual(seen["contents"], "cp 'a' 'b'\ncp 'c' 'd'\n")
        self.assertFalse(seen["path"].exists(), "batch file must be removed")
        self.assertEqual(result.num_commands, 2)
        self.assertTrue(result.ok)

    def test_batch_file_is_removed_even_when_the_call_raises(self):
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["path"] = Path(cmd[-1])
            raise OSError("boom")

        with mock.patch.object(s5cmd.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(OSError):
                s5cmd.run_commands(["cp 'a' 'b'"])
        self.assertFalse(seen["path"].exists())

    def test_per_command_failures_are_surfaced(self):
        stderr = "ERROR \"cp s3://b/x /y\": object not found\nsome chatter\n"
        with mock.patch.object(s5cmd.subprocess, "run",
                               return_value=_completed(stderr=stderr, returncode=1)):
            result = s5cmd.run_commands(["cp 'a' 'b'"])
        self.assertFalse(result.ok)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("object not found", result.failures[0])

    def test_dry_run_passes_the_flag(self):
        with mock.patch.object(s5cmd.subprocess, "run", return_value=_completed()) as run:
            s5cmd.run_commands(["cp 'a' 'b'"], dry_run=True)
        self.assertIn("--dry-run", run.call_args.args[0])


class HelperTest(unittest.TestCase):
    def test_quote_preserves_wildcards(self):
        self.assertEqual(s5cmd.quote("s3://b/p/*"), "'s3://b/p/*'")

    def test_quote_rejects_unsafe_paths(self):
        with self.assertRaises(ValueError):
            s5cmd.quote("/tmp/it's")

    def test_format_bytes(self):
        self.assertEqual(s5cmd.format_bytes(512), "512 B")
        self.assertEqual(s5cmd.format_bytes(9932), "9.70 KB")
        self.assertEqual(s5cmd.format_bytes(1_139_539_558), "1.06 GB")


if __name__ == "__main__":
    unittest.main()
