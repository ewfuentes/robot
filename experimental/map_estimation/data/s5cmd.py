"""Thin wrapper around the `s5cmd` binary.

`s5cmd` is a parallel S3 client that is dramatically faster than the AWS CLI for the
many-small-objects workload that Argoverse presents (a single sensor log is ~3000 objects).
This module owns every subprocess invocation so that callers never build command lines
themselves.

Two things this wrapper exists to get right:

* **Region pinning.** The Argoverse bucket lives in ``us-east-1``. If ``AWS_REGION`` is unset
  or set to something else in the ambient environment, every call fails with
  ``BucketRegionError``. We inject the region into the child environment rather than trusting
  whatever the machine happens to be configured for.
* **Unsigned requests.** The bucket is public, so requests must be sent with
  ``--no-sign-request``; otherwise s5cmd tries to load credentials and fails on machines that
  have AWS config for an unrelated account.

Deliberately depends only on the standard library and msgspec so it stays trivially testable.
"""

import collections
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Sequence

import msgspec

logger = logging.getLogger(__name__)

# The Argoverse bucket's region. s5cmd raises BucketRegionError when this disagrees with the
# bucket, so it must be set explicitly rather than inherited from the environment.
DEFAULT_REGION = "us-east-1"

# s5cmd's own default is 256, which is more parallelism than is polite on a shared machine.
DEFAULT_NUM_WORKERS = 32

# How much of a failed batch's output to quote back when s5cmd gives no parseable error line.
_OUTPUT_TAIL_LINES = 20

# Emit a progress line at INFO at most this often during a transfer.
_PROGRESS_INTERVAL_S = 5.0


class S5cmdNotFoundError(RuntimeError):
    """The s5cmd binary could not be located."""


class S5cmdError(RuntimeError):
    """An s5cmd invocation exited non-zero."""


class Options(msgspec.Struct, frozen=True):
    """How to invoke s5cmd. The defaults are correct for the public Argoverse bucket."""

    binary: Path | None = None
    """Explicit path to the binary. None means look it up on PATH."""
    num_workers: int = DEFAULT_NUM_WORKERS
    region: str = DEFAULT_REGION
    retry_count: int = 5
    no_sign_request: bool = True
    """The Argoverse bucket is public; signing would require credentials we don't have."""


class Object(msgspec.Struct):
    """One record from `s5cmd ls --json`.

    Only the fields we use are declared; msgspec ignores the rest (etag, last_modified,
    storage_class).
    """

    key: str
    """Full URI, e.g. 's3://argoverse/datasets/av2/sensor/val/<log>/annotations.feather'."""
    type: str
    """'file' or 'directory'."""
    size: int = 0
    """Absent for directory records, hence the default."""

    @property
    def is_file(self) -> bool:
        return self.type == "file"


class Result(msgspec.Struct):
    """Outcome of a batch of transfer commands."""

    returncode: int
    num_commands: int
    elapsed_s: float
    failures: tuple[str, ...] = ()
    """Per-object errors s5cmd reported. Never empty when `ok` is False -- see
    :func:`run_commands`, which synthesizes an entry from the exit status if s5cmd failed
    without printing a recognizable error line."""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.failures


def binary_path(explicit: Path | None = None) -> Path:
    """Locate the s5cmd binary.

    Order: the explicit argument, then $S5CMD_BINARY, then PATH. Mirrors the lookup-or-raise
    pattern in common/ollama/pyollama.py.
    """
    if explicit is not None:
        candidate = Path(explicit)
        if not candidate.exists():
            raise S5cmdNotFoundError(f"s5cmd not found at {candidate}")
        return candidate

    from_env = os.environ.get("S5CMD_BINARY")
    if from_env:
        candidate = Path(from_env)
        if not candidate.exists():
            raise S5cmdNotFoundError(f"$S5CMD_BINARY points at {candidate}, which does not exist")
        return candidate

    found = shutil.which("s5cmd")
    if found is None:
        raise S5cmdNotFoundError(
            "s5cmd not found on PATH. Install it by running setup.sh, or set $S5CMD_BINARY."
        )
    return Path(found)


def _global_flags(opts: Options, *, as_json: bool = False, dry_run: bool = False) -> list[str]:
    """Build the argv prefix. s5cmd requires global flags *before* the subcommand."""
    cmd = [str(binary_path(opts.binary))]
    if as_json:
        cmd.append("--json")
    if opts.no_sign_request:
        cmd.append("--no-sign-request")
    if dry_run:
        cmd.append("--dry-run")
    cmd += ["--numworkers", str(opts.num_workers)]
    cmd += ["--retry-count", str(opts.retry_count)]
    return cmd


def _child_env(opts: Options) -> dict[str, str]:
    """The environment for the child process, with the bucket's region forced.

    Both names are set because s5cmd's SDK consults AWS_REGION and AWS_DEFAULT_REGION
    depending on version.
    """
    return {**os.environ, "AWS_REGION": opts.region, "AWS_DEFAULT_REGION": opts.region}


def _run(cmd: list[str], opts: Options, *, check: bool = True) -> subprocess.CompletedProcess:
    logger.info("running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd, env=_child_env(opts), capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise S5cmdError(
            f"s5cmd exited {result.returncode}: {' '.join(cmd)}\n{result.stderr.strip()}"
        )
    return result


def version(opts: Options = Options()) -> str:
    """Return the s5cmd version string, e.g. 'v2.3.0-991c9fb'.

    Recorded in the catalog so a listing can be attributed to a specific binary.
    """
    cmd = [str(binary_path(opts.binary)), "version"]
    return _run(cmd, opts).stdout.strip()


def _parse_jsonl(stdout: str) -> list[Object]:
    """Parse s5cmd's JSONL output, skipping blank lines.

    s5cmd emits one JSON object per line rather than a JSON array, so this cannot be a single
    msgspec.json.decode call.
    """
    decoder = msgspec.json.Decoder(Object)
    out = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(decoder.decode(line.encode()))
    return out


def list_prefixes(uri: str, *, opts: Options = Options()) -> list[str]:
    """List the immediate child "directories" of `uri`.

    `uri` must end in '/'. Returns bare names with no trailing slash, e.g. log ids. Used to
    enumerate a split's logs, which takes ~0.2 s for 150 entries.
    """
    if not uri.endswith("/"):
        raise ValueError(f"list_prefixes needs a trailing slash: {uri!r}")
    cmd = _global_flags(opts, as_json=True) + ["ls", uri]
    records = _parse_jsonl(_run(cmd, opts).stdout)
    return [record.key.rstrip("/").rsplit("/", 1)[-1] for record in records if not record.is_file]


def list_objects(uri_prefix: str, *, opts: Options = Options()) -> list[Object]:
    """Recursively list every object under `uri_prefix`.

    `uri_prefix` should end in '/'; a '*' is appended to make s5cmd recurse. Takes ~0.4 s for
    a 3000-object sensor log.
    """
    if not uri_prefix.endswith("/"):
        raise ValueError(f"list_objects needs a trailing slash: {uri_prefix!r}")
    cmd = _global_flags(opts, as_json=True) + ["ls", f"{uri_prefix}*"]
    return [record for record in _parse_jsonl(_run(cmd, opts).stdout) if record.is_file]


def run_commands(
    commands: Sequence[str],
    *,
    opts: Options = Options(),
    dry_run: bool = False,
) -> Result:
    """Execute a batch of s5cmd commands through `s5cmd run`.

    Batching matters: a single `run` lets --numworkers parallelize across *all* the commands,
    whereas one subprocess per copy would serialize them. `commands` are subcommand lines
    without the binary name, e.g. "cp -n 's3://bucket/a/*' '/local/a/'".

    An empty batch is a no-op, so callers can hand over a plan that turned out to need nothing.
    """
    if not commands:
        return Result(returncode=0, num_commands=0, elapsed_s=0.0)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="s5cmd-batch-", delete=False
    ) as handle:
        handle.write("\n".join(commands) + "\n")
        batch_file = Path(handle.name)

    try:
        cmd = _global_flags(opts, dry_run=dry_run) + ["run", str(batch_file)]
        logger.info("running: %s", " ".join(cmd))
        logger.info("batch of %d commands in %s", len(commands), batch_file)
        start = time.monotonic()

        # Streamed rather than captured: s5cmd prints a line per copied object, so a large pull
        # (700 logs x 3000 objects) would otherwise buffer hundreds of MB and show the user
        # nothing until the whole transfer ended. stderr is merged so ordering is preserved.
        failures: list[str] = []
        tail: collections.deque[str] = collections.deque(maxlen=_OUTPUT_TAIL_LINES)
        num_lines = 0
        last_progress = time.monotonic()
        with subprocess.Popen(
            cmd,
            env=_child_env(opts),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            for line in process.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                num_lines += 1
                tail.append(line)
                if line.strip().startswith("ERROR"):
                    failures.append(line)
                    logger.warning("%s", line)
                else:
                    logger.debug("%s", line)
                    # A periodic summary at INFO rather than a line per object: streaming
                    # exists so a multi-hour transfer isn't silent, but 2M individual lines
                    # would be unreadable. Throttled by time rather than by count so the
                    # cadence is the same for a 150-object log and a 2M-object split.
                    now = time.monotonic()
                    if now - last_progress >= _PROGRESS_INTERVAL_S:
                        last_progress = now
                        logger.info("... %d objects transferred (%.0fs)", num_lines,
                                    now - start)
            returncode = process.wait()
        elapsed = time.monotonic() - start

        if returncode != 0 and not failures:
            # s5cmd can fail without printing a parseable ERROR line (bad flag, abort, bad
            # region). Without this the caller would raise "reported 0 failure(s)" with no
            # detail at all, and the CLI would print a success-shaped "0 failures".
            detail = "; ".join(tail) if tail else "no output"
            failures.append(f"ERROR s5cmd exited {returncode} with no error line: {detail}")

        logger.info("s5cmd emitted %d line(s), %d failure(s)", num_lines, len(failures))
        return Result(
            returncode=returncode,
            num_commands=len(commands),
            elapsed_s=elapsed,
            failures=tuple(failures),
        )
    finally:
        batch_file.unlink(missing_ok=True)


def quote(value: str | Path) -> str:
    """Quote a path for an s5cmd batch-file line.

    s5cmd parses its batch file with shell-like word splitting, so wildcards must survive
    unexpanded and spaces must not split arguments. Single quotes do both.
    """
    text = str(value)
    if "'" in text:
        raise ValueError(f"cannot safely quote a path containing a single quote: {text!r}")
    return f"'{text}'"


def format_bytes(num_bytes: int) -> str:
    """Human-readable byte count, for CLI tables and log lines."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")
