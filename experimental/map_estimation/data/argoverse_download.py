"""Get Argoverse 2 data onto local disk, fetching only what is missing.

The entry point is :func:`ensure_logs`, whose contract is *validate the request, check what is
already on disk, download the remainder, hand back local paths*. It is idempotent: calling it
twice costs one catalog load and one directory scan the second time, with no network traffic.

Everything except the transfer itself is offline. :func:`plan` turns catalog rows into an
inspectable list of copies with byte totals, so a caller (or `--dry_run`) can see exactly what
would happen before anything is written.

Completeness is judged by **object count against the catalog's count**, not by bytes. Counting
files is a cheap `scandir`, and it is the same signal `s5cmd cp -n` acts on when resuming.
Pass ``verify_bytes=True`` for the stricter comparison when a truncated transfer is suspected.
"""

import enum
import logging
import os
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import msgspec

from common.python.serialization import MSGSPEC_STRUCT_OPTS
from experimental.map_estimation.data import argoverse_catalog as ac
from experimental.map_estimation.data import argoverse_layout as al
from experimental.map_estimation.data import s5cmd

logger = logging.getLogger(__name__)


class MissingDataError(RuntimeError):
    """Requested data is absent locally and downloading was not permitted."""


class InsufficientSpaceError(RuntimeError):
    """The plan would not fit in the destination filesystem."""


class DownloadFailedError(RuntimeError):
    """s5cmd reported failures, or data was still incomplete after transferring."""


class State(enum.Enum):
    """Local availability of one item."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"
    NOT_PRESENT = "not_present"
    """The item does not exist remotely for this log, so there is nothing to fetch."""


class SkipReason(enum.Enum):
    ALREADY_COMPLETE = "already_complete"
    NOT_PRESENT = "not_present"


class ItemStatus(msgspec.Struct, frozen=True):
    """Local vs remote size of one item of one log."""

    item: str
    """Item token; str rather than the enum so this serializes without a union key."""
    expected: ac.ItemStat
    local: ac.ItemStat
    state: State

    @property
    def is_satisfied(self) -> bool:
        return self.state in (State.COMPLETE, State.NOT_PRESENT)


class LogStatus(msgspec.Struct, frozen=True):
    log_id: str
    path: Path
    items: tuple[ItemStatus, ...]

    @property
    def state(self) -> State:
        """COMPLETE only when every requested item is satisfied."""
        if all(status.is_satisfied for status in self.items):
            return State.COMPLETE
        if any(status.local.num_objects for status in self.items):
            return State.PARTIAL
        return State.MISSING

    def missing(self) -> tuple[ItemStatus, ...]:
        return tuple(status for status in self.items if not status.is_satisfied)

    def local_total(self) -> ac.ItemStat:
        total = ac.EMPTY_STAT
        for status in self.items:
            total = total + status.local
        return total


class Transfer(msgspec.Struct, frozen=True):
    """One `s5cmd cp` invocation. Sizes come from the catalog, not from a fresh listing."""

    log_id: str
    item: str
    src: str
    dst: Path
    num_bytes: int
    num_objects: int
    is_dir: bool = False
    """Whether `dst` names a directory. Carried explicitly because pathlib strips the trailing
    slash that s5cmd needs in order to treat a destination as a directory."""
    force_overwrite: bool = False
    """Set when local files exist but are the wrong size, so `-n` must be dropped for this
    transfer specifically -- otherwise s5cmd would skip the very files that need replacing."""

    def to_command(self, *, overwrite: bool = False) -> str:
        """The batch-file line for this copy.

        `-n` (--no-clobber) makes a re-run skip files already present, which is what turns an
        interrupted download into a cheap resume. It is dropped when the caller asked to
        overwrite, or when this transfer exists *because* the local bytes are wrong: keeping
        `-n` there would make a truncated file unrepairable, since s5cmd would decline to
        replace it while the size check kept flagging it forever.
        """
        flags = "" if (overwrite or self.force_overwrite) else " -n"
        destination = f"{self.dst}/" if self.is_dir else str(self.dst)
        return f"cp{flags} {s5cmd.quote(self.src)} {s5cmd.quote(destination)}"

    @property
    def dst_dir(self) -> Path:
        """Directory that must exist before s5cmd writes."""
        return self.dst if self.is_dir else self.dst.parent


class DownloadPlan(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """What :func:`execute` would do. Built entirely offline."""

    spec: str
    root: Path
    items: tuple[str, ...]
    transfers: tuple[Transfer, ...]
    skipped: tuple[tuple[Transfer, SkipReason], ...] = ()
    free_bytes: int | None = None

    @property
    def num_logs(self) -> int:
        return len({transfer.log_id for transfer in self.transfers})

    @property
    def total_bytes(self) -> int:
        return sum(transfer.num_bytes for transfer in self.transfers)

    @property
    def total_objects(self) -> int:
        return sum(transfer.num_objects for transfer in self.transfers)

    @property
    def is_empty(self) -> bool:
        return not self.transfers

    def num_skipped(self, reason: SkipReason | None = None) -> int:
        if reason is None:
            return len(self.skipped)
        return sum(1 for _, skip_reason in self.skipped if skip_reason is reason)

    def to_commands(self, *, overwrite: bool = False) -> list[str]:
        return [transfer.to_command(overwrite=overwrite) for transfer in self.transfers]

    def summary(self) -> str:
        return (
            f"{self.num_logs} logs, {self.total_objects} objects, "
            f"{s5cmd.format_bytes(self.total_bytes)} in {len(self.transfers)} transfers"
        )


class LocalLog(msgspec.Struct, frozen=True):
    """A log confirmed present on disk with the requested items."""

    log_id: str
    path: Path
    items: tuple[str, ...]

    def item_path(self, item: al._Item) -> Path:
        """Local path of one item within this log."""
        return self.path / item.resolve_relpath(self.log_id)


def _count_local(path: Path, is_dir: bool) -> ac.ItemStat:
    """Count the objects and bytes present locally for one item.

    os.scandir rather than Path.rglob: `status sensor/train --items all` walks ~2M entries, and
    scandir gets file-vs-directory from the directory entry itself, halving the syscalls.
    """
    if not is_dir:
        try:
            return ac.ItemStat(num_bytes=path.stat().st_size, num_objects=1)
        except OSError:
            return ac.EMPTY_STAT

    num_bytes = 0
    num_objects = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            # Absent or unreadable directory: nothing local here, which is the answer we want.
            continue
        for entry in entries:
            # Per-entry, so one unreadable file (permissions, or a race with a concurrent
            # download) skips that file rather than abandoning the rest of the directory and
            # under-counting the whole item. A file we cannot stat is left uncounted rather
            # than counted-with-unknown-size, so the item reads as incomplete and gets
            # re-fetched -- the conservative direction.
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                size = entry.stat().st_size
            except OSError:
                continue
            num_objects += 1
            num_bytes += size
    return ac.ItemStat(num_bytes=num_bytes, num_objects=num_objects)


def _item_status(
    entry: ac.LogEntry,
    request: al.Request,
    item: al._Item,
    root: Path,
    verify_bytes: bool,
) -> ItemStatus:
    expected = entry.stat(item)
    if expected.num_objects == 0:
        # Absent remotely, e.g. annotations on the sensor test split. This reads a *missing*
        # catalog stat as "there is nothing to fetch", which is only sound because a catalog
        # entry is written all-or-nothing: argoverse_catalog._build_per_log_detail drops a log
        # entirely if its listing fails rather than recording a partial row, and build() refuses
        # to emit a catalog with no rows at all.
        return ItemStatus(item=item.token, expected=expected, local=ac.EMPTY_STAT,
                          state=State.NOT_PRESENT)

    local = _count_local(al.local_path(request, entry.log_id, item, root), item.is_dir)
    if local.num_objects == 0:
        state = State.MISSING
    elif local.num_objects >= expected.num_objects and (
        not verify_bytes or local.num_bytes >= expected.num_bytes
    ):
        state = State.COMPLETE
    else:
        state = State.PARTIAL
    return ItemStatus(item=item.token, expected=expected, local=local, state=state)


def local_status(
    request: al.Request,
    entries: Sequence[ac.LogEntry],
    *,
    root: Path = al.DEFAULT_ROOT,
    verify_bytes: bool = False,
) -> list[LogStatus]:
    """Report what of `request` is already on disk. Offline: only stats the filesystem."""
    statuses = []
    for entry in entries:
        item_statuses = tuple(
            _item_status(entry, request, item, root, verify_bytes) for item in request.items
        )
        statuses.append(
            LogStatus(
                log_id=entry.log_id,
                path=al.log_dir(request, entry.log_id, root),
                items=item_statuses,
            )
        )
    return statuses


def plan(
    request: al.Request,
    entries: Sequence[ac.LogEntry],
    *,
    root: Path = al.DEFAULT_ROOT,
    overwrite: bool = False,
    ignore_free_space: bool = False,
    verify_bytes: bool = False,
) -> DownloadPlan:
    """Work out which copies `request` needs. Pure and offline.

    Items already complete locally are recorded in `skipped` rather than dropped, so callers can
    report "N already complete" instead of silently doing less than asked.

    For a *partially* present item the plan counts the whole item, because the transfer is a
    single wildcard and only `cp -n` (inside s5cmd) knows which individual files it will skip.
    Byte totals are therefore an upper bound when resuming -- deliberately conservative, so the
    free-space check below cannot under-estimate.

    Raises InsufficientSpaceError when the total would not fit under `root`; a tool whose whole
    job is bulk transfer should refuse rather than fill the disk mid-copy.
    """
    transfers: list[Transfer] = []
    skipped: list[tuple[Transfer, SkipReason]] = []

    for entry in entries:
        for item in request.items:
            status = _item_status(entry, request, item, root, verify_bytes)
            destination = al.local_path(request, entry.log_id, item, root)
            transfer = Transfer(
                log_id=entry.log_id,
                item=item.token,
                src=al.s3_uri(request, entry.log_id, item),
                dst=destination,
                num_bytes=status.expected.num_bytes,
                num_objects=status.expected.num_objects,
                is_dir=item.is_dir,
                # Under --verify_bytes a PARTIAL item may hold short files rather than merely
                # be missing some, and `cp -n` cannot replace those. Re-fetching a few intact
                # files is the cheap side of this trade.
                force_overwrite=verify_bytes and status.state is State.PARTIAL,
            )
            if status.state is State.NOT_PRESENT:
                skipped.append((transfer, SkipReason.NOT_PRESENT))
            elif status.state is State.COMPLETE and not overwrite:
                skipped.append((transfer, SkipReason.ALREADY_COMPLETE))
            else:
                transfers.append(transfer)

    free_bytes = None
    probe = Path(root)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if probe.exists():
        free_bytes = shutil.disk_usage(probe).free

    result = DownloadPlan(
        spec=request.spec(),
        root=Path(root),
        items=tuple(item.token for item in request.items),
        transfers=tuple(transfers),
        skipped=tuple(skipped),
        free_bytes=free_bytes,
    )

    if not ignore_free_space and free_bytes is not None and result.total_bytes > free_bytes:
        shortfall = result.total_bytes - free_bytes
        raise InsufficientSpaceError(
            f"{result.spec} needs {s5cmd.format_bytes(result.total_bytes)} but only "
            f"{s5cmd.format_bytes(free_bytes)} is free under {root} "
            f"(short by {s5cmd.format_bytes(shortfall)}). "
            "Narrow --items or --limit, or pass ignore_free_space to proceed anyway."
        )
    return result


def execute(
    download_plan: DownloadPlan,
    *,
    opts: s5cmd.Options = s5cmd.Options(),
    overwrite: bool = False,
    dry_run: bool = False,
) -> s5cmd.Result:
    """Run a plan's copies as a single s5cmd batch.

    Batching matters: one `s5cmd run` lets --numworkers parallelize across every copy in the
    plan, whereas one subprocess per copy would serialize them.
    """
    if download_plan.is_empty:
        logger.info("nothing to download")
        return s5cmd.Result(returncode=0, num_commands=0, elapsed_s=0.0)

    if not dry_run:
        # s5cmd will create leaf directories, but creating them up front keeps failures about
        # the transfer rather than about the tree.
        for transfer in download_plan.transfers:
            transfer.dst_dir.mkdir(parents=True, exist_ok=True)

    commands = download_plan.to_commands(overwrite=overwrite)
    logger.info("downloading %s", download_plan.summary())
    return s5cmd.run_commands(commands, opts=opts, dry_run=dry_run)


def ensure_logs(
    request: al.Request,
    *,
    root: Path = al.DEFAULT_ROOT,
    download: bool = True,
    dry_run: bool = False,
    overwrite: bool = False,
    ignore_free_space: bool = False,
    verify_bytes: bool = False,
    opts: s5cmd.Options = s5cmd.Options(),
    cache_dir: Path = ac.CACHE_DIR,
    refresh_catalog: bool = False,
) -> list[LocalLog]:
    """Ensure `request`'s items exist under `root`, fetching whatever is missing.

    Three phases, in order, so failures happen as early and as cheaply as possible:

    1. **Validate** (offline). The request type has already constrained items and split by
       construction; here the log ids are resolved against the catalog, so a typo'd id fails
       before any transfer starts.
    2. **Check** (offline). Local object counts are compared against the catalog's.
    3. **Download**. One batched `s5cmd run` for the remainder, skipped entirely when nothing is
       missing -- which is what makes repeat calls cheap.

    With ``download=False`` this becomes an assertion: it raises :class:`MissingDataError`
    naming what is absent and the CLI command that would fetch it. Use that in library code
    which must never silently start a multi-gigabyte transfer.

    Returns one :class:`LocalLog` per requested log, sorted by log id.
    """
    catalog = ac.load_or_build(
        request, cache_dir=cache_dir, refresh=refresh_catalog, opts=opts
    )
    entries = ac.filter_logs(catalog, log_ids=request.log_ids)
    if not entries:
        raise MissingDataError(f"no logs selected for {request.spec()}")

    download_plan = plan(
        request,
        entries,
        root=root,
        overwrite=overwrite,
        ignore_free_space=ignore_free_space,
        verify_bytes=verify_bytes,
    )

    if not download_plan.is_empty:
        if not download:
            missing = ", ".join(
                sorted({transfer.item for transfer in download_plan.transfers})
            )
            raise MissingDataError(
                f"{download_plan.num_logs} log(s) of {request.spec()} are missing [{missing}] "
                f"under {root} ({download_plan.summary()}). Fetch them with:\n"
                f"  bazel run //experimental/map_estimation/data:argoverse -- "
                f"download {request.spec()} --items {','.join(download_plan.items)}"
            )
        result = execute(download_plan, opts=opts, overwrite=overwrite, dry_run=dry_run)
        if not result.ok:
            raise DownloadFailedError(
                f"s5cmd reported {len(result.failures)} failure(s):\n"
                + "\n".join(result.failures[:10])
            )
        if not dry_run:
            _verify(request, entries, root=root, verify_bytes=verify_bytes)

    return [
        LocalLog(
            log_id=entry.log_id,
            path=al.log_dir(request, entry.log_id, root),
            items=tuple(item.token for item in request.items),
        )
        for entry in entries
    ]


def _verify(
    request: al.Request,
    entries: Sequence[ac.LogEntry],
    *,
    root: Path,
    verify_bytes: bool,
) -> None:
    """Confirm the download actually landed, so ensure_logs' guarantee is real."""
    statuses = local_status(request, entries, root=root, verify_bytes=verify_bytes)
    incomplete = [status for status in statuses if status.state is not State.COMPLETE]
    if not incomplete:
        return
    details = []
    for status in incomplete[:5]:
        for item_status in status.missing():
            details.append(
                f"  {status.log_id} {item_status.item}: "
                f"{item_status.local.num_objects}/{item_status.expected.num_objects} objects"
            )
    raise DownloadFailedError(
        f"{len(incomplete)} log(s) still incomplete after downloading:\n"
        + "\n".join(details)
        + "\nIf the local files are the wrong size rather than absent, re-run with "
          "--overwrite to replace them."
    )


def parse_size(text: str) -> int:
    """Parse a human byte size such as '500MB', '10GB', '2TB', or a bare byte count."""
    cleaned = str(text).strip().upper().replace("_", "")
    multipliers = {"TB": 1024**4, "GB": 1024**3, "MB": 1024**2, "KB": 1024, "B": 1}
    for suffix, multiplier in multipliers.items():
        if cleaned.endswith(suffix):
            number = cleaned[: -len(suffix)].strip()
            if not number:
                break
            return int(float(number) * multiplier)
    try:
        return int(float(cleaned))
    except ValueError:
        raise ValueError(
            f"cannot parse {text!r} as a size; use forms like '500MB', '10GB', '2TB'"
        ) from None


def summarize_status(statuses: Iterable[LogStatus]) -> dict[State, list[LogStatus]]:
    """Group log statuses by state, for the `status` subcommand's summary table."""
    grouped: dict[State, list[LogStatus]] = {
        State.COMPLETE: [], State.PARTIAL: [], State.MISSING: []
    }
    for status in statuses:
        grouped[status.state].append(status)
    return grouped
