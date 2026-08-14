"""Explore and download the Argoverse 2 dataset.

A thin shell over experimental.map_estimation.data.argoverse_download; every subcommand is a few
lines of formatting over the library API.

    # index a split once (cached under ~/.cache/robot), then explore offline
    argoverse -- index sensor/val
    argoverse -- list sensor/val --city PIT --limit 5
    argoverse -- show sensor/val 02678d04-cc9f-3148-9f95-1ba66347dff9

    # metadata only (~2 MB/log) for a whole split
    argoverse -- download sensor/val

    # lidar plus one camera for a few logs, previewed first
    argoverse -- download sensor/train --items metadata,lidar,ring_front_center \
        --log_id_file my_logs.txt --dry_run

    # what is already on disk
    argoverse -- status sensor/val --items metadata,lidar

The positional <spec> selects the dataset, which in turn selects which item names are legal --
so `--items ring_front_center` is accepted for sensor/val and rejected for lidar/val.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import msgspec

from experimental.map_estimation.data import argoverse_catalog as ac
from experimental.map_estimation.data import argoverse_download as ad
from experimental.map_estimation.data import argoverse_layout as al
from experimental.map_estimation.data import s5cmd

logger = logging.getLogger(__name__)

DEFAULT_CONFIRM_ABOVE = "10GB"


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(message)s"
    )


def print_json(data) -> None:
    """Print data as formatted JSON, handling msgspec structs and enums."""
    print(json.dumps(json.loads(msgspec.json.encode(data, enc_hook=str)), indent=2))


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a simple aligned table. Numeric-looking columns are right-aligned."""
    if not rows:
        return
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    def justify(cell, width, header):
        text = str(cell)
        return text.rjust(width) if header.isupper() and _is_numeric(text) else text.ljust(width)

    header_row = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    print(header_row)
    print("-" * len(header_row))
    for row in rows:
        print(" | ".join(
            justify(cell, width, header)
            for cell, width, header in zip(row, widths, headers)
        ))


def _is_numeric(text: str) -> bool:
    try:
        float(text.replace(",", ""))
        return True
    except ValueError:
        return False


def _progress_reporter(label: str):
    """Emit a single rewriting progress line, so indexing 20 000 logs is not 20 000 lines."""
    def report(done: int, total: int) -> None:
        if done == total or done % 25 == 0:
            print(f"\r{label} {done}/{total}", end="", file=sys.stderr, flush=True)
        if done == total:
            print(file=sys.stderr)
    return report


def _build_request(args) -> al.Request:
    """Turn the positional spec plus --items into a typed request.

    This is the single place strings become enums; everything downstream is typed.
    """
    # Build a default request first so group aliases can be expanded against what this
    # dataset *and split* actually has -- otherwise `--items all` on sensor/test would expand
    # to include the annotations that split does not ship, and be rejected.
    probe = al.make_request(args.spec)
    tokens = getattr(args, "items", None)
    items = (
        al.resolve_items(probe.item_type, tokens, available=probe.available_items())
        if tokens
        else None
    )

    log_ids = list(args.log_id) if getattr(args, "log_id", None) else []
    if getattr(args, "log_id_file", None):
        log_ids.extend(ac.read_log_id_file(args.log_id_file))

    return al.make_request(args.spec, items=items, log_ids=log_ids or None)


def _load_catalog(args, request: al.Request) -> ac.Catalog:
    return ac.load_or_build(
        request,
        cache_dir=args.cache_dir,
        catalog_path=args.catalog,
        refresh=args.refresh,
        opts=_options(args),
        progress=_progress_reporter("indexing logs"),
    )


def _options(args) -> s5cmd.Options:
    return s5cmd.Options(num_workers=args.num_workers)


def _select(args, catalog: ac.Catalog, request: al.Request) -> list[ac.LogEntry]:
    return ac.filter_logs(
        catalog,
        cities=args.city,
        log_ids=request.log_ids,
        sort_by=args.sort,
        limit=args.limit,
    )


def _mb(num_bytes: int) -> str:
    return f"{num_bytes / 1024 / 1024:.1f}"


def cmd_index(args) -> int:
    """Rebuild the cached catalog for a dataset+split."""
    request = _build_request(args)

    if request.log_ids:
        # Elsewhere --log_id accepts fnmatch patterns, but here the ids become S3 prefixes:
        # listing '.../aaa*/' would file every matched object under the literal string 'aaa*'
        # and invent a phantom log row. Patterns can only be expanded against a catalog, which
        # is the thing being built.
        patterns = [value for value in request.log_ids if any(c in value for c in "*?[")]
        if patterns:
            raise ValueError(
                f"index needs literal log ids, not patterns: {', '.join(patterns)}. "
                "Rebuild the whole split, or name the ids explicitly."
            )

    out_path = args.out or ac.cache_path(request, args.cache_dir)

    if request.log_ids:
        # A partial refresh patches rows into an existing catalog. With nothing to patch, the
        # one-log result would be saved as the canonical catalog for the whole split, and every
        # later list/download would quietly act on that single log.
        if not out_path.exists():
            raise ValueError(
                f"--log_id patches an existing catalog, but {out_path} does not exist. "
                f"Index the whole split first: index {request.spec()}"
            )
        existing = ac.load(out_path)
        if existing.spec != request.spec():
            raise ValueError(f"{out_path} describes {existing.spec}, not {request.spec()}")

    catalog = ac.build(
        request,
        opts=_options(args),
        progress=_progress_reporter("indexing logs"),
        log_ids=request.log_ids,
    )

    if request.log_ids:
        updated = {entry.log_id: entry for entry in existing.logs}
        updated.update({entry.log_id: entry for entry in catalog.logs})
        catalog = msgspec.structs.replace(
            catalog, logs=tuple(sorted(updated.values(), key=lambda e: e.log_id))
        )

    ac.save(catalog, out_path)

    total = catalog.total()
    print(f"wrote {out_path} ({len(catalog)} logs, {catalog.strategy.value})")
    if catalog.cities():
        print("cities: " + ", ".join(f"{city} {n}" for city, n in catalog.cities().items()))
    print(f"total:  {s5cmd.format_bytes(total.num_bytes)}, {total.num_objects} objects"
          + (" (sizes sampled)" if catalog.sizes_are_inferred else ""))
    return 0


def cmd_list(args) -> int:
    """List a dataset+split's logs with their sizes."""
    request = _build_request(args)
    catalog = _load_catalog(args, request)
    entries = _select(args, catalog, request)

    if args.json:
        print_json(entries)
        return 0
    if not entries:
        print(f"no logs of {catalog.spec} matched")
        return 0

    item_type = catalog.item_type
    has_lidar = any(item.is_lidar for item in item_type)
    has_cameras = bool(item_type.cameras())

    headers = ["LOG_ID", "CITY"]
    if has_lidar:
        headers.append("SWEEPS")
    if has_cameras:
        headers.append("CAMERA_MB")
    if has_lidar:
        headers.append("LIDAR_MB")
    headers.append("TOTAL_MB")
    if args.local:
        headers.append("LOCAL")

    statuses = {}
    if args.local:
        statuses = {
            status.log_id: status
            for status in ad.local_status(request, entries, root=args.root,
                                          verify_bytes=args.verify_bytes)
        }

    rows = []
    for entry in entries:
        row = [entry.log_id, entry.city or "-"]
        if has_lidar:
            row.append(str(entry.num_lidar_sweeps))
        if has_cameras:
            row.append(_mb(entry.total(item_type.cameras()).num_bytes))
        if has_lidar:
            row.append(_mb(entry.stat(item_type.LIDAR).num_bytes))
        row.append(_mb(entry.total().num_bytes))
        if args.local:
            row.append(statuses[entry.log_id].state.value)
        rows.append(row)

    print_table(headers, rows)
    print()
    shown = f"{len(entries)} shown" if len(entries) != len(catalog) else "all shown"
    source = "sampled sizes" if catalog.sizes_are_inferred else "measured"
    print(f"{len(catalog)} logs in {catalog.spec}, {shown}. "
          f"catalog built {catalog.built_at.date()} ({source})")
    return 0


def cmd_show(args) -> int:
    """Show one log's per-item sizes and local state."""
    request = _build_request(args)
    catalog = _load_catalog(args, request)
    entry = catalog.get(args.log_id_positional)

    # `show` describes everything the split has, not just the default item selection.
    full_request = msgspec.structs.replace(request, items=request.available_items())
    status = ad.local_status(
        full_request, [entry], root=args.root, verify_bytes=args.verify_bytes
    )[0]

    if args.json:
        print_json({"entry": entry, "status": status})
        return 0

    print(f"log_id  {entry.log_id}")
    print(f"dataset {catalog.spec}    city {entry.city or '-'}    "
          f"sweeps {entry.num_lidar_sweeps}")
    print(f"s3      {request.s3_prefix()}{entry.log_id}/")
    print(f"local   {status.path}")
    print()

    rows = []
    for item_status in status.items:
        if item_status.state is ad.State.NOT_PRESENT:
            local = "not in this dataset/split"
        elif item_status.state is ad.State.COMPLETE:
            local = "complete"
        elif item_status.state is ad.State.MISSING:
            local = "missing"
        else:
            local = (f"{item_status.local.num_objects}/"
                     f"{item_status.expected.num_objects} partial")
        rows.append([
            item_status.item,
            str(item_status.expected.num_objects),
            s5cmd.format_bytes(item_status.expected.num_bytes),
            local,
        ])
    total = entry.total()
    rows.append(["total", str(total.num_objects), s5cmd.format_bytes(total.num_bytes),
                 status.state.value])
    print_table(["ITEM", "OBJECTS", "SIZE", "LOCAL"], rows)
    return 0


def cmd_download(args) -> int:
    """Download the requested items, skipping whatever is already complete."""
    request = _build_request(args)
    catalog = _load_catalog(args, request)
    entries = _select(args, catalog, request)
    if not entries:
        print(f"no logs of {catalog.spec} matched")
        return 1

    try:
        download_plan = ad.plan(
            request,
            entries,
            root=args.root,
            overwrite=args.overwrite,
            ignore_free_space=args.ignore_free_space,
            verify_bytes=args.verify_bytes,
        )
    except ad.InsufficientSpaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        # Emit the plan and stop: printing progress and the final summary into the same stream
        # would corrupt the JSON, and a --json caller has no tty to answer the size prompt.
        print_json(download_plan)
        return 0
    else:
        print(f"plan: {download_plan.spec} -> {request.local_dir(args.root)}")
        print(f"  items: {', '.join(download_plan.items)}")
        print(f"  {download_plan.summary()}")
        if catalog.sizes_are_inferred:
            # Free-space decisions rest on these numbers, so say when they are extrapolated.
            print(f"  NOTE: sizes extrapolated from {catalog.sampled_logs} sampled logs")
        already = download_plan.num_skipped(ad.SkipReason.ALREADY_COMPLETE)
        absent = download_plan.num_skipped(ad.SkipReason.NOT_PRESENT)
        if already:
            print(f"  skipped: {already} already complete")
        if absent:
            print(f"  skipped: {absent} not present in this dataset/split")
        if download_plan.free_bytes is not None and download_plan.total_bytes:
            after = download_plan.free_bytes - download_plan.total_bytes
            print(f"  free space: {s5cmd.format_bytes(download_plan.free_bytes)} -> "
                  f"{s5cmd.format_bytes(after)}")

    if args.print_commands:
        for command in download_plan.to_commands(overwrite=args.overwrite):
            print(command)
        return 0

    if download_plan.is_empty:
        print("nothing to download; everything requested is already present.")
        return 0

    if args.dry_run:
        for transfer in download_plan.transfers[:10]:
            print(f"  {transfer.log_id[:12]}... {transfer.item:<20} "
                  f"{s5cmd.format_bytes(transfer.num_bytes):>10}  {transfer.src}")
        if len(download_plan.transfers) > 10:
            print(f"  ... {len(download_plan.transfers) - 10} more "
                  "(--json for the full plan)")
        print("dry run: nothing transferred.")
        return 0

    threshold = ad.parse_size(args.confirm_above)
    if not args.yes and download_plan.total_bytes > threshold:
        prompt = (f"Download {s5cmd.format_bytes(download_plan.total_bytes)} to "
                  f"{request.local_dir(args.root)}? [y/N]: ")
        try:
            answer = input(prompt)
        except EOFError:
            # No tty (CI, piped bazel run, nohup). Decline rather than crash, and say how to
            # proceed without a prompt.
            print("\nno input available to confirm; aborting. Pass --yes to skip this prompt.")
            return 1
        if answer.strip().lower() not in ("y", "yes"):
            print("aborted.")
            return 1

    result = ad.execute(download_plan, opts=_options(args), overwrite=args.overwrite)
    rate = download_plan.total_bytes / result.elapsed_s if result.elapsed_s else 0
    print(f"done: {download_plan.total_objects} objects, "
          f"{s5cmd.format_bytes(download_plan.total_bytes)} in {result.elapsed_s:.1f} s "
          f"({s5cmd.format_bytes(int(rate))}/s). {len(result.failures)} failures.")
    for failure in result.failures[:10]:
        print(f"  {failure}", file=sys.stderr)
    if not result.ok:
        return 1

    # A zero exit is not proof the data landed -- `cp -n` silently declines to replace a
    # wrong-sized file, so without this the command would report success having changed
    # nothing. ensure_logs() does the same check for library callers.
    remaining = [
        status for status in ad.local_status(request, entries, root=args.root,
                                             verify_bytes=args.verify_bytes)
        if status.state is not ad.State.COMPLETE
    ]
    if remaining:
        print(f"warning: {len(remaining)} log(s) still incomplete after downloading",
              file=sys.stderr)
        for status in remaining[:5]:
            for item_status in status.missing():
                print(f"  {status.log_id} {item_status.item}: "
                      f"{item_status.local.num_objects}/{item_status.expected.num_objects} "
                      "objects", file=sys.stderr)
        print("  if local files are the wrong size rather than absent, re-run with "
              "--overwrite", file=sys.stderr)
        return 1
    return 0


def cmd_status(args) -> int:
    """Report what of a dataset+split is already on disk."""
    request = _build_request(args)
    catalog = _load_catalog(args, request)
    entries = _select(args, catalog, request)
    statuses = ad.local_status(request, entries, root=args.root,
                               verify_bytes=args.verify_bytes)

    if args.json:
        print_json(statuses)
        return 0

    if args.detail:
        rows = [
            [
                status.log_id,
                status.state.value,
                f"{status.local_total().num_objects}",
                s5cmd.format_bytes(status.local_total().num_bytes),
            ]
            for status in statuses
        ]
        print_table(["LOG_ID", "STATE", "OBJECTS", "SIZE"], rows)
        return 0

    grouped = ad.summarize_status(statuses)
    expected_objects = sum(
        item.expected.num_objects for status in statuses for item in status.items
    )
    local_objects = sum(status.local_total().num_objects for status in statuses)
    local_bytes = sum(status.local_total().num_bytes for status in statuses)
    expected_bytes = sum(
        item.expected.num_bytes for status in statuses for item in status.items
    )

    rows = [
        [
            state.value,
            str(len(group)),
            s5cmd.format_bytes(sum(s.local_total().num_bytes for s in group)),
        ]
        for state, group in grouped.items()
    ]
    percent = 100.0 * local_bytes / expected_bytes if expected_bytes else 100.0
    rows.append(["total", str(len(statuses)),
                 f"{s5cmd.format_bytes(local_bytes)} / "
                 f"{s5cmd.format_bytes(expected_bytes)} ({percent:.0f}%)"])
    print_table(["STATE", "LOGS", "SIZE"], rows)
    print(f"\nobjects: {local_objects}/{expected_objects}")

    partial = grouped[ad.State.PARTIAL]
    if partial:
        print("\npartial logs:")
        for status in partial[:10]:
            detail = ", ".join(
                f"{item.item} {item.local.num_objects}/{item.expected.num_objects}"
                for item in status.missing()
            )
            print(f"  {status.log_id}  {detail}")

    if grouped[ad.State.PARTIAL] or grouped[ad.State.MISSING]:
        print(f"\nre-run: argoverse -- download {catalog.spec} "
              f"--items {','.join(item.token for item in request.items)}")
    return 0


def _add_selector_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("spec", type=str,
                        help="dataset[/split], e.g. sensor/val, tbv, lidar/train")
    parser.add_argument("--city", action="append", default=None,
                        help="filter by city code, e.g. PIT (repeatable, comma-separated)")
    parser.add_argument("--log_id", action="append", default=None,
                        help="select a log id; fnmatch patterns allowed (repeatable)")
    parser.add_argument("--log_id_file", type=Path, default=None,
                        help="file of log ids, one per line ('#' comments allowed)")
    parser.add_argument("--limit", type=int, default=None,
                        help="keep only the first N logs after sorting")
    parser.add_argument("--sort", type=str, default="log_id",
                        choices=["log_id", "city", "bytes", "sweeps"],
                        help="sort order applied before --limit (default: log_id)")


def _add_item_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--items", action="append", default=None,
        help="items to act on: item names or the groups metadata/cameras/ring/stereo/"
             "sensors/all. Validated against the dataset in <spec>. Default: metadata.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="argoverse",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="log every s5cmd invocation")
    parser.add_argument("--root", type=Path, default=al.DEFAULT_ROOT,
                        help=f"local dataset root (default: {al.DEFAULT_ROOT})")
    parser.add_argument("--cache_dir", type=Path, default=ac.CACHE_DIR,
                        help=f"catalog cache directory (default: {ac.CACHE_DIR})")
    parser.add_argument("--catalog", type=Path, default=None,
                        help="explicit catalog json, bypassing the cache")
    parser.add_argument("--refresh", action="store_true",
                        help="rebuild the catalog from S3 before running")
    parser.add_argument("--num_workers", type=int, default=s5cmd.DEFAULT_NUM_WORKERS,
                        help=f"s5cmd parallelism (default: {s5cmd.DEFAULT_NUM_WORKERS})")
    parser.add_argument("--verify_bytes", action="store_true",
                        help="compare local byte counts, not just object counts")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_list = subparsers.add_parser("list", help="list a dataset+split's logs")
    _add_selector_args(p_list)
    p_list.add_argument("--local", action="store_true",
                        help="add a LOCAL column (scans the destination tree)")
    p_list.set_defaults(func=cmd_list)

    p_show = subparsers.add_parser("show", help="show one log's items and local state")
    p_show.add_argument("spec", type=str, help="dataset[/split], e.g. sensor/val")
    p_show.add_argument("log_id_positional", metavar="log_id", type=str)
    p_show.set_defaults(func=cmd_show)

    p_download = subparsers.add_parser("download", help="download items, skipping what is local")
    _add_selector_args(p_download)
    _add_item_args(p_download)
    p_download.add_argument("--dry_run", action="store_true",
                            help="print the plan without transferring")
    p_download.add_argument("--overwrite", action="store_true",
                            help="re-fetch files that already exist locally")
    p_download.add_argument("--yes", "-y", action="store_true",
                            help="skip the size confirmation prompt")
    p_download.add_argument("--confirm_above", type=str, default=DEFAULT_CONFIRM_ABOVE,
                            help=f"prompt above this size (default: {DEFAULT_CONFIRM_ABOVE})")
    p_download.add_argument("--ignore_free_space", action="store_true",
                            help="proceed even if the plan may not fit on disk")
    p_download.add_argument("--print_commands", action="store_true",
                            help="dump the s5cmd batch lines and exit")
    p_download.set_defaults(func=cmd_download)

    p_status = subparsers.add_parser("status", help="report what is already downloaded")
    _add_selector_args(p_status)
    _add_item_args(p_status)
    p_status.add_argument("--detail", action="store_true", help="one row per log")
    p_status.set_defaults(func=cmd_status)

    # `index` deliberately does not take the shared selector flags: a catalog always describes a
    # whole dataset+split, so --city/--limit/--sort would silently do nothing.
    p_index = subparsers.add_parser("index", help="rebuild the cached catalog")
    p_index.add_argument("spec", type=str,
                         help="dataset[/split], e.g. sensor/val, tbv, lidar/train")
    p_index.add_argument("--log_id", action="append", default=None,
                         help="re-list only these literal log ids and merge them into the "
                              "existing catalog (no patterns)")
    p_index.add_argument("--log_id_file", type=Path, default=None,
                         help="file of literal log ids, one per line")
    p_index.add_argument("--out", type=Path, default=None,
                         help="write the catalog here instead of the cache")
    p_index.set_defaults(func=cmd_index)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        return args.func(args)
    except (al.UnknownItemError, al.UnknownSplitError, ac.CatalogError, ad.MissingDataError,
            ad.DownloadFailedError, ad.InsufficientSpaceError, s5cmd.S5cmdNotFoundError,
            s5cmd.S5cmdError, KeyError, ValueError, OSError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"error: {message}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
