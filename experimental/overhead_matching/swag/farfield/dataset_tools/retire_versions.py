"""Move superseded artifact versions under ``<lane>/<dataset>/retired/``.

Everything in the lane directory that is not a kept version (or one of its
sidecars: ``<version>.<suffix>`` and ``<version>--...``) moves, manifest and
all, and a row is appended to ``retired/RETIRED.md`` saying what replaced it.
Index refresh skips ``retired/``. Older runs and builds recorded the absolute
path of what they consumed; those records keep resolving only by hand, which
is the trade accepted on 2026-09-10 for a lane that shows its current state.
"""

import argparse
import datetime
import shutil
import sys
from pathlib import Path

RETIRED_DIRECTORY = "retired"
LEDGER_NAME = "RETIRED.md"
_LEDGER_HEADER = (
    "# Retired versions\n\n"
    "Moved out of the live lane by `dataset_tools:retire_versions`; "
    "manifests are intact. Runs and builds that recorded the old absolute "
    "path resolve these only by hand.\n\n"
    "| date | version | replaced by | reason |\n|---|---|---|---|\n")


def _is_kept(name: str, keep: set[str]) -> bool:
    return name in keep or any(
        name.startswith(f"{kept}.") or name.startswith(f"{kept}--")
        for kept in keep)


def retire(lane_dir: Path, *, keep: set[str], replaced_by: str,
           reason: str, dry_run: bool = False) -> list[str]:
    """Move every non-kept version directory; return the names moved."""
    if not lane_dir.is_dir():
        raise SystemExit(f"not a lane directory: {lane_dir}")
    missing = sorted(k for k in keep if not (lane_dir / k).is_dir())
    if missing:
        raise SystemExit(f"kept versions are not in {lane_dir}: {missing}")
    moved = sorted(
        entry.name for entry in lane_dir.iterdir()
        if entry.is_dir() and not entry.is_symlink()
        and not entry.name.startswith(".")
        and entry.name != RETIRED_DIRECTORY
        and not _is_kept(entry.name, keep))
    if dry_run or not moved:
        return moved
    retired_dir = lane_dir / RETIRED_DIRECTORY
    retired_dir.mkdir(exist_ok=True)
    for name in moved:
        if (retired_dir / name).exists():
            raise SystemExit(f"{retired_dir / name} already exists")
    for name in moved:
        shutil.move(str(lane_dir / name), str(retired_dir / name))
    ledger = retired_dir / LEDGER_NAME
    if not ledger.exists():
        ledger.write_text(_LEDGER_HEADER)
    today = datetime.date.today().isoformat()
    with ledger.open("a", encoding="utf-8") as handle:
        for name in moved:
            handle.write(f"| {today} | `{name}` | `{replaced_by}` | {reason} |\n")
    return moved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farfield_root", type=Path, required=True)
    parser.add_argument("--lane", required=True, help="artifact kind, e.g. catalogs")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--keep", action="append", required=True,
                        help="version to keep live (repeatable)")
    parser.add_argument("--replaced_by", required=True,
                        help="the version that supersedes the moved ones")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    lane_dir = args.farfield_root / "artifacts" / args.lane / args.dataset
    moved = retire(lane_dir, keep=set(args.keep), replaced_by=args.replaced_by,
                   reason=args.reason, dry_run=args.dry_run)
    verb = "would move" if args.dry_run else "moved"
    print(f"{lane_dir}: {verb} {len(moved)} version(s) to {RETIRED_DIRECTORY}/: "
          + ", ".join(moved))
    return 0


if __name__ == "__main__":
    sys.exit(main())
