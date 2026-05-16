"""Dump scalar tags + recent values from a tensorboard event directory.

Usage:
  bazel run //experimental/overhead_matching/swag/scripts:dump_tb_scalars -- \\
      /path/to/run_dir [tag_filter]

If tag_filter is given, only tags containing that substring are printed in detail.
"""
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main(run_dir: Path, tag_filter: str | None = None):
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    tags = sorted(ea.Tags()["scalars"])
    print(f"Scalar tags ({len(tags)}):")
    for t in tags:
        print(f"  {t}")
    print()

    selected = tags if tag_filter is None else [t for t in tags if tag_filter in t]
    for tag in selected:
        events = ea.Scalars(tag)
        if not events:
            continue
        print(f"[{tag}] {len(events)} steps")
        # Print up to ~40 evenly-spaced rows so we can eyeball the trajectory.
        stride = max(1, len(events) // 40)
        for e in events[::stride]:
            print(f"  step={e.step:>6}  value={e.value:.6f}")
        last = events[-1]
        print(f"  step={last.step:>6}  value={last.value:.6f}  (last)")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    run_dir = Path(sys.argv[1])
    tag_filter = sys.argv[2] if len(sys.argv) > 2 else None
    main(run_dir, tag_filter)
