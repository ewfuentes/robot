"""Regenerate the navigation index chain across the farfield data root.

Every stage calls indexes.refresh() itself after writing; this CLI exists
for manual refreshes and for the data migration.

  bazel run //experimental/overhead_matching/swag/farfield/viewers:refresh_indexes
  bazel run ...:refresh_indexes -- --data_root /mnt/mirror/farfield_matching
"""

import argparse
from pathlib import Path

from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.viewers import indexes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=None,
                        help=f"default: ${paths_lib.ROOT_ENV_VAR} or "
                             f"{paths_lib.DEFAULT_ROOT}")
    args = parser.parse_args()
    root = args.data_root or paths_lib.default_root()
    result = indexes.refresh(root)
    print(f"wrote {len(result['written'])} index pages under {root}")
    for path in result["skipped"]:
        print(f"  left alone (not ours): {path}")


if __name__ == "__main__":
    main()
