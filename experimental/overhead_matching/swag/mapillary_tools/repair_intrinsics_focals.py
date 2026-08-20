#!/usr/bin/env python3
"""Repair unphysical focal lengths in an already-converted dataset's intrinsics.

`mapillary_to_vigor.py` does this inline now, but datasets converted before that
-- and any whose `_raw/` staging has been pruned, so stage 3 cannot be re-run --
need it applied in place. The rule and the threshold are imported from the
converter so there is exactly one definition of "implausible".

Idempotent: rows already labelled `substituted_implausible` are left alone and
excluded from the median, so running twice changes nothing.

    python repair_intrinsics_focals.py /data/.../tokyo_bay [more datasets...]
    python repair_intrinsics_focals.py --all          # every built dataset
    python repair_intrinsics_focals.py --all --dry_run
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from experimental.overhead_matching.swag.mapillary_tools.mapillary_to_vigor import MIN_PLAUSIBLE_BASIS, PLAUSIBLE_HFOV_DEG
from experimental.overhead_matching.swag.mapillary_tools.run_farfield_collection import DEFAULT_OUTPUT_BASE

FIELDS = ["idx", "pano_id", "projection", "width", "height", "focal_norm",
          "k1", "k2", "hfov_deg", "vfov_deg", "heading_deg",
          "heading_reference", "heading_source", "focal_source"]


def fovs(focal, width, height):
    """(hfov, vfov) in degrees, matching fov_from_camera_parameters().

    Recomputed from the *stored* width/height in intrinsics.csv. That is the
    same normalization the converter used, because focal_norm is normalized by
    max(w, h) and a uniform resize leaves the ratio unchanged.
    """
    norm = max(width, height)
    return (2 * math.degrees(math.atan((width / norm) / (2 * focal))),
            2 * math.degrees(math.atan((height / norm) / (2 * focal))))


def repair(dataset: Path, dry_run=False):
    path = dataset / "intrinsics.csv"
    if not path.exists():
        print(f"{dataset.name:24s} no intrinsics.csv, skipping")
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print(f"{dataset.name:24s} empty intrinsics.csv, skipping")
        return None
    if rows[0]["projection"] != "perspective":
        print(f"{dataset.name:24s} equirectangular, nothing to repair")
        return None

    lo, hi = PLAUSIBLE_HFOV_DEG
    already = [r for r in rows if r.get("focal_source") == "substituted_implausible"]
    plausible, suspect = [], []
    for row in rows:
        if row.get("focal_source") == "substituted_implausible":
            continue
        if not row["focal_norm"]:
            continue
        hfov, _ = fovs(float(row["focal_norm"]), int(row["width"]), int(row["height"]))
        (plausible if lo <= hfov <= hi else suspect).append(row)

    if not suspect:
        print(f"{dataset.name:24s} all {len(rows)} row(s) plausible"
              + (f" ({len(already)} already substituted)" if already else ""))
        return 0
    if len(plausible) < MIN_PLAUSIBLE_BASIS:
        print(f"{dataset.name:24s} ERROR: only {len(plausible)} plausible row(s), "
              f"too thin a basis for a median ({MIN_PLAUSIBLE_BASIS} required)")
        return -1

    focals = sorted(float(r["focal_norm"]) for r in plausible)
    median = focals[len(focals) // 2]
    bad = sorted(fovs(float(r["focal_norm"]), int(r["width"]), int(r["height"]))[0]
                 for r in suspect)
    share = 100.0 * len(suspect) / len(rows)
    print(f"{dataset.name:24s} {len(suspect)}/{len(rows)} ({share:.1f}%) "
          f"implausible ({bad[0]:.2f}-{bad[-1]:.2f}°, outside {lo}-{hi}°) "
          f"-> focal {median:.4f} "
          f"(≈{fovs(median, 4, 3)[0]:.1f}° on a 4:3 frame)")
    if dry_run:
        return len(suspect)

    ids = {r["pano_id"] for r in suspect}
    for row in rows:
        if row["pano_id"] in ids:
            hfov, vfov = fovs(median, int(row["width"]), int(row["height"]))
            row["focal_norm"] = median
            row["hfov_deg"] = round(hfov, 4)
            row["vfov_deg"] = round(vfov, 4)
            row["focal_source"] = "substituted_implausible"
        else:
            row.setdefault("focal_source", "api")
            row["focal_source"] = row.get("focal_source") or "api"

    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})
    tmp.replace(path)

    meta_path = dataset / "pipeline_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["focals_substituted"] = len(suspect) + len(already)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{'':24s} rewrote {path.name} and pipeline_metadata.json")
    return len(suspect)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true",
                        help=f"every dataset under {DEFAULT_OUTPUT_BASE}")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    targets = list(args.datasets)
    if args.all:
        targets = sorted(d for d in DEFAULT_OUTPUT_BASE.iterdir()
                         if d.is_dir() and not d.name.startswith("_"))
    if not targets:
        parser.error("pass dataset paths or --all")

    failed = [d for d in targets if repair(d, args.dry_run) == -1]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
