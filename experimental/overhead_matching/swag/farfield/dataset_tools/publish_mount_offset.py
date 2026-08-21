"""Publish a calibrated mount offset into a dataset's pipeline_metadata.json.

This is the ONLY writer of dataset `mount_offset` blocks. The estimators
(`calibration/mount_offset_sweep.py`, `calibration/sun_offset_check.py`)
write sidecars in the artifact tree and never touch the frozen dataset lane
(REORG.md rule 7); publishing is a separate, deliberate act performed here,
with the accuracy-validated guard and the checksum regeneration the old
in-sweep writer printed a reminder about but never performed — which left
every published dataset failing its own `checksums.sha256`.

What gets written (shaped to satisfy `dataset.mount_offset_record` by
construction — the block is validated through it before the file is touched):

- `mount_offset_deg`, from the sidecar;
- `frame`: `geometry.MOUNT_OFFSET_FRAME` (the sidecar must already carry it —
  a sidecar quoting any other frame is refused, because a column-0 offset is
  exactly 180 degrees out);
- `applied_to_heading_deg: false`: publishing records the calibration, it
  does not rewrite intrinsics.csv, so heading_deg does NOT include it;
- `convention`: `geometry.MOUNT_OFFSET_CONVENTION` verbatim;
- a `source` pointer back to the sidecar (path, generator, run, git commit)
  and a `published_by` provenance stamp.

**Refuses to overwrite an accuracy-validated offset.** A block carrying
`accuracy_validated: true` was earned by evidence external to any sweep
(boston_harbor_leg1's 214.0 deg came from a surveyed building over 72
keyframes); replacing it with an unvalidated number silently desynchronises
the metadata from every artifact already built against it.
`--supersede_validated` forces it, and even then the old block is preserved
under `superseded`.

**Refuses an unusable sidecar.** The estimators publish their own verdict
(`usable`); an angle whose curve or sun model rejected it must not become
dataset truth because someone forgot to read the verdict.

    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:publish_mount_offset -- \\
        --dataset boston_harbor_leg2 \\
        --from_sidecar /data/.../runs/r004/mount_offset_sweep.json
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

from experimental.overhead_matching.swag.farfield import dataset as ds_lib
from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield import provenance
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    checksums,
)

GENERATOR = "farfield/dataset_tools/publish_mount_offset.py"


def load_sidecar(path: Path) -> dict:
    """Read and validate an estimator sidecar; SystemExit on anything
    unpublishable.

    The three refusals here are the three ways a wrong angle has actually
    reached a dataset: an estimator whose own verdict rejected the number, a
    sidecar quoting a different frame (the 180-degree trap), and a record
    with no numeric angle at all.
    """
    if not path.exists():
        raise SystemExit(f"sidecar not found: {path}")
    record = json.loads(path.read_text())
    problems = []
    if record.get("frame") != geo.MOUNT_OFFSET_FRAME:
        problems.append(
            f"frame={record.get('frame')!r} (must be "
            f"{geo.MOUNT_OFFSET_FRAME!r}; a column-0 offset is exactly 180 "
            f"deg out — see geometry.MOUNT_OFFSET_CONVENTION)")
    if record.get("usable") is not True:
        problems.append(
            f"usable={record.get('usable')!r} — the estimator's own verdict "
            f"({record.get('verdict')!r}: {record.get('detail')!r}) does not "
            f"support publishing this angle")
    offset = record.get("mount_offset_deg", record.get("offset_deg"))
    if not isinstance(offset, (int, float)) or isinstance(offset, bool):
        problems.append("no numeric mount_offset_deg / offset_deg")
    if problems:
        raise SystemExit(
            f"refusing to publish from {path}:\n"
            + "\n".join(f"  - {p}" for p in problems))
    record["_offset_deg"] = float(offset)
    return record


def build_block(sidecar_path: Path, sidecar: dict) -> dict:
    return {
        "mount_offset_deg": sidecar["_offset_deg"],
        "frame": geo.MOUNT_OFFSET_FRAME,
        # Publishing records the calibration; it does not rewrite the
        # intrinsics heading_deg column, so the offset is NOT baked in and
        # consumers must apply it (geometry.apply_mount_offset).
        "applied_to_heading_deg": False,
        "status": str(sidecar.get("verdict", "published")),
        "accuracy_validated": False,
        "convention": geo.MOUNT_OFFSET_CONVENTION,
        "source": {
            "sidecar": str(sidecar_path),
            "generator": sidecar.get("generator"),
            "run": sidecar.get("run"),
            "detail": sidecar.get("detail"),
            "sidecar_git_commit": sidecar.get("git_commit"),
        },
        "published_by": {
            "generator": GENERATOR,
            "git_commit": provenance.git_commit(),
            "argv": list(sys.argv),
            "created": datetime.datetime.now(datetime.timezone.utc)
                       .isoformat(timespec="seconds"),
        },
    }


def publish(dataset_base: Path, sidecar_path: Path,
            supersede_validated: bool) -> dict:
    """Validate, guard, write the block, regenerate checksums.

    Returns the written block. SystemExit on every refusal, with the file
    untouched.
    """
    sidecar = load_sidecar(sidecar_path)
    metadata_path = Path(dataset_base) / "pipeline_metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"{metadata_path} not found — is {dataset_base} a "
                         f"dataset directory?")
    meta = json.loads(metadata_path.read_text())
    previous = meta.get("mount_offset") or {}

    if previous.get("accuracy_validated") and not supersede_validated:
        old = previous.get("mount_offset_deg")
        lines = [
            f"REFUSING to overwrite {metadata_path}:",
            f"  existing: {old} deg, accuracy_validated=true",
            f"    source: {previous.get('source', 'unrecorded')}",
            f"  sidecar:  {sidecar['_offset_deg']} deg "
            f"(accuracy_validated=false)",
        ]
        if isinstance(old, (int, float)):
            delta = abs(float(geo.circular_diff_deg(
                sidecar["_offset_deg"], old)))
            lines.append(f"  they differ by {delta:.1f} deg")
        lines.append(
            "  The existing value has external evidence this estimate does "
            "not. Pass --supersede_validated only if you mean to replace "
            "it; the old block will be preserved under 'superseded'.")
        raise SystemExit("\n".join(lines))

    block = build_block(sidecar_path, sidecar)
    # Keep whatever was there. A calibration's history is part of its
    # evidence, and artifacts already built under the old value need it to
    # stay legible.
    if previous:
        block["superseded"] = previous

    # Self-check: the block must round-trip through the one consumer-side
    # validator before it lands on disk. A publish that writes a block
    # mount_offset_record refuses is worse than no publish.
    trial = dict(meta)
    trial["mount_offset"] = block
    ds_lib.mount_offset_record(trial, dataset_base)

    meta["mount_offset"] = block
    metadata_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote mount_offset {block['mount_offset_deg']:.2f} deg "
          f"(applied_to_heading_deg=false) to {metadata_path}")
    if previous:
        print(f"  previous value ({previous.get('mount_offset_deg')} deg) "
              f"kept under mount_offset.superseded")

    # The regeneration the old in-sweep writer always skipped: the metadata
    # file is covered by checksums.sha256, so publishing without this leaves
    # the dataset failing its own integrity check.
    n_sums = checksums.regenerate(dataset_base)
    if n_sums is None:
        print(f"  note: {dataset_base} carries no "
              f"{checksums.CHECKSUM_FILE}; nothing to regenerate")
    else:
        print(f"  regenerated {checksums.CHECKSUM_FILE} over {n_sums} files")
    return block


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    paths_lib.add_arguments(parser)
    parser.add_argument("--from_sidecar", type=Path, required=True,
                        help="estimator sidecar JSON (mount_offset_sweep."
                             "json or sun_offset_check.json in a run dir)")
    parser.add_argument("--supersede_validated", action="store_true",
                        help="replace an accuracy-validated block (the old "
                             "block is preserved under 'superseded')")
    args = parser.parse_args(argv)
    paths = paths_lib.resolve(parser, args, require=("dataset_base",))
    publish(paths.dataset_base, args.from_sidecar, args.supersede_validated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
