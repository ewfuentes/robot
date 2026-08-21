"""Collate the triage state of a dataset collection into one markdown table.

The triage tools each leave their answer in a different place -- the
mount-offset block in `pipeline_metadata.json` (written only by
`publish_mount_offset`), `_manifests/recording_seams.json` and
`_manifests/vehicle_anchor.json` (written by their annotators) -- and the
useful judgements only appear when they are read side by side. This writes
that joined view so nobody has to open fourteen JSON files to get it.

Every column is read back from the datasets themselves; nothing here is
hand-maintained. Regenerate after any change.

    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:dataset_status_table -- \\
        --dataset_path /data/farfield_matching/datasets/*
"""

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield import provenance

HEADER = ("| dataset | proj | frames | km | trims | offset | frame ok | "
          "applied | status | validated | anchor | seams | worst m/s |")

# What each column is read from -- facts about the mechanism, not judgements
# about any particular dataset (the old preamble narrated one collection's
# calibration history, which was stale the day another collection was added).
LEGEND = """\
Columns and their sources:

- **proj / frames / km / trims**: `pipeline_metadata.json` and
  `frames_gps.csv`.
- **offset / frame ok / applied / status / validated**: the dataset's
  `mount_offset` block. `frame ok` checks the block's `frame` tag against
  `geometry.MOUNT_OFFSET_FRAME`; `applied` is its `applied_to_heading_deg`;
  `validated` is `accuracy_validated`. Only `publish_mount_offset` writes
  this block.
- **anchor**: verdict from `_manifests/vehicle_anchor.json`
  (`detect_vehicle_anchor`).
- **seams / worst m/s**: `_manifests/recording_seams.json`
  (`annotate_recording_seams`; falls back to a legacy in-metadata
  `recording_seams` block), and the largest implied speed across its seams.

`—` means the producing tool has not been run on that dataset.
"""


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def collect_one(ds: Path) -> dict | None:
    meta = read_json(ds / "pipeline_metadata.json")
    if meta is None:
        return None
    gps_path = ds / "frames_gps.csv"
    gps = (list(csv.DictReader(open(gps_path))) if gps_path.exists() else [])
    mo = meta.get("mount_offset") or {}
    # The sidecar is the current home; the in-metadata block is the legacy
    # one (and what trim_dataset rebases on datasets that still carry it).
    seams = (read_json(ds / "_manifests" / "recording_seams.json")
             or meta.get("recording_seams") or {})
    anchor = read_json(ds / "_manifests" / "vehicle_anchor.json") or {}
    worst = max([s.get("implied_speed_mps") or 0.0
                 for s in seams.get("seams", [])], default=0.0)
    km = 0.0
    if gps:
        dists = [float(r["dist_m"]) for r in gps]
        km = (dists[-1] - dists[0]) / 1000.0
    frame_ok = (mo.get("frame") == geo.MOUNT_OFFSET_FRAME) if mo else None
    return {
        "name": ds.name,
        "proj": str(meta.get("projection", "?"))[:5],
        "n": meta.get("num_images", len(gps)),
        "km": km,
        "trims": len(meta.get("trims", [])),
        "off": mo.get("mount_offset_deg"),
        "frame_ok": frame_ok,
        "applied": mo.get("applied_to_heading_deg"),
        "status": mo.get("status") if mo else None,
        "validated": mo.get("accuracy_validated") if mo else None,
        "anchor": anchor.get("verdict", "—"),
        "seams": seams.get("n_seams") if seams else None,
        "worst": worst,
    }


def render(rows) -> str:
    def num(value, spec, suffix=""):
        return "—" if value is None else format(value, spec) + suffix

    def flag(value):
        return {None: "—", True: "yes", False: "**no**"}[value]

    lines = [HEADER, "|" + "---|" * 13]
    for r in sorted(rows, key=lambda x: x["name"]):
        lines.append(
            f"| `{r['name']}` | {r['proj']} | {r['n']} | {r['km']:.1f} | "
            f"{r['trims']} | {num(r['off'], '.1f', '°')} | "
            f"{flag(r['frame_ok'])} | {flag(r['applied'])} | "
            f"{r['status'] or '—'} | {flag(r['validated'])} | "
            f"{r['anchor']} | "
            f"{r['seams'] if r['seams'] is not None else '—'} | "
            f"{r['worst']:.0f} |")
    return "\n".join(lines)


def preamble() -> str:
    """Neutral, generated header: what this file is and how it was made.

    Deliberately carries no dataset-specific narrative or tuning claims --
    those live with the data that earned them, and a hardcoded story here is
    stale the day the collection changes (REORG.md rule 8).
    """
    created = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")
    return (
        "# Dataset triage status\n\n"
        "Generated file — do not edit. Every column is read back from the\n"
        "datasets themselves; regenerate after any change with:\n\n"
        "    bazel run //experimental/overhead_matching/swag/farfield/"
        "dataset_tools:dataset_status_table\n\n"
        f"- generated: {created}\n"
        f"- git_commit: {provenance.git_commit()}\n"
        f"- argv: `{' '.join(sys.argv)}`\n\n"
        + LEGEND
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", nargs="+", required=True, type=Path,
                   help="dataset directories to include")
    p.add_argument("--output", type=Path, default=None,
                   help="markdown path (default: <common parent>/_manifests/"
                        "STATUS.md when every dataset shares one parent)")
    args = p.parse_args(argv)

    rows = []
    for ds in args.dataset_path:
        row = collect_one(ds)
        if row is None:
            print(f"{ds}: no pipeline_metadata.json, skipping")
            continue
        rows.append(row)
    if not rows:
        p.error("none of the given paths is a dataset directory")

    out = args.output
    if out is None:
        parents = {Path(ds).resolve().parent for ds in args.dataset_path}
        if len(parents) != 1:
            p.error("datasets span multiple parent directories; pass "
                    "--output explicitly")
        out = parents.pop() / "_manifests" / "STATUS.md"

    table = render(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(preamble() + "\n" + table + "\n")
    print(table)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
