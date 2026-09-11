"""Collate the triage state of a dataset collection into one markdown table.

The triage tools each leave their answer in a different place --
`pipeline_metadata.json`, `_manifests/recording_seams.json`, and
`_manifests/vehicle_anchor.json` (written by their annotators) -- and the
useful judgements only appear when they are read side by side. This writes
that joined view so nobody has to open every JSON file to get it.

Every column is read back from the datasets themselves; nothing here is
hand-maintained. Regenerate after any change.

    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:dataset_status_table -- \\
        --dataset_path /path/to/datasets/*
"""

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

from experimental.overhead_matching.swag.farfield import provenance

HEADER = ("| dataset | proj | frames | km | trims | anchor | seams | "
          "worst m/s |")

# What each column is read from -- facts about the mechanism, not judgements
# about any particular dataset.
LEGEND = """\
Columns and their sources:

- **proj / frames / km / trims**: `pipeline_metadata.json` and
  `frames_gps.csv`.
- **anchor**: verdict from `_manifests/vehicle_anchor.json`
  (`detect_vehicle_anchor`).
- **seams / worst m/s**: `_manifests/recording_seams.json`
  (`annotate_recording_seams`), and the largest implied speed across its seams.

`—` means the producing tool has not been run on that dataset.
`CORRUPT JSON` means the expected file exists but cannot be decoded; it is
never treated as equivalent to a tool that has not run.
"""

CORRUPT = "CORRUPT JSON"


class CorruptJsonError(ValueError):
    pass


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CorruptJsonError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorruptJsonError(f"JSON in {path} is not an object")
    return value


def collect_one(ds: Path) -> dict | None:
    try:
        meta = read_json(ds / "pipeline_metadata.json")
    except CorruptJsonError:
        return {
            "name": ds.name, "proj": CORRUPT, "n": "—", "km": None,
            "trims": "—", "anchor": "—", "seams": "—", "worst": None,
        }
    if meta is None:
        return None
    gps_path = ds / "frames_gps.csv"
    gps = (list(csv.DictReader(open(gps_path))) if gps_path.exists() else [])
    try:
        seams = (read_json(ds / "_manifests" / "recording_seams.json")
                 or {})
        seams_value = seams.get("n_seams") if seams else None
        worst = max([s.get("implied_speed_mps") or 0.0
                     for s in seams.get("seams", [])], default=0.0)
    except CorruptJsonError:
        seams_value, worst = CORRUPT, None
    try:
        anchor = read_json(ds / "_manifests" / "vehicle_anchor.json") or {}
        anchor_value = anchor.get("verdict", "—")
    except CorruptJsonError:
        anchor_value = CORRUPT
    km = 0.0
    if gps:
        dists = [float(r["dist_m"]) for r in gps]
        km = (dists[-1] - dists[0]) / 1000.0
    return {
        "name": ds.name,
        "proj": str(meta.get("projection", "?"))[:5],
        "n": meta.get("num_images", len(gps)),
        "km": km,
        "trims": len(meta.get("trims", [])),
        "anchor": anchor_value,
        "seams": seams_value,
        "worst": worst,
    }


def render(rows) -> str:
    lines = [HEADER, "|" + "---|" * 8]
    for r in sorted(rows, key=lambda x: x["name"]):
        km = "—" if r["km"] is None else f"{r['km']:.1f}"
        worst = "—" if r["worst"] is None else f"{r['worst']:.0f}"
        lines.append(
            f"| `{r['name']}` | {r['proj']} | {r['n']} | {km} | "
            f"{r['trims']} | {r['anchor']} | "
            f"{r['seams'] if r['seams'] is not None else '—'} | "
            f"{worst} |")
    return "\n".join(lines)


def preamble() -> str:
    """Neutral, generated header: what this file is and how it was made.

    Dataset-specific notes and tuning claims belong in dataset-owned records.
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
