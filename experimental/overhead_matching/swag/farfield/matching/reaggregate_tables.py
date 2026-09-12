"""Re-aggregate a published landmark_matches artifact without new model calls.

The shipped ``digest_chunks`` matcher labels each endorsed row ``instance`` or
``category``. As published, a category-labelled row carries the model's kind
confidence as if it were an identity claim, and only the rows of that kind
the model happened to endorse in some slice carry it at all (the per-slice
lottery, see docs/farfield/localization.md). ``catexpand_divided`` re-reads
the same responses under the ``category_chunks`` encoding rule: every kind a
track was matched to expands to every catalog row of that kind at c/N under
the category clip floor, instance rows keep their confidence, and the default
follows the derived global no-match over the resulting rows. ``baseline``
rebuilds compatibility.json unchanged and is the plumbing check.

Output is a JSON list of CompatibilityTable for
``localization:grid_filter --tables_override``. This is the 2026-09-12
experiment's table path (ported from Harel's scratch ``reaggregate.py``).
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import msgspec

from experimental.overhead_matching.swag.farfield.localization import structs
from experimental.overhead_matching.swag.farfield.matching import (
    match_landmarks as ml,
)

POLICIES = ("baseline", "catexpand_divided")
# A kind endorsed at confidence c is spread over the kind's N catalog rows at
# c/N each, so the kind's rows collectively carry ~c under the filter's odds
# split and a row-level identity claim keeps its weight. c/N for a large kind
# sits far below logit(0.05) = -2.9, so the table floor must drop below the
# matcher's -4 or the whole expansion would collapse onto the default and read
# as "unendorsed".
CATEGORY_CLIP_LO = -12.0
# First present key names a signature's kind. Ordered so that a wind turbine
# tagged building=yes is a generator, not a building, and a bridge carrying a
# highway is a bridge.
CATEGORY_KEYS = (
    "generator:source", "power", "man_made", "aeroway", "natural", "water",
    "landuse", "leisure", "amenity", "tourism", "historic", "railway",
    "waterway", "industrial", "bridge", "highway", "military", "building",
    "place", "seamark:type", "object_class")
UNCATEGORISED = "other"


def signature_category(tags: dict) -> str:
    """The kind a category match to this signature expands over."""
    for key in CATEGORY_KEYS:
        if key in tags:
            return f"{key}={tags[key]}"
    return UNCATEGORISED


def reaggregate(matches: dict, signatures: dict,
                tables: list, policy: str) -> list:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    divided = policy == "catexpand_divided"
    by_tracklet = {table.tracklet_id: table for table in tables}
    kind_of = {sid: signature_category(entry["canonical_tags"])
               for sid, entry in signatures.items()}
    kind_signatures = defaultdict(list)
    kind_rows = Counter()
    for sid, kind in kind_of.items():
        kind_signatures[kind].append(sid)
        kind_rows[kind] += len(signatures[sid]["landmark_ids"])
    clip_lo = CATEGORY_CLIP_LO if divided else -ml.DEFAULT_CLIP
    out = []
    for tracklet_id, record in matches.items():
        base = by_tracklet[tracklet_id]
        rows = {}   # landmark_id -> confidence
        kinds = {}  # kind -> best category confidence
        for match in record["matches"]:
            confidence = match["aggregate_confidence"]
            kind = kind_of[match["signature_id"]]
            # A category row of an unrecognised kind has nothing to expand
            # over and is kept as the model reported it.
            if (divided and match["match_type"] == "category"
                    and kind != UNCATEGORISED):
                kinds[kind] = max(kinds.get(kind, 0.0), confidence)
            else:
                rows[match["landmark_id"]] = max(
                    rows.get(match["landmark_id"], 0.0), confidence)
        for kind, confidence in kinds.items():
            per_row = confidence / kind_rows[kind]
            for sid in kind_signatures[kind]:
                for landmark_id in signatures[sid]["landmark_ids"]:
                    rows.setdefault(landmark_id, per_row)
        # The publish loop's global_no_match rule applied to the rows as
        # encoded: for a kind-only track this is 1 - c/N, so its default sits
        # well below the expanded rows and they stay endorsed.
        no_match = (round(1.0 - max(rows.values()), 4) if rows
                    else record["aggregate_no_match_confidence"])
        default_log_lr = ml.to_log_lr(
            max(1e-4, 1.0 - no_match) / max(1, len(signatures)),
            clip_lo=clip_lo)
        out.append(ml.to_compatibility_table(
            tracklet_id,
            {lid: ml.to_log_lr(c, clip_lo=clip_lo) for lid, c in rows.items()},
            matcher_version=(base.matcher_version if policy == "baseline"
                             else f"{base.matcher_version}+reagg_{policy}"),
            default_log_lr=default_log_lr, status=base.status,
            clip_lo=clip_lo))
    return out


def load_artifact(matches_dir: Path):
    matches = json.loads((matches_dir / "matches.json").read_text())
    signatures = json.loads((matches_dir / "signatures.json").read_text())
    tables = msgspec.json.decode(
        (matches_dir / "compatibility.json").read_bytes(),
        type=list[structs.CompatibilityTable])
    return matches, signatures, tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches_dir", type=Path, required=True,
                        help="published landmark_matches artifact directory")
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    matches, signatures, tables = load_artifact(args.matches_dir)
    out = reaggregate(matches, signatures, tables, args.policy)
    args.out.write_bytes(msgspec.json.encode(out))
    n_rows = sum(len(table.entries) for table in out)
    print(f"{args.policy}: {len(out)} tables, {n_rows} endorsed rows -> "
          f"{args.out}")
    if args.policy == "baseline":
        shipped = {t.tracklet_id: t for t in tables}
        differing = sum(
            1 for t in out
            if {(e.landmark_id, round(e.log_lr, 6)) for e in t.entries}
            != {(e.landmark_id, round(e.log_lr, 6))
                for e in shipped[t.tracklet_id].entries}
            or abs(t.default_log_lr - shipped[t.tracklet_id].default_log_lr)
            > 1e-6)
        print(f"baseline check: {differing} tables differ from "
              "compatibility.json")


if __name__ == "__main__":
    main()
