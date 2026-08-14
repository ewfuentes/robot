"""M11: assemble a localization export from the M9 (unprivileged) matches.

Builds the export layout `bearing_only_localization/export_ingest.py` reads,
combining:

  - bearings, truth, and body-frame odometry copied UNCHANGED from a base
    export (they are matcher-independent; odometry should already be the
    §5.2 body-frame derivation, e.g. a gps_to_odometry output),
  - CompatibilityTables from the run's `matching/compatibility.json` — the
    M9 whole-map matcher, which never saw a position-based shortlist (see
    the wedge-removal note in object-tracking-pipeline.md), and
  - a catalog of EVERY feather row, because the honest candidate universe
    is the whole map. This is the "full regional catalog + distractors"
    experiment the first localization runs could not claim.

Track merges can postdate matching, so a measured tracklet may be a
superset-merge of a matched one (e.g. LT232_T262_T269 vs LT232_T262). Such
measurements are aliased onto the matched ancestor's table — same physical
object, scored without the later frames — and every alias is printed and
recorded in export_meta.

Usage:
  bazel run //experimental/overhead_matching/swag/landmark_filtering/object_tracking:m11_localization_export -- \
    --run_dir /data/.../runs/r003_full_leg1 \
    --base_export /data/.../runs/r003_full_leg1/localization_export_v02 \
    --output_dir /data/.../runs/r003_full_leg1/localization_export_llm_chunked
"""

import argparse
import json
import shutil
from pathlib import Path

import msgspec

from common.python.serialization import msgspec_dec_hook, msgspec_enc_hook
from experimental.overhead_matching.swag.bearing_only_localization import (
    geodesy,
    structs,
)
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    harbor_catalog,
)

DEFAULT_FEATHER = Path(
    "/data/farfield_matching/boston_harbor_dataset/landmarks/"
    "harbor_osm_enc_trimmed_v1.feather")

# Coarse display type for the viewer's glyphs; the filter itself only uses
# ids and positions.
_TYPE_TAGS = ("seamark:type", "man_made", "leisure", "amenity", "natural",
              "building", "place", "highway")


def _type_key(tags: dict) -> str:
    for key in _TYPE_TAGS:
        if key in tags:
            return f"{key}={tags[key]}"
    return "landmark"


def _load_tables(path: Path) -> dict:
    raw = json.loads(path.read_text())
    return {tid: msgspec.json.decode(json.dumps(table).encode(),
                                     type=structs.CompatibilityTable,
                                     dec_hook=msgspec_dec_hook)
            for tid, table in raw.items()}


def _alias_tables(tables: dict, measured_ids: set) -> tuple[dict, dict]:
    """Cover measured tracklets whose merge name postdates matching."""
    aliases = {}
    for tid in sorted(measured_ids - set(tables)):
        ancestors = [known for known in tables
                     if tid.startswith(known + "_T")]
        if len(ancestors) != 1:
            raise ValueError(
                f"measured tracklet {tid!r} has no table and "
                f"{len(ancestors)} merge ancestors {ancestors!r}; cannot "
                f"alias unambiguously")
        source = tables[ancestors[0]]
        aliases[tid] = ancestors[0]
        tables = dict(tables)
        tables[tid] = structs.CompatibilityTable(
            tracklet_id=tid,
            matcher_version=source.matcher_version,
            entries=source.entries,
            default_log_lr=source.default_log_lr,
            clip_lo=source.clip_lo,
            clip_hi=source.clip_hi,
            status=source.status)
    return tables, aliases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--base_export", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--feather", type=Path, default=DEFAULT_FEATHER)
    args = parser.parse_args()

    base_meta = json.loads((args.base_export / "export_meta.json").read_text())
    if base_meta["schema_version"] != structs.SCHEMA_VERSION:
        raise ValueError(
            f"base export schema {base_meta['schema_version']!r}; expected "
            f"{structs.SCHEMA_VERSION!r} (run gps_to_odometry first)")

    tables = _load_tables(args.run_dir / "matching" / "compatibility.json")
    measured = {
        msgspec.json.decode(line, type=structs.TrackletMeasurement,
                            dec_hook=msgspec_dec_hook).tracklet_id
        for line in (args.base_export
                     / "tier1_measurements.jsonl").read_bytes().splitlines()
        if line.strip()}
    tables, aliases = _alias_tables(tables, measured)
    for merged, ancestor in aliases.items():
        print(f"aliased measurement tracklet {merged} -> table {ancestor}")

    # The whole trimmed map is the candidate universe (no shortlist).
    anchor_lat = base_meta["anchor_lat_deg"]
    anchor_lon = base_meta["anchor_lon_deg"]
    entries = harbor_catalog.load_catalog(args.feather, anchor_lat,
                                          anchor_lon, keep_hulls=False)
    frame = geodesy.RegionFrame(anchor_lat, anchor_lon)
    landmarks = []
    for entry in entries:
        lat, lon = frame.latlon_from_enu(entry.east_m, entry.north_m)
        landmarks.append(structs.LandmarkEntry(
            landmark_id=entry.landmark_id, lat_deg=float(lat),
            lon_deg=float(lon), type_key=_type_key(entry.tags)))
    known = {lm.landmark_id for lm in landmarks}
    scored = {e.landmark_id for t in tables.values() for e in t.entries}
    if scored - known:
        raise ValueError(f"{len(scored - known)} scored landmarks missing "
                         f"from the feather, e.g. {sorted(scored - known)[:3]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("truth.jsonl", "tier1_measurements.jsonl",
                 "tier1_odometry.jsonl"):
        shutil.copy2(args.base_export / name, args.output_dir / name)
    with open(args.output_dir / "landmarks.json", "wb") as f:
        f.write(msgspec.json.encode(landmarks, enc_hook=msgspec_enc_hook))
    with open(args.output_dir / "tier1_tables.json", "wb") as f:
        f.write(msgspec.json.encode(
            sorted(tables.values(), key=lambda t: t.tracklet_id),
            enc_hook=msgspec_enc_hook))

    matcher_versions = {t.matcher_version for t in tables.values()}
    meta = dict(base_meta)
    meta.update(
        schema_version=structs.SCHEMA_VERSION,
        scenario_name=base_meta["scenario_name"].replace("_TEMP", "")
        + "_llm_chunked",
        matcher_version="+".join(sorted(matcher_versions)),
        log_lr_scheme={"source": "matching/compatibility.json (M9)",
                       "aliased_tracklets": aliases})
    (args.output_dir / "export_meta.json").write_text(
        json.dumps(meta, indent=1))

    print(f"{len(landmarks)} landmarks, {len(tables)} tables "
          f"({len(aliases)} aliased), matcher {meta['matcher_version']}")
    print(f"export written to {args.output_dir}")


if __name__ == "__main__":
    main()
