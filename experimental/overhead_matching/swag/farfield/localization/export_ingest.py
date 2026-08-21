"""Read a localization export into the filter's inputs.

An export directory carries the Tier-1 inputs and the catalog but not the
Tier-0/Tier-2 outputs a completed run has, so `run_io.read_run` cannot read
it — that function is for reading a run back, this one is for starting one.
It decodes, builds the ENU catalog from the export's anchor, and checks the
preconditions the filter would otherwise hit as assertion failures deep in a
run.

Expected layout:
  export_meta.json          anchor, scenario, matcher, mount-offset provenance
  landmarks.json            LandmarkEntry list (lat/lon)
  tier1_tables.json         CompatibilityTable list
  tier1_measurements.jsonl  TrackletMeasurement events
  tier1_odometry.jsonl      OdometryDelta per keyframe
  truth.jsonl               TruthPose per keyframe (optional; diagnostics)

Mount-offset provenance is ENFORCED here, not merely carried: the export's
bearings have the offset baked in, a 180-degree frame slip produces perfectly
well-formed numbers, and this boundary is the last place the mistake is
cheap. An export whose meta does not state the offset, its source, and the
frame constant it follows is refused. (The old loader declared the field and
then never read it — the exact silent-acceptance this replaces.)
"""

import dataclasses
import math
from pathlib import Path

import msgspec
import numpy as np

from common.python.serialization import msgspec_dec_hook
from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.localization import (
    filter_catalog,
    run_io,
    structs,
)


class ExportMeta(msgspec.Struct):
    schema_version: str
    scenario_name: str
    anchor_lat_deg: float
    anchor_lon_deg: float
    n_keyframes: int
    matcher_version: str
    # Mount-offset provenance, required (see module docstring).
    mount_offset_deg: float
    mount_offset_source: str
    mount_offset_frame: str
    log_lr_scheme: dict | None = None


@dataclasses.dataclass
class ExportData:
    meta: ExportMeta
    frame: geo.RegionFrame
    catalog: filter_catalog.LandmarkCatalog
    landmarks: list
    odometry: list
    measurements: list
    tables: dict
    truth: list

    @property
    def n_keyframes(self) -> int:
        return len(self.odometry) + 1


def load(export_dir: Path, *, max_visible_range_m: float) -> ExportData:
    """max_visible_range_m is required and flows into the run manifest: the
    replay contract depends on the recorded value being the used value."""
    export_dir = Path(export_dir)
    meta = msgspec.json.decode((export_dir / "export_meta.json").read_bytes(),
                               type=ExportMeta)
    if meta.schema_version != structs.SCHEMA_VERSION:
        raise ValueError(
            f"export schema version {meta.schema_version!r}, this build "
            f"reads {structs.SCHEMA_VERSION!r}")
    if meta.mount_offset_frame != geo.MOUNT_OFFSET_FRAME:
        raise ValueError(
            f"export {export_dir} states mount_offset_frame="
            f"{meta.mount_offset_frame!r}; this build consumes only "
            f"{geo.MOUNT_OFFSET_FRAME!r}. A column-0 offset is exactly 180 "
            f"deg out — regenerate the export rather than reinterpreting it.")

    landmarks = msgspec.json.decode(
        (export_dir / "landmarks.json").read_bytes(),
        type=list[structs.LandmarkEntry], dec_hook=msgspec_dec_hook)
    frame = geo.RegionFrame(meta.anchor_lat_deg, meta.anchor_lon_deg)
    east, north = frame.enu_from_latlon(
        np.array([lm.lat_deg for lm in landmarks]),
        np.array([lm.lon_deg for lm in landmarks]))
    catalog = filter_catalog.LandmarkCatalog(
        [lm.landmark_id for lm in landmarks], east, north,
        max_visible_range_m=max_visible_range_m)

    tables = {t.tracklet_id: t for t in msgspec.json.decode(
        (export_dir / "tier1_tables.json").read_bytes(),
        type=list[structs.CompatibilityTable], dec_hook=msgspec_dec_hook)}

    data = ExportData(
        meta=meta, frame=frame, catalog=catalog, landmarks=landmarks,
        odometry=run_io.read_jsonl(export_dir / "tier1_odometry.jsonl",
                                   structs.OdometryDelta),
        measurements=run_io.read_jsonl(
            export_dir / "tier1_measurements.jsonl",
            structs.TrackletMeasurement),
        tables=tables,
        truth=run_io.read_jsonl(export_dir / "truth.jsonl",
                                structs.TruthPose))
    validate(data)
    return data


def validate(data: ExportData) -> None:
    """Fail loudly at the boundary rather than deep inside a run."""
    problems = []
    expected = list(range(1, len(data.odometry) + 1))
    if [o.keyframe_idx for o in data.odometry] != expected:
        problems.append("odometry keyframe indices are not contiguous 1..N")
    if data.truth and len(data.truth) != data.n_keyframes:
        problems.append(
            f"{len(data.truth)} truth poses for {data.n_keyframes} keyframes")

    seen = set()
    for meas in data.measurements:
        key = (meas.tracklet_id, meas.anchor_keyframe_idx)
        if key in seen:
            problems.append(f"duplicate information epoch {key}")
        seen.add(key)
        if meas.tracklet_id not in data.tables:
            problems.append(f"no table for tracklet {meas.tracklet_id!r}")
        if not 0 <= meas.anchor_keyframe_idx < data.n_keyframes:
            problems.append(
                f"measurement anchored outside the run: {key}")
        if not math.isfinite(meas.kappa) or meas.kappa <= 0.0:
            problems.append(f"non-positive kappa on {key}")
        if not 0.0 <= meas.bearing_body_deg < 360.0:
            problems.append(
                f"bearing_body_deg out of the serialized [0, 360) range on "
                f"{key}: {meas.bearing_body_deg}")

    known = set(data.catalog.landmark_ids)
    for table in data.tables.values():
        unknown = [e.landmark_id for e in table.entries
                   if e.landmark_id not in known]
        if unknown:
            problems.append(
                f"table {table.tracklet_id!r} scores {len(unknown)} landmarks "
                f"absent from the catalog, e.g. {unknown[0]!r}")
    if problems:
        raise ValueError("export failed validation:\n  - "
                         + "\n  - ".join(problems))


def region_box(data: ExportData, margin_m: float) -> structs.UniformBoxInit:
    """A uniform prior spanning everything the catalog could explain."""
    return structs.UniformBoxInit(
        east_min_m=float(data.catalog.east_m.min()) - margin_m,
        east_max_m=float(data.catalog.east_m.max()) + margin_m,
        north_min_m=float(data.catalog.north_m.min()) - margin_m,
        north_max_m=float(data.catalog.north_m.max()) + margin_m)


def describe(data: ExportData) -> str:
    box = region_box(data, 0.0)
    tied = sum(1 for t in data.tables.values()
               if len(t.entries) > 1
               and len({e.log_lr for e in t.entries}) == 1)
    kappas = [m.kappa for m in data.measurements]
    sigmas = [math.degrees(1.0 / math.sqrt(k)) for k in kappas]
    return "\n".join([
        f"export      : {data.meta.scenario_name}",
        f"matcher     : {data.meta.matcher_version}",
        f"mount offset: {data.meta.mount_offset_deg:.1f} deg "
        f"({data.meta.mount_offset_source})",
        f"anchor      : {data.meta.anchor_lat_deg:.6f}, "
        f"{data.meta.anchor_lon_deg:.6f}",
        f"keyframes   : {data.n_keyframes}",
        f"catalog     : {data.catalog.n} landmarks spanning "
        f"{(box.east_max_m - box.east_min_m) / 1000:.1f} x "
        f"{(box.north_max_m - box.north_min_m) / 1000:.1f} km",
        f"measurements: {len(data.measurements)} over "
        f"{len({m.tracklet_id for m in data.measurements})} tracklets"
        + (f"; bearing sigma {min(sigmas):.1f}-{max(sigmas):.1f} deg"
           if sigmas else ""),
        f"tables      : {len(data.tables)} ({tied} are ties — disjunctive "
        f"matches with no unique identity)",
        f"truth       : {len(data.truth)} poses"
        + (" (diagnostics only)" if data.truth else " — none"),
    ])
