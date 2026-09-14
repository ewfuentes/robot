"""Natural-closure availability for existing localization artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.calibration import audit_io
from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
)
from experimental.overhead_matching.swag.farfield.localization import structs
from experimental.overhead_matching.swag.farfield.tracking import tracklets


SIDECAR_SCHEMA = "farfield_natural_closure_release_schedule/v1"
_SIDECAR_KEYS = frozenset({
    "schema",
    "localization_inputs",
    "policy",
    "final_keyframe_idx",
    "release_keyframe_by_tracklet",
})
_POLICY = {
    "closed_track_release": "source_track.last_keyframe",
    "alive_track_release": "final_keyframe_idx",
    "tracklet_release": "atomic",
    "measurement_anchors": "preserved",
}


@dataclass(frozen=True)
class TrackRelease:
    """One matched track and all of its historical measurements."""

    release_keyframe_idx: int
    tracklet_id: str
    measurements: tuple[structs.TrackletMeasurement, ...]
    table: structs.CompatibilityTable


def _schedule_from_releases(
        releases: Mapping[str, int],
        measurements: Iterable[structs.TrackletMeasurement],
        tables: Mapping[str, structs.CompatibilityTable],
        final_keyframe_idx: int) -> tuple[TrackRelease, ...]:
    if (isinstance(final_keyframe_idx, bool)
            or not isinstance(final_keyframe_idx, int)
            or final_keyframe_idx < 0):
        raise ValueError("final_keyframe_idx must be a nonnegative integer")
    if not isinstance(releases, Mapping):
        raise ValueError("release_keyframe_by_tracklet must be an object")
    for tracklet_id, release in releases.items():
        if not isinstance(tracklet_id, str) or not tracklet_id:
            raise ValueError("release tracklet IDs must be non-empty strings")
        if (isinstance(release, bool) or not isinstance(release, int)
                or not 0 <= release <= final_keyframe_idx):
            raise ValueError(
                f"tracklet {tracklet_id!r} has invalid release keyframe "
                f"{release!r}")

    accepted_ids = set(releases)
    table_ids = set(tables)
    if table_ids != accepted_ids:
        raise ValueError(
            "matched tables do not exactly cover accepted tracklets: "
            f"missing={sorted(accepted_ids - table_ids)}, "
            f"extra={sorted(table_ids - accepted_ids)}")
    for tracklet_id, table in tables.items():
        if table.tracklet_id != tracklet_id:
            raise ValueError(
                f"table key {tracklet_id!r} disagrees with its tracklet_id "
                f"{table.tracklet_id!r}")

    by_tracklet = defaultdict(list)
    seen = set()
    for measurement in measurements:
        tracklet_id = measurement.tracklet_id
        if tracklet_id not in releases:
            raise ValueError(
                f"measurement names unaccepted tracklet {tracklet_id!r}")
        anchor = measurement.anchor_keyframe_idx
        if (isinstance(anchor, bool) or not isinstance(anchor, int)
                or not 0 <= anchor <= releases[tracklet_id]):
            raise ValueError(
                f"measurement for {tracklet_id!r} has anchor {anchor!r} "
                f"after release {releases[tracklet_id]}")
        key = (tracklet_id, anchor)
        if key in seen:
            raise ValueError(f"duplicate information epoch {key}")
        seen.add(key)
        by_tracklet[tracklet_id].append(measurement)

    missing = sorted(accepted_ids - set(by_tracklet))
    if missing:
        raise ValueError(
            f"accepted tracklets have no localization measurements: {missing}")

    return tuple(sorted(
        (TrackRelease(
            release_keyframe_idx=release,
            tracklet_id=tracklet_id,
            measurements=tuple(sorted(
                by_tracklet[tracklet_id],
                key=lambda item: item.anchor_keyframe_idx)),
            table=tables[tracklet_id])
         for tracklet_id, release in releases.items()),
        key=lambda item: (item.release_keyframe_idx, item.tracklet_id)))


def natural_closure_schedule(
        accepted_tracklets: Iterable[tracklets.AcceptedTracklet],
        measurements: Iterable[structs.TrackletMeasurement],
        tables: Mapping[str, structs.CompatibilityTable],
        final_keyframe_idx: int) -> tuple[TrackRelease, ...]:
    """Return one atomic release per accepted, matched source track.

    Closed tracks become available at their source ``last_keyframe``. Tracks
    still alive in a finite artifact become available at the explicit EOF.
    Measurement objects retain their historical anchor keyframes.
    """
    releases = {}
    for item in accepted_tracklets:
        tracklet_id = item.tracklet_id
        if tracklet_id in releases:
            raise ValueError(f"duplicate accepted tracklet {tracklet_id!r}")
        source = item.source_track
        status = source.get("status")
        last = source.get("last_keyframe")
        if (isinstance(last, bool) or not isinstance(last, int)
                or last < 0):
            raise ValueError(
                f"tracklet {tracklet_id!r} has invalid last_keyframe {last!r}")
        if status == "closed":
            release = last
        elif status == "alive":
            release = final_keyframe_idx
        else:
            raise ValueError(
                f"tracklet {tracklet_id!r} has invalid status {status!r}")
        releases[tracklet_id] = release

    return _schedule_from_releases(
        releases, measurements, tables, final_keyframe_idx)


def _one_upstream(manifest: artifact.ArtifactManifest, kind: str,
                  label: str) -> artifact.ArtifactRef:
    matches = [item for item in manifest.upstreams if item.kind == kind]
    if len(matches) != 1:
        raise ValueError(
            f"{label} must bind exactly one {kind} artifact; "
            f"found {len(matches)}")
    return matches[0]


def natural_closure_schedule_from_export(
        data: export_ingest.ExportData) -> tuple[TrackRelease, ...]:
    """Resolve the export's exact audited tracks and build its schedule."""
    inputs_label = "localization_inputs"
    bearing_ref = _one_upstream(
        data.manifest, paths_lib.BEARING_OBSERVATIONS, inputs_label)
    opened_bearing = artifact.open_artifact(
        bearing_ref.path,
        expected_kind=paths_lib.BEARING_OBSERVATIONS,
        expected_dataset=data.artifact_ref.dataset,
        expected_version=bearing_ref.version)
    if opened_bearing != bearing_ref:
        raise ValueError(
            "localization_inputs bearing_observations reference is stale")

    bearing_manifest = artifact.load_manifest(opened_bearing.path)
    expected_kinds = (
        paths_lib.OBJECT_TRACKS, paths_lib.SEMANTIC_AUDITS)
    actual_kinds = tuple(item.kind for item in bearing_manifest.upstreams)
    if actual_kinds != expected_kinds:
        raise ValueError(
            "bearing_observations upstreams must be exactly object_tracks "
            f"then semantic_audits; found {actual_kinds}")
    tracks_ref, audits_ref = bearing_manifest.upstreams

    audits = audit_io.load_audits(
        Path(tracks_ref.path), Path(audits_ref.path))
    if (audits.tracks_ref != tracks_ref
            or audits.semantic_audits_ref != audits_ref):
        raise ValueError(
            "bearing_observations track/audit references are stale")
    accepted = tracklets.build_accepted_tracklets(
        audits.source_tracks, audits)
    return natural_closure_schedule(
        accepted, data.measurements, data.tables, data.n_keyframes - 1)


def write_sidecar(
        path: Path, data: export_ingest.ExportData,
        schedule: Iterable[TrackRelease]) -> None:
    """Write the portable release mapping for one exact input artifact."""
    releases = {}
    for item in schedule:
        if item.tracklet_id in releases:
            raise ValueError(
                f"duplicate release tracklet {item.tracklet_id!r}")
        releases[item.tracklet_id] = item.release_keyframe_idx
    # Validate the only information persisted by the sidecar against the
    # current artifact before writing it.
    _schedule_from_releases(
        releases, data.measurements, data.tables, data.n_keyframes - 1)
    artifact.atomic_write_json(path, {
        "schema": SIDECAR_SCHEMA,
        "localization_inputs": data.artifact_ref.to_dict(),
        "policy": _POLICY,
        "final_keyframe_idx": data.n_keyframes - 1,
        "release_keyframe_by_tracklet": releases,
    })


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_sidecar_document(path: Path):
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid non-finite JSON constant {value!r}")))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid release schedule sidecar {path}: {exc}") \
            from exc


def load_sidecar(
        path: Path,
        data: export_ingest.ExportData) -> tuple[TrackRelease, ...]:
    """Load release times, rebuilding measurement bundles from ``data``."""
    document = _load_sidecar_document(path)
    if not isinstance(document, dict) or set(document) != _SIDECAR_KEYS:
        actual = set(document) if isinstance(document, dict) else set()
        raise ValueError(
            "release schedule sidecar fields differ: "
            f"missing={sorted(_SIDECAR_KEYS - actual)}, "
            f"unknown={sorted(actual - _SIDECAR_KEYS)}")
    if document["schema"] != SIDECAR_SCHEMA:
        raise ValueError(
            f"release schedule schema must be {SIDECAR_SCHEMA!r}")
    if not artifact.records_same_artifact(
            document["localization_inputs"], data.artifact_ref):
        raise ValueError(
            "release schedule is bound to a different localization_inputs "
            "artifact")
    if document["policy"] != _POLICY:
        raise ValueError("release schedule policy differs from this reader")
    final_keyframe_idx = data.n_keyframes - 1
    if document["final_keyframe_idx"] != final_keyframe_idx:
        raise ValueError(
            "release schedule final keyframe disagrees with localization "
            "inputs")
    return _schedule_from_releases(
        document["release_keyframe_by_tracklet"],
        data.measurements,
        data.tables,
        final_keyframe_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        data = export_ingest.load(args.input_dir)
        schedule = natural_closure_schedule_from_export(data)
        write_sidecar(args.out, data, schedule)
    except (artifact.ArtifactError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"wrote {len(schedule)} track releases to {args.out}")


if __name__ == "__main__":
    main()
