"""Tracklets: localization-ready bearings straight from tracks + audit.

This library replaces the merge stage (m6_merge_tracks / track_merge on the
checkpoint branch). There is no consolidation step and no materialized
`merged/` artifact any more: each audited track IS a tracklet, and the three
consumers (offset sweep, matching, localization export) call this one library
on `tracks_*.json` + the semantic-audit artifact with the run's recorded
fusion parameters.

Why no merging: the old consolidator's own contract conceded that
under-merging is cheap — two tracklets of one physical object both match the
same map feature and the filter's data association copes — while over-merging
welds two objects into one landmark with a bimodal bearing. Meanwhile its
consumers had already drifted apart from it (a second bearing fusion with a
different kappa rule in the export, a different support bar in matching,
three copies of the epoch constant). Deleting the weld removes a stage, an
artifact, and that entire disagreement surface. If over-splitting ever proves
harmful, the fix belongs in the filter's association model, not in a
pipeline-side weld.

The support gate: a tracklet exists only for tracks that HAVE a semantic
audit. Audit membership already encodes the evidence bar (only tracks with
enough detector supports are auditable), and a track that was never audited
has no canonical semantics, so it cannot be matched and must not reach the
filter. This replaces three independently-defaulted `--min_supports` flags
with one recorded decision made at audit time.

Bearings are CAMERA-frame azimuths (pano_geometry convention, CW positive);
converting to the body frame is `geometry.apply_mount_offset`, applied by the
export, not here.
"""

from dataclasses import dataclass

from experimental.overhead_matching.swag.farfield import geometry as geo


@dataclass
class TrackletParams:
    """Fusion parameters. No defaults on purpose (REORG.md rule 2): values
    come from the run's recorded config, so the numbers used are the numbers
    recorded."""
    # Keyframes fused into one bearing measurement. Consecutive bearings on
    # one object are strongly correlated (same mask, same tracker), so the
    # filter consumes one fused bearing per tracklet per information epoch
    # rather than one per keyframe.
    epoch_keyframes: int
    # Per-observation bearing noise floor, degrees.
    bearing_sigma_deg: float


@dataclass
class Measurement:
    tracklet_id: str          # "T<track_id>" — one tracklet per audited track
    anchor_keyframe_idx: int
    bearing_camera_deg: float
    kappa: float


def mask_boxes_by_keyframe(track: dict, valid_segments=None) -> dict:
    """keyframe -> mask bbox in pano coords, restricted to valid segments.

    Segments come from the semantic audit (relative time indices); applying
    them matters because a track's own drifted tail otherwise contaminates
    its bearing series.
    """
    birth = track["birth_keyframe"]
    spans = None
    if valid_segments:
        spans = [(birth + s["start_t"], birth + s["end_t"])
                 for s in valid_segments]
    out = {}
    for rec in track["records"]:
        mb = rec.get("mask_bbox_window")
        if mb is None:
            continue
        kf = rec["keyframe"]
        if spans is not None and not any(a <= kf <= b for a, b in spans):
            continue
        ox, oy = rec["window_origin"]
        out[kf] = (ox + mb[0], oy + mb[1], ox + mb[2], oy + mb[3])
    return out


def bearing_series(track: dict, pano_w: int, valid_segments=None) -> list:
    """[(keyframe, az_cw_deg, angular_width_deg)] from the tracked mask.

    Camera-frame azimuth of the mask centroid column; angular width is the
    mask's own extent, the basis for the measurement's concentration.
    """
    out = []
    for kf, box in sorted(
            mask_boxes_by_keyframe(track, valid_segments).items()):
        centre = (box[0] + box[2]) / 2.0
        az = geo.azimuth_of_pano_column(centre, pano_w)
        width = (box[2] - box[0]) / pano_w * 360.0
        out.append((kf, az, width))
    return out


def fuse_bearings(series: list, params: TrackletParams) -> list:
    """Fuse a bearing series into sparse per-epoch measurements.

    Returns [(anchor_keyframe, az_cw_deg, kappa)]. kappa combines the
    per-observation concentration with the object's own angular width — an
    extended object's centroid is not a point bearing — and does NOT grow
    with the number of fused keyframes, which is the conservative choice
    while the intra-epoch correlation is unmodelled.
    """
    if not series:
        return []
    import math

    fused = []
    epoch = max(1, params.epoch_keyframes)
    start_kf = series[0][0]
    bucket = []

    def flush(bucket):
        if not bucket:
            return
        mean_az = geo.circular_mean_deg([a for _, a, _ in bucket])
        mean_width = sum(w for _, _, w in bucket) / len(bucket)
        anchor = bucket[len(bucket) // 2][0]
        # Width contributes a centroid ambiguity of about a quarter-width.
        sigma = math.hypot(params.bearing_sigma_deg, mean_width / 4.0)
        kappa = 1.0 / math.radians(sigma) ** 2
        fused.append((anchor, mean_az, kappa))

    for entry in series:
        if entry[0] - start_kf >= epoch:
            flush(bucket)
            bucket = []
            start_kf = entry[0]
        bucket.append(entry)
    flush(bucket)
    return fused


def tracklet_id(track: dict) -> str:
    return f"T{track['track_id']}"


def build_measurements(tracks: dict, audits: dict, pano_w: int,
                       params: TrackletParams) -> list:
    """Fused camera-frame bearings for every AUDITED track.

    tracks: {track_id: track dict} from tracks_*.json.
    audits: {track_id: audit dict} from the semantic-audit artifact. Audit
    membership is the support gate: tracks absent from `audits` produce no
    measurements (they have no canonical semantics and must not reach
    matching or the filter).

    Sorted by (anchor keyframe, tracklet id), the order the export consumes.
    """
    measurements = []
    for tid, audit in audits.items():
        track = tracks.get(tid)
        if track is None or not track.get("records"):
            continue
        series = bearing_series(track, pano_w,
                                (audit or {}).get("valid_segments"))
        for anchor, az, kappa in fuse_bearings(series, params):
            measurements.append(Measurement(
                tracklet_id=tracklet_id(track),
                anchor_keyframe_idx=anchor,
                bearing_camera_deg=az,
                kappa=kappa))
    measurements.sort(key=lambda m: (m.anchor_keyframe_idx, m.tracklet_id))
    return measurements
