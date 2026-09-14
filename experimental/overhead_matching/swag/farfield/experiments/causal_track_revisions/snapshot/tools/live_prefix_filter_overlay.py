"""Typed, explicitly diagnostic input overlay for the unchanged grid filter.

The validated base export supplies the static map, prior and benchmark motion.
Its historical measurements and matching tables are completely replaced.
Release clocks are rounded forward to an output keyframe, never backward.
"""
import bisect
import dataclasses
import math
import msgspec
from experimental.overhead_matching.swag.farfield.localization import structs, release_schedule


@dataclasses.dataclass(frozen=True)
class OverlayReference:
    base: object
    identity: dict

    @property
    def dataset(self):
        return self.base.dataset

    def to_dict(self):
        return dict(kind='diagnostic_live_prefix_overlay', base=self.base.to_dict(),
                    overlay=self.identity)


def prepare_releases(evidence, timestamps, end):
    if not 0 <= end < len(timestamps)-1:
        raise ValueError('Screen must exclude the final/EOF interval')
    if any(not math.isfinite(t) for t in timestamps) or any(
            a >= b for a,b in zip(timestamps,timestamps[1:])):
        raise ValueError('Output timestamps must be finite and increasing')
    releases, diagnostics, seen = [], [], set()
    for event in evidence['emissions']:
        clock = event['available_time_s']
        observed = event['release_keyframe']
        if (not math.isfinite(clock) or type(observed) is not int
                or not 0 <= observed < len(timestamps) or timestamps[observed] > clock):
            raise ValueError('Release precedes its observed track state')
        frame = bisect.bisect_left(timestamps, clock)
        # Unavailable future events are not interpreted or scored.
        if frame > end:
            continue
        tid = event['tracklet_id']
        physical_id=tid.rsplit('#',1)[-1]
        if physical_id in seen:
            raise ValueError('A physical prefix cannot be consumed twice')
        seen.add(physical_id)
        measurements = tuple(structs.TrackletMeasurement(**m) for m in event['measurements'])
        table = msgspec.convert(event['table'], type=structs.CompatibilityTable)
        if table.tracklet_id != tid or any(m.tracklet_id != tid for m in measurements):
            raise ValueError('Mismatched prefix identities')
        if not measurements or any(
                not 0 <= m.anchor_keyframe_idx <= observed
                or not math.isfinite(m.kappa) or m.kappa <= 0
                or not 0 <= m.bearing_forward_cw_deg < 360
                or (m.range_max_m is not None and (not math.isfinite(m.range_max_m) or m.range_max_m <= 0))
                for m in measurements):
            raise ValueError('Invalid or future-anchored measurements')
        releases.append(release_schedule.TrackRelease(frame,tid,measurements,table))
        diagnostics.append(dict(tracklet_id=tid,observed_keyframe=observed,
            available_time_s=clock,consumed_keyframe=frame,output_time_s=timestamps[frame],
            quantization_delay_seconds=timestamps[frame]-clock))
    # Stable order by release time and immutable identity, independent of file order.
    releases.sort(key=lambda r:(r.release_keyframe_idx,r.tracklet_id))
    return tuple(releases), diagnostics


def overlay_data(base, releases, identity):
    return dataclasses.replace(base,
        artifact_ref=OverlayReference(base.artifact_ref,identity),
        measurements=[m for r in releases for m in r.measurements],
        tables={r.tracklet_id:r.table for r in releases})
