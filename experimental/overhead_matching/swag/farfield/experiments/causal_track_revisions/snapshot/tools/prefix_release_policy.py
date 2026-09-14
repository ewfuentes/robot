"""One-shot early track release experiment with immutable prefix evidence.

This is a screening policy, not an adopted replacement for whole-track updates.
Later evidence is intentionally not re-emitted, avoiding duplicate consumption.
"""
import copy
import math
from experimental.overhead_matching.swag.farfield.tracking.track_builder import SUPPORT_CLASSES


class OneShotPrefixRelease:
    def __init__(self, min_supported_frames=3, min_age_seconds=10.):
        if type(min_supported_frames) is not int or min_supported_frames<1:
            raise ValueError('Positive support count required')
        if not math.isfinite(min_age_seconds) or min_age_seconds<0:
            raise ValueError('Finite nonnegative age required')
        self.min_supported_frames = min_supported_frames
        self.min_age_seconds = min_age_seconds
        self.emitted = set()

    def consider(self, track, frame_times, keyframe, available_time_s):
        tid = track['track_id']
        if tid in self.emitted:
            return None
        records = track['records']
        if not records:
            return None
        if track['status'] not in {'alive','closed'}:
            raise ValueError('Unknown track state')
        if (frame_times[keyframe]>available_time_s or track['birth_keyframe']>keyframe
                or any(r['keyframe']>keyframe for r in records)
                or any(frame_times[r['keyframe']]>available_time_s for r in records)):
            raise ValueError('Future evidence in release prefix')
        support_frames = {track['birth_keyframe']}
        support_frames.update(r['keyframe'] for r in records
            if any(s['class'] in SUPPORT_CLASSES for s in r.get('supports',[])))
        ready = (len(support_frames)>=self.min_supported_frames
                 and frame_times[keyframe]-frame_times[track['birth_keyframe']]>=self.min_age_seconds)
        closed = track['status']=='closed'
        if not ready and not closed:
            return None
        self.emitted.add(tid)
        return dict(track=copy.deepcopy(track),release_keyframe=keyframe,
                    available_time_s=available_time_s,
                    reason='confirmed_close' if closed else 'age_and_support_prefix',
                    supported_frames=len(support_frames),
                    later_evidence_discarded=True,uses_final_audit=False)
