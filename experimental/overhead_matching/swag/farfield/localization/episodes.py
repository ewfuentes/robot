"""Evaluation episodes: fixed-length segments of a recorded trajectory, half
of them traversed in reverse, spread evenly along the whole recording.

Different seeds of one long run re-randomize the sampler but see the same
evidence in the same order; a segment starting elsewhere, or driven the other
way, is a genuinely different localization problem on the same data (the LOCI
protocol: 3 km segments, half forward, half reverse). A segment is derived
from the localization export without touching the filter:

  forward  [a, b]: keyframes a..b re-indexed to 0..b-a; odometry, epochs and
           truth sliced and shifted.
  reverse  [a, b]: new keyframe j is original keyframe b - j. The platform
           drives the path backwards, so nominal forward flips by 180 deg:
           every bearing gains 180 deg, truth course gains 180 deg, and the
           odometry increment INTO new keyframe j+1 is the original increment
           into keyframe b-j, inverted under the filter's rotate-then-move
           semantics (yaw_k = yaw_{k-1} + dyaw_k, then translate by
           R(yaw_k)[forward, left]):
               dyaw'    = -dyaw
               forward' =  forward cos(dyaw) + left sin(dyaw)
               left'    = -forward sin(dyaw) + left cos(dyaw)
           with the declared sigmas unchanged. (Exact, not an approximation:
           the original step moves by v = R(yaw_k)[f, l]; the reversed step
           moves by -v from a platform whose heading is yaw_k + pi - dyaw,
           i.e. the same displacement expressed in the frame yaw_{k-1} + pi.)

Compatibility tables and the catalog are direction-independent and shared.
Short recordings (shorter than the segment) yield two episodes, the whole
recording forward and reverse.
"""
import dataclasses
import math

import msgspec

from experimental.overhead_matching.swag.farfield.localization import (
    metrics,
    structs,
)


@dataclasses.dataclass(frozen=True)
class Episode:
    index: int
    start_keyframe: int  # inclusive, original indexing
    end_keyframe: int  # inclusive, original indexing
    reverse: bool
    start_distance_m: float
    length_m: float

    @property
    def name(self) -> str:
        direction = "rev" if self.reverse else "fwd"
        return f"ep{self.index:02d}_{direction}_kf{self.start_keyframe}-{self.end_keyframe}"

    @property
    def n_keyframes(self) -> int:
        return self.end_keyframe - self.start_keyframe + 1


def plan(truth: list, segment_length_m: float, *, min_episodes: int = 2,
         episodes_per_length: float = 2.0) -> list[Episode]:
    """Segments of `segment_length_m` with starts spread evenly along the
    recording, alternating forward / reverse. The count is
    `episodes_per_length` per segment length of recording (so segments
    overlap by half), at least `min_episodes`. A recording shorter than one
    segment gives the whole recording forward and reverse."""
    if segment_length_m <= 0.0:
        raise ValueError("segment_length_m must be positive")
    distances = metrics.cumulative_distance_m(truth)
    keyframes = [record.keyframe_idx for record in truth]
    total = distances[keyframes[-1]]
    if total <= segment_length_m:
        starts = [0.0, 0.0]
        length = total
    else:
        count = max(min_episodes,
                    int(round(episodes_per_length * total / segment_length_m)))
        span = total - segment_length_m
        starts = [span * i / (count - 1) for i in range(count)]
        length = segment_length_m
    episodes = []
    for index, start_distance in enumerate(starts):
        start_kf = next(kf for kf in keyframes if distances[kf] >= start_distance - 1e-9)
        end_kf = max(kf for kf in keyframes
                     if distances[kf] <= distances[start_kf] + length + 1e-9)
        if end_kf <= start_kf:
            end_kf = keyframes[min(keyframes.index(start_kf) + 1, len(keyframes) - 1)]
        episodes.append(Episode(
            index=index, start_keyframe=start_kf, end_keyframe=end_kf,
            reverse=(index % 2 == 1), start_distance_m=distances[start_kf],
            length_m=distances[end_kf] - distances[start_kf]))
    return episodes


def _reverse_delta(delta: structs.OdometryDelta, new_keyframe: int) -> structs.OdometryDelta:
    dyaw = delta.delta_yaw_cw_rad
    return msgspec.structs.replace(
        delta, keyframe_idx=new_keyframe,
        forward_m=delta.forward_m * math.cos(dyaw) + delta.left_m * math.sin(dyaw),
        left_m=-delta.forward_m * math.sin(dyaw) + delta.left_m * math.cos(dyaw),
        delta_yaw_cw_rad=-dyaw)


def derive(odometry: list, measurements: list, truth: list,
           episode: Episode) -> tuple[list, list, list]:
    """(odometry, measurements, truth) of the episode, re-indexed from 0."""
    a, b = episode.start_keyframe, episode.end_keyframe
    by_kf = {delta.keyframe_idx: delta for delta in odometry}
    if episode.reverse:
        # new j <- original b - j; increment into new j+1 is the original
        # increment into b - j (which moved the platform from b-j-1 to b-j).
        new_odometry = [
            _reverse_delta(by_kf[b - j], j + 1) for j in range(b - a)]
        new_measurements = [
            msgspec.structs.replace(
                meas, anchor_keyframe_idx=b - meas.anchor_keyframe_idx,
                bearing_forward_cw_deg=(meas.bearing_forward_cw_deg + 180.0) % 360.0)
            for meas in measurements if a <= meas.anchor_keyframe_idx <= b]
        new_truth = [
            msgspec.structs.replace(
                pose, keyframe_idx=b - pose.keyframe_idx,
                course_world_cw_deg=(pose.course_world_cw_deg + 180.0) % 360.0)
            for pose in truth if a <= pose.keyframe_idx <= b]
    else:
        new_odometry = [
            msgspec.structs.replace(by_kf[k], keyframe_idx=k - a)
            for k in range(a + 1, b + 1)]
        new_measurements = [
            msgspec.structs.replace(meas, anchor_keyframe_idx=meas.anchor_keyframe_idx - a)
            for meas in measurements if a <= meas.anchor_keyframe_idx <= b]
        new_truth = [
            msgspec.structs.replace(pose, keyframe_idx=pose.keyframe_idx - a)
            for pose in truth if a <= pose.keyframe_idx <= b]
    new_measurements.sort(key=lambda m: (m.anchor_keyframe_idx, m.tracklet_id))
    new_truth.sort(key=lambda p: p.keyframe_idx)
    return new_odometry, new_measurements, new_truth
