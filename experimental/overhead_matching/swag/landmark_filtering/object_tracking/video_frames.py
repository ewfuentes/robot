"""Frame access into the full-rate leg video.

The landmark keyframes are ~3.5 s apart, but the source video runs at 3 fps;
the ~10 intermediate frames between keyframes are the tracking substrate.
Keyframe timestamps come from frames_gps.csv (video_t_s), so
frame index = round(t * fps) addresses the video exactly.
"""

import numpy as np

import cv2


class VideoFrameProvider:
    """Sequential-friendly random access to video frames as RGB arrays."""

    def __init__(self, video_path):
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"could not open video {video_path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._next_idx = 0
        # Most recently decoded frame, kept because consecutive keyframe
        # intervals share an endpoint: interval [k, k+1] ends on the frame that
        # interval [k+1, k+2] starts on. Re-reading it means seeking backwards
        # one frame, and in a long-GOP stream that costs a decode from the
        # previous keyframe -- measured at 1754 ms against 87 ms for a
        # sequential read on charles' 8K HEVC (60-frame GOP), once per interval.
        # One cached frame keeps the whole pass sequential.
        self._cached_idx = None
        self._cached_frame = None

    def index_at_time(self, t_s: float) -> int:
        return int(round(t_s * self.fps))

    def time_at_index(self, idx: int) -> float:
        return idx / self.fps

    def frame(self, idx: int) -> np.ndarray:
        """RGB frame at index. Seeks only when access is non-sequential.

        Returns the cached frame when asked for the one just decoded, so the
        endpoint shared by two consecutive intervals costs nothing.
        """
        if idx == self._cached_idx:
            return self._cached_frame
        if idx != self._next_idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = self._cap.read()
        if not ok:
            raise IndexError(f"failed to read video frame {idx}")
        self._next_idx = idx + 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._cached_idx, self._cached_frame = idx, rgb
        return rgb

    def frames_between(self, t0_s: float, t1_s: float, stride: int = 1,
                       include_end: bool = True):
        """Yield (video_idx, t_s, frame) from t0 to t1 inclusive."""
        i0 = self.index_at_time(t0_s)
        i1 = self.index_at_time(t1_s)
        indices = list(range(i0, i1, stride))
        if include_end and (not indices or indices[-1] != i1):
            indices.append(i1)
        for idx in indices:
            yield idx, self.time_at_index(idx), self.frame(idx)

    def close(self):
        self._cap.release()


class KeyframeProvider:
    """The keyframes themselves as the tracking substrate, for datasets with
    no source video (Mapillary collections retain only the posted frames).

    Covers the slice of VideoFrameProvider's interface the tracking loop
    touches: `frames_between(t0, t1)` yielding (idx, t_s, rgb). Times must be
    the same time_s values ingest put on the frames, so a keyframe interval
    [t_k, t_k+1] selects exactly its two endpoint keyframes and SAM2
    propagates straight across the keyframe baseline with no intermediates.
    """

    def __init__(self, times_s, image_paths):
        if len(times_s) != len(image_paths):
            raise ValueError(
                f"{len(times_s)} times vs {len(image_paths)} image paths")
        self._times = [float(t) for t in times_s]
        self._paths = [str(p) for p in image_paths]

    def frames_between(self, t0_s: float, t1_s: float, stride: int = 1,
                       include_end: bool = True):
        """Yield (idx, t_s, frame) for keyframes with t0 <= time_s <= t1."""
        picked = [i for i, t in enumerate(self._times) if t0_s <= t <= t1_s]
        indices = picked[::stride]
        if include_end and picked and (
                not indices or indices[-1] != picked[-1]):
            indices.append(picked[-1])
        for i in indices:
            bgr = cv2.imread(self._paths[i])
            if bgr is None:
                raise FileNotFoundError(f"could not read {self._paths[i]}")
            yield i, self._times[i], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        pass
