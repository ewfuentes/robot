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

    def index_at_time(self, t_s: float) -> int:
        return int(round(t_s * self.fps))

    def time_at_index(self, idx: int) -> float:
        return idx / self.fps

    def frame(self, idx: int) -> np.ndarray:
        """RGB frame at index. Seeks only when access is non-sequential."""
        if idx != self._next_idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = self._cap.read()
        if not ok:
            raise IndexError(f"failed to read video frame {idx}")
        self._next_idx = idx + 1
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

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
