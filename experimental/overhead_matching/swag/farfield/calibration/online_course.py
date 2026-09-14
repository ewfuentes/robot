"""Append-only course preprocessing with explicit data-availability times.

The smoothing sample count is a fixed configuration input, never estimated from
future timestamps. Unresolved values wait for the necessary moving fixes. EOF
padding is available only after an explicit finish() call.
"""
import bisect
import math


class OnlineCourse:
    def __init__(self, smoothing_samples, min_displacement_m=3.0):
        if type(smoothing_samples) is not int or smoothing_samples < 1:
            raise ValueError("smoothing_samples must be a positive integer")
        if not math.isfinite(min_displacement_m) or min_displacement_m <= 0:
            raise ValueError("min_displacement_m must be positive and finite")
        self.n = smoothing_samples
        self.minimum = min_displacement_m
        self.fixes = []
        self.anchor = None
        self.midpoints = []
        self.headings = []
        self.pending = []
        self.finished = False

    def append(self, time_s, east_m, north_m):
        if self.finished:
            raise ValueError("cannot append after EOF")
        if not all(math.isfinite(v) for v in (time_s, east_m, north_m)):
            raise ValueError("fix values must be finite")
        if self.fixes and time_s <= self.fixes[-1][0]:
            raise ValueError("fix times must strictly increase")
        fix = (time_s, east_m, north_m)
        self.fixes.append(fix)
        self.pending.append(len(self.fixes) - 1)
        if self.anchor is None:
            self.anchor = fix
        elif math.hypot(east_m - self.anchor[1], north_m - self.anchor[2]) >= self.minimum:
            h = math.atan2(east_m - self.anchor[1], north_m - self.anchor[2])
            if self.headings:
                h = self.headings[-1] + (h - self.headings[-1] + math.pi) % (2 * math.pi) - math.pi
            self.midpoints.append((time_s + self.anchor[0]) / 2)
            self.headings.append(h)
            self.anchor = fix
        return self._drain(False)

    def finish(self):
        if self.finished:
            raise ValueError("EOF already declared")
        self.finished = True
        return self._drain(True)

    def _value(self, t, eof):
        count = len(self.headings)
        if count == 0:
            return None
        if not eof and (t > self.midpoints[-1] or (self.n > 1 and count < 3)):
            return None
        right = min(bisect.bisect_left(self.midpoints, t), count - 1)
        left = max(0, right - 1)
        width = self.n if count > 2 else 1
        after = width - 1 - width // 2
        if not eof and right + after >= count:
            return None

        def smooth(index):
            return sum(self.headings[min(max(j, 0), count - 1)]
                       for j in range(index - width // 2, index + after + 1)) / width

        if t <= self.midpoints[0]:
            return smooth(0)
        if t >= self.midpoints[-1]:
            return smooth(count - 1)
        fraction = (t - self.midpoints[left]) / (self.midpoints[right] - self.midpoints[left])
        return smooth(left) * (1 - fraction) + smooth(right) * fraction

    def _drain(self, eof):
        emitted = []
        remaining = []
        for index in self.pending:
            value = self._value(self.fixes[index][0], eof)
            if value is None:
                remaining.append(index)
            else:
                emitted.append({"source_keyframe": index,
                                "available_keyframe": len(self.fixes) - 1,
                                "course_rad": value, "requires_eof": eof})
        self.pending = remaining
        return emitted
