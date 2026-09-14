"""Streaming course with a fixed window in seconds, independent of cadence.

Experimental replacement for the legacy sample-count smoother. Heading is
piecewise linear between moving-segment midpoints. A centered time average is
emitted only after the entire right half-window is observed. Initial extension
uses the first observed heading; final extension requires explicit EOF. This
component does not make historical tracking artifacts compatible by itself.
"""
import bisect
import collections
import math


class OnlineTimeCourse:
    def __init__(self, window_seconds=10.0, min_displacement_m=3.0):
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError('window_seconds must be positive and finite')
        if not math.isfinite(min_displacement_m) or min_displacement_m <= 0:
            raise ValueError('min_displacement_m must be positive and finite')
        self.window = float(window_seconds)
        self.minimum = float(min_displacement_m)
        self.midpoints, self.headings, self.areas = [], [], []
        self.pending = collections.deque()
        self.anchor = None
        self.last_time = None
        self.index = -1
        self.finished = False

    def append(self, time_s, east_m, north_m):
        if self.finished:
            raise ValueError('cannot append after EOF')
        if not all(math.isfinite(v) for v in [time_s, east_m, north_m]):
            raise ValueError('fix values must be finite')
        if self.last_time is not None and time_s <= self.last_time:
            raise ValueError('times must strictly increase')
        self.index += 1
        self.last_time = time_s
        self.pending.append((self.index, time_s))
        fix = (time_s, east_m, north_m)
        if self.anchor is None:
            self.anchor = fix
        elif math.hypot(east_m-self.anchor[1], north_m-self.anchor[2]) >= self.minimum:
            heading = math.atan2(east_m-self.anchor[1], north_m-self.anchor[2])
            midpoint = .5*(self.anchor[0]+time_s)
            area = 0.0
            if self.headings:
                heading = self.headings[-1] + (heading-self.headings[-1]+math.pi) % (2*math.pi)-math.pi
                area = self.areas[-1] + .5*(heading+self.headings[-1])*(midpoint-self.midpoints[-1])
            self.midpoints.append(midpoint)
            self.headings.append(heading)
            self.areas.append(area)
            self.anchor = fix
        return self._drain(False)

    def _primitive(self, t):
        if t <= self.midpoints[0]:
            return (t-self.midpoints[0])*self.headings[0]
        if t >= self.midpoints[-1]:
            return self.areas[-1] + (t-self.midpoints[-1])*self.headings[-1]
        left = bisect.bisect_right(self.midpoints, t)-1
        dt = t-self.midpoints[left]
        slope = ((self.headings[left+1]-self.headings[left]) /
                 (self.midpoints[left+1]-self.midpoints[left]))
        return self.areas[left] + self.headings[left]*dt + .5*slope*dt*dt

    def _drain(self, eof):
        rows = []
        if not self.headings:
            return rows
        while self.pending:
            index, time_s = self.pending[0]
            if not eof and time_s+self.window/2 > self.midpoints[-1]:
                break
            self.pending.popleft()
            value = (self._primitive(time_s+self.window/2) -
                     self._primitive(time_s-self.window/2))/self.window
            rows.append({'source_keyframe':index, 'source_time_s':time_s,
                         'available_keyframe':self.index, 'available_time_s':self.last_time,
                         'course_rad':value, 'requires_eof':eof})
        return rows

    def finish(self):
        if self.finished:
            raise ValueError('EOF already declared')
        self.finished = True
        return self._drain(True)
