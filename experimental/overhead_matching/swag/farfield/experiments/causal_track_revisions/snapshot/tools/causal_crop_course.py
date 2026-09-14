"""Immutable course view for a tracker interval ending at the current clock.

Uses the same 10-second time window on every stream. The currently unobserved
right side holds the last known heading. No future fix is requested. A snapshot
is frozen before processing its interval; later fixes cannot revise its crop
rotations. Without a moving segment, rotation compensation explicitly abstains.
This is a tracker-input prototype, not a rewrite of historical SAM masks.
"""
from dataclasses import dataclass
import math
from online_time_course import OnlineTimeCourse


@dataclass(frozen=True)
class CropCourseSnapshot:
    midpoints: tuple
    headings: tuple
    areas: tuple
    window: float
    available_time_s: float
    available_keyframe: int

    def _check_time(self, t):
        if not math.isfinite(t) or t > self.available_time_s:
            raise ValueError('crop query exceeds the observed clock or is not finite')

    def course_world_cw_deg_at(self, t):
        self._check_time(t)
        if not self.headings:
            return None
        area = (OnlineTimeCourse._primitive(self,t+self.window/2) -
                OnlineTimeCourse._primitive(self,t-self.window/2))
        return math.degrees(area/self.window)

    def delta_course_cw_deg(self, t, reference_t):
        self._check_time(t)
        self._check_time(reference_t)
        if not self.headings:
            return 0.0  # Explicitly no rotation compensation, not world heading0.
        return self.course_world_cw_deg_at(t)-self.course_world_cw_deg_at(reference_t)

    @property
    def rotation_compensation_available(self):
        return bool(self.headings)


class CausalCropCourse:
    def __init__(self, window_seconds=10., min_displacement_m=3.):
        self.producer = OnlineTimeCourse(window_seconds,min_displacement_m)

    def append(self, time_s, east_m, north_m):
        self.producer.append(time_s,east_m,north_m)
        p = self.producer
        return CropCourseSnapshot(tuple(p.midpoints),tuple(p.headings),tuple(p.areas),
                                  p.window,time_s,p.index)
