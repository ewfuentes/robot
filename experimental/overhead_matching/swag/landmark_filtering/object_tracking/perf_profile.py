"""Opt-in phase timing for the M3 tracking loop.

Off unless `TRACK_PROFILE=1`, so it can live permanently in the hot path: the
disabled `phase()` is a generator that yields immediately, which costs far less
than the work it wraps. It exists because guessing where this loop spends its
time has been wrong twice (JPEG round-trip, video decode), and hand-patching
timers in and out risks leaving instrumentation behind.

GPU work is asynchronous, so a timer that does not synchronize charges the wait
for a kernel to whichever phase happens to read its result. `phase()` therefore
synchronizes CUDA on both edges of a GPU phase (`gpu=True`). That serializes the
stream and makes the profiled run slightly slower than the real one -- which is
the point of a profile, and why wall-clock comparisons are always taken with
profiling off.

    TRACK_PROFILE=1 bazel run ...:m3_track_viewer -- --dataset X --range r 0 20
"""

import contextlib
import os
import time
from collections import Counter

ENABLED = os.environ.get("TRACK_PROFILE", "") not in ("", "0")


class Profile:
    def __init__(self, enabled: bool = ENABLED):
        self.enabled = enabled
        self.seconds = Counter()
        self.calls = Counter()
        self.items = Counter()
        self._sync = None

    def _cuda_sync(self):
        """Resolved lazily: torch must not be imported by a CPU-only consumer."""
        if self._sync is None:
            try:
                import common.torch.load_torch_deps  # noqa: F401
                import torch
                self._sync = (torch.cuda.synchronize
                              if torch.cuda.is_available() else lambda: None)
            except ImportError:
                self._sync = lambda: None
        return self._sync

    @contextlib.contextmanager
    def phase(self, name: str, items: int = 0, gpu: bool = False):
        """Time a named phase. `items` accumulates a unit count (frames, crops,
        tracks) so the report can show cost per unit."""
        if not self.enabled:
            yield
            return
        sync = self._cuda_sync() if gpu else None
        if sync:
            sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            if sync:
                sync()
            self.seconds[name] += time.perf_counter() - started
            self.calls[name] += 1
            self.items[name] += items

    def report(self, log=print, wall: float | None = None, label: str = ""):
        if not self.enabled or not self.seconds:
            return
        total = sum(self.seconds.values())
        denominator = wall if wall else total
        log(f"--- profile {label}".rstrip())
        log(f"    {'phase':<22}{'seconds':>9}{'share':>8}{'calls':>8}"
            f"{'items':>9}{'ms/item':>9}")
        for name, seconds in self.seconds.most_common():
            items = self.items[name]
            per_item = f"{seconds / items * 1000:.1f}" if items else "-"
            log(f"    {name:<22}{seconds:9.1f}{seconds / denominator * 100:7.1f}%"
                f"{self.calls[name]:8d}{items:9d}{per_item:>9}")
        log(f"    {'SUM OF PHASES':<22}{total:9.1f}"
            + (f"{total / wall * 100:7.1f}% of {wall:.1f}s wall" if wall else ""))

    def reset(self):
        self.seconds.clear()
        self.calls.clear()
        self.items.clear()


PROFILE = Profile()
