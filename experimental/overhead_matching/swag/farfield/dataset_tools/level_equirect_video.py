"""Level an equirectangular video whose exported horizon is not on the equator row.

Why this exists: GoPro Player's "horizon lock" left the Portland flight exports
5-20 degrees off level, drifting over minutes. For a landmark below the horizon
(everything, from an aircraft) a frame tilt `beta` biases its azimuth by about
`beta * tan|el|` -- 2-6 degrees at the tilts seen -- a slowly varying,
landmark-dependent bias the localization heading state cannot absorb. The fix
is a per-frame rotation about a HORIZONTAL axis that puts the fitted horizon
plane on the equator. Yaw is never touched, so the camera's forward column and
the nominal-forward calibration are preserved (docs/farfield/conventions.md).

Pipeline (two decode passes over the source, both on the `output_fps` grid
that `prepare_selfcollect` and `audit_dataset` address, frame n <-> t = n/fps):

  pass A  decode at 960x480 -> per-frame horizon-plane fit -> self-check on a
          levelled low-res copy -> temporal gate (running median, continuity,
          self-check) -> per-frame rotation, interpolated where gated
  pass B  decode full resolution -> rotate on the GPU -> encode the levelled
          derivative; a stacked before/after review video and a re-detection
          on every written frame are produced alongside

Horizon fit: skyness = (blue AND bright) OR (white AND bright, low saturation);
the brightness gate is what separates pale sky at the horizon from water.
Candidate boundary pixels are lifted to unit vectors in the camera frame and a
plane `p . n = d` is RANSAC-fit with one sample per azimuth third; the plane
must be near-horizontal and sit within a few degrees of elevation zero
(shorelines and cloud shadows do not). `d` absorbs the ~0.5 degree horizon dip
and the haze band; only `n` (the frame's "up") is used.

Outputs in --output_dir, all no-clobber:
  <name>_<fps>fps_levelled.mp4     the derivative to declare as video.source_video
  <name>_leveling.csv              one row per output frame
  <name>_leveling_review.mp4       before/after stacked, equator + fitted horizon drawn
  <name>_leveling_manifest.json    source/output digests, settings, gate statistics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from experimental.overhead_matching.swag.farfield import artifact, code_provenance
from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    anonymize_video as av,
)

SCHEMA = "farfield_horizon_leveling/v1"
DET_W, DET_H = 960, 480             # fit resolution
BAND_EL_DEG = 30.0                  # horizon searched within +-30 deg elevation
EL_RANGE_DEG = (-7.5, 2.5)          # plane offset must be near el 0 whatever the tilt
RANSAC_ITERS = 1500
RANSAC_GATE_DEG = 1.0
REFINE_GATES_DEG = (1.0, 0.6, 0.35)
MIN_INLIERS = 30
MIN_CANDIDATES = 50
GRAD_MIN_COLS = 150                 # independent gradient-edge plane needs this support
TEMPORAL_WINDOW = 9                 # frames, at output_fps
CONTINUITY_GATE_DEG = 1.5
SELF_CHECK_GATE_DEG = 0.5
CSV_COLUMNS = [
    "frame_index", "video_t_s", "status", "tilt_deg", "roll_deg", "pitch_deg",
    "lean_az_deg", "horizon_el_deg", "rms_deg", "cols", "coverage",
    "post_tilt_deg", "grad_plane_delta_deg", "grad_cols",
    "applied_tilt_deg", "applied_roll_deg", "applied_pitch_deg",
    "applied_n_x", "applied_n_y", "applied_n_z", "written_post_tilt_deg",
]


# ---------------------------------------------------------------------------
# Camera-frame maths. Vectorised twins of geometry.direction_from_pano_px /
# pano_px_from_direction (parity is unit-tested); x forward, y left, z up, so a
# clockwise azimuth points toward -y.
# ---------------------------------------------------------------------------

def unit_vectors(az_cw_deg, el_up_deg):
    az = np.radians(az_cw_deg); el = np.radians(el_up_deg)
    return np.stack([np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)], -1)


def az_el_from_vectors(d):
    az = np.degrees(np.arctan2(-d[..., 1], d[..., 0])) % 360.0
    el = np.degrees(np.arcsin(np.clip(d[..., 2], -1.0, 1.0)))
    return az, el


_DIRS: dict[tuple[int, int], np.ndarray] = {}


def pano_dirs(w: int, h: int) -> np.ndarray:
    """(h, w, 3) camera-frame unit vector of every pixel centre."""
    if (w, h) not in _DIRS:
        x = (np.arange(w) + 0.5) / w
        y = (np.arange(h) + 0.5) / h
        az = (x - 0.5) * 360.0
        el = (0.5 - y) * 180.0
        AZ, EL = np.meshgrid(az, el)
        _DIRS[(w, h)] = unit_vectors(AZ, EL)
    return _DIRS[(w, h)]


def dirs_to_px(d, w: int, h: int):
    az, el = az_el_from_vectors(d)
    x = ((az / 360.0 + 0.5) % 1.0) * w - 0.5
    y = (0.5 - el / 180.0) * h - 0.5
    return x, y


def level_rotation(n: np.ndarray) -> np.ndarray:
    """Minimal rotation R with R @ n = z. Its axis is horizontal: zero yaw."""
    z = np.array([0.0, 0.0, 1.0])
    k = np.cross(n, z)
    s = float(np.linalg.norm(k))
    c = float(n @ z)
    if s < 1e-12:
        return np.eye(3)
    k = k / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def tilt_params(n: np.ndarray) -> dict:
    """tilt: angle between frame up and level up; lean_az: azimuth (cw from forward)
    the frame's up leans toward; pitch > 0 nose-down; roll > 0 right wing down."""
    return {
        "tilt_deg": math.degrees(math.acos(min(1.0, float(n[2])))),
        "lean_az_deg": math.degrees(math.atan2(-n[1], n[0])) % 360.0,
        "pitch_deg": math.degrees(math.atan2(n[0], n[2])),
        "roll_deg": math.degrees(math.atan2(-n[1], n[2])),
    }


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b)))))


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    omega = math.acos(max(-1.0, min(1.0, float(a @ b))))
    if omega < 1e-9:
        return a.copy()
    return (math.sin((1 - t) * omega) * a + math.sin(t * omega) * b) / math.sin(omega)


# ---------------------------------------------------------------------------
# Horizon detection
# ---------------------------------------------------------------------------

def skyness(bgr: np.ndarray) -> np.ndarray:
    f = bgr.astype(np.float32)
    r, g, b = f[..., 2], f[..., 1], f[..., 0]
    v = f.max(-1)
    s = (v - f.min(-1)) / np.maximum(v, 1.0)
    # sky at the horizon is pale and BRIGHT (V >= ~210); water is blue but darker
    blue = np.clip((b - np.maximum(r, g)) / 40.0, 0, 1) * np.clip((v - 175.0) / 35.0, 0, 1)
    white = np.clip((v - 205.0) / 30.0, 0, 1) * np.clip((0.25 - s) / 0.15, 0, 1)
    return np.maximum(blue, white)


def candidates(bgr: np.ndarray):
    """Per column: rows where smoothed skyness crosses 0.5 downward (sky above)."""
    sk = cv2.blur(skyness(bgr), (1, 7))
    h = sk.shape[0]
    y0 = int(round(h * (0.5 - BAND_EL_DEG / 180.0)))
    y1 = int(round(h * (0.5 + BAND_EL_DEG / 180.0)))
    above = sk[y0 - 1:y1 - 1] > 0.5
    below = sk[y0:y1] <= 0.5
    ys, xs = np.nonzero(above & below)
    return xs, ys + y0


def plane_from_points(P: np.ndarray):
    """Least-squares plane p.n = d through unit vectors P; unit n with n_z > 0."""
    A = np.c_[P, -np.ones(len(P))]
    v = np.linalg.svd(A, full_matrices=False)[2][-1]
    n, d = v[:3], v[3]
    s = np.linalg.norm(n)
    n, d = n / s, d / s
    if n[2] < 0:
        n, d = -n, -d
    return n, float(d)


def ang_resid(P: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    return np.degrees(np.arcsin(np.clip(P @ n, -1, 1)) - math.asin(max(-1.0, min(1.0, d))))


def horizon_curve(n: np.ndarray, d: float, w: int, h: int) -> np.ndarray:
    """Row of the plane's horizon at every column (for drawing)."""
    az = np.radians(((np.arange(w) + 0.5) / w - 0.5) * 360.0)
    A = n[0] * np.cos(az) - n[1] * np.sin(az)
    B = n[2]
    el = np.arcsin(np.clip(d / np.hypot(A, B), -1, 1)) - np.arctan2(A, B)
    return (0.5 - el / np.pi) * h - 0.5


@dataclass
class HorizonFit:
    n: np.ndarray
    d: float
    rms_deg: float
    cols: int
    coverage: float
    n_grad: np.ndarray | None
    d_grad: float | None
    grad_cols: int
    xs: np.ndarray
    ys: np.ndarray
    inliers: np.ndarray

    @property
    def horizon_el_deg(self) -> float:
        return math.degrees(math.asin(max(-1.0, min(1.0, self.d))))

    @property
    def grad_plane_delta_deg(self) -> float | None:
        return None if self.n_grad is None else angle_between_deg(self.n, self.n_grad)


def fit_horizon(bgr: np.ndarray, rng: np.random.Generator) -> HorizonFit | None:
    h, w = bgr.shape[:2]
    xs, ys = candidates(bgr)
    if len(xs) < MIN_CANDIDATES:
        return None
    P = pano_dirs(w, h)[ys, xs]
    thirds = [np.nonzero((xs >= w * k / 3) & (xs < w * (k + 1) / 3))[0] for k in range(3)]
    if min(len(t) for t in thirds) < 5:
        return None
    lo, hi = math.sin(math.radians(EL_RANGE_DEG[0])), math.sin(math.radians(EL_RANGE_DEG[1]))
    best_k, best_inl = 0, None
    for _ in range(RANSAC_ITERS):
        i = [rng.choice(t) for t in thirds]        # one point per azimuth third
        p1, p2, p3 = P[i]
        n = np.cross(p2 - p1, p3 - p1)
        s = np.linalg.norm(n)
        if s < 1e-9:
            continue
        n = n / s
        if n[2] < 0:
            n = -n
        if n[2] < 0.5:
            continue
        d = float(n @ p1)
        if not (lo <= d <= hi):
            continue
        inl = np.abs(ang_resid(P, n, d)) < RANSAC_GATE_DEG
        k = len(np.unique(xs[inl]))                # distinct columns, not points
        if k > best_k:
            best_k, best_inl = k, inl
    if best_inl is None:
        return None
    inl = best_inl
    for gate in REFINE_GATES_DEG:
        n, d = plane_from_points(P[inl])
        inl = np.abs(ang_resid(P, n, d)) < gate
        if inl.sum() < MIN_INLIERS:
            return None
    n, d = plane_from_points(P[inl])
    if not (lo <= d <= hi):
        return None
    res = ang_resid(P[inl], n, d)
    cols = len(np.unique(xs[inl]))
    # Independent check: snap each supporting column to the strongest luminance
    # edge near the plane and refit. Different cue; trusted only with wide support.
    n_grad, d_grad, grad_cols = None, None, 0
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gy = np.abs(cv2.Sobel(cv2.GaussianBlur(L, (1, 5), 0), cv2.CV_32F, 0, 1, ksize=3))
    pred = horizon_curve(n, d, w, h)
    ucols = np.unique(xs[inl])
    rows = []
    for x in ucols:
        c = int(round(pred[x]))
        a, b = max(1, c - 12), min(h - 2, c + 12)
        rows.append(a + int(np.argmax(gy[a:b, x])))
    Pg = pano_dirs(w, h)[np.array(rows), ucols]
    ng, dg = plane_from_points(Pg)
    keep = np.abs(ang_resid(Pg, ng, dg)) < REFINE_GATES_DEG[-1]
    if keep.sum() >= GRAD_MIN_COLS:
        n_grad, d_grad = plane_from_points(Pg[keep])
        grad_cols = int(keep.sum())
    return HorizonFit(n=n, d=d, rms_deg=float(np.sqrt(np.mean(res ** 2))), cols=int(cols),
                      coverage=cols / w, n_grad=n_grad, d_grad=d_grad, grad_cols=grad_cols,
                      xs=xs, ys=ys, inliers=inl)


# ---------------------------------------------------------------------------
# Levelling (image rotation on the sphere)
# ---------------------------------------------------------------------------

def level_image_cpu(bgr: np.ndarray, R: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    d_src = pano_dirs(w, h).reshape(-1, 3) @ R           # R^T d_out, row form
    mx, my = dirs_to_px(d_src, w, h)
    return cv2.remap(bgr, mx.reshape(h, w).astype(np.float32),
                     my.reshape(h, w).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


class GpuLeveler:
    """torch grid_sample version for 8K frames; the source is wrapped by two
    columns so bilinear taps across the azimuth seam read real pixels."""

    PAD = 2

    def __init__(self, w: int, h: int, device: str):
        import common.torch.load_torch_deps  # noqa: F401  (CUDA libs before torch)
        import torch
        self.torch = torch
        self.w, self.h = w, h
        self.device = torch.device(device)
        self.dirs = torch.from_numpy(
            pano_dirs(w, h).reshape(-1, 3).astype(np.float32)).to(self.device)

    def level(self, bgr: np.ndarray, R: np.ndarray) -> np.ndarray:
        torch, w, h, pad = self.torch, self.w, self.h, self.PAD
        with torch.no_grad():
            Rt = torch.from_numpy(np.ascontiguousarray(R, dtype=np.float32)).to(self.device)
            d = self.dirs @ Rt                            # R^T d_out
            az = torch.atan2(-d[:, 1], d[:, 0])
            el = torch.asin(torch.clamp(d[:, 2], -1.0, 1.0))
            x = torch.remainder(az / (2 * math.pi) + 0.5, 1.0) * w - 0.5 + pad
            y = (0.5 - el / math.pi) * h - 0.5
            gx = (x + 0.5) / (w + 2 * pad) * 2 - 1
            gy = (y + 0.5) / h * 2 - 1
            grid = torch.stack([gx, gy], -1).reshape(1, h, w, 2)
            src = np.concatenate([bgr[:, -pad:], bgr, bgr[:, :pad]], axis=1)
            t = torch.from_numpy(np.ascontiguousarray(src)).to(self.device)
            t = t.permute(2, 0, 1).unsqueeze(0).float()
            out = torch.nn.functional.grid_sample(
                t, grid, mode="bilinear", padding_mode="border", align_corners=False)
            out = out[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
            return out.cpu().numpy()


# ---------------------------------------------------------------------------
# Temporal gate
# ---------------------------------------------------------------------------

def temporal_gate(ns: list[np.ndarray | None], post_tilts: list[float | None],
                  window: int = TEMPORAL_WINDOW,
                  continuity_gate_deg: float = CONTINUITY_GATE_DEG,
                  self_check_gate_deg: float = SELF_CHECK_GATE_DEG):
    """Return (applied_n list, status list). A frame keeps its own fit unless it
    failed, deviates from the running median of its neighbours, or failed its
    self-check; gated frames take the SLERP of the nearest accepted neighbours."""
    count = len(ns)
    if count == 0:
        raise ValueError("no frames to gate")
    xy = np.full((count, 2), np.nan)
    for i, n in enumerate(ns):
        if n is not None:
            xy[i] = n[:2]
    half = window // 2
    status = []
    for i in range(count):
        if ns[i] is None:
            status.append("no_fit")
            continue
        lo, hi = max(0, i - half), min(count, i + half + 1)
        block = xy[lo:hi]
        block = block[~np.isnan(block[:, 0])]
        med = np.nanmedian(block, axis=0)
        med_n = np.array([med[0], med[1], math.sqrt(max(0.0, 1 - med[0] ** 2 - med[1] ** 2))])
        if angle_between_deg(ns[i], med_n) > continuity_gate_deg:
            status.append("continuity_gate")
        elif post_tilts[i] is None or post_tilts[i] > self_check_gate_deg:
            status.append("self_check_gate")
        else:
            status.append("fit")
    accepted = [i for i, s in enumerate(status) if s == "fit"]
    if not accepted:
        raise ValueError("every frame was gated; the horizon fit does not work on this video")
    applied = []
    for i in range(count):
        if status[i] == "fit":
            applied.append(ns[i])
            continue
        prev = max((j for j in accepted if j < i), default=None)
        nxt = min((j for j in accepted if j > i), default=None)
        if prev is None:
            applied.append(ns[nxt])
        elif nxt is None:
            applied.append(ns[prev])
        else:
            applied.append(slerp(ns[prev], ns[nxt], (i - prev) / (nxt - prev)))
    return applied, status


# ---------------------------------------------------------------------------
# Review drawing
# ---------------------------------------------------------------------------

def draw_panel(bgr: np.ndarray, fit: HorizonFit | None, title: str, lines: list[str]):
    out = bgr.copy()
    h, w = out.shape[:2]
    for x in range(0, w, 24):
        cv2.line(out, (x, h // 2), (x + 12, h // 2), (255, 255, 255), 1)
    if fit is not None:
        yc = horizon_curve(fit.n, fit.d, w, h)
        pts = np.c_[np.arange(w), yc].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, (0, 0, 255), 2)
    cv2.rectangle(out, (0, 0), (w, 28 + 22 * len(lines)), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for i, s in enumerate(lines):
        cv2.putText(out, s, (8, 44 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _small(bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(bgr, (DET_W, DET_H), interpolation=cv2.INTER_AREA)


def fit_pass(source: Path, info: dict, output_fps: float, start_frame: int,
             end_frame: int | None, log=print) -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    reader = av.RawVideoReader(source, info, output_fps, width=DET_W, height=DET_H,
                               start_frame=start_frame, end_frame=end_frame,
                               scale_flags="area")
    t0 = time.time()
    for i, small in enumerate(reader):
        fit = fit_horizon(small, rng)
        row = {"frame_index": start_frame + i,
               "video_t_s": (start_frame + i) / output_fps, "n": None, "post_tilt_deg": None}
        if fit is not None:
            levelled = level_image_cpu(small, level_rotation(fit.n))
            refit = fit_horizon(levelled, rng)
            row.update(n=fit.n, **tilt_params(fit.n), horizon_el_deg=fit.horizon_el_deg,
                       rms_deg=fit.rms_deg, cols=fit.cols, coverage=fit.coverage,
                       post_tilt_deg=(tilt_params(refit.n)["tilt_deg"] if refit else None),
                       grad_plane_delta_deg=fit.grad_plane_delta_deg, grad_cols=fit.grad_cols)
        rows.append(row)
        if i % 300 == 0:
            log(f"  fit pass: frame {start_frame + i}  ({time.time() - t0:.0f} s)")
    return rows


def render_pass(source: Path, info: dict, output_fps: float, rows: list[dict],
                start_frame: int, end_frame: int | None, out_video: Path,
                review_video: Path, encoder: str, device: str, review_width: int,
                log=print) -> None:
    w, h = info["width"], info["height"]
    rw, rh = review_width, review_width // 2
    leveler = GpuLeveler(w, h, device) if device != "cpu" else None
    reader = av.RawVideoReader(source, info, output_fps, start_frame=start_frame,
                               end_frame=end_frame)
    writer = av.RawVideoWriter(out_video, w, h, output_fps, encoder=encoder)
    review = av.RawVideoWriter(review_video, rw, rh * 2, output_fps, review=True,
                               encoder=encoder)
    rng = np.random.default_rng(0)
    t0 = time.time()
    written = 0
    try:
        for i, frame in enumerate(reader):
            if i >= len(rows):
                raise RuntimeError("render pass decoded more frames than the fit pass")
            row = rows[i]
            R = level_rotation(row["applied_n"])
            levelled = leveler.level(frame, R) if leveler else level_image_cpu(frame, R)
            writer.write(levelled)
            small = _small(levelled)
            refit = fit_horizon(small, rng)
            row["written_post_tilt_deg"] = tilt_params(refit.n)["tilt_deg"] if refit else None
            before = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
            after = cv2.resize(levelled, (rw, rh), interpolation=cv2.INTER_AREA)
            tp = tilt_params(row["applied_n"])
            before_fit = None
            if row["n"] is not None:
                before_fit = HorizonFit(n=row["n"], d=math.sin(math.radians(row["horizon_el_deg"])),
                                        rms_deg=0, cols=0, coverage=0, n_grad=None, d_grad=None,
                                        grad_cols=0, xs=np.array([]), ys=np.array([]), inliers=np.array([]))
            top = draw_panel(before, before_fit, f"t={row['video_t_s']:.3f}s  BEFORE",
                             [f"applied tilt {tp['tilt_deg']:.1f} deg (roll {tp['roll_deg']:+.1f}, "
                              f"pitch {tp['pitch_deg']:+.1f})  status={row['status']}"])
            bottom = draw_panel(after, refit, "AFTER levelling (yaw untouched)",
                                [f"re-detected tilt {row['written_post_tilt_deg']:.2f} deg"
                                 if row["written_post_tilt_deg"] is not None else "re-detection: no fit"])
            review.write(np.vstack([top, bottom]))
            written += 1
            if i % 300 == 0:
                log(f"  render pass: frame {start_frame + i}  ({time.time() - t0:.0f} s)")
    except BaseException:
        writer.close(publish=False)
        review.close(publish=False)
        raise
    if written != len(rows):
        writer.close(publish=False)
        review.close(publish=False)
        raise RuntimeError(f"render pass wrote {written} frames, fit pass had {len(rows)}")
    writer.close()
    review.close()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        wr.writeheader()
        for r in rows:
            n = r["applied_n"]
            ap = tilt_params(n)
            out = {k: r.get(k) for k in CSV_COLUMNS}
            out.update(applied_tilt_deg=ap["tilt_deg"], applied_roll_deg=ap["roll_deg"],
                       applied_pitch_deg=ap["pitch_deg"], applied_n_x=n[0], applied_n_y=n[1],
                       applied_n_z=n[2])
            wr.writerow({k: ("" if v is None else (f"{v:.6f}" if isinstance(v, float) else v))
                         for k, v in out.items()})


def gate_statistics(rows: list[dict]) -> dict:
    status = [r["status"] for r in rows]
    tilts = np.array([r["tilt_deg"] for r in rows if r["n"] is not None])
    written = np.array([r["written_post_tilt_deg"] for r in rows
                        if r.get("written_post_tilt_deg") is not None])
    return {
        "frames": len(rows),
        "status_counts": {s: status.count(s) for s in sorted(set(status))},
        "interpolated_fraction": 1.0 - status.count("fit") / len(rows),
        "tilt_as_delivered_deg": {"min": float(tilts.min()), "median": float(np.median(tilts)),
                                  "max": float(tilts.max())} if len(tilts) else None,
        "written_post_tilt_deg": {"median": float(np.median(written)),
                                  "p95": float(np.percentile(written, 95)),
                                  "max": float(written.max()),
                                  "fraction_below_0p5": float(np.mean(written < 0.5)),
                                  "n": int(len(written))} if len(written) else None,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--name", required=True, help="output stem, e.g. leg1")
    p.add_argument("--output_fps", type=float, default=3.0)
    p.add_argument("--start_s", type=float, default=0.0, help="inclusive, on the output grid")
    p.add_argument("--end_s", type=float, default=None, help="exclusive, on the output grid")
    p.add_argument("--encoder", choices=sorted(av.VIDEO_ENCODERS), default="nvenc")
    p.add_argument("--device", default="cuda", help="torch device for the 8K remap, or cpu")
    p.add_argument("--review_width", type=int, default=1280)
    p.add_argument("--fit_only", action="store_true",
                   help="stop after the fit pass and the CSV (no derivative)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    source = args.source.resolve()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}"
    manifest_path = out_dir / f"{stem}_leveling_manifest.json"
    csv_path = out_dir / f"{stem}_leveling.csv"
    out_video = out_dir / f"{stem}_{args.output_fps:g}fps_levelled.mp4"
    review_video = out_dir / f"{stem}_leveling_review.mp4"
    for p in (manifest_path, csv_path, out_video, review_video):
        if p.exists():
            raise FileExistsError(f"refusing to replace {p}")
    info = av.probe_video(source)
    if abs(info["width"] / info["height"] - 2.0) > 0.01:
        raise ValueError(f"not an equirectangular 2:1 video: {info['width']}x{info['height']}")
    start_frame = int(round(args.start_s * args.output_fps))
    end_frame = None if args.end_s is None else int(round(args.end_s * args.output_fps))
    print(f"source {source.name}: {info['width']}x{info['height']} {info['codec']} "
          f"{info['media_fps']:.4f} fps, {info['duration_s']:.1f} s -> output {args.output_fps:g} fps")
    rows = fit_pass(source, info, args.output_fps, start_frame, end_frame)
    applied, status = temporal_gate([r["n"] for r in rows], [r["post_tilt_deg"] for r in rows])
    for r, n, s in zip(rows, applied, status):
        r["applied_n"], r["status"] = n, s
    print(f"fit pass: {len(rows)} frames; status {gate_statistics(rows)['status_counts']}")
    if not args.fit_only:
        render_pass(source, info, args.output_fps, rows, start_frame, end_frame, out_video,
                    review_video, args.encoder, args.device, args.review_width)
    write_csv(csv_path, rows)
    manifest = {
        "schema": SCHEMA,
        "source": {"path": str(source), "sha256": artifact.sha256_file(source), **info},
        "settings": {
            "output_fps": args.output_fps, "start_frame": start_frame, "end_frame": end_frame,
            "fit_resolution": [DET_W, DET_H], "band_el_deg": BAND_EL_DEG,
            "el_range_deg": list(EL_RANGE_DEG), "ransac_iters": RANSAC_ITERS,
            "refine_gates_deg": list(REFINE_GATES_DEG), "temporal_window": TEMPORAL_WINDOW,
            "continuity_gate_deg": CONTINUITY_GATE_DEG, "self_check_gate_deg": SELF_CHECK_GATE_DEG,
            "encoder": None if args.fit_only else av.video_encoder_profile(args.encoder),
            "device": args.device,
        },
        "rotation_convention": (
            "Per frame, pixels are resampled by the minimal rotation taking the fitted "
            "horizon-plane normal to the camera's up axis. The rotation axis is horizontal, "
            "so yaw is untouched: the centre column stays the camera forward. "
            f"Camera frame: {geo.CAMERA_FRAME}"),
        "outputs": {
            "leveling_csv": {"path": csv_path.name, "sha256": artifact.sha256_file(csv_path)},
        },
        "statistics": gate_statistics(rows),
        "code_provenance": code_provenance.record(),
    }
    if not args.fit_only:
        manifest["outputs"]["levelled_video"] = {
            "path": out_video.name, "sha256": artifact.sha256_file(out_video),
            **av.probe_video(out_video)}
        manifest["outputs"]["review_video"] = {
            "path": review_video.name, "sha256": artifact.sha256_file(review_video)}
    artifact.atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest["statistics"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
