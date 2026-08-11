"""Pinhole-face <-> equirectangular panorama geometry.

Implements the exact inverse of scripts/panorama_to_pinhole.py, verified
empirically against the stored pinhole renders (regenerating faces from the
panorama with this math reproduces them to JPEG noise).

Conventions:
- Pinhole bbox coordinates are normalized 0-1000 per face, y down
  (Gemini convention).
- The render maps pinhole column fraction u in [0 (left), 1 (right)] at face
  yaw phi to a CCW-positive azimuth az_ccw = phi + atan((1 - 2u) tan(fov/2)),
  and pano column x = ((180 - az_ccw) / 360 mod 1) * W. Faces therefore sit
  left-to-right in the panorama as 180 | 90 | 0 | 270.
- Public azimuth here is **clockwise-positive from camera forward**
  (az_cw = -az_ccw), so pano x is increasing in az_cw:
  x = ((az_cw / 360 + 0.5) mod 1) * W.
- Public elevation is up-positive; pano row y = (0.5 - el_up / 180) * H.
- NOTE: this differs from bearing_geometry.bearing_camera_deg, whose per-face
  bearings mirror the true direction within each face (they only agree at
  face centers of the 0/180 faces). Use this module for anything that must
  land on panorama pixels.

Pano-space boxes may straddle the +-180 wrap; they are represented unwrapped:
x_min in [0, W), x_max in (x_min, x_min + W], so x_max > W means the box
wraps around the seam.
"""

import math

import numpy as np

BBOX_NORM_MAX = 1000.0


def direction_from_face_px(face_yaw_deg: float, x_norm: float, y_norm: float,
                           fov_deg: float = 90.0):
    """(az_cw_deg, el_up_deg) of a normalized pinhole-face pixel."""
    half_tan = math.tan(math.radians(fov_deg) / 2.0)
    # Render ray in face frame, z normalized to 1: [c, r, 1], c right-negative.
    c = (1.0 - 2.0 * x_norm / BBOX_NORM_MAX) * half_tan
    r = (2.0 * y_norm / BBOX_NORM_MAX - 1.0) * half_tan
    az_ccw_deg = face_yaw_deg + math.degrees(math.atan(c))
    el_down_deg = math.degrees(math.atan2(r, math.hypot(c, 1.0)))
    return (-az_ccw_deg) % 360.0, -el_down_deg


def pano_px_from_direction(az_cw_deg: float, el_up_deg: float,
                           pano_w: int, pano_h: int):
    """(x, y) pano pixel of a direction. x in [0, W), y clamped to [0, H)."""
    x = ((az_cw_deg / 360.0 + 0.5) % 1.0) * pano_w
    y = (0.5 - el_up_deg / 180.0) * pano_h
    return x, min(max(y, 0.0), pano_h - 1.0)


def direction_from_pano_px(x: float, y: float, pano_w: int, pano_h: int):
    """(az_cw_deg, el_up_deg) of a pano pixel; inverse of pano_px_from_direction."""
    az_cw_deg = (x / pano_w - 0.5) * 360.0 % 360.0
    el_up_deg = (0.5 - y / pano_h) * 180.0
    return az_cw_deg, el_up_deg


def _edge_samples(xmin: float, ymin: float, xmax: float, ymax: float,
                  n_per_edge: int):
    """Points along the four edges of a bbox, in normalized face coords."""
    ts = np.linspace(0.0, 1.0, n_per_edge)
    xs = xmin + (xmax - xmin) * ts
    ys = ymin + (ymax - ymin) * ts
    top = np.stack([xs, np.full_like(xs, ymin)], axis=1)
    bottom = np.stack([xs, np.full_like(xs, ymax)], axis=1)
    left = np.stack([np.full_like(ys, xmin), ys], axis=1)
    right = np.stack([np.full_like(ys, xmax), ys], axis=1)
    return np.concatenate([top, bottom, left, right], axis=0)


def pano_bbox_from_face_bbox(face_yaw_deg: float, xmin: float, ymin: float,
                             xmax: float, ymax: float, pano_w: int,
                             pano_h: int, fov_deg: float = 90.0,
                             n_per_edge: int = 9):
    """Pano-pixel bbox (x_min, y_min, x_max, y_max) of one pinhole-face bbox.

    Edge points are sampled because straight pinhole edges curve in the
    equirectangular projection. x is unwrapped (see module docstring).
    """
    pts = _edge_samples(xmin, ymin, xmax, ymax, n_per_edge)
    center_az, _ = direction_from_face_px(
        face_yaw_deg, (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, fov_deg)
    xs, ys = [], []
    for x_norm, y_norm in pts:
        az, el = direction_from_face_px(face_yaw_deg, x_norm, y_norm, fov_deg)
        # Unwrap azimuth to within +-180 of the box center, keep x unwrapped.
        az_unwrapped = center_az + (az - center_az + 180.0) % 360.0 - 180.0
        xs.append((az_unwrapped / 360.0 + 0.5) * pano_w)
        ys.append(min(max((0.5 - el / 180.0) * pano_h, 0.0), pano_h - 1.0))
    x_min, x_max = min(xs), max(xs)
    # Normalize so x_min lands in [0, W).
    shift = math.floor(x_min / pano_w) * pano_w
    return x_min - shift, min(ys), x_max - shift, max(ys)


def pano_bbox_union(boxes_pano, pano_w: int):
    """Union of unwrapped pano bboxes (e.g. a seam-merged observation).

    Boxes are re-unwrapped around the first box's center so a group that
    straddles the seam stays contiguous.
    """
    ref = (boxes_pano[0][0] + boxes_pano[0][2]) / 2.0
    xs, ys = [], []
    for x_min, y_min, x_max, y_max in boxes_pano:
        center = (x_min + x_max) / 2.0
        shift = round((center - ref) / pano_w) * pano_w
        xs.extend([x_min - shift, x_max - shift])
        ys.extend([y_min, y_max])
    x_min, x_max = min(xs), max(xs)
    shift = math.floor(x_min / pano_w) * pano_w
    return x_min - shift, min(ys), x_max - shift, max(ys)


def pano_bbox_for_observation(obs_boxes, pano_w: int, pano_h: int,
                              fov_deg: float = 90.0):
    """Pano bbox of an ingest Observation's (possibly seam-merged) box group.

    obs_boxes: iterable of objects with face_yaw_deg/xmin/ymin/xmax/ymax
    attributes (artifact_schema.BBox is not imported to keep this module
    numpy-only).
    """
    boxes = [
        pano_bbox_from_face_bbox(b.face_yaw_deg, b.xmin, b.ymin, b.xmax,
                                 b.ymax, pano_w, pano_h, fov_deg)
        for b in obs_boxes
    ]
    return pano_bbox_union(boxes, pano_w)


def x_offset_in_window(x: float, window_x0: float, pano_w: int) -> float:
    """Horizontal offset of unwrapped pano x inside a window starting at
    window_x0, respecting the circular wrap."""
    return (x - window_x0) % pano_w


def signed_x_offset(x: float, window_x0: float, pano_w: int) -> float:
    """Signed horizontal offset of pano x from window_x0 in [-W/2, W/2)."""
    return (x - window_x0 + pano_w / 2.0) % pano_w - pano_w / 2.0


def extract_window(pano: np.ndarray, x0: float, y0: float, width: int,
                   height: int):
    """Crop a (H, W, C) pano with horizontal wrap; vertical range is clamped.

    x0/y0 are the requested top-left corner in pano pixels (x may be
    unwrapped). Returns (crop, y_start) where y_start is the actual top row
    after clamping the window inside the pano vertically.
    """
    pano_h, pano_w = pano.shape[:2]
    cols = (np.arange(int(round(x0)), int(round(x0)) + width) % pano_w)
    y_start = int(round(min(max(y0, 0), pano_h - height)))
    return pano[y_start:y_start + height][:, cols], y_start
