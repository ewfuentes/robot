"""Shared drawing helpers for object_tracking visualizations."""

import hashlib

from PIL import ImageDraw, ImageFont

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    pano_geometry as pg,
)

PALETTE = [
    (230, 60, 60), (60, 160, 230), (90, 200, 90), (240, 170, 40),
    (200, 90, 220), (70, 220, 200), (240, 110, 180), (170, 200, 60),
    (120, 120, 250), (250, 140, 80),
]


def load_font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def obs_semantic_label(obs) -> str:
    tags = dict(tuple(t) for t in obs.additional_tags)
    name = tags.get("name", "")
    label = f"lm{obs.landmark_idx} {obs.primary_tag_key}={obs.primary_tag_value}"
    if name:
        label += f" '{name}'"
    return label + f" ({obs.confidence[0]})"


def obs_color(obs):
    """Stable box color keyed on (primary tag, name) so the same semantic
    object keeps its color across frames."""
    tags = dict(tuple(t) for t in obs.additional_tags)
    key = f"{obs.primary_tag_key}={obs.primary_tag_value}|{tags.get('name', '')}"
    digest = hashlib.md5(key.encode()).digest()
    return PALETTE[digest[0] % len(PALETTE)]


def draw_caption(draw: ImageDraw.ImageDraw, text: str, font, xy=(5, 5)):
    draw.text((xy[0] + 1, xy[1] + 1), text, fill=(0, 0, 0), font=font)
    draw.text(xy, text, fill=(255, 255, 60), font=font)


def draw_obs_box(draw: ImageDraw.ImageDraw, obs, pano_w: int, pano_h: int,
                 window_x0: float, window_y0: float, view_w: int, view_h: int,
                 scale: float, font, highlight: bool = False,
                 with_label: bool = True):
    """Draw one observation's pano bbox on a window crop (or scaled pano).

    window_x0/window_y0 are the crop's top-left in pano pixels; scale maps
    pano pixels to view pixels. Returns True if the box intersected the view.
    """
    x_min, y_min, x_max, y_max = pg.pano_bbox_for_observation(
        obs.boxes, pano_w, pano_h)
    box_w = x_max - x_min
    rel_x = pg.signed_x_offset(x_min, window_x0, pano_w)
    if rel_x + box_w < 0 or rel_x > view_w / scale:
        return False
    x0 = rel_x * scale
    y0 = (y_min - window_y0) * scale
    x1 = (rel_x + box_w) * scale
    y1 = (y_max - window_y0) * scale
    if y1 < 0 or y0 > view_h:
        return False
    color = obs_color(obs)
    width = 4 if highlight else 2
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    if with_label:
        label = obs_semantic_label(obs)
        tx = min(max(x0, 2), view_w - 200)
        ty = y0 - 16 if y0 >= 16 else y1 + 2
        draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), label, fill=color, font=font)
    return True
