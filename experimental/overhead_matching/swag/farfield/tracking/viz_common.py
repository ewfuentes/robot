"""Shared drawing helpers for tracking visualizations."""

import hashlib

from PIL import Image, ImageDraw, ImageFont

from experimental.overhead_matching.swag.farfield import geometry as geo

PALETTE = [
    (230, 60, 60), (60, 160, 230), (90, 200, 90), (240, 170, 40),
    (200, 90, 220), (70, 220, 200), (240, 110, 180), (170, 200, 60),
    (120, 120, 250), (250, 140, 80),
]

DETECTION_COLOR = (40, 200, 70)   # green
MASK_COLOR = (255, 60, 60)        # red


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


def render_chip(pano, det_box, mask_box, out_path, chip_height: int):
    """Crop around the union of detection box and mask bbox (wrap-safe);
    detection box green, mask bbox red.

    Boxes are pano-pixel (possibly unwrapped) coordinates. The keyframe viewer
    and semantic-audit request builder share this helper so they render
    byte-identical chips.
    """
    pano_w = pano.shape[1]
    x0, y0, x1, y1 = det_box
    ux0, uy0, ux1, uy1 = x0, y0, x1, y1
    mask_rel = None
    if mask_box is not None:
        dx = geo.signed_x_offset(mask_box[0], x0, pano_w)
        mxa = x0 + dx
        mask_rel = (mxa, mask_box[1], mxa + (mask_box[2] - mask_box[0]),
                    mask_box[3])
        ux0, uy0 = min(ux0, mask_rel[0]), min(uy0, mask_rel[1])
        ux1, uy1 = max(ux1, mask_rel[2]), max(uy1, mask_rel[3])
    w, h = ux1 - ux0, uy1 - uy0
    mx, my = max(30, 0.25 * w), max(30, 0.25 * h)
    cw, ch = int(w + 2 * mx), int(h + 2 * my)
    crop, cy0 = geo.extract_window(pano, ux0 - mx, uy0 - my, cw, ch)
    img = Image.fromarray(crop)
    draw = ImageDraw.Draw(img)
    line_w = max(2, int(ch / 130))
    cx0 = geo.signed_x_offset(ux0, ux0 - mx, pano_w)  # = mx, wrap-safe
    draw.rectangle([cx0 + (x0 - ux0), y0 - cy0, cx0 + (x1 - ux0), y1 - cy0],
                   outline=DETECTION_COLOR, width=line_w)
    if mask_rel is not None:
        draw.rectangle([cx0 + (mask_rel[0] - ux0), mask_rel[1] - cy0,
                        cx0 + (mask_rel[2] - ux0), mask_rel[3] - cy0],
                       outline=MASK_COLOR, width=line_w)
    scale = chip_height / img.height
    img = img.resize((max(80, int(img.width * scale)), chip_height),
                     Image.BILINEAR)
    img.save(out_path, quality=88)
