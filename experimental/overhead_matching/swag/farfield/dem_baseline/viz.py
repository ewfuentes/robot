"""Shared imagery + page helpers for the dem_baseline viewers.

Follows the farfield viewer conventions (viewers/page.py): self-contained
static HTML, relative links, no network — `python -m http.server` at the data
root must render everything. Pages are click-through-by-index with prev/next
links and arrow-key navigation.

Depth views are colormapped on a FIXED log-depth scale so the same color
means the same range on every page; sky is drawn as the page background so
terrain silhouettes read like skylines.
"""

import math

import numpy as np
from PIL import Image

# Fixed display range: log10(2 m) .. log10(30 km). One scale everywhere.
_LOG_MIN = math.log10(2.0)
_LOG_MAX = math.log10(30000.0)
_SKY_RGB = (16, 20, 26)  # page background (#10141a)


def _turbo_lut() -> np.ndarray:
    from matplotlib import cm
    return (cm.get_cmap("turbo")(np.linspace(0, 1, 256))[:, :3]
            * 255).astype(np.uint8)


def depth_image(depth_m: np.ndarray, max_px: int = 320) -> Image.Image:
    """Colormapped depth PNG; +inf/NaN/negative sky pixels use the page bg."""
    depth = np.asarray(depth_m, dtype=np.float32)
    finite = np.isfinite(depth) & (depth > 0)
    logd = np.zeros_like(depth)
    logd[finite] = np.log10(np.clip(depth[finite], 10 ** _LOG_MIN, None))
    norm = np.clip((logd - _LOG_MIN) / (_LOG_MAX - _LOG_MIN), 0.0, 1.0)
    lut = _turbo_lut()
    rgb = lut[(norm * 255).astype(np.uint8)]
    rgb[~finite] = _SKY_RGB
    image = Image.fromarray(rgb)
    if max(image.size) > max_px:
        scale = max_px / max(image.size)
        image = image.resize((round(image.width * scale),
                              round(image.height * scale)), Image.BILINEAR)
    return image


def depth_colorbar(width_px: int = 640, height_px: int = 14) -> Image.Image:
    """The one legend strip for the fixed log-depth scale."""
    lut = _turbo_lut()
    ramp = lut[np.linspace(0, 255, width_px).astype(np.uint8)][None]
    return Image.fromarray(np.repeat(ramp, height_px, axis=0))


def colorbar_html(rel_src: str) -> str:
    marks = "".join(
        f"<span>{label}</span>"
        for label in ("2 m", "20 m", "200 m", "2 km", "20 km"))
    return (
        '<div class="cbar"><img src="' + rel_src + '" alt="depth scale">'
        f'<div class="cbar-marks">{marks}</div>'
        "<div class='muted'>log-scaled metric depth; background = sky</div>"
        "</div>")


def hillshade_u8(elevation: np.ndarray, res_m: float,
                 max_px: int = 1200) -> np.ndarray:
    """Grayscale hillshade (azimuth 315, altitude 45), downsampled."""
    step = max(1, int(math.ceil(max(elevation.shape) / max_px)))
    elev = elevation[::step, ::step].astype(np.float32)
    gy, gx = np.gradient(elev, res_m * step)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azimuth, altitude = math.radians(315.0), math.radians(45.0)
    shade = (np.sin(altitude) * np.cos(slope)
             + np.cos(altitude) * np.sin(slope)
             * np.cos(azimuth - aspect))
    return (np.clip(shade, 0, 1) * 200 + 20).astype(np.uint8)


class MapRenderer:
    """Hillshade base map for one HeightField, reused across pages."""

    def __init__(self, height_field):
        self.hf = height_field
        self.shade = hillshade_u8(height_field.elevation, height_field.res)

    def render(self, out_path, *, markers: list,
               bounds_xy: tuple | None = None, figsize_px: int = 560,
               legend: bool = True) -> None:
        """markers: [(x, y, style_kwargs, label)] in the surface CRS."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x0, y_min, x1, y0 = (self.hf.bounds[0], self.hf.bounds[1],
                             self.hf.bounds[2], self.hf.bounds[3])
        km = 1e-3
        fig, ax = plt.subplots(
            figsize=(figsize_px / 100, figsize_px / 100), dpi=100)
        ax.imshow(self.shade, cmap="gray", vmin=0, vmax=255,
                  extent=[x0 * km, x1 * km, y_min * km, y0 * km])
        if bounds_xy is not None:
            bx0, by0, bx1, by1 = bounds_xy
            ax.plot([bx0 * km, bx1 * km, bx1 * km, bx0 * km, bx0 * km],
                    [by0 * km, by0 * km, by1 * km, by1 * km, by0 * km],
                    color="#7fb4e8", linewidth=0.8, linestyle="--")
        seen_labels = set()
        for x, y, style, label in markers:
            show = label if label not in seen_labels else None
            seen_labels.add(label)
            ax.scatter([x * km], [y * km], label=show, **style)
        if bounds_xy is not None:
            pad = 0.15 * max(bounds_xy[2] - bounds_xy[0],
                             bounds_xy[3] - bounds_xy[1])
            ax.set_xlim((bounds_xy[0] - pad) * km, (bounds_xy[2] + pad) * km)
            ax.set_ylim((bounds_xy[1] - pad) * km, (bounds_xy[3] + pad) * km)
        ax.set_xlabel("easting (km)")
        ax.set_ylabel("northing (km)")
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        if legend and seen_labels:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
        fig.tight_layout(pad=0.4)
        fig.savefig(out_path, facecolor="#14171c")
        plt.close(fig)


VIEWER_STYLE = """
.nav { display: flex; gap: 14px; align-items: baseline; margin: 10px 0 16px;
       font-size: 14px; }
.nav .idx { color: #8a94a3; }
.strip { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0 14px; }
.strip figure { margin: 0; text-align: center; }
.strip img { display: block; border-radius: 3px; }
.strip figcaption { font-size: 10px; color: #79828f; margin-top: 2px; }
.row2 { display: grid; grid-auto-flow: column; gap: 4px;
        justify-content: start; margin: 6px 0 14px; }
.pane { display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start; }
.pano img { max-width: 100%; border-radius: 4px; }
.cbar { margin: 10px 0 4px; }
.cbar img { display: block; width: 640px; max-width: 100%; height: 12px; }
.cbar-marks { display: flex; justify-content: space-between; width: 640px;
              max-width: 100%; font-size: 10px; color: #79828f; }
.mapimg img { border-radius: 4px; }
"""


def nav_html(index: int, count: int, page_name) -> str:
    """Prev/next links + position readout + arrow-key navigation.

    page_name: callable index -> relative filename.
    """
    prev_href = page_name(index - 1) if index > 0 else None
    next_href = page_name(index + 1) if index < count - 1 else None

    def link(label, href):
        if href is None:
            return f'<span class="muted">{label}</span>'
        return f'<a href="{href}">{label}</a>'

    keys = (
        "<script>document.addEventListener('keydown',e=>{"
        f"if(e.key==='ArrowLeft'{'' if prev_href else '&&false'})"
        f"location.href='{prev_href or ''}';"
        f"if(e.key==='ArrowRight'{'' if next_href else '&&false'})"
        f"location.href='{next_href or ''}';"
        "});</script>")
    return (f'<div class="nav">{link("&larr; prev", prev_href)}'
            f'<span class="idx">{index + 1} / {count}</span>'
            f'{link("next &rarr;", next_href)}'
            f'<a href="index.html">index</a></div>{keys}')


def thumb_strip(entries: list) -> str:
    """entries: [(rel_src, caption_html)] rendered as a labeled image row."""
    figures = "".join(
        f'<figure><img src="{src}"><figcaption>{caption}</figcaption>'
        "</figure>"
        for src, caption in entries)
    return f'<div class="strip">{figures}</div>'
