"""Height-field depth rendering for CrossLocate-Depth reference views.

Renders rings of square perspective depth views (default: 12 headings at 30
degree spacing, 60 degree FOV, 500x500 -- the published CrossLocate database
geometry) from a terrain height field, on GPU via torch.

Algorithm: column scan. With zero pitch and roll, every image column sees a
single azimuth, so one 1-D march along that azimuth serves all rows: walk
outward computing the terrain's apparent elevation angle, keep the running
maximum, and each pixel's first hit is where the running max first reaches
that pixel's elevation angle (a searchsorted, since running maxima are
nondecreasing). This is exact for the crossing sample and ~500x cheaper than
per-pixel ray marching.

Earth curvature and standard atmospheric refraction enter as the usual
apparent-height drop ``s^2 / (2 R_eff)`` with ``R_eff = R / (1 - k)``; both
are fixed physical settings recorded in the render manifest, never fit to
imagery (plan section 7.3).

Conventions: yaw/azimuth is compass style, clockwise-positive from grid north
(+y of the height field), matching the farfield heading convention. Columns
run left to right; column x right of center is clockwise of the view axis.
The canonical product is metric slant range (float32, meters) with +inf sky;
the network tensor is derived from it separately (crosslocate_net).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import common.torch.load_torch_deps  # noqa: F401  (must precede torch)
import torch

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry
from experimental.overhead_matching.swag.farfield.dem_baseline import terrain

STANDARD_REFRACTION_K = 0.13


@dataclass(frozen=True)
class RenderConfig:
    """Geometry of one reference ring and the march that renders it."""

    n_yaw: int = 12
    fov_deg: float = 60.0
    width: int = 500
    height: int = 500
    observer_height_m: float = 1.7
    max_range_m: float = 30000.0
    min_range_m: float = 1.0
    # March step: max(min_step, min(res_scale * grid res, s * near_growth),
    # s * angular). Near field grows geometrically (interpolation error is
    # second order in step/s, so a fixed fraction of range bounds it); the
    # middle is capped at the grid resolution so no terrain cell is skipped;
    # far away the angular term tracks the pixel footprint instead.
    step_res_scale: float = 1.0
    step_angular: float = 0.001
    step_near_growth: float = 0.05
    min_step_m: float = 0.5
    curvature: bool = True
    refraction_k: float = STANDARD_REFRACTION_K

    def yaw_degrees(self) -> np.ndarray:
        return np.arange(self.n_yaw) * (360.0 / self.n_yaw)


@dataclass
class TerrainTensor:
    """HeightField staged on a torch device for repeated rendering.

    An optional ``background`` is a second, coarser TerrainTensor consulted
    wherever this one has no source data -- outside its grid, and in interior
    holes. A view from inside a 1 m region needs geometry out to the declared
    far range, and carrying that at 1 m would be billions of cells (a 27 km
    box grown by 30 km is 7.7e9), so the far field is a 30 m surface behind a
    fine foreground.
    """

    elevation: torch.Tensor  # (1, 1, H, W) float32
    valid: torch.Tensor  # (1, 1, H, W) float32, 0 where source had no data
    x0: float
    y0: float
    res: float
    step_distances: torch.Tensor = field(default=None)  # cached march schedule
    background: "TerrainTensor | None" = None

    @classmethod
    def from_height_field(cls, hf: terrain.HeightField,
                          device: str = "cuda",
                          background: "TerrainTensor | None" = None) \
            -> "TerrainTensor":
        elevation = torch.from_numpy(
            np.ascontiguousarray(hf.elevation)).float()
        valid = torch.from_numpy(
            (~hf.nodata_mask).astype(np.float32))
        return cls(elevation=elevation[None, None].to(device),
                   valid=valid[None, None].to(device),
                   x0=hf.x0, y0=hf.y0, res=hf.res, background=background)

    @classmethod
    def chain_from_height_fields(cls, fields: "Sequence[terrain.HeightField]",
                                 device: str = "cuda") \
            -> "TerrainTensor | None":
        """Stage fine-to-coarse surfaces as a fallback chain.

        ``fields[0]`` is the foreground; each later field is consulted only
        where every earlier one lacks source data. A bare-earth ring with its
        own ocean holes, backed by a global surface model, is the case this
        exists for.
        """
        staged = None
        for hf in reversed(list(fields)):
            staged = cls.from_height_field(hf, device=device,
                                           background=staged)
        return staged

    @property
    def device(self) -> torch.device:
        return self.elevation.device

    def _grid(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h, w = self.elevation.shape[-2:]
        gx = ((x - self.x0) / self.res - 0.5) / (w - 1) * 2.0 - 1.0
        gy = ((self.y0 - y) / self.res - 0.5) / (h - 1) * 2.0 - 1.0
        return torch.stack([gx, gy], dim=-1)[None]

    def _own_valid(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.grid_sample(
            self.valid, self._grid(x, y), mode="nearest",
            padding_mode="zeros", align_corners=True)
        return out[0, 0]

    def sample(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Bilinear elevation at metric (x, y).

        Border-clamped outside this grid, unless a background is attached, in
        which case the background's elevation is used wherever this grid has
        no source data (the border clamp would otherwise smear the edge row
        across the whole far field).
        """
        out = torch.nn.functional.grid_sample(
            self.elevation, self._grid(x, y), mode="bilinear",
            padding_mode="border", align_corners=True)[0, 0]
        if self.background is None:
            return out
        return torch.where(self._own_valid(x, y) > 0.5, out,
                           self.background.sample(x, y))

    def sample_valid(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Source-data validity at metric (x, y); 0 outside all grids."""
        out = self._own_valid(x, y)
        if self.background is None:
            return out
        return torch.maximum(out, self.background.sample_valid(x, y))


def march_schedule(config: RenderConfig, grid_res_m: float,
                   device) -> torch.Tensor:
    """Monotone horizontal march distances s_1..s_N for the column scan."""
    base = config.step_res_scale * grid_res_m
    steps = [config.min_range_m]
    s = config.min_range_m
    while s < config.max_range_m:
        step = max(config.min_step_m,
                   min(base, s * config.step_near_growth),
                   s * config.step_angular)
        s = s + step
        steps.append(min(s, config.max_range_m))
    return torch.tensor(steps, dtype=torch.float32, device=device)


def column_azimuths_rad(config: RenderConfig,
                        yaw_rad: torch.Tensor) -> torch.Tensor:
    """(n_yaw, W) azimuth of each image column, CW from north."""
    half_tan = math.tan(math.radians(config.fov_deg) / 2.0)
    x_norm = (torch.arange(config.width, dtype=torch.float32,
                           device=yaw_rad.device) + 0.5) \
        / config.width * 2.0 - 1.0
    # Column right of center (+x_norm) is clockwise of the view axis.
    offsets = torch.atan(x_norm * half_tan)
    return yaw_rad[:, None] + offsets[None, :]


def row_elevation_angles_rad(config: RenderConfig,
                             device) -> torch.Tensor:
    """(H, W) elevation angle of each pixel, up-positive.

    Rows are top to bottom; the off-axis column correction (rays in an outer
    column are closer to horizontal for the same row) is included.
    """
    half_tan = math.tan(math.radians(config.fov_deg) / 2.0)
    y_norm = (torch.arange(config.height, dtype=torch.float32, device=device)
              + 0.5) / config.height * 2.0 - 1.0
    x_norm = (torch.arange(config.width, dtype=torch.float32, device=device)
              + 0.5) / config.width * 2.0 - 1.0
    tan_up = -y_norm * half_tan  # top of image looks up
    horiz = torch.sqrt(1.0 + (x_norm * half_tan) ** 2)
    return torch.atan(tan_up[:, None] / horiz[None, :])


@dataclass
class RingDepth:
    """Rendered ring: metric slant depth with +inf sky, plus coverage."""

    depth_m: torch.Tensor  # (n_yaw, H, W) float32, +inf where sky
    yaw_deg: np.ndarray  # (n_yaw,)
    observer_xy: tuple[float, float]
    observer_z_m: float
    # Fraction of marched terrain samples per view that carried valid source
    # data (1.0 = full coverage through the render range).
    coverage: np.ndarray  # (n_yaw,)


def render_ring(tt: TerrainTensor, config: RenderConfig,
                observer_x: float, observer_y: float,
                observer_z_m: float | None = None) -> RingDepth:
    """Render one location's ring of depth views.

    ``observer_z_m`` defaults to terrain height at the observer plus
    ``config.observer_height_m``; pass an explicit value for water surfaces
    or surveyed camera heights.
    """
    device = tt.device
    if tt.step_distances is None or tt.step_distances.device != device:
        tt.step_distances = march_schedule(config, tt.res, device)
    s = tt.step_distances  # (S,)

    if observer_z_m is None:
        obs_terrain = tt.sample(
            torch.tensor([[observer_x]], device=device, dtype=torch.float32),
            torch.tensor([[observer_y]], device=device, dtype=torch.float32))
        observer_z_m = float(obs_terrain.item()) + config.observer_height_m

    yaw_rad = torch.deg2rad(
        torch.tensor(config.yaw_degrees(), dtype=torch.float32,
                     device=device))
    az = column_azimuths_rad(config, yaw_rad)  # (n_yaw, W)
    n_cols = config.n_yaw * config.width
    az_flat = az.reshape(n_cols)

    # Terrain sample positions for every (column, step): (n_cols, S).
    dx = torch.sin(az_flat)[:, None] * s[None, :]
    dy = torch.cos(az_flat)[:, None] * s[None, :]
    heights = tt.sample(observer_x + dx, observer_y + dy)
    if config.curvature:
        r_eff = geometry.MEAN_EARTH_RADIUS_M / (1.0 - config.refraction_k)
        heights = heights - s[None, :] ** 2 / (2.0 * r_eff)

    # Apparent elevation angle of each sample, then its running maximum.
    elev_angle = torch.atan2(heights - observer_z_m, s[None, :])
    running_max = torch.cummax(elev_angle, dim=1).values  # nondecreasing

    # First step index where the running max reaches each pixel's elevation.
    pixel_elev = row_elevation_angles_rad(config, device)  # (H, W)
    pixel_per_col = pixel_elev.t().reshape(1, config.width, config.height) \
        .expand(config.n_yaw, -1, -1).reshape(n_cols, config.height)
    hit_idx = torch.searchsorted(running_max.contiguous(),
                                 pixel_per_col.contiguous())  # (n_cols, H)
    sky = hit_idx >= s.shape[0]
    hit_clamped = hit_idx.clamp(max=s.shape[0] - 1)

    # Sub-step refinement: linear interpolation of the running-max curve
    # across the crossing interval.
    s_hi = s[hit_clamped]
    e_hi = torch.gather(running_max, 1, hit_clamped)
    prev = (hit_clamped - 1).clamp(min=0)
    s_lo = s[prev]
    e_lo = torch.gather(running_max, 1, prev)
    denom = (e_hi - e_lo).clamp(min=1e-9)
    frac = ((pixel_per_col - e_lo) / denom).clamp(0.0, 1.0)
    s_hit = torch.where(hit_clamped > 0, s_lo + frac * (s_hi - s_lo), s_hi)

    depth = s_hit / torch.cos(pixel_per_col)
    depth = torch.where(sky, torch.full_like(depth, float("inf")), depth)
    # (n_cols, H) -> (n_yaw, W, H) -> (n_yaw, H, W)
    depth = depth.reshape(config.n_yaw, config.width, config.height) \
        .permute(0, 2, 1).contiguous()

    valid = tt.sample_valid(observer_x + dx, observer_y + dy)
    coverage = valid.reshape(config.n_yaw, config.width, -1) \
        .mean(dim=(1, 2)).cpu().numpy()

    return RingDepth(depth_m=depth,
                     yaw_deg=config.yaw_degrees(),
                     observer_xy=(observer_x, observer_y),
                     observer_z_m=observer_z_m,
                     coverage=coverage)


@dataclass
class CylinderDepth:
    """360-degree cylindrical depth strip, aligned to an equirectangular
    panorama: columns are uniform azimuth (col 0 = north, CW-increasing),
    rows are uniform elevation angle (row 0 = ``elev_max_deg``, top)."""

    depth_m: torch.Tensor  # (n_rows, n_az) float32, +inf where sky
    elev_deg: np.ndarray  # (n_rows,) top to bottom
    observer_xy: tuple[float, float]
    observer_z_m: float
    coverage: float  # fraction of marched samples with valid source data


def render_cylinder(tt: TerrainTensor, config: RenderConfig,
                    observer_x: float, observer_y: float, *,
                    n_az: int = 1440, elev_min_deg: float = -20.0,
                    elev_max_deg: float = 20.0, n_rows: int = 320,
                    observer_z_m: float | None = None) -> CylinderDepth:
    """Render one cylindrical (panoramic) depth strip.

    Same column-scan core as ``render_ring``; each column is its own
    vertical plane, so rows are exactly uniform in elevation angle -- the
    vertical mapping of an equirectangular panorama. Column azimuths are
    compass style (CW from grid north), so the strip aligns column-for-column
    with a heading-rolled equirectangular photo.
    """
    device = tt.device
    if tt.step_distances is None or tt.step_distances.device != device:
        tt.step_distances = march_schedule(config, tt.res, device)
    s = tt.step_distances

    if observer_z_m is None:
        obs_terrain = tt.sample(
            torch.tensor([[observer_x]], device=device, dtype=torch.float32),
            torch.tensor([[observer_y]], device=device, dtype=torch.float32))
        observer_z_m = float(obs_terrain.item()) + config.observer_height_m

    az_flat = (torch.arange(n_az, dtype=torch.float32, device=device) + 0.5) \
        / n_az * 2.0 * math.pi
    dx = torch.sin(az_flat)[:, None] * s[None, :]
    dy = torch.cos(az_flat)[:, None] * s[None, :]
    heights = tt.sample(observer_x + dx, observer_y + dy)
    if config.curvature:
        r_eff = geometry.MEAN_EARTH_RADIUS_M / (1.0 - config.refraction_k)
        heights = heights - s[None, :] ** 2 / (2.0 * r_eff)
    elev_angle = torch.atan2(heights - observer_z_m, s[None, :])
    running_max = torch.cummax(elev_angle, dim=1).values

    elev_deg = np.linspace(elev_max_deg, elev_min_deg, n_rows)
    pixel_elev = torch.deg2rad(
        torch.tensor(elev_deg, dtype=torch.float32, device=device))
    pixel_per_col = pixel_elev[None, :].expand(n_az, -1)
    hit_idx = torch.searchsorted(running_max.contiguous(),
                                 pixel_per_col.contiguous())
    sky = hit_idx >= s.shape[0]
    hit_clamped = hit_idx.clamp(max=s.shape[0] - 1)
    s_hi = s[hit_clamped]
    e_hi = torch.gather(running_max, 1, hit_clamped)
    prev = (hit_clamped - 1).clamp(min=0)
    s_lo = s[prev]
    e_lo = torch.gather(running_max, 1, prev)
    denom = (e_hi - e_lo).clamp(min=1e-9)
    frac = ((pixel_per_col - e_lo) / denom).clamp(0.0, 1.0)
    s_hit = torch.where(hit_clamped > 0, s_lo + frac * (s_hi - s_lo), s_hi)
    depth = s_hit / torch.cos(pixel_per_col)
    depth = torch.where(sky, torch.full_like(depth, float("inf")), depth)

    valid = tt.sample_valid(observer_x + dx, observer_y + dy)
    return CylinderDepth(depth_m=depth.t().contiguous(),
                         elev_deg=elev_deg,
                         observer_xy=(observer_x, observer_y),
                         observer_z_m=observer_z_m,
                         coverage=float(valid.mean().item()))
