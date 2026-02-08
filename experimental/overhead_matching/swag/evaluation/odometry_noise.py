"""Odometry noise model for corrupting perfect motion deltas.

Per-step noise applied independently to each motion delta:
- Sample north/east offsets in meters, proportional to step distance
- Convert offsets to lat/lon degrees and add to the original deltas

All computation happens in a local metric frame (meters, float64) to avoid lat/lon
precision issues, then converts back to lat/lon degrees at the end.
"""

import math
import torch
from dataclasses import dataclass

from common.gps.web_mercator import METERS_PER_DEG_LAT


@dataclass
class OdometryNoiseConfig:
    """Per-step odometry noise applied to each motion delta independently.

    sigma_noise_frac: noise std as fraction of step distance.
        For a step of d meters, north and east offsets are each
        sampled from N(0, sigma_noise_frac * d) meters.
    """
    sigma_noise_frac: float = 0.05    # fraction of step size
    seed: int = 7919

    def __post_init__(self):
        if self.sigma_noise_frac < 0:
            raise ValueError(
                f"sigma_noise_frac must be non-negative, got {self.sigma_noise_frac}"
            )


def compute_positions_from_deltas(
    start_latlon: torch.Tensor,
    motion_deltas: torch.Tensor,
) -> torch.Tensor:
    """Integrate motion deltas to get positions.

    Args:
        start_latlon: (2,) start position [lat, lon] in degrees.
        motion_deltas: (N, 2) deltas [delta_lat, delta_lon] in degrees.

    Returns:
        (N+1, 2) positions [lat, lon] in degrees, starting with start_latlon.
    """
    if start_latlon.shape != (2,):
        raise ValueError(f"start_latlon must have shape (2,), got {start_latlon.shape}")
    if motion_deltas.ndim != 2 or motion_deltas.shape[1] != 2:
        raise ValueError(f"motion_deltas must have shape (N, 2), got {motion_deltas.shape}")

    cumulative = torch.cumsum(motion_deltas, dim=0)
    positions = torch.cat([start_latlon.unsqueeze(0), start_latlon.unsqueeze(0) + cumulative], dim=0)
    return positions


def add_noise_to_motion_deltas(
    motion_deltas: torch.Tensor,
    start_latlon: torch.Tensor,
    config: OdometryNoiseConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add noise to perfect odometry motion deltas.

    For each step, computes the step distance in meters, samples i.i.d. north/east
    offsets proportional to that distance, and adds them to the delta in lat/lon.

    Args:
        motion_deltas: (N, 2) [delta_lat, delta_lon] in degrees.
        start_latlon: (2,) start position [lat, lon] in degrees.
        config: Noise configuration.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        (N, 2) noised deltas in degrees, same device as input.
    """
    if motion_deltas.ndim != 2 or motion_deltas.shape[1] != 2:
        raise ValueError(f"motion_deltas must have shape (N, 2), got {motion_deltas.shape}")
    if start_latlon.shape != (2,):
        raise ValueError(f"start_latlon must have shape (2,), got {start_latlon.shape}")

    original_dtype = motion_deltas.dtype
    original_device = motion_deltas.device

    # Work in float64 for precision
    deltas_f64 = motion_deltas.to(dtype=torch.float64)
    start_f64 = start_latlon.to(dtype=torch.float64)

    # Reference latitude for lon↔meters conversion
    ref_lat_rad = math.radians(start_f64[0].item())
    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(ref_lat_rad)

    # Compute step distances in meters
    north_m = deltas_f64[:, 0] * METERS_PER_DEG_LAT
    east_m = deltas_f64[:, 1] * meters_per_deg_lon
    distance = torch.sqrt(north_m ** 2 + east_m ** 2)

    # Sample north/east offsets in meters, proportional to step distance
    N = deltas_f64.shape[0]
    noise_north_m = torch.randn(N, dtype=torch.float64, device=deltas_f64.device, generator=generator) * (
        config.sigma_noise_frac * distance)
    noise_east_m = torch.randn(N, dtype=torch.float64, device=deltas_f64.device, generator=generator) * (
        config.sigma_noise_frac * distance)

    # Convert offsets to lat/lon degrees and add to original deltas
    noise_lat = noise_north_m / METERS_PER_DEG_LAT
    noise_lon = noise_east_m / meters_per_deg_lon

    noised_deltas = deltas_f64 + torch.stack([noise_lat, noise_lon], dim=1)

    return noised_deltas.to(dtype=original_dtype, device=original_device)
