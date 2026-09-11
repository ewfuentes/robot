"""Panorama -> ring of calibrated perspective crops for CrossLocate queries.

A cylindrical/equirectangular panorama becomes M square pinhole crops whose
FOV and yaw spacing match the reference depth ring (default 12 x 30 degrees,
60 degree FOV, 500x500), per plan section 5.3. The relative yaw between crops
is known; the ring's global yaw is what retrieval searches.

Conventions follow ``farfield/geometry.py`` (the authority): camera-frame
azimuth is clockwise-positive and azimuth 0 is the CENTRE column of the
panorama; elevation is up-positive. Crops are indexed by camera-frame azimuth
``m * (360 / M)`` so crop index m at reference yaw bin n implies robot
heading ``wrap(psi_n - alpha_m)`` -- verified by the yaw round-trip test.

The existing ``extraction/panorama_to_pinhole.py`` renders CCW-yaw faces with
a per-channel wrap asymmetry; this module is the corrected CW-convention
implementation for dem_baseline queries (documented divergence, not reuse).
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from experimental.overhead_matching.swag.farfield import geometry


@dataclass(frozen=True)
class CropRingConfig:
    n_crops: int = 12
    fov_deg: float = 60.0
    width: int = 500
    height: int = 500

    def azimuths_deg(self) -> np.ndarray:
        """Camera-frame azimuth (CW from pano centre) of each crop's axis."""
        return np.arange(self.n_crops) * (360.0 / self.n_crops)


def crop_directions(config: CropRingConfig,
                    crop_azimuth_cw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """(az_cw_deg, el_up_deg) of every crop pixel, each (H, W)."""
    half_tan = math.tan(math.radians(config.fov_deg) / 2.0)
    x_norm = (np.arange(config.width) + 0.5) / config.width * 2.0 - 1.0
    y_norm = (np.arange(config.height) + 0.5) / config.height * 2.0 - 1.0
    tan_x = x_norm * half_tan  # +x right of axis == clockwise
    tan_up = -y_norm * half_tan  # row 0 is the top of the image
    az_offset_deg = np.degrees(np.arctan(tan_x))
    horiz = np.sqrt(1.0 + tan_x ** 2)
    el_deg = np.degrees(np.arctan(tan_up[:, None] / horiz[None, :]))
    az_deg = (crop_azimuth_cw_deg + az_offset_deg[None, :]) \
        * np.ones((config.height, 1))
    return az_deg % 360.0, el_deg


def extract_crop(panorama: np.ndarray, config: CropRingConfig,
                 crop_azimuth_cw_deg: float) -> np.ndarray:
    """One square pinhole crop from an equirectangular pano (H, W, C) uint8."""
    pano_h, pano_w = panorama.shape[:2]
    az_deg, el_deg = crop_directions(config, crop_azimuth_cw_deg)
    # Vectorized twin of geometry.pano_px_from_direction (same formulas; the
    # geometry helper is scalar-only).
    x = ((az_deg / 360.0 + 0.5) % 1.0) * pano_w
    y = np.clip((0.5 - el_deg / 180.0) * pano_h, 0.0, pano_h - 1.0)
    # Sample at pixel centers; wrap horizontally, clamp vertically.
    coords = np.stack([np.clip(y - 0.5, 0, pano_h - 1), (x - 0.5) % pano_w])
    channels = [
        ndimage.map_coordinates(panorama[:, :, c].astype(np.float32), coords,
                                order=1, mode="grid-wrap")
        for c in range(panorama.shape[2])
    ]
    crop = np.stack(channels, axis=-1)
    if panorama.dtype == np.uint8:
        crop = np.clip(np.rint(crop), 0, 255).astype(np.uint8)
    return crop


def extract_crop_ring(panorama: np.ndarray,
                      config: CropRingConfig) -> np.ndarray:
    """(M, H, W, C) ring of crops at the config's azimuths."""
    return np.stack([extract_crop(panorama, config, az)
                     for az in config.azimuths_deg()])


def implied_heading_cw_deg(reference_yaw_map_cw_deg: float,
                           crop_azimuth_cw_deg: float) -> float:
    """Robot heading implied by matching a crop to a reference view.

    Reference view at map yaw psi matching crop at body azimuth alpha means
    the platform's forward axis points at ``wrap(psi - alpha)`` (plan
    section 5.3's convention test).
    """
    return (reference_yaw_map_cw_deg - crop_azimuth_cw_deg) % 360.0
