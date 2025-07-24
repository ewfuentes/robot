import common.torch.load_torch_deps
import torch
import numpy as np


def haversine(theta_rad: float):
    if isinstance(theta_rad, torch.Tensor):
        mod = torch
    else:
        mod = np

    return mod.sin(theta_rad / 2.0) ** 2


def haversine_lat_lon(p1_deg, p2_deg):
    if isinstance(p1_deg, torch.Tensor):
        mod = torch
    else:
        mod = np
    p1_rad = mod.deg2rad(p1_deg)
    p2_rad = mod.deg2rad(p2_deg)

    delta_lat_rad = p2_rad[..., 0] - p1_rad[..., 0]
    delta_lon_rad = p2_rad[..., 1] - p1_rad[..., 1]

    return (haversine(delta_lat_rad) +
            mod.cos(p1_rad[..., 0]) * mod.cos(p2_rad[..., 0]) * haversine(delta_lon_rad))


def find_d_on_unit_circle(point_one_lat_long_deg: tuple[float, float],
                          point_two_lat_long_deg: tuple[float, float]) -> float:
    """
    Calculate the distance between p1 (lat, long) and p2 (lat, long) on a unit circle.
    Scale by radius for distance on non-unit circle

    https://en.wikipedia.org/wiki/Haversine_formula

    """
    if isinstance(point_one_lat_long_deg, torch.Tensor):
        mod = torch
    else:
        point_one_lat_long_deg = np.asarray(point_one_lat_long_deg, dtype=np.float64)
        point_two_lat_long_deg = np.asarray(point_two_lat_long_deg, dtype=np.float64)
        mod = np

    return 2 * mod.arcsin(
            mod.sqrt(haversine_lat_lon(point_one_lat_long_deg, point_two_lat_long_deg)))
