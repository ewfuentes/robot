import common.torch.load_torch_deps
import torch

import math 
import numpy as np 

def find_d_on_unit_circle(point_one_lat_long_deg: tuple[float, float], 
                          point_two_lat_long_deg: tuple[float, float])-> float:
    """
    Calculate the distance between p1 (lat, long) and p2 (lat, long) on a unit circle.
    Scale by radius for distance on non-unit circle

    https://en.wikipedia.org/wiki/Haversine_formula

    """
    if isinstance(point_one_lat_long_deg, torch.Tensor):
        deg2rad = torch.deg2rad
        cos = torch.cos
        sqrt = torch.sqrt
        arcsin = torch.asin
    else:
        point_one_lat_long_deg = np.asarray(point_one_lat_long_deg, dtype=np.float64)
        point_two_lat_long_deg = np.asarray(point_two_lat_long_deg, dtype=np.float64)
        deg2rad = np.deg2rad
        cos = np.cos
        sqrt = np.sqrt
        arcsin = np.arcsic

    p1_rad = deg2rad(point_one_lat_long_deg)
    p2_rad = deg2rad(point_two_lat_long_deg)
    dphi_dlam = p2_rad - p1_rad
    delta_phi = dphi_dlam[..., 0]
    delta_lambda = dphi_dlam[..., 1]

    numerator = (
        1 - cos(delta_phi) 
        + cos(p1_rad[..., 0]) * cos(p2_rad[..., 0]) * (1 - cos(delta_lambda))
    )
    d = 2 * arcsin(sqrt(numerator / 2))
    return d
