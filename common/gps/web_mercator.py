
import common.torch.load_torch_deps
import torch
import numpy as np

MAX_LAT_RAD = 2 * np.arctan(np.pi) - np.pi / 2.0


def latlon_to_pixel_coords(lat_deg: float, lon_deg: float, zoom_level: int):
    if isinstance(lat_deg, torch.Tensor):
        deg2rad = torch.deg2rad
        abs_val = torch.abs
        all = torch.all
        log = torch.log
        tan = torch.tan
    else:
        deg2rad = np.radians
        abs_val = np.abs
        all = np.all
        log = np.log
        tan = np.tan

    lat_rad = deg2rad(lat_deg)
    lon_rad = deg2rad(lon_deg)
    assert all(abs_val(lat_rad) <= MAX_LAT_RAD)
    assert all(abs_val(lon_rad) <= np.pi)

    map_size = 2 ** (8 + zoom_level)
    pixel_from_rad = map_size / (2 * np.pi)

    x = pixel_from_rad * (np.pi + lon_rad)
    y = pixel_from_rad * (np.pi - log(tan(np.pi / 4.0 + lat_rad / 2.0)))
    # Note that return type is row and column
    return y, x


def pixel_coords_to_latlon(py: float, px: float, zoom_level: int):
    if isinstance(py, torch.Tensor):
        rad2deg = torch.rad2deg
        all = torch.all
        exp = torch.exp
        arctan = torch.atan
    else:
        rad2deg = np.degrees
        all = np.all
        exp = np.exp
        arctan = np.arctan

    map_size = 2 ** (8 + zoom_level)

    assert all(0 <= py) and all(py <= map_size)
    assert all(0 <= px) and all(px <= map_size)

    rad_from_pixel = 2 * np.pi / map_size

    lon_rad = rad_from_pixel * px - np.pi
    lat_rad = 2 * arctan(exp(np.pi - rad_from_pixel * py)) - np.pi / 2.0

    lon_deg = rad2deg(lon_rad)
    lat_deg = rad2deg(lat_rad)
    return lat_deg, lon_deg
