
import numpy as np


def latlon_to_pixel_coords(lat_deg: float, lon_deg: float, zoom_level: int):
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)

    map_size = 2 ** (8 + zoom_level)
    scale_factor = map_size / (2 * np.pi)

    x = scale_factor * (np.pi - lon_rad)
    y = scale_factor * (np.pi - np.log(np.tan(np.pi / 4.0 - lat_rad / 2.0)))
    # Note that return type is row and column
    return y, x


def pixel_coords_to_latlon(py, px, zoom_level):
    map_size = 2 ** (8 + zoom_level)
    scale_factor = 2 * np.pi / map_size

    lon_rad = px * scale_factor - np.pi
    lat_rad = 2 * (np.arctan(np.exp(np.pi - py * scale_factor)) - np.pi / 4.0)

    lon_deg = np.degrees(lon_rad)
    lat_deg = np.degrees(lat_rad)
    return lat_deg, lon_deg
