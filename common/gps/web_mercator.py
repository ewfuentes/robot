
import numpy as np

MAX_LAT_RAD = 2 * np.arctan(np.pi) - np.pi / 2.0


def latlon_to_pixel_coords(lat_deg: float, lon_deg: float, zoom_level: int):
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    assert abs(lat_rad) <= MAX_LAT_RAD
    assert abs(lon_rad) <= np.pi

    map_size = 2 ** (8 + zoom_level)
    pixel_from_rad = map_size / (2 * np.pi)

    x = pixel_from_rad * (np.pi + lon_rad)
    y = pixel_from_rad * (np.pi - np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0)))
    # Note that return type is row and column
    return y, x


def pixel_coords_to_latlon(py: float, px: float, zoom_level: int):
    map_size = 2 ** (8 + zoom_level)

    assert 0 <= py and py <= map_size
    assert 0 <= px and px <= map_size

    rad_from_pixel = 2 * np.pi / map_size

    lon_rad = rad_from_pixel * px - np.pi
    lat_rad = 2 * np.arctan(np.exp(np.pi - rad_from_pixel * py)) - np.pi / 2.0

    lon_deg = np.degrees(lon_rad)
    lat_deg = np.degrees(lat_rad)
    return lat_deg, lon_deg
