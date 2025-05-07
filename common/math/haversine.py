import math 
import numpy as np 

def find_d_on_unit_circle(point_one_lat_long_deg: tuple[float, float], 
                          point_two_lat_long_deg: tuple[float, float])-> float:
    """
    Calculate the distance between p1 (lat, long) and p2 (lat, long) on a unit circle.
    Scale by radius for distance on non-unit circle

    https://en.wikipedia.org/wiki/Haversine_formula

    """
    point_one_lat_long_deg = np.asarray(point_one_lat_long_deg, dtype=np.float64)
    point_two_lat_long_deg = np.asarray(point_two_lat_long_deg, dtype=np.float64)

    p1_rad = np.deg2rad(point_one_lat_long_deg)
    p2_rad = np.deg2rad(point_two_lat_long_deg)
    delta_phi, delta_lambda = p2_rad - p1_rad

    numerator = (
        1 - np.cos(delta_phi) 
        + np.cos(p1_rad[0]) * np.cos(p2_rad[0]) * (1 - np.cos(delta_lambda))
    )
    d = 2 * np.arcsin(np.sqrt(numerator / 2))
    return d
