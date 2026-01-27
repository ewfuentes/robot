"""Convergence metrics for histogram filter evaluation.

This module provides functions to measure how quickly the histogram filter
converges to the true location by tracking probability mass within specified
radii of the ground truth position over the course of a path.
"""

import common.torch.load_torch_deps
import torch
import math

from common.gps import web_mercator
from experimental.overhead_matching.swag.filter.histogram_belief import (
    HistogramBelief,
)


def get_meters_per_pixel(lat_deg: float, zoom_level: int) -> float:
    """Compute meters per pixel at a given latitude and zoom level.

    In Web Mercator projection, the scale factor varies with latitude.
    At the equator, the circumference is 2*pi*R meters, and at zoom level z,
    the map is 2^(8+z) pixels wide. At other latitudes, the east-west scale
    is reduced by cos(lat).

    Args:
        lat_deg: Latitude in degrees
        zoom_level: Web Mercator zoom level

    Returns:
        Meters per pixel at the given latitude
    """
    earth_circumference_m = 2 * math.pi * 6_371_000.0
    map_size_px = 2 ** (8 + zoom_level)
    meters_per_pixel_equator = earth_circumference_m / map_size_px
    # Scale by cos(lat) for latitude correction
    lat_rad = math.radians(lat_deg)
    return meters_per_pixel_equator * math.cos(lat_rad)


def compute_probability_mass_within_radius(
    belief: HistogramBelief,
    true_latlon: torch.Tensor,
    radius_meters: float,
) -> float:
    """Sum probability mass within radius of true position.

    Args:
        belief: Current histogram belief distribution
        true_latlon: (2,) tensor [lat, lon] of ground truth position in degrees
        radius_meters: Radius in meters within which to sum probability mass

    Returns:
        Probability mass within the specified radius (value in [0, 1])
    """
    # Get cell centers in pixel coordinates
    cell_centers_px = belief.grid_spec.get_all_cell_centers_px(belief.device)

    # Get true position in pixel coordinates
    true_lat = true_latlon[0]
    true_lon = true_latlon[1]
    if isinstance(true_lat, torch.Tensor):
        true_lat = true_lat.item()
        true_lon = true_lon.item()

    true_y_px, true_x_px = web_mercator.latlon_to_pixel_coords(
        true_lat, true_lon, belief.grid_spec.zoom_level
    )

    # Compute pixel distances from all cell centers to true position
    true_pos_px = torch.tensor(
        [[true_y_px, true_x_px]], device=belief.device, dtype=torch.float32
    )
    delta = cell_centers_px - true_pos_px
    dist_px = torch.norm(delta, dim=1)

    # Convert pixel distance to meters
    meters_per_pixel = get_meters_per_pixel(true_lat, belief.grid_spec.zoom_level)
    dist_meters = dist_px * meters_per_pixel

    # Sum probability within radius
    within_mask = dist_meters <= radius_meters
    prob_mass = belief.get_belief().flatten()[within_mask].sum()

    return prob_mass.item()


def compute_convergence_cost(
    prob_mass: torch.Tensor,
    distance_traveled: torch.Tensor,
) -> float:
    """Compute integrated convergence cost over path.

    The convergence cost is the area under the "missing probability" curve,
    computed as sum((1 - prob_mass) * delta_distance). Lower values indicate
    faster convergence to the true location.

    Args:
        prob_mass: (path_len + 1,) tensor of probability mass at each step
                   (index 0 is initial, indices 1..path_len are after each observation)
        distance_traveled: (path_len,) tensor of cumulative distance in meters

    Returns:
        Integrated convergence cost (lower is better)
    """
    if len(prob_mass) < 3 or len(distance_traveled) < 2:
        return 0.0

    # Compute distance increments: delta_dist[i] = distance from step i to step i+1
    # This has (path_len - 1) elements
    delta_dist = distance_traveled[1:] - distance_traveled[:-1]

    # Align prob_mass with distance segments:
    # For segment from step i to step i+1, use prob_mass at step i+1
    # prob_mass indices: 0=initial, 1=after step 0, 2=after step 1, ...
    # So for delta_dist[i] (step i to i+1), use prob_mass[i+2] (after step i+1)
    # This means we use prob_mass[2:] which has (path_len - 1) elements
    missing_prob = 1.0 - prob_mass[2:len(distance_traveled) + 1]

    # Ensure same length
    min_len = min(len(missing_prob), len(delta_dist))
    return (missing_prob[:min_len] * delta_dist[:min_len]).sum().item()


def compute_probability_mass_history(
    belief_history: list[HistogramBelief],
    true_latlons: torch.Tensor,
    radius_meters: float,
) -> torch.Tensor:
    """Compute probability mass within radius for a sequence of beliefs.

    Args:
        belief_history: List of histogram beliefs at each step
        true_latlons: (path_len, 2) tensor of ground truth positions [lat, lon]
        radius_meters: Radius in meters

    Returns:
        (path_len,) tensor of probability mass values
    """
    prob_masses = []
    for belief, true_latlon in zip(belief_history, true_latlons):
        prob_mass = compute_probability_mass_within_radius(
            belief, true_latlon, radius_meters
        )
        prob_masses.append(prob_mass)
    return torch.tensor(prob_masses)
