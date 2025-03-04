import common.torch.load_torch_deps
import torch

def spherical_projection(points: torch.Tensor):
    """
    This function takes a ... x 3 tensor representing points in a frame and produces
    a ... x 3 tensor with the points in spherical coordinates where the coordinates
    are ordered as (distance, inclination, azimuth) where inclination refers to the
    angle from the x-y plane and azimuth increases from the +x axis to the + y axis
    """

    x = points[..., :1]
    y = points[..., 1:2]
    z = points[..., 2:3]

    azimuth_rad = torch.atan2(y, x)
    xy_dist = torch.hypot(x, y)
    inclination_rad = torch.atan2(z, xy_dist)
    r = torch.hypot(xy_dist, z)

    return torch.concatenate([r, inclination_rad, azimuth_rad], axis=-1)
