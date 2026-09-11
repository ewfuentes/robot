"""Candidate-location lattices over a height field (plan section 5.1).

A lattice is a regular grid of candidate observer locations in the height
field's CRS. The search region is declared up front (bounds in metric
coordinates, from the experiment's prior -- never cropped around the truth
route); cells whose terrain has no valid source data are dropped and counted.

Hierarchical refinement (coarse global grid, fine re-render around retrieved
neighborhoods) composes from this: build a second, finer lattice over the
bounding boxes of the surviving coarse candidates.
"""

from dataclasses import dataclass

import numpy as np

from experimental.overhead_matching.swag.farfield.dem_baseline import terrain


@dataclass
class Lattice:
    """Candidate locations; index is the stable location ID."""

    x_m: np.ndarray  # (N,) easting
    y_m: np.ndarray  # (N,) northing
    spacing_m: float
    crs: str
    bounds_xy: tuple[float, float, float, float]
    n_dropped_nodata: int

    def __len__(self) -> int:
        return len(self.x_m)


def build_lattice(hf: terrain.HeightField, *, spacing_m: float,
                  bounds_xy: tuple[float, float, float, float] | None = None
                  ) -> Lattice:
    """Regular grid clipped to the height field and its valid data.

    ``bounds_xy`` (x_min, y_min, x_max, y_max) defaults to the height field's
    own bounds. Grid nodes are offset half a spacing from the bound edges so
    a shared declared region yields the same lattice for every method.
    """
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    hf_bounds = hf.bounds
    if bounds_xy is None:
        bounds_xy = hf_bounds
    x_min = max(bounds_xy[0], hf_bounds[0])
    y_min = max(bounds_xy[1], hf_bounds[1])
    x_max = min(bounds_xy[2], hf_bounds[2])
    y_max = min(bounds_xy[3], hf_bounds[3])
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(
            f"bounds {bounds_xy} do not intersect height field {hf_bounds}")

    xs = np.arange(x_min + spacing_m / 2.0, x_max, spacing_m)
    ys = np.arange(y_min + spacing_m / 2.0, y_max, spacing_m)
    gx, gy = np.meshgrid(xs, ys)
    x_flat = gx.ravel()
    y_flat = gy.ravel()

    # Drop candidates standing on no-data cells (elevation is untrustworthy
    # there); rendering *through* distant no-data is reported as coverage by
    # the renderer instead, so the lattice only vets the observer's own cell.
    col = np.clip(((x_flat - hf.x0) / hf.res).astype(np.int64), 0,
                  hf.elevation.shape[1] - 1)
    row = np.clip(((hf.y0 - y_flat) / hf.res).astype(np.int64), 0,
                  hf.elevation.shape[0] - 1)
    keep = ~hf.nodata_mask[row, col]

    return Lattice(x_m=x_flat[keep], y_m=y_flat[keep], spacing_m=spacing_m,
                   crs=hf.crs, bounds_xy=(x_min, y_min, x_max, y_max),
                   n_dropped_nodata=int((~keep).sum()))
