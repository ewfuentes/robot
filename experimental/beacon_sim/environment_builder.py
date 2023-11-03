from experimental.beacon_sim.world_map_config_pb2 import (
    WorldMapConfig,
    CorrelatedBeaconsConfig,
)
from experimental.beacon_sim.mapped_landmarks_pb2 import MappedLandmarks
from experimental.beacon_sim.correlated_beacons_python import (
    create_correlated_beacons,
    BeaconClique,
)

from typing import Protocol, runtime_checkable

import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from spatialmath import SE2


@runtime_checkable
class Drawable(Protocol):
    def draw(self, ax: plt.Axes) -> None:
        ...


class GridLandmark:
    def __init__(
        self, world_from_anchor: SE2, num_rows: int, num_cols: int, spacing_m: float
    ):
        """
        A grid landmark has rows that span the +x direction where landmarks are
        Additional rows start in the +y direction
        """
        self._world_from_anchor = world_from_anchor
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._spacing_m = spacing_m

    def draw(self, ax: plt.Axes):
        max_x = self._spacing_m * max(self._num_cols - 1, 1)
        max_y = self._spacing_m * max(self._num_rows - 1, 1)

        pts_in_anchor = np.array(
            [(0.0, 0.0), (max_x, 0.0), (max_x, max_y), (0.0, max_y), (0.0, 0.0)]
        ).T

        pts_in_world = self._world_from_anchor * pts_in_anchor

        outline = mpl.patches.Polygon(pts_in_world.T, alpha=0.25)

        ax.add_patch(outline)

        to_plot = []
        for r, c in itertools.product(range(self._num_rows), range(self._num_cols)):
            pt_in_anchor = np.array([[c * self._spacing_m, r * self._spacing_m]]).T
            to_plot.append(self._world_from_anchor * pt_in_anchor)
        to_plot = np.hstack(to_plot)
        ax.scatter(to_plot[0, :], to_plot[1, :])

    def __len__(self):
        return self._num_cols * self._num_rows


class DiffuseLandmark:
    def __init__(
        self,
        world_from_anchor: SE2,
        width_m: float,
        height_m: float,
        density: float,
        rng: np.random.Generator,
    ):
        self._world_from_anchor = world_from_anchor
        self._width_m = width_m
        self._height_m = height_m

        area = self._width_m * self._height_m
        num_pts = int(density * area)

        xs = rng.uniform(0, self._width_m, num_pts)
        ys = rng.uniform(0, self._height_m, num_pts)

        self._pts_in_anchor = np.stack([xs, ys])

    def draw(self, ax: plt.Axes):
        pts_in_anchor = np.array(
            [
                (0.0, 0.0),
                (self._width_m, 0.0),
                (self._width_m, self._height_m),
                (0.0, self._height_m),
                (0.0, 0.0),
            ]
        ).T

        pts_in_world = self._world_from_anchor * pts_in_anchor

        outline = mpl.patches.Polygon(pts_in_world.T, alpha=0.25, facecolor="red")

        ax.add_patch(outline)

        to_plot = []
        for pt in self._pts_in_anchor.T:
            to_plot.append(self._world_from_anchor * pt)
        to_plot = np.hstack(to_plot)
        ax.scatter(to_plot[0, :], to_plot[1, :])

    def __len__(self):
        return self._pts_in_anchor.shape[0]


class Boardwalk(GridLandmark):
    def __init__(
        self, x_pos_m: int, p_beacon: float, p_no_beacons: float, beacon_ns: str
    ):
        assert len(beacon_ns) == 1
        super().__init__(
            SE2(x_pos_m, 50.0, -np.pi / 2.0), num_rows=3, num_cols=11, spacing_m=5.0
        )

        clique = BeaconClique(
            p_beacon=p_beacon,
            p_no_beacons=p_no_beacons,
            members=[(ord(beacon_ns) << 16) + i for i in range(len(self))],
        )

        self._dist = create_correlated_beacons(clique)


def draw_landmarks(ax: plt.Axes, objs: list[Drawable]):
    for obj in objs:
        obj.draw(ax)
