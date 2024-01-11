from experimental.beacon_sim.world_map_config_pb2 import (
    WorldMapConfig,
    CorrelatedBeaconsConfig,
    Beacon,
)
from experimental.beacon_sim.beacon_potential_pb2 import BeaconPotential
from experimental.beacon_sim.mapped_landmarks_pb2 import MappedLandmarks
from experimental.beacon_sim.correlated_beacons_python import (
    create_correlated_beacons,
    BeaconClique,
)
from common.math.matrix_pb2 import Matrix
from planning.probabilistic_road_map_python import create_road_map, MapBounds, RoadmapCreationConfig
from planning.road_map_pb2 import RoadMap

from typing import Protocol, runtime_checkable

import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

from spatialmath import SE2


@runtime_checkable
class Drawable(Protocol):
    def draw(self, ax: plt.Axes) -> None:
        ...


class GridLandmark:
    def __init__(
        self,
        world_from_anchor: SE2,
        num_rows: int,
        num_cols: int,
        spacing_m: float,
        p_beacon: float,
        p_no_beacons: float,
        beacon_ns: str,
    ):
        """
        A grid landmark has rows that span the +x direction where landmarks are
        Additional rows start in the +y direction
        """
        self._world_from_anchor = world_from_anchor
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._spacing_m = spacing_m
        max_x = max(self._spacing_m * (self._num_cols - 1), 1)
        max_y = max(self._spacing_m * (self._num_rows - 1), 1)
        self._pts_in_anchor = np.array(
            [(0.0, 0.0), (max_x, 0.0), (max_x, max_y), (0.0, max_y), (0.0, 0.0)]
        ).T

        clique = BeaconClique(
            p_beacon=p_beacon,
            p_no_beacons=p_no_beacons,
            members=[(ord(beacon_ns) << 16) + i for i in range(len(self))],
        )

        self._dist = create_correlated_beacons(clique)

    def draw(self, ax: plt.Axes):
        pts_in_world = self._world_from_anchor * self._pts_in_anchor

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
        p_beacon: float,
        p_no_beacons: float,
        beacon_ns: str,
        rng: np.random.Generator,
    ):
        assert len(beacon_ns) == 1
        self._world_from_anchor = world_from_anchor
        self._width_m = width_m
        self._height_m = height_m

        area = self._width_m * self._height_m
        num_pts = int(density * area)

        xs = rng.uniform(0, self._width_m, num_pts)
        ys = rng.uniform(0, self._height_m, num_pts)

        self._pts_in_anchor = np.stack([xs, ys])

        clique = BeaconClique(
            p_beacon=p_beacon,
            p_no_beacons=p_no_beacons,
            members=[(ord(beacon_ns) << 16) + i for i in range(len(self))],
        )

        self._dist = create_correlated_beacons(clique)

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
        return self._pts_in_anchor.shape[1]


class Boardwalk(GridLandmark):
    def __init__(
        self,
        x_pos_m: float,
        p_beacon: float = 0.95,
        p_no_beacons: float = 0.05,
        beacon_ns: str = "b",
    ):
        assert len(beacon_ns) == 1
        super().__init__(
            SE2(x_pos_m, 50.0, -np.pi / 2.0),
            num_rows=3,
            num_cols=11,
            spacing_m=5.0,
            beacon_ns=beacon_ns,
            p_beacon=p_beacon,
            p_no_beacons=p_no_beacons,
        )


class BeachChairs(DiffuseLandmark):
    def __init__(
        self,
        x_pos_m: float,
        y_pos_m: float,
        width_m: float,
        height_m: float,
        rng: np.random.Generator,
        density: float = 0.02,
        p_beacon: float = 0.3,
        p_no_beacons: float = 0.5,
        beacon_ns: str = "c",
    ):
        assert len(beacon_ns) == 1
        super().__init__(
            SE2(x_pos_m, y_pos_m, 0.0),
            width_m,
            height_m,
            density=density,
            rng=rng,
            p_beacon=p_beacon,
            p_no_beacons=p_no_beacons,
            beacon_ns=beacon_ns,
        )


class Driftwood(DiffuseLandmark):
    def __init__(
        self,
        x_pos_m: float,
        y_pos_m: float,
        width_m: float,
        height_m: float,
        rng: np.random.Generator,
        p_beacon: float = 0.9,
        density: float = 0.005,
        beacon_ns: str = "w",
    ):
        num_beacons = int(width_m * height_m * density)
        p_no_beacons = (1 - p_beacon) ** num_beacons
        super().__init__(
            SE2(x_pos_m, y_pos_m, 0.0),
            width_m,
            height_m,
            density=density,
            rng=rng,
            p_beacon=p_beacon,
            p_no_beacons=p_no_beacons,
            beacon_ns=beacon_ns,
        )


def draw_landmarks(ax: plt.Axes, objs: list[Drawable]):
    for obj in objs:
        obj.draw(ax)


def map_config_to_proto(objs: list[DiffuseLandmark | GridLandmark]):
    pts_in_world = []
    ids = []
    combined_dist = None
    for obj in objs:
        pts_in_world.append(obj._world_from_anchor * obj._pts_in_anchor)
        ids += obj._dist.members()

        if combined_dist is None:
            combined_dist = obj._dist
        else:
            combined_dist *= obj._dist

    pts_in_world = np.concatenate(pts_in_world, axis=-1)

    beacons = []
    for (x, y), beacon_id in zip(pts_in_world.T, ids):
        b = Beacon()
        b.id = beacon_id
        b.pos_x_m = x
        b.pos_y_m = y
        beacons.append(b)

    beacon_potential = BeaconPotential()
    beacon_potential.ParseFromString(combined_dist.to_proto_string())

    beacon_config = CorrelatedBeaconsConfig()
    beacon_config.beacons.extend(beacons)
    beacon_config.potential.CopyFrom(beacon_potential)

    map_proto = WorldMapConfig()
    map_proto.correlated_beacons.CopyFrom(beacon_config)

    return map_proto


def mapped_landmarks_to_proto(objs: list[DiffuseLandmark | GridLandmark]):
    pts_in_world = []
    ids = []
    for obj in objs:
        pts_in_world.append(obj._world_from_anchor * obj._pts_in_anchor)
        ids += obj._dist.members()

    pts_in_world = np.concatenate(pts_in_world, axis=-1)

    def pt_in_world_to_matrix_proto(pt_in_world):
        out = Matrix()
        out.num_rows = 2
        out.num_cols = 1
        out.data.extend(pt_in_world)
        return out

    out = MappedLandmarks()
    out.beacon_ids.extend(ids)
    out.beacon_in_local.extend(map(pt_in_world_to_matrix_proto, pts_in_world.T))

    cov_in_local = Matrix()
    num_cols = 2 * len(ids)
    cov_in_local.num_rows = num_cols
    cov_in_local.num_cols = num_cols
    cov_in_local.data.extend((np.identity(num_cols) * 1e-9).flatten().tolist())

    out.cov_in_local.CopyFrom(cov_in_local)
    return out


def create_road_map_from_objs(objs: list[DiffuseLandmark | GridLandmark], random_seed: int):
    pts_in_world = []
    ids = []
    for obj in objs:
        pts_in_world.append(obj._world_from_anchor * obj._pts_in_anchor)
        ids += obj._dist.members()

    pts_in_world = np.concatenate(pts_in_world, axis=-1)

    map_bounds = MapBounds()
    map_bounds.bottom_left = np.min(pts_in_world, axis=1)
    map_bounds.top_right = np.max(pts_in_world, axis=1)

    config = RoadmapCreationConfig()
    config.seed = random_seed
    config.num_valid_points = 50
    config.desired_node_degree = 4

    road_map = create_road_map(map_bounds, config)

    road_map_proto = RoadMap()
    road_map_proto.ParseFromString(road_map.to_proto_string())
    return road_map_proto


def write_environment_to_file(output_path: Path, objs: list[DiffuseLandmark | GridLandmark]):
    # write out map config
    output_path.mkdir(exist_ok=True)
    map_config_proto = map_config_to_proto(objs)
    with open(output_path / 'world_map_config.pb', 'wb') as file_out:
        file_out.write(map_config_proto.SerializeToString())

    # write out mapped landmarks
    mapped_landmarks_proto = mapped_landmarks_to_proto(objs)
    with open(output_path / 'mapped_landmarks.pb', 'wb') as file_out:
        file_out.write(mapped_landmarks_proto.SerializeToString())

    # write out road map
    random_seed = sum([ord(c) for c in str(output_path)])
    road_map_proto = create_road_map_from_objs(objs, random_seed)
    with open(output_path / 'road_map.pb', 'wb') as file_out:
        file_out.write(road_map_proto.SerializeToString())
