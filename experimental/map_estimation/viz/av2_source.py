"""Reads one Argoverse 2 log off disk.

Paths come from :mod:`argoverse_layout` -- the same module the download manager writes with --
so the viewer cannot disagree with the downloader about where anything lives. Parsing is the
upstream ``av2`` devkit's job; this module only locates files and decides what is present.

**Every stream is optional.** A log's items are downloaded independently, so a `sensor/val` log
commonly has lidar and annotations but no imagery at all, while a `tbv` log has seven cameras
and no annotations. :meth:`LogSource.present_items` reports what is actually on disk, and the
accessors raise :class:`MissingStreamError` rather than returning something empty that a caller
would have to distinguish from a genuinely empty stream.

No rerun import: this half is exercised by tests that do not need a renderer, and prediction
tooling can reuse it for ego poses without pulling in a viewer.
"""

from pathlib import Path
from typing import Iterator

import numpy as np
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.sweep import Sweep
from av2.utils.io import read_city_SE3_ego, read_ego_SE3_sensor, read_feather

from experimental.map_estimation.data import argoverse_layout as al


class MissingStreamError(RuntimeError):
    """A stream was requested that is not present on disk for this log."""


class UnsupportedDatasetError(ValueError):
    """The dataset has no log-directory shape this module can read."""


def ensure_supported(request: al.Request) -> None:
    """Raise if this dataset has no log-directory shape to read.

    Separate from :class:`LogSource` so a caller that only wants to *list* logs can reject the
    dataset before printing anything. Listing motion-forecasting would otherwise find its
    scenario directories, announce them, and only then fail per-directory.
    """
    if request.dataset is al.Dataset.MOTION_FORECASTING:
        raise UnsupportedDatasetError(
            "motion-forecasting scenarios are a single parquet file, not a log directory "
            "with sensor streams; this viewer reads sensor, tbv, and lidar logs"
        )


def discover_log_ids(request: al.Request, root: Path = al.DEFAULT_ROOT) -> list[str]:
    """Log ids present on disk for `request`'s dataset+split, sorted.

    A directory is reported whether or not it is complete; completeness is per-item and is
    :meth:`LogSource.present_items`' business.
    """
    split_dir = request.local_dir(root)
    if not split_dir.is_dir():
        return []
    return sorted(entry.name for entry in split_dir.iterdir() if entry.is_dir())


class LogSource:
    """One log's streams, loaded lazily from disk.

    Args:
        request: dataset+split this log belongs to. Its ``items`` selection is ignored -- the
            source reports what is on disk, not what someone once asked to download.
        log_id: the log directory's name.
        root: local dataset root, mirroring S3.
    """

    def __init__(self, request: al.Request, log_id: str, root: Path = al.DEFAULT_ROOT) -> None:
        ensure_supported(request)
        self.request = request
        self.log_id = log_id
        self.root = Path(root)
        self.log_dir = al.log_dir(request, log_id, self.root)
        if not self.log_dir.is_dir():
            raise MissingStreamError(f"no log directory at {self.log_dir}")

    # -- what is here -------------------------------------------------------------------

    def _item_path(self, item: al._Item) -> Path:
        return al.local_path(self.request, self.log_id, item, self.root)

    def has(self, item: al._Item) -> bool:
        """Whether `item` is on disk. Directory items must also be non-empty."""
        path = self._item_path(item)
        if item.is_dir:
            return path.is_dir() and any(path.iterdir())
        return path.is_file()

    def present_items(self) -> tuple[al._Item, ...]:
        """Every item of this dataset that is actually on disk, in enum order."""
        return tuple(item for item in self.request.available_items() if self.has(item))

    def _require_named(self, member: str) -> Path:
        """Resolve an item by enum member name, then require it on disk.

        The per-dataset item enums omit what a dataset does not have -- ``TbvItem`` has no
        ANNOTATIONS, ``LidarItem`` has no cameras -- so a plain ``item_type.ANNOTATIONS`` raises
        AttributeError from inside the accessor. Going through here turns "this dataset has no
        such stream" and "this log did not download it" into the same actionable error.
        """
        item = getattr(self.request.item_type, member, None)
        if item is None:
            raise MissingStreamError(
                f"the {self.request.dataset.value} dataset has no {member.lower()}; "
                f"its items are: {', '.join(self.request.item_type.tokens())}"
            )
        if not self.has(item):
            raise MissingStreamError(
                f"{self.log_id} has no {item.token} at {self._item_path(item)}; "
                f"present items: {', '.join(i.token for i in self.present_items()) or 'none'}"
            )
        return self._item_path(item)

    # -- streams ------------------------------------------------------------------------

    def city_SE3_ego(self) -> dict[int, SE3]:
        """Egovehicle pose in the city frame, keyed by nanosecond timestamp.

        Sampled far denser than any sensor (~170 Hz), which is what will let each sensor stream
        be logged on its own timestamps without anyone having to synchronize them first.
        """
        self._require_named("POSES")
        return read_city_SE3_ego(self.log_dir)

    def static_map(self) -> ArgoverseStaticMap:
        """The log's vector HD map -- lane segments, crosswalks, drivable areas -- in city coords.

        ``build_raster=False`` because this is the vector map and nothing here rasterizes it:
        True would additionally rasterize every drivable-area polygon and load the ~1.6 MB
        ground-height surface, which costs seconds per log. Pass True only for
        ``get_ground_height_at_xy`` and friends.
        """
        map_dir = self._require_named("MAP")
        try:
            return ArgoverseStaticMap.from_map_dir(map_dir, build_raster=False)
        except RuntimeError as error:
            # has() only asks whether map/ is non-empty, and a partial download leaves the
            # ground-height .npy and the Sim2 json behind without the archive JSON. from_map_dir
            # reports that as a bare RuntimeError; make it the same error every other stream
            # raises so callers have one thing to catch.
            raise MissingStreamError(f"{self.log_id} has an unreadable map: {error}") from error

    def lidar_sweeps(self) -> Iterator[Sweep]:
        """The log's lidar sweeps in timestamp order, in the **egovehicle** frame.

        AV2 stacks two 32-beam Velodynes with overlapping fields of view, and the release has
        already egomotion-compensated both into the egovehicle frame at the sweep's reference
        timestamp -- so ``sweep.xyz`` needs no transform to be drawn on the vehicle, and
        ``offset_ns`` (which spans the full 106 ms revolution) has already been accounted for.

        A generator rather than a list: a 157-sweep log is 14.3 million points, and a caller that
        logs each sweep and drops it never has to hold more than one.

        NOT ``Sweep.from_feather``, which cannot read a tbv log: **tbv sweeps ship without the
        ``offset_ns`` column** -- their columns are x, y, z, intensity, laser_number -- and that
        loader indexes it unconditionally, so it raises KeyError on the entire dataset. Zeros
        stand in, which is what the column would say if the release had bothered to write it for
        data it has already motion-compensated. Reading here also hoists the sensor extrinsics
        out of the loop; the devkit re-reads that file once per sweep.

        Sorted explicitly, because the timestamp comes from the filename and ``glob`` order is
        arbitrary.
        """
        lidar_dir = self._require_named("LIDAR")
        # Required because Sweep carries the lidar extrinsics, not because anything drawing a
        # sweep needs them -- the points are already in the ego frame.
        self._require_named("CALIBRATION")
        sensor_poses = read_ego_SE3_sensor(self.log_dir)

        for sweep_path in sorted(lidar_dir.glob("*.feather")):
            table = read_feather(sweep_path)
            columns = {name: table[name].to_numpy() for name in table.columns}
            count = len(table)
            yield Sweep(
                # float16 on disk; widened to match what the devkit's own loader returns.
                xyz=np.stack([columns["x"], columns["y"], columns["z"]], axis=-1).astype(float),
                intensity=columns["intensity"],
                laser_number=columns["laser_number"],
                offset_ns=columns.get("offset_ns", np.zeros(count, dtype=np.int32)),
                timestamp_ns=int(sweep_path.stem),
                ego_SE3_up_lidar=sensor_poses["up_lidar"],
                ego_SE3_down_lidar=sensor_poses["down_lidar"],
            )

    def cameras(self) -> tuple[al._Item, ...]:
        """Every camera of this dataset that is on disk, in enum order.

        Empty rather than an error when there are none, which is the common case: cameras are
        downloaded per-camera, ``LidarItem`` declares none at all, and a log with no imagery is
        still worth drawing. Callers iterate this instead of asking for a camera by name.
        """
        return tuple(item for item in self.present_items() if item.is_camera)

    def camera_frames(self, item: al._Item) -> Iterator[tuple[int, Path]]:
        """``(timestamp_ns, jpeg path)`` for one camera, in timestamp order.

        Paths, not decoded images: rerun stores the jpeg bytes verbatim and decodes in the
        viewer, so nothing here should pay to decompress a frame nobody looks at.

        Sorted explicitly for the same reason :meth:`lidar_sweeps` is -- the timestamp lives in
        the filename and ``glob`` order is arbitrary.
        """
        camera_dir = self._require_named(item.name)
        for frame_path in sorted(camera_dir.glob("*.jpg")):
            yield int(frame_path.stem), frame_path

    def camera_model(self, item: al._Item) -> PinholeCamera:
        """Intrinsics and ``ego_SE3_cam`` for one camera.

        The devkit's loader is used as-is here, unlike for sweeps: it reads two *log-level*
        calibration files rather than one file per frame, so there is nothing to hoist out of a
        loop and no dataset it chokes on.

        ``intrinsics.feather`` also carries ``k1``, ``k2``, ``k3``, which this drops along with
        the devkit -- the released imagery is already undistorted, and nothing in ``av2`` reads
        those columns.
        """
        self._require_named("CALIBRATION")
        return PinholeCamera.from_feather(self.log_dir, item.token)
