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

from av2.geometry.se3 import SE3
from av2.utils.io import read_city_SE3_ego

from experimental.map_estimation.data import argoverse_layout as al


class MissingStreamError(RuntimeError):
    """A stream was requested that is not present on disk for this log."""


class UnsupportedDatasetError(ValueError):
    """The dataset has no log-directory shape this module can read."""


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
        if request.dataset is al.Dataset.MOTION_FORECASTING:
            raise UnsupportedDatasetError(
                "motion-forecasting scenarios are a single parquet file, not a log directory "
                "with sensor streams; this viewer reads sensor, tbv, and lidar logs"
            )
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
