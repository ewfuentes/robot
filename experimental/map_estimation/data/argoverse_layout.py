"""Argoverse 2 dataset layout: what exists, where it lives, and what may be requested.

This module is the single source of truth for both S3 URIs and local paths, and it is pure --
it performs no I/O, so it is cheap to unit test.

The organizing idea is that **the four AV2 datasets have genuinely different contents**, so each
gets its own item enum:

    dataset             splits            annotations  lidar  cameras
    ------------------  ----------------  -----------  -----  ---------------------
    sensor              train/val/test    yes*         yes    9 (7 ring + 2 stereo)
    tbv                 none (flat)       no           yes    7 (ring only)
    lidar               train/val/test    no           yes    none
    motion-forecasting  train/val/test    n/a          no     none

    * except the test split, which ships no annotations.

Because ``LidarItem`` has no camera members, asking for a camera from the lidar dataset is not
a mistake that can be spelled -- it fails at the call site with ``AttributeError``, with or
without a static type checker. Likewise ``TbvRequest`` has no ``split`` field at all, because
TBV is a flat listing. The single constraint the type system cannot express is annotations on
``sensor/test``, which is checked in ``SensorRequest.__post_init__``.
"""

import enum
import re
from pathlib import Path
from typing import Iterable, Sequence, TypeAlias

import msgspec

from common.python.serialization import MSGSPEC_STRUCT_OPTS

# Root of the AV2 release in the public Argoverse bucket.
S3_ROOT = "s3://argoverse/datasets/av2"

# Where downloaded data lands. The tree below this mirrors S3 exactly, which is what makes the
# result directly loadable by the upstream `av2` package and makes the S3 URI of any local file
# a pure string substitution.
DEFAULT_ROOT = Path("/data/map_estimation/datasets/argoverse")

# Relpath prefixes used to derive item groupings, so the groups cannot drift out of sync with
# the per-dataset member sets.
_LIDAR_RELPATH = "sensors/lidar"
_CAMERA_RELPATH_PREFIX = "sensors/cameras/"


class UnknownItemError(ValueError):
    """A requested item name is not valid for the dataset in question."""


class UnknownSplitError(ValueError):
    """A requested split is not valid for the dataset in question."""


class Dataset(enum.Enum):
    """The four AV2 datasets. Values are the S3 (and local) directory names."""

    SENSOR = "sensor"
    TBV = "tbv"
    LIDAR = "lidar"
    MOTION_FORECASTING = "motion-forecasting"


class _Item(str, enum.Enum):
    """Base for the per-dataset item enums.

    Members are declared as ``(token, relpath, is_dir)``:

    * ``token`` is the CLI spelling and the enum *value*. It is always the lowercased member
      name -- stated explicitly because msgspec only serializes enums whose values are all
      strings, and ``__new__`` cannot see the member name. ``test_token_matches_member_name``
      keeps the two from drifting.
    * ``relpath`` is where the item sits inside a log directory. It may contain a ``{log_id}``
      placeholder for datasets whose filenames embed the id.
    * ``is_dir`` marks items that span many objects, whose S3 source must be wildcarded.
    """

    def __new__(cls, token: str, relpath: str, is_dir: bool) -> "_Item":
        obj = str.__new__(cls, token)
        obj._value_ = token
        obj.relpath = relpath
        obj.is_dir = is_dir
        return obj

    def __str__(self) -> str:
        return self.token

    @property
    def token(self) -> str:
        """The CLI spelling of this item, e.g. 'ring_front_center'."""
        return self._value_

    @property
    def is_camera(self) -> bool:
        return self.relpath.startswith(_CAMERA_RELPATH_PREFIX)

    @property
    def is_lidar(self) -> bool:
        return self.relpath == _LIDAR_RELPATH

    def resolve_relpath(self, log_id: str) -> str:
        """The item's path inside a log dir, with any ``{log_id}`` placeholder filled in."""
        if "{log_id}" in self.relpath:
            return self.relpath.format(log_id=log_id)
        return self.relpath

    @classmethod
    def tokens(cls) -> tuple[str, ...]:
        return tuple(item.token for item in cls)

    @classmethod
    def dataset_label(cls) -> str:
        """The dataset this item enum belongs to, for error messages.

        Resolved through ITEM_TYPES at call time rather than stored, since ITEM_TYPES is defined
        after the enums.
        """
        for dataset, item_type in ITEM_TYPES.items():
            if item_type is cls:
                return dataset.value
        return cls.__name__

    @classmethod
    def from_token(cls, token: str) -> "_Item":
        """Look up an item by its CLI spelling.

        Raises UnknownItemError naming the dataset and its valid items, which is what makes a
        CLI typo (or a camera requested from the lidar dataset) actionable.
        """
        try:
            return cls[token.strip().upper()]
        except KeyError:
            raise UnknownItemError(
                f"{token!r} is not a valid item for the {cls.dataset_label()} dataset. "
                f"valid items: {', '.join(cls.tokens())}"
            ) from None

    # Group helpers. These are derived from relpaths rather than hardcoded, so every dataset's
    # groups are automatically correct for its own member set.
    @classmethod
    def metadata(cls) -> tuple["_Item", ...]:
        """Everything that is not a sensor stream: map, calibration, poses, annotations.

        Tiny (~2 MB per sensor log, 0.2% of the log) and almost always wanted, which is why
        this is the default selection.
        """
        return tuple(item for item in cls if not item.is_camera and not item.is_lidar)

    @classmethod
    def cameras(cls) -> tuple["_Item", ...]:
        return tuple(item for item in cls if item.is_camera)

    @classmethod
    def ring_cameras(cls) -> tuple["_Item", ...]:
        return tuple(item for item in cls.cameras() if item.name.startswith("RING_"))

    @classmethod
    def stereo_cameras(cls) -> tuple["_Item", ...]:
        return tuple(item for item in cls.cameras() if item.name.startswith("STEREO_"))

    @classmethod
    def sensors(cls) -> tuple["_Item", ...]:
        """Lidar plus every camera -- the items that account for the bytes."""
        return tuple(item for item in cls if item.is_camera or item.is_lidar)

    @classmethod
    def all_items(cls) -> tuple["_Item", ...]:
        return tuple(cls)


class SensorItem(_Item):
    """Contents of a `sensor` dataset log. ~1.1 GB / 3036 objects per log."""

    MAP = ("map", "map", True)
    CALIBRATION = ("calibration", "calibration", True)
    POSES = ("poses", "city_SE3_egovehicle.feather", False)
    ANNOTATIONS = ("annotations", "annotations.feather", False)
    LIDAR = ("lidar", _LIDAR_RELPATH, True)
    RING_FRONT_CENTER = ("ring_front_center", f"{_CAMERA_RELPATH_PREFIX}ring_front_center", True)
    RING_FRONT_LEFT = ("ring_front_left", f"{_CAMERA_RELPATH_PREFIX}ring_front_left", True)
    RING_FRONT_RIGHT = ("ring_front_right", f"{_CAMERA_RELPATH_PREFIX}ring_front_right", True)
    RING_SIDE_LEFT = ("ring_side_left", f"{_CAMERA_RELPATH_PREFIX}ring_side_left", True)
    RING_SIDE_RIGHT = ("ring_side_right", f"{_CAMERA_RELPATH_PREFIX}ring_side_right", True)
    RING_REAR_LEFT = ("ring_rear_left", f"{_CAMERA_RELPATH_PREFIX}ring_rear_left", True)
    RING_REAR_RIGHT = ("ring_rear_right", f"{_CAMERA_RELPATH_PREFIX}ring_rear_right", True)
    STEREO_FRONT_LEFT = ("stereo_front_left", f"{_CAMERA_RELPATH_PREFIX}stereo_front_left", True)
    STEREO_FRONT_RIGHT = ("stereo_front_right", f"{_CAMERA_RELPATH_PREFIX}stereo_front_right",
                          True)


class TbvItem(_Item):
    """Contents of a `tbv` (Trust But Verify, map change) log.

    Same shape as a sensor log minus annotations and the two stereo cameras.
    """

    MAP = ("map", "map", True)
    CALIBRATION = ("calibration", "calibration", True)
    POSES = ("poses", "city_SE3_egovehicle.feather", False)
    LIDAR = ("lidar", _LIDAR_RELPATH, True)
    RING_FRONT_CENTER = ("ring_front_center", f"{_CAMERA_RELPATH_PREFIX}ring_front_center", True)
    RING_FRONT_LEFT = ("ring_front_left", f"{_CAMERA_RELPATH_PREFIX}ring_front_left", True)
    RING_FRONT_RIGHT = ("ring_front_right", f"{_CAMERA_RELPATH_PREFIX}ring_front_right", True)
    RING_SIDE_LEFT = ("ring_side_left", f"{_CAMERA_RELPATH_PREFIX}ring_side_left", True)
    RING_SIDE_RIGHT = ("ring_side_right", f"{_CAMERA_RELPATH_PREFIX}ring_side_right", True)
    RING_REAR_LEFT = ("ring_rear_left", f"{_CAMERA_RELPATH_PREFIX}ring_rear_left", True)
    RING_REAR_RIGHT = ("ring_rear_right", f"{_CAMERA_RELPATH_PREFIX}ring_rear_right", True)


class LidarItem(_Item):
    """Contents of a `lidar` dataset log: lidar sweeps only, no imagery, no annotations.

    ~292 MB / 302 objects per log.
    """

    MAP = ("map", "map", True)
    CALIBRATION = ("calibration", "calibration", True)
    POSES = ("poses", "city_SE3_egovehicle.feather", False)
    LIDAR = ("lidar", _LIDAR_RELPATH, True)


class MotionForecastingItem(_Item):
    """Contents of a `motion-forecasting` scenario: two files, ~220 KB total.

    Both filenames embed the scenario id, hence the ``{log_id}`` placeholders.
    """

    SCENARIO = ("scenario", "scenario_{log_id}.parquet", False)
    MAP = ("map", "log_map_archive_{log_id}.json", False)


Item: TypeAlias = SensorItem | TbvItem | LidarItem | MotionForecastingItem


class SensorSplit(enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class LidarSplit(enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MotionForecastingSplit(enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# NOTE: there is deliberately no TbvSplit. TBV is a flat listing of logs with no split dirs, and
# TbvRequest correspondingly has no split field.
Split: TypeAlias = SensorSplit | LidarSplit | MotionForecastingSplit

ITEM_TYPES: dict[Dataset, type[_Item]] = {
    Dataset.SENSOR: SensorItem,
    Dataset.TBV: TbvItem,
    Dataset.LIDAR: LidarItem,
    Dataset.MOTION_FORECASTING: MotionForecastingItem,
}

SPLIT_TYPES: dict[Dataset, type[enum.Enum] | None] = {
    Dataset.SENSOR: SensorSplit,
    Dataset.TBV: None,
    Dataset.LIDAR: LidarSplit,
    Dataset.MOTION_FORECASTING: MotionForecastingSplit,
}


class _Request(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Shared behaviour for the per-dataset requests.

    Deliberately declares no fields: msgspec orders inherited fields first, and a defaulted
    field in the base would prevent subclasses from declaring required ones.
    """

    @property
    def dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    def item_type(self) -> type[_Item]:
        return ITEM_TYPES[self.dataset]

    @property
    def split_name(self) -> str | None:
        """The split's directory name, or None for datasets without splits."""
        split = getattr(self, "split", None)
        return None if split is None else split.value

    def available_items(self) -> tuple[_Item, ...]:
        """Every item that exists for this dataset *and split*.

        Distinct from ``tuple(self.item_type)`` because a split can lack an item the dataset
        otherwise has -- sensor/test ships no annotations. Use this when describing a log rather
        than requesting one.
        """
        return tuple(self.item_type)

    def default_items(self) -> tuple[_Item, ...]:
        """What ``items=None`` selects: this split's metadata, which is ~0.2% of a log."""
        return tuple(item for item in self.available_items() if not item.is_camera
                     and not item.is_lidar)

    def _resolve_items(self) -> None:
        """Fill in the default selection, or validate an explicit one.

        Splitting these two cases matters: requesting annotations for sensor/test is an error,
        but *defaulting* into them must not be, or every sensor/test command would fail until
        the user passed --items by hand.
        """
        if self.items is None:
            msgspec.structs.force_setattr(self, "items", self.default_items())
            return

        if not self.items:
            raise UnknownItemError(f"no items requested for {self.slug()}")
        expected = self.item_type
        for item in self.items:
            # The type system already prevents this for anyone using a checker; this catches
            # the dynamic case (CLI strings, deserialized configs).
            if not isinstance(item, expected):
                raise UnknownItemError(
                    f"{item!r} is not a valid item for the {self.dataset.value} dataset. "
                    f"valid items: {', '.join(expected.tokens())}"
                )
        unavailable = set(self.items) - set(self.available_items())
        if unavailable:
            names = ", ".join(sorted(item.token for item in unavailable))
            raise UnknownItemError(
                f"{self.spec()} has no {names}; valid items for this split: "
                f"{', '.join(item.token for item in self.available_items())}"
            )

    def s3_prefix(self) -> str:
        """URI of the directory holding this request's logs, with a trailing slash."""
        parts = [S3_ROOT, self.dataset.value]
        if self.split_name is not None:
            parts.append(self.split_name)
        return "/".join(parts) + "/"

    def local_dir(self, root: Path = DEFAULT_ROOT) -> Path:
        """Local directory holding this request's logs. Mirrors :meth:`s3_prefix`."""
        path = Path(root) / self.dataset.value
        if self.split_name is not None:
            path = path / self.split_name
        return path

    def slug(self) -> str:
        """Stable identifier for this dataset+split, used for cache filenames and messages.

        e.g. 'sensor_val', or just 'tbv' for the split-less dataset.
        """
        if self.split_name is None:
            return self.dataset.value.replace("-", "_")
        return f"{self.dataset.value.replace('-', '_')}_{self.split_name}"

    def spec(self) -> str:
        """The CLI spelling of this dataset+split, e.g. 'sensor/val' or 'tbv'."""
        if self.split_name is None:
            return self.dataset.value
        return f"{self.dataset.value}/{self.split_name}"

    def with_log_ids(self, log_ids: Sequence[str] | None) -> "Request":
        """Return a copy narrowed to `log_ids`. Requests are frozen, hence the copy."""
        return msgspec.structs.replace(
            self, log_ids=None if log_ids is None else tuple(log_ids)
        )


class SensorRequest(_Request):
    """A request against the `sensor` dataset."""

    split: SensorSplit
    items: tuple[SensorItem, ...] | None = None
    """None selects :meth:`default_items`; always a concrete tuple after construction."""
    log_ids: tuple[str, ...] | None = None
    """None means every log in the split."""

    def __post_init__(self) -> None:
        self._resolve_items()

    def available_items(self) -> tuple[SensorItem, ...]:
        """All sensor items, minus annotations on the test split, which ships none.

        This is the one dataset constraint the type system cannot carry, since it depends on the
        split rather than the dataset.
        """
        if self.split is SensorSplit.TEST:
            return tuple(item for item in SensorItem if item is not SensorItem.ANNOTATIONS)
        return tuple(SensorItem)

    @property
    def dataset(self) -> Dataset:
        return Dataset.SENSOR


class TbvRequest(_Request):
    """A request against the `tbv` dataset.

    NOTE: no ``split`` field -- TBV is a flat listing of 1043 logs named
    ``<log_id>__<Season>_<Year>``.
    """

    items: tuple[TbvItem, ...] | None = None
    log_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        self._resolve_items()

    @property
    def dataset(self) -> Dataset:
        return Dataset.TBV


class LidarRequest(_Request):
    """A request against the `lidar` dataset (20 000 logs, no imagery)."""

    split: LidarSplit
    items: tuple[LidarItem, ...] | None = None
    log_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        self._resolve_items()

    @property
    def dataset(self) -> Dataset:
        return Dataset.LIDAR


class MotionForecastingRequest(_Request):
    """A request against the `motion-forecasting` dataset (~250 000 scenarios).

    Both of its items are metadata (there are no sensor streams), so the default selection is
    everything a scenario has.
    """

    split: MotionForecastingSplit
    items: tuple[MotionForecastingItem, ...] | None = None
    log_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        self._resolve_items()

    @property
    def dataset(self) -> Dataset:
        return Dataset.MOTION_FORECASTING


Request: TypeAlias = SensorRequest | TbvRequest | LidarRequest | MotionForecastingRequest

_REQUEST_TYPES: dict[Dataset, type[_Request]] = {
    Dataset.SENSOR: SensorRequest,
    Dataset.TBV: TbvRequest,
    Dataset.LIDAR: LidarRequest,
    Dataset.MOTION_FORECASTING: MotionForecastingRequest,
}


def parse_spec(spec: str) -> tuple[Dataset, enum.Enum | None]:
    """Parse a CLI dataset spec such as 'sensor/val', 'tbv', or 'motion-forecasting/train'.

    Returns the dataset and its split (None for TBV). Raises UnknownSplitError with the valid
    forms on anything unrecognized.
    """
    text = spec.strip().strip("/")
    dataset_name, _, split_name = text.partition("/")
    try:
        dataset = Dataset(dataset_name)
    except ValueError:
        valid = ", ".join(d.value for d in Dataset)
        raise UnknownSplitError(
            f"{dataset_name!r} is not an AV2 dataset. valid datasets: {valid}"
        ) from None

    split_type = SPLIT_TYPES[dataset]
    if split_type is None:
        if split_name:
            raise UnknownSplitError(
                f"the {dataset.value} dataset has no splits; use {dataset.value!r} alone"
            )
        return dataset, None

    if not split_name:
        valid = ", ".join(s.value for s in split_type)
        raise UnknownSplitError(
            f"the {dataset.value} dataset needs a split: {dataset.value}/<{valid.replace(', ', '|')}>"
        )
    try:
        return dataset, split_type(split_name)
    except ValueError:
        valid = ", ".join(s.value for s in split_type)
        raise UnknownSplitError(
            f"{split_name!r} is not a split of {dataset.value}. valid splits: {valid}"
        ) from None


def make_request(
    spec: str,
    *,
    items: Sequence[_Item] | None = None,
    log_ids: Sequence[str] | None = None,
) -> Request:
    """Build the request type appropriate to `spec`.

    This is how the CLI crosses from strings into the typed world; library callers should
    construct the concrete request (SensorRequest, TbvRequest, ...) directly instead.
    """
    dataset, split = parse_spec(spec)
    request_type = _REQUEST_TYPES[dataset]
    kwargs: dict = {}
    if split is not None:
        kwargs["split"] = split
    if items is not None:
        kwargs["items"] = tuple(items)
    if log_ids is not None:
        kwargs["log_ids"] = tuple(log_ids)
    return request_type(**kwargs)


# Group aliases accepted wherever items are named on the command line. Each resolves against
# the dataset's own item enum, so 'cameras' means 9 items for sensor and 0 for lidar.
_GROUP_ALIASES = {
    "metadata": "metadata",
    "meta": "metadata",
    "cameras": "cameras",
    "ring": "ring_cameras",
    "stereo": "stereo_cameras",
    "sensors": "sensors",
    "all": "all_items",
}


def resolve_items(
    item_type: type[_Item],
    tokens: Iterable[str],
    available: Sequence[_Item] | None = None,
) -> tuple[_Item, ...]:
    """Expand a mix of item names and group aliases into a deduplicated, ordered tuple.

    >>> resolve_items(SensorItem, ["metadata", "lidar"])       # doctest: +SKIP
    (MAP, CALIBRATION, POSES, ANNOTATIONS, LIDAR)

    Group aliases resolve against `item_type`, so 'ring' yields 7 cameras for both sensor and
    tbv, and nothing at all for lidar. An alias that expands to nothing is an error rather than
    a silent empty selection.

    `available` narrows what a *group* may expand to -- pass a request's
    :meth:`_Request.available_items` so that `--items all` on sensor/test quietly omits the
    annotations that split does not ship. Items named individually are never filtered, so
    `--items annotations` on sensor/test still raises. Without this split, every group alias
    would be a hard error on sensor/test and the user would have to enumerate items by hand.
    """
    allowed = None if available is None else set(available)
    selected: list[_Item] = []
    for raw in tokens:
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            alias = _GROUP_ALIASES.get(token.lower())
            if alias is not None:
                expanded = getattr(item_type, alias)()
                if allowed is not None:
                    expanded = tuple(item for item in expanded if item in allowed)
                if not expanded:
                    raise UnknownItemError(
                        f"group {token!r} is empty for the {item_type.dataset_label()} "
                        f"dataset. valid items: {', '.join(item_type.tokens())}"
                    )
                selected.extend(expanded)
            else:
                selected.append(item_type.from_token(token))

    # Deduplicate while preserving the declaration order of the enum, so plans are stable.
    unique = set(selected)
    return tuple(item for item in item_type if item in unique)


def s3_uri(request: Request, log_id: str, item: _Item) -> str:
    """S3 source URI for one item of one log.

    Directory-valued items get a trailing '/*' so a single `s5cmd cp` fetches the whole stream;
    single-file items get their exact key.
    """
    base = f"{request.s3_prefix()}{log_id}/{item.resolve_relpath(log_id)}"
    return f"{base}/*" if item.is_dir else base


def local_path(request: Request, log_id: str, item: _Item, root: Path = DEFAULT_ROOT) -> Path:
    """Local destination for one item of one log.

    For directory items this is the directory itself (the s5cmd `cp` destination); for
    single-file items it is the file path.
    """
    return request.local_dir(root) / log_id / item.resolve_relpath(log_id)


def log_dir(request: Request, log_id: str, root: Path = DEFAULT_ROOT) -> Path:
    """Local directory for one log."""
    return request.local_dir(root) / log_id


def classify_key(item_type: type[_Item], rel_key: str, log_id: str) -> _Item | None:
    """Map a log-relative S3 key to the item it belongs to, or None if unrecognized.

    Longest relpath first, so a nested item would win over a shorter prefix.
    """
    for item in sorted(item_type, key=lambda i: len(i.relpath), reverse=True):
        relpath = item.resolve_relpath(log_id)
        if item.is_dir:
            if rel_key.startswith(f"{relpath}/"):
                return item
        elif rel_key == relpath:
            return item
    return None


# Matches the city code that AV2 embeds in map filenames. Covers both forms observed in the
# bucket: '____PIT_city_71109.json' / '____PIT.npy' (sensor, tbv) and
# '__Summer____ATX_city_77093.json' (lidar).
_CITY_RE = re.compile(r"____(?P<city>[A-Z]{2,4})(?:_city_\d+)?\.(?:json|npy)$")


def city_from_map_key(rel_key: str) -> str | None:
    """Extract the city code from a map filename, or None if absent.

    The city is *only* recoverable from map filenames -- no metadata file records it -- which is
    why the catalog has to list each log's map/ contents. Motion-forecasting map filenames carry
    no city code, so this returns None for them.
    """
    match = _CITY_RE.search(rel_key)
    return match.group("city") if match else None
