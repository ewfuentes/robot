"""A cached index of what Argoverse 2 logs exist remotely and how big their pieces are.

Nothing about a log's contents is discoverable without listing S3 -- in particular the **city**
is only recoverable from map filenames, since no metadata file records it. Listing on every
`list`/`show`/`download` would be intolerably slow, so we list once per dataset+split and cache
the aggregate.

The catalog stores *per-log, per-item* byte and object counts. That is enough to

* answer `list` and `show` offline,
* compute download sizes before transferring anything, and
* decide whether a local copy is complete (compare local object counts against these).

A file-level index is deliberately *not* kept: `s5cmd cp 'prefix/*'` expands server-side, so
knowing individual keys would buy nothing and would cost millions of rows.

The cache lives under ~/.cache/robot and is disposable -- AV2 is a frozen release, so a stale
catalog is a non-event, and a schema change just triggers a rebuild.
"""

import concurrent.futures
import datetime
import difflib
import enum
import fnmatch
import logging
from pathlib import Path
from typing import Callable, Iterable, Sequence

import msgspec

from common.python.serialization import (
    MSGSPEC_STRUCT_OPTS,
    msgspec_dec_hook,
    msgspec_enc_hook,
)
from experimental.map_estimation.data import argoverse_layout as al
from experimental.map_estimation.data import s5cmd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("~/.cache/robot/map_estimation/argoverse").expanduser()

# Bumped whenever LogEntry/Catalog change shape. A mismatch rebuilds rather than errors.
SCHEMA_VERSION = 1

# How many logs to list concurrently. Each listing is one s5cmd subprocess taking ~0.4 s, so
# this is the difference between 12 s and 400 s for a 1000-log dataset.
DEFAULT_LIST_CONCURRENCY = 32

# Motion-forecasting has ~250 000 scenarios, all with exactly the same two files. Measuring
# every one would take hours, so we measure this many and extrapolate.
PREFIX_ONLY_SAMPLE_SIZE = 24


class CatalogError(RuntimeError):
    """The catalog could not be built or loaded."""


class BuildStrategy(enum.Enum):
    """How a catalog's numbers were obtained."""

    PER_LOG_DETAIL = "per_log_detail"
    """Every log was listed. Sizes are measured."""
    PREFIX_ONLY = "prefix_only"
    """Only log ids were listed; sizes come from a sample. Used for motion-forecasting."""


# Datasets small enough to list exhaustively. Motion-forecasting (250k scenarios) is not.
_STRATEGIES: dict[al.Dataset, BuildStrategy] = {
    al.Dataset.SENSOR: BuildStrategy.PER_LOG_DETAIL,
    al.Dataset.TBV: BuildStrategy.PER_LOG_DETAIL,
    al.Dataset.LIDAR: BuildStrategy.PER_LOG_DETAIL,
    al.Dataset.MOTION_FORECASTING: BuildStrategy.PREFIX_ONLY,
}


class ItemStat(msgspec.Struct, frozen=True):
    """Size of one item of one log, as it exists remotely."""

    num_bytes: int
    num_objects: int

    def __add__(self, other: "ItemStat") -> "ItemStat":
        return ItemStat(
            num_bytes=self.num_bytes + other.num_bytes,
            num_objects=self.num_objects + other.num_objects,
        )


EMPTY_STAT = ItemStat(num_bytes=0, num_objects=0)


class LogEntry(msgspec.Struct, frozen=True):
    """One log's remote contents.

    `items` is keyed by item *token* rather than by enum member: the catalog is stored as JSON,
    and a union-typed enum key could not be decoded unambiguously. Use :meth:`stat` to look up
    by member. A missing key means the item does not exist remotely for this log.
    """

    log_id: str
    city: str | None
    items: dict[str, ItemStat]

    def stat(self, item: al._Item) -> ItemStat:
        """Remote size of `item`, or a zero stat when the log does not have it."""
        return self.items.get(item.token, EMPTY_STAT)

    def has(self, item: al._Item) -> bool:
        return item.token in self.items

    def total(self, items: Iterable[al._Item] | None = None) -> ItemStat:
        """Summed size of `items` (default: everything this log has)."""
        if items is None:
            stats = self.items.values()
        else:
            stats = [self.stat(item) for item in items]
        total = EMPTY_STAT
        for stat in stats:
            total = total + stat
        return total

    @property
    def num_lidar_sweeps(self) -> int:
        """Sweep count, i.e. the number of objects in the lidar item. 0 if no lidar."""
        return self.items.get("lidar", EMPTY_STAT).num_objects


class Catalog(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Everything known about one dataset+split, as of `built_at`."""

    schema_version: int
    dataset: al.Dataset
    split: str | None
    built_at: datetime.datetime
    s5cmd_version: str
    strategy: BuildStrategy
    logs: tuple[LogEntry, ...]
    sampled_logs: int | None = None
    """For PREFIX_ONLY catalogs, how many logs were actually measured."""

    def __len__(self) -> int:
        return len(self.logs)

    def __iter__(self):
        return iter(self.logs)

    @property
    def item_type(self) -> type[al._Item]:
        return al.ITEM_TYPES[self.dataset]

    @property
    def sizes_are_inferred(self) -> bool:
        """True when byte counts came from a sample rather than from listing every log."""
        return self.strategy is BuildStrategy.PREFIX_ONLY

    @property
    def spec(self) -> str:
        return self.dataset.value if self.split is None else f"{self.dataset.value}/{self.split}"

    def get(self, log_id: str) -> LogEntry:
        """Look up one log, with a near-miss hint when the id is not present.

        difflib rather than a prefix compare, so a typo anywhere in a 36-character uuid still
        produces a useful suggestion.
        """
        for entry in self.logs:
            if entry.log_id == log_id:
                return entry
        close = difflib.get_close_matches(
            log_id, [entry.log_id for entry in self.logs], n=3, cutoff=0.6
        )
        hint = f" did you mean: {', '.join(close)}?" if close else ""
        raise KeyError(f"{log_id!r} is not a log of {self.spec}.{hint}")

    def cities(self) -> dict[str, int]:
        """City code -> log count, for logs whose city is known."""
        counts: dict[str, int] = {}
        for entry in self.logs:
            if entry.city is not None:
                counts[entry.city] = counts.get(entry.city, 0) + 1
        return dict(sorted(counts.items()))

    def total(self) -> ItemStat:
        total = EMPTY_STAT
        for entry in self.logs:
            total = total + entry.total()
        return total

    @property
    def age(self) -> datetime.timedelta:
        return datetime.datetime.now(datetime.timezone.utc) - self.built_at


def cache_path(request_or_slug: al.Request | str, cache_dir: Path = CACHE_DIR) -> Path:
    """Where a dataset+split's catalog is cached."""
    slug = request_or_slug if isinstance(request_or_slug, str) else request_or_slug.slug()
    return Path(cache_dir) / f"{slug}.json"


def _log_relative_key(key: str, s3_prefix: str) -> tuple[str, str] | None:
    """Split a full S3 URI into (log_id, path relative to the log dir).

    Returns None for keys that sit directly under the split prefix (there are none in practice,
    but a stray file must not crash a build).
    """
    if not key.startswith(s3_prefix):
        return None
    remainder = key[len(s3_prefix):]
    log_id, sep, rel_key = remainder.partition("/")
    if not sep or not rel_key:
        return None
    return log_id, rel_key


def _entry_from_objects(
    log_id: str, objects: Sequence[s5cmd.Object], s3_prefix: str, item_type: type[al._Item]
) -> LogEntry:
    """Aggregate one log's object listing into a LogEntry."""
    stats: dict[str, ItemStat] = {}
    city: str | None = None
    for obj in objects:
        split_key = _log_relative_key(obj.key, s3_prefix)
        if split_key is None:
            continue
        _, rel_key = split_key
        if city is None:
            city = al.city_from_map_key(rel_key)
        item = al.classify_key(item_type, rel_key, log_id)
        if item is None:
            logger.debug("unrecognized key for %s: %s", log_id, rel_key)
            continue
        stats[item.token] = stats.get(item.token, EMPTY_STAT) + ItemStat(
            num_bytes=obj.size, num_objects=1
        )
    return LogEntry(log_id=log_id, city=city, items=stats)


def _list_log_ids(s3_prefix: str, opts: s5cmd.Options) -> list[str]:
    logger.info("listing log prefixes under %s", s3_prefix)
    log_ids = sorted(s5cmd.list_prefixes(s3_prefix, opts=opts))
    if not log_ids:
        raise CatalogError(f"no logs found under {s3_prefix}")
    return log_ids


def _build_per_log_detail(
    request: al.Request,
    log_ids: Sequence[str],
    opts: s5cmd.Options,
    concurrency: int,
    progress: Callable[[int, int], None] | None,
) -> tuple[list[LogEntry], list[str]]:
    """List every log concurrently and aggregate. Returns (entries, failed log ids).

    One s5cmd subprocess per log. Threads are the right tool here despite the GIL: the work is
    entirely spent waiting on child processes.

    A log either lists completely or not at all, so every returned entry is a full picture of
    that log -- which is what lets a missing item stat be read as "absent remotely" rather than
    "we never looked". Failures are returned rather than swallowed so the caller can decide.
    """
    s3_prefix = request.s3_prefix()
    item_type = request.item_type
    entries: dict[str, LogEntry] = {}
    failed: list[str] = []

    def list_one(log_id: str) -> LogEntry:
        objects = s5cmd.list_objects(f"{s3_prefix}{log_id}/", opts=opts)
        return _entry_from_objects(log_id, objects, s3_prefix, item_type)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(list_one, log_id): log_id for log_id in log_ids}
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            log_id = futures[future]
            try:
                entries[log_id] = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad log must not sink a whole build
                logger.warning("failed to list %s: %s", log_id, exc)
                failed.append(log_id)
            if progress is not None:
                progress(done, len(log_ids))

    # Preserve the caller's (sorted) order rather than completion order, so catalogs are
    # byte-identical across rebuilds.
    return [entries[log_id] for log_id in log_ids if log_id in entries], failed


def _build_prefix_only(
    request: al.Request,
    log_ids: Sequence[str],
    opts: s5cmd.Options,
    concurrency: int,
    progress: Callable[[int, int], None] | None,
) -> tuple[list[LogEntry], int]:
    """Enumerate log ids, then extrapolate sizes from a small measured sample.

    Motion-forecasting scenarios are uniform -- exactly one parquet and one map json -- so a
    sample gives usable plan totals at a fraction of the cost of 250 000 listings. Callers can
    tell these numbers apart via ``Catalog.sizes_are_inferred``.
    """
    sample_ids = list(log_ids[:PREFIX_ONLY_SAMPLE_SIZE])
    measured, _failed = _build_per_log_detail(request, sample_ids, opts, concurrency, progress)
    if not measured:
        raise CatalogError(f"could not measure any sample logs under {request.s3_prefix()}")

    mean_stats: dict[str, ItemStat] = {}
    for item in request.item_type:
        present = [entry.stat(item) for entry in measured if entry.has(item)]
        if not present:
            continue
        mean_stats[item.token] = ItemStat(
            num_bytes=round(sum(s.num_bytes for s in present) / len(present)),
            num_objects=round(sum(s.num_objects for s in present) / len(present)),
        )

    by_id = {entry.log_id: entry for entry in measured}
    entries = [
        by_id.get(log_id, LogEntry(log_id=log_id, city=None, items=mean_stats))
        for log_id in log_ids
    ]
    return entries, len(measured)


def build(
    request: al.Request,
    *,
    opts: s5cmd.Options = s5cmd.Options(),
    concurrency: int = DEFAULT_LIST_CONCURRENCY,
    progress: Callable[[int, int], None] | None = None,
    log_ids: Sequence[str] | None = None,
) -> Catalog:
    """List `request`'s dataset+split from S3 and aggregate it into a Catalog.

    `request.items` is ignored -- a catalog always describes everything the dataset has, so it
    can serve any later request. `log_ids` restricts which logs are listed, which is how a
    partial refresh patches a few rows.
    """
    s3_prefix = request.s3_prefix()
    strategy = _STRATEGIES[request.dataset]
    explicit_ids = log_ids is not None
    ids = list(log_ids) if explicit_ids else _list_log_ids(s3_prefix, opts)
    if not ids:
        raise CatalogError(f"no logs requested for {request.spec()}")
    logger.info("indexing %d logs of %s (%s)", len(ids), request.spec(), strategy.value)

    sampled_logs = None
    failed: list[str] = []
    if strategy is BuildStrategy.PER_LOG_DETAIL:
        entries, failed = _build_per_log_detail(request, ids, opts, concurrency, progress)
    else:
        entries, sampled_logs = _build_prefix_only(request, ids, opts, concurrency, progress)

    if not entries:
        # Never return an empty catalog. Caching one would make every later list/download
        # report an empty split until the user thought to pass --refresh.
        raise CatalogError(
            f"listed none of the {len(ids)} requested log(s) of {request.spec()}. "
            + (f"is {ids[0]!r} a real log id?" if explicit_ids and len(ids) == 1
               else f"{len(failed)} listing(s) failed.")
        )
    if explicit_ids and failed:
        # An explicit id list is a precise request; silently dropping part of it would leave the
        # catalog quietly missing rows the caller asked for.
        raise CatalogError(
            f"could not list {len(failed)} of {len(ids)} requested log(s) of "
            f"{request.spec()}: {', '.join(failed[:5])}"
        )
    if failed:
        logger.warning("indexed %d of %d logs; %d failed to list",
                       len(entries), len(ids), len(failed))

    return Catalog(
        schema_version=SCHEMA_VERSION,
        dataset=request.dataset,
        split=request.split_name,
        built_at=datetime.datetime.now(datetime.timezone.utc),
        s5cmd_version=s5cmd.version(opts),
        strategy=strategy,
        logs=tuple(entries),
        sampled_logs=sampled_logs,
    )


def save(catalog: Catalog, path: Path) -> None:
    """Write a catalog as indented JSON, so cache files are diffable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = msgspec.json.encode(catalog, enc_hook=msgspec_enc_hook)
    path.write_bytes(msgspec.json.format(encoded, indent=2))
    logger.info("wrote %s (%d logs, %s)", path, len(catalog),
                s5cmd.format_bytes(path.stat().st_size))


def load(path: Path) -> Catalog:
    """Read a catalog. Raises CatalogError on a schema mismatch or corrupt file."""
    try:
        catalog = msgspec.json.decode(
            Path(path).read_bytes(), type=Catalog, dec_hook=msgspec_dec_hook
        )
    except msgspec.DecodeError as exc:
        raise CatalogError(f"{path} is not a readable catalog: {exc}") from exc
    if catalog.schema_version != SCHEMA_VERSION:
        raise CatalogError(
            f"{path} has schema version {catalog.schema_version}, expected {SCHEMA_VERSION}"
        )
    return catalog


def load_or_build(
    request: al.Request,
    *,
    cache_dir: Path = CACHE_DIR,
    catalog_path: Path | None = None,
    refresh: bool = False,
    opts: s5cmd.Options = s5cmd.Options(),
    concurrency: int = DEFAULT_LIST_CONCURRENCY,
    progress: Callable[[int, int], None] | None = None,
) -> Catalog:
    """Return the catalog for `request`, building and caching it if needed.

    Resolution order: an explicit `catalog_path`, then the cache, then a fresh build.
    `refresh=True` skips straight to building. A cache file that fails to load (old schema,
    truncated write) is rebuilt rather than raised, since the cache is disposable.
    """
    if catalog_path is not None:
        catalog = load(catalog_path)
        if catalog.spec != request.spec():
            # Otherwise log ids from one dataset would be pasted into another's URIs, and every
            # transfer would 404 with nothing to explain why.
            raise CatalogError(
                f"{catalog_path} describes {catalog.spec}, not {request.spec()}"
            )
        return catalog

    path = cache_path(request, cache_dir)
    if not refresh and path.exists():
        try:
            catalog = load(path)
            logger.info("loaded catalog %s (%d logs, built %s)", path, len(catalog),
                        catalog.built_at.date())
            return catalog
        except CatalogError as exc:
            logger.warning("rebuilding %s: %s", path, exc)

    catalog = build(request, opts=opts, concurrency=concurrency, progress=progress)
    save(catalog, path)
    return catalog


def filter_logs(
    catalog: Catalog,
    *,
    cities: Sequence[str] | None = None,
    log_ids: Sequence[str] | None = None,
    has_items: Sequence[al._Item] | None = None,
    min_sweeps: int | None = None,
    sort_by: str = "log_id",
    limit: int | None = None,
) -> list[LogEntry]:
    """Select logs from `catalog`.

    Sorting happens before `limit`, so `--limit N` is reproducible rather than dependent on
    listing order. `log_ids` entries containing a fnmatch wildcard are treated as patterns;
    literal ids that match nothing raise KeyError rather than silently shrinking the result,
    since a typo'd log id should not look like an empty dataset.
    """
    entries = list(catalog.logs)

    if cities is not None:
        # Accept both repeated flags and comma-separated values: --city PIT --city ATX and
        # --city PIT,ATX mean the same thing.
        wanted = {
            part.strip().upper()
            for value in cities
            for part in str(value).split(",")
            if part.strip()
        }
        if wanted:
            unknown = wanted - set(catalog.cities())
            if unknown:
                known = ", ".join(catalog.cities()) or "none (no cities in this catalog)"
                raise ValueError(
                    f"no logs in {catalog.spec} for city {', '.join(sorted(unknown))}; "
                    f"known cities: {known}"
                )
            entries = [entry for entry in entries if entry.city in wanted]

    # `if log_ids:` rather than `is not None`: an empty selection means "no filter", matching
    # the None case. Treating it as "match nothing" would make with_log_ids([]) silently yield
    # zero logs.
    if log_ids:
        patterns = [str(value).strip() for value in log_ids if str(value).strip()]
        literals = [p for p in patterns if not any(c in p for c in "*?[")]
        globs = [p for p in patterns if p not in literals]
        for literal in literals:
            catalog.get(literal)  # raises KeyError with a hint if absent
        selected = {p for p in literals}
        matched = [
            entry
            for entry in entries
            if entry.log_id in selected
            or any(fnmatch.fnmatch(entry.log_id, pattern) for pattern in globs)
        ]
        if patterns and not matched:
            raise KeyError(f"no logs of {catalog.spec} match {patterns}")
        entries = matched

    if has_items is not None:
        entries = [entry for entry in entries if all(entry.has(i) for i in has_items)]

    if min_sweeps is not None:
        entries = [entry for entry in entries if entry.num_lidar_sweeps >= min_sweeps]

    sorters: dict[str, Callable[[LogEntry], object]] = {
        "log_id": lambda e: e.log_id,
        "city": lambda e: (e.city or "", e.log_id),
        "bytes": lambda e: (-e.total().num_bytes, e.log_id),
        "sweeps": lambda e: (-e.num_lidar_sweeps, e.log_id),
    }
    if sort_by not in sorters:
        raise ValueError(f"unknown sort key {sort_by!r}; valid: {', '.join(sorters)}")
    entries.sort(key=sorters[sort_by])

    if limit is not None:
        entries = entries[:limit]
    return entries


def read_log_id_file(path: Path) -> list[str]:
    """Read log ids from a file, one per line. '#' comments and blank lines are ignored."""
    ids = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    if not ids:
        raise ValueError(f"{path} contains no log ids")
    return ids
