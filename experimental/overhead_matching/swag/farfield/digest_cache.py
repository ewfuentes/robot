"""Remember a file's digest so a frozen input is not rehashed every stage.

`paths.dataset_source_digests` hashes every panorama in a dataset, and it is
called once per stage. On `pohang_canal_04` that is 1,450 JPEGs and 3.3 GB,
so an eight-stage run reads and hashes ~26 GB purely to re-confirm that a
directory documented as frozen has not changed. Cold, that dominates the
wall-clock of a run whose stages are otherwise cache hits.

So key each digest on the file's identity as the filesystem reports it --
`(device, inode, size, mtime_ns)` -- and rehash only when that changes. Git's
index and Bazel's file cache make exactly this trade.

THE TRADE, stated plainly because it is a real weakening: a file whose bytes
change while its inode, size and mtime all stay identical returns a stale
digest. That takes deliberate effort (a same-length rewrite plus `utimes`), it
is the same exposure git accepts for every `git status`, and the guarantee is
not gone -- `audit_dataset` and `checksums.verify` still read every byte. What
this removes is re-proving a frozen input on every stage of every run; what it
keeps is proving it when asked.

Nanosecond mtime matters: a same-second rewrite is entirely plausible during
development, and a one-second-resolution key would miss it. Where a filesystem
does not provide `st_mtime_ns` at full resolution the key degrades to whatever
it does provide, which is why size and inode are in the key too.

The cache is disposable. Deleting it costs one slow run and nothing else, so
it lives outside every lane that means something: never in `datasets/` (frozen
by contract), never in `artifacts/` (immutable and content-addressed).
"""

from __future__ import annotations

import atexit
import fcntl
import json
from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact

SCHEMA = "farfield_digest_cache/v1"
CACHE_DIR_NAME = ".cache"
CACHE_FILE_NAME = "file_digests.json"

# Bound the file so a long-lived root cannot grow one unboundedly. Entries are
# cheap (~200 bytes) and the working set is a few datasets' panoramas, so this
# is generous; eviction is "start again", which costs one slow run.
MAX_ENTRIES = 200_000

_MEMO: dict[tuple, str] = {}


class DigestCacheError(ValueError):
    """The cache file exists but cannot be used."""


def _key(path: Path) -> tuple[str, int, int, int, int]:
    """Filesystem identity of the resolved file behind `path`."""
    resolved = path.resolve()
    stat = resolved.stat()
    return (str(resolved), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns)


def cache_path(root: Path) -> Path:
    return Path(root) / CACHE_DIR_NAME / CACHE_FILE_NAME


def _load(path: Path) -> dict[str, str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as error:
        raise DigestCacheError(f"cannot read digest cache {path}: {error}") \
            from error
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        # A truncated cache is not an error worth failing a run over: it is
        # a cache. Start again.
        return {}
    if (not isinstance(document, dict)
            or document.get("schema") != SCHEMA
            or not isinstance(document.get("entries"), dict)):
        return {}
    return document["entries"]


def _flush(path: Path, pending: dict[str, str]) -> None:
    """Merge `pending` into the stored cache under an exclusive lock.

    Merge rather than overwrite: several stages run concurrently against one
    root, and a last-writer-wins replace would silently discard the digests
    another process just paid to compute.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    # Lock a separate file, so the atomic replace below never swaps the cache
    # out from under a descriptor another process is holding the lock on.
    with open(lock, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            merged = _load(path)
            merged.update(pending)
            if len(merged) > MAX_ENTRIES:
                merged = dict(list(merged.items())[-MAX_ENTRIES:])
            artifact.atomic_write_file(path, json.dumps(
                {"schema": SCHEMA, "entries": merged},
                separators=(",", ":"), sort_keys=True).encode("utf-8"))
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# Loaded once per root per process. Reading and parsing the cache file per
# *file* would have made a 1,450-panorama dataset re-parse it 1,450 times,
# which is how a cache becomes slower than the work it replaces.
_LOADED: dict[str, dict[str, str]] = {}
_PENDING: dict[str, dict[str, str]] = {}


def _entries(store: Path) -> dict[str, str]:
    key = str(store)
    if key not in _LOADED:
        _LOADED[key] = _load(store)
    return _LOADED[key]


def sha256_file(path: Path | str, *, root: Path | str | None = None) -> str:
    """`artifact.sha256_file`, skipping the read when nothing changed.

    With no `root` this memoizes within the process only, which is still the
    right answer for a single tool that hashes one file twice. Cross-process
    reuse -- the case that matters, since every stage is its own `bazel run` --
    needs the on-disk cache and therefore a root.

    New digests accumulate and are written by `flush()`; a caller that hashes
    many files pays one cache write rather than one per file.
    """
    target = Path(path)
    key = _key(target)
    memo = _MEMO.get(key)
    if memo is not None:
        return memo
    if root is None:
        digest = artifact.sha256_file(target)
        _MEMO[key] = digest
        return digest

    store = cache_path(Path(root))
    text_key = json.dumps(key, separators=(",", ":"))
    cached = _entries(store).get(text_key)
    if isinstance(cached, str) and len(cached) == 64:
        _MEMO[key] = cached
        return cached
    digest = artifact.sha256_file(target)
    _MEMO[key] = digest
    _PENDING.setdefault(str(store), {})[text_key] = digest
    _entries(store)[text_key] = digest
    return digest


def flush() -> None:
    """Write accumulated digests. Never fails a run that only wanted speed."""
    for store, pending in list(_PENDING.items()):
        if not pending:
            continue
        try:
            _flush(Path(store), pending)
        except OSError:
            # A read-only or full data root costs a slow next run, nothing
            # more. Losing a cache write is never a correctness event.
            pass
        _PENDING[store] = {}


atexit.register(flush)


def clear_process_memo() -> None:
    """Drop every in-process cache. For tests that mutate files in place."""
    _MEMO.clear()
    _LOADED.clear()
    _PENDING.clear()
