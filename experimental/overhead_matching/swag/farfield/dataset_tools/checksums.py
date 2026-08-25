"""The one implementation of a dataset's `checksums.sha256` regeneration.

Datasets are frozen (REORG.md rule 7). ``trim_dataset`` is the one explicit,
checksum-regenerating mutator in this package and calls ``regenerate`` here so
the manifest format and exclusion list have one owner. Calibration diagnostics
never mutate dataset metadata; approved nominal-forward records are immutable
build inputs.

Format matches `sha256sum` output with `./`-relative paths sorted as bytes
(C locale), covering everything except:

- the manifest itself;
- the `panorama/` symlink tree (it aliases `frames/`, which is covered);
- derived per-dataset products (`_manifests/`, `catalog_cache/`,
  `__pycache__/`): these are rebuildable and rewritten whenever a triage tool
  runs, so checksumming them would report every tool run as corruption.
"""

import hashlib
from pathlib import Path

CHECKSUM_FILE = "checksums.sha256"
# Rebuildable, tool-rewritten directories. `_manifests/` holds the triage
# sidecars (recording_seams.json, vehicle_anchor.json, regenerated views);
# it is derived data living beside the frozen definition, not part of it.
EXCLUDED_DIRS = frozenset({"catalog_cache", "__pycache__", "_manifests"})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regenerate(dataset_base: Path) -> int | None:
    """Rewrite `checksums.sha256` over every real file in the dataset.

    Returns the number of manifest lines, or None when the dataset carries no
    manifest (nothing is invented: a dataset that never had integrity checking
    does not gain it as a side effect of an unrelated tool).
    """
    dataset_base = Path(dataset_base)
    target = dataset_base / CHECKSUM_FILE
    if not target.exists():
        return None
    entries = []
    for path in dataset_base.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(dataset_base)
        if rel.parts[0] == "panorama" or rel.name == CHECKSUM_FILE:
            continue
        if set(rel.parts) & EXCLUDED_DIRS:
            continue
        entries.append(("./" + rel.as_posix(), path))
    lines = [f"{file_sha256(path)}  {rel}\n"
             for rel, path in sorted(entries, key=lambda e: e[0].encode())]
    target.write_text("".join(lines))
    return len(lines)
