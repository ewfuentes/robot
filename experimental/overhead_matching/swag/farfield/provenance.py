"""Lightweight provenance records for source and review side outputs.

Canonical pipeline artifacts use the typed, content-addressed manifest in
``farfield.artifact``. This module records the inputs and resolved recipe for
outputs that are intentionally outside those artifact lanes. It is not a
completion marker; callers remain responsible for transactional publication.

The manifest records the five things a reader needs to reproduce or audit the
output: the git commit of the producing code, the exact argv, the resolved
input paths, the config that shaped the result, and when it happened. Values
are recorded, never defaulted: if a producer does not know one of these, that
is the producer's bug to fix, not this module's to paper over.

`content_digest` (optional, via `digest_dir`) lets an artifact-version writer
detect the "new version, identical bytes" failure mode: a vN directory whose
content digest equals a sibling version's is versioning noise, not a new
artifact, and `check_version_is_new` refuses it.
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "manifest.json"
SCHEMA = "farfield_provenance/v1"


def git_commit() -> str:
    """HEAD of the source workspace. `unknown` if unavailable.

    `bazel run` sets BUILD_WORKSPACE_DIRECTORY to the source workspace;
    without it the runfiles tree is not a git checkout.
    """
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    try:
        return subprocess.check_output(
            ["git", "-C", workspace or ".", "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, TypeError):
        return "unknown"


def write(target_dir: Path, *, generator: str, inputs: dict, config: dict,
          notes: str = "", content_digest: str | None = None,
          extra: dict | None = None) -> Path:
    """Write `manifest.json` into `target_dir` and return its path.

    generator: the producing entry point (module or bazel target).
    inputs:    {name: path-or-value} -- the *resolved* inputs, absolute or
               data-root-relative paths, never flag spellings.
    config:    every value that shaped the result (thresholds, models,
               versions). What is recorded here is what readers must use;
               a reader constructing a fresh default instead is a bug.
    extra:     producer-specific fields (e.g. kind/dataset/version for
               artifact lanes); merged at the top level, cannot shadow the
               standard fields.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": SCHEMA,
        "generator": generator,
        "git_commit": git_commit(),
        "argv": list(sys.argv),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {k: str(v) for k, v in inputs.items()},
        "config": config,
        "notes": notes,
    }
    if content_digest:
        manifest["content_digest"] = content_digest
    for key, value in (extra or {}).items():
        if key in manifest:
            raise ValueError(f"extra field {key!r} shadows a standard field")
        manifest[key] = value
    path = target_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    return path


def read(target_dir: Path) -> dict:
    """Parsed manifest of an artifact directory.

    Raises FileNotFoundError with a pointed message when absent: consumers
    are entitled to a manifest, and a missing one is the producer's bug.
    """
    path = Path(target_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Every artifact directory carries a "
            f"manifest (see farfield/provenance.py); whatever produced "
            f"{Path(target_dir)} skipped it.")
    return json.loads(path.read_text())


def digest_dir(target_dir: Path, exclude: tuple = (MANIFEST_NAME,)) -> str:
    """Content digest of a directory: sha256 over (relpath, file sha256) pairs.

    Excludes the manifest itself so the digest is stable across manifest
    rewrites. Use with `check_version_is_new` when writing versioned
    artifacts.
    """
    target_dir = Path(target_dir)
    entries = []
    for path in sorted(p for p in target_dir.rglob("*") if p.is_file()):
        rel = str(path.relative_to(target_dir))
        if rel in exclude:
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        entries.append(f"{rel}:{h.hexdigest()}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def check_version_is_new(version_dir: Path, content_digest: str) -> None:
    """Refuse a new artifact version whose bytes equal an existing sibling's.

    `version_dir` is `<kind>/<dataset>/<version>`; siblings are the other
    version dirs of the same dataset. A byte-identical re-release under a new
    version name destroys the meaning of versions, so it is an error rather
    than a warning.
    """
    version_dir = Path(version_dir)
    for sibling in version_dir.parent.iterdir() if version_dir.parent.exists() else ():
        if sibling == version_dir or not sibling.is_dir():
            continue
        manifest_path = sibling / MANIFEST_NAME
        if not manifest_path.exists():
            continue
        try:
            recorded = json.loads(manifest_path.read_text()).get(
                "content_digest")
        except (json.JSONDecodeError, OSError):
            continue
        if recorded and recorded == content_digest:
            raise ValueError(
                f"refusing to write {version_dir.name}: content is "
                f"byte-identical to existing version {sibling.name} "
                f"(digest {content_digest[:12]}...). A new version must "
                f"contain new content.")
