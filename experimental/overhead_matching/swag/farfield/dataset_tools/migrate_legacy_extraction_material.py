"""Bring the pre-contract v4/v5 extraction directories into the artifact era.

Eight directories under `artifacts/frame_landmarks/<dataset>/{v4,v5}` carry a
hand-written manifest from before the artifact contract existed: no schema, no
content digest, no declared outputs, and an `inputs` list of bare paths rather
than typed upstream refs. Nothing can open them -- `artifact.load_manifest`
refuses, which is why the identity survey reports them as unreadable.

They are worth keeping. They hold the raw request and result material that
`legacy_extraction_adoption` read to publish the adopted `frame_landmarks`
artifacts without re-billing a provider, and that material is referenced by
file path from the adoption's own `llm-work` records.

THEY ARE NOT CURRENT `frame_landmarks`, and this does not pretend otherwise.
The current kind promises one canonical `predictions.jsonl`; these hold the
older `sentences/` and `sentence_requests/` trees. Writing a current-schema
manifest that still said `kind: frame_landmarks` would make
`open_artifact(expected_kind=FRAME_LANDMARKS)` succeed on a directory that
cannot satisfy the contract -- a silent wrong answer in place of today's loud
one. So they are re-kinded to `legacy_extraction_material`, which is not in
`ARTIFACT_KINDS` and therefore cannot be addressed as a pipeline input at all.
A consumer that asks for frame_landmarks here now gets an exact kind mismatch
rather than a parse error.

They stay where they are. The adoption records absolute paths into these
directories, so moving them would break provenance that is currently intact.

Rewriting the manifest is safe, and that was checked rather than assumed:
nothing references these artifacts by manifest digest (the adoption cites
files inside the payload), and `sha256_directory` excludes `manifest.json`, so
the content digest does not depend on the manifest being rewritten.

Report-only by default; applying requires the printed plan digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from experimental.overhead_matching.swag.farfield import (
    artifact,
    code_provenance,
    paths as paths_lib,
)

SCHEMA = "farfield.legacy_extraction_material_migration/v1"
LEGACY_KIND = "legacy_extraction_material"
# The layout these directories actually have. A directory missing both is not
# one of the things this tool knows how to describe, and is left alone.
LEGACY_MARKERS = ("sentences", "sentence_requests")


class LegacyMigrationError(ValueError):
    """A legacy directory cannot be described as an artifact."""


def _legacy_manifest(version_dir: Path) -> dict | None:
    """The old hand-written manifest, or None if this is not one."""
    path = version_dir / artifact.MANIFEST_NAME
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict) or document.get("schema") is not None:
        return None
    if not {"kind", "dataset", "version"} <= set(document):
        return None
    return document


def _declared_outputs(version_dir: Path) -> tuple[str, ...]:
    """Top-level entries, as the artifact contract's declared outputs.

    Directory names rather than every leaf: these trees hold thousands of
    request and result files, and a declared-output list per file would be
    larger than the manifest it lives in without saying anything more.
    """
    return tuple(sorted(
        entry.name for entry in version_dir.iterdir()
        if entry.name != artifact.MANIFEST_NAME and not entry.is_symlink()))


def plan_directory(version_dir: Path) -> dict:
    record: dict[str, Any] = {"path": str(version_dir), "status": "pending"}
    legacy = _legacy_manifest(version_dir)
    if legacy is None:
        record["status"] = "not_a_legacy_manifest"
        return record
    present = [name for name in LEGACY_MARKERS
               if (version_dir / name).is_dir()]
    if not present:
        record.update(status="unrecognized_layout",
                      reason=f"none of {list(LEGACY_MARKERS)} present")
        return record
    if any(path.is_symlink() for path in version_dir.rglob("*")):
        # `sha256_directory` refuses symlinks, since their targets are mutable
        # and may point outside the artifact.
        record.update(status="contains_symlinks")
        return record
    record.update(
        status="migrate",
        dataset=legacy.get("dataset"),
        version=legacy.get("version"),
        from_kind=legacy.get("kind"),
        to_kind=LEGACY_KIND,
        generator=legacy.get("generator") or "unknown",
        git_commit=legacy.get("git_commit") or code_provenance.UNKNOWN,
        created=legacy.get("created") or "unknown",
        legacy_manifest=legacy,
        declared_outputs=list(_declared_outputs(version_dir)),
    )
    return record


def build_plan(root: Path) -> dict:
    root = Path(root)
    lane = root / "artifacts" / paths_lib.FRAME_LANDMARKS
    records = []
    if lane.is_dir():
        for dataset_dir in sorted(p for p in lane.iterdir() if p.is_dir()):
            for version_dir in sorted(p for p in dataset_dir.iterdir()
                                      if p.is_dir()):
                if (version_dir / artifact.MANIFEST_NAME).is_file():
                    record = plan_directory(version_dir)
                    if record["status"] != "not_a_legacy_manifest":
                        records.append(record)
    plan = {
        "schema": SCHEMA,
        "root": str(root),
        "by_status": dict(Counter(record["status"] for record in records)),
        "directories": records,
    }
    plan["plan_digest"] = artifact.sha256_json(
        {key: value for key, value in plan.items() if key != "plan_digest"})
    return plan


def migrated_manifest(version_dir: Path, record: dict) -> artifact.ArtifactManifest:
    legacy = record["legacy_manifest"]
    return artifact.ArtifactManifest(
        kind=LEGACY_KIND,
        dataset=record["dataset"],
        version=record["version"],
        generator=record["generator"],
        git_commit=record["git_commit"],
        created=record["created"],
        arguments=(),
        content_digest=artifact.sha256_directory(version_dir),
        # No typed upstream refs exist for these: the legacy `inputs` were
        # bare paths, and the artifacts they named may not survive under those
        # names. Recording the strings in config rather than inventing refs
        # that would fail to open.
        upstreams=(),
        config={
            "schema": SCHEMA,
            "migrated_from": "pre_contract_hand_written_manifest",
            "original_kind": record["from_kind"],
            "original_manifest": legacy,
            "payload_layout": "sentences/ and sentence_requests/ trees; this "
                              "is NOT the current frame_landmarks contract of "
                              "one canonical predictions.jsonl",
        },
        declared_outputs=tuple(record["declared_outputs"]),
        # The producing commit is preserved from the legacy manifest; no diff
        # was ever recorded, and inventing a clean one would claim the tree
        # was clean when nobody knows.
        code_provenance={
            "schema": code_provenance.SCHEMA,
            "commit": record["git_commit"],
            "diff": "",
            "diff_sha256": hashlib.sha256(b"").hexdigest(),
            "computational_diff_sha256":
                hashlib.sha256(b"").hexdigest(),
            "dirty": False,
            "computationally_dirty": False,
            "note": "commit preserved from the pre-contract manifest; no "
                    "working diff was recorded at the time",
        },
    )


def apply_plan(root: Path, plan: dict) -> list[Path]:
    written = []
    for record in plan["directories"]:
        if record["status"] != "migrate":
            continue
        version_dir = Path(record["path"])
        manifest = migrated_manifest(version_dir, record)
        backup = version_dir / "manifest.pre_contract.json"
        if not backup.exists():
            # The original is kept verbatim beside the new one. It is the only
            # copy of what this directory claimed about itself.
            artifact.atomic_create_json(backup, record["legacy_manifest"])
        artifact.atomic_write_json(
            version_dir / artifact.MANIFEST_NAME, manifest.to_dict())
        # Prove the result is readable rather than trusting that it is.
        artifact.load_manifest(version_dir)
        written.append(version_dir)
    return written


def _print_plan(plan: dict) -> None:
    print(f"root:        {plan['root']}")
    print(f"plan digest: {plan['plan_digest']}")
    print("\nstatus:")
    for status, count in sorted(plan["by_status"].items()):
        print(f"  {status:<28} {count:>4}")
    for record in plan["directories"]:
        if record["status"] == "migrate":
            print(f"  {record['dataset']}/{record['version']}: "
                  f"{record['from_kind']} -> {record['to_kind']}, "
                  f"outputs {record['declared_outputs']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--farfield_root", type=Path, default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm_plan_digest", default=None)
    args = parser.parse_args(argv)
    root = Path(args.farfield_root or paths_lib.default_root())
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    plan = build_plan(root)
    _print_plan(plan)
    if not args.apply:
        return 0
    if args.confirm_plan_digest != plan["plan_digest"]:
        parser.error(
            "--confirm_plan_digest does not match the plan just computed")
    for path in apply_plan(root, plan):
        print(f"  migrated {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
