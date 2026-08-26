"""Compute the identity of artifacts published before identity existed.

115 artifacts on disk record no `artifact_identity`. Every term needed to
compute one survives, so this is arithmetic rather than an assumption:

- kind, dataset, and the upstream manifest digests are in the manifest;
- the stage's resolved config digest is `config.orchestration.config_digest`;
- the build inputs are in the build recipe that produced it, joined by
  `config.build_identity`. Measured on the real root: 24 build recipes survive
  and all 56 artifacts that record a build identity join to one, none missing.

With code out of identity there is no term left that has to be guessed. That
is the payoff of `artifact_identity` being data lineage: a backfill is a
computation, not a claim.

WHY A SIDECAR AND NOT THE MANIFEST. An artifact's `manifest_digest` is the
sha256 of its `manifest.json`, and every downstream artifact records that
digest in its `ArtifactRef`. Editing a published manifest to add a field
changes its digest and silently invalidates every reference to it -- and
publishing immutable artifacts is the contract the whole system rests on. So
the backfill is a derived index beside the data, never a retroactive edit
inside it. `artifact_identity.resolve` consults it for artifacts whose
manifest predates the field.

Report-only by default. Applying is a separate invocation that must spell the
reported plan digest back, so a reviewed plan cannot silently grow.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from experimental.overhead_matching.swag.farfield import (
    artifact,
    artifact_identity,
    paths as paths_lib,
    pipeline,
    provenance,
)

SCHEMA = "farfield.artifact_identity_backfill/v1"
INDEX_NAME = "artifact_identity_backfill.json"


class BackfillError(ValueError):
    """The backfill cannot be computed or applied."""


def index_path(root: Path) -> Path:
    return Path(root) / INDEX_NAME


def load_builds(root: Path) -> dict[str, dict]:
    """Every surviving build recipe, keyed by its build identity."""
    builds: dict[str, dict] = {}
    builds_root = Path(root) / "builds"
    if not builds_root.is_dir():
        return builds
    for config in sorted(builds_root.glob("*/*/build_config.json")):
        try:
            document = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        identity = document.get("build_identity")
        if isinstance(identity, str) and identity:
            # Two builds cannot share an identity: it is a digest of their
            # config and inputs, so a collision would mean identical recipes,
            # and either copy answers the question.
            builds.setdefault(identity, document)
    return builds


def plan_artifact(version_dir: Path, builds: dict[str, dict]) -> dict:
    """What identity this artifact should carry, or why it cannot get one."""
    record: dict[str, Any] = {"path": str(version_dir), "status": "pending"}
    # A viewer or review bundle carries a provenance manifest under the same
    # filename. Feeding it to the artifact loader would report a healthy side
    # output as corrupt -- the survey made exactly that mistake, so the two
    # tools classify identically here.
    try:
        document = json.loads(
            (version_dir / artifact.MANIFEST_NAME).read_text(encoding="utf-8"))
        if isinstance(document, dict) and document.get(
                "schema") == provenance.SCHEMA:
            record["status"] = "side_output"
            return record
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    try:
        manifest = artifact.load_manifest(version_dir)
    except (artifact.ArtifactError, OSError, ValueError) as error:
        record.update(status="unreadable", reason=str(error))
        return record
    record.update(kind=manifest.kind, dataset=manifest.dataset,
                  version=manifest.version)
    if artifact_identity.recorded(manifest) != artifact_identity.UNATTRIBUTED:
        record["status"] = "already_attributed"
        return record

    owner = pipeline.PIPELINE_ARTIFACT_OWNER.get(manifest.kind)
    if owner is None:
        # Catalogs, coverage plots and viewers are not produced by a pipeline
        # stage, so no stage config or exclusion list describes them. Left
        # alone rather than given an identity nothing would check.
        record.update(status="not_a_pipeline_stage_output")
        return record

    orchestration = manifest.config.get("orchestration")
    if not isinstance(orchestration, dict) or not orchestration.get(
            "config_digest"):
        record.update(status="no_stage_config_digest")
        return record

    build_identity = manifest.config.get("build_identity")
    build = builds.get(build_identity) if isinstance(build_identity, str) \
        else None
    if build is None:
        record.update(status="no_surviving_build", build_identity=build_identity)
        return record

    try:
        identity = artifact_identity.compute(
            kind=manifest.kind,
            dataset=manifest.dataset,
            stage_config_digest=orchestration["config_digest"],
            upstreams=manifest.upstreams,
            build_inputs=build["inputs"],
            inputs_not_consumed=pipeline.STAGE_SPECS[owner].inputs_not_consumed,
        )
    except (artifact_identity.ArtifactIdentityError, KeyError,
            TypeError) as error:
        record.update(status="uncomputable", reason=str(error))
        return record
    record.update(status="computed", identity=identity, stage=owner,
                  build_identity=build_identity)
    return record


def build_plan(root: Path) -> dict:
    root = Path(root)
    builds = load_builds(root)
    records = []
    artifacts_root = root / "artifacts"
    roots = []
    if artifacts_root.is_dir():
        roots += sorted(artifacts_root.glob("*/*/*"))
    runs_root = root / "runs"
    if runs_root.is_dir():
        roots += sorted(runs_root.glob("*/*"))
    for candidate in roots:
        if (candidate.is_dir()
                and (candidate / artifact.MANIFEST_NAME).is_file()):
            records.append(plan_artifact(candidate, builds))
    plan = {
        "schema": SCHEMA,
        "root": str(root),
        "n_builds": len(builds),
        "by_status": dict(Counter(record["status"] for record in records)),
        "artifacts": records,
    }
    plan["plan_digest"] = artifact.sha256_json(
        {key: value for key, value in plan.items() if key != "plan_digest"})
    return plan


def apply_plan(root: Path, plan: dict) -> Path:
    """Write the derived index. Never touches a published artifact."""
    entries = {record["path"]: {
        "identity": record["identity"],
        "kind": record["kind"],
        "dataset": record["dataset"],
        "version": record["version"],
        "stage": record["stage"],
        "build_identity": record["build_identity"],
        # Says plainly how the value was obtained, so a reader never has to
        # wonder whether it was computed or assumed.
        "basis": "computed_from_manifest_and_surviving_build_recipe",
    } for record in plan["artifacts"] if record["status"] == "computed"}
    target = index_path(Path(root))
    if target.exists():
        raise BackfillError(
            f"{target} already exists; remove it to recompute the backfill")
    artifact.atomic_create_json(target, {
        "schema": SCHEMA,
        "plan_digest": plan["plan_digest"],
        "entries": entries,
    })
    return target


def load_index(root: Path) -> dict[str, str]:
    """Backfilled identities by artifact path, or empty when absent."""
    try:
        document = json.loads(
            index_path(Path(root)).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeError, OSError):
        return {}
    entries = document.get("entries")
    if document.get("schema") != SCHEMA or not isinstance(entries, dict):
        return {}
    return {path: value["identity"] for path, value in entries.items()
            if isinstance(value, dict) and isinstance(value.get("identity"), str)}


def _print_plan(plan: dict) -> None:
    print(f"root:        {plan['root']}")
    print(f"build recipes found: {plan['n_builds']}")
    print(f"plan digest: {plan['plan_digest']}")
    print("\nstatus:")
    for status, count in sorted(plan["by_status"].items()):
        print(f"  {status:<32} {count:>5}")
    blocked = [r for r in plan["artifacts"]
               if r["status"] in ("no_surviving_build", "uncomputable",
                                  "no_stage_config_digest")]
    if blocked:
        print(f"\n{len(blocked)} artifact(s) cannot be computed:")
        for record in blocked[:10]:
            print(f"  {record['status']:<28} {record['path']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--farfield_root", type=Path, default=None)
    parser.add_argument("--apply", action="store_true",
                        help="write the derived index (requires the digest)")
    parser.add_argument("--confirm_plan_digest", default=None,
                        help="the digest printed while planning")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.farfield_root or paths_lib.default_root())
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    plan = build_plan(root)
    if args.json:
        json.dump(plan, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        _print_plan(plan)
    if not args.apply:
        return 0
    if args.confirm_plan_digest != plan["plan_digest"]:
        parser.error(
            "--confirm_plan_digest does not match the plan just computed; "
            "review the plan and pass the digest it printed")
    written = apply_plan(root, plan)
    print(f"\nwrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
