"""What per-artifact identity would mean for the artifacts already on disk.

Read-only, always. Before the pipeline starts checking `artifact_identity`,
somebody has to know what that check would say about the corpus that exists --
how many artifacts carry no identity, what they were built from, and which of
them are reachable from a frozen recipe. Answering that by running the switch
and seeing what breaks is the expensive way round.

Every artifact published before per-artifact identity records none, so it reads
as UNATTRIBUTED. That is neither an error nor a licence: the manifest simply
does not say which code or which resolved stage config produced it, and no
amount of reading it can recover that. There are exactly three honest
resolutions and this tool is here to size them:

1. REBUILD -- cheapest for anything derived and quick to recompute.
2. ADOPT -- record the identity the artifact WOULD have under today's code and
   config, marked as assumed rather than proven, so downstream inherits a
   weakened claim it can see. Right for the paid artifacts, where rebuilding
   means re-billing a provider for bytes already in hand.
3. LEAVE -- for artifacts nothing current consumes.

The report prints the counts and the per-artifact detail; it decides nothing
and writes nothing. `--json` emits the same content for a migration plan to
consume.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    artifact,
    artifact_identity,
    paths as paths_lib,
    provenance,
)

SCHEMA = "farfield.artifact_identity_survey/v1"


def _manifest_schema(version_dir: Path) -> str | None:
    """The `schema` a directory's manifest.json declares, if it parses.

    Two manifest formats share the filename: the typed artifact manifest and
    `provenance`'s lighter record for side outputs. Feeding one to the other's
    loader reports a perfectly healthy viewer sidecar as a corrupt artifact,
    which is what this survey did until it was run against the real root.
    """
    try:
        document = json.loads(
            (version_dir / artifact.MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    schema = document.get("schema") if isinstance(document, dict) else None
    return schema if isinstance(schema, str) else None


def _manifest_or_reason(version_dir: Path):
    """The manifest, or why it could not be read.

    The reason is the point. "unreadable" with no detail is the same
    unhelpful shape as a bearing count that says nothing about whether the
    bearings existed: a migration has to distinguish an older manifest schema
    -- which is expected, and adoptable -- from genuine corruption, which is
    a bug to look at.
    """
    try:
        return artifact.load_manifest(version_dir), None
    except (artifact.ArtifactError, OSError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"


def survey_artifact(version_dir: Path) -> dict:
    """One artifact's identity state, without judging what to do about it."""
    record = {
        "path": str(version_dir),
        "kind": None,
        "dataset": None,
        "version": version_dir.name,
        "identity": None,
        "generator": None,
        "git_commit": None,
        "created": None,
        "n_upstreams": None,
        "state": "unreadable",
    }
    if _manifest_schema(version_dir) == provenance.SCHEMA:
        # A side output (a viewer, a review bundle), not a pipeline artifact.
        # It carries provenance but has no identity to hold, by design.
        record["state"] = "side_output"
        return record
    manifest, reason = _manifest_or_reason(version_dir)
    if manifest is None:
        record["detail"] = reason
        return record
    record.update({
        "kind": manifest.kind,
        "dataset": manifest.dataset,
        "version": manifest.version,
        "generator": manifest.generator,
        "git_commit": manifest.git_commit,
        "created": manifest.created,
        "n_upstreams": len(manifest.upstreams),
    })
    try:
        identity = artifact_identity.recorded(manifest)
    except artifact_identity.ArtifactIdentityError as error:
        record["state"] = "malformed_identity"
        record["detail"] = str(error)
        return record
    if identity == artifact_identity.UNATTRIBUTED:
        record["state"] = "unattributed"
    else:
        record["state"] = "attributed"
        record["identity"] = identity
    return record


def survey(root: Path) -> dict:
    """Every artifact under `root`'s artifact lanes."""
    artifacts_root = Path(root) / "artifacts"
    records = []
    if artifacts_root.is_dir():
        for kind_dir in sorted(p for p in artifacts_root.iterdir()
                               if p.is_dir()):
            for dataset_dir in sorted(p for p in kind_dir.iterdir()
                                      if p.is_dir()):
                for version_dir in sorted(p for p in dataset_dir.iterdir()
                                          if p.is_dir()):
                    if (version_dir / artifact.MANIFEST_NAME).is_file():
                        records.append(survey_artifact(version_dir))
    runs_root = Path(root) / "runs"
    if runs_root.is_dir():
        for experiment in sorted(p for p in runs_root.iterdir() if p.is_dir()):
            for run in sorted(p for p in experiment.iterdir() if p.is_dir()):
                if (run / artifact.MANIFEST_NAME).is_file():
                    records.append(survey_artifact(run))
    return {
        "schema": SCHEMA,
        "root": str(Path(root)),
        "n_artifacts": len(records),
        "by_state": dict(Counter(record["state"] for record in records)),
        # Records with no kind are those whose manifest could not be read as
        # an artifact one; label them by their state rather than lumping a
        # side output in with a corrupt artifact.
        "by_kind": dict(Counter(
            record["kind"] or f"<{record['state']}>" for record in records)),
        "artifacts": records,
    }


def _print_report(report: dict) -> None:
    print(f"root: {report['root']}")
    print(f"artifacts with a manifest: {report['n_artifacts']}")
    print("\nidentity state:")
    for state, count in sorted(report["by_state"].items()):
        print(f"  {state:<20} {count:>5}")
    print("\nby kind:")
    for kind, count in sorted(report["by_kind"].items()):
        print(f"  {kind:<24} {count:>5}")
    unattributed = [r for r in report["artifacts"]
                    if r["state"] == "unattributed"]
    if unattributed:
        print(f"\n{len(unattributed)} artifact(s) record no identity. Each "
              "needs one of: rebuild, adopt-as-assumed, or leave.")
        print("  Distinct producing commits among them:")
        for commit, count in sorted(
                Counter(r["git_commit"] for r in unattributed).items()):
            print(f"    {str(commit)[:12]:<14} {count:>5}")
    side = [r for r in report["artifacts"] if r["state"] == "side_output"]
    if side:
        print(f"\n{len(side)} side output(s) carry a provenance manifest "
              "rather than an artifact one. They hold no identity by design.")
    broken = [r for r in report["artifacts"]
              if r["state"] in ("unreadable", "malformed_identity")]
    if broken:
        print(f"\n{len(broken)} artifact(s) could not be read:")
        reasons = Counter(
            str(record.get("detail", "")).split(";")[0][:90]
            for record in broken)
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>4}x  {reason}")
        print("  examples:")
        for record in broken[:5]:
            print(f"    {record['path']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--farfield_root", type=Path, default=None,
                        help="data root to survey (default: the resolved root)")
    parser.add_argument("--json", action="store_true",
                        help="emit the full report as JSON instead of text")
    args = parser.parse_args(argv)
    root = args.farfield_root or paths_lib.default_root()
    if not Path(root).is_dir():
        parser.error(f"not a directory: {root}")
    report = survey(Path(root))
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
