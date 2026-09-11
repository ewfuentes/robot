#!/usr/bin/env python3
"""Seal an existing ``<version>.incomplete`` directory as a typed artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact, code_provenance


def publish_staged(
    destination: Path,
    *,
    generator: str,
    producer_command: str,
    declared_outputs: list[str],
    upstream_paths: list[Path],
    config: dict,
) -> artifact.ArtifactRef:
    """Validate, manifest, and atomically publish one staged artifact."""
    destination = destination.expanduser().resolve()
    staging = destination.with_name(destination.name + artifact.INCOMPLETE_SUFFIX)
    if destination.exists() or destination.is_symlink():
        raise artifact.ArtifactExistsError(
            f"completed artifact already exists: {destination}")
    if not staging.is_dir() or staging.is_symlink():
        raise artifact.ArtifactValidationError(
            f"staging directory does not exist: {staging}")
    manifest_path = staging / artifact.MANIFEST_NAME
    if manifest_path.exists() or manifest_path.is_symlink():
        raise artifact.ArtifactValidationError(
            f"staging directory already contains {artifact.MANIFEST_NAME}")

    provenance = code_provenance.record()
    outputs = tuple(sorted(set(declared_outputs)))
    if len(outputs) != len(declared_outputs):
        raise artifact.ArtifactValidationError("declared outputs must be unique")
    actual_outputs = tuple(sorted(
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file()
    ))
    if outputs != actual_outputs:
        missing = sorted(set(outputs) - set(actual_outputs))
        undeclared = sorted(set(actual_outputs) - set(outputs))
        raise artifact.ArtifactValidationError(
            f"artifact output mismatch: missing={missing}, "
            f"undeclared={undeclared}")
    upstreams = tuple(
        artifact.reference_from_manifest(path) for path in upstream_paths)
    manifest = artifact.ArtifactManifest(
        kind=destination.parent.parent.name,
        dataset=destination.parent.name,
        version=destination.name,
        generator=generator,
        git_commit=provenance["commit"],
        created=datetime.now(timezone.utc).isoformat(),
        arguments=(producer_command,),
        content_digest=artifact.sha256_directory(staging),
        upstreams=upstreams,
        config=config,
        declared_outputs=outputs,
        code_provenance=provenance,
    )
    artifact.atomic_write_json(manifest_path, manifest.to_dict())
    artifact.publish_directory_no_clobber(staging, destination)
    return artifact.open_artifact(
        destination,
        expected_kind=manifest.kind,
        expected_dataset=manifest.dataset,
        expected_version=manifest.version,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--producer-command", required=True)
    parser.add_argument("--declared-output", action="append", required=True)
    parser.add_argument("--upstream", type=Path, action="append", default=[])
    parser.add_argument("--config-json", default="{}")
    args = parser.parse_args()
    config = json.loads(args.config_json)
    if not isinstance(config, dict):
        parser.error("--config-json must decode to an object")
    reference = publish_staged(
        args.destination,
        generator=args.generator,
        producer_command=args.producer_command,
        declared_outputs=args.declared_output,
        upstream_paths=args.upstream,
        config=config,
    )
    print(json.dumps(reference.to_dict(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
