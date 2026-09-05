"""Stable localization-run identity derived from immutable build inputs."""

import re

from experimental.overhead_matching.swag.farfield import artifact

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def localization_run_version(run_name: str, object_tracks_version: str,
                             build_identity: str) -> str:
    """Return a collision-resistant, path-safe localization artifact version."""
    run_name = artifact.require_identifier(run_name, "localization run name")
    tracks = artifact.require_identifier(
        object_tracks_version, "object-track artifact version")
    if not isinstance(build_identity, str) or not _SHA256_RE.fullmatch(
            build_identity):
        raise ValueError("build identity must be a lowercase SHA-256 digest")
    return f"{run_name}--tracks-{tracks}--build-{build_identity}"


def from_build_document(document: dict) -> str:
    config = document.get("config")
    if not isinstance(config, dict):
        raise ValueError("build document has no config object")
    localization = config.get("localization")
    artifacts = config.get("artifacts")
    if not isinstance(localization, dict) or not isinstance(artifacts, dict):
        raise ValueError(
            "build config must record localization and artifact identities")
    return localization_run_version(
        localization.get("run_name"),
        artifacts.get("object_tracks_version"),
        document.get("build_identity"))
