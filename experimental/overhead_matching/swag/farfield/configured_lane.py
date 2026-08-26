"""An input must come from its configured lane, not a copy that matches it.

`open_artifact` proves a directory is the artifact it claims to be. It does
not prove it is the artifact the recipe MEANT, because a byte-identical copy
elsewhere on disk passes every content check: same kind, same dataset, same
version, same digests. Only its path differs, and the path is the whole
question -- "did this stage read the tracks the build configured, or a copy
somebody left in a scratch directory?"

That check used to live inside `stage_reuse`, which is gone: it existed to
grant human-attested exceptions to a whole-build identity check, and with
identity now data lineage (`artifact_identity`) there is nothing to except.
But the lane check was never about reuse, and it is the guarantee that stops
the wrong-directory class of bug -- a stage handed leg2's tracks against
leg1's audits, every number well-formed and every answer wrong. So it survives
here, on its own, doing one thing.
"""

from __future__ import annotations

from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact


class ConfiguredLaneError(ValueError):
    """An artifact was read from somewhere other than its configured lane."""


def expected_lane(document: dict, kind: str) -> Path:
    """Where a build's recipe says an artifact of `kind` must live."""
    inputs = document.get("inputs")
    root = inputs.get("farfield_root") if isinstance(inputs, dict) else None
    if not isinstance(root, str) or not root:
        raise ConfiguredLaneError(
            "build does not record inputs.farfield_root, so no lane can be "
            "checked")
    try:
        version = document["config"]["artifacts"][f"{kind}_version"]
        dataset = document["dataset"]
    except (KeyError, TypeError) as error:
        raise ConfiguredLaneError(
            f"build does not configure an exact {kind} lane") from error
    return Path(root) / "artifacts" / kind / dataset / version


def require(reference: artifact.ArtifactRef, *, document: dict,
            kind: str) -> artifact.ArtifactManifest:
    """Reopen `reference` only at its exact configured lane."""
    if reference.kind != kind or reference.dataset != document.get("dataset"):
        raise ConfiguredLaneError(
            f"configured-lane check received the wrong {kind} ref")
    lane = expected_lane(document, kind)
    version = document["config"]["artifacts"][f"{kind}_version"]
    if reference.version != version:
        raise ConfiguredLaneError(
            f"{kind} ref does not use the configured version {version!r}")
    if Path(reference.path).resolve() != lane.resolve():
        raise ConfiguredLaneError(
            f"{kind} must be read from its exact configured lane {lane}, not "
            f"{reference.path}")
    try:
        reopened = artifact.open_artifact(
            lane, expected_kind=kind, expected_dataset=document["dataset"],
            expected_version=version)
    except artifact.ArtifactError as error:
        raise ConfiguredLaneError(
            f"configured {kind} lane {lane} is not a valid artifact: {error}"
        ) from error
    if reopened.to_dict() != reference.to_dict():
        raise ConfiguredLaneError(
            f"{kind} at its configured lane is not the artifact that was read")
    return artifact.load_manifest(lane)
