"""An input must still be what the configured lane holds right now.

`open_artifact` proves a directory is the artifact it claims to be, from the
manifest inside that directory. What it cannot prove is that the artifact is
still the one the build's recipe points at: a ref recorded by an upstream
stage names a path, a version and two digests, and the lane it came from can
have been republished since. So this reopens the configured lane and requires
what is there NOW to be the same artifact that was read.

This deliberately does NOT care which path the ref names. An earlier version
rejected a byte-identical copy read from outside the lane, on the reasoning
that "the path is the whole question". It is not, and `ArtifactRef` says so
itself: `path` is declared `field(compare=False)` because "moving an
immutable artifact does not change its identity; the two digests do". A copy
with the same kind, dataset, version and digests IS the artifact. The
wrong-directory bug that check claimed to stop -- a stage handed leg2's
tracks against leg1's audits -- is caught by the kind/dataset/version checks
below, which read the copy's own manifest and do not care where it sits.

What is left is one narrow, real guarantee: the artifact a stage consumed
agrees with the configured lane's current contents. That fails when a ref has
gone stale, and stale-but-well-formed is the failure that produces confident
wrong numbers.
"""

from __future__ import annotations

from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact


class ConfiguredLaneError(ValueError):
    """An input disagrees with what its configured lane holds."""

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
    """Check `reference` against the configured lane and return its manifest."""
    if reference.kind != kind or reference.dataset != document.get("dataset"):
        raise ConfiguredLaneError(
            f"configured-lane check received the wrong {kind} ref")
    lane = expected_lane(document, kind)
    version = document["config"]["artifacts"][f"{kind}_version"]
    if reference.version != version:
        raise ConfiguredLaneError(
            f"{kind} ref does not use the configured version {version!r}")
    try:
        reopened = artifact.open_artifact(
            lane, expected_kind=kind, expected_dataset=document["dataset"],
            expected_version=version)
    except artifact.ArtifactError as error:
        raise ConfiguredLaneError(
            f"configured {kind} lane {lane} is not a valid artifact: {error}"
        ) from error
    # `ArtifactRef` equality already excludes `path` by design, so this is
    # the comparison the type intends and not a hand-rolled field list.
    if reopened != reference:
        differing = [
            field for field, value in reopened.to_dict().items()
            if field != "path" and value != getattr(reference, field)]
        raise ConfiguredLaneError(
            f"{kind} at its configured lane {lane} is not the artifact that "
            f"was read: {', '.join(differing)} differ. The input is stale -- "
            "the lane was republished after this ref was recorded.")
    return artifact.load_manifest(lane)
