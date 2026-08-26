"""What determines an artifact, hashed -- one node of the build's Merkle DAG.

The pipeline used to identify an artifact by `build_identity`, a digest over
the *entire* build config and *all* its inputs. That is sound but far too
broad: changing `localization.pi0`, a knob only the last stage reads, produced
a new build identity, so the paid `frame_landmarks` and the hours of
`object_tracks` upstream of it were declared to "belong to a different
immutable build identity" and had to be republished.

`stage_reuse.py` existed entirely to undo that, and could only do so by asking
a human to attest that the prefix-computing code had not changed -- the one
unverified claim in an otherwise machine-checked system.

The fix is to identify an artifact by what actually determines it:

    identity = H(kind, dataset,
                 the stage's own resolved config,
                 the manifest digests of its upstreams,
                 the fingerprint of the code that computes it,
                 the dataset source digests, when the stage reads them)

Nothing else. Then a downstream config change simply does not move an upstream
artifact's identity, no attestation is needed, and "did the code change?"
becomes a machine question (see `code_fingerprint`).

The upstream term needs no recursion: an `ArtifactRef` already carries the
`manifest_digest` of the artifact it names, and that manifest already records
*its* upstreams and *its* config. The chain is a Merkle DAG already; this
module only names the node.

WHAT THIS DELIBERATELY DOES NOT COVER, so nobody reads more into an identity
match than it means:

- provider non-determinism. Two extraction runs with identical inputs return
  different landmarks; identity says "the same recipe", never "the same
  bytes". Byte identity is what `content_digest` is for.
- anything outside the farfield package (see `code_fingerprint`'s limits).
- the mutable orchestration state in `builds/`. A build directory is where a
  run is driven from, not part of what its products are.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from experimental.overhead_matching.swag.farfield import (
    artifact,
    code_fingerprint,
)

SCHEMA = "farfield_artifact_identity/v2"

# A manifest written before per-artifact identity existed. Such an artifact is
# not *wrong*, it is unattributed: nothing on disk says which code or which
# resolved stage config produced it. Consumers must say what they want to do
# about that rather than have it decided for them; see `pipeline`'s
# `--assume-current`.
UNATTRIBUTED = "unattributed"


class ArtifactIdentityError(ValueError):
    """An identity cannot be computed or does not match what was recorded."""


def compute(*, kind: str, dataset: str, stage_config_digest: str,
            upstreams: Iterable[artifact.ArtifactRef],
            entry_module: str,
            dataset_source_digests: Mapping[str, str] | None = None) -> str:
    """The identity of an artifact this recipe would produce.

    `upstreams` is order-insensitive on purpose: a stage that takes tracks and
    audits produces the same thing whichever order the refs were listed in, and
    an identity that changed with listing order would report false differences.
    """
    references = list(upstreams)
    for reference in references:
        if not isinstance(reference, artifact.ArtifactRef):
            raise ArtifactIdentityError(
                "upstreams must be ArtifactRef values")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": artifact.require_identifier(kind, "artifact kind"),
        "dataset": artifact.require_identifier(dataset, "dataset"),
        "stage_config_digest": _digest(
            stage_config_digest, "stage_config_digest"),
        # Sorted by the manifest digest, not by kind: two upstreams of the
        # same kind (a full catalog and its trim) must not collide on the key.
        # The version name is deliberately absent -- it is a label, and two
        # refs with one manifest digest name one artifact whatever it is
        # called.
        "upstreams": sorted(
            _digest(reference.manifest_digest,
                    f"{reference.kind} manifest_digest")
            for reference in references),
        "code_fingerprint": code_fingerprint.fingerprint(entry_module),
        "dataset_source_digests": (
            dict(sorted(dataset_source_digests.items()))
            if dataset_source_digests is not None else None),
    }
    return artifact.sha256_json(payload)


def _digest(value: Any, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or not all(character in "0123456789abcdef" for character in value)):
        raise ArtifactIdentityError(f"{field} must be a sha256 hex digest")
    return value


def recorded(manifest: artifact.ArtifactManifest) -> str:
    """The identity a published artifact claims, or `UNATTRIBUTED`."""
    value = manifest.config.get("artifact_identity")
    if value is None:
        return UNATTRIBUTED
    return _digest(value, "recorded artifact_identity")


def explain(*, expected: str, manifest: artifact.ArtifactManifest,
            kind: str) -> str:
    """Why an artifact was rejected, in terms a rebuild decision needs.

    An identity is a digest, so a mismatch on its own tells a reader nothing
    about what to do. The recorded stage config digest and code fingerprint are
    kept beside the identity precisely so this message can name which of the
    two moved.
    """
    found = recorded(manifest)
    if found == UNATTRIBUTED:
        return (f"{kind} artifact {manifest.version!r} predates per-artifact "
                "identity and records none. Rebuild it, or accept it as-is "
                f"with --assume-current {kind} (which records that the claim "
                "was assumed, not proven).")
    return (f"{kind} artifact {manifest.version!r} was built from a different "
            f"recipe: identity {found[:12]} != {expected[:12]}. Compare its "
            "manifest's stage_config_digest and code_fingerprint against the "
            "current build to see which moved.")
