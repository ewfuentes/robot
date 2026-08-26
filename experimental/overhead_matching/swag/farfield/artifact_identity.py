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
                 every recorded build input the stage reads)

The last term is default-include: an input is in unless `identity_inputs`
excludes it by name, so a build that starts recording something new is covered
before anyone remembers it exists. An enumeration of what to INCLUDE fails
silently when an entry is forgotten -- the identity still matches and the
stale artifact is still used. This way round the failure is an identity that
moves when it need not, which is a rebuild, and loud.

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
    identity_inputs,
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
            build_inputs: Mapping[str, str],
            inputs_not_consumed: tuple[str, ...] = ()) -> str:
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
        # Default-include: every recorded build input is here unless
        # `identity_inputs` excludes it by name. A new input is covered before
        # anyone remembers it exists, and the cost of a missing exclusion is
        # an identity that moves when it need not -- loud -- rather than one
        # that matches when it must not.
        "build_inputs": identity_inputs.contributing(
            build_inputs, inputs_not_consumed),
    }
    return artifact.sha256_json(payload)


def for_stage(*, kind: str, dataset: str, orchestration: Mapping[str, Any],
              upstreams: Iterable[artifact.ArtifactRef], source_file: str,
              build_inputs: Mapping[str, str],
              inputs_not_consumed: tuple[str, ...] = ()) -> str:
    """`compute` for a producer, from the values it already holds.

    `source_file` is the producer's own `__file__`. Under `bazel run` a stage
    sees `__name__ == "__main__"` and cannot name itself, and a hand-written
    module constant is one more thing to keep in step with the file it
    describes -- so the file is the input, and the module name is derived.
    """
    if not isinstance(orchestration, Mapping):
        raise ArtifactIdentityError("orchestration must be the stage contract")
    digest = orchestration.get("config_digest")
    if digest is None:
        raise ArtifactIdentityError(
            "stage contract records no config_digest")
    return compute(
        kind=kind, dataset=dataset, stage_config_digest=digest,
        upstreams=upstreams,
        entry_module=code_fingerprint.module_of(source_file),
        build_inputs=build_inputs,
        inputs_not_consumed=inputs_not_consumed)


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
