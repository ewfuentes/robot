"""What determines an artifact, hashed -- one node of the build's Merkle DAG.

The pipeline used to identify an artifact by `build_identity`, a digest over
the *entire* build config and *all* its inputs. That is sound but far too
broad: changing `localization.pi0`, a knob only the last stage reads, produced
a new build identity, so the paid `frame_landmarks` and the hours of
`object_tracks` upstream of it were declared to "belong to a different
immutable build identity" and had to be republished.

`stage_reuse.py` existed entirely to undo that, and could only do so by asking
a human to attest that the prefix-computing code had not changed. With code
out of identity there is nothing left to attest, and that whole mechanism --
along with its one unverified human claim -- has no reason to exist.

The fix is to identify an artifact by what actually determines it:

    identity = H(kind, dataset,
                 the stage's own resolved config,
                 the manifest digests of its upstreams,
                 every recorded build input the stage reads)

DATA LINEAGE ONLY. Code is deliberately absent, and `code_provenance` records
it instead. A v2 of this schema hashed a fingerprint of the producing code;
measured against the real tree, three commits touching only viewer HTML
invalidated all eight producing stages, and one ordinary day's work
invalidated two of eight even with presentation excluded. Code changes
constantly in a research tree and data does not, so gating on it means
near-permanent invalidation of the artifacts that cost money -- and these
artifacts are not byte-reproducible anyway, since extraction and matching are
provider calls with real variance. Gating and recording are separable; only
recording was ever needed.

The last term is default-include: an input is in unless `identity_inputs`
excludes it by name, so a build that starts recording something new is covered
before anyone remembers it exists. An enumeration of what to INCLUDE fails
silently when an entry is forgotten -- the identity still matches and the
stale artifact is still used. This way round the failure is an identity that
moves when it need not, which is a rebuild, and loud.

Nothing else. A downstream config change simply does not move an upstream
artifact's identity, and "did the code change?" is answered by reading
`code_provenance` when someone asks -- not by refusing to reuse.

The upstream term needs no recursion: an `ArtifactRef` already carries the
`manifest_digest` of the artifact it names, and that manifest already records
*its* upstreams and *its* config. The chain is a Merkle DAG already; this
module only names the node.

WHAT THIS DELIBERATELY DOES NOT COVER, so nobody reads more into an identity
match than it means:

- provider non-determinism. Two extraction runs with identical inputs return
  different landmarks; identity says "the same recipe", never "the same
  bytes". Byte identity is what `content_digest` is for.
- the code that produced the artifact. That is recorded by `code_provenance`
  and surfaced when a lineage spans more than one code state, but it never
  gates: see this module's opening note for the measurements behind that.
- the mutable orchestration state in `builds/`. A build directory is where a
  run is driven from, not part of what its products are.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from experimental.overhead_matching.swag.farfield import (
    artifact,
    identity_inputs,
)

SCHEMA = "farfield_artifact_identity/v3"

# A manifest that records no identity. After the legacy artifacts were signed
# in place this means one of two things: an artifact no gated stage produced
# (a catalog, a viewer sidecar) which nothing asks about, or one published by
# a producer run outside the pipeline, which had no identity to record. Such
# an artifact is not *wrong*, it is unattributed -- nothing on disk says which
# resolved recipe produced it, and no amount of reading it can recover that.
UNATTRIBUTED = "unattributed"



class ArtifactIdentityError(ValueError):
    """An identity cannot be computed or does not match what was recorded."""


def compute(*, kind: str, dataset: str, stage_config_digest: str,
            upstreams: Iterable[artifact.ArtifactRef],
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
              upstreams: Iterable[artifact.ArtifactRef],
              build_inputs: Mapping[str, str],
              inputs_not_consumed: tuple[str, ...] = ()) -> str:
    """`compute` for a producer, from the values it already holds."""
    if not isinstance(orchestration, Mapping):
        raise ArtifactIdentityError("orchestration must be the stage contract")
    digest = orchestration.get("config_digest")
    if digest is None:
        raise ArtifactIdentityError(
            "stage contract records no config_digest")
    return compute(
        kind=kind, dataset=dataset, stage_config_digest=digest,
        upstreams=upstreams, build_inputs=build_inputs,
        inputs_not_consumed=inputs_not_consumed)


def _digest(value: Any, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or not all(character in "0123456789abcdef" for character in value)):
        raise ArtifactIdentityError(f"{field} must be a sha256 hex digest")
    return value


def recorded(manifest: artifact.ArtifactManifest) -> str:
    """The identity a published artifact claims, or `UNATTRIBUTED`.

    A top-level manifest field rather than a `config` entry: `config` is the
    stage's resolved recipe, and the identity is a property OF the artifact
    computed partly from that recipe. Putting it in `config` would also have
    made it one of the inputs to its own stage config digest.

    The manifest is the ONLY place an identity is read from. There was for a
    while a second path -- a derived index beside the data, for artifacts
    published before identity existed -- and two lookup paths for one fact is
    one too many. `manifest_digest` excluding `artifact_identity` is what let
    those artifacts be signed in place instead; see the decision journal.
    """
    if manifest.artifact_identity is None:
        return UNATTRIBUTED
    return _digest(manifest.artifact_identity, "recorded artifact_identity")


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
        # Deliberately does not offer a flag. An earlier version of this
        # message told the reader to pass `--assume-current`, which never
        # existed -- advice that sends someone hunting for a flag is worse
        # than no advice.
        return (f"{kind} artifact {manifest.version!r} records no identity, "
                "so it was not published by this pipeline. Rebuild it through "
                "`pipeline run`, which records the identity of what it "
                "builds.")
    return (f"{kind} artifact {manifest.version!r} was built from a different "
            f"recipe: identity {found[:12]} != {expected[:12]}. Compare its "
            "manifest's stage_config_digest, upstream refs and build inputs "
            "against the current build to see which moved. Its code state is "
            "recorded separately (see code_provenance) and is NOT part of "
            "this identity.")
