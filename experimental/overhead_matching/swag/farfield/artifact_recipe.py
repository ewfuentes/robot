"""What an artifact records so it can answer for itself.

Two questions have to be answerable about any artifact on disk:

1. **How do I reproduce it?** -- what settings and what inputs produced it.
2. **Are the things it was made from out of date?** -- `artifact_identity`
   already answers this, by recomputing the identity from the current recipe
   and comparing with what the artifact recorded.

Both were answerable only by joining a manifest to
`builds/<dataset>/<build>/build_config.json` through `build_identity`. That
join works -- measured on the real root, 24 surviving recipes covered all 56
signed artifacts with zero orphans -- but it is a join to a directory the
docs describe as "only orchestration state ... not a scientific artifact
lane". Nothing protects it. Lose one recipe and that artifact is both
irreproducible AND its identity uncomputable, which is exactly the hole the
identity backfill had to be written to paper over.

Of the four terms in an identity, one is already in every manifest -- kind and
dataset. The other three are recorded here: the stage's resolved config, the
build inputs that stage read, and the upstream digests that entered the
identity.

That last one needs saying. Identity is computed over the stage's CONFIGURED
upstreams, which is not the same list as `manifest.upstreams`. A
`frame_landmarks` manifest records its pinhole artifact and the canonical LLM
result artifact, but `extract` declares no artifact upstreams at all -- and
the orchestrator could not predict those two anyway, since the result
artifact only exists once the stage has run. So `manifest.upstreams` is the
fuller lineage record and the identity term is a subset of it. Recomputation
uses the recorded subset, and `verify_self_describing` checks it really is a
subset, so a recipe cannot invent lineage the manifest does not show.

The payoff is a check that is total rather than illustrative: an identity
recomputed FROM THE MANIFEST ALONE must equal the identity the manifest
records. If a producer ever stops recording a term that identity depends on,
that assertion fails -- it cannot be satisfied by a manifest that is missing
something.

Three ad-hoc conventions preceded this and none of them was complete:
`resolved_stage_config` (flat dotted keys, in `semantic_audits` and
`landmark_matches`), `resolved` (nested blocks, in `object_tracks`), and
per-domain blocks (in `localization_inputs`). `semantic_audits` recorded no
`source_digests` at all, so for that kind you could not even verify you had
found the right build recipe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from experimental.overhead_matching.swag.farfield import (
    artifact,
    artifact_identity,
    identity_inputs,
)

SCHEMA = "farfield_artifact_recipe/v1"


class ArtifactRecipeError(ValueError):
    """A recipe is missing, malformed, or disagrees with its artifact."""


def build(*, stage: str, stage_config: Mapping[str, Any],
          build_inputs: Mapping[str, str],
          identity_upstreams: Iterable[artifact.ArtifactRef] = (),
          inputs_not_consumed: tuple[str, ...] = ()) -> dict[str, Any]:
    """The block a producer stores, from the values the orchestrator resolved.

    `stage_config` must be exactly what the stage's `config_digest` is
    computed over, and `build_inputs` is stored already reduced to the
    contributing set -- so a reader needs no exclusion list, and the recorded
    inputs are the ones that actually shaped this artifact rather than
    everything the build happened to know.
    """
    return {
        "schema": SCHEMA,
        "stage": stage,
        "stage_config": dict(sorted(stage_config.items())),
        "build_inputs": identity_inputs.contributing(
            build_inputs, inputs_not_consumed),
        # Sorted, matching how `artifact_identity` orders them: the identity
        # is order-insensitive, so a recorded order would be noise that could
        # disagree without meaning anything.
        "identity_upstreams": sorted(
            reference.manifest_digest for reference in identity_upstreams),
    }


def load(path: Path | str | None) -> dict[str, Any] | None:
    """Read the recipe the orchestrator wrote, or None when run by hand.

    A producer driven directly gets no recipe and its artifact is honestly
    not self-describing, which is the same shape as `--artifact_identity`.
    """
    if path is None:
        return None
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactRecipeError(
            f"cannot read artifact recipe {path}: {error}") from error
    return validate(document)


def validate(recipe: Any) -> dict[str, Any]:
    if not isinstance(recipe, Mapping):
        raise ArtifactRecipeError("artifact recipe must be an object")
    missing = sorted({"schema", "stage", "stage_config", "build_inputs",
                      "identity_upstreams"} - set(recipe))
    if missing:
        raise ArtifactRecipeError(f"artifact recipe is missing {missing}")
    if recipe["schema"] != SCHEMA:
        raise ArtifactRecipeError(
            f"unsupported artifact-recipe schema {recipe['schema']!r}")
    for field in ("stage_config", "build_inputs"):
        if not isinstance(recipe[field], Mapping):
            raise ArtifactRecipeError(
                f"artifact recipe {field} must be an object")
    upstreams = recipe["identity_upstreams"]
    if not isinstance(upstreams, (list, tuple)) or not all(
            isinstance(item, str) for item in upstreams):
        raise ArtifactRecipeError(
            "artifact recipe identity_upstreams must be a list of digests")
    return dict(recipe)


def stage_config_digest(recipe: Mapping[str, Any]) -> str:
    """The digest the stage contract would have computed for this config."""
    return artifact.sha256_json(dict(validate(recipe)["stage_config"]))


def identity_from_manifest(manifest: artifact.ArtifactManifest) -> str:
    """Recompute this artifact's identity using only its own manifest.

    Nothing else is read -- no build directory, no upstream artifact, no
    exclusion list. `build_inputs` was stored already reduced, so
    `inputs_not_consumed` is empty here by construction.
    """
    if manifest.recipe is None:
        raise ArtifactRecipeError(
            f"{manifest.kind} artifact {manifest.version!r} records no "
            "recipe, so its identity cannot be recomputed from its manifest")
    recipe = validate(manifest.recipe)
    # A set is enough: `ArtifactManifest` refuses duplicate upstreams
    # ("artifact upstream identities must be unique"), so one digest selects
    # at most one ref and the count `compute` sees cannot be inflated.
    recorded = set(recipe["identity_upstreams"])
    return artifact_identity.compute(
        kind=manifest.kind,
        dataset=manifest.dataset,
        stage_config_digest=stage_config_digest(recipe),
        upstreams=[reference for reference in manifest.upstreams
                   if reference.manifest_digest in recorded],
        build_inputs=recipe["build_inputs"],
    )


def verify_self_describing(manifest: artifact.ArtifactManifest) -> None:
    """Assert the manifest carries every term its identity depends on.

    This is the whole point of the module, so it is a function rather than a
    comment. A manifest that omits a term cannot pass: the recomputed digest
    would differ from the recorded one.
    """
    recorded = artifact_identity.recorded(manifest)
    if recorded == artifact_identity.UNATTRIBUTED:
        raise ArtifactRecipeError(
            f"{manifest.kind} artifact {manifest.version!r} records no "
            "identity, so there is nothing to verify the recipe against")
    declared = set(validate(manifest.recipe)["identity_upstreams"]) \
        if manifest.recipe else set()
    present = {reference.manifest_digest
               for reference in manifest.upstreams}
    invented = sorted(declared - present)
    if invented:
        raise ArtifactRecipeError(
            f"{manifest.kind} artifact {manifest.version!r} recipe names "
            f"upstream digests its manifest does not record: {invented}. A "
            "recipe may name a subset of the manifest's lineage, never "
            "something outside it.")
    recomputed = identity_from_manifest(manifest)
    if recomputed != recorded:
        raise ArtifactRecipeError(
            f"{manifest.kind} artifact {manifest.version!r} is not "
            f"self-describing: identity recomputed from its manifest "
            f"({recomputed[:12]}) differs from the identity it records "
            f"({recorded[:12]}). Its recipe is missing or misrecords a term "
            "that identity depends on.")


def describe(manifest: artifact.ArtifactManifest) -> str:
    """A human-readable account of how to reproduce this artifact."""
    recipe = validate(manifest.recipe) if manifest.recipe else None
    lines = [
        f"{manifest.kind} / {manifest.dataset} / {manifest.version}",
        f"  generator:  {manifest.generator}",
        f"  git commit: {manifest.git_commit}",
        f"  identity:   {artifact_identity.recorded(manifest)}",
    ]
    if recipe is None:
        lines.append("  recipe:     NONE RECORDED -- reproduce it by finding "
                     "the build recipe whose build_identity this manifest's "
                     "config names")
        return "\n".join(lines)
    lines.append(f"  stage:      {recipe['stage']}")
    lines.append("  resolved stage config:")
    for key, value in recipe["stage_config"].items():
        lines.append(f"    {key} = {value!r}")
    lines.append("  build inputs that shaped it:")
    for key, value in recipe["build_inputs"].items():
        lines.append(f"    {key} = {value}")
    if manifest.upstreams:
        lines.append("  upstream artifacts:")
        for reference in manifest.upstreams:
            lines.append(
                f"    {reference.kind}/{reference.dataset}/"
                f"{reference.version}  manifest={reference.manifest_digest[:12]}")
    return "\n".join(lines)
