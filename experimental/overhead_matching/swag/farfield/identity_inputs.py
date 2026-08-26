"""Which recorded build inputs belong in an artifact's identity.

The old `build_identity` hashed the whole `inputs` map, so every input was
covered whether or not anyone had thought about it. Replacing it with a
per-artifact identity means naming the inputs again -- and an enumeration of
what to INCLUDE is the wrong shape for that, because forgetting an entry is
silent: the identity still matches, the stale artifact is still reused, and
nothing anywhere raises.

So the list runs the other way. **Everything recorded is in the identity
unless this module excludes it by name.** A build that starts recording a new
input is covered from the moment it exists. The cost of forgetting to exclude
something is an identity that moves when it did not need to -- a rebuild
nobody wanted, which is loud, annoying, and immediately obvious. The cost of
forgetting to include something is a wrong answer nobody sees. Those are not
symmetric, and the default should fall on the side that shouts.

Two kinds of exclusion, each stated with its reason:

- GLOBAL: values that are locations rather than content. An artifact copied to
  a mirror root, or a checkpoint reached by a different path, is the same
  artifact; an identity keyed on absolute paths would say otherwise and would
  make every mirror a full rebuild. Each excluded path has a `_sha256` twin
  that IS included, so the bytes still count -- only the route to them does
  not.
- PER STAGE: an input this stage does not read. Extraction does not consult
  the mount calibration, so a corrected `nominal_forward.json` should not
  invalidate paid extraction. These live on `StageSpec.inputs_not_consumed`
  beside the stage's other declarations, and the same asymmetry applies: an
  input left off a stage's exclusion list is merely over-counted.
"""

from __future__ import annotations

from collections.abc import Mapping

# Locations, not content. Every one of these has a `_sha256` companion that is
# included, except where noted.
GLOBAL_EXCLUSIONS: Mapping[str, str] = {
    "farfield_root":
        "absolute path to the data root; a mirror holds the same artifacts",
    "dataset_base":
        "absolute path to the dataset; its bytes enter through the "
        "dataset_*_sha256 digests",
    "sam2_checkpoint":
        "path to the weights; sam2_checkpoint_sha256 carries their bytes",
    "motion_source":
        "path to the motion table; motion_source_sha256 carries its bytes",
    "nominal_forward_calibration":
        "path to the calibration; nominal_forward_sha256 carries its bytes",
    "video":
        "path to the source video; video_sha256 carries its bytes",
    "source_config":
        "path to the YAML a build was created from; the RESOLVED config is "
        "the authority and is already hashed per stage",
    "source_config_sha256":
        "the source YAML's bytes are provenance, not a determinant: two "
        "files differing only in comments resolve to one config, and the "
        "resolved config is what every stage reads",
    "identity_review_output_dir":
        "path to a review gate's output location; the review artifact itself "
        "enters as a typed upstream when a stage consumes it",
    "identity_review_phase":
        "a constant label describing when the gate runs, not an input value",
}


class IdentityInputError(ValueError):
    """A build's inputs cannot be reduced to an identity contribution."""


def contributing(inputs: Mapping[str, str],
                 not_consumed: tuple[str, ...] = ()) -> dict[str, str]:
    """The inputs that shape one stage's output, from what a build recorded.

    Default-include: an unfamiliar key is kept. That is the point -- a new
    input is covered before anyone remembers it exists.
    """
    if not isinstance(inputs, Mapping):
        raise IdentityInputError("build inputs must be a mapping")
    unknown_exclusion = sorted(set(not_consumed) - set(inputs))
    if unknown_exclusion:
        # A stage excluding a key that is not recorded is a stale declaration:
        # harmless today, wrong the moment the key comes back under a new
        # meaning. Fail while the mistake is cheap to fix.
        raise IdentityInputError(
            f"stage excludes inputs that no build records: {unknown_exclusion}")
    dropped = set(GLOBAL_EXCLUSIONS) | set(not_consumed)
    contributed = {key: value for key, value in inputs.items()
                   if key not in dropped}
    for key, value in contributed.items():
        if not isinstance(value, str):
            raise IdentityInputError(
                f"build input {key!r} must be a string to enter an identity")
    return dict(sorted(contributed.items()))


def audit(recorded_keys: frozenset[str],
          consumed_by_any_stage: frozenset[str]) -> list[str]:
    """Problems with the exclusion lists, for the guard test to assert on.

    The list is written by hand, so it needs something that reads the real
    key set back and complains. Returns human-readable findings; empty means
    every recorded key is accounted for and every exclusion is real.
    """
    findings = []
    stale = sorted(set(GLOBAL_EXCLUSIONS) - recorded_keys)
    if stale:
        findings.append(
            f"global exclusions name inputs no build records: {stale}")
    orphaned = sorted(
        recorded_keys - set(GLOBAL_EXCLUSIONS) - consumed_by_any_stage)
    if orphaned:
        findings.append(
            "recorded inputs that no stage consumes and that are not "
            f"globally excluded: {orphaned}. They are still hashed into every "
            "artifact's identity, which is safe but over-strict; either give "
            "them to the stage that reads them or exclude them by name.")
    undocumented = sorted(key for key, reason in GLOBAL_EXCLUSIONS.items()
                          if not reason.strip())
    if undocumented:
        findings.append(f"global exclusions without a reason: {undocumented}")
    return findings
