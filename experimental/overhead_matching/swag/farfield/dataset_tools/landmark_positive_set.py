"""Freeze a completed matching artifact into a catalog-trimming recall guard.

The matching stage is the authority for correspondence. This tool consumes
only a complete, typed ``LANDMARK_MATCHES`` artifact and copies its final
``matches.json`` decisions into a small, immutable-identity JSON document.
It never reconstructs decisions from provider requests or retry results.

The matching manifest supplies the exact ``CATALOGS`` upstream. The final
``signatures.json`` table is checked against every matched landmark id before
publication, so a positive set cannot silently preserve a stale or malformed
correspondence.

Example::

    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:landmark_positive_set -- \\
        --matching_dir <completed-landmark-matches-artifact> \\
        --output <positive-set-v2.json>
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield import provenance


CATALOG_PAYLOAD = "catalog.feather"
MATCHES_PAYLOAD = "matches.json"
SIGNATURES_PAYLOAD = "signatures.json"
MATCHING_OUTPUTS = tuple(sorted((
    "canonical_results.jsonl",
    "compatibility.json",
    MATCHES_PAYLOAD,
    "request_set.json",
    "requests.jsonl",
    "settings.json",
    SIGNATURES_PAYLOAD,
)))
POSITIVE_SET_SCHEMA = "farfield.landmark_positive_set.v2"
MATCH_KEYS = frozenset({
    "landmark_id", "per_call_candidate_scores", "aggregate_confidence",
    "aggregation_rule", "match_type", "signature_id", "signature_display",
})
POSITIVE_KEYS = frozenset({
    "tracklet_id", "landmark_id", "signature_id", "signature_display",
    "match_type", "aggregate_confidence",
})
SIGNATURE_KEYS = frozenset({
    "canonical_tags", "display_label", "landmark_ids",
})
CANDIDATE_AGGREGATION_RULE = "maximum_per_landmark_across_calls_v1"
DOCUMENT_KEYS = frozenset({
    "schema", "generator", "git_commit", "matching", "catalog",
    "n_tracklets", "n_positives", "positives",
})


class PositiveSetError(ValueError):
    """Raised when a matching artifact or positive-set document is invalid."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise PositiveSetError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path, what: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PositiveSetError(f"{what} is not a regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PositiveSetError(
                    f"{what} contains non-finite JSON constant {token!r}")),
        )
    except PositiveSetError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PositiveSetError(f"cannot decode {what} at {path}: {exc}") \
            from exc
    if not isinstance(value, dict):
        raise PositiveSetError(f"{what} must be a JSON object")
    return value


def _count(value: Any, field: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise PositiveSetError(f"{field} must be a {qualifier} integer")
    return value


def _exact_keys(value: dict, expected: frozenset[str], what: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise PositiveSetError(
            f"invalid {what} keys: missing={missing}, unknown={unknown}")


def open_catalog_artifact(
        catalog_dir: Path, *, expected_dataset: str | None = None,
) -> tuple[artifact.ArtifactRef, Path]:
    """Validate one published CATALOGS artifact and return its payload."""
    catalog_dir = Path(catalog_dir)
    ref = artifact.open_artifact(
        catalog_dir,
        expected_kind=paths_lib.CATALOGS,
        expected_dataset=expected_dataset,
    )
    manifest = artifact.load_manifest(catalog_dir)
    if manifest.declared_outputs != (CATALOG_PAYLOAD,):
        raise artifact.ArtifactValidationError(
            "CATALOGS must declare exactly catalog.feather; found "
            f"{list(manifest.declared_outputs)}")
    return ref, catalog_dir / CATALOG_PAYLOAD


def format_signature(tags: dict) -> str:
    """Return the matcher's display label (never its machine identity)."""
    return "; ".join(f"{key}={value}" for key, value in sorted(tags.items()))


def signature_id(tags: dict) -> str:
    """Return the matcher's collision-resistant canonical signature identity."""
    return f"sha256:{artifact.sha256_json(tags)}"


def _validate_signatures(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not value:
        raise PositiveSetError("matching signatures.json must not be empty")
    result = {}
    owner = {}
    for canonical_id, entry in value.items():
        if not isinstance(canonical_id, str) or not canonical_id:
            raise PositiveSetError("matching contains an empty signature id")
        if not isinstance(entry, dict):
            raise PositiveSetError(
                f"signature {canonical_id!r} metadata must be an object")
        _exact_keys(entry, SIGNATURE_KEYS, f"signature {canonical_id!r}")
        tags = entry["canonical_tags"]
        if (not isinstance(tags, dict) or not tags
                or any(not isinstance(key, str) or not key
                       or not isinstance(tag_value, str) or not tag_value
                       for key, tag_value in tags.items())):
            raise PositiveSetError(
                f"signature {canonical_id!r} canonical_tags must be a "
                "non-empty string map")
        expected_id = signature_id(tags)
        if canonical_id != expected_id:
            raise PositiveSetError(
                f"signature {canonical_id!r} digest does not match its "
                f"canonical_tags (expected {expected_id!r})")
        display_label = entry["display_label"]
        expected_display = format_signature(tags)
        if display_label != expected_display:
            raise PositiveSetError(
                f"signature {canonical_id!r} display label does not match "
                "its canonical_tags")
        landmark_ids = entry["landmark_ids"]
        if not isinstance(landmark_ids, list) or not landmark_ids:
            raise PositiveSetError(
                f"signature {canonical_id!r} must name a non-empty landmark "
                "id list")
        if not all(isinstance(item, str) and item for item in landmark_ids):
            raise PositiveSetError(
                f"signature {canonical_id!r} contains an invalid landmark id")
        if len(landmark_ids) != len(set(landmark_ids)):
            raise PositiveSetError(
                f"signature {canonical_id!r} repeats a landmark id")
        for landmark_id in landmark_ids:
            previous = owner.setdefault(landmark_id, canonical_id)
            if previous != canonical_id:
                raise PositiveSetError(
                    f"landmark {landmark_id!r} belongs to multiple signatures")
        result[canonical_id] = {
            "canonical_tags": dict(tags),
            "display_label": display_label,
            "landmark_ids": tuple(landmark_ids),
        }
    return result


def _finite_confidence(value: Any, what: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0):
        raise PositiveSetError(f"{what} must be finite and in [0, 1]")
    return float(value)


def _validate_matches(
        value: dict[str, Any], signatures: dict[str, dict[str, Any]],
        *, n_tracklets: int,
) -> dict[str, tuple[dict[str, Any], ...]]:
    if len(value) != n_tracklets:
        raise PositiveSetError(
            "matches.json tracklet count does not match the completed "
            f"manifest: expected {n_tracklets}, found {len(value)}")
    result = {}
    for tracklet_id, record in value.items():
        if not isinstance(tracklet_id, str) or not tracklet_id:
            raise PositiveSetError("matches.json contains an empty tracklet id")
        if not isinstance(record, dict) or not isinstance(
                record.get("matches"), list):
            raise PositiveSetError(
                f"tracklet {tracklet_id!r} lacks a matches list")
        matches = []
        seen_ids = set()
        for index, match in enumerate(record["matches"]):
            if not isinstance(match, dict):
                raise PositiveSetError(
                    f"tracklet {tracklet_id!r} match {index} is not an object")
            _exact_keys(match, MATCH_KEYS,
                        f"tracklet {tracklet_id!r} match {index}")
            landmark_id = match["landmark_id"]
            canonical_id = match["signature_id"]
            match_type = match["match_type"]
            if not isinstance(landmark_id, str) or not landmark_id:
                raise PositiveSetError(
                    f"tracklet {tracklet_id!r} match {index} has an invalid "
                    "landmark id")
            if landmark_id in seen_ids:
                raise PositiveSetError(
                    f"tracklet {tracklet_id!r} repeats landmark {landmark_id!r}")
            seen_ids.add(landmark_id)
            if (not isinstance(canonical_id, str)
                    or canonical_id not in signatures):
                raise PositiveSetError(
                    f"tracklet {tracklet_id!r} references unknown signature "
                    f"{canonical_id!r}")
            signature = signatures[canonical_id]
            if landmark_id not in signature["landmark_ids"]:
                raise PositiveSetError(
                    f"landmark {landmark_id!r} is not bound to signature "
                    f"{canonical_id!r}")
            if match["signature_display"] != signature["display_label"]:
                raise PositiveSetError(
                    f"tracklet {tracklet_id!r} match {index} signature display "
                    "does not match signatures.json")
            if match_type not in ("instance", "category"):
                raise PositiveSetError(
                    f"tracklet {tracklet_id!r} has invalid match_type "
                    f"{match_type!r}")
            per_call_scores = match["per_call_candidate_scores"]
            if not isinstance(per_call_scores, list) or not per_call_scores:
                raise PositiveSetError(
                    f"tracklet {tracklet_id!r} match {index} must record "
                    "per-call candidate scores")
            scores = [
                _finite_confidence(
                    score,
                    f"tracklet {tracklet_id!r} match {index} per-call score")
                for score in per_call_scores
            ]
            aggregate_confidence = _finite_confidence(
                match["aggregate_confidence"],
                f"tracklet {tracklet_id!r} aggregate confidence")
            if aggregate_confidence != max(scores):
                raise PositiveSetError(
                    f"tracklet {tracklet_id!r} match {index} aggregate "
                    "confidence is not the maximum per-call score")
            if match["aggregation_rule"] != CANDIDATE_AGGREGATION_RULE:
                raise PositiveSetError(
                    f"tracklet {tracklet_id!r} match {index} has an unknown "
                    "candidate aggregation rule")
            matches.append({
                "tracklet_id": tracklet_id,
                "landmark_id": landmark_id,
                "signature_id": canonical_id,
                "signature_display": signature["display_label"],
                "match_type": match_type,
                "aggregate_confidence": aggregate_confidence,
            })
        n_landmarks = _count(record.get("n_landmarks"),
                             f"tracklet {tracklet_id!r} n_landmarks")
        if n_landmarks != len(matches):
            raise PositiveSetError(
                f"tracklet {tracklet_id!r} n_landmarks does not match its "
                "matches list")
        n_signatures = _count(record.get("n_signatures"),
                              f"tracklet {tracklet_id!r} n_signatures")
        if n_signatures != len({item["signature_id"] for item in matches}):
            raise PositiveSetError(
                f"tracklet {tracklet_id!r} n_signatures is inconsistent")
        result[tracklet_id] = tuple(matches)
    return result


def open_matching_artifact(
        matching_dir: Path,
        *, expected_catalog_ref: artifact.ArtifactRef | None = None,
) -> tuple[artifact.ArtifactRef, artifact.ArtifactRef,
           dict[str, tuple[dict[str, Any], ...]], dict[str, dict[str, Any]]]:
    """Load a complete typed matching artifact and validate its final join."""
    matching_dir = Path(matching_dir)
    expected_dataset = (expected_catalog_ref.dataset
                        if expected_catalog_ref is not None else None)
    matching_ref = artifact.open_artifact(
        matching_dir,
        expected_kind=paths_lib.LANDMARK_MATCHES,
        expected_dataset=expected_dataset,
    )
    manifest = artifact.load_manifest(matching_dir)
    if manifest.declared_outputs != MATCHING_OUTPUTS:
        raise PositiveSetError(
            "LANDMARK_MATCHES must declare exactly the canonical final "
            f"outputs; found {list(manifest.declared_outputs)}")
    if manifest.config.get("phase") != "canonical_results":
        raise PositiveSetError(
            "matching manifest must attest phase='canonical_results'")
    if manifest.config.get("coverage") != "complete":
        raise PositiveSetError(
            "matching manifest must attest coverage='complete'")
    n_expected = _count(manifest.config.get("n_expected"),
                        "matching n_expected", positive=True)
    n_successful = _count(manifest.config.get("n_successful"),
                          "matching n_successful")
    if n_successful != n_expected:
        raise PositiveSetError(
            "matching does not have one successful result per expected request")
    n_tracklets = _count(manifest.config.get("n_tracklets_expected"),
                         "matching n_tracklets_expected", positive=True)
    n_tracklets_successful = _count(
        manifest.config.get("n_tracklets_successful"),
        "matching n_tracklets_successful")
    if n_tracklets_successful != n_tracklets:
        raise PositiveSetError(
            "matching does not have one final record per expected tracklet")
    expected_upstream_kinds = (
        paths_lib.OBJECT_TRACKS,
        paths_lib.SEMANTIC_AUDITS,
        paths_lib.CATALOGS,
    )
    if tuple(ref.kind for ref in manifest.upstreams) != expected_upstream_kinds:
        raise PositiveSetError(
            "LANDMARK_MATCHES upstreams must be exactly object_tracks, "
            "semantic_audits, then catalogs")
    if any(ref.dataset != matching_ref.dataset for ref in manifest.upstreams):
        raise PositiveSetError(
            "matching artifact and all upstreams must name the same dataset")
    catalog_ref = manifest.upstreams[-1]
    if expected_catalog_ref is not None and catalog_ref != expected_catalog_ref:
        raise PositiveSetError(
            "matching artifact is bound to a different catalog: expected "
            f"{expected_catalog_ref.to_dict()}, found {catalog_ref.to_dict()}")
    signatures = _validate_signatures(_load_json_object(
        matching_dir / SIGNATURES_PAYLOAD, "matching signatures.json"))
    matches = _validate_matches(
        _load_json_object(matching_dir / MATCHES_PAYLOAD,
                          "matching matches.json"),
        signatures,
        n_tracklets=n_tracklets,
    )
    return matching_ref, catalog_ref, matches, signatures


def build(matching_dir: Path) -> dict[str, Any]:
    """Build a schema-v2 guard from final, completely covered matches."""
    matching_ref, catalog_ref, matches, _ = open_matching_artifact(matching_dir)
    positives = [
        match
        for tracklet_id in sorted(matches)
        for match in sorted(
            matches[tracklet_id],
            key=lambda item: (item["landmark_id"], item["signature_id"]),
        )
    ]
    return {
        "schema": POSITIVE_SET_SCHEMA,
        "generator": "farfield/dataset_tools/landmark_positive_set.py",
        "git_commit": provenance.git_commit(),
        "matching": matching_ref.to_dict(),
        "catalog": catalog_ref.to_dict(),
        "n_tracklets": len(matches),
        "n_positives": len(positives),
        "positives": positives,
    }


def validate_positive_set(
        positive_set: dict[str, Any], source: Path | str,
) -> tuple[artifact.ArtifactRef, artifact.ArtifactRef]:
    """Validate schema-v2 structure and return matching/catalog identities."""
    source = Path(source)
    if not isinstance(positive_set, dict):
        raise PositiveSetError(f"positive set {source} must be a JSON object")
    _exact_keys(positive_set, DOCUMENT_KEYS, f"positive set {source}")
    if positive_set["schema"] != POSITIVE_SET_SCHEMA:
        raise PositiveSetError(
            f"positive set {source} must use schema {POSITIVE_SET_SCHEMA!r}")
    if (not isinstance(positive_set["generator"], str)
            or not positive_set["generator"]):
        raise PositiveSetError(f"positive set {source} has invalid generator")
    if (not isinstance(positive_set["git_commit"], str)
            or not positive_set["git_commit"]):
        raise PositiveSetError(f"positive set {source} has invalid git_commit")
    try:
        matching_ref = artifact.ArtifactRef.from_dict(positive_set["matching"])
        catalog_ref = artifact.ArtifactRef.from_dict(positive_set["catalog"])
    except (artifact.ArtifactError, TypeError) as exc:
        raise PositiveSetError(
            f"positive set {source} has an invalid exact ArtifactRef") from exc
    if matching_ref.kind != paths_lib.LANDMARK_MATCHES:
        raise PositiveSetError(
            f"positive set {source} matching ref is not LANDMARK_MATCHES")
    if catalog_ref.kind != paths_lib.CATALOGS:
        raise PositiveSetError(
            f"positive set {source} catalog ref is not CATALOGS")
    if matching_ref.dataset != catalog_ref.dataset:
        raise PositiveSetError(
            f"positive set {source} matching/catalog datasets differ")
    n_tracklets = _count(positive_set["n_tracklets"],
                         f"positive set {source} n_tracklets", positive=True)
    positives = positive_set["positives"]
    if not isinstance(positives, list):
        raise PositiveSetError(f"positive set {source} positives must be a list")
    n_positives = _count(positive_set["n_positives"],
                         f"positive set {source} n_positives")
    if n_positives != len(positives):
        raise PositiveSetError(
            f"positive set {source} n_positives is inconsistent")
    tracklet_ids = set()
    identities = set()
    for index, record in enumerate(positives):
        if not isinstance(record, dict):
            raise PositiveSetError(
                f"positive set {source} positive {index} is not an object")
        _exact_keys(record, POSITIVE_KEYS,
                    f"positive set {source} positive {index}")
        tracklet_id = record["tracklet_id"]
        landmark_id = record["landmark_id"]
        canonical_id = record["signature_id"]
        if not isinstance(tracklet_id, str) or not tracklet_id:
            raise PositiveSetError(
                f"positive set {source} positive {index} has invalid tracklet")
        if not isinstance(landmark_id, str) or not landmark_id:
            raise PositiveSetError(
                f"positive set {source} positive {index} has invalid landmark")
        if not isinstance(canonical_id, str) or not canonical_id:
            raise PositiveSetError(
                f"positive set {source} positive {index} has invalid signature "
                "id")
        if (not isinstance(record["signature_display"], str)
                or not record["signature_display"]):
            raise PositiveSetError(
                f"positive set {source} positive {index} has invalid signature "
                "display")
        if record["match_type"] not in ("instance", "category"):
            raise PositiveSetError(
                f"positive set {source} positive {index} has invalid "
                "match_type")
        _finite_confidence(
            record["aggregate_confidence"],
            f"positive set {source} positive {index} aggregate confidence")
        identity = (tracklet_id, landmark_id)
        if identity in identities:
            raise PositiveSetError(
                f"positive set {source} repeats {tracklet_id}/{landmark_id}")
        identities.add(identity)
        tracklet_ids.add(tracklet_id)
    if len(tracklet_ids) > n_tracklets:
        raise PositiveSetError(
            f"positive set {source} contains more tracklets than declared")
    return matching_ref, catalog_ref


def load_positive_set(
        source: Path | str,
) -> tuple[dict[str, Any], artifact.ArtifactRef, artifact.ArtifactRef]:
    """Strictly load one schema-v2 positive-set JSON document."""
    source = Path(source)
    document = _load_json_object(source, "positive set")
    matching_ref, catalog_ref = validate_positive_set(document, source)
    return document, matching_ref, catalog_ref


def recall(positive_set: dict,
           surviving_signatures: set[str]) -> tuple[float, list[dict]]:
    """Fraction of distinct positive signatures with a surviving landmark."""
    signatures = {
        record["signature_id"] for record in positive_set["positives"]
    }
    if not signatures:
        return 1.0, []
    lost = signatures - surviving_signatures
    detail = [record for record in positive_set["positives"]
              if record["signature_id"] in lost]
    return (len(signatures) - len(lost)) / len(signatures), detail


def main(matching_dir: Path, output: Path) -> dict[str, Any]:
    positive_set = build(matching_dir)
    artifact.atomic_create_json(output, positive_set)
    print(f"Wrote {output}")
    return positive_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matching_dir", required=True, type=Path,
        help="completed typed LANDMARK_MATCHES artifact")
    parser.add_argument(
        "--output", required=True, type=Path,
        help="schema-v2 positive-set JSON to write atomically")
    args = parser.parse_args()
    try:
        main(args.matching_dir, args.output)
    except (artifact.ArtifactError, PositiveSetError) as exc:
        parser.error(str(exc))
