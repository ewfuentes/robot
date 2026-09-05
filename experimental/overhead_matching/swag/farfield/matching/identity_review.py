"""Publish immutable human identity decisions over one exact matching artifact.

The viewer writes a typed draft whose candidate rows are copied from the
matching artifact. A reviewer fills decisions in that draft; this command
validates every decision against the same immutable matching identity and
publishes it transactionally. Stale drafts and off-candidate landmark ids fail
closed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    artifact,
    paths as paths_lib,
    publication,
    provenance,
)


IDENTITY_REVIEW_KIND = "identity_reviews"
REVIEW_NAME = "identity_review.json"
DRAFT_NAME = "identity_review_draft.json"
DRAFT_SCHEMA = "farfield.identity_review_draft/v1"
REVIEW_SCHEMA = "farfield.identity_review/v1"
GENERATOR = ("//experimental/overhead_matching/swag/farfield/"
             "matching:identity_review")
_DRAFT_KEYS = {"schema", "source_matching", "rows"}
_DRAFT_ROW_KEYS = {
    "tracklet_id", "candidate_landmark_ids", "decision", "landmark_ids",
    "reviewer", "timestamp", "notes",
}
_REVIEW_KEYS = {"schema", "source_matching", "decisions"}
_DECISION_KEYS = {
    "tracklet_id", "decision", "landmark_ids", "reviewer", "timestamp",
    "notes",
}
_VERDICTS = {"confirmed", "rejected", "ambiguous"}


class IdentityReviewError(ValueError):
    """An identity review is malformed, stale, or out of scope."""


@dataclass(frozen=True)
class IdentityDecision:
    tracklet_id: str
    decision: str
    landmark_ids: tuple[str, ...]
    reviewer: str
    timestamp: str
    notes: str


@dataclass(frozen=True)
class IdentityReview:
    source_matching: artifact.ArtifactRef
    decisions: tuple[IdentityDecision, ...]


def _exact_keys(value, expected, what):
    actual = set(value) if isinstance(value, dict) else set()
    if not isinstance(value, dict) or actual != set(expected):
        raise IdentityReviewError(
            f"{what} must have exact keys {sorted(expected)}; "
            f"missing={sorted(set(expected) - actual)}, "
            f"unknown={sorted(actual - set(expected))}")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IdentityReviewError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(
                stream, object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    IdentityReviewError(
                        f"non-finite JSON constant {value!r}")))
    except IdentityReviewError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IdentityReviewError(f"cannot read {path}: {error}") from error


def matching_candidates(matching_dir: Path):
    """Return exact matching ref and tracklet -> published candidate ids."""
    matching_dir = Path(matching_dir)
    ref = artifact.open_artifact(
        matching_dir, expected_kind=paths_lib.LANDMARK_MATCHES)
    manifest = artifact.load_manifest(matching_dir)
    n_expected = manifest.config.get("n_expected")
    n_successful = manifest.config.get("n_successful")
    if (manifest.config.get("phase") != "canonical_results"
            or manifest.config.get("coverage") != "complete"
            or type(n_expected) is not int or n_expected < 1
            or type(n_successful) is not int
            or n_successful != n_expected):
        raise IdentityReviewError(
            "identity review requires a complete canonical matching artifact")
    document = _load_json(matching_dir / "matches.json")
    if not isinstance(document, dict):
        raise IdentityReviewError("matching matches.json must be an object")
    candidates = {}
    for tracklet_id, entry in document.items():
        if not isinstance(tracklet_id, str) or not tracklet_id:
            raise IdentityReviewError("matching contains an empty tracklet id")
        rows = entry.get("matches") if isinstance(entry, dict) else None
        if not isinstance(rows, list):
            raise IdentityReviewError(
                f"matching tracklet {tracklet_id!r} has no matches list")
        ids = []
        for row in rows:
            landmark_id = row.get("landmark_id") if isinstance(row, dict) \
                else None
            if not isinstance(landmark_id, str) or not landmark_id:
                raise IdentityReviewError(
                    f"matching tracklet {tracklet_id!r} has an invalid "
                    "landmark id")
            ids.append(landmark_id)
        if len(ids) != len(set(ids)):
            raise IdentityReviewError(
                f"matching tracklet {tracklet_id!r} repeats a candidate id")
        candidates[tracklet_id] = tuple(sorted(ids))
    n_tracklets_expected = manifest.config.get("n_tracklets_expected")
    n_tracklets_successful = manifest.config.get("n_tracklets_successful")
    if (type(n_tracklets_expected) is not int
            or type(n_tracklets_successful) is not int
            or n_tracklets_expected < 1
            or n_tracklets_successful != n_tracklets_expected
            or len(candidates) != n_tracklets_expected):
        raise IdentityReviewError(
            "matching manifest does not attest complete tracklet coverage")
    return ref, candidates


def draft_document(matching_ref: artifact.ArtifactRef,
                   candidates: dict[str, tuple[str, ...]]) -> dict:
    """Create an exact, editable review form; None means not reviewed yet."""
    return {
        "schema": DRAFT_SCHEMA,
        "source_matching": matching_ref.to_dict(),
        "rows": [{
            "tracklet_id": tracklet_id,
            "candidate_landmark_ids": list(candidates[tracklet_id]),
            "decision": None,
            "landmark_ids": [],
            "reviewer": "",
            "timestamp": "",
            "notes": "",
        } for tracklet_id in sorted(candidates)],
    }


def write_draft(path: Path, matching_dir: Path) -> None:
    reference, candidates = matching_candidates(matching_dir)
    artifact.atomic_write_json(
        Path(path), draft_document(reference, candidates))


def _source_ref(value, expected: artifact.ArtifactRef):
    try:
        source = artifact.ArtifactRef.from_dict(value)
    except artifact.ArtifactError as error:
        raise IdentityReviewError(
            f"invalid source_matching identity: {error}") from error
    if source != expected:
        raise IdentityReviewError(
            "identity review is stale: source_matching does not exactly "
            "match the supplied landmark_matches artifact")
    return source


def _timestamp(value, what):
    if not isinstance(value, str) or not value:
        raise IdentityReviewError(f"{what} must be a non-empty timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise IdentityReviewError(f"{what} is not ISO-8601: {value!r}") \
            from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise IdentityReviewError(f"{what} must include a timezone")
    return value


def _decision(value, *, candidates, what):
    _exact_keys(value, _DECISION_KEYS, what)
    tracklet_id = value["tracklet_id"]
    if tracklet_id not in candidates:
        raise IdentityReviewError(
            f"{what} names unknown tracklet {tracklet_id!r}")
    verdict = value["decision"]
    if verdict not in _VERDICTS:
        raise IdentityReviewError(
            f"{what} decision must be one of {sorted(_VERDICTS)}")
    ids = value["landmark_ids"]
    if (not isinstance(ids, list)
            or not all(isinstance(item, str) and item for item in ids)
            or len(ids) != len(set(ids))):
        raise IdentityReviewError(
            f"{what} landmark_ids must be a unique string list")
    unknown = set(ids) - set(candidates[tracklet_id])
    if unknown:
        raise IdentityReviewError(
            f"{what} names a landmark outside the exact machine candidates: "
            f"{sorted(unknown)[0]!r}")
    if verdict in ("confirmed", "rejected") and not ids:
        raise IdentityReviewError(
            f"{what} {verdict} decision requires landmark_ids")
    if verdict == "ambiguous" and ids:
        raise IdentityReviewError(
            f"{what} ambiguous decision must not name landmark_ids")
    reviewer = value["reviewer"]
    notes = value["notes"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise IdentityReviewError(f"{what} reviewer is empty")
    if not isinstance(notes, str):
        raise IdentityReviewError(f"{what} notes must be a string")
    return IdentityDecision(
        tracklet_id=tracklet_id,
        decision=verdict,
        landmark_ids=tuple(ids),
        reviewer=reviewer.strip(),
        timestamp=_timestamp(value["timestamp"], f"{what}.timestamp"),
        notes=notes,
    )


def parse_input(value, *, matching_ref, candidates) -> IdentityReview:
    """Parse either the editable draft or the final strict document."""
    if not isinstance(value, dict):
        raise IdentityReviewError("identity review input must be an object")
    schema = value.get("schema")
    if schema == DRAFT_SCHEMA:
        _exact_keys(value, _DRAFT_KEYS, "identity review draft")
        source = _source_ref(value["source_matching"], matching_ref)
        rows = value["rows"]
        if not isinstance(rows, list):
            raise IdentityReviewError("identity review draft rows must be a list")
        decisions = []
        seen = set()
        for index, row in enumerate(rows):
            _exact_keys(row, _DRAFT_ROW_KEYS, f"rows[{index}]")
            tracklet_id = row["tracklet_id"]
            if tracklet_id in seen:
                raise IdentityReviewError(
                    f"identity review repeats tracklet {tracklet_id!r}")
            seen.add(tracklet_id)
            if tracklet_id not in candidates:
                raise IdentityReviewError(
                    f"draft names unknown tracklet {tracklet_id!r}")
            if row["candidate_landmark_ids"] != list(candidates[tracklet_id]):
                raise IdentityReviewError(
                    f"draft candidates for {tracklet_id!r} are stale")
            if row["decision"] is None:
                if (row["landmark_ids"] or row["reviewer"]
                        or row["timestamp"] or row["notes"]):
                    raise IdentityReviewError(
                        f"unreviewed row {tracklet_id!r} has filled fields")
                continue
            decision_value = {key: row[key] for key in _DECISION_KEYS}
            decisions.append(_decision(
                decision_value, candidates=candidates,
                what=f"rows[{index}]"))
    elif schema == REVIEW_SCHEMA:
        _exact_keys(value, _REVIEW_KEYS, "identity review")
        source = _source_ref(value["source_matching"], matching_ref)
        raw_decisions = value["decisions"]
        if not isinstance(raw_decisions, list):
            raise IdentityReviewError("identity review decisions must be a list")
        decisions = [_decision(
            item, candidates=candidates, what=f"decisions[{index}]")
                     for index, item in enumerate(raw_decisions)]
    else:
        raise IdentityReviewError(
            f"unsupported identity review schema {schema!r}")
    tracklet_ids = [decision.tracklet_id for decision in decisions]
    if len(tracklet_ids) != len(set(tracklet_ids)):
        raise IdentityReviewError("identity review repeats a tracklet decision")
    decisions.sort(key=lambda item: item.tracklet_id)
    return IdentityReview(source, tuple(decisions))


def review_document(review: IdentityReview) -> dict:
    return {
        "schema": REVIEW_SCHEMA,
        "source_matching": review.source_matching.to_dict(),
        "decisions": [{
            "tracklet_id": item.tracklet_id,
            "decision": item.decision,
            "landmark_ids": list(item.landmark_ids),
            "reviewer": item.reviewer,
            "timestamp": item.timestamp,
            "notes": item.notes,
        } for item in review.decisions],
    }


def publish(*, dataset: str, matching_dir: Path, input_json: Path,
            output_dir: Path, version: str) -> artifact.ArtifactRef:
    matching_ref, candidates = matching_candidates(matching_dir)
    if matching_ref.dataset != dataset:
        raise IdentityReviewError(
            f"matching artifact belongs to {matching_ref.dataset!r}, not "
            f"{dataset!r}")
    review = parse_input(
        _load_json(input_json), matching_ref=matching_ref,
        candidates=candidates)
    document = review_document(review)
    with publication.published_artifact(
            output_dir, kind=IDENTITY_REVIEW_KIND, dataset=dataset,
            version=version, generator=GENERATOR,
            git_commit=provenance.git_commit(), arguments=sys.argv,
            upstreams=(matching_ref,), config={
                "schema": REVIEW_SCHEMA,
                "precedence_policy": "human_identity_over_machine_v1",
                "n_decisions": len(review.decisions),
                "source_matching": matching_ref.to_dict(),
            }, declared_outputs=(REVIEW_NAME,)) as builder:
        artifact.atomic_write_json(builder.output_path(REVIEW_NAME), document)
    assert builder.artifact_ref is not None
    return builder.artifact_ref


def load(review_dir: Path, *, expected_matching_ref: artifact.ArtifactRef,
         matching_dir: Path | None = None):
    review_ref = artifact.open_artifact(
        review_dir, expected_kind=IDENTITY_REVIEW_KIND,
        expected_dataset=expected_matching_ref.dataset)
    manifest = artifact.load_manifest(review_dir)
    if manifest.upstreams != (expected_matching_ref,):
        raise IdentityReviewError(
            "identity review is not bound to the exact matching artifact")
    actual_matching_ref, candidates = matching_candidates(
        matching_dir or expected_matching_ref.path)
    if actual_matching_ref != expected_matching_ref:
        raise IdentityReviewError(
            "supplied matching directory does not have the expected identity")
    review = parse_input(
        _load_json(Path(review_dir) / REVIEW_NAME),
        matching_ref=expected_matching_ref, candidates=candidates)
    expected_config = {
        "schema": REVIEW_SCHEMA,
        "precedence_policy": "human_identity_over_machine_v1",
        "n_decisions": len(review.decisions),
        "source_matching": expected_matching_ref.to_dict(),
    }
    if manifest.config != expected_config:
        raise IdentityReviewError(
            "identity review manifest disagrees with its typed decision file")
    return review_ref, review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--matching_dir", type=Path, required=True)
    parser.add_argument("--input_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    try:
        reference = publish(
            dataset=args.dataset, matching_dir=args.matching_dir,
            input_json=args.input_json, output_dir=args.output_dir,
            version=args.version)
    except (IdentityReviewError, artifact.ArtifactError,
            publication.PublicationValidationError, OSError) as error:
        parser.error(str(error))
    print(f"published immutable identity review: {reference.path}")


if __name__ == "__main__":
    main()
