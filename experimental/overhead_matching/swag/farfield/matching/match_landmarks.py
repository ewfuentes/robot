"""Match tracklets against the whole map, position-free, via chunked LLM calls.

Set 1 is built per canonical accepted tracklet from its validated semantic
audit. Audit membership is the support gate: a track that was never audited
has no canonical semantics and gets no Set 1 entry, and `verdict: drop` is
excluded. Output is a `CompatibilityTable` per tracklet for the bearing-only filter
(farfield.localization.structs).

**No observer-position information is used anywhere.** Gating candidates with
the vessel's GPS position would leak the localization answer. Set 2 is the
complete bound catalog. Catalog-internal containment may add identity context
such as an apartment complex's name to one of its buildings, but it never
selects candidates relative to the observer.

Shape of the work: the map is far larger than a prompt, so it is split into
signature chunks and the tracklets into small batches, and every batch is
asked about every chunk. Distinct *signatures* are matched rather than rows -
identical tag bundles are things the model cannot tell apart by construction -
and a matched signature expands to every landmark carrying it.

The mutable provider boundary lives in a sibling ``<output_dir>.llm-work``
directory. It is not a scientific artifact and is never consumed downstream.
Each provider response is retained there as one atomically created immutable
file under ``attempts/``; there is no shared append log.
The requested ``output_dir`` is published atomically as one typed
``landmark_matches`` artifact only after every expected request has exactly one
schema-valid success. It contains:

  request_set.json      immutable request snapshot
  requests.jsonl        exact provider inputs
  canonical_results.jsonl complete, unique, validated results
  signatures.json       signature -> [landmark_id, ...] expansion table
  matches.json          per tracklet: matched landmark_ids, confidence, type
  compatibility.json    list[structs.CompatibilityTable], msgspec-encoded so
                        the localization loader decodes it directly
  settings.json         the provenance reference for this matching run
  manifest.json         typed artifact identity and upstream/config contract

Run:
  bazel run //experimental/overhead_matching/swag/farfield/matching:match_landmarks -- \\
      --tracks_dir ... --audit_dir ... --catalog_dir ... --output_dir ... \
      --build_config .../build_config.json --orchestration_config_digest ... \
      --build_only
"""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import msgspec
import shapely

from common.python.serialization import msgspec_enc_hook
from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import artifact_recipe
from experimental.overhead_matching.swag.farfield import configured_lane
from experimental.overhead_matching.swag.farfield import build_config
from experimental.overhead_matching.swag.farfield import llm_lifecycle
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield import publication
from experimental.overhead_matching.swag.farfield import provenance
from experimental.overhead_matching.swag.farfield.calibration import audit_io
from experimental.overhead_matching.swag.farfield.catalog import (
    catalog as catalog_lib,
    lineage as catalog_lineage,
    schema,
)
from experimental.overhead_matching.swag.farfield.extraction import (
    vertex_batch_manager as vbm,
)
from experimental.overhead_matching.swag.farfield.localization import structs
from experimental.overhead_matching.swag.farfield.tracking import tracklets

SYSTEM_PROMPT = """You are a landmark matching expert. Given two sets of OpenStreetMap-style tag
bundles, identify which landmarks in Set 1 (observed in images) are the same physical object as a
landmark in Set 2 (a map database). Both use key=value notation.

Set 2 is one arbitrary slice of a much larger map. Most of it is irrelevant to any given Set 1
landmark, and for most Set 1 landmarks the correct answer is not in this slice at all. Returning no
match is the expected outcome, not a failure. Never settle for the closest thing present.

Match the observed object ITSELF, not what it stands on, contains, or sits beside. A structure on an
island is not the island; a light on a pier is not the pier. When only the container, the contents,
or a neighbour appears in Set 2, that is a no match - not a weaker match.

A Set 1 landmark may match several Set 2 entries when the map holds more than one row for the same
physical object. Several rows for one object is a real multiple match. Several related-but-distinct
objects is not.

Each Set 1 entry spans several lines. It is the output of a prior review stage that examined every
detection of the object across many frames alongside the images, and reported both what it concluded
and how sure it was. Every number below is that stage's belief, not a measurement:
  tags        - candidate tags, each with its belief (0-1). A low weight is a possibility that stage
                could not rule out, not a claim.
  names       - candidate names, each with a belief and a basis. "both" means the review stage could
                also corroborate the name from the images it was shown. "reported_by_detections"
                means it could not, which does not eliminate the name as a possibility.
  kind/extent - what sort of object it is, and whether it is point-like or spatially extended.
  description - one sentence on intrinsic appearance. Set 2 rows sometimes carry free text as well;
                compare them directly.
  features    - distinguishing visual details.
  unresolved  - what that stage could not settle. Read it as a warning.

For each match report:
  - match_type:
      "instance" - this exact physical object, identified uniquely by a matching name or by a tag
        combination no other candidate shares.
      "category" - the right kind of object, but the tags cannot say WHICH one.
  - confidence 0.0-1.0 - how sure you are that this Set 2 entry is the same physical object as the
    Set 1 landmark. This is about the match, not about how distinctive the landmark is.

Also report, once per Set 1 landmark:
  - no_match_confidence 0.0-1.0 - how sure you are that none of the Set 2 entries SHOWN HERE is this
    landmark. It is a statement about this slice only, not about the map as a whole, and it will
    usually be high. Report it honestly rather than lowering it to justify a weak match.
  - uniqueness_score 1-5 - how distinctive the SET 1 landmark is on its own, independent of any match
    and of your confidence: 1 generic (building=yes), 3 moderately specific (man_made=water_tower),
    5 unmistakable (a named lighthouse).

Not evidence against a match: small numeric differences (height 40 vs 45 - the observer is often
off); different tag specificity for the same thing (man_made=tower vs man_made=water_tower); one
name being a longer or shorter variant of another; tags present on one side only."""


SCHEMA = {
    "type": "object", "required": ["matches"],
    "additionalProperties": False,
    "properties": {"matches": {"type": "array", "items": {
        "type": "object",
        "required": ["set_1_id", "set_2_matches", "no_match_confidence",
                     "uniqueness_score"],
        "additionalProperties": False,
        "properties": {
            "set_1_id": {"type": "integer", "minimum": 0},
            "set_2_matches": {"type": "array", "items": {
                "type": "object",
                "required": ["set_2_id", "match_type", "confidence"],
                "additionalProperties": False,
                "properties": {
                    "set_2_id": {"type": "integer", "minimum": 0},
                    "match_type": {"type": "string",
                                   "enum": ["instance", "category"]},
                    "confidence": {"type": "number", "minimum": 0,
                                   "maximum": 1}}}},
            "no_match_confidence": {"type": "number", "minimum": 0,
                                    "maximum": 1},
            "uniqueness_score": {"type": "integer", "minimum": 1,
                                 "maximum": 5}}}}}}

ATTEMPTS_DIR_NAME = llm_lifecycle.ATTEMPTS_DIR_NAME
TRANSPORT_RESULTS_NAME = "transport_results.jsonl"
SIGNATURES_NAME = "signatures.json"
WORK_SNAPSHOT_NAME = "matching_snapshot.json"
WORK_SNAPSHOT_SCHEMA = "farfield.matching_work_snapshot/v3"
SETTINGS_NAME = "settings.json"
MATCHES_NAME = "matches.json"
COMPATIBILITY_NAME = "compatibility.json"
FINAL_OUTPUTS = (
    WORK_SNAPSHOT_NAME,
    llm_lifecycle.REQUEST_SET_NAME,
    llm_lifecycle.REQUESTS_NAME,
    llm_lifecycle.CANONICAL_RESULTS_NAME,
    SIGNATURES_NAME,
    SETTINGS_NAME,
    MATCHES_NAME,
    COMPATIBILITY_NAME,
)
GENERATOR = ("//experimental/overhead_matching/swag/farfield/"
             "matching:match_landmarks")

MATCHING_CONFIG_KEYS = (
    "matching.model",
    "matching.query_batch",
    "matching.chunk_size",
    "matching.thinking_level",
    "matching.confidence_floor",
    "matching.instance_max_rows",
    "execution.llm_transport",
    "execution.batch_gcs_prefix",
    "execution.approve_cost",
    "cost.limit_usd",
    "artifacts.landmark_matches_version",
)

# The affine seam the design doc SS6 specifies for an uncalibrated matcher:
# clipped log odds. Mirrors structs.CompatibilityTable's clip contract.
DEFAULT_CLIP = 4.0
SCORE_CONTRACT = {
    "per_call_candidate_score_semantics":
        "self_reported_model_score_0_to_1",
    "candidate_aggregation_rule": "maximum_per_landmark_across_calls_v1",
    "uniqueness_aggregation_rule": "arithmetic_mean_across_calls_v1",
    "score_policy": "clipped_logit_compat_v1",
    "score_calibration": "uncalibrated",
}


def signature_display(tags: dict) -> str:
    return "; ".join(f"{k}={v}" for k, v in sorted(tags.items()))


def signature(tags: dict) -> str:
    """Collision-resistant identity for one canonical pruned tag bundle."""
    return f"sha256:{artifact.sha256_json(tags)}"


COMPLEX_LANDUSE = frozenset(
    {"residential", "commercial", "retail", "industrial"})


def enclosing_complex_names(frame, records) -> dict[int, str]:
    """Map building rows to the smallest named complex containing them.

    East Asian apartment mapping commonly puts the complex name on a landuse
    polygon while individual towers carry only unit labels such as ``101``.
    The inherited name is matching context, not a persisted catalog mutation.
    """
    geometries = list(frame.geometry.values)
    complex_rows = []
    for index, (tags, geometry) in enumerate(zip(records, geometries)):
        if (tags.get("name")
                and tags.get("landuse") in COMPLEX_LANDUSE
                and geometry is not None
                and geometry.geom_type in ("Polygon", "MultiPolygon")):
            complex_rows.append(index)
    if not complex_rows:
        return {}

    tree = shapely.STRtree([geometries[index] for index in complex_rows])
    result = {}
    for index, (tags, geometry) in enumerate(zip(records, geometries)):
        if "building" not in tags or geometry is None:
            continue
        centroid = shapely.centroid(geometry)
        if centroid is None or shapely.is_empty(centroid):
            continue
        hits = tree.query(centroid, predicate="within")
        candidates = [complex_rows[int(hit)] for hit in hits
                      if complex_rows[int(hit)] != index]
        if not candidates:
            continue
        owner = min(candidates,
                    key=lambda row: (geometries[row].area, row))
        result[index] = records[owner]["name"]
    return result


def build_map_signatures(feather_path: Path):
    """Digest -> canonical tags, display label, and unique landmark ids.

    Identical bundles are indistinguishable
    to a text matcher, so they are asked about once and expanded after.

    Tags come from `catalog.schema.tag_dicts`, the canonical validated reader,
    so a schema change fails at one explicit boundary.
    """
    frame = schema.read_frame(Path(feather_path))
    records = schema.tag_dicts(frame)
    ids = frame["id"].values
    sources = frame["landmark_type"].values
    complex_names = enclosing_complex_names(frame, records)
    table = {}
    seen_landmark_ids = set()
    for i in range(len(frame)):
        source = "enc" if sources[i] == "enc" else "osm"
        text = catalog_lib._id_text(ids[i])
        landmark_id = (text if text.startswith(f"{source}:")
                       else f"{source}:{text}")
        if landmark_id in seen_landmark_ids:
            raise ValueError(
                "catalog repeats globally namespaced landmark id "
                f"{landmark_id!r}")
        seen_landmark_ids.add(landmark_id)
        tags = catalog_lib.prune_far_field_tags(records[i])
        if not tags:
            continue
        if i in complex_names:
            tags = {**tags, "complex:name": complex_names[i]}
        signature_id = signature(tags)
        entry = table.setdefault(signature_id, {
            "canonical_tags": dict(sorted(tags.items())),
            "display_label": signature_display(tags),
            "landmark_ids": [],
        })
        if entry["canonical_tags"] != dict(sorted(tags.items())):
            raise ValueError(
                f"signature digest collision for {signature_id}")
        entry["landmark_ids"].append(landmark_id)
    return table


def format_query(audit) -> str:
    """The Set 1 block for one tracklet, from its audit record.

    Carries the audit's *uncertainty*, not just its conclusion. Description and
    distinctive features help distinguish tracklets with otherwise identical
    coarse tags, while weighted tags and names retain marginal alternatives.
    """
    primary = audit["primary_object"]
    lines = []
    tags = sorted(primary.get("tags", []),
                  key=lambda t: -(t.get("weight") or 0))
    if tags:
        lines.append("tags: " + "; ".join(
            f"{t['tag']} ({(t.get('weight') or 0):.2f})" for t in tags))
    names = sorted(primary.get("name_candidates", []),
                   key=lambda c: -(c.get("weight") or 0))
    if names:
        lines.append("names: " + "; ".join(
            f"{c['name']} ({(c.get('weight') or 0):.2f}, {c.get('basis', '?')})"
            for c in names if c.get("name")))
    kind = audit.get("landmark_kind")
    extent = primary.get("extent")
    if kind or extent:
        lines.append(f"kind: {kind or '?'}, extent: {extent or '?'}")
    if primary.get("description"):
        lines.append(f'description: "{primary["description"]}"')
    if primary.get("distinctive_features"):
        lines.append("features: " + "; ".join(primary["distinctive_features"]))
    if audit.get("unresolved"):
        lines.append(f'unresolved: "{audit["unresolved"]}"')
    return "\n".join(lines)


def query_bundles(tracks: dict, audits: dict) -> dict:
    """Artifact-scoped AcceptedTracklet id -> Set 1 block.

    Acceptance, source binding, valid segments, and global identity all come
    from tracking/tracklets.py's single policy. There is deliberately no
    fallback to raw tag votes or run-local ids at this stage.
    """
    accepted = tracklets.build_accepted_tracklets(tracks, audits)
    out = {
        item.tracklet_id: format_query(item.audit)
        for item in accepted
    }
    n_dropped = sum(
        1 for audit in audits.values()
        if isinstance(audit, dict) and audit.get("verdict") == "drop")
    if n_dropped:
        print(f"  excluded {n_dropped} audited tracks with verdict=drop")
    return out


def build_requests(queries, sig_chunks, signatures, query_batch, thinking):
    """One request per (tracklet batch x map chunk)."""
    keys = sorted(queries)
    records = []
    for qi in range(0, len(keys), query_batch):
        batch = keys[qi:qi + query_batch]
        for ci, chunk in enumerate(sig_chunks):
            set1_parts = []
            for i, k in enumerate(batch):
                block = queries[k].splitlines()
                set1_parts.append(f" {i}. {block[0]}")
                set1_parts += [f"    {line}" for line in block[1:]]
            set1 = "\n".join(set1_parts)
            set2 = "\n".join(
                f" {i}. {signatures[s]['display_label']}"
                for i, s in enumerate(chunk))
            user = (f"Set 1 (observed from the vessel):\n{set1}\n\n"
                    f"Set 2 (map database, arbitrary slice):\n{set2}")
            records.append({
                "key": f"q{qi:04d}_c{ci:04d}",
                "batch_keys": batch,
                "chunk_index": ci,
                "chunk_signature_ids": list(chunk),
                "request": {
                    "contents": [{"parts": [{"text": user}], "role": "user"}],
                    "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseSchema": SCHEMA,
                        "thinkingConfig": {"thinkingLevel": thinking}}}})
    return records


def _exact_keys(value, expected, what):
    if not isinstance(value, dict) or set(value) != set(expected):
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{what} must have exact keys {sorted(expected)}; "
            f"missing={sorted(set(expected) - actual)}, "
            f"unknown={sorted(actual - set(expected))}")


def _integer(value, what, *, minimum=None, maximum=None):
    if type(value) is not int:
        raise ValueError(f"{what} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{what} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{what} must be <= {maximum}")
    return value


def _probability(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{what} must be finite and in [0, 1]")
    return value


def _reject_duplicate_json_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def validate_matching_response(key, response, metadata):
    """Decode one provider response and enforce complete inner coverage."""
    if not isinstance(response, dict):
        raise ValueError("provider response must be an object")
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("provider response must contain exactly one candidate")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise ValueError("response candidate must be an object")
    content = candidate.get("content")
    if not isinstance(content, dict):
        raise ValueError("response candidate content must be an object")
    parts = content.get("parts")
    if not isinstance(parts, list) or len(parts) != 1:
        raise ValueError("response content must contain exactly one part")
    part = parts[0]
    if not isinstance(part, dict):
        raise ValueError("response part must be an object")
    part_keys = set(part)
    if part_keys not in ({"text"}, {"text", "thoughtSignature"}):
        raise ValueError(
            "response part must contain text and only the optional provider "
            "thoughtSignature metadata")
    if not isinstance(part["text"], str):
        raise ValueError("response part text must be a string")
    if "thoughtSignature" in part and (
            not isinstance(part["thoughtSignature"], str)
            or not part["thoughtSignature"]):
        raise ValueError(
            "response part thoughtSignature must be a nonempty string")
    try:
        payload = json.loads(
            part["text"], object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value!r}")))
    except json.JSONDecodeError as error:
        raise ValueError(f"response text is not JSON: {error}") from error
    _exact_keys(payload, {"matches"}, "matching payload")
    entries = payload["matches"]
    batch_keys = metadata["batch_keys"]
    chunk = metadata["chunk_signature_ids"]
    if not isinstance(entries, list) or len(entries) != len(batch_keys):
        raise ValueError(
            f"request {key!r} must return exactly {len(batch_keys)} Set 1 "
            f"entries; found {len(entries) if isinstance(entries, list) else '?'}")
    validated = []
    seen_set1 = set()
    for entry_index, entry in enumerate(entries):
        _exact_keys(entry, {"set_1_id", "set_2_matches",
                            "no_match_confidence", "uniqueness_score"},
                    f"matches[{entry_index}]")
        set1_id = _integer(entry["set_1_id"],
                           f"matches[{entry_index}].set_1_id",
                           minimum=0, maximum=len(batch_keys) - 1)
        if set1_id in seen_set1:
            raise ValueError(f"duplicate set_1_id {set1_id} in request {key!r}")
        seen_set1.add(set1_id)
        raw_matches = entry["set_2_matches"]
        if not isinstance(raw_matches, list):
            raise ValueError("set_2_matches must be a list")
        set2_matches = []
        seen_set2 = set()
        for match_index, match in enumerate(raw_matches):
            _exact_keys(match, {"set_2_id", "match_type", "confidence"},
                        f"set_2_matches[{match_index}]")
            set2_id = _integer(
                match["set_2_id"], f"set_2_matches[{match_index}].set_2_id",
                minimum=0, maximum=len(chunk) - 1)
            if set2_id in seen_set2:
                raise ValueError(
                    f"duplicate set_2_id {set2_id} for set_1_id {set1_id}")
            seen_set2.add(set2_id)
            match_type = match["match_type"]
            if match_type not in ("instance", "category"):
                raise ValueError(f"invalid match_type {match_type!r}")
            set2_matches.append({
                "set_2_id": set2_id,
                "match_type": match_type,
                "confidence": _probability(
                    match["confidence"],
                    f"set_2_matches[{match_index}].confidence"),
            })
        validated.append({
            "set_1_id": set1_id,
            "set_2_matches": set2_matches,
            "no_match_confidence": _probability(
                entry["no_match_confidence"],
                f"matches[{entry_index}].no_match_confidence"),
            "uniqueness_score": _integer(
                entry["uniqueness_score"],
                f"matches[{entry_index}].uniqueness_score",
                minimum=1, maximum=5),
        })
    if seen_set1 != set(range(len(batch_keys))):
        raise ValueError(f"request {key!r} does not cover every Set 1 entry")
    return {"matches": sorted(validated, key=lambda item: item["set_1_id"])}


def aggregate(canonical_results, records_by_key, signatures):
    """Fold per-(batch, chunk) answers into per-tracklet matches.

    A matched signature expands to EVERY landmark carrying it, each at the
    signature's full confidence (chosen deliberately: a category match to
    `man_made=pier` genuinely admits all 327 piers equally, and the filter's
    mixture is the right place to represent that, not a pre-divided weight).

    Handling of no_match_confidence deserves care, because the obvious fusion
    is wrong. The model is asked "is it in THIS slice", and by construction it
    is not in ~14 of 15 slices, so a high value is the trivially correct answer
    almost every time and carries no information about whether the object is in
    the map AT ALL. Fusing per-slice values cannot recover the global question:
    a product drives it to zero (0.9^15 = 0.2) whether or not the object exists,
    and a minimum lets one spurious slice veto fourteen honest ones.

    So the per-slice value is kept for what it is good for - letting the model
    decline within a call, which is what stops it settling for the nearest thing
    - and the GLOBAL null is derived from the evidence instead:

        no match globally = 1 - (best match confidence anywhere)

    with the mean of the per-slice values used only when nothing matched at all.
    The per-slice values are still recorded, because disagreement between them
    and the derived value is a useful inconsistency signal.
    """
    # tid -> landmark_id -> selected candidate plus every score which entered
    # the explicit maximum aggregation rule.
    per_tracklet = defaultdict(dict)
    no_match = defaultdict(list)
    uniqueness = defaultdict(list)
    expanded = defaultdict(int)
    for record in canonical_results:
        meta = records_by_key[record.key]
        chunk = meta["chunk_signature_ids"]
        for entry in record.result["matches"]:
            tid = meta["batch_keys"][entry["set_1_id"]]
            no_match[tid].append(entry["no_match_confidence"])
            uniqueness[tid].append(entry["uniqueness_score"])
            for match in entry["set_2_matches"]:
                signature_id = chunk[match["set_2_id"]]
                signature_entry = signatures[signature_id]
                for landmark_id in signature_entry["landmark_ids"]:
                    expanded[tid] += 1
                    previous = per_tracklet[tid].get(landmark_id)
                    if previous is None:
                        previous = {
                            "aggregate_confidence": match["confidence"],
                            "per_call_candidate_scores": [
                                match["confidence"]],
                            "match_type": match["match_type"],
                            "signature_id": signature_id,
                        }
                        per_tracklet[tid][landmark_id] = previous
                    else:
                        if previous["signature_id"] != signature_id:
                            raise ValueError(
                                f"landmark {landmark_id!r} appears in more "
                                "than one canonical signature")
                        previous["per_call_candidate_scores"].append(
                            match["confidence"])
                        if match["confidence"] > previous[
                                "aggregate_confidence"]:
                            previous["aggregate_confidence"] = match[
                                "confidence"]
                            previous["match_type"] = match["match_type"]
    catalog_size = sum(
        len(entry["landmark_ids"]) for entry in signatures.values())
    max_calls = len(records_by_key)
    for tid, count in expanded.items():
        # Every signature occurs in one chunk, so a tracklet can expand each
        # catalog row at most once. This makes the work linear in the catalog.
        if count > catalog_size:
            raise ValueError(
                f"tracklet {tid!r} expansion exceeded the {catalog_size} "
                "catalog rows; request chunks overlap")
    if max_calls and any(len(values) > max_calls
                         for values in uniqueness.values()):
        raise ValueError("uniqueness aggregation exceeded request count")
    return per_tracklet, no_match, uniqueness


def global_no_match(matches, per_slice):
    """P(this landmark is nowhere in the map), from the evidence.

    See aggregate(): per-slice values answer a different question and cannot
    be fused into this one.
    """
    if matches:
        return round(1.0 - max(
            item["aggregate_confidence"] for item in matches.values()), 4)
    if per_slice:
        return round(sum(per_slice) / len(per_slice), 4)
    return 1.0


def to_log_lr(confidence, clip=DEFAULT_CLIP):
    """Confidence -> log odds, clipped. The seam the design doc specifies is
    an uncalibrated matcher behind a tuned transform; this is that transform,
    and the clip is what keeps a confident-but-wrong match survivable."""
    c = min(max(confidence, 1e-4), 1 - 1e-4)
    return max(-clip, min(clip, math.log(c / (1 - c))))


def to_compatibility_table(tracklet_id, scores, matcher_version,
                           scale=1.0, offset=0.0, clip=DEFAULT_CLIP,
                           default_log_lr=-2.0, status="fast"):
    """Raw scores -> the filter's structs.CompatibilityTable.

    Built as the struct itself (not a lookalike dict) and msgspec-encoded, so
    the localization loader decodes `compatibility.json` directly as
    `list[structs.CompatibilityTable]`. Only entries that differ from
    `default_log_lr` are emitted; everything absent scores the default, per
    that struct's contract.
    """
    entries = []
    for landmark_id, raw in scores.items():
        log_lr = max(-clip, min(clip, scale * raw + offset))
        if abs(log_lr - default_log_lr) > 1e-9:
            entries.append(structs.CompatibilityEntry(
                landmark_id=landmark_id, log_lr=float(log_lr)))
    entries.sort(key=lambda e: -e.log_lr)
    return structs.CompatibilityTable(
        tracklet_id=tracklet_id,
        matcher_version=matcher_version,
        entries=entries,
        default_log_lr=float(default_log_lr),
        clip_lo=float(-clip),
        clip_hi=float(clip),
        status=status,
    )



def make_request_set(records, *, model, thinking_level, build_identity,
                     orchestration_config_digest, upstreams):
    """Bind the complete matching workload, including ordered unit context."""
    input_digests = {
        "build_identity": build_identity,
        "orchestration_config": orchestration_config_digest,
    }
    input_digests.update({ref.kind: ref.content_digest for ref in upstreams})
    units = tuple(llm_lifecycle.RequestUnit(
        key=record["key"],
        request=record["request"],
        metadata={
            "batch_keys": record["batch_keys"],
            "chunk_index": record["chunk_index"],
            "chunk_signature_ids": record["chunk_signature_ids"],
        },
    ) for record in records)
    return llm_lifecycle.RequestSet.create(
        stage="landmark_matching",
        model=model,
        system_prompt=SYSTEM_PROMPT,
        response_schema=SCHEMA,
        media_settings={
            "response_mime_type": "application/json",
            "thinking_level": thinking_level,
        },
        input_digests=input_digests,
        upstreams=upstreams,
        units=units,
    )


def _request_metadata(request_set):
    return {
        unit.key: {
            "batch_keys": list(unit.metadata["batch_keys"]),
            "chunk_index": unit.metadata["chunk_index"],
            "chunk_signature_ids": list(
                unit.metadata["chunk_signature_ids"]),
        }
        for unit in request_set.units
    }


def matching_work_dir(output_dir: Path) -> Path:
    """Mutable retry state beside, never inside, the published artifact."""
    output_dir = Path(output_dir)
    return output_dir.with_name(output_dir.name + ".llm-work")


def _write_once_or_verify(path: Path, data: bytes, label: str) -> None:
    """Create immutable work input once, or require byte identity on resume."""
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise ValueError(
                f"recorded {label} at {path} differs from the requested "
                "workload; choose a new output artifact version")
        return
    artifact.atomic_write_file(path, data)


def make_work_snapshot(*, dataset, output_version, request_set, queries,
                       signatures, selected, build_identity, orchestration,
                       catalog_source, target_git_commit, target_build_path):
    """Freeze every semantic input needed by aggregate-only publication."""
    return {
        "schema": WORK_SNAPSHOT_SCHEMA,
        "dataset": dataset,
        "output_version": output_version,
        "request_set": request_set.to_dict(),
        "queries": queries,
        "signatures": signatures,
        "resolved_stage_config": selected,
        "build_identity": build_identity,
        "target_git_commit": target_git_commit,
        "target_build_path": str(Path(target_build_path).resolve()),
        "orchestration": orchestration,
        "catalog_source": str(catalog_source),
        "score_contract": SCORE_CONTRACT,
    }


def _load_json_strict(path: Path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(
                stream, object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {value!r}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON document {path}: {error}") from error


def validate_work_snapshot(value):
    """Validate and return the RequestSet embedded in a work snapshot."""
    expected = {
        "schema", "dataset", "output_version", "request_set", "queries",
        "signatures", "resolved_stage_config", "build_identity",
        "orchestration", "catalog_source", "score_contract",
        "target_git_commit", "target_build_path",
    }
    _exact_keys(value, expected, "matching work snapshot")
    if value["schema"] != WORK_SNAPSHOT_SCHEMA:
        raise ValueError(
            f"unsupported matching work snapshot {value['schema']!r}")
    for key in ("dataset", "output_version", "build_identity",
                "target_git_commit", "target_build_path",
                "catalog_source"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"matching work snapshot {key} is empty")
    if not Path(value["target_build_path"]).is_absolute():
        raise ValueError("matching work snapshot target_build_path is relative")
    selected = value["resolved_stage_config"]
    _exact_keys(selected, MATCHING_CONFIG_KEYS,
                "matching work resolved_stage_config")
    validate_selected_config(selected)
    if value["output_version"] != selected[
            "artifacts.landmark_matches_version"]:
        raise ValueError("matching snapshot output version disagrees with config")
    if value["score_contract"] != SCORE_CONTRACT:
        raise ValueError("unsupported matching score contract")
    orchestration = value["orchestration"]
    _exact_keys(orchestration, {"schema", "stage", "config_digest"},
                "matching orchestration")
    if (orchestration["schema"] != "farfield_pipeline_stage/v1"
            or orchestration["stage"] != "match"
            or orchestration["config_digest"]
            != artifact.sha256_json(selected)):
        raise ValueError("matching snapshot has invalid orchestration identity")

    request_set = llm_lifecycle.RequestSet.from_dict(value["request_set"])
    if request_set.stage != "landmark_matching":
        raise ValueError("matching snapshot has the wrong request-set stage")
    request_document = request_set.to_dict()
    if (request_set.model != selected["matching.model"]
            or request_set.system_prompt != SYSTEM_PROMPT
            or request_document["response_schema"] != SCHEMA
            or request_document["media_settings"] != {
                "response_mime_type": "application/json",
                "thinking_level": selected["matching.thinking_level"],
            }):
        raise ValueError(
            "matching snapshot prompt, schema, model, or media settings are "
            "not supported by this aggregator")
    if request_set.input_digests.get("build_identity") != value[
            "build_identity"]:
        raise ValueError("matching snapshot build identity is inconsistent")
    if request_set.input_digests.get("orchestration_config") != orchestration[
            "config_digest"]:
        raise ValueError(
            "matching snapshot orchestration digest is inconsistent")
    expected_kinds = (
        paths_lib.OBJECT_TRACKS, paths_lib.SEMANTIC_AUDITS,
        paths_lib.CATALOGS)
    expected_digest_keys = {
        "build_identity", "orchestration_config",
        *expected_kinds,
    }
    if set(request_set.input_digests) != expected_digest_keys:
        raise ValueError(
            "matching snapshot request input digests have an invalid shape")
    if tuple(ref.kind for ref in request_set.upstreams) != expected_kinds:
        raise ValueError(
            "matching snapshot must bind tracks, audits, then catalog")
    if any(ref.dataset != value["dataset"] for ref in request_set.upstreams):
        raise ValueError("matching snapshot crosses dataset identities")
    for ref in request_set.upstreams:
        if request_set.input_digests.get(ref.kind) != ref.content_digest:
            raise ValueError(
                f"matching snapshot input digest disagrees with {ref.kind}")

    queries = value["queries"]
    signatures = value["signatures"]
    if (not isinstance(queries, dict) or not queries
            or not all(isinstance(k, str) and k and isinstance(v, str) and v
                       for k, v in queries.items())):
        raise ValueError("matching snapshot queries must be a non-empty map")
    if not isinstance(signatures, dict) or not signatures:
        raise ValueError("matching snapshot signatures must be a non-empty map")
    all_landmarks = set()
    for signature_id, entry in signatures.items():
        _exact_keys(entry, {"canonical_tags", "display_label", "landmark_ids"},
                    f"signature {signature_id!r}")
        tags = entry["canonical_tags"]
        if (not isinstance(tags, dict) or not tags
                or signature(tags) != signature_id
                or signature_display(tags) != entry["display_label"]):
            raise ValueError(f"signature {signature_id!r} identity mismatch")
        ids = entry["landmark_ids"]
        if (not isinstance(ids, list) or not ids
                or not all(isinstance(item, str) and item for item in ids)
                or len(ids) != len(set(ids))):
            raise ValueError(
                f"signature {signature_id!r} has invalid landmark ids")
        overlap = all_landmarks.intersection(ids)
        if overlap:
            raise ValueError(
                "matching snapshot repeats globally namespaced landmark id "
                f"{sorted(overlap)[0]!r}")
        all_landmarks.update(ids)

    seen_by_tracklet = {key: [] for key in queries}
    for unit in request_set.units:
        metadata = dict(unit.metadata)
        _exact_keys(metadata,
                    {"batch_keys", "chunk_index", "chunk_signature_ids"},
                    f"request unit {unit.key!r} metadata")
        batch_keys = list(metadata["batch_keys"])
        chunk_ids = list(metadata["chunk_signature_ids"])
        if (not batch_keys or len(batch_keys) != len(set(batch_keys))
                or any(key not in queries for key in batch_keys)):
            raise ValueError(f"request unit {unit.key!r} has invalid batch keys")
        if (not chunk_ids or len(chunk_ids) != len(set(chunk_ids))
                or any(item not in signatures for item in chunk_ids)):
            raise ValueError(
                f"request unit {unit.key!r} has invalid signature chunk")
        _integer(metadata["chunk_index"],
                 f"request unit {unit.key!r} chunk_index", minimum=0)
        for key in batch_keys:
            seen_by_tracklet[key].extend(chunk_ids)
    expected_signatures = set(signatures)
    for key, seen in seen_by_tracklet.items():
        if len(seen) != len(set(seen)) or set(seen) != expected_signatures:
            raise ValueError(
                f"request snapshot does not cover every signature exactly "
                f"once for tracklet {key!r}")
    signature_ids = sorted(signatures)
    chunk_size = selected["matching.chunk_size"]
    expected_records = build_requests(
        queries,
        [signature_ids[index:index + chunk_size]
         for index in range(0, len(signature_ids), chunk_size)],
        signatures, selected["matching.query_batch"],
        selected["matching.thinking_level"])
    expected_request_set = make_request_set(
        expected_records, model=selected["matching.model"],
        thinking_level=selected["matching.thinking_level"],
        build_identity=value["build_identity"],
        orchestration_config_digest=orchestration["config_digest"],
        upstreams=request_set.upstreams)
    if expected_request_set.fingerprint != request_set.fingerprint:
        raise ValueError(
            "matching request units do not exactly encode the frozen queries, "
            "signatures, chunking, prompts, schema, and local mappings")
    return request_set


def record_work_snapshot(work_dir: Path, snapshot: dict) -> None:
    """Record the immutable request boundary in mutable orchestration state."""
    work_dir = Path(work_dir)
    if work_dir.is_symlink() or (work_dir.exists() and not work_dir.is_dir()):
        raise ValueError(f"matching work path is not a directory: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    request_set = validate_work_snapshot(snapshot)
    request_path = work_dir / llm_lifecycle.REQUEST_SET_NAME
    if request_path.exists() or request_path.is_symlink():
        recorded = llm_lifecycle.load_request_set(request_path)
        if recorded.fingerprint != request_set.fingerprint:
            raise ValueError(
                f"recorded matching request set at {request_path} differs "
                "from the requested workload; choose a new output artifact "
                "version")
    else:
        artifact.atomic_write_json(request_path, request_set.to_dict())
    _write_once_or_verify(
        work_dir / llm_lifecycle.REQUESTS_NAME,
        llm_lifecycle.transport_requests_bytes(request_set),
        "matching transport requests")
    _write_once_or_verify(
        work_dir / SIGNATURES_NAME,
        artifact.canonical_json_bytes(snapshot["signatures"]) + b"\n",
        "matching signature table")
    _write_once_or_verify(
        work_dir / WORK_SNAPSHOT_NAME,
        artifact.canonical_json_bytes(snapshot) + b"\n",
        "matching semantic snapshot")


def load_work_snapshot(work_dir: Path):
    """Load one self-contained immutable aggregation input."""
    work_dir = Path(work_dir)
    snapshot = _load_json_strict(work_dir / WORK_SNAPSHOT_NAME)
    request_set = validate_work_snapshot(snapshot)
    recorded_request_set = llm_lifecycle.load_request_set(
        work_dir / llm_lifecycle.REQUEST_SET_NAME)
    if recorded_request_set.fingerprint != request_set.fingerprint:
        raise ValueError(
            "matching semantic snapshot and request_set.json disagree")
    if ((work_dir / SIGNATURES_NAME).read_bytes()
            != artifact.canonical_json_bytes(snapshot["signatures"]) + b"\n"):
        raise ValueError(
            "matching semantic snapshot and signatures.json disagree")
    if ((work_dir / llm_lifecycle.REQUESTS_NAME).read_bytes()
            != llm_lifecycle.transport_requests_bytes(request_set)):
        raise ValueError(
            "matching semantic snapshot and requests.jsonl disagree")
    return snapshot, request_set


def validate_selected_config(selected):
    """Validate every field that shapes requests, execution, or aggregation."""
    specs = {
        "matching.model": build_config.ValueSpec((str,), nonempty=True),
        "matching.query_batch": build_config.ValueSpec((int,), minimum=1),
        "matching.chunk_size": build_config.ValueSpec((int,), minimum=1),
        "matching.thinking_level": build_config.ValueSpec(
            (str,), nonempty=True),
        "matching.confidence_floor": build_config.ValueSpec(
            (int, float), minimum=0.0, maximum=1.0),
        "matching.instance_max_rows": build_config.ValueSpec(
            (int,), minimum=1),
        "execution.llm_transport": build_config.ValueSpec(
            (str,), choices=("batch", "on_demand")),
        "execution.batch_gcs_prefix": build_config.ValueSpec(
            (str,), allow_none=True, nonempty=True),
        "execution.approve_cost": build_config.ValueSpec((bool,)),
        "cost.limit_usd": build_config.ValueSpec(
            (int, float), minimum=0.0),
        "artifacts.landmark_matches_version": build_config.ValueSpec(
            (str,), nonempty=True),
    }
    for key, spec in specs.items():
        spec.validate(key, selected[key])


def load_matching_config(args):
    """Validate the exact immutable recipe and matching stage digest."""
    config_path = Path(args.build_config)
    if config_path.name != build_config.BUILD_CONFIG_NAME:
        raise ValueError(
            f"--build_config must name {build_config.BUILD_CONFIG_NAME}")
    document = build_config.load(config_path.parent)
    if config_path.resolve() != (
            config_path.parent / build_config.BUILD_CONFIG_NAME).resolve():
        raise ValueError("--build_config does not name the loaded recipe")
    if document["dataset"] != args.dataset:
        raise ValueError(
            f"build config dataset {document['dataset']!r} does not match "
            f"--dataset {args.dataset!r}")
    if args.dataset_base is not None:
        recorded_base = document["inputs"].get("dataset_base")
        if recorded_base is None or Path(recorded_base).resolve() != Path(
                args.dataset_base).resolve():
            raise ValueError(
                "--dataset_base does not match build_config inputs.dataset_base")
    selected = {key: build_config.value(document, key)
                for key in MATCHING_CONFIG_KEYS}
    validate_selected_config(selected)
    actual_digest = artifact.sha256_json(selected)
    if args.orchestration_config_digest != actual_digest:
        raise ValueError(
            "--orchestration_config_digest does not match the immutable "
            "matching/execution/cost recipe")
    return document, selected, {
        "schema": "farfield_pipeline_stage/v1",
        "stage": "match",
        "config_digest": actual_digest,
    }


def validate_execution_args(args, selected) -> None:
    """Refuse execution flags that disagree with the recorded recipe."""
    expected_online = selected["execution.llm_transport"] == "on_demand"
    if bool(args.online) != expected_online:
        raise ValueError(
            "--online selection disagrees with execution.llm_transport")
    expected_prefix = selected["execution.batch_gcs_prefix"]
    if ((expected_online and args.gcs_prefix is not None)
            or (not expected_online and args.gcs_prefix != expected_prefix)):
        raise ValueError(
            "--gcs_prefix disagrees with execution.batch_gcs_prefix")
    if bool(args.approve_cost) != selected["execution.approve_cost"]:
        raise ValueError(
            "--approve_cost disagrees with execution.approve_cost")
    expected_limit = float(selected["cost.limit_usd"])
    if args.cost_limit is not None and args.cost_limit != expected_limit:
        raise ValueError("--cost_limit disagrees with cost.limit_usd")
    args.cost_limit = expected_limit
    args.model = selected["matching.model"]


def settings_document(*, snapshot, request_set):
    """Human-readable reproduction summary within the immutable artifact."""
    selected = snapshot["resolved_stage_config"]
    return {
        "generator": GENERATOR,
        "dataset": snapshot["dataset"],
        "catalog_feather": snapshot["catalog_source"],
        "model": selected["matching.model"],
        "transport": selected["execution.llm_transport"],
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode()).hexdigest(),
        "thinking_level": selected["matching.thinking_level"],
        "support_gate": ("audit membership: a track without a semantic audit "
                         "has no Set 1 entry, and verdict=drop is excluded "
                         "by the canonical accepted-tracklet contract"),
        "query_batch": selected["matching.query_batch"],
        "chunk_size": selected["matching.chunk_size"],
        "confidence_floor": selected["matching.confidence_floor"],
        "instance_max_rows": selected["matching.instance_max_rows"],
        "n_set1": len(snapshot["queries"]),
        "n_requests": len(request_set.units),
        "n_signatures": len(snapshot["signatures"]),
        "request_set_fingerprint": request_set.fingerprint,
        "required_result_coverage": "complete",
        "build_identity": snapshot["build_identity"],
        "upstreams": [ref.to_dict() for ref in request_set.upstreams],
        "score_contract": SCORE_CONTRACT,
        "spatial_gating": ("none - set 2 is the whole catalog, so the catalog's "
                           "extent bounds what can match"),
    }


def _require_nonaggregate_inputs(parser, args):
    missing = [name for name in (
        "tracks_dir", "audit_dir", "catalog_dir", "build_config",
        "orchestration_config_digest") if getattr(args, name) is None]
    if missing:
        parser.error(
            "building matching work requires "
            + ", ".join(f"--{name}" for name in missing))


def _build_snapshot(args, parser):
    """Resolve mutable upstreams exactly once, before provider execution."""
    _require_nonaggregate_inputs(parser, args)
    try:
        document, selected, orchestration = load_matching_config(args)
        validate_execution_args(args, selected)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    expected_versions = {
        paths_lib.OBJECT_TRACKS: build_config.value(
            document, "artifacts.object_tracks_version"),
        paths_lib.SEMANTIC_AUDITS: build_config.value(
            document, "artifacts.semantic_audits_version"),
    }
    try:
        tracks_ref = artifact.open_artifact(
            args.tracks_dir, expected_kind=paths_lib.OBJECT_TRACKS,
            expected_dataset=args.dataset,
            expected_version=expected_versions[paths_lib.OBJECT_TRACKS])
        audits_ref = artifact.open_artifact(
            args.audit_dir, expected_kind=paths_lib.SEMANTIC_AUDITS,
            expected_dataset=args.dataset,
            expected_version=expected_versions[paths_lib.SEMANTIC_AUDITS])
        catalog_ref = artifact.open_artifact(
            args.catalog_dir, expected_kind=paths_lib.CATALOGS,
            expected_dataset=args.dataset,
            expected_version=build_config.value(
                document, "artifacts.catalogs_version"))
        catalog_lineage.require_passed_source_coverage(catalog_ref)
    except (artifact.ArtifactError, audit_io.AuditArtifactError) as error:
        raise SystemExit(f"invalid matching input artifact: {error}") from error
    build_dir = Path(args.build_config).parent
    try:
        # The opens above prove each input is the artifact it claims to be at
        # the version this recipe names. Which GENERATION it belongs to is the
        # orchestrator's question, answered by `artifact_identity`.
        configured_lane.require(
            tracks_ref, document=document, kind=paths_lib.OBJECT_TRACKS)
        configured_lane.require(
            catalog_ref, document=document, kind=paths_lib.CATALOGS)
        configured_lane.require(
            audits_ref, document=document, kind=paths_lib.SEMANTIC_AUDITS)
        catalog_digest_keys = {
            "catalog_manifest_digest", "catalog_content_digest"}
        recorded_catalog_keys = catalog_digest_keys & set(document["inputs"])
        if recorded_catalog_keys and (
                recorded_catalog_keys != catalog_digest_keys
                or catalog_ref.manifest_digest
                    != document["inputs"]["catalog_manifest_digest"]
                or catalog_ref.content_digest
                    != document["inputs"]["catalog_content_digest"]):
            raise ValueError(
                "matching catalog differs from target build digests")
    except (artifact.ArtifactError, configured_lane.ConfiguredLaneError,
            ValueError) as error:
        raise SystemExit(f"invalid matching input binding: {error}") from error
    try:
        audits = audit_io.load_audits(args.tracks_dir, args.audit_dir)
    except (artifact.ArtifactError, audit_io.AuditArtifactError) as error:
        raise SystemExit(f"invalid matching audit artifact: {error}") from error
    if (audits.tracks_ref != tracks_ref
            or audits.semantic_audits_ref != audits_ref):
        raise SystemExit(
            "matching audit loader changed the exact authorized refs")
    upstreams = (tracks_ref, audits_ref, catalog_ref)
    catalog_path = Path(args.catalog_dir) / "catalog.feather"
    if not catalog_path.is_file() or catalog_path.is_symlink():
        raise SystemExit(
            f"catalog artifact lacks regular catalog.feather: {catalog_path}")

    signatures = build_map_signatures(catalog_path)
    signature_ids = sorted(signatures)
    if not signature_ids:
        raise SystemExit("catalog contains no far-field signature to match")
    chunk_size = selected["matching.chunk_size"]
    signature_chunks = [
        signature_ids[i:i + chunk_size]
        for i in range(0, len(signature_ids), chunk_size)]
    queries = query_bundles(audits.source_tracks, audits)
    if not queries:
        raise SystemExit(
            "no matchable tracklet: every audited track was dropped; "
            "nothing to match")
    records = build_requests(
        queries, signature_chunks, signatures,
        selected["matching.query_batch"],
        selected["matching.thinking_level"])
    request_set = make_request_set(
        records, model=selected["matching.model"],
        thinking_level=selected["matching.thinking_level"],
        build_identity=document["build_identity"],
        orchestration_config_digest=orchestration["config_digest"],
        upstreams=upstreams)
    snapshot = make_work_snapshot(
        dataset=args.dataset,
        output_version=selected["artifacts.landmark_matches_version"],
        request_set=request_set, queries=queries, signatures=signatures,
        selected=selected, build_identity=document["build_identity"],
        orchestration=orchestration, catalog_source=catalog_path,
        target_git_commit=document["git_commit"],
        target_build_path=build_dir)
    print(f"map: {sum(len(v['landmark_ids']) for v in signatures.values())} "
          f"landmarks -> {len(signatures)} signatures in "
          f"{len(signature_chunks)} chunks")
    print(f"queries: {len(queries)} tracklets in "
          f"{math.ceil(len(queries) / selected['matching.query_batch'])} "
          "batches")
    return snapshot, request_set


def revalidate_work_snapshot_inputs(snapshot, request_set):
    """Re-prove what the frozen workload says about itself.

    Deliberately reopens NOTHING. Aggregation runs long after the requests
    were built, and the upstream artifacts may legitimately be gone by then --
    `test_aggregate_does_not_reopen_mutable_semantic_inputs` deletes them and
    requires this to succeed. Everything aggregation needs was frozen into the
    snapshot and the request set, and re-reading a mutable directory here
    would let an edit change what a frozen workload means.

    Whether those upstreams were the right generation was decided when the
    requests were built, and is recorded; it is not re-litigated here.
    """
    tracks_ref, audits_ref, catalog_ref = request_set.upstreams
    for reference, kind in ((tracks_ref, paths_lib.OBJECT_TRACKS),
                            (audits_ref, paths_lib.SEMANTIC_AUDITS),
                            (catalog_ref, paths_lib.CATALOGS)):
        if reference.kind != kind:
            raise SystemExit(
                f"matching snapshot upstream order is wrong: expected {kind}, "
                f"found {reference.kind}")
        if reference.dataset != snapshot["dataset"]:
            raise SystemExit(
                f"matching snapshot {kind} belongs to another dataset")
    return snapshot["target_git_commit"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    paths_lib.add_arguments(parser, dataset_required=True)
    parser.add_argument("--tracks_dir", type=Path)
    parser.add_argument("--audit_dir", type=Path)
    parser.add_argument("--catalog_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--build_config", type=Path)
    parser.add_argument("--orchestration_config_digest")
    # The identity the orchestrator computed for this stage's artifact; see
    # `pipeline.stage_identity_flags`. Optional so a producer stays runnable
    # by hand -- the artifact is then honestly unattributed.
    parser.add_argument("--artifact_identity", default=None)
    parser.add_argument("--artifact_recipe", default=None,
                        help="path to the resolved stage config and build "
                             "inputs this artifact should record, written by "
                             "`pipeline run`")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--gcs_prefix", default=None)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--poll_interval", type=int, default=120)
    parser.add_argument("--cost_limit", type=float, default=None)
    parser.add_argument("--approve_cost", action="store_true")
    parser.add_argument("--submit", action="store_true",
                        help="Execute pending immutable requests and carry "
                             "straight on to aggregation. "
                             "Batch by default; --online for on-demand. "
                             "Resumable: reruns retry only failures.")
    parser.add_argument("--build_only", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()
    if args.parallel <= 0:
        parser.error("--parallel must be a positive integer")
    if args.poll_interval <= 0:
        parser.error("--poll_interval must be a positive integer")
    if args.build_only and args.aggregate_only:
        parser.error("--build_only and --aggregate_only are mutually exclusive")
    if args.submit and (args.build_only or args.aggregate_only):
        parser.error("--submit cannot be combined with a one-phase mode")
    output_dir = Path(args.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise SystemExit(
            f"completed matching artifact already exists: {output_dir}")
    work_dir = matching_work_dir(output_dir)
    if args.aggregate_only:
        supplied = [name for name in (
            "farfield_root", "dataset_base", "landmark_base",
            "frame_landmarks_version", "object_tracks_version",
            "semantic_audits_version", "bearing_observations_version",
            "landmark_matches_version", "alignment_diagnostics_version",
            "localization_inputs_version", "tracks_dir", "audit_dir",
            "catalog_dir", "build_config", "orchestration_config_digest")
                    if getattr(args, name) is not None]
        if supplied:
            parser.error(
                "--aggregate_only consumes only the recorded immutable "
                "snapshot; do not supply "
                + ", ".join(f"--{name}" for name in supplied))
        if (args.online or args.gcs_prefix is not None or args.approve_cost
                or args.cost_limit is not None):
            parser.error(
                "execution options are invalid with --aggregate_only")
        try:
            snapshot, request_set = load_work_snapshot(work_dir)
        except (OSError, ValueError,
                llm_lifecycle.LlmLifecycleError) as error:
            raise SystemExit(f"invalid matching work state: {error}") from error
    else:
        snapshot, request_set = _build_snapshot(args, parser)
        try:
            record_work_snapshot(work_dir, snapshot)
            snapshot, request_set = load_work_snapshot(work_dir)
        except (OSError, ValueError,
                llm_lifecycle.LlmLifecycleError) as error:
            raise SystemExit(f"invalid matching work state: {error}") from error
    if snapshot["dataset"] != args.dataset:
        raise SystemExit(
            f"matching snapshot belongs to {snapshot['dataset']!r}, not "
            f"{args.dataset!r}")
    target_git_commit = revalidate_work_snapshot_inputs(
        snapshot, request_set)
    output_version = snapshot["output_version"]
    if output_dir.name != output_version:
        parser.error(
            f"--output_dir must end in recorded version {output_version!r}")
    selected = snapshot["resolved_stage_config"]
    queries = snapshot["queries"]
    signatures = snapshot["signatures"]
    signature_ids = sorted(signatures)
    upstreams = request_set.upstreams
    thinking_level = selected["matching.thinking_level"]
    confidence_floor = selected["matching.confidence_floor"]
    instance_max_rows = selected["matching.instance_max_rows"]
    print(f"requests: {len(request_set.units)}")
    est = sum(len(unit.request["contents"][0]["parts"][0]["text"])
              for unit in request_set.units) // 4
    print(f"estimated input: ~{est / 1e6:.2f}M tokens "
          f"(+ system {len(request_set.system_prompt) // 4 * len(request_set.units) / 1e6:.2f}M)")
    if args.build_only:
        print(f"\nwrote immutable requests to {work_dir}; rerun with "
              "--submit to execute them")
        return

    transport_path = work_dir / TRANSPORT_RESULTS_NAME
    attempts_dir = work_dir / ATTEMPTS_DIR_NAME
    transport_paths = ([transport_path] if transport_path.exists() else [])
    transport_paths.extend(vbm.completed_submission_results(work_dir))
    for existing_transport in transport_paths:
        imported = llm_lifecycle.import_transport_results(
            existing_transport, attempts_dir, request_set)
        if imported:
            print(f"preserved {imported} new transport result(s) in "
                  f"{attempts_dir}")
    metadata = _request_metadata(request_set)
    attempts = (llm_lifecycle.load_attempts(attempts_dir)
                if attempts_dir.exists() else ())
    if args.submit:
        pending = llm_lifecycle.pending_request_keys(
            request_set, attempts,
            lambda key, response: validate_matching_response(
                key, response, metadata[key]))
        if pending:
            round_index, pending_path, round_transport = (
                vbm.next_submission_paths(work_dir))
            artifact.atomic_create_file(
                pending_path,
                llm_lifecycle.transport_requests_bytes(request_set, pending))
            print(f"submitting {len(pending)}/{len(request_set.units)} "
                  f"requests without a validated success (round "
                  f"{round_index})")
            vbm.run_requests(
                args, pending_path, round_transport,
                tag=(f"{args.dataset}_matching_{output_version}_"
                     f"r{round_index:04d}"))
            for completed in vbm.completed_submission_results(work_dir):
                imported = llm_lifecycle.import_transport_results(
                    completed, attempts_dir, request_set)
                if imported:
                    print(f"preserved {imported} new transport result(s) in "
                          f"{attempts_dir}")
            attempts = (llm_lifecycle.load_attempts(attempts_dir)
                        if attempts_dir.exists() else ())
        else:
            print("all matching requests already have a validated success")
    if not attempts_dir.exists():
        raise SystemExit(
            f"no bound attempts at {attempts_dir}; submit the immutable "
            "request set first (--submit, or write provider output to "
            f"{transport_path})")
    canonical_results = llm_lifecycle.compile_canonical_results(
        request_set,
        attempts,
        lambda key, response: validate_matching_response(
            key, response, metadata[key]),
    )
    per_tracklet, no_match, uniqueness = aggregate(
        canonical_results, metadata, signatures)
    print(f"validated complete coverage: {len(canonical_results)}/"
          f"{len(request_set.units)} requests; aggregated "
          f"{len(per_tracklet)} tracklets with >=1 match")

    matches = {}
    tables = []
    expected_calls = {
        tid: sum(tid in unit.metadata["batch_keys"]
                 for unit in request_set.units)
        for tid in queries
    }
    for tid in sorted(queries):
        if (len(no_match.get(tid, ())) != expected_calls[tid]
                or len(uniqueness.get(tid, ())) != expected_calls[tid]):
            raise llm_lifecycle.IncompleteCoverageError(
                f"tracklet {tid!r} does not have one validated answer for "
                "every catalog chunk")
        got = per_tracklet.get(tid, {})
        kept = {}
        downgraded = 0
        for lid, candidate in got.items():
            conf = candidate["aggregate_confidence"]
            if conf < confidence_floor:
                continue
            kind = candidate["match_type"]
            sig = candidate["signature_id"]
            if (kind == "instance"
                    and len(signatures[sig]["landmark_ids"])
                    > instance_max_rows):
                kind = "category"
                downgraded += 1
            kept[lid] = dict(candidate, match_type=kind)
        nm = global_no_match(kept, no_match.get(tid, []))
        slice_values = no_match.get(tid, [])
        matches[tid] = {
            "query": queries[tid],
            "aggregate_no_match_confidence": nm,
            "per_call_no_match_confidence": {
                "n": len(slice_values),
                "mean": round(sum(slice_values) / len(slice_values), 3)
                if slice_values else None,
                "min": round(min(slice_values), 3) if slice_values else None,
                "scores": slice_values,
            },
            "uniqueness": {
                "aggregate_score": round(
                    sum(uniqueness[tid]) / len(uniqueness[tid]), 4),
                "per_call_scores": uniqueness[tid],
                "aggregation_rule": SCORE_CONTRACT[
                    "uniqueness_aggregation_rule"],
            },
            "n_landmarks": len(kept),
            "n_signatures": len({
                v["signature_id"] for v in kept.values()}),
            "n_downgraded_to_category": downgraded,
            "matches": [{
                         "landmark_id": lid,
                         "per_call_candidate_scores": v[
                             "per_call_candidate_scores"],
                         "aggregate_confidence": v[
                             "aggregate_confidence"],
                         "aggregation_rule": SCORE_CONTRACT[
                             "candidate_aggregation_rule"],
                         "match_type": v["match_type"],
                         "signature_id": v["signature_id"],
                         "signature_display": signatures[
                             v["signature_id"]]["display_label"],
                         }
                        for lid, v in sorted(kept.items(),
                                             key=lambda kv: -kv[1][
                                                 "aggregate_confidence"])]}
        scores = {lid: v["aggregate_confidence"]
                  for lid, v in kept.items()}
        tables.append(to_compatibility_table(
            tid, {lid: to_log_lr(c) for lid, c in scores.items()},
            matcher_version=f"llm_chunked_v2_{thinking_level.lower()}",
            scale=1.0, offset=0.0,
            default_log_lr=to_log_lr(
                max(1e-4, 1.0 - nm) / max(1, len(signature_ids)))))
    if set(matches) != set(queries) or {table.tracklet_id for table in tables} != set(
            queries):
        raise llm_lifecycle.IncompleteCoverageError(
            "matching aggregation did not cover every accepted tracklet")

    settings = settings_document(snapshot=snapshot, request_set=request_set)
    manifest_config = {
        "phase": "canonical_results",
        "coverage": "complete",
        "n_expected": len(request_set.units),
        "n_successful": len(canonical_results),
        "n_tracklets_expected": len(queries),
        "n_tracklets_successful": len(matches),
        "request_set_fingerprint": request_set.fingerprint,
        "build_identity": snapshot["build_identity"],
        "orchestration": snapshot["orchestration"],
        "resolved_stage_config": selected,
        "semantic_snapshot_sha256": artifact.sha256_json(snapshot),
        "score_contract": SCORE_CONTRACT,
    }
    with publication.published_artifact(
            output_dir,
            kind=paths_lib.LANDMARK_MATCHES,
            dataset=args.dataset,
            version=output_version,
            generator=GENERATOR,
            git_commit=target_git_commit,
            arguments=sys.argv,
            upstreams=upstreams,
            config=manifest_config,
            artifact_identity=getattr(args, "artifact_identity", None),
            recipe=artifact_recipe.load(
                getattr(args, "artifact_recipe", None)),
            declared_outputs=FINAL_OUTPUTS) as builder:
        artifact.atomic_write_json(
            builder.output_path(WORK_SNAPSHOT_NAME), snapshot)
        artifact.atomic_write_json(
            builder.output_path(llm_lifecycle.REQUEST_SET_NAME),
            request_set.to_dict())
        artifact.atomic_write_file(
            builder.output_path(llm_lifecycle.REQUESTS_NAME),
            llm_lifecycle.transport_requests_bytes(request_set))
        artifact.atomic_write_file(
            builder.output_path(llm_lifecycle.CANONICAL_RESULTS_NAME),
            llm_lifecycle.canonical_results_bytes(
                request_set, canonical_results))
        artifact.atomic_write_json(
            builder.output_path(SIGNATURES_NAME), signatures)
        artifact.atomic_write_json(
            builder.output_path(SETTINGS_NAME), settings)
        artifact.atomic_write_json(
            builder.output_path(MATCHES_NAME), matches)
        artifact.atomic_write_file(
            builder.output_path(COMPATIBILITY_NAME),
            msgspec.json.encode(tables, enc_hook=msgspec_enc_hook))
    assert builder.artifact_ref is not None
    n_hit = sum(1 for m in matches.values() if m["n_landmarks"])
    print(f"tracklets with a match above the floor: {n_hit}/{len(matches)}")
    print(f"published complete {paths_lib.LANDMARK_MATCHES} artifact: "
          f"{output_dir}")


if __name__ == "__main__":
    main()
