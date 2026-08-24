"""Match tracklets against the whole map, position-free, via chunked LLM calls.

This is the matcher. Set 1 is built PER AUDITED TRACK -- the merge stage is
gone (REORG.md), so each audited track is its own tracklet with id
"T<track_id>" and its Set 1 entry comes straight from its semantic-audit
record. Audit membership is the support gate: a track that was never audited
has no canonical semantics and gets no Set 1 entry, and `verdict: drop` is
honoured here (replacing the old --min_supports flag entirely). Output is a
`CompatibilityTable` per tracklet for the bearing-only filter
(farfield.localization.structs).

**No spatial information is used anywhere.** An earlier design gated
candidates by a bearing wedge computed from GPS, which leaked the answer:
selecting candidates by where the vessel was, then asking the filter to
recover where the vessel was, is circular. Set 2 here is the map, full stop.

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

from common.python.serialization import msgspec_enc_hook
from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import build_config
from experimental.overhead_matching.swag.farfield import llm_lifecycle
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield import publication
from experimental.overhead_matching.swag.farfield import provenance
from experimental.overhead_matching.swag.farfield.calibration import audit_io
from experimental.overhead_matching.swag.farfield.catalog import (
    catalog as catalog_lib,
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
                means it could not, which does not elimnate the name as a possibility.
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
SETTINGS_NAME = "settings.json"
MATCHES_NAME = "matches.json"
COMPATIBILITY_NAME = "compatibility.json"
FINAL_OUTPUTS = (
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


def signature(tags: dict) -> str:
    return "; ".join(f"{k}={v}" for k, v in sorted(tags.items()))


def build_map_signatures(feather_path: Path):
    """signature -> [landmark_id, ...]. Identical bundles are indistinguishable
    to a text matcher, so they are asked about once and expanded after.

    Tags come from `catalog.schema.tag_dicts`, the shared reader, rather than
    from this module's own parse of the columns. The feather moved from ~1700
    sparse tag columns to a single JSON `tags` column, and a private reader
    silently survived that change: it saw one unparsed column, matched almost
    nothing against the keep-list, and reported 13,210 landmarks as 16 distinct
    signatures. Nothing raised. Use the shared reader so a schema change breaks
    loudly in one place instead of quietly everywhere.
    """
    frame = schema.read_frame(Path(feather_path))
    records = schema.tag_dicts(frame)
    ids = frame["id"].values
    sources = frame["landmark_type"].values
    table = defaultdict(list)
    for i in range(len(frame)):
        tags = catalog_lib.prune_far_field_tags(records[i])
        if not tags:
            continue
        source = "enc" if sources[i] == "enc" else "osm"
        text = catalog_lib._id_text(ids[i])
        landmark_id = (text if text.startswith(f"{source}:")
                       else f"{source}:{text}")
        table[signature(tags)].append(landmark_id)
    return dict(table)


def format_query(audit) -> str:
    """The Set 1 block for one tracklet, from its audit record.

    Carries the audit's *uncertainty*, not just its conclusion. Flattening
    weighted tags into a bare list hid which reading was believed and which
    was marginal - LT17's stray "Logan Airport Control Tower" votes were
    invisible in the old format. The description and distinctive features are
    the only fields that can separate the many tracklets whose tags are just
    `building=commercial`, and the map side carries free-text `description`
    too (the ENC's own "visually conspicuous"), so this is not asymmetric.
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


def build_requests(queries, sig_chunks, query_batch, thinking):
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
            set2 = "\n".join(f" {i}. {s}" for i, s in enumerate(chunk))
            user = (f"Set 1 (observed from the vessel):\n{set1}\n\n"
                    f"Set 2 (map database, arbitrary slice):\n{set2}")
            records.append({
                "key": f"q{qi:04d}_c{ci:04d}",
                "batch_keys": batch,
                "chunk_index": ci,
                "chunk_signatures": list(chunk),
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
    if not isinstance(part, dict) or set(part) != {"text"} or not isinstance(
            part["text"], str):
        raise ValueError("response part must contain exactly one text string")
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
    chunk = metadata["chunk_signatures"]
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


def aggregate(canonical_results, records_by_key, sig_to_ids):
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
    per_tracklet = defaultdict(dict)      # tid -> landmark_id -> (conf, kind)
    no_match = defaultdict(list)
    for record in canonical_results:
        meta = records_by_key[record.key]
        chunk = meta["chunk_signatures"]
        for entry in record.result["matches"]:
            tid = meta["batch_keys"][entry["set_1_id"]]
            no_match[tid].append(entry["no_match_confidence"])
            for match in entry["set_2_matches"]:
                signature_text = chunk[match["set_2_id"]]
                for landmark_id in sig_to_ids[signature_text]:
                    previous = per_tracklet[tid].get(landmark_id)
                    candidate = (match["confidence"], match["match_type"],
                                 signature_text)
                    if previous is None or candidate[0] > previous[0]:
                        per_tracklet[tid][landmark_id] = candidate
    return per_tracklet, no_match


def global_no_match(matches, per_slice):
    """P(this landmark is nowhere in the map), from the evidence.

    See aggregate(): per-slice values answer a different question and cannot
    be fused into this one.
    """
    if matches:
        return round(1.0 - max(c for c, _, _ in matches.values()), 4)
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
            "chunk_signatures": record["chunk_signatures"],
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
            "chunk_signatures": list(unit.metadata["chunk_signatures"]),
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


def record_work_snapshot(work_dir: Path,
                         request_set: llm_lifecycle.RequestSet,
                         sig_to_ids: dict) -> None:
    """Record the immutable request boundary in mutable orchestration state."""
    work_dir = Path(work_dir)
    if work_dir.is_symlink() or (work_dir.exists() and not work_dir.is_dir()):
        raise ValueError(f"matching work path is not a directory: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
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
        artifact.canonical_json_bytes(sig_to_ids) + b"\n",
        "matching signature table")


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
    actual_digest = artifact.sha256_json(selected)
    if args.orchestration_config_digest != actual_digest:
        raise ValueError(
            "--orchestration_config_digest does not match the immutable "
            "matching/execution/cost recipe")

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


def settings_document(*, args, selected, records, sigs, queries,
                      request_set, catalog_path, upstreams, build_identity):
    """Human-readable reproduction summary within the immutable artifact."""
    return {
        "generator": GENERATOR,
        "dataset": args.dataset,
        "catalog_feather": str(catalog_path),
        "model": selected["matching.model"],
        "transport": selected["execution.llm_transport"],
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode()).hexdigest(),
        "thinking_level": selected["matching.thinking_level"],
        "support_gate": ("audit membership: a track without a semantic audit "
                         "has no Set 1 entry, and verdict=drop is excluded "
                         "(replaces the retired --min_supports flag)"),
        "query_batch": selected["matching.query_batch"],
        "chunk_size": selected["matching.chunk_size"],
        "confidence_floor": selected["matching.confidence_floor"],
        "instance_max_rows": selected["matching.instance_max_rows"],
        "n_set1": len(queries),
        "n_requests": len(records),
        "n_signatures": len(sigs),
        "request_set_fingerprint": request_set.fingerprint,
        "required_result_coverage": "complete",
        "build_identity": build_identity,
        "upstreams": [ref.to_dict() for ref in upstreams],
        "spatial_gating": ("none - set 2 is the whole catalog, so the catalog's "
                           "extent bounds what can match"),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    paths_lib.add_arguments(parser, dataset_required=True)
    parser.add_argument("--tracks_dir", type=Path, required=True)
    parser.add_argument("--audit_dir", type=Path, required=True)
    parser.add_argument("--catalog_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--build_config", type=Path, required=True)
    parser.add_argument("--orchestration_config_digest", required=True)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--gcs_prefix", default=None)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--poll_interval", type=int, default=120)
    parser.add_argument("--cost_limit", type=float, default=None)
    parser.add_argument("--approve_cost", action="store_true")
    parser.add_argument("--submit", action="store_true",
                        help="Execute the requests and carry straight on to "
                             "aggregation, instead of stopping so "
                             "vertex_batch_manager can be invoked by hand. "
                             "Batch by default; --online for on-demand. "
                             "Resumable: reruns retry only failures.")
    parser.add_argument("--build_only", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()
    if args.build_only and args.aggregate_only:
        parser.error("--build_only and --aggregate_only are mutually exclusive")
    if args.submit and (args.build_only or args.aggregate_only):
        parser.error("--submit cannot be combined with a one-phase mode")
    try:
        document, selected, orchestration = load_matching_config(args)
        validate_execution_args(args, selected)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    output_dir = Path(args.output_dir)
    output_version = selected["artifacts.landmark_matches_version"]
    if output_dir.name != output_version:
        parser.error(
            f"--output_dir must end in configured version {output_version!r}")
    if output_dir.exists() or output_dir.is_symlink():
        raise SystemExit(
            f"completed matching artifact already exists: {output_dir}")

    try:
        audits = audit_io.load_audits(args.tracks_dir, args.audit_dir)
        tracks_ref = audits.tracks_ref
        audits_ref = audits.semantic_audits_ref
        catalog_ref = artifact.open_artifact(
            args.catalog_dir, expected_kind=paths_lib.CATALOGS,
            expected_dataset=args.dataset,
            expected_version=build_config.value(
                document, "artifacts.catalogs_version"))
    except (artifact.ArtifactError, audit_io.AuditArtifactError) as error:
        raise SystemExit(f"invalid matching input artifact: {error}") from error
    expected_versions = {
        paths_lib.OBJECT_TRACKS: build_config.value(
            document, "artifacts.object_tracks_version"),
        paths_lib.SEMANTIC_AUDITS: build_config.value(
            document, "artifacts.semantic_audits_version"),
    }
    for ref in (tracks_ref, audits_ref):
        if ref.dataset != args.dataset or ref.version != expected_versions[ref.kind]:
            raise SystemExit(
                f"{ref.kind} artifact identity does not match build_config")
    upstreams = (tracks_ref, audits_ref, catalog_ref)
    catalog_path = Path(args.catalog_dir) / "catalog.feather"
    if not catalog_path.is_file() or catalog_path.is_symlink():
        raise SystemExit(
            f"catalog artifact lacks regular catalog.feather: {catalog_path}")

    sig_to_ids = build_map_signatures(catalog_path)
    sigs = sorted(sig_to_ids)
    if not sigs:
        raise SystemExit("catalog contains no far-field signature to match")
    query_batch = selected["matching.query_batch"]
    chunk_size = selected["matching.chunk_size"]
    thinking_level = selected["matching.thinking_level"]
    confidence_floor = selected["matching.confidence_floor"]
    instance_max_rows = selected["matching.instance_max_rows"]
    sig_chunks = [sigs[i:i + chunk_size]
                  for i in range(0, len(sigs), chunk_size)]
    tracks = audits.source_tracks
    queries = query_bundles(tracks, audits)
    if not queries:
        raise SystemExit(
            "no matchable tracklet: every audited track was dropped; "
            "nothing to match")
    print(f"map: {sum(len(v) for v in sig_to_ids.values())} landmarks -> "
          f"{len(sigs)} signatures in {len(sig_chunks)} chunks")
    print(f"queries: {len(queries)} tracklets in "
          f"{math.ceil(len(queries) / query_batch)} batches")

    records = build_requests(
        queries, sig_chunks, query_batch, thinking_level)
    expected_request_set = make_request_set(
        records,
        model=selected["matching.model"],
        thinking_level=thinking_level,
        build_identity=document["build_identity"],
        orchestration_config_digest=orchestration["config_digest"],
        upstreams=upstreams)
    print(f"requests: {len(records)}")
    work_dir = matching_work_dir(output_dir)
    if (args.aggregate_only
            and not (work_dir / llm_lifecycle.REQUEST_SET_NAME).is_file()):
        raise SystemExit(
            f"no matching request snapshot at {work_dir}; run --build_only")
    try:
        record_work_snapshot(work_dir, expected_request_set, sig_to_ids)
        request_set = llm_lifecycle.load_request_set(
            work_dir / llm_lifecycle.REQUEST_SET_NAME)
    except (OSError, ValueError, llm_lifecycle.LlmLifecycleError) as error:
        raise SystemExit(f"invalid matching work state: {error}") from error
    if request_set.fingerprint != expected_request_set.fingerprint:
        raise SystemExit(
            "recorded matching request set does not match the current "
            "catalog, tracks, audits, prompt, schema, model, or settings; "
            "choose a new matching artifact version")
    est = sum(len(r["request"]["contents"][0]["parts"][0]["text"])
              for r in records) // 4
    print(f"estimated input: ~{est / 1e6:.2f}M tokens "
          f"(+ system {len(SYSTEM_PROMPT) // 4 * len(records) / 1e6:.2f}M)")
    if args.build_only:
        print(f"\nwrote immutable requests to {work_dir}; submit "
              f"{work_dir / llm_lifecycle.REQUESTS_NAME} with "
              "vertex_batch_manager")
        return

    transport_path = work_dir / TRANSPORT_RESULTS_NAME
    attempts_dir = work_dir / ATTEMPTS_DIR_NAME
    transport_paths = ([transport_path] if transport_path.exists() else [])
    transport_paths.extend(sorted(work_dir.glob("transport_submit_*.jsonl")))
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
            round_index = 1 + len(tuple(
                work_dir.glob("transport_submit_*.jsonl")))
            pending_path = work_dir / f"requests_submit_{round_index:04d}.jsonl"
            round_transport = (
                work_dir / f"transport_submit_{round_index:04d}.jsonl")
            artifact.atomic_write_file(
                pending_path,
                llm_lifecycle.transport_requests_bytes(request_set, pending))
            print(f"submitting {len(pending)}/{len(request_set.units)} "
                  f"requests without a validated success (round "
                  f"{round_index})")
            vbm.run_requests(
                args, pending_path, round_transport,
                tag=(f"{args.dataset}_matching_{output_version}_"
                     f"r{round_index:04d}"))
            if round_transport.exists():
                imported = llm_lifecycle.import_transport_results(
                    round_transport, attempts_dir, request_set)
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
    per_tracklet, no_match = aggregate(
        canonical_results, metadata, sig_to_ids)
    print(f"validated complete coverage: {len(canonical_results)}/"
          f"{len(request_set.units)} requests; aggregated "
          f"{len(per_tracklet)} tracklets with >=1 match")

    matches = {}
    tables = []
    for tid in sorted(queries):
        if len(no_match.get(tid, ())) != len(sig_chunks):
            raise llm_lifecycle.IncompleteCoverageError(
                f"tracklet {tid!r} does not have one validated answer for "
                "every catalog chunk")
        got = per_tracklet.get(tid, {})
        kept = {}
        downgraded = 0
        for lid, (conf, kind, sig) in got.items():
            if conf < confidence_floor:
                continue
            if kind == "instance" and len(
                    sig_to_ids.get(sig, [])) > instance_max_rows:
                kind = "category"
                downgraded += 1
            kept[lid] = (conf, kind, sig)
        nm = global_no_match(kept, no_match.get(tid, []))
        slice_values = no_match.get(tid, [])
        matches[tid] = {
            "query": queries[tid],
            "no_match_confidence": nm,
            "per_slice_no_match": {
                "n": len(slice_values),
                "mean": round(sum(slice_values) / len(slice_values), 3)
                if slice_values else None,
                "min": round(min(slice_values), 3) if slice_values else None},
            "n_landmarks": len(kept),
            "n_signatures": len({v[2] for v in kept.values()}),
            "n_downgraded_to_category": downgraded,
            "matches": [{"landmark_id": lid, "confidence": v[0],
                         "match_type": v[1], "signature": v[2]}
                        for lid, v in sorted(kept.items(),
                                             key=lambda kv: -kv[1][0])]}
        scores = {lid: v[0] for lid, v in kept.items()}
        tables.append(to_compatibility_table(
            tid, {lid: to_log_lr(c) for lid, c in scores.items()},
            matcher_version=f"llm_chunked_v1_{thinking_level.lower()}",
            scale=1.0, offset=0.0,
            default_log_lr=to_log_lr(max(1e-4, 1.0 - nm) / max(1, len(sigs)))))
    if set(matches) != set(queries) or {table.tracklet_id for table in tables} != set(
            queries):
        raise llm_lifecycle.IncompleteCoverageError(
            "matching aggregation did not cover every accepted tracklet")

    settings = settings_document(
        args=args,
        selected=selected,
        records=records,
        sigs=sigs,
        queries=queries,
        request_set=request_set,
        catalog_path=catalog_path,
        upstreams=upstreams,
        build_identity=document["build_identity"])
    manifest_config = {
        "phase": "canonical_results",
        "coverage": "complete",
        "n_expected": len(request_set.units),
        "n_successful": len(canonical_results),
        "n_tracklets_expected": len(queries),
        "n_tracklets_successful": len(matches),
        "request_set_fingerprint": request_set.fingerprint,
        "build_identity": document["build_identity"],
        "orchestration": orchestration,
        "resolved_stage_config": selected,
    }
    with publication.published_artifact(
            output_dir,
            kind=paths_lib.LANDMARK_MATCHES,
            dataset=args.dataset,
            version=output_version,
            generator=GENERATOR,
            git_commit=provenance.git_commit(),
            arguments=sys.argv,
            upstreams=upstreams,
            config=manifest_config,
            declared_outputs=FINAL_OUTPUTS) as builder:
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
            builder.output_path(SIGNATURES_NAME), sig_to_ids)
        artifact.atomic_write_json(
            builder.output_path(SETTINGS_NAME), settings)
        artifact.atomic_write_json(
            builder.output_path(MATCHES_NAME), matches)
        artifact.atomic_write_file(
            builder.output_path(COMPATIBILITY_NAME),
            msgspec.json.encode(tables, enc_hook=msgspec_enc_hook))
    n_hit = sum(1 for m in matches.values() if m["n_landmarks"])
    print(f"tracklets with a match above the floor: {n_hit}/{len(matches)}")
    print(f"published complete {paths_lib.LANDMARK_MATCHES} artifact: "
          f"{output_dir}")


if __name__ == "__main__":
    main()
