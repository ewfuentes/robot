"""Match tracklets against the whole map, position-free, via chunked LLM calls.

This is the matcher. It takes merged landmarks out of M6 and produces a
`CompatibilityTable` per tracklet for the bearing-only filter.

**No spatial information is used anywhere.** An earlier design gated
candidates by a bearing wedge computed from GPS, which leaked the answer:
selecting candidates by where the vessel was, then asking the filter to
recover where the vessel was, is circular. Set 2 here is the map, full stop.

Shape of the work: the map is far larger than a prompt, so it is split into
signature chunks and the tracklets into small batches, and every batch is
asked about every chunk. Distinct *signatures* are matched rather than rows -
identical tag bundles are things the model cannot tell apart by construction -
and a matched signature expands to every landmark carrying it.

Outputs under <run_dir>/matching/:
  requests.jsonl        one per (tracklet batch x map chunk)
  results.jsonl         raw model output
  signatures.json       signature -> [landmark_id, ...] expansion table
  matches.json          per tracklet: matched landmark_ids, confidence, type
  compatibility.json    per tracklet: the filter's CompatibilityTable

Run:
  bazel run //...object_tracking:m9_match_landmarks -- \\
      --run_dir <runs>/r003_full_leg1 --build_only
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from experimental.overhead_matching.swag.data import farfield_paths
from experimental.overhead_matching.swag.data import landmark_schema
from experimental.overhead_matching.swag.scripts import (
    vertex_batch_manager as vbm,
)
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    bearing_matcher as bm,
    harbor_catalog as hc,
    semantic_audit as sa,
)

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
    "properties": {"matches": {"type": "array", "items": {
        "type": "object",
        "required": ["set_1_id", "set_2_matches", "no_match_confidence",
                     "uniqueness_score"],
        "properties": {
            "set_1_id": {"type": "integer"},
            "set_2_matches": {"type": "array", "items": {
                "type": "object",
                "required": ["set_2_id", "match_type", "confidence"],
                "properties": {
                    "set_2_id": {"type": "integer"},
                    "match_type": {"type": "string",
                                   "enum": ["instance", "category"]},
                    "confidence": {"type": "number"}}}},
            "no_match_confidence": {"type": "number"},
            "uniqueness_score": {"type": "integer"}}}}}}


def signature(tags: dict) -> str:
    return "; ".join(f"{k}={v}" for k, v in sorted(tags.items()))


def build_map_signatures(feather_path: Path):
    """signature -> [landmark_id, ...]. Identical bundles are indistinguishable
    to a text matcher, so they are asked about once and expanded after.

    Tags come from `landmark_schema.tag_dicts`, the shared reader, rather than
    from this module's own parse of the columns. The feather moved from ~1700
    sparse tag columns to a single JSON `tags` column, and a private reader
    silently survived that change: it saw one unparsed column, matched almost
    nothing against the keep-list, and reported 13,210 landmarks as 16 distinct
    signatures. Nothing raised. Use the shared reader so a schema change breaks
    loudly in one place instead of quietly everywhere.
    """
    frame = landmark_schema.read_frame(Path(feather_path))
    records = landmark_schema.tag_dicts(frame)
    ids = frame["id"].values
    sources = frame["landmark_type"].values
    table = defaultdict(list)
    for i in range(len(frame)):
        tags = hc.prune_far_field_tags(records[i])
        if not tags:
            continue
        source = "enc" if sources[i] == "enc" else "osm"
        text = hc._id_text(ids[i])
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


def query_bundles(run_dir: Path, min_supports: int):
    """tracklet_id -> Set 1 block. AUDITED TRACKLETS ONLY.

    There is deliberately no fallback to raw tag votes. A track that was
    never audited has no canonical semantics, no reconciled name, and no
    description - matching it would mean matching un-adjudicated detector
    output, and any match it earned could not be trusted downstream. Such
    tracks are dropped from the pipeline, not silently downgraded.
    """
    landmarks = json.loads(
        (run_dir / "merged" / "landmarks.json").read_text())
    meta_path = run_dir / "semantic_audit" / "audit_meta.json"
    results = run_dir / "semantic_audit" / "results.jsonl"
    if not (meta_path.exists() and results.exists()):
        raise SystemExit(
            f"no semantic audit under {run_dir}; matching consumes audit "
            "output only - run m5 and its Vertex pass first")
    meta = json.loads(meta_path.read_text())
    by_track = {v["track_id"]: k for k, v in meta.items()}
    raw = {}
    with open(results) as f:
        for line in f:
            if line.strip():
                key, audit, _ = sa.parse_result_line(json.loads(line))
                if audit:
                    raw[key] = audit

    out, skipped = {}, 0
    for lm in landmarks:
        if lm["n_supports"] < min_supports:
            continue
        audit = None
        for tid in lm["track_ids"]:
            if by_track.get(tid) in raw:
                audit = raw[by_track[tid]]
                break
        if audit is None:
            skipped += 1
            continue
        if audit.get("verdict") == "drop":
            skipped += 1
            continue
        out[lm["landmark_id"]] = format_query(audit)
    if skipped:
        print(f"  skipped {skipped} landmarks with no usable audit "
              "(un-audited or verdict=drop)")
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
                "request": {
                    "contents": [{"parts": [{"text": user}], "role": "user"}],
                    "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseSchema": SCHEMA,
                        "thinkingConfig": {"thinkingLevel": thinking}}}})
    return records


def aggregate(results_path, records_by_key, sig_chunks, sig_to_ids):
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
    errors = 0
    with open(results_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            meta = records_by_key.get(record.get("key"))
            if meta is None or record.get("error"):
                errors += 1
                continue
            try:
                text = record["response"]["candidates"][0]["content"][
                    "parts"][0]["text"]
                payload = json.loads(text)
            except Exception:  # noqa: BLE001
                errors += 1
                continue
            chunk = sig_chunks[meta["chunk_index"]]
            for entry in payload.get("matches", []):
                idx = int(entry.get("set_1_id", -1))
                if not 0 <= idx < len(meta["batch_keys"]):
                    continue
                tid = meta["batch_keys"][idx]
                if entry.get("no_match_confidence") is not None:
                    no_match[tid].append(float(entry["no_match_confidence"]))
                for match in entry.get("set_2_matches", []):
                    sid = int(match.get("set_2_id", -1))
                    if not 0 <= sid < len(chunk):
                        continue
                    conf = float(match.get("confidence") or 0.0)
                    kind = match.get("match_type", "category")
                    for landmark_id in sig_to_ids.get(chunk[sid], []):
                        prev = per_tracklet[tid].get(landmark_id)
                        if prev is None or conf > prev[0]:
                            per_tracklet[tid][landmark_id] = (
                                conf, kind, chunk[sid])
    return per_tracklet, no_match, errors


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


def to_log_lr(confidence, clip=4.0):
    """Confidence -> log odds, clipped. The seam the design doc specifies is
    an uncalibrated matcher behind a tuned transform; this is that transform,
    and the clip is what keeps a confident-but-wrong match survivable."""
    c = min(max(confidence, 1e-4), 1 - 1e-4)
    return max(-clip, min(clip, math.log(c / (1 - c))))


def write_settings(out: Path, args, paths, records, sigs):
    """Everything needed to reproduce this matching run, beside its outputs.

    Without this the artifact was unreproducible: `request_meta.json` holds only
    a key-to-chunk mapping, and the model was never recorded anywhere at all
    (`results.jsonl` carries request and response but no model field), so
    "which model produced these matches" had no answer from the artifact. The
    prompt is hashed rather than named for the same reason the extraction's is:
    `SYSTEM_PROMPT` is a module constant that can be edited in place.
    """
    settings = {
        "generator": ("//experimental/overhead_matching/swag/landmark_filtering/"
                      "object_tracking:m9_match_landmarks"),
        "git_commit": farfield_paths.git_commit(),
        "dataset": paths.dataset,
        "run": out.parent.name,
        "feather": str(paths.feather),
        "model": args.model,
        "submitted_by_m9": bool(args.submit),
        "transport": "online" if args.online else "batch",
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode()).hexdigest(),
        "thinking_level": args.thinking_level,
        "min_supports": args.min_supports,
        "query_batch": args.query_batch,
        "chunk_size": args.chunk_size,
        "confidence_floor": args.confidence_floor,
        "instance_max_rows": args.instance_max_rows,
        "n_requests": len(records),
        "n_signatures": len(sigs),
        "spatial_gating": ("none - set 2 is the whole catalog, so the catalog's "
                           "extent bounds what can match"),
    }
    (out / "settings.json").write_text(json.dumps(settings, indent=1) + "\n")
    print(f"wrote {out}/settings.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    farfield_paths.add_arguments(parser, feather=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--min_supports", type=int, default=1)
    parser.add_argument("--query_batch", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--thinking_level", default="HIGH")
    parser.add_argument("--confidence_floor", type=float, default=0.05,
                        help="Matches below this are dropped from the table")
    parser.add_argument("--instance_max_rows", type=int, default=5,
                        help="A signature covering more than this many map "
                             "rows cannot be an 'instance' match by "
                             "definition - the tags do not identify one "
                             "object - so the label is downgraded to "
                             "'category' in code rather than trusted.")
    vbm.add_execution_arguments(parser)
    parser.add_argument("--submit", action="store_true",
                        help="Execute the requests and carry straight on to "
                             "aggregation, instead of stopping so "
                             "vertex_batch_manager can be invoked by hand. "
                             "Batch by default; --online for on-demand. "
                             "Resumable: reruns retry only failures.")
    parser.add_argument("--build_only", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()
    paths = farfield_paths.resolve(
        parser, args, infer_from=args.run_dir,
        require=("dataset_base", "feather"))

    out = args.run_dir / "matching"
    out.mkdir(parents=True, exist_ok=True)

    sig_to_ids = build_map_signatures(paths.feather)
    sigs = sorted(sig_to_ids)
    sig_chunks = [sigs[i:i + args.chunk_size]
                  for i in range(0, len(sigs), args.chunk_size)]
    queries = query_bundles(args.run_dir, args.min_supports)
    print(f"map: {sum(len(v) for v in sig_to_ids.values())} landmarks -> "
          f"{len(sigs)} signatures in {len(sig_chunks)} chunks")
    print(f"queries: {len(queries)} tracklets in "
          f"{math.ceil(len(queries) / args.query_batch)} batches")

    records = build_requests(queries, sig_chunks, args.query_batch,
                             args.thinking_level)
    print(f"requests: {len(records)}")
    if not args.aggregate_only:
        with open(out / "requests.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps({"key": r["key"],
                                    "request": r["request"]}) + "\n")
        (out / "request_meta.json").write_text(json.dumps(
            {r["key"]: {"batch_keys": r["batch_keys"],
                        "chunk_index": r["chunk_index"]} for r in records}))
        (out / "signatures.json").write_text(json.dumps(sig_to_ids))
        write_settings(out, args, paths, records, sigs)
        est = sum(len(r["request"]["contents"][0]["parts"][0]["text"])
                  for r in records) // 4
        print(f"estimated input: ~{est / 1e6:.2f}M tokens "
              f"(+ system {len(SYSTEM_PROMPT) // 4 * len(records) / 1e6:.2f}M)")
    if args.build_only:
        print(f"\nwrote {out}/requests.jsonl - submit with vertex_batch_manager")
        return

    results_path = out / "results.jsonl"
    if args.submit:
        vbm.run_requests(args, out / "requests.jsonl", results_path,
                         tag=f"{paths.dataset}_matching_{args.run_dir.name}")
    if not results_path.exists():
        raise SystemExit(
            f"no results at {results_path}; submit the batch first (--submit, "
            f"or vertex_batch_manager run-online)")
    meta = {r["key"]: {"batch_keys": r["batch_keys"],
                       "chunk_index": r["chunk_index"]} for r in records}
    per_tracklet, no_match, errors = aggregate(
        results_path, meta, sig_chunks, sig_to_ids)
    print(f"aggregated: {len(per_tracklet)} tracklets with >=1 match, "
          f"{errors} unusable responses")

    matches = {}
    tables = {}
    for tid in sorted(queries):
        got = per_tracklet.get(tid, {})
        kept = {}
        downgraded = 0
        for lid, (conf, kind, sig) in got.items():
            if conf < args.confidence_floor:
                continue
            if kind == "instance" and len(
                    sig_to_ids.get(sig, [])) > args.instance_max_rows:
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
        tables[tid] = bm.to_compatibility_table(
            tid, {lid: to_log_lr(c) for lid, c in scores.items()},
            matcher_version=f"llm_chunked_v1_{args.thinking_level.lower()}",
            scale=1.0, offset=0.0,
            default_log_lr=to_log_lr(max(1e-4, 1.0 - nm) / max(1, len(sigs))))
    (out / "matches.json").write_text(json.dumps(matches, indent=1))
    (out / "compatibility.json").write_text(json.dumps(tables, indent=1))
    n_hit = sum(1 for m in matches.values() if m["n_landmarks"])
    print(f"tracklets with a match above the floor: {n_hit}/{len(matches)}")
    print(f"wrote {out}/matches.json and compatibility.json")


if __name__ == "__main__":
    main()
