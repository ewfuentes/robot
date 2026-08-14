"""Freeze the pairing run's labelled matches into a recall guard for trimming.

Trimming the map is only safe if you can measure what it costs. The pairing
run (`<run>/pairing/{requests,results}.jsonl`) already contains a model's
tracklet -> map-landmark labels, which is a far better recall guard than any
heuristic: every labelled match is a map entry a real observation was matched
to, so a trim that removes one has demonstrably destroyed a match we had.

**What a positive is, precisely.** `set_2_id` in the results is an index into
that tracklet's Set 2 list, not a landmark id, and recovering the id needs the
wedge that produced the list. What survives the round trip unambiguously is the
Set 2 *line*: `format_tags(sorted(entry.tags.items()))` over
`harbor_catalog.prune_harbor_tags`. So a positive is a **tag signature**, and
the guard asks "does at least one landmark with this signature survive?".
That is weaker than pinning the exact row - 394 unnamed piers share one
signature - but it is honest, cheap, and catches exactly the failure that
matters: a trim rule that wipes out a whole class the labels rely on.

The JSONL parsing mirrors `m8_pairing_results_viewer.{parse_sets,load_results}`
and tolerates both the bare-int and `{set_2_id, match_type}` schema versions.
Unresolved signatures are reported loudly, so a prompt-format change shows up
as a visible count rather than a silently empty guard.

Example:
    bazel run //experimental/overhead_matching/swag/scripts:landmark_positive_set -- \\
        --pairing_dir /data/.../runs/r003_full_leg1/pairing \\
        --feather /data/.../landmarks/harbor_osm_enc_v1.feather \\
        --dataset_base /data/.../processed/leg1 \\
        --output /data/.../landmarks/positive_set_r003_leg1.json
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    harbor_catalog as hc,
)


def format_signature(tags: dict) -> str:
    """The Set 2 line for a tag bundle (m7's format_tags over sorted items)."""
    return "; ".join(f"{k}={v}" for k, v in sorted(tags.items()))


def parse_sets(prompt_text: str) -> tuple[list[str], list[str]]:
    """(set1, set2) as lists of tag lines, indexed as the prompt numbered them."""
    set1, set2, current = [], [], None
    for line in prompt_text.splitlines():
        if line.startswith("Set 1"):
            current = set1
            continue
        if line.startswith("Set 2"):
            current = set2
            continue
        match = re.match(r"^ (\d+)\. (.*)$", line)
        if match and current is not None:
            current.append(match.group(2))
    return set1, set2


def load_requests(pairing_dir: Path) -> dict[str, tuple[list[str], list[str]]]:
    out = {}
    with open(pairing_dir / "requests.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record["request"]["contents"][0]["parts"][0]["text"]
            out[record["key"]] = parse_sets(text)
    return out


def load_results(pairing_dir: Path) -> tuple[dict[str, list[dict]], dict[str, str]]:
    results, errors = {}, {}
    with open(pairing_dir / "results.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("key", "?")
            if record.get("error"):
                errors[key] = record["error"]
                continue
            try:
                text = record["response"]["candidates"][0]["content"][
                    "parts"][0]["text"]
                payload = json.loads(text)
                normalised = []
                for match in payload.get("matches", []):
                    pairs = []
                    for item in match.get("set_2_matches", []):
                        if isinstance(item, dict):
                            pairs.append((int(item["set_2_id"]),
                                          item.get("match_type", "instance")))
                        else:
                            pairs.append((int(item), "instance"))
                    normalised.append({
                        "set_1_id": int(match.get("set_1_id", 0)),
                        "matches": pairs,
                        "uniqueness": match.get("uniqueness_score"),
                        "negatives": match.get("negatives", []),
                    })
                results[key] = normalised
            except Exception as exc:  # noqa: BLE001
                errors[key] = f"{type(exc).__name__}: {exc}"
    return results, errors


def anchor_from_dataset(dataset_base: Path) -> tuple[float, float]:
    """Mean frame lat/lon, matching how m7 anchors its ENU frame."""
    with open(dataset_base / "frames_gps.csv") as f:
        rows = list(csv.DictReader(f))
    return (sum(float(r["latitude"]) for r in rows) / len(rows),
            sum(float(r["longitude"]) for r in rows) / len(rows))


def catalog_signature_index(entries) -> dict[str, list[str]]:
    """Set 2 line -> landmark ids that produce it."""
    index: dict[str, list[str]] = {}
    for entry in entries:
        index.setdefault(format_signature(entry.tags), []).append(
            entry.landmark_id)
    return index


def build(pairing_dir: Path, feather: Path, dataset_base: Path) -> dict:
    anchor_lat, anchor_lon = anchor_from_dataset(dataset_base)
    print(f"anchor {anchor_lat:.5f},{anchor_lon:.5f}")
    entries = hc.load_catalog_cached(feather, anchor_lat, anchor_lon)
    index = catalog_signature_index(entries)
    print(f"{len(entries)} catalog entries, {len(index)} distinct signatures")

    requests = load_requests(pairing_dir)
    results, errors = load_results(pairing_dir)
    print(f"{len(requests)} requests, {len(results)} labelled, {len(errors)} errors")

    positives, negatives, unresolved = [], [], []
    kinds = Counter()
    for key, matches in sorted(results.items()):
        _, set2 = requests.get(key, ([], []))
        for match in matches:
            for set2_id, kind in match["matches"]:
                if not 0 <= set2_id < len(set2):
                    unresolved.append({"tracklet": key, "set_2_index": set2_id,
                                       "reason": "index out of range"})
                    continue
                signature = set2[set2_id]
                ids = index.get(signature, [])
                kinds[kind] += 1
                record = {"tracklet": key, "set_2_index": set2_id,
                          "match_type": kind, "signature": signature,
                          "landmark_ids": ids,
                          "uniqueness": match.get("uniqueness")}
                if ids:
                    positives.append(record)
                else:
                    record["reason"] = "signature not found in catalog"
                    unresolved.append(record)
            for neg in match["negatives"]:
                set2_id = int(neg.get("set_2_id", -1))
                if 0 <= set2_id < len(set2):
                    negatives.append({"tracklet": key,
                                      "signature": set2[set2_id],
                                      "difficulty": neg.get("difficulty", "?")})

    print(f"\npositives: {len(positives)} ({dict(kinds)}), "
          f"unresolved: {len(unresolved)}, negatives: {len(negatives)}")
    by_signature = Counter(p["signature"] for p in positives)
    print(f"distinct positive signatures: {len(by_signature)}")
    for signature, count in by_signature.most_common(15):
        n_rows = len(index.get(signature, []))
        print(f"  x{count:2d}  [{n_rows:4d} row(s)]  {signature[:88]}")
    if unresolved:
        print("\nunresolved (these would silently weaken the guard):")
        for record in unresolved[:10]:
            print(f"  {record['tracklet']} #{record['set_2_index']}: "
                  f"{record.get('reason')}")

    return {
        "pairing_dir": str(pairing_dir),
        "feather": str(feather),
        "anchor": [anchor_lat, anchor_lon],
        "n_tracklets_labelled": len(results),
        "positives": positives,
        "negatives": negatives,
        "unresolved": unresolved,
    }


def recall(positive_set: dict, surviving_signatures: set[str]) -> tuple[float, list[dict]]:
    """Fraction of distinct positive signatures with a surviving landmark."""
    signatures = {p["signature"] for p in positive_set["positives"]}
    if not signatures:
        return 1.0, []
    lost = sorted(signatures - surviving_signatures)
    kept = len(signatures) - len(lost)
    detail = [p for p in positive_set["positives"] if p["signature"] in set(lost)]
    return kept / len(signatures), detail


def main(pairing_dir: Path, feather: Path, dataset_base: Path, output: Path) -> dict:
    positive_set = build(pairing_dir, feather, dataset_base)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(positive_set, f, indent=2)
    print(f"\nWrote {output}")
    return positive_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairing_dir", required=True, type=Path)
    parser.add_argument("--feather", required=True, type=Path)
    parser.add_argument("--dataset_base", required=True, type=Path,
                        help="leg dir with frames_gps.csv, for the ENU anchor")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    main(args.pairing_dir, args.feather, args.dataset_base, args.output)
