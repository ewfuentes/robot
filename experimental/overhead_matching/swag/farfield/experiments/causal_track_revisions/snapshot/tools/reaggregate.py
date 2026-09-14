#!/usr/bin/env python3
"""Rebuild compatibility tables from an existing landmark_matches artifact.

Offline re-aggregation of the model's already-paid responses. Policies:
  baseline       reproduce compatibility.json (sanity check)
  instance_only  keep only match_type == instance rows (category evidence -> uninformative)
  catexpand      every category endorsement expands to ALL catalog signatures sharing its
                 primary category (key=value), at the endorsing signature's confidence.
                 Fixes the per-chunk recall lottery.
  catexpand_flat like catexpand but all category rows get one flat confidence (--cat_conf)
Common options: --min_conf drops endorsements below a threshold before expansion.
"""
import argparse, json, math, collections
from pathlib import Path

CLIP = 4.0
# order matters: first present key names the category
KIND_KEYS = {"generator:source", "generator:method", "power", "man_made", "aeroway", "natural", "water",
             "landuse", "leisure", "amenity", "tourism", "historic", "railway", "waterway", "industrial",
             "bridge", "highway", "military", "building", "place", "seamark:type", "object_class",
             "tower:type", "tower:construction", "crane:type", "attraction", "sport"}
CATEGORY_KEYS = ["generator:source", "power", "man_made", "aeroway", "natural", "water",
                 "landuse", "leisure", "amenity", "tourism", "historic", "railway", "waterway",
                 "industrial", "bridge", "highway", "military", "building", "place",
                 "seamark:type", "object_class"]

def to_log_lr(c, clip=CLIP, clip_lo=None):
    c = min(max(c, 1e-6), 1 - 1e-6)
    lo = -clip if clip_lo is None else clip_lo
    return max(lo, min(clip, math.log(c / (1 - c))))

def category_of(tags):
    for k in CATEGORY_KEYS:
        if k in tags:
            v = tags[k]
            if k == "bridge" and v == "yes" and "highway" in tags:
                return ("bridge=yes",)  # any bridge
            return (f"{k}={v}",)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches_dir", required=True)
    ap.add_argument("--policy", required=True,
                    choices=["baseline", "instance_only", "catexpand", "catexpand_flat", "catexpand_divided", "catexpand_common", "catexpand_divided2", "catexpand_split"])
    ap.add_argument("--min_conf", type=float, default=0.0)
    ap.add_argument("--cat_conf", type=float, default=0.1)
    ap.add_argument("--rho", type=float, default=0.5,
                    help="catexpand_split: share of a kind's confidence kept on the rows the model picked")
    ap.add_argument("--clip_lo", type=float, default=-CLIP,
                    help="table clip floor; catexpand_divided needs a lower floor than -4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.matches_dir)
    matches = json.load(open(d / "matches.json"))
    sig = json.load(open(d / "signatures.json"))
    comp = json.load(open(d / "compatibility.json"))
    by_tid = {t["tracklet_id"]: t for t in comp}
    sig_cat = {s: category_of(v["canonical_tags"]) for s, v in sig.items()}
    cat_sigs = collections.defaultdict(list)
    for s, c in sig_cat.items():
        if c: cat_sigs[c].append(s)
    n_sig = len(sig)
    sig_kindtags = {s: {f"{k}={v}" for k, v in v_["canonical_tags"].items() if k in KIND_KEYS}
                    for s, v_ in sig.items()}
    out = []
    stats = collections.Counter()
    for tid, rec in matches.items():
        base = by_tid[tid]
        rows = {}  # landmark_id -> confidence
        cats = {}  # category -> best conf among category endorsements
        for m in rec["matches"]:
            c = m["aggregate_confidence"]
            if c < a.min_conf: continue
            if m["match_type"] == "instance":
                rows[m["landmark_id"]] = max(rows.get(m["landmark_id"], 0), c)
            else:
                if a.policy == "instance_only": continue
                if a.policy == "baseline":
                    rows[m["landmark_id"]] = max(rows.get(m["landmark_id"], 0), c)
                else:
                    cat = sig_cat.get(m["signature_id"])
                    if cat is None:  # no recognisable category: keep the row as-is
                        rows[m["landmark_id"]] = max(rows.get(m["landmark_id"], 0), c)
                    else:
                        cats[cat] = max(cats.get(cat, 0), c)
        if a.policy == "catexpand_common":
            # kind = the kind-tags shared by every endorsed signature with the
            # same primary key; expand to every signature carrying all of them
            clusters = collections.defaultdict(list)
            for m in rec["matches"]:
                if m["aggregate_confidence"] < a.min_conf or m["match_type"] == "instance":
                    continue
                cat = sig_cat.get(m["signature_id"])
                if cat is not None:
                    clusters[cat].append(m)
            cats = {}
            for cat, ms in clusters.items():
                tagsets = [{f"{k}={v}" for k, v in sig[m["signature_id"]]["canonical_tags"].items()
                            if k in KIND_KEYS} for m in ms]
                common = frozenset(set.intersection(*tagsets)) if tagsets else frozenset()
                if not common:
                    common = frozenset(cat)
                cats[common] = max(m["aggregate_confidence"] for m in ms)
            for common, c in cats.items():
                members = [s for s, ts in sig_kindtags.items() if common <= ts]
                n_rows = sum(len(sig[s]["landmark_ids"]) for s in members)
                conf = c / max(1, n_rows)
                for s in members:
                    for lid in sig[s]["landmark_ids"]:
                        if lid not in rows:
                            rows[lid] = max(rows.get(lid, 0), conf)
                stats["expanded_categories"] += 1
            cats = {}
        if a.policy == "catexpand_split":
            # picked rows of a kind share rho*c; the kind's unpicked rows share (1-rho)*c
            by_kind = collections.defaultdict(list)
            for m in rec["matches"]:
                if m["aggregate_confidence"] < a.min_conf or m["match_type"] == "instance":
                    continue
                cat = sig_cat.get(m["signature_id"])
                if cat is not None:
                    by_kind[cat].append(m)
            for cat, ms in by_kind.items():
                c = max(m["aggregate_confidence"] for m in ms)
                picked = {m["signature_id"] for m in ms}
                picked_rows = sum(len(sig[s_]["landmark_ids"]) for s_ in picked)
                other = [s_ for s_ in cat_sigs[cat] if s_ not in picked]
                other_rows = sum(len(sig[s_]["landmark_ids"]) for s_ in other)
                rho = a.rho if other_rows else 1.0
                for m in ms:
                    for lid in sig[m["signature_id"]]["landmark_ids"]:
                        if lid not in rows:
                            rows[lid] = max(rows.get(lid, 0), m["aggregate_confidence"] * rho / max(1, picked_rows))
                for s_ in other:
                    for lid in sig[s_]["landmark_ids"]:
                        if lid not in rows:
                            rows[lid] = max(rows.get(lid, 0), c * (1 - rho) / max(1, other_rows))
                stats["expanded_categories"] += 1
            cats = {}
        if a.policy == "catexpand_divided2":
            # a kind cluster whose endorsed named signatures all carry ONE name is
            # several rows of one object: an identity claim, kept at its confidence
            by_kind = collections.defaultdict(list)
            for m in rec["matches"]:
                if m["aggregate_confidence"] < a.min_conf or m["match_type"] == "instance":
                    continue
                cat = sig_cat.get(m["signature_id"])
                if cat is not None:
                    by_kind[cat].append(m)
            for cat, ms in by_kind.items():
                named = [(m, sig[m["signature_id"]]["canonical_tags"].get("name")) for m in ms]
                names = {n for _, n in named if n}
                n_named = sum(1 for _, n in named if n)
                if len(names) == 1 and n_named >= 2:
                    for m, n in named:
                        if n:
                            for lid in sig[m["signature_id"]]["landmark_ids"]:
                                rows[lid] = max(rows.get(lid, 0), m["aggregate_confidence"])
                    stats["single_object_kinds"] += 1
        for cat, c in cats.items():
            n_rows = sum(len(sig[s]["landmark_ids"]) for s in cat_sigs[cat])
            if a.policy == "catexpand_flat":
                conf = a.cat_conf
            elif a.policy in ("catexpand_divided", "catexpand_divided2"):
                conf = c / max(1, n_rows)  # the kind's rows collectively carry ~c
            else:
                conf = c
            for s in cat_sigs[cat]:
                for lid in sig[s]["landmark_ids"]:
                    if lid not in rows:  # instance rows keep their own confidence
                        rows[lid] = max(rows.get(lid, 0), conf)
            stats["expanded_categories"] += 1
        stats["rows"] += len(rows)
        best = max(rows.values(), default=0.0)
        nm = round(1.0 - best, 4) if rows else rec["aggregate_no_match_confidence"]
        default_log_lr = to_log_lr(max(1e-4, 1.0 - nm) / max(1, n_sig), clip_lo=a.clip_lo)
        entries = [{"kind": "CompatibilityEntry", "landmark_id": lid, "log_lr": to_log_lr(c, clip_lo=a.clip_lo)}
                   for lid, c in rows.items()]
        entries = [e for e in entries if abs(e["log_lr"] - default_log_lr) > 1e-9]
        entries.sort(key=lambda e: (-e["log_lr"], e["landmark_id"]))
        out.append({"kind": "CompatibilityTable", "tracklet_id": tid,
                    "matcher_version": base["matcher_version"] + f"+reagg_{a.policy}",
                    "entries": entries, "default_log_lr": default_log_lr,
                    "clip_lo": a.clip_lo, "clip_hi": CLIP, "status": base["status"]})
    json.dump(out, open(a.out, "w"))
    print(f"{a.policy}: {len(out)} tables, {stats['rows']} rows, "
          f"{stats['expanded_categories']} category expansions, "
          f"{stats['single_object_kinds']} single-object kinds -> {a.out}")
    if a.policy == "baseline":
        diffs = 0
        for t in out:
            b = by_tid[t["tracklet_id"]]
            be = {(e["landmark_id"], round(e["log_lr"], 6)) for e in b["entries"]}
            te = {(e["landmark_id"], round(e["log_lr"], 6)) for e in t["entries"]}
            if be != te or abs(b["default_log_lr"] - t["default_log_lr"]) > 1e-6: diffs += 1
        print(f"baseline check: {diffs} tables differ from compatibility.json")

if __name__ == "__main__":
    main()
