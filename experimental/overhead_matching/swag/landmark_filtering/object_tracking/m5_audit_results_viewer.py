"""Review site for per-track semantic audit results.

Joins <run_dir>/semantic_audit/{results.jsonl, requests.jsonl,
audit_meta.json} with the run artifact and renders

  <run_dir>/semantic_audit/review/index.html

Contents:
- headline stats (verdicts, kinds, extents, strikes, secondaries)
- a NAME / ALIAS COLLISION table: every name claimed by more than one track,
  with the role it was claimed in (primary vs alias). Cross-role collisions
  (one track's alias is another track's primary name) are the alias-theft
  signature and are highlighted.
- per-track sections grouped by verdict (drops first, then keep_partial,
  then keeps with edits, then clean keeps): the model's canonical record, a
  lifetime bar showing valid_segments / strike marks / secondary-object
  marks, the chips the model saw, and a chip for every strike and secondary
  object so each claim can be verified visually.

Strike / secondary chips are rendered on demand (requires the dataset for
pano decoding); pass --no_extra_chips to skip that and reuse only the chips
already rendered for the requests.

Run:
  bazel run //...object_tracking:m5_audit_results_viewer -- \\
      --run_dir <runs>/r002_full_leg1
"""

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.landmark_filtering import ingest
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    semantic_audit as sa,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    IngestConfig,
)

DEFAULT_DATASET = Path("/data/farfield_matching/boston_harbor_dataset/processed/leg1")
DEFAULT_LANDMARKS = Path(
    "/data/farfield_matching/boston_harbor_dataset/panorama_landmarks/boston_harbor_leg1")

VERDICT_CSS = {"keep": "#3c3", "keep_partial": "#fa2", "drop": "#e55"}


def esc(x):
    return html.escape(str(x))


def load_results(audit_dir: Path):
    audits, errors = {}, {}
    with open(audit_dir / "results.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            key, audit, err = sa.parse_result_line(json.loads(line))
            if audit is not None:
                audits[key] = audit
            else:
                errors[key] = err
    return audits, errors


def load_request_texts(audit_dir: Path):
    """key -> (dossier_text, [caption, ...]) from the request file."""
    texts = {}
    with open(audit_dir / "requests.jsonl") as f:
        for line in f:
            r = json.loads(line)
            parts = r["request"]["contents"][0]["parts"]
            captions = [p["text"] for p in parts[1:] if "text" in p]
            texts[r["key"]] = (parts[0]["text"], captions)
    return texts


def collect_extra_chip_wants(audits, meta):
    """{key: sorted set of t} needing chips beyond the request chips: every
    strike t and up to 2 ts per secondary object."""
    wants = defaultdict(set)
    for key, audit in audits.items():
        chipped = {int(t) for t in meta[key].get("chipped_ts", [])}
        for strike in audit["strike_votes"]:
            wants[key].add(strike["t"])
        for sec in audit["secondary_objects"]:
            for t in sorted(sec["ts"])[:2]:
                wants[key].add(t)
        wants[key] -= chipped
        if not wants[key]:
            del wants[key]
    return wants


def render_extra_chips(wants, artifact, meta, dataset_base, landmark_base,
                       chips_dir, cfg):
    """Render chips for (key, t) pairs. Returns {(key, t): filename}."""
    if not wants:
        return {}
    result = ingest.run_ingest(dataset_base, landmark_base, IngestConfig())
    obs_by_id = {o.obs_id: o for o in result.observations}
    frames_by_idx = {f.frame_idx: f for f in result.frames}
    tracks_by_id = {t["track_id"]: t for t in artifact["tracks"]}

    probe = Image.open(dataset_base / "panorama"
                       / f"{result.frames[0].pano_stem}.jpg")
    pano_w, pano_h = probe.size

    by_keyframe = defaultdict(list)
    for key, ts in wants.items():
        track = tracks_by_id[meta[key]["track_id"]]
        supports, context = sa.collect_evidence(track, obs_by_id, cfg)
        by_t = defaultdict(list)
        for e in supports + context:
            by_t[e["t"]].append(e)
        for t in ts:
            if t in by_t:
                e = by_t[t][0]
                by_keyframe[e["keyframe"]].append((key, t, e))

    rendered = {}
    for keyframe in sorted(by_keyframe):
        frame = frames_by_idx.get(keyframe)
        if frame is None:
            continue
        pano = np.asarray(Image.open(
            dataset_base / "panorama" / f"{frame.pano_stem}.jpg"))
        for key, t, e in by_keyframe[keyframe]:
            det_box, mask_box = sa.chip_boxes_for_entry(
                e, e["obs"], pano_w, pano_h)
            name = f"{key}_t{t:04d}_extra.jpg"
            sa.render_chip(pano, det_box, mask_box, chips_dir / name,
                           cfg.chip_height_px)
            rendered[(key, t)] = name
    return rendered


def kf_link(birth_keyframe, t, label=None):
    """Link a relative time index to its keyframe page (from review/)."""
    kf = birth_keyframe + t
    return (f"<a href='../../keyframes/f{kf:04d}.html'>"
            f"{label if label is not None else f't{t}'}</a>")


def timeline_bar(lifetime, segments, strike_ts, secondary_ts):
    """Lifetime bar: valid segments green, strikes red ticks, secondary-
    object detections purple ticks."""
    parts = ["<div class='tlbar'>"]
    for seg in segments:
        left = 100.0 * seg["start_t"] / lifetime
        width = 100.0 * (seg["end_t"] - seg["start_t"] + 1) / lifetime
        parts.append(f"<div class='tlseg' style='left:{left:.1f}%;"
                     f"width:{width:.1f}%'></div>")
    for t, cls in [(t, "tlstrike") for t in strike_ts] + \
                  [(t, "tlsec") for t in secondary_ts]:
        left = 100.0 * t / lifetime
        parts.append(f"<div class='{cls}' style='left:{left:.1f}%'></div>")
    parts.append("</div>")
    return "".join(parts)


def name_collision_rows(audits):
    """[(name, [(key, role), ...])] for names claimed by more than one
    track; role is 'primary' or 'alias'."""
    claims = defaultdict(list)
    for key, audit in audits.items():
        po = audit["primary_object"]
        if po["name"]:
            claims[po["name"].strip().lower()].append(
                (po["name"], key, "primary"))
        for alias in po["name_aliases"]:
            if alias.strip():
                claims[alias.strip().lower()].append((alias, key, "alias"))
    rows = []
    for entries in claims.values():
        if len(entries) > 1:
            display = entries[0][0]
            rows.append((display, [(k, role) for _, k, role in entries]))
    rows.sort(key=lambda r: (-len({role for _, role in r[1]}), r[0]))
    return rows


def chip_div(rel_path, caption):
    return (f"<div class='chip'><img src='{rel_path}' loading='lazy'>"
            f"{caption}</div>")


def track_section(key, audit, meta_entry, track, texts, extra_chips):
    parts = []
    po = audit["primary_object"]
    verdict = audit["verdict"]
    lifetime = track["end_keyframe"] - track["birth_keyframe"] + 1
    strike_ts = [s["t"] for s in audit["strike_votes"]]
    sec_ts = sorted({t for s in audit["secondary_objects"] for t in s["ts"]})

    tid = meta_entry["track_id"]
    parts.append(
        f"<h2 id='{key}'><span class='v_{verdict}'>{verdict}</span> {key} "
        f"&mdash; {esc(po['name'] or po['tags'][0]['tag'] if po['tags'] else '?')}"
        f"</h2>")
    parts.append(
        f"<p>{esc(audit['landmark_kind'])} | extent: {esc(po['extent'])} | "
        f"confidence: {esc(audit['confidence'])} | single_object: "
        f"{audit['single_object']} | drop_reason: "
        f"{esc(audit['drop_reason'])} | supports: "
        f"{meta_entry['n_supports']} | "
        f"<a href='../../track_full_leg1_T{tid}.html'>track page</a> | "
        f"<a href='../preview/index.html#T{tid}'>request preview</a> | "
        f"born {kf_link(meta_entry['birth_keyframe'], 0, 'keyframe page')}"
        "</p>")

    segs = ", ".join(f"t{s['start_t']}..t{s['end_t']}"
                     for s in audit["valid_segments"]) or "(none)"
    parts.append(f"<p>valid segments over t0..t{lifetime - 1}: {segs}</p>")
    parts.append(timeline_bar(lifetime, audit["valid_segments"], strike_ts,
                              sec_ts))

    tag_txt = ", ".join(f"{esc(t['tag'])} ({t['weight']:.2f})"
                        for t in po["tags"])
    parts.append(f"<p><b>tags:</b> {tag_txt}</p>")
    if po["name"]:
        alias_txt = (" | aliases: " + ", ".join(
            f"'{esc(a)}'" for a in po["name_aliases"])
            if po["name_aliases"] else "")
        parts.append(f"<p><b>name:</b> '{esc(po['name'])}'{alias_txt}</p>")
    parts.append(f"<p><b>description:</b> {esc(po['description'])}<br>"
                 f"<b>features:</b> "
                 f"{esc(', '.join(po['distinctive_features']))}</p>")
    if audit["unresolved"]:
        parts.append(f"<p class='unres'><b>unresolved:</b> "
                     f"{esc(audit['unresolved'])}</p>")

    dossier_text, captions = texts.get(key, ("", []))
    chip_files = [Path(p).name for p in meta_entry.get("chips", [])]
    if chip_files:
        parts.append("<details><summary>chips the model saw "
                     f"({len(chip_files)})</summary>")
        for fname, caption in zip(chip_files, captions):
            parts.append(chip_div(f"../chips/{fname}", esc(caption)))
        parts.append("</details>")
    if dossier_text:
        parts.append("<details><summary>dossier text</summary>"
                     f"<pre>{esc(dossier_text)}</pre></details>")

    birth = meta_entry["birth_keyframe"]
    if audit["strike_votes"]:
        parts.append("<h3>strikes</h3>")
        for s in audit["strike_votes"]:
            chip = extra_chips.get((key, s["t"]))
            caption = (f"<b class='v_drop'>strike "
                       f"{kf_link(birth, s['t'])}</b> {esc(s['reason'])}")
            if chip:
                parts.append(chip_div(f"../chips/{chip}", caption))
            else:
                parts.append(f"<p>{caption} <i>(no chip)</i></p>")

    if audit["secondary_objects"]:
        parts.append("<h3>secondary objects</h3>")
        for s in audit["secondary_objects"]:
            tag_txt = ", ".join(f"{esc(t['tag'])} ({t['weight']:.2f})"
                                for t in s["tags"])
            own = ("own landmark" if s["worth_own_landmark"]
                   else "not own landmark")
            ts_txt = ", ".join(kf_link(birth, t) for t in s["ts"])
            head = (f"<b class='sec'>{esc(s['relation'])}</b> [{own}] "
                    f"{tag_txt}"
                    + (f" '{esc(s['name'])}'" if s["name"] else "")
                    + f" @ {ts_txt}<br>{esc(s['description'])}")
            shown = False
            for t in sorted(s["ts"])[:2]:
                chip = extra_chips.get((key, t))
                if chip:
                    parts.append(chip_div(f"../chips/{chip}", head))
                    shown = True
                    break
            if not shown:
                parts.append(f"<p>{head}</p>")
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--dataset_base", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--landmark_base", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--no_extra_chips", action="store_true",
                        help="Skip rendering chips for strikes/secondaries")
    args = parser.parse_args()

    audit_dir = args.run_dir / "semantic_audit"
    audits, errors = load_results(audit_dir)
    meta = json.loads((audit_dir / "audit_meta.json").read_text())
    texts = load_request_texts(audit_dir)
    artifact = json.loads(
        next(args.run_dir.glob("tracks_*.json")).read_text())
    tracks_by_id = {t["track_id"]: t for t in artifact["tracks"]}
    cfg = sa.AuditConfig()

    # ts already chipped in the request (avoid re-rendering those).
    for key, m in meta.items():
        m["chipped_ts"] = [int(Path(p).stem.split("_t")[1].split("_")[0])
                           for p in m.get("chips", [])]

    extra_chips = {}
    if not args.no_extra_chips:
        wants = collect_extra_chip_wants(audits, meta)
        n_wanted = sum(len(v) for v in wants.values())
        print(f"rendering {n_wanted} extra chips "
              f"(strikes/secondaries) for {len(wants)} tracks")
        extra_chips = render_extra_chips(
            wants, artifact, meta, args.dataset_base, args.landmark_base,
            audit_dir / "chips", cfg)
        print(f"rendered {len(extra_chips)}")

    verdicts = Counter(a["verdict"] for a in audits.values())
    kinds = Counter(a["landmark_kind"] for a in audits.values())
    extents = Counter(a["primary_object"]["extent"] for a in audits.values())
    n_strikes = sum(len(a["strike_votes"]) for a in audits.values())
    n_secs = sum(len(a["secondary_objects"]) for a in audits.values())

    parts = [
        "<html><head><title>semantic audit results</title><style>",
        "body{font-family:sans-serif;background:#161616;color:#ddd;margin:16px}",
        "a{color:#8bf}pre{background:#1d1d1d;padding:10px;border-radius:6px;",
        "white-space:pre-wrap;font-size:12px;max-width:1100px}",
        ".chip{display:inline-block;background:#222;margin:3px;padding:4px;",
        "border-radius:4px;vertical-align:top;max-width:380px;font-size:12px}",
        ".chip img{height:190px;display:block;border-radius:3px}",
        "table{border-collapse:collapse}td,th{padding:3px 10px;font-size:13px;",
        "border-bottom:1px solid #333;text-align:left}",
        ".v_keep{color:#3c3}.v_keep_partial{color:#fa2}.v_drop{color:#e55}",
        ".sec{color:#b9f}.unres{color:#fa2}",
        ".collide{background:#3a2222}",
        ".tlbar{position:relative;height:16px;background:#333;max-width:700px;",
        "border-radius:3px;margin:6px 0}",
        ".tlseg{position:absolute;top:0;height:100%;background:#274;",
        "border-radius:3px}",
        ".tlstrike{position:absolute;top:0;height:100%;width:2px;",
        "background:#e55}",
        ".tlsec{position:absolute;top:8px;height:8px;width:2px;",
        "background:#b9f}",
        "h2{margin-top:44px;border-top:1px solid #333;padding-top:16px}",
        "details{margin:8px 0}summary{cursor:pointer;color:#89a}",
        "</style></head><body>",
        "<h1>semantic audit results</h1>",
        f"<p>{len(audits)} tracks audited"
        + (f", <b class='v_drop'>{len(errors)} parse errors</b>" if errors
           else "") + "</p>",
        "<p>verdicts: " + ", ".join(
            f"<span class='v_{v}'>{v} x{n}</span>"
            for v, n in verdicts.most_common())
        + f" | strikes: {n_strikes} | secondary objects: {n_secs}</p>",
        "<p>kinds: " + ", ".join(f"{k} x{n}" for k, n in kinds.most_common())
        + " | extents: " + ", ".join(
            f"{k} x{n}" for k, n in extents.most_common()) + "</p>",
        "<p>timeline bars: <span style='color:#274'>&#9632;</span> valid "
        "segment | <span class='v_drop'>|</span> strike | "
        "<span class='sec'>|</span> secondary-object detection</p>",
    ]

    if errors:
        parts.append("<h3>parse errors</h3><pre>" + "\n".join(
            f"{k}: {esc(e)}" for k, e in errors.items()) + "</pre>")

    collisions = name_collision_rows(audits)
    parts.append("<h3>name / alias collisions</h3>")
    if collisions:
        parts.append(
            "<p>Names claimed by more than one track. primary+primary = "
            "duplicate tracks of the same object (consolidation input); "
            "primary+alias across tracks = possible alias theft "
            "(highlighted).</p>")
        parts.append("<table><tr><th>name</th><th>claims</th></tr>")
        for name, claims in collisions:
            roles = {role for _, role in claims}
            cls = " class='collide'" if len(roles) > 1 else ""
            claim_txt = ", ".join(
                f"<a href='#{k}'>{k}</a> ({role})" for k, role in claims)
            parts.append(f"<tr{cls}><td>'{esc(name)}'</td>"
                         f"<td>{claim_txt}</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<p>(none)</p>")

    group_order = {"drop": 0, "keep_partial": 1, "keep": 2}

    def sort_key(item):
        key, audit = item
        has_edits = bool(audit["strike_votes"] or audit["secondary_objects"])
        return (group_order[audit["verdict"]],
                0 if has_edits else 1,
                -meta[key]["n_supports"])

    for key, audit in sorted(audits.items(), key=sort_key):
        parts.extend(track_section(
            key, audit, meta[key],
            tracks_by_id[meta[key]["track_id"]], texts, extra_chips))

    parts.append("</body></html>")
    out = audit_dir / "review"
    out.mkdir(exist_ok=True)
    (out / "index.html").write_text("\n".join(parts))
    print(f"wrote {out / 'index.html'}")
    print("verdicts:", dict(verdicts))


if __name__ == "__main__":
    main()
