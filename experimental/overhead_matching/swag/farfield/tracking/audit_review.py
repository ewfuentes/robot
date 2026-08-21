"""Review site for per-track semantic audit results.

Joins <run_dir>/semantic_audit/{results.jsonl, requests.jsonl,
audit_meta.json, settings.json} with the run's tracks artifacts and renders

  <run_dir>/semantic_audit/review/index.html

Contents:
- headline stats (verdicts, kinds, extents, strikes, secondaries) and the
  audit's recorded provenance (model, thinking level, support bar, prompt
  hash)
- a NAME / ALIAS COLLISION table: every name claimed by more than one track,
  with the role it was claimed in (primary vs alias). Cross-role collisions
  (one track's alias is another track's primary name) are the alias-theft
  signature and are highlighted.
- per-track sections grouped by verdict (drops first, then keep_partial,
  then keeps with edits, then clean keeps): the model's canonical record, a
  lifetime bar showing valid_segments / strike marks / secondary-object
  marks, the chips the model saw, and a chip for every strike and secondary
  object so each claim can be verified visually.

Audit payloads are read through calibration.audit_io.load_audits -- the one
canonical reader every consumer shares -- so what this page shows is exactly
what matching and the export consume. The results JSONL is re-parsed only for
what that reader deliberately skips: the per-key errors. Support
classification and chip rendering use the RECORDED settings (settings.json +
each tracks artifact's own TrackBuilderConfig), never a fresh default; an
audit directory without a settings record is refused.

Strike / secondary chips are rendered on demand (requires the dataset for
pano decoding); pass --no_extra_chips to skip that and reuse only the chips
already rendered for the requests.

Run:
  bazel run //experimental/overhead_matching/swag/farfield/tracking:audit_review -- \\
      --run_dir <runs>/r003 [required flags...]
"""

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.farfield import (
    dataset,
    paths as paths_lib,
)
from experimental.overhead_matching.swag.farfield.calibration import audit_io
from experimental.overhead_matching.swag.farfield.tracking import (
    keyframe_viewer as kv,
    semantic_audit as sa,
)
from experimental.overhead_matching.swag.farfield.viewers import (
    page as page_lib,
)

GENERATOR = ("//experimental/overhead_matching/swag/farfield/tracking"
             ":audit_review")

# Pages go through the one shared farfield.viewers.page helper (one CSS, a
# provenance footer on every page); only the classes specific to this viewer
# live here.
EXTRA_STYLE = """
pre{white-space:pre-wrap;font-size:12px;max-width:1100px}
.chip{display:inline-block;background:#222;margin:3px;padding:4px;
border-radius:4px;vertical-align:top;max-width:380px;font-size:12px}
.chip img{height:190px;display:block;border-radius:3px}
.v_keep{color:#3c3}.v_keep_partial{color:#fa2}.v_drop{color:#e55}
.sec{color:#b9f}.unres{color:#fa2}
.collide{background:#3a2222}
.tlbar{position:relative;height:16px;background:#333;max-width:700px;
border-radius:3px;margin:6px 0}
.tlseg{position:absolute;top:0;height:100%;background:#274;
border-radius:3px}
.tlstrike{position:absolute;top:0;height:100%;width:2px;background:#e55}
.tlsec{position:absolute;top:8px;height:8px;width:2px;background:#b9f}
h2{margin-top:44px;border-top:1px solid #333;padding-top:16px}
details{margin:8px 0}summary{cursor:pointer;color:#89a}
"""


def esc(x):
    return html.escape(str(x))


def name_candidates(primary_object):
    """[(name, weight, basis)] highest-weight first.

    audit_io hands back the model's payload verbatim (no legacy upgrade), so
    audits written before names became a weighted list still carry a single
    `name`; the fallback lifts it into a lone candidate the way
    semantic_audit.upgrade_legacy_audit does.
    """
    cands = primary_object.get("name_candidates")
    if cands is None:
        name = primary_object.get("name", "")
        cands = ([{"name": name, "weight": 1.0,
                   "basis": "reported_by_detections"}] if name else [])
    return sorted(((c["name"], c.get("weight", 0.0),
                    c.get("basis", "reported_by_detections"))
                   for c in cands if c.get("name")), key=lambda c: -c[1])


def top_name(primary_object):
    cands = name_candidates(primary_object)
    return cands[0][0] if cands else ""


def load_settings(audit_dir: Path) -> dict:
    """The audit's recorded settings (written by audit_requests).

    Refused when absent: this viewer classifies supports and renders chips
    under the RECORDED values, and an audit with no record predates the
    provenance fix (or was built by the retired m5 stage) -- rebuild the
    requests with farfield/tracking:audit_requests rather than guessing.
    """
    path = audit_dir / "settings.json"
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist: this audit recorded no settings, so "
            f"the review cannot know which thresholds and chip parameters "
            f"produced it. Rebuild the requests with "
            f"//experimental/overhead_matching/swag/farfield/tracking:"
            f"audit_requests (which records them).")
    return json.loads(path.read_text())


def collect_errors(audit_dir: Path, consumed_keys: set) -> dict:
    """{key: error} for result lines the canonical reader delivered nothing
    for. audit_io.load_audits deliberately skips error lines and unparseable
    payloads; this re-parse exists ONLY to surface those per key, so any key
    whose payload WAS consumed is not an error here."""
    errors = {}
    with open(audit_dir / "results.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            key, audit, err = sa.parse_result_line(json.loads(line))
            if err is not None and key not in consumed_keys:
                errors[key] = err
    return errors


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


def render_extra_chips(wants, tracks_by_id, range_by_track, cfg_by_range,
                       meta, dataset_base, landmark_base, chips_dir,
                       ingest_params, chip_height_px):
    """Render chips for (key, t) pairs. Returns {(key, t): filename}.

    Evidence is re-collected under each track's own range config
    (cfg_by_range), the same classification the requests were built with.
    """
    if not wants:
        return {}
    result = dataset.run_ingest(dataset_base, landmark_base, ingest_params)
    obs_by_id = {o.obs_id: o for o in result.observations}
    frames_by_idx = {f.frame_idx: f for f in result.frames}

    probe = Image.open(dataset_base / "panorama"
                       / f"{result.frames[0].pano_stem}.jpg")
    pano_w, pano_h = probe.size

    by_keyframe = defaultdict(list)
    for key, ts in wants.items():
        tid = meta[key]["track_id"]
        track = tracks_by_id[tid]
        cfg = cfg_by_range[range_by_track[tid]]
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
                e, e["obs"], pano_w, pano_h, ingest_params.fov_deg)
            name = f"{key}_t{t:04d}_extra.jpg"
            sa.render_chip(pano, det_box, mask_box, chips_dir / name,
                           chip_height_px)
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
        cands = name_candidates(po)
        for rank, (name, weight, _) in enumerate(cands):
            role = "primary" if rank == 0 else f"cand {weight:.2f}"
            claims[name.strip().lower()].append((name, key, role))
        for alias in po.get("name_aliases", []):
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


def track_section(key, audit, meta_entry, track, texts, extra_chips,
                  range_name):
    parts = []
    po = audit["primary_object"]
    verdict = audit["verdict"]
    lifetime = track["end_keyframe"] - track["birth_keyframe"] + 1
    strike_ts = [s["t"] for s in audit["strike_votes"]]
    sec_ts = sorted({t for s in audit["secondary_objects"] for t in s["ts"]})

    tid = meta_entry["track_id"]
    parts.append(
        f"<h2 id='{key}'><span class='v_{verdict}'>{verdict}</span> {key} "
        f"&mdash; {esc(top_name(po) or (po['tags'][0]['tag'] if po['tags'] else '?'))}"
        f"</h2>")
    parts.append(
        f"<p>{esc(audit['landmark_kind'])} | extent: {esc(po['extent'])} | "
        f"confidence: {esc(audit['confidence'])} | single_object: "
        f"{audit['single_object']} | drop_reason: "
        f"{esc(audit['drop_reason'])} | supports: "
        f"{meta_entry['n_supports']} | "
        f"<a href='../../track_{range_name}_T{tid}.html'>track page</a> | "
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
    cands = name_candidates(po)
    if cands:
        cand_txt = ", ".join(
            f"'{esc(n)}' {w:.2f}<span style='color:#889'> ({esc(b)})</span>"
            for n, w, b in cands)
        alias_txt = (" | aliases: " + ", ".join(
            f"'{esc(a)}'" for a in po["name_aliases"])
            if po.get("name_aliases") else "")
        multi = (" <span class='contested'>[multiple candidates &mdash; "
                 "matcher resolves]</span>" if len(cands) > 1 else "")
        parts.append(f"<p><b>name candidates:</b> {cand_txt}{alias_txt}"
                     f"{multi}</p>")
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
    paths_lib.add_arguments(parser)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--no_extra_chips", action="store_true",
                        help="Skip rendering chips for strikes/secondaries")
    # TODO(run_config): the ingest values move into the run's recorded config
    # (run_config.json, REORG.md PR 12); required on the CLI until then.
    parser.add_argument("--fov_deg", type=float, required=True,
                        help="Pinhole-face FOV the extraction rendered with, "
                             "degrees (previously 90.0 on all datasets)")
    parser.add_argument("--seam_gap_norm", type=float, required=True,
                        help="Seam-merge margin in bbox units 0-1000 "
                             "(previously 25)")
    parser.add_argument("--seam_min_y_iou", type=float, required=True,
                        help="Vertical IoU to accept a seam continuation "
                             "(previously 0.3)")
    args = parser.parse_args()
    paths = paths_lib.resolve(
        parser, args, infer_from=args.run_dir,
        require=("dataset_base", "frame_landmarks"))

    audit_dir = args.run_dir / "semantic_audit"
    if not (audit_dir / "audit_meta.json").exists():
        raise SystemExit(
            f"no audit_meta.json under {audit_dir} -- build the requests "
            f"first (farfield/tracking:audit_requests)")
    if not (audit_dir / "results.jsonl").exists():
        raise SystemExit(
            f"no results.jsonl under {audit_dir} -- submit the requests "
            f"first (audit_requests --submit, or vertex_batch_manager)")
    settings = load_settings(audit_dir)
    meta = json.loads((audit_dir / "audit_meta.json").read_text())

    # The canonical reader (what matching/export consume), joined back to
    # request keys through the meta record.
    audits_by_tid = audit_io.load_audits(args.run_dir)
    audits = {key: audits_by_tid[m["track_id"]] for key, m in meta.items()
              if m["track_id"] in audits_by_tid}
    errors = collect_errors(audit_dir, set(audits))
    texts = load_request_texts(audit_dir)

    # Every range's tracks, each classified under ITS recorded config plus
    # the audit's recorded knobs -- never a freshly-defaulted AuditConfig.
    artifacts = kv.load_track_artifacts(args.run_dir)
    tracks_by_id, range_by_track = sa.merge_tracks(artifacts)
    cfg_by_range = {
        rn: sa.AuditConfig(**settings["audit_config"],
                           classifier=kv.recorded_config(artifact))
        for rn, artifact in artifacts.items()}
    chip_height_px = settings["audit_config"]["chip_height_px"]
    ingest_params = dataset.IngestParams(
        fov_deg=args.fov_deg, seam_gap_norm=args.seam_gap_norm,
        seam_min_y_iou=args.seam_min_y_iou)

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
            wants, tracks_by_id, range_by_track, cfg_by_range, meta,
            paths.dataset_base, paths.frame_landmarks, audit_dir / "chips",
            ingest_params, chip_height_px)
        print(f"rendered {len(extra_chips)}")

    verdicts = Counter(a["verdict"] for a in audits.values())
    kinds = Counter(a["landmark_kind"] for a in audits.values())
    extents = Counter(a["primary_object"]["extent"] for a in audits.values())
    n_strikes = sum(len(a["strike_votes"]) for a in audits.values())
    n_secs = sum(len(a["secondary_objects"]) for a in audits.values())

    parts = [
        f"<p>{len(audits)} tracks audited"
        + (f", <b class='v_drop'>{len(errors)} result errors</b>" if errors
           else "") + "</p>",
        # The recorded provenance, so the page states what produced what it
        # shows (model/thresholds were previously recorded nowhere at all).
        f"<p class='muted'>model {esc(settings['model'])} | thinking "
        f"{esc(settings['thinking_level'])} | support bar "
        f">= {esc(settings['min_supports'])} | prompt sha256 "
        f"{esc(settings['system_prompt_sha256'][:12])}</p>",
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
        parts.append("<h3>result errors</h3><pre>" + "\n".join(
            f"{esc(k)}: {esc(e)}" for k, e in errors.items()) + "</pre>")

    collisions = name_collision_rows(audits)
    parts.append("<h3>name / alias collisions</h3>")
    if collisions:
        parts.append(
            "<p>Names claimed by more than one track. primary+primary = "
            "duplicate tracks of the same object (the filter's data "
            "association copes); primary+alias across tracks = possible "
            "alias theft (highlighted).</p>")
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
        tid = meta[key]["track_id"]
        parts.extend(track_section(
            key, audit, meta[key], tracks_by_id[tid], texts, extra_chips,
            range_by_track[tid]))

    out = audit_dir / "review"
    out.mkdir(exist_ok=True)
    (out / "index.html").write_text(page_lib.page(
        "semantic audit results", "\n".join(parts), generator=GENERATOR,
        extra_style=EXTRA_STYLE))
    print(f"wrote {out / 'index.html'}")
    print("verdicts:", dict(verdicts))


if __name__ == "__main__":
    main()
