"""Side-by-side review of what we saw against what we matched it to.

Renders <run_dir>/matching/review/index.html: per tracklet, the observation
(query tags + the chips the audit looked at) beside every map landmark the
matcher proposed, with confidence, match type, and how many map rows the
matched signature expanded to.

The page is built so a wrong match is findable without opening JSON: matches
sort by confidence, instance matches are marked distinctly from category
ones, wide expansions are called out, and tracklets with no match are listed
with their no-match confidence so an unexplained silence is visible too.

Run:
  bazel run //...object_tracking:m10_match_viewer -- \\
      --run_dir <runs>/r003_full_leg1
"""

import argparse
import html
import json
from collections import Counter
from pathlib import Path


def esc(x):
    return html.escape(str(x))


def source_links(landmark_id, lm, meta_by_track, range_name="full_leg1"):
    """Links back to every artifact this landmark came from.

    Chips only exist for tracks that went through the semantic audit, so a
    landmark below the audit's support bar shows none. That is a reason to
    link the raw track and keyframe pages rather than leave the row blank -
    the evidence exists, it was just never chipped.
    """
    out = []
    for tid in lm.get("track_ids", []):
        out.append(f"<a href='../../track_{range_name}_T{tid}.html'>track "
                   f"T{tid}</a>")
        key = meta_by_track.get(tid, {}).get("_key")
        if key:
            out.append(f"<a href='../../semantic_audit/review/index.html"
                       f"#{key}'>audit {key}</a>")
    span = lm.get("keyframe_span") or []
    if len(span) == 2 and span[0] is not None:
        out.append(f"<a href='../../keyframes/f{int(span[0]):04d}.html'>"
                   f"keyframe f{int(span[0]):04d}</a>")
        out.append(f"<a href='../../keyframes/f{int(span[1]):04d}.html'>"
                   f"f{int(span[1]):04d}</a>")
    out.append("<a href='../../merged/index.html'>merged</a>")
    return " &middot; ".join(out)


def chips_for(landmark_id, landmarks, meta):
    """Audit chips belonging to the tracks that make up this landmark."""
    tracks = next((lm["track_ids"] for lm in landmarks
                   if lm["landmark_id"] == landmark_id), [])
    by_track = {v["track_id"]: v for v in meta.values()}
    out = []
    for tid in tracks:
        out += by_track.get(tid, {}).get("chips", [])
    return out[:4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    args = parser.parse_args()

    match_dir = args.run_dir / "matching"
    matches = json.loads((match_dir / "matches.json").read_text())
    signatures = json.loads((match_dir / "signatures.json").read_text())
    landmarks = json.loads(
        (args.run_dir / "merged" / "landmarks.json").read_text())
    meta_path = args.run_dir / "semantic_audit" / "audit_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    span = {lm["landmark_id"]: lm for lm in landmarks}
    meta_by_track = {}
    for key, value in meta.items():
        meta_by_track[value["track_id"]] = dict(value, _key=key)

    out = match_dir / "review"
    (out / "chips").mkdir(parents=True, exist_ok=True)
    # Chips live under semantic_audit/chips; symlink so the page can reach
    # them with a stable relative path even if that tree is regenerated.
    kinds = Counter(x["match_type"] for v in matches.values()
                    for x in v["matches"])
    hit = [k for k, v in matches.items() if v["n_landmarks"]]
    miss = [k for k, v in matches.items() if not v["n_landmarks"]]

    parts = [
        "<html><head><title>matches</title><style>",
        "body{font-family:sans-serif;background:#161616;color:#ddd;margin:16px}",
        "a{color:#8bf}code{color:#cfc}",
        ".q{background:#1b2430;border-left:3px solid #4af;padding:8px 12px;",
        "border-radius:5px;max-width:1150px;margin:6px 0}",
        "table{border-collapse:collapse;margin:8px 0;width:100%;max-width:1150px}",
        "td,th{padding:4px 10px;font-size:13px;border-bottom:1px solid #303030;",
        "text-align:left;vertical-align:top}th{color:#89a;font-size:11.5px;",
        "text-transform:uppercase;letter-spacing:.08em}",
        ".instance{color:#3c8;font-weight:bold}.category{color:#89f}",
        ".wide{color:#fa2}.conf{font-variant-numeric:tabular-nums}",
        "img{height:120px;border-radius:4px;margin:3px 3px 0 0;",
        "vertical-align:top}",
        "h2{margin-top:38px;border-top:1px solid #2c2c2c;padding-top:14px}",
        ".nomatch{color:#999}",
        ".links{font-size:12.5px;margin:6px 0}",
        ".nochips{color:#a88;font-size:12.5px;font-style:italic;margin:6px 0}",
        "</style></head><body>",
        "<h1>Observation &rarr; map landmark</h1>",
        f"<p>{len(matches)} tracklets | {len(hit)} with a match | "
        f"{len(miss)} without | matches: "
        f"<span class='instance'>{kinds.get('instance', 0)} instance</span>, "
        f"<span class='category'>{kinds.get('category', 0)} category</span>"
        " | <a href='../../index.html'>run index</a></p>",
        "<p><b>instance</b> = this exact object identified. <b>category</b> = "
        "right kind of object, cannot say which. A matched <i>signature</i> "
        "expands to every map row carrying it; the expansion count is shown "
        "and flagged when large, because those contribute little information "
        "however confident the match.</p>"]

    ordered = sorted(matches.items(),
                     key=lambda kv: -(kv[1]["matches"][0]["confidence"]
                                      if kv[1]["matches"] else -1))
    for key, entry in ordered:
        if not entry["matches"]:
            continue
        lm = span.get(key, {})
        parts.append(f"<h2 id='{esc(key)}'>{esc(key)}</h2>")
        parts.append(f"<div class='q'><b>observed:</b> "
                     f"<code>{esc(entry['query'])}</code><br>"
                     f"<span class='nomatch'>no_match_confidence "
                     f"{entry['no_match_confidence']} &middot; "
                     f"{lm.get('n_supports', '?')} supports &middot; "
                     f"tracks {lm.get('track_ids', [])}</span></div>")
        parts.append(f"<div class='links'>{source_links(key, lm, meta_by_track)}"
                     "</div>")
        chips = chips_for(key, landmarks, meta)
        for chip in chips:
            parts.append(f"<img src='../../semantic_audit/chips/"
                         f"{Path(chip).name}' loading='lazy'>")
        if not chips:
            parts.append("<div class='nochips'>no chips: this track was below "
                         "the semantic audit's support bar, so it was never "
                         "chipped. Use the track and keyframe links above.</div>")
        parts.append("<table><tr><th>conf</th><th>type</th><th>map rows</th>"
                     "<th>signature</th></tr>")
        shown = set()
        for match in entry["matches"]:
            sig = match["signature"]
            if sig in shown or match["confidence"] < args.min_confidence:
                continue
            shown.add(sig)
            n = len(signatures.get(sig, []))
            wide = " class='wide'" if n >= 50 else ""
            parts.append(
                f"<tr><td class='conf'>{match['confidence']:.2f}</td>"
                f"<td class='{match['match_type']}'>{match['match_type']}</td>"
                f"<td{wide}>{n}</td><td><code>{esc(sig)}</code></td></tr>")
        parts.append("</table>")

    parts.append("<h2>Returned no match</h2>")
    parts.append("<p>Note this means <i>the matcher found nothing</i>, not "
                 "that the object is absent from the map. For an unnamed "
                 "<code>building=commercial</code> the right building is "
                 "almost certainly in the catalog; the query simply cannot "
                 "discriminate it.</p><table><tr><th>tracklet</th>"
                 "<th>no-match conf</th><th>observed</th></tr>")
    for key in sorted(miss, key=lambda k: -matches[k]["no_match_confidence"]):
        entry = matches[key]
        lm = span.get(key, {})
        parts.append(f"<tr><td>{esc(key)}<br>"
                     f"<span class='links'>"
                     f"{source_links(key, lm, meta_by_track)}</span></td>"
                     f"<td class='conf'>{entry['no_match_confidence']}</td>"
                     f"<td><code>{esc(entry['query'])}</code></td></tr>")
    parts.append("</table></body></html>")
    (out / "index.html").write_text("\n".join(parts))
    print(f"wrote {out}/index.html ({len(hit)} matched, {len(miss)} not)")


if __name__ == "__main__":
    main()
