"""Diff two M3 tracking runs and render a comparison page.

Answers "was this change a step in the right direction?" by putting three
things on one page:
1. What changed: each run's notes (from run_meta.json) and the exact config
   deltas pulled from the artifacts.
2. Whether it moved the needle: per-range headline metrics side by side
   (closures by reason, supported keyframes, window-edge clipping, erosion),
   with deltas colored by desired direction.
3. Where to look: tracks matched across runs by their founding detection,
   sorted by how much they changed, with side-by-side thumbnails linking to
   each run's full track page (video + evidence table).

Run:
  bazel run //experimental/overhead_matching/swag/landmark_filtering/object_tracking:m3_run_diff -- \
    --run_a <runs_root>/r000_baseline --run_b <runs_root>/r001_...
Writes diff_vs_<run_a>.html into run_b's directory.
"""

import argparse
import html
import json
from pathlib import Path

EDGE_MARGIN = 3
# Keep in sync with track_builder.SUPPORT_CLASSES (this tool stays
# dependency-free; "context" and "none" are not support).
SUPPORT_CLASSES = {"continue_clean", "merge_superset", "split_child", "weak"}


def load_run(run_dir: Path):
    meta = json.loads((run_dir / "run_meta.json").read_text())
    artifacts = {}
    for p in sorted(run_dir.glob("tracks_*.json")):
        a = json.loads(p.read_text())
        artifacts[a["range"]["name"]] = a
    return meta, artifacts


def track_metrics(t):
    """Per-track summary metrics derived from its records."""
    clipped = 0
    last_supported_rec = None
    for r in t["records"]:
        mb = r.get("mask_bbox_window")
        if mb is not None:
            win = r.get("window_px", 1024)
            if mb[0] <= EDGE_MARGIN or mb[2] >= win - 1 - EDGE_MARGIN:
                clipped += 1
        if any(s["class"] in SUPPORT_CLASSES for s in r.get("supports", [])):
            last_supported_rec = r
    erosion = None
    if last_supported_rec is not None \
            and last_supported_rec.get("mask_bbox_window") is not None:
        mb = last_supported_rec["mask_bbox_window"]
        mask_w = mb[2] - mb[0]
        box_w = max((s["box_window"][2] - s["box_window"][0]
                     for s in last_supported_rec["supports"]
                     if s["class"] in SUPPORT_CLASSES), default=None)
        if box_w:
            erosion = round(mask_w / box_w, 2)
    status = t["close_reason"] if t["status"] == "closed" else "alive"
    return {
        "sup": t["n_supported_keyframes"],
        "status": status,
        "clipped": clipped,
        "erosion": erosion,  # end mask width / end supporting box width
        "span": (t["birth_keyframe"], t["end_keyframe"]),
    }


def summarize(artifact):
    tracks = [t for t in artifact["tracks"] if t["records"]]
    metrics = [track_metrics(t) for t in tracks]
    by_status = {}
    for m in metrics:
        by_status[m["status"]] = by_status.get(m["status"], 0) + 1
    eroded = sum(1 for m in metrics
                 if m["erosion"] is not None and m["erosion"] < 0.3)
    return {
        "tracks": len(tracks),
        "alive": by_status.get("alive", 0),
        "starved": by_status.get("starved", 0),
        "drift_alarm": by_status.get("drift_alarm", 0),
        "mask_dead": by_status.get("mask_dead", 0),
        "birth_rejected": sum(v for k, v in by_status.items()
                              if k.startswith("birth_")),
        "supported_kf": sum(m["sup"] for m in metrics),
        "clipped_kf": sum(m["clipped"] for m in metrics),
        "eroded_tracks (end mask/box < 0.3)": eroded,
    }


# Whether an increase in the metric is good (+), bad (-), or neutral (0).
METRIC_SIGN = {
    "tracks": 0, "alive": 1, "starved": -1, "drift_alarm": -1,
    "mask_dead": -1, "birth_rejected": 0, "supported_kf": 1,
    "clipped_kf": -1, "eroded_tracks (end mask/box < 0.3)": -1,
}

STYLE = """
body{font-family:sans-serif;background:#161616;color:#ddd;margin:16px;
     max-width:1500px}
a{color:#8bf}
table{border-collapse:collapse;margin:8px 0}
td,th{padding:3px 10px;text-align:left;border-bottom:1px solid #333;
      font-size:14px;vertical-align:top}
.good{color:#4d4;font-weight:bold}.bad{color:#e55;font-weight:bold}
.neutral{color:#aaa}
.notes{background:#222;border-left:5px solid #58f;border-radius:4px;
       padding:8px 14px;margin:8px 0;max-width:1100px}
img.thumb{width:150px;height:150px;object-fit:cover;border-radius:4px}
.chip{display:inline-block;padding:1px 7px;border-radius:9px;font-size:12px;
      background:#333}
"""

TOGGLE_JS = """
function toggleUnchanged(){
  const show=document.getElementById('show_unchanged').checked;
  document.querySelectorAll('tr.unchanged').forEach(
    r=>r.style.display=show?'':'none');
}
window.addEventListener('DOMContentLoaded',toggleUnchanged);
"""


def delta_cell(key, a, b):
    if a == b:
        return f"<td class='neutral'>{b}</td>"
    d = (b or 0) - (a or 0)
    sign = METRIC_SIGN.get(key, 0) * (1 if d > 0 else -1)
    css = {1: "good", -1: "bad", 0: "neutral"}[sign]
    return f"<td class='{css}'>{b} ({d:+d})</td>"


def fmt_track_cell(m):
    ero = "-" if m["erosion"] is None else f"{m['erosion']:.2f}"
    span = (f"f{m['span'][0]:04d}..f{m['span'][1]:04d}"
            if m["span"][1] is not None else "-")
    return (f"sup={m['sup']} <span class='chip'>{html.escape(m['status'])}"
            f"</span><br><small>{span} | clipped={m['clipped']} | "
            f"end mask/box={ero}</small>")


def change_score(ma, mb):
    score = abs(mb["sup"] - ma["sup"])
    if ma["status"] != mb["status"]:
        score += 10
    score += abs(mb["clipped"] - ma["clipped"])
    if (ma["erosion"] is not None and mb["erosion"] is not None
            and abs(mb["erosion"] - ma["erosion"]) > 0.15):
        score += 5
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_a", type=Path, required=True)
    parser.add_argument("--run_b", type=Path, required=True)
    args = parser.parse_args()

    meta_a, arts_a = load_run(args.run_a)
    meta_b, arts_b = load_run(args.run_b)
    name_a, name_b = meta_a["run_name"], meta_b["run_name"]
    out_path = args.run_b / f"diff_vs_{name_a}.html"
    # Viewer roots, relative to the diff page's directory (run_b).
    rel_a = str(Path(meta_a.get("viewer_rel", ".")))
    rel_b = str(Path(meta_b.get("viewer_rel", ".")))
    view_a = (Path("..") / args.run_a.name / rel_a)
    view_b = Path(rel_b)

    parts = [f"<html><head><title>{name_b} vs {name_a}</title>"
             f"<style>{STYLE}</style><script>{TOGGLE_JS}</script></head><body>",
             f"<h1>Run diff: {html.escape(name_b)} vs "
             f"{html.escape(name_a)}</h1>",
             "<h2>What changed</h2>",
             f"<div class='notes'><b>{html.escape(name_a)}</b> "
             f"({html.escape(meta_a.get('created', '?'))}):<br>"
             f"{html.escape(meta_a.get('notes', ''))}</div>",
             f"<div class='notes'><b>{html.escape(name_b)}</b> "
             f"({html.escape(meta_b.get('created', '?'))}):<br>"
             f"{html.escape(meta_b.get('notes', ''))}</div>"]

    # Config diff from the first range's artifact.
    cfg_a = next(iter(arts_a.values()))["config"]
    cfg_b = next(iter(arts_b.values()))["config"]
    keys = sorted(set(cfg_a) | set(cfg_b))
    diff_rows = [(k, cfg_a.get(k, "(absent)"), cfg_b.get(k, "(absent)"))
                 for k in keys if cfg_a.get(k) != cfg_b.get(k)]
    if diff_rows:
        parts.append("<h3>Config deltas</h3><table><tr><th>field</th>"
                     f"<th>{name_a}</th><th>{name_b}</th></tr>")
        parts.extend(f"<tr><td><code>{k}</code></td><td>{a}</td>"
                     f"<td class='good'>{b}</td></tr>" for k, a, b in diff_rows)
        parts.append("</table>")

    parts.append("<h2>Headline metrics</h2>"
                 "<p>green/red = moved in the desired/undesired direction "
                 "(e.g. fewer clipped keyframes is good)</p>")
    totals_a, totals_b = {}, {}
    for range_name in sorted(set(arts_a) | set(arts_b)):
        parts.append(f"<h3>{html.escape(range_name)}</h3>")
        sa = summarize(arts_a[range_name]) if range_name in arts_a else {}
        sb = summarize(arts_b[range_name]) if range_name in arts_b else {}
        parts.append(f"<table><tr><th>metric</th><th>{name_a}</th>"
                     f"<th>{name_b}</th></tr>")
        for k in sa:
            totals_a[k] = totals_a.get(k, 0) + sa[k]
            totals_b[k] = totals_b.get(k, 0) + sb.get(k, 0)
            parts.append(f"<tr><td>{k}</td><td>{sa[k]}</td>"
                         + delta_cell(k, sa[k], sb.get(k, 0)) + "</tr>")
        parts.append("</table>")
    parts.append("<h3>All ranges</h3><table><tr><th>metric</th>"
                 f"<th>{name_a}</th><th>{name_b}</th></tr>")
    for k in totals_a:
        parts.append(f"<tr><td>{k}</td><td>{totals_a[k]}</td>"
                     + delta_cell(k, totals_a[k], totals_b[k]) + "</tr>")
    parts.append("</table>")

    # Track-by-track diff, matched on (range, birth_obs_id).
    parts.append("<h2>Tracks</h2>"
                 "<label><input type='checkbox' id='show_unchanged' "
                 "onchange='toggleUnchanged()'> show unchanged tracks"
                 "</label>")
    parts.append("<table><tr><th>founding detection</th>"
                 f"<th colspan='2'>{name_a}</th>"
                 f"<th colspan='2'>{name_b}</th></tr>")
    rows = []
    for range_name in sorted(set(arts_a) | set(arts_b)):
        ta = {t["birth_obs_id"]: t
              for t in arts_a.get(range_name, {"tracks": []})["tracks"]
              if t["records"]}
        tb_ = {t["birth_obs_id"]: t
               for t in arts_b.get(range_name, {"tracks": []})["tracks"]
               if t["records"]}
        for obs_id in sorted(set(ta) | set(tb_)):
            a, b = ta.get(obs_id), tb_.get(obs_id)
            ma = track_metrics(a) if a else None
            mb = track_metrics(b) if b else None
            score = (change_score(ma, mb) if ma and mb else 20)
            label = (a or b)["modal_label"]
            rows.append((score, range_name, obs_id, label, a, b, ma, mb))
    rows.sort(key=lambda r: -r[0])

    def side(run_view, range_name, t, m):
        if t is None:
            return "<td colspan='2' class='neutral'>(no track)</td>"
        key = f"{range_name}_T{t['track_id']}"
        return (f"<td><a href='{run_view}/track_{key}.html'>"
                f"<img class='thumb' src='{run_view}/thumbs/{key}.jpg' "
                f"loading='lazy'></a></td><td>T{t['track_id']}<br>"
                + fmt_track_cell(m) + "</td>")

    for score, range_name, obs_id, label, a, b, ma, mb in rows:
        cls = " class='unchanged'" if score == 0 else ""
        parts.append(
            f"<tr{cls}><td><code>{html.escape(obs_id)}</code><br>"
            f"{html.escape(label[:48])}<br>"
            f"<small>{range_name}</small></td>"
            + side(view_a.as_posix(), range_name, a, ma)
            + side(view_b.as_posix(), range_name, b, mb) + "</tr>")
    parts.append("</table></body></html>")
    out_path.write_text("\n".join(parts))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
