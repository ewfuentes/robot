"""Landing page for a tracking run: one door into every stage's viewer.

Walks the run, reports what each stage actually produced, and links the
viewers in pipeline order so a run can be browsed end to end.

STAGES is the list of stages that still exist. The wedge-based pairing
stages (m7/m8) were removed when candidate generation stopped using the
vessel's position - gating candidates by where the boat was, then asking the
filter to recover where the boat was, is circular. Matching is now m9 (whole
map, no spatial gating) with m10 as its viewer.

Writes <run_dir>/index.html (overwriting the m3 track board, which it links
to as "tracks").

Run:
  bazel run //...object_tracking:run_index -- --run_dir <runs>/r003_full_leg1
"""

import argparse
import html
import json
from pathlib import Path

STAGES = [
    ("M3", "tracks", "board.html",
     "SAM2 mask tracks with per-keyframe evidence, videos and diffs"),
    ("&mdash;", "keyframes", "keyframes/index.html",
     "every detection on every keyframe: births, supports, rejections"),
    ("M5", "semantic audit", "semantic_audit/review/index.html",
     "per-track VLM canonicalization: verdicts, name candidates, strikes"),
    ("M5", "audit requests", "semantic_audit/preview/index.html",
     "the exact prompt and chips each audit call was given"),
    ("M6", "merged landmarks", "merged/index.html",
     "co-visibility merging, parent/child links, ambiguous pairs"),
    ("M9/M10", "matches", "matching/review/index.html",
     "observation vs matched map landmark: confidence, instance/category, "
     "expansion width"),
]



def summarise(run_dir: Path) -> dict:
    """Cheap counts per stage; missing files are simply absent."""
    out = {}
    tracks = next(run_dir.glob("tracks_*.json"), None)
    if tracks:
        data = json.loads(tracks.read_text())
        out["tracks"] = f"{len(data['tracks'])} tracks"
    kf = run_dir / "keyframes"
    if kf.exists():
        out["keyframes"] = f"{len(list(kf.glob('f*.html')))} keyframes"
    audit = run_dir / "semantic_audit" / "results.jsonl"
    if audit.exists():
        n = sum(1 for line in open(audit) if line.strip())
        out["semantic audit"] = f"{n} tracks audited"
    merged = run_dir / "merged" / "landmarks.json"
    if merged.exists():
        data = json.loads(merged.read_text())
        multi = sum(1 for lm in data if len(lm["track_ids"]) > 1)
        out["merged landmarks"] = f"{len(data)} landmarks, {multi} merged"
    match_path = run_dir / "matching" / "matches.json"
    if match_path.exists():
        data = json.loads(match_path.read_text())
        hit = sum(1 for v in data.values() if v.get("n_landmarks"))
        out["matches"] = f"{hit}/{len(data)} matched"
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--runs_root", type=Path, default=None,
                        help="If given, also write an index of all runs")
    args = parser.parse_args()

    run_dir = args.run_dir
    # The m3 board owns index.html; move it aside once so this page can be
    # the entry point without losing it.
    board = run_dir / "board.html"
    index = run_dir / "index.html"
    if index.exists() and not board.exists():
        board.write_text(index.read_text())

    counts = summarise(run_dir)
    meta = {}
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    parts = [
        "<html><head><title>", html.escape(run_dir.name), "</title><style>",
        "body{font-family:sans-serif;background:#151515;color:#ddd;margin:0}",
        "main{max-width:900px;margin:0 auto;padding:36px 22px 70px}",
        "a{color:#8bf;text-decoration:none}a:hover{text-decoration:underline}",
        "h1{margin:0 0 6px;font-size:26px}",
        ".notes{color:#9aa;max-width:70ch;margin:0 0 26px;font-size:14px}",
        ".stage{display:flex;gap:14px;align-items:baseline;padding:13px 14px;",
        "border:1px solid #2c2c2c;border-radius:7px;margin-bottom:9px;",
        "background:#1b1b1b}",
        ".stage.missing{opacity:.4}",
        ".tag{font:11px ui-monospace,monospace;color:#7a8;min-width:26px}",
        ".name{font-size:16px;font-weight:600;min-width:170px}",
        ".desc{color:#9aa;font-size:13.5px;flex:1}",
        ".count{color:#dc8;font:12.5px ui-monospace,monospace;",
        "white-space:nowrap}",
        "</style></head><body><main>",
        f"<h1>{html.escape(run_dir.name)}</h1>"]
    if meta.get("notes"):
        parts.append(f"<p class='notes'>{html.escape(meta['notes'])}</p>")
    for tag, name, rel, desc in STAGES:
        exists = (run_dir / rel).exists()
        count = counts.get(name, "")
        cls = "stage" if exists else "stage missing"
        label = (f"<a href='{rel}'>{html.escape(name)}</a>" if exists
                 else html.escape(name) + " (not built)")
        parts.append(
            f"<div class='{cls}'><span class='tag'>{tag}</span>"
            f"<span class='name'>{label}</span>"
            f"<span class='desc'>{html.escape(desc)}</span>"
            f"<span class='count'>{html.escape(count)}</span></div>")
    parts.append("</main></body></html>")
    index.write_text("\n".join(parts))
    print(f"wrote {index}")

    if args.runs_root:
        runs = sorted(p for p in args.runs_root.iterdir() if p.is_dir())
        rows = []
        for run in runs:
            if not (run / "index.html").exists():
                continue
            info = summarise(run)
            rows.append(
                f"<div class='stage'><span class='name'>"
                f"<a href='{run.name}/index.html'>{html.escape(run.name)}</a>"
                f"</span><span class='desc'>"
                + html.escape(", ".join(info.values())) + "</span></div>")
        (args.runs_root / "index.html").write_text(
            "\n".join(parts[:20] + ["<h1>runs</h1>"] + rows
                      + ["</main></body></html>"]))
        print(f"wrote {args.runs_root}/index.html")


if __name__ == "__main__":
    main()
