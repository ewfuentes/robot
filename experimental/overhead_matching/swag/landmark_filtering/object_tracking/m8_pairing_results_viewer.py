"""Review site for pairing (tracklet -> map landmark) labels.

Joins <run_dir>/pairing/{requests,results}.jsonl and renders

  <run_dir>/pairing/review/index.html

The labels are training data for the retrained correspondence model, so the
job of this page is to make a wrong label findable before it is trained on.
Each tracklet shows the Set 1 bundle, the wedge map, every proposed match
with the Set 2 bundle it points at, and the negatives - so a match can be
checked against what the model was actually shown.

`match_type` is surfaced prominently: "instance" (this exact object) and
"category" (right kind, cannot say which) are both valid labels with very
different downstream meaning, and conflating them is the failure this
column exists to catch.

Run:
  bazel run //...object_tracking:m8_pairing_results_viewer -- \\
      --run_dir <runs>/r003_full_leg1
"""

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path

VERDICT_CSS = {"instance": "#2c8", "category": "#89f",
               "hard": "#fa2", "easy": "#888"}


def esc(x):
    return html.escape(str(x))


def parse_sets(prompt_text):
    """(set1_entries, set2_entries) as lists of tag strings, by index."""
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


def load_requests(pairing_dir: Path):
    out = {}
    with open(pairing_dir / "requests.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record["request"]["contents"][0]["parts"][0]["text"]
            out[record["key"]] = parse_sets(text)
    return out


def load_results(pairing_dir: Path):
    """key -> (matches, error). Tolerates both the plain-int and the
    {set_2_id, match_type} forms of set_2_matches."""
    results, errors = {}, {}
    path = pairing_dir / "results.jsonl"
    if not path.exists():
        return results, errors
    with open(path) as f:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()

    pairing = args.run_dir / "pairing"
    requests = load_requests(pairing)
    results, errors = load_results(pairing)
    print(f"{len(requests)} requests, {len(results)} labelled, "
          f"{len(errors)} errors")

    n_match = n_instance = n_category = n_neg = 0
    uniqueness = Counter()
    unmatched = []
    for key, matches in results.items():
        got = False
        for match in matches:
            for _, kind in match["matches"]:
                n_match += 1
                got = True
                if kind == "instance":
                    n_instance += 1
                else:
                    n_category += 1
            n_neg += len(match["negatives"])
            if match["uniqueness"] is not None:
                uniqueness[match["uniqueness"]] += 1
        if not got:
            unmatched.append(key)

    parts = [
        "<html><head><title>pairing labels</title><style>",
        "body{font-family:sans-serif;background:#161616;color:#ddd;margin:16px}",
        "a{color:#8bf}code{color:#cfc}",
        "img{max-width:820px;border-radius:6px;background:#fff;display:block;",
        "margin:10px 0}",
        "pre{background:#1d1d1d;padding:10px;border-radius:6px;font-size:12px;",
        "white-space:pre-wrap;max-width:1100px}",
        "table{border-collapse:collapse;margin:8px 0}",
        "td,th{padding:3px 10px;font-size:13px;border-bottom:1px solid #333;",
        "text-align:left;vertical-align:top}th{color:#89a}",
        ".instance{color:#2c8;font-weight:bold}.category{color:#89f}",
        ".hard{color:#fa2}.easy{color:#888}",
        "h2{margin-top:42px;border-top:1px solid #333;padding-top:16px}",
        ".s1{background:#1b2430;padding:8px 12px;border-radius:5px;",
        "border-left:3px solid #4af;max-width:1100px}",
        "</style></head><body>",
        "<h1>Pairing labels: tracklet &rarr; map landmark</h1>",
        f"<p>{len(results)} tracklets labelled"
        + (f", <b style='color:#e55'>{len(errors)} errors</b>" if errors
           else "")
        + f" | {n_match} matches (<span class='instance'>{n_instance} "
        f"instance</span>, <span class='category'>{n_category} category"
        "</span>) | " + f"{n_neg} negatives | "
        f"{len(unmatched)} tracklets with no match</p>",
        "<p><b>instance</b> = this exact object is identified. "
        "<b>category</b> = the right kind of object, but the tags cannot say "
        "which one. Both are valid labels; a category match can be certain "
        "and still fail to pin a position.</p>",
        "<p>uniqueness scores: " + ", ".join(
            f"{k}&rarr;{v}" for k, v in sorted(uniqueness.items())) + "</p>",
        "<p><a href='../index.html'>&larr; run index</a> | "
        "<a href='../pairing/index.html'>wedge maps</a></p>"]

    if errors:
        parts.append("<h3>errors</h3><pre>" + "\n".join(
            f"{k}: {esc(v)}" for k, v in errors.items()) + "</pre>")

    for key in sorted(results, key=lambda k: -len(results[k])):
        set1, set2 = requests.get(key, ([], []))
        parts.append(f"<h2 id='{esc(key)}'>{esc(key)}</h2>")
        for entry in set1:
            parts.append(f"<div class='s1'><b>Set 1:</b> <code>"
                         f"{esc(entry)}</code></div>")
        fig = pairing / "figures" / f"{key}.png"
        if fig.exists():
            parts.append(f"<img src='../figures/{key}.png' loading='lazy'>")
        parts.append("<table><tr><th>kind</th><th>Set 2 #</th>"
                     "<th>map landmark</th></tr>")
        for match in results[key]:
            for set2_id, kind in match["matches"]:
                tags = set2[set2_id] if set2_id < len(set2) else "(out of range)"
                parts.append(
                    f"<tr><td class='{kind}'>{kind}</td><td>{set2_id}</td>"
                    f"<td><code>{esc(tags)}</code></td></tr>")
            for neg in match["negatives"]:
                set2_id = int(neg.get("set_2_id", -1))
                tags = set2[set2_id] if 0 <= set2_id < len(set2) else "(?)"
                difficulty = neg.get("difficulty", "?")
                parts.append(
                    f"<tr><td class='{difficulty}'>&minus; {difficulty}</td>"
                    f"<td>{set2_id}</td><td><code>{esc(tags)}</code></td></tr>")
            if match["uniqueness"] is not None:
                parts.append(
                    f"<tr><td colspan='3' style='color:#89a'>uniqueness "
                    f"{match['uniqueness']}/5</td></tr>")
        parts.append("</table>")
    if unmatched:
        parts.append("<h2>tracklets with no proposed match</h2><p>"
                     + ", ".join(f"<a href='#{esc(k)}'>{esc(k)}</a>"
                                 for k in sorted(unmatched)) + "</p>")
    parts.append("</body></html>")

    out = pairing / "review"
    out.mkdir(exist_ok=True)
    (out / "index.html").write_text("\n".join(parts))
    print(f"wrote {out}/index.html")


if __name__ == "__main__":
    main()
