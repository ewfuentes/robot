"""Debug viewer for CVG-Text panos.

Per pano, shows:
- CVG-Text text description (the CT2L text query)
- raw pano image
- GT satellite tile + GT OSM-rasterised tile
- Gemini-extracted landmark list (from the pano_v2 batch predictions JSONL)
- GT retrieval rank under every similarity matrix discovered for this city
  (CT2L {train×kind}, Exp A WAG, Exp B correspondence, plus any `extra_roots`
  reproductions), sorted best-to-worst for this pano.

One pano per page; prev/next nav (or ← / → keys) walks through the test set.
Toggle `?empty=1` to walk only panos Gemini returned zero landmarks for.
`/list` gives a terse filename index.

Defaults assume the standard local layout:
  CVG-Text root: /data/overhead_matching/datasets/cvgtext
  pano_v2 base:  /data/overhead_matching/datasets/semantic_landmark_embeddings/cvgtext_pano_v2_base
  results root:  /data/overhead_matching/evaluation/results/cvgtext
  VIGOR root:    /data/overhead_matching/datasets/VIGOR   (for Exp B row mapping)

Usage:
  bazel run //experimental/overhead_matching/swag/scripts:cvgtext_pano_landmark_viewer -- \
    --city Brisbane --port 5001 \
    [--extra_roots /data/overhead_matching/evaluation/results/cvgtext_repro]
Then open http://localhost:5001/.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import numpy as np
import torch
from flask import Flask, abort, redirect, request, send_file, url_for


CITIES = ("Brisbane", "NewYork", "Tokyo")

_CVGTEXT_FN = re.compile(
    r"^(?P<lat>-?[\d.]+),(?P<lon>-?[\d.]+)_"
    r"(?P<date>\d{4}-\d{2})_"
    r"(?P<pano_id>.+)_"
    r"d(?P<yaw>\d+)_"
    r"z(?P<zoom>\d+)"
    r"\.(?:png|jpg|jpeg)$"
)


def _canonical_key(fn: str) -> str:
    m = _CVGTEXT_FN.match(fn)
    if m is None:
        raise ValueError(f"cannot parse CVG-Text filename: {fn!r}")
    g = m.groupdict()
    return f"{g['pano_id']}_{g['date']}_d{g['yaw']}"


def _google_maps_links(fn: str) -> tuple[str, str]:
    """Return (maps_url, streetview_url) from a CVG-Text filename."""
    m = _CVGTEXT_FN.match(fn)
    if m is None:
        return "", ""
    g = m.groupdict()
    lat, lon, yaw = g["lat"], g["lon"], g["yaw"]
    maps = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
    street = (f"https://www.google.com/maps/@?api=1&map_action=pano"
              f"&viewpoint={lat},{lon}&heading={yaw}")
    return maps, street


def load_gemini_predictions(pano_v2_base: Path, city: str) -> dict[str, dict]:
    """Return {key: {raw, landmarks, location_type}} from predictions.jsonl."""
    results_dir = pano_v2_base / city / "sentences" / "results" / "panorama_request_000"
    if not results_dir.exists():
        raise FileNotFoundError(f"No sentences results dir at {results_dir}")
    pred_dirs = [d for d in sorted(results_dir.iterdir())
                 if d.is_dir() and d.name.startswith("prediction-model-")]
    if not pred_dirs:
        raise FileNotFoundError(f"No prediction-model-* under {results_dir}")
    jsonl = pred_dirs[-1] / "predictions.jsonl"
    print(f"Loading {jsonl}")

    out: dict[str, dict] = {}
    with open(jsonl) as f:
        for line in f:
            d = json.loads(line)
            key = d["key"]
            cands = d.get("response", {}).get("candidates", [])
            text = cands[0].get("content", {}).get("parts", [{}])[0].get("text", "") if cands else ""
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            out[key] = {
                "raw": text,
                "landmarks": parsed.get("landmarks", []) if parsed else [],
                "location_type": parsed.get("location_type", "?") if parsed else "?",
                "parse_ok": parsed is not None,
            }
    return out


def build_pano_records(cvgtext_root: Path, pano_v2_base: Path, city: str) -> list[dict]:
    ann_path = cvgtext_root / "annotation" / city / "test.json"
    ann = json.load(open(ann_path))
    gemini = load_gemini_predictions(pano_v2_base, city)
    missing_keys = 0
    panos = []
    for fn, text in ann.items():
        key = Path(fn).stem
        g = gemini.get(key)
        if g is None:
            missing_keys += 1
            g = {"raw": "", "landmarks": [], "location_type": "?", "parse_ok": False}
        panos.append({"filename": fn, "text": text, "gemini": g})
    if missing_keys:
        print(f"  [warn] {missing_keys} test-split filenames had no predictions.jsonl entry")
    return panos


def _short_method_name(dir_name: str, suffix: str = "") -> str:
    if m := re.match(r"^crosstext2loc_train-(\w+)_test-(\w+)_(sat|osm)$", dir_name):
        return f"CT2L {m.group(1)}→{m.group(2)} {m.group(3)}" + suffix
    if m := re.match(rf"^expA_test-({'|'.join(CITIES)})_", dir_name):
        return f"WAG {m.group(1)}" + suffix
    if m := re.match(rf"^expB_({'|'.join(CITIES)})$", dir_name):
        return f"ExpB {m.group(1)}" + suffix
    return dir_name + suffix


def _compute_ranks(sim: torch.Tensor, gt_idxs: np.ndarray) -> np.ndarray:
    rankings = torch.argsort(sim, dim=1, descending=True)
    gt = torch.as_tensor(gt_idxs, dtype=torch.long)
    mask = rankings == gt.unsqueeze(1)
    return mask.long().argmax(dim=1).cpu().numpy().astype(np.int64)


def load_method_ranks(
    results_roots: list[Path],
    cvgtext_root: Path,
    vigor_root: Path,
    city: str,
    pano_filenames: list[str],
) -> list[dict]:
    """Return a list of {short_name, n_gallery, ranks_by_fn, mrr}.

    `ranks_by_fn[filename]` gives the 0-indexed GT rank for that pano under
    this method's similarity matrix. All methods' ranks are keyed by the same
    CVG-Text pano filename so the detail page can look up rows uniformly.
    """
    sat_gallery_names = sorted(
        p.name for p in (cvgtext_root / "reference" / f"{city}-satellite").iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    sat_name_to_idx = {n: i for i, n in enumerate(sat_gallery_names)}

    osm_dir = cvgtext_root / "reference" / f"{city}-OSM"
    osm_sat_name_to_idx: dict[str, int] = {}
    if osm_dir.exists():
        osm_gallery_names = sorted(
            p.name for p in osm_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        osm_sat_name_to_idx = {n: i for i, n in enumerate(osm_gallery_names)}

    # VIGOR pano row → canonical key, for Exp B
    staged = vigor_root / f"cvgtext_{city}"
    vigor_perm_by_key: dict[str, int] | None = None
    vigor_sat_gt: dict[str, int] | None = None
    if staged.exists():
        pano_files = sorted((staged / "panorama").iterdir())
        sat_files = sorted((staged / "satellite").iterdir())
        vigor_perm_by_key = {}
        for i, pp in enumerate(pano_files):
            key = pp.stem.split(",")[0]
            vigor_perm_by_key[key] = i
        vigor_sat_latlon_to_idx: dict[tuple[str, str], int] = {}
        for i, sp in enumerate(sat_files):
            parts = sp.stem.split("_")
            vigor_sat_latlon_to_idx[(parts[-2], parts[-1])] = i
        vigor_sat_gt_list: list[int] = []
        for pp in pano_files:
            parts = pp.name.split(",")
            vigor_sat_gt_list.append(vigor_sat_latlon_to_idx[(parts[1], parts[2])])
        vigor_sat_gt = vigor_sat_gt_list

    pano_keys = [_canonical_key(fn) for fn in pano_filenames]

    methods: list[dict] = []

    for root_idx, root in enumerate(results_roots):
        if not root.exists():
            continue
        suffix = "" if root_idx == 0 else f" ({root.name})"
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            name = d.name
            short = None
            sim_path: Path | None = None
            kind = None
            if m := re.match(r"^crosstext2loc_train-(\w+)_test-(\w+)_(sat|osm)$", name):
                if m.group(2) != city:
                    continue
                short = _short_method_name(name, suffix)
                sim_path = d / "similarity.pt"
                kind = "sat" if m.group(3) == "sat" else "osm"
            elif m := re.match(rf"^expA_test-({'|'.join(CITIES)})_", name):
                if m.group(1) != city:
                    continue
                short = _short_method_name(name, suffix)
                sim_path = d / "similarity.pt"
                kind = "sat"
            elif m := re.match(rf"^expB_({'|'.join(CITIES)})$", name):
                if m.group(1) != city:
                    continue
                short = _short_method_name(name, suffix)
                sim_path = d / "simple_v1_raw_similarity.pt"
                kind = "vigor_sat"
            else:
                continue
            if not sim_path.exists():
                print(f"  [skip] missing {sim_path}")
                continue
            sim = torch.load(sim_path, map_location="cpu", weights_only=True).float()

            if kind == "sat":
                gt_idxs = np.array(
                    [sat_name_to_idx[fn] for fn in pano_filenames], dtype=np.int64
                )
                ranks = _compute_ranks(sim, gt_idxs)
                ranks_by_fn = dict(zip(pano_filenames, ranks.tolist()))
            elif kind == "osm":
                if not osm_sat_name_to_idx:
                    print(f"  [skip] no OSM gallery for {city}")
                    continue
                gt_idxs = np.array(
                    [osm_sat_name_to_idx[fn] for fn in pano_filenames], dtype=np.int64
                )
                ranks = _compute_ranks(sim, gt_idxs)
                ranks_by_fn = dict(zip(pano_filenames, ranks.tolist()))
            elif kind == "vigor_sat":
                if vigor_perm_by_key is None or vigor_sat_gt is None:
                    print(f"  [skip] Exp B needs VIGOR staging at {staged}")
                    continue
                ranks_vigor = _compute_ranks(sim, np.array(vigor_sat_gt, dtype=np.int64))
                ranks_by_fn = {}
                for fn, key in zip(pano_filenames, pano_keys):
                    vi = vigor_perm_by_key.get(key)
                    if vi is None:
                        continue
                    ranks_by_fn[fn] = int(ranks_vigor[vi])
            else:
                continue

            if not ranks_by_fn:
                continue
            r = np.array(list(ranks_by_fn.values()), dtype=np.int64)
            mrr = float((1.0 / (r + 1.0)).mean())
            methods.append({
                "short": short,
                "dir_name": name,
                "n_gallery": int(sim.shape[1]),
                "ranks_by_fn": ranks_by_fn,
                "mrr": mrr,
                "kind": kind,
            })
            print(f"  loaded {short}: n_gallery={sim.shape[1]}, MRR={mrr:.4f}")
    return methods


def _fmt_landmark(lm) -> str:
    if isinstance(lm, str):
        return f"<li>{lm}</li>"
    if not isinstance(lm, dict):
        return f"<li>{lm!r}</li>"
    primary = lm.get("primary_tag") or {}
    key = primary.get("key", "?")
    value = primary.get("value", "?")
    additional = lm.get("additional_tags") or []
    name = next(
        (a.get("value") for a in additional
         if isinstance(a, dict) and a.get("key") == "name"),
        None,
    )
    conf = lm.get("confidence", "")
    desc = lm.get("description", "")
    extras = ", ".join(
        f"{a.get('key')}={a.get('value')}"
        for a in additional if isinstance(a, dict) and a.get("key") != "name"
    )
    head = f"<b>{key}={value}</b>"
    if name:
        head += f' &ldquo;{name}&rdquo;'
    if conf:
        head += f" <span class='conf'>[{conf}]</span>"
    body = f"<div class='desc'>{desc}</div>" if desc else ""
    if extras:
        body += f"<div class='extras'>{extras}</div>"
    return f"<li>{head}{body}</li>"


def _rank_row_html(method: dict, fn: str) -> str | None:
    rank = method["ranks_by_fn"].get(fn)
    if rank is None:
        return None
    n = method["n_gallery"]
    rank_1idx = rank + 1
    cls = ""
    if rank == 0:
        cls = "hit1"
    elif rank < 5:
        cls = "hit5"
    elif rank < 10:
        cls = "hit10"
    pct = 100.0 * rank / max(1, n - 1)
    return (
        f"<tr class='{cls}'>"
        f"<td class='m'>{method['short']}</td>"
        f"<td class='r'>{rank_1idx}</td>"
        f"<td class='n'>/ {n}</td>"
        f"<td class='p'>{pct:.1f}%</td>"
        f"<td class='k'>{'R@1' if rank < 1 else ('R@5' if rank < 5 else ('R@10' if rank < 10 else ''))}</td>"
        f"</tr>"
    )


DETAIL_HTML = """<!doctype html>
<html><head>
<title>{city} · {pos} of {total} · {fn}</title>
<style>
  body {{ font-family: sans-serif; margin: 0; padding: 0; background: #fafafa; color: #222; }}
  .nav {{ position: sticky; top: 0; background: #fff; padding: 0.6em 1em;
          border-bottom: 1px solid #ccc; z-index: 10;
          display: flex; align-items: center; gap: 1em; flex-wrap: wrap; }}
  .nav a {{ text-decoration: none; padding: 0.3em 0.7em; border: 1px solid #ccc;
           border-radius: 3px; color: #222; background: #f5f5f5; }}
  .nav a:hover {{ background: #eaeaea; }}
  .nav .disabled {{ color: #aaa; border-color: #eee; background: #fafafa; pointer-events: none; }}
  .body {{ padding: 1em 1.5em; max-width: 1600px; }}
  h2.fn {{ margin: 0.5em 0 0.2em 0; font-family: monospace; font-size: 0.95em;
          color: #666; word-break: break-all; font-weight: normal; }}
  .loc-links {{ margin: 0.3em 0 0.6em 0; font-size: 0.9em; }}
  .loc-links a {{ padding: 0.2em 0.6em; margin-right: 0.5em;
                  border: 1px solid #ccc; border-radius: 3px;
                  text-decoration: none; color: #036; background: #f0f4f8; }}
  .loc-links a:hover {{ background: #e0eaf0; }}
  .text {{ font-size: 1.15em; font-style: italic; color: #111; background: #fff;
           border-left: 4px solid #88b; padding: 0.8em 1em; margin: 0.8em 0 1.2em 0; }}
  img.pano {{ width: 100%; max-width: 1600px; height: auto; display: block;
              margin: 0 auto; background: #eee; border: 1px solid #ddd; }}
  .row2 {{ display: flex; gap: 1.5em; margin: 1.2em 0; align-items: flex-start; }}
  .tiles {{ display: flex; gap: 1em; }}
  .tile {{ text-align: center; }}
  .tile h3 {{ font-size: 0.85em; color: #666; margin: 0 0 0.3em 0; font-weight: normal; }}
  .tile img {{ width: 500px; height: 500px; object-fit: contain; background: #eee;
               border: 1px solid #ddd; }}
  .ranks {{ flex: 1; background: #fff; border: 1px solid #ddd; padding: 0.8em 1em;
            font-size: 0.9em; }}
  .ranks h3 {{ margin: 0 0 0.5em 0; font-size: 1em; }}
  .ranks table {{ border-collapse: collapse; width: 100%; }}
  .ranks td {{ padding: 0.25em 0.5em; border-bottom: 1px solid #f0f0f0;
               font-variant-numeric: tabular-nums; }}
  .ranks td.m {{ font-family: sans-serif; }}
  .ranks td.r {{ text-align: right; font-weight: bold; }}
  .ranks td.n {{ color: #888; font-size: 0.85em; }}
  .ranks td.p {{ text-align: right; color: #666; font-size: 0.85em; }}
  .ranks td.k {{ color: #080; font-size: 0.85em; }}
  .ranks tr.hit1 td {{ background: #e0f5d6; }}
  .ranks tr.hit5 td {{ background: #f5f5d0; }}
  .ranks tr.hit10 td {{ background: #fbf3d0; }}
  .gemini {{ background: #fff; padding: 1em; margin: 1em 0; border: 1px solid #ddd; }}
  .gemini h3 {{ margin: 0 0 0.5em 0; }}
  .gemini .loctype {{ color: #555; font-size: 0.9em; margin-bottom: 0.5em; }}
  .gemini ul {{ margin: 0; padding-left: 1.2em; }}
  .gemini li {{ margin: 0.6em 0; }}
  .gemini .desc {{ color: #333; font-size: 0.9em; margin-top: 0.2em; }}
  .gemini .extras {{ color: #777; font-size: 0.85em; font-family: monospace; }}
  .gemini .conf {{ color: #888; font-size: 0.85em; }}
  .empty {{ color: #a00; font-weight: bold; }}
</style>
</head><body>
<div class="nav">
  <strong>{city}</strong>
  <span>{pos} of {total}{filter_tag}</span>
  <a href="{first_url}">« first</a>
  {prev_a}
  {next_a}
  <a href="{last_url}">last »</a>
  <a href="{toggle_url}">{toggle_label}</a>
  <a href="/list{empty_q}">list view</a>
</div>
<div class="body">
  <h2 class="fn">{fn}</h2>
  <div class="loc-links">
    <a href="{maps_url}" target="_blank" rel="noopener">Google Maps ↗</a>
    <a href="{street_url}" target="_blank" rel="noopener">Street View ↗</a>
  </div>
  <div class="text">&ldquo;{text}&rdquo;</div>
  <img class="pano" src="/image/pano/{fn}" alt="pano">
  <div class="row2">
    <div class="tiles">
      <div class="tile"><h3>Ground-truth satellite tile</h3><img src="/image/satellite/{fn}"></div>
      <div class="tile"><h3>Ground-truth OSM tile</h3><img src="/image/OSM/{fn}"></div>
    </div>
    <div class="ranks">
      <h3>GT rank per method (sorted best-first)</h3>
      <table>{ranks_rows}</table>
    </div>
  </div>
  <div class="gemini">
    <h3>Gemini-extracted pano landmarks ({n_lm})</h3>
    <div class="loctype">location_type: <em>{loctype}</em></div>
    {lm_block}
  </div>
</div>
<script>
  const prev = {prev_js}, next = {next_js};
  document.addEventListener('keydown', e => {{
    if (e.key === 'ArrowLeft' && prev) window.location = prev;
    if (e.key === 'ArrowRight' && next) window.location = next;
  }});
</script>
</body></html>"""


LIST_HTML = """<!doctype html>
<html><head>
<title>{city} pano list{filter_tag}</title>
<style>
  body {{ font-family: sans-serif; margin: 1em; }}
  .nav {{ padding: 0.5em; border-bottom: 1px solid #ccc; margin-bottom: 1em; }}
  ol {{ padding-left: 2em; }}
  li {{ margin: 0.4em 0; }}
  li .fn {{ font-family: monospace; font-size: 0.85em; color: #666; }}
  li .snip {{ color: #333; font-style: italic; }}
  li .n {{ color: #060; font-size: 0.85em; }}
  li .n0 {{ color: #a00; font-weight: bold; font-size: 0.85em; }}
</style>
</head><body>
<div class="nav">
  <strong>{city}</strong> ·
  {n_total} entries{filter_tag} ·
  <a href="{toggle_url}">{toggle_label}</a> ·
  <a href="/">detail view</a>
</div>
<ol>
{body}
</ol>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--city", choices=CITIES, default="Brisbane")
    ap.add_argument("--cvgtext_root", type=Path,
                    default=Path("/data/overhead_matching/datasets/cvgtext"))
    ap.add_argument("--pano_v2_base", type=Path,
                    default=Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/cvgtext_pano_v2_base"))
    ap.add_argument("--results_root", type=Path,
                    default=Path("/data/overhead_matching/evaluation/results/cvgtext"))
    ap.add_argument("--extra_roots", type=Path, nargs="*", default=[])
    ap.add_argument("--vigor_root", type=Path,
                    default=Path("/data/overhead_matching/datasets/VIGOR"))
    ap.add_argument("--port", type=int, default=5001)
    args = ap.parse_args()

    panos = build_pano_records(args.cvgtext_root, args.pano_v2_base, args.city)
    total_empty = sum(1 for p in panos if not p["gemini"]["landmarks"])
    print(f"{args.city}: {len(panos)} test panos, {total_empty} with 0 Gemini landmarks "
          f"({100 * total_empty / len(panos):.1f}%)")

    methods = load_method_ranks(
        [args.results_root, *args.extra_roots],
        args.cvgtext_root, args.vigor_root, args.city,
        [p["filename"] for p in panos],
    )
    print(f"Loaded {len(methods)} retrieval methods for rank display")

    app = Flask(__name__)

    def _subset(empty: bool) -> list[dict]:
        if empty:
            return [p for p in panos if not p["gemini"]["landmarks"]]
        return panos

    def _pano_url(fn: str, empty: bool) -> str:
        return url_for("pano", filename=fn) + ("?empty=1" if empty else "")

    @app.route("/")
    def home():
        empty = request.args.get("empty") == "1"
        subset = _subset(empty)
        if not subset:
            return "No panos match the current filter.", 404
        return redirect(_pano_url(subset[0]["filename"], empty))

    @app.route("/pano/<path:filename>")
    def pano(filename):
        empty = request.args.get("empty") == "1"
        subset = _subset(empty)
        names = [p["filename"] for p in subset]
        if filename not in names:
            if filename not in [p["filename"] for p in panos]:
                abort(404)
            empty = False
            subset = panos
            names = [p["filename"] for p in subset]
        i = names.index(filename)
        p = subset[i]
        n = len(names)
        prev_fn = names[i - 1] if i > 0 else None
        next_fn = names[i + 1] if i < n - 1 else None

        lms = p["gemini"]["landmarks"]
        if lms:
            lm_block = "<ul>" + "".join(_fmt_landmark(lm) for lm in lms) + "</ul>"
        else:
            parse_note = "" if p["gemini"]["parse_ok"] else " (JSON parse failed)"
            lm_block = f'<div class="empty">0 landmarks extracted{parse_note}</div>'

        # Rank rows sorted best-first for this pano
        rows = []
        for m in methods:
            rank = m["ranks_by_fn"].get(p["filename"])
            if rank is None:
                continue
            rows.append((rank, _rank_row_html(m, p["filename"])))
        rows.sort(key=lambda t: t[0])
        ranks_rows = "".join(html for _, html in rows) or "<tr><td>(no ranks available)</td></tr>"

        filter_tag = " · filter: 0-landmark only" if empty else ""
        empty_q = "?empty=1" if empty else ""
        # Toggle: turning filter OFF keeps us on the current pano (it's always in the
        # unfiltered set). Turning it ON keeps us here if the current pano is itself
        # 0-landmark; otherwise jump to the first 0-landmark pano (`/?empty=1` redirects).
        if empty:
            toggle_url = _pano_url(p["filename"], False)
        elif not p["gemini"]["landmarks"]:
            toggle_url = _pano_url(p["filename"], True)
        else:
            toggle_url = "/?empty=1"
        toggle_label = "show all" if empty else "show 0-landmark only"

        def nav_link(label, fn):
            if fn is None:
                return f'<span class="disabled">{label}</span>'
            return f'<a href="{_pano_url(fn, empty)}">{label}</a>'

        maps_url, street_url = _google_maps_links(p["filename"])
        return DETAIL_HTML.format(
            city=args.city,
            pos=i + 1, total=n, fn=p["filename"],
            maps_url=maps_url, street_url=street_url,
            filter_tag=filter_tag,
            first_url=_pano_url(names[0], empty),
            last_url=_pano_url(names[-1], empty),
            prev_a=nav_link("‹ prev", prev_fn),
            next_a=nav_link("next ›", next_fn),
            toggle_url=toggle_url, toggle_label=toggle_label,
            empty_q=empty_q,
            text=p["text"].replace("<", "&lt;"),
            n_lm=len(lms), loctype=p["gemini"]["location_type"],
            lm_block=lm_block,
            ranks_rows=ranks_rows,
            prev_js=json.dumps(_pano_url(prev_fn, empty) if prev_fn else None),
            next_js=json.dumps(_pano_url(next_fn, empty) if next_fn else None),
        )

    @app.route("/list")
    def list_view():
        empty = request.args.get("empty") == "1"
        subset = _subset(empty)
        lines = []
        for p in subset:
            n_lm = len(p["gemini"]["landmarks"])
            cls = "n0" if n_lm == 0 else "n"
            snip = p["text"][:120].replace("<", "&lt;")
            lines.append(
                f'<li><a href="{_pano_url(p["filename"], empty)}">'
                f'<span class="fn">{p["filename"]}</span></a> '
                f'<span class="{cls}">[{n_lm} lm]</span> '
                f'<span class="snip">{snip}</span></li>'
            )
        return LIST_HTML.format(
            city=args.city, n_total=len(subset),
            filter_tag=" (0-landmark only)" if empty else "",
            toggle_url="/list" if empty else "/list?empty=1",
            toggle_label="show all" if empty else "show 0-landmark only",
            body="\n".join(lines),
        )

    @app.route("/image/<kind>/<path:filename>")
    def image(kind, filename):
        if kind == "pano":
            fpath = args.cvgtext_root / "data" / "query" / f"{args.city}-ground" / filename
        elif kind == "satellite":
            fpath = args.cvgtext_root / "reference" / f"{args.city}-satellite" / filename
        elif kind == "OSM":
            fpath = args.cvgtext_root / "reference" / f"{args.city}-OSM" / filename
        else:
            abort(404)
        if not fpath.exists():
            abort(404)
        return send_file(fpath)

    print(f"Serving on http://127.0.0.1:{args.port}  (try /?empty=1)")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
