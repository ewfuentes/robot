"""Interactive run viewer: the five §7.4 views as one self-contained page.

  1. **Run overview strip** — Tier-0 scalars as sparklines, mode lifespans as
     ribbons whose thickness is their weight, and the §7.3 event index as a
     clickable glyph rail. The entry point: open a run, look at the strip,
     click.
  2. **Map view** — optional satellite imagery, matcher-referenced landmarks
     glyphed by type, the particle cloud at the selected keyframe drawn as a
     *weighted* sample and coloured by mode, per-mode 1-sigma circles with
     heading ticks, bearing wedges from the selected mode, correspondence lines
     with opacity proportional to association posterior, and a red flag where
     the matcher's best claim disagrees with where that landmark actually lies.
  3. **Tracklet inspector** — one tracklet's completed-run evidence:
     bearing/kappa series, LLR bars per candidate, per-mode association
     evolution and attribution series, and a truth-privileged culpability
     verdict.
  4. **Mode ledger / genealogy** — modes as rows with weight trajectories,
     birth provenance, death keyframe, and a pre-computed death waterfall.
  5. **What-if console** — counterfactual runs ghost-overlaid on the map, with
     their own final/median error and mode count beside the baseline.

The page renders from `viewer_payload.build` and nothing else, so
`viewer_server.py` shows the same thing from the same data. Where the payload is
thin — no attribution cache or no ground truth — the
affected panel says why rather than rendering an empty box.

The static page publishes transactionally to
``<run_dir>.viewer/viewer.html`` by default. It is a reproducible side output;
the completed run directory is read-only and is never extended with viewer
files.

The stylesheet and the application script live as real files in
`viewer_assets/` (build-time data deps) and are INLINED here at render time:
editable and reviewable as CSS/JS, while the emitted page stays a single
self-contained offline file.

Two rules the page keeps:

**Truth-privileged content is fenced.** Anything derived from GPS truth is
visually marked and is evaluation output only; it is never a filter input.

**Nothing is silently truncated.** Where a cap applies — particles per frame
or table entries shown — the page states it.
"""

import argparse
import json
from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact, provenance
from experimental.overhead_matching.swag.farfield.localization import (
    side_outputs,
    viewer_payload,
)
from experimental.overhead_matching.swag.farfield.viewers import page

_ASSET_DIR = Path(__file__).parent / "viewer_assets"
# Read once at import: the files are bazel data deps, laid out next to this
# module in the runfiles tree exactly as in the source tree.
_STYLE = (_ASSET_DIR / "style.css").read_text()
_SCRIPT = (_ASSET_DIR / "app.js").read_text()

GENERATOR = ("//experimental/overhead_matching/swag/farfield/"
             "localization:viewer")


def render_html(payload: dict, body_only: bool = False) -> str:
    run = payload["run"]
    notes = payload.get("notes") or []
    notes_html = ""
    if notes:
        notes_html = ("<details class=\"notes\"><summary>"
                      f"{len(notes)} note(s) about this run's completeness"
                      "</summary><ul>"
                      + "".join(f"<li>{_escape(n)}</li>" for n in notes)
                      + "</ul></details>")
    replay_pill = ("ok" if run["replayable"] else "bad")
    replay_text = ("replayable" if run["replayable"]
                   else "not faithfully replayable")
    primary_evaluation = (
        run["runKind"] == "evaluation"
        and run["initialization"] == "uniform"
        and run["bearingsConsumed"])
    evaluation_pill = "ok" if primary_evaluation else "warn"
    evaluation_text = (
        "PRIMARY EVALUATION · uniform prior · bearings consumed"
        if primary_evaluation else
        f"{run['runKind']} · {run['initialization']} initialization")
    body = f"""<div class="wrap">
<header>
<div class="eyebrow">Bearing-only localization &middot; run viewer</div>
<h1>{_escape(run['scenario'])}</h1>
<div class="meta"><b>{run['nParticles']:,}</b> particles &middot;
<b>{run['nCatalog']:,}</b> catalog landmarks &middot;
<b>{run['nKeyframes']}</b> keyframes &middot; seed <b>{run['seed']}</b> &middot;
&pi;<sub>0</sub> <b>{run['pi0']}</b> &middot;
recall <b>{run['matcherRecall']}</b> &middot;
backend <b>{_escape(run['backend'])}</b> &middot;
matcher <b>{_escape(run['matcher'])}</b> &middot;
history <code>{_escape(run['historyHash'])}</code>
<span class="pill info" id="liveStatus">static viewer</span>
<span class="pill {replay_pill}">{replay_text}</span>
<span class="pill {evaluation_pill}">{_escape(evaluation_text)}</span></div>
<div class="meta" style="margin-top:4px">triage (truth-privileged):
{_escape(payload.get('triageSummary', ''))}</div>
{notes_html}
</header>
<div class="tiles" id="tiles"></div>
<main>
<section class="full">
<h2>Run overview <span class="hint">&mdash; &sect;7.4 view 1 &middot; click to
scrub, click a mode ribbon to isolate it, click an event glyph to jump</span></h2>
<div class="controls">
<button id="play">&#9654; play</button>
<input id="slider" type="range" min="0" value="0">
<span>kf <b id="kf">0</b></span>
<button id="clear">clear selection</button>
</div>
<svg id="strip"></svg>
</section>
<section>
<h2>Map <span class="hint">&mdash; &sect;7.4 view 2</span></h2>
<div class="controls" style="margin-bottom:8px">
<button id="tgPart" class="on">particles</button>
<button id="tgFull" disabled>full particles</button>
<button id="tgGhost" class="on">ghosts</button>
<button id="tgSat" class="on">satellite</button>
<button id="tgFitTrack">fit track</button>
<button id="tgFitAll">full extent</button>
<span class="axis" id="mapzoom" style="align-self:center"></span>
</div>
<svg id="map"></svg>
<div class="legend"><span id="mapnote"></span></div>
<div class="legend"><b>Scroll to zoom, drag to pan, double-click to zoom in.</b>
Opens fitted to the complete truth track (the estimate when truth is absent);
<b>full extent</b> restores the matched-landmark extent and <b>fit track</b>
returns.
Only landmarks used by the matcher are shown; unrelated OSM context is omitted.
The live server keeps the fast weighted particle sample unless <b>full
particles</b> is explicitly enabled. The zoom is in the projection, so landmark
glyphs, labels and flag rings keep a constant size while 1&sigma; circles and the
scale bar stay true to the ground.</div>
<div class="legend">Dashed grey is ground truth, magenta is the MAP trail
(latest 60 keyframes emphasized, older history faint), dashed blue are
counterfactual ghosts. Circles are per-mode 1&sigma; with a
heading tick; magenta rays are bearing wedges (&plusmn;2&sigma;) from the
selected mode; dashed coloured lines are correspondence with opacity &prop;
association posterior. A <b style="color:var(--port)">red dotted line and
ring</b> mark an LLR/geometry disagreement: the matcher's best claim lies more
than 15&deg; off the measured bearing under this mode. When a tracklet is
selected, its thick <b style="color:var(--privileged)">purple ray</b> is the
per-frame measured bearing projected from GPS truth at the current keyframe
(truth-privileged debugging only). Glyphs by type:
&#9651; light &middot; &#9671; navaid &middot; &#9711; tank &middot;
&#9770; tower/crane &middot; &#8801; bridge &middot; &#8852; pier/dock
&middot; &#9723; building. Filled and labelled means the filter is putting
association mass there at this keyframe.</div>
</section>
<section>
<div class="tabs" id="tabbar">
<button class="tab on" data-tab="state">State</button>
<button class="tab" data-tab="modes">Modes</button>
<button class="tab" data-tab="trklist">Tracklets</button>
<button class="tab" data-tab="attr">Attribution</button>
<button class="tab" data-tab="whatif">What-if</button>
</div>
<div><div id="state"></div></div>
<div><div id="modes"></div></div>
<div><div id="trklist"></div></div>
<div><div id="attr"></div></div>
<div><div id="whatif"></div></div>
</section>
<section class="full" id="inspector"></section>
<section class="full">
<h2>Event index <span class="hint">&mdash; &sect;7.3 &middot; the debugging
table of contents; click a row to jump</span></h2>
<div id="events"></div>
</section>
</main>
{page.provenance_footer(GENERATOR, css_class="prov", style="margin:0 0 40px")}
</div>
<script>window.__RUN__ = {_inline_json(payload)};</script>
<script>{_SCRIPT}</script>"""
    if body_only:
        # A fragment for embedding: no document, and therefore no generated
        # mark. Whoever embeds it owns the page it lands in.
        return f"<style>{_STYLE}</style>" + body
    return page.document(f"{run['scenario']} — run viewer", body,
                         style=_STYLE)


def _escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _inline_json(payload: dict) -> str:
    """JSON safe to embed inside a <script> element.

    `json.dumps` leaves `<` alone, so any string in the payload containing
    `</script>` would close the element and the rest of the payload would be
    parsed as HTML. Scenario names and landmark ids come from data files rather
    than from this code, so that is reachable input, not a hypothetical. The
    three escapes below are valid JSON string escapes, so the parsed value is
    unchanged.
    """
    return (json.dumps(payload, separators=(",", ":"))
            .replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026")
            # U+2028/U+2029 are literal line terminators in JS source but legal
            # inside a JSON string, so they break the script the same way.
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def write_viewer(run_dir: Path, payload: dict, *, output_dir: Path | None,
                 body_only: bool, inputs: dict, config: dict) -> Path:
    """Publish a self-contained viewer beside, never inside, its run."""
    with side_outputs.publish_directory(
            run_dir, output_dir=output_dir, suffix=".viewer") as output:
        viewer_path = output.staging_dir / "viewer.html"
        artifact.atomic_write_file(
            viewer_path, render_html(payload, body_only).encode("utf-8"))
        provenance.write(
            output.staging_dir,
            generator=GENERATOR,
            inputs=inputs,
            config=config)
    return output.destination / "viewer.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="defaults to the sibling <run_dir>.viewer; the "
                             "immutable run directory is never modified")
    parser.add_argument("--tracks_dir", type=Path, default=None,
                        help="exact object_tracks artifact that produced the run")
    parser.add_argument("--audit_dir", type=Path, default=None,
                        help="exact semantic_audits artifact bound to tracks_dir")
    parser.add_argument("--matcher_page", type=Path, default=None,
                        help="exact generated matcher review index.html to "
                             "link from each source tracklet")
    parser.add_argument("--audit_page", type=Path, default=None,
                        help="exact generated semantic-audit review index.html "
                             "to link from each source tracklet")
    parser.add_argument("--no_source_chips", action="store_true",
                        help="validate sources without embedding audit chips")
    parser.add_argument("--feather", type=Path, default=None,
                        help="accepted for command compatibility; unmatched "
                             "OSM context is no longer rendered")
    parser.add_argument("--ghost", type=Path, action="append", default=[],
                        help="counterfactual run directory to overlay "
                             "(repeatable)")
    parser.add_argument("--max_particles", type=int,
                        default=viewer_payload.MAX_PARTICLES_PER_FRAME)
    parser.add_argument("--satellite", type=Path, default=None,
                        help="directory written by satellite_underlay.py: "
                             "satellite.json naming the mosaic layers "
                             "(wide.jpg/fine.jpg) and their ENU bounds, "
                             "embedded as an imagery underlay. This must be "
                             "a side output separate from the run.")
    parser.add_argument("--basemap_detail", type=float, default=1.0,
                        help="deprecated compatibility option; unmatched OSM "
                             "context is no longer rendered")
    parser.add_argument("--body_only", action="store_true",
                        help="emit a fragment for embedding rather than a "
                             "standalone document")
    args = parser.parse_args()

    # Auto-discover the run's satellite sibling: satellite_underlay publishes
    # to `<run>.satellite` by convention, so an explicit flag should only be
    # needed to point somewhere else. The absence is printed rather than
    # silent — an imagery-less page kept getting read as "imagery was never
    # generated" when it was only never wired in.
    if args.satellite is None:
        sibling = side_outputs.default_directory(args.run_dir, ".satellite")
        if (sibling / "satellite.json").is_file():
            args.satellite = sibling
            print(f"satellite: using {sibling}")
        else:
            print(f"satellite: none at {sibling} — run "
                  "localization:satellite_underlay --run_dir <run> "
                  "--date <yyyy-mm> to add imagery")

    viewer_dir = (args.output_dir if args.output_dir is not None else
                  side_outputs.default_directory(args.run_dir, ".viewer"))
    payload = viewer_payload.build(
        args.run_dir, tracks_dir=args.tracks_dir, audit_dir=args.audit_dir,
        feather=args.feather,
        ghost_dirs=args.ghost, max_particles=args.max_particles,
        embed_source_chips=not args.no_source_chips,
        basemap_detail=args.basemap_detail,
        satellite=args.satellite,
        viewer_dir=viewer_dir,
        matcher_page=args.matcher_page,
        audit_page=args.audit_page)
    run_ref = artifact.open_artifact(
        args.run_dir, expected_kind="localization_run")
    output = write_viewer(
        args.run_dir,
        payload,
        output_dir=args.output_dir,
        body_only=args.body_only,
        inputs={
            "run_dir": args.run_dir.resolve(),
            "run_manifest_digest": run_ref.manifest_digest,
            "tracks_dir": (args.tracks_dir.resolve()
                           if args.tracks_dir is not None else ""),
            "tracks_manifest_digest": (
                artifact.open_artifact(args.tracks_dir).manifest_digest
                if args.tracks_dir is not None else ""),
            "audit_dir": (args.audit_dir.resolve()
                          if args.audit_dir is not None else ""),
            "audit_manifest_digest": (
                artifact.open_artifact(args.audit_dir).manifest_digest
                if args.audit_dir is not None else ""),
            "matcher_page": (args.matcher_page.resolve()
                             if args.matcher_page is not None else ""),
            "matcher_page_sha256": (
                artifact.sha256_file(args.matcher_page)
                if args.matcher_page is not None else ""),
            "audit_page": (args.audit_page.resolve()
                           if args.audit_page is not None else ""),
            "audit_page_sha256": (
                artifact.sha256_file(args.audit_page)
                if args.audit_page is not None else ""),
            "feather": (args.feather.resolve()
                        if args.feather is not None else ""),
            "feather_sha256": (artifact.sha256_file(args.feather)
                               if args.feather is not None else ""),
            "ghosts": [path.resolve() for path in args.ghost],
            "satellite": (args.satellite.resolve()
                          if args.satellite is not None else ""),
        },
        config={
            "max_particles": args.max_particles,
            "basemap_detail": args.basemap_detail,
            "body_only": args.body_only,
            "embed_source_chips": not args.no_source_chips,
        })

    size_kb = output.stat().st_size / 1024
    print(f"Wrote {output} ({size_kb:,.0f} KB)")
    print(f"  {len(payload['health'])} keyframes, "
          f"{len(payload['checkpoints'])} checkpoints, "
          f"{len(payload['tracklets'])} tracklets, "
          f"{len(payload['events'])} events, "
          f"{len(payload['landmarks'])} referenced landmarks, "
          f"{len(payload['ghosts'])} ghosts")
    print(f"  attribution: "
          + ("absent" if not payload["attribution"]
             else f"{payload['attribution']['nRows']} rows, "
                  f"verified={payload['attribution']['verified']}"))
    print(f"  triage: {payload['triageSummary']}")
    for note in payload["notes"]:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
