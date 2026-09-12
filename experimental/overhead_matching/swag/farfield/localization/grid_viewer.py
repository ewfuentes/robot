"""Self-contained viewer for one ``farfield_causal_grid/v1`` result.

The grid filter has no particles and does not smooth old states.  This page
therefore shows the online position-MAP estimate emitted at each keyframe, the
final sparse grid modes, and the exact matcher-endorsed landmark candidates.
Natural-closure measurements are shown at their original anchors even though
the complete track becomes available atomically at its later release frame.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import urllib.parse
from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact, paths
from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    forensics,
    release_schedule,
)
from experimental.overhead_matching.swag.farfield.viewers import page


SCHEMA = "farfield_causal_grid/v1"
GENERATOR = ("//experimental/overhead_matching/swag/farfield/"
             "localization:grid_viewer")
_STYLE = (Path(__file__).parent / "viewer_assets" / "style.css").read_text()
_TRACKLET_RE = re.compile(r"#(T[0-9]+)\Z")
_MATCHER_VIEWER_GENERATOR = (
    "//experimental/overhead_matching/swag/farfield/matching:match_viewer")

_EXTRA_STYLE = r"""
body{margin:0;background:#11151b;color:#d8dce2;font-family:system-ui,sans-serif}
.wrap{max-width:1500px;margin:auto;padding:22px}.meta,.note{color:#96a0ad}
.note{font-size:12px;margin:7px 0}.grid{display:grid;grid-template-columns:2fr 1fr;
gap:14px}.panel{background:#171c23;border:1px solid #2b323d;border-radius:7px;
padding:12px;min-width:0}h1{margin:0 0 4px;font-size:23px}h2{font-size:15px;
margin:0 0 9px;color:#b9c2ce}button,input{accent-color:#d05a98}button{background:#252c36;
color:#d8dce2;border:1px solid #3b4553;border-radius:4px;padding:5px 9px}
button.on{border-color:#d05a98}.controls{display:flex;align-items:center;gap:8px;
margin:7px 0}.controls input{flex:1}svg{width:100%;display:block;background:#10141a;
border:1px solid #29303a;border-radius:4px}#map{height:620px}#metric{height:150px}
.truth{stroke:#9ba4ae;stroke-dasharray:7 5;fill:none}.estimate{stroke:#df5ba2;
fill:none}.candidate{fill:#63a1c9;opacity:.35}.selected{fill:#ffd166;opacity:.9}
.mode{fill:#b680d7;stroke:#eddcff}.anchor{fill:none;stroke:#ffd166;
stroke-width:2}.state{stroke:#fff;stroke-width:2;fill:none}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#9da7b4}
.sw{display:inline-block;width:12px;height:3px;margin-right:4px;vertical-align:middle}
.scroll{max-height:430px;overflow:auto}table{border-collapse:collapse;width:100%}
th,td{font-size:12px;padding:5px 7px;border-bottom:1px solid #2b323d;text-align:left}
tr.pick{cursor:pointer}tr.pick:hover,tr.pick.on{background:#27303b}.new{color:#ffd166}
.available{color:#74c69d}.pending{color:#ed9b61}.disabled{color:#8b93a3}
.links a{margin-right:9px}.candidate-list{max-height:310px;overflow:auto}
.candidate-list code{font-size:11px}.metric-label{font-size:11px;fill:#aab3bf}
@media(max-width:900px){.grid{grid-template-columns:1fr}#map{height:480px}}
"""


def _load_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"result must be a regular file: {path}")

    def reject_duplicates(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key {key!r}")
            out[key] = value
        return out

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid result JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("result JSON must contain an object")
    return value


def _one_upstream(manifest, kind: str, label: str):
    found = [item for item in manifest.upstreams if item.kind == kind]
    if len(found) != 1:
        raise ValueError(f"{label} must bind exactly one {kind} artifact")
    return found[0]


def _open_ref(reference):
    opened = artifact.open_artifact(
        reference.path, expected_kind=reference.kind,
        expected_dataset=reference.dataset, expected_version=reference.version)
    if opened != reference:
        raise ValueError(f"stale {reference.kind} artifact reference")
    return artifact.load_manifest(opened.path)


def _lineage(data):
    """Exact matching/tracks/audits/catalog ancestors of the export."""
    matching = _one_upstream(data.manifest, paths.LANDMARK_MATCHES,
                             "localization_inputs")
    bearing = _one_upstream(data.manifest, paths.BEARING_OBSERVATIONS,
                            "localization_inputs")
    catalog = _one_upstream(data.manifest, paths.CATALOGS,
                            "localization_inputs")
    matching_manifest = _open_ref(matching)
    bearing_manifest = _open_ref(bearing)
    _open_ref(catalog)
    tracks = _one_upstream(matching_manifest, paths.OBJECT_TRACKS,
                           "landmark_matches")
    audits = _one_upstream(matching_manifest, paths.SEMANTIC_AUDITS,
                           "landmark_matches")
    matching_catalog = _one_upstream(
        matching_manifest, paths.CATALOGS, "landmark_matches")
    bearing_tracks = _one_upstream(
        bearing_manifest, paths.OBJECT_TRACKS, "bearing_observations")
    bearing_audits = _one_upstream(
        bearing_manifest, paths.SEMANTIC_AUDITS, "bearing_observations")
    if (matching_catalog != catalog or bearing_tracks != tracks
            or bearing_audits != audits):
        raise ValueError("export branches disagree on tracks, audits, or catalog")
    return matching, matching_manifest, tracks, _open_ref(tracks), audits, \
        _open_ref(audits), catalog


def _portable_href(output: Path, target: Path, anchor: str | None = None):
    if target.is_symlink() or not target.is_file():
        return None
    value = Path(os.path.relpath(target.resolve(), output.parent.resolve())) \
        .as_posix()
    if anchor:
        value += "#" + urllib.parse.quote(anchor, safe="")
    return value


def _matching_review(output: Path, matching, tracks, audits, catalog):
    directory = Path(matching.path).with_name(
        Path(matching.path).name + ".matcher-review")
    manifest_path = directory / artifact.MANIFEST_NAME
    index = directory / "index.html"
    if (directory.is_symlink() or manifest_path.is_symlink()
            or not manifest_path.is_file() or index.is_symlink()
            or not index.is_file()):
        return None
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    inputs = record.get("inputs")
    expected = {
        "matching": Path(matching.path),
        "tracks": Path(tracks.path),
        "semantic_audits": Path(audits.path),
        "catalog": Path(catalog.path),
    }
    if (record.get("generator") != _MATCHER_VIEWER_GENERATOR
            or not isinstance(inputs, dict)):
        return None
    for key, wanted in expected.items():
        try:
            if Path(inputs.get(key, "")).resolve() != wanted.resolve():
                return None
        except (OSError, RuntimeError):
            return None
    return index


def _finite_number(value, label: str, *, low=None, high=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (low is not None and value < low) \
            or (high is not None and value > high):
        raise ValueError(f"{label} is outside its valid range")
    return value


def _state_trace(result: dict, data) -> list[dict]:
    record = result.get("online_map_state_by_keyframe")
    states = record.get("states") if isinstance(record, dict) else record
    if not isinstance(states, list) or len(states) != data.n_keyframes:
        raise ValueError(
            "result has no complete online MAP state trace; rerun grid_filter "
            "with state recording enabled")
    out = []
    for keyframe, state in enumerate(states):
        if not isinstance(state, dict):
            raise ValueError(f"MAP state {keyframe} must be an object")
        east = _finite_number(state.get("east_m"), f"state {keyframe} east")
        north = _finite_number(state.get("north_m"), f"state {keyframe} north")
        heading = _finite_number(
            state.get("heading_world_cw_deg"), f"state {keyframe} heading",
            low=0.0, high=360.0)
        if heading == 360.0:
            raise ValueError(f"state {keyframe} heading must be in [0, 360)")
        out.append({"e": east, "n": north, "h": heading})
    return out


def _validate_result(result: dict, data, states: list[dict]):
    if result.get("schema") != SCHEMA:
        raise ValueError(f"result schema must be {SCHEMA!r}")
    if not artifact.records_same_artifact(
            result.get("localization_inputs"), data.artifact_ref):
        raise ValueError("result is bound to different localization inputs")
    n = data.n_keyframes
    mass = (result.get("mass_by_keyframe") or {}).get("500")
    errors = result.get("map_error_m_by_keyframe")
    if not isinstance(mass, list) or len(mass) != n:
        raise ValueError("mass@500 series length disagrees with the export")
    if not isinstance(errors, list) or len(errors) != n:
        raise ValueError("MAP-error series length disagrees with the export")
    for keyframe, (value, error, state, truth) in enumerate(zip(
            mass, errors, states, data.truth, strict=True)):
        # Float32 reductions can miss the probability simplex by one ulp.
        _finite_number(
            value, f"mass@500[{keyframe}]", low=-1e-6, high=1.0 + 1e-6)
        error = _finite_number(error, f"map_error[{keyframe}]", low=0.0)
        implied = math.hypot(state["e"] - truth.east_m,
                             state["n"] - truth.north_m)
        if not math.isclose(error, implied, abs_tol=1e-3):
            raise ValueError(
                f"MAP state {keyframe} does not produce its recorded error; "
                "the viewer requires the position-marginal MAP trace")
    return [min(1.0, max(0.0, float(value))) for value in mass], \
        [float(value) for value in errors]


def _final_modes(result: dict, data) -> list[dict]:
    record = result.get("filtered_final_top_modes")
    if not isinstance(record, dict) or record.get("reference_keyframe_idx") \
            != data.n_keyframes - 1 or not isinstance(record.get("modes"), list):
        raise ValueError("result has no valid final grid modes")
    modes = []
    for index, mode in enumerate(record["modes"]):
        if not isinstance(mode, dict):
            raise ValueError(f"final mode {index} must be an object")
        modes.append({
            "e": _finite_number(mode.get("east_m"), f"mode {index} east"),
            "n": _finite_number(mode.get("north_m"), f"mode {index} north"),
            "h": _finite_number(mode.get("heading_world_cw_deg"),
                                f"mode {index} heading", low=0.0, high=360.0),
            "p": _finite_number(mode.get("source_probability"),
                                f"mode {index} probability", low=0.0),
        })
        if modes[-1]["h"] >= 360.0:
            raise ValueError(f"mode {index} heading must be in [0, 360)")
    return modes


def _tracklet_label(tracklet_id: str) -> str:
    match = _TRACKLET_RE.search(tracklet_id)
    if match is None:
        raise ValueError(f"cannot resolve local track id from {tracklet_id!r}")
    return match.group(1)


def build_payload(result_path: Path, output: Path) -> dict:
    result = _load_json(result_path)
    if result.get("schema") != SCHEMA:
        raise ValueError(f"result schema must be {SCHEMA!r}")
    config = result.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("input_dir"), str):
        raise ValueError("result has no recorded input_dir")
    data = export_ingest.load(Path(config["input_dir"]))
    states = _state_trace(result, data)
    mass, errors = _validate_result(result, data, states)
    modes = _final_modes(result, data)
    matching, _, tracks, tracks_manifest, audits, audits_manifest, catalog = \
        _lineage(data)

    availability = config.get("availability")
    if availability not in ("natural", "eager", "none"):
        raise ValueError("result availability is invalid")
    release_by_tracklet = {}
    if availability == "natural":
        sidecar = config.get("release_schedule")
        if not isinstance(sidecar, str) or not sidecar:
            raise ValueError("natural result has no release schedule")
        releases = release_schedule.load_sidecar(Path(sidecar), data)
        release_by_tracklet = {
            item.tracklet_id: item.release_keyframe_idx for item in releases}

    matcher_page = _matching_review(
        output, matching, tracks, audits, catalog)
    audit_page = Path(audits.path) / "preview" / "index.html"
    if "preview/index.html" not in audits_manifest.declared_outputs:
        audit_page = None
    landmark_by_id = {
        item.landmark_id: (item, float(east), float(north))
        for item, east, north in zip(
            data.landmarks, data.catalog.east_m, data.catalog.north_m,
            strict=True)}

    measurements = {}
    for item in data.measurements:
        measurements.setdefault(item.tracklet_id, []).append(item)
    tracks_payload = []
    referenced = set()
    for tracklet_id in sorted(
            measurements, key=lambda value: int(_tracklet_label(value)[1:])):
        local_id = _tracklet_label(tracklet_id)
        epochs = sorted(
            measurements[tracklet_id], key=lambda item: item.anchor_keyframe_idx)
        table = data.tables[tracklet_id]
        endorsed = forensics.endorsed_entries(table)
        candidates = []
        for landmark_id, log_lr in sorted(
                endorsed.items(), key=lambda item: (-item[1], item[0])):
            if landmark_id not in landmark_by_id:
                raise ValueError(f"endorsed landmark is absent: {landmark_id}")
            candidates.append({"id": landmark_id, "lr": float(log_lr)})
            referenced.add(landmark_id)
        track_page = Path(tracks.path) / f"track_full_{local_id}.html"
        if track_page.name not in tracks_manifest.declared_outputs:
            track_page = None
        tracks_payload.append({
            "id": tracklet_id,
            "label": local_id,
            "release": release_by_tracklet.get(tracklet_id),
            "anchors": [{
                "kf": item.anchor_keyframe_idx,
                "bearing": float(item.bearing_forward_cw_deg),
                "sigma": math.degrees(1.0 / math.sqrt(item.kappa)),
                "range": item.range_max_m,
            } for item in epochs],
            "candidates": candidates,
            "floor": forensics.clipped_log_lr(table, table.default_log_lr),
            "trackHref": (_portable_href(output, track_page)
                          if track_page is not None else None),
            "matcherHref": (_portable_href(output, matcher_page, local_id)
                            if matcher_page is not None else None),
            "auditHref": (_portable_href(output, audit_page, local_id)
                          if audit_page is not None else None),
        })

    landmarks = []
    for landmark_id in sorted(referenced):
        item, east, north = landmark_by_id[landmark_id]
        osm = re.fullmatch(r"osm:(node|way|relation):([0-9]+)", landmark_id)
        landmarks.append({
            "id": landmark_id, "type": item.type_key, "e": east, "n": north,
            "osm": (f"https://www.openstreetmap.org/{osm.group(1)}/"
                    f"{osm.group(2)}" if osm else None),
        })

    grid = result.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("result grid must be an object")
    box = grid.get("box")
    if (not isinstance(box, list) or len(box) != 4
            or any(isinstance(value, bool) for value in box)
            or not all(isinstance(value, (int, float))
                       and math.isfinite(value) for value in box)
            or box[0] >= box[1] or box[2] >= box[3]):
        raise ValueError("result grid box is invalid")
    cell_m = _finite_number(grid.get("cell_m"), "grid cell_m", low=0.0)
    n_heading = grid.get("n_heading")
    if cell_m == 0.0 or isinstance(n_heading, bool) \
            or not isinstance(n_heading, int) or n_heading <= 0:
        raise ValueError("result grid dimensions are invalid")
    return {
        "scenario": data.meta.scenario_name,
        "dataset": data.meta.dataset,
        "availability": availability,
        "tail": config.get("tail"),
        "nKeyframes": data.n_keyframes,
        "grid": {"box": [float(value) for value in box],
                 "cellM": cell_m, "nHeading": n_heading},
        "summary": result.get("summary") or {},
        "truth": [{"e": pose.east_m, "n": pose.north_m}
                  for pose in data.truth],
        "states": states,
        "mass500": mass,
        "errors": errors,
        "finalModes": modes,
        "landmarks": landmarks,
        "tracks": tracks_payload,
    }


def _inline_json(value: dict) -> str:
    return (json.dumps(value, separators=(",", ":"), allow_nan=False)
            .replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026").replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029"))


_SCRIPT = r"""
const D=window.__GRID__,N=D.nKeyframes,$=id=>document.getElementById(id);
let t=0,chosen=null,fit="track",timer=null;
const esc=s=>String(s??"").replace(/[&<>\"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const link=(href,label)=>href?`<a href="${esc(href)}" target="_blank" rel="noopener">${esc(label)}</a>`:"";
const lmById=new Map(D.landmarks.map(x=>[x.id,x]));
const trkById=new Map(D.tracks.map(x=>[x.id,x]));
const slider=$("slider");slider.max=N-1;
function bounds(){
  if(fit==="full"){const b=D.grid.box;return {x0:b[0],x1:b[1],y0:b[2],y1:b[3]};}
  const pts=D.truth.concat(D.states);let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
  pts.forEach(p=>{x0=Math.min(x0,p.e);x1=Math.max(x1,p.e);y0=Math.min(y0,p.n);y1=Math.max(y1,p.n)});
  const pad=Math.max(500,.08*Math.max(x1-x0,y1-y0));return{x0:x0-pad,x1:x1+pad,y0:y0-pad,y1:y1+pad};
}
function project(){const b=bounds(),w=900,h=620,s=Math.min(w/(b.x1-b.x0),h/(b.y1-b.y0));
 const ox=(w-s*(b.x1-b.x0))/2,oy=(h-s*(b.y1-b.y0))/2;
 return {x:e=>ox+(e-b.x0)*s,y:n=>h-oy-(n-b.y0)*s,scale:s};}
const path=(pts,p)=>pts.map((v,i)=>(i?"L":"M")+p.x(v.e).toFixed(1)+","+p.y(v.n).toFixed(1)).join(" ");
function status(track){const first=track.anchors[0].kf;if(first>t)return null;
 if(D.availability==="none")return "disabled";
 if(D.availability==="eager")return track.anchors.some(a=>a.kf===t)?"new":"available";
 return track.release===t?"new":track.release<t?"available":"pending";}
function eventEpochs(track){if(D.availability==="natural")return track.release===t?track.anchors:[];
 if(D.availability==="eager")return track.anchors.filter(a=>a.kf===t);return [];}
function drawMap(){const p=project(),selected=chosen&&trkById.get(chosen),selectedIds=new Set((selected?.candidates||[]).map(c=>c.id));
 const shownIds=new Set();D.tracks.forEach(track=>{const st=status(track);if(st==="new"||st==="available"||track.id===chosen)track.candidates.forEach(c=>shownIds.add(c.id));});
 let out=`<path class="truth" stroke-width="2" d="${path(D.truth,p)}"/><path class="estimate" stroke-width="3" d="${path(D.states.slice(0,t+1),p)}"/>`;
 D.landmarks.filter(l=>shownIds.has(l.id)).forEach(l=>{const cls=selectedIds.has(l.id)?"selected":"candidate",title=esc(l.id+" · "+l.type);
  const dot=`<circle class="${cls}" cx="${p.x(l.e)}" cy="${p.y(l.n)}" r="${selectedIds.has(l.id)?4:2}"><title>${title}</title></circle>`;
  out+=l.osm?`<a href="${esc(l.osm)}" target="_blank" rel="noopener">${dot}</a>`:dot;});
 if(t===N-1){const max=Math.max(...D.finalModes.map(m=>m.p),1e-30);D.finalModes.forEach(m=>{const r=2+5*Math.sqrt(m.p/max);
  out+=`<circle class="mode" cx="${p.x(m.e)}" cy="${p.y(m.n)}" r="${r}"><title>final grid mode · p(cell) ${m.p.toPrecision(3)} · heading ${m.h.toFixed(0)}°</title></circle>`;});}
 D.tracks.forEach(track=>eventEpochs(track).forEach(a=>{const origin=D.states[a.kf];
  out+=`<circle class="anchor" cx="${p.x(origin.e)}" cy="${p.y(origin.n)}" r="7"><title>${esc(track.label)} original anchor kf ${a.kf}; bearing ${a.bearing.toFixed(1)}°</title></circle>`;}));
 const s=D.states[t],ang=s.h*Math.PI/180,len=45/p.scale;out+=`<circle cx="${p.x(s.e)}" cy="${p.y(s.n)}" r="5" fill="#df5ba2"/><line class="state" x1="${p.x(s.e)}" y1="${p.y(s.n)}" x2="${p.x(s.e+len*Math.sin(ang))}" y2="${p.y(s.n+len*Math.cos(ang))}"/>`;
 $("map").innerHTML=out;}
function drawMetric(){const w=900,h=150,x=i=>10+i*(w-20)/Math.max(1,N-1),massY=v=>65-55*v;
 const logs=D.errors.map(v=>Math.log10(Math.max(1,v))),lo=Math.min(...logs),hi=Math.max(...logs),errY=v=>140-50*(v-lo)/Math.max(1e-9,hi-lo);
 const line=(values,y)=>values.map((v,i)=>(i?"L":"M")+x(i).toFixed(1)+","+y(v).toFixed(1)).join(" ");
 $("metric").innerHTML=`<text class="metric-label" x="10" y="10">mass@500 (0–1)</text><path d="${line(D.mass500,massY)}" fill="none" stroke="#74c69d" stroke-width="2"/><text class="metric-label" x="10" y="88">MAP error (log scale)</text><path d="${line(logs,errY)}" fill="none" stroke="#ed9b61" stroke-width="2"/><line x1="${x(t)}" x2="${x(t)}" y1="0" y2="150" stroke="#fff" opacity=".7"/>`;}
function chooseVisible(rows){if(chosen&&rows.some(x=>x.track.id===chosen))return;chosen=(rows.find(x=>x.st==="new")||rows.find(x=>x.st==="available")||rows[0]||{}).track?.id||null;}
function drawTracks(){const rows=D.tracks.map(track=>({track,st:status(track)})).filter(x=>x.st);chooseVisible(rows);
 const rank={new:0,pending:1,available:2,disabled:3};rows.sort((a,b)=>rank[a.st]-rank[b.st]||a.track.label.localeCompare(b.track.label,undefined,{numeric:true}));
 $("tracks").innerHTML=rows.length?`<table><thead><tr><th>state</th><th>track</th><th>release</th><th>original anchors</th><th>matches</th></tr></thead><tbody>${rows.map(({track,st})=>`<tr class="pick ${track.id===chosen?'on':''}" data-id="${esc(track.id)}"><td class="${st}">${st}</td><td>${esc(track.label)}</td><td>${track.release??'—'}</td><td>${track.anchors.filter(a=>a.kf<=t).map(a=>a.kf).join(', ')}</td><td>${track.candidates.length}</td></tr>`).join('')}</tbody></table>`:`<div class="note">No observed tracks yet.</div>`;
 document.querySelectorAll("tr.pick").forEach(row=>row.onclick=()=>{chosen=row.dataset.id;drawTracks();drawDetail();drawMap();});}
function drawDetail(){const track=chosen&&trkById.get(chosen);if(!track){$("detail").innerHTML='<div class="note">Select a track.</div>';return;}
 const links=[link(track.trackHref,"track evidence"),link(track.matcherHref,"matcher review"),link(track.auditHref,"audit")].filter(Boolean).join(" ");
 const epochs=track.anchors.map(a=>`kf ${a.kf}: ${a.bearing.toFixed(1)}° ± ${a.sigma.toFixed(1)}°${a.range?' · range ≤ '+a.range.toFixed(0)+' m':''}`).join('<br>');
 const candidates=track.candidates.map(c=>{const l=lmById.get(c.id),name=l?.osm?link(l.osm,c.id):esc(c.id);return `<tr><td>${name}</td><td>${esc(l?.type||'')}</td><td>${c.lr.toFixed(3)}</td></tr>`}).join('');
 $("detail").innerHTML=`<h2>${esc(track.label)}</h2><div class="links">${links}</div><p class="note">${epochs}</p><div class="candidate-list"><table><thead><tr><th>endorsed landmark</th><th>type</th><th>clipped log-LR</th></tr></thead><tbody>${candidates||'<tr><td colspan="3">No endorsed candidates</td></tr>'}</tbody></table></div><p class="note">Unlisted/default floor: ${track.floor.toFixed(3)}.</p>`;}
function render(){t=Number(slider.value);$("kf").textContent=t;$("mass").textContent=D.mass500[t].toFixed(4);$("error").textContent=D.errors[t].toFixed(0)+" m";drawMetric();drawTracks();drawDetail();drawMap();}
slider.oninput=render;$("play").onclick=()=>{if(timer){clearInterval(timer);timer=null;$("play").textContent="▶ play";return;}$("play").textContent="■ stop";timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%N;render();},250);};
$("fitTrack").onclick=()=>{fit="track";$("fitTrack").className="on";$("fitFull").className="";drawMap();};$("fitFull").onclick=()=>{fit="full";$("fitFull").className="on";$("fitTrack").className="";drawMap();};render();
"""


def render_html(payload: dict) -> str:
    availability = html.escape(payload["availability"])
    tail_note = (
        "Unendorsed catalog rows contribute only aggregate uniform angular "
        "tail mass." if payload["tail"] == "uniform" else
        "Unendorsed catalog rows receive the compatibility-table default and "
        "are omitted from this diagnostic map.")
    body = f"""<div class="wrap"><header><div class="eyebrow">Exact grid
localization &middot; causal run viewer</div><h1>{html.escape(payload['scenario'])}</h1>
<div class="meta">{html.escape(payload['dataset'])} &middot; {availability}
availability &middot; {payload['nKeyframes']} keyframes &middot;
{payload['grid']['cellM']:g} m cells &middot; {payload['grid']['nHeading']} headings</div>
<p class="note"><b>Semantics:</b> magenta is the online position-marginal MAP
estimate emitted at each step, not a smoothed or Viterbi trajectory. This exact
grid method has no particles. Gold rings on a release frame mark the archived
online estimates at the observations' original anchors; they are not replayed
historical states or bearing-ray geometry. {html.escape(tail_note)}</p></header>
<div class="panel"><div class="controls"><button id="play">&#9654; play</button>
<input id="slider" type="range" min="0" value="0"><span>kf <b id="kf">0</b>
&middot; mass@500 <b id="mass"></b> &middot; MAP error <b id="error"></b></span></div>
<svg id="metric" viewBox="0 0 900 150"></svg></div>
<div class="grid"><section class="panel"><h2>Map</h2><div class="controls">
<button id="fitTrack" class="on">fit trajectory</button><button id="fitFull">full prior</button></div>
<svg id="map" viewBox="0 0 900 620"></svg><div class="legend">
<span><i class="sw" style="background:#9ba4ae"></i>truth (evaluation only)</span>
<span><i class="sw" style="background:#df5ba2"></i>online MAP</span>
<span><i class="sw" style="background:#63a1c9"></i>endorsed candidates</span>
<span><i class="sw" style="background:#b680d7"></i>final modes (final kf)</span></div></section>
<section class="panel"><h2>Selected track</h2><div id="detail"></div></section></div>
<section class="panel"><h2>Tracks observed by this keyframe</h2><p class="note">
Natural: <span class="new">new</span> arrives atomically now,
<span class="available">available</span> is already in the current belief, and
<span class="pending">pending</span> has been observed but is withheld until
closure. Anchors remain the original observation keyframes.</p><div id="tracks" class="scroll"></div></section>
{page.provenance_footer(GENERATOR, css_class='prov', style='margin-top:24px')}</div>
<script>window.__GRID__={_inline_json(payload)};</script><script>{_SCRIPT}</script>"""
    return page.document(
        f"{payload['scenario']} — grid viewer", body,
        style=_STYLE + _EXTRA_STYLE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="HTML path; defaults to <result-stem>.viewer/viewer.html")
    args = parser.parse_args()
    output = args.output or (
        args.result.parent / f"{args.result.stem}.viewer" / "viewer.html")
    try:
        payload = build_payload(args.result, output)
        if output.is_symlink():
            raise ValueError(f"refusing symlink output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        artifact.atomic_write_file(output, render_html(payload).encode("utf-8"))
    except (artifact.ArtifactError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
