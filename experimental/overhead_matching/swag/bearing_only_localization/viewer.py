"""Interactive run viewer: the §7.4 views as one self-contained HTML page.

Builds the three views the design doc says to build first, because they need
only Tier 0/1 plus checkpoints (§10.6):

1. **Run overview strip** — health scalars as sparklines, mode lifespans as
   bands whose thickness is their weight, proposal and mode events as glyphs.
   The entry point: open a run, look at the strip, click.
2. **Map view** — landmarks, truth, MAP trail, the particle cloud at the
   selected keyframe coloured BY MODE, per-mode ellipses, and bearing wedges
   drawn from the selected mode's centroid with correspondence lines whose
   opacity is that mode's association posterior.
3. **Mode ledger** — every mode with its birth keyframe, provenance ("spawned
   by proposal #1 from tracklets {a,b,c} ↔ landmarks {…}"), weight
   trajectory, and death.

Deliberately NOT here: the attribution waterfall and the what-if console.
Both need the Tier 3 replay service, and §7.5 is emphatic that the replay
path must be the production filter in replay mode rather than a
reimplementation — a viewer that recomputed likelihoods in JavaScript would
make viewer-vs-filter divergence possible, which is exactly what that
requirement exists to prevent.

The page is one file with its data inlined, so a run directory stays
portable and a viewer session needs no server.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    filter as pf,
    geodesy,
    run_log,
)

# Enough particles to read the shape of a cloud, few enough to inline.
MAX_PARTICLES_PER_FRAME = 900
# Mode colours are drawn from chart symbology and kept clear of the
# starboard-green / port-red semantic pair, so "which mode" never reads as
# "good or bad".
MODE_COLORS = ["#C21E76", "#2E7FA8", "#B07A16", "#7A4FBF",
               "#0F8C86", "#A8447E", "#5C6BC0", "#77702A"]


def _round(values, decimals=1):
    return [round(float(v), decimals) for v in values]


def _landmark_positions(data):
    frame = geodesy.RegionFrame(data.manifest.anchor_lat_deg,
                               data.manifest.anchor_lon_deg)
    east, north = frame.enu_from_latlon(
        np.array([lm.lat_deg for lm in data.manifest.landmarks]),
        np.array([lm.lon_deg for lm in data.manifest.landmarks]))
    return east, north


def _referenced_landmark_ids(data) -> set:
    """Landmarks the run actually talks about: endorsed table entries,
    proposal-hypothesis identities, and anything that ever carried reported
    association mass. Everything else renders as catalog backdrop — a
    whole-map catalog (13k rows on the harbor runs) drawn as labelled
    glyphs is unreadable and unresponsive."""
    ids = set()
    for table in data.tables.values():
        default = min(max(table.default_log_lr, table.clip_lo),
                      table.clip_hi)
        for entry in table.entries:
            clipped = min(max(entry.log_lr, table.clip_lo), table.clip_hi)
            if clipped > default + 1e-12:
                ids.add(entry.landmark_id)
    for event in data.proposal_events:
        for landmark_ids in event.hypothesis_landmark_ids:
            ids.update(landmark_ids)
    for record in data.health:
        for assoc in record.associations:
            for landmark_id, value in assoc.responsibilities.items():
                if value > 1e-3:
                    ids.add(landmark_id)
    return ids


def build_payload(data, max_particles=MAX_PARTICLES_PER_FRAME) -> dict:
    """Everything the page needs, shaped for the browser."""
    east, north = _landmark_positions(data)
    truth_by_kf = {t.keyframe_idx: t for t in data.truth}
    referenced = _referenced_landmark_ids(data)

    checkpoints = {}
    rng = np.random.default_rng(0)
    for keyframe_idx, arrays in sorted(data.checkpoints.items()):
        count = arrays["east_m"].shape[0]
        if count > max_particles:
            index = rng.choice(count, size=max_particles, replace=False)
        else:
            index = np.arange(count)
        checkpoints[str(keyframe_idx)] = {
            "e": _round(arrays["east_m"][index], 0),
            "n": _round(arrays["north_m"][index], 0),
            "m": [int(v) for v in arrays.get(
                "mode_id", np.full(count, -1))[index]],
        }

    health = []
    for record in data.health:
        truth = truth_by_kf.get(record.keyframe_idx)
        entry = {
            "kf": record.keyframe_idx,
            "ess": round(record.ess, 1),
            "sigma": round(record.position_std_m, 1),
            "headingSigma": round(record.heading_std_deg, 2),
            "meanE": round(record.mean_east_m, 1),
            "meanN": round(record.mean_north_m, 1),
            "mapE": round(record.map_east_m, 1),
            "mapN": round(record.map_north_m, 1),
            "entropy": round(record.mode_entropy_nats, 3),
            "proposalShare": round(record.proposal_weight_share, 3),
            "nMeas": record.n_measurements,
            "modes": [{
                "id": mode.mode_id,
                "w": round(mode.weight, 4),
                "e": round(mode.mean_east_m, 1),
                "n": round(mode.mean_north_m, 1),
                "h": round(mode.mean_heading_deg, 1),
                "std": round(mode.position_std_m, 1),
                "born": mode.birth_keyframe_idx,
                "prov": {k: str(v) for k, v in mode.provenance.items()},
            } for mode in record.modes],
            "assoc": [{
                "mode": a.mode_id,
                "trk": a.tracklet_id,
                "null": round(a.null_share, 4),
                "resp": {k: round(v, 4) for k, v in
                         sorted(a.responsibilities.items(),
                                key=lambda kv: -kv[1])[:4] if v > 1e-4},
            } for a in record.associations],
        }
        if truth is not None:
            entry["truthE"] = round(truth.east_m, 1)
            entry["truthN"] = round(truth.north_m, 1)
            entry["err"] = round(math.hypot(record.mean_east_m - truth.east_m,
                                            record.mean_north_m - truth.north_m), 1)
            entry["mapErr"] = round(
                math.hypot(record.map_east_m - truth.east_m,
                           record.map_north_m - truth.north_m), 1)
            entry["headingErr"] = round(abs(math.degrees(float(geodesy.wrap_rad(
                math.radians(record.mean_heading_deg)
                - math.radians(truth.heading_deg))))), 2)
        health.append(entry)

    measurements = {}
    for meas in data.measurements:
        measurements.setdefault(str(meas.anchor_keyframe_idx), []).append({
            "trk": meas.tracklet_id,
            "bearing": round(meas.bearing_body_deg, 2),
            "sigma": round(math.degrees(1.0 / math.sqrt(max(meas.kappa, 1e-9))),
                           2),
        })

    return {
        "scenario": data.manifest.scenario_name,
        "nKeyframes": data.manifest.n_keyframes,
        "nParticles": data.manifest.filter_config.n_particles,
        "seed": data.manifest.filter_config.seed,
        "matcher": data.manifest.matcher_version,
        "historyHash": data.manifest.particle_history_sha256[:12],
        "landmarks": [{"id": lm.landmark_id, "type": lm.type_key,
                       "e": round(float(e), 1), "n": round(float(n), 1)}
                      for lm, e, n in zip(data.manifest.landmarks, east,
                                          north)
                      if lm.landmark_id in referenced],
        "backdrop": [[int(round(float(e))), int(round(float(n)))]
                     for lm, e, n in zip(data.manifest.landmarks, east,
                                         north)
                     if lm.landmark_id not in referenced],
        "truth": [[round(t.east_m, 1), round(t.north_m, 1)] for t in data.truth],
        "health": health,
        "checkpoints": checkpoints,
        "measurements": measurements,
        "proposalEvents": [{
            "id": e.event_id, "kf": e.keyframe_idx, "trigger": e.trigger,
            "nHyp": e.n_hypotheses, "nInj": e.n_injected,
            "skipped": e.n_combinations_skipped,
            "hyp": [{"trk": t, "lm": l} for t, l in
                    zip(e.hypothesis_tracklet_ids[:8],
                        e.hypothesis_landmark_ids[:8])],
        } for e in data.proposal_events],
        "modeEvents": [{"kf": e.keyframe_idx, "kind": e.kind,
                        "id": e.mode_id,
                        "parents": e.parent_mode_ids,
                        "detail": {k: str(v) for k, v in e.detail.items()}}
                       for e in data.mode_events],
        "colors": MODE_COLORS,
    }


_STYLE = """
/* Palette from NOAA chart symbology: buff chart paper, blue-biased ink,
   magenta for navaids and lights, starboard-green / port-red for status. */
:root{
  --paper:#FBF8F3; --panel:#FFFFFF; --sunk:#F2EEE6;
  --rule:#DCD6CA; --rule-soft:#EAE5DA;
  --ink:#16202B; --ink-soft:#5B6875; --ink-faint:#8B96A3;
  --accent:#C21E76; --water:#2E7FA8;
  --starboard:#1B7F52; --port:#B3372C; --caution:#B07A16;
  --truth:#7C8894; --grid:rgba(22,32,43,.08);
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --paper:#0E1620; --panel:#16202B; --sunk:#111A24;
  --rule:#2A3644; --rule-soft:#212C39;
  --ink:#E4EAF0; --ink-soft:#9DAAB8; --ink-faint:#6F7C8B;
  --accent:#E85BA6; --water:#5AA9C9;
  --starboard:#3FB584; --port:#E0705F; --caution:#D8A63C;
  --truth:#7A8796; --grid:rgba(228,234,240,.09);
}}
:root[data-theme="dark"]{
  --paper:#0E1620; --panel:#16202B; --sunk:#111A24;
  --rule:#2A3644; --rule-soft:#212C39;
  --ink:#E4EAF0; --ink-soft:#9DAAB8; --ink-faint:#6F7C8B;
  --accent:#E85BA6; --water:#5AA9C9;
  --starboard:#3FB584; --port:#E0705F; --caution:#D8A63C;
  --truth:#7A8796; --grid:rgba(228,234,240,.09);
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased}
.mono,code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
header{padding:20px 24px 0;max-width:1500px;margin:0 auto}
.eyebrow{font-size:11px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--accent);font-weight:600}
h1{margin:6px 0 4px;font-size:21px;font-weight:640;letter-spacing:-.01em;
  text-wrap:balance}
.meta{color:var(--ink-soft);font-size:12.5px}
.meta b{color:var(--ink);font-weight:600}
/* Summary before detail: the run at a glance. */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(122px,1fr));
  gap:9px;max-width:1500px;margin:16px auto 0;padding:0 24px}
.tile{background:var(--panel);border:1px solid var(--rule);border-radius:7px;
  padding:9px 11px}
.tile .k{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink-faint);font-weight:600}
.tile .v{font-size:19px;font-weight:600;margin-top:3px;letter-spacing:-.01em}
.tile .u{font-size:11.5px;color:var(--ink-faint);font-weight:500}
.pill{display:inline-flex;align-items:center;gap:5px;font-size:11px;
  font-weight:650;padding:2px 8px;border-radius:999px;letter-spacing:.02em}
.pill::before{content:"";width:6px;height:6px;border-radius:50%;
  background:currentColor}
.pill.ok{color:var(--starboard);background:color-mix(in srgb,var(--starboard) 14%,transparent)}
.pill.warn{color:var(--caution);background:color-mix(in srgb,var(--caution) 16%,transparent)}
.pill.bad{color:var(--port);background:color-mix(in srgb,var(--port) 14%,transparent)}
main{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(300px,1fr);
  gap:14px;max-width:1500px;margin:0 auto;padding:14px 24px 36px}
@media (max-width:960px){main{grid-template-columns:minmax(0,1fr)}}
section{background:var(--panel);border:1px solid var(--rule);border-radius:9px;
  padding:14px;min-width:0}
.strip{grid-column:1/-1}
h2{margin:0 0 10px;font-size:11px;font-weight:650;letter-spacing:.11em;
  text-transform:uppercase;color:var(--ink-faint)}
h2+h2{margin-top:18px}
svg{display:block;width:100%;overflow:visible}
.axis{fill:var(--ink-faint);font-size:9.5px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.cursor{stroke:var(--accent);stroke-width:1.5}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;padding:5px 7px;border-bottom:1px solid var(--rule-soft)}
th{color:var(--ink-faint);font-weight:650;font-size:10.5px;
  letter-spacing:.07em;text-transform:uppercase}
td.num{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums}
tr.mode{cursor:pointer}
tr.mode:hover td{background:var(--sunk)}
tr.mode.sel td{background:color-mix(in srgb,var(--accent) 12%,transparent)}
.swatch{display:inline-block;width:9px;height:9px;border-radius:2px;
  margin-right:6px;vertical-align:baseline}
.prov{color:var(--ink-soft);font-size:11.5px}
.controls{display:flex;gap:11px;align-items:center;margin-bottom:11px;
  flex-wrap:wrap}
input[type=range]{flex:1;min-width:190px;accent-color:var(--accent);height:20px}
button{background:var(--panel);color:var(--ink);border:1px solid var(--rule);
  border-radius:6px;padding:4px 11px;cursor:pointer;font:inherit;
  font-size:12.5px;font-weight:550}
button:hover{border-color:var(--accent);color:var(--accent)}
button:focus-visible,input:focus-visible{outline:2px solid var(--accent);
  outline-offset:2px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:12.5px}
.kv dt{color:var(--ink-soft)}
.kv dd{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums}
.legend{color:var(--ink-soft);font-size:11.5px;margin-top:9px;
  line-height:1.55;max-width:74ch}
.scroll{overflow-x:auto}
@media (prefers-reduced-motion:reduce){*{animation:none!important;
  transition:none!important}}
"""

_SCRIPT = r"""
const D = window.__RUN__;
const H = D.health, KF = D.nKeyframes - 1;
let t = 0, selMode = null;
const color = id => id == null || id < 0 ? "#8b93a3"
  : D.colors[id % D.colors.length];
const $ = id => document.getElementById(id);
const ckKeys = Object.keys(D.checkpoints).map(Number).sort((a,b)=>a-b);
const nearestCk = kf => ckKeys.reduce((best,k) =>
  Math.abs(k-kf) < Math.abs(best-kf) ? k : best, ckKeys[0]);

// ---------- bounds ----------
let X0=1e9,X1=-1e9,Y0=1e9,Y1=-1e9;
const grow=(e,n)=>{X0=Math.min(X0,e);X1=Math.max(X1,e);Y0=Math.min(Y0,n);Y1=Math.max(Y1,n);};
D.landmarks.forEach(l=>grow(l.e,l.n));
(D.backdrop||[]).forEach(p=>grow(p[0],p[1]));
D.truth.forEach(p=>grow(p[0],p[1]));
H.forEach(h=>grow(h.mapE,h.mapN));
const pad=(X1-X0+Y1-Y0)*0.06+150; X0-=pad;X1+=pad;Y0-=pad;Y1+=pad;
const span=Math.max(X1-X0,Y1-Y0);
const MW=760, MH=560;
const px=e=>((e-X0)/span)*MW, py=n=>MH-((n-Y0)/span)*MH;
// Unreferenced catalog rows: one static path of 1-px dots (a single DOM
// node), rendered behind everything. Built once — scrubbing re-renders
// the rest of the map every frame.
const BACKDROP=(D.backdrop&&D.backdrop.length)
  ? `<path d="${D.backdrop.map(p=>"M"+px(p[0]).toFixed(1)+" "
      +py(p[1]).toFixed(1)+"h.1").join("")}" stroke="#59606e"
      stroke-width="1.6" stroke-linecap="round" fill="none" opacity=".4"/>`
  : "";

// ---------- overview strip ----------
const SW=980, ROW=30;
function series(key){return H.map(h=>h[key]);}
function sparkline(y0,vals,label,col,log){
  if(vals.every(v=>v===undefined))return "";
  const fin=vals.filter(v=>v!==undefined&&isFinite(v));
  if(!fin.length)return "";
  let lo=Math.min(...fin),hi=Math.max(...fin);
  if(log){lo=Math.max(lo,1e-3);hi=Math.max(hi,lo*1.01);}
  if(hi-lo<1e-9)hi=lo+1;
  const base=y0+ROW-3;
  const sc=v=>{const a=log?Math.log(Math.max(v,lo)):v,
    b=log?Math.log(lo):lo, c=log?Math.log(hi):hi;
    return y0+ROW-3-((a-b)/(c-b))*(ROW-8);};
  let d="",area="",last=null;
  vals.forEach((v,i)=>{if(v===undefined||!isFinite(v))return;
    const X=(i/KF)*SW, Y=sc(v);
    d+=(d?"L":"M")+X.toFixed(1)+" "+Y.toFixed(1);
    area+=(area?"L":"M"+X.toFixed(1)+" "+base+"L")+X.toFixed(1)+" "+Y.toFixed(1);
    last=[X,Y];});
  const uid="g"+label.replace(/\W/g,"");
  return `<line x1="0" y1="${base}" x2="${SW}" y2="${base}"
    stroke="var(--grid)" stroke-width="1"/>
  <line x1="0" y1="${y0+3}" x2="${SW}" y2="${y0+3}"
    stroke="var(--grid)" stroke-width="1"/>
  <path d="${area}L${last?last[0].toFixed(1):0} ${base}Z" fill="${col}"
    opacity=".10"/>
  <path d="${d}" fill="none" stroke="${col}" stroke-width="1.3"
    stroke-linejoin="round"/>
  ${last?`<circle cx="${last[0].toFixed(1)}" cy="${last[1].toFixed(1)}" r="2.4"
    fill="${col}"/>`:""}
  <text class="axis" x="2" y="${y0+10}">${label}</text>
  <text class="axis" x="${SW-2}" y="${y0+10}" text-anchor="end">${
    hi.toFixed(hi<10?2:0)}</text>
  <text class="axis" x="${SW-2}" y="${base}" text-anchor="end">${
    lo.toFixed(lo<10?2:0)}</text>`;
}
function modeBands(y0){
  // Lifespan per mode: first..last keyframe seen, thickness = mean weight.
  const life={};
  H.forEach(h=>h.modes.forEach(m=>{
    const L=life[m.id]||(life[m.id]={a:h.kf,b:h.kf,w:[],born:m.born});
    L.b=h.kf; L.w.push(m.w);}));
  const ids=Object.keys(life).map(Number).sort((a,b)=>a-b);
  let out="", y=y0;
  ids.forEach(id=>{
    const L=life[id], w=L.w.reduce((a,b)=>a+b,0)/L.w.length;
    const x=(L.a/KF)*SW, x2=(L.b/KF)*SW, th=3+w*11;
    out+=`<rect x="${x.toFixed(1)}" y="${(y+7-th/2).toFixed(1)}"
      width="${Math.max(x2-x,1.5).toFixed(1)}" height="${th.toFixed(1)}"
      fill="${color(id)}" opacity="${selMode===id?1:.72}" rx="2"
      class="band" data-mode="${id}" style="cursor:pointer"/>
      <text class="axis" x="${(x+3).toFixed(1)}" y="${y+5}">m${id}</text>`;
    y+=15;
  });
  return {svg:out, height:Math.max(y-y0, 16)};
}
function drawStrip(){
  let y=4, out="";
  const css=getComputedStyle(document.documentElement);
  const C=n=>css.getPropertyValue(n).trim();
  out+=sparkline(y,series("err"),"pos err (m)",C("--port"),true); y+=ROW;
  out+=sparkline(y,series("sigma"),"reported σ (m)",C("--starboard"),true); y+=ROW;
  out+=sparkline(y,series("ess"),"ESS",C("--water"),true); y+=ROW;
  out+=sparkline(y,series("entropy"),"mode entropy",C("--accent"),false); y+=ROW;
  const bands=modeBands(y); out+=bands.svg; y+=bands.height+6;
  // event glyphs
  D.proposalEvents.forEach(e=>{const X=(e.kf/KF)*SW;
    out+=`<line x1="${X}" y1="0" x2="${X}" y2="${y}" stroke="var(--warn)"
      stroke-width="1" stroke-dasharray="3 2" opacity=".8"/>
      <text class="axis" x="${X+2}" y="${y-2}" fill="var(--warn)">
      ⟐ ${e.trigger}</text>`;});
  D.modeEvents.filter(e=>e.kind!=="death").forEach(e=>{
    const X=(e.kf/KF)*SW;
    out+=`<circle cx="${X}" cy="${y+8}" r="3" fill="${color(e.id)}"
      opacity=".9"><title>${e.kind} m${e.id} @kf${e.kf}</title></circle>`;});
  y+=16;
  const cx=(t/KF)*SW;
  out+=`<line class="cursor" x1="${cx}" y1="0" x2="${cx}" y2="${y}"/>`;
  const svg=$("strip");
  svg.setAttribute("viewBox",`0 0 ${SW} ${y}`);
  svg.innerHTML=out;
  svg.querySelectorAll(".band").forEach(b=>b.onclick=ev=>{
    ev.stopPropagation();
    selMode = selMode===+b.dataset.mode ? null : +b.dataset.mode; render();});
}

// ---------- map ----------
function drawMap(){
  const h=H[t], ck=nearestCk(t), P=D.checkpoints[ck];
  let out=BACKDROP;
  // particles, coloured by mode
  for(let i=0;i<P.e.length;i++){
    const m=P.m[i];
    if(selMode!==null && m!==selMode) continue;
    out+=`<circle cx="${px(P.e[i]).toFixed(1)}" cy="${py(P.n[i]).toFixed(1)}"
      r="1.5" fill="${color(m)}" opacity=".5"/>`;
  }
  // truth + MAP trails
  let dt="",dm="";
  D.truth.forEach((p,i)=>{dt+=(i?"L":"M")+px(p[0]).toFixed(1)+" "+py(p[1]).toFixed(1);});
  H.slice(0,t+1).forEach((r,i)=>{dm+=(i?"L":"M")+px(r.mapE).toFixed(1)+" "+py(r.mapN).toFixed(1);});
  out+=`<path d="${dt}" fill="none" stroke="var(--truth)" stroke-width="1.5"
    stroke-dasharray="4 3"/>`;
  out+=`<path d="${dm}" fill="none" stroke="var(--accent)" stroke-width="1.5"/>`;
  // mode ellipses + heading ticks
  h.modes.forEach(m=>{
    if(selMode!==null && m.id!==selMode) return;
    const r=Math.max(px(X0+m.std)-px(X0),2.5);
    out+=`<circle cx="${px(m.e).toFixed(1)}" cy="${py(m.n).toFixed(1)}"
      r="${r.toFixed(1)}" fill="none" stroke="${color(m.id)}"
      stroke-width="1.4" opacity=".95"/>`;
    const a=m.h*Math.PI/180, L=22;
    out+=`<line x1="${px(m.e).toFixed(1)}" y1="${py(m.n).toFixed(1)}"
      x2="${(px(m.e)+L*Math.sin(a)).toFixed(1)}"
      y2="${(py(m.n)-L*Math.cos(a)).toFixed(1)}"
      stroke="${color(m.id)}" stroke-width="1.4"/>`;
    out+=`<text class="axis" x="${(px(m.e)+6).toFixed(1)}"
      y="${(py(m.n)-6).toFixed(1)}" fill="${color(m.id)}">m${m.id} ${(m.w*100).toFixed(0)}%</text>`;
  });
  // bearing wedges from the selected (or heaviest) mode
  const anchor = h.modes.find(m=>m.id===selMode) || h.modes[0];
  const meas = D.measurements[String(t)]||[];
  if(anchor) meas.forEach(mm=>{
    const world=(anchor.h+mm.bearing)*Math.PI/180, R=span*0.55;
    const wedge=(off,op,w)=>`<line x1="${px(anchor.e).toFixed(1)}"
      y1="${py(anchor.n).toFixed(1)}"
      x2="${(px(anchor.e)+ (R/span)*MW*Math.sin(world+off)).toFixed(1)}"
      y2="${(py(anchor.n)- (R/span)*MH*Math.cos(world+off)).toFixed(1)}"
      stroke="var(--warn)" stroke-width="${w}" opacity="${op}"/>`;
    const s=mm.sigma*Math.PI/180;
    out+=wedge(0,.85,1.3)+wedge(2*s,.35,.8)+wedge(-2*s,.35,.8);
    // correspondence lines: opacity = this mode's association posterior
    const a=h.assoc.find(x=>x.trk===mm.trk && x.mode===(anchor?anchor.id:null))
         || h.assoc.find(x=>x.trk===mm.trk && x.mode===null);
    if(a) Object.entries(a.resp).forEach(([lm,p])=>{
      const L=D.landmarks.find(x=>x.id===lm); if(!L||p<0.02)return;
      out+=`<line x1="${px(anchor.e).toFixed(1)}" y1="${py(anchor.n).toFixed(1)}"
        x2="${px(L.e).toFixed(1)}" y2="${py(L.n).toFixed(1)}"
        stroke="${color(anchor.id)}" stroke-width="${(1+2*p).toFixed(1)}"
        opacity="${(0.15+0.6*p).toFixed(2)}" stroke-dasharray="2 3"/>`;
    });
  });
  // Referenced landmarks on top. Labels only where the filter is
  // currently putting association mass (or everywhere, in small worlds) —
  // a whole-map run references thousands of tie members and labelling
  // them all makes the map unreadable.
  const active={};
  h.assoc.forEach(a=>Object.entries(a.resp).forEach(([lm,p])=>{
    if(p>=0.05) active[lm]=Math.max(active[lm]||0,p);}));
  const many=D.landmarks.length>60;
  D.landmarks.forEach(l=>{
    const hot=active[l.id]!==undefined;
    const r=many?(hot?4:2.2):5;
    out+=`<circle cx="${px(l.e).toFixed(1)}" cy="${py(l.n).toFixed(1)}"
      r="${r}" fill="#f5c542" stroke="#7a5c00"
      stroke-width="${many&&!hot?0.6:1.2}" opacity="${many&&!hot?0.55:1}"/>`;
    if(hot||!many)
      out+=`<text class="axis" x="${(px(l.e)+8).toFixed(1)}"
        y="${(py(l.n)+3).toFixed(1)}">${l.id}</text>`;
  });
  // truth marker at t
  if(h.truthE!==undefined)
    out+=`<circle cx="${px(h.truthE).toFixed(1)}" cy="${py(h.truthN).toFixed(1)}"
      r="4" fill="none" stroke="var(--truth)" stroke-width="2"/>`;
  const svg=$("map");
  svg.setAttribute("viewBox",`0 0 ${MW} ${MH}`);
  svg.innerHTML=out;
  $("mapnote").textContent =
    `particles from checkpoint kf ${ck}${ck!==t?" (nearest to "+t+")":""}`;
}

// ---------- side panels ----------
function drawLedger(){
  const h=H[t];
  let rows="";
  h.modes.forEach(m=>{
    const p=m.prov||{};
    const origin = p.source==="proposal"
      ? `proposal #${p.proposal_event_id} (${p.trigger||"?"})` +
        (p.landmark_ids?`<br><code>${p.landmark_ids}</code>`:"")
      : "motion";
    rows+=`<tr class="mode ${selMode===m.id?"sel":""}" data-mode="${m.id}">
      <td><span class="swatch" style="background:${color(m.id)}"></span>m${m.id}</td>
      <td class="num">${(m.w*100).toFixed(1)}%</td>
      <td class="num">${m.std.toFixed(0)}</td>
      <td class="num">${m.born}</td><td class="prov">${origin}</td></tr>`;
  });
  if(!rows) rows=`<tr><td colspan="5" class="prov">no modes above weight
    threshold</td></tr>`;
  $("ledger").innerHTML=`<table><thead><tr><th>mode</th><th>weight</th>
    <th>σ (m)</th><th>born</th><th>origin</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
  $("ledger").querySelectorAll("tr.mode").forEach(r=>r.onclick=()=>{
    selMode = selMode===+r.dataset.mode ? null : +r.dataset.mode; render();});
}
function drawAssoc(){
  const h=H[t];
  const rows=h.assoc.filter(a=>selMode===null ? a.mode===null
                                              : a.mode===selMode);
  let out="";
  rows.forEach(a=>{
    const parts=Object.entries(a.resp)
      .map(([lm,p])=>`${lm} <b>${(p*100).toFixed(0)}%</b>`).join(" · ");
    out+=`<tr><td><code>${a.trk}</code></td>
      <td class="num">${(a.null*100).toFixed(0)}%</td>
      <td class="prov">${parts||"—"}</td></tr>`;
  });
  if(!out) out=`<tr><td colspan="3" class="prov">no measurement at this
    keyframe</td></tr>`;
  $("assoc").innerHTML=`<table><thead><tr><th>tracklet</th><th>null</th>
    <th>believes it is${selMode!==null?` (mode ${selMode})`:" (all modes)"}
    </th></tr></thead><tbody>${out}</tbody></table>`;
}
function drawTiles(){
  const h=H[t], last=H[H.length-1];
  // Status encodes state in form, not just number: consistent means the
  // reported sigma actually covers the error being made.
  let status="ok", label="tracking";
  if(h.err!==undefined){
    if(h.err>3*Math.max(h.sigma,1)){status="bad";label="overconfident";}
    else if(h.modes.length>1){status="warn";label="multimodal";}
    else if(h.err>300){status="warn";label="searching";}
  } else {status="warn";label="no ground truth";}
  const tile=(k,v,u="")=>`<div class="tile"><div class="k">${k}</div>
    <div class="v mono">${v}<span class="u"> ${u}</span></div></div>`;
  const f=(v,d=0)=>v===undefined?"—":v.toFixed(d);
  $("tiles").innerHTML =
    `<div class="tile"><div class="k">status @ kf ${h.kf}</div>
      <div class="v"><span class="pill ${status}">${label}</span></div></div>`
    + tile("mean error", f(h.err), "m")
    + tile("reported σ", f(h.sigma), "m")
    + tile("MAP error", f(h.mapErr), "m")
    + tile("modes", h.modes.length, `H=${f(h.entropy,2)}`)
    + tile("ESS", f(h.ess), `/ ${D.nParticles}`)
    + tile("final error", f(last.err), "m")
    + tile("proposals", D.proposalEvents.length,
           D.proposalEvents.length ? D.proposalEvents[D.proposalEvents.length-1].trigger : "none");
}
function drawStats(){
  const h=H[t];
  const f=(v,d=1)=>v===undefined?"—":v.toFixed(d);
  $("stats").innerHTML=`
    <dt>keyframe</dt><dd>${h.kf} / ${KF}</dd>
    <dt>heading error</dt><dd>${f(h.headingErr,2)}°</dd>
    <dt>heading σ</dt><dd>${f(h.headingSigma,2)}°</dd>
    <dt>measurements</dt><dd>${h.nMeas}</dd>
    <dt>proposal-descended</dt><dd>${(h.proposalShare*100).toFixed(0)}%</dd>`;
}
function render(){
  $("kf").textContent=t; $("slider").value=t;
  drawTiles(); drawStrip(); drawMap(); drawLedger(); drawAssoc(); drawStats();
}
$("slider").max=KF;
$("slider").oninput=e=>{t=+e.target.value; render();};
$("strip").onclick=e=>{
  const r=$("strip").getBoundingClientRect();
  t=Math.max(0,Math.min(KF,Math.round(((e.clientX-r.left)/r.width)*KF)));
  render();};
$("clear").onclick=()=>{selMode=null; render();};
document.addEventListener("keydown",e=>{
  if(e.key==="ArrowRight"){t=Math.min(KF,t+1);render();}
  if(e.key==="ArrowLeft"){t=Math.max(0,t-1);render();}});
let timer=null;
$("play").onclick=()=>{
  if(timer){clearInterval(timer);timer=null;$("play").textContent="▶ play";return;}
  $("play").textContent="⏸ pause";
  timer=setInterval(()=>{t=(t+1)%(KF+1);render();},90);};
render();
"""


def render_html(payload: dict, body_only: bool = False) -> str:
    events = payload["proposalEvents"]
    event_summary = " · ".join(
        f"#{e['id']} kf{e['kf']} {e['trigger']} → {e['nInj']} particles "
        f"from {e['nHyp']} hypotheses" for e in events) or "none"
    body = f"""<header>
<div class="eyebrow">Bearing-only localization · run viewer</div>
<h1>{payload['scenario'].replace('_', ' ')}</h1>
<div class="meta"><b>{payload['nParticles']:,}</b> particles ·
seed <b>{payload['seed']}</b> · <b>{payload['nKeyframes']}</b> keyframes ·
matcher <b>{payload['matcher']}</b> ·
history <code>{payload['historyHash']}</code></div>
<div class="meta">proposal events: {event_summary}</div>
</header>
<div class="tiles" id="tiles"></div>
<main>
<section class="strip">
<h2>Run overview</h2>
<div class="controls">
<button id="play">▶ play</button>
<input id="slider" type="range" min="0" value="0">
<span>kf <b id="kf">0</b></span>
<button id="clear">show all modes</button>
</div>
<svg id="strip"></svg>
<div class="legend">Click the strip to scrub; click a mode band or ledger row
to isolate that mode. Dashed red verticals are proposal events; dots are mode
births and merges.</div>
</section>
<section>
<h2>Map</h2>
<svg id="map"></svg>
<div class="legend"><span id="mapnote"></span> — dashed grey is ground truth,
blue is the MAP trail, circles are per-mode 1σ with a heading tick, red rays
are bearing wedges (±2σ) from the selected mode, dashed coloured lines are
correspondence with opacity ∝ association posterior.</div>
</section>
<section>
<h2>State</h2>
<dl class="kv" id="stats"></dl>
<h2 style="margin-top:14px">Mode ledger</h2>
<div class="scroll" id="ledger"></div>
<h2 style="margin-top:14px">Association posteriors</h2>
<div class="scroll" id="assoc"></div>
</section>
</main>
<script>window.__RUN__ = {json.dumps(payload, separators=(',', ':'))};</script>
<script>{_SCRIPT}</script>"""
    style = f"<style>{_STYLE}</style>"
    if body_only:
        return style + body
    return (f"<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,"
            f"initial-scale=1\">"
            f"<title>{payload['scenario']} — bearing-only localization</title>"
            f"{style}</head><body>{body}</body></html>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="defaults to <run_dir>/viewer.html")
    parser.add_argument("--body_only", action="store_true",
                        help="emit a fragment for embedding rather than a "
                             "standalone document")
    parser.add_argument("--max_particles", type=int,
                        default=MAX_PARTICLES_PER_FRAME)
    args = parser.parse_args()

    data = run_log.read_run(args.run_dir)
    payload = build_payload(data, args.max_particles)
    output = args.output or (args.run_dir / "viewer.html")
    output.write_text(render_html(payload, args.body_only))
    size_kb = output.stat().st_size / 1024
    print(f"Wrote {output} ({size_kb:.0f} KB): "
          f"{len(payload['health'])} keyframes, "
          f"{len(payload['checkpoints'])} checkpoints, "
          f"{len(payload['proposalEvents'])} proposal events, "
          f"{len(payload['modeEvents'])} mode events")


if __name__ == "__main__":
    main()
