const D = window.__RUN__;
const H = D.health, KF = D.run.nKeyframes - 1, RUN = D.run;
const $ = id => document.getElementById(id);
let t = 0, selMode = null, selTrk = null, tab = "state";
let showGhosts = true, showBase = true, showParticles = true;
// Imagery defaults ON when supplied: if someone went to the
// trouble of fetching a mosaic, they want to see it.
let showSat = true;

const color = id => id == null || id < 0 ? "#8b93a3"
  : D.colors[id % D.colors.length];
const fmt = (v, d = 0) => (v === undefined || v === null || !isFinite(v))
  ? "—" : v.toFixed(d);
const esc = s => String(s == null ? "" : s)
  .replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const ckKeys = Object.keys(D.checkpoints).map(Number).sort((a, b) => a - b);
const nearestCk = kf => ckKeys.reduce((best, k) =>
  Math.abs(k - kf) < Math.abs(best - kf) ? k : best, ckKeys[0]);
const TRK = new Map(D.tracklets.map(x => [x.id, x]));
const LM = new Map(D.landmarks.map(l => [l.id, l]));
const wrap180 = a => ((a % 360) + 540) % 360 - 180;

// ---------- optional localhost server ----------
const LIVE = {
  features: new Set(), requestedCheckpoints: new Set(),
  loadedCheckpoints: new Set(), health: null
};

async function apiJson(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok)
    throw new Error(body.error || ("HTTP " + response.status));
  return body;
}

async function detectLiveServer() {
  if (!/^https?:$/.test(window.location.protocol)) return;
  try {
    LIVE.health = await apiJson("/api/health");
    LIVE.features = new Set(LIVE.health.features || []);
    const status = $("liveStatus");
    status.textContent = "live server";
    status.className = "pill ok";
    drawTiles();
    drawMap();
    if (tab === "whatif") drawWhatIf();
  } catch (_) {
    // A static page is a fully supported deployment; an absent server is not
    // an error and deliberately leaves the static label in place.
  }
}

async function loadFullCheckpoint(keyframe) {
  if (!LIVE.features.has("checkpoint")
      || LIVE.loadedCheckpoints.has(keyframe)
      || LIVE.requestedCheckpoints.has(keyframe)) return;
  LIVE.requestedCheckpoints.add(keyframe);
  try {
    const body = await apiJson("/api/checkpoint/" + keyframe);
    D.checkpoints[String(keyframe)] = {
      e: body.e, n: body.n_m, h: body.h, m: body.mode,
      w: body.w, event: body.event
    };
    LIVE.loadedCheckpoints.add(keyframe);
    const status = $("liveStatus");
    status.textContent = "live · " + body.n.toLocaleString()
      + " particles at kf " + keyframe;
    drawMap();
  } catch (error) {
    LIVE.requestedCheckpoints.delete(keyframe);
    const status = $("liveStatus");
    status.textContent = "live checkpoint error";
    status.className = "pill bad";
    status.title = error.message;
  }
}

async function runLiveReplay(edits) {
  return apiJson("/api/replay", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({edits: edits})
  });
}

// ---------- projection ----------
let X0 = 1e9, X1 = -1e9, Y0 = 1e9, Y1 = -1e9;
const grow = (e, n) => { if (!isFinite(e) || !isFinite(n)) return;
  X0 = Math.min(X0, e); X1 = Math.max(X1, e);
  Y0 = Math.min(Y0, n); Y1 = Math.max(Y1, n); };
D.landmarks.forEach(l => grow(l.e, l.n));
(D.backdrop || []).forEach(p => grow(p[0], p[1]));
(D.landmarkGeometry || []).forEach(g => g.points.forEach(p => grow(p[0], p[1])));
D.truth.forEach(p => grow(p[0], p[1]));
H.forEach(h => grow(h.mapE, h.mapN));
const pad = (X1 - X0 + Y1 - Y0) * 0.05 + 150;
X0 -= pad; X1 += pad; Y0 -= pad; Y1 += pad;
const MW = 780, MH = 600;
// Base projection: metres -> the full-extent pixel box. Static layers are built
// once in this space and reached afterwards by a transform.
//
// ONE scale for both axes. The earlier version divided east by MW/span and north
// by MH/span, which squashed north by 600/780 = 0.77: a 1-sigma circle came out
// 30% too large north-south, every shape was distorted, and imagery could not be
// laid over it undistorted. Barely visible at a 25 km extent, badly wrong once
// you zoom in to read geometry. Fitting by the tighter axis letterboxes instead,
// which is the honest trade.
const SC = Math.min(MW / Math.max(X1 - X0, 1e-6),
                    MH / Math.max(Y1 - Y0, 1e-6));
const EMID = (X0 + X1) / 2, NMID = (Y0 + Y1) / 2;
const px0 = e => MW / 2 + (e - EMID) * SC;
const py0 = n => MH / 2 - (n - NMID) * SC;
const mPerPx = 1 / SC;

// ---------- map viewport ----------
// A whole-map run spans 25 km and the thing worth looking at is a few hundred
// metres of trajectory inside it, so the map zooms. The zoom lives in the
// PROJECTION, not in the viewBox, and that choice is the whole design:
//
//   * viewBox zoom scales everything indiscriminately. Landmark glyphs, label
//     offsets and flag rings are sized in screen units on purpose -- a 2.2-unit
//     square is meant to be a 2.2-pixel square -- and at 20x they became
//     40-pixel squares that swallowed the map. Counter-scaling each of them is a
//     list of literals someone has to remember to extend.
//   * Projection zoom keeps the output in one fixed 0..MW x 0..MH screen space.
//     Screen-sized literals stay screen-sized with no bookkeeping, and anything
//     that should scale is expressed in metres and divided by mppx().
//
// VIEW is the visible rectangle in BASE pixel space: k is metres-per-screen-pixel
// relative to the full extent, so k = 1 shows everything and k = 0.05 is 20x.
const VIEW = {x: 0, y: 0, k: 1};
const VIEW_MIN_K = 1 / 500;
const VIEW_MAX_K = 1.5;
const px = e => (px0(e) - VIEW.x) / VIEW.k;
const py = n => (py0(n) - VIEW.y) / VIEW.k;
// Metres per screen pixel at the current zoom. Anything geometric -- a 1-sigma
// radius, a ray length, the scale bar -- goes through this and nothing else.
const mppx = () => mPerPx * VIEW.k;
// Maps base pixel space into the current view, for the static layers.
const staticTransform = () =>
  `translate(${(-VIEW.x / VIEW.k).toFixed(3)} ${(-VIEW.y / VIEW.k).toFixed(3)})`
  + ` scale(${(1 / VIEW.k).toFixed(6)})`;

function applyView() {
  const z = 1 / VIEW.k;
  const el = $("mapzoom");
  if (el) el.textContent = z < 1.02 ? "full extent" : `${z.toFixed(1)}x`;
  drawMap();          // the projection moved, so everything is re-projected
}

function viewFit(x0, y0, x1, y1) {          // rect in BASE pixel space
  let w = x1 - x0, h = y1 - y0;
  const pad = Math.max(w, h) * 0.08 + 4;
  x0 -= pad; y0 -= pad; w += 2 * pad; h += 2 * pad;
  const k = Math.min(VIEW_MAX_K,
                     Math.max(VIEW_MIN_K, Math.max(w / MW, h / MH)));
  VIEW.k = k;
  VIEW.x = x0 + w / 2 - MW * k / 2;         // centre the requested rect
  VIEW.y = y0 + h / 2 - MH * k / 2;
  applyView();
}

function viewFitAll() { viewFit(0, 0, MW, MH); }

// The truth track plus the complete estimate trajectory. With no truth, the
// estimate alone still defines a useful fit instead of falling back to all.
function viewFitTrack() {
  let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
  const g = (x, y) => { if (!isFinite(x) || !isFinite(y)) return;
    x0 = Math.min(x0, x); x1 = Math.max(x1, x);
    y0 = Math.min(y0, y); y1 = Math.max(y1, y); };
  D.truth.forEach(p => g(px0(p[0]), py0(p[1])));
  H.forEach(h => g(px0(h.mapE), py0(h.mapN)));
  if (x1 <= x0 || y1 <= y0) { viewFitAll(); return; }
  viewFit(x0, y0, x1, y1);
}

// Zoom about a screen point, keeping the metre under the cursor put.
function viewZoomBy(factor, sx, sy) {
  const k = Math.min(VIEW_MAX_K, Math.max(VIEW_MIN_K, VIEW.k * factor));
  if (k === VIEW.k) return;
  VIEW.x += sx * (VIEW.k - k);
  VIEW.y += sy * (VIEW.k - k);
  VIEW.k = k;
  applyView();
}

// Client coordinates -> the fixed 0..MW x 0..MH screen space.
function eventToScreen(ev) {
  const r = $("map").getBoundingClientRect();
  return {sx: (ev.clientX - r.left) / r.width * MW,
          sy: (ev.clientY - r.top) / r.height * MH};
}

function wireMapZoom() {
  const svg = $("map");
  if (!svg) return;
  svg.addEventListener("wheel", ev => {
    ev.preventDefault();
    const at = eventToScreen(ev);
    viewZoomBy(Math.exp(ev.deltaY * 0.0016), at.sx, at.sy);
  }, {passive: false});
  let drag = null;
  svg.addEventListener("pointerdown", ev => {
    drag = eventToScreen(ev); drag.id = ev.pointerId;
    svg.setPointerCapture(ev.pointerId);
    svg.classList.add("dragging");
  });
  svg.addEventListener("pointermove", ev => {
    if (!drag || ev.pointerId !== drag.id) return;
    const now = eventToScreen(ev);
    VIEW.x -= (now.sx - drag.sx) * VIEW.k;
    VIEW.y -= (now.sy - drag.sy) * VIEW.k;
    drag.sx = now.sx; drag.sy = now.sy;
    applyView();
  });
  const end = ev => {
    if (!drag || ev.pointerId !== drag.id) return;
    drag = null; svg.classList.remove("dragging");
  };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);
  svg.addEventListener("dblclick", ev => {
    const at = eventToScreen(ev); viewZoomBy(0.5, at.sx, at.sy);
  });
  const bAll = $("tgFitAll"), bTrk = $("tgFitTrack");
  if (bAll) bAll.onclick = viewFitAll;
  if (bTrk) bTrk.onclick = viewFitTrack;
}

// ---------- static map layers, built once ----------
// The backdrop and basemap do not change with the scrubber, and rebuilding
// ~12k dots and 37k basemap vertices every frame is what makes a viewer feel
// broken. Each becomes one path string, concatenated ahead of the live layers.
const BASE_STYLE = {
  land:      {fill: "var(--land)",  stroke: "var(--struct)", w: 0.5, op: 1},
  water:     {fill: "var(--sea)",   stroke: "none",          w: 0,   op: 1},
  coastline: {fill: "none",         stroke: "var(--struct)", w: 0.9, op: .95},
  pier:      {fill: "none",         stroke: "var(--struct)", w: 0.7, op: .8},
  bridge:    {fill: "none",         stroke: "var(--struct)", w: 0.8, op: .8},
  building:  {fill: "var(--struct)", stroke: "none",         w: 0,   op: .35},
};
function buildBasemap() {
  const layers = (D.basemap && D.basemap.layers) || [];
  return layers.map(layer => {
    const st = BASE_STYLE[layer.name] || BASE_STYLE.coastline;
    let d = "";
    for (const path of layer.paths) {
      for (let i = 0; i < path.length; i += 2)
        d += (i ? "L" : "M") + px0(path[i]).toFixed(1) + " "
          + py0(path[i + 1]).toFixed(1);
      if (layer.kind === "polygon") d += "Z";
    }
    return `<path d="${d}" fill="${st.fill}" stroke="${st.stroke}"
      stroke-width="${st.w}" opacity="${st.op}" stroke-linejoin="round"/>`;
  }).join("");
}
// A georeferenced raster underlay, if one was supplied. Positioned in BASE
// pixel space and emitted inside #staticmap, so it inherits the zoom transform
// and needs no per-frame work. Beneath the vector layers, which are drawn over
// it as thin outlines rather than fills once imagery is showing.
const SATELLITE = (() => {
  const layers = (D.satellite && D.satellite.layers) || [];
  return layers.map(t => {
    const x = px0(t.e0), y = py0(t.n1);          // north maps to the SMALLER y
    const w = px0(t.e1) - px0(t.e0), h = py0(t.n0) - py0(t.n1);
    // image-rendering:pixelated keeps a coarse wide layer honest about being
    // coarse rather than smearing it into something that looks like detail.
    return `<image href="${t.uri}" x="${x.toFixed(2)}" y="${y.toFixed(2)}"
      width="${w.toFixed(2)}" height="${h.toFixed(2)}"
      preserveAspectRatio="none" opacity=".92"
      style="image-rendering:pixelated"/>`;
  }).join("");
})();
const BASEMAP = buildBasemap();
const BACKDROP = (D.backdrop && D.backdrop.length)
  ? `<path d="${D.backdrop.map(p => "M" + px0(p[0]).toFixed(1) + " "
      + py0(p[1]).toFixed(1) + "h.1").join("")}" stroke="var(--ink-faint)"
      stroke-width="1.5" stroke-linecap="round" fill="none" opacity=".38"/>`
  : "";
const LANDMARK_GEOMETRY = !(D.landmarkGeometry || []).length ? ""
  : D.landmarkGeometry.map(g => {
    let d = "";
    g.points.forEach((p, i) => {
      d += (i ? "L" : "M") + px0(p[0]).toFixed(2) + " "
        + py0(p[1]).toFixed(2); });
    if (g.kind === "polygon") d += "Z";
    const stroke = g.referenced ? "var(--caution)" : "var(--ink-faint)";
    return `<path d="${d}" fill="none" stroke="${stroke}" stroke-width="1"
      stroke-linejoin="round" opacity="${g.referenced ? .75 : .38}"/>`;
  }).join("");

// Scale bar: a map with no basemap and no scale is unreadable at a glance.
// Recomputed for the current viewport rather than built once, for two reasons:
// anchored at fixed coordinates it slides out of frame as soon as you pan, and a
// bar sized for a 25 km view reads 5 km when you are looking at 200 m of quay.
function scaleBar() {
  // Screen space is fixed, so the bar sits in the corner and only its LENGTH and
  // label follow the zoom -- it reads 200 m on a quay and 5 km on the full box.
  const raw = 130 * mppx();
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const nice = [1, 2, 5, 10].map(m => m * pow)
    .reduce((a, b) => Math.abs(b - raw) < Math.abs(a - raw) ? b : a);
  const w = nice / mppx();
  const y = MH - 16, x = 14, tick = 4;
  const label = nice >= 1000 ? (nice / 1000) + " km" : Math.round(nice) + " m";
  return `<g opacity=".85"><line x1="${x}" y1="${y}" x2="${x + w}" y2="${y}"
    stroke="var(--ink)" stroke-width="1.6"/>
    <line x1="${x}" y1="${y - tick}" x2="${x}" y2="${y + tick}"
      stroke="var(--ink)" stroke-width="1.6"/>
    <line x1="${x + w}" y1="${y - tick}" x2="${x + w}" y2="${y + tick}"
      stroke="var(--ink)" stroke-width="1.6"/>
    <text class="axis" x="${x + w / 2}" y="${y - tick * 1.8}"
      text-anchor="middle" fill="var(--ink)">${label}</text></g>`;
}

// ---------- landmark glyphs by type (§7.4) ----------
// Shape carries class, so "which kind of thing is the filter believing in"
// survives a screenshot. Fill intensity carries whether it is live right now.
function glyph(l, r, hot) {
  const x = px(l.e), y = py(l.n);
  const col = hot ? "var(--accent)" : "var(--caution)";
  const sw = hot ? 1.3 : 0.6, op = hot ? 1 : 0.62;
  const a = {stroke: `stroke="${col}" stroke-width="${sw}" opacity="${op}"`,
             fill: `fill="${hot ? col : "none"}"`};
  const A = `${a.fill} ${a.stroke}`;
  switch (l.g) {
    case "light": // triangle point-up: fixed light / lighthouse
      return `<path d="M${x} ${y - r * 1.2}L${x + r} ${y + r * .8}L${x - r} ${y + r * .8}Z" ${A}/>`;
    case "navaid": // diamond: floating or minor aid
      return `<path d="M${x} ${y - r}L${x + r} ${y}L${x} ${y + r}L${x - r} ${y}Z" ${A}/>`;
    case "tank": // circle: storage tank, silo, chimney
      return `<circle cx="${x}" cy="${y}" r="${r * .95}" ${A}/>`;
    case "tower": // vertical bar with a cap: tower, mast, crane, monument
      return `<path d="M${x} ${y + r}L${x} ${y - r * 1.3}" ${a.stroke}
        stroke-width="${sw + 0.9}"/><circle cx="${x}" cy="${y - r * 1.3}"
        r="${r * .48}" ${A}/>`;
    case "bridge": // horizontal double rule
      return `<path d="M${x - r * 1.2} ${y - r * .4}h${r * 2.4}M${x - r * 1.2} ${y + r * .4}h${r * 2.4}" ${a.stroke}/>`;
    case "water": // half-square open to the water: pier, dock, marina
      return `<path d="M${x - r} ${y + r}L${x - r} ${y - r}L${x + r} ${y - r}L${x + r} ${y + r}" ${a.fill} ${a.stroke}/>`;
    case "nature": // hollow rounded blob: island, cape, beach
      return `<circle cx="${x}" cy="${y}" r="${r}" fill="none" ${a.stroke}
        stroke-dasharray="2 1.5"/>`;
    default: // square: building and everything unclassified
      return `<rect x="${x - r * .85}" y="${y - r * .85}" width="${r * 1.7}"
        height="${r * 1.7}" ${A}/>`;
  }
}

// ---------- view 1: run overview strip ----------
// Gutters, not overlays: the row name goes left of the plot and the axis
// extremes go right of it, so a label can never sit on top of the data it
// describes. Everything time-indexed goes through X() and shares one axis.
const SW = 1000, GL = 120, GR = 54, PW = SW - GL - GR, ROW = 32;
const X = kf => GL + (kf / KF) * PW;
function sparkline(y0, vals, label, col, log) {
  const fin = vals.filter(v => v !== undefined && v !== null && isFinite(v));
  if (!fin.length)
    return `<text class="axis" x="2" y="${y0 + 13}"
      opacity=".55">${label}</text>
      <text class="axis" x="${GL + 4}" y="${y0 + 13}" opacity=".55">
      no data</text>`;
  let lo = Math.min(...fin), hi = Math.max(...fin);
  if (log) { lo = Math.max(lo, 1e-3); hi = Math.max(hi, lo * 1.01); }
  if (hi - lo < 1e-9) hi = lo + 1;
  const top = y0 + 4, base = y0 + ROW - 6;
  const sc = v => {
    const a = log ? Math.log(Math.max(v, lo)) : v;
    const b = log ? Math.log(lo) : lo, c = log ? Math.log(hi) : hi;
    return base - ((a - b) / (c - b)) * (base - top);
  };
  let d = "", area = "", last = null;
  vals.forEach((v, i) => {
    if (v === undefined || v === null || !isFinite(v)) return;
    const x = X(i), y = sc(v);
    d += (d ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
    area += (area ? "L" : "M" + x.toFixed(1) + " " + base + "L")
      + x.toFixed(1) + " " + y.toFixed(1);
    last = [x, y];
  });
  const num = v => Math.abs(v) < 10 ? v.toFixed(2)
    : Math.abs(v) < 100000 ? String(Math.round(v))
    : (v / 1000).toFixed(0) + "k";
  return `<line x1="${GL}" y1="${base}" x2="${GL + PW}" y2="${base}"
    stroke="var(--grid)"/>
  <line x1="${GL}" y1="${top}" x2="${GL + PW}" y2="${top}"
    stroke="var(--grid)"/>
  <path d="${area}L${last ? last[0].toFixed(1) : GL} ${base}Z" fill="${col}"
    opacity=".10"/>
  <path d="${d}" fill="none" stroke="${col}" stroke-width="1.3"
    stroke-linejoin="round"/>
  <text class="axis" x="2" y="${(top + base) / 2 + 3}"
    fill="${col}">${label}</text>
  <text class="axis" x="${SW - 2}" y="${top + 4}" text-anchor="end">${
    num(hi)}</text>
  <text class="axis" x="${SW - 2}" y="${base + 2}" text-anchor="end">${
    num(lo)}</text>`;
}

// Mode ribbons: thickness is weight over time, so a mode that dies thins to
// nothing instead of simply stopping. This is the §7.4 view-4 "weight
// trajectory" living in the strip where it can be compared across modes.
function modeRibbons(y0) {
  let out = "", y = y0;
  const H_RIB = 15;
  for (const mode of D.modes) {
    if (!mode.kf.length) continue;
    const mid = y + H_RIB / 2;
    let top = "", bot = "";
    mode.kf.forEach((kf, i) => {
      const x = X(kf), half = Math.max(mode.w[i] * (H_RIB / 2 - 1), 0.3);
      top += (i ? "L" : "M") + x.toFixed(1) + " " + (mid - half).toFixed(2);
      bot = "L" + x.toFixed(1) + " " + (mid + half).toFixed(2) + bot;
    });
    const peak = Math.max(...mode.w);
    out += `<line x1="${GL}" y1="${mid}" x2="${GL + PW}" y2="${mid}"
      stroke="var(--grid)"/>
      <path d="${top}${bot}Z" fill="${color(mode.id)}"
      opacity="${selMode === mode.id ? .95 : .6}" class="ribbon"
      data-mode="${mode.id}" style="cursor:pointer"><title>mode ${mode.id}: born kf ${mode.born}${
      mode.died !== undefined ? ", died kf " + mode.died : ""}, peak weight ${
      (peak * 100).toFixed(0)}%</title></path>
      <text class="axis" x="2" y="${mid + 3}"
        fill="${color(mode.id)}" font-weight="${selMode === mode.id ? 700 : 400}"
        >mode ${mode.id}</text>
      <text class="axis" x="${SW - 2}" y="${mid + 3}" text-anchor="end">${
      (peak * 100).toFixed(0)}%</text>`;
    y += H_RIB;
  }
  return {svg: out, height: Math.max(y - y0, 14)};
}

const EV_STYLE = {
  proposal: {glyph: "◇", col: "var(--accent)"},
  map_jump: {glyph: "⤴", col: "var(--port)"},
  ess_crash: {glyph: "↓", col: "var(--port)"},
  resample_storm: {glyph: "≈", col: "var(--caution)"},
  null_spike: {glyph: "⊘", col: "var(--caution)"},
  association_flip: {glyph: "⇄", col: "var(--caution)"},
  mode_birth: {glyph: "●", col: "var(--water)"},
  mode_death: {glyph: "×", col: "var(--ink-faint)"},
  mode_merge: {glyph: "⊕", col: "var(--water)"},
};
function eventRail(y0) {
  // One row per kind, so a hundred association flips cannot bury the one MAP
  // jump that explains the run, and the count per kind is visible at a glance.
  const order = Object.keys(EV_STYLE);
  const kinds = [...new Set(D.events.map(e => e.kind))].sort(
    (a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99));
  let out = "", y = y0;
  for (const kind of kinds) {
    const st = EV_STYLE[kind] || {glyph: "•", col: "var(--ink-soft)"};
    const evs = D.events.filter(e => e.kind === kind);
    out += `<text class="axis" x="2" y="${y + 9}" fill="${st.col}">${
      st.glyph} ${kind}</text>
      <text class="axis" x="${SW - 2}" y="${y + 9}" text-anchor="end">${
      evs.length}</text>
      <line x1="${GL}" y1="${y + 6}" x2="${GL + PW}" y2="${y + 6}"
        stroke="var(--grid)"/>`;
    for (const ev of evs) {
      out += `<text x="${X(ev.keyframe_idx).toFixed(1)}" y="${y + 9.5}"
        fill="${st.col}" font-size="9.5" text-anchor="middle" class="evg"
        data-kf="${ev.keyframe_idx}" style="cursor:pointer"
        opacity="${ev.source === "derived" ? .7 : 1}">${st.glyph}<title>kf ${
        ev.keyframe_idx} — ${esc(ev.label)}: ${esc(ev.detail)}</title></text>`;
    }
    y += 13;
  }
  return {svg: out, height: Math.max(y - y0, 13)};
}

function drawStrip() {
  let y = 4, out = "";
  const S = k => H.map(h => h[k]);
  out += sparkline(y, S("err"), "pos err (m)", "var(--port)", true); y += ROW;
  const massMetric = RUN.positionMassMetric;
  if (massMetric) for (const radius of massMetric.radiiM) {
    const key = `${massMetric.id}@${massMetric.version}:radius_m=${Number(radius)}`;
    const values = H.map(h => h.positionMass ? h.positionMass[key] : undefined);
    out += sparkline(y, values, `mass ≤ ${Number(radius)} m`,
      "var(--starboard)", false);
    y += ROW;
  }
  out += sparkline(y, S("sigma"), "reported σ (m)", "var(--starboard)", true); y += ROW;
  out += sparkline(y, S("ess"), "ESS", "var(--water)", true); y += ROW;
  out += sparkline(y, S("null"), "null share", "var(--caution)", false); y += ROW;
  out += sparkline(y, S("entropy"), "mode entropy", "var(--accent)", false);
  y += ROW + 4;
  const rib = modeRibbons(y); out += rib.svg; y += rib.height + 6;
  const rail = eventRail(y); out += rail.svg; y += rail.height + 4;
  // Keyframe ruler, so the horizontal axis is readable without hovering.
  const ticks = 8;
  for (let i = 0; i <= ticks; i++) {
    const kf = Math.round(KF * i / ticks);
    out += `<text class="axis" x="${X(kf).toFixed(1)}" y="${y + 8}"
      text-anchor="middle">${kf}</text>`;
  }
  y += 12;
  const cx = X(t);
  out += `<line class="cursor" x1="${cx.toFixed(1)}" y1="0"
    x2="${cx.toFixed(1)}" y2="${y - 12}"/>
    <text class="axis" x="${cx.toFixed(1)}" y="${y + 8}" text-anchor="middle"
      fill="var(--accent)" font-weight="700">kf ${t}</text>`;
  const svg = $("strip");
  svg.setAttribute("viewBox", `0 0 ${SW} ${y + 2}`);
  svg.innerHTML = out;
  svg.querySelectorAll(".ribbon").forEach(el => el.onclick = ev => {
    ev.stopPropagation();
    selMode = selMode === +el.dataset.mode ? null : +el.dataset.mode;
    render();
  });
  svg.querySelectorAll(".evg").forEach(el => el.onclick = ev => {
    ev.stopPropagation(); t = +el.dataset.kf; render();
  });
}

// ---------- view 2: map ----------
function drawMap() {
  const h = H[t], ck = nearestCk(t), P = D.checkpoints[ck];
  loadFullCheckpoint(ck);
  // Static layers stay in base coordinates and are moved by one transform:
  // re-emitting 138k basemap vertices on every wheel tick is not affordable.
  let out = `<g id="staticmap" transform="${staticTransform()}">`
    + (showSat ? SATELLITE : "")
    + (showBase ? BASEMAP : "") + LANDMARK_GEOMETRY + BACKDROP
    + `</g>`;

  if (showParticles && P) {
    // One path per mode rather than one circle per particle: 900 DOM nodes per
    // frame is what makes scrubbing stutter.
    const byMode = new Map();
    for (let i = 0; i < P.e.length; i++) {
      const m = P.m[i];
      if (selMode !== null && m !== selMode) continue;
      let d = byMode.get(m); if (!d) byMode.set(m, d = []);
      d.push("M" + px(P.e[i]).toFixed(1) + " " + py(P.n[i]).toFixed(1) + "h.1");
    }
    for (const [m, d] of byMode)
      out += `<path d="${d.join("")}" stroke="${color(m)}" stroke-width="2.4"
        stroke-linecap="round" fill="none" opacity=".5"/>`;
  }

  // Truth then baseline MAP trail then ghosts, so the baseline reads on top.
  let dt = "";
  D.truth.forEach((p, i) => {
    dt += (i ? "L" : "M") + px(p[0]).toFixed(1) + " " + py(p[1]).toFixed(1); });
  if (dt) out += `<path d="${dt}" fill="none" stroke="var(--truth)"
    stroke-width="1.5" stroke-dasharray="4 3"/>`;
  if (showGhosts) D.ghosts.forEach(g => {
    let d = "";
    g.trail.slice(0, t + 1).forEach((p, i) => {
      d += (i ? "L" : "M") + px(p[0]).toFixed(1) + " " + py(p[1]).toFixed(1); });
    out += `<path d="${d}" fill="none" stroke="var(--water)" stroke-width="1.4"
      stroke-dasharray="6 3" opacity=".8"><title>${esc(g.label)}</title></path>`;
  });
  let dm = "";
  H.slice(0, t + 1).forEach((r, i) => {
    dm += (i ? "L" : "M") + px(r.mapE).toFixed(1) + " " + py(r.mapN).toFixed(1); });
  out += `<path d="${dm}" fill="none" stroke="var(--accent)" stroke-width="1.6"/>`;

  // Per-mode 1-sigma circle + heading tick.
  h.modes.forEach(m => {
    if (selMode !== null && m.id !== selMode) return;
    const r = Math.max(m.std / mppx(), 2.5);   // metres -> current pixels
    const a = m.h * Math.PI / 180, L = 20;
    out += `<circle cx="${px(m.e).toFixed(1)}" cy="${py(m.n).toFixed(1)}"
      r="${r.toFixed(1)}" fill="none" stroke="${color(m.id)}"
      stroke-width="1.4"/>
      <line x1="${px(m.e).toFixed(1)}" y1="${py(m.n).toFixed(1)}"
        x2="${(px(m.e) + L * Math.sin(a)).toFixed(1)}"
        y2="${(py(m.n) - L * Math.cos(a)).toFixed(1)}"
        stroke="${color(m.id)}" stroke-width="1.4"/>
      <text class="axis" x="${(px(m.e) + 6).toFixed(1)}"
        y="${(py(m.n) - 6).toFixed(1)}" fill="${color(m.id)}">m${m.id} ${
      (m.w * 100).toFixed(0)}%</text>`;
  });

  // Bearing wedges and correspondence, from the selected (else heaviest) mode.
  const anchor = h.modes.find(m => m.id === selMode) || h.modes[0];
  const meas = D.measurements[String(t)] || [];
  const flags = [];
  if (anchor) meas.forEach(mm => {
    // Forward->world bearing: forward-world orientation + forward-frame
    // bearing, degrees clockwise from north. JS cannot import Python, so this
    // restates the convention owned
    // by farfield.geometry.forward_to_world_bearing_cw_deg — keep them in step.
    const world = (anchor.h + mm.bearing) * Math.PI / 180;
    const R = 3 * MW;      // long enough to leave the frame at any zoom
    const ray = (off, op, w) => `<line x1="${px(anchor.e).toFixed(1)}"
      y1="${py(anchor.n).toFixed(1)}"
      x2="${(px(anchor.e) + R * Math.sin(world + off)).toFixed(1)}"
      y2="${(py(anchor.n) - R * Math.cos(world + off)).toFixed(1)}"
      stroke="var(--accent)" stroke-width="${w}" opacity="${op}"/>`;
    const s = mm.sigma * Math.PI / 180;
    out += ray(0, .8, 1.3) + ray(2 * s, .3, .8) + ray(-2 * s, .3, .8);

    // §7.4 red flag: the matcher's best claim vs where that landmark is.
    // Computed against THIS mode's pose, so it re-evaluates as you switch
    // modes — a claim can be geometrically fine under one hypothesis and
    // absurd under another, which is the whole point of per-mode views.
    const top = mm.topLm ? LM.get(mm.topLm) : null;
    if (top) {
      const predicted = Math.atan2(top.e - anchor.e, top.n - anchor.n)
        * 180 / Math.PI;
      const disagree = Math.abs(wrap180(predicted - (anchor.h + mm.bearing)));
      if (disagree > 15) {
        flags.push({trk: mm.trk, lm: mm.topLm, deg: disagree});
        out += `<line x1="${px(anchor.e).toFixed(1)}" y1="${py(anchor.n).toFixed(1)}"
          x2="${px(top.e).toFixed(1)}" y2="${py(top.n).toFixed(1)}"
          stroke="var(--port)" stroke-width="1.5" stroke-dasharray="1 3"
          opacity=".85"><title>${esc(mm.trk)}: matcher's best claim ${
          esc(mm.topLm)} lies ${disagree.toFixed(0)}° off the measured
          bearing under mode ${anchor.id}</title></line>
          <circle cx="${px(top.e).toFixed(1)}" cy="${py(top.n).toFixed(1)}"
            r="7" fill="none" stroke="var(--port)" stroke-width="1.6"/>`;
      }
    }
    const a = h.assoc.find(x => x.trk === mm.trk && x.mode === anchor.id)
           || h.assoc.find(x => x.trk === mm.trk && x.mode === null);
    if (a) Object.entries(a.resp).forEach(([lm, p]) => {
      const L = LM.get(lm); if (!L || p < 0.02) return;
      out += `<line x1="${px(anchor.e).toFixed(1)}" y1="${py(anchor.n).toFixed(1)}"
        x2="${px(L.e).toFixed(1)}" y2="${py(L.n).toFixed(1)}"
        stroke="${color(anchor.id)}" stroke-width="${(1 + 2 * p).toFixed(1)}"
        opacity="${(0.15 + 0.6 * p).toFixed(2)}" stroke-dasharray="2 3"/>`;
    });
  });

  // Landmarks on top. Labels only where the filter is putting association
  // mass right now (or everywhere, in a small world): a whole-map run
  // references thousands of tie members and labelling them all is illegible.
  const active = {};
  h.assoc.forEach(a => Object.entries(a.resp).forEach(([lm, p]) => {
    if (p >= 0.05) active[lm] = Math.max(active[lm] || 0, p); }));
  meas.forEach(mm => { if (mm.topLm) active[mm.topLm] = active[mm.topLm] || 0.05; });
  const many = D.landmarks.length > 60;
  D.landmarks.forEach(l => {
    const hot = active[l.id] !== undefined;
    const r = many ? (hot ? 4.5 : 2.2) : 5.5;
    out += glyph(l, r, hot);
    if ((hot || !many) && (selTrk === null || true))
      out += `<text class="axis" x="${(px(l.e) + 8).toFixed(1)}"
        y="${(py(l.n) + 3).toFixed(1)}">${esc(l.id)}</text>`;
  });
  if (h.truthE !== undefined)
    out += `<circle cx="${px(h.truthE).toFixed(1)}" cy="${py(h.truthN).toFixed(1)}"
      r="4.5" fill="none" stroke="var(--truth)" stroke-width="2"/>`;
  out += scaleBar();

  const svg = $("map");
  svg.setAttribute("viewBox", `0 0 ${MW} ${MH}`);   // fixed: the zoom is in VIEW
  svg.innerHTML = out;
  $("mapnote").innerHTML =
    `particles: weighted sample of ${RUN.nParticles.toLocaleString()} from `
    + `checkpoint kf ${ck}${ck !== t ? ` (nearest to ${t})` : ""}`
    + (flags.length ? ` · <b style="color:var(--port)">${flags.length} `
        + `LLR/geometry disagreement${flags.length > 1 ? "s" : ""}: `
        + flags.map(f => `${esc(f.trk)} ${f.deg.toFixed(0)}°`).join(", ")
        + "</b>" : "");
}

// ---------- tiles + state ----------
function drawTiles() {
  const h = H[t], last = H[H.length - 1];
  let status = "ok", label = "tracking";
  if (h.err !== undefined) {
    if (h.err > 3 * Math.max(h.sigma, 1)) { status = "bad"; label = "overconfident"; }
    else if (h.modes.length > 1) { status = "warn"; label = "multimodal"; }
    else if (h.err > 500) { status = "warn"; label = "searching"; }
  } else { status = "info"; label = "no ground truth"; }
  const tile = (k, v, u = "") => `<div class="tile"><div class="k">${k}</div>
    <div class="v mono">${v}<span class="u"> ${u}</span></div></div>`;
  $("tiles").innerHTML =
    `<div class="tile"><div class="k">status @ kf ${h.kf}</div>
      <div class="v"><span class="pill ${status}">${label}</span></div></div>`
    + tile("mean error", fmt(h.err), "m")
    + tile("reported σ", fmt(h.sigma), "m")
    + tile("MAP error", fmt(h.mapErr), "m")
    + tile("modes", h.modes.length, `H=${fmt(h.entropy, 2)}`)
    + tile("ESS", fmt(h.ess), `/ ${RUN.nParticles.toLocaleString()}`)
    + tile("null share", fmt(h.null, 2))
    + tile("final error", fmt(last.err), "m")
    + `<div class="tile"><div class="k">replay</div><div class="v">
      <span class="pill ${RUN.replayable ? "ok" : "bad"}">${
      RUN.replayable ? "exact" : "not replayable"}</span></div></div>`;
}

function drawState() {
  const h = H[t];
  $("state").innerHTML = `<dl class="kv">
    <dt>keyframe</dt><dd>${h.kf} / ${KF}</dd>
    <dt>mean pose</dt><dd>${fmt(h.meanE)}, ${fmt(h.meanN)} @ ${fmt(h.meanH)}°</dd>
    <dt>heading error</dt><dd>${fmt(h.headingErr, 2)}°</dd>
    <dt>heading σ</dt><dd>${fmt(h.headingSigma, 2)}°</dd>
    <dt>measurements</dt><dd>${h.nMeas}</dd>
    <dt>resampled</dt><dd>${h.resampled ? "yes" : "no"}</dd>
    <dt>proposal-descended</dt><dd>${(h.proposalShare * 100).toFixed(0)}%</dd>
    </dl>
    <h3>Association posteriors ${selMode !== null
      ? "(mode " + selMode + ")" : "(whole belief)"}</h3>` + assocTable();
}
function assocTable() {
  const h = H[t];
  const rows = h.assoc.filter(a => selMode === null ? a.mode === null
                                                    : a.mode === selMode);
  if (!rows.length) return `<div class="empty">no measurement at this keyframe</div>`;
  let out = "";
  for (const a of rows) {
    const parts = Object.entries(a.resp)
      .map(([lm, p]) => `${esc(lm)} <b>${(p * 100).toFixed(0)}%</b>`).join(" · ");
    out += `<tr class="click" data-trk="${esc(a.trk)}">
      <td><code>${esc(a.trk)}</code></td>
      <td class="num">${(a.null * 100).toFixed(0)}%</td>
      <td class="num">${(a.surprise * 100).toFixed(0)}%</td>
      <td class="prov">${parts || "—"}</td></tr>`;
  }
  return `<div class="scroll"><table><thead><tr><th>tracklet</th>
    <th class="num">null</th><th class="num">surprise</th>
    <th>believes it is</th></tr></thead><tbody>${out}</tbody></table></div>`;
}

// ---------- view 4: mode ledger + death waterfalls ----------
function waterfallHtml(wf) {
  if (!wf) return `<div class="empty">no attribution for this mode</div>`;
  const max = Math.max(...wf.terms.map(term => Math.abs(term.nats)), 1e-6);
  const marker = {tracklet: "", recluster: "✱ ", settle: "✱ ",
                  injection: "! ", resample: "≈ "};
  let rows = "";
  for (const term of wf.terms.slice(0, 12)) {
    const w = Math.abs(term.nats) / max * 100;
    const col = term.kind !== "tracklet" ? "var(--ink-faint)"
      : term.nats < 0 ? "var(--port)" : "var(--starboard)";
    rows += `<div class="lab" title="${esc(term.kind)}">${
      marker[term.kind] || ""}${esc(term.label)}</div>
      <div style="display:flex;align-items:center;gap:6px;justify-content:flex-end">
        <span class="bar" style="width:${w.toFixed(1)}px;max-width:110px;
          background:${col}"></span>
        <span class="mono" style="min-width:52px;text-align:right">${
        term.nats >= 0 ? "+" : ""}${term.nats.toFixed(2)}</span></div>`;
  }
  const extra = wf.terms.length > 12 ? `<div class="prov">+ ${
    wf.terms.length - 12} smaller terms</div>` : "";
  return `<div class="prov" style="margin-bottom:6px">
    kf ${wf.range[0]}–${wf.range[1]}: <b>${wf.total >= 0 ? "+" : ""}${
    wf.total.toFixed(2)} nats</b> of log-share
    (<span style="color:var(--starboard)">${wf.evidence >= 0 ? "+" : ""}${
    wf.evidence.toFixed(2)} evidence</span>,
    <span style="color:var(--ink-faint)">${wf.structural >= 0 ? "+" : ""}${
    wf.structural.toFixed(2)} structural</span>)</div>
    <div class="wf">${rows}</div>${extra}
    ${wf.residual !== null ? `<div class="prov" style="margin-top:5px">
      residual vs Tier 0: ${wf.residual >= 0 ? "+" : ""}${wf.residual.toFixed(3)}
      nats — the decomposition is checked against the independently
      recorded mode weight.</div>` : ""}`;
}

function drawModes() {
  const h = H[t];
  const live = new Set(h.modes.map(m => m.id));
  let rows = "";
  for (const mode of D.modes) {
    const now = h.modes.find(m => m.id === mode.id);
    const p = mode.prov || {};
    const origin = p.source === "proposal"
      ? `proposal #${p.proposal_event_id} (${p.trigger || "?"})`
        + (p.landmark_ids ? `<br><code>${esc(p.landmark_ids)}</code>` : "")
      : "motion";
    // Sparkline of this mode's weight over its whole life: the §7.4 "weight
    // trajectory" at row scale, so the ledger answers "was it ever strong".
    const W = 92, Hh = 16;
    let d = "";
    mode.kf.forEach((kf, i) => {
      d += (i ? "L" : "M") + (kf / KF * W).toFixed(1) + " "
        + (Hh - mode.w[i] * (Hh - 1)).toFixed(1); });
    rows += `<tr class="click ${selMode === mode.id ? "sel" : ""}"
      data-mode="${mode.id}">
      <td><span class="swatch" style="background:${color(mode.id)}"></span>m${mode.id}</td>
      <td><svg viewBox="0 0 ${W} ${Hh}" width="${W}" height="${Hh}"
        style="width:${W}px"><path d="${d}" fill="none"
        stroke="${color(mode.id)}" stroke-width="1.3"/>
        <line x1="${(t / KF * W).toFixed(1)}" y1="0"
          x2="${(t / KF * W).toFixed(1)}" y2="${Hh}" stroke="var(--accent)"
          stroke-width=".8" opacity=".7"/></svg></td>
      <td class="num">${now ? (now.w * 100).toFixed(1) + "%"
        : `<span class="prov">${live.has(mode.id) ? "" : "gone"}</span>`}</td>
      <td class="num">${now ? fmt(now.std) : "—"}</td>
      <td class="num">${mode.born}</td>
      <td class="num">${mode.died !== undefined ? mode.died : "—"}</td>
      <td class="prov">${origin}</td></tr>`;
  }
  const deaths = (D.attribution && D.attribution.deaths) || {};
  let deathHtml = "";
  for (const [id, wf] of Object.entries(deaths))
    deathHtml += `<h3>Why mode ${id} died</h3>${waterfallHtml(wf)}`;
  if (!Object.keys(deaths).length)
    deathHtml = `<h3>Mode deaths</h3><div class="empty">${
      D.attribution ? "no mode died in this run"
        : "needs the Tier-3 attribution cache"}</div>`;
  $("modes").innerHTML = `<div class="scroll"><table><thead><tr><th>mode</th>
    <th>weight trajectory</th><th class="num">now</th><th class="num">σ</th>
    <th class="num">born</th><th class="num">died</th><th>origin</th></tr>
    </thead><tbody>${rows || `<tr><td colspan="7" class="empty">no modes
    above threshold in this run</td></tr>`}</tbody></table></div>` + deathHtml;
  $("modes").querySelectorAll("tr.click").forEach(r => r.onclick = () => {
    selMode = selMode === +r.dataset.mode ? null : +r.dataset.mode; render(); });
}

// ---------- view 3: tracklet inspector ----------
const VERDICT_PILL = {consistent: "ok", "geometry-unexplained": "bad",
                      "matcher-fault": "bad", "filter-fault": "warn",
                      "no-evidence": "info"};
function drawTrackletList() {
  const rows = D.tracklets.map(trk => {
    const tri = trk.triage;
    const pill = tri ? `<span class="pill ${VERDICT_PILL[tri.verdict] || "info"}"
      >${tri.verdict}</span>` : "";
    const nats = trk.attributionTotal;
    return `<tr class="click ${selTrk === trk.id ? "sel" : ""}"
      data-trk="${esc(trk.id)}">
      <td><code>${esc(trk.id)}</code></td>
      <td class="num">${trk.epochs.length}</td>
      <td class="num">${trk.table ? trk.table.nEndorsed : "—"}</td>
      <td class="num" style="color:${nats === undefined ? "inherit"
        : nats < 0 ? "var(--port)" : "var(--starboard)"}">${
      nats === undefined ? "—" : (nats >= 0 ? "+" : "") + nats.toFixed(1)}</td>
      <td>${pill}${tri && tri.antiEvidence
        ? ' <span class="pill bad">anti</span>' : ""}</td></tr>`;
  }).join("");
  $("trklist").innerHTML = `<div class="scroll scrollY"><table><thead><tr>
    <th>tracklet</th><th class="num">epochs</th><th class="num">endorsed</th>
    <th class="num">nats</th><th>verdict</th></tr></thead>
    <tbody>${rows}</tbody></table></div>
    <div class="legend">"nats" is this tracklet's total contribution to the
    whole belief's log-likelihood (§7.2). "verdict" is truth-privileged.
    </div>`;
  $("trklist").querySelectorAll("tr.click").forEach(r => r.onclick = () => {
    selTrk = selTrk === r.dataset.trk ? null : r.dataset.trk;
    render(); if (selTrk) $("inspector").scrollIntoView({block: "nearest"}); });
}

function seriesChart(rows, xKey, yKey, opts) {
  // A tiny shared line/bar chart: bearing series, attribution series and
  // association evolution are all the same shape over the keyframe axis.
  const o = Object.assign({w: 300, h: 74, col: "var(--water)", bars: false,
                           band: null, zero: false}, opts || {});
  if (!rows.length) return `<div class="empty">no data</div>`;
  const ys = rows.map(r => r[yKey]);
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (o.band) {
    lo = Math.min(lo, ...rows.map(r => r[yKey] - 2 * r[o.band]));
    hi = Math.max(hi, ...rows.map(r => r[yKey] + 2 * r[o.band]));
  }
  if (o.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (hi - lo < 1e-9) { hi = lo + 1; }
  // Local axis: these charts live in a narrow panel and carry their own
  // keyframe range, so they do not share the strip's gutters.
  const X = kf => (kf / KF) * o.w;
  const Y = v => o.h - 12 - ((v - lo) / (hi - lo)) * (o.h - 20);
  let out = `<line x1="0" y1="${Y(lo)}" x2="${o.w}" y2="${Y(lo)}"
    stroke="var(--grid)"/>`;
  if (o.zero && lo < 0 && hi > 0)
    out += `<line x1="0" y1="${Y(0)}" x2="${o.w}" y2="${Y(0)}"
      stroke="var(--ink-faint)" stroke-width=".7" opacity=".6"/>`;
  if (o.band) {
    let up = "", dn = "";
    rows.forEach((r, i) => {
      up += (i ? "L" : "M") + X(r[xKey]).toFixed(1) + " "
        + Y(r[yKey] + 2 * r[o.band]).toFixed(1);
      dn = "L" + X(r[xKey]).toFixed(1) + " "
        + Y(r[yKey] - 2 * r[o.band]).toFixed(1) + dn; });
    out += `<path d="${up}${dn}Z" fill="${o.col}" opacity=".16"/>`;
  }
  if (o.bars) {
    for (const r of rows) {
      const y0 = Y(0), y1 = Y(r[yKey]);
      out += `<line x1="${X(r[xKey]).toFixed(1)}" y1="${y0.toFixed(1)}"
        x2="${X(r[xKey]).toFixed(1)}" y2="${y1.toFixed(1)}"
        stroke="${r[yKey] < 0 ? "var(--port)" : "var(--starboard)"}"
        stroke-width="1.8"><title>kf ${r[xKey]}: ${r[yKey].toFixed(2)}</title></line>`;
    }
  } else {
    let d = "";
    rows.forEach((r, i) => {
      d += (i ? "L" : "M") + X(r[xKey]).toFixed(1) + " " + Y(r[yKey]).toFixed(1); });
    out += `<path d="${d}" fill="none" stroke="${o.col}" stroke-width="1.4"/>`;
    for (const r of rows)
      out += `<circle cx="${X(r[xKey]).toFixed(1)}" cy="${Y(r[yKey]).toFixed(1)}"
        r="1.7" fill="${o.col}"><title>kf ${r[xKey]}: ${
        r[yKey].toFixed(2)}</title></circle>`;
  }
  out += `<line x1="${X(t).toFixed(1)}" y1="0" x2="${X(t).toFixed(1)}"
    y2="${o.h - 12}" stroke="var(--accent)" stroke-width="1" opacity=".8"/>
    <text class="axis" x="0" y="${o.h - 2}">kf 0</text>
    <text class="axis" x="${o.w}" y="${o.h - 2}" text-anchor="end">kf ${KF}</text>
    <text class="axis" x="${o.w}" y="9" text-anchor="end">${hi.toFixed(1)}</text>
    <text class="axis" x="0" y="9">${lo.toFixed(1)}</text>`;
  return `<svg viewBox="0 0 ${o.w} ${o.h}">${out}</svg>`;
}

function drawInspector() {
  if (!selTrk) {
    $("inspector").innerHTML = `<h2>Tracklet inspector <span class="hint">
      &mdash; §7.4 view 3</span></h2>
      <div class="empty">Pick a tracklet from the list, the association table,
      or a correspondence line on the map. This panel shows the completed-run
      bearing observations, compatibility table, and filter responsibilities
      side by side.</div>`;
    return;
  }
  const trk = TRK.get(selTrk);
  if (!trk) { $("inspector").innerHTML = `<div class="empty">unknown tracklet</div>`;
    return; }
  const tbl = trk.table;

  // --- canonical tracker/audit source ---
  const src = trk.source;
  let source = `<h3>Tracker + semantic audit</h3>`;
  if (src) {
    if (src.chip) source += `<img class="crop" src="${src.chip}"
      alt="audited evidence chip for ${esc(src.localId)}" loading="lazy">`;
    source += `<dl class="kv" style="margin-top:8px">
      <dt>source track</dt><dd>${esc(src.localId)} (track ${
        src.sourceTrackId})</dd>
      <dt>audit</dt><dd><span class="pill ${
        src.verdict === "drop" ? "bad" : src.verdict === "keep_partial"
          ? "warn" : "ok"}">${esc(src.verdict)}</span>
        ${esc(src.confidence)} confidence</dd>
      <dt>name</dt><dd>${esc(src.name || "—")}</dd>
      <dt>supports</dt><dd>${src.nSupports}</dd>
      <dt>span</dt><dd>kf ${src.span[0]}–${src.span[1]}</dd>
      <dt>accepted</dt><dd>${src.validSegments.map(
        segment => `kf ${segment[0]}–${segment[1]}`).join(", ")}</dd></dl>`;
    if (src.tags.length)
      source += `<div style="margin-top:6px">${src.tags.map(
        tag => `<span class="chip">${esc(tag)}</span>`).join("")}</div>`;
    if (src.description)
      source += `<div class="legend">${esc(src.description)}</div>`;
    if (src.features.length)
      source += `<div class="legend">distinctive: ${
        src.features.map(esc).join("; ")}</div>`;
    if (src.unresolved)
      source += `<div class="legend" style="color:var(--caution)">unresolved:
        ${esc(src.unresolved)}</div>`;
  } else {
    source += `<div class="empty">No canonical source artifacts were supplied.
      Pass the exact object_tracks and semantic_audits artifacts.</div>`;
  }

  // --- bearing series ---
  const bearings = `<h3>Nominal-forward bearing &amp; κ</h3>` + seriesChart(
    trk.epochs, "kf", "bearing", {band: "sigma", col: "var(--accent)"})
    + `<div class="legend">Nominal-forward-frame clockwise bearing, shaded ±2σ from the
    fused κ. A step here that the vehicle did not make is a tracker
    problem, not a matcher one.</div>`;

  // --- matcher column ---
  let matcher = `<h3>Matcher — log-LR per candidate</h3>`;
  if (tbl) {
    const maxAbs = Math.max(...tbl.entries.map(e => Math.abs(e.lr)),
                            Math.abs(tbl.default), 1e-6);
    let bars = "";
    for (const e of tbl.entries.slice(0, 14)) {
      const w = Math.abs(e.lr) / maxAbs * 96;
      bars += `<div class="lab">${e.endorsed ? "" : "· "}${esc(e.lm)}</div>
        <div style="display:flex;align-items:center;gap:6px;justify-content:flex-end">
        <span class="bar" style="width:${w.toFixed(1)}px;background:${
        e.endorsed ? "var(--water)" : "var(--ink-faint)"}"></span>
        <span class="mono" style="min-width:44px;text-align:right">${
        e.lr >= 0 ? "+" : ""}${e.lr.toFixed(2)}</span></div>`;
    }
    matcher += `<div class="wf">${bars}</div>
      <dl class="kv" style="margin-top:8px">
      <dt>status</dt><dd>${esc(tbl.status)}</dd>
      <dt>entries</dt><dd>${tbl.nEntries} (${tbl.nEndorsed} endorsed)</dd>
      <dt>tied at top</dt><dd>${tbl.nTied}${tbl.nTied > 1
        ? " — a disjunction, not an identity" : ""}</dd>
      <dt>default</dt><dd>${tbl.default.toFixed(2)}</dd>
      <dt>clip</dt><dd>[${tbl.clipLo}, ${tbl.clipHi}]</dd></dl>`;
    if (tbl.nEntries > tbl.entries.length)
      matcher += `<div class="legend">showing the top ${tbl.entries.length}
        of ${tbl.nEntries} entries.</div>`;
  } else {
    matcher += `<div class="empty">no compatibility table for this tracklet
      — the filter had no semantic evidence at all, only geometry.</div>`;
  }

  // --- filter column ---
  let filter = `<h3>Filter — attribution &amp; belief</h3>`;
  if (trk.attribution)
    filter += seriesChart(trk.attribution, "kf", "nats",
                          {bars: true, zero: true})
      + `<div class="legend">Per-epoch contribution to the whole belief's
      log-likelihood (§7.2), total <b>${
      trk.attributionTotal >= 0 ? "+" : ""}${trk.attributionTotal}</b> nats.
      </div>`;
  else
    filter += `<div class="empty">needs the Tier-3 attribution cache
      (<code>forensics attribute</code>)</div>`;
  // Association evolution: which identity the posterior favoured over time.
  const whole = trk.assoc.filter(a => a.mode === null);
  if (whole.length) {
    const names = new Set();
    whole.forEach(a => Object.keys(a.resp).forEach(k => names.add(k)));
    const list = [...names].slice(0, 5);
    const W = 300, Hh = 62;
    let paths = "";
    list.forEach((lm, k) => {
      let d = "";
      whole.forEach((a, i) => {
        const v = a.resp[lm] || 0;
        d += (i ? "L" : "M") + (a.kf / KF * W).toFixed(1) + " "
          + (Hh - 10 - v * (Hh - 18)).toFixed(1); });
      paths += `<path d="${d}" fill="none" stroke="${
        D.colors[k % D.colors.length]}" stroke-width="1.3"><title>${
        esc(lm)}</title></path>`;
    });
    let dn = "";
    whole.forEach((a, i) => {
      dn += (i ? "L" : "M") + (a.kf / KF * W).toFixed(1) + " "
        + (Hh - 10 - a.null * (Hh - 18)).toFixed(1); });
    filter += `<h3>Association posterior over time</h3>
      <svg viewBox="0 0 ${W} ${Hh}">${paths}
      <path d="${dn}" fill="none" stroke="var(--ink-faint)" stroke-width="1.2"
        stroke-dasharray="3 2"><title>null</title></path>
      <line x1="${(t / KF * W).toFixed(1)}" y1="0"
        x2="${(t / KF * W).toFixed(1)}" y2="${Hh - 10}" stroke="var(--accent)"
        stroke-width="1" opacity=".8"/>
      <text class="axis" x="0" y="${Hh - 1}">0</text>
      <text class="axis" x="0" y="9">1</text></svg>
      <div class="legend">${list.map((lm, k) =>
        `<span style="color:${D.colors[k % D.colors.length]}">▬</span> ${
        esc(lm)}`).join(" ")}
        <span style="color:var(--ink-faint)">╌ null</span></div>`;
  }

  // --- truth-privileged band ---
  let priv = "";
  const tri = trk.triage;
  if (tri) {
    const fitLine = (label, fit) => fit
      ? `<dt>${label}</dt><dd><code>${esc(fit.lm)}</code> — RMS ${
        fmt(fit.rms, 2)}°, worst ${fmt(fit.max, 1)}°, ${
        (fit.rangeM / 1000).toFixed(1)} km${fit.lr !== null
          ? `, LLR ${fit.lr >= 0 ? "+" : ""}${fit.lr.toFixed(2)}` : ""}${
        fit.rms !== null && fit.rms <= tri.toleranceDeg
          ? ' <span class="pill ok">explains it</span>'
          : ' <span class="pill bad">does not explain it</span>'}</dd>`
      : `<dt>${label}</dt><dd>—</dd>`;
    const rows = tri.epochs.map(e => `<tr>
      <td class="num">${e.kf}</td>
      <td class="num">${e.bearing.toFixed(1)}</td>
      <td class="num">±${e.sigma.toFixed(1)}</td>
      <td class="num">${e.worldBearing.toFixed(0)}</td>
      <td class="num">${fmt(e.bestRes, 2)}</td>
      <td class="num">${fmt(e.topRes, 1)}</td>
      <td><code>${esc(e.filterTop || "—")}</code>
        <span class="prov">${(e.filterTopShare * 100).toFixed(0)}%</span></td>
      <td class="num">${(e.bestFitShare * 100).toFixed(0)}%</td>
      <td class="num">${(e.null * 100).toFixed(0)}%</td>
      <td class="num">${(e.surprise * 100).toFixed(0)}%</td></tr>`).join("");
    priv = `<div class="privileged">
      <div class="tag">⚠ Truth-privileged — uses GPS ground truth;
        a debugging instrument, never evidence of localization performance</div>
      <div style="margin:7px 0 6px"><span class="pill ${
        VERDICT_PILL[tri.verdict] || "info"}">${tri.verdict}</span>
        ${tri.antiEvidence ? '<span class="pill bad">anti-evidence</span>' : ""}
        ${tri.ambiguous ? '<span class="pill warn">geometrically ambiguous</span>'
          : ""}</div>
      <dl class="kv">
      <dt>tolerance</dt><dd>${tri.toleranceDeg}° (3× this tracklet's median
        σ of ${tri.medianSigmaDeg}°, clamped)</dd>
      ${fitLine("best in catalog", tri.bestFit)}
      ${fitLine("best endorsed", tri.bestEndorsed)}
      ${fitLine("matcher's top claim", tri.topEndorsed)}
      <dt>catalog rows that fit</dt><dd>${tri.nConsistent}${tri.ambiguous
        ? " — too many to identify anything, so this verdict is a weak claim"
        : " — the geometry discriminates"}</dd>
      <dt>endorsed entries</dt><dd>${tri.nEndorsed} of ${tri.nTableEntries}</dd>
      <dt>max mass on a fit</dt><dd>${(tri.bestFilterShare * 100).toFixed(0)}%</dd>
      </dl>
      <div class="scroll" style="margin-top:8px"><table><thead><tr>
        <th class="num">kf</th><th class="num">bearing</th><th class="num">σ</th>
        <th class="num">world</th><th class="num">best res°</th>
        <th class="num">top res°</th><th>filter believes</th>
        <th class="num">mass on fit</th><th class="num">null</th>
        <th class="num">surprise</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      <div class="legend">"best res" is the angular residual of the
      best-fitting catalog landmark at that epoch; "top res" is the residual of
      the matcher's highest-scoring claim. A tracklet is only identifiable
      because one landmark has to explain <em>every</em> epoch — a single
      bearing in a 13,210-row catalog is satisfied by dozens of rows.</div>
      </div>`;
  } else {
    priv = `<div class="empty">no ground truth in this run, so culpability
      cannot be assigned automatically.</div>`;
  }

  $("inspector").innerHTML = `<h2>Tracklet <code>${esc(trk.id)}</code>
    <span class="hint">&mdash; §7.4 view 3</span>
    <button id="closetrk">close</button></h2>
    <div class="grid2"><div>${source}</div><div>${bearings}</div>
    <div>${matcher}</div>
    <div>${filter}</div></div>${priv}`;
  $("closetrk").onclick = () => { selTrk = null; render(); };
}

// ---------- view 5: what-if console ----------
function drawWhatIf() {
  const rows = D.ghosts.map(g => `<tr>
    <td>${esc(g.label)}</td>
    <td class="num">${fmt(g.finalErr)}</td>
    <td class="num">${fmt(g.medianErr)}</td>
    <td class="num">${g.nModes}</td></tr>`).join("");
  const last = H[H.length - 1];
  const baseMedian = (() => {
    const errs = H.map(h => h.mapErr).filter(v => v !== undefined);
    if (!errs.length) return undefined;
    const s = [...errs].sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)];
  })();
  const cmd = `bazel run //experimental/overhead_matching/swag/`
    + `farfield/localization:forensics -- replay \\\n  --run_dir ${RUN.runDir} \\\n`
    + `  --without_tracklet ${selTrk || "<TRACKLET>"}\n\n`
    + `# the ghost is written under ${RUN.runDir}/counterfactuals/;\n`
    + `# then re-render with it overlaid:\n`
    + `bazel run //experimental/overhead_matching/swag/farfield/localization:viewer -- \\\n`
    + `  --run_dir ${RUN.runDir} --ghost <the printed counterfactual dir>`;
  $("whatif").innerHTML = `<div class="scroll"><table><thead><tr>
    <th>counterfactual</th><th class="num">final err (m)</th>
    <th class="num">median err (m)</th><th class="num">modes</th></tr></thead>
    <tbody><tr><td><b>baseline</b> (this run)</td>
      <td class="num">${fmt(last.mapErr)}</td>
      <td class="num">${fmt(baseMedian)}</td>
      <td class="num">${last.modes.length}</td></tr>
    ${rows || `<tr><td colspan="4" class="empty">no counterfactuals loaded
      — pass --ghost &lt;run_dir&gt;</td></tr>`}</tbody></table></div>
    <div class="legend">A counterfactual is a full replay from keyframe 0 with
    one input changed, written as its own run directory, so it can be opened,
    diffed and archived like any other run. On this run a replay takes about
    ${RUN.nCatalog > 5000 ? "30 s on the GPU backend" : "a couple of minutes"}.
    </div>
    <h3>Run one</h3>
    <pre class="mono" style="white-space:pre-wrap;font-size:11px;
      background:var(--sunk);padding:9px;border-radius:6px;overflow-x:auto">${
    esc(cmd)}</pre>`;
  if (LIVE.features.has("replay")) {
    const controls = document.createElement("div");
    controls.innerHTML =
      '<h3>Live replay</h3><div class="controls">'
      + '<button id="liveReplay">drop selected tracklet and replay</button>'
      + '<span id="liveReplayStatus" class="legend"></span></div>'
      + '<div class="legend">Select a tracklet in the inspector, then run an '
      + 'exact counterfactual through the localhost server.</div>';
    $("whatif").appendChild(controls);
    const button = $("liveReplay");
    const status = $("liveReplayStatus");
    button.disabled = !selTrk || !RUN.replayable;
    if (!selTrk)
      status.textContent = "Select a tracklet first.";
    else if (!RUN.replayable)
      status.textContent = "This baseline is not faithfully replayable.";
    button.onclick = async () => {
      const tracklet = selTrk;
      button.disabled = true;
      status.textContent = "replaying from keyframe 0…";
      try {
        const result = await runLiveReplay({drop_tracklets: [tracklet]});
        if (result.ghost) {
          D.ghosts.push(result.ghost);
          showGhosts = true;
        }
        status.textContent = "complete in " + result.elapsed_s + " s";
        render();
      } catch (error) {
        status.textContent = "replay failed: " + error.message;
        button.disabled = false;
      }
    };
  }
}

// ---------- event index ----------
let evFilter = null;
function drawEvents() {
  const kinds = [...new Set(D.events.map(e => e.kind))];
  const chips = kinds.map(k => `<button class="tab ${evFilter === k ? "on" : ""}"
    data-kind="${k}">${k} (${D.events.filter(e => e.kind === k).length})</button>`
    ).join("");
  const shown = D.events.filter(e => !evFilter || e.kind === evFilter);
  const rows = shown.map(e => `<tr class="click" data-kf="${e.keyframe_idx}">
    <td class="num">${e.keyframe_idx}</td>
    <td><span class="pill ${e.severity === "alarm" ? "bad"
      : e.severity === "warn" ? "warn" : "info"}">${e.kind}</span></td>
    <td>${esc(e.label)}</td><td class="prov">${esc(e.detail)}</td>
    <td class="prov">${e.source}</td></tr>`).join("");
  $("events").innerHTML = `<div class="tabs">
    <button class="tab ${evFilter === null ? "on" : ""}" data-kind="">all (${
    D.events.length})</button>${chips}</div>
    <div class="scroll scrollY"><table><thead><tr><th class="num">kf</th>
    <th>kind</th><th>what</th><th>detail</th><th>source</th></tr></thead>
    <tbody>${rows || `<tr><td colspan="5" class="empty">no events</td></tr>`}
    </tbody></table></div>
    <div class="legend"><b>logged</b> events were emitted by the filter as it
    ran. <b>derived</b> events were reconstructed from the Tier-0 stream
    afterwards — real findings, but the filter did not react to them at the
    time (§7.3 asks for these to be detected online and checkpointed;
    they are not yet).</div>`;
  $("events").querySelectorAll(".tab").forEach(b => b.onclick = () => {
    evFilter = b.dataset.kind || null; drawEvents(); });
  $("events").querySelectorAll("tr.click").forEach(r => r.onclick = () => {
    t = +r.dataset.kf; render(); });
}

// ---------- attribution tab ----------
function drawAttribution() {
  if (!D.attribution) {
    $("attr").innerHTML = `<div class="empty">No Tier-3 attribution cache.
      Build it with <code>forensics attribute --run_dir ...</code>; it replays the
      run under instrumentation and caches the whole decomposition (about 45 KB
      for a 379-keyframe run), after which every waterfall here is a lookup.
      </div>`;
    return;
  }
  const a = D.attribution;
  let out = `<div class="prov">${a.nRows.toLocaleString()} contribution rows`
    + (a.verified ? `, verified against the run's recorded history hash`
       : `, <b style="color:var(--port)">not verified against the run</b>`)
    + `.</div>`;
  const ids = Object.keys(a.modes).sort((x, y) => +x - +y);
  for (const id of ids) {
    if (selMode !== null && +id !== selMode) continue;
    out += `<h3><span class="swatch" style="background:${color(+id)}"></span>
      mode ${id} over the whole run</h3>` + waterfallHtml(a.modes[id]);
  }
  if (selMode !== null && !a.modes[String(selMode)])
    out += `<div class="empty">no attribution rows for mode ${selMode}</div>`;
  out += `<div class="legend">A <b>tracklet</b> term is evidence: a bearing
    changed this mode's share of the belief. The starred <b>re-clustering</b>,
    <b>proposal injection</b> and <b>resampling</b> terms are belief
    bookkeeping — a mode whose rise is mostly structural has not been
    confirmed by anything the vehicle saw. §7.2 lists only the tracklet and
    resample terms; the others are needed for the decomposition to add up,
    because a mode is a tracked cluster rather than a fixed set of particles.
    </div>`;
  $("attr").innerHTML = out;
}

// ---------- wiring ----------
const TABS = {state: drawState, modes: drawModes, attr: drawAttribution,
              trklist: drawTrackletList, whatif: drawWhatIf};
function drawTab() {
  for (const key of Object.keys(TABS)) {
    const el = $(key);
    el.parentElement.style.display = key === tab ? "" : "none";
  }
  TABS[tab]();
  document.querySelectorAll("#tabbar .tab").forEach(b =>
    b.classList.toggle("on", b.dataset.tab === tab));
}
function render() {
  $("kf").textContent = t; $("slider").value = t;
  drawTiles(); drawStrip(); drawMap(); drawTab(); drawInspector();
  // Association rows are a way into the inspector from wherever you are.
  document.querySelectorAll("[data-trk]").forEach(el => {
    if (el.tagName === "TR" && el.closest("#state"))
      el.onclick = () => { selTrk = el.dataset.trk; render(); };
  });
}
$("slider").max = KF;
$("slider").oninput = e => { t = +e.target.value; render(); };
$("strip").onclick = e => {
  // Map the click through the same gutter geometry the plots use, so clicking
  // under a point selects that point rather than one shifted by the gutter.
  const r = $("strip").getBoundingClientRect();
  const frac = ((e.clientX - r.left) / r.width * SW - GL) / PW;
  t = Math.max(0, Math.min(KF, Math.round(frac * KF)));
  render();
};
$("clear").onclick = () => { selMode = null; selTrk = null; render(); };
document.querySelectorAll("#tabbar .tab").forEach(b =>
  b.onclick = () => { tab = b.dataset.tab; drawTab(); });
$("tgBase").onclick = e => { showBase = !showBase;
  e.target.classList.toggle("on", showBase); drawMap(); };
$("tgPart").onclick = e => { showParticles = !showParticles;
  e.target.classList.toggle("on", showParticles); drawMap(); };
$("tgGhost").onclick = e => { showGhosts = !showGhosts;
  e.target.classList.toggle("on", showGhosts); drawMap(); };
if ($("tgSat")) {
  if (!SATELLITE) $("tgSat").disabled = true;
  $("tgSat").onclick = e => { showSat = !showSat;
    e.target.classList.toggle("on", showSat); drawMap(); };
}
wireMapZoom();
// Open on the trajectory rather than the whole box: the 25 km extent is context
// you ask for, not what you want to read first.
viewFitTrack();
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const step = e.shiftKey ? 10 : 1;
  if (e.key === "ArrowRight") { t = Math.min(KF, t + step); render(); }
  if (e.key === "ArrowLeft") { t = Math.max(0, t - step); render(); }
});
let timer = null;
$("play").onclick = () => {
  if (timer) { clearInterval(timer); timer = null; $("play").textContent = "▶ play";
    return; }
  $("play").textContent = "⏸ pause";
  timer = setInterval(() => { t = (t + 1) % (KF + 1); render(); }, 80);
};
drawEvents();
render();
detectLiveServer();
