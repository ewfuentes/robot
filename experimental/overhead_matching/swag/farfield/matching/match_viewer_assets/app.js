const M = MAP_DATA;
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const tip = document.getElementById('tip');
let view = {cx:0, cy:0, scale:1};
let sel = null, hits = [], showAll = true, showCtx = true;

const LAYER_STYLE = {
  water:{fill:'#152230'}, land:{fill:'#1b2119'}, wetland:{fill:'#182018'},
  buildings:{fill:'#262a30'}, piers:{line:'#3d4a58', w:1.6},
  breakwaters:{line:'#465360', w:1.6}, bridges:{line:'#4a5666', w:1.4},
  coastline:{line:'#3f5163', w:1.3}, roads:{line:'#2b333c', w:1},
  railways:{line:'#333b44', w:1}
};

// Residual suffix for a marker's readout; null where the tracklet has no
// bearing to check against.
function dtxt(g){
  return g[7] == null ? '' : ', signed ray residual '
    + (g[7] >= 0 ? '+' : '') + g[7] + ' deg';
}

function esc(t){
  return String(t).replace(/[&<>"]/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function trackletLabel(key){
  const tk = M.tracklets[key];
  if(tk && tk.label) return tk.label;
  const text = String(key), split = text.lastIndexOf('#');
  return split >= 0 ? text.slice(split + 1) : text;
}

function canonicalTracklet(label){
  if(M.tracklets[label]) return label;
  return Object.keys(M.tracklets).find(key => trackletLabel(key) === label)
    || label;
}

// Inverse of the equirectangular ENU the whole page is drawn in; the scale
// factors come from the payload so this matches geometry.RegionFrame exactly
// rather than re-deriving a projection in the browser.
function latlon(e, n){
  return [M.enu.lat0 + n * M.enu.dlat_per_m,
          M.enu.lon0 + e * M.enu.dlon_per_m];
}
// Slippy-map zoom whose ground resolution matches the canvas', so opening OSM
// lands at the scale you were already looking at.
function zoomFor(lat){
  const z = Math.log2(156543.03 * Math.cos(lat * Math.PI / 180) * view.scale);
  return Math.max(3, Math.min(19, Math.round(z)));
}
function osmHref(lat, lon, z){
  return 'https://www.openstreetmap.org/?mlat=' + lat.toFixed(6)
    + '&mlon=' + lon.toFixed(6) + '#map=' + z + '/' + lat.toFixed(6)
    + '/' + lon.toFixed(6);
}
function gmHref(lat, lon){
  return 'https://www.google.com/maps?q=' + lat.toFixed(6) + ','
    + lon.toFixed(6);
}
function osmObject(lid){
  const p = String(lid).split(':');
  if(p.length !== 3 || p[0] !== 'osm') return null;
  if(['node','way','relation'].indexOf(p[1]) < 0) return null;
  return 'https://www.openstreetmap.org/' + p[1] + '/' + p[2];
}

// The link row follows the selection: with a tracklet chosen it points at the
// vessel's true position when that observation was made, which is the
// coordinate you want to go look at; otherwise at wherever the map is centred.
function llUpdate(){
  const tk = sel && M.tracklets[sel];
  let e, n, what;
  if(tk && tk.rays.length){
    e = tk.rays[0][1]; n = tk.rays[0][2];
    what = 'robot at keyframe ' + tk.rays[0][0] + ' ('
      + trackletLabel(sel) + ')';
  } else {
    e = view.cx; n = view.cy; what = 'view centre';
  }
  const ll = latlon(e, n), z = zoomFor(ll[0]);
  document.getElementById('ll-what').textContent = what;
  document.getElementById('ll-osm').href = osmHref(ll[0], ll[1], z);
  document.getElementById('ll-gm').href = gmHref(ll[0], ll[1]);
  document.getElementById('ll-coord').textContent =
    ll[0].toFixed(5) + ', ' + ll[1].toFixed(5) + '  z' + z;
}

// Readout for a clicked marker: what it is, plus a way out to the map it came
// from. Catalog rows are OSM objects, so the id is a link.
function showTip(hit){
  let html = esc(hit.label);
  const obj = hit.lid ? osmObject(hit.lid) : null;
  if(obj) html += " &middot; <a href='" + obj + "' target='_blank' "
    + "rel='noopener'>open " + esc(hit.lid) + " in OSM</a>";
  if(hit.e != null){
    const ll = latlon(hit.e, hit.n), z = zoomFor(ll[0]);
    html += " &middot; <a href='" + osmHref(ll[0], ll[1], z) + "' "
      + "target='_blank' rel='noopener'>OSM here</a>"
      + " &middot; <a href='" + gmHref(ll[0], ll[1]) + "' target='_blank' "
      + "rel='noopener'>Google Maps</a>"
      + " <span class='pin'>" + ll[0].toFixed(5) + ', '
      + ll[1].toFixed(5) + "</span>";
  }
  tip.innerHTML = html;
}

function resize(){
  const r = cv.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.max(1, Math.round(r.width * dpr));
  cv.height = Math.max(1, Math.round(r.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
function size(){
  const dpr = window.devicePixelRatio || 1;
  return {w: cv.width / dpr, h: cv.height / dpr};
}
function sx(e){ return (e - view.cx) * view.scale + size().w / 2; }
function sy(n){ return size().h / 2 - (n - view.cy) * view.scale; }
function fit(emin, emax, nmin, nmax){
  const s = size();
  const w = Math.max(emax - emin, 40), h = Math.max(nmax - nmin, 40);
  view.scale = Math.min(s.w / w, s.h / h) * 0.9;
  view.cx = (emin + emax) / 2; view.cy = (nmin + nmax) / 2;
  draw();
}
function fitAll(){ fit(M.bounds[0], M.bounds[1], M.bounds[2], M.bounds[3]); }
function fitTruth(){
  const t = M.truth; let a=1e18,b=-1e18,c=1e18,d=-1e18;
  for(let i=0;i<t.length;i+=2){
    a=Math.min(a,t[i]); b=Math.max(b,t[i]);
    c=Math.min(c,t[i+1]); d=Math.max(d,t[i+1]);
  }
  const pad = Math.max(200, (b-a)*0.15);
  fit(a-pad, b+pad, c-pad, d+pad);
}
function fitSel(){
  const t = M.tracklets[sel]; if(!t) return fitAll();
  let a=1e18,b=-1e18,c=1e18,d=-1e18;
  const add = (e,n)=>{a=Math.min(a,e);b=Math.max(b,e);
                      c=Math.min(c,n);d=Math.max(d,n);};
  t.rays.forEach(r=>{
    add(r[1], r[2]);
    const rad = r[3] * Math.PI / 180;
    add(r[1] + Math.sin(rad) * t.ray_m, r[2] + Math.cos(rad) * t.ray_m);
  });
  t.targets.forEach(g=>{
    add(g[0],g[1]);
    const hull = g[10] || [];
    for(let i=0;i<hull.length;i+=2) add(hull[i],hull[i+1]);
  });
  if(a>b) return fitAll();
  const pad = Math.max(250, Math.max(b-a, d-c) * 0.18);
  fit(a-pad, b+pad, c-pad, d+pad);
}

function draw(){
  const s = size();
  ctx.clearRect(0, 0, s.w, s.h);
  hits = [];
  // Basemap, in payload order: fills first, then the lines that give a
  // harbour its edges.
  (M.basemap.layers || []).forEach(layer => {
    const style = LAYER_STYLE[layer.name] || {line:'#2a323a', w:1};
    ctx.beginPath();
    layer.paths.forEach(p => {
      for(let i=0;i<p.length;i+=2){
        const x = sx(p[i]), y = sy(p[i+1]);
        if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }
      if(layer.kind === 'polygon') ctx.closePath();
    });
    if(layer.kind === 'polygon' && style.fill){
      ctx.fillStyle = style.fill; ctx.fill();
    } else {
      ctx.strokeStyle = style.line || '#2a323a';
      ctx.lineWidth = style.w || 1; ctx.stroke();
    }
  });

  if(showCtx && view.scale > 0.0015){
    ctx.fillStyle = '#3d454e';
    const C = M.context;
    for(let i=0;i<C.e.length;i++){
      const x = sx(C.e[i]), y = sy(C.n[i]);
      if(x<-10||y<-10||x>s.w+10||y>s.h+10) continue;
      ctx.fillRect(x-1, y-1, 2, 2);
      hits.push({x, y, r:5, kind:'ctx', label:C.l[i] || 'unnamed catalog row',
                 lid:C.i[i], e:C.e[i], n:C.n[i]});
    }
  }

  // Truth track.
  const t = M.truth;
  ctx.beginPath();
  for(let i=0;i<t.length;i+=2){
    const x = sx(t[i]), y = sy(t[i+1]);
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.strokeStyle = '#e6ecf2'; ctx.lineWidth = 2; ctx.stroke();
  ctx.strokeStyle = '#93a7b8'; ctx.lineWidth = 1;
  M.ticks.forEach(k => arrow(k[0], k[1], k[2], 9));
  let ta=1e18, tb=-1e18, tc=1e18, td=-1e18;
  for(let i=0;i<t.length;i+=2){
    const x = sx(t[i]), y = sy(t[i+1]);
    ta=Math.min(ta,x); tb=Math.max(tb,x); tc=Math.min(tc,y); td=Math.max(td,y);
  }
  // Ring the track whenever it is small against the whole view. The default
  // extent is set by how far the matches reach, which on a run whose matches
  // are 20 km away leaves the vessel a thin squiggle among thousands of
  // catalog dots -- and where the vessel actually went is the one thing the
  // reader must not have to hunt for.
  const span = Math.max(tb-ta, td-tc);
  if(span < s.w * 0.22){
    const cx = (ta+tb)/2, cy = (tc+td)/2;
    ctx.beginPath(); ctx.arc(cx, cy, Math.max(14, span/2 + 10), 0, 7);
    ctx.strokeStyle = '#e6ecf2'; ctx.lineWidth = 1.4; ctx.stroke();
    ctx.fillStyle = '#e6ecf2'; ctx.font = '12px sans-serif';
    ctx.fillText('vessel track',
                 cx + Math.max(14, span/2 + 10) + 5, cy + 4);
  }

  // Every match at once, so the overview shows where the matcher is pointing
  // before you pick a tracklet.
  if(showAll && !sel){
    Object.keys(M.tracklets).forEach(k => {
      M.tracklets[k].targets.forEach(g => {
        const x = sx(g[0]), y = sy(g[1]);
        ctx.fillStyle = g[3] === 'instance'
          ? 'rgba(62,207,142,.5)' : 'rgba(111,155,255,.38)';
        ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill();
        hits.push({x, y, r:7, kind:'target', tracklet:k, label:
          trackletLabel(k) + ' -> ' + (g[5] || g[4])
          + '  aggregate confidence '
          + g[2] + ', ' + g[3]
          + ', ' + g[6] + ' map rows' + dtxt(g), lid:g[4],
          e:g[0], n:g[1]});
      });
    });
  }

  if(sel && M.tracklets[sel]){
    const tk = M.tracklets[sel];
    // Rays, then poses, then targets on top.
    tk.rays.forEach(r => {
      const rad = r[3] * Math.PI / 180;
      const ex = r[1] + Math.sin(rad) * tk.ray_m;
      const en = r[2] + Math.cos(rad) * tk.ray_m;
      ctx.beginPath();
      ctx.moveTo(sx(r[1]), sy(r[2])); ctx.lineTo(sx(ex), sy(en));
      ctx.strokeStyle = 'rgba(255,179,71,.75)'; ctx.lineWidth = 1.4;
      ctx.stroke();
    });
    tk.rays.forEach(r => {
      const x = sx(r[1]), y = sy(r[2]);
      ctx.fillStyle = '#fff';
      ctx.beginPath(); ctx.arc(x, y, 3.5, 0, 7); ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.6;
      arrow(r[1], r[2], r[4], 16);
      hits.push({x, y, r:8, kind:'pose', tracklet:sel, label:
        'true pose at keyframe ' + r[0] + ': GPS course ' + r[4]
        + ' deg, bearing ' + r[3] + ' deg world, kappa ' + r[5],
        e:r[1], n:r[2]});
    });
    tk.targets.forEach(g => {
      const hull = g[10] || [];
      if(hull.length >= 4){
        ctx.beginPath();
        for(let i=0;i<hull.length;i+=2){
          const hx=sx(hull[i]), hy=sy(hull[i+1]);
          if(i===0) ctx.moveTo(hx,hy); else ctx.lineTo(hx,hy);
        }
        if(hull.length >= 6) ctx.closePath();
        ctx.fillStyle = g[3] === 'instance'
          ? 'rgba(62,207,142,.10)' : 'rgba(111,155,255,.08)';
        ctx.fill();
        ctx.strokeStyle = g[3] === 'instance' ? '#3ecf8e' : '#6f9bff';
        ctx.lineWidth = 1.2; ctx.stroke();
      }
      const x = sx(g[0]), y = sy(g[1]);
      ctx.beginPath(); ctx.arc(x, y, 6, 0, 7);
      ctx.fillStyle = g[3] === 'instance' ? '#3ecf8e' : '#6f9bff';
      ctx.fill();
      ctx.strokeStyle = '#0e1319'; ctx.lineWidth = 2; ctx.stroke();
      if(g[8] != null && g[8] > 45){
        ctx.beginPath(); ctx.arc(x, y, 10.5, 0, 7);
        ctx.strokeStyle = '#f2777a'; ctx.lineWidth = 1.6; ctx.stroke();
      }
      if(g[5]){
        ctx.fillStyle = '#dfe8f0'; ctx.font = '12px sans-serif';
        ctx.fillText(g[5], x + 9, y + 4);
      }
      hits.push({x, y, r:9, kind:'target', tracklet:sel, label:
        (g[5] || g[4]) + '  aggregate confidence ' + g[2] + ', '
        + g[3] + ', ' + g[6]
        + ' map rows' + dtxt(g), lid:g[4], e:g[0], n:g[1]});
    });
  }
  scalebar();
  llUpdate();
}

function arrow(e, n, deg, px){
  const rad = deg * Math.PI / 180;
  const x = sx(e), y = sy(n);
  const dx = Math.sin(rad) * px, dy = -Math.cos(rad) * px;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + dx, y + dy); ctx.stroke();
  const a = Math.atan2(dy, dx);
  ctx.beginPath();
  ctx.moveTo(x + dx, y + dy);
  ctx.lineTo(x + dx - 5 * Math.cos(a - 0.4), y + dy - 5 * Math.sin(a - 0.4));
  ctx.moveTo(x + dx, y + dy);
  ctx.lineTo(x + dx - 5 * Math.cos(a + 0.4), y + dy - 5 * Math.sin(a + 0.4));
  ctx.stroke();
}

function scalebar(){
  const s = size();
  const target = 120 / view.scale;
  const pow = Math.pow(10, Math.floor(Math.log10(target)));
  const nice = [1,2,5,10].map(v=>v*pow).find(v=>v>=target*0.6) || pow;
  const px = nice * view.scale;
  const y = s.h - 14, x = 14;
  ctx.strokeStyle = '#8fa3b5'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x, y-4); ctx.lineTo(x, y); ctx.lineTo(x+px, y);
  ctx.lineTo(x+px, y-4); ctx.stroke();
  ctx.fillStyle = '#8fa3b5'; ctx.font = '11px sans-serif';
  ctx.fillText(nice >= 1000 ? (nice/1000)+' km' : nice+' m', x + px + 6, y);
}

function select(key, andFit){
  sel = key;
  if(window.selectMatchNote) window.selectMatchNote(key);
  document.querySelectorAll('.card').forEach(c =>
    c.classList.toggle('sel', c.dataset.key === key));
  const tk = M.tracklets[key];
  if(tk){
    let msg = trackletLabel(key) + ': ' + tk.rays.length + ' bearing'
      + (tk.rays.length===1?'':'s') + ', ' + tk.n_shown + ' of '
      + tk.n_resolved + ' placed map rows drawn';
    if(tk.n_rows > tk.n_resolved)
      msg += ' (' + tk.n_rows + ' rows before dedup/placement)';
    tip.textContent = msg;
  } else {
    tip.textContent = trackletLabel(key)
      + ': nothing to draw (no bearings and no placed match)';
  }
  if(andFit) fitSel(); else draw();
}

let drag = null;
cv.addEventListener('pointerdown', ev => {
  drag = {x:ev.clientX, y:ev.clientY, cx:view.cx, cy:view.cy, moved:false};
  cv.classList.add('drag'); cv.setPointerCapture(ev.pointerId);
});
cv.addEventListener('pointermove', ev => {
  const r = cv.getBoundingClientRect();
  if(drag){
    const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
    if(Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    view.cx = drag.cx - dx / view.scale;
    view.cy = drag.cy + dy / view.scale;
    draw();
    return;
  }
  const hit = pick(ev.clientX - r.left, ev.clientY - r.top);
  cv.style.cursor = hit ? 'pointer' : 'grab';
  if(hit) showTip(hit);
});
cv.addEventListener('pointerup', ev => {
  const r = cv.getBoundingClientRect();
  const wasDrag = drag && drag.moved;
  drag = null; cv.classList.remove('drag');
  if(wasDrag) return;
  const hit = pick(ev.clientX - r.left, ev.clientY - r.top);
  if(!hit) return;
  showTip(hit);
  if(hit.tracklet){
    select(hit.tracklet, false);
    const card = document.querySelector('[data-key="' + hit.tracklet + '"]');
    if(card) card.scrollIntoView({block:'center', behavior:'smooth'});
  }
});
cv.addEventListener('wheel', ev => {
  ev.preventDefault();
  const r = cv.getBoundingClientRect();
  const mx = ev.clientX - r.left, my = ev.clientY - r.top;
  const s = size();
  const we = view.cx + (mx - s.w/2) / view.scale;
  const wn = view.cy - (my - s.h/2) / view.scale;
  const k = Math.exp(-ev.deltaY * 0.0015);
  view.scale = Math.min(4, Math.max(2e-5, view.scale * k));
  view.cx = we - (mx - s.w/2) / view.scale;
  view.cy = wn + (my - s.h/2) / view.scale;
  draw();
}, {passive:false});

function pick(x, y){
  let best = null, bestD = 1e9;
  for(const h of hits){
    const d = (h.x-x)*(h.x-x) + (h.y-y)*(h.y-y);
    if(d < h.r*h.r && d < bestD){ best = h; bestD = d; }
  }
  return best;
}

document.querySelectorAll('.card h2').forEach(h => {
  h.addEventListener('click', () => select(h.parentElement.dataset.key, true));
});
document.querySelectorAll('[data-jump]').forEach(a => {
  a.addEventListener('click', ev => {
    ev.preventDefault(); select(a.dataset.jump, true);
    cv.scrollIntoView({block:'nearest'});
  });
});
document.getElementById('b-all').onclick = () => { sel=null; fitAll();
  tip.textContent = 'all matches, ' + Object.keys(M.tracklets).length
    + ' tracklets drawn'; };
document.getElementById('b-truth').onclick = () => fitTruth();
document.getElementById('b-sel').onclick = () => fitSel();
document.getElementById('b-ctx').onclick = ev => {
  showCtx = !showCtx;
  ev.currentTarget.setAttribute('aria-pressed', showCtx); draw();
};
new ResizeObserver(resize).observe(cv);
resize(); fitAll();
if(location.hash)
  select(canonicalTracklet(decodeURIComponent(location.hash.slice(1))), true);
