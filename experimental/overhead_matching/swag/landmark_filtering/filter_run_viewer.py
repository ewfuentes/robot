"""Flask viewer for landmark-filter run artifacts.

Renders exclusively from a RunArtifact JSON (plus the pinhole/panorama images
on disk): dispositions, reasons, and details are treated as opaque strings and
dicts grouped generically, so new filters need no viewer changes.

Example:
    bazel run //experimental/overhead_matching/swag/landmark_filtering:filter_run_viewer -- \\
        --artifact /tmp/filter_runs/walk_river_ingest.json --port 5006
"""

import argparse
import hashlib
import html
import json
from pathlib import Path

from flask import Flask, abort, request, send_file

from experimental.overhead_matching.swag.landmark_filtering import (
    artifact_schema as schema,
    bearing_geometry as bg,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    config_to_yaml,
)

FACES = (0, 90, 180, 270)

LEAFLET_HEAD = (
    '<link rel="stylesheet" '
    'href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>'
    '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>')

# Injected via str.replace("{map_json}", ...) - not str.format - so plain JS
# braces (and Leaflet's {z}/{x}/{y} URL template) stay literal.
MAP_SCRIPT = """<script>
const data = {map_json};
const map = L.map('map');
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
            {maxZoom: 19, attribution: 'OSM'}).addTo(map);
const traj = L.polyline(data.trajectory, {color: '#333', weight: 2})
    .addTo(map);
map.fitBounds(traj.getBounds().pad(0.3));
for (const f of data.frames) {
  L.circleMarker([f.lat, f.lon], {radius: 3, color: '#555', weight: 1})
    .bindPopup(`<a href="/frame/${f.i}">frame ${f.i}</a>`).addTo(map);
}
const colors = {near: '#2a2', far: '#26c', degenerate: '#999'};
const layers = {near: L.layerGroup(), far: L.layerGroup(),
                degenerate: L.layerGroup()};
for (const t of data.tracks) {
  const layer = layers[t.obs] || layers.degenerate;
  const popup = `<a href="/track/${t.id}">track #${t.id}</a> ` +
      `(${t.obs}, n=${t.n})<br>${t.label}`;
  for (const ray of t.rays) {
    L.polyline(ray, {color: colors[t.obs] || '#999', weight: 1,
                     opacity: 0.5, dashArray: t.obs === 'far' ? '4' : null})
      .addTo(layer);
  }
  if (t.lat !== null) {
    L.circleMarker([t.lat, t.lon],
                   {radius: 6, color: colors[t.obs] || '#999',
                    fillOpacity: 0.8})
      .bindPopup(popup).addTo(layer);
    if (t.sigma) {
      L.circle([t.lat, t.lon], {radius: t.sigma, color: colors[t.obs],
                                weight: 1, fill: false}).addTo(layer);
    }
  } else if (t.rays.length) {
    L.circleMarker(t.rays[0][1], {radius: 1, opacity: 0})
      .bindPopup(popup).addTo(layer);
  }
  layer.addTo(map);
}
const overlayColors = ['#c60', '#909', '#066', '#a33', '#361'];
(data.landmark_overlays || []).forEach((ov, i) => {
  const group = L.layerGroup();
  const color = overlayColors[i % overlayColors.length];
  for (const m of ov.points) {
    L.circleMarker([m.lat, m.lon],
                   {radius: 4, color: color, weight: 1.5, fillOpacity: 0.4})
      .bindPopup(m.label).addTo(group);
  }
  layers[`${ov.name} (${ov.points.length})`] = group;
  group.addTo(map);
});
L.control.layers(null, layers).addTo(map);
</script>"""

PAGE_SHELL = """<!DOCTYPE html>
<html><head><title>{title}</title>{head_extra}<style>
body {{ font-family: sans-serif; margin: 16px; background: #fafafa; }}
a {{ color: #06c; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
nav a {{ margin-right: 14px; font-weight: bold; }}
table {{ border-collapse: collapse; margin: 10px 0; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; font-size: 13px;
          text-align: left; vertical-align: top; }}
th {{ background: #eee; }}
.faces {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.face {{ position: relative; flex: 1; min-width: 320px; }}
.face img {{ width: 100%; display: block; }}
.face .label {{ position: absolute; top: 2px; left: 4px; color: #fff;
                background: rgba(0,0,0,0.55); padding: 1px 6px;
                font-size: 12px; z-index: 5; }}
.obsbox {{ position: absolute; border: 2px solid; box-sizing: border-box;
           z-index: 4; }}
.obsbox:hover {{ background: rgba(255,255,255,0.25); }}
.obsbox .tag {{ position: absolute; top: -1.4em; left: -2px; font-size: 11px;
                color: #fff; padding: 0 4px; white-space: nowrap; }}
.bar {{ background: #4a90d9; height: 14px; display: inline-block; }}
.crop {{ position: relative; overflow: hidden; background: #ddd; }}
.crop img {{ position: absolute; max-width: none; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 12px; }}
.card {{ border: 1px solid #ccc; background: #fff; padding: 8px;
         width: 260px; font-size: 12px; }}
.kept {{ color: #182; }} .filtered {{ color: #c22; }}
pre {{ background: #eee; padding: 8px; overflow-x: auto; font-size: 12px; }}
.pill {{ display: inline-block; padding: 0 6px; border-radius: 8px;
         background: #ddd; font-size: 11px; margin-right: 4px; }}
</style></head><body>
<nav><a href="/">Stats</a><a href="/frame/0">Frames</a>
<a href="/gallery">Gallery</a><a href="/tracks">Tracks</a>
<a href="/semantics">Semantics</a><a href="/map">Map</a></nav>
{body}
{script}
</body></html>"""

NAV_SCRIPT = """<script>
document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft' && {prev} >= 0) window.location = '/frame/{prev}';
  if (e.key === 'ArrowRight' && {next} < {n_frames})
    window.location = '/frame/{next}';
}});
</script>"""


def render(title: str, body: str, script: str = "",
           head_extra: str = "") -> str:
    return PAGE_SHELL.format(title=title, body=body, script=script,
                             head_extra=head_extra)


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def reason_color(reason: str) -> str:
    """Stable hue per reason string, red-ish family for filtered."""
    if not reason:
        return "#2a2"
    hue = int(hashlib.md5(reason.encode()).hexdigest(), 16) % 70
    return f"hsl({(hue + 330) % 360}, 85%, 45%)"


class ViewerState:
    def __init__(self, artifact: schema.RunArtifact, pinhole_base: Path,
                 dataset_base: Path):
        self.artifact = artifact
        self.pinhole_base = pinhole_base
        self.dataset_base = dataset_base
        self.obs_by_id = {o.obs_id: o for o in artifact.observations}
        self.obs_by_frame: dict[int, list[schema.Observation]] = {}
        for obs in artifact.observations:
            self.obs_by_frame.setdefault(obs.frame_idx, []).append(obs)
        self.track_by_id = {t.track_id: t for t in artifact.tracks}
        self.obs_by_reason: dict[str, list[schema.Observation]] = {}
        for obs in artifact.observations:
            if obs.final_disposition == schema.FILTERED:
                self.obs_by_reason.setdefault(obs.final_reason, []).append(obs)

    def frame(self, frame_idx: int) -> schema.Frame:
        if not 0 <= frame_idx < len(self.artifact.frames):
            abort(404)
        return self.artifact.frames[frame_idx]

    def face_image_path(self, frame_idx: int, face: int) -> Path:
        frame = self.frame(frame_idx)
        return self.pinhole_base / frame.pano_stem / f"yaw_{face:03d}.jpg"


def decision_trail_html(obs: schema.Observation) -> str:
    if not obs.decisions:
        return "<i>none</i>"
    parts = []
    for d in obs.decisions:
        detail = ", ".join(f"{k}={v:.2f}" for k, v in d.details.items())
        cls = "kept" if d.disposition == schema.KEPT else "filtered"
        label = d.reason if d.reason else "ok"
        parts.append(
            f'<span class="{cls}">{esc(d.filter_name)}: {esc(label)}'
            f'{" (" + esc(detail) + ")" if detail else ""}</span>')
    return "<br>".join(parts)


def obs_boxes_overlay_html(obs: schema.Observation, face: int) -> str:
    """Absolutely-positioned percentage divs for this obs's boxes on a face."""
    out = []
    color = reason_color(
        obs.final_reason if obs.final_disposition == schema.FILTERED else "")
    tooltip = (f"{obs.obs_id} | {obs.primary_tag_key}={obs.primary_tag_value}"
               f" | bearing {obs.bearing_camera_deg:.1f} | "
               f"{obs.final_disposition} {obs.final_reason}")
    for box in obs.boxes:
        if box.face_yaw_deg != face:
            continue
        left = box.xmin / 10.0
        top = box.ymin / 10.0
        width = (box.xmax - box.xmin) / 10.0
        height = (box.ymax - box.ymin) / 10.0
        out.append(
            f'<div class="obsbox" title="{esc(tooltip)}" style="left:{left}%;'
            f'top:{top}%;width:{width}%;height:{height}%;'
            f'border-color:{color};">'
            f'<span class="tag" style="background:{color};">'
            f'{esc(obs.primary_tag_value or obs.primary_tag_key)}</span></div>')
    return "".join(out)


def crop_html(obs: schema.Observation, frame_idx: int, size: int = 240) -> str:
    """CSS-only crop of the observation's (first) bbox from its face image."""
    box = obs.boxes[0]
    dim = max(box.xmax - box.xmin, box.ymax - box.ymin, 1)
    img_size = size * 1000.0 / dim
    cx = (box.xmin + box.xmax) / 2.0
    cy = (box.ymin + box.ymax) / 2.0
    left = size / 2.0 - cx / 1000.0 * img_size
    top = size / 2.0 - cy / 1000.0 * img_size
    return (
        f'<div class="crop" style="width:{size}px;height:{size}px;">'
        f'<img src="/image/{frame_idx}/{box.face_yaw_deg}" '
        f'style="width:{img_size:.0f}px;left:{left:.0f}px;top:{top:.0f}px;">'
        f'</div>')


OVERLAY_META_COLUMNS = {"id", "geometry", "landmark_type", "object_class"}
OVERLAY_POINT_CAP = 20_000


def load_overlay_feathers(paths: list[Path]) -> list[dict]:
    """Load landmark feathers (OSM or ENC pipeline format) into Leaflet-ready
    overlay dicts: {"name", "points": [{"lat", "lon", "label"}]}."""
    import geopandas as gpd  # deferred: only needed when overlays are requested

    overlays = []
    for path in paths:
        gdf = gpd.read_feather(path)
        if len(gdf) > OVERLAY_POINT_CAP:
            print(f"WARNING: {path.name}: {len(gdf)} landmarks, "
                  f"showing the first {OVERLAY_POINT_CAP}")
            gdf = gdf.iloc[:OVERLAY_POINT_CAP]
        tag_columns = [c for c in gdf.columns if c not in OVERLAY_META_COLUMNS]
        points = []
        for _, row in gdf.iterrows():
            if row.geometry is None:
                continue
            point = row.geometry.representative_point()
            tags = {k: row[k] for k in tag_columns
                    if isinstance(row[k], str) and row[k]}
            name = tags.pop("name", None)
            label = "<br>".join(
                ([f"<b>{esc(name)}</b>"] if name else [])
                + [f"{esc(k)}={esc(v)}" for k, v in sorted(tags.items())[:8]])
            points.append({"lat": point.y, "lon": point.x,
                           "label": label or "(no tags)"})
        overlays.append({"name": path.stem, "points": points})
        print(f"Overlay {path.stem}: {len(points)} landmarks")
    return overlays


def make_app(state: ViewerState, landmark_overlays: list[dict] | None = None) -> Flask:
    app = Flask(__name__)
    artifact = state.artifact
    landmark_overlays = landmark_overlays or []

    @app.route("/")
    def stats_page():
        s = artifact.stats
        meta_rows = "".join(
            f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in [
                ("created", artifact.created_at),
                ("git hash", artifact.git_hash),
                ("stages run", ", ".join(artifact.stages_run)),
                ("dataset", artifact.dataset_base),
                ("landmarks", artifact.landmark_base),
                ("yaw offset",
                 f"{artifact.yaw_offset_deg:.1f} deg "
                 f"({artifact.yaw_offset_method}) "
                 f"{artifact.yaw_offset_details or ''}"),
                ("anchor",
                 f"{artifact.anchor_lat:.6f}, {artifact.anchor_lon:.6f}"),
            ])
        count_rows = "".join(
            f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in [
                ("frames", s.n_frames),
                ("raw landmark entries", s.n_raw_landmark_entries),
                ("parse failures", s.n_parse_failures),
                ("invalid-yaw boxes", s.n_boxes_invalid_yaw),
                ("observations", s.n_observations),
                ("kept", s.n_kept),
                ("filtered", s.n_filtered),
                ("tracks", s.n_tracks),
                ("tracks by observability",
                 s.tracks_by_observability or "-"),
                ("singleton kept obs", s.n_singleton_obs),
            ])

        reason_rows = "".join(
            f'<tr><td><a href="/gallery?reason={esc(reason)}">'
            f'{esc(reason)}</a></td><td>{count}</td>'
            f'<td><span class="bar" style="width:{count * 300 // max(1, s.n_filtered)}px;'
            f'background:{reason_color(reason)}"></span></td></tr>'
            for reason, count in sorted(
                s.filtered_by_reason.items(), key=lambda kv: -kv[1]))
        reason_table = (
            f"<h3>Filtered by reason</h3><table><tr><th>reason</th>"
            f"<th>count</th><th></th></tr>{reason_rows}</table>"
            if reason_rows else "<p>No observations filtered.</p>")

        hist_items = sorted(
            ((int(k), v) for k, v in s.obs_per_frame_histogram.items()))
        max_count = max((v for _, v in hist_items), default=1)
        hist_rows = "".join(
            f'<tr><td>{k}</td><td>{v}</td><td><span class="bar" '
            f'style="width:{v * 300 // max_count}px"></span></td></tr>'
            for k, v in hist_items)

        body = (
            f"<h2>Filter run</h2><table>{meta_rows}</table>"
            f"<h3>Counts</h3><table>{count_rows}</table>"
            f"{reason_table}"
            f"<h3>Observations per frame</h3><table>"
            f"<tr><th>obs</th><th>frames</th><th></th></tr>{hist_rows}</table>"
            f"<h3>Config</h3><pre>{esc(config_to_yaml(artifact.config))}</pre>")
        return render(title="Filter run stats", body=body,
                                 script="")

    @app.route("/frame/<int:frame_idx>")
    def frame_page(frame_idx: int):
        frame = state.frame(frame_idx)
        obs_list = state.obs_by_frame.get(frame_idx, [])

        face_divs = []
        for face in FACES:
            overlays = "".join(
                obs_boxes_overlay_html(o, face) for o in obs_list)
            face_divs.append(
                f'<div class="face"><span class="label">yaw {face:03d}</span>'
                f'<img src="/image/{frame_idx}/{face}">{overlays}</div>')

        rows = []
        for obs in sorted(obs_list, key=lambda o: o.bearing_camera_deg):
            color = reason_color(
                obs.final_reason
                if obs.final_disposition == schema.FILTERED else "")
            tags = ", ".join(f"{k}={v}" for k, v in obs.additional_tags)
            track_link = (
                f'<a href="/track/{obs.track_id}">#{obs.track_id}</a>'
                if obs.track_id is not None else "-")
            rows.append(
                f'<tr><td>{esc(obs.obs_id)}</td>'
                f'<td>{esc(obs.primary_tag_key)}={esc(obs.primary_tag_value)}'
                f'<br><small>{esc(tags)}</small></td>'
                f'<td>{esc(obs.confidence)}</td>'
                f'<td>{obs.bearing_camera_deg:.1f} / '
                f'{obs.bearing_global_deg:.1f}</td>'
                f'<td>{obs.elevation_deg:.1f}</td>'
                f'<td>{obs.angular_width_deg:.1f}</td>'
                f'<td style="color:{color}">{esc(obs.final_disposition)}'
                f'{" - " + esc(obs.final_reason) if obs.final_reason else ""}'
                f'</td><td>{decision_trail_html(obs)}</td>'
                f'<td>{track_link}</td>'
                f'<td><small>{esc(obs.description)}</small></td></tr>')

        prev_idx, next_idx = frame_idx - 1, frame_idx + 1
        nav = (
            f'<p>{f"<a href=\"/frame/{prev_idx}\">&larr; prev</a>" if prev_idx >= 0 else ""} '
            f'frame {frame_idx} / {len(artifact.frames) - 1} '
            f'&mdash; {esc(frame.pano_id)} ({frame.lat:.6f}, {frame.lon:.6f})'
            f' {f"<a href=\"/frame/{next_idx}\">next &rarr;</a>" if next_idx < len(artifact.frames) else ""}'
            f' &mdash; <a href="/pano/{frame_idx}">raw panorama</a></p>')

        body = (
            f"<h2>Frame {frame_idx}: {esc(frame.pano_id)}</h2>{nav}"
            f'<div class="faces">{"".join(face_divs)}</div>'
            f"<table><tr><th>obs</th><th>tags</th><th>conf</th>"
            f"<th>bearing cam/global</th><th>elev</th><th>width</th>"
            f"<th>disposition</th><th>decision trail</th><th>track</th>"
            f"<th>description</th></tr>{''.join(rows)}</table>")
        script = NAV_SCRIPT.format(prev=prev_idx, next=next_idx,
                                   n_frames=len(artifact.frames))
        return render(
            title=f"Frame {frame_idx}", body=body, script=script)

    @app.route("/image/<int:frame_idx>/<int:face>")
    def face_image(frame_idx: int, face: int):
        if face not in FACES:
            abort(404)
        path = state.face_image_path(frame_idx, face)
        if not path.exists():
            abort(404)
        return send_file(path)

    @app.route("/pano/<int:frame_idx>")
    def pano_image(frame_idx: int):
        frame = state.frame(frame_idx)
        path = state.dataset_base / "panorama" / f"{frame.pano_stem}.jpg"
        if not path.exists():
            abort(404)
        return send_file(path)

    @app.route("/tracks")
    def tracks_page():
        observability = request.args.get("observability", "")
        disposition = request.args.get("disposition", "")
        sort_key = request.args.get("sort", "n_obs")

        tracks = list(artifact.tracks)
        if observability:
            tracks = [t for t in tracks if t.triangulation
                      and t.triangulation.observability == observability]
        if disposition:
            tracks = [t for t in tracks if t.disposition == disposition]
        if sort_key == "n_obs":
            tracks.sort(key=lambda t: -len(t.obs_ids))
        elif sort_key == "first_frame":
            tracks.sort(key=lambda t: t.first_frame_idx)
        elif sort_key == "similarity":
            tracks.sort(key=lambda t: -(t.mean_pairwise_similarity or 0.0))

        rows = []
        for t in tracks:
            rep = state.obs_by_id.get(t.representative_obs_id)
            tri = t.triangulation
            tri_txt = "-"
            if tri is not None:
                tri_txt = (f"{tri.observability}"
                           f"{' - ' + tri.degenerate_reason if tri.degenerate_reason else ''}"
                           f"{f'<br>range {tri.mean_range_m:.0f}m' if tri.mean_range_m else ''}"
                           f"<br>parallax {tri.parallax_deg:.1f}&deg;")
            sim_txt = (f"{t.mean_pairwise_similarity:.2f}"
                       if t.mean_pairwise_similarity is not None else "-")
            rows.append(
                f'<tr><td><a href="/track/{t.track_id}">#{t.track_id}</a></td>'
                f'<td>{esc(rep.primary_tag_key)}={esc(rep.primary_tag_value)}'
                f'<br><small>{esc(rep.description[:70])}</small></td>'
                f'<td>{len(t.obs_ids)}</td>'
                f'<td>{t.first_frame_idx}-{t.last_frame_idx}</td>'
                f'<td>{sim_txt}</td>'
                f'<td class="{t.disposition}">{esc(t.disposition)}'
                f'{" - " + esc(t.reason) if t.reason else ""}</td>'
                f'<td>{tri_txt}</td></tr>')

        filters = " ".join([
            '<a class="pill" href="/tracks">all</a>',
            '<a class="pill" href="/tracks?disposition=kept">kept</a>',
            '<a class="pill" href="/tracks?disposition=filtered">filtered</a>',
            '<a class="pill" href="/tracks?observability=near">near</a>',
            '<a class="pill" href="/tracks?observability=far">far</a>',
            '<a class="pill" href="/tracks?observability=degenerate">degenerate</a>',
            '<a class="pill" href="/tracks?sort=similarity">by similarity</a>',
            '<a class="pill" href="/tracks?sort=first_frame">by first frame</a>',
        ])
        body = (
            f"<h2>Tracks ({len(tracks)})</h2><p>{filters}</p>"
            f"<table><tr><th>track</th><th>representative</th><th>#obs</th>"
            f"<th>frames</th><th>mean sim</th><th>disposition</th>"
            f"<th>triangulation</th></tr>{''.join(rows)}</table>")
        return render(title="Tracks", body=body, script="")

    @app.route("/track/<int:track_id>")
    def track_page(track_id: int):
        track = state.track_by_id.get(track_id)
        if track is None:
            abort(404)
        members = [state.obs_by_id[oid] for oid in track.obs_ids
                   if oid in state.obs_by_id]

        cards = []
        for obs in members:
            cards.append(
                f'<div class="card">{crop_html(obs, obs.frame_idx, 200)}'
                f'<a href="/frame/{obs.frame_idx}">frame {obs.frame_idx}</a> '
                f'| bearing {obs.bearing_camera_deg:.1f}&deg; '
                f'| {esc(obs.confidence)}<br>'
                f'<b>{esc(obs.primary_tag_value or obs.primary_tag_key)}</b> '
                f'<small>{esc(", ".join(f"{k}={v}" for k, v in obs.additional_tags))}'
                f'</small><br><small>{esc(obs.description[:90])}</small></div>')

        info_rows = [
            ("members", len(members)),
            ("frames", f"{track.first_frame_idx} - {track.last_frame_idx}"),
            ("mean pairwise similarity",
             f"{track.mean_pairwise_similarity:.3f}"
             if track.mean_pairwise_similarity is not None else "-"),
            ("disposition",
             f"{track.disposition} {track.reason}".strip()),
        ]
        tri = track.triangulation
        if tri is not None:
            info_rows += [
                ("observability",
                 f"{tri.observability} {tri.degenerate_reason}".strip()),
                ("position",
                 f"({tri.lat:.6f}, {tri.lon:.6f})" if tri.lat else "-"),
                ("mean range",
                 f"{tri.mean_range_m:.0f} m" if tri.mean_range_m else "-"),
                ("residual RMS",
                 f"{tri.residual_rms_deg:.2f} deg"
                 if tri.residual_rms_deg is not None else "-"),
                ("parallax", f"{tri.parallax_deg:.1f} deg"),
                ("sigma major/minor",
                 f"{tri.sigma_major_m:.0f} / {tri.sigma_minor_m:.0f} m"
                 if tri.sigma_major_m else "-"),
                ("inliers/outliers", f"{tri.n_inliers}/{tri.n_outliers}"),
            ]
        info = "".join(f"<tr><th>{esc(k)}</th><td>{v}</td></tr>"
                       for k, v in info_rows)

        body = (f"<h2>Track #{track_id}</h2><table>{info}</table>"
                f'<p><a href="/track/{track_id}/map">ray map for this track'
                f'</a></p>'
                f'<h3>Member observations</h3>'
                f'<div class="cards">{"".join(cards)}</div>')
        return render(
            title=f"Track {track_id}", body=body, script="")

    def pair_card(pair: schema.SimilarityPairExample, label: str) -> str:
        a = state.obs_by_id.get(pair.obs_id_a)
        b = state.obs_by_id.get(pair.obs_id_b)
        if a is None or b is None:
            return ""
        halves = []
        for obs in (a, b):
            track_link = (f'<a href="/track/{obs.track_id}">#{obs.track_id}'
                          f'</a>' if obs.track_id is not None else "-")
            halves.append(
                f'<div style="flex:1">{crop_html(obs, obs.frame_idx, 120)}'
                f'<small>{esc(obs.primary_tag_value or obs.primary_tag_key)}'
                f'<br><a href="/frame/{obs.frame_idx}">f{obs.frame_idx}</a> '
                f'trk {track_link}</small></div>')
        return (f'<div class="card" style="width:300px">'
                f'<b>{label} = {pair.score:.3f}</b>'
                f'{" (same track)" if pair.same_track else ""}'
                f'<div style="display:flex;gap:6px">{"".join(halves)}</div>'
                f'</div>')

    def histogram_html(hist: dict[str, int], color: str) -> str:
        if not hist:
            return "<i>empty</i>"
        items = sorted((float(k), v) for k, v in hist.items())
        max_count = max(v for _, v in items)
        rows = "".join(
            f'<tr><td>{left:.2f}</td><td>{count}</td><td><span class="bar" '
            f'style="width:{count * 220 // max_count}px;background:{color}">'
            f'</span></td></tr>'
            for left, count in items)
        return f"<table>{rows}</table>"

    def track_map_entry(track: schema.Track, max_rays: int,
                        far_ray_length_m: float = 3000.0) -> dict:
        """Leaflet-ready geometry for one track: member rays + fit point."""
        members = [state.obs_by_id[oid] for oid in track.obs_ids
                   if oid in state.obs_by_id]
        if len(members) > max_rays:
            step = len(members) / max_rays
            members = [members[int(i * step)] for i in range(max_rays)]
        tri = track.triangulation
        ray_length = far_ray_length_m
        if tri is not None and tri.solved and tri.mean_range_m:
            ray_length = tri.mean_range_m * 1.25
        rays = []
        for obs in members:
            frame = artifact.frames[obs.frame_idx]
            east, north = bg.bearing_unit_vector(obs.bearing_global_deg)
            end = bg.latlon_from_enu(
                frame.x_m + east * ray_length, frame.y_m + north * ray_length,
                artifact.anchor_lat, artifact.anchor_lon)
            rays.append([[frame.lat, frame.lon], [end[0], end[1]]])
        rep = state.obs_by_id.get(track.representative_obs_id)
        return {
            "id": track.track_id,
            "n": len(track.obs_ids),
            "obs": tri.observability if tri else "degenerate",
            "lat": tri.lat if tri else None,
            "lon": tri.lon if tri else None,
            "sigma": tri.sigma_major_m if tri else None,
            "label": (f"{rep.primary_tag_key}={rep.primary_tag_value}"
                      if rep else ""),
            "rays": rays,
        }

    def map_page_html(tracks: list[schema.Track], title: str,
                      max_rays: int) -> str:
        map_data = {
            "trajectory": [[f.lat, f.lon] for f in artifact.frames],
            "frames": [{"i": f.frame_idx, "lat": f.lat, "lon": f.lon}
                       for f in artifact.frames],
            "tracks": [track_map_entry(t, max_rays) for t in tracks],
            "landmark_overlays": landmark_overlays,
        }
        body = (f"<h2>{esc(title)}</h2>"
                '<div id="map" style="height: 80vh"></div>')
        return render(
            title, body,
            script=MAP_SCRIPT.replace("{map_json}", json.dumps(map_data)),
            head_extra=LEAFLET_HEAD)

    @app.route("/map")
    def whole_run_map():
        tracks = [t for t in artifact.tracks
                  if t.disposition == schema.KEPT and t.triangulation]
        return map_page_html(
            tracks, f"Map: {len(tracks)} kept tracks", max_rays=4)

    @app.route("/track/<int:track_id>/map")
    def single_track_map(track_id: int):
        track = state.track_by_id.get(track_id)
        if track is None:
            abort(404)
        return map_page_html([track], f"Map: track #{track_id}", max_rays=20)

    @app.route("/semantics")
    def semantics_page():
        sections = []
        for diag in artifact.semantic_diagnostics:
            if diag.missing_embedding_values:
                missing = "".join(
                    f"<li>{esc(v)}</li>"
                    for v in diag.missing_embedding_values[:100])
                sections.append(
                    f"<h3>{esc(diag.backend)}: SKIPPED - "
                    f"{len(diag.missing_embedding_values)} tag values missing "
                    f"text embeddings</h3><ul>{missing}</ul>")
                continue
            top_pairs = "".join(
                pair_card(p, "sim") for p in diag.top_cross_track_pairs[:24])
            bottom_pairs = "".join(
                pair_card(p, "sim")
                for p in diag.bottom_intra_track_pairs[:24])
            sections.append(
                f"<h3>{esc(diag.backend)}</h3>"
                f'<div style="display:flex;gap:40px">'
                f"<div><h4>intra-track similarity</h4>"
                f"{histogram_html(diag.intra_track_similarity_histogram, '#2a2')}"
                f"</div><div><h4>inter-track similarity</h4>"
                f"{histogram_html(diag.inter_track_similarity_histogram, '#c22')}"
                f"</div></div>"
                f"<h4>Top cross-track pairs (candidate aliases / merges)</h4>"
                f'<div class="cards">{top_pairs}</div>'
                f"<h4>Bottom intra-track pairs (candidate mis-associations)"
                f'</h4><div class="cards">{bottom_pairs}</div>')

        for agreement in artifact.backend_agreements:
            examples = "".join(
                pair_card(p, f"{agreement.backend_a[:12]}")
                for p in agreement.example_disagreements[:24])
            sections.append(
                f"<h3>Agreement: {esc(agreement.backend_a)} vs "
                f"{esc(agreement.backend_b)}</h3>"
                f"<p>correlation {agreement.correlation:.3f} over "
                f"{agreement.n_pairs} pairs; "
                f"{agreement.n_large_disagreements} large disagreements</p>"
                f"<h4>Largest disagreements (score shown: "
                f"{esc(agreement.backend_a)})</h4>"
                f'<div class="cards">{examples}</div>')

        if not sections:
            sections = ["<p>No semantic diagnostics in this artifact "
                        "(run with --stage track or all).</p>"]
        body = "<h2>Semantic-similarity diagnostics</h2>" + "".join(sections)
        return render(title="Semantics", body=body, script="")

    @app.route("/gallery")
    def gallery_page():
        reason = request.args.get("reason", "")
        disposition = request.args.get(
            "disposition",
            schema.FILTERED if reason else schema.KEPT)
        page = int(request.args.get("page", "0"))
        page_size = 48

        if reason:
            matching = state.obs_by_reason.get(reason, [])
        else:
            matching = [o for o in artifact.observations
                        if o.final_disposition == disposition]
        total = len(matching)
        matching = matching[page * page_size:(page + 1) * page_size]

        cards = []
        for obs in matching:
            cards.append(
                f'<div class="card">{crop_html(obs, obs.frame_idx)}'
                f'<b>{esc(obs.primary_tag_key)}={esc(obs.primary_tag_value)}'
                f'</b><br>conf {esc(obs.confidence)} | '
                f'bearing {obs.bearing_camera_deg:.1f} | '
                f'width {obs.angular_width_deg:.1f}<br>'
                f'{decision_trail_html(obs)}<br>'
                f'<a href="/frame/{obs.frame_idx}">frame {obs.frame_idx}</a>'
                f' | {esc(obs.obs_id)}</div>')

        reason_links = " ".join(
            f'<a class="pill" href="/gallery?reason={esc(r)}">{esc(r)} '
            f'({len(v)})</a>'
            for r, v in sorted(state.obs_by_reason.items()))
        pager = []
        if page > 0:
            pager.append(f'<a href="/gallery?reason={esc(reason)}'
                         f'&disposition={esc(disposition)}&page={page - 1}">'
                         f'&larr; prev</a>')
        if (page + 1) * page_size < total:
            pager.append(f'<a href="/gallery?reason={esc(reason)}'
                         f'&disposition={esc(disposition)}&page={page + 1}">'
                         f'next &rarr;</a>')

        title = reason if reason else disposition
        body = (
            f"<h2>Gallery: {esc(title)} ({total})</h2>"
            f'<p>{reason_links} <a class="pill" href="/gallery?disposition=kept">'
            f'kept</a></p>'
            f'<div class="cards">{"".join(cards)}</div>'
            f'<p>{" | ".join(pager)}</p>')
        return render(title="Gallery", body=body, script="")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--pinhole_base", type=Path, default=None,
                        help="Defaults to the path recorded in the artifact")
    parser.add_argument("--dataset_base", type=Path, default=None,
                        help="Defaults to the path recorded in the artifact")
    parser.add_argument("--overlay_feather", type=Path, action="append",
                        default=[],
                        help="Landmark feather(s) (OSM or ENC pipeline format) "
                             "to overlay on the /map page; repeatable")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5006)
    args = parser.parse_args()

    artifact = schema.load_artifact(args.artifact)
    state = ViewerState(
        artifact,
        args.pinhole_base or Path(artifact.pinhole_base),
        args.dataset_base or Path(artifact.dataset_base))
    app = make_app(state, load_overlay_feathers(args.overlay_feather))
    print(f"Serving {args.artifact} on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
