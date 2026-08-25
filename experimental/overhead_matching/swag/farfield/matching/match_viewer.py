"""Side-by-side review of what we saw against what we matched it to.

Renders a separate review output directory: a pannable map of the dataset
beside a scrollable list of tracklets. Per tracklet the list holds the observation
(query tags + the chips the audit looked at) and every map landmark the matcher
proposed, with confidence, match type, and how many map rows the matched
signature expanded to. Selecting a tracklet draws it on the map; clicking
something on the map selects it in the list.

The map is the half that makes a wrong match *visible*. In the run's own ENU
frame it carries:

  - the vessel's true track from GPS, with GPS-course ticks, and at each anchor
    keyframe of the selected tracklet the true pose and GPS course it was
    observed from;
  - a bearing ray from each of those anchors, rotated through the exact
    human-approved nominal-forward calibration consumed by localization;
  - every map row the match expanded to.

A ray that misses its matched landmark is a wrong match, and nothing in the
JSON shows that as fast as one look at the geometry. The bearings come from
`tracking/tracklets.py` -- the same per-track fused bearings the localization
export consumes -- so what is drawn here is what the filter is fed.

The catalog it draws is the one bound by the typed matching artifact. The
nominal-forward calibration is an explicit required input, validated for the
same dataset. Sun checks and alignment sweeps are diagnostics, never silently
promoted to calibration authority.

Everything is embedded in the page: no tile host, no network, and the file
stays a frozen record that can be copied elsewhere and still render.

Run:
  bazel run //experimental/overhead_matching/swag/farfield/matching:match_viewer -- \\
      --matches_dir ... --tracks_dir ... --audit_dir ... --catalog_dir ... \
      --nominal_forward_calibration ... --epoch_keyframes N \
      --bearing_sigma_deg S
"""

import argparse
import html
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import dataset
from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield import llm_lifecycle
from experimental.overhead_matching.swag.farfield import nominal_forward
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield import provenance
from experimental.overhead_matching.swag.farfield.calibration import (
    audit_io,
    heading as course_model,
)
from experimental.overhead_matching.swag.farfield.catalog import (
    catalog as catalog_lib,
)
from experimental.overhead_matching.swag.farfield.matching import (
    match_landmarks,
)
from experimental.overhead_matching.swag.farfield.tracking import tracklets

# A signature can expand to hundreds of map rows. Drawing all of them buries
# the geometry under identical dots, so the map draws the nearest few to the
# observing pose and the page says how many it left out -- a silent cap would
# read as "this is everything it matched".
MAX_TARGETS_DRAWN = 40
# Faint context dots, so "is the right object even in the catalog near here?"
# is answerable. Sorted named-first then by distance to the track, and the
# count that was dropped is reported.
MAX_CONTEXT_POINTS = 6000
# GPS-course ticks along the whole truth track, in keyframes.
COURSE_TICK_EVERY = 20
# Ray length as a multiple of the distance to the furthest drawn target, so a
# ray always overshoots what it is supposed to hit and a miss is visible.
RAY_OVERSHOOT = 1.15
MIN_RAY_M = 800.0

NAME_KEYS = ("name:en", "name", "name:ko", "ref")
TYPE_KEYS = ("seamark:type", "man_made", "amenity", "tourism", "natural",
             "waterway", "building", "place", "landuse", "leisure")


def esc(x):
    return html.escape(str(x))


def load_settings(match_dir: Path) -> dict:
    """The matching artifact's immutable reproduction summary."""
    path = match_dir / "settings.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- run match_landmarks first; the viewer "
            f"requires the published reproduction summary")
    return json.loads(path.read_text())


def track_info(tracks: dict, audits: dict, audit_meta: dict) -> dict:
    """Per-tracklet display facts, straight from the primary artifacts.

    The merged/landmarks.json this used to come from is gone (the merge stage
    is eliminated); each audited track is its own tracklet, so the span comes
    from the track's own records and the support count from the audit's meta.
    """
    info = {}
    for accepted in tracklets.build_accepted_tracklets(tracks, audits):
        track = accepted.source_track
        tid = track["track_id"]
        key = accepted.tracklet_id
        keyframes = [r["keyframe"] for r in track.get("records", [])]
        meta = audit_meta.get(accepted.local_id) or {}
        info[key] = {
            "local_id": accepted.local_id,
            "track_ids": [tid],
            "keyframe_span": ([min(keyframes), max(keyframes)]
                              if keyframes else []),
            "n_supports": meta.get("n_supports"),
        }
    return info


def source_links(info: dict, range_by_track: dict) -> str:
    """Links back to every artifact this tracklet came from."""
    out = []
    for tid in info.get("track_ids", []):
        range_name = range_by_track.get(tid)
        if range_name:
            out.append(f"<a href='../../track_{range_name}_T{tid}.html'>"
                       f"track T{tid}</a>")
        out.append(f"<a href='../../semantic_audit/review/index.html"
                   f"#T{tid}'>audit T{tid}</a>")
    span = info.get("keyframe_span") or []
    if len(span) == 2 and span[0] is not None:
        out.append(f"<a href='../../keyframes/f{int(span[0]):04d}.html'>"
                   f"keyframe f{int(span[0]):04d}</a>")
        out.append(f"<a href='../../keyframes/f{int(span[1]):04d}.html'>"
                   f"f{int(span[1]):04d}</a>")
    return " &middot; ".join(out)


def chips_for(key: str, audit_meta: dict) -> list:
    """Audit chips belonging to this tracklet's track (key == audit key)."""
    return (audit_meta.get(key) or {}).get("chips", [])[:4]


def label_for(tags: dict) -> str:
    """Shortest thing that identifies a map row to a human reading the map."""
    for key in NAME_KEYS:
        if tags.get(key):
            return str(tags[key])
    for key in TYPE_KEYS:
        if tags.get(key):
            return f"{key}={tags[key]}"
    return ""


def rows_table(key: str, payload, log_lrs: dict,
               log_lr_defaults: dict) -> str:
    """The individual map rows this tracklet matched, one row each.

    The signature table above is per *signature*; this is per *object*, which is
    the level the filter and OpenStreetMap both work at. Each id links straight
    to the OSM object so a suspicious match can be looked at in the map's own
    terms, and carries the log-LR the filter was actually handed next to the
    geometry's verdict on it.
    """
    if payload is None or key not in payload["tracklets"]:
        return ""
    tracklet = payload["tracklets"][key]
    targets = tracklet["targets"]
    if not targets:
        return ""
    floor = log_lr_defaults.get(key)
    out = ["<table class='rows'><tr><th>map row</th><th>name</th>"
           "<th>log-LR</th><th>ray &Delta;</th><th>range</th></tr>"]
    for east, north, conf, kind, lid, label, n_rows, residual, dist in targets:
        url = osm_url(lid)
        link = (f"<a class='osm' href='{url}' target='_blank' "
                f"rel='noopener'>{esc(lid)}</a>" if url
                else f"<span class='pin'>{esc(lid)}</span>")
        score = log_lrs.get(key, {}).get(lid)
        gap = ("" if score is None or floor is None
               else f"<span class='pin'> ({score - floor:+.1f} vs floor)"
                    f"</span>")
        style = ("agree" if residual is not None and residual < 15.0
                 else "iffy" if residual is not None and residual < 45.0
                 else "disagree" if residual is not None else "pin")
        out.append(
            f"<tr><td>{link}</td><td>{esc(label) or '<i>unnamed</i>'}</td>"
            f"<td class='conf'>{'--' if score is None else f'{score:+.2f}'}"
            f"{gap}</td>"
            f"<td class='{style} conf'>"
            f"{'--' if residual is None else f'{residual:.0f}&deg;'}</td>"
            f"<td class='conf'>{dist / 1000.0:.1f} km</td></tr>")
    if tracklet["n_resolved"] > tracklet["n_shown"]:
        out.append(f"<tr><td colspan='5' class='pin'>"
                   f"{tracklet['n_resolved'] - tracklet['n_shown']} further "
                   f"placed rows not listed (nearest and most confident "
                   f"{tracklet['n_shown']} shown; the signature expanded to "
                   f"{tracklet['n_rows']} rows in total)</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def stats_line(key: str, entry: dict, info: dict, uniqueness: dict,
               log_lr_defaults: dict) -> str:
    """Every scalar the matcher produced for this tracklet, in one line.

    `matches.json` carries more than the confidence the page used to show, and
    the numbers it left out are the ones that say how much to trust the rest:
    how many independent slices the matcher scored this tracklet in, the worst
    of those slices' no-match confidences, whether an instance claim was
    downgraded to a category, and its own `uniqueness_score` -- which the
    matcher requires in the response schema and preserves in the validated
    canonical result artifact. `default_log_lr` is the filter's floor:
    it is what every landmark *not* named here scores, and a named landmark's
    weight is the gap to it.
    """
    slices = entry.get("per_slice_no_match") or {}
    bits = [f"no_match_confidence {entry['no_match_confidence']}"]
    if slices:
        bits.append(f"over {slices.get('n', '?')} slices "
                    f"(mean {slices.get('mean', '?')}, "
                    f"min {slices.get('min', '?')})")
    scores = uniqueness.get(key) or []
    if scores:
        bits.append(f"uniqueness {statistics.median(scores):.0f}/5 "
                    f"({len(scores)} slices)")
    if entry.get("n_signatures"):
        bits.append(f"{entry['n_signatures']} signatures matched")
    if entry.get("n_downgraded_to_category"):
        bits.append(f"{entry['n_downgraded_to_category']} downgraded to "
                    f"category")
    floor = log_lr_defaults.get(key)
    if floor is not None:
        bits.append(f"filter floor log-LR {floor:+.2f}")
    if info.get("n_supports") is not None:
        bits.append(f"{info['n_supports']} supports")
    bits.append(f"tracks {info.get('track_ids', [])}")
    return " &middot; ".join(bits)


def osm_url(landmark_id: str):
    """openstreetmap.org URL for a catalog id, or None if it is not OSM.

    Ids are `osm:<kind>:<id>`; ENC-sourced rows have no OSM object to open, so
    they get no link rather than a broken one.
    """
    parts = str(landmark_id).split(":")
    if len(parts) != 3 or parts[0] != "osm":
        return None
    kind, ident = parts[1], parts[2]
    if kind not in ("node", "way", "relation"):
        return None
    return f"https://www.openstreetmap.org/{kind}/{ident}"


def load_log_lrs(match_dir: Path) -> tuple:
    """`compatibility.json` as {tracklet: {landmark_id: log_lr}} + defaults.

    This is the file the *filter* reads: confidence never reaches it, only the
    log likelihood ratio the matcher derived from it, with every unlisted
    landmark getting `default_log_lr`. A match's real weight in the
    localization is the gap between those two numbers, so the page shows both
    -- a +4.0 against a -4.0 default is the matcher asserting e^8, and that
    assertion is what moves the filter, not the 1.00 next to it.

    The file is a msgspec-encoded `list[structs.CompatibilityTable]` (tagged
    with `kind`); reading the plain JSON here keeps the viewer's dependency
    surface small.
    """
    path = match_dir / "compatibility.json"
    if not path.is_file():
        raise SystemExit(f"missing matching compatibility artifact {path}")
    try:
        tables = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid matching compatibility artifact {path}: "
                         f"{error}") from error
    if not isinstance(tables, list):
        raise SystemExit(f"{path} must contain a list of compatibility tables")
    scores, defaults = {}, {}
    for table in tables:
        tracklet_id = table["tracklet_id"]
        if tracklet_id in scores:
            raise SystemExit(
                f"{path} contains duplicate table for {tracklet_id!r}")
        scores[tracklet_id] = {e["landmark_id"]: e["log_lr"]
                               for e in table.get("entries", [])}
        defaults[tracklet_id] = table.get("default_log_lr")
    return scores, defaults


def load_uniqueness(match_dir: Path) -> dict:
    """Per-tracklet uniqueness scores from complete canonical LLM results.

    The one typed ``landmark_matches`` artifact is validated first, including
    its content digest, complete-coverage attestation, request fingerprint,
    and exact tracks/audits/catalog upstream edge. There is no fallback to
    mutable transport output and no partial viewer when a request unit is
    absent.
    """
    try:
        match_ref = artifact.open_artifact(
            match_dir, expected_kind=paths_lib.LANDMARK_MATCHES)
        request_set = llm_lifecycle.load_request_set(
            match_dir / llm_lifecycle.REQUEST_SET_NAME)
        manifest = artifact.load_manifest(match_dir)
        expected_upstream_kinds = (
            paths_lib.OBJECT_TRACKS,
            paths_lib.SEMANTIC_AUDITS,
            paths_lib.CATALOGS,
        )
        if tuple(ref.kind for ref in manifest.upstreams) != (
                expected_upstream_kinds):
            raise llm_lifecycle.LlmLifecycleError(
                "matching artifact does not bind exact tracks, audits, and "
                "catalog upstreams")
        if request_set.upstreams != manifest.upstreams:
            raise llm_lifecycle.LlmLifecycleError(
                "matching request set is not bound to artifact upstreams")
        if any(ref.dataset != match_ref.dataset
               for ref in request_set.upstreams):
            raise llm_lifecycle.LlmLifecycleError(
                "matching request set crosses dataset identities")
        if (manifest.config.get("request_set_fingerprint")
                != request_set.fingerprint):
            raise llm_lifecycle.LlmLifecycleError(
                "matching artifact has the wrong request fingerprint")
        n_expected = len(request_set.units)
        if (manifest.config.get("phase") != "canonical_results"
                or manifest.config.get("coverage") != "complete"
                or manifest.config.get("n_expected") != n_expected
                or manifest.config.get("n_successful") != n_expected):
            raise llm_lifecycle.IncompleteCoverageError(
                "matching artifact does not attest complete canonical coverage")
        results = llm_lifecycle.load_canonical_results(
            match_dir / llm_lifecycle.CANONICAL_RESULTS_NAME, request_set)
    except (artifact.ArtifactError, llm_lifecycle.LlmLifecycleError,
            OSError) as error:
        raise SystemExit(
            f"invalid canonical matching results under {match_dir}: {error}") \
            from error

    metadata = {unit.key: unit.metadata for unit in request_set.units}
    out = {}
    for record in results:
        meta = metadata[record.key]
        # Revalidate the stage payload at the read boundary too. A forged
        # manifest cannot turn a malformed response into usable viewer data.
        provider_shape = {"candidates": [{"content": {"parts": [{
            "text": json.dumps(record.result),
        }]}}]}
        validated = match_landmarks.validate_matching_response(
            record.key, provider_shape, meta)
        keys = meta["batch_keys"]
        for row in validated["matches"]:
            out.setdefault(keys[row["set_1_id"]], []).append(
                row["uniqueness_score"])
    return out


def bearing_residual_deg(rays, east_m: float, north_m: float):
    """Median |bearing we measured - direction to this map row|, over a
    tracklet's anchor keyframes.

    This is the arbiter the confidence score is not. A tracklet's bearings say
    where the camera was pointing from a known GPS position; if the row the
    matcher named does not lie that way, the match is wrong however sure the
    matcher was. Returns None when the tracklet has no bearings.

    Read it with one caveat, which the page repeats: for a signature that
    expands to many rows this is the *best* row's residual, and with enough
    candidates one aligns by chance. It is strong evidence only where the
    expansion is small.
    """
    if not rays:
        return None
    out = []
    for _kf, east, north, world, _course, _kappa in rays:
        to_row = geo.compass_bearing_deg(east_m - east, north_m - north)
        out.append(abs(float(geo.circular_diff_deg(world, to_row))))
    return statistics.median(out)


def residual_cell(residual, n_rows: int) -> str:
    """The `ray delta` cell: how far off the bearing the matched row sits.

    Colour is a shortcut, never the whole story -- the number is always shown,
    and a wide expansion is marked because there the residual is the best of
    many rows rather than a claim about one.
    """
    if residual is None:
        return "<td class='pin'>no bearing</td>"
    style = ("agree" if residual < 15.0
             else "iffy" if residual < 45.0 else "disagree")
    mark = "" if n_rows <= 3 else "<span class='pin'> best of "\
                                  f"{n_rows}</span>"
    return f"<td class='{style} conf'>{residual:.0f}&deg;{mark}</td>"


def truth_track(dataset_base: Path, *, gps_course_min_displacement_m: float,
                gps_course_smooth_window_s: float):
    """Poses and course per keyframe, plus the ENU anchor they are about.

    Positions come from `dataset.load_frames`/`fill_enu` and the course from
    the same GPS-course model the localization export uses
    (`calibration/heading.py`), so the map's idea of where the vessel was and
    its direction of travel is the filter's idea. The viewer refuses a course
    model that abstains; it never substitutes intrinsics heading or a second
    finite-difference model.
    """
    frames = dataset.load_frames(Path(dataset_base))
    if not frames:
        raise SystemExit(f"no panoramas under {dataset_base}/panorama")
    anchor_lat, anchor_lon = dataset.fill_enu(frames)
    frames.sort(key=lambda fr: fr.frame_idx)
    east = [fr.x_m for fr in frames]
    north = [fr.y_m for fr in frames]
    times = [fr.time_s for fr in frames]
    model = course_model.gps_course_model_from_positions(
        east, north, times,
        min_displacement_m=gps_course_min_displacement_m,
        smooth_window_s=gps_course_smooth_window_s)
    if model is None:
        raise ValueError(
            "GPS-course model abstained for inadequate displacement; the map "
            "cannot draw world-frame bearings")
    course = [float(model.course_world_cw_deg_at(t)) for t in times]
    source = ("GPS-course model over positions (calibration/heading.py, "
              "the model the localization export uses)")
    return frames, east, north, course, anchor_lat, anchor_lon, source


def build_map_payload(paths, feather: Path, matches: dict,
                      signatures: dict, min_confidence: float, tracks: dict,
                      audits: dict, fusion,
                      calibration: nominal_forward.NominalForward,
                      calibration_path: Path,
                      catalog_cache_dir: Path,
                      landmark_position_sigma_m: float,
                      gps_course_min_displacement_m: float,
                      gps_course_smooth_window_s: float) -> tuple:
    """Everything the map draws, in ENU metres about the run's anchor.

    Returns (payload, notes, sig_resid). Notes are surfaced on the page -- an
    unresolved landmark id or a truncated expansion changes what the map is
    showing, so it is reported rather than absorbed.
    """
    notes = []
    frames, east, north, course, anchor_lat, anchor_lon, course_source = \
        truth_track(
            paths.dataset_base,
            gps_course_min_displacement_m=gps_course_min_displacement_m,
            gps_course_smooth_window_s=gps_course_smooth_window_s)

    with Image.open(
            paths.panorama_dir / f"{frames[0].pano_stem}.jpg") as probe:
        pano_w = probe.size[0]
    accepted = tracklets.build_accepted_tracklets(tracks, audits)
    observations = tracklets.build_camera_bearing_observations(
        accepted, pano_w, fusion.bearing_sigma_deg)
    measurements = tracklets.epoch_fused_compat_v1(observations, fusion)

    catalog = catalog_lib.load_catalog_cached(
        feather, anchor_lat, anchor_lon,
        cache_dir=catalog_cache_dir, keep_hulls=False,
        position_sigma_m=landmark_position_sigma_m)
    by_id = {entry.landmark_id: entry for entry in catalog}

    rays = {}
    for meas in measurements:
        kf = int(meas.anchor_keyframe_idx)
        if kf >= len(east):
            notes.append(f"measurement on keyframe {kf} beyond the "
                         f"{len(east)} frames on disk; not drawn")
            continue
        forward = nominal_forward.camera_to_forward_cw_deg(
            meas.bearing_camera_cw_deg, calibration)
        world = float(geo.forward_to_world_bearing_cw_deg(
            forward_world_cw_deg=course[kf],
            bearing_forward_cw_deg=forward))
        rays.setdefault(meas.tracklet_id, []).append(
            [kf, round(east[kf], 1), round(north[kf], 1), round(world, 2),
             round(course[kf], 2), round(float(meas.kappa), 1)])

    unresolved = set()
    tracklets_payload = {}
    # {tracklet: {signature: best residual}} -- rendered in the list table, not
    # embedded in the map payload, so the signature strings are not stored a
    # second time in a page that already carries them.
    sig_resid = {}
    for key, entry in matches.items():
        my_rays = rays.get(key, [])
        anchor_e = my_rays[0][1] if my_rays else 0.0
        anchor_n = my_rays[0][2] if my_rays else 0.0
        rows, seen = [], set()
        n_total = 0
        for match in entry["matches"]:
            if match["confidence"] < min_confidence:
                continue
            expansion = signatures.get(match["signature"], [])
            for lid in expansion:
                n_total += 1
                if lid in seen:
                    continue
                seen.add(lid)
                found = by_id.get(lid)
                if found is None:
                    unresolved.add(lid)
                    continue
                residual = bearing_residual_deg(my_rays, found.east_m,
                                                found.north_m)
                rows.append((
                    [round(found.east_m, 1), round(found.north_m, 1),
                     round(float(match["confidence"]), 3),
                     match["match_type"], lid, label_for(found.tags),
                     len(expansion),
                     None if residual is None else round(residual, 1),
                     round(math.hypot(found.east_m - anchor_e,
                                      found.north_m - anchor_n))],
                    math.hypot(found.east_m - anchor_e,
                               found.north_m - anchor_n)))
                if residual is not None:
                    best = sig_resid.setdefault(key, {})
                    sig = match["signature"]
                    if sig not in best or residual < best[sig]:
                        best[sig] = residual
        rows.sort(key=lambda row: (-row[0][2], row[1]))
        drawn = rows[:MAX_TARGETS_DRAWN]
        if not drawn and not my_rays:
            continue
        far = max((row[1] for row in drawn), default=0.0)
        tracklets_payload[key] = {
            "rays": my_rays,
            "targets": [row[0] for row in drawn],
            "n_shown": len(drawn), "n_resolved": len(rows),
            "n_rows": n_total,
            "ray_m": round(max(MIN_RAY_M, far * RAY_OVERSHOOT), 1),
        }
    if unresolved:
        notes.append(f"{len(unresolved)} matched landmark ids are not in "
                     f"{Path(feather).name} and could not be placed "
                     f"(e.g. {sorted(unresolved)[:3]})")

    all_e = list(east) + [t[0] for tk in tracklets_payload.values()
                          for t in tk["targets"]]
    all_n = list(north) + [t[1] for tk in tracklets_payload.values()
                           for t in tk["targets"]]
    pad_e = max(300.0, (max(all_e) - min(all_e)) * 0.08)
    pad_n = max(300.0, (max(all_n) - min(all_n)) * 0.08)
    bounds = [min(all_e) - pad_e, max(all_e) + pad_e,
              min(all_n) - pad_n, max(all_n) + pad_n]

    # The vector basemap (land/water/pier fills) is not ported yet; the page
    # degrades to catalog dots + truth track, which still carry the geometry
    # a match check needs. Announced rather than silent.
    notes.append("vector basemap not drawn (module not yet ported); catalog "
                 "dots carry the map geometry")

    mid_e = sum(east) / len(east)
    mid_n = sum(north) / len(north)
    inside = [e for e in catalog
              if bounds[0] <= e.east_m <= bounds[1]
              and bounds[2] <= e.north_m <= bounds[3]]
    inside.sort(key=lambda e: (not any(e.tags.get(k) for k in NAME_KEYS),
                               math.hypot(e.east_m - mid_e,
                                          e.north_m - mid_n)))
    context = inside[:MAX_CONTEXT_POINTS]
    if len(inside) > len(context):
        notes.append(f"context dots capped at {MAX_CONTEXT_POINTS}: "
                     f"{len(inside) - len(context)} of {len(inside)} catalog "
                     f"rows in view are not drawn (named and nearest kept)")

    # Inverse of the equirectangular ENU (geometry.RegionFrame), as two scale
    # factors: going back is linear about the anchor, so the page can do it
    # without a projection library.
    per_m = 1.0 / geo.METERS_PER_DEG_LAT
    payload = {
        "anchor": [anchor_lat, anchor_lon],
        "enu": {"lat0": anchor_lat, "lon0": anchor_lon,
                "dlat_per_m": per_m,
                "dlon_per_m": per_m / math.cos(math.radians(anchor_lat))},
        "truth": [round(v, 1) for pair in zip(east, north) for v in pair],
        "ticks": [[round(east[i], 1), round(north[i], 1), round(course[i], 1)]
                  for i in range(0, len(east), COURSE_TICK_EVERY)],
        "basemap": {"layers": [], "source": None},
        "context": {
            "e": [round(e.east_m, 1) for e in context],
            "n": [round(e.north_m, 1) for e in context],
            "l": [label_for(e.tags) for e in context],
            "i": [e.landmark_id for e in context],
        },
        "tracklets": tracklets_payload,
        "bounds": [round(v, 1) for v in bounds],
        "nominal_forward": {
            "camera_bearing_cw_deg": round(
                calibration.bearing_camera_cw_deg, 6),
            "version": calibration.version,
            "mounting_id": calibration.mounting_id,
            "uncertainty_deg": calibration.uncertainty_deg,
            "source": str(calibration_path),
            "authority": "human-approved nominal-forward calibration",
        },
        "gps_course_source": course_source,
        "n_catalog": len(catalog),
        "n_measurements": len(measurements),
        "notes": notes,
    }
    return payload, notes, sig_resid


MAP_CSS = """
.wrap{display:grid;grid-template-columns:minmax(0,1fr) minmax(420px,46vw);
gap:18px;align-items:start}
/* A grid child defaults to min-width:auto, which is its *content* width -- a
   1000px panorama chip then widens the column and slides under the map. This
   plus the scrolling .chips strip keeps wide evidence inside its own pane.
   min-width:0 is the whole fix; the column needs no overflow of its own. */
.list{min-width:0}
.chips{display:flex;gap:5px;overflow-x:auto;padding-bottom:5px;
scrollbar-width:thin}
.chips img{flex:0 0 auto;height:120px;width:auto;max-width:none}
.rows{font-size:12.5px}
.rows td{padding:3px 8px}
.osm{white-space:nowrap}
@media (max-width:1100px){.wrap{grid-template-columns:1fr}
.mapcol{position:static !important;height:70vh !important}}
.mapcol{position:sticky;top:12px;height:calc(100vh - 28px);display:flex;
flex-direction:column;gap:6px}
#cv{flex:1;width:100%;background:#0e1319;border:1px solid #2b3540;
border-radius:6px;cursor:grab;touch-action:none;display:block;min-height:0}
#cv.drag{cursor:grabbing}
.mapbar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:12px;
color:#8fa3b5}
.mapbar button{background:#1b2430;color:#cfe0f0;border:1px solid #34455a;
border-radius:4px;padding:3px 9px;font-size:12px;cursor:pointer;
font-family:inherit}
.mapbar button:hover{background:#243244}
.mapbar a{background:#1b2430;border:1px solid #34455a;border-radius:4px;
padding:3px 9px;text-decoration:none}
.mapbar a:hover{background:#243244}
#ll-what{color:#cfe0f0}
.mapbar button[aria-pressed=true]{background:#2d4358;border-color:#4af}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:11.5px;color:#8fa3b5}
.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;
margin-right:4px;vertical-align:middle}
.tip{min-height:34px;font-size:12px;color:#cfe0f0;background:#131a22;
border:1px solid #263140;border-radius:5px;padding:5px 8px;
overflow-wrap:anywhere}
.card{border-left:3px solid #2b3540;padding-left:10px;margin:0 0 6px 0}
.card.sel{border-left-color:#fa2;background:#1a1c1f}
h2{cursor:pointer}
h2:hover{color:#fff}
.mapnote{color:#c93;font-size:12px;margin:4px 0}
.pin{color:#8fa3b5;font-size:11.5px}
"""

# The map is one canvas redrawn from the payload on every interaction: at this
# vertex count a full redraw is cheaper than tracking dirty regions, and it
# keeps pan, zoom and selection from drifting out of sync with each other.
MAP_JS = """
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
  return g[7] == null ? '' : ', ray delta ' + g[7] + ' deg';
}

function esc(t){
  return String(t).replace(/[&<>"]/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
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
    what = 'robot at keyframe ' + tk.rays[0][0] + ' (' + sel + ')';
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
  t.targets.forEach(g=>add(g[0],g[1]));
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
          k + ' -> ' + (g[5] || g[4]) + '  conf ' + g[2] + ', ' + g[3]
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
      const x = sx(g[0]), y = sy(g[1]);
      ctx.beginPath(); ctx.arc(x, y, 6, 0, 7);
      ctx.fillStyle = g[3] === 'instance' ? '#3ecf8e' : '#6f9bff';
      ctx.fill();
      ctx.strokeStyle = '#0e1319'; ctx.lineWidth = 2; ctx.stroke();
      if(g[7] != null && g[7] > 45){
        ctx.beginPath(); ctx.arc(x, y, 10.5, 0, 7);
        ctx.strokeStyle = '#f2777a'; ctx.lineWidth = 1.6; ctx.stroke();
      }
      if(g[5]){
        ctx.fillStyle = '#dfe8f0'; ctx.font = '12px sans-serif';
        ctx.fillText(g[5], x + 9, y + 4);
      }
      hits.push({x, y, r:9, kind:'target', tracklet:sel, label:
        (g[5] || g[4]) + '  conf ' + g[2] + ', ' + g[3] + ', ' + g[6]
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
  document.querySelectorAll('.card').forEach(c =>
    c.classList.toggle('sel', c.dataset.key === key));
  const tk = M.tracklets[key];
  if(tk){
    let msg = key + ': ' + tk.rays.length + ' bearing'
      + (tk.rays.length===1?'':'s') + ', ' + tk.n_shown + ' of '
      + tk.n_resolved + ' placed map rows drawn';
    if(tk.n_rows > tk.n_resolved)
      msg += ' (' + tk.n_rows + ' rows before dedup/placement)';
    tip.textContent = msg;
  } else {
    tip.textContent = key + ': nothing to draw (no bearings and no placed '
      + 'match)';
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
if(location.hash) select(decodeURIComponent(location.hash.slice(1)), true);
"""


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matches_dir", type=Path, required=True)
    parser.add_argument("--tracks_dir", type=Path, required=True)
    parser.add_argument("--audit_dir", type=Path, required=True)
    parser.add_argument("--catalog_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Separate review output; published matching "
                             "artifacts are immutable")
    parser.add_argument("--nominal_forward_calibration", type=Path,
                        required=True,
                        help="Exact human-approved calibration also supplied "
                             "to localization input export")
    parser.add_argument("--min_confidence", type=float, default=0.0,
                        help="Hide matches below this on the page (0 shows "
                             "everything; a view knob, not a result knob)")
    parser.add_argument("--no_map", action="store_true",
                        help="skip the map pane (and its feather read)")
    # tracklets.TrackletParams for the bearing overlay: no defaults on
    # purpose (REORG.md rule 2). Must match the values the export uses or the
    # rays drawn are not the rays the filter is fed.
    parser.add_argument("--epoch_keyframes", type=int, default=None,
                        help="Keyframes fused into one bearing measurement; "
                             "required unless --no_map (previously 5 on the "
                             "harbor datasets)")
    parser.add_argument("--bearing_sigma_deg", type=float, default=None,
                        help="Per-observation bearing noise floor, degrees; "
                             "required unless --no_map (previously 1.0 on "
                             "the harbor datasets)")
    parser.add_argument(
        "--landmark_position_sigma_m", type=float, default=None,
        help="One uniform catalog position uncertainty in metres; required "
             "when drawing the map")
    parser.add_argument(
        "--gps_course_min_displacement_m", type=float, default=None,
        help="Minimum displacement for a GPS-course sample; required when "
             "drawing the map")
    parser.add_argument(
        "--gps_course_smooth_window_s", type=float, default=None,
        help="GPS-course smoothing window in seconds; required when drawing "
             "the map")
    paths_lib.add_arguments(parser, dataset_required=True)
    args = parser.parse_args()

    match_dir = Path(args.matches_dir)
    try:
        artifact.open_artifact(
            match_dir, expected_kind=paths_lib.LANDMARK_MATCHES,
            expected_dataset=args.dataset)
        audits = audit_io.load_audits(args.tracks_dir, args.audit_dir)
        catalog_ref = artifact.open_artifact(
            args.catalog_dir, expected_kind=paths_lib.CATALOGS,
            expected_dataset=args.dataset)
        match_manifest = artifact.load_manifest(match_dir)
        expected_upstreams = (
            audits.tracks_ref, audits.semantic_audits_ref, catalog_ref)
        if match_manifest.upstreams != expected_upstreams:
            raise ValueError(
                "matching artifact was not built from the supplied tracks, "
                "audits, and catalog artifacts")
        calibration = nominal_forward.load(
            args.nominal_forward_calibration,
            expected_dataset=args.dataset)
    except (artifact.ArtifactError, audit_io.AuditArtifactError,
            OSError, ValueError) as error:
        raise SystemExit(f"invalid viewer input: {error}") from error
    load_settings(match_dir)
    tracks = audits.source_tracks
    matches = json.loads((match_dir / match_landmarks.MATCHES_NAME).read_text())
    signatures = json.loads(
        (match_dir / match_landmarks.SIGNATURES_NAME).read_text())
    log_lrs, log_lr_defaults = load_log_lrs(match_dir)
    uniqueness = load_uniqueness(match_dir)
    for label, keys in (("canonical results", set(uniqueness)),
                        ("compatibility tables", set(log_lr_defaults))):
        if keys != set(matches):
            raise SystemExit(
                f"matching output coverage mismatch: matches.json has "
                f"{len(matches)} tracklets but {label} has {len(keys)}")
    meta_path = Path(args.audit_dir) / "audit_meta.json"
    meta_document = json.loads(meta_path.read_text())
    if (meta_document.get("schema") != audit_io.META_SCHEMA
            or not isinstance(meta_document.get("requests"), dict)):
        raise SystemExit(
            f"{meta_path} is not a {audit_io.META_SCHEMA} artifact")
    audit_meta = meta_document["requests"]
    range_by_track = {
        request["track_id"]: request["range"]
        for request in audit_meta.values()
        if "range" in request
    }
    info_by_key = track_info(tracks, audits, audit_meta)

    payload, notes, sig_resid = None, [], {}
    feather = Path(args.catalog_dir) / "catalog.feather"
    if not args.no_map:
        if (args.epoch_keyframes is None or args.bearing_sigma_deg is None
                or args.landmark_position_sigma_m is None
                or args.gps_course_min_displacement_m is None
                or args.gps_course_smooth_window_s is None):
            parser.error(
                "--epoch_keyframes, --bearing_sigma_deg, and "
                "--landmark_position_sigma_m plus both GPS-course parameters "
                "are required to "
                "draw the map (tracklets.TrackletParams; no defaults on "
                "purpose -- they must match the export's values). Pass "
                "--no_map to skip the map pane.")
        if not feather.exists():
            raise SystemExit(
                f"catalog artifact does not contain {feather.name}: {feather}")
        paths = paths_lib.resolve(
            parser, args,
            require=("dataset_base", "panorama_dir"))
        fusion = tracklets.TrackletParams(
            epoch_keyframes=args.epoch_keyframes,
            bearing_sigma_deg=args.bearing_sigma_deg)
        payload, notes, sig_resid = build_map_payload(
            paths, feather, matches, signatures,
            args.min_confidence, tracks, audits, fusion,
            calibration, Path(args.nominal_forward_calibration),
            Path(args.output_dir) / "catalog_cache",
            args.landmark_position_sigma_m,
            args.gps_course_min_displacement_m,
            args.gps_course_smooth_window_s)

    out = Path(args.output_dir)
    if out == match_dir or match_dir in out.parents:
        raise SystemExit(
            "--output_dir must be outside the immutable matching artifact")
    out.mkdir(parents=True, exist_ok=True)
    kinds = Counter(x["match_type"] for v in matches.values()
                    for x in v["matches"])
    hit = [k for k, v in matches.items() if v["n_landmarks"]]
    miss = [k for k, v in matches.items() if not v["n_landmarks"]]

    # The unambiguous matches are the only ones whose residual is a clean
    # verdict, so they get their own headline rather than being averaged in
    # with wide expansions that align by chance.
    unambiguous = [residual
                   for key, sigs in sig_resid.items()
                   for sig, residual in sigs.items()
                   if len(signatures.get(sig, [])) == 1]
    if unambiguous:
        off = sum(1 for r in unambiguous if r > 90.0)
        residual_summary = (
            f"<p class='q'><b>Geometry check:</b> of "
            f"{len(unambiguous)} matches naming exactly one map row, the "
            f"median ray &Delta; is "
            f"<b>{statistics.median(unambiguous):.0f}&deg;</b> and "
            f"<b class='disagree'>{off}</b> sit more than 90&deg; off the "
            f"bearing they were seen on. Those cannot be the object observed. "
            f"No single rotation fixes them, so this is the matcher naming "
            f"wrong rows, not a frame or mount-offset error &mdash; a frame "
            f"error would move every ray by the same angle.</p>")
    else:
        residual_summary = ("<p class='pin'>no geometry check: no match names "
                            "exactly one map row</p>" if payload is not None
                            else "")
    parts = [
        "<html><head><title>matches</title><meta charset='utf-8'><style>",
        "body{font-family:sans-serif;background:#161616;color:#ddd;margin:16px}",
        "a{color:#8bf}code{color:#cfc}",
        ".q{background:#1b2430;border-left:3px solid #4af;padding:8px 12px;",
        "border-radius:5px;max-width:100%;margin:6px 0}",
        "table{border-collapse:collapse;margin:8px 0;width:100%;",
        "max-width:100%}",
        "td,th{padding:4px 10px;font-size:13px;border-bottom:1px solid #303030;",
        "text-align:left;vertical-align:top}th{color:#89a;font-size:11.5px;",
        "text-transform:uppercase;letter-spacing:.08em}",
        ".instance{color:#3c8;font-weight:bold}.category{color:#89f}",
        ".wide{color:#fa2}.conf{font-variant-numeric:tabular-nums}",
        ".agree{color:#3ecf8e}.iffy{color:#e3b341}",
        ".disagree{color:#f2777a;font-weight:bold}",
        "img{height:120px;border-radius:4px;margin:3px 3px 0 0;",
        "vertical-align:top}",
        "h2{margin-top:26px;border-top:1px solid #2c2c2c;padding-top:14px;",
        "font-size:19px}",
        ".nomatch{color:#999}",
        ".links{font-size:12.5px;margin:6px 0}",
        ".nochips{color:#a88;font-size:12.5px;font-style:italic;margin:6px 0}",
        MAP_CSS,
        "</style></head><body>",
        "<h1>Observation &rarr; map landmark</h1>",
        f"<p>{len(matches)} tracklets | {len(hit)} with a match | "
        f"{len(miss)} without | matches: "
        f"<span class='instance'>{kinds.get('instance', 0)} instance</span>, "
        f"<span class='category'>{kinds.get('category', 0)} category</span>"
        " | <a href='../../index.html'>run index</a></p>",
        "<p><b>instance</b> = this exact object identified. <b>category</b> = "
        "right kind of object, cannot say which. A matched <i>signature</i> "
        "expands to every map row carrying it; the expansion count is shown "
        "and flagged when large, because those contribute little information "
        "however confident the match.</p>",
        "<p><b>ray &Delta;</b> is the angle between the bearing we measured "
        "and the direction from the vessel's GPS position to the matched map "
        "row, median over the tracklet's anchor keyframes. It is an "
        "independent check on the match: the matcher never saw the geometry, "
        "so a row it named that does not lie along the bearing is wrong "
        "however confident it was. Where a signature expands to many rows the "
        "figure is the <i>best</i> row's and is weak evidence -- with enough "
        "candidates one aligns by chance -- so it is marked as such.</p>",
        residual_summary,
        "<div class='wrap'><div class='list'>"]

    ordered = sorted(matches.items(),
                     key=lambda kv: -(kv[1]["matches"][0]["confidence"]
                                      if kv[1]["matches"] else -1))
    for key, entry in ordered:
        if not entry["matches"]:
            continue
        info = info_by_key.get(key, {})
        parts.append(f"<div class='card' data-key='{esc(key)}' "
                     f"id='{esc(key)}'>")
        onmap = ("" if payload is None or key not in payload["tracklets"]
                 else f" <a href='#' data-jump='{esc(key)}' "
                      f"class='pin'>[show on map]</a>")
        local_id = info.get("local_id", key)
        parts.append(f"<h2>{esc(local_id)}{onmap}</h2>")
        stats = stats_line(key, entry, info, uniqueness, log_lr_defaults)
        parts.append(f"<div class='q'><b>observed:</b> "
                     f"<code>{esc(entry['query'])}</code><br>"
                     f"<span class='nomatch'>{stats}</span></div>")
        parts.append(f"<div class='links'>"
                     f"{source_links(info, range_by_track)}</div>")
        chips = chips_for(local_id, audit_meta)
        if chips:
            parts.append("<div class='chips'>")
            for chip in chips:
                parts.append(f"<img src='../../semantic_audit/chips/"
                             f"{Path(chip).name}' loading='lazy'>")
            parts.append("</div>")
        if not chips:
            parts.append("<div class='nochips'>no chips: the audit builder "
                         "rendered none for this track. Use the track and "
                         "keyframe links above.</div>")
        parts.append("<table><tr><th>conf</th><th>type</th>"
                     "<th>ray &Delta;</th><th>map rows</th>"
                     "<th>signature</th></tr>")
        shown = set()
        for match in entry["matches"]:
            sig = match["signature"]
            if sig in shown or match["confidence"] < args.min_confidence:
                continue
            shown.add(sig)
            n = len(signatures.get(sig, []))
            wide = " class='wide'" if n >= 50 else ""
            parts.append(
                f"<tr><td class='conf'>{match['confidence']:.2f}</td>"
                f"<td class='{match['match_type']}'>{match['match_type']}</td>"
                f"{residual_cell(sig_resid.get(key, {}).get(sig), n)}"
                f"<td{wide}>{n}</td><td><code>{esc(sig)}</code></td></tr>")
        parts.append("</table>")
        parts.append(rows_table(key, payload, log_lrs, log_lr_defaults))
        parts.append("</div>")

    parts.append("<h2>Returned no match</h2>")
    parts.append("<p>Note this means <i>the matcher found nothing</i>, not "
                 "that the object is absent from the map. For an unnamed "
                 "<code>building=commercial</code> the right building is "
                 "almost certainly in the catalog; the query simply cannot "
                 "discriminate it.</p><table><tr><th>tracklet</th>"
                 "<th>no-match conf</th><th>observed</th></tr>")
    for key in sorted(miss, key=lambda k: -matches[k]["no_match_confidence"]):
        entry = matches[key]
        info = info_by_key.get(key, {})
        onmap = ("" if payload is None or key not in payload["tracklets"]
                 else f" &middot; <a href='#' data-jump='{esc(key)}' "
                      f"class='pin'>bearings on map</a>")
        slices = entry.get("per_slice_no_match") or {}
        scores = uniqueness.get(key) or []
        parts.append(f"<tr><td>{esc(key)}{onmap}<br>"
                     f"<span class='links'>"
                     f"{source_links(info, range_by_track)}</span></td>"
                     f"<td class='conf'>{entry['no_match_confidence']}"
                     f"<br><span class='pin'>{slices.get('n', '?')} slices, "
                     f"min {slices.get('min', '?')}"
                     + (f", uniq {statistics.median(scores):.0f}/5"
                        if scores else "") +
                     f"</span></td>"
                     f"<td><code>{esc(entry['query'])}</code></td></tr>")
    parts.append("</table></div>")

    if payload is None:
        parts.append("<div class='mapcol'><p class='mapnote'>map skipped "
                     "(--no_map)</p></div>")
    else:
        parts.append("<div class='mapcol'>")
        parts.append("<div class='mapbar'>"
                     "<button id='b-all'>all matches</button>"
                     "<button id='b-truth'>fit track</button>"
                     "<button id='b-sel'>fit selection</button>"
                     "<button id='b-ctx' aria-pressed='true'>catalog dots"
                     "</button>"
                     "<span>drag to pan &middot; wheel to zoom &middot; "
                     "click a marker</span></div>")
        parts.append(
            "<div class='mapbar'><span id='ll-what'>view centre</span>"
            "<a id='ll-osm' href='#' target='_blank' rel='noopener'>OSM"
            "</a><a id='ll-gm' href='#' target='_blank' rel='noopener'>"
            "Google Maps</a><span id='ll-coord' class='pin'></span></div>")
        parts.append("<canvas id='cv'></canvas>")
        parts.append("<div class='tip' id='tip'>Click a tracklet title, or "
                     "&ldquo;show on map&rdquo;, to draw its bearings and "
                     "matches.</div>")
        parts.append(
            "<div class='legend'>"
            "<span><i style='background:#e6ecf2'></i>true track + GPS course"
            "</span>"
            "<span><i style='background:#ffb347'></i>bearing ray</span>"
            "<span><i style='background:#3ecf8e'></i>instance match</span>"
            "<span><i style='background:#6f9bff'></i>category match</span>"
            "<span><i style='background:#3d454e'></i>catalog row</span>"
            "<span><i style='border:1.5px solid #f2777a;background:none'></i>"
            "ray &Delta; &gt; 45&deg;</span>"
            "</div>")
        parts.append(
            f"<div class='pin'>ENU about {payload['anchor'][0]:.5f}, "
            f"{payload['anchor'][1]:.5f} &middot; camera nominal forward "
            f"{payload['nominal_forward']['camera_bearing_cw_deg']}&deg; "
            f"from <b>{esc(payload['nominal_forward']['authority'])}</b> "
            f"({esc(payload['nominal_forward']['source'])}) &middot; GPS course: "
            f"{esc(payload['gps_course_source'])} &middot; "
            f"{payload['n_catalog']} catalog rows &middot; "
            f"{payload['n_measurements']} fused bearings &middot; catalog "
            f"{esc(Path(feather).name)}</div>")
        for note in notes:
            parts.append(f"<div class='mapnote'>note: {esc(note)}</div>")
        parts.append("</div>")

    parts.append("</div>")
    if payload is not None:
        blob = json.dumps(payload, separators=(",", ":"),
                          ensure_ascii=False).replace("</", "<\\/")
        parts.append(f"<script>const MAP_DATA={blob};</script>")
        parts.append(f"<script>{MAP_JS}</script>")
    parts.append("</body></html>")
    (out / "index.html").write_text("\n".join(parts))
    size_mb = (out / "index.html").stat().st_size / 1e6
    print(f"wrote {out}/index.html ({len(hit)} matched, {len(miss)} not, "
          f"{size_mb:.1f} MB)")
    provenance.write(
        out,
        generator=("//experimental/overhead_matching/swag/farfield/"
                   "matching:match_viewer"),
        inputs={"matching": match_dir,
                "tracks": args.tracks_dir,
                "semantic_audits": args.audit_dir,
                "catalog": args.catalog_dir,
                "nominal_forward_calibration":
                    args.nominal_forward_calibration},
        config={
            "min_confidence": args.min_confidence,
            "no_map": bool(args.no_map),
            "nominal_forward": {
                "schema": nominal_forward.SCHEMA,
                "version": calibration.version,
                "mounting_id": calibration.mounting_id,
                "bearing_camera_cw_deg":
                    calibration.bearing_camera_cw_deg,
            },
            "epoch_keyframes": args.epoch_keyframes,
            "bearing_sigma_deg": args.bearing_sigma_deg,
            "landmark_position_sigma_m": args.landmark_position_sigma_m,
            "gps_course_min_displacement_m":
                args.gps_course_min_displacement_m,
            "gps_course_smooth_window_s": args.gps_course_smooth_window_s,
        })
    if payload is not None:
        print(f"  map: {len(payload['tracklets'])} tracklets drawable, "
              f"{len(payload['truth']) // 2} truth poses, "
              f"{len(payload['context']['e'])} context dots")
        for note in notes:
            print(f"  note: {note}")


if __name__ == "__main__":
    main()
