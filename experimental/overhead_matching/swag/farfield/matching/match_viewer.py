"""Side-by-side review of what we saw against what we matched it to.

Renders a separate review output directory: a pannable map of the dataset
beside a scrollable list of tracklets. Per tracklet the list holds the observation
(query tags + the chips the audit looked at) and every map landmark the matcher
proposed, with an uncalibrated model score, match type, and how many map rows
the matched
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

The maintained stylesheet and application script live in
`match_viewer_assets/` and are inlined when rendering: there is no tile host
or network dependency, and the frozen page can be copied elsewhere unchanged.
The document around them comes from `viewers.page`, which is what gives the
page its generated mark and provenance footer.

Run:
  bazel run //experimental/overhead_matching/swag/farfield/matching:match_viewer -- \\
      --matching_dir ... --tracks_dir ... --audit_dir ... --catalog_dir ... \
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
from experimental.overhead_matching.swag.farfield.viewers import page
from experimental.overhead_matching.swag.farfield.calibration import (
    audit_io,
    heading as course_model,
)
from experimental.overhead_matching.swag.farfield.catalog import (
    catalog as catalog_lib,
)
from experimental.overhead_matching.swag.farfield.matching import (
    identity_review,
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
    """Per-tracklet display facts from validated tracks and audit metadata."""
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
    terms, and carries the clipped-logit compatibility heuristic the filter
    was actually handed next to the geometry's verdict on it.
    """
    if payload is None or key not in payload["tracklets"]:
        return ""
    tracklet = payload["tracklets"][key]
    targets = tracklet["targets"]
    if not targets:
        return ""
    floor = log_lr_defaults.get(key)
    out = ["<table class='rows'><tr><th>map row</th><th>name</th>"
           "<th>clipped-logit heuristic</th>"
           "<th>signed ray residual</th><th>range</th></tr>"]
    for (east, north, aggregate_confidence, kind, lid, label, n_rows,
         signed_residual, absolute_residual, dist, hull) in targets:
        url = osm_url(lid)
        link = (f"<a class='osm' href='{url}' target='_blank' "
                f"rel='noopener'>{esc(lid)}</a>" if url
                else f"<span class='pin'>{esc(lid)}</span>")
        score = log_lrs.get(key, {}).get(lid)
        gap = ("" if score is None or floor is None
               else f"<span class='pin'> ({score - floor:+.1f} vs floor)"
                    f"</span>")
        style = ("agree" if absolute_residual is not None
                 and absolute_residual < 15.0
                 else "iffy" if absolute_residual is not None
                 and absolute_residual < 45.0
                 else "disagree" if absolute_residual is not None else "pin")
        out.append(
            f"<tr><td>{link}</td><td>{esc(label) or '<i>unnamed</i>'}</td>"
            f"<td class='conf'>{'--' if score is None else f'{score:+.2f}'}"
            f"{gap}</td>"
            f"<td class='{style} conf'>"
            f"{'--' if signed_residual is None else f'{signed_residual:+.0f}&deg;'}</td>"
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

    `matches.json` carries uncalibrated per-call candidate scores and their
    explicit aggregation rules. The additional numbers show:
    how many independent slices the matcher scored this tracklet in, the worst
    of those calls' no-match scores, whether an instance claim was
    downgraded to a category, and its own `uniqueness_score` -- which the
    matcher requires in the response schema and preserves in the validated
    canonical result artifact. `default_log_lr` is the filter's floor:
    it is what every landmark *not* named here scores, and a named landmark's
    weight is the gap to it.
    """
    slices = entry.get("per_call_no_match_confidence") or {}
    bits = [f"aggregate no-match score "
            f"{entry['aggregate_no_match_confidence']} (uncalibrated)"]
    if slices:
        bits.append(f"over {slices.get('n', '?')} slices "
                    f"(mean {slices.get('mean', '?')}, "
                    f"min {slices.get('min', '?')})")
    scores = uniqueness.get(key) or []
    if scores:
        bits.append(f"uniqueness {entry['uniqueness']['aggregate_score']:.1f}/5 "
                    f"({len(scores)} calls, arithmetic mean)")
    if entry.get("n_signatures"):
        bits.append(f"{entry['n_signatures']} signatures matched")
    if entry.get("n_downgraded_to_category"):
        bits.append(f"{entry['n_downgraded_to_category']} downgraded to "
                    f"category")
    floor = log_lr_defaults.get(key)
    if floor is not None:
        bits.append(f"filter floor clipped-logit heuristic {floor:+.2f}")
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

    This is the file the *filter* reads: the raw model score never reaches it,
    only a deliberately uncalibrated clipped-logit compatibility heuristic.
    Every unlisted landmark gets `default_log_lr`; the viewer labels the
    quantity as a heuristic rather than implying calibrated statistical
    evidence.

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
    """Per-tracklet uniqueness scores recorded by canonical aggregation.

    The one typed ``landmark_matches`` artifact is validated first, including
    its content digest, complete-coverage attestation, request fingerprint,
    and exact tracks/audits/catalog upstream edge. There is no fallback to
    mutable transport output and no partial viewer when a request unit is
    absent.
    """
    try:
        match_ref = artifact.open_artifact(
            match_dir, expected_kind=paths_lib.LANDMARK_MATCHES)
        _snapshot, request_set = match_landmarks.load_work_snapshot(match_dir)
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
        matches = json.loads(
            (match_dir / match_landmarks.MATCHES_NAME).read_text())
    except (artifact.ArtifactError, llm_lifecycle.LlmLifecycleError,
            OSError, ValueError) as error:
        raise SystemExit(
            f"invalid canonical matching results under {match_dir}: {error}") \
            from error

    out = {}
    for key, entry in matches.items():
        uniqueness = entry.get("uniqueness")
        if (not isinstance(uniqueness, dict)
                or uniqueness.get("aggregation_rule")
                != match_landmarks.SCORE_CONTRACT[
                    "uniqueness_aggregation_rule"]
                or not isinstance(uniqueness.get("per_call_scores"), list)):
            raise SystemExit(
                f"invalid aggregated uniqueness record for {key!r}")
        out[key] = uniqueness["per_call_scores"]
    return out


def circular_fit_deg(values):
    """Circular mean, concentration, and post-fit absolute residual."""
    if not values:
        return None
    radians = [math.radians(float(value)) for value in values]
    mean_sin = sum(math.sin(value) for value in radians) / len(radians)
    mean_cos = sum(math.cos(value) for value in radians) / len(radians)
    mean = float(geo.wrap_deg(math.degrees(math.atan2(mean_sin, mean_cos))))
    concentration = math.hypot(mean_sin, mean_cos)
    postfit = [abs(float(geo.circular_diff_deg(value, mean)))
               for value in values]
    return {
        "rotation_deg": mean,
        "resultant_length": concentration,
        "median_abs_postfit_deg": statistics.median(postfit),
        "n": len(values),
    }


def bearing_residual_deg(rays, east_m: float, north_m: float):
    """Signed measured-minus-target bearing residual, circularly averaged.

    A tracklet's bearings say where the camera was pointing from a known GPS
    position. A large correction is an inconsistency under the currently
    approved frame/calibration contract; several same-signed corrections may
    instead expose one shared rotation. Returns None without bearings.

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
        out.append(float(geo.circular_diff_deg(world, to_row)))
    return circular_fit_deg(out)["rotation_deg"]


def residual_cell(residual, n_rows: int) -> str:
    """The `ray delta` cell: how far off the bearing the matched row sits.

    Colour is a shortcut, never the whole story -- the number is always shown,
    and a wide expansion is marked because there the residual is the best of
    many rows rather than a claim about one.
    """
    if residual is None:
        return "<td class='pin'>no bearing</td>"
    absolute = abs(residual)
    style = ("agree" if absolute < 15.0
             else "iffy" if absolute < 45.0 else "disagree")
    mark = "" if n_rows <= 3 else "<span class='pin'> best of "\
                                  f"{n_rows}</span>"
    return f"<td class='{style} conf'>{residual:+.0f}&deg;{mark}</td>"


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


def frame_pose_lookup(frames, east, north, course):
    """Map real frame ids to aligned pose arrays; ids need not be dense."""
    if not (len(frames) == len(east) == len(north) == len(course)):
        raise ValueError("truth pose arrays do not have equal lengths")
    result = {}
    for index, frame in enumerate(frames):
        frame_idx = int(frame.frame_idx)
        if frame_idx in result:
            raise ValueError(f"truth poses repeat frame id {frame_idx}")
        result[frame_idx] = (east[index], north[index], course[index])
    return result


def build_map_payload(paths, feather: Path, matches: dict,
                      signatures: dict, min_aggregate_confidence: float,
                      tracks: dict,
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
        cache_dir=catalog_cache_dir, keep_hulls=True,
        position_sigma_m=landmark_position_sigma_m)
    by_id = {entry.landmark_id: entry for entry in catalog}
    if len(by_id) != len(catalog):
        raise SystemExit("catalog repeats a globally namespaced landmark id")

    pose_by_frame_idx = frame_pose_lookup(frames, east, north, course)

    rays = {}
    for meas in measurements:
        kf = int(meas.anchor_keyframe_idx)
        pose = pose_by_frame_idx.get(kf)
        if pose is None:
            notes.append(f"measurement on absent keyframe {kf}; not drawn")
            continue
        pose_e, pose_n, pose_course = pose
        forward = nominal_forward.camera_to_forward_cw_deg(
            meas.bearing_camera_cw_deg, calibration)
        world = float(geo.forward_to_world_bearing_cw_deg(
            forward_world_cw_deg=pose_course,
            bearing_forward_cw_deg=forward))
        rays.setdefault(meas.tracklet_id, []).append(
            [kf, round(pose_e, 1), round(pose_n, 1), round(world, 2),
             round(pose_course, 2), round(float(meas.kappa), 1)])

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
        grouped = {}
        for match in entry["matches"]:
            signature_id = match["signature_id"]
            if signature_id not in signatures:
                raise SystemExit(
                    f"match refers to absent signature {signature_id!r}")
            grouped.setdefault(signature_id, []).append(match)
        for signature_id, signature_matches in grouped.items():
            signature_entry = signatures[signature_id]
            expansion = signature_entry["landmark_ids"]
            actual_ids = [match["landmark_id"]
                          for match in signature_matches]
            if (len(actual_ids) != len(set(actual_ids))
                    or set(actual_ids) != set(expansion)):
                raise SystemExit(
                    f"match expansion for {signature_id!r} is incomplete or "
                    "repeats rows")
            match = signature_matches[0]
            score = match["aggregate_confidence"]
            if any(item["aggregate_confidence"] != score
                   or item["match_type"] != match["match_type"]
                   for item in signature_matches):
                raise SystemExit(
                    f"expanded signature {signature_id!r} has inconsistent "
                    "aggregate scores or match types")
            if score < min_aggregate_confidence:
                continue
            n_total += len(expansion)
            for lid in expansion:
                if lid in seen:
                    raise SystemExit(
                        f"landmark {lid!r} occurs in multiple signatures")
                seen.add(lid)
                found = by_id.get(lid)
                if found is None:
                    unresolved.add(lid)
                    continue
                residual = bearing_residual_deg(my_rays, found.east_m,
                                                found.north_m)
                if len(found.hull_east_m) != len(found.hull_north_m):
                    raise SystemExit(
                        f"catalog hull coordinate lengths differ for {lid!r}")
                hull = [float(value) for pair in zip(
                    found.hull_east_m, found.hull_north_m) for value in pair]
                rows.append((
                    [round(found.east_m, 1), round(found.north_m, 1),
                     round(float(score), 3),
                     match["match_type"], lid, label_for(found.tags),
                     len(expansion),
                     None if residual is None else round(residual, 1),
                     None if residual is None else round(abs(residual), 1),
                     round(math.hypot(found.east_m - anchor_e,
                                      found.north_m - anchor_n)), hull],
                    math.hypot(found.east_m - anchor_e,
                               found.north_m - anchor_n)))
                if residual is not None:
                    best = sig_resid.setdefault(key, {})
                    if (signature_id not in best
                            or abs(residual) < abs(best[signature_id])):
                        best[signature_id] = residual
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

    all_e, all_n = list(east), list(north)
    for tracklet_payload in tracklets_payload.values():
        for target in tracklet_payload["targets"]:
            all_e.append(target[0])
            all_n.append(target[1])
            all_e.extend(target[10][0::2])
            all_n.extend(target[10][1::2])
    pad_e = max(300.0, (max(all_e) - min(all_e)) * 0.08)
    pad_n = max(300.0, (max(all_n) - min(all_n)) * 0.08)
    bounds = [min(all_e) - pad_e, max(all_e) + pad_e,
              min(all_n) - pad_n, max(all_n) + pad_n]

    # This artifact carries catalog geometry and the truth track, but no vector
    # land/water/pier basemap. State that limitation in the rendered page.
    notes.append("vector basemap not included; catalog dots carry the map "
                 "geometry")

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
        "bounds": bounds,
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


GENERATOR = ("//experimental/overhead_matching/swag/farfield/"
             "matching:match_viewer")
PAGE_TITLE = "Landmark matches — review"

_ASSET_DIR = Path(__file__).parent / "match_viewer_assets"
# Bazel lays these data deps next to this module in the runfiles tree. Keep the
# newlines that the old inline literals contributed so rendered pages remain
# byte-for-byte stable while the sources become independently maintainable.
_STYLE = (_ASSET_DIR / "style.css").read_text(
    encoding="utf-8").rstrip("\n") + "\n"
_SCRIPT = "\n" + (_ASSET_DIR / "app.js").read_text(
    encoding="utf-8").rstrip("\n") + "\n"


def render_page(body_parts: list[str]) -> str:
    """Inline the maintained assets into one frozen HTML page.

    The document skeleton comes from `viewers.page` rather than from a local
    template. This page used to build its own, and paid for it: no doctype, no
    viewport, the title "matches", no provenance footer, and -- the one that
    matters -- no `GENERATED_MARK`, which is what `indexes.refresh` checks
    before it is willing to overwrite a page. The design here is still this
    module's; only the skeleton is shared.
    """
    body = "\n".join(body_parts) + "\n" + page.provenance_footer(GENERATOR)
    return page.document(PAGE_TITLE, body, style=_STYLE)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matching_dir", type=Path, required=True)
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
    parser.add_argument("--min_aggregate_confidence", type=float, default=0.0,
                        help="Hide aggregate confidences below this on "
                             "the page (0 shows "
                             "everything; a view knob, not a result knob)")
    parser.add_argument("--no_map", action="store_true",
                        help="skip the map pane (and its feather read)")
    # These TrackletParams must match the export so the overlay draws the same
    # bearings supplied to the filter.
    parser.add_argument("--epoch_keyframes", type=int, default=None,
                        help="Keyframes fused into one bearing measurement; "
                             "required unless --no_map")
    parser.add_argument("--bearing_sigma_deg", type=float, default=None,
                        help="Per-observation bearing noise floor, degrees; "
                             "required unless --no_map")
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
    if (not math.isfinite(args.min_aggregate_confidence)
            or not 0.0 <= args.min_aggregate_confidence <= 1.0):
        parser.error(
            "--min_aggregate_confidence must be finite and in [0, 1]")

    match_dir = Path(args.matching_dir)
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
                "draw the map; they must match the export's values. Pass "
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
            args.min_aggregate_confidence, tracks, audits, fusion,
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
    identity_review.write_draft(
        out / identity_review.DRAFT_NAME, match_dir)
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
                   if len(signatures[sig]["landmark_ids"]) == 1]
    if unambiguous:
        off = sum(1 for r in unambiguous if abs(r) > 90.0)
        fit = circular_fit_deg(unambiguous)
        coherent = (len(unambiguous) >= 2
                    and fit["resultant_length"] >= 0.8
                    and fit["median_abs_postfit_deg"] <= 20.0)
        interpretation = (
            "The signed errors support a shared rotation; investigate the "
            "frame/calibration contract before classifying these rows as "
            "independent identity failures."
            if coherent else
            "The signed errors do not form a tight shared rotation, so this "
            "sample does not support one common frame correction.")
        residual_summary = (
            f"<p class='q'><b>Geometry check:</b> of "
            f"{len(unambiguous)} matches naming exactly one map row, the "
            f"best shared signed residual (measured minus target) is "
            f"<b>{fit['rotation_deg']:+.0f}&deg;</b> "
            f"(circular concentration {fit['resultant_length']:.2f}; "
            f"median post-fit absolute error "
            f"{fit['median_abs_postfit_deg']:.0f}&deg;). "
            f"<b class='disagree'>{off}</b> sit more than 90&deg; off the "
            f"unadjusted bearing. {interpretation}</p>")
    else:
        residual_summary = ("<p class='pin'>no geometry check: no match names "
                            "exactly one map row</p>" if payload is not None
                            else "")
    parts = [
        "<h1>Observation &rarr; map landmark</h1>",
        f"<p>{len(matches)} tracklets | {len(hit)} with a match | "
        f"{len(miss)} without | matches: "
        f"<span class='instance'>{kinds.get('instance', 0)} instance</span>, "
        f"<span class='category'>{kinds.get('category', 0)} category</span>"
        " | <a href='identity_review_draft.json'>identity review draft</a>"
        " | <a href='../../index.html'>run index</a></p>",
        "<p><b>instance</b> = this exact object identified. <b>category</b> = "
        "right kind of object, cannot say which. A matched <i>signature</i> "
        "expands to every map row carrying it; the expansion count is shown "
        "and flagged when large, because those contribute little information "
        "however high the uncalibrated aggregate confidence.</p>",
        "<p><b>signed ray residual</b> is measured world bearing minus the "
        "direction from the vessel's GPS position to the matched map row "
        "(CW-positive, wrapped), circularly averaged over the tracklet's "
        "anchor keyframes. Its absolute value is an "
        "independent check on the match: the matcher never saw the geometry, "
        "so a row it named that does not lie along the bearing is inconsistent "
        "under the current frame/calibration contract, however high its model "
        "score. Where a signature expands to many rows "
        "the "
        "figure is the <i>best</i> row's and is weak evidence -- with enough "
        "candidates one aligns by chance -- so it is marked as such. GPS "
        "course uncertainty, crab/current, catalog position error, timing "
        "error, and extended landmark geometry can also move a residual.</p>",
        residual_summary,
        "<div class='wrap'><div class='list'>"]

    ordered = sorted(matches.items(),
                     key=lambda kv: -(
                         kv[1]["matches"][0]["aggregate_confidence"]
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
        parts.append("<table><tr><th>aggregate confidence</th>"
                     "<th>type</th><th>signed ray residual</th>"
                     "<th>map rows</th>"
                     "<th>signature</th></tr>")
        shown = set()
        for match in entry["matches"]:
            sig = match["signature_id"]
            if (sig in shown or match["aggregate_confidence"]
                    < args.min_aggregate_confidence):
                continue
            shown.add(sig)
            n = len(signatures[sig]["landmark_ids"])
            wide = " class='wide'" if n >= 50 else ""
            parts.append(
                f"<tr><td class='conf'>"
                f"{match['aggregate_confidence']:.2f} "
                f"<span class='pin'>(uncalibrated; max of "
                f"{len(match['per_call_candidate_scores'])} call "
                f"score(s))</span>"
                f"</td>"
                f"<td class='{match['match_type']}'>{match['match_type']}</td>"
                f"{residual_cell(sig_resid.get(key, {}).get(sig), n)}"
                f"<td{wide}>{n}</td><td><code>"
                f"{esc(match['signature_display'])}</code><br>"
                f"<span class='pin'>{esc(sig)}</span></td></tr>")
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
    for key in sorted(
            miss,
            key=lambda k: -matches[k]["aggregate_no_match_confidence"]):
        entry = matches[key]
        info = info_by_key.get(key, {})
        onmap = ("" if payload is None or key not in payload["tracklets"]
                 else f" &middot; <a href='#' data-jump='{esc(key)}' "
                      f"class='pin'>bearings on map</a>")
        slices = entry.get("per_call_no_match_confidence") or {}
        scores = uniqueness.get(key) or []
        parts.append(f"<tr><td>{esc(key)}{onmap}<br>"
                     f"<span class='links'>"
                     f"{source_links(info, range_by_track)}</span></td>"
                     f"<td class='conf'>"
                     f"{entry['aggregate_no_match_confidence']}"
                     f" <span class='pin'>uncalibrated</span>"
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
            "absolute signed ray residual &gt; 45&deg;</span>"
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
        parts.append(f"<script>{_SCRIPT}</script>")
    (out / "index.html").write_text(render_page(parts), encoding="utf-8")
    size_mb = (out / "index.html").stat().st_size / 1e6
    print(f"wrote {out}/index.html ({len(hit)} matched, {len(miss)} not, "
          f"{size_mb:.1f} MB)")
    provenance.write(
        out,
        generator=GENERATOR,
        inputs={"matching": match_dir,
                "tracks": args.tracks_dir,
                "semantic_audits": args.audit_dir,
                "catalog": args.catalog_dir,
                "nominal_forward_calibration":
                    args.nominal_forward_calibration},
        config={
            "min_aggregate_confidence": args.min_aggregate_confidence,
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
