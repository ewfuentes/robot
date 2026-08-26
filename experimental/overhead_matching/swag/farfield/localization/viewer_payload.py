"""Assemble everything the viewer renders, from a run directory and its context.

One builder, two consumers: `viewer.py` inlines this into a self-contained page
and `viewer_server.py` serves it over HTTP. They therefore cannot disagree about
what a run means, which is the same reason §7.5 wants the replay path to be the
production filter.

Inputs, in decreasing order of necessity:

  run_dir      required. Tier 0/1 + checkpoints + the manifest.
  attribution  the Tier-3 cache if present (`forensics attribute` writes it).
               Absent, every view still works except the waterfalls.
  feather      landmark geometry, for the offline vector basemap.
  ghosts       counterfactual run directories to overlay.

Everything except `run_dir` degrades to a note rather than an error, because a
viewer that refuses to open a run until its whole context is present is a viewer
you stop using.

One deliberate difference from the previous viewer: **particles are drawn as a
weighted sample.** Checkpoints hold the weighted posterior before resampling, so
drawing a uniform subsample renders every particle as equally believed and makes
a cloud that is 99% dead weight look alive. Systematic resampling by weight
gives a fair draw from the posterior instead, which is what "the particle cloud"
is supposed to show.
"""

import base64
import dataclasses
import json
import math
import re
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.localization import (
    attribution as attribution_mod,
    basemap as basemap_mod,
    forensics,
    metrics,
    replay as replay_mod,
    run_io,
    sources as sources_mod,
)

# Enough particles to read a cloud's shape, few enough to inline for every
# checkpoint of a long run.
MAX_PARTICLES_PER_FRAME = 900
# Mode colours, drawn from chart symbology and kept clear of the
# starboard-green / port-red semantic pair so "which mode" never reads as
# "good or bad".
MODE_COLORS = ("#C21E76", "#2E7FA8", "#B07A16", "#7A4FBF",
               "#0F8C86", "#A8447E", "#5C6BC0", "#77702A")
# Landmark glyph classes, resolved from the catalog's type_key. §7.4 asks for
# landmarks "glyphed by type"; these are the classes a harbour run needs to
# tell apart at a glance.
GLYPH_RULES = (
    ("light", ("lighthouse", "light", "beacon", "seamark")),
    ("navaid", ("buoy", "daymark", "cardinal", "lateral")),
    ("tank", ("storage_tank", "silo", "tank", "chimney", "gasometer")),
    ("tower", ("tower", "mast", "antenna", "crane", "monument", "obelisk")),
    ("bridge", ("bridge", "viaduct")),
    ("water", ("pier", "dock", "marina", "breakwater", "ferry")),
    ("nature", ("island", "cape", "beach", "peak", "wood", "water")),
)
DEFAULT_GLYPH = "building"


def _round(values, decimals=1):
    return [round(float(v), decimals) for v in values]


def _glyph_for(type_key: str) -> str:
    lowered = (type_key or "").lower()
    for glyph, needles in GLYPH_RULES:
        if any(needle in lowered for needle in needles):
            return glyph
    return DEFAULT_GLYPH


def _weighted_sample(log_weight: np.ndarray, count: int,
                     rng: np.random.Generator) -> np.ndarray:
    """A fair draw from the weighted posterior, by systematic resampling.

    Systematic rather than multinomial for the same reason the filter uses it:
    it has lower variance, so the drawn cloud is a more faithful picture of the
    posterior at the small sample sizes a page can carry.

    Kept local rather than imported: `filter.systematic_resample` is the
    canonical systematic resampler, but it mutates a whole ParticleBelief and
    applies kernel regularization — this needs only an index draw from raw
    log-weights loaded off disk. Keep the two in step.
    """
    n = log_weight.shape[0]
    if n <= count:
        return np.arange(n)
    weights = np.exp(log_weight - log_weight.max())
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        return rng.choice(n, size=count, replace=False)
    positions = (rng.random() + np.arange(count)) / count
    return np.searchsorted(np.cumsum(weights / total), positions)


def _landmark_positions(manifest, frame):
    return frame.enu_from_latlon(
        np.array([lm.lat_deg for lm in manifest.landmarks]),
        np.array([lm.lon_deg for lm in manifest.landmarks]))


def _landmark_geometry(landmark) -> dict | None:
    if len(landmark.hull_east_m) < 2:
        return None
    points = [[float(east), float(north)] for east, north in zip(
        landmark.hull_east_m, landmark.hull_north_m)]
    kind = ("polygon" if len(points) >= 4 and points[0] == points[-1]
            else "linestring")
    return {
        "id": landmark.landmark_id,
        "kind": kind,
        "points": points,
    }


def _catalog_bounds(east, north, landmarks, margin: float = 2000.0):
    all_east = [float(value) for value in east]
    all_north = [float(value) for value in north]
    for landmark in landmarks:
        all_east.extend(float(value) for value in landmark.hull_east_m)
        all_north.extend(float(value) for value in landmark.hull_north_m)
    if not all_east:
        return None
    return (min(all_east) - margin, max(all_east) + margin,
            min(all_north) - margin, max(all_north) + margin)


def referenced_landmark_ids(data) -> set:
    """Landmarks the run actually talks about.

    Everything else renders as backdrop: a whole-map catalog (13,210 rows on
    the harbour runs) drawn as labelled glyphs is unreadable and unresponsive.
    """
    ids = set()
    for table in data.tables.values():
        ids.update(forensics.endorsed_entries(table))
    for event in data.proposal_events:
        for landmark_ids in event.hypothesis_landmark_ids:
            ids.update(landmark_ids)
    for record in data.health:
        for assoc in record.associations:
            ids.update(landmark_id
                       for landmark_id, value in assoc.responsibilities.items()
                       if value > 1e-3)
    return ids


def _error_series(health, truth) -> dict:
    """keyframe_idx -> (mean_err, map_err, heading_err), where truth exists.

    The error math is metrics.py's, never re-derived here: those helpers
    already own the skip-keyframes-without-truth rule.
    """
    have_truth = [r for r in health
                  if r.keyframe_idx in {t.keyframe_idx for t in truth}]
    if not have_truth:
        return {}
    errors = metrics.position_errors_m(health, truth)
    map_errors = metrics.map_position_errors_m(health, truth)
    heading_errors = metrics.heading_errors_deg(health, truth)
    return {r.keyframe_idx: (float(e), float(me), float(he))
            for r, e, me, he in zip(have_truth, errors, map_errors,
                                    heading_errors)}


def _health_payload(data, truth_by_kf) -> list:
    errors_by_kf = _error_series(data.health, data.truth)
    out = []
    for record in data.health:
        entry = {
            "kf": record.keyframe_idx,
            "ess": round(record.ess, 1),
            "resampled": bool(record.resampled),
            "sigma": round(record.position_std_m, 1),
            "headingSigma": round(record.heading_std_deg, 2),
            "meanE": round(record.mean_east_m, 1),
            "meanN": round(record.mean_north_m, 1),
            "meanH": round(record.mean_heading_deg, 1),
            "mapE": round(record.map_east_m, 1),
            "mapN": round(record.map_north_m, 1),
            "entropy": round(record.mode_entropy_nats, 3),
            "proposalShare": round(record.proposal_weight_share, 3),
            "nMeas": record.n_measurements,
            "positionMass": {
                key: round(value, 8) for key, value in sorted(
                    record.position_probability_mass.items())},
            "modes": [{
                "id": mode.mode_id, "w": round(mode.weight, 4),
                "e": round(mode.mean_east_m, 1),
                "n": round(mode.mean_north_m, 1),
                "h": round(mode.mean_heading_deg, 1),
                "std": round(mode.position_std_m, 1),
                "hstd": round(mode.heading_std_deg, 2),
                "born": mode.birth_keyframe_idx,
                "prov": {k: str(v) for k, v in mode.provenance.items()},
            } for mode in record.modes],
            "assoc": [{
                "mode": a.mode_id, "trk": a.tracklet_id,
                "null": round(a.null_share, 4),
                "surprise": round(a.surprise_share, 4),
                "resp": {k: round(v, 4) for k, v in
                         sorted(a.responsibilities.items(),
                                key=lambda kv: -kv[1])[:6] if v > 1e-4},
            } for a in record.associations],
        }
        nulls = [a.null_share for a in record.associations
                 if a.mode_id is None]
        if nulls:
            entry["null"] = round(float(np.mean(nulls)), 4)
        truth = truth_by_kf.get(record.keyframe_idx)
        if truth is not None:
            mean_err, map_err, heading_err = errors_by_kf[record.keyframe_idx]
            entry["truthE"] = round(truth.east_m, 1)
            entry["truthN"] = round(truth.north_m, 1)
            entry["err"] = round(mean_err, 1)
            entry["mapErr"] = round(map_err, 1)
            entry["headingErr"] = round(heading_err, 2)
        out.append(entry)
    return out


def _mode_trajectories(data) -> list:
    """Per-mode weight over time, for the §7.4 view-4 ledger."""
    trajectories: dict[int, dict] = {}
    for record in data.health:
        for mode in record.modes:
            entry = trajectories.setdefault(mode.mode_id, {
                "id": mode.mode_id, "born": mode.birth_keyframe_idx,
                "kf": [], "w": [], "prov": {k: str(v) for k, v
                                            in mode.provenance.items()},
                "parents": list(mode.parent_mode_ids)})
            entry["kf"].append(record.keyframe_idx)
            entry["w"].append(round(mode.weight, 4))
    for event in data.mode_events:
        entry = trajectories.get(event.mode_id)
        if entry is not None and event.kind == "death":
            entry["died"] = event.keyframe_idx
    return sorted(trajectories.values(), key=lambda m: m["id"])


def _natural_key(value: str) -> tuple:
    """Compare numeric runs numerically, so LT2 sorts before LT10."""
    parts = re.split(r"([0-9]+)", value)
    return tuple((1, int(part)) if part.isdigit()
                 else (0, part.casefold()) for part in parts)


def _tracklet_dossiers(data, cache, triage, bundle,
                       max_table_entries: int = 40) -> list:
    """The §7.4 view-3 payload: one tracklet's whole life in one object."""
    epochs_by_tracklet: dict[str, list] = {}
    for meas in data.measurements:
        epochs_by_tracklet.setdefault(meas.tracklet_id, []).append(meas)

    # Per-mode association evolution, keyed by tracklet.
    assoc_by_tracklet: dict[str, list] = {}
    for record in data.health:
        for assoc in record.associations:
            assoc_by_tracklet.setdefault(assoc.tracklet_id, []).append({
                "kf": record.keyframe_idx, "mode": assoc.mode_id,
                "null": round(assoc.null_share, 4),
                "surprise": round(assoc.surprise_share, 4),
                "resp": {k: round(v, 4) for k, v in
                         sorted(assoc.responsibilities.items(),
                                key=lambda kv: -kv[1])[:5] if v > 1e-3},
            })

    out = []
    for tracklet_id in sorted(epochs_by_tracklet, key=_natural_key):
        epochs = sorted(epochs_by_tracklet[tracklet_id],
                        key=lambda m: m.anchor_keyframe_idx)
        table = data.tables.get(tracklet_id)
        entry = {
            "id": tracklet_id,
            "epochs": [{
                "kf": m.anchor_keyframe_idx,
                "bearing": round(m.bearing_forward_cw_deg, 2),
                "sigma": round(math.degrees(
                    1.0 / math.sqrt(max(m.kappa, 1e-9))), 2),
            } for m in epochs],
            "assoc": assoc_by_tracklet.get(tracklet_id, []),
        }
        if table is not None:
            default = forensics.clipped_log_lr(table, table.default_log_lr)
            ranked = sorted(table.entries, key=lambda e: -e.log_lr)
            entry["table"] = {
                "status": table.status,
                "matcher": table.matcher_version,
                "default": round(default, 3),
                "clipLo": table.clip_lo, "clipHi": table.clip_hi,
                "nEntries": len(table.entries),
                "nEndorsed": len(forensics.endorsed_entries(table)),
                "entries": [{
                    "lm": e.landmark_id,
                    "lr": round(forensics.clipped_log_lr(table, e.log_lr), 3),
                    "raw": round(e.log_lr, 3),
                    "endorsed": forensics.clipped_log_lr(table, e.log_lr)
                    > default + forensics.ENDORSEMENT_EPS,
                } for e in ranked[:max_table_entries]],
                # A tie is a disjunction the matcher could not resolve; the
                # count is what says whether "top entry" means anything.
                "nTied": sum(1 for e in table.entries
                             if abs(e.log_lr - ranked[0].log_lr) < 1e-9)
                if ranked else 0,
            }
        if cache is not None:
            series = attribution_mod.tracklet_series(cache, tracklet_id)
            entry["attribution"] = [
                {"kf": row.keyframe_idx, "nats": round(row.self_nats, 4)}
                for row in series]
            entry["attributionTotal"] = round(
                sum(row.self_nats for row in series), 3)
        verdict = triage.get(tracklet_id)
        if verdict is not None:
            entry["triage"] = _triage_payload(verdict)
        source = bundle.get(tracklet_id) if bundle is not None else None
        if source is not None:
            entry["source"] = {
                "localId": source.local_id,
                "sourceTrackId": source.source_track_id,
                "name": source.name,
                "tags": list(source.tags),
                "description": source.description,
                "features": list(source.features),
                "unresolved": source.unresolved,
                "nSupports": source.n_supports,
                "span": list(source.keyframe_span),
                "validSegments": [
                    list(segment) for segment in source.valid_segments],
                "verdict": source.verdict,
                "confidence": source.confidence,
                "chip": source.chip_data_uri,
                # Resolved from the exact ancestor artifact's manifest, never
                # by searching old tracking-run filenames.
                "evidencePage": source.evidence_page,
                "evidenceHref": (
                    (Path(bundle.tracks_ref.path) / source.evidence_page)
                    .resolve().as_uri()
                    if source.evidence_page is not None else None),
            }
        out.append(entry)
    return out


def _fit_payload(fit) -> dict | None:
    if fit is None:
        return None
    return {
        "lm": fit.landmark_id,
        "rms": (round(fit.rms_deg, 2) if math.isfinite(fit.rms_deg) else None),
        "max": (round(fit.max_deg, 2) if math.isfinite(fit.max_deg) else None),
        "rangeM": round(fit.median_range_m),
        "lr": round(fit.log_lr, 2) if fit.log_lr is not None else None,
        "endorsed": fit.endorsed,
    }


def _triage_payload(verdict) -> dict:
    return {
        "verdict": verdict.verdict,
        "nEpochs": verdict.n_epochs,
        "span": list(verdict.keyframe_span),
        "toleranceDeg": round(verdict.tolerance_deg, 1),
        "medianSigmaDeg": round(verdict.median_sigma_deg, 2),
        "bestFit": _fit_payload(verdict.best_fit),
        "bestEndorsed": _fit_payload(verdict.best_endorsed_fit),
        "topEndorsed": _fit_payload(verdict.top_endorsed_fit),
        "nConsistent": verdict.n_consistent_catalog,
        "ambiguous": verdict.ambiguous,
        "antiEvidence": verdict.anti_evidence,
        "tableStatus": verdict.table_status,
        "nTableEntries": verdict.n_table_entries,
        "nEndorsed": verdict.n_endorsed,
        "bestFilterShare": round(verdict.best_filter_share, 3),
        "epochs": [{
            "kf": e.keyframe_idx,
            "bearing": round(e.bearing_forward_cw_deg, 2),
            "sigma": round(e.sigma_deg, 2),
            "worldBearing": round(e.true_world_bearing_deg, 1),
            "bestRes": (round(e.best_fit_residual_deg, 2)
                        if e.best_fit_residual_deg is not None else None),
            "topRes": (round(e.top_endorsed_residual_deg, 2)
                       if e.top_endorsed_residual_deg is not None else None),
            "filterTop": e.filter_top_id,
            "filterTopShare": round(e.filter_top_share, 3),
            "bestFitShare": round(e.best_fit_share, 3),
            "null": round(e.null_share, 3),
            "surprise": round(e.surprise_share, 3),
        } for e in verdict.epochs],
    }


def _attribution_payload(cache, data) -> dict | None:
    if cache is None:
        return None
    mode_ids = sorted({w.group for w in cache.group_weights})
    waterfalls = {}
    for mode_id in mode_ids:
        waterfall = attribution_mod.attribute(cache, mode_id)
        waterfalls[str(mode_id)] = _waterfall_payload(waterfall)
    deaths = attribution_mod.death_waterfall(cache, data.mode_events)
    return {
        "verified": cache.verified_against_manifest,
        "modes": waterfalls,
        "deaths": {str(mode_id): _waterfall_payload(w)
                   for mode_id, w in deaths.items()},
        "nRows": len(cache.contributions),
    }


def _waterfall_payload(waterfall) -> dict:
    return {
        "range": list(waterfall.keyframe_range),
        "total": round(waterfall.total_nats, 3),
        "evidence": round(waterfall.evidence_nats, 3),
        "structural": round(waterfall.structural_nats, 3),
        "observed": (round(waterfall.observed_nats, 3)
                     if waterfall.observed_nats is not None else None),
        "residual": (round(waterfall.residual_nats, 4)
                     if waterfall.residual_nats is not None else None),
        "terms": [{"label": label, "nats": round(nats, 3), "kind": kind}
                  for label, nats, kind in waterfall.terms],
    }


def _ghost_payload(ghost_dirs) -> tuple:
    """Counterfactual trails to overlay, plus notes for the ones that failed."""
    ghosts, notes = [], []
    for index, ghost_dir in enumerate(ghost_dirs or ()):
        ghost_dir = Path(ghost_dir)
        try:
            ghost = run_io.read_run(ghost_dir)
        except Exception as exc:  # noqa: BLE001 - a bad ghost is not fatal
            notes.append(f"ghost {ghost_dir}: unreadable ({exc})")
            continue
        errors = metrics.map_position_errors_m(ghost.health, ghost.truth)
        label = ghost.manifest.scenario_name
        detail = (ghost_dir / "counterfactual.json")
        if detail.exists():
            try:
                label = json.loads(detail.read_text()).get("describe", label)
            except ValueError:
                pass
        ghosts.append({
            "id": index,
            "label": label,
            "dir": str(ghost_dir),
            "trail": [[round(r.map_east_m, 1), round(r.map_north_m, 1)]
                      for r in ghost.health],
            "sigma": [round(r.position_std_m, 1) for r in ghost.health],
            "finalErr": round(float(errors[-1]), 1) if errors.size else None,
            "medianErr": (round(float(np.median(errors)), 1)
                          if errors.size else None),
            "nModes": len(ghost.health[-1].modes) if ghost.health else 0,
        })
    return ghosts, notes


def _satellite_payload(directory, notes) -> dict | None:
    """Optional raster underlay layers, embedded as data URIs.

    Expects the directory `satellite_underlay.py` writes: a `satellite.json`
    listing layers coarse-to-fine, each naming an image and the ENU box it
    covers in the run's own frame. Layers are returned in that order and drawn
    in it, so a fine mosaic lands on top of a wide one.

    Embedded rather than referenced, for the same reason the vector basemap is
    built offline: the page is a record that has to survive being copied. The
    cost is bytes, and imagery is easily the largest thing in a payload, which is
    why it is opt-in. The licence note travels into the notes so a page built
    with non-redistributable imagery says so.
    """
    if directory is None:
        return None
    directory = Path(directory)
    meta = directory / "satellite.json"
    if not meta.exists():
        notes.append(f"satellite underlay: no {meta}; skipped")
        return None
    try:
        spec = json.loads(meta.read_text())
    except Exception as exc:  # noqa: BLE001 - a bad underlay must not be fatal
        notes.append(f"satellite underlay unreadable ({exc}); skipped")
        return None
    layers, total = [], 0
    for entry in spec.get("layers", []):
        image = directory / entry.get("image", "")
        if not image.exists():
            notes.append(f"satellite underlay: {image} missing; layer skipped")
            continue
        blob = image.read_bytes()
        total += len(blob)
        layers.append({
            "e0": float(entry["east_min"]), "e1": float(entry["east_max"]),
            "n0": float(entry["north_min"]), "n1": float(entry["north_max"]),
            "zoom": entry.get("zoom"),
            "uri": "data:image/jpeg;base64," + base64.b64encode(blob).decode(
                "ascii"),
        })
    if not layers:
        notes.append("satellite underlay: no usable layers; skipped")
        return None
    notes.append(f"satellite underlay: {len(layers)} layer(s), "
                 f"{total / 1e6:.1f} MB, from "
                 f"{spec.get('source', 'unstated source')}")
    if spec.get("licence"):
        notes.append(f"satellite underlay licence: {spec['licence']}")
    return {"layers": layers, "source": spec.get("source", "unstated")}


def build(run_dir: Path, tracks_dir: Path | None = None,
          audit_dir: Path | None = None, feather: Path | None = None, ghost_dirs=(),
          max_particles: int = MAX_PARTICLES_PER_FRAME,
          embed_source_chips: bool = True,
          with_basemap: bool = True,
          basemap_detail: float = 1.0,
          satellite: Path | None = None) -> dict:
    """The whole payload. See the module docstring for what is optional.

    The catalog visibility radius comes from the run's manifest — required
    since schema 0.3, so there is no override and no fallback: the page shows
    the geometry the run was actually built with.
    """
    run_dir = Path(run_dir)
    data = run_io.read_run(run_dir)
    manifest = data.manifest
    notes: list[str] = []

    status = replay_mod.replayability(run_dir)
    if not status.replayable:
        notes.append("this run is not faithfully replayable by this build: "
                     + status.report().replace("\n", " ").strip())
    notes.extend(status.notes)

    frame = geo.RegionFrame(manifest.anchor_lat_deg, manifest.anchor_lon_deg)
    visible_range = manifest.max_visible_range_m
    catalog = replay_mod.build_catalog(manifest, visible_range, frame)
    east, north = _landmark_positions(manifest, frame)

    cache = None
    try:
        cache = attribution_mod.read_cache(
            run_dir, expected_sha256=manifest.particle_history_sha256 or None)
    except ValueError as exc:
        notes.append(str(exc))
    if cache is None:
        notes.append("no Tier-3 attribution cache: run `forensics attribute` "
                     "on this run to enable the waterfall and per-tracklet "
                     "attribution series")

    triage = forensics.triage_tracklets(data, catalog)
    events = forensics.derive_events(data)

    if (tracks_dir is None) != (audit_dir is None):
        raise ValueError("tracks_dir and audit_dir must be supplied together")
    bundle = None
    if tracks_dir is not None:
        tracklet_ids = {
            measurement.tracklet_id for measurement in data.measurements}
        bundle = sources_mod.load(
            run_dir, tracks_dir, audit_dir, tracklet_ids,
            embed_chips=embed_source_chips)
        notes.extend(bundle.notes)

    referenced = referenced_landmark_ids(data)
    truth_by_kf = {t.keyframe_idx: t for t in data.truth}

    rng = np.random.default_rng(0)
    checkpoints = {}
    for keyframe_idx, arrays in sorted(data.checkpoints.items()):
        index = _weighted_sample(arrays["log_weight"], max_particles, rng)
        checkpoints[str(keyframe_idx)] = {
            "e": _round(arrays["east_m"][index], 0),
            "n": _round(arrays["north_m"][index], 0),
            "h": _round(np.degrees(arrays["heading_rad"][index]), 0),
            "m": [int(v) for v in arrays["mode_id"][index]],
        }

    bounds = _catalog_bounds(east, north, manifest.landmarks)
    basemap = (basemap_mod.build(feather, manifest.anchor_lat_deg,
                                 manifest.anchor_lon_deg, bounds_enu=bounds,
                                 detail=basemap_detail)
               if with_basemap else basemap_mod.Basemap([], None, ()))
    notes.extend(basemap.notes)

    ghosts, ghost_notes = _ghost_payload(ghost_dirs)
    notes.extend(ghost_notes)

    measurements: dict[str, list] = {}
    for meas in data.measurements:
        table = data.tables.get(meas.tracklet_id)
        endorsed = forensics.endorsed_entries(table) if table else {}
        top = max(endorsed, key=endorsed.get) if endorsed else None
        measurements.setdefault(str(meas.anchor_keyframe_idx), []).append({
            "trk": meas.tracklet_id,
            "bearing": round(meas.bearing_forward_cw_deg, 2),
            "sigma": round(math.degrees(
                1.0 / math.sqrt(max(meas.kappa, 1e-9))), 2),
            # The matcher's best claim, so the map can red-flag the case where
            # it disagrees with where that landmark actually lies (§7.4).
            "topLm": top,
            "topLr": round(endorsed[top], 2) if top else None,
            "nEndorsed": len(endorsed),
        })

    landmark_geometry = []
    for landmark in manifest.landmarks:
        geometry = _landmark_geometry(landmark)
        if geometry is not None:
            geometry["referenced"] = landmark.landmark_id in referenced
            landmark_geometry.append(geometry)
    metric_config = manifest.position_mass_metric
    metric_summary = (
        metrics.position_mass_summary(data.health, metric_config)
        if metric_config is not None else None)

    return {
        "run": {
            "scenario": manifest.scenario_name,
            "runKind": manifest.run_kind,
            "initialization": manifest.initialization_kind,
            "bearingsConsumed": manifest.bearings_consumed,
            "ablationTags": list(manifest.ablation_tags),
            "truthPositionArtifact": manifest.truth_position_artifact,
            "truthPositionSchema": manifest.truth_position_schema,
            "positionMassMetric": (None if metric_config is None else {
                "id": metric_config.metric_id,
                "version": metric_config.metric_version,
                "radiiM": list(metric_config.radii_m),
                "aggregate": {
                    "id": metric_summary["metric_id"],
                    "version": metric_summary["metric_version"],
                    "referencePosition": metric_summary[
                        "reference_position"],
                    "normalization": metric_summary["normalization"],
                    "higherIsBetter": metric_summary["higher_is_better"],
                    "primaryRadiusM": metric_summary["primary_radius_m"],
                    "scores": {
                        radius: value["time_normalized_mass"]
                        for radius, value in metric_summary["radii"].items()
                    },
                },
            }),
            "nKeyframes": manifest.n_keyframes,
            "nParticles": manifest.filter_config.n_particles,
            "seed": manifest.filter_config.seed,
            "matcher": manifest.matcher_version,
            "backend": manifest.filter_config.measurement_backend,
            "pi0": manifest.filter_config.pi0,
            "matcherRecall": manifest.filter_config.matcher_recall,
            "persistence": manifest.filter_config.association_persistence,
            "historyHash": (manifest.particle_history_sha256 or "")[:12],
            "maxVisibleRangeM": visible_range,
            "replayable": status.replayable,
            "nCatalog": len(manifest.landmarks),
            "runDir": str(run_dir),
        },
        "landmarks": [{"id": lm.landmark_id, "type": lm.type_key,
                       "g": _glyph_for(lm.type_key),
                       "e": round(float(e), 1), "n": round(float(n), 1)}
                      for lm, e, n in zip(manifest.landmarks, east, north)
                      if lm.landmark_id in referenced],
        "backdrop": [[int(round(float(e))), int(round(float(n)))]
                     for lm, e, n in zip(manifest.landmarks, east, north)
                     if lm.landmark_id not in referenced],
        "landmarkGeometry": landmark_geometry,
        "basemap": basemap.to_payload(),
        "satellite": _satellite_payload(satellite, notes),
        "truth": [[round(t.east_m, 1), round(t.north_m, 1)]
                  for t in data.truth],
        "health": _health_payload(data, truth_by_kf),
        "checkpoints": checkpoints,
        "measurements": measurements,
        "modes": _mode_trajectories(data),
        "tracklets": _tracklet_dossiers(data, cache, triage, bundle),
        "attribution": _attribution_payload(cache, data),
        "events": [dataclasses.asdict(e) for e in events],
        "triageSummary": forensics.triage_summary(triage),
        "ghosts": ghosts,
        "notes": notes,
        "colors": list(MODE_COLORS),
    }
