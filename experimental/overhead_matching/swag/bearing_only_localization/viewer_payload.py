"""Assemble everything the viewer renders, from a run directory and its context.

One builder, two consumers: `viewer.py` inlines this into a self-contained page
and `viewer_server.py` serves it over HTTP. They therefore cannot disagree about
what a run means, which is the same reason §7.5 wants the replay path to be the
production filter.

Inputs, in decreasing order of necessity:

  run_dir      required. Tier 0/1 + checkpoints + the manifest.
  attribution  the Tier-3 cache if present (`runlog attribute` writes it).
               Absent, every view still works except the waterfalls.
  sources_dir  the object-track run, for crops and matcher payload.
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

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    attribution as attribution_mod,
    basemap as basemap_mod,
    forensics,
    geodesy,
    replay as replay_mod,
    run_log,
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


def _clipped(table, log_lr: float) -> float:
    return min(max(log_lr, table.clip_lo), table.clip_hi)


def _endorsed_ids(table) -> dict:
    """landmark_id -> clipped log-LR, for entries the matcher vouches for."""
    default = _clipped(table, table.default_log_lr)
    return {entry.landmark_id: _clipped(table, entry.log_lr)
            for entry in table.entries
            if _clipped(table, entry.log_lr) > default + 1e-12}


def referenced_landmark_ids(data) -> set:
    """Landmarks the run actually talks about.

    Everything else renders as backdrop: a whole-map catalog (13,210 rows on
    the harbour runs) drawn as labelled glyphs is unreadable and unresponsive.
    """
    ids = set()
    for table in data.tables.values():
        ids.update(_endorsed_ids(table))
    for event in data.proposal_events:
        for landmark_ids in event.hypothesis_landmark_ids:
            ids.update(landmark_ids)
    for record in data.health:
        for assoc in record.associations:
            ids.update(landmark_id
                       for landmark_id, value in assoc.responsibilities.items()
                       if value > 1e-3)
    return ids


def _health_payload(data, truth_by_kf) -> list:
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
            entry["truthE"] = round(truth.east_m, 1)
            entry["truthN"] = round(truth.north_m, 1)
            entry["err"] = round(math.hypot(record.mean_east_m - truth.east_m,
                                            record.mean_north_m - truth.north_m),
                                 1)
            entry["mapErr"] = round(
                math.hypot(record.map_east_m - truth.east_m,
                           record.map_north_m - truth.north_m), 1)
            entry["headingErr"] = round(abs(math.degrees(float(geodesy.wrap_rad(
                math.radians(record.mean_heading_deg)
                - math.radians(truth.heading_deg))))), 2)
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
    for tracklet_id in sorted(epochs_by_tracklet):
        epochs = sorted(epochs_by_tracklet[tracklet_id],
                        key=lambda m: m.anchor_keyframe_idx)
        table = data.tables.get(tracklet_id)
        entry = {
            "id": tracklet_id,
            "epochs": [{
                "kf": m.anchor_keyframe_idx,
                "bearing": round(m.bearing_body_deg, 2),
                "sigma": round(math.degrees(
                    1.0 / math.sqrt(max(m.kappa, 1e-9))), 2),
            } for m in epochs],
            "assoc": assoc_by_tracklet.get(tracklet_id, []),
        }
        if table is not None:
            default = _clipped(table, table.default_log_lr)
            ranked = sorted(table.entries, key=lambda e: -e.log_lr)
            entry["table"] = {
                "status": table.status,
                "matcher": table.matcher_version,
                "default": round(default, 3),
                "clipLo": table.clip_lo, "clipHi": table.clip_hi,
                "nEntries": len(table.entries),
                "nEndorsed": len(_endorsed_ids(table)),
                "entries": [{
                    "lm": e.landmark_id,
                    "lr": round(_clipped(table, e.log_lr), 3),
                    "raw": round(e.log_lr, 3),
                    "endorsed": _clipped(table, e.log_lr) > default + 1e-12,
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
        source = bundle.get(tracklet_id) if bundle else None
        if source is not None:
            entry["source"] = {
                "name": source.best_name,
                "tags": source.best_tags,
                "nameContested": source.name_contested,
                "description": source.description,
                "features": list(source.features),
                "unresolved": source.unresolved,
                "nSupports": source.n_supports,
                "span": list(source.keyframe_span or ()),
                "trackIds": list(source.track_ids),
                "handoffs": [{"with": w, "gap": g, "status": s}
                             for w, g, s in source.handoff_proposals[:6]],
                "noMatchRate": (round(source.no_match_rate, 3)
                                if source.no_match_rate is not None else None),
                "uniqueness": source.median_uniqueness,
                "nChunks": source.n_matcher_chunks,
                "thumb": source.thumbnail_data_uri,
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
            "bearing": round(e.bearing_body_deg, 2),
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
            ghost = run_log.read_run(ghost_dir)
        except Exception as exc:  # noqa: BLE001 - a bad ghost is not fatal
            notes.append(f"ghost {ghost_dir}: unreadable ({exc})")
            continue
        truth_by_kf = {t.keyframe_idx: t for t in ghost.truth}
        errors = [math.hypot(r.map_east_m - truth_by_kf[r.keyframe_idx].east_m,
                             r.map_north_m - truth_by_kf[r.keyframe_idx].north_m)
                  for r in ghost.health if r.keyframe_idx in truth_by_kf]
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
            "finalErr": round(errors[-1], 1) if errors else None,
            "medianErr": round(float(np.median(errors)), 1) if errors else None,
            "nModes": len(ghost.health[-1].modes) if ghost.health else 0,
        })
    return ghosts, notes


def build(run_dir: Path, sources_dir: Path | None = None,
          feather: Path | None = None, ghost_dirs=(),
          max_particles: int = MAX_PARTICLES_PER_FRAME,
          max_visible_range_m: float | None = None,
          embed_thumbnails: bool = True,
          with_basemap: bool = True) -> dict:
    """The whole payload. See the module docstring for what is optional."""
    run_dir = Path(run_dir)
    data = run_log.read_run(run_dir)
    manifest = data.manifest
    notes: list[str] = []

    status = replay_mod.replayability(run_dir)
    if not status.replayable:
        notes.append("this run is not faithfully replayable by this build: "
                     + status.report().replace("\n", " ").strip())
    notes.extend(status.notes)

    frame = geodesy.RegionFrame(manifest.anchor_lat_deg,
                               manifest.anchor_lon_deg)
    visible_range = (max_visible_range_m
                     if max_visible_range_m is not None
                     else manifest.max_visible_range_m
                     or replay_mod.FALLBACK_MAX_VISIBLE_RANGE_M)
    catalog = replay_mod.build_catalog(manifest, visible_range, frame)
    east, north = _landmark_positions(manifest, frame)

    cache = None
    try:
        cache = attribution_mod.read_cache(
            run_dir, expected_sha256=manifest.particle_history_sha256 or None)
    except ValueError as exc:
        notes.append(str(exc))
    if cache is None:
        notes.append("no Tier-3 attribution cache: run `runlog attribute` on "
                     "this run to enable the waterfall and per-tracklet "
                     "attribution series")

    triage = forensics.triage_tracklets(data, catalog)
    events = forensics.derive_events(data)

    tracklet_ids = {m.tracklet_id for m in data.measurements}
    bundle = sources_mod.load(sources_dir, tracklet_ids,
                              embed_thumbnails=embed_thumbnails)
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

    bounds = None
    if east.size:
        margin = 2000.0
        bounds = (float(east.min()) - margin, float(east.max()) + margin,
                  float(north.min()) - margin, float(north.max()) + margin)
    basemap = (basemap_mod.build(feather, manifest.anchor_lat_deg,
                                 manifest.anchor_lon_deg, bounds_enu=bounds)
               if with_basemap else basemap_mod.Basemap([], None, ()))
    notes.extend(basemap.notes)

    ghosts, ghost_notes = _ghost_payload(ghost_dirs)
    notes.extend(ghost_notes)

    measurements: dict[str, list] = {}
    for meas in data.measurements:
        table = data.tables.get(meas.tracklet_id)
        endorsed = _endorsed_ids(table) if table else {}
        top = max(endorsed, key=endorsed.get) if endorsed else None
        measurements.setdefault(str(meas.anchor_keyframe_idx), []).append({
            "trk": meas.tracklet_id,
            "bearing": round(meas.bearing_body_deg, 2),
            "sigma": round(math.degrees(
                1.0 / math.sqrt(max(meas.kappa, 1e-9))), 2),
            # The matcher's best claim, so the map can red-flag the case where
            # it disagrees with where that landmark actually lies (§7.4).
            "topLm": top,
            "topLr": round(endorsed[top], 2) if top else None,
            "nEndorsed": len(endorsed),
        })

    return {
        "run": {
            "scenario": manifest.scenario_name,
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
        "basemap": basemap.to_payload(),
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
