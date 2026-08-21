"""Derived findings a run log implies but does not state (design doc §7.3/§7.4).

Two things live here, both computed from Tier 0/1 alone — no replay, so they
are available for any run directory including ones this build cannot faithfully
replay:

**Auto-bookmarked events (§7.3).** The doc lists ten event kinds and says they
should be detected online and trigger checkpoints. Four of them are: mode
birth/death/merge and proposal firings, which the filter logs. The rest —
MAP jumps, ESS crashes, resample storms, association flips, null-share spikes,
gate rejections — are derivable after the fact from the Tier-0 stream, and are
derived here rather than left undone. They are tagged `derived` so the strip
never implies the filter noticed something at the time that it did not.

**Truth-privileged triage (§7.4 view 3).** The tracklet inspector's headline
question is "did the tracker, the matcher, or the filter get this wrong?" With
GPS truth available that is largely decidable, but *not* by the obvious method.

The obvious method — reproject one bearing from the truth pose and see which
catalog landmark it points at — is degenerate on a whole-map catalog and was
tried first. With 13,210 landmarks over a 23 x 21 km region the angular density
from any pose is about 37 landmarks per degree, so the nearest-in-angle landmark
is always a hundredth of a degree away and is an arbitrary catalog row rather
than the observed object. Measured on the harbour run it produced 0.0 deg
"bearing error" for 44 of 44 tracklets and a uniform matcher-fault verdict:
a number that looked like a finding and meant nothing.

What breaks the degeneracy is the same thing that lets the filter work at all:
**one landmark must explain every epoch of a tracklet**. So the test here is
per-candidate rather than per-bearing. For a candidate landmark L, compute the
angular residual between each measured bearing and the direction from that
keyframe's truth pose to L, and take the RMS over the tracklet's epochs. A
tracklet observed across real vessel motion admits only landmarks near the true
one; a tracklet seen once admits a whole ray of them, which is honest ambiguity
and is reported as such via `n_consistent_catalog`.

  tracker-fault  no catalog landmark at all explains this tracklet's bearings
                 from the truth poses — the bearings are mutually inconsistent
                 or wrongly referenced (drifted mask, heading compensation,
                 panorama seam)
  no-evidence    the bearings are explicable but the table endorses nothing;
                 the filter had geometry and no semantics. Silence, not error
  matcher-fault  the bearings are explicable but nothing the table endorses
                 explains them. `anti-evidence` when the table's *best* claim
                 is geometrically wrong: worse than silence, since it argues
                 for a false fix
  filter-fault   bearings explicable, matcher endorsed something that explains
                 them, and the posterior still put its mass elsewhere
  consistent     all three agree

`n_consistent_catalog` is the ambiguity measure that keeps the verdict honest:
it counts how many catalog landmarks fit within tolerance. When it is large the
geometry is not discriminating and a "matcher-fault" verdict means "the matcher
did not pick one of the many that fit", which is a much weaker claim than the
same verdict on a tracklet with a unique geometric explanation.

The tolerance scales with the tracklet's own declared precision — a bearing
with sigma 25 deg genuinely cannot discriminate, and holding it to the same
standard as a sigma 1 deg bearing would invent faults.

[HAZARD] Everything under `triage` uses ground truth and therefore cannot
appear in, or support, any claim about how well the system localizes. It is a
debugging instrument. Callers render it behind an explicit truth-privileged
marking; `TrackletTriage.truth_privileged` is True so that marking cannot be
forgotten by accident.

This module also owns the SHARED table-reading helpers (`clipped_log_lr`,
`table_lookup`, `endorsed_entries`): "endorsed" has one definition, used by the
triage, the viewer payload and the forensics CLI alike — the old tree carried
three drifting copies of the same clip-and-compare.
"""

import dataclasses
import math
from collections import Counter

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo

# A candidate landmark "explains" a tracklet when the RMS angular residual over
# its epochs is under a tolerance. The tolerance is `sigma_scale` times the
# tracklet's median declared bearing sigma, clamped: the floor keeps a very
# confident tracklet from being condemned by ordinary catalog position error,
# and the ceiling keeps a hopeless sigma from making every landmark acceptable.
RESIDUAL_SIGMA_SCALE = 3.0
MIN_RESIDUAL_TOLERANCE_DEG = 5.0
MAX_RESIDUAL_TOLERANCE_DEG = 25.0
# Above this many catalog landmarks fitting within tolerance, the geometry is
# not identifying anything and the verdict is reported as weakly supported.
AMBIGUOUS_CATALOG_FITS = 25
# Reported association mass below this is not a claim about identity.
MEANINGFUL_RESPONSIBILITY = 0.05
# An entry counts as endorsed only when its clipped score exceeds the table's
# clipped default by more than this — strict inequality with float headroom,
# shared by every consumer of "endorsed".
ENDORSEMENT_EPS = 1e-12


# ---------------------------------------------------------------------------
# Compatibility-table reading, defined once
# ---------------------------------------------------------------------------

def clipped_log_lr(table, log_lr: float) -> float:
    """A log-LR as the filter consumes it: clipped to the table's bounds."""
    return min(max(log_lr, table.clip_lo), table.clip_hi)


def table_lookup(table, landmark_id: str):
    """A landmark's clipped log-LR under a table, and whether it is endorsed.

    Endorsed means "scored above what the table gives an unlisted landmark":
    the matcher affirmatively vouched for it rather than merely not excluding
    it. The comparison is on clipped values because that is what the filter
    uses.
    """
    if table is None:
        return None, False
    default = clipped_log_lr(table, table.default_log_lr)
    for entry in table.entries:
        if entry.landmark_id == landmark_id:
            clipped = clipped_log_lr(table, entry.log_lr)
            return clipped, clipped > default + ENDORSEMENT_EPS
    return default, False


def endorsed_entries(table) -> dict:
    """landmark_id -> clipped log-LR, for entries the matcher vouches for."""
    if table is None:
        return {}
    default = clipped_log_lr(table, table.default_log_lr)
    out = {}
    for entry in table.entries:
        clipped = clipped_log_lr(table, entry.log_lr)
        if clipped > default + ENDORSEMENT_EPS:
            out[entry.landmark_id] = clipped
    return out


@dataclasses.dataclass
class Event:
    """One entry in the run's debugging table of contents (§7.3)."""
    keyframe_idx: int
    kind: str
    severity: str  # "info" | "warn" | "alarm"
    label: str
    detail: str
    # "logged" — the filter emitted it at the time. "derived" — reconstructed
    # from Tier 0 afterwards. Kept distinct so the strip never implies the
    # filter reacted to something it never saw.
    source: str


def _series(health, attr):
    return np.array([getattr(r, attr) for r in health], dtype=float)


def derive_events(data, *, map_jump_sigma: float = 3.0,
                  map_jump_floor_m: float = 100.0,
                  ess_crash_frac: float = 0.05,
                  resample_storm_window: int = 10,
                  resample_storm_count: int = 8,
                  null_spike: float = 0.7) -> list:
    """The §7.3 event index: logged events plus the derivable ones."""
    events: list[Event] = []
    health = data.health
    if not health:
        return events

    for event in data.proposal_events:
        injected = event.n_injected
        if injected:
            severity, label = "alarm", f"proposal #{event.event_id} injected"
            detail = (f"{event.trigger} trigger, {injected} particles from "
                      f"{event.n_hypotheses} hypotheses")
        else:
            severity, label = "warn", f"proposal #{event.event_id} rejected"
            detail = (f"{event.trigger} trigger; evidence gate refused "
                      f"{event.n_hypotheses} hypotheses")
            if event.gate_best_hypothesis_nats is not None:
                detail += (f" (best {event.gate_best_hypothesis_nats:.2f} vs "
                           f"reference {event.gate_reference_nats:.2f} nats)")
        events.append(Event(event.keyframe_idx, "proposal", severity, label,
                            detail, "logged"))

    for event in data.mode_events:
        severity = {"birth": "info", "merge": "info",
                    "death": "warn"}.get(event.kind, "info")
        detail = ", ".join(f"{k}={v}" for k, v in event.detail.items())
        if event.parent_mode_ids:
            detail = (f"parents {event.parent_mode_ids}"
                      + (f"; {detail}" if detail else ""))
        events.append(Event(event.keyframe_idx, f"mode_{event.kind}", severity,
                            f"mode {event.mode_id} {event.kind}", detail,
                            "logged"))

    # MAP jump: the estimate moved further in one keyframe than its own
    # claimed uncertainty allows. The floor keeps a confidently-tracking
    # filter (small sigma) from flagging ordinary motion.
    map_e = _series(health, "map_east_m")
    map_n = _series(health, "map_north_m")
    sigma = _series(health, "position_std_m")
    steps = np.hypot(np.diff(map_e), np.diff(map_n))
    for i, step in enumerate(steps):
        threshold = max(map_jump_sigma * sigma[i], map_jump_floor_m)
        if step > threshold:
            events.append(Event(
                health[i + 1].keyframe_idx, "map_jump", "alarm",
                f"MAP jumped {step / 1000:.1f} km" if step > 1000
                else f"MAP jumped {step:.0f} m",
                f"reported sigma was {sigma[i]:.0f} m "
                f"({step / max(sigma[i], 1e-9):.1f} sigma)", "derived"))

    # ESS crash: the belief has no viable hypotheses left and resampling
    # cannot invent one. Reported on the leading edge only, so a long
    # starvation is one bookmark rather than a hundred.
    n_particles = data.manifest.filter_config.n_particles
    ess = _series(health, "ess")
    crashed = ess < ess_crash_frac * n_particles
    for i, is_crashed in enumerate(crashed):
        if is_crashed and (i == 0 or not crashed[i - 1]):
            events.append(Event(
                health[i].keyframe_idx, "ess_crash", "alarm",
                f"ESS crashed to {ess[i]:.0f}",
                f"{ess[i] / n_particles * 100:.1f}% of {n_particles} "
                f"particles carry the belief", "derived"))

    # Resample storm: resampling nearly every keyframe means the weights are
    # being destroyed as fast as they are earned.
    resampled = np.array([r.resampled for r in health], dtype=int)
    if len(resampled) >= resample_storm_window:
        window = np.convolve(resampled,
                             np.ones(resample_storm_window, dtype=int), "valid")
        in_storm = window >= resample_storm_count
        for i, flag in enumerate(in_storm):
            if flag and (i == 0 or not in_storm[i - 1]):
                events.append(Event(
                    health[i].keyframe_idx, "resample_storm", "warn",
                    f"resampled {window[i]}/{resample_storm_window} keyframes",
                    "weights are being rebuilt from scratch nearly every "
                    "keyframe", "derived"))

    # Null-share spike: the semantic evidence became unusable — nothing the
    # matcher offered explains the bearings under the current belief.
    for record in health:
        whole = [a for a in record.associations if a.mode_id is None]
        if not whole:
            continue
        mean_null = float(np.mean([a.null_share for a in whole]))
        if mean_null >= null_spike:
            events.append(Event(
                record.keyframe_idx, "null_spike", "warn",
                f"null share {mean_null:.2f}",
                f"{len(whole)} measurement(s) mostly explained as clutter",
                "derived"))

    # Association flip: a tracklet's best-guess identity changed. Under §5.3
    # persistence a tracklet is one physical object, so a flip is either the
    # filter revising a mistake or making one.
    previous: dict[tuple, str] = {}
    for record in health:
        for assoc in record.associations:
            if not assoc.responsibilities:
                continue
            best = max(assoc.responsibilities,
                       key=assoc.responsibilities.get)
            if assoc.responsibilities[best] < MEANINGFUL_RESPONSIBILITY:
                continue
            key = (assoc.mode_id, assoc.tracklet_id)
            if key in previous and previous[key] != best:
                scope = ("belief" if assoc.mode_id is None
                         else f"mode {assoc.mode_id}")
                events.append(Event(
                    record.keyframe_idx, "association_flip", "warn",
                    f"{assoc.tracklet_id} flipped identity",
                    f"{scope}: {previous[key]} -> {best} "
                    f"({assoc.responsibilities[best] * 100:.0f}%)", "derived"))
            previous[key] = best

    events.sort(key=lambda e: (e.keyframe_idx, e.kind))
    return events


@dataclasses.dataclass
class CandidateFit:
    """How well one catalog landmark explains a whole tracklet's bearings.

    Computed from the truth poses, so this is "could this tracklet have been
    this landmark" answered geometrically, independent of what the matcher
    thought. `rms_deg` over all epochs is the discriminating number: a single
    epoch admits everything along a ray, several epochs across real motion
    admit only a neighbourhood of the true object.
    """
    landmark_id: str
    rms_deg: float
    max_deg: float
    median_range_m: float
    log_lr: float | None = None
    endorsed: bool = False

    def explains(self, tolerance_deg: float) -> bool:
        return self.rms_deg <= tolerance_deg


@dataclasses.dataclass
class EpochTriage:
    """One tracklet epoch, judged against truth."""
    keyframe_idx: int
    bearing_body_deg: float
    sigma_deg: float
    # World bearing the truth heading implies for this body bearing.
    true_world_bearing_deg: float
    # Residual of the tracklet's best-fitting catalog landmark at this epoch,
    # and of the matcher's best endorsed claim. Per-epoch so a tracklet that
    # was fine and then drifted is visible as a trend rather than an average.
    best_fit_residual_deg: float | None
    top_endorsed_residual_deg: float | None
    # What the filter did with this epoch.
    filter_top_id: str | None
    filter_top_share: float
    best_fit_share: float
    null_share: float
    surprise_share: float


@dataclasses.dataclass
class TrackletTriage:
    """A tracklet's whole life, judged against truth (§7.4 view 3).

    [HAZARD] Truth-privileged. See the module docstring.
    """
    tracklet_id: str
    n_epochs: int
    keyframe_span: tuple
    verdict: str
    # Tolerance actually applied, which scales with this tracklet's declared
    # precision — reported so a verdict can be read against the standard it
    # was held to.
    tolerance_deg: float
    median_sigma_deg: float
    # Geometry: the best explanation available anywhere in the catalog, the
    # best among endorsed entries, and the matcher's highest-scoring claim
    # regardless of geometry.
    best_fit: CandidateFit | None
    best_endorsed_fit: CandidateFit | None
    top_endorsed_fit: CandidateFit | None
    # How many catalog landmarks explain the bearings within tolerance. Large
    # means the geometry is not identifying anything, so the verdict is a weak
    # claim; this is the honest counterweight to a dense catalog.
    n_consistent_catalog: int
    ambiguous: bool
    anti_evidence: bool
    table_status: str
    n_table_entries: int
    n_endorsed: int
    # Mass the filter ever put on a geometrically-consistent endorsed entry.
    best_filter_share: float
    epochs: list
    truth_privileged: bool = True

    @property
    def geometry_explicable(self) -> bool:
        return (self.best_fit is not None
                and self.best_fit.explains(self.tolerance_deg))


def _residual_matrix(epochs, truth_by_kf, catalog):
    """(n_epochs, n_catalog) absolute angular residuals, in degrees.

    Each row asks, for one epoch, "how far off would each catalog landmark be
    if this bearing were pointing at it, given the truth pose". Landmarks
    beyond their visibility radius are set to infinity: they were not what was
    seen, so they may not be offered as an explanation, which is the same
    constraint the proposal applies.
    """
    east = np.array([truth_by_kf[m.anchor_keyframe_idx].east_m
                     for m in epochs])
    north = np.array([truth_by_kf[m.anchor_keyframe_idx].north_m
                      for m in epochs])
    bearings, ranges = catalog.bearings_from(east, north)
    observed_deg = geo.body_to_world_bearing_deg(
        np.array([truth_by_kf[m.anchor_keyframe_idx].heading_deg
                  for m in epochs]),
        np.array([m.bearing_body_deg for m in epochs]))
    residual = np.abs(geo.circular_diff_deg(np.degrees(bearings),
                                            observed_deg[:, None]))
    visible = ranges <= catalog.max_visible_range_m[None, :]
    return np.where(visible, residual, np.inf), ranges


def _fit(landmark_id, residuals, ranges, index, table) -> CandidateFit:
    column = residuals[:, index]
    finite = column[np.isfinite(column)]
    if finite.size == 0:
        rms = max_deg = float("inf")
    else:
        rms = float(np.sqrt(np.mean(np.square(finite))))
        max_deg = float(finite.max())
    log_lr, endorsed = table_lookup(table, landmark_id)
    return CandidateFit(
        landmark_id=landmark_id, rms_deg=rms, max_deg=max_deg,
        median_range_m=float(np.median(ranges[:, index])),
        log_lr=log_lr, endorsed=endorsed)


def triage_tracklets(data, catalog) -> dict:
    """Per-tracklet culpability, using GPS truth. Empty without truth."""
    if not data.truth:
        return {}
    truth_by_kf = {t.keyframe_idx: t for t in data.truth}

    assoc_by_key = {}
    for record in data.health:
        for assoc in record.associations:
            if assoc.mode_id is None:
                assoc_by_key[(assoc.tracklet_id,
                              assoc.anchor_keyframe_idx)] = assoc

    by_tracklet: dict[str, list] = {}
    for meas in data.measurements:
        if meas.anchor_keyframe_idx in truth_by_kf:
            by_tracklet.setdefault(meas.tracklet_id, []).append(meas)

    out = {}
    for tracklet_id, epochs in sorted(by_tracklet.items()):
        epochs = sorted(epochs, key=lambda m: m.anchor_keyframe_idx)
        table = data.tables.get(tracklet_id)
        out[tracklet_id] = _triage_tracklet(
            tracklet_id, epochs, truth_by_kf, catalog, table, assoc_by_key)
    return out


def _triage_tracklet(tracklet_id, epochs, truth_by_kf, catalog, table,
                     assoc_by_key) -> TrackletTriage:
    sigmas = [math.degrees(1.0 / math.sqrt(max(m.kappa, 1e-9)))
              for m in epochs]
    median_sigma = float(np.median(sigmas))
    tolerance = min(max(RESIDUAL_SIGMA_SCALE * median_sigma,
                        MIN_RESIDUAL_TOLERANCE_DEG),
                    MAX_RESIDUAL_TOLERANCE_DEG)

    residuals, ranges = _residual_matrix(epochs, truth_by_kf, catalog)
    # RMS over the epochs where the landmark was within visibility. A landmark
    # out of range at every epoch has no finite residual at all and must score
    # infinity rather than an empty mean.
    visible = np.isfinite(residuals)
    n_visible = visible.sum(axis=0)
    squared = np.where(visible, residuals, 0.0) ** 2
    rms = np.where(n_visible > 0,
                   np.sqrt(squared.sum(axis=0) / np.maximum(n_visible, 1)),
                   np.inf)

    best_index = int(np.argmin(rms))
    best_fit = (_fit(catalog.landmark_ids[best_index], residuals, ranges,
                     best_index, table)
                if np.isfinite(rms[best_index]) else None)
    n_consistent = int(np.sum(rms <= tolerance))

    endorsed = endorsed_entries(table)
    endorsed_fits = [
        _fit(landmark_id, residuals, ranges, catalog.index_of(landmark_id),
             table)
        for landmark_id in endorsed if landmark_id in catalog]
    best_endorsed_fit = (min(endorsed_fits, key=lambda f: f.rms_deg)
                         if endorsed_fits else None)
    top_endorsed_fit = (max(endorsed_fits,
                            key=lambda f: (f.log_lr if f.log_lr is not None
                                           else -math.inf))
                        if endorsed_fits else None)

    # Mass the filter ever put on a geometrically-consistent endorsed entry:
    # this is the "and the filter believed it" half of a consistent verdict.
    consistent_ids = {f.landmark_id for f in endorsed_fits
                      if f.explains(tolerance)}
    best_filter_share = 0.0
    epoch_rows = []
    for row, meas in enumerate(epochs):
        assoc = assoc_by_key.get((tracklet_id, meas.anchor_keyframe_idx))
        responsibilities = assoc.responsibilities if assoc else {}
        share = max((responsibilities.get(landmark_id, 0.0)
                     for landmark_id in consistent_ids), default=0.0)
        best_filter_share = max(best_filter_share, share)
        filter_top_id = (max(responsibilities, key=responsibilities.get)
                         if responsibilities else None)
        best_residual = (float(residuals[row, best_index])
                         if best_fit is not None else None)
        top_residual = None
        if top_endorsed_fit is not None:
            value = residuals[row, catalog.index_of(
                top_endorsed_fit.landmark_id)]
            top_residual = float(value) if math.isfinite(value) else None
        epoch_rows.append(EpochTriage(
            keyframe_idx=meas.anchor_keyframe_idx,
            bearing_body_deg=meas.bearing_body_deg,
            sigma_deg=sigmas[row],
            true_world_bearing_deg=float(geo.body_to_world_bearing_deg(
                truth_by_kf[meas.anchor_keyframe_idx].heading_deg,
                meas.bearing_body_deg)),
            best_fit_residual_deg=(best_residual
                                   if best_residual is not None
                                   and math.isfinite(best_residual) else None),
            top_endorsed_residual_deg=top_residual,
            filter_top_id=filter_top_id,
            filter_top_share=float(responsibilities.get(filter_top_id, 0.0)
                                   if filter_top_id else 0.0),
            best_fit_share=float(share),
            null_share=float(assoc.null_share) if assoc else 1.0,
            surprise_share=float(assoc.surprise_share) if assoc else 0.0))

    explicable = best_fit is not None and best_fit.explains(tolerance)
    if not explicable:
        verdict = "tracker-fault"
    elif not endorsed:
        verdict = "no-evidence"
    elif best_endorsed_fit is None or not best_endorsed_fit.explains(tolerance):
        verdict = "matcher-fault"
    elif best_filter_share < MEANINGFUL_RESPONSIBILITY:
        verdict = "filter-fault"
    else:
        verdict = "consistent"

    # The matcher's best claim is geometrically wrong: it is not merely absent
    # but arguing for a false fix. Only meaningful when the geometry could
    # discriminate in the first place.
    anti_evidence = bool(
        explicable and top_endorsed_fit is not None
        and not top_endorsed_fit.explains(tolerance))

    return TrackletTriage(
        tracklet_id=tracklet_id, n_epochs=len(epochs),
        keyframe_span=(epochs[0].anchor_keyframe_idx,
                       epochs[-1].anchor_keyframe_idx),
        verdict=verdict, tolerance_deg=tolerance,
        median_sigma_deg=median_sigma, best_fit=best_fit,
        best_endorsed_fit=best_endorsed_fit,
        top_endorsed_fit=top_endorsed_fit,
        n_consistent_catalog=n_consistent,
        ambiguous=n_consistent > AMBIGUOUS_CATALOG_FITS,
        anti_evidence=anti_evidence,
        table_status=(table.status if table else "no table"),
        n_table_entries=(len(table.entries) if table else 0),
        n_endorsed=len(endorsed), best_filter_share=best_filter_share,
        epochs=epoch_rows)


def triage_summary(triage: dict) -> str:
    """The run-level error budget, per §7.4's "who is culpable" question."""
    if not triage:
        return "no ground truth: triage unavailable"
    counts = Counter(t.verdict for t in triage.values())
    total = len(triage)
    parts = [f"{counts.get(v, 0)}/{total} {v}"
             for v in ("consistent", "tracker-fault", "no-evidence",
                       "matcher-fault", "filter-fault") if counts.get(v)]
    extra = []
    anti = sum(1 for t in triage.values() if t.anti_evidence)
    if anti:
        extra.append(f"{anti} carry anti-evidence")
    empty = sum(1 for t in triage.values() if t.n_table_entries == 0)
    if empty:
        extra.append(f"{empty} have an empty table")
    ambiguous = sum(1 for t in triage.values() if t.ambiguous)
    if ambiguous:
        extra.append(f"{ambiguous} geometrically ambiguous, so their verdict "
                     f"is a weak claim")
    return "; ".join(parts) + ((" (" + ", ".join(extra) + ")") if extra else "")
