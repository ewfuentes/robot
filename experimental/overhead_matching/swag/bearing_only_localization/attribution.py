"""Log-domain attribution: why a mode gained or lost belief (design doc §7.2).

The filter multiplies each particle's weight by one factor per measurement, so
a group of particles' total mass decomposes *exactly* into per-measurement
terms. For a group g (a tracked mode, or the whole belief) with log-weights
l_i before a measurement and l'_i after:

    delta_g = logsumexp_{i in g}(l'_i) - logsumexp_{i in g}(l_i)

which is the log of the group's likelihood-averaged mass multiplier — the
number of nats that measurement gave to, or took from, that group. Because the
updates are sequential multiplications, these terms sum over a keyframe's
measurements to the group's whole log-mass change. Nothing is approximated and
nothing is attributed twice.

The number a debugging session actually wants is *relative*: a group can gain
mass in absolute terms and still lose the weight war, because normalization is
what makes weights a distribution. So each term is reported twice:

    self_nats     = delta_g                (mass this group gained)
    relative_nats = delta_g - delta_all    (mass it gained ON the others)

`relative_nats` is the quantity in the doc's example — "mode B lost 6.2 nats at
t=412: -5.8 from tracklet 7" — and it is exactly the change in log(share), so
the terms sum over tracklets and keyframes to log(share_end / share_start).

**Where §7.2 is incomplete.** The doc gives the decomposition as
`Sum_tracklets + motion/resample effects`, which holds only if a mode is a
fixed set of particles. It is not: `mode_tracker` re-clusters the belief every
keyframe and assigns mode ids by lineage, so a mode's membership changes
between updates and its share moves without any evidence arriving. Measured on
the synthetic harbour loop, the leading mode gained 10.7 nats of share over 193
keyframes while its measurement terms summed to 0.04 — essentially all of its
win came from the clusterer absorbing neighbouring particles. Attributing that
to bearings would be a fiction, and dropping it would leave the waterfall not
adding up, so re-clustering is reported as its own term. The distinction is
diagnostically useful in its own right: "this mode won because the bearings
said so" and "this mode won because the clusterer merged its rival into it" are
different findings, and only the first is evidence.

The full set of terms, which telescope exactly to the observed change because
every point at which a share can move emits one:

  tracklet   one measurement's likelihood ratio, per §7.2
  recluster  mode_tracker reassigning particles between keyframes, measured
             before that keyframe's measurements land
  settle     whatever moved a share between the last measurement and the end
             of the keyframe; ~0 unless a proposal fired
  injection  a proposal replacing belief mass (§5.5), including the
             re-clustering and evidence re-application that follow it
  resample   a mass-weighted group becoming a count-weighted one

Motion contributes nothing: `motion_update` moves particles and never touches
weights, so there is no motion term to show — a checkable fact about this
filter rather than a general one. Normalization contributes nothing either: it
subtracts a constant from every log-weight, which cancels out of every share.

Everything here is derived by observing the production filter through
`filter.RunObserver`, so the arithmetic the viewer displays is the arithmetic
the filter did (§7.5 [CONTRACT]).

Cost note: the whole-map harbour run — 379 keyframes, 44 tracklets, 4 modes —
decomposes into 2,510 rows, about 560 KB, from a 31 s instrumented replay. That
is small enough that there is no reason to compute it lazily per query, so
`compute` does the entire run once and caches it, and `attribute` is then a
dictionary lookup rather than a replay. Row count scales with keyframes times
live modes rather than with particles, so it stays flat as the filter grows.
"""

import dataclasses
import math
from pathlib import Path

import msgspec
import numpy as np
from scipy import special

from common.python.serialization import (
    MSGSPEC_STRUCT_OPTS,
    msgspec_dec_hook,
    msgspec_enc_hook,
)
from experimental.overhead_matching.swag.bearing_only_localization import (
    filter as pf,
    replay as replay_mod,
)

CACHE_NAME = "tier3_attribution.jsonl"
META_NAME = "tier3_attribution_meta.json"
# The whole-belief group. Not a mode id: mode ids are small non-negative ints
# assigned by the tracker, and -1 already means "unclustered particle".
ALL_GROUPS = -1000


class Contribution(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """One event's effect on one group's share of the belief, in nats.

    `relative_nats` is `log(share_after / share_before)`, so the terms for a
    group telescope over any keyframe range to its total log-share change.
    """
    keyframe_idx: int
    group: int  # mode id, or ALL_GROUPS for the whole belief
    term: str  # tracklet | recluster | settle | injection | resample
    # The tracklet for a "tracklet" term; "" for the structural terms.
    tracklet_id: str
    # Absolute log-mass change. For a measurement this is the group's
    # likelihood-averaged log multiplier; for the structural terms it is not
    # meaningful on its own, since mass is being moved rather than scaled.
    self_nats: float
    relative_nats: float
    # Group share of the belief before this event. Contextualizes the nats:
    # -6 nats off a group holding 0.1% of the belief is bookkeeping, off the
    # leading mode it is the story of the run.
    mass_share_before: float
    mass_share_after: float
    # 1 for the re-application pass that follows a proposal injection (§5.5).
    pass_index: int = 0


class GroupWeight(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """A group's posterior share at the end of a keyframe."""
    keyframe_idx: int
    group: int
    mass_share: float


class AttributionCache(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """A run's complete §7.2 decomposition, plus what it was computed from.

    `particle_history_sha256` is the staleness guard: it is the hash of the
    replay that produced these numbers, so a cache written against a different
    run — or against a filter whose behaviour has since changed — is detected
    rather than trusted.
    """
    particle_history_sha256: str
    scenario_name: str
    n_keyframes: int
    contributions: list[Contribution]
    group_weights: list[GroupWeight]
    verified_against_manifest: bool = False


def _group_shares(log_weight: np.ndarray, mode_id: np.ndarray) -> dict:
    """Each mode's share of the belief, plus its unnormalized log-mass.

    Returns group -> (share, log_mass). ALL_GROUPS carries the whole belief,
    whose share is 1 by construction.
    """
    total = float(special.logsumexp(log_weight))
    out = {ALL_GROUPS: (1.0, total)}
    for group in np.unique(mode_id):
        group = int(group)
        if group < 0:
            continue
        member = mode_id == group
        mass = float(special.logsumexp(log_weight[member]))
        out[group] = (math.exp(min(mass - total, 0.0)), mass)
    return out


class AttributionRecorder(pf.RunObserver):
    """Collects the §7.2 decomposition while the production filter runs.

    Works by tracking each group's share of the belief and emitting a term at
    every point where a share can move. That construction is what makes the
    decomposition complete: there is no residual to explain away, because
    nothing changes a share without passing through one of these hooks.

    Groups are read from `belief.mode_id` as the filter has it, deliberately
    not re-clustered here — "mode A believes tracklet 7 is X" has to refer to
    the mode that existed before tracklet 7 was seen, which is the convention
    `run_filter` already uses when it clusters before the measurement block.
    """

    def __init__(self):
        self.contributions: list[Contribution] = []
        self.group_weights: list[GroupWeight] = []
        # group -> share, as of the last emitted term.
        self._share: dict[int, float] = {}
        self._injected_at: set[int] = set()

    def _emit(self, keyframe_idx: int, term: str, shares: dict,
              tracklet_id: str = "", pass_index: int = 0,
              log_mass_before: dict | None = None) -> None:
        """Record the move from the tracked shares to `shares`."""
        for group, (share_after, mass_after) in shares.items():
            share_before = self._share.get(group)
            if share_before is None or share_before <= 0.0 or share_after <= 0.0:
                # A group that just appeared, or one whose share underflowed:
                # there is no ratio to report. Adopt the new share so later
                # terms are measured from somewhere real.
                continue
            self_nats = 0.0
            if log_mass_before is not None and group in log_mass_before:
                self_nats = mass_after - log_mass_before[group]
            self.contributions.append(Contribution(
                keyframe_idx=keyframe_idx, group=group, term=term,
                tracklet_id=tracklet_id, self_nats=self_nats,
                relative_nats=math.log(share_after / share_before),
                mass_share_before=share_before, mass_share_after=share_after,
                pass_index=pass_index))
        # Groups absent from `shares` have died; forget them so a later mode
        # reusing the id does not inherit a stale baseline.
        self._share = {group: share for group, (share, _) in shares.items()}

    def keyframe_start(self, keyframe_idx, belief):
        self._emit(keyframe_idx, "recluster",
                   _group_shares(belief.log_weight, belief.mode_id))

    def measurement(self, keyframe_idx, meas, log_weight_before, belief,
                    pass_index):
        before = _group_shares(log_weight_before, belief.mode_id)
        after = _group_shares(belief.log_weight, belief.mode_id)
        # The measurement is applied to a fixed partition, so `before` is the
        # right baseline even if a structural term has not been emitted since.
        self._share = {group: share for group, (share, _) in before.items()}
        self._emit(keyframe_idx, "tracklet", after,
                   tracklet_id=meas.tracklet_id, pass_index=pass_index,
                   log_mass_before={g: mass for g, (_, mass) in before.items()})

    def injection(self, keyframe_idx, event, n_injected):
        if n_injected:
            self._injected_at.add(keyframe_idx)

    def keyframe_end(self, keyframe_idx, belief, health):
        # Anything that moved a share since the last term without passing
        # through a hook of its own: proposal injection replacing mass, plus
        # the re-clustering that follows it. At a keyframe with no injection
        # this is ~0, which is itself worth being able to see.
        #
        # Named separately from the keyframe-start "recluster" because the two
        # are measured at different instants: this one is contemporaneous with
        # the HealthRecord and so is the only one `pointwise_check` may compare
        # against it. Conflating them made that check fail by 0.02 — the size
        # of one keyframe's measurement effect.
        term = "injection" if keyframe_idx in self._injected_at else "settle"
        self._emit(keyframe_idx, term,
                   _group_shares(belief.log_weight, belief.mode_id))
        for mode in health.modes:
            self.group_weights.append(GroupWeight(
                keyframe_idx=keyframe_idx, group=mode.mode_id,
                mass_share=mode.weight))

    def resample(self, keyframe_idx, log_weight_before, mode_id_before,
                 belief):
        self._share = {
            group: share for group, (share, _) in
            _group_shares(log_weight_before, mode_id_before).items()}
        self._emit(keyframe_idx, "resample",
                   _group_shares(belief.log_weight, belief.mode_id))

    def cache(self, history: pf.FilterHistory, scenario_name: str,
              n_keyframes: int, verified: bool) -> AttributionCache:
        return AttributionCache(
            particle_history_sha256=history.particle_history_sha256,
            scenario_name=scenario_name, n_keyframes=n_keyframes,
            contributions=self.contributions,
            group_weights=self.group_weights,
            verified_against_manifest=verified)


def compute(run_dir: Path, max_visible_range_m: float | None = None,
            verify: bool = True) -> tuple[AttributionCache,
                                          replay_mod.ReplayResult]:
    """Replay a run under instrumentation and decompose it.

    `verify=True` requires the replay to reproduce the run's recorded history
    hash. An attribution computed from a divergent replay would describe a run
    that never happened, which is a worse failure than having no attribution.
    """
    recorder = AttributionRecorder()
    result = replay_mod.replay(run_dir, observer=recorder,
                               max_visible_range_m=max_visible_range_m,
                               verify=verify)
    cache = recorder.cache(result.history,
                           result.inputs.manifest.scenario_name,
                           result.inputs.manifest.n_keyframes,
                           verified=bool(result.hash_match))
    return cache, result


def write_cache(run_dir: Path, cache: AttributionCache) -> Path:
    """Write the Tier-3 cache beside the run it describes."""
    run_dir = Path(run_dir)
    path = run_dir / CACHE_NAME
    with open(path, "wb") as f:
        for record in cache.contributions + cache.group_weights:
            f.write(msgspec.json.encode(record, enc_hook=msgspec_enc_hook))
            f.write(b"\n")
    (run_dir / META_NAME).write_bytes(msgspec.json.encode({
        "particle_history_sha256": cache.particle_history_sha256,
        "scenario_name": cache.scenario_name,
        "n_keyframes": cache.n_keyframes,
        "verified_against_manifest": cache.verified_against_manifest,
        "n_contributions": len(cache.contributions),
        "n_group_weights": len(cache.group_weights),
    }, enc_hook=msgspec_enc_hook))
    return path


def read_cache(run_dir: Path, expected_sha256: str | None = None
               ) -> AttributionCache | None:
    """Read a cached decomposition, or None if absent.

    Raises when `expected_sha256` disagrees with the cache: a stale
    attribution silently describing an older run is the failure mode the
    recorded hash exists to prevent.
    """
    run_dir = Path(run_dir)
    meta_path = run_dir / META_NAME
    cache_path = run_dir / CACHE_NAME
    if not meta_path.exists() or not cache_path.exists():
        return None
    meta = msgspec.json.decode(meta_path.read_bytes())
    if (expected_sha256 and meta.get("particle_history_sha256")
            and meta["particle_history_sha256"] != expected_sha256):
        raise ValueError(
            f"attribution cache in {run_dir} was computed against history "
            f"{meta['particle_history_sha256'][:12]} but the run records "
            f"{expected_sha256[:12]}; recompute it")

    contributions, group_weights = [], []
    # The msgspec tag field is "kind", and Contribution also has a semantic
    # field called kind; the tag is what selects the record type.
    by_tag = {"Contribution": (Contribution, contributions),
              "GroupWeight": (GroupWeight, group_weights)}
    for line in cache_path.read_bytes().splitlines():
        if not line.strip():
            continue
        record_type, sink = by_tag[msgspec.json.decode(line)["kind"]]
        sink.append(msgspec.json.decode(line, type=record_type,
                                        dec_hook=msgspec_dec_hook))
    return AttributionCache(
        particle_history_sha256=meta.get("particle_history_sha256", ""),
        scenario_name=meta.get("scenario_name", ""),
        n_keyframes=meta.get("n_keyframes", 0),
        contributions=contributions, group_weights=group_weights,
        verified_against_manifest=meta.get("verified_against_manifest", False))


@dataclasses.dataclass
class Waterfall:
    """A group's log-share change over a keyframe range, itemized.

    `terms` is sorted most-negative first, because the debugging question is
    almost always "what killed this mode" and the answer should be the first
    row. `residual_nats` closes the books against Tier 0, which recorded the
    mode's weight independently — so a near-zero residual is evidence the
    decomposition is right rather than merely self-consistent.
    """
    group: int
    keyframe_range: tuple[int, int]
    # (label, nats, kind); kind in {"tracklet", "recluster", "injection",
    # "resample"}.
    terms: list
    total_nats: float
    observed_nats: float | None

    @property
    def evidence_nats(self) -> float:
        """The part of the change that bearings are responsible for."""
        return sum(nats for _, nats, kind in self.terms if kind == "tracklet")

    @property
    def structural_nats(self) -> float:
        """The part caused by clustering, injection and resampling — belief
        bookkeeping rather than evidence. A mode whose rise is mostly this has
        not been confirmed by anything the vehicle saw."""
        return self.total_nats - self.evidence_nats

    @property
    def residual_nats(self) -> float | None:
        if self.observed_nats is None:
            return None
        return self.observed_nats - self.total_nats

    def report(self, top: int = 8) -> str:
        low, high = self.keyframe_range
        span = f"kf {low}" if low == high else f"kf {low}-{high}"
        verb = "lost" if self.total_nats < 0 else "gained"
        lines = [f"mode {self.group} {verb} {abs(self.total_nats):.1f} nats of "
                 f"log-share over {span} "
                 f"({self.evidence_nats:+.1f} evidence, "
                 f"{self.structural_nats:+.1f} structural)"]
        marker = {"tracklet": " ", "resample": "~", "recluster": "*",
                  "injection": "!"}
        for label, nats, kind in self.terms[:top]:
            lines.append(f"  {nats:+7.2f}  {marker.get(kind, ' ')}{label}")
        if len(self.terms) > top:
            rest = sum(n for _, n, _ in self.terms[top:])
            lines.append(f"  {rest:+7.2f}   {len(self.terms) - top} others")
        if self.residual_nats is not None:
            lines.append(f"  residual vs Tier 0: {self.residual_nats:+.3f}")
        return "\n".join(lines)


def _live_terms(cache: AttributionCache, group: int, low: int, high: int
                ) -> list:
    """A group's contribution rows over a range, with superseded rows dropped.

    At a keyframe where a proposal injected, the filter re-applies the
    measurement block; pass 0 describes weights it then recomputed, so
    summing both passes would double-count that keyframe's evidence.
    """
    rows = [c for c in cache.contributions
            if low <= c.keyframe_idx <= high and c.group == group]
    superseded = {c.keyframe_idx for c in rows if c.pass_index == 1}
    return [c for c in rows
            if c.pass_index == 1 or c.keyframe_idx not in superseded]


def attribute(cache: AttributionCache, group: int,
              keyframe_range: tuple[int, int] | None = None,
              include_structural: bool = True) -> Waterfall:
    """`attribute(mode_id, t_range) -> per-tracklet contribution series`.

    The §7.2 central viewer API. Per-tracklet terms are aggregated over the
    range, so the answer is "which tracklet did this to this mode" rather than
    a list of individual epochs; the structural terms (§7.2 note above) are
    aggregated by kind.
    """
    if keyframe_range is None:
        keyframe_range = (0, cache.n_keyframes - 1)
    low, high = keyframe_range

    by_tracklet: dict[str, float] = {}
    by_kind: dict[str, float] = {}
    for row in _live_terms(cache, group, low, high):
        if not math.isfinite(row.relative_nats):
            continue
        if row.term == "tracklet":
            by_tracklet[row.tracklet_id] = (
                by_tracklet.get(row.tracklet_id, 0.0) + row.relative_nats)
        else:
            by_kind[row.term] = by_kind.get(row.term, 0.0) + row.relative_nats

    terms = [(tracklet_id, nats, "tracklet")
             for tracklet_id, nats in by_tracklet.items()]
    if include_structural:
        label = {"recluster": "mode re-clustering",
                 "settle": "post-update re-clustering",
                 "injection": "proposal injection",
                 "resample": "resampling"}
        terms.extend((label.get(kind, kind), nats, kind)
                     for kind, nats in by_kind.items()
                     if abs(nats) > 1e-9)
    terms.sort(key=lambda term: term[1])

    return Waterfall(
        group=group, keyframe_range=(low, high), terms=terms,
        total_nats=sum(nats for _, nats, _ in terms),
        observed_nats=_observed_log_share(cache, group, low, high))


def _observed_log_share(cache: AttributionCache, group: int, low: int,
                        high: int) -> float | None:
    """A group's log-share change over the range, from Tier 0 alone.

    Derived from the HealthRecord mode weights, which the filter computed
    independently of anything here.

    Caveat worth knowing before reading `residual_nats` as an error: the two
    are measured between slightly different instants. This spans the first and
    last keyframe at which the filter *reported* the mode, and a mode is only
    reported once it clears `ModeConfig.min_mode_weight`, whereas the terms
    span observation points that bracket the requested range. On the harbour
    run that difference shows up as a few tenths of a nat on the leading mode.
    For a tight check of the arithmetic itself, use `pointwise_check`, which
    compares the two at the same instant.
    """
    shares = {w.keyframe_idx: w.mass_share for w in cache.group_weights
              if w.group == group and low <= w.keyframe_idx <= high}
    if len(shares) < 2:
        return None
    start, end = shares[min(shares)], shares[max(shares)]
    if start <= 0.0 or end <= 0.0:
        return None
    return math.log(end / start)


def pointwise_check(cache: AttributionCache) -> tuple:
    """Compare the recorder's shares against the filter's, instant by instant.

    Every keyframe-end term carries the group's share as the recorder measured
    it; every `GroupWeight` carries the same quantity as `filter._mode_records`
    computed it, in code that knows nothing about this module. Where both exist
    for the same (keyframe, group) they must agree to floating-point noise —
    and unlike `Waterfall.residual_nats`, this comparison has no range
    ambiguity to hide behind.

    Returns (max_abs_difference, n_compared).
    """
    recorded = {(w.keyframe_idx, w.group): w.mass_share
                for w in cache.group_weights}
    worst, n = 0.0, 0
    for row in cache.contributions:
        # Only the keyframe-end terms are contemporaneous with the
        # HealthRecord. A "recluster" row is measured before that keyframe's
        # measurements land, so comparing it here would be comparing two
        # different instants.
        if row.term not in ("settle", "injection"):
            continue
        key = (row.keyframe_idx, row.group)
        if key not in recorded:
            continue
        worst = max(worst, abs(row.mass_share_after - recorded[key]))
        n += 1
    return worst, n


def tracklet_series(cache: AttributionCache, tracklet_id: str,
                    group: int | None = None) -> list:
    """One tracklet's attribution series: its contribution over time.

    The §7.4 view-3 panel. `group=None` uses the whole belief, which is the
    right default for a run that stayed unimodal and misleading for one that
    did not — hence the parameter.
    """
    want = ALL_GROUPS if group is None else group
    rows = [c for c in cache.contributions
            if c.tracklet_id == tracklet_id and c.group == want
            and c.term == "tracklet"]
    superseded = {c.keyframe_idx for c in rows if c.pass_index == 1}
    rows = [c for c in rows
            if c.pass_index == 1 or c.keyframe_idx not in superseded]
    rows.sort(key=lambda c: c.keyframe_idx)
    return rows


def death_waterfall(cache: AttributionCache, mode_events: list,
                    lookback_keyframes: int = 10) -> dict:
    """Pre-computed waterfalls for every mode death (§7.4 view 4).

    "Why did the right mode die" should be one click, not one replay, so the
    window ending at each death event is decomposed up front. The lookback is
    a window, not a claim: a mode usually dies from evidence spread over
    several keyframes rather than from a single one.
    """
    out = {}
    for event in mode_events:
        if event.kind != "death":
            continue
        low = max(0, event.keyframe_idx - lookback_keyframes)
        out[event.mode_id] = attribute(cache, event.mode_id,
                                      (low, event.keyframe_idx))
    return out
