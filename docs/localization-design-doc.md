# Design Document: Semantic Bearing-Based Localization & Matching Infrastructure

**Status:** Draft for refinement against codebase
**Audience:** Implementing agent + team. This document is deliberately high-level: it fixes the architecture, contracts, invariants, tests, and tooling requirements, and leaves data structures, module layout, and naming to be aligned with the existing codebase. Where a decision is grounded in the literature, the intuition is stated inline so refactors don't accidentally violate the reason the decision exists.

**Note to the refining agent:** Sections marked `[CONTRACT]` are load-bearing — changing them changes system behavior or breaks the seam between components; do not weaken them without flagging. Sections marked `[ADAPT]` are expected to be reshaped to fit existing code.

---

## 1. Problem statement

A vehicle moves through a large outdoor area (initial target: Boston Harbor, ~10² km²; design must not preclude ~625 km² regions — the scale of the existing histogram-filter evaluation regions). It observes **far-field landmarks**: objects potentially kilometers away, extracted from panoramic video by a VLM, with **no range measurement** — each observation is a bearing (unit direction) plus a semantic payload. Upstream perception (out of scope here) produces **tracklets**: temporally consistent tracks of a single physical object across frames, already consistency-filtered. In this codebase, "upstream" is the `swag/landmark_filtering` pipeline: ingest → heuristic filters → per-frame Hungarian association with dustbin (`tracking.py`) → bearing-only triangulation (`triangulation.py`), whose "far" observability class is precisely the far-field tracklet population this filter consumes. Relative odometry between sequential states is available with covariance.

The system must maintain a **belief over vehicle pose** by corresponding these tracklets against a **catalog of known-position landmarks** (OSM + NOAA ENC), where correspondence is uncertain, many observed objects are not in the catalog, and some catalog objects will never be observed.

Two boxes, one seam:

- **Localization box** (primary focus): Bayes filter over pose with an association-marginalized bearing likelihood; mixture-MCL with a resection-based proposal for global initialization and recovery.
- **Semantic matcher** (behind a hard interface): produces calibrated compatibility scores between a tracklet's semantic payload and catalog entries. Multiple implementations will be trialed (trained pairwise classifier — current approach; Fellegi–Sunter; OT over attribute sets; LLM-adjudicated). The localization box must never know which is running.

---

## 2. Architecture overview

```
                         ┌────────────────────────────────────────┐
  video/panos ──►  VLM + │ tracker (SAM3/CoTracker) → tracklets   │  (upstream, out of scope)
                         └───────────────┬────────────────────────┘
                                         │ TrackletMsg (keyframe rate)
        odometry (SE(3) deltas + cov) ───┤
                                         ▼
   ┌──────────────┐  candidates   ┌──────────────────┐
   │ Landmark     │◄──────────────│  Localization    │──► BeliefMsg (particles/GMM, MAP+cov)
   │ Catalog Svc  │──────────────►│  Box (filter)    │──► AssociationPosteriorMsg (per tracklet)
   └──────┬───────┘  landmarks    └────────▲─────────┘──► EventLog / RunLog (§7)
          │ prepare_catalog()              │ CompatibilityTable
          ▼                                │
   ┌─────────────────────────────────────────────┐
   │ Semantic Matcher (swappable, own process ok) │
   └─────────────────────────────────────────────┘
```

Key structural fact `[CONTRACT]`: **semantic compatibility depends only on (tracklet, landmark), never on the pose hypothesis.** Therefore the matcher runs *outside* the particle loop, once per tracklet per candidate set at keyframe rate, and communicates through a `CompatibilityTable`. The filter's inner loop is pure arithmetic on cached values. This is what makes matcher implementations swappable with zero filter changes, and what makes deterministic replay (§7) cheap.

---

## 3. Design principles, with literature intuition

These are the "why"s. Each maps to concrete requirements later.

1. **The tracklet is the observation unit, not the frame.** Frames within a track are heavily correlated; treating them as independent measurements makes any Bayes filter overconfident by orders of magnitude. The closed-set label-fusion literature learned this the hard way (SemanticFusion-style multiplicative fusion lets one overconfident frame lock in an error; Morilla-Cabello et al.'s robust-fusion critique). Updates happen at tracklet level, at keyframe rate. (§5.3, test T-F1.)

2. **Association is marginalized, not decided.** The core lesson of the semantic-SLAM data-association line (Bowman et al. 2017's EM soft weights; Doherty et al.'s max-mixtures/null hypotheses; Atanasov et al. 2016's matrix-permanent localization — the closest prior formulation to this exact problem): never make a hard, irreversible correspondence commitment inside the estimator. Every measurement update sums over candidate landmarks *plus an explicit null hypothesis* (clutter / not-in-catalog / hallucination). (§5.3.)

3. **Geometry gates weakly here, so semantics carries more weight — and must be calibrated.** Far-field bearings are insensitive to translation: geometry separates hypotheses well in *heading* and poorly in *which of two nearby distant objects*. This inverts the near-field object-SLAM regime (ConceptGraphs et al., where 3D overlap dominates). Consequently the semantic term does real inferential work, and an uncalibrated score will silently dominate or vanish against the geometric term. Hence the calibrated-LLR contract on the matcher seam (§6). The VLM-uncertainty literature (CLIP miscalibration — LeVine et al.; hallucination context-dependence — POPE/ROPE) is why calibration is a *contract*, not a nicety.

4. **Bearings live on the sphere; use directional statistics.** von Mises–Fisher likelihoods, tangent-plane gating as the Mahalanobis analogue. Inverse-depth intuition (Civera et al. 2008): points at infinity are regular, parallax is a bonus not a requirement.

5. **Belief must represent multimodality until evidence kills it.** Symmetric environments (two similar lighthouses) create genuinely multimodal posteriors; MHT → MH-iSAM2 → multimodal-PDA lineage all exist because premature unimodality is unrecoverable. Particle/mixture representation, with explicit mode bookkeeping (§5.6) — which doubles as the backbone of the debugging story (§7).

6. **Sample from the likelihood, not just the motion model (Mixture MCL).** Over large areas, motion-model-only proposals need hopeless particle counts. Bearings admit direct likelihood inversion: two identified landmarks → inscribed-angle arc; three → resection fix. Global init = type-pair retrieval from the catalog + resection + particle injection. Particle count then scales with the number of *plausible modes*, not with area — the property that lets the same filter run at harbor scale and at 625 km². (Analogues: star-tracker lost-in-space identification; skyline geolocalization at country scale.)

7. **Semantic payloads are distributions, not points.** Keep caption sets and embedding mode-clusters, not averages ("Bare Necessities" result: view-averaged VLM embeddings degrade association). Per-tracklet semantic entropy modulates how much the matcher's evidence is trusted — this is the calibrated-inconsistency idea the literature has not yet exploited; it lives in the matcher, but the payload format must preserve the raw material for it.

8. **Determinism is a feature of the filter, not the tooling.** Fixed seeds + logged inputs ⇒ bit-exact replay. Every debugging, visualization, attribution, and A/B capability in §7–§8 is built on this single property. `[CONTRACT]` The filter must be a pure function of (config, seed, ordered input log). No wall-clock, no unordered map iteration affecting results, no global RNG. The existing filter code already passes explicit `torch.Generator`s (`swag/filter/particle_filter.py`) — keep that discipline. Additionally, the filter core must run on CPU: GPU reductions are not bit-reproducible, and the filter is cheap arithmetic on cached values (§2), so nothing is lost.

---

## 4. Frames, conventions, and the catalog `[ADAPT]`

- **Working frame:** single local metric ENU frame per region, anchored at the region centroid (fine to ≥625 km²: at a 25 km half-extent, equirectangular tangent-plane distortion is ~0.3% in scale and ≲0.2° in bearing at the corners — sub-noise for angular measurements; bounds documented by T-U4). This extends the per-run anchor convention already in the codebase (`landmark_filtering/ingest.py:fill_enu` → `Frame.x_m/y_m` via `bearing_geometry.enu_from_latlon`) from per-artifact to per-region. Prefer anchored ENU over literal UTM-zone coordinates: UTM grid north deviates from true north by the meridian convergence (~1.3° for Boston, near the edge of zone 19) — a systematic bearing bias larger than projected map error that would need correcting at every bearing computation. Note the two *other* frames already live in this codebase, which the wrapper must convert to at the boundaries only: raw lat/lon degrees (dataset metadata, `vigor_dataset`, the WAG particle filter — which carries a live TODO about latitude non-uniformity in `evaluation/swag_algorithm.py`, the cautionary precedent for degree-space state) and Web Mercator pixels at fixed zoom (`common/gps/web_mercator`, `filter/histogram_belief.GridSpec`, basemap tiles — beware y grows downward). All geodesy through one wrapper module; no inline lat/lon math anywhere else.
- **Bearing conventions:** already fixed in `landmark_filtering/bearing_geometry.py` — compass bearing = degrees clockwise from true north = atan2(east, north) in ENU — and this filter adopts it (enforced by test, T-U3). Internally, bearings are stored as unit vectors, not angles, to avoid wraparound bugs. `[CONTRACT]` The filter consumes **body-frame** bearings (relative to vehicle heading) and rotates them through each pose hypothesis's heading; it must not build on upstream "global" bearings (`bearing_global_deg`), because the per-dataset yaw calibration behind them has documented drift and sign subtleties (`yaw_offset.py`; a measured ~0.24°/frame compass drift on one real run). The `bearing_geometry.py` KNOWN ISSUE header (camera bearings mirrored on the 90°/270° faces) is a live instance of exactly the bug class T-U3 exists to catch — bearings must come from `object_tracking/pano_geometry.py`-derived paths, not `bearing_camera_deg`.
- **Catalog:** merged OSM + ENC records: id, position (+accuracy class; ENC surveyed ≫ OSM), canonical `type_key` from a small controlled vocabulary (~30/domain), raw tags, height/`max_visible_range` estimate, regional rarity weight (IDF-style). Ingest largely exists: `scripts/download_enc_cells.py` + `scripts/extract_landmarks_from_enc.py` (ENC S-57), the OSM landmark-extraction scripts, and the vocabulary builders (`scripts/build_tag_vocabularies.py`); what's new is the merge, the indices, and versioning. Precomputed indices: spatial (grid or KD-tree with visibility-radius expansion → "landmarks visible from cell" is one lookup; `cKDTree` is the codebase idiom), type-pair table for the proposal (pairs within joint visibility, keyed by type pair, filtered by baseline), matcher artifacts from `prepare_catalog()`. Catalog is versioned; a run log records the catalog version (§7).
- **Map error intuition:** OSM position error projects to angle as ~atan(err/range); 5 m at 2 km ≈ 0.14° — usually below detection noise. κ in the measurement model absorbs detection noise + projected map error; ENC vs OSM accuracy classes give per-landmark κ adjustments.

---

## 5. The localization box

### 5.1 Belief representation

Weighted particle set over SE(2) (position + heading; extend to altitude only if pitch matters for elevation-angle evidence). Decided vs. the incumbent `filter/histogram_belief.py` grid: with relative-only bearings, heading is a load-bearing state dimension, and a dense position×heading grid at useful resolution (e.g. 512×512×72 ≈ 19M cells) is exactly the cost explosion particles avoid — the grid stays relevant only as a position-only prior (§9). KLD-adaptive particle count. A lightweight **mode tracker** clusters particles (position + heading metric) each keyframe and maintains persistent mode identities across time (birth/death/merge), including **provenance** for injected particles (§5.5). Modes are the unit of explanation in the visualizer; treat mode bookkeeping as a first-class filter output, not a post-hoc analysis.

Published outputs per keyframe: particle set (or GMM summary), MAP pose + covariance when effectively unimodal, multimodality flag, per-tracklet association posteriors p(cᵢ = j | Z) — the latter feed back to mapping/smoothing and are diagnostic gold.

### 5.2 Motion update

Compose logged SE(3) odometry deltas (projected to the working manifold) with sampled noise per particle, between keyframe timestamps. Nothing exotic; the requirement is that deltas and their covariances come from the log so replay is exact. Reality check for current datasets: real odometry with covariance mostly doesn't exist yet — trajectories are GPS-positioned panorama sequences, and odometry is synthesized from position deltas plus configured noise (`evaluation/odometry_noise.py` is the precedent). That's fine; the synthesized deltas are what get logged. One consequence of the relative-bearings-only contract (§4): heading is anchored *only* by landmark bearings plus whatever heading signal the motion side supplies. Two structural facts make weak heading affordable: (1) GPS-derived translation deltas are already **world-frame**, so position propagation never routes through the heading estimate — heading error widens bearing gates but does not leak into position drift; (2) course over ground from ~1 Hz GPS *is* a usable heading measurement. Differenced-fix course error ≈ σ_rel·√2/(v·Δt) (consecutive-fix errors are correlated, σ_rel ~1 m): tens of degrees for a single 1 s interval at boat speed, a few degrees smoothed over ~10 s — so consume course as a weak heading measurement, speed-gated (dropped when slow) with noise ∝ 1/(v·window). Heading = course + a slowly-varying bias (leeway/crab from current and wind, ~5–10° worst case, plus constant mount offset): model the bias as a random-walk state; bearings observe heading, course observes heading − bias, so the bias is observable whenever landmarks are in view. Optional upgrade if course proves too weak: relative yaw between consecutive panoramas via circular cross-correlation of an azimuth signature (boat masked out) — rotation-only estimation from a 360° pano is far better conditioned than full VO over featureless water, needing only a few degrees of textured azimuth (harbor skylines qualify). Heading uncertainty growing between landmark observations is correct behavior, not a bug.

### 5.3 Measurement update `[CONTRACT]`

Per tracklet z = (bearing b with concentration κ_z, payload → CompatibilityTable T):

```
p(z | x) = π₀(class)·(1/2π)
         + (1−π₀) · Σ_j∈cand(x)  w_j · vMF(b; bearing(x, m_j), κ_eff(z, m_j)) · exp(T.log_lr[j])
```

- `b` is the tracklet's fused **body-frame** bearing (§4 — no north alignment of the sensor is assumed anywhere); `bearing(x, m_j)` is the predicted body-frame bearing: the world-frame bearing from the particle's position to landmark m_j, rotated by the particle's heading. North exists only on the map side. Fusing bearings across a tracklet's frames into the anchor-keyframe bearing requires compensating vehicle rotation across those frames — an upstream obligation of the tracklet contract, using the same Δheading source as §5.2.
- `cand(x)`: spatial-index lookup of landmarks plausibly visible from x (gate on `max_visible_range`, generous).
- `w_j`: prior over candidates given a detection (uniform over `cand(x)` in v1; hook for visibility/saliency weighting). `(π₀, (1−π₀)·w_j)` must form a proper mixture — this is easy to silently break when gating changes `|cand(x)|` per particle.
- On SE(2) the "sphere" is the circle: vMF reduces to the von Mises distribution and the null density 1/2π is uniform on the circle. The formula is written as vMF so the elevation extension (§9) is a representation change, not a model change.
- κ_eff combines tracklet κ with the landmark's map-accuracy class.
- **Null hypothesis is mandatory** and π₀ is per-landmark-class config (buoy ≠ lighthouse miss/hallucination rates).
- All arithmetic in log domain with log-sum-exp (T-U5).
- **Information-epoch rule:** a tracklet contributes a *new* measurement update only when it carries new information (new anchor keyframe with refined fused bearing), never by re-submitting the same evidence. When a refined CompatibilityTable arrives (two-tier matcher), it replaces the table for *future* updates only — no retroactive reweighting. This keeps the filter simple and prevents double-counting (T-F1, T-F8).
- **Mutual exclusion** (two tracklets claiming one landmark) is ignored in v1 (independent-sum), with a hook for k-best-assignment marginals (Michael et al. 2022) when candidate density demands it. Log a counter of exclusion violations in the health stream so the v1 approximation's cost is observable (T-F9).
- **Negative information** (not seeing a landmark that should be prominent): off by default; if enabled, only for the highest-saliency class with generous miss probability. VLM missed detections are too common for aggressive use.

### 5.4 Association posteriors

By-product of the update: for each tracklet, the normalized per-landmark (and null) responsibility, averaged over particles (or reported per mode — per mode is more useful; see §7). `[CONTRACT]` This output must be computed per mode, because "mode A believes tracklet 7 is Graves Light; mode B believes it's Boston Light" is exactly the explanation the visualizer needs to surface.

### 5.5 Mixture proposal (global init & recovery) `[CONTRACT on provenance]`

- Trigger: initialization, kidnapped detection (sustained likelihood collapse), or ESS starvation.
- Mechanism: pick high-quality tracklets weighted by type rarity → retrieve candidate landmark pairs/triples from the type-pair index (feasibility-pruned by observed angular separation vs. baseline and visibility) → resect candidate poses (inscribed-angle arcs for pairs, fixes for triples; reject degenerate/near-collinear geometry) → inject particles with weights from the full measurement model.
- **Use one landmark too.** A single bearing does not constrain position, but it determines heading *for every candidate position* (θ = bearing(L, p) − β), collapsing the heading axis exactly — worth ~2π/σ in particle count against a uniform prior, and available whenever any confident tracklet exists. So the proposal is a hierarchy: 1 landmark → 2-D set (heading pinned), 2 → 1-D arc, 3 → discrete fixes. A filter needs *samples*, not a solution, so the lower-order sets are as usable as a point fix; restricting to triples throws away every sparse-visibility case, which is exactly when recovery matters most. Sample the single-landmark disc uniformly in **area** (r = R√u), not in range, or the injected cloud piles up near the landmark and biases the proposal density.
- **The sign of the subtended angle picks the arc.** Both inscribed-angle arcs carry the same *unsigned* angle, but once the two bearings are assigned to landmarks the signed angle is observable and only one side is a real hypothesis. Injecting into both puts half the particles where the measurement flatly contradicts them.
- **Candidate identities come from the matcher, not a type index** (implemented): each tracklet's top-k CompatibilityTable entries, paired across tracklets. Matcher-agnostic and bounded by k; the type-pair index is the scaling path for when tables are uninformative, not a prerequisite. `[HAZARD]` This gives an uncalibrated matcher an attack surface the §6 clip bounds do *not* cover: clipping bounds an LLR's contribution to the measurement update, but nothing bounds a wrong *identity*, and the proposal will place hypotheses wherever that identity implies. Measured, the residual guarantee is weaker but real — the belief is displaced yet stays consistent (NEES ≈ 2), because the injected mixture keeps it broad. See the `inject_fraction` non-monotonicity in §8.4.
- **Bearings must be treated as simultaneous across a short window**, since the information-epoch rule staggers tracklet anchors and one keyframe rarely carries two bearings. The error is translation/range (~0.6° per keyframe at 2.5 km, under the bearing noise) and costs proposal recall, not posterior accuracy — every injected particle is re-scored under the exact measurement model.
- **Provenance:** every injected particle records (proposal event id, generating tracklet ids, hypothesized landmark ids). Modes inherit provenance from their founding particles. This is cheap (a couple of ints per particle) and is the single most valuable piece of metadata for debugging "where did this wrong mode come from" (§7).
- Note the terrestrial correction to the star-tracker analogy: pairwise angles are *not* observer-independent here; the index retrieves by type pair + baseline feasibility, and the observed angle constrains pose via the arc — it is not a hash key by itself.

### 5.6 Health monitoring

Per keyframe, compute and log (Tier 0, §7): ESS, mode count and per-mode weight/entropy, MAP jump magnitude, per-tracklet null-share (fraction of responsibility on the null hypothesis), exclusion-violation counter, proposal activity, timing. Kidnapped-detection and event-bookmark logic (§7.3) read this stream.

---

## 6. The matcher seam (summary) `[CONTRACT]`

Full rationale in the matcher design discussion; the localization-box-facing contract is:

- Input to filter: `CompatibilityTable{tracklet_id, matcher_version, entries: [(landmark_id, log_lr)], default_log_lr, clip: [lo, hi], status: fast|refined}`.
- `log_lr` is a **calibrated log-likelihood ratio** whose non-match reference population is *gated candidate pairs in-region* (not random pairs). Calibration lives inside the matcher; the filter never compensates.
- LLRs are clipped to declared bounds so no matcher saturates and steamrolls the geometric term.
- Every matcher version ships a calibration report (reliability/ECE/ROC) on a frozen shared benchmark; `matcher_version` in every table makes filter regressions traceable to matcher changes.
- Tracklet semantic payloads are opaque, channelized containers (captions, attributes, embedding clusters, crop refs, entropy); matchers declare required channels; the filter reads none of it.
- Matcher API: `prepare_catalog` (offline), `block` (cheap high-recall gate, optional), `score` (batched), `calibration_report`.

**v1 posture (decided):** the interface above is binding from day one, but v1 fills it with the *existing, uncalibrated* matcher — the trained pairwise correspondence model — mapped into `log_lr` by a fixed monotone transform (temperature/affine; the same role sigma plays in the existing similarity→likelihood conversions, cf. `scripts/calibrate_sigma.py`, `scripts/grid_search_combined_likelihood.py`) and clipped tightly. With an uncalibrated matcher, the clip bounds and π₀ carry the safety burden: tight clips make the semantic term advisory and geometry dominant, and the transform+clip settings are tuned on the regression suite like any other config. Proper calibration (reliability/ECE against gated in-region pairs) is deferred until symptoms demand it — the health stream is the tripwire: persistent null-share anomalies, T-F5-style saturation behavior, or matcher A/B results that don't transfer across regions. In v1 `calibration_report` may return "uncalibrated: transform+clip only"; `matcher_version` must still change whenever the model, transform, or clips change.

---

## 7. Visualization & debugging infrastructure

This section is deliberately detailed: the visualizer is the debugging surface for the *entire pipeline*, and its design determines whether failures take minutes or days to diagnose.

### 7.1 The core storage/compute strategy: log inputs, checkpoint sparsely, recompute on demand

The naive options are both wrong: logging everything the filter computes (per-particle, per-tracklet, per-candidate likelihood terms) is ~GBs/hour and mostly never looked at; logging only summaries makes "why" questions unanswerable. The resolution is the determinism contract (§3.8), which turns storage into a classic video-codec-style tradeoff:

- **Tier 0 — health scalars** (every keyframe, ~1 KB): the §5.6 stream. Always in RAM for a whole run; drives the timeline UI.
- **Tier 1 — inputs** (every keyframe, ~10–50 KB): tracklets, odometry deltas, CompatibilityTables, config/catalog/matcher versions, RNG seed. `[CONTRACT]` Tier 1 is *sufficient for bit-exact replay*. Append-only JSONL/Parquet.
- **Tier 2 — belief checkpoints** (sparse): full compressed particle sets (with provenance) every N keyframes (N ~ 50–200, tunable) **and** at auto-detected events (§7.3). ~1–10 MB each compressed. These are the "I-frames."
- **Tier 3 — nothing, recomputed**: every internal quantity (per-particle weights, per-tracklet log-likelihood contributions, per-mode responsibilities, gating decisions) is reconstructed exactly by replaying from the nearest checkpoint ≤ the query time. Bounded recompute: ≤ N keyframes of filter time, which is fast because the filter is cheap without rendering.

Budget consequence: a multi-hour run is tens of MB (Tier 0+1) plus checkpoints — small enough to keep every run forever, which is what makes regression suites over real logs (§8) possible. RAM in the viewer is bounded by one checkpoint + one replay window + Tier 0, regardless of run length.

Raw video/panoramas are **not** stored in the run log; tracklets carry frame/crop *references* (timestamps + source URIs). A bounded LRU thumbnail cache materializes crops for the tracklet inspector on demand. This keeps run logs portable and small while preserving drill-down to pixels when the source store is reachable.

### 7.2 The explanation primitive: log-domain attribution

Because each tracklet contributes an additive term to each particle's log-weight, any change in belief decomposes exactly:

> For mode m over update t: Δlog W_m = Σ_tracklets (per-tracklet contribution) + motion/resample effects.

The viewer computes this decomposition on demand (Tier 3 replay) and renders it as a **waterfall chart**: "mode B lost 6.2 nats at t=412: −5.8 from tracklet 7 (bearing inconsistent with Boston Light under mode B), −0.4 from tracklet 12." This single primitive answers most "why" questions, and it requires nothing logged beyond Tiers 1–2. Implementations should treat it as the central viewer API: `attribute(mode_id, t_range) → per-tracklet contribution series`.

### 7.3 Auto-bookmarked events

Detected online from the Tier 0 stream, logged as an index, each triggering a Tier 2 checkpoint:

mode birth (with proposal provenance) · mode death · mode merge · MAP jump > threshold·σ · ESS crash / resample storm · association flip (a tracklet's per-mode argmax landmark changed) · null-share spike (semantic evidence went unusable) · kidnapped trigger · refined-table arrival that changed any argmax · exclusion-violation burst.

The event index is the debugging table of contents: the workflow is "open run → look at event strip → click."

### 7.4 Views

1. **Run overview strip.** Timeline: Tier 0 scalars as sparklines, event bookmarks as glyphs, mode lifespans as horizontal bands (width = weight). Click anywhere → map view at that time. This is the entry point; it must render in <1 s from Tier 0 alone.

2. **Map view** (main canvas; basemap = OSM/ENC tiles). At selected time t: particle cloud (subsampled draw, colored by mode), mode ellipses, MAP trail (± ground truth if available), catalog landmarks glyphed by type. **Per selected mode** (not just MAP — this matters): active tracklet bearings drawn as wedges from the mode centroid (wedge width ∝ 1/κ), candidate correspondence lines from wedge to landmarks with opacity ∝ that mode's association posterior, red-flagged when LLR and geometry disagree. Scrub time with the strip; the map re-renders from checkpoint+replay.

3. **Tracklet inspector.** Click a tracklet anywhere: its full life — bearing/κ series, payload (captions, attributes, entropy, crops via the thumbnail cache), CompatibilityTable entries over time (LLR bars per candidate, fast vs refined), per-mode association posterior evolution, and its **attribution series** (contribution to each mode's log-weight over time). The question "did the tracker, the matcher, or the filter get this wrong?" should be answerable from this one panel — it shows the raw track (tracker), the LLRs (matcher), and the responsibilities (filter) side by side.

4. **Mode ledger / genealogy.** Modes as rows; birth event with provenance ("spawned by proposal #14: tracklets {3,7} ↔ landmarks {Graves Light, Deer Island tank}"), weight trajectory, death event with its attribution waterfall pre-computed. This view answers the two most common debugging questions — "why did the right mode die" and "where did the wrong mode come from" — in one click each.

5. **What-if console** (replay-powered). Toggle a tracklet off / edit an LLR / change π₀ → replay the window from the nearest checkpoint → ghost-overlay the counterfactual trajectory on the map. Cheap because of §7.1; enormously effective for "was this tracklet actually the culprit" confirmation, and for pipeline debugging upstream of the filter (e.g., demonstrating a run would have converged if the matcher hadn't scored a false positive).

### 7.5 Implementation shape `[ADAPT]`

Run directory = self-describing artifact: `manifest.json` (versions, seed, config), Tier 0/1 Parquet/JSONL, checkpoint blobs, event index. This is the pattern `landmark_filtering` already uses — `artifact_schema.py` + `filter_run_viewer.py`, where the viewer renders *exclusively* from a self-describing artifact and new pipeline stages require no schema/viewer changes; reuse its msgspec serialization conventions (`common/python/serialization`). Viewer: local web app (map stack: deck.gl/MapLibre or similar) + a small replay service (same filter code, headless) for Tier 3 queries. The replay service is the *production filter binary* in replay mode, not a reimplementation — divergence between viewer math and filter math must be structurally impossible. A CLI (`runlog attribute`, `runlog replay --without-tracklet 7`) should expose the same primitives for scripted forensics and CI.

---

## 8. Testing strategy

### 8.1 Unit tests

- **T-U1 vMF correctness:** density integrates to 1 (quadrature); sampling vs density agreement (KS test); κ→∞ and κ→0 limits.
- **T-U2 Resection geometry:** synthetic pose + landmarks → bearings → resection recovers pose; inscribed-angle arc membership; degenerate (collinear, near-coincident) configurations rejected, not silently accepted.
- **T-U3 Frame conventions:** golden fixture — known pose, known landmark lat/lons, hand-computed true bearings; catches N/E swaps, CW/CCW, magnetic/true, ENU/NED errors. This class of bug is historically the most common and the most embarrassing; the fixture must be hand-derived, not generated by the code under test.
- **T-U4 Geodesy round-trips:** lat/lon ↔ working frame at region corners; error bounds documented.
- **T-U5 Log-domain numerics:** likelihood with 1 candidate at LLR clip bounds, 10³ candidates, all-null — no NaN/underflow; log-sum-exp path exercised.
- **T-U6 Resampler unbiasedness:** systematic resampler preserves expected weights (statistical test over many trials).
- **T-U7 Determinism:** same (config, seed, input log) → bit-identical particle history hash. Run in CI on every commit; this test protects the entire §7 edifice.

### 8.2 Filter-level statistical tests (synthetic scenarios)

- **T-F1 Correlated-evidence guard:** inject the same tracklet twice (or re-emit unchanged) → posterior must not sharpen beyond single-inclusion bound. Catches information-epoch violations (§5.3). *Failure mode: overconfidence from re-counted evidence — the classic frame-vs-tracklet error.*
- **T-F2 Consistency (NEES):** synthetic runs with known truth; average NEES within chi-squared bounds, over seeds. *Failure mode: generic over/under-confidence.* `[CONTRACT]` **This test gates the rest of the suite.** Every other assertion here scores error *magnitude*; for a component whose published output is a belief (§5.1), a filter that is accurate-looking and 150× overconfident passes all of them. Land T-F2 before, not after, the accuracy envelopes — an accuracy number from an inconsistent filter measures nothing.
- **T-F3 Multimodality preservation:** symmetric two-lighthouse world; filter must hold bimodal belief (both modes > threshold weight) until a disambiguating observation, then collapse to the correct mode. *Failure mode: premature unimodality — unrecoverable by design; must be caught here.*
- **T-F4 Null-hypothesis function:** (a) all-hallucinated tracklets (no catalog counterpart) → belief degrades gracefully to odometry-only, no confident wrong fix; (b) π₀ = 0 ablation demonstrates the failure this prevents. *Failure mode: null starvation — every tracklet forced onto some landmark.*
- **T-F5 LLR saturation:** adversarial table with log_lr at clip bounds on a geometrically wrong candidate → geometry must still win given enough bearings. *Failure mode: semantic term steamrolls geometric term (calibration/clipping breach).*
- **T-F6 Kidnapped recovery:** teleport mid-run; recovery within bounded keyframes via proposal; wrong-mode mass decays. *Failure modes: proposal starvation (no rare pairs in view — test both rich and poor visibility), dead recovery trigger.*
- **T-F7 Invariances:** tracklet order permutation within a keyframe → identical posterior; global map translation/rotation with matching truth → equivariant posterior.
- **T-F8 Refined-table semantics:** refined CompatibilityTable arrival mid-run → affects only subsequent updates; replay with refined-from-start differs (documented, intended). *Failure mode: retroactive reweighting / double counting.*
- **T-F9 Exclusion-violation observability:** dense-candidate scenario where independent-sum is wrong → violation counter fires (the v1 approximation must be *visible*, even though tolerated).
- **T-F10 Map-error robustness:** perturb catalog positions per accuracy class → κ_eff absorbs it; localization error degrades smoothly, no association flips at perturbation scale.
- **T-F11 Particle-count invariance:** results statistically stable across particle counts above KLD floor (catches accidental hard dependence on particle count).

### 8.3 Scenario regression suite

Golden scenarios (synthetic + replayed real logs once available), each with recorded metric envelopes: time-to-converge, wrong-mode mass over time, RMSE after convergence, association precision/recall vs truth. CI compares against envelopes, not exact values (except T-U7 exact-replay hashes). Matcher A/B = regenerate CompatibilityTables offline, rerun suite, diff metric tables — no filter changes (§6). *The end-to-end metric is "does the vehicle know where it is sooner/more reliably"; pair-level matcher ROC is only a proxy.*

### 8.4 Failure-mode register (watch list)

Beyond those pinned to tests above: **non-monotonic injection fraction** — tuning the mixture proposal's inject fraction *down* to be cautious makes things worse, not better. Measured against a matcher that confidently misidentifies a landmark: at φ=0.5 the belief is displaced ~180 m but stays consistent (σ ≈ 1000 m, NEES ≈ 2); at φ=0.2 and 0.05 it ends up 2500 m out at NEES ≈ 150. A partial injection drags the belief toward the wrong hypothesis without leaving enough mass on the alternatives to keep the posterior honest about the ambiguity. **Kidnap detection cannot use a consecutive-run counter** — a displaced belief does not put *all* its evidence on the null: some bearings coincidentally align with the wrong landmark and score well, so null-share alternates (measured: 1.0 and ~0.00 alternating after a 1900 m kidnap) and any consecutive run resets before it fires. Use the fraction of recent measurements that were null-dominated. Note also that ESS is useless here: a stranded belief has a *flat* likelihood, so ESS reads full health (20000/20000) at exactly the moment the filter is most lost. **course/heading evidence reuse** — consuming one course reading both as a heading-propagation input (Δcourse) and as an absolute heading measurement lets the prior and the likelihood confirm the same noise sample, so the posterior concentrates without ever averaging the noise down. Signature: a reported heading σ *below* the single-sample course σ while the actual heading error sits above it. This is the frame-vs-tracklet error of §3.1 wearing a different hat, and it deserves the same billing; it was a live defect in Milestone 0 (audit A-1). The resolution is to consume course exactly once, as increments — which also cancels a constant COG-vs-heading offset, so the §5.2 bias state is only needed by a design that wants the *absolute* course information. **Particle impoverishment masquerading as confidence** — plain resampling redraws only from locations already present, so a long run collapses to duplicated atoms and the filter reports a spread it no longer represents (measured: NEES 67 at 4k particles, 12 at 20k, against an ideal of 2.0). Kernel-regularized resampling with a bandwidth tied to the posterior's own spread is the fix, and NEES is the only thing that sees the problem — every accuracy metric looks fine throughout. **Null-floor stranding** (observed empirically in the milestone-0 synthetic harness): the null hypothesis flattens the likelihood gradient a few σ beyond each bearing's vM support, so a belief that ends up parked outside every bearing's basin of attraction feels no pull back — ESS looks healthy while the fix stays biased; recovery is the mixture proposal's job (§5.5), not the gradient's, and sustained per-tracklet null-share with a stable MAP is the health-stream signature to alarm on; panorama seam/pole distortion corrupting bearings near image edges (guard: per-tracklet bearing sanity vs. tracker pixel positions — belongs upstream but the filter should flag statistical outliers); upstream yaw-calibration drift — per-dataset compass offsets have measured *time-varying* drift on real runs (~0.24°/frame on one), which is a systematic heading-rate error, not zero-mean noise; the body-frame bearing contract (§4) confines it to the heading state where odometry+bearings can fight it, but the health stream should watch for sustained one-sided bearing residuals; timestamp skew between odometry and keyframes (guard: monotonicity + max-skew assertions on ingest); catalog staleness (moved buoys — seasonal repositioning is real; mitigations: ENC update discipline, higher π₀ for repositionable classes); heading-only degeneracy when all visible landmarks cluster in one narrow azimuth range (health metric: bearing-space dispersion; expect elongated position uncertainty); silent unit drift at interfaces (radians/degrees, true/magnetic) — mitigated by typed wrappers at every boundary.

---

## 9. Extension hooks (design for, don't build)

- **Smoothing promotion:** confident associations (persistently high posterior, unimodal) become bearing factors (max-mixture with null component) in a GTSAM-based smoother — the filter's association posteriors are already the right currency.
- **Coarse-prior fusion:** `filter/histogram_belief.py` is the incumbent belief representation at 625 km² (appearance-likelihood grids). A histogram posterior over the region can serve as a position prior for proposal injection or as a cross-check baseline in the regression suite, without touching the filter or seam.
- **Mutual exclusion upgrade:** k-best assignment marginals behind the same measurement-update interface.
- **Elevation-angle weak range:** landmark heights (ENC) + pitch → coarse range likelihoods; additive term in the same per-tracklet update.
- **Ephemeris landmarks (absolute heading):** the sun (and moon) fit the existing measurement model as catalog entries with time-indexed, exactly known azimuth, effectively infinite range, and π₀ ≈ 0 — one confident detection per pano is an absolute heading fix to a couple of degrees given timestamps (GPS-accurate) and the belief's coarse position (even 25 km of position ambiguity moves sun azimuth < ~0.3°). Daytime/weather-opportunistic; detector is upstream work; no filter or seam changes.
- **Semantic-entropy modulation:** per-tracklet entropy tempering the LLR — implemented inside the matcher, transparent to the filter; the payload format already preserves what it needs.

---

## 10. Suggested build order

**Milestone 0 (`swag/bearing_only_localization/`):** steps 1+3+4 in synthetic form — anchored-ENU geodesy wrapper with T-U3 golden fixtures, event-driven filter core (sparse information-epoch tracklet measurements over a keyframe odometry timebase), `LandmarkCatalog` owning positions/accuracy classes/candidate priors, scenario generator with identity-stub matcher *and model-mismatch knobs* (crab bias, bearing bias, outliers, catalog position error, dropout, clutter), run-directory logging (§7.5 shape), matplotlib run plots, and the test suite: T-F2 consistency (the gate), plus T-F1/F3/F4/F5/F7/F10/F11 and the E2E accuracy envelopes.

Milestone 0 was audited (`localization-design-doc.audit.md`) and the audit found the filter statistically broken behind a green test suite — mean NEES 309 against an ideal of 2.0, from course double-counting (§5.2) compounded by particle impoverishment, invisible because T-F2 had never been written. Both are fixed and measured; the register there is the record. Two lessons generalize beyond this milestone and are folded into §5.2/§8.2/§8.4 above: **a generator that shares the filter's assumptions exactly measures only "does Bayes invert this generator"** — the mismatch knobs are what make the suite mean anything — and **assertions on error magnitude cannot see a miscalibrated belief**, which is why T-F2 gates the rest.

**Milestone 1 (`resection.py`, `proposal.py`):** step 5's mixture proposal, minus the mode tracker. Resection geometry with degeneracy rejection (collinear, near-0/π subtended angles, danger circle), the 1/2/3-landmark hypothesis hierarchy, matcher-driven candidate identities, per-particle provenance plumbed through checkpoints and the run-directory event index, and triggers for init / sustained null-share / sustained ESS floor. Demonstrated: global init over a 100×100 km box converges at keyframe 0 with **1500 particles**, where brute-force uniform init never converges — the §5.5 scaling claim. Kidnap recovery succeeds on 6 of 8 seeds and is consistent (NEES ≈ 0.3) where it succeeds; the two failures latch onto a self-consistent *mirror* hypothesis and stop re-triggering, which is the mode tracker's job and is recorded as such in `proposal_test`.

**Milestone 2 (`mode_tracker.py`, `viewer.py`):** step 5's mode tracker and step 6's viewer.

Modes are clustered by grid connected components over (east, north, heading) — O(N), no extra dependency, and exactly deterministic, which a distance-based clusterer with iteration order would not be. Identity is carried by **particle lineage** rather than centroid matching: every particle holds the mode id it belonged to last keyframe, so a cluster's identity is whichever ancestor holds the most weight inside it. Merges, splits and births then fall out rather than being detected — and a cluster with no ancestor is a birth, which for proposal-injected particles carries the §5.5 provenance. This closes the §5.4 `[CONTRACT]`: association posteriors are now emitted per mode alongside the whole-belief average, and mode-weight entropy is the §5.1 multimodality flag as a number.

The viewer builds the three Tier-0/1+checkpoint views (overview strip, map, mode ledger) as one self-contained HTML file per run. The attribution waterfall and what-if console remain deferred *by design*, not by convenience: §7.5 requires the replay path to be the production filter in replay mode, and a viewer recomputing likelihoods in JavaScript would make viewer-vs-filter divergence possible — precisely what that requirement exists to prevent.

Two things the mode tracker immediately exposed. The T-F3 test asserted "posterior mass on both sides of the symmetry axis", which **a single unimodal cloud straddling the axis passes trivially** — it had been asserting nothing since it was written; it now asserts on tracked modes. And the correct T-F3 property is not "stays bimodal" but *hold both hypotheses until evidence kills one, then commit to the right one*: measured, the symmetric world holds two modes with entropy 0.69 through keyframe ~20 and resolves by keyframe 60.

**Milestone 3 (`export_ingest.py`, `run_export.py`): first real data.** Boston Harbor leg 1 — 344 VLM-matched bearings over 379 keyframes against a 100-landmark OSM+ENC catalog, from a uniform prior over the 16x10 km harbour. Field names already matched `structs.py`, so ingest only decodes, builds the ENU catalog from the export anchor, and validates preconditions at the boundary.

Three things real data broke that synthetic data could not:

- **The initial proposal never fired.** `on_init` was keyed to keyframe 0; leg 1's first bearing anchors at keyframe 3, so the uniform prior was silently left to brute force. Now it fires at the first keyframe carrying bearings.
- **Real matchers emit disjunctions.** Every multi-entry table is a *tie* (41 storage tanks at one log_lr); only 30 of 44 have a unique top scorer. Taking `top_k=3` from a 41-way tie resects from a coin-flip identity — the §5.5 `[HAZARD]` in practice. Ties are now enumerated whole, under per-kind combination budgets spent cheapest-kind-first, and a combination that does not fit is skipped entire rather than swept part-way.
- **Real bearings are coarse — and the first fix for that was wrong.** σ ranges to 25.5°. The residual gate that rejects spurious circle intersections was given an absolute 12° ceiling on the reasoning that a 3σ gate of 77° rejects nothing. That reasoning ignored what the gate is measuring: the *true-identity* fix's own residual grows with bearing noise (measured on a 3-landmark fixture: median 0.0° / 4.4° / 20.5° / 33.9° at σ = 1° / 5° / 15° / 25°), so a ceiling below the noise floor discards real solutions — 26% of true fixes at σ=5°, 79% at σ=25°. What a coarse bearing implies is an **imprecise fix, not an untrustworthy one**: at σ=25° against 3 km landmarks the fix still lands in the right neighbourhood, ~800 m out. So the gate stays at 3σ (with a 90° backstop where the geometry is meaningless anyway), and the honest correction is on the other side — each hypothesis now carries its own injection spread, estimated as σ_bearing × range and floored/capped, instead of every fix being injected as the same 100 m blob. Injecting a diffuse fix tightly asserts a precision the geometry never had, which is the Milestone 0 overconfidence class in a new place.

Results, read as the export's provenance allows (GPS odometry, GPS-selected candidates, so position error is a sanity check rather than evidence): bearing residuals against the filter's own pose are median 0.66°, p90 5.3°, 90% under 5°, **none over 60°** — against median 2.64°/p90 11.3°/1% over 60° for the same bearings scored against GPS truth. The association posterior independently isolates the one bad tracklet: LT267 carries mean null share 0.61 with top responsibility 0.13 spread over 7 candidates, and LT267 is exactly the tracklet owning the entire >60° tail in the truth-residual check — two unrelated mechanisms agreeing. The odometry-only control settles the "dead reckoning would nearly solve this" question: **from a uniform prior it does nothing** (4002 m final, 6825 m median, σ 3884 m, no mode above threshold), because odometry only translates an unlocated cloud. The bearings are doing the whole job.

Steps 2/6/7 below remain (real catalog service, replay service and the views that need it, calibrated matcher). Also still deferred from the audit: per-mode association posteriors (§5.4 `[CONTRACT]`, needs the mode tracker), per-particle spatial gating of `cand(x)` (until then the candidate prior w_j = 1/|catalog| couples the posterior to catalog size — real, measured, and documented in `catalog.py`), exclusion-violation and mode-count health fields (§5.6), and the §7.3 event index.

1. Frames/geodesy wrapper + golden convention fixtures (T-U3 first — everything downstream depends on it). Build on `bearing_geometry.enu_from_latlon` + `common/gps/web_mercator`; the `bearing_geometry.py` KNOWN ISSUE is the fixture-worthy cautionary tale.
2. Catalog ingest + indices; freeze a Boston Harbor snapshot as the test catalog. ENC/OSM extraction and vocabularies largely exist (§4) — the new work is the merged record schema, indices, and versioning.
3. Filter core with stub matcher (type-equality LLRs) + determinism harness (T-U7) + Tier 0/1 logging from day one — *do not* bolt logging on later; the replay contract shapes internal structure. `filter/particle_filter.py` has the primitive precedents (explicit `torch.Generator`s; note its resampler is multinomial — this design wants systematic, T-U6).
4. Synthetic scenario generator (the T-F suite's engine and the demo engine — same code). Reuse `evaluation/odometry_noise.py` and `evaluation/convergence_metrics.py` for the metric envelopes.
5. Mixture proposal + mode tracker with provenance.
6. Viewer: overview strip + map view first (they're Tier 0/1+checkpoint only), then attribution/what-if (needs replay service). Follow the `filter_run_viewer.py` artifact-driven pattern (§7.5).
7. Real matcher behind the seam; A/B harness. v1 = the existing trained pairwise correspondence model (`model/`, exported via `scripts/export_correspondence_similarity.py`) behind the transform+clip adapter of §6 — no calibration work up front; `landmark_filtering/semantic_similarity.py` is the existing pluggable-backend precedent, and `evaluation/correspondence_matching.py`'s dustbin idiom is the matcher-side sibling of the null hypothesis.

---

*Prepared for the Robust Robotics Group far-field semantic localization effort. Companion documents: literature review (semantic landmark association over time), matcher-seam discussion (calibrated LLR contract). Both available in the project conversation; fold in as `docs/` neighbors of this file.*

