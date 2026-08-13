# Far-field matching → localization plan (boston_harbor leg1)

Status: drafted 2026-08-11. Companion to `docs/localization-design-doc.md`
(filter design, already implemented at Milestone 0) and the
`swag/landmark_filtering/object_tracking/` pipeline (M0–M6, built).

Goal: take merged landmarks out of M6, match them against the harbor
OSM+ENC catalog, and run the bearing-only filter over leg1 with a tight
prior on the true start position.

---

## 0. Camera → body frame

**The measurement.** The bow sits ~80/1280 of the way across the frame from
the left edge, i.e. fraction 0.0625. Panorama width is 7680 px, so the bow is
at pano x ≈ 480 px. Using the verified convention in
`pano_geometry.direction_from_pano_px` (`az_cw = (x/W − 0.5)·360 mod 360`):

```
bow_azimuth_camera = (480/7680 − 0.5) · 360 mod 360 = 202.5°
bearing_body_deg   = (bearing_camera_deg − 202.5) mod 360
```

**Is one number sufficient?** For yaw, yes — that is the entire transform,
given three assumptions worth stating:

1. The camera is rigidly mounted and the offset is constant over the leg.
   (Fit per leg; do not assume it carries across legs 2/3.)
2. The camera is approximately level. Roll shifts apparent azimuth by
   roughly `roll · tan(elevation)`; our landmarks sit near the horizon, so
   this is second-order. Pitch moves elevation, not azimuth, to first order.
3. Bearings are relative-only. The design doc §4 already commits to this —
   nothing upstream is north-aligned, and the filter must never consume
   `bearing_global_deg`.

**RESOLVED 2026-08-11: the offset is 214 deg, calibrated map-free.**

The bow estimate could not be sharpened by looking harder - `bow_calibration.py`
renders the vessel razor-sharp by temporal median, but the **deckhouse occludes
the bow**, so 202.5 deg lands on the pilothouse face and the bow tip is never
visible in any frame. Three readings disagreed by 30 deg (deckhouse centreline
~190, wake ~218, LT20-matched-to-One-International-Place 222).

What settled it needs no map, no assumed match, and no hand-read image: a wrong
offset rotates every bearing by a constant, which stops the rays of a static
object from intersecting. So **sweep the offset and minimise the median
triangulation residual**. Over 26 well-conditioned tracklets the curve is smooth
and unimodal - 5.95 deg at 180, **1.33 deg at 214**, 4.34 deg at 270 - and the
minimum is the calibration. Independent corroboration: the wake implies ~218,
LT20-to-OIP implies 222.

Verification of the whole chain at that offset:
- per-keyframe residual against One International Place (a surveyed 46-storey
  building, 72 keyframes): **mean +0.6 deg, std 2.42 deg**
- correlation of that residual with mask angular width: **+0.07**; with course
  change: **-0.09**. Neither the mask reference point nor the epoch fusion is a
  material error source at this scale.
- triangulation residuals across tracklets: **0.35-3.73 deg**, median ~1.3.

So bbox -> pano -> tracklet -> bearing is well formed. Remaining refinement is
map-side, not ours: we compare a mask centre against a building *centroid*,
while the mask tracks the visible facade - for a 60 m building at 1 km that
alone is ~1.7 deg of the residual. `bearing_span_from` already returns the
hull's angular span, so matching span-to-span rather than centre-to-centroid
is the next available gain.

**Retracted: "circular std of world bearing" as a quality metric.** It measured
parallax, not error - a static object 700 m off sweeps ~74 deg of genuine
bearing as the vessel passes. It inverted the ranking: LT160, labelled a
failure at 50.5 deg spread, has a 0.99 deg triangulation residual and the best
conditioning in the run (cond 1), while LT139, praised at 1.6 deg spread, has
condition 1259 - a short arc whose position along the line of sight is nearly
free. Use `bearing_matcher.triangulate`: residual for consistency, condition
number for whether the geometry determines a position at all. Both, never one.

---

## 1. Phase 0 — pipeline re-run (in progress)

r002 predated both the `context` class and the `weak_min_complement` floor,
so its *tracking* used the old claiming/seeding behaviour even though every
viewer recomputes classes for display. The re-run makes artifacts and rules
agree.

- [x] Archive r002 → `object_track_runs/m3_tracks/archive/r002_full_leg1_artifacts.tgz`
      (28 MB; JSON + HTML + audit requests/results, excludes regenerable
      chips/keyframe images/videos)
- [x] `m3_track_viewer --run_name r003_full_leg1 --range full_leg1 0 378` (running)
- [ ] `m5_build_audit_requests --run_dir <r003>` → new `name_candidates` schema
- [ ] Vertex online run (~86 tracks, ~1 M tokens, ~6 min)
- [ ] `m6_merge_tracks --run_dir <r003>`
- [ ] `keyframe_viewer`, `m5_audit_results_viewer` regeneration

Expected difference vs r002: context-class detections no longer claim their
detections, so island/tank/fort boxes that previously fed a dying mask should
now seed their own tracks. Track count will rise; T172-style multi-island
tracks should not recur.

---

## 2. Phase 1 — bearings in body frame

- Bow calibration (above) → `mount_offset_deg`.
- Extend `m6_merge_tracks` to emit `bearing_body_deg = camera − offset`
  alongside the camera-frame value, and record the offset in the artifact so
  a re-calibration is traceable.
- Build `OdometryDelta` per keyframe from `frames_gps.csv`: world-frame ENU
  deltas (`ingest.fill_enu` already computes ENU positions), `speed_mps`,
  and course over ground with `course_sigma_deg`. Course is a *weak,
  speed-gated* heading measurement consumed as Δcourse increments — the
  design doc's audit is explicit that consuming absolute course breaks NEES.

---

## 3. Phase 2 — catalog construction (filtered to far-field-visible classes)

`landmarks/harbor_osm_enc_v1.feather` holds 184,805 rows: 183,682 OSM
(`landmark_type=historical`) + 1,123 ENC. Overwhelmingly streets, footpaths
and `building=yes`.

**Bbox: W −71.093528, S 42.245833, E −70.831944, N 42.395.** The north edge was
widened from 42.373333 (`42°22'24"N`) on 2026-08-13 and the tables rebuilt in
place. The old edge clipped the Tobin Bridge by ~52 m — its southernmost way
starts at 42.3738 — along with the Logan control tower (42.3743) and the Donald
McKay monument (42.3741), all landmarks leg 1 observed and named. It ran
straight through the Charlestown / East Boston waterfront the boat looks at, so
a thin strip of exactly the wrong things was missing. Re-extract with the same
bbox if these tables are ever rebuilt.

**Decision reversed 2026-08-12: we now do filter to far-field-visible
classes.** An earlier draft proposed it, a later draft retracted it, and the
retraction is itself now withdrawn. The retraction's technical argument was
sound and most of it still stands — it is recorded below, because the reasons
it gave are exactly the constraints the filter has to respect. What changed is
the trade: per-particle gating is *still unimplemented* (§10 item 4), so the
dilution coupling it was supposed to remove is live today, while 47 % of the
table is untagged building footprints and ~30 % is street furniture that no
observation from the water can ever correspond to.

**The full table is never modified.** `trim_landmark_feather.py` writes a
separate `harbor_osm_enc_trimmed_v1.feather`; `harbor_osm_enc_v1.feather`
remains the source of record and stays the right input for anything that
needs completeness (occlusion, compound skylines, shoreline — see below).
Consumers choose which file they load.

**The trim is guarded by the pairing labels, not by taste.**
`landmark_positive_set.py` freezes the run's labelled matches
(`runs/r003_full_leg1/pairing/`) into 58 distinct tag signatures — 132
positives, 53 instance and 79 category — and the trim reports recall against
them on every run. Current state: **184,805 → 13,210 rows (7.1 %) at recall
1.0000**. A rule that drops a labelled match is a bug in the rule.

The frozen JSON stores tag *signatures*, not catalog row ids, which is what
lets it keep working when the catalog is rebuilt — it guarded the widened-bbox
rebuild after the `r003` pairing directory it was derived from had been
deleted. `positive_set_r003_leg1.json` is now the only surviving copy of those
132 labels.

Guard coverage is 58 signatures from one leg, which is thin, and it has
already missed a real defect once: Bunker Hill Monument, a 67 m granite
obelisk tagged only `name` + `tourism=information`, was dropped while recall
still read 1.0. Spot-checks against known landmarks are part of the loop, not
optional.

What the retraction argued, and how the trim respects it:

- **Dilution is the wrong reason to filter.** The uniform
  `log_prior = −log(n)` couples the posterior to catalog size, but the fix is
  §5.3 `cand(x)` **per-particle spatial gating**, not a smaller map — measured
  at a ±2° wedge returning 579 candidates downtown versus 25–36 out in the
  harbor, so the dilution problem is local to the city. Gating remains the
  long-term answer and remains a prerequisite; trimming is not a substitute
  for it and does not close §10 item 4.
- **LOCI gates spatially too.** Set 2 in `landmark_pairing_cli.py` is the OSM
  landmarks on the satellite tiles covering that panorama. Our analogue is the
  bearing wedge, and it is unaffected by the trim.
- **The surveyor's prior survives.** All 883 ENC rows are kept, including the
  80 carrying `description: "visually conspicuous"`.
- **Container cranes survive.** They carry no name and no height and would
  fall to any prominence-based filter; `man_made=crane` is structural in the
  trim and there is a test asserting it.
- **The coastline survives** — all 85 `natural=coastline` ways.
- Generic entries are still handled by weight where they are kept: uniqueness
  weighting down-weights continuously, which stays strictly better than a hard
  cutoff for everything inside the trimmed set.

Rules, each voting independently so "only-this" is a real ablation:
`no_harbor_tags`, `unobservable_only`, `generic_small_building`. Unobservable
tags are two-tier — HARD (`highway=*`, benches, bus stops: a name does not
rescue them) and SOFT (`tourism=information`: a proper noun outranks a weak
tag). Bridges are deliberately exempt from the highway block, named or not.

Beyond the trim, generic entries are handled by weight, not exclusion:

- **Uniqueness weighting, not filtering.** LOCI's labeller already assigns a
  1–5 uniqueness score per landmark (1 = `building=yes`, 5 = Cloud Gate) and
  `export_correspondence_similarity --uniqueness_weighted` applies
  `1/log2(1 + N_matches)` (paper §III-C). A nondescript building is
  *down-weighted continuously*, which is strictly better than a hard cutoff
  and is already built and validated.
- Decode WKB geometry to region-anchored ENU (local ENU — **not** UTM, **not**
  raw lat/lon).
- `position_sigma_m` per source: ENC surveyed (tight) vs OSM (looser). The
  catalog projects this into the angular domain via `kappa_eff`, so the
  accuracy *class* matters more than the position.

**Dependency this creates:** per-particle gating (§5.3 `cand(x)`) is
specified but **not yet implemented** in the filter. It now moves from
"nice-to-have" to a prerequisite for using the full catalog. That is the
right trade — it is a bounded piece of filter work that permanently removes
the dilution coupling, versus a filter that permanently removes landmarks.

### Going beyond: what roads and nondescript buildings are actually for

Ordered by how soon they could pay off.

1. **Extended geometry instead of centroids.** OSM ships polygons and
   polylines; a bridge is a line, an island a polygon. Our measurement
   already carries an angular *width*, and the audit already classifies
   objects as `point_like` / `small_extended` / `large_extended`. Matching a
   bearing interval against a silhouette is both more informative and fixes
   the reference-point bias — the centroid of an extended object drifts with
   viewing aspect, which is a systematic bearing error, not noise.
2. **Compound landmarks synthesized from the boring rows.** A single
   `building=yes` is useless; ten thousand of them *are* the Boston skyline,
   which our detector reports as one landmark. Cluster footprints (with
   `building:levels` for height) into compound catalog entries so that
   large_extended detections like "city skyline" or "dense waterfront
   cluster" have something to match. Right now they have no counterpart at
   all — a recall hole that class-filtering would have made permanent.
3. **Occlusion and visibility.** Footprints + heights say what is actually
   visible from a pose over water, and what is hidden behind what. This is
   the direct use for the 83 k buildings: they are the occluders that explain
   absence.
4. **Negative / absence evidence.** A pose that predicts a prominent landmark
   where the panorama shows open water is evidence *against* that pose. This
   requires a detection-recall model to avoid punishing poses for our own
   misses, but it is the highest-information use of a complete catalog.
5. **Shoreline as a signature.** The land–water boundary as a function of
   bearing is a strong, continuous signal that no point-landmark formulation
   captures — and it comes from exactly the coastline ways a class filter
   would have discarded.

Items 1–2 are the ones that plausibly matter for this milestone; 3–5 are
research directions worth keeping visible.

---

## 4. Phase 3 — matching (four methods behind one seam)

All four emit the same thing: a `CompatibilityTable` per tracklet
(`landmark_id → log_lr`, plus `default_log_lr`, `clip_lo/hi`). The design
doc's posture is an **uncalibrated matcher behind a tuned transform + clip**,
with the clips and `pi0` carrying the safety burden — so a method does not
need calibrated probabilities to be usable, it needs a sane ranking.

**Method D — tag/name rule scoring (baseline + gate).**
Score a candidate from the merged landmark's weighted tags against the
candidate's OSM tags, plus name match (exact / alias / token-overlap), using
the ~15-entry tag-affinity table from the level-1 review. No training, fully
interpretable, and it doubles as the candidate gate. This is the control the
learned methods must beat.

**Method C — text-embedding retrieval.**
Embed the merged landmark's canonical description + distinctive features,
embed each candidate's tag-text, cosine similarity → `log_lr` via a tuned
affine transform. Reuses the existing sentence-embedding infrastructure.
Cheap and captures free-text ("red and white banded standpipe") that tags
cannot.

All of this reuses the LOCI stack (`~/code/loci-release`), whose stages map
onto ours one-for-one — the only substantive change is how Set 2 candidates
are gathered (satellite tiles → bearing wedge):

| LOCI stage | script | our analogue |
|---|---|---|
| Set 2 candidates | `landmark_pairing_cli.osm_landmarks_from_pano_id` (sat tiles) | bearing wedge from known camera pose |
| Label generation | `landmark_pairing_cli --with_negatives` → Gemini | same prompt, our Set 1 = merged landmarks |
| Text embeddings | `precompute_value_embeddings` (`text-embedding-005`, 768-d) | same, extend via `--base_embeddings` |
| Training | `train_landmark_correspondence` | fine-tune from `simple_v1_v5` |
| Scoring | `export_correspondence_similarity` | per-tracklet `log_lr` |

**Method A — the existing `simple_v1_v5` correspondence model.**
`landmark_correspondence_model.py` is a **tag-bundle** model: it encodes a
bundle of (tag-key index, 768-d tag text embedding, mask) for the pano side
and the OSM side, then classifies from
`[pano_repr, osm_repr, pano·osm, cross_features(4)]` → **a logit**. That
logit is already a log-odds, which drops straight into the `log_lr` seam with
a scale + clip. Two integration risks to settle before trusting it:

- *Embedding provenance*: the checkpoint was trained against
  `eval_text_embeddings_panov2_tuned_v5_all.pkl`. Our tag text must be
  embedded with the **same** embedder or the projection is meaningless.
- *`cross_features` (4)*: need to confirm what they encode. If any of them
  carry relative geometry, they are unavailable at match time — we do not
  know where we are, which is the whole point. If so, they must be zeroed or
  the model re-headed, and that changes what "using the existing model"
  means. **Check this first; it gates the method.**

Domain gap is real: trained on Chicago/Seattle VIGOR street-level panoramas
at short range, applied to a maritime far-field harbor. Expect a starting
point, not an answer — which is precisely what Method B is for.

**Method B — fine-tune on this environment, labelled the LOCI way.**
Reuse `landmark_pairing_cli`'s labeller verbatim — same system prompt, same
hard/easy negative definitions, same uniqueness 1–5 score, same JSON schema —
and change only how Set 2 is gathered. LOCI takes the OSM landmarks on the
panorama's satellite tiles; we take the OSM landmarks in the **bearing
wedge**: from the known camera position, along the tracklet's measured
bearing, spanning the bearing uncertainty and the measured angular width, out
to the visibility horizon. Gemini then labels matches + negatives inside that
wedge, and the trainer parses them into `CorrespondencePair`s unchanged.

Two things make the wedge better than a tile: it is *narrow* (a few degrees
rather than a whole tile), so the LLM sees a short, high-precision candidate
list; and it *accumulates* — a tracklet observed from many positions
intersects many wedges, and the intersection is far tighter than any one of
them. That intersection is triangulation, which doubles as the label check
below.

Guards, because auto-labels can be confidently wrong:
- Require a real baseline — bearing-only triangulation is ill-conditioned for
  distant objects seen over a short arc; gate on triangulation residual and
  on the conditioning of the intersection.
- Gate on local uniqueness: if two catalog entries of the same class sit
  within the triangulation uncertainty, the label is ambiguous — drop it.
- Hold out a spatial split (e.g. train on the first half of the leg, validate
  on the second) so we are not scoring memorization.
- Fine-tune from `simple_v1_v5` rather than training from scratch; leg1 alone
  will not support a from-scratch model.
- Keep the LLM's `uniqueness_score` — it feeds the same uniqueness weighting
  the catalog relies on instead of class filtering.

**Method E (stretch) — joint geometric consistency.** All tracklets must be
simultaneously consistent with one trajectory. This is really the filter's
data-association job, but a pre-pass that prunes candidates inconsistent with
*any* plausible pose is cheap insurance.

---

## 5. Phase 4 — localization

Inputs: catalog (Phase 2), `TrackletMeasurement`s with `bearing_body_deg`
(Phases 1–2), `CompatibilityTable`s (Phase 3), `OdometryDelta`s (Phase 1).

Run 1, as requested: **tight Gaussian prior on the true start position**
(`GaussianInit` at the first GPS fix, small sigma), local particle count.

**A caveat to be honest about up front.** GPS supplies both the odometry and
the ground truth here. With accurate world-frame GPS deltas and a tight
prior, dead reckoning alone will track well — so a small final position error
does *not* demonstrate that the landmarks are helping. In this configuration
the informative outputs are:

- **bearing residuals** — do matched bearings actually point at their catalog
  landmarks from the true pose?
- **association posteriors** (`AssociationPosterior`, `null_share`) — is the
  filter picking the intended landmark, or parking mass on the null?
- **health stream** — ESS, NEES (~2.0 ideal), reported vs actual error.

That makes Run 1 a *matcher* test, which is the right first question. Only
after it passes do we degrade the setup to test localization proper:
inject drift/noise into odometry to emulate SLAM, widen to a loose or global
prior, and compare against the dead-reckoning-only baseline. Deployment has
no GPS at all — it is a stand-in for SLAM — so the drift-injected run is
closer to the real problem than the clean one.

---

## 6. Evaluation harness (shared)

Triangulation from GPS + bearings is both the label generator for Method B
and the evaluation metric for all four:

- Per tracklet: triangulated position, uncertainty, and distance to the
  matched candidate.
- Per method: top-1 / top-k match rate against hand-checked tracklets,
  and the distribution of `log_lr` for intended vs unintended candidates
  (the input to tuning the seam transform and clips).
- Hand-label the ~40 substantial leg1 tracklets against real chart/OSM
  objects once. Every downstream stage inherits that endpoint metric, and it
  is the only way to tell a matcher improvement from a filter improvement.

---

## 7. Open risks

1. ~~`cross_features` semantics~~ **resolved**: pure tag-text, no geometry.
2. ~~Embedding provenance~~ **resolved**: `text-embedding-005`, extend the
   released pickle via `--base_embeddings`.
3. **Mount-offset precision** dominates far-field position error; the bow
   median-image calibration is the mitigation, not the ±2.8° eyeball value.
4. **Per-particle gating (§5.3 `cand(x)`) is now a prerequisite**, not an
   optimization: it is what lets the full 156 k-row catalog be used without
   the `log_prior = −log(n)` dilution coupling. Unimplemented today.
5. **Null-floor stranding** (design doc §8.4): a belief parked just outside
   the von Mises support feels no gradient. Relevant as soon as we move to a
   loose prior; the §5.5 proposal is the remedy and is not yet implemented.
6. **Leg1 alone is thin** for fine-tuning. Legs 2/3 exist and have panoramas
   prepared; extraction has not been run on them.
