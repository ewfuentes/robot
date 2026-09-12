# 2026-09-11/12: why Flevoland and the Portland flight did not localize, and what moved

Scope: the first grid-filter pass on the regenerated paper inputs
(`runs/260911_causal_promatch_epson_v1`, four datasets, natural / eager / none
availability). Boston leg 1 (0.64) and Mount Washington leg 2 (0.86) were fine;
Flevoland (0.057) and the Portland flight (0.036) were not. Everything below
uses `dn_mass_500` (distance-normalized truth mass within 500 m) unless stated.
Truth-conditioned numbers are diagnostics, never results.

Code: branch `offline-localization` (archived as
`archive/farfield-matcher-prompt-v3-260912`); the tracker fix is its own PR
against `farfield-crossview-base`. Pre-change snapshot tag
`snapshot-260911-pre-debug`.

## 1. Diagnosis

A per-measurement attribution tool (`localization/grid_truth_attrib.py`,
privileged: it reads truth) records, for every measurement applied by the eager
grid filter, the factor it multiplies onto the truth state's mass and which
catalog candidate dominates the likelihood at truth versus at the posterior
argmax. Truth-initialized runs (σ = 150 m) lose truth on both datasets, so the
likelihood contradicted truth; it was not a search failure.

### Portland flight

1. **Identity posterior ignored the matcher's absolute confidence.**
   `_identity_log_weights` gave the endorsed set a fixed `matcher_recall` = 0.5
   however weak the table. One track (T195) had a single 0.05-confidence
   category guess 40 km away with matcher no-match 0.95; it received weight 0.5
   and its 16 epochs cost truth −17 nats on their own. Fix:
   `--identity_share noisy_or` (share = recall · (1 − ∏(1 − c_j))).
2. **Within-table log-odds softmax amplified arbitrary category splits**
   (quarry rows at 0.6 vs the true Gray Pit at 0.05 → 50:1). Fix:
   `--identity_within uniform|prob`.
3. **Information vacuum kf 80-360.** Zero truth-consistent endorsed candidates
   among ~160 measurements; a quarter of tables empty. A privileged override
   that injects the 38 labelled correct OSM ids lifts natural 0.24 → 0.69,
   eager 0.88. In 7 of 8 misses checked in the request logs the correct row
   was in the slice the model saw (school, WPOR-FM and WKZS-FM towers, Parsons
   Pond, Blackstrap Road overpass, Westbrook Quarry, Evergreen Cemetery with
   the name extracted) and it answered no-match 0.9-1.0: prompt policy
   ("Returning no match is the expected outcome … Never settle for the closest
   thing present"), not retrieval.
4. **Duplicate tracks** (see §2).

### Flevoland

1. Identical wind turbines on a lattice. At truth exactly one endorsed turbine
   is bearing-consistent (near, 550-980 m, softened by cell quantization);
   at the posterior mode 3 km away 4-24 far turbines are co-aligned (a row
   seen end-on), so the sum-mixture pays 4-24× per epoch.
2. Turbine tables list 245-336 rows; the LLM gave ten of them confidence 1.00
   and the rest 0.05 for no visible reason, and the odds softmax turned that
   into ~1000:1, leaving the true turbine (annotated "good") at weight ≈ 0.
3. Range buckets: the reducer takes the tightest bucket in the epoch; 75 % of
   tracks carry 2-3 distinct buckets and the minimum excludes the true object
   for 7/53 Flevoland and 6/16 Portland tracks (mode: 1/53, 3/16). Yet turning
   the cap off or flooring it is worse everywhere: the cap also suppresses far
   aliases.

## 2. Track splintering (fixed; PR against `farfield-crossview-base`)

11-18 % of non-rejected tracks per dataset sat in sustained co-alive duplicate
groups (Portland 122 tracks / 51 groups). Every labelled example was concurrent,
not sequential:

* **Same-keyframe twins**: one predicted landmark with boxes in several pitched
  faces that fail the seam test becomes `__box0`, `__box1` observations, each
  seeding a track (28 pairs on Portland, 1-3 elsewhere).
* **Re-birth on an eroded track**: a same-class detection classifies
  `none`/`context` against a shrunken SAM2 mask, seeds a new track, the old one
  starves (48 / 34 / 33 pairs on Portland / Flevoland / Boston).

`track_builder.seed_unassigned` now (a) re-anchors an unsupported alive track
onto a same-class detection that covers ≥ 50 % of its mask bbox or lands
within the drift gate (`reanchor_rebirth`), absorbs it as a `duplicate` vote
when the track is already supported, and (b) folds same-class unclaimed
detections whose pano boxes overlap (IoU ≥ 0.5 or containment ≥ 0.8) into one
seed (`seed_duplicates` in the birth record). Thresholds are recorded config
fields. Portland re-track (`object_tracks/…/v3dedup_20260912_v1`): tracks
780 → 654, tracks in duplicate pairs 122 → 71, 103 re-anchors, 37 twins folded,
matched tracks 196 → 177 with the same length distribution.

## 3. Results

Portland, natural / eager `dn_mass_500`, baseline filter config:

| inputs | labelled targets endorsed | empty tables | landmarks / table p50 / p90 | natural | eager |
|---|---|---|---|---|---|
| original tracks, prompt v2 | 14 / 38 | 24 % | 5 / 55 | 0.036 | 0.050 |
| de-duplicated tracks, prompt v2 | 8 / 35 | 40 % | 2 / 35 | **0.220** | **0.529** |
| de-duplicated tracks, prompt v3 | **27 / 35** | 1 % | 225 / 1030 | 0.212 | 0.400 |
| privileged oracle identities | 38 / 38 | – | 1 | 0.69 | 0.88 |

Identity-fixed filter configs on the same inputs: noisy-OR + uniform 0.181 /
0.361 (v2) and 0.122 / 0.208 (v3); noisy-OR + prob 0.198 / 0.320 (v3). On the
original inputs the ordering was reversed (noisy-OR + uniform 0.243 / 0.389 vs
baseline 0.036 / 0.050) because those inputs contained the stray
confident-wrong singletons; the de-duplicated inputs contain none (worst track
−1.6 nats), so the sharper baseline weighting wins there.

Flevoland natural (matches unchanged):

| config | dn500 |
|---|---|
| baseline | 0.057 |
| noisy-OR + uniform | 0.054 |
| whole-track joint factor at release, motion slack, temper 0.5 | 0.230 |
| joint + max-mixture (`--mixture max`) | **0.265** |
| joint + max + per-track evidence cap 5 / 20 / 100 | 0.03 / 0.02 / 0.01 |
| baseline, reducer range cap = mode / track_median | 0.086 / 0.084 |
| joint + max, reducer range cap = mode / track_median | 0.193 / 0.196 |

Boston leg 1 / Mount Washington leg 2 across all of the above: 0.64-0.76 /
0.82-0.87, no config regressed either below its baseline except the joint on
Mount Washington (0.864 → 0.82).

## 4. Matcher prompt v3 (archived, not adopted)

`matching.prompt_variant = dossier_category_sets_v3` asks for every same-kind
row as a category match with probability-like confidence (≈ 1/n, floor 0.05)
and reserves no-match for "no row of this kind". It fixed recall (27 / 35
labelled targets, empty tables 70 → 1) and destroyed precision: median 225
landmarks per table, confidences flat inside a set (the correct row's share is
0.006 under prob weighting vs 0.0054 uniform; it is top-confidence in 3 of 27
hits). A bearing to "one of 1000 buildings" is uninformative, so localization
did not move. The sets that carried its hits were the rare kinds: hangars (21
rows), quarries (46), runways (30-107), golf course (30). That is the ceiling
of category matching for nameless objects: it helps in proportion to how rare
the kind is in the catalog.

Matcher variance is large: an identical bridge dossier scored 0.7 (Veterans
Memorial Bridge) in one v2 run and no-match 0.95 in the next; labelled-target
endorsement went 14 / 38 → 8 / 35 on the same prompt. Single-run prompt A/Bs
need to clear that noise.

Cost: audits $2, each matching run ≈ $20 (batch), total ≈ $43.

## 5. Dead ends (all measured on four datasets)

range cap off; range-cap floor 0.2; hull-extent bearing variance (30 km rivers
and forests become flat votes); 50 m / 5° grid (no gain, Portland OOM at chunk
256); `max_conf` identity share (hurts Flevoland); max-mixture on independent
epochs (Flevoland 0.006); per-track joint evidence cap (kills Flevoland,
barely helps Portland); softer range gate σ = 1.0 (neutral).

## 6. Recommendations

1. Merge the tracker de-duplication (this alone: Portland 0.036 → 0.220
   natural). Re-track Flevoland, Boston and Mount Washington with it before
   the next paper pass so all lineages share one tracker.
2. Matcher v4: keep v3's head, add a hard cap (rank same-kind rows by tag and
   description agreement, report at most ~30, and say "too common" for kinds
   with hundreds of rows in the slice). Judge on labelled-target endorsement
   *and* set size, then dn500, and run it twice or on Boston + MtW as well
   before adopting.
3. Keep `--identity_share noisy_or` available: it is insurance against the
   stray confident-wrong singleton that the v2 prompt emits stochastically.
4. Flevoland needs a range cue that is not the extractor bucket and not a
   per-class size model; the joint factor is the right structure but is only
   as good as the epoch-level caps that gate it.

## 7. Where things are

Artifacts (Portland): `object_tracks/…/v3dedup_20260912_v1`,
`semantic_audits/…/v3dedup_20260912_v1`, `bearing_observations/…/v3dedup_20260912_v1`,
`landmark_matches/…/promatch_20260912_dedup_v1` (v2 prompt) and
`…/promatch_20260912_dedupv3_v1` (v3), `localization_inputs/…/promatch_20260912_dedup_epson_v1`
and `…dedupv3_epson_v1`. Builds `builds/portland_flight_20260906_leg1/v3dedup_20260912_m1`
and `…_m2_v3prompt`; recipes in `~/scratch/farfield_recipes/260912_dedup/`.
Flevoland / Boston / MtW / Portland re-exports with `range_cap_reduction =
mode|track_median` live under `localization_inputs/<ds>/promatch_20260911_epson_v1_cap*_v1`.
Pipeline nit: the alignment-diagnostics stage refuses plugged-in bearings from
another build, so A/B builds that reuse tracks run `--only localization_inputs`
after `match`.
