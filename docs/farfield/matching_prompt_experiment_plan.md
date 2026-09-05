# Matching prompt experiment plan

Status: Phases 1–5 and the full Flevoland production-candidate rerun are
complete; the repository-default production prompt has not been promoted.

## Problem

The matcher currently asks one score to mean both:

1. whether a map object is compatible with the observed landmark; and
2. whether the evidence identifies that exact physical instance.

The prompt also defines `category` as “the right kind, but we cannot tell
which one” while telling the model not to return several related-but-distinct
objects. This makes exhaustive category matching ill-defined. Large requests
compound the problem: the production setting presents 10 observed tracks and
500 map signatures in each call.

Map data is open-world. An attribute absent from a map row is unknown, not a
contradiction. A matching optional attribute such as `colour=white` may be
positive evidence, but must not exclude otherwise-compatible rows that omit
`colour`. Conversely, an explicit incompatible value is negative evidence.
Names, references, inscriptions, and distinctive attributes can support an
instance match when the observation itself supports them. A unique value that
appears only in the map is not identifying evidence.

## Prompt designs

### Baseline

Use the current production prompt unchanged.

### Repaired single-score prompt

Keep the current response schema but define confidence conditional on the
reported match type:

- `category`: confidence that the map entry is compatible with the observed
  class and observed attributes. Return every compatible entry, including
  distinct physical objects. Candidates differing only in unobserved map
  attributes should receive equal scores.
- `instance`: confidence that this exact object is identified by evidence
  present on both sides. Map-side uniqueness alone is insufficient.

Missing map attributes are neutral. Explicit conflicts are negative evidence.
Supported names and references retain their identifying value.

### Split-score schema (follow-up only)

If the repaired prompt cannot produce calibrated category coverage and
instance precision with the current schema, separately request category
compatibility and exact-instance confidence. Do not implement this before the
single-score experiment shows it is needed.

### Equivalence-group schema (follow-up only)

If equivalent candidates still receive unstable scores, let the matcher return
sets of equally compatible map IDs plus a shared score. This is also deferred
until the simpler prompt is tested.

Hierarchical category/instance matching is explicitly out of scope.

## Phase 1: controlled small candidate sets

Run Gemini Pro with HIGH thinking on several observed tracks at once. Compare
the baseline and repaired prompts using fixed 10-signature candidate lists and
three deterministic orderings.

The panels cover:

- Flevoland wind turbines against one turbine plus unrelated objects;
- the same tracks against ten differently tagged/named turbines;
- labeled Flevoland power towers, farm/industrial buildings, and mast cases;
- labeled Pohang name-supported retail, apartment, bridge, and steelworks
  cases; and
- Pohang category/over-specificity controls: lighthouses, beacons, chimneys,
  industrial structures, and the incorrectly instance-matched museum.

The wind-turbine expectations are deliberately open-world: a name, operator,
output rating, or missing color on the map side cannot by itself distinguish a
turbine when the observation did not supply that information. Pohang
name-supported cases ensure the prompt does not achieve category recall by
discarding valid identifying evidence.

Measure:

- inclusion of labeled/expected signatures;
- coverage and score spread across compatible category candidates;
- incompatible-category false positives;
- instance precision, especially map-only unique names;
- stability across candidate order; and
- no-match confidence.

## Phase 2: attention and list-size scaling

Use the repaired prompt with two 10-track panels: labeled Flevoland tracks and
labeled Pohang controls. Embed a fixed block of relevant signatures in map
lists of 10, 50, 100, 250, and 500 signatures. Put the block at the beginning,
middle, and end while holding all other content fixed.

Interpretation:

- degradation with candidate count indicates attention/response-load effects;
- degradation with block position indicates positional sensitivity;
- failure at size 10 indicates prompt semantics rather than scale;
- strong category coverage but weak instance precision indicates that the
  split-score schema should be tested next; and
- good matching followed by poor localization moves the investigation to
  proposal coverage or trajectory likelihood.

## Downstream gate

Do not change the production matcher from these probes alone. Promote a prompt
only after it improves category recall and order stability without regressing
Pohang name-supported instance matches or increasing the museum-style
over-specific matches. A subsequent localization comparison should disable
hard association persistence and use multiple seeds.

## Phase 3: observed-track batch size

Hold each dataset's 500-signature list and ordering fixed, then submit the same
twenty Phase 2 tracks in batches of 1, 5, and 10. This isolates Set 1 batching
from candidate count and candidate order. Choose the largest batch whose
expected-ID recall, category enumeration, and instance typing are not worse
than the smaller batches.

## Phase 4: labeled non-turbine generalization

Run a balanced panel drawn from the human-reviewed Flevoland and Pohang
annotations, prioritizing tracks not used in Phases 1–3. The panel includes
explicit OSM-ID corrections, accepted original matches, and rejected/not-in-map
controls in shared 50-signature slices. This checks that the turbine repair
generalizes to buildings, towers, bridges, named businesses, maritime objects,
and absent objects.

## Phase 5: every reviewed track against the complete catalog

Run all 38 reviewed Flevoland tracks and all 80 reviewed Pohang tracks against
every signature in their frozen catalogs, in 500-signature chunks. Score
explicit OSM IDs when the referenced object exists in the frozen catalog,
retention of human-approved original matches, and instance false positives for
human-rejected original matches. Preserve unscorable reviewed tracks in the
output for qualitative inspection rather than inventing gold labels.

The complete-catalog run is an offline matcher evaluation only; it does not
rerun localization. Total expected spend for Phases 3–5 must keep cumulative
prompt-experiment spend below $50.

## Results (2026-09-04)

All calls used `gemini-3.1-pro-preview` with HIGH thinking and batches of
multiple observed tracks. Raw immutable requests, provider attempts, canonical
results, and summaries are under
`/tmp/farfield_matching_prompt_experiments/`.

### Phase 1

Fifteen calls per prompt covered five panels and three candidate orderings.

| Metric | Production prompt | Repaired prompt |
| --- | ---: | ---: |
| Expected-signature inclusion | 53/84 (63.1%) | 78/84 (92.9%) |
| Compatible-category coverage | 74/288 (25.7%) | 281/288 (97.6%) |
| Dense-turbine category coverage | 14/210 (6.7%) | 210/210 (100%) |
| Flevoland non-turbine expected inclusion | 12/18 | 18/18 |
| Pohang name-supported instance inclusion | 15/18 | 16/18 |
| Pohang general/category coverage | 12/21 | 16/21 |
| Forbidden museum instance match | 0/3 | 0/3 |

The repaired dense-turbine calls returned all ten turbine signatures for all
seven observed turbine tracks. Scores were equal within each call (0.9 or 1.0),
although the common score moved between calls. In the sparse panel both prompts
found the sole turbine. The production prompt therefore understands turbine
compatibility but fails to enumerate a dense set of equivalent candidates;
the semantic contradiction in the prompt is material.

The repaired prompt cleanly separated the test panels: every Flevoland,
lighthouse, beacon, chimney, and industrial-category result was `category`,
while every returned name-supported Pohang result was `instance`. It did not
match the map-only museum. The remaining expected misses were order-dependent
Sejan apartment matches and the POSCO candidate for an unnamed blast-furnace
observation. One lighthouse ordering also omitted the annotated green-light
entry, apparently conflating the observed white tower body with
`seamark:light:colour=green`.

### Phase 2

The first scaling pass used deterministic but non-nested filler samples. It was
retained as a robustness result but not used to attribute changes to list size.
The causal rerun used nested lists: every smaller candidate list is an exact
subset of the larger list. It issued 30 calls across two 10-track panels, five
candidate counts, and three positions for the fixed relevant block.

| Candidates | Flevoland expected | Flevoland compatible | Pohang expected |
| ---: | ---: | ---: | ---: |
| 10 | 30/30 | 156/156 | 26/27 |
| 50 | 30/30 | 156/156 | 26/27 |
| 100 | 26/30 | 152/156 | 27/27 |
| 250 | 26/30 | 152/156 | 27/27 |
| 500 | 24/30 | 150/156 | 27/27 |

All seven relevant wind-turbine signatures were returned for all seven
Flevoland turbine tracks at every size and position: 735/735 category decisions.
Scores remained equal or within 0.1 inside each call. All 15 museum traps were
rejected as instance matches. Pohang's only two misses were the Sejan apartment
in one ordering at sizes 10 and 50; it was retained at every larger size, so
there is no monotonic Pohang load failure or consistent position bias.

The Flevoland degradation above 50 candidates is entirely T76 and T358:

- T76 is a visually specific farmhouse whose annotated row expands from the
  very generic `building=yes` signature. At 500 candidates the matcher instead
  chose a more specific `building=farm` row and omitted `building=yes` in all
  three positions.
- T358 is an industrial building whose annotated row is
  `building=yes; complex:name=Trekkersveld III`. At 500 candidates the matcher
  returned five explicitly `building=industrial` signatures but omitted the
  less-specific annotated signature in all three positions.

This is a candidate-competition/tag-specificity failure, not a universal
500-entry attention collapse. Smaller chunks help, but the prompt still does
not reliably enumerate generic and specific compatible rows together when a
large candidate list offers more semantically specific alternatives.

### Interim decision after Phase 2

The repaired single-score prompt is substantially better and does not require
hierarchical matching. Do not move to the split-score or equivalence-group
schemas yet: the existing schema cleanly separated category and instance in
these probes. Before promotion, test one small follow-up focused on generic
versus more-specific tag bundles and decide whether production should use
roughly 50-entry chunks or an additional exhaustiveness instruction.

The useful baseline, repaired Phase 1, and nested Phase 2 calls cost about
$4.12 at recorded on-demand Pro token usage. The discarded non-nested scaling
pass cost about $3.15, for approximately $7.27 total experimental spend.

### Phase 3

Twenty fixed Phase 2 tracks were tested against the same 500 signatures at
Set 1 batch sizes 1, 5, and 10. All included expected signatures had the
intended match type, and no forbidden museum instance was returned.

| Set 1 batch | Expected IDs | Compatible rows |
| ---: | ---: | ---: |
| 1 | 18/19 | 60/61 |
| 5 | 16/19 | 58/61 |
| 10 | 17/19 | 59/61 |

Batch 1 recovered Flevoland T76's generic `building=yes` row, but all batch
sizes missed T358's generic industrial-building row. Batch 5 alone missed the
Pohang T156 apartment truth; batch 10 retained every scored Pohang truth.
There is no monotonic batching failure. Batch 10 was selected for Phase 5:
batch 1's one additional controlled inclusion would increase the full run from
144 to 1,424 requests.

### Phase 4

Seven calls used 50-signature slices and five observed tracks per request
(the final Flevoland request had two). The panel contained all 12 reviewed
non-wind Flevoland tracks and 20 varied Pohang tracks. Eighteen annotations had
a defensible scored positive; the other reviewed tracks remained qualitative
or absent-map controls.

- Expected-ID inclusion: 17/18. Flevoland was 6/6; Pohang was 11/12. The sole
  miss was T13, whose extraction says bridge/public building while the reviewed
  OSM truth is a ferry terminal.
- Match type: 15/17 included positives had the expected type. T127 promoted a
  name reported only by detections to `instance`; T152 promoted a map-side
  hospital name despite no supported name in Set 1.
- Rejected-original instance errors: 1/64. T111, an artwork absent from OSM,
  was instance-matched to a named viewpoint.

This generalizes the category repair beyond turbines, but the prompt still
occasionally treats uncorroborated or map-only names as instance evidence.

### Phase 5

The repaired prompt ran all 38 reviewed Flevoland tracks against all 3,753
frozen Flevoland signatures and all 80 reviewed Pohang tracks against all
6,508 frozen Pohang signatures. The run issued 144 batch-10/chunk-500 requests;
all 144 completed and passed schema validation.

Scoring is intentionally conservative. An explicit OSM URL is scored only if
that object exists in the frozen catalog. An unnegated `good`, `perfect`, or
`correct` review retains the original top match as a positive. Notes saying
`bad`, `not a good match`, or absent from OSM turn original returned rows into
instance-negative controls. Qualitative notes without a defensible identity
are preserved but not converted into invented gold.

| Cohort | Expected-ID inclusion | Correct type among all expected |
| --- | ---: | ---: |
| Flevoland approved original | 18/18 | 18/18 |
| Flevoland explicit OSM ID | 12/13 | 12/13 |
| Pohang approved original | 23/23 | 17/23 |
| Pohang explicit OSM ID | 10/14 | 10/14 |
| **Total** | **63/68 (92.6%)** | **57/68** |

The five missed truths have different causes:

- Flevoland T76 is the remaining clear prompt failure: a visually specific
  farmhouse is compatible with the generic truth `building=yes`, but the model
  returns more-specific farm rows and omits the generic row.
- Pohang T13 is extracted as a bridge/public pavilion but its truth is
  `amenity=ferry_terminal; ferry=yes; name=포항크루즈`.
- Pohang T239 is extracted as a commercial/office building but its truth is a
  named hospital.
- Pohang T223 and T390 observe apartment-tower clusters while their truth is a
  named `landuse=residential` complex. The current “object, not container” rule
  discourages exactly this cross-granularity match.

The repaired prompt returned all 78 wind-generator signatures for every one of
the 26 reviewed Flevoland turbine tracks: 2,028/2,028 category inclusions. It
also returned no museum row and no instance match for Pohang T295. However,
equivalent turbines can receive different scores when they occur in different
catalog chunks: the within-track score range was 0 for 10 tracks, 0.05 for 9,
0.1 for 5, and 0.2 for 2.

Four of 147 human-rejected original candidates were still returned as
`instance`: a named viewpoint for absent artwork T111, Songdo Bridge for T166,
and two named/ref-tagged distant buoys for T258. The six positive type errors
were T127, T152, T191, T364, T411, and T450; all promoted map-only or
uncorroborated identity details. Category result volume is intentionally large:
Flevoland returned a median of 78 rows per reviewed track (range 1–107), while
Pohang returned a median of 7 (range 0–731). Generic apartment and bridge
observations produce hundreds of plausible rows. Those are not counted as
false positives without exhaustive human labels.

## Current matcher-to-localizer probability pipeline

This section records the implementation as of 2026-09-05. It is a description,
not an endorsement of the model. In particular, the implementation currently
answers a different probabilistic question from the observation-likelihood
semantics proposed below.

### 1. LLM matching

`farfield/matching/match_landmarks.py` sends batches of observed tracks (Set 1)
and chunks of OSM signature groups (Set 2) to the LLM. Each returned row has a
map signature ID, `match_type` (`category` or `instance`), and confidence in
`[0, 1]`. The model also reports a track-level no-match confidence.

The same map signature may be returned in several chunk calls. Aggregation
keeps the maximum confidence for each signature; the `match_type` from that
maximum-confidence response wins. If at least one candidate is returned,
global no-match confidence is `1 - max(candidate confidence)`. Otherwise it is
the mean of the per-chunk no-match confidences.

### 2. Signature expansion and compatibility table

A signature can represent several physical OSM rows with the same canonical
tags. It is expanded back to all of those landmark IDs. After the configured
confidence floor, each retained confidence `c` becomes a clipped log odds:

```
llr_j = clip(log(c / (1 - c)), -4, 4)
```

The localization input stores only `{landmark_id: llr_j}` per observed track.
The LLM's `category` versus `instance` output does **not** reach the particle
filter. There is an audit-time downgrade from `instance` to `category` when a
signature expands to too many rows, but because match type is discarded, this
does not change localization likelihoods.

Unreturned catalog rows are assigned a common default log odds derived from
global no-match confidence and divided by the number of signature IDs. Thus
both returned confidences and the LLM's no-match result can influence the
relative landmark weights downstream.

### 3. Normalized latent-identity weights

`farfield/localization/filter.py::_identity_log_weights` converts the table
into a probability distribution over **all** catalog landmarks for that track.
With `r = matcher_recall` (currently `0.5`):

- returned landmarks collectively receive probability mass `r`, distributed
  by `softmax(llr_j)`;
- unreturned landmarks collectively receive mass `1 - r`, uniformly; and
- edge cases with no returned rows, or with every row returned, are normalized
  over the entire catalog.

Consequently `sum_j p(j | appearance) = 1`. If 500 indistinguishable turbines
are all returned with equal scores, each receives approximately `r / 500`.
Adding otherwise identical candidates therefore reduces the contribution of
the turbine that is geometrically consistent with a given particle. LLM
confidence is not merely an audit annotation: after a nonlinear logit and
softmax, it allocates likelihood between returned landmarks.

### 4. Per-frame observation likelihood

For particle state `x`, observed bearing `z`, map landmark `j`, and wrapped
bearing residual `delta_j(x)`, the non-persistent path evaluates a mixture:

```
p_current(z | x, M) =
    pi0 / (2*pi)
    + (1 - pi0) * sum_j p(j | appearance) * VM(delta_j(x); kappa)
```

`VM` is a von Mises bearing likelihood. The current Flevoland settings use
`pi0 = 0.2`, so every observation reserves a fixed 20% probability for a
uniform null/outlier process regardless of the track, frame, matcher output,
or local map density. Geometry enters through `delta_j(x)`, but the globally
normalized identity weight multiplies that geometric evidence first.

### 5. Persistence and no-persistence modes

Without persistence, the expression above is recomputed independently at each
epoch. A track seen in ten frames can associate to a different catalog row in
each frame, and it pays the normalized catalog/identity allocation again at
every observation.

With persistence, each particle also carries a latent landmark association.
The association is updated by an HMM-like mixture of retaining the previous
identity and renewing it from the current identity distribution. Current
defaults use renewal probability `0.1` and an association-specific uniform
outlier probability `0.1`. The outer fixed-null mixture remains part of the
measurement model. Persistence therefore rewards a single map identity that
explains a track over time, but it also makes an early wrong association sticky.

### 6. Why this design is disputed

The desired observation likelihood is simply

```
L(x) = p(observations | state=x, fixed OSM map M).
```

OSM is conditioning information, not a mutually exclusive class label to be
predicted. A turbine elsewhere in the catalog that is geometrically
inconsistent with state `x` should contribute essentially zero; its existence
should not reduce the evidence supplied by a compatible turbine at the correct
bearing. In particular, adding 499 remote but visually identical turbines
should not impose a `log(500)` penalty on the locally consistent explanation.

The current global identity normalization violates that invariance. It may be
appropriate for the question "which one catalog object generated this
appearance?", but that need not be the right factorization for evaluating
`p(z | x, M)`. It also creates three coupled design choices that need separate
justification:

- uncalibrated LLM confidence changes numeric particle likelihood, rather than
  only deciding which landmarks are compatible;
- `matcher_recall=0.5` reserves fixed mass for every unreturned map row, even
  though that is not an empirical detector-recall estimate; and
- `pi0=0.2` supplies a fixed null likelihood to every observation, rather than
  modeling track-specific reliability or map clutter.

The newly repaired prompt makes category enumeration deliberately exhaustive.
Under the current normalized-identity model, that success can perversely dilute
each returned category candidate. Category/instance is then discarded before
localization, so a visually proven exact identity and one of 500 generic class
compatibilities differ only through the LLM confidence number.

### 7. Candidate redesigns to revisit

No replacement is selected here. Useful alternatives are:

1. **Unnormalized compatible-cause potential.** Sum a compatibility gate times
   geometric likelihood without globally normalizing over OSM IDs. Equal
   category matches then have equal local strength. This needs an explicit
   treatment of local landmark density so that many co-located candidates do
   not inflate evidence without bound.
2. **Best-explanation likelihood.** Use the maximum compatible geometric term.
   It is invariant to remote duplicates and simple, but throws away legitimate
   support from multiple nearby observations and introduces hard switches.
3. **Noisy-or likelihood.** Treat each compatible row as a possible cause and
   combine bounded probabilities with `1 - product_j(1 - q_j K_j)`. This avoids
   linear count growth but requires calibrated per-row probabilities.
4. **Point-process/clutter model.** Model landmark-generated observation
   intensity plus an explicit background intensity. This is the most direct
   probabilistic route to accounting for local map density without a global
   identity simplex, but is a larger modeling change.
5. **Separate compatibility from identity.** Let category matching be a
   boolean/equal-weight compatibility gate. Use instance evidence only when an
   observed name, reference, or other identifying feature is corroborated on
   both sides. Do not use raw LLM confidence as likelihood unless it is
   calibrated against labeled data.

Null handling should be evaluated independently: a track-specific reliability
estimate, an explicit clutter intensity, or a robust contamination model may
all be better founded than a fixed null weight. The LLM's no-match score could
be evaluated as one input, but should not be assumed calibrated.

The immediate Flevoland rerun deliberately leaves this likelihood machinery
unchanged. It isolates the effect of the repaired matching prompt and then
compares persistence on versus off under otherwise identical settings. Those
runs are diagnostics of the current system, not validation of its probability
model.

### Final decision and next prompt changes

The repaired semantics solve the turbine exclusion failure and generalize well
enough to justify a production-candidate prompt, but not an unqualified rollout.
The next small regression should add three explicit rules without hierarchical
matching:

1. A generic map value such as `building=yes` remains category-compatible with
   a more specific observed subtype unless another tag explicitly conflicts.
2. `instance` requires a visually corroborated (`basis=both`) name/reference or
   other identifying evidence; `reported_by_detections` and map-only identity
   fields are insufficient by themselves.
3. For `large_extended` cluster observations, a mapped complex/landuse area may
   be a category match even though it contains the visible component objects.

Per-candidate confidence should not silently reintroduce a preference among
otherwise-equivalent category matches. Either normalize category scores after
chunk aggregation or add an explicit equivalence representation if a targeted
probe shows normalization would merge genuinely different hypotheses. Hard
association persistence should remain disabled for localization comparisons.

### Spend

Recorded Phase 3/4/5 on-demand Pro usage cost approximately $2.22, $0.49, and
$23.47 respectively, including thinking tokens at the output rate: $26.18 for
the requested phases. Including the earlier approximately $7.27 gives a
cumulative experiment spend of approximately **$33.45**, below the $50 cap by
about $16.55.

Phase 5 exceeded its $21.21 preflight guard estimate because its exhaustive
responses averaged about 10,989 output-plus-thinking tokens per request, above
the estimator's fixed 8,000-token assumption. The total authorization cap was
still respected, but future full-catalog Pro matching estimates should use at
least a 12,000-token response allowance.

## Full Flevoland repaired-prompt rerun (2026-09-05)

The exact Phase 5 repaired prompt was run against all accepted Flevoland
tracks. Every other matching setting remained the same as the earlier Pro run:
`gemini-3.1-pro-preview`, HIGH thinking, Set 1 batches of 10, Set 2 chunks of
500, confidence floor `0.05`, and batch transport. The immutable artifact is:

```
/data/farfield_matching/artifacts/landmark_matches/flevoland_polder/
  stage3_b847f55_osmv2_pro_repaired_v1
```

All 136 requests passed schema validation and covered all 169 tracklets. Actual
usage was 2,732,632 prompt tokens, 451,233 output tokens, and 808,658 thinking
tokens, approximately **$10.29** at the recorded batch rates.

| Matcher artifact | Tracks with >=1 retained row | Retained category rows | Retained instance rows | Median rows/track |
| --- | ---: | ---: | ---: | ---: |
| Earlier Pro prompt | 142/169 | 27,685 | 5 | 156 |
| Repaired prompt | 166/169 | 78,760 | 6 | 336 |

The repaired prompt therefore did what it was asked to do: it returned far
more compatible category candidates. The current localizer then globally
normalized across that larger set, so improved category coverage also diluted
the per-candidate identity weights.

Four 50,000-particle Torch/CUDA localization runs compared association
persistence on and off for seeds 0 and 1. All other settings, including
`matcher_recall=0.5` and `pi0=0.2`, were unchanged.

| Prompt | Persistence | Seed | Distance-normalized mass within 500 m | Final MAP error |
| --- | --- | ---: | ---: | ---: |
| Earlier Pro | on | 0 | 0.000105 | — |
| Earlier Pro | on | 1 | 0.000079 | — |
| Earlier Pro | off | 0 | 0.002297 | — |
| Earlier Pro | off | 1 | 0.002133 | — |
| Repaired | on | 0 | 0.000091 | 14.1 km |
| Repaired | on | 1 | 0.000081 | 19.2 km |
| Repaired | off | 0 | 0.001240 | 15.4 km |
| Repaired | off | 1 | 0.001063 | 11.0 km |

The earlier no-persistence values are recorded counterfactuals; the repaired
values are complete primary pipeline runs with their own manifests and
viewers. Disabling persistence improved the repaired-prompt 500 m metric by
roughly 13x on average, but neither seed localized. Moreover, the repaired
prompt's no-persistence mean (0.00115) was about half the earlier prompt's
no-persistence mean (0.00221).

This does not by itself prove that identity normalization is the sole cause:
the added candidates can also introduce genuinely confusing local geometry.
It is nevertheless the result predicted by the dilution objection. Better
category recall did not become better state evidence because the current
model divided fixed identity mass across the expanded compatible set. This
result should be retained as a regression target for any replacement
observation-likelihood formulation.
