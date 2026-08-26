# Localization: bearings, matching, filter runs, and forensics

## The scientific seam

Localization does not read tracker pages or raw provider responses. It reads
typed products with explicit lineage:

1. the semantic audit defines the valid evidence segments for each accepted
   track;
2. `tracking/tracklets.py` applies the support gate and exposes only those
   accepted segments;
3. `tracking:build_bearing_observations` fuses supported observations into
   body/camera-frame bearing measurements;
4. matching supplies a weighted compatibility table, including a null
   alternative;
5. `localization:build_export` publishes `localization_inputs` from bearings,
   matches, the typed catalog, motion, and approved nominal-forward data.

There is no geometric merge stage. Multiple tracks of one physical object may
remain separate; uncertainty belongs in association rather than an irreversible
pre-filter weld.

Tracking distinguishes the last supported keyframe from the last propagated
keyframe. Audit segments are bounded by supported evidence, so an unsupported
or mask-dead tail can be reviewed without leaking into bearings. A `keep`
decision may trim an unreliable same-object tail; a partial-identity decision
is reserved for a real object switch. Dropped or malformed evidence remains
counted in the audit ledger.

## Frames and nominal forward

`nominal_forward.py` owns the approved camera-to-platform-forward rotation.
Nominal forward is a fixed platform axis: it is not GPS course and is never
inferred automatically from a sun check or landmark sweep. Diagnostics may
produce reviewable evidence, but only an approved, dataset-bound calibration
record can rotate localization bearings.

The export records the calibration bytes, frame identifier, source path, and
all upstream artifact references. A missing, wrong-dataset, unapproved, or
frame-inconsistent calibration fails before publication.

## Matching semantics

Machine-only matching is a valid primary autonomous experiment. Low-confidence
or ambiguous candidates are not silently converted into a single identity;
they remain weighted hypotheses alongside a null score. Configuring a human
identity-review directory creates a distinct human-assisted lineage and must
be labeled as such.

Exact pair counts can grow with catalog signature expansion and should not be
read as independent identity claims. Regression checks emphasize request and
tracklet coverage, retained instance identity, null mass, and calibrated
uncertainty—not raw category-expanded row count alone.

## Running the filter

`localization:run_export` consumes an immutable `localization_inputs` artifact
and the recorded localization section of `build_config.json`. Scientific
settings cannot be overridden at the run command. The run records the build,
input artifact, backend, replay inputs, and complete resolved configuration.

**Only a uniform-prior, bearing-enabled run is a primary evaluation.** A
truth-centered initialization, odometry-only run, oracle association, or
other privileged input is a labeled diagnostic. These controls are valuable
for attribution but do not answer whether a uniformly initialized agent can
localize cross-view.

The measurement backend may be NumPy or Torch. The Torch backend is packaged
with the localization launcher, uses CUDA when available, and chunks the
candidate dimension to bound device memory. It is required to be numerically
equivalent within tested tolerances, not bit-for-bit identical; the selected
backend is part of the run record.

The main regression metric is the **time-normalized posterior probability mass
within 500 m of the true position**.  The 100 m score is calculated alongside
it by default as a tighter companion metric.  At each keyframe, the evaluation
observer records

    p_N(k) = P(||X_k - x_truth(k)|| <= N).

The run score is the trapezoidal area under `p_N(k)` divided by the keyframe
span.  It is therefore normalized to `[0, 1]`, comparable across run lengths,
and higher is better.  A score of 1 means all posterior mass stayed within the
radius for the whole run; a lucky MAP point cannot hide probability mass held
by wrong modes.  The every-keyframe series is stored in `tier0_health.jsonl`,
and the immutable run also publishes the aggregate in `metrics.json`.

Truth is used only by this evaluation observer after each posterior update. It
does not initialize the prior, alter a measurement, trigger a proposal, or feed
back into the filter.  Only uniform-prior, bearing-enabled, non-ablation runs
may present the 500 m score as a primary evaluation; other runs label it as a
diagnostic-control score.

Plots, the run viewer, and the run CLI put the 500 m score first and mark it as
primary.  The 100 m score follows it, while MAP/mean error are secondary point
diagnostics.  Also inspect null share, effective sample size/resampling,
proposal accept/reject events, residual distributions, mode weights, and
entropy.  Rejected proposal events remain linked in health/forensics output
instead of disappearing from the record.

## Reading a run

- `localization:viewer` builds the self-contained run viewer; the optional
  server adds live replay/crop endpoints over the same recorded payload.
- `localization:plot_run` writes maps, strips, and animation products.
- `localization:satellite_underlay` adds an optional licensed imagery layer;
  source tiles are not redistributed.
- `localization:forensics` checks invariants, attributes measurements,
  replays exact inputs, explores explicitly labeled counterfactuals, and
  traces one tracklet or event.

Presentation outputs are derived from the completed run. They do not become
scientific inputs and do not invalidate the run when viewer code evolves.
