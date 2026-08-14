# Audit: Milestone 0 — `swag/bearing_only_localization/`

> **Resolution (2026-08-11).** Every finding below was reproduced with the appendix probe before anything was changed; all the reported numbers replicated exactly. A-1 through A-5 and A-8 through A-11 are now fixed, and the probe re-run shows:
>
> | | audit | now |
> |---|---:|---:|
> | mean NEES (2 dof, ideal 2.0) | 309.3 | **1.0** |
> | seeds outside the 95% bound | 100% | **0%** |
> | position error vs reported σ | 111.9 m vs 8.8 m | **37.9 m vs 48.4 m** |
> | heading error vs reported σ | 1.62° vs 0.80° | **0.51° vs 1.27°** |
> | 5° crab: error / null-share | 590.7 m / 0.43 | **34.6 m / 0.006** |
> | duplicate epoch re-submitted | accepted, σ sharpens 1.29× | **rejected** |
> | heading with bearings removed | pinned to 1.3° | **uninformative (185°), as intended** |
>
> Two corrections to the audit's prescriptions, both empirical:
> - **A-1 fix.** The audit recommends dropping the Δ-course rotation and keeping course as an absolute measurement. That reintroduces a failure it did not test for: at a 90° waypoint corner an absolute-course update cannot rotate the cloud (all particles are ~90° off, so weights barely differentiate) and heading only creeps over at the random-walk rate — ~90 keyframes. The implemented fix keeps the Δ-course rotation and **deletes the absolute update**, so course is still consumed exactly once. Differencing has a bonus the audit's version needs a bias state to get: a constant COG-vs-heading offset cancels, which is why A-3's table is now flat without any bias state at all.
> - **A-2 root cause.** Fixing A-1 alone moved NEES 309 → 156, not to 2.0. The remaining overconfidence was **particle impoverishment** (NEES 67 / 12 / 4.6 at 4k / 20k / 100k particles). Fixed with kernel-regularized resampling, bandwidth `σ·n^(−1/6)`, now on by default — plain resampling is not a consistent option, so this is a correction to the earlier "roughening defaults to 0" decision, and `consistency_test.ResamplingRegularizationTest` pins it.
>
> Deliberately not done: **A-6** (per-mode association posteriors) needs the mode tracker; **A-7** is partly done (`kappa_eff` and map-accuracy classes now exist; speed-gated course noise, exclusion counters, mode/timing health fields and the event index do not); **A-5**'s underlying catalog-size coupling is *correct Bayes* for a uniform candidate prior and is resolved by per-particle spatial gating, not by a normalization change — it is now documented and its home is `catalog.py`; the `filter.py` module rename is cosmetic and was skipped.
>
> The findings below are left as written — they are the record of what was wrong, not a to-do list.

> **Superseded note (2026-08-13).** The §5.2 motion model this audit examined — world-frame GPS deltas with course consumed as Δ-increments — has since been redesigned to a body-frame dead-reckoning contract (parent §5.2, Milestone 4); the A-1 Δcourse machinery now lives in the odometry *producer*, not the filter. The findings and numbers below are the record of the archived design (branch `archive/bearing-loc-first-experiments`) and are intentionally not updated.

**Status:** Findings, not fixes. No code changed.
**Parent document:** [`localization-design-doc.md`](localization-design-doc.md). Section references of the form §N point into the parent; this document adds no new design decisions, it only reports where the Milestone 0 implementation (parent §10) diverges from what the parent already specifies.
**Audience:** Whoever picks up Milestone 0 remediation. Every finding carries a reproduction, so nothing here needs to be taken on trust.
**Scope:** All 2004 lines of `experimental/overhead_matching/swag/bearing_only_localization/` (13 files) as of the untracked working tree on branch `farfield-crossview`, read against the parent document.

---

## 1. Verdict

The architecture is sound and matches the parent document: the matcher seam is genuinely behind a contract, the arithmetic is log-domain throughout, the null hypothesis is mandatory and correctly floors the likelihood, the resampler is systematic rather than multinomial (as §10.3 asks), the geodesy fixtures are hand-derived (as T-U3 demands), and the run directory has the §7.5 shape. The `i0e`-scaled von Mises normalizer is correct and stable to κ=3000. The code is clean and unusually well-commented.

**The filter is nonetheless statistically broken, and the test suite cannot see it.** `bazel test //experimental/overhead_matching/swag/bearing_only_localization/...` is 5/5 green. Under a consistency check the filter reports ±8.8 m while sitting 111.9 m from truth — a mean NEES of 309 against an ideal of 2.0, with 20 of 20 seeds outside the 95% bound.

The root cause is a single modelling error (A-1), amplified by a missing state (A-3), and invisible because the one test class that would catch it (§8.2 T-F2) was never implemented (A-2). The remaining findings are secondary to those three.

Two structural observations about how this happened, which matter more than any individual bug:

- **The scenario generator and the filter share their assumptions exactly.** Same σ, same κ, exact von Mises, zero bias, perfect catalog positions, no outliers. The E2E suite therefore measures "does Bayes invert this generator," not "does this filter survive its own design document's stated operating conditions." Every model-mismatch case in §8.2 is absent.
- **Every assertion in the suite is on error *magnitude*, never on whether reported uncertainty is honest.** For a component whose entire output contract (§5.1: "MAP pose + covariance", association posteriors) is a belief consumed by something downstream, calibration is the property that matters most, and it is the one property untested.

---

## 2. Findings register

Severity: **C** = the filter produces confidently wrong answers today · **H** = contract in the parent document is unimplemented or violated · **M** = correct now, breaks at the next scale step · **L** = hygiene.

### A-1 · **C** · Course over ground is double-counted

`filter.py:241-261`. Each `course_deg` is consumed twice: once as a Δ-rotation of the heading state (`course_rad - prev_course_rad`, line 247) and again as an absolute von Mises measurement (`course_update`, line 258). The same random draw `c_k` enters both the prior and the likelihood.

The Δ-rotation telescopes: heading ≈ `h₀ + c_k − c₀`, so the propagated state already carries `c_k`'s noise, and the absolute update then "confirms" it. The two agree by construction. The filter therefore **never averages course noise down, but its confidence keeps rising anyway**:

```
final heading |error| over 20 seeds: mean = 1.62 deg, max = 3.24 deg
filter-reported heading sigma:       mean = 0.80 deg
scenario course_sigma_deg = 1.5
```

Over ~240 keyframes an honest filter reaches ≈0.1° (if the course samples are independent) *or* stays pinned at 1.5° (if they are not). This one reports 0.80° and delivers 1.62°.

The noise inflation at line 253-254 — `heading_sigma = sqrt(2·σ_course² + rw²)` — is a symptom, and its own comment says so: the proposal noise has to be widened "or bearings get nulled instead of correcting heading." That is the propagation and the update fighting each other.

**Fix:** keep exactly one mechanism. Recommend dropping the Δ-course rotation and retaining course as a measurement, which is what §5.2 describes.

**Corollary — the harness never exercises bearing-driven heading estimation.** A single course update collapses the uniform heading prior at the first keyframe, after which bearings are nearly irrelevant to heading:

```
             WITH bearings          WITHOUT bearings
kf   0    err  0.45  σ  36.79     err 145.71  σ 185.05
kf   1    err  1.01  σ   1.33     err   0.98  σ   1.52
kf   2    err  4.34  σ   1.32     err   4.33  σ   1.32
kf  20    err  0.81  σ   0.80     err   1.26  σ   1.31
```

From keyframe 1 onward the two columns are the same filter. §5.2's claim that "bearings observe heading" is untested by construction.

### A-2 · **C** · No consistency test; the filter is ~150× overconfident

§8.2 **T-F2** requires NEES within χ² bounds on synthetic runs with known truth. It is not implemented. It is also precisely the test that catches A-1 and A-3:

```
mean NEES = 309.3            (2 dof; ideal ≈ 2.0, 95% single-run bound = 5.99)
frac of runs with NEES > 5.99 = 1.00   (want ≈ 0.05)
mean position error = 111.9 m; mean reported sigma = 8.8 m
```

`e2e_convergence_test.py:46-51` asserts `median < 80 m` and `max < 200 m`, which the filter satisfies while being wrong about its own uncertainty by more than an order of magnitude in σ.

**Fix:** NEES-over-seeds and credible-set coverage as a first-class test, gating everything else.

### A-3 · **C** · The COG↔heading bias state specified in §5.2 is missing

§5.2 is explicit: *"Heading = course + a slowly-varying bias (leeway/crab from current and wind, ~5–10° worst case, plus constant mount offset): **model the bias as a random-walk state**; bearings observe heading, course observes heading − bias, so the bias is observable whenever landmarks are in view."*

There is no bias state. `course_update` (`filter.py:96-101`) treats course as a direct, unbiased observation of heading. For a vessel in Boston Harbor this is the expected operating condition, not an edge case:

| crab bias | position error | reported σ | heading error | mean null share |
|---:|---:|---:|---:|---:|
| 0.0° | 135.9 m | 8.5 m | 2.04° | 0.009 |
| 2.0° | 322.2 m | 9.0 m | 2.88° | 0.216 |
| 5.0° | 590.7 m | 8.5 m | 6.14° | 0.433 |
| 10.0° | 955.5 m | 10.0 m | 7.37° | 0.582 |

A 5° crab — mid-range in the parent document's own estimate — yields **591 m of error reported as 8.5 m**.

One thing worth recording as a success: `null_share` climbs 0.009 → 0.58, exactly the health-stream signature §8.4 predicts for this failure. The tripwire fires. Nothing reads it.

### A-4 · **H** · No information-epoch guard; duplicate evidence silently double-counts

§5.3 makes the information-epoch rule `[CONTRACT]` and T-F1 exists to police it. `run_filter` enforces nothing — it processes whatever is in `measurements`. Re-submitting the same tracklet set:

```
single inclusion:  sigma 8.5 m   heading sigma 0.794 deg   error 135.9 m
double inclusion:  sigma 7.3 m   heading sigma 0.617 deg   error 140.9 m
```

Posterior sharpens by ≈1.29× (≈√2 modulo resampling); accuracy does not improve. This is the frame-vs-tracklet error of §3.1 reappearing at the filter's own input boundary.

**Fix:** assert uniqueness of `(tracklet_id, anchor_keyframe_idx)` in `run_filter`, and implement T-F1.

### A-5 · **H** · `w_j = 1/n_cand` over the whole catalog; behavior is catalog-size dependent

`filter.py:132` normalizes by the global candidate count. §5.3 flags this exact hazard: *"(π₀, (1−π₀)·w_j) must form a proper mixture — this is easy to silently break when gating changes |cand(x)| per particle."* With M=3 it is invisible. Adding decoy landmarks shows the coupling — and that it is **not monotonic**:

```
n_cand =    3  ->  position error 135.9 m,  mean null share 0.0090
n_cand =   33  ->  position error  46.9 m,  mean null share 0.0125
n_cand =  303  ->  position error  45.8 m,  mean null share 0.0285
n_cand = 3003  ->  position error 169.4 m,  mean null share 0.1055
```

The 3→33 *improvement* is the tell: current accuracy rests on an accidental balance between an overconfident bearing term and the null floor. Any real catalog moves that balance. Per-particle spatial gating will move it again, per particle.

### A-6 · **H** · Association posteriors are not per-mode

§5.4 marks this `[CONTRACT]`: *"This output must be computed per mode, because 'mode A believes tracklet 7 is Graves Light; mode B believes it's Boston Light' is exactly the explanation the visualizer needs to surface."* `measurement_update` (`filter.py:140-149`) averages responsibilities over all particles. For a bimodal belief this averages two contradictory explanations into a number that describes neither. Acknowledged in the `AssociationPosterior` docstring as deferred to the mode tracker (§10.5), but it is a `[CONTRACT]` deferral and should be recorded as such.

### A-7 · **H** · Specified-but-absent state and health outputs

| Specified in | Requirement | Status |
|---|---|---|
| §5.2 | Course speed-gated, noise ∝ 1/(v·window) | `OdometryDelta.speed_mps` (`structs.py:70`) is logged and **never read**; `course_sigma_deg` is a fixed config constant |
| §5.3, §4 | `κ_eff` combines tracklet κ with landmark map-accuracy class | Not implemented; `LandmarkEntry` (`structs.py:22-27`) has no accuracy field |
| §5.3, T-F9 | Exclusion-violation counter in the health stream | Absent from `HealthRecord` |
| §5.6 | Mode count / per-mode weight+entropy, MAP jump magnitude, timing | Absent from `HealthRecord` |
| §7.3, §7.5 | Event index in the run directory | Not written |
| §4 | Bearings stored internally as unit vectors, not angles | Stored as angles throughout. Defensible for SE(2) given `wrap_rad` is applied consistently and `geodesy_test.py` covers it — but it is a `[CONTRACT]` deviation and is currently undeclared |

### A-8 · **M** · `measurement_update` allocates ~5 (N, M) temporaries with no chunking

`filter.py:125-141` builds `bearing_world`, `predicted_body`, `delta`, `log_cand`, `log_all`, `resp` at full (N particles × M candidates) float64. Fine at M=3; fatal at catalog scale:

```
N = 150 000 x M =     3   ->   ~0.02 GB
N = 150 000 x M =  2000   ->  ~12.0 GB
N =  20 000 x M = 50000   ->  ~40.0 GB
```

The §5.3 spatial gate will reduce M, but the update should block over the candidate axis regardless, accumulating log-sum-exp per block. Related: `filter.py:118-122` rebuilds the LLR lookup and runs an O(M) Python loop on every measurement.

### A-9 · **M** · Convergence is scored by the weighted mean pose

`position_errors_m` / `heading_errors_deg` (`filter.py:296-315`) use `mean_east_m`/`mean_north_m`. For the global-init test — uniform over 5×5 km, genuinely multimodal by design — the mean of the cloud sits between modes and describes no hypothesis the filter holds. §5.1 asks for MAP + covariance and a multimodality flag; the metrics should follow. Credible-set coverage is the honest scalar here.

### A-10 · **M** · Test-suite quality issues that will bite

- **`e2e_convergence_test.py:131` `test_low_pi0_ablation_collapses` is an anti-test.** It asserts the filter *does* produce a confident wrong fix (`final_std < 150 and error > 300`) with tight two-sided thresholds. Any improvement in clutter robustness breaks it. The §8.2 T-F4(b) ablation is a legitimate thing to demonstrate, but it should assert the *contrast* against the T-F4(a) configuration, not pin absolute failure magnitudes.
- **`e2e_convergence_test.py:98` `test_no_confident_wrong_fix` was tuned off the defaults** to `pi0=0.5`, `identity_default_log_lr=-4.0`. Its own comment states that at the shipped defaults (`pi0=0.05`, `-2.0`) "Bayes chases accidental alignments into a confident wrong fix." That is honest, and the reasoning is right — but it means the configuration anyone actually runs is known-unsafe under clutter and untested. Either the defaults should change or a test should hold the line on them.
- **Determinism is weaker than claimed.** `run_log.py:14-15` states Tier 1 is sufficient to "re-run the filter bit-exactly"; the tests only compare hashes in-process (`filter_test.py:179-189`). A numpy/BLAS/scipy version change silently invalidates the §7.1 replay contract. Pin a golden hash in CI or soften the claim to same-environment.
- Missing entirely from §8.2: **T-F3** (multimodality preservation — the symmetric two-lighthouse case, and the failure mode the parent calls "unrecoverable by design"), **T-F5** (LLR saturation vs. geometry, which is what makes the §6 clip bounds safe), **T-F10** (map-error robustness), **T-F11** (particle-count invariance).

### A-11 · **L** · Hygiene

- No config validation. `pi0 = 0` or `pi0 = 1` raises `ValueError` from `math.log`/`math.log1p` (`filter.py:132-134`). No check that `clip_lo ≤ clip_hi`, `n_particles > 0`, `kappa > 0`, or that `len(landmark_ids) == len(landmark_east_m) == len(landmark_north_m)` — a length mismatch broadcasts silently into wrong bearings rather than raising.
- No κ ceiling. A matcher handing an implausible κ gets unbounded authority over the posterior; §6's safety argument rests on clipping LLRs but nothing clips concentration.
- Missing compatibility table raises a bare `KeyError` at `filter.py:266`.
- `SCHEMA_VERSION` is written into the manifest and never validated on read (`run_log.py:88-95`).
- Stale docstring: `scenario.py:8` advertises "sqrt-distance noise"; the code uses constant `odom_sigma_m`, as the `ScenarioConfig` comment correctly explains.
- Weak typing throughout the seam: bare `list`/`dict` on `FilterHistory`, `RunData`, and every `run_filter` parameter. `run_filter` takes 7 positional arguments including three parallel landmark arrays.
- `filter.py` shadows the builtin (every consumer aliases it `as pf`) and reads confusingly against the existing `swag/filter/` package.
- Health records mix timing: `ess` is pre-resample, `mean_*`/`position_std_m` are post-resample-and-roughening. Correct as written, undocumented, and easy to misread in the viewer.

---

## 3. Recommended remediation order

1. **Fix the heading model** (A-1, A-3). Two states — heading and course bias — with bias as a random walk; course observes `heading − bias`; bearings observe `heading`. Delete the Δ-course rotation and the compensating noise inflation. One localized change addressing the two critical findings and most of the third.
2. **Make consistency the gate** (A-2). NEES over ~20 seeds within χ² bounds, plus credible-set coverage. Land this before any further features — it is the regression net for everything below.
3. **Put the mismatch knobs in the generator.** `course_bias_deg`, `bearing_bias_deg`, `outlier_frac`, `catalog_position_sigma_m`. The T-F suite then largely writes itself (T-F5, T-F10), and the §8.3 metric envelopes start meaning something. The A-3 table above is a ~6-line generator change.
4. **Introduce a `LandmarkCatalog` type** owning ids, positions, and accuracy class, with `candidates(particles) → CandidateSet`. One place for gating, one place for the `w_j` normalization (A-5), one place for `κ_eff` (A-7), and it removes the parallel-array threading and the length-mismatch hazard (A-11).
5. **Block the candidate axis** in `measurement_update` (A-8).
6. **Report MAP and credible set alongside the mean** (A-9); stop scoring multimodal runs by mean pose.
7. Then the guard (A-4), the health fields (A-7), and the hygiene pass (A-11).

## 4. Suggested amendments to the parent document

- §8.4 failure-mode register: add **course/heading evidence reuse** — using a course sample in both the propagation and the update prevents noise averaging while still concentrating the posterior; signature is a reported heading σ *below* the single-sample course σ. It is a sibling of the frame-vs-tracklet error in §3.1 and deserves the same billing.
- §10 Milestone 0 is marked "done." Recommend qualifying it: the harness, geodesy, logging, and plots are done; the heading model and the T-F suite are not, and the E2E convergence numbers currently recorded were produced by an inconsistent filter.

---

## Appendix — reproduction

Every number above comes from one throwaway `py_binary` in the package (added, run, removed; the working tree is unchanged). To regenerate, drop the script below at `experimental/overhead_matching/swag/bearing_only_localization/_audit_probe.py`, append the target to that package's `BUILD`, and run it.

```python
py_binary(
    name = "_audit_probe",
    srcs = ["_audit_probe.py"],
    deps = [
        requirement("numpy"),
        ":filter", ":geodesy", ":scenario", ":structs",
    ],
)
```

```python
"""Temporary audit probe -- DELETE ME."""
import dataclasses
import math

import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    filter as pf, geodesy, scenario, structs)

P = 5.0


def run(data, fc, meas=None, tables=None):
    return pf.run_filter(fc, data.landmark_ids, data.landmark_east_m,
                         data.landmark_north_m, data.odometry,
                         data.measurements if meas is None else meas,
                         data.tables if tables is None else tables)


def local_init(data, off_e=300.0, off_n=-200.0, s=500.0):
    t = data.truth[0]
    return structs.GaussianInit(t.east_m + off_e, t.north_m + off_n, s)


def circ_std_deg(h, w):
    R = math.hypot(float(w @ np.sin(h)), float(w @ np.cos(h)))
    R = min(R, 1.0 - 1e-15)
    return math.degrees(math.sqrt(-2.0 * math.log(R)))


def belief_stats(b, truth):
    w = b.normalized_weights()
    me, mn = float(w @ b.east_m), float(w @ b.north_m)
    de, dn = b.east_m - me, b.north_m - mn
    cov = np.array([[float(w @ (de * de)), float(w @ (de * dn))],
                    [float(w @ (de * dn)), float(w @ (dn * dn))]])
    err = np.array([me - truth.east_m, mn - truth.north_m])
    nees = float(err @ np.linalg.solve(cov + 1e-9 * np.eye(2), err))
    hd = math.atan2(float(w @ np.sin(b.heading_rad)),
                    float(w @ np.cos(b.heading_rad)))
    herr = abs(math.degrees(float(geodesy.wrap_rad(
        hd - math.radians(truth.heading_deg)))))
    return dict(pos_err=float(np.hypot(*err)), nees=nees,
                pos_sigma=math.sqrt(0.5 * np.trace(cov)),
                head_err=herr, head_sigma=circ_std_deg(b.heading_rad, w))


# A-1 corollary: is heading pinned by course alone?
cfg = scenario.harbor_loop(keyframe_period_s=P)
data = scenario.generate(cfg)
fc = structs.FilterConfig(n_particles=20000, seed=3,
                          init=structs.UniformBoxInit(-2500, 2500, -2500, 2500),
                          checkpoint_every=1)
h_meas = run(data, fc)
h_nomeas = run(data, fc, meas=[], tables={})
for kf in [0, 1, 2, 5, 20]:
    a = belief_stats(h_meas.checkpoints[kf], data.truth[kf])
    b = belief_stats(h_nomeas.checkpoints[kf], data.truth[kf])
    print(f"  kf{kf:3d} WITH: err={a['head_err']:6.2f} sig={a['head_sigma']:7.2f}"
          f" | NO: err={b['head_err']:6.2f} sig={b['head_sigma']:7.2f}")

# A-1: heading consistency -- reported sigma vs actual error.
errs, sigmas = [], []
for seed in range(20):
    fc = structs.FilterConfig(n_particles=4000, seed=seed,
                              init=local_init(data), checkpoint_every=1000)
    s = belief_stats(run(data, fc).final_belief, data.truth[-1])
    errs.append(s["head_err"]); sigmas.append(s["head_sigma"])
print(f"  heading |err| mean={np.mean(errs):.3f} max={np.max(errs):.3f}; "
      f"reported sigma={np.mean(sigmas):.3f}; course_sigma={cfg.course_sigma_deg}")

# A-2: position consistency (NEES, 2 dof, ideal mean ~2.0).
neeses, pes, pss = [], [], []
for seed in range(20):
    fc = structs.FilterConfig(n_particles=4000, seed=seed,
                              init=local_init(data), checkpoint_every=1000)
    s = belief_stats(run(data, fc).final_belief, data.truth[-1])
    neeses.append(s["nees"]); pes.append(s["pos_err"]); pss.append(s["pos_sigma"])
print(f"  mean NEES={np.mean(neeses):.1f}; frac>5.99="
      f"{np.mean(np.array(neeses) > 5.99):.2f}; err={np.mean(pes):.1f} m; "
      f"reported sigma={np.mean(pss):.1f} m")

# A-3: COG != heading (crab/leeway bias).
for bias_deg in [0.0, 2.0, 5.0, 10.0]:
    d2 = scenario.generate(scenario.harbor_loop(keyframe_period_s=P))
    d2 = dataclasses.replace(
        d2,
        truth=[structs.TruthPose(t.keyframe_idx, t.east_m, t.north_m,
                                 (t.heading_deg - bias_deg) % 360.0)
               for t in d2.truth],
        measurements=[structs.TrackletMeasurement(
            m.tracklet_id, m.anchor_keyframe_idx,
            m.bearing_body_deg + bias_deg, m.kappa) for m in d2.measurements])
    fc = structs.FilterConfig(n_particles=4000, seed=5,
                              init=local_init(d2), checkpoint_every=1000)
    h = run(d2, fc)
    s = belief_stats(h.final_belief, d2.truth[-1])
    nulls = [a.null_share for r in h.health for a in r.associations]
    print(f"  bias={bias_deg:5.1f} -> err={s['pos_err']:7.1f} m, "
          f"sigma={s['pos_sigma']:6.1f} m, head_err={s['head_err']:5.2f}, "
          f"null={np.mean(nulls):.3f}")

# A-5: catalog dilution -- w_j = 1/n_cand over the whole catalog.
base = scenario.generate(scenario.harbor_loop(keyframe_period_s=P))
rng = np.random.default_rng(0)
for n_decoy in [0, 30, 300, 3000]:
    de = np.concatenate([base.landmark_east_m, rng.uniform(-3e4, 3e4, n_decoy)])
    dn = np.concatenate([base.landmark_north_m, rng.uniform(-3e4, 3e4, n_decoy)])
    ids = base.landmark_ids + [f"decoy_{i}" for i in range(n_decoy)]
    fc = structs.FilterConfig(n_particles=4000, seed=5, init=local_init(base),
                              checkpoint_every=1000)
    h = pf.run_filter(fc, ids, de, dn, base.odometry, base.measurements,
                      base.tables)
    s = belief_stats(h.final_belief, base.truth[-1])
    nulls = [a.null_share for r in h.health for a in r.associations]
    print(f"  n_cand={len(ids):5d} -> err={s['pos_err']:7.1f} m, "
          f"null={np.mean(nulls):.4f}")

# A-4: correlated evidence -- resubmit the same tracklet.
fc = structs.FilterConfig(n_particles=4000, seed=5, init=local_init(base),
                          checkpoint_every=1000)
s1 = belief_stats(run(base, fc).final_belief, base.truth[-1])
dup = sorted(base.measurements * 2,
             key=lambda m: (m.anchor_keyframe_idx, m.tracklet_id))
s2 = belief_stats(run(base, fc, meas=dup).final_belief, base.truth[-1])
print(f"  single: sigma={s1['pos_sigma']:.1f} head_sigma={s1['head_sigma']:.3f}"
      f" err={s1['pos_err']:.1f}")
print(f"  double: sigma={s2['pos_sigma']:.1f} head_sigma={s2['head_sigma']:.3f}"
      f" err={s2['pos_err']:.1f}")
```
