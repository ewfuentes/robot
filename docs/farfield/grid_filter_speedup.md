# Fused joint-factor backend

The Portland joint filter spends almost all its forward-pass time computing
whole-track likelihoods. Each epoch previously produced many grid-by-candidate
intermediates through separate PyTorch operations. The optional fused backend
compiles that epoch calculation with `torch.compile(dynamic=True, fullgraph=True)`
and reduces intermediate allocations, GPU kernel launches, and memory traffic.

Use the existing natural-release joint configuration with:

```
--joint_backend fused --joint_chunk 128
```

`reference` remains the default backend and the default chunk remains 64.
`eager_optimized` runs the new expression without compilation for diagnosis.
The first compiled invocation incurs compilation/cache-loading overhead.

## Changes

- Preserve the full heading/position grid and all endorsed candidate identities.
  The mixture, null/tail mass, range gate, uncertainty terms, tempering, and cap
  remain the same model in FP32. No hypotheses or observations are pruned.
- Precompute heading/epoch scalar parameters and pass them as tensors. Their
  values can change between tracks without compiling a separate specialized
  graph for each observation's Python scalar values.
- Fuse the per-epoch coordinate transforms, distance/bearing calculations,
  variance and von Mises likelihood, range penalty, and log-product update.
  Candidate reduction and final mixture/cap stay outside this kernel.
- Expose candidate chunk size; the 16/32/64/128 microbenchmark selected 128.
- Keep a reference path and numerical tests for empty/singleton/multiple
  candidate sets, short/long tracks, cap/no cap, range floor, tempering,
  quantization compensation, and candidate chunks of 16/64/128.

## Measured validation

On an RTX 3090 Ti with Torch 2.7.0+cu128, Portland seed 0, divided tables,
natural releases, joint slack, cap 20, 100 m grid cells and 36 headings:

| Metric | Reference | Fused |
|---|---:|---:|
| Instrumented causal forward pass | 1027.70 s | 143.71 s |
| p95 update | 11.76 s | 1.64 s |
| Maximum update | 96.94 s | 11.07 s |
| Causal posterior mass within 500 m | 0.3646487915 | 0.3646487818 |

The causal speedup is 7.15x. Maximum per-keyframe 500 m mass difference was
2.38e-7; aggregate causal and legacy lag-30-keyframe differences were below
1e-8. The uncapped Portland aggregate difference was 4.1e-9. This is numerical
agreement, not bit identity. The representative warmed factor microbenchmark
fell from 4.73 s to 0.57 s; most of the gain came from fusion, with chunk tuning
providing a smaller additional gain.

These measurements came from the isolated overnight experiment implementation
at commit 821e3ca, as reported by Harel. This port onto PR #721 retains its
sum-mixture API and removes the experimental extent/max-mixture options.
It preserves #721's log-space accumulation of co-released smoothing factors.
The timings above have not been remeasured on this port.
Compilation caches were warmed; a fresh-cache startup benchmark is still needed.
The 7.15x claim concerns causal filtering, not end-to-end extraction/matching or
the full smoothing workload. Thirty keyframes are not thirty seconds in Portland.

Local provenance: `/data/farfield_matching/runs/260913_overnight/`, including
`checks/backend_cuda.json`, `checks/full_run_gate.json`, run metadata, and logs.

## Follow-up robustness fixes

Panorama-box unions use the smallest circular enclosing interval, so seam
merges cannot exceed one full turn. Legacy overwrapped boxes are treated as
full-circle observations and excluded from bearing/range fusion. Localization
exports record these exclusions and omit matching tables only for tracks with
no remaining directional measurements; the paid matching artifact is retained.
Natural-release schedule construction verifies the exclusions against the
bound source geometry. These input-generation fixes apply with or without
smoothing.

The optional likelihood cache now evicts least-recently-used entries within
its host-memory budget. Invalid budgets are rejected and oversized tensors
bypass the cache without evicting useful entries. `--likelihood_cache_gb 0`
still disables it.

Fixed-lag and final smoothing combine forward/backward messages through the
existing log-space normalization helper. This preserves tiny overlapping
tails that underflow when multiplied directly in float32, while genuinely
disjoint support still fails. These two smoother paths are inactive in
causal-only runs. Regression tests cover circular geometry, exclusion/export
lineage, cache accounting and numerical parity, and tiny-tail normalization.
