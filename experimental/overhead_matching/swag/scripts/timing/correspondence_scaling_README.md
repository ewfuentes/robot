# Local-LLM correspondence timing benchmark

Measures how long a local LLM would take to replace the learned correspondence model:
prompted to match a panorama's landmarks directly against a city's full OpenStreetMap
landmark set. This backs the timing comparison in the paper (Gemma 3 27B: ~5.4 min /
panorama without thinking, ~7.6 min with chain-of-thought, vs the 3.7 s learned step).

## What it does

The correspondence model cheaply scores each panorama landmark against the whole city
stockpile. An LLM must instead read candidate OSM landmarks into a finite context window,
so covering the full stockpile (Chicago: 28,582 unique tag bundles) requires streaming the
candidates through in chunks. The benchmark measures single-stream per-call latency
(prefill + decode) on the local GPU at a range of context fills and query-batch sizes, then
extrapolates to per-landmark / per-panorama / full-city cost.

- `correspondence_scaling_common.py` — library: loads the Chicago OSM stockpile (VIGOR
  `v4_202001` landmarks) and panorama query landmarks (pano_v2 `panov2_tuned_prompt`),
  builds the match prompt, validates JSON output, and extrapolates per-call timings to a
  whole city.
- `benchmark_correspondence_scaling_ollama.py` — runnable benchmark for `gemma3:27b` via
  ollama.

## Setup (one-time, not CI-reproducible — needs a local GPU + ollama)

1. Install ollama (https://ollama.com) and ensure the `ollama` binary is on PATH. The
   `gemma3:27b` weights (~17 GB) are pulled automatically on first run.
2. The VIGOR dataset and pano_v2 embeddings must be present under
   `/data/overhead_matching/datasets/` (see `data/README.md`).

## Run

```bash
bazel run //experimental/overhead_matching/swag/scripts/timing:benchmark_correspondence_scaling_ollama -- \
  --candidate_fills 500 --pano_batches 1,2,3,5 --thinking off \
  --gpu_safe_num_ctx 65536 --output_headroom 8192 \
  --output_dir /tmp/corr_scaling/gemma3_off

# thinking on (chain-of-thought; gemma3 has no native thinking mode):
#   --thinking on
```

## Output

- stdout: one line per measured call (prompt/output tokens, prefill & decode tok/s, latency)
  plus an extrapolation report (operating-point curve, chosen point, per-pano-landmark time,
  full-city totals).
- `<output_dir>/sweep_points.jsonl`: machine-readable per-point record — the traceable
  source of the reported numbers.

## Reference results

The sweep results behind the paper figures are stored at
`/data/overhead_matching/evaluation/timing/`:
- `gemma3_27b_thinking_off_per_panorama.jsonl` — ~5.4 min / panorama
- `gemma3_27b_thinking_on_cot_per_panorama.jsonl` — ~7.6 min / panorama

## Notes

- "single-stream" = one request at a time (the latency a robot processing observations
  sequentially would see), no concurrent request batching.
- Gemma 3 has no native thinking mode (ollama reports `completion`+`vision` only), so
  `--thinking on` emulates it with a chain-of-thought instruction; that output is free text,
  so schema-valid JSON is not enforced in that mode.
