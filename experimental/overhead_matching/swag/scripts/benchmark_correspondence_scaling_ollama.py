"""Sweep gemma3:27b (via ollama) over growing OSM-candidate context fills.

Measures prefill/decode throughput and the max usable context on the local GPU for
the whole-stockpile landmark-correspondence task, then extrapolates the calls and
wall-clock to match a whole city (Chicago by default).

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:benchmark_correspondence_scaling_ollama -- \
        --output_dir /tmp/corr_scaling/gemma3_27b \
        --candidate_fills 100,1000,5000,10000,20000 \
        --pano_batch 10 --thinking both

The Gemma 3 family is not a native reasoning model, so "thinking on" is best-effort:
we try ollama's think=True and fall back to a chain-of-thought system instruction
(free-text output, JSON parsed from the tail), flagged as emulated in the report.
"""

import argparse
import functools
import math
import time
from pathlib import Path

from common.ollama.pyollama import Ollama

print = functools.partial(print, flush=True)  # noqa: A001 - stream progress to redirected logs
from experimental.overhead_matching.swag.scripts import correspondence_scaling_common as common


def _next_pow2(n: int) -> int:
    return 1 << max(0, math.ceil(math.log2(max(1, n))))


def _choose_num_ctx(prompt_tokens_est: int, output_headroom: int, max_num_ctx: int) -> int:
    return min(max_num_ctx, max(4096, _next_pow2(prompt_tokens_est + output_headroom)))


def _run_one(client, model, system, user, num_ctx, think, use_schema):
    """One chat call. Returns (response_obj, error_str_or_None)."""
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        options={"num_ctx": num_ctx, "temperature": 0.0},
    )
    if use_schema:
        kwargs["format"] = common.MATCH_SCHEMA
    if think is not None:
        kwargs["think"] = think
    try:
        return client.chat(**kwargs), None
    except Exception as e:  # noqa: BLE001 - want to record OOM/unsupported-think as data
        return None, f"{type(e).__name__}: {e}"


def _ns_to_s(ns) -> float:
    return (ns or 0) / 1e9


def _sweep_thinking_mode(client, model, thinking, fills, pano_batch, queries, stockpile,
                         gpu_safe_num_ctx, output_headroom, tokens_per_cand):
    """Run the candidate-fill sweep for one thinking mode. Returns list[SweepPoint]."""
    points = []
    emulated = False
    for fill in fills:
        candidates = stockpile[:fill]
        query_landmarks = queries[:pano_batch]
        user = common.build_match_prompt(query_landmarks, candidates)
        system = common.MATCH_SYSTEM_PROMPT

        think_arg = None
        use_schema = True
        if thinking == "on":
            think_arg = True  # first attempt: native thinking

        prompt_est = int(tokens_per_cand * (fill + pano_batch)) + 2000
        needed_ctx = prompt_est + output_headroom
        # Don't even submit a prompt that would need a KV cache too big for the GPU --
        # record it as the max-usable-context ceiling and stop growing the fill.
        if needed_ctx > gpu_safe_num_ctx:
            note = f"needs num_ctx~{needed_ctx} > gpu_safe ceiling {gpu_safe_num_ctx}"
            print(f"  [b{pano_batch} {thinking}] fill={fill}: STOP ({note})")
            points.append(common.SweepPoint(
                model=model, thinking=thinking, thinking_emulated=emulated,
                n_candidates=fill, n_queries=pano_batch, prompt_tokens=0, output_tokens=0,
                prefill_s=0.0, decode_s=0.0, total_latency_s=0.0,
                parsed_ok=False, schema_ok=False, ok=False, note=note))
            break
        num_ctx = _choose_num_ctx(prompt_est, output_headroom, gpu_safe_num_ctx)
        t0 = time.perf_counter()
        resp, err = _run_one(client, model, system, user, num_ctx, think_arg, use_schema)

        # gemma3 doesn't support native thinking -> emulate with CoT, free-text output.
        if err is not None and thinking == "on" and "think" in err.lower():
            emulated = True
            system = common.MATCH_SYSTEM_PROMPT + common.MATCH_THINKING_SUFFIX
            t0 = time.perf_counter()
            resp, err = _run_one(client, model, system, user, num_ctx,
                                 think=None, use_schema=False)

        if err is not None:
            print(f"  [b{pano_batch} {thinking}] fill={fill}: STOP ({err})")
            points.append(common.SweepPoint(
                model=model, thinking=thinking, thinking_emulated=emulated,
                n_candidates=fill, n_queries=pano_batch, prompt_tokens=0, output_tokens=0,
                prefill_s=0.0, decode_s=0.0, total_latency_s=0.0,
                parsed_ok=False, schema_ok=False, ok=False, note=err))
            break  # larger fills will only fail harder

        wall = time.perf_counter() - t0
        content = resp.message.content or ""
        prompt_tokens = resp.prompt_eval_count or 0
        output_tokens = resp.eval_count or 0
        prefill_s = _ns_to_s(resp.prompt_eval_duration)
        decode_s = _ns_to_s(resp.eval_duration)
        total = prefill_s + decode_s if (prefill_s + decode_s) > 0 else wall
        parsed_ok, schema_ok = common.validate_json(common.extract_json_blob(content))

        points.append(common.SweepPoint(
            model=model, thinking=thinking, thinking_emulated=emulated,
            n_candidates=fill, n_queries=pano_batch,
            prompt_tokens=prompt_tokens, output_tokens=output_tokens,
            prefill_s=prefill_s, decode_s=decode_s, total_latency_s=total,
            parsed_ok=parsed_ok, schema_ok=schema_ok, ok=True,
            note=f"num_ctx={num_ctx}"))
        p = points[-1]
        print(f"  [b{pano_batch} {thinking}] fill={fill}: {prompt_tokens} ptok, {output_tokens} otok, "
              f"prefill {p.prefill_tok_s:,.0f} tok/s, decode {p.decode_tok_s:,.0f} tok/s, "
              f"latency {total:.1f}s, json_ok={schema_ok}")
    return points


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemma3:27b")
    ap.add_argument("--city", default=common.DEFAULT_CITY)
    ap.add_argument("--thinking", choices=["on", "off", "both"], default="both")
    ap.add_argument("--candidate_fills", default="100,500,1000,2000,4000",
                    help="Comma-separated OSM-candidate counts to pack into context")
    ap.add_argument("--pano_batches", default="1,10,40",
                    help="Comma-separated pano-query batch sizes to sweep (queries per call)")
    ap.add_argument("--gpu_safe_num_ctx", type=int, default=65536,
                    help="Largest num_ctx to submit; fills needing more are recorded as the ceiling")
    ap.add_argument("--output_headroom", type=int, default=4096,
                    help="Tokens reserved for the model's output in num_ctx sizing")
    ap.add_argument("--call_timeout_s", type=float, default=900.0,
                    help="Per-request HTTP timeout; a slower call is recorded as the ceiling")
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()

    fills = [int(x) for x in args.candidate_fills.split(",") if x.strip()]
    pano_batches = [int(x) for x in args.pano_batches.split(",") if x.strip()]
    modes = ["off", "on"] if args.thinking == "both" else [args.thinking]

    print(f"Loading OSM stockpile and pano queries for {args.city}...")
    stockpile = common.load_osm_stockpile(city=args.city)
    queries = common.load_pano_query_landmarks(city=args.city)
    print(f"  stockpile: {len(stockpile):,} OSM landmarks | query pool: {len(queries):,} pano landmarks")
    fills = [f for f in fills if f <= len(stockpile)] or [len(stockpile)]

    all_points = []
    with Ollama(args.model) as server:
        client = server.client
        # Bound each request so a pathological large-context prefill is recorded as the
        # ceiling instead of hanging the whole sweep.
        client._client.timeout = args.call_timeout_s
        # Warm-up + tokens-per-candidate estimate (excluded from results).
        warm_user = common.build_match_prompt(queries[:pano_batches[0]], stockpile[:50])
        warm, err = _run_one(client, args.model, common.MATCH_SYSTEM_PROMPT, warm_user,
                             num_ctx=8192, think=None, use_schema=True)
        if err is not None:
            raise RuntimeError(f"Warm-up call failed: {err}")
        tokens_per_cand = max(5.0, (warm.prompt_eval_count or 2500) / 50.0)
        print(f"Warm-up: ~{tokens_per_cand:.1f} prompt tokens per candidate landmark")

        for mode in modes:
            for pano_batch in pano_batches:
                print(f"\n=== thinking={mode}, pano_batch={pano_batch} ===")
                all_points += _sweep_thinking_mode(
                    client, args.model, mode, fills, pano_batch, queries, stockpile,
                    args.gpu_safe_num_ctx, args.output_headroom, tokens_per_cand)

    out_path = args.output_dir / "sweep_points.jsonl"
    common.dump_sweep_points(all_points, out_path)
    print(f"\nWrote {len(all_points)} sweep points to {out_path}")

    num_query_landmarks = len(queries)
    reports = common.extrapolate(all_points, stockpile_size=len(stockpile),
                                 num_query_landmarks=num_query_landmarks)
    common.print_report(reports, stockpile_size=len(stockpile),
                        num_query_landmarks=num_query_landmarks)


if __name__ == "__main__":
    main()
