"""Shared helpers for the local-LLM correspondence scaling benchmark.

The question this benchmark answers: *if we replaced the learned correspondence
model with a local LLM, how long would it take?* The correspondence model cheaply
scores, for each panorama (street-level) landmark, which OSM landmarks in the entire
city stockpile it could correspond to. An LLM instead has to read candidate OSM
landmarks into a finite context window, so covering the whole stockpile needs many
large-context calls per query.

This module provides:
  - data loading for one city (Chicago by default): the full OSM landmark stockpile
    and a pool of realistic panorama query landmarks,
  - a prompt builder framed for the *whole-stockpile search* task,
  - JSON validation for free-text model outputs,
  - a SweepPoint record + JSONL dump, and
  - an extrapolation/report that turns measured per-call latency and max usable
    context into a total call count and wall-clock for the full city.

It is imported by the benchmark binary (benchmark_correspondence_scaling_ollama).
"""

import dataclasses
import json
import math
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401  (must precede torch-importing modules)
import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.model.additional_panorama_extractors import load_v2_pickle
from experimental.overhead_matching.swag.scripts.landmark_pairing_cli import (
    extract_tags_from_pano_data, format_tags)


DEFAULT_VIGOR_BASE = Path("/data/overhead_matching/datasets/VIGOR/")
DEFAULT_PANO_BASE = Path(
    "/data/overhead_matching/datasets/semantic_landmark_embeddings/panov2_tuned_prompt")
DEFAULT_CITY = "Chicago"
DEFAULT_LANDMARK_VERSION = "v4_202001"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_osm_stockpile(city: str = DEFAULT_CITY,
                       vigor_base: Path = DEFAULT_VIGOR_BASE,
                       landmark_version: str = DEFAULT_LANDMARK_VERSION) -> list[str]:
    """Return the deduplicated OSM landmark stockpile for a city as tag strings.

    Each string is ``key=value; key=value`` (same formatting used in the real
    pairing prompts). Deduplication is by pruned tag set, matching
    ``landmark_pairing_cli.osm_landmarks_from_pano_id``.
    """
    config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version=landmark_version,
    )
    dataset = vd.VigorDataset(dataset_path=Path(vigor_base) / city, config=config)
    props = dataset._landmark_metadata["pruned_props"].values
    unique_props = sorted(set(props), key=lambda fs: sorted(fs))
    return [format_tags(sorted(p)) for p in unique_props if len(p) > 0]


def load_pano_query_landmarks(city: str = DEFAULT_CITY,
                              pano_base: Path = DEFAULT_PANO_BASE) -> list[str]:
    """Return a flat pool of panorama (street-level) landmark tag strings for a city.

    Reads only ``<pano_base>/<city>/embeddings/embeddings.pkl`` directly (rather than
    every city) and applies the same keep-list tag pruning as the pairing CLI.
    """
    pickle_path = Path(pano_base) / city / "embeddings" / "embeddings.pkl"
    data = load_v2_pickle(pickle_path)
    if data is None or "panoramas" not in data:
        raise FileNotFoundError(f"No v2.0 panorama pickle at {pickle_path}")
    queries: list[str] = []
    for pano_id, pano_data in data["panoramas"].items():
        landmarks = extract_tags_from_pano_data(pano_id.split(",")[0], pano_data)
        if not landmarks:
            continue
        for lm in landmarks:
            queries.append(format_tags(lm["tags"]))
    return queries


# ---------------------------------------------------------------------------
# Prompt construction (whole-stockpile search framing)
# ---------------------------------------------------------------------------

MATCH_SYSTEM_PROMPT = """You are a landmark matching expert. You are given two lists of \
OpenStreetMap-style tag bundles in key=value notation.

"Query landmarks" were extracted from street-level imagery. "Candidate landmarks" come \
from an OpenStreetMap map database. For each query landmark, identify which candidate \
landmarks (by id) could plausibly be the same physical object. A query may match zero, \
one, or several candidates. Only report matches you are confident about."""

MATCH_THINKING_SUFFIX = (
    "\n\nThink step by step about each query before answering, then give the final answer.")

MATCH_SCHEMA = {
    "type": "object",
    "required": ["matches"],
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["query_id", "candidate_ids"],
                "properties": {
                    "query_id": {"type": "integer"},
                    "candidate_ids": {"type": "array", "items": {"type": "integer"}},
                },
            },
        }
    },
}


def itemized(items) -> str:
    return "\n".join(f" {i}. {v}" for i, v in enumerate(items))


def build_match_prompt(query_landmarks: list[str],
                       candidate_landmarks: list[str]) -> str:
    """Build the user prompt for one whole-stockpile-search call."""
    return (f"Query landmarks (seen from street level):\n{itemized(query_landmarks)}\n\n"
            f"Candidate map landmarks (OpenStreetMap database):\n{itemized(candidate_landmarks)}")


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def extract_json_blob(text: str) -> str:
    """Best-effort: the substring from the first '{' to the last '}'.

    Lets the validators tolerate models (or emulated chain-of-thought) that wrap the
    JSON object in surrounding prose.
    """
    start = text.find("{")
    end = text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else text


def validate_json(text: str) -> tuple[bool, bool]:
    """Return (parsed_ok, schema_ok) for a model's free-text output.

    parsed_ok: text parses as JSON. schema_ok: it has a ``matches`` list whose items
    carry an int ``query_id`` and an int-list ``candidate_ids``.
    """
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False, False
    matches = obj.get("matches") if isinstance(obj, dict) else None
    if not isinstance(matches, list):
        return True, False
    for m in matches:
        if not isinstance(m, dict):
            return True, False
        if not isinstance(m.get("query_id"), int):
            return True, False
        cids = m.get("candidate_ids")
        if not isinstance(cids, list) or not all(isinstance(c, int) for c in cids):
            return True, False
    return True, True


# ---------------------------------------------------------------------------
# Sweep records
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SweepPoint:
    model: str
    thinking: str            # "on" | "off"
    thinking_emulated: bool  # True if "thinking on" was approximated (e.g. gemma3 CoT)
    n_candidates: int        # OSM candidates packed into context
    n_queries: int           # pano query landmarks per call (the batch size)
    prompt_tokens: int
    output_tokens: int
    prefill_s: float
    decode_s: float
    total_latency_s: float
    parsed_ok: bool
    schema_ok: bool
    ok: bool                 # call completed without OOM / fatal error
    note: str = ""

    @property
    def prefill_tok_s(self) -> float:
        return self.prompt_tokens / self.prefill_s if self.prefill_s > 0 else 0.0

    @property
    def decode_tok_s(self) -> float:
        return self.output_tokens / self.decode_s if self.decode_s > 0 else 0.0


def dump_sweep_points(points: list[SweepPoint], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in points:
            row = dataclasses.asdict(p)
            row["prefill_tok_s"] = p.prefill_tok_s
            row["decode_tok_s"] = p.decode_tok_s
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Extrapolation to a full city
# ---------------------------------------------------------------------------

def _batched_wall_s(point: SweepPoint, stockpile_size: int, num_query_landmarks: int) -> float:
    ctx_chunks = math.ceil(stockpile_size / point.n_candidates)
    query_batches = math.ceil(num_query_landmarks / point.n_queries)
    return ctx_chunks * query_batches * point.total_latency_s


def extrapolate(points: list[SweepPoint],
                stockpile_size: int,
                num_query_landmarks: int) -> list[dict]:
    """Project measured per-call cost to matching a whole city's stockpile.

    Groups points by (model, thinking). To give the LLM its best shot, "best-case
    batched" picks the *wall-clock-minimizing* operating point across the sweep (not
    simply the largest context): packing more candidates per call cuts the number of
    context-chunks but raises per-call latency (more prefill, and the model emits more
    claimed matches), so the optimum is usually an interior point. For the chosen
    point we report:

      best_case_batched : ceil(stockpile/lpc) * ceil(queries/pano_batch) calls
      one_at_a_time     : ceil(stockpile/lpc) * queries calls (pano_batch -> 1)

    one_at_a_time reuses the chosen point's per-call latency, so it is an *optimistic*
    lower bound for that strategy and still strictly worse than batched. Each group
    also carries the full per-operating-point curve.
    """
    groups: dict[tuple[str, str], list[SweepPoint]] = {}
    for p in points:
        if p.ok and p.n_candidates > 0:
            groups.setdefault((p.model, p.thinking), []).append(p)

    reports = []
    for (model, thinking), pts in sorted(groups.items()):
        curve = []
        for p in sorted(pts, key=lambda p: p.n_candidates):
            wall = _batched_wall_s(p, stockpile_size, num_query_landmarks)
            curve.append({
                "n_candidates": p.n_candidates,
                "latency_s": p.total_latency_s,
                "output_tokens": p.output_tokens,
                "prompt_tokens": p.prompt_tokens,
                "batched_wall_clock_s": wall,
                "batched_wall_clock_days": wall / 86400.0,
            })

        best = min(pts, key=lambda p: _batched_wall_s(p, stockpile_size, num_query_landmarks))
        lpc = best.n_candidates
        pano_batch = best.n_queries
        latency = best.total_latency_s
        ctx_chunks = math.ceil(stockpile_size / lpc)
        batched_calls = ctx_chunks * math.ceil(num_query_landmarks / pano_batch)
        one_at_a_time_calls = ctx_chunks * num_query_landmarks
        reports.append({
            "model": model,
            "thinking": thinking,
            "thinking_emulated": best.thinking_emulated,
            "landmarks_per_ctx": lpc,
            "pano_batch": pano_batch,
            "latency_at_chosen_ctx_s": latency,
            "prompt_tokens_at_chosen_ctx": best.prompt_tokens,
            "output_tokens_at_chosen_ctx": best.output_tokens,
            "prefill_tok_s": best.prefill_tok_s,
            "decode_tok_s": best.decode_tok_s,
            "ctx_chunks_for_stockpile": ctx_chunks,
            "operating_points": curve,
            "best_case_batched": {
                "calls": batched_calls,
                "wall_clock_s": batched_calls * latency,
                "wall_clock_days": batched_calls * latency / 86400.0,
                # Cost to compare ONE pano query landmark against the whole stockpile,
                # amortized over the pano_batch queries that share each context-chunk:
                #   ctx_chunks * latency / pano_batch
                "seconds_per_query_landmark": ctx_chunks * latency / pano_batch,
            },
            "one_at_a_time": {
                "calls": one_at_a_time_calls,
                "wall_clock_s": one_at_a_time_calls * latency,
                "wall_clock_days": one_at_a_time_calls * latency / 86400.0,
            },
        })
    return reports


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.1f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}min"
    if seconds < 2 * 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def print_report(reports: list[dict], stockpile_size: int, num_query_landmarks: int) -> None:
    print("\n" + "=" * 100)
    print(f"CORRESPONDENCE SCALING EXTRAPOLATION  "
          f"(stockpile={stockpile_size:,} OSM landmarks, "
          f"queries={num_query_landmarks:,} pano landmarks)")
    print("=" * 100)
    for r in reports:
        emu = " (emulated)" if r["thinking_emulated"] and r["thinking"] == "on" else ""
        print(f"\n{r['model']}  |  thinking={r['thinking']}{emu}")
        print(f"  operating points (candidates/ctx -> latency, out_tok, full-city days):")
        for c in r["operating_points"]:
            marker = "  *" if c["n_candidates"] == r["landmarks_per_ctx"] else "   "
            print(f"   {marker} {c['n_candidates']:>6,} -> {_fmt_duration(c['latency_s']):>7}/call, "
                  f"{c['output_tokens']:>6,} out_tok, {c['batched_wall_clock_days']:>8,.1f} days")
        print(f"  chosen (min wall)   : {r['landmarks_per_ctx']:,} candidates/ctx "
              f"(~{r['prompt_tokens_at_chosen_ctx']:,} prompt tok, "
              f"{r['output_tokens_at_chosen_ctx']:,} out tok), pano_batch={r['pano_batch']}")
        print(f"  throughput          : prefill {r['prefill_tok_s']:,.0f} tok/s, "
              f"decode {r['decode_tok_s']:,.0f} tok/s, "
              f"latency/call {_fmt_duration(r['latency_at_chosen_ctx_s'])}")
        print(f"  stockpile coverage  : {r['ctx_chunks_for_stockpile']} context-chunks per query-batch")
        b = r["best_case_batched"]
        o = r["one_at_a_time"]
        print(f"  PER PANO LANDMARK   : {_fmt_duration(b['seconds_per_query_landmark'])} "
              f"to compare 1 landmark against the whole stockpile "
              f"({r['ctx_chunks_for_stockpile']} chunks / pano_batch {r['pano_batch']})")
        print(f"  BEST-CASE BATCHED   : {b['calls']:,} calls  ->  "
              f"{_fmt_duration(b['wall_clock_s'])}  ({b['wall_clock_days']:,.1f} days)")
        print(f"  one-at-a-time       : {o['calls']:,} calls  ->  "
              f"{_fmt_duration(o['wall_clock_s'])}  ({o['wall_clock_days']:,.1f} days)")
    print("\n" + "=" * 100)
