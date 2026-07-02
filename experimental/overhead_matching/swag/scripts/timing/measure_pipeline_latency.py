"""Measure per-observation wall-clock latency for the WAG/SWAG pipeline.

Times each stage independently on N sample panos from a VIGOR city so we can
report a per-observation latency budget. The paper framing is "online-capable
under a ground-robot motion budget" (the histogram filter tolerates arbitrary
inter-observation gaps via its motion model; what matters is that a new
observation finishes before the robot has moved farther than the filter's
noise can absorb), not strict realtime. The local LLM and sentence encoder
stand in for Gemini / Vertex text-embedding-005.

Stages timed (see /home/erick/.claude/plans/this-isn-t-quite-right-eventual-tide.md):
  1. pano_encode       - pano_model forward on one pano image
  2. ollama             - single-shot Ollama call on 4 yaw pinhole images
  3. sentence_embed    - local SBERT on any novel text values from stage 2
  4. correspondence    - CorrespondenceClassifier: (K pano x N osm) cost matrix
  5. aggregate         - per-sat Hungarian matching + aggregation -> (num_sats,)
  6. filter_step       - HistogramBelief.apply_observation + apply_motion

Stage 6 uses a dummy obs_log_ll of the correct shape (the output of stage 5
is in similarity space, not log-likelihood; converting requires a wag config
that isn't load-bearing for the per-step wall clock claim).

Usage:
    # With Ollama running at localhost:11434
    bazel run //experimental/overhead_matching/swag/scripts:measure_pipeline_latency -- \\
        --num-samples 5 --warmup 1 \\
        --city Chicago \\
        --dataset-path /data/overhead_matching/datasets/VIGOR/Chicago \\
        --paths-path <paths.json> \\
        --landmark-version <version> \\
        --pano-model-path <...> --sat-model-path <...> \\
        --correspondence-model-path <...> \\
        --osm-text-embeddings-path <eval_text_embeddings.pkl>
"""

from __future__ import annotations

import common.torch.load_torch_deps  # noqa: F401 - before torch

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import ollama as ollama_sdk
import torch

from common.gps import web_mercator
from experimental.overhead_matching.swag.data import vigor_dataset as vd
from experimental.overhead_matching.swag.data.landmark_correspondence_dataset import (
    load_text_embeddings,
)
from experimental.overhead_matching.swag.evaluation import (
    correspondence_matching as cm,
)
from experimental.overhead_matching.swag.filter.histogram_belief import (
    GridSpec,
    HistogramBelief,
    build_cell_to_patch_mapping,
)
from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    CorrespondenceClassifier,
    CorrespondenceClassifierConfig,
    TagBundleEncoderConfig,
)
from experimental.overhead_matching.swag.model.semantic_landmark_extractor import (
    SYSTEM_PROMPTS as GEMINI_SYSTEM_PROMPTS,
    encode_image_to_base64,
    get_osm_tags_schema,
    osm_tags_user_prompt,
)
from experimental.overhead_matching.swag.scripts.evaluate_histogram_on_paths import (
    HistogramFilterConfig,
    get_dataset_bounds,
    get_patch_positions_px,
)
from experimental.overhead_matching.swag.scripts.ollama_osm_extraction import (
    OSM_EXTRACTION_SCHEMA,
    OSM_TAGS_SYSTEM_PROMPT,
    resolve_pano_images,
)
from sentence_transformers import SentenceTransformer


STAGE_ORDER = [
    "pano_encode",
    "ollama",
    "sentence_embed",
    "correspondence",
    "aggregate",
    "filter_step",
]


# Hand-picked Chicago panos labeled "useful" in
# ~/scratch/overhead_matching/landmark_tagger.labels.json — spread across the
# city, each has pinhole images on disk and is in the VIGOR Chicago dataset.
# Labeled "useful" means the pano is likely (not guaranteed) to contain
# visually distinctive, extractable landmarks.
DEFAULT_CITY = "Chicago"
DEFAULT_PANO_IDS = [
    "--3sSXMbqhZCCvFQ6MdcLw",
    "-DPsUZCxedknT0Dpm_V3aQ",
    "-THW7X9eiN_P3Oe6kvbhhg",
    "-jKwcfMnYuqH57z1RfZUDQ",
    "3EzUc70iL-thb0PoE5FlJw",
    "bZpr7sZuhvAKghFO5BHOoQ",
]
# Synthesized motion delta for stage 6. Magnitude ~8 m N + 8 m E at mid-Chicago
# latitude (each step in a real filter run is typically 1-10 m).
DEFAULT_MOTION_DELTA_DEG = (7.2e-5, 9.6e-5)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_block(fn, device: torch.device):
    _sync(device)
    t0 = time.perf_counter_ns()
    result = fn()
    _sync(device)
    return result, (time.perf_counter_ns() - t0) / 1e6  # ms


def parse_ollama_landmarks(content: str) -> list[dict[str, str]]:
    """Convert Ollama structured-output JSON into a list of tag dicts."""
    parsed = json.loads(content)
    tag_dicts: list[dict[str, str]] = []
    for lm in parsed.get("landmarks", []):
        tags: dict[str, str] = {}
        primary = lm.get("primary_tag") or {}
        if primary.get("key") and primary.get("value"):
            tags[primary["key"]] = primary["value"]
        for at in lm.get("additional_tags", []) or []:
            if at.get("key") and at.get("value"):
                tags[at["key"]] = at["value"]
        if tags:
            tag_dicts.append(tags)
    return tag_dicts


def build_histogram_inputs(vigor_dataset, device: torch.device):
    """Set up GridSpec, CellToPatchMapping, and uniform belief (mirrors
    evaluate_histogram_on_paths.py setup). Returns (belief, mapping, config)."""
    config = HistogramFilterConfig()
    min_lat, max_lat, min_lon, max_lon = get_dataset_bounds(vigor_dataset)
    footprint_px = config.footprint_px
    cell_size_px = footprint_px / config.subdivision_factor
    patch_half_size_px = footprint_px / 2.0

    center_lat = (min_lat + max_lat) / 2
    ref_y, ref_x = web_mercator.latlon_to_pixel_coords(
        center_lat, min_lon, config.zoom_level
    )
    buf_lat, _ = web_mercator.pixel_coords_to_latlon(
        ref_y - patch_half_size_px, ref_x, config.zoom_level
    )
    _, buf_lon = web_mercator.pixel_coords_to_latlon(
        ref_y, ref_x + patch_half_size_px, config.zoom_level
    )
    lat_buffer = buf_lat - center_lat
    lon_buffer = buf_lon - min_lon

    grid_spec = GridSpec.from_bounds_and_cell_size(
        min_lat=min_lat - lat_buffer,
        max_lat=max_lat + lat_buffer,
        min_lon=min_lon - lon_buffer,
        max_lon=max_lon + lon_buffer,
        zoom_level=config.zoom_level,
        cell_size_px=cell_size_px,
    )
    patch_positions_px = get_patch_positions_px(vigor_dataset, device)
    mapping = build_cell_to_patch_mapping(
        grid_spec=grid_spec,
        patch_positions_px=patch_positions_px,
        patch_half_size_px=patch_half_size_px,
        device=device,
        max_chunk_bytes=int(config.max_chunk_gib * 1024 ** 3),
    )
    belief = HistogramBelief.from_uniform(grid_spec, device)
    return belief, mapping, config


def build_osm_tag_lookup(vigor_dataset):
    """Collect unique OSM landmark tag dicts + per-sat column lookup.

    Returns:
        osm_tags_list: [dict[str, str]] indexed by column in the cost matrix.
        sat_col_positions: list of list[int]; sat_col_positions[sat_idx] gives
            the cost-matrix columns for that sat patch's landmarks.
    """
    osm_lm_idx_to_tags: dict[int, dict[str, str]] = {}
    num_sats = len(vigor_dataset._satellite_metadata)
    for sat_idx in range(num_sats):
        lm_idxs = vigor_dataset._satellite_metadata.iloc[sat_idx].get(
            "landmark_idxs", []
        )
        if lm_idxs is None:
            continue
        for lm_idx in lm_idxs:
            if lm_idx in osm_lm_idx_to_tags:
                continue
            lm_row = vigor_dataset._landmark_metadata.iloc[lm_idx]
            pruned = lm_row.get("pruned_props", frozenset())
            if pruned:
                osm_lm_idx_to_tags[lm_idx] = dict(pruned)

    unique_osm_lm_idxs = sorted(osm_lm_idx_to_tags.keys())
    osm_tags_list = [osm_lm_idx_to_tags[idx] for idx in unique_osm_lm_idxs]
    osm_idx_to_col = {idx: col for col, idx in enumerate(unique_osm_lm_idxs)}

    sat_col_positions: list[list[int]] = []
    for sat_idx in range(num_sats):
        lm_idxs = vigor_dataset._satellite_metadata.iloc[sat_idx].get(
            "landmark_idxs", []
        )
        if lm_idxs is None:
            sat_col_positions.append([])
        else:
            sat_col_positions.append(
                [osm_idx_to_col[i] for i in lm_idxs if i in osm_idx_to_col]
            )
    return osm_tags_list, sat_col_positions


def run_ollama_once(
    client: ollama_sdk.Client,
    model: str,
    image_paths: list[Path],
    max_retries: int = 2,
):
    """Single-shot Ollama call; retries on invalid-JSON output. Does NOT include
    retry time in the returned wall-clock (retries count as stage failure)."""
    images = [encode_image_to_base64(p) for p in image_paths]
    messages = [
        {"role": "system", "content": OSM_TAGS_SYSTEM_PROMPT},
        {"role": "user", "images": images, "content": osm_tags_user_prompt},
    ]

    for attempt in range(max_retries + 1):
        t0 = time.perf_counter_ns()
        response = client.chat(
            model=model, messages=messages, format=OSM_EXTRACTION_SCHEMA
        )
        ms = (time.perf_counter_ns() - t0) / 1e6
        content = response.message.content or ""
        try:
            tags = parse_ollama_landmarks(content)
            return tags, ms
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(
                f"  [ollama retry {attempt + 1}/{max_retries}] invalid JSON: {e}",
                file=sys.stderr,
            )
    return [], ms


def run_gemini_once(
    client,  # google.genai.Client
    model: str,
    image_paths: list[Path],
    thinking_level: str = "low",
):
    """Synchronous (non-batch) Gemini call with the same 4-yaw + JSON-schema
    setup the batch pipeline uses. Times the wall clock end-to-end."""
    parts = []
    for p in image_paths:
        with open(p, "rb") as f:
            data = f.read()
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg")
                             else "image/png",
                "data": data,
            }
        })
    parts.append({"text": osm_tags_user_prompt})

    config = {
        "system_instruction": GEMINI_SYSTEM_PROMPTS["osm_tags"],
        "response_mime_type": "application/json",
        "response_schema": get_osm_tags_schema(),
    }
    if thinking_level and thinking_level != "off":
        # gemini-3.x uses thinking_level (low/medium/high); 2.5 uses thinking_budget (int).
        if thinking_level.isdigit() or (thinking_level.startswith("-")
                                        and thinking_level[1:].isdigit()):
            config["thinking_config"] = {"thinking_budget": int(thinking_level)}
        else:
            config["thinking_config"] = {"thinking_level": thinking_level}

    t0 = time.perf_counter_ns()
    response = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": parts}],
        config=config,
    )
    ms = (time.perf_counter_ns() - t0) / 1e6
    try:
        tags = parse_ollama_landmarks(response.text or "")
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  [gemini parse error] {e}", file=sys.stderr)
        tags = []
    return tags, ms


def pct(samples: list[float], q: float) -> float:
    return float(np.percentile(samples, q)) if samples else float("nan")


def summarize(timings: dict[str, list[float]]) -> dict:
    summary = {}
    for stage in STAGE_ORDER:
        samples = timings[stage]
        if not samples:
            summary[stage] = {"n": 0}
            continue
        summary[stage] = {
            "n": len(samples),
            "median_ms": statistics.median(samples),
            "p10_ms": pct(samples, 10),
            "p90_ms": pct(samples, 90),
            "mean_ms": statistics.mean(samples),
        }
    summed_median = sum(
        summary[s]["median_ms"] for s in STAGE_ORDER if summary[s]["n"] > 0
    )
    summary["summed_median_ms"] = summed_median
    return summary


def print_table(summary: dict) -> None:
    print()
    print(f"{'stage':<18} {'n':>3}  {'median':>10}  {'p10':>10}  {'p90':>10}  {'mean':>10}")
    print("-" * 70)
    for stage in STAGE_ORDER:
        s = summary[stage]
        if s["n"] == 0:
            print(f"{stage:<18} {0:>3}  (no samples)")
            continue
        print(
            f"{stage:<18} {s['n']:>3}  {s['median_ms']:>8.2f}ms  "
            f"{s['p10_ms']:>8.2f}ms  {s['p90_ms']:>8.2f}ms  {s['mean_ms']:>8.2f}ms"
        )
    print("-" * 70)
    print(f"{'summed median':<18}      {summary['summed_median_ms']:>8.2f}ms")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure per-pano pipeline latency for the realtime claim"
    )
    # Dataset + models
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--landmark-version", type=str, required=True)
    parser.add_argument("--pano-model-path", type=Path, required=True)
    parser.add_argument("--sat-model-path", type=Path, required=True)
    parser.add_argument("--correspondence-model-path", type=Path, required=True)
    parser.add_argument("--osm-text-embeddings-path", type=Path, required=True)
    parser.add_argument("--city", type=str, default=DEFAULT_CITY,
                        help="City name for resolve_pano_images pinhole lookup "
                             f"(default: {DEFAULT_CITY})")
    parser.add_argument("--pano-ids", type=str, default=None,
                        help="Comma-separated pano_ids to time; if unset, uses a "
                             "hardcoded set of Chicago panos labeled 'useful' in "
                             "~/scratch/overhead_matching/landmark_tagger.labels.json")

    # Optional
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--llm-backend", choices=["ollama", "gemini"], default="ollama",
                        help="Which LLM to use for stage 2 landmark extraction")
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--ollama-model", type=str, default="qwen3.5:35b")
    parser.add_argument("--gemini-model", type=str, default="gemini-3-flash-preview",
                        help="Gemini model for realtime (non-batch) generate_content")
    parser.add_argument("--gemini-thinking-level", type=str, default="low",
                        help="Thinking effort for Gemini realtime call")
    parser.add_argument("--pinhole-base", type=Path,
                        default=Path("/data/overhead_matching/datasets/pinhole_images"))
    parser.add_argument("--sentence-encoder", type=str,
                        default="BAAI/bge-base-en-v1.5",
                        help="Local sentence-transformers model; output dim must "
                             "match correspondence model text_input_dim")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--panorama-neighbor-radius-deg", type=float, default=0.0005)
    parser.add_argument("--panorama-landmark-radius-px", type=int, default=640)
    parser.add_argument("--prob-threshold", type=float, default=0.3,
                        help="Match threshold used by stage 5 aggregation")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load text embeddings + sentence encoder (with dim check) ---
    print(f"Loading text embeddings from {args.osm_text_embeddings_path}")
    text_embeddings = load_text_embeddings(args.osm_text_embeddings_path)
    text_input_dim = next(iter(text_embeddings.values())).shape[0]
    print(f"  {len(text_embeddings):,} entries, dim={text_input_dim}")

    print(f"Loading sentence encoder: {args.sentence_encoder}")
    sbert = SentenceTransformer(args.sentence_encoder, device=str(device))
    sbert_dim = sbert.get_sentence_embedding_dimension()
    if sbert_dim != text_input_dim:
        raise ValueError(
            f"Sentence encoder {args.sentence_encoder} produces {sbert_dim}-d "
            f"vectors but correspondence model text_input_dim={text_input_dim}. "
            f"Pick a 768-d encoder (e.g. BAAI/bge-base-en-v1.5, "
            f"sentence-transformers/all-mpnet-base-v2)."
        )

    # --- Correspondence model ---
    print(f"Loading correspondence model from {args.correspondence_model_path}")
    encoder_config = TagBundleEncoderConfig(
        text_input_dim=text_input_dim, text_proj_dim=128
    )
    classifier_config = CorrespondenceClassifierConfig(encoder=encoder_config)
    corr_model = CorrespondenceClassifier(classifier_config).to(device)
    corr_model.load_state_dict(
        torch.load(args.correspondence_model_path, map_location=device,
                   weights_only=True)
    )
    corr_model.eval()

    # --- VIGOR dataset + pano/sat models ---
    # construct_path_eval_inputs_from_args takes a paths file but only uses it
    # to set the dataset factor and error out on old-format paths. We sidestep
    # it by loading the dataset + models directly with the same config shape.
    print("Loading VIGOR dataset + pano/sat models")
    from experimental.overhead_matching.swag.scripts.evaluate_histogram_on_paths import (
        load_model as _load_histogram_model,
    )
    pano_model = _load_histogram_model(args.pano_model_path, device=device)
    sat_model = _load_histogram_model(args.sat_model_path, device=device)
    dataset_config = vd.VigorDatasetConfig(
        panorama_tensor_cache_info=vd.TensorCacheInfo(
            dataset_keys=[args.dataset_path.name],
            model_type="panorama",
            landmark_version=args.landmark_version,
            panorama_landmark_radius_px=args.panorama_landmark_radius_px,
            landmark_correspondence_inflation_factor=1.0,
            extractor_info=pano_model.cache_info(),
        ),
        satellite_tensor_cache_info=vd.TensorCacheInfo(
            dataset_keys=[args.dataset_path.name],
            model_type="satellite",
            landmark_version=args.landmark_version,
            panorama_landmark_radius_px=args.panorama_landmark_radius_px,
            landmark_correspondence_inflation_factor=1.0,
            extractor_info=sat_model.cache_info(),
        ),
        panorama_neighbor_radius=args.panorama_neighbor_radius_deg,
        satellite_patch_size=sat_model.patch_dims,
        panorama_size=pano_model.patch_dims,
        factor=1.0,
        landmark_version=args.landmark_version,
        should_load_images=True,
        should_load_landmarks=True,
    )
    vigor_dataset = vd.VigorDataset(args.dataset_path, dataset_config)
    pano_model.eval()

    # --- Histogram belief / mapping ---
    print("Building histogram grid + cell-to-patch mapping")
    belief, mapping, hist_config = build_histogram_inputs(vigor_dataset, device)
    num_sats = len(vigor_dataset._satellite_metadata)

    # --- OSM landmarks + per-sat column lookup (amortized) ---
    print("Collecting OSM landmark tags")
    osm_tags_list, sat_col_positions = build_osm_tag_lookup(vigor_dataset)
    print(f"  {len(osm_tags_list):,} unique OSM landmarks across {num_sats:,} sats")

    # --- Sample panos + synthesized motion delta ---
    if args.pano_ids:
        candidate_ids = [p.strip() for p in args.pano_ids.split(",") if p.strip()]
    else:
        candidate_ids = list(DEFAULT_PANO_IDS)
    total_needed = args.warmup + args.num_samples
    if len(candidate_ids) < total_needed:
        raise ValueError(
            f"Need {total_needed} pano_ids (warmup={args.warmup} + "
            f"num_samples={args.num_samples}) but only have {len(candidate_ids)}. "
            f"Pass more via --pano-ids."
        )
    sample_pano_ids = candidate_ids[:total_needed]

    # Manual pano_id → idx lookup because VigorDataset.pano_ids_to_indices uses
    # pd.Index.get_indexer which fails on non-unique pano_id columns (which VIGOR
    # metadata can have after stitching).
    pano_id_to_idx: dict[str, int] = {}
    for idx, pid in enumerate(vigor_dataset._panorama_metadata["pano_id"].values):
        pano_id_to_idx.setdefault(pid, idx)
    missing = [p for p in sample_pano_ids if p not in pano_id_to_idx]
    if missing:
        raise ValueError(f"pano_ids not found in dataset: {missing[:5]}...")
    pano_indices = [pano_id_to_idx[p] for p in sample_pano_ids]

    # Synthesize a representative motion delta for stage 6. apply_motion's cost
    # is dominated by the grid shift + Gaussian blur; magnitude matters only
    # through the blur sigma (noise_percent * |delta|).
    motion_delta = torch.tensor(
        list(DEFAULT_MOTION_DELTA_DEG), dtype=torch.float32, device=device
    )

    # Pre-load batched pano items so stage 1 times only the forward pass, not
    # image decode / collate. Each batch has batch_size=1 so we get realistic
    # per-pano encoder input.
    pano_view = vigor_dataset.get_pano_view()
    pano_subset = torch.utils.data.Subset(pano_view, pano_indices)
    pano_loader = vd.get_dataloader(
        pano_subset, batch_size=1, num_workers=0, shuffle=False
    )
    pano_batches = list(pano_loader)

    # --- LLM client ---
    ollama_client = None
    gemini_client = None
    if args.llm_backend == "ollama":
        print(f"Connecting to Ollama at {args.ollama_base_url} "
              f"(model={args.ollama_model})")
        ollama_client = ollama_sdk.Client(host=args.ollama_base_url)
    else:
        print(f"Initializing Gemini client (model={args.gemini_model}, "
              f"thinking={args.gemini_thinking_level})")
        from google import genai
        gemini_client = genai.Client(vertexai=True)

    # --- Timing loop ---
    timings: dict[str, list[float]] = {s: [] for s in STAGE_ORDER}
    print(f"\nTiming {args.warmup} warmup + {args.num_samples} samples\n")

    for i, (pano_id, pano_idx) in enumerate(zip(sample_pano_ids, pano_indices)):
        is_warmup = i < args.warmup
        label = "warmup" if is_warmup else f"sample {i - args.warmup + 1}"
        print(f"[{label}] pano_id={pano_id}")

        # Resolve pinhole images (skip-on-miss)
        try:
            image_paths = resolve_pano_images(
                args.pinhole_base, args.city, pano_id
            )
        except FileNotFoundError as e:
            print(f"  [skip] no pinhole images: {e}", file=sys.stderr)
            continue

        # --- Stage 1: pano encode ---
        batch = pano_batches[i]
        if batch.panorama is None or torch.isnan(batch.panorama).any():
            print("  [skip] pano tensor is nan (dataset returned placeholder)",
                  file=sys.stderr)
            continue

        def _encode_pano():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model_input = pano_model.model_input_from_batch(batch).to(device)
                out = pano_model(model_input)
                return out[0] if isinstance(out, tuple) else out
        _, pano_ms = _time_block(_encode_pano, device)

        # --- Stage 2: LLM landmark extraction ---
        if args.llm_backend == "ollama":
            pano_tags, ollama_ms = run_ollama_once(
                ollama_client, args.ollama_model, image_paths
            )
        else:
            pano_tags, ollama_ms = run_gemini_once(
                gemini_client, args.gemini_model, image_paths,
                thinking_level=args.gemini_thinking_level,
            )
        if not pano_tags:
            print(f"  [warn] {args.llm_backend} returned 0 tags; "
                  "stages 3-5 will be near-trivial")

        # --- Stage 3: sentence embed of novel values ---
        novel_values: list[str] = []
        seen: set[str] = set()
        for tags in pano_tags:
            for v in tags.values():
                if v not in text_embeddings and v not in seen:
                    novel_values.append(v)
                    seen.add(v)

        def _embed_novel():
            if not novel_values:
                return np.empty((0, text_input_dim), dtype=np.float32)
            return sbert.encode(
                novel_values,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        novel_embs, sent_ms = _time_block(_embed_novel, device)
        for v, e in zip(novel_values, novel_embs):
            text_embeddings[v] = torch.from_numpy(np.asarray(e)).float()

        # --- Stage 4: correspondence classifier ---
        def _correspondence():
            with torch.no_grad():
                return cm.compute_pairs_cost_matrix(
                    pano_tags_list=pano_tags,
                    osm_tags_list=osm_tags_list,
                    model=corr_model,
                    text_embeddings=text_embeddings,
                    text_input_dim=text_input_dim,
                    device=device,
                    max_pairs_per_batch=50_000,
                    allow_missing_text_embeddings=True,
                )
        cost_matrix, corr_ms = _time_block(_correspondence, device)

        # --- Stage 5: aggregate per-sat ---
        def _aggregate():
            sim = np.zeros(num_sats, dtype=np.float32)
            if cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
                return sim
            for sat_idx in range(num_sats):
                cols = sat_col_positions[sat_idx]
                if not cols:
                    continue
                sub = cost_matrix[:, cols]
                res = cm.match_and_aggregate(
                    sub,
                    cm.MatchingMethod.HUNGARIAN,
                    cm.AggregationMode.SUM,
                    prob_threshold=args.prob_threshold,
                    use_dustbin=True,
                )
                sim[sat_idx] = res.similarity_score
            return sim
        _, agg_ms = _time_block(_aggregate, device)

        # --- Stage 6: filter step (on a cloned belief so state doesn't drift) ---
        # Representative obs_log_ll: zeros (shape is what matters for timing)
        obs_log_ll = torch.zeros(num_sats, device=device)

        def _filter_step():
            b = belief.clone()
            b.apply_observation(obs_log_ll, mapping)
            b.apply_motion(motion_delta, hist_config.motion_noise_frac)
            return b
        _, filter_ms = _time_block(_filter_step, device)

        print(
            f"  pano={pano_ms:.1f}ms  ollama={ollama_ms:.0f}ms  "
            f"sent={sent_ms:.1f}ms  corr={corr_ms:.1f}ms  "
            f"agg={agg_ms:.1f}ms  filter={filter_ms:.1f}ms  "
            f"(K={len(pano_tags)}, novel={len(novel_values)})"
        )

        if is_warmup:
            continue

        timings["pano_encode"].append(pano_ms)
        timings["ollama"].append(ollama_ms)
        timings["sentence_embed"].append(sent_ms)
        timings["correspondence"].append(corr_ms)
        timings["aggregate"].append(agg_ms)
        timings["filter_step"].append(filter_ms)

    summary = summarize(timings)
    print_table(summary)

    payload = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "text_input_dim": text_input_dim,
        "num_sats": num_sats,
        "num_osm_landmarks": len(osm_tags_list),
        "raw_timings_ms": timings,
        "summary": summary,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON to {args.output_json}")
    else:
        print()
        json.dump(payload, sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
