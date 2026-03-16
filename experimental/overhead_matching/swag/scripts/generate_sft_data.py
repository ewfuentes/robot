"""Generate SFT training data for finetuning OSM landmark extraction.

Supports two backends:
  - **ollama** : Uses a local Ollama model with tool calling.
  - **gemini** (default): Uses Google Gemini API with tool calling.

Both backends use the same ``query_landmarks`` tool against a running
``osm_tag_server``. Conversations are saved in a unified format regardless
of backend, with image file paths instead of base64 data.

Optionally injects ground-truth hints (without names/brands) as a separate
user message to bias the model toward known landmarks. At training time, hint
messages are stripped via the ``is_hint`` flag.

Requires osm_tag_server running for both backends.

Usage (gemini, default)::

    bazel run //experimental/overhead_matching/swag/scripts:generate_sft_data -- \
      --city Chicago \
      --labels /home/erick/scratch/overhead_matching/landmark_tagger.labels.json \
      --feather /data/overhead_matching/datasets/VIGOR/Chicago/landmarks/v4_202001.feather \
      --output-dir /home/erick/scratch/overhead_matching/sft_conversations \
      --limit 5

Usage (ollama)::

    bazel run //experimental/overhead_matching/swag/scripts:generate_sft_data -- \
      --city Chicago \
      --labels /home/erick/scratch/overhead_matching/landmark_tagger.labels.json \
      --feather /data/overhead_matching/datasets/VIGOR/Chicago/landmarks/v4_202001.feather \
      --output-dir /home/erick/scratch/overhead_matching/sft_conversations \
      --backend ollama --ollama-base-url http://localhost:11434 \
      --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

from experimental.overhead_matching.swag.model.semantic_landmark_extractor import (
    OSMTagExtraction,
    encode_image_to_base64,
    osm_tags_user_prompt,
)
from experimental.overhead_matching.swag.scripts.evaluate_extraction import (
    extract_osm_tags,
)
from experimental.overhead_matching.swag.scripts.ollama_osm_extraction import (
    DEFAULT_PINHOLE_BASE,
    DEFAULT_TAG_SERVER_URL,
    OSM_TAGS_SYSTEM_PROMPT,
    OSM_EXTRACTION_SCHEMA,
    TOOL_INSTRUCTIONS,
    TOOLS,
    build_user_message,
    dispatch_tool,
    load_examples,
    resolve_pano_images,
)

DEFAULT_LABELS = "/home/erick/scratch/overhead_matching/landmark_tagger.labels.json"


def build_hints(osm_id_strs: list[str], osm_id_to_row: dict[str, int], df: pd.DataFrame) -> str:
    """Build a <ground_truth_hints> block from labeled OSM IDs."""
    lines = []
    for osm_id_str in osm_id_strs:
        row_idx = osm_id_to_row.get(osm_id_str)
        if row_idx is None:
            continue
        tags = extract_osm_tags(df.iloc[row_idx])
        # Filter out identifying tags — the model should discover these from images
        tags = [(k, v) for k, v in tags if k not in ("name", "brand", "operator")]
        if not tags:
            continue
        tag_str = ", ".join(f"{k}={v}" for k, v in tags)
        lines.append(f"- {tag_str}")

    if not lines:
        return ""

    return (f"""
        <ground_truth_hints>
        We are generating training data for supervised finetuning of an LLM. The following OSM landmarks are believed
        to be visible in the panorama. However, you should confirm that each tag that is emitted is justifiable
        from the images. For example, if you see that one of the following landmarks is a park with a name, if there
        is no sign identifying in the park, the name shouldn't be emitted. However, we can still include a `leisure=park` tag
        because we can see that it is a park. Do not reference the existence of this ground_truth_hints block.

        {"\n".join(lines)}
        </ground_truth_hints>
        """)


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------


def _serialize_ollama_message(msg) -> dict:
    """Convert an ollama message object to a serializable dict."""
    d: dict = {"role": msg.role}
    if msg.content:
        d["content"] = msg.content
    # Intentionally skip images — they are replaced with file paths post-hoc
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [
            {
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            }
            for tc in msg.tool_calls
        ]
    return d


def run_ollama_extraction(
    client,
    model: str,
    image_paths: list[Path],
    server_url: str,
    hints: str,
    max_tool_rounds: int = 15,
    examples_dir: Path | None = None,
) -> tuple[list[dict], dict]:
    """Run extraction via ollama with tool calling.

    Returns (saved_messages, extraction_result).
    """
    examples_text = load_examples(examples_dir) if examples_dir else ""
    system_prompt = OSM_TAGS_SYSTEM_PROMPT + TOOL_INSTRUCTIONS + examples_text

    live_messages = [
        {"role": "system", "content": system_prompt},
        build_user_message(image_paths),
        {"role": "user", "content": hints},
    ]

    saved_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "image_paths": [str(p) for p in image_paths],
            "content": osm_tags_user_prompt,
        },
        {"role": "user", "content": hints, "is_hint": True},
    ]

    chat_kwargs = dict(model=model, think=True)

    # Tool-calling loop
    for round_idx in range(max_tool_rounds):
        response = client.chat(messages=live_messages, tools=TOOLS, **chat_kwargs)

        if getattr(response.message, "thinking", None):
            print(f"  [thinking] {response.message.thinking[:200]}...", file=sys.stderr)
        if response.message.content:
            print(f"  [response] {response.message.content[:200]}", file=sys.stderr)

        serialized = _serialize_ollama_message(response.message)

        if not response.message.tool_calls:
            saved_messages.append(serialized)
            live_messages.append(response.message)
            break

        # Process tool calls
        live_messages.append(response.message)
        saved_messages.append(serialized)
        for tool_call in response.message.tool_calls:
            name = tool_call.function.name
            args = tool_call.function.arguments
            print(f"  [tool round {round_idx + 1}] {name}({json.dumps(args)})", file=sys.stderr)
            result = dispatch_tool(name, args, server_url)
            print(f"    -> {result[:200]}", file=sys.stderr)
            tool_msg = {"role": "tool", "content": result}
            live_messages.append(tool_msg)
            saved_messages.append(tool_msg)
    else:
        print(f"  [max tool rounds reached]", file=sys.stderr)

    # Final structured output call
    print(f"  [generating structured output]", file=sys.stderr)
    final_user_msg = {"role": "user", "content": "Now output your final answer as JSON."}
    live_messages.append(final_user_msg)
    saved_messages.append(final_user_msg)

    max_retries = 3
    for attempt in range(max_retries):
        response = client.chat(messages=live_messages, format=OSM_EXTRACTION_SCHEMA, **chat_kwargs)
        if getattr(response.message, "thinking", None):
            print(f"  [thinking] {response.message.thinking[:200]}...", file=sys.stderr)

        content = response.message.content
        if not content or not content.strip():
            print(f"  [empty response, retry {attempt + 1}/{max_retries}]", file=sys.stderr)
            continue

        try:
            raw = json.loads(content)
            extraction = OSMTagExtraction.model_validate(raw)
            saved_messages.append(_serialize_ollama_message(response.message))
            return saved_messages, extraction.model_dump()
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [bad response, retry {attempt + 1}/{max_retries}]: {e}", file=sys.stderr)
            serialized = _serialize_ollama_message(response.message)
            retry_msg = {
                "role": "user",
                "content": f"Your response was not valid JSON: {e}. Please try again.",
            }
            live_messages.append(response.message)
            live_messages.append(retry_msg)
            saved_messages.append(serialized)
            saved_messages.append(retry_msg)

    raise RuntimeError("Failed to get valid structured output after retries")


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

GEMINI_TOOL_DECLARATION = {
    "name": "query_landmarks",
    "description": (
        "Execute a read-only SQL query against the OSM landmark database. "
        "Available tables: tags(key, value, count) and "
        "landmark_tags(landmark_id, key, value). "
        "Returns {columns, rows} or {error}."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "SQL SELECT query to execute",
            },
        },
        "required": ["sql"],
    },
}


def run_gemini_extraction(
    client,
    model: str,
    image_paths: list[Path],
    server_url: str,
    hints: str,
    max_tool_rounds: int = 15,
) -> tuple[list[dict], dict]:
    """Run extraction via Gemini API with tool calling.

    Returns (saved_messages, extraction_result) in the same format as the
    ollama backend.
    """
    from google.genai import types

    system_prompt = OSM_TAGS_SYSTEM_PROMPT + TOOL_INSTRUCTIONS

    # Build image parts
    image_parts = []
    for img_path in image_paths:
        ext = img_path.suffix.lower()
        mime_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        image_data = img_path.read_bytes()
        image_parts.append(types.Part(inline_data=types.Blob(
            mime_type=mime_type,
            data=image_data,
        )))

    # Build initial contents
    contents = [
        types.Content(role="user", parts=[
            *image_parts,
            types.Part(text=osm_tags_user_prompt),
        ]),
        types.Content(role="user", parts=[types.Part(text=hints)]),
    ]

    # Saved messages (unified format)
    saved_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "image_paths": [str(p) for p in image_paths],
            "content": osm_tags_user_prompt,
        },
        {"role": "user", "content": hints, "is_hint": True},
    ]

    tool = types.Tool(function_declarations=[
        types.FunctionDeclaration(**GEMINI_TOOL_DECLARATION),
    ])
    thinking_config = types.ThinkingConfig(thinking_budget=1024)

    tool_config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[tool],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=thinking_config,
    )

    # Tool-calling loop
    for round_idx in range(max_tool_rounds):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=tool_config,
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts

        # Check for function calls
        function_calls = [p for p in parts if p.function_call]
        text_parts = [p.text for p in parts if p.text]

        if text_parts:
            text = "\n".join(text_parts)
            print(f"  [response] {text[:200]}", file=sys.stderr)

        if not function_calls:
            # No tool calls — model is done with tool phase
            if text_parts:
                saved_messages.append({"role": "assistant", "content": "\n".join(text_parts)})
            contents.append(candidate.content)
            break

        # Serialize assistant message with tool calls
        assistant_msg: dict = {"role": "assistant"}
        if text_parts:
            assistant_msg["content"] = "\n".join(text_parts)
        assistant_msg["tool_calls"] = [
            {
                "function": {
                    "name": fc.function_call.name,
                    "arguments": dict(fc.function_call.args),
                }
            }
            for fc in function_calls
        ]
        saved_messages.append(assistant_msg)
        contents.append(candidate.content)

        # Dispatch each tool call and build function response parts
        response_parts = []
        for fc in function_calls:
            name = fc.function_call.name
            args = dict(fc.function_call.args)
            print(f"  [tool round {round_idx + 1}] {name}({json.dumps(args)})", file=sys.stderr)
            result = dispatch_tool(name, args, server_url)
            print(f"    -> {result[:200]}", file=sys.stderr)
            saved_messages.append({"role": "tool", "content": result})
            response_parts.append(types.Part.from_function_response(
                name=name,
                response={"result": json.loads(result)},
            ))

        contents.append(types.Content(role="user", parts=response_parts))
    else:
        print(f"  [max tool rounds reached]", file=sys.stderr)

    # Final structured output call (without tools, with JSON schema)
    print(f"  [generating structured output]", file=sys.stderr)
    final_user_msg = {"role": "user", "content": "Now output your final answer as JSON."}
    saved_messages.append(final_user_msg)
    contents.append(types.Content(role="user", parts=[types.Part(text=final_user_msg["content"])]))

    final_config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=OSM_EXTRACTION_SCHEMA,
    )

    max_retries = 3
    for attempt in range(max_retries):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=final_config,
        )

        content = response.text
        if not content or not content.strip():
            print(f"  [empty response, retry {attempt + 1}/{max_retries}]", file=sys.stderr)
            continue

        try:
            raw = json.loads(content)
            extraction = OSMTagExtraction.model_validate(raw)
            saved_messages.append({"role": "assistant", "content": content})
            return saved_messages, extraction.model_dump()
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [bad response, retry {attempt + 1}/{max_retries}]: {e}", file=sys.stderr)
            saved_messages.append({"role": "assistant", "content": content})
            retry_msg = {
                "role": "user",
                "content": f"Your response was not valid JSON: {e}. Please try again.",
            }
            saved_messages.append(retry_msg)
            contents.append(response.candidates[0].content)
            contents.append(types.Content(role="user", parts=[types.Part(text=retry_msg["content"])]))

    raise RuntimeError("Failed to get valid structured output after retries")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT training data with tool-calling extraction"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(DEFAULT_LABELS),
        help="Path to landmark_tagger labels JSON",
    )
    parser.add_argument("--city", type=str, required=True, help="VIGOR city name")
    parser.add_argument(
        "--feather",
        type=Path,
        required=True,
        help="Path to feather file with OSM landmarks",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for output JSONs"
    )
    parser.add_argument(
        "--pinhole-base",
        type=Path,
        default=Path(DEFAULT_PINHOLE_BASE),
        help=f"Base directory for pinhole images (default: {DEFAULT_PINHOLE_BASE})",
    )
    parser.add_argument(
        "--tag-server-url",
        type=str,
        default=DEFAULT_TAG_SERVER_URL,
        help=f"URL of the OSM tag server (default: {DEFAULT_TAG_SERVER_URL})",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["ollama", "gemini"],
        default="gemini",
        help="Model backend (default: gemini)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: gemini-3-flash for gemini, qwen3.5:35b for ollama)",
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=None,
        help="URL to an existing Ollama server (ollama backend only)",
    )
    parser.add_argument(
        "--max-tool-rounds", type=int, default=15, help="Max tool-calling iterations"
    )
    parser.add_argument(
        "--no-hints",
        action="store_true",
        help="Disable ground-truth hints (pure extraction)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of panoramas to process (for testing)",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=None,
        help="Directory with example conversation JSONs for in-context learning",
    )
    args = parser.parse_args()

    # Default model per backend
    if args.model is None:
        args.model = "gemini-3-flash-preview" if args.backend == "gemini" else "qwen3.5:35b"

    # Load labels
    with open(args.labels) as f:
        labels_data = json.load(f)
    relevant_landmarks = labels_data.get("relevant_landmarks", {})
    # Only process panos that have relevant_landmarks labels
    pano_ids = sorted(pid for pid, osm_ids in relevant_landmarks.items() if osm_ids)
    print(f"Loaded {len(pano_ids)} panoramas with relevant landmarks", file=sys.stderr)

    # Load feather and build OSM ID index
    df = pd.read_feather(args.feather)
    osm_id_to_row = {str(df.iloc[i]["id"]): i for i in range(len(df))}
    print(f"Loaded feather: {len(df)} landmarks", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to those not yet processed
    remaining = [
        pid for pid in pano_ids if not (args.output_dir / f"{pid}.json").exists()
    ]
    if args.limit:
        remaining = remaining[: args.limit]
    print(
        f"{len(remaining)} remaining (of {len(pano_ids)} total)", file=sys.stderr
    )

    def run_extraction_for_pano(run_fn, pano_id, model):
        """Process a single pano with the given extraction function."""
        image_paths = resolve_pano_images(args.pinhole_base, args.city, pano_id)

        osm_id_strs = relevant_landmarks[pano_id]
        if args.no_hints:
            hints = ""
        else:
            hints = build_hints(osm_id_strs, osm_id_to_row, df)

        messages, extraction_result = run_fn(
            model=model,
            image_paths=image_paths,
            server_url=args.tag_server_url,
            hints=hints,
            max_tool_rounds=args.max_tool_rounds,
        )

        return {
            "pano_id": pano_id,
            "city": args.city,
            "hints": hints,
            "ground_truth_osm_ids": osm_id_strs,
            "messages": messages,
            "extraction_result": extraction_result,
        }

    def process_all(run_fn, model):
        success = 0
        skipped = 0
        failed = 0
        for i, pano_id in enumerate(remaining):
            output_path = args.output_dir / f"{pano_id}.json"
            print(
                f"\n[{i + 1}/{len(remaining)}] {pano_id}",
                file=sys.stderr,
            )

            try:
                output_data = run_extraction_for_pano(run_fn, pano_id, model)
            except FileNotFoundError as e:
                print(f"  SKIP (no images): {e}", file=sys.stderr)
                skipped += 1
                continue
            except Exception:
                print(f"  FAIL:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                failed += 1
                continue

            output_path.write_text(json.dumps(output_data, indent=2))
            num_lm = len(output_data["extraction_result"].get("landmarks", []))
            print(f"  OK: {num_lm} landmarks -> {output_path}", file=sys.stderr)
            success += 1

        print(
            f"\nDone: {success} success, {skipped} skipped, {failed} failed",
            file=sys.stderr,
        )

    if args.backend == "gemini":
        from google import genai

        gemini_client = genai.Client()
        print(f"Using Gemini model: {args.model}", file=sys.stderr)

        def gemini_run_fn(*, model, image_paths, server_url, hints, max_tool_rounds):
            return run_gemini_extraction(
                gemini_client, model, image_paths, server_url, hints, max_tool_rounds,
            )

        process_all(gemini_run_fn, args.model)
    else:
        import ollama as ollama_sdk
        from common.ollama.pyollama import Ollama

        def ollama_run_fn(client):
            def fn(*, model, image_paths, server_url, hints, max_tool_rounds):
                return run_ollama_extraction(
                    client, model, image_paths, server_url, hints, max_tool_rounds,
                    examples_dir=args.examples_dir,
                )
            return fn

        if args.ollama_base_url:
            print(
                f"Using existing Ollama server at {args.ollama_base_url}",
                file=sys.stderr,
            )
            client = ollama_sdk.Client(host=args.ollama_base_url)
            process_all(ollama_run_fn(client), args.model)
        else:
            print(
                f"Starting managed Ollama server for model {args.model}...",
                file=sys.stderr,
            )
            with Ollama(args.model) as server:
                client = ollama_sdk.Client(host=server.base_url())
                process_all(ollama_run_fn(client), args.model)


if __name__ == "__main__":
    main()
