"""Extract OSM tags from panorama images using Ollama with tool calling.

Local alternative to OpenAI/Gemini batch APIs. Uses an Ollama model (default:
qwen3.5:35b) with two tools that let the LLM fuzzy-search the local OSM tag
database while extracting landmarks.

Usage::

    # Explicit image paths:
    bazel run //experimental/overhead_matching/swag/scripts:ollama_osm_extraction -- \
      --images /path/to/0.jpg /path/to/90.jpg /path/to/180.jpg /path/to/270.jpg \
      --feather /data/.../landmarks.feather \
      --ollama-base-url http://localhost:11434 \
      --output /tmp/result.json

    # By city + panorama ID (auto-resolves paths):
    bazel run //experimental/overhead_matching/swag/scripts:ollama_osm_extraction -- \
      --city Chicago --pano-id 005T5CAKugPnibKVcndUSA \
      --ollama-base-url http://localhost:11434
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ollama as ollama_sdk
import pandas as pd

from common.ollama.pyollama import Ollama
from experimental.overhead_matching.swag.model.semantic_landmark_extractor import (
    OSMTagExtraction,
    encode_image_to_base64,
    get_osm_tags_schema,
    osm_tags_user_prompt,
)
from experimental.overhead_matching.swag.scripts.search_osm_tags import TagSearchIndex

OSM_TAGS_SYSTEM_PROMPT = """<role>
You are an expert at identifying landmarks in street-level imagery and mapping them to OpenStreetMap (OSM) tags.
</role>

<instructions>
Given four images which show the same location from yaws 0°, 90°, 180°, and 270° respectively, identify distinctive, permanent landmarks and classify them using OSM's key=value tagging system.

For each landmark:
- Assign a primary OSM tag (e.g., amenity=cafe, shop=pharmacy, building=apartments)
- Add relevant additional tags (name, brand, cuisine, building:levels, etc.)
- Specify which yaw angle(s)/images the landmark appears in and provide bounding boxes for each
- Rate your confidence (high/medium/low)
- Provide a brief description for debugging

Based on the images, classify the location type (e.g., urban_commercial, suburban, rural).
Finally, review your work and confirm you have not included any information you cannot confidently make out from the images.
</instructions>

<landmark_selection>
Focus on landmarks that are VISUALLY DISTINCTIVE and useful for identifying a specific location. Prioritize:
- Named businesses, restaurants, shops with visible signage
- Architecturally unique buildings (churches, historic buildings, distinctive facades)
- Branded locations (gas stations, chain stores, banks)
- Named parks, monuments, public art, memorials
- Distinctive infrastructure (water towers, clock towers, unique bridges)

DO NOT include generic, ubiquitous features such as:
- Traffic signals, street lamps, fire hydrants
- Crosswalks, stop signs, generic road markings
- Plain sidewalks, curbs, gutters
- Trees, bushes, or grass unless they are a notable landmark (e.g., a named park)
- Generic buildings described only by their appearance (e.g., "a multi-story brick building",
  "a low-rise commercial building"). Only include a building if you can identify at least one of:
  - A readable building number or address
  - A visible name or sign on the building
  - A well-known or historically significant building (e.g., Willis Tower, Chicago Stock Exchange)

The goal is to identify landmarks that distinguish THIS location from others, not to catalog every object in the scene.
</landmark_selection>

<osm_tag_guidelines>
## Primary OSM Tag Categories

- `amenity`: facilities providing services (restaurants, cafes, banks, hospitals, fuel stations, parking)
- `shop`: retail stores (grocery, clothing, hardware, pharmacy, convenience)
- `building`: structures with a roof (residential, commercial, retail, church, industrial). Use `building=yes` if unclear.
- `tourism`: visitor attractions (hotels, museums, viewpoints)
- `leisure`: recreation (parks, playgrounds, sports facilities)
- `office`: professional services (lawyer, accountant, insurance)
- `craft`: custom workshops (carpenter, tailor, jeweller)
- `highway`: road infrastructure (traffic_signals, crossing, bus_stop, street_lamp)
- `man_made`: non-building structures (towers, piers, bridges, chimneys, water towers)
- `historic`: historically significant features (monuments, memorials, ruins)
- `natural`: natural features (trees, water bodies)

## Key Distinctions

- **amenity vs shop**: Use amenity for services (eating, banking, fuel); shop for goods to take away
- **man_made vs building**: Use building if it has walls and roof for human use; man_made for towers, piers, etc.
- **leisure vs tourism**: Use leisure for local recreation; tourism for visitor attractions

## Branded Locations
For chains, include both category and brand:
- Starbucks -> amenity=cafe, brand=Starbucks
- Shell -> amenity=fuel, brand=Shell
- CVS -> shop=pharmacy, brand=CVS
</osm_tag_guidelines>

<constraints>
- Focus on OSM-mappable features within ~100 meters
- Extract visible text for name/brand tags only if clearly readable
- Be conservative: only output tags you can confidently identify
- Exclude transient objects (cars, pedestrians, temporary items)
- Do not mention location in image or relative to other landmarks
</constraints>

<output_format>
Provide your response as a JSON object conforming to the assigned schema.
Bounding box coordinates are normalized 0-1000, where (0,0) is top-left and (1000,1000) is bottom-right.
</output_format>
"""

DEFAULT_DATASET_BASE = "/data/overhead_matching/datasets/VIGOR"
DEFAULT_PINHOLE_BASE = "/data/overhead_matching/datasets/pinhole_images"
DEFAULT_FEATHER_NAME = "v4_202001.feather"

TOOL_INSTRUCTIONS = """

<tools>
You have access to two tools that let you search the local OSM tag database for
the city these images were taken in. Use them to find the correct tag key=value
pairs for the landmarks you identify. You may call the tools multiple times.

1. **search_tags(query, limit)** - fuzzy-search for OSM tag values matching a
   query string. Returns [{key, value, count}].
2. **get_tag_context(key, value, limit)** - given a specific tag, show which
   other tags commonly co-occur with it. Returns {total, co_occurring_tags}.

search_tags uses fzf for fuzzy matching. Every word in your query must appear
(in some fuzzy form) in the result, so keep queries short — ideally a single
word or substring. If you can only partially read a sign (e.g. "harma" from
"Pharmacy"), query for just that substring and the fuzzy matcher will find it.
Multi-word queries like "italian restaurant" will only match values containing
both words.

Workflow: first identify landmarks visually, then use search_tags to find
appropriate OSM tags, optionally use get_tag_context to discover additional tags,
and finally output the JSON result.
</tools>"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_tags",
            "description": (
                "Fuzzy-search for OSM key=value tags matching a query string. "
                "Returns a list of matching tags sorted by occurrence count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. 'cafe', 'pharmacy', 'residential')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tag_context",
            "description": (
                "Given a specific OSM tag (key=value), show which other tags "
                "commonly co-occur with it in the local dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "OSM tag key (e.g. 'amenity')",
                    },
                    "value": {
                        "type": "string",
                        "description": "OSM tag value (e.g. 'cafe')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of co-occurring tags to return (default 10)",
                    },
                },
                "required": ["key", "value"],
            },
        },
    },
]


def dispatch_tool(name: str, args: dict, index: TagSearchIndex) -> str:
    if name == "search_tags":
        results = index.search_tags(args["query"], limit=args.get("limit", 10))
        return json.dumps([{"key": k, "value": v, "count": c} for k, v, c in results])
    elif name == "get_tag_context":
        results, total = index.get_tag_context(
            args["key"], args["value"], limit=args.get("limit", 10)
        )
        return json.dumps(
            {"total": total, "co_occurring": [{"key": k, "value": v, "count": c} for k, v, c in results]}
        )
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


def parse_json_response(text: str) -> dict:
    """Parse JSON from model response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (with optional language tag) and closing fence
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


def build_user_message(image_paths: list[Path]) -> dict:
    """Build a user message with 4 panorama images and the extraction prompt."""
    images = [encode_image_to_base64(p) for p in image_paths]
    return {
        "role": "user",
        "images": images,
        "content": osm_tags_user_prompt,
    }


def run_extraction(
    client: ollama_sdk.Client,
    model: str,
    image_paths: list[Path],
    index: TagSearchIndex,
    max_tool_rounds: int = 5,
) -> OSMTagExtraction:
    system_prompt = OSM_TAGS_SYSTEM_PROMPT + TOOL_INSTRUCTIONS

    messages = [
        {"role": "system", "content": system_prompt},
        build_user_message(image_paths),
    ]

    chat_kwargs = dict(model=model, options={"think": True})
    schema = get_osm_tags_schema()

    # Tool-calling loop
    for round_idx in range(max_tool_rounds):
        response = client.chat(messages=messages, tools=TOOLS, **chat_kwargs)

        if getattr(response.message, "thinking", None):
            print(f"  [thinking] {response.message.thinking}", file=sys.stderr)
        if response.message.content:
            print(f"  [response] {response.message.content}", file=sys.stderr)

        if not response.message.tool_calls:
            break

        # Process tool calls
        messages.append(response.message)
        for tool_call in response.message.tool_calls:
            name = tool_call.function.name
            args = tool_call.function.arguments
            print(f"  [tool round {round_idx + 1}] {name}({json.dumps(args)})", file=sys.stderr)
            result = dispatch_tool(name, args, index)
            print(f"    -> {result[:200]}", file=sys.stderr)
            messages.append({"role": "tool", "content": result})

    # Final structured output call (format= can't be combined with tools=)
    if response.message.tool_calls:
        # Exhausted tool rounds — append last response and continue
        messages.append(response.message)
        print(f"  [max tool rounds reached]", file=sys.stderr)

    print(f"  [generating structured output]", file=sys.stderr)
    messages.append({"role": "user", "content": "Now output your final answer as JSON."})
    response = client.chat(messages=messages, format=schema, **chat_kwargs)
    if getattr(response.message, "thinking", None):
        print(f"  [thinking] {response.message.thinking}", file=sys.stderr)
    if response.message.content:
        print(f"  [response] {response.message.content}", file=sys.stderr)

    raw = json.loads(response.message.content)
    return OSMTagExtraction.model_validate(raw)


def resolve_pano_images(pinhole_base: Path, city: str, pano_id: str) -> list[Path]:
    """Find the 4 yaw images for a panorama ID under pinhole_base/city/.

    Directories are named ``{pano_id},{lat},{lon},``.  We match by the pano_id
    prefix so callers only need the ID, not the full stem.
    """
    city_dir = pinhole_base / city
    if not city_dir.is_dir():
        raise FileNotFoundError(f"City directory not found: {city_dir}")

    # Match dirs starting with "{pano_id}," to avoid partial ID collisions
    matches = sorted(d for d in city_dir.iterdir() if d.is_dir() and d.name.startswith(f"{pano_id},"))
    if not matches:
        raise FileNotFoundError(
            f"No pinhole directory found for pano_id={pano_id!r} under {city_dir}"
        )
    if len(matches) > 1:
        print(
            f"  Warning: multiple matches for {pano_id!r}, using {matches[0].name}",
            file=sys.stderr,
        )
    pano_dir = matches[0]

    yaw_angles = [0, 90, 180, 270]
    images = []
    for yaw in yaw_angles:
        for ext in ("jpg", "png"):
            p = pano_dir / f"yaw_{yaw:03d}.{ext}"
            if p.exists():
                images.append(p)
                break
        else:
            raise FileNotFoundError(f"Missing yaw image: yaw_{yaw:03d}.* in {pano_dir}")
    return images


def resolve_feather(dataset_base: Path, city: str, feather_name: str = DEFAULT_FEATHER_NAME) -> Path:
    """Find the feather file for a city."""
    p = dataset_base / city / "landmarks" / feather_name
    if not p.exists():
        raise FileNotFoundError(f"Feather file not found: {p}")
    return p


def main():
    parser = argparse.ArgumentParser(
        description="Extract OSM tags from panorama images using Ollama with tool calling"
    )

    # Image source: either explicit paths or city+pano-id
    img_group = parser.add_argument_group("image source (pick one)")
    img_group.add_argument(
        "--images",
        type=Path,
        nargs=4,
        default=None,
        metavar=("IMG_0", "IMG_90", "IMG_180", "IMG_270"),
        help="4 panorama image paths (0, 90, 180, 270 degrees)",
    )
    img_group.add_argument(
        "--city",
        type=str,
        default=None,
        help="VIGOR city name (e.g. Chicago, NewYork). Used with --pano-id.",
    )
    img_group.add_argument(
        "--pano-id",
        type=str,
        default=None,
        help="Panorama ID (e.g. 005T5CAKugPnibKVcndUSA). Used with --city.",
    )

    # Directories for city+pano-id resolution
    dir_group = parser.add_argument_group("directory configuration (for --city/--pano-id)")
    dir_group.add_argument(
        "--dataset-base",
        type=Path,
        default=Path(DEFAULT_DATASET_BASE),
        help=f"Base dataset directory (default: {DEFAULT_DATASET_BASE})",
    )
    dir_group.add_argument(
        "--pinhole-base",
        type=Path,
        default=Path(DEFAULT_PINHOLE_BASE),
        help=f"Base directory for pinhole images (default: {DEFAULT_PINHOLE_BASE})",
    )
    dir_group.add_argument(
        "--feather-name",
        type=str,
        default=DEFAULT_FEATHER_NAME,
        help=f"Feather filename under <city>/landmarks/ (default: {DEFAULT_FEATHER_NAME})",
    )

    parser.add_argument(
        "--feather",
        type=Path,
        default=None,
        help="OSM landmarks feather file (overrides auto-resolution from --city)",
    )
    parser.add_argument(
        "--model", type=str, default="qwen3.5:35b", help="Ollama model tag"
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=None,
        help="URL to an existing Ollama server (skip managed startup)",
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=5,
        help="Max tool-calling iterations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: stdout)",
    )
    args = parser.parse_args()

    # Resolve image paths
    if args.images:
        image_paths = args.images
    elif args.city and args.pano_id:
        image_paths = resolve_pano_images(args.pinhole_base, args.city, args.pano_id)
        print(f"Resolved images: {[p.name for p in image_paths]}", file=sys.stderr)
    else:
        parser.error("Provide either --images or both --city and --pano-id")

    # Resolve feather path
    if args.feather:
        feather_path = args.feather
    elif args.city:
        feather_path = resolve_feather(args.dataset_base, args.city, args.feather_name)
        print(f"Resolved feather: {feather_path}", file=sys.stderr)
    else:
        parser.error("Provide --feather or --city (to auto-resolve)")

    # Validate image paths
    for p in image_paths:
        if not p.exists():
            parser.error(f"Image not found: {p}")

    # Build tag search index
    print(f"Loading feather file: {feather_path}", file=sys.stderr)
    df = pd.read_feather(feather_path)
    print(f"Building tag search index from {len(df)} landmarks...", file=sys.stderr)
    index = TagSearchIndex(df)
    print(f"Index ready: {len(index._values)} unique values", file=sys.stderr)

    # Create ollama client
    if args.ollama_base_url:
        print(f"Using existing Ollama server at {args.ollama_base_url}", file=sys.stderr)
        client = ollama_sdk.Client(host=args.ollama_base_url)
        result = run_extraction(client, args.model, image_paths, index, args.max_tool_rounds)
    else:
        print(f"Starting managed Ollama server for model {args.model}...", file=sys.stderr)
        with Ollama(args.model) as server:
            client = ollama_sdk.Client(host=server.base_url())
            result = run_extraction(client, args.model, image_paths, index, args.max_tool_rounds)

    # Output
    output_json = result.model_dump_json(indent=2)
    if args.output:
        args.output.write_text(output_json)
        print(f"Wrote result to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
