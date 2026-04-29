"""Extract OSM tags from panorama images using Ollama with tool calling.

Local alternative to OpenAI/Gemini batch APIs. Uses an Ollama model (default:
qwen3.5:35b) with a SQL tool that queries a running ``osm_tag_server`` for
OSM tag lookups during landmark extraction.

Usage::

    # Start the tag server first (separate terminal):
    bazel run //experimental/overhead_matching/swag/scripts:osm_tag_server -- \
      --feather /data/.../landmarks.feather

    # Then run extraction:
    bazel run //experimental/overhead_matching/swag/scripts:ollama_osm_extraction -- \
      --city Chicago --pano-id 005T5CAKugPnibKVcndUSA \
      --ollama-base-url http://localhost:11434 \
      --tag-server-url http://localhost:8421
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ollama as ollama_sdk
import requests

from common.ollama.pyollama import Ollama
from experimental.overhead_matching.swag.model.semantic_landmark_extractor import (
    OSMTagExtraction,
    encode_image_to_base64,
    osm_tags_user_prompt,
)

OSM_TAGS_SYSTEM_PROMPT = """<role>
You are an expert at identifying landmarks in street-level imagery and mapping them to OpenStreetMap (OSM) tags.
</role>

<instructions>
Given four images which show the same location from yaws 0° (facing north), 90° (facing west), 180° (facing south), and 270° (facing east) respectively, identify distinctive, permanent landmarks and classify them using OSM's key=value tagging system.

Your workflow should be:
 1. Analyze the images for any visually distinctive landmarks and useful signs. Summarize what you have found.
 2. Use tools at your disposal to discover what OSM tags are appropriate and justifiable for each identified landmark.
 3. Report your results using the specified JSON schema.

For each landmark:
- Assign a primary OSM tag (e.g., amenity=cafe, shop=pharmacy, building=apartments)
- Add relevant additional tags (name, brand, cuisine, building:levels, etc.)
- Specify which yaw angle(s)/images the landmark appears in and provide bounding boxes for each
- Rate your confidence (high/medium/low)
- Provide a brief description for debugging

If you cannot confidently idenitify any visually distinct landmarks, it is acceptable to return an empty list of landmarks.
Based on the images, classify the location type (e.g., urban_commercial, suburban, rural).
Finally, review your work and confirm you have not included any information you cannot confidently make out from the images.
</instructions>

<landmark_selection>
Focus on landmarks that are VISUALLY DISTINCTIVE and useful for identifying a specific location. Prioritize:
- Readable street names on signs (e.g., "Adams St", "Michigan Ave") — these are extremely
  informative for localization. Tag the street itself: highway=residential (or tertiary,
  secondary, primary depending on size) with name=<street name>. Use the tool to look up
  the street name and find its exact tags.
- Named businesses, restaurants, shops with visible signage
- Architecturally unique buildings (churches, historic buildings)
- Branded locations (gas stations, chain stores, banks)
- Named parks, monuments, public art, memorials
- Distinctive infrastructure (clock towers, unique bridges)

DO NOT include generic, ubiquitous features such as:
- Traffic signals, street lamps, fire hydrants
- Crosswalks, stop signs, generic road markings
- Plain sidewalks, curbs, gutters
- Trees, bushes, or grass unless they are a notable landmark (e.g., a named park)
- Apartment buildings or residential complexes, even if visually distinctive — these are
  rarely represented in the map data and are not useful for localization
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
- Exclude transient objects (cars, pedestrians, temporary items, construction areas)
- Do not mention location in image or relative to other landmarks
</constraints>

<output_format>
Provide your response as a JSON object conforming to the assigned schema.
Bounding box coordinates are normalized 0-1000, where (0,0) is top-left and (1000,1000) is bottom-right.
</output_format>
"""

DEFAULT_PINHOLE_BASE = "/data/overhead_matching/datasets/pinhole_images"
DEFAULT_TAG_SERVER_URL = "http://localhost:8421"

TOOL_INSTRUCTIONS = """
<tools>
You have a tool `query_landmarks` that executes SQL against a database of ALL
OSM landmarks in the city these images were taken in.

IMPORTANT: Use this tool to find the correct OSM tag vocabulary for landmarks you
see. Do NOT use it to identify which specific landmark something is — the database
contains every landmark in the entire city, not just ones near this location.

For example, if you see a church, query for what tags churches typically have, but do NOT pick
a specific church name from the results unless you can read that name in the image.
Only output tags that are directly supported by what you see in the panorama.

The database is NOT all-inclusive — some landmarks visible in the panorama may not
be in the map. That's fine; tag them with whatever you can justify from the image.
However, when you DO have a confident match (e.g. you can read a name or brand),
query the database for that landmark and extract as many tags as you can justify
from the image. Some landmarks are unique (e.g. "Statue of Liberty" appears once)
while others have multiple locations (e.g. "CVS" appears many times) — a unique
match lets you pull more tags confidently than a chain with many locations.

IMPORTANT: The tool can only tell you that a landmark exists in the map, it cannot
confirm that the landmark you see is in fact the one that you suspect. For example,
if you see that you are on a bridge, but you do not see any indications of which bridge it is,
just because you think that it might be the Lake Street bridge, verifying that the Lake Street
bridge is in the map DOES NOT confirm that you are on the Lake Street bridge.

Tables:
- tags(key, value, count) — all unique key=value tags across the city with occurrence counts
- landmark_tags(landmark_id, key, value) — per-landmark tags for co-occurrence queries
- landmarks(landmark_id, geom) — spatial geometries (SpatiaLite); use ST_Distance(a.geom, b.geom, 1) for meters

Workflow:
1. Look at the images and identify candidate landmarks
2. For each candidate, query the database to find appropriate tags:
   - What tags does this type of landmark use? e.g. query for amenity=place_of_worship to see co-occurring tags
   - If you can read a name/brand in the image, verify it exists: SELECT * FROM landmark_tags WHERE key='name' AND value LIKE '%readable text%'
   - If the name is unique (count=1), pull all its tags — many will apply
   - If the name has multiple locations (count>1), only include tags common to all or visible in the image
3. Only output tags you can justify from the image:
   - You CAN read "Walgreens" on a sign → include name=Walgreens, query for its full tag set
   - You see a church but cannot read its name → include amenity=place_of_worship, but do NOT guess a name
   - You see a generic storefront → do NOT assume what shop it is

Example queries:
- Find all tags for a named landmark: SELECT DISTINCT lt2.key, lt2.value FROM landmark_tags lt1 JOIN landmark_tags lt2 ON lt1.landmark_id = lt2.landmark_id WHERE lt1.key='name' AND lt1.value LIKE '%walgreen%'
- Browse names: SELECT value, count FROM tags WHERE key='name' ORDER BY count DESC LIMIT 20
- Browse brands: SELECT value, count FROM tags WHERE key='brand' ORDER BY count DESC LIMIT 20
- Find tags: SELECT key, value, count FROM tags WHERE value LIKE '%cafe%' ORDER BY count DESC LIMIT 10
- Check existence: SELECT count FROM tags WHERE key='shop' AND value='pharmacy'

Use LIKE '%substring%' for partial matches. Keep queries simple.
</tools>"""

TOOLS = [
    {
        "type": "function",
        "function": {
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
        },
    },
]


OSM_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "location_type": {
            "type": "string",
            "description": "Scene type classification (e.g., 'urban_commercial', 'suburban', 'rural')",
        },
        "landmarks": {
            "type": "array",
            "description": "List of landmarks with OSM tags",
            "items": {
                "type": "object",
                "properties": {
                    "primary_tag": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "enum": [
                                    "amenity", "shop", "building", "tourism",
                                    "leisure", "highway", "man_made", "historic",
                                    "natural", "office", "craft", "railway",
                                    "power", "landuse", "emergency", "public_transport",
                                ],
                                "description": "OSM tag key",
                            },
                            "value": {
                                "type": "string",
                                "description": "OSM tag value",
                            },
                        },
                        "required": ["key", "value"],
                    },
                    "additional_tags": {
                        "type": "array",
                        "description": "Additional OSM tags (name, brand, cuisine, building:levels, etc.)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {
                                    "type": "string",
                                    "description": "OSM tag key (e.g., 'name', 'brand', 'cuisine')",
                                },
                                "value": {
                                    "type": "string",
                                    "description": "OSM tag value",
                                },
                            },
                            "required": ["key", "value"],
                        },
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Confidence level for this identification",
                    },
                    "bounding_boxes": {
                        "type": "array",
                        "description": "List of bounding boxes showing where this landmark appears",
                        "items": {
                            "type": "object",
                            "properties": {
                                "yaw_angle": {
                                    "type": "string",
                                    "enum": ["0", "90", "180", "270"],
                                    "description": "Which yaw image this bounding box refers to",
                                },
                                "ymin": {"type": "integer", "description": "Min y (0-1000)"},
                                "xmin": {"type": "integer", "description": "Min x (0-1000)"},
                                "ymax": {"type": "integer", "description": "Max y (0-1000)"},
                                "xmax": {"type": "integer", "description": "Max x (0-1000)"},
                            },
                            "required": ["yaw_angle", "ymin", "xmin", "ymax", "xmax"],
                        },
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description for debugging purposes",
                    },
                },
                "required": [
                    "primary_tag", "additional_tags", "confidence",
                    "bounding_boxes", "description",
                ],
            },
        },
    },
    "required": ["location_type", "landmarks"],
}


def load_examples(conversations_dir: Path) -> str:
    """Load conversation JSONs and format as text examples for the system prompt.

    Serializes the reasoning, tool calls, tool results, and final answer from
    each conversation into an ``<examples>`` block.  Images and system prompt
    are omitted (redundant with the live prompt).
    """
    if not conversations_dir or not conversations_dir.is_dir():
        return ""

    parts = []
    for json_file in sorted(conversations_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
        messages = data.get("messages", [])

        lines = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")

            if role == "system":
                continue
            elif role == "user":
                # Skip the initial image prompt — it's the same every time
                if msg.get("image_paths") or "four images above" in content:
                    continue
                lines.append(f"User: {content}")
            elif role == "assistant":
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc["function"]
                        sql = fn["arguments"].get("sql", "")
                        lines.append(f"Assistant: [calls query_landmarks]\nSQL: {sql}")
                elif content:
                    lines.append(f"Assistant: {content}")
            elif role == "tool":
                lines.append(f"Tool result: {content}")

        if lines:
            parts.append("\n\n".join(lines))

    if not parts:
        return ""

    examples_text = "\n\n---\n\n".join(
        f"<example_{i + 1}>\n{text}\n</example_{i + 1}>"
        for i, text in enumerate(parts)
    )
    return f"""
<examples>
The following are examples of good extraction conversations. Note how the assistant:
- Describes what it sees before querying
- Uses the tool to discover correct OSM tag vocabulary
- Checks co-occurring tags to find additional relevant tags
- Only includes tags that can be justified from the images
- Explicitly states when it cannot justify additional tags

{examples_text}
</examples>"""


def dispatch_tool(name: str, args: dict, server_url: str) -> str:
    if name == "query_landmarks":
        if "sql" not in args or not args["sql"]:
            return json.dumps({"error": "Missing required 'sql' argument"})
        try:
            resp = requests.post(
                f"{server_url}/query",
                json={"sql": args["sql"]},
                timeout=10,
            )
            return resp.text
        except requests.RequestException as e:
            return json.dumps({"error": f"Server request failed: {e}"})
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
    server_url: str,
    max_tool_rounds: int = 15,
    examples_dir: Path | None = None,
) -> OSMTagExtraction:
    examples_text = load_examples(examples_dir) if examples_dir else ""
    system_prompt = OSM_TAGS_SYSTEM_PROMPT + TOOL_INSTRUCTIONS + examples_text

    messages = [
        {"role": "system", "content": system_prompt},
        build_user_message(image_paths),
    ]

    chat_kwargs = dict(model=model, think=True)
    schema = OSM_EXTRACTION_SCHEMA

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
            result = dispatch_tool(name, args, server_url)
            print(f"    -> {result[:200]}", file=sys.stderr)
            messages.append({"role": "tool", "content": result})

    # Final structured output call (format= can't be combined with tools=)
    if response.message.tool_calls:
        # Exhausted tool rounds — append last response and continue
        messages.append(response.message)
        print(f"  [max tool rounds reached]", file=sys.stderr)

    print(f"  [generating structured output]", file=sys.stderr)
    messages.append({"role": "user", "content": "Now output your final answer as JSON."})

    max_retries = 3
    for attempt in range(max_retries):
        response = client.chat(messages=messages, format=schema, **chat_kwargs)
        if getattr(response.message, "thinking", None):
            print(f"  [thinking] {response.message.thinking}", file=sys.stderr)

        content = response.message.content
        if not content or not content.strip():
            print(f"  [empty response, retry {attempt + 1}/{max_retries}]", file=sys.stderr)
            continue

        try:
            raw = json.loads(content)
            return OSMTagExtraction.model_validate(raw)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [bad response, retry {attempt + 1}/{max_retries}]: {e}", file=sys.stderr)
            print(f"  [content] {content[:200]}", file=sys.stderr)
            messages.append(response.message)
            messages.append({
                "role": "user",
                "content": f"Your response was not valid JSON: {e}. Please try again.",
            })

    raise RuntimeError("Failed to get valid structured output after retries")


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
        default=15,
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

    # Validate image paths
    for p in image_paths:
        if not p.exists():
            parser.error(f"Image not found: {p}")

    server_url = args.tag_server_url

    # Create ollama client
    if args.ollama_base_url:
        print(f"Using existing Ollama server at {args.ollama_base_url}", file=sys.stderr)
        client = ollama_sdk.Client(host=args.ollama_base_url)
        result = run_extraction(client, args.model, image_paths, server_url, args.max_tool_rounds)
    else:
        print(f"Starting managed Ollama server for model {args.model}...", file=sys.stderr)
        with Ollama(args.model) as server:
            client = ollama_sdk.Client(host=server.base_url())
            result = run_extraction(client, args.model, image_paths, server_url, args.max_tool_rounds)

    # Output
    output_json = result.model_dump_json(indent=2)
    if args.output:
        args.output.write_text(output_json)
        print(f"Wrote result to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
