"""Farfield VLM extraction prompts, response schema, and request building.

Import-light on purpose. `swag/model/semantic_landmark_extractor.py` drags
in torch, openai and pandas for what is, for this stage, prompt text plus a
pydantic schema plus one request-shaping function; the farfield extraction
stage runs where there is no torch. So the schema and request building live
here, and the prompt text lives in a dependency-free module both this and
that extractor import:

- `SYSTEM_PROMPTS` / `USER_PROMPT`: re-exported from
  `swag/model/farfield_prompt_text.py`, the single owner of the prompt text
  (a dependency-free module of strings, so importing it costs this stage
  nothing). Which prompt a run uses is a modeling choice, so the extraction
  stage requires `--prompt_type` explicitly, and the recorded
  `prompt_sha256` pins the text that actually went out.
- The response schema: a copy of main's OSM-tag pydantic models with ONE
  deliberate difference -- `place` is a valid primary tag key, because the
  farfield prompts direct islands and settlements to `place=island` etc.
  (that enum addition is the reason the models are copied rather than
  imported, over and above the dependency weight).
- `build_request` / `write_requests`: the Gemini batch request format, ported
  from `_create_panorama_batch_request` / `create_panorama_description_requests`,
  including the stale-render guard (requests are restricted to the stems the
  dataset currently contains -- a pinhole render can outlive a dataset trim,
  and every extra stem is a paid model call for a frame nothing reads).

A prompt name is a lookup key whose text can change under it, so consumers
that need to pin an extraction record `prompt_sha256(...)` (and the request
JSONL keeps the text verbatim as it went out).
"""

import base64
import hashlib
import json
from enum import Enum
from multiprocessing import Pool
from pathlib import Path
from typing import List

import tqdm
from pydantic import BaseModel, Field

from experimental.overhead_matching.swag.model import farfield_prompt_text

# ---------------------------------------------------------------------------
# Prompt registry
# ---------------------------------------------------------------------------

# The prompt TEXT has one owner, swag/model/farfield_prompt_text.py: a
# dependency-free module of strings that this stage and the shared VIGOR
# extractor both import. It used to be duplicated here to keep this module
# torch-free, but a second copy of a prompt is a silent fork -- every
# extraction run records prompt_sha256, so two registries drifting means
# two runs claiming one prompt_type were given different instructions.
SYSTEM_PROMPTS = dict(farfield_prompt_text.FARFIELD_SYSTEM_PROMPTS)

PROMPT_TYPES = tuple(sorted(SYSTEM_PROMPTS))

USER_PROMPT = farfield_prompt_text.OSM_TAGS_USER_PROMPT

MEDIA_RESOLUTIONS = ("MEDIA_RESOLUTION_LOW", "MEDIA_RESOLUTION_MEDIUM",
                     "MEDIA_RESOLUTION_HIGH", "MEDIA_RESOLUTION_ULTRA_HIGH")
THINKING_LEVELS = ("OFF", "LOW", "MEDIUM", "HIGH")


def prompt_sha256(prompt_type: str) -> str:
    """Digest of a registered prompt's TEXT, the thing that pins an extraction."""
    return hashlib.sha256(SYSTEM_PROMPTS[prompt_type].encode()).hexdigest()


# ---------------------------------------------------------------------------
# Response schema
#
# Copied from swag/model/semantic_landmark_extractor.py on main, with `place`
# added to OSMPrimaryTagKey. Everything downstream that reads the resulting
# predictions (farfield/dataset.py) keys on this shape.
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Bounding box for a landmark in a specific yaw image"""
    yaw_angle: str = Field(
        description="Which yaw image this bounding box refers to (0, 90, 180, or 270)")
    ymin: int = Field(
        description="Minimum y coordinate (0-1000), normalized to image height",
        ge=0, le=1000)
    xmin: int = Field(
        description="Minimum x coordinate (0-1000), normalized to image width",
        ge=0, le=1000)
    ymax: int = Field(
        description="Maximum y coordinate (0-1000), normalized to image height",
        ge=0, le=1000)
    xmax: int = Field(
        description="Maximum x coordinate (0-1000), normalized to image width",
        ge=0, le=1000)


class OSMPrimaryTagKey(str, Enum):
    """Primary OSM tag keys"""
    AMENITY = "amenity"
    SHOP = "shop"
    BUILDING = "building"
    TOURISM = "tourism"
    LEISURE = "leisure"
    HIGHWAY = "highway"
    MAN_MADE = "man_made"
    HISTORIC = "historic"
    NATURAL = "natural"
    OFFICE = "office"
    CRAFT = "craft"
    RAILWAY = "railway"
    POWER = "power"
    LANDUSE = "landuse"
    EMERGENCY = "emergency"
    PUBLIC_TRANSPORT = "public_transport"
    PLACE = "place"


class OSMPrimaryTag(BaseModel):
    """Primary OSM tag (key=value pair)"""
    key: OSMPrimaryTagKey = Field(description="OSM tag key")
    value: str = Field(description="OSM tag value")


class OSMAdditionalTag(BaseModel):
    """Additional OSM tag (key=value pair)"""
    key: str = Field(description="OSM tag key (e.g., 'name', 'brand', 'cuisine')")
    value: str = Field(description="OSM tag value")


class Confidence(str, Enum):
    """Confidence level for landmark identification"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class OSMLandmarkWithBBox(BaseModel):
    """A landmark with OSM tags and bounding boxes"""
    primary_tag: OSMPrimaryTag = Field(
        description="Primary OSM tag categorizing this landmark")
    additional_tags: List[OSMAdditionalTag] = Field(
        description="Additional OSM tags (name, brand, cuisine, building:levels, etc.)")
    confidence: Confidence = Field(
        description="Confidence level for this identification")
    bounding_boxes: List[BoundingBox] = Field(
        description="List of bounding boxes showing where this landmark appears across different yaw angles")
    description: str = Field(
        description="Brief description for debugging purposes")


class OSMTagExtraction(BaseModel):
    """OSM tag extraction from panorama images"""
    location_type: str = Field(
        description="Scene type classification (e.g., 'urban_commercial', 'suburban', 'rural')")
    landmarks: List[OSMLandmarkWithBBox] = Field(
        description="List of landmarks with OSM tags")


def _add_required_no_add_props(schema: dict) -> dict:
    """Mark every object's properties required and drop pydantic titles."""
    if isinstance(schema, dict):
        if "title" in schema:
            del schema["title"]
        if schema.get("type") == "object" and "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
        for value in schema.values():
            if isinstance(value, dict):
                _add_required_no_add_props(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _add_required_no_add_props(item)
    return schema


def _resolve_refs(schema, defs: dict = None):
    """Recursively resolve $ref in schema by inlining definitions."""
    if defs is None:
        defs = schema.get("$defs", {}) or schema.get("definitions", {})
    if isinstance(schema, dict):
        if "$ref" in schema:
            ref_name = schema["$ref"].split("/")[-1]
            if ref_name in defs:
                return _resolve_refs(defs[ref_name], defs)
            return schema
        return {key: _resolve_refs(value, defs)
                for key, value in schema.items()
                if key not in ("$defs", "definitions")}
    if isinstance(schema, list):
        return [_resolve_refs(item, defs) for item in schema]
    return schema


def response_schema() -> dict:
    """The Gemini responseSchema every farfield extraction request carries."""
    schema = OSMTagExtraction.model_json_schema()
    schema = _resolve_refs(schema)
    schema = _add_required_no_add_props(schema)
    return schema


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

# Faces are rendered at 90-degree yaw intervals with a 90-degree FOV, which is
# what panorama_to_pinhole emits and what geometry.direction_from_face_px
# assumes when it maps a detection's box back to a camera-frame azimuth.
PINHOLE_FACES = ("yaw_000", "yaw_090", "yaw_180", "yaw_270")
FACE_YAWS = (0, 90, 180, 270)

# Vertex batch input file size limit; requests are split into multiple JSONL
# files below it (a 2048px extraction runs a few MB per request).
MAX_BATCH_FILE_SIZE_GCP = 1_900_000_000

# Encode-and-write chunk size, to bound peak memory while batching.
_PANORAMA_CHUNK_SIZE = 1000


def build_request(key: str, images: list, *, prompt_type: str,
                  media_resolution: str, thinking_level: str) -> dict:
    """One Gemini batch request record: `{key, request}`.

    `images` is a list of (mime_type, base64_data), in face-yaw order.
    ULTRA_HIGH must be set per-part; every other media resolution is set
    globally in generationConfig (a Gemini API quirk, preserved verbatim from
    the reference implementation).
    """
    system_prompt = SYSTEM_PROMPTS[prompt_type]
    parts = []
    for mime_type, b64_data in images:
        part = {
            "inline_data": {
                "mime_type": mime_type,
                "data": b64_data,
            }
        }
        if media_resolution == "MEDIA_RESOLUTION_ULTRA_HIGH":
            part["media_resolution"] = {"level": media_resolution}
        parts.append(part)
    parts.append({"text": USER_PROMPT})

    generation_config = {
        "responseMimeType": "application/json",
        "responseSchema": response_schema(),
        "thinkingConfig": {"thinkingLevel": thinking_level},
    }
    if media_resolution != "MEDIA_RESOLUTION_ULTRA_HIGH":
        generation_config["mediaResolution"] = media_resolution

    return {
        "key": key,
        "request": {
            "contents": [{
                "parts": parts,
                "role": "user",
            }],
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "generationConfig": generation_config,
        },
    }


def collect_faces(pinhole_dir: Path, panorama_dir: Path) -> dict:
    """{pano_stem: [(mime_type, face_path) x4]} for the dataset's stems.

    Restricted to the stems `panorama_dir` currently contains. The pinhole
    render is an artifact that can outlive a dataset trim (folkestone_dover
    2026-08-17: 399 rendered stems vs 105 kept panoramas), and every extra
    stem here is a paid model call for a frame nothing downstream reads. A
    kept panorama with no render is an error: re-run the pinhole stage.
    """
    pinhole_dir, panorama_dir = Path(pinhole_dir), Path(panorama_dir)
    exts = ('.jpg', '.jpeg', '.png')
    wanted = sorted(p.stem for p in panorama_dir.iterdir()
                    if p.suffix.lower() in exts)
    if not wanted:
        raise RuntimeError(f"no panoramas found in {panorama_dir}")
    rendered = {d.name for d in pinhole_dir.iterdir() if d.is_dir()} \
        if pinhole_dir.exists() else set()
    missing = [s for s in wanted if s not in rendered]
    if missing:
        raise RuntimeError(
            f"{len(missing)} panorama(s) in {panorama_dir} have no pinhole "
            f"render, e.g. {missing[:3]}; re-run the pinhole stage")
    n_stale = len(rendered) - len(wanted)
    if n_stale:
        print(f"Excluding {n_stale} rendered stem(s) not in {panorama_dir} "
              f"(stale pinhole render)")

    faces_by_stem = {}
    for stem in wanted:
        faces = []
        for yaw in FACE_YAWS:
            path = pinhole_dir / stem / f"yaw_{yaw:03d}.jpg"
            if not path.exists():
                path = pinhole_dir / stem / f"yaw_{yaw:03d}.png"
            if not path.exists():
                raise RuntimeError(f"pinhole face not found: {path}")
            mime = "image/jpeg" if path.suffix.lower() == ".jpg" else "image/png"
            faces.append((mime, path))
        faces_by_stem[stem] = faces
    return faces_by_stem


def _encode_image_to_base64(image_path: Path) -> str:
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def write_requests(pinhole_dir: Path, panorama_dir: Path, out_dir: Path, *,
                   prompt_type: str, media_resolution: str,
                   thinking_level: str, num_workers: int = 8,
                   max_requests_per_batch: int = 10000,
                   disable_tqdm: bool = False) -> list:
    """Build every request for a dataset and write size-capped JSONL files.

    Returns the list of files written (`requests_000.jsonl`, ...). Requests
    are keyed by panorama stem -- the identity that names the pinhole
    subdirectories and joins the predictions back to frames downstream -- and
    ordered deterministically by stem.
    """
    faces_by_stem = collect_faces(pinhole_dir, panorama_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = sorted(faces_by_stem.items())
    written = []
    current, current_size = [], 0

    def flush():
        nonlocal current, current_size
        if not current:
            return
        path = out_dir / f"requests_{len(written):03d}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in current) + "\n")
        print(f"Wrote {path} ({len(current)} requests)")
        written.append(path)
        current, current_size = [], 0

    for start in tqdm.tqdm(range(0, len(items), _PANORAMA_CHUNK_SIZE),
                           desc="Building requests", disable=disable_tqdm):
        chunk = items[start:start + _PANORAMA_CHUNK_SIZE]
        chunk_paths = [path for _, faces in chunk for _, path in faces]
        if num_workers > 1:
            with Pool(num_workers) as pool:
                encoded = pool.map(_encode_image_to_base64, chunk_paths)
        else:
            encoded = [_encode_image_to_base64(p) for p in chunk_paths]
        b64_by_path = dict(zip(chunk_paths, encoded))

        for stem, faces in chunk:
            request = build_request(
                stem,
                [(mime, b64_by_path[path]) for mime, path in faces],
                prompt_type=prompt_type,
                media_resolution=media_resolution,
                thinking_level=thinking_level)
            size = len(json.dumps(request).encode('utf-8'))
            if current and (current_size + size > MAX_BATCH_FILE_SIZE_GCP
                            or len(current) >= max_requests_per_batch):
                flush()
            current.append(request)
            current_size += size
    flush()
    return written
