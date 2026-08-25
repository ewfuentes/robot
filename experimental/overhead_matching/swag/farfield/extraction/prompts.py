"""Farfield VLM extraction prompts, response schema, and request building.

This module owns the far-field prompt and request contract:

- `SYSTEM_PROMPTS` and `USER_PROMPT`: prompt text selected explicitly by a run;
- the response schema, including `place` for islands and settlements;
- one provider-neutral semantic request with Batch and online adapters;
- request writers restricted to the panorama stems in the current dataset, so
  stale renders cannot produce unconsumable model calls.

A prompt name is a lookup key whose text can change under it, so consumers
that need to pin an extraction record `prompt_sha256(...)` (and the request
JSONL keeps the text verbatim as it went out).
"""

import base64
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool
from pathlib import Path
from typing import Any, List, Mapping

import tqdm
from pydantic import BaseModel, Field

from experimental.overhead_matching.swag.farfield import paths as paths_lib

# ---------------------------------------------------------------------------
# Prompt registry
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    'osm_tags_farfield': """<role>
You are an expert at identifying distant landmarks in outdoor imagery and mapping them to OpenStreetMap (OSM) tags.
</role>

<context>
The four images come from a camera on a moving platform — a boat, a road vehicle, or a person on foot.
They show the same location at relative yaws 0°, 90°, 180°, and 270° (camera frame — NOT compass-aligned; do not assume any cardinal direction).
The setting may be any outdoor environment: harbour or open water, a river, a mountain range, forest or trail, farmland or open plain, or a built-up area.
Much of each image is usually sky, water, vegetation or bare ground; the landmarks are the exceptions.
The platform itself (deck, railings, canopy, bonnet, dashboard, handlebars, the operator, passengers, safety equipment) is visible in most images and must be completely ignored.
</context>

<instructions>
Identify permanent, distinctive landmarks that plausibly appear in OpenStreetMap and classify them using OSM's key=value tagging system.

Your workflow should be:
 1. Scan the full horizon in all four images — skyline, ridgelines, shoreline, and the middle distance — for distinctive permanent features. Summarize what you have found.
 2. Identify what OSM tags are appropriate and justifiable for each identified landmark.

For each landmark:
- Assign a primary OSM tag (e.g., natural=peak, man_made=crane, man_made=silo, historic=fort, building=commercial, place=island)
- Add relevant additional tags (name, height, colour, etc. Do not give 2 of the same tags to a single landmark). Include name=<name> ONLY under the naming rules below.
- Add a distance_estimate additional tag with exactly one of these values: "under_100m", "100m_to_500m", "500m_to_2km", "2km_to_10km", "over_10km"
- Specify which yaw angle(s)/images the landmark appears in and provide bounding boxes for each. Boxes must be TIGHT around the landmark itself — never a whole skyline, ridgeline or shoreline in one box.
- Rate your confidence (high/medium/low) using the rubric below
- Provide a brief description following the description rules below

If you cannot confidently identify any visually distinct landmarks, it is acceptable to return an empty list of landmarks.
Based on the images, classify the location type in free text (e.g., open_water, inner_harbor, river_valley, alpine_ridge, forest_trail, open_farmland, high_desert, urban_waterfront).
Finally, review your work and remove anything you cannot confidently make out from the images, along with any tag you cannot confidently justify.
</instructions>

<landmark_selection>
A good far-field landmark is FIXED in place, VISIBLE from a long way off, DISTINCTIVE enough to tell from its neighbours, and plausibly mapped in OSM. There is no distance limit — far-away features are the primary target, as long as you are confident what they are. Prioritize by how strongly a feature identifies itself:

1. Named or recognizable features — a summit you can name, a famous bridge or tower, a structure with a readable sign. These are worth the most; see the naming rules.
2. Features whose category plus appearance narrows them down — a glaciated peak, a lighthouse, a red-and-white banded chimney, a grain elevator of six linked silos.
3. Features that repeat and are individually generic — wind turbines, transmission pylons, silos. Report each one you can see as its own landmark; several weak detections combined with other evidence can still locate you. Where instances are so dense or numerous that they cannot be told apart at all, they are not useful landmarks.

Examples across settings, not an exhaustive list:
- Terrain: summits and named high points (natural=peak), saddles and cols (natural=saddle), ridges (natural=ridge), glaciers and permanent snowfields (natural=glacier), cliffs and rock faces (natural=cliff), buttes and mesas, islands (place=island)
- Trail and summit markers: cairns (man_made=cairn), summit survey markers (man_made=survey_point), signed summit posts
- Tall structures: radio and communications masts (man_made=mast, tower:type=communication), transmission pylons (power=tower), water towers, chimneys and smokestacks, church spires, clock towers, skyscrapers with a distinctive top, silos and grain elevators (man_made=silo), storage tanks, wind turbines (power=generator with generator:source=wind), fire lookouts and summit huts
- Water and coast: lighthouses and daybeacons on pilings or rocks (seamark:type=beacon_lateral), channel, lateral and special-purpose buoys (seamark:type=buoy_lateral or buoy_special_purpose), dams and weirs, locks, bridges (man_made=bridge), piers, wharves, breakwaters, marinas (leisure=marina — a dense cluster of sailboat masts marks one), container and gantry cranes
- Built and historic: forts, monuments and memorials (historic=*), ski lifts and aerial cableways (aerialway=*), large barns and farm complexes, buildings ONLY if at least one of: unique shape/colour/silhouette, a readable sign or name, or you recognize the specific building

Identifying attributes — ALWAYS include the ones you can actually see, since these are what distinguish one instance from hundreds of its neighbours:
- A number or letter painted or mounted on the structure (e.g. a buoy's "8", "13", "1SC"; a pylon's line number) → give it as name=<exactly as read>, ONLY if legible and not inferred
- colour=<red|green|yellow|white|white;orange|...> for anything with a deliberate colour scheme (buoys, beacons, banded chimneys, painted tanks)
- Shape, where the category has standard shapes — e.g. seamark:<type>:shape=<can|nun|pillar|spar> ("can" is a flat-topped cylinder, "nun" a cone)

DO NOT include:
- Anything that moves: watercraft (even docked or anchored), road and rail vehicles, aircraft, livestock, people. A marina is a landmark; the boats in it are not.
- The platform the camera is on, or anything mounted on it
- Wakes, waves, sun glare, clouds, snow patches that are plainly seasonal, birds
- Generic shoreline, generic tree lines, generic forest, riprap, ordinary fields
- Rows of visually identical generic buildings (e.g., condo blocks) with nothing to tell them apart

Report each physical feature as its own landmark, including each member of a repeated group.
</landmark_selection>

<naming_rules>
A name is the most valuable thing you can attach to a landmark: it narrows a match
from "one of the hundreds of peaks, buoys or towers in this region" to one specific
feature. Give one whenever you honestly can.

Two routes to a name are equally legitimate:
- READING it: a sign, a painted number or letter, a summit marker, a building's name on its facade.
- RECOGNIZING it: you know this specific mountain, bridge, tower or building from its shape, profile and setting. Recognition is a real source of identity, not a guess, and for distant natural features it is usually the ONLY route — a summit 10 km away carries no signage. Name the peaks, ranges, islands and well-known structures you genuinely know.

The test is whether you are confident in THIS feature's identity. It is NOT whether
other things nearby look similar. Similar-looking neighbours are the normal case in
every setting this prompt covers — a ridgeline of peaks, a field of turbines, a row
of channel buoys — so treat them as a reason to look for what distinguishes this
one, never as a reason to withhold a name you are sure of.

Express your certainty through the `confidence` field rather than by staying silent.
When you give a name, `confidence` describes your certainty in THE NAME, not merely
in the category:
- high: you are sure of the identity — read clearly, or recognized unmistakably
- medium: the identification is probable, but you would not stake a position fix on it
- low: do not name it at all; report the category and description instead

Never take a name from a billboard, advertising banner or other commercial signage.
Never derive one from geographic context alone ("we are near X, so this must be X"),
and never infer a number you cannot read. A confidently wrong name points at a real
feature that may be many kilometres away — but so does withholding a name you
actually know, by leaving the feature indistinguishable from its neighbours. Report
what you know, and rate how sure you are.
</naming_rules>

<description_rules>
Descriptions must be stable across viewpoints so the same landmark can be re-identified from other locations:
- Describe intrinsic properties only: shape, colour, material, relative height, count of elements (e.g., "granite fort with sloped walls and a flagpole", "red-and-white banded smokestack", "glaciated pyramidal summit with a rocky north face").
- If you recognize the landmark, lead with its canonical name.
- NEVER mention: position in the image, direction relative to the observer, distance, lighting, weather, or nearby transient objects.
</description_rules>

<confidence_rubric>
Rate the landmark as you have reported it — including its name, if you gave one:
- high: an identity you are sure of (name read clearly or recognized unmistakably), or an unnamed feature whose category is unmistakable (a container crane, a lighthouse, a wind turbine)
- medium: the category is clear but the instance is generic (an unnamed pier, an unnumbered buoy, one silo among several), or a name you believe but cannot confirm
- low: category is uncertain — prefer omitting these unless the feature is very distinctive visually, and never attach a name to one
</confidence_rubric>

<osm_tag_guidelines>
## Primary OSM Tag Categories

- `natural`: terrain and natural features (peak, saddle, ridge, glacier, cliff, rock, beach, wood, water, coastline)
- `man_made`: non-building structures (mast, tower, silo, storage_tank, chimney, water_tower, crane, lighthouse, pier, breakwater, bridge, cairn, survey_point)
- `place`: islands and settlements (island, islet, village, town)
- `historic`: historically significant features (fort, monument, memorial, lighthouse)
- `building`: structures with a roof (commercial, church, industrial, farm, hotel). Use `building=yes` if unclear.
- `power`: power infrastructure (tower for a transmission pylon, generator for a wind turbine or solar plant, plant, line)
- `leisure`: recreation (marina, park, nature_reserve, sports_centre)
- `tourism`: visitor attractions (hotels, museums, viewpoints, alpine_hut)
- `amenity`: facilities providing services (ferry_terminal, shelter, restaurants)
- `landuse`: land use areas (industrial, port, military, farmland, quarry)
- `aerialway`: cable cars, chairlifts, gondolas
- `railway`: rail infrastructure
- `seamark:type`: navigational aids (buoy_lateral, buoy_special_purpose, beacon_lateral, light_major)

## Key Distinctions

- **man_made vs building**: Use building if it has walls and a roof for human use; man_made for towers, masts, silos, tanks, piers, cranes
- **natural=peak vs natural=ridge**: peak for a distinct summit point; ridge for an extended crest
- **power=tower vs man_made=mast/tower**: power=tower is a transmission pylon carrying lines; man_made=mast is a guyed communications mast; man_made=tower is a freestanding tower
- **historic=fort vs building**: Use historic=fort for fortifications
- **leisure vs tourism**: Use leisure for local recreation; tourism for visitor attractions
</osm_tag_guidelines>


<output_format>
Provide your response as a JSON object conforming to the assigned schema.
Bounding box coordinates are normalized 0-1000, where (0,0) is top-left and (1000,1000) is bottom-right.
</output_format>
""",
    # v2 grounds identity in the structure's own visible features: name the
    # structure rather than its scene, require a visible differentiator for a
    # lookalike, and record a painted designator as `ref` rather than `name`.
    'osm_tags_farfield_v2': """<role>
You are an expert at identifying distant landmarks in outdoor imagery and mapping them to OpenStreetMap (OSM) tags.
</role>

<context>
The four images come from a camera on a moving platform — a boat, a road vehicle, or a person on foot.
They show the same location at relative yaws 0°, 90°, 180°, and 270° (camera frame — NOT compass-aligned; do not assume any cardinal direction).
The setting may be any outdoor environment: harbour or open water, a river, a mountain range, forest or trail, farmland or open plain, or a built-up area.
Much of each image is usually sky, water, vegetation or bare ground; the landmarks are the exceptions.
The platform itself (deck, railings, canopy, bonnet, dashboard, handlebars, the operator, passengers, safety equipment) is visible in most images and must be completely ignored.
</context>

<instructions>
Identify permanent, distinctive landmarks that plausibly appear in OpenStreetMap and classify them using OSM's key=value tagging system.

Your workflow should be:
 1. Scan the full horizon in all four images — skyline, ridgelines, shoreline, and the middle distance — for distinctive permanent features. Summarize what you have found.
 2. Identify what OSM tags are appropriate and justifiable for each identified landmark.

For each landmark:
- Assign a primary OSM tag (e.g., natural=peak, man_made=crane, man_made=silo, historic=fort, building=commercial, place=island)
- Add relevant additional tags (name, height, colour, etc. Do not give 2 of the same tags to a single landmark). Include name=<name> ONLY under the naming rules below.
- Add a distance_estimate additional tag with exactly one of these values: "under_100m", "100m_to_500m", "500m_to_2km", "2km_to_10km", "over_10km"
- Specify which yaw angle(s)/images the landmark appears in and provide bounding boxes for each. Boxes must be TIGHT around the landmark itself — never a whole skyline, ridgeline or shoreline in one box.
- Rate your confidence (high/medium/low) using the rubric below
- Provide a brief description following the description rules below

If you cannot confidently identify any visually distinct landmarks, it is acceptable to return an empty list of landmarks.
Based on the images, classify the location type in free text (e.g., open_water, inner_harbor, river_valley, alpine_ridge, forest_trail, open_farmland, high_desert, urban_waterfront).
Finally, review your work and remove anything you cannot confidently make out from the images, along with any tag you cannot confidently justify.
</instructions>

<landmark_selection>
A good far-field landmark is FIXED in place, VISIBLE from a long way off, DISTINCTIVE enough to tell from its neighbours, and plausibly mapped in OSM. There is no distance limit — far-away features are the primary target, as long as you are confident what they are. Prioritize by how strongly a feature identifies itself:

1. Named or recognizable features — a summit you can name, a famous bridge or tower, a structure with a readable sign. These are worth the most; see the naming rules.
2. Features whose category plus appearance narrows them down — a glaciated peak, a lighthouse, a red-and-white banded chimney, a grain elevator of six linked silos.
3. Features that repeat and are individually generic — wind turbines, transmission pylons, silos. Report each one you can see as its own landmark; several weak detections combined with other evidence can still locate you. Where instances are so dense or numerous that they cannot be told apart at all, they are not useful landmarks.

Examples across settings, not an exhaustive list:
- Terrain: summits and named high points (natural=peak), saddles and cols (natural=saddle), ridges (natural=ridge), glaciers and permanent snowfields (natural=glacier), cliffs and rock faces (natural=cliff), buttes and mesas, islands (place=island)
- Trail and summit markers: cairns (man_made=cairn), summit survey markers (man_made=survey_point), signed summit posts
- Tall structures: radio and communications masts (man_made=mast, tower:type=communication), transmission pylons (power=tower), water towers, chimneys and smokestacks, church spires, clock towers, skyscrapers with a distinctive top, silos and grain elevators (man_made=silo), storage tanks, wind turbines (power=generator with generator:source=wind), fire lookouts and summit huts
- Water and coast: lighthouses and daybeacons on pilings or rocks (seamark:type=beacon_lateral), channel, lateral and special-purpose buoys (seamark:type=buoy_lateral or buoy_special_purpose), dams and weirs, locks, bridges (man_made=bridge), piers, wharves, breakwaters, marinas (leisure=marina — a dense cluster of sailboat masts marks one), container and gantry cranes
- Built and historic: forts, monuments and memorials (historic=*), ski lifts and aerial cableways (aerialway=*), large barns and farm complexes, buildings ONLY if at least one of: unique shape/colour/silhouette, a readable sign or name, or you recognize the specific building

Identifying attributes — ALWAYS include the ones you can actually see, since these are what distinguish one instance from hundreds of its neighbours:
- A number or letter painted or mounted on the structure (e.g. a buoy's "8", "13", "1SC"; a pylon's line number) → give it as name=<exactly as read>, ONLY if legible and not inferred
- colour=<red|green|yellow|white|white;orange|...> for anything with a deliberate colour scheme (buoys, beacons, banded chimneys, painted tanks)
- Shape, where the category has standard shapes — e.g. seamark:<type>:shape=<can|nun|pillar|spar> ("can" is a flat-topped cylinder, "nun" a cone)

DO NOT include:
- Anything that moves: watercraft (even docked or anchored), road and rail vehicles, aircraft, livestock, people. A marina is a landmark; the boats in it are not.
- The platform the camera is on, or anything mounted on it
- Wakes, waves, sun glare, clouds, snow patches that are plainly seasonal, birds
- Generic shoreline, generic tree lines, generic forest, riprap, ordinary fields
- Rows of visually identical generic buildings (e.g., condo blocks) with nothing to tell them apart

Report each physical feature as its own landmark, including each member of a repeated group.
</landmark_selection>

<naming_rules>
Two routes to a name are legitimate:
- READING it: a sign, a summit marker, a building's name on its facade.
- RECOGNIZING it: you know this specific item (e.g., mountain, bridge, tower, building) from its
  own shape, profile and proportions. Name the landmarks you genuinely know, do not guess if there is ambiguity.

A name must be justified by what you can see of
that structure itself - its outline, its top, its proportions, its colour, its
signage, its position relative to other features in the same image. It must NEVER
rest on the overall view resembling a place you know. 

Express your certainty through the `confidence` field rather than by staying silent.
When you give a name, `confidence` describes your certainty in THE NAME, not merely
in the category:
- high: you are sure of the identity - read clearly, or recognized from this
  structure's own form
- medium: the identification is probable, but you would not stake a position fix on it
- low: do not name it at all; report the category and description instead

Never take a name from a billboard, advertising banner or other commercial signage.
Never derive one from geographic context alone ("we are near X, so this must be X").
Report what you know, and rate how sure you are.
</naming_rules>

<description_rules>
Descriptions must be stable across viewpoints so the same landmark can be re-identified from other locations:
- Describe intrinsic properties only: shape, colour, material, relative height, count of elements (e.g., "granite fort with sloped walls and a flagpole", "red-and-white banded smokestack", "glaciated pyramidal summit with a rocky north face").
- If you recognize the landmark, lead with its canonical name.
- NEVER mention: position in the image, direction relative to the observer, distance, lighting, weather, or nearby transient objects.
</description_rules>

<confidence_rubric>
Rate the landmark as you have reported it — including its name, if you gave one:
- high: an identity you are sure of (name read clearly or recognized unmistakably), or an unnamed feature whose category is unmistakable (a container crane, a lighthouse, a wind turbine)
- medium: the category is clear but the instance is generic (an unnamed pier, an unnumbered buoy, one silo among several), or a name you believe but cannot confirm
- low: category is uncertain — prefer omitting these unless the feature is very distinctive visually, and never attach a name to one
</confidence_rubric>

<osm_tag_guidelines>
## Primary OSM Tag Categories

- `natural`: terrain and natural features (peak, saddle, ridge, glacier, cliff, rock, beach, wood, water, coastline)
- `man_made`: non-building structures (mast, tower, silo, storage_tank, chimney, water_tower, crane, lighthouse, pier, breakwater, bridge, cairn, survey_point)
- `place`: islands and settlements (island, islet, village, town)
- `historic`: historically significant features (fort, monument, memorial, lighthouse)
- `building`: structures with a roof (commercial, church, industrial, farm, hotel). Use `building=yes` if unclear.
- `power`: power infrastructure (tower for a transmission pylon, generator for a wind turbine or solar plant, plant, line)
- `leisure`: recreation (marina, park, nature_reserve, sports_centre)
- `tourism`: visitor attractions (hotels, museums, viewpoints, alpine_hut)
- `amenity`: facilities providing services (ferry_terminal, shelter, restaurants)
- `landuse`: land use areas (industrial, port, military, farmland, quarry)
- `aerialway`: cable cars, chairlifts, gondolas
- `railway`: rail infrastructure
- `seamark:type`: navigational aids (buoy_lateral, buoy_special_purpose, beacon_lateral, light_major)

## Key Distinctions

- **man_made vs building**: Use building if it has walls and a roof for human use; man_made for towers, masts, silos, tanks, piers, cranes
- **natural=peak vs natural=ridge**: peak for a distinct summit point; ridge for an extended crest
- **power=tower vs man_made=mast/tower**: power=tower is a transmission pylon carrying lines; man_made=mast is a guyed communications mast; man_made=tower is a freestanding tower
- **historic=fort vs building**: Use historic=fort for fortifications
- **leisure vs tourism**: Use leisure for local recreation; tourism for visitor attractions
</osm_tag_guidelines>


<output_format>
Provide your response as a JSON object conforming to the assigned schema.
Bounding box coordinates are normalized 0-1000, where (0,0) is top-left and (1000,1000) is bottom-right.
</output_format>
""",
}

PROMPT_TYPES = tuple(sorted(SYSTEM_PROMPTS))

USER_PROMPT = """
Based on the four images above (which show the same location from yaws 0°, 90°, 180°, and 270° respectively), identify all landmarks and classify them using OSM tags.
"""

MEDIA_RESOLUTIONS = ("MEDIA_RESOLUTION_LOW", "MEDIA_RESOLUTION_MEDIUM",
                     "MEDIA_RESOLUTION_HIGH", "MEDIA_RESOLUTION_ULTRA_HIGH")
THINKING_LEVELS = ("OFF", "LOW", "MEDIUM", "HIGH")


def prompt_sha256(prompt_type: str) -> str:
    """Digest of a registered prompt's TEXT, the thing that pins an extraction."""
    return hashlib.sha256(SYSTEM_PROMPTS[prompt_type].encode()).hexdigest()


# ---------------------------------------------------------------------------
# Response schema. This module owns the persisted far-field prediction shape
# consumed by `farfield/dataset.py`.
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


def _json_copy(value: Any, where: str) -> Any:
    """Detach one JSON value while rejecting provider-invalid constants."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{where} must be finite JSON: {error}") from error


def _canonical_part(value: Mapping[str, Any], where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    part = _json_copy(dict(value), where)
    if set(part) == {"text"}:
        if not isinstance(part["text"], str):
            raise ValueError(f"{where}.text must be a string")
        return part
    if set(part) == {"inline_data"}:
        inline = part["inline_data"]
        if (not isinstance(inline, dict)
                or set(inline) != {"mime_type", "data"}
                or not isinstance(inline["mime_type"], str)
                or not inline["mime_type"]
                or not isinstance(inline["data"], str)
                or not inline["data"]):
            raise ValueError(
                f"{where}.inline_data must contain exact non-empty "
                "mime_type and data strings")
        return part
    raise ValueError(
        f"{where} must contain exactly one canonical text or inline_data part")


@dataclass(frozen=True)
class SemanticRequest:
    """Provider-neutral request from which both Vertex transports are built.

    Media resolution is a semantic property here. Its provider-specific
    placement (global for LOW/MEDIUM/HIGH, per image part for ULTRA_HIGH) is
    owned exclusively by the adapters below.
    """

    key: str
    system_instruction: str
    parts: tuple[dict[str, Any], ...]
    response_schema: dict[str, Any]
    thinking_level: str
    media_resolution: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("request key must be a non-empty string")
        if (not isinstance(self.system_instruction, str)
                or not self.system_instruction):
            raise ValueError("system instruction must be a non-empty string")
        if (not isinstance(self.thinking_level, str)
                or not self.thinking_level):
            raise ValueError("thinking level must be a non-empty string")
        if (self.media_resolution is not None
                and self.media_resolution not in MEDIA_RESOLUTIONS):
            raise ValueError(
                f"unsupported media resolution {self.media_resolution!r}")
        if not self.parts:
            raise ValueError("request must contain at least one content part")
        object.__setattr__(self, "parts", tuple(
            _canonical_part(part, f"parts[{index}]")
            for index, part in enumerate(self.parts)))
        if (self.media_resolution is not None
                and not any("inline_data" in part for part in self.parts)):
            raise ValueError(
                "media resolution requires at least one image part")
        schema = _json_copy(self.response_schema, "response_schema")
        if not isinstance(schema, dict):
            raise ValueError("response_schema must be an object")
        object.__setattr__(self, "response_schema", schema)


def semantic_request(
        key: str, *, system_instruction: str,
        parts: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
        response_schema: Mapping[str, Any], thinking_level: str,
        media_resolution: str | None = None) -> SemanticRequest:
    """Build the single canonical representation used by both transports."""
    return SemanticRequest(
        key=key,
        system_instruction=system_instruction,
        parts=tuple(dict(part) for part in parts),
        response_schema=dict(response_schema),
        thinking_level=thinking_level,
        media_resolution=media_resolution,
    )


def batch_request(request: SemanticRequest) -> dict[str, Any]:
    """Adapt a semantic request to the exact Vertex Batch request object."""
    parts = [_json_copy(part, "request part") for part in request.parts]
    if request.media_resolution == "MEDIA_RESOLUTION_ULTRA_HIGH":
        for part in parts:
            if "inline_data" in part:
                part["media_resolution"] = {
                    "level": request.media_resolution,
                }

    generation_config = {
        "responseMimeType": "application/json",
        "responseSchema": _json_copy(
            request.response_schema, "response_schema"),
        "thinkingConfig": {"thinkingLevel": request.thinking_level},
    }
    if (request.media_resolution is not None
            and request.media_resolution != "MEDIA_RESOLUTION_ULTRA_HIGH"):
        generation_config["mediaResolution"] = request.media_resolution

    return {
        "contents": [{"parts": parts, "role": "user"}],
        "systemInstruction": {
            "parts": [{"text": request.system_instruction}],
        },
        "generationConfig": generation_config,
    }


def batch_record(request: SemanticRequest) -> dict[str, Any]:
    """Adapt a semantic request to one keyed Vertex Batch JSONL record."""
    return {"key": request.key, "request": batch_request(request)}


def semantic_request_from_batch(
        key: str, value: Mapping[str, Any]) -> SemanticRequest:
    """Recover and validate the semantic request stored in batch JSONL."""
    if not isinstance(value, Mapping) or set(value) != {
            "contents", "systemInstruction", "generationConfig"}:
        raise ValueError("batch request has an invalid top-level shape")
    contents = value["contents"]
    if (not isinstance(contents, list) or len(contents) != 1
            or not isinstance(contents[0], Mapping)
            or set(contents[0]) != {"parts", "role"}
            or contents[0]["role"] != "user"
            or not isinstance(contents[0]["parts"], list)):
        raise ValueError("batch request must contain one user content")
    instruction = value["systemInstruction"]
    if (not isinstance(instruction, Mapping)
            or set(instruction) != {"parts"}
            or not isinstance(instruction["parts"], list)
            or len(instruction["parts"]) != 1
            or not isinstance(instruction["parts"][0], Mapping)
            or set(instruction["parts"][0]) != {"text"}):
        raise ValueError("batch request has an invalid system instruction")

    generation = value["generationConfig"]
    required_generation = {
        "responseMimeType", "responseSchema", "thinkingConfig",
    }
    if (not isinstance(generation, Mapping)
            or set(generation) not in (
                required_generation,
                required_generation | {"mediaResolution"})
            or generation["responseMimeType"] != "application/json"):
        raise ValueError("batch request has an invalid generation config")
    thinking = generation["thinkingConfig"]
    if (not isinstance(thinking, Mapping)
            or set(thinking) != {"thinkingLevel"}):
        raise ValueError("batch request has an invalid thinking config")

    parts = []
    per_part_resolution = []
    image_count = 0
    for index, raw_part in enumerate(contents[0]["parts"]):
        if not isinstance(raw_part, Mapping):
            raise ValueError(f"batch request part {index} is not an object")
        part = _json_copy(dict(raw_part), f"batch request part {index}")
        has_marker = "media_resolution" in part
        marker = part.pop("media_resolution", None)
        if "inline_data" in part:
            image_count += 1
            if has_marker:
                if (not isinstance(marker, dict)
                        or set(marker) != {"level"}):
                    raise ValueError(
                        f"batch request part {index} has an invalid media "
                        "resolution")
                per_part_resolution.append(marker["level"])
        elif has_marker:
            raise ValueError(
                "per-part media resolution is valid only on image parts")
        parts.append(_canonical_part(part, f"batch request part {index}"))

    global_resolution = generation.get("mediaResolution")
    if global_resolution is not None and per_part_resolution:
        raise ValueError(
            "batch request mixes global and per-part media resolution")
    if per_part_resolution:
        if (len(per_part_resolution) != image_count
                or set(per_part_resolution) != {
                    "MEDIA_RESOLUTION_ULTRA_HIGH"}):
            raise ValueError(
                "ULTRA_HIGH media resolution must be set on every image part")
        media_resolution = "MEDIA_RESOLUTION_ULTRA_HIGH"
    else:
        media_resolution = global_resolution
        if media_resolution == "MEDIA_RESOLUTION_ULTRA_HIGH":
            raise ValueError(
                "ULTRA_HIGH media resolution must be set per image part")

    return semantic_request(
        key,
        system_instruction=instruction["parts"][0]["text"],
        parts=parts,
        response_schema=generation["responseSchema"],
        thinking_level=thinking["thinkingLevel"],
        media_resolution=media_resolution,
    )


def online_request(request: SemanticRequest) -> dict[str, Any]:
    """Adapt a semantic request to models.generate_content keyword args."""
    adapted = batch_request(request)
    generation = adapted["generationConfig"]
    config = {
        "system_instruction": request.system_instruction,
        "response_mime_type": generation["responseMimeType"],
        "response_schema": generation["responseSchema"],
        "thinking_config": {
            "thinking_level": request.thinking_level,
        },
    }
    if "mediaResolution" in generation:
        config["media_resolution"] = generation["mediaResolution"]
    return {"contents": adapted["contents"], "config": config}


def online_request_from_batch(
        key: str, value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a persisted batch request and adapt it for the online SDK."""
    return online_request(semantic_request_from_batch(key, value))


# Faces are rendered at 90-degree yaw intervals with a 90-degree FOV, which is
# what panorama_to_pinhole emits and what geometry.direction_from_face_px
# assumes when it maps a detection's box back to a camera-frame azimuth.
PINHOLE_FACES = paths_lib.PINHOLE_FACES
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
    parts = [{
        "inline_data": {
            "mime_type": mime_type,
            "data": b64_data,
        },
    } for mime_type, b64_data in images]
    parts.append({"text": USER_PROMPT})
    return batch_record(semantic_request(
        key,
        system_instruction=SYSTEM_PROMPTS[prompt_type],
        parts=parts,
        response_schema=response_schema(),
        thinking_level=thinking_level,
        media_resolution=media_resolution,
    ))


def collect_faces(pinhole_dir: Path, panorama_dir: Path) -> dict:
    """{pano_stem: [(mime_type, face_path) x4]} for the dataset's stems.

    Restricted to the stems `panorama_dir` currently contains because a pinhole
    render may outlive a dataset trim. Extra stems would create model calls no
    downstream reader can consume. A kept panorama without a render is an error.
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
