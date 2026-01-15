"""Template-based natural language sentence generator for OSM tags.

Generates varied natural language descriptions from OpenStreetMap tags,
designed for training sentence embedding models via contrastive or
predictive tasks.

Variety is achieved through:
1. Multiple structural templates per category type
2. Synonym banks for common category values
3. Probabilistic inclusion of optional attributes
4. Deterministic randomness via tag content hashing
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class GeneratedSentence:
    """Result of sentence generation with tag tracking."""

    sentence: str
    used_tags: dict[str, str]
    unused_tags: dict[str, str]
    template_type: str


@dataclass
class LandmarkRecord:
    """A landmark with its OSM tags."""

    landmark_id: int
    osm_type: str
    osm_id: int
    tags: dict[str, str]


@dataclass
class TagTemplateConfig:
    """Configuration for template-based sentence generation."""

    # Primary tag types that define the landmark category
    category_tags: tuple[str, ...] = (
        "amenity",
        "building",
        "shop",
        "tourism",
        "leisure",
        "highway",
        "landuse",
        "natural",
        "railway",
        "power",
        "man_made",
        "historic",
        "emergency",
        "office",
        "public_transport",
        "craft",
        "military",
    )

    # Tags that provide descriptive attributes
    attribute_tags: tuple[str, ...] = (
        "name",
        "brand",
        "cuisine",
        "operator",
        "height",
        "building:levels",
        "levels",
        "surface",
        "material",
        "denomination",
        "religion",
        "sport",
        "ref",              # Road/exit numbers (visible on signs)
        "addr:street",      # Street name (visible on signs)
        "addr:housenumber", # House number (visible on buildings)
        "addr:city",        # City name
    )

    # Tag prefixes to exclude (pure metadata, not visible)
    excluded_tag_prefixes: tuple[str, ...] = (
        "tiger:",      # TIGER import metadata
        "gnis:",       # GNIS import metadata
        "source:",     # Source metadata
        "brand:",      # Brand wikidata links (brand name itself is kept)
        "payment:",    # Payment methods
        "contact:",    # Contact info
        "ref:",        # Internal reference IDs (ref:walmart, ref:penndot, etc.)
    )

    # Specific tags to exclude (pure metadata, not visible/useful)
    excluded_tags: tuple[str, ...] = (
        "source",
        "created_by",
        "wikidata",
        "wikipedia",
        "website",
        "phone",
        "fax",
        "email",
        "opening_hours",
        "unsigned_ref",   # Reference without sign (not visible)
        "gtfs_id",        # Transit database ID
        "ntd_id",         # Transit database ID
        "ele",            # Elevation (not visible)
        "note",
        "fixme",
        "FIXME",
        "description",    # Usually internal notes
        "is_in",
        "import_uuid",
        "layer",          # Rendering layer (not visible)
    )


class OSMSentenceGenerator:
    """Generate varied natural language descriptions from OSM tags."""

    def __init__(self, config: TagTemplateConfig | None = None):
        self.config = config or TagTemplateConfig()
        self._init_templates()
        self._init_synonyms()

    def _init_templates(self) -> None:
        """Initialize template banks for different sentence structures."""
        # Templates for named entities (amenity/shop/building with name)
        self._named_entity_templates = [
            "{name}, {article} {category}",
            "{article} {category} called {name}",
            "{article} {category} named {name}",
            "{name} ({category})",
            "The {category} known as {name}",
        ]

        # Templates for anonymous entities (no name)
        self._anonymous_entity_templates = [
            "{article} {category}",
            "A {category}",
            "A local {category}",
            "{article} {category} here",
        ]

        # Templates for buildings with size info
        self._building_size_templates = [
            "{article} {size_descriptor} building",
            "{article} {levels}-story building",
            "A building with {levels} floors",
            "{article} {size_descriptor} {building_type}",
        ]

        # Templates for roads/highways
        self._road_templates = [
            "{article} {road_type}",
            "A {road_type}",
            "{article} {surface_type} {road_type}",
            "A road classified as {road_type}",
        ]

        # Templates for named roads
        self._named_road_templates = [
            "{name}, {article} {road_type}",
            "{article} {road_type} called {name}",
            "{name} ({road_type})",
        ]

        # Templates for natural features
        self._natural_templates = [
            "{article} {natural_type}",
            "A natural {natural_type}",
            "{article} {natural_type} feature",
            "An area of {natural_type}",
        ]

        # Templates for landuse
        self._landuse_templates = [
            "{article} {landuse_type} area",
            "An area of {landuse_type}",
            "{landuse_type} land",
            "A {landuse_type} zone",
        ]

        # Templates for railway
        self._railway_templates = [
            "{article} {railway_type}",
            "A {railway_type}",
            "A railway {railway_type}",
        ]

        # Templates for power infrastructure
        self._power_templates = [
            "{article} {power_type}",
            "A {power_type}",
            "{article} {power_type} structure",
        ]

        # Templates for man_made features
        self._man_made_templates = [
            "{article} {man_made_type}",
            "A {man_made_type}",
            "A man-made {man_made_type}",
        ]

        # Templates for historic features
        self._historic_templates = [
            "{article} historic {historic_type}",
            "A {historic_type} of historical significance",
            "{article} {historic_type}",
        ]

        # Templates for emergency features
        self._emergency_templates = [
            "{article} emergency {emergency_type}",
            "An {emergency_type}",
            "{article} {emergency_type}",
        ]

        # Templates for office features
        self._office_templates = [
            "{article} {office_type} office",
            "An office ({office_type})",
            "{article} {office_type} office building",
        ]

        # Templates for public transport
        self._public_transport_templates = [
            "{article} {transport_type}",
            "A {transport_type}",
            "A public transport {transport_type}",
        ]

        # Templates for craft
        self._craft_templates = [
            "{article} {craft_type}",
            "A {craft_type} workshop",
            "A {craft_type} business",
        ]

        # Templates for military
        self._military_templates = [
            "{article} military {military_type}",
            "A {military_type}",
            "{article} {military_type}",
        ]

        # Attribute suffix templates
        self._attribute_suffixes = {
            "cuisine": [
                " serving {cuisine}",
                " with {cuisine} cuisine",
                " specializing in {cuisine}",
            ],
            "brand": [" operated by {brand}", " ({brand})"],
            "operator": [" operated by {operator}", " run by {operator}"],
            "denomination": [" ({denomination})", ", {denomination}"],
            "religion": [" ({religion})", ", a {religion} facility"],
            "sport": [" for {sport}", " ({sport})"],
            "surface": [" with {surface} surface", " ({surface})"],
            # Address attributes (visible on signs/buildings)
            "addr:street": [" on {addr:street}", " at {addr:street}"],
            "addr:housenumber": [" at #{addr:housenumber}", " ({addr:housenumber})"],
            # Road attributes (visible)
            "ref": [" ({ref})", ", route {ref}"],
            "lanes": [" with {lanes} lanes", " ({lanes} lanes)"],
            "maxspeed": [" with {maxspeed} speed limit"],
        }

    def _init_synonyms(self) -> None:
        """Initialize synonym banks for lexical variation."""
        # Synonyms for amenity values
        self._category_synonyms = {
            # Amenities
            "restaurant": ["restaurant", "dining establishment", "eatery", "dining spot"],
            "cafe": ["cafe", "coffee shop", "coffeehouse", "coffee spot"],
            "fast_food": [
                "fast food restaurant",
                "quick-service restaurant",
                "fast food spot",
            ],
            "bar": ["bar", "pub", "tavern", "drinking establishment"],
            "bank": ["bank", "banking branch", "financial institution"],
            "pharmacy": ["pharmacy", "drugstore", "chemist"],
            "parking": ["parking lot", "parking area", "car park"],
            "school": ["school", "educational facility", "learning institution"],
            "place_of_worship": [
                "place of worship",
                "religious building",
                "house of worship",
            ],
            "hospital": ["hospital", "medical center", "healthcare facility"],
            "fuel": ["gas station", "fuel station", "petrol station"],
            "post_office": ["post office", "postal facility"],
            "library": ["library", "public library"],
            "fire_station": ["fire station", "fire department"],
            "police": ["police station", "police department"],
            "toilets": ["public restroom", "public toilets", "restroom"],
            "bench": ["bench", "public bench", "seating"],
            "shelter": ["shelter", "covered shelter"],
            "waste_basket": ["trash can", "waste bin", "garbage can"],
            "bicycle_parking": ["bike parking", "bicycle rack", "bike rack"],
            "fountain": ["fountain", "water fountain"],
            # Buildings
            "yes": ["building", "structure"],
            "house": ["house", "single-family home", "residence", "dwelling"],
            "apartments": [
                "apartment building",
                "apartment complex",
                "residential complex",
            ],
            "residential": ["residential building", "residential structure"],
            "commercial": ["commercial building", "business building", "commercial space"],
            "retail": ["retail building", "store building", "retail space"],
            "industrial": ["industrial building", "industrial facility"],
            "garage": ["garage", "car garage"],
            "garages": ["garages", "garage complex"],
            "shed": ["shed", "storage shed", "outbuilding"],
            "detached": ["detached building", "detached structure"],
            "warehouse": ["warehouse", "storage warehouse"],
            "office": ["office building", "office space"],
            "church": ["church", "church building"],
            "school": ["school building", "school facility"],
            "university": ["university building", "campus building"],
            "roof": ["covered structure", "roof structure"],
            "terrace": ["terrace", "row of buildings"],
            "static_caravan": ["mobile home", "static caravan", "manufactured home"],
            # Highways
            "service": ["service road", "access road"],
            "footway": ["footpath", "walkway", "pedestrian path"],
            "crossing": ["pedestrian crossing", "crosswalk"],
            "cycleway": ["bike path", "bicycle lane", "cycling path"],
            "path": ["path", "trail", "pathway"],
            "steps": ["steps", "stairway", "stairs"],
            "track": ["track", "unpaved road", "dirt road"],
            "primary": ["primary road", "main road", "major road"],
            "secondary": ["secondary road", "connector road"],
            "tertiary": ["tertiary road", "local road"],
            "residential": ["residential street", "neighborhood street"],
            "unclassified": ["unclassified road", "minor road"],
            "motorway": ["motorway", "highway", "freeway"],
            "motorway_link": ["motorway ramp", "highway ramp"],
            "trunk": ["trunk road", "major highway"],
            "living_street": ["living street", "shared space street"],
            "pedestrian": ["pedestrian street", "pedestrian zone"],
            "bus_stop": ["bus stop", "bus station"],
            "turning_circle": ["turning circle", "cul-de-sac"],
            "traffic_signals": ["traffic light", "traffic signal", "stoplight"],
            "stop": ["stop sign", "stop"],
            "give_way": ["yield sign", "give way sign"],
            "street_lamp": ["street light", "lamp post", "street lamp"],
            # Natural
            "tree": ["tree", "standing tree"],
            "tree_row": ["tree row", "line of trees", "row of trees"],
            "water": ["water body", "body of water"],
            "wood": ["woodland", "wooded area", "forest"],
            "wetland": ["wetland", "marsh", "swamp"],
            "scrub": ["scrubland", "shrubland", "brush"],
            "sand": ["sandy area", "sand"],
            # Landuse
            "grass": ["grassy area", "grass", "lawn"],
            "meadow": ["meadow", "grassland"],
            "farmland": ["farmland", "agricultural land", "farm"],
            "forest": ["forest", "forested area"],
            "brownfield": ["brownfield", "previously developed land"],
            # Leisure
            "park": ["park", "public park"],
            "playground": ["playground", "play area"],
            "pitch": ["sports pitch", "playing field", "athletic field"],
            "swimming_pool": ["swimming pool", "pool"],
            "garden": ["garden", "public garden"],
            "picnic_table": ["picnic table", "picnic area"],
            # Shops
            "convenience": ["convenience store", "corner store", "mini-mart"],
            "supermarket": ["supermarket", "grocery store", "food market"],
            "clothes": ["clothing store", "apparel store", "clothes shop"],
            "car_repair": ["auto repair shop", "car repair shop", "mechanic"],
            "car": ["car dealership", "auto dealer"],
            "hairdresser": ["hair salon", "barbershop", "hairdresser"],
            "beauty": ["beauty salon", "beauty shop"],
            # Tourism
            "hotel": ["hotel", "lodging", "accommodation"],
            "artwork": ["public artwork", "art installation", "sculpture"],
            "information": ["information point", "tourist information"],
            "viewpoint": ["viewpoint", "scenic overlook"],
            "museum": ["museum"],
            "attraction": ["tourist attraction", "attraction"],
            # Railway
            "rail": ["railroad track", "railway line", "train tracks"],
            "level_crossing": ["railroad crossing", "level crossing", "train crossing"],
            "station": ["train station", "railway station"],
            "platform": ["train platform", "railway platform"],
            "subway": ["subway line", "metro line"],
            "tram": ["tram line", "streetcar line"],
            # Power
            "pole": ["utility pole", "power pole", "electric pole"],
            "tower": ["power tower", "transmission tower", "electric tower"],
            "line": ["power line", "electric line", "transmission line"],
            "substation": ["electrical substation", "power substation"],
            "generator": ["power generator", "electric generator"],
            "transformer": ["transformer", "electrical transformer"],
            "busbar": ["busbar", "electrical busbar"],
            # Man-made
            "utility_pole": ["utility pole", "telephone pole", "electric pole"],
            "tower": ["tower", "structure"],
            "bridge": ["bridge", "overpass"],
            "pipeline": ["pipeline", "pipe"],
            "water_tower": ["water tower"],
            "chimney": ["chimney", "smokestack"],
            "silo": ["silo", "grain silo"],
            "storage_tank": ["storage tank", "tank"],
            # Historic
            "monument": ["monument", "memorial"],
            "memorial": ["memorial", "monument"],
            "castle": ["castle", "fortress"],
            "ruins": ["ruins", "historic ruins"],
            "archaeological_site": ["archaeological site", "historic site"],
            "building": ["historic building", "heritage building"],
            "boundary_stone": ["boundary marker", "boundary stone"],
            # Emergency
            "fire_hydrant": ["fire hydrant", "hydrant"],
            "phone": ["emergency phone", "emergency telephone"],
            "defibrillator": ["defibrillator", "AED"],
            "assembly_point": ["emergency assembly point", "evacuation point"],
            # Office
            "company": ["company office", "corporate office"],
            "government": ["government office", "public office"],
            "insurance": ["insurance office", "insurance agency"],
            "lawyer": ["law office", "attorney office"],
            "estate_agent": ["real estate office", "realty office"],
            "financial": ["financial office", "finance office"],
            # Public transport
            "stop_position": ["transit stop", "public transit stop", "stop"],
            "platform": ["transit platform", "platform"],
            "station": ["transit station", "station"],
            "stop_area": ["transit stop area", "stop area"],
            # Craft
            "signmaker": ["sign maker", "sign shop"],
            "sculptor": ["sculptor", "sculpture studio"],
            "carpenter": ["carpenter", "carpentry shop"],
            "electrician": ["electrician", "electrical contractor"],
            "plumber": ["plumber", "plumbing contractor"],
            "painter": ["painter", "painting contractor"],
            "photographer": ["photographer", "photography studio"],
            "jeweller": ["jeweler", "jewelry shop"],
            "tailor": ["tailor", "tailor shop"],
            "shoemaker": ["shoemaker", "cobbler"],
            "locksmith": ["locksmith", "locksmith shop"],
            "brewery": ["brewery", "craft brewery"],
            "winery": ["winery", "wine maker"],
            "distillery": ["distillery", "spirits maker"],
            # Military
            "barracks": ["military barracks", "barracks"],
            "bunker": ["military bunker", "bunker"],
            "checkpoint": ["military checkpoint", "checkpoint"],
            "danger_area": ["military danger area", "restricted area"],
            "naval_base": ["naval base", "navy base"],
            "airfield": ["military airfield", "air base"],
            "training_area": ["military training area", "training grounds"],
        }

    def _get_deterministic_seed(self, tags: dict[str, str], seed_offset: int = 0) -> int:
        """Generate deterministic seed from tag content for reproducible randomness."""
        tag_str = "|".join(f"{k}={v}" for k, v in sorted(tags.items()))
        hash_val = int(hashlib.sha256(tag_str.encode()).hexdigest()[:8], 16)
        return hash_val + seed_offset

    def generate_sentence(
        self,
        tags: dict[str, str],
        rng: random.Random | None = None,
        seed_offset: int = 0,
    ) -> GeneratedSentence:
        """Generate a varied natural language description from OSM tags.

        Args:
            tags: Dictionary of OSM tag key-value pairs
            rng: Random number generator to use. If None, creates one from
                 deterministic seed based on tag content.
            seed_offset: Offset to add to seed when creating RNG (ignored if rng provided)

        Returns:
            GeneratedSentence with sentence, used_tags, unused_tags, and template_type
        """
        # Filter out excluded tags (metadata, addresses, etc.)
        filtered_tags = {
            k: v for k, v in tags.items() if not self._is_excluded_tag(k)
        }

        if rng is None:
            seed = self._get_deterministic_seed(filtered_tags, seed_offset)
            rng = random.Random(seed)

        # Identify primary category
        category_key, category_value = self._extract_primary_category(filtered_tags)
        if category_key is None:
            return self._generate_generic_sentence(filtered_tags, rng)

        # Route to appropriate template generator
        if category_key == "highway":
            return self._generate_road_sentence(filtered_tags, category_value, rng)
        elif category_key == "building":
            return self._generate_building_sentence(filtered_tags, category_value, rng)
        elif category_key in ("amenity", "shop", "tourism", "leisure"):
            return self._generate_poi_sentence(filtered_tags, category_key, category_value, rng)
        elif category_key == "natural":
            return self._generate_natural_sentence(filtered_tags, category_value, rng)
        elif category_key == "landuse":
            return self._generate_landuse_sentence(filtered_tags, category_value, rng)
        elif category_key == "railway":
            return self._generate_railway_sentence(filtered_tags, category_value, rng)
        elif category_key == "power":
            return self._generate_power_sentence(filtered_tags, category_value, rng)
        elif category_key == "man_made":
            return self._generate_man_made_sentence(filtered_tags, category_value, rng)
        elif category_key == "historic":
            return self._generate_historic_sentence(filtered_tags, category_value, rng)
        elif category_key == "emergency":
            return self._generate_emergency_sentence(filtered_tags, category_value, rng)
        elif category_key == "office":
            return self._generate_office_sentence(filtered_tags, category_value, rng)
        elif category_key == "public_transport":
            return self._generate_public_transport_sentence(filtered_tags, category_value, rng)
        elif category_key == "craft":
            return self._generate_craft_sentence(filtered_tags, category_value, rng)
        elif category_key == "military":
            return self._generate_military_sentence(filtered_tags, category_value, rng)
        else:
            return self._generate_generic_sentence(filtered_tags, rng)

    def generate_sentences(
        self, tags: dict[str, str], n: int = 3
    ) -> list[GeneratedSentence]:
        """Generate multiple sentences with different tag subsets.

        Args:
            tags: Dictionary of OSM tag key-value pairs
            n: Number of sentences to generate

        Returns:
            List of GeneratedSentence objects with different tag subsets
        """
        return [self.generate_sentence(tags, seed_offset=i) for i in range(n)]

    def _extract_primary_category(
        self, tags: dict[str, str]
    ) -> tuple[str | None, str | None]:
        """Find the primary category tag (amenity, building, shop, etc.)."""
        for tag_key in self.config.category_tags:
            if tag_key in tags:
                return tag_key, tags[tag_key]
        return None, None

    def _get_category_variant(self, category: str, rng: random.Random) -> str:
        """Get a synonym variant for a category value."""
        if category in self._category_synonyms:
            return rng.choice(self._category_synonyms[category])
        # Convert underscores to spaces for unknown categories
        return category.replace("_", " ")

    def _get_article(self, word: str) -> str:
        """Get appropriate article (a/an) for a word."""
        if not word:
            return "a"
        return "an" if word[0].lower() in "aeiou" else "a"

    def _is_excluded_tag(self, tag_key: str) -> bool:
        """Check if a tag should be excluded from tracking."""
        if tag_key in self.config.excluded_tags:
            return True
        for prefix in self.config.excluded_tag_prefixes:
            if tag_key.startswith(prefix):
                return True
        return False

    def _maybe_add_suffix(
        self,
        base: str,
        tags: dict[str, str],
        rng: random.Random,
        used_tags: dict[str, str],
        unused_tags: dict[str, str],
    ) -> str:
        """Optionally append attribute suffixes based on available tags."""
        # Check each attribute that has suffix templates
        for attr, templates in self._attribute_suffixes.items():
            if attr in tags and attr not in used_tags:
                # 50% chance to include each attribute
                if rng.random() < 0.5:
                    # Use string replace instead of .format() to handle keys with colons
                    suffix = rng.choice(templates).replace(f"{{{attr}}}", tags[attr])
                    base = base + suffix
                    used_tags[attr] = tags[attr]
                    # Remove from unused if it was added earlier
                    unused_tags.pop(attr, None)
                elif attr not in unused_tags:
                    unused_tags[attr] = tags[attr]

        return base

    def _generate_poi_sentence(
        self,
        tags: dict[str, str],
        category_key: str,
        category_value: str,
        rng: random.Random,
    ) -> GeneratedSentence:
        """Generate sentence for points of interest (amenity, shop, tourism)."""
        used_tags: dict[str, str] = {category_key: category_value}
        unused_tags: dict[str, str] = {}

        name = tags.get("name")
        category = self._get_category_variant(category_value, rng)
        article = self._get_article(category)

        # Select template based on whether name exists
        if name and rng.random() < 0.8:  # 80% chance to include name if present
            template = rng.choice(self._named_entity_templates)
            base = template.format(name=name, article=article, category=category)
            used_tags["name"] = name
            template_type = "poi_named"
        else:
            if name:
                unused_tags["name"] = name
            template = rng.choice(self._anonymous_entity_templates)
            base = template.format(article=article, category=category)
            template_type = "poi_anonymous"

        # Track remaining tags
        for key in tags:
            if key not in used_tags and key not in unused_tags:
                if key in self.config.attribute_tags:
                    unused_tags[key] = tags[key]

        # Optionally add attribute suffixes
        base = self._maybe_add_suffix(base, tags, rng, used_tags, unused_tags)

        return GeneratedSentence(
            sentence=base,
            used_tags=used_tags,
            unused_tags=unused_tags,
            template_type=template_type,
        )

    def _generate_building_sentence(
        self, tags: dict[str, str], building_type: str, rng: random.Random
    ) -> GeneratedSentence:
        """Generate sentence for buildings."""
        used_tags: dict[str, str] = {"building": building_type}
        unused_tags: dict[str, str] = {}

        height = tags.get("height")
        levels = tags.get("building:levels") or tags.get("levels")
        name = tags.get("name")

        # Determine if we use size info (only if levels is a valid integer)
        used_size = False
        if levels and rng.random() < 0.7:
            try:
                lvl = int(levels)
                size_descriptor = self._get_size_descriptor(lvl, rng)
                template = rng.choice(self._building_size_templates)
                building_variant = self._get_category_variant(building_type, rng)
                article = self._get_article(size_descriptor)

                sentence = template.format(
                    article=article,
                    size_descriptor=size_descriptor,
                    levels=levels,
                    building_type=building_variant,
                )
                used_tags["building:levels"] = levels
                if "levels" in tags:
                    used_tags["levels"] = tags["levels"]
                template_type = "building_sized"
                used_size = True
            except ValueError:
                # Invalid levels value, fall back to generic
                pass

        if not used_size:
            category = self._get_category_variant(building_type, rng)
            article = self._get_article(category)
            template = rng.choice(self._anonymous_entity_templates)
            sentence = template.format(article=article, category=category)
            template_type = "building_typed"

            if levels:
                unused_tags["building:levels"] = levels
            if height:
                unused_tags["height"] = height

        # Prepend name if exists and random says so
        if name:
            if rng.random() < 0.7:
                sentence = f"{name}, {sentence[0].lower()}{sentence[1:]}"
                used_tags["name"] = name
            else:
                unused_tags["name"] = name

        # Track remaining unused attribute tags
        for key in tags:
            if key not in used_tags and key not in unused_tags:
                if key in self.config.attribute_tags:
                    unused_tags[key] = tags[key]

        return GeneratedSentence(
            sentence=sentence,
            used_tags=used_tags,
            unused_tags=unused_tags,
            template_type=template_type,
        )

    def _get_size_descriptor(self, levels: int, rng: random.Random) -> str:
        """Generate size descriptor from number of levels."""
        if levels <= 2:
            options = ["low-rise", "small", "compact", "single-story"]
        elif levels <= 5:
            options = ["mid-rise", "medium-sized", "modest"]
        elif levels <= 10:
            options = ["mid-rise", "substantial", "sizable"]
        else:
            options = ["high-rise", "tall", "towering"]
        return rng.choice(options)

    def _generate_road_sentence(
        self, tags: dict[str, str], highway_type: str, rng: random.Random
    ) -> GeneratedSentence:
        """Generate sentence for highways/roads."""
        used_tags: dict[str, str] = {"highway": highway_type}
        unused_tags: dict[str, str] = {}

        name = tags.get("name")
        surface = tags.get("surface")

        road_type = self._get_category_variant(highway_type, rng)
        article = self._get_article(road_type)

        # Check if we include surface
        surface_type = "paved"
        if surface:
            if rng.random() < 0.5:
                surface_type = surface
                used_tags["surface"] = surface
            else:
                unused_tags["surface"] = surface

        # Named vs anonymous
        if name and rng.random() < 0.7:
            template = rng.choice(self._named_road_templates)
            sentence = template.format(
                name=name,
                article=article,
                road_type=road_type,
                surface_type=surface_type,
            )
            used_tags["name"] = name
        else:
            if name:
                unused_tags["name"] = name
            template = rng.choice(self._road_templates)
            sentence = template.format(
                article=article,
                road_type=road_type,
                surface_type=surface_type,
            )

        return GeneratedSentence(
            sentence=sentence,
            used_tags=used_tags,
            unused_tags=unused_tags,
            template_type="highway",
        )

    def _generate_natural_sentence(
        self, tags: dict[str, str], natural_type: str, rng: random.Random
    ) -> GeneratedSentence:
        """Generate sentence for natural features."""
        used_tags: dict[str, str] = {"natural": natural_type}
        unused_tags: dict[str, str] = {}

        variant = self._get_category_variant(natural_type, rng)
        article = self._get_article(variant)

        template = rng.choice(self._natural_templates)
        sentence = template.format(article=article, natural_type=variant)

        # Handle name if present
        name = tags.get("name")
        if name:
            if rng.random() < 0.5:
                sentence = f"{name}, {sentence[0].lower()}{sentence[1:]}"
                used_tags["name"] = name
            else:
                unused_tags["name"] = name

        return GeneratedSentence(
            sentence=sentence,
            used_tags=used_tags,
            unused_tags=unused_tags,
            template_type="natural",
        )

    def _generate_landuse_sentence(
        self, tags: dict[str, str], landuse_type: str, rng: random.Random
    ) -> GeneratedSentence:
        """Generate sentence for landuse areas."""
        used_tags: dict[str, str] = {"landuse": landuse_type}
        unused_tags: dict[str, str] = {}

        variant = self._get_category_variant(landuse_type, rng)
        article = self._get_article(variant)

        template = rng.choice(self._landuse_templates)
        sentence = template.format(article=article, landuse_type=variant)

        # Handle name if present
        name = tags.get("name")
        if name:
            if rng.random() < 0.5:
                sentence = f"{name}, {sentence[0].lower()}{sentence[1:]}"
                used_tags["name"] = name
            else:
                unused_tags["name"] = name

        return GeneratedSentence(
            sentence=sentence,
            used_tags=used_tags,
            unused_tags=unused_tags,
            template_type="landuse",
        )

    def _generate_railway_sentence(
        self, tags: dict[str, str], railway_type: str, rng: random.Random
    ) -> GeneratedSentence:
        """Generate sentence for railway features."""
        used_tags: dict[str, str] = {"railway": railway_type}
        unused_tags: dict[str, str] = {}

        variant = self._get_category_variant(railway_type, rng)
        article = self._get_article(variant)

        template = rng.choice(self._railway_templates)
        sentence = template.format(article=article, railway_type=variant)

        # Handle name if present
        name = tags.get("name")
        if name:
            if rng.random() < 0.5:
                sentence = f"{name}, {sentence[0].lower()}{sentence[1:]}"
                used_tags["name"] = name
            else:
                unused_tags["name"] = name

        return GeneratedSentence(
            sentence=sentence,
            used_tags=used_tags,
            unused_tags=unused_tags,
            template_type="railway",
        )

    def _generate_power_sentence(
        self, tags: dict[str, str], power_type: str, rng: random.Random
    ) -> GeneratedSentence:
        """Generate sentence for power infrastructure."""
        used_tags: dict[str, str] = {"power": power_type}
        unused_tags: dict[str, str] = {}

        variant = self._get_category_variant(power_type, rng)
        article = self._get_article(variant)

        template = rng.choice(self._power_templates)
        sentence = template.format(article=article, power_type=variant)

        # Handle name if present
        name = tags.get("name")
        if name:
            if rng.random() < 0.5:
                sentence = f"{name}, {sentence[0].lower()}{sentence[1:]}"
                used_tags["name"] = name
            else:
                unused_tags["name"] = name

        return GeneratedSentence(
            sentence=sentence,
            used_tags=used_tags,
            unused_tags=unused_tags,
            template_type="power",
        )

    def _generate_man_made_sentence(
        self, tags: dict[str, str], man_made_type: str, rng: random.Random
    ) -> GeneratedSentence:
        """Generate sentence for man-made features."""
        used_tags: dict[str, str] = {"man_made": man_made_type}
        unused_tags: dict[str, str] = {}

        variant = self._get_category_variant(man_made_type, rng)
        article = self._get_article(variant)

        template = rng.choice(self._man_made_templates)
        sentence = template.format(article=article, man_made_type=variant)

        # Handle name if present
        name = tags.get("name")
        if name:
            if rng.random() < 0.5:
                sentence = f"{name}, {sentence[0].lower()}{sentence[1:]}"
                used_tags["name"] = name
            else:
                unused_tags["name"] = name

        return GeneratedSentence(
            sentence=sentence,
            used_tags=used_tags,
            unused_tags=unused_tags,
            template_type="man_made",
        )

    def _generate_historic_sentence(
        self, tags: dict[str, str], historic_type: str, rng: random.Random
    ) -> GeneratedSentence:
        """Generate sentence for historic features."""
        used_tags: dict[str, str] = {"historic": historic_type}
        unused_tags: dict[str, str] = {}

        variant = self._get_category_variant(historic_type, rng)
        article = self._get_article(variant)

        template = rng.choice(self._historic_templates)
        sentence = template.format(article=article, historic_type=variant)

        # Handle name if present
        name = tags.get("name")
        if name:
            if rng.random() < 0.6:
                sentence = f"{name}, {sentence[0].lower()}{sentence[1:]}"
                used_tags["name"] = name
            else:
                unused_tags["name"] = name

        return GeneratedSentence(
            sentence=sentence,
            used_tags=used_tags,
            unused_tags=unused_tags,
            template_type="historic",
        )

    def _generate_emergency_sentence(
        self, tags: dict[str, str], emergency_type: str, rng: random.Random
    ) -> GeneratedSentence:
        """Generate sentence for emergency features."""
        used_tags: dict[str, str] = {"emergency": emergency_type}
        unused_tags: dict[str, str] = {}

        variant = self._get_category_variant(emergency_type, rng)
        article = self._get_article(variant)

        template = rng.choice(self._emergency_templates)
        sentence = template.format(article=article, emergency_type=variant)

        return GeneratedSentence(
            sentence=sentence,
            used_tags=used_tags,
            unused_tags=unused_tags,
            template_type="emergency",
        )

    def _generate_office_sentence(
        self, tags: dict[str, str], office_type: str, rng: random.Random
    ) -> GeneratedSentence:
        """Generate sentence for office features."""
        used_tags: dict[str, str] = {"office": office_type}
        unused_tags: dict[str, str] = {}

        variant = self._get_category_variant(office_type, rng)
        article = self._get_article(variant)

        template = rng.choice(self._office_templates)
        sentence = template.format(article=article, office_type=variant)

        # Handle name if present
        name = tags.get("name")
        if name:
            if rng.random() < 0.7:
                sentence = f"{name}, {sentence[0].lower()}{sentence[1:]}"
                used_tags["name"] = name
            else:
                unused_tags["name"] = name

        return GeneratedSentence(
            sentence=sentence,
            used_tags=used_tags,
            unused_tags=unused_tags,
            template_type="office",
        )

    def _generate_public_transport_sentence(
        self, tags: dict[str, str], transport_type: str, rng: random.Random
    ) -> GeneratedSentence:
        """Generate sentence for public transport features."""
        used_tags: dict[str, str] = {"public_transport": transport_type}
        unused_tags: dict[str, str] = {}

        variant = self._get_category_variant(transport_type, rng)
        article = self._get_article(variant)

        template = rng.choice(self._public_transport_templates)
        sentence = template.format(article=article, transport_type=variant)

        # Handle name if present
        name = tags.get("name")
        if name:
            if rng.random() < 0.7:
                sentence = f"{name}, {sentence[0].lower()}{sentence[1:]}"
                used_tags["name"] = name
            else:
                unused_tags["name"] = name

        return GeneratedSentence(
            sentence=sentence,
            used_tags=used_tags,
            unused_tags=unused_tags,
            template_type="public_transport",
        )

    def _generate_craft_sentence(
        self, tags: dict[str, str], craft_type: str, rng: random.Random
    ) -> GeneratedSentence:
        """Generate sentence for craft businesses."""
        used_tags: dict[str, str] = {"craft": craft_type}
        unused_tags: dict[str, str] = {}

        variant = self._get_category_variant(craft_type, rng)
        article = self._get_article(variant)

        template = rng.choice(self._craft_templates)
        sentence = template.format(article=article, craft_type=variant)

        # Handle name if present
        name = tags.get("name")
        if name:
            if rng.random() < 0.7:
                sentence = f"{name}, {sentence[0].lower()}{sentence[1:]}"
                used_tags["name"] = name
            else:
                unused_tags["name"] = name

        return GeneratedSentence(
            sentence=sentence,
            used_tags=used_tags,
            unused_tags=unused_tags,
            template_type="craft",
        )

    def _generate_military_sentence(
        self, tags: dict[str, str], military_type: str, rng: random.Random
    ) -> GeneratedSentence:
        """Generate sentence for military features."""
        used_tags: dict[str, str] = {"military": military_type}
        unused_tags: dict[str, str] = {}

        variant = self._get_category_variant(military_type, rng)
        article = self._get_article(variant)

        template = rng.choice(self._military_templates)
        sentence = template.format(article=article, military_type=variant)

        # Handle name if present
        name = tags.get("name")
        if name:
            if rng.random() < 0.5:
                sentence = f"{name}, {sentence[0].lower()}{sentence[1:]}"
                used_tags["name"] = name
            else:
                unused_tags["name"] = name

        return GeneratedSentence(
            sentence=sentence,
            used_tags=used_tags,
            unused_tags=unused_tags,
            template_type="military",
        )

    def _generate_generic_sentence(
        self, tags: dict[str, str], rng: random.Random
    ) -> GeneratedSentence:
        """Fallback for landmarks without recognized category."""
        used_tags: dict[str, str] = {}
        unused_tags = dict(tags)

        # Try to use name if available
        if "name" in tags:
            sentence = f"A landmark called {tags['name']}"
            used_tags["name"] = tags.pop("name")
        else:
            # List first few tags
            tag_list = ", ".join(f"{k}={v}" for k, v in list(tags.items())[:3])
            if tag_list:
                sentence = f"An OSM feature ({tag_list})"
            else:
                sentence = "An unnamed OSM feature"

        return GeneratedSentence(
            sentence=sentence,
            used_tags=used_tags,
            unused_tags=unused_tags,
            template_type="generic_fallback",
        )


def iter_landmarks_from_db(
    db_path: Path, batch_size: int = 10000
) -> Iterator[list[LandmarkRecord]]:
    """Iterate over landmarks from SQLite database in batches.

    Efficiently loads landmarks with their tags using JOIN queries.

    Args:
        db_path: Path to SQLite database
        batch_size: Number of landmarks per batch

    Yields:
        Lists of LandmarkRecord objects
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get total count for progress tracking
    cursor = conn.execute("SELECT MAX(id) FROM landmarks")
    max_id = cursor.fetchone()[0] or 0

    offset = 0
    while offset <= max_id:
        # Load batch of landmarks with tags via JOIN
        # Note: representative_osm_type/id are from deduplicated schema
        query = """
            SELECT
                l.id,
                l.representative_osm_type,
                l.representative_osm_id,
                GROUP_CONCAT(tk.key || '=' || tv.value, '|') as tag_pairs
            FROM landmarks l
            LEFT JOIN tags t ON l.id = t.landmark_id
            LEFT JOIN tag_keys tk ON t.key_id = tk.id
            LEFT JOIN tag_values tv ON t.value_id = tv.id
            WHERE l.id > ? AND l.id <= ?
            GROUP BY l.id
        """
        cursor = conn.execute(query, (offset, offset + batch_size))

        batch = []
        for row in cursor:
            tags: dict[str, str] = {}
            if row["tag_pairs"]:
                for pair in row["tag_pairs"].split("|"):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        tags[k] = v

            batch.append(
                LandmarkRecord(
                    landmark_id=row["id"],
                    osm_type=row["representative_osm_type"],
                    osm_id=row["representative_osm_id"],
                    tags=tags,
                )
            )

        if batch:
            yield batch
        offset += batch_size

    conn.close()


def compute_coverage_stats(sentences: list[GeneratedSentence]) -> dict:
    """Compute breakdown of template types used.

    Args:
        sentences: List of generated sentences

    Returns:
        Dictionary with counts, percentages, and fallback_rate
    """
    counts = Counter(s.template_type for s in sentences)
    total = len(sentences)
    if total == 0:
        return {"counts": {}, "percentages": {}, "fallback_rate": 0.0}

    return {
        "counts": dict(counts),
        "percentages": {k: v / total * 100 for k, v in counts.items()},
        "fallback_rate": counts.get("generic_fallback", 0) / total * 100,
    }


def generate_sentences_for_db(
    db_path: Path,
    output_path: Path,
    config: TagTemplateConfig | None = None,
    batch_size: int = 10000,
    sentences_per_landmark: int = 1,
    limit: int | None = None,
    show_progress: bool = True,
) -> tuple[int, dict]:
    """Generate sentences for all landmarks in database.

    Writes output as JSONL with format:
    {"landmark_id": 123, "osm_type": "node", "osm_id": 456,
     "sentence": "...", "used_tags": {...}, "unused_tags": {...},
     "template_type": "..."}

    Args:
        db_path: Path to SQLite database
        output_path: Path for output JSONL file
        config: Optional template configuration
        batch_size: Landmarks per batch
        sentences_per_landmark: Number of sentences to generate per landmark
        limit: Optional limit on total landmarks to process
        show_progress: Whether to print progress updates

    Returns:
        Tuple of (sentence_count, coverage_stats)
    """
    generator = OSMSentenceGenerator(config)
    all_sentences: list[GeneratedSentence] = []
    count = 0
    landmark_count = 0

    with open(output_path, "w") as f:
        for batch in iter_landmarks_from_db(db_path, batch_size):
            for record in batch:
                if limit and landmark_count >= limit:
                    break

                sentences = generator.generate_sentences(
                    record.tags, n=sentences_per_landmark
                )
                for sentence in sentences:
                    line = {
                        "landmark_id": record.landmark_id,
                        "osm_type": record.osm_type,
                        "osm_id": record.osm_id,
                        "sentence": sentence.sentence,
                        "used_tags": sentence.used_tags,
                        "unused_tags": sentence.unused_tags,
                        "template_type": sentence.template_type,
                    }
                    f.write(json.dumps(line) + "\n")
                    all_sentences.append(sentence)
                    count += 1

                landmark_count += 1

            if limit and landmark_count >= limit:
                break

            if show_progress and landmark_count % 100000 == 0:
                print(f"Processed {landmark_count} landmarks, {count} sentences")

    stats = compute_coverage_stats(all_sentences)
    return count, stats
