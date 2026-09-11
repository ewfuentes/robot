#!/usr/bin/env python3
"""Candidate regions to search for far-field trajectories.

These are places worth searching before any seed link exists. A far-field site
needs three properties at once:

  1. an empty foreground, so nothing occludes the 5-50 km band;
  2. landmarks tall enough to clear the horizon at that range and mapped in OSM;
  3. Mapillary coverage from a moving platform.

A fourth property is azimuthal spread: landmarks concentrated in one wedge
constrain position across their sightlines more strongly than along them,
whereas a surrounding basin can provide bearings around the full circle.

`geometry` records which of the categories an entry is testing, so a search that
comes back empty says something about the category rather than just the place.

Bboxes are (west, south, east, north), matching Mapillary and the collection
pipeline. Consumers that require another order perform the conversion.
"""

REGIONS = {
    # --- intermontane basins: 360 degrees of mapped peaks, dense urban coverage
    "geneva_lakeshore": {
        "bbox": (6.10, 46.30, 7.00, 46.60),
        "geometry": "alpine_lake",
        "note": "Lausanne-Montreux shore road; Alps across 13 km of water, "
                "Dents du Midi and the Chablais at 30-60 km",
    },
    "salt_lake_valley": {
        "bbox": (-112.20, 40.45, -111.75, 40.90),
        "geometry": "intermontane_basin",
        "note": "Wasatch east, Oquirrh west; peaks on both sides at 10-40 km",
    },
    "denver_front_range": {
        "bbox": (-105.30, 39.55, -104.85, 39.95),
        "geometry": "intermontane_basin",
        "note": "Front Range from the plains; one-sided, a deliberate contrast "
                "with Salt Lake's two-sided basin",
    },
    "innsbruck_inn_valley": {
        "bbox": (11.20, 47.20, 11.60, 47.35),
        "geometry": "intermontane_basin",
        "note": "steep two-sided valley, very short sightlines but extreme "
                "elevation angles -- tests the opposite regime",
    },
    "santiago_chile": {
        "bbox": (-70.80, -33.60, -70.45, -33.30),
        "geometry": "intermontane_basin",
        "note": "Andes wall east at 30-60 km; southern-hemisphere check",
    },

    # --- isolated stratovolcanoes: one node visible from 100 km, full bearing sweep
    "fuji_kanto": {
        "bbox": (138.55, 35.25, 139.30, 35.65),
        "geometry": "isolated_peak",
        "note": "Fuji from the Kanto plain; Japanese OSM extracts already staged",
    },
    "tenerife_teide": {
        "bbox": (-16.95, 28.15, -16.25, 28.60),
        "geometry": "isolated_peak",
        "note": "Teide at 3715 m dominates the whole island",
    },
    "etna_sicily": {
        "bbox": (14.85, 37.55, 15.35, 37.90),
        "geometry": "isolated_peak",
        "note": "Etna from the Catania plain and coast road",
    },
    "rainier_puget": {
        "bbox": (-122.50, 47.10, -121.85, 47.65),
        "geometry": "isolated_peak",
        "note": "Rainier from the I-5 corridor at ~90 km; the land-side "
                "counterpart to the existing `seattle` water track",
    },

    # --- flat plains: curvature-limited, landmarks are man-made and vertical
    "flevoland_polder": {
        "bbox": (5.20, 52.30, 5.95, 52.80),
        "geometry": "flat_plain",
        "note": "turbines, spires and water towers at 10-25 km; near-complete "
                "OSM turbine mapping, and Veluwe is already familiar ground",
    },
    "kansas_plains": {
        "bbox": (-98.20, 37.90, -97.40, 38.50),
        "geometry": "flat_plain",
        "note": "grain elevators and masts; tests whether a US plains catalog "
                "is dense enough without FAA DOF",
    },

    # --- bridges and causeways: ferry geometry, but a wheeled vehicle on a mapped road
    "oresund_bridge": {
        "bbox": (12.55, 55.53, 13.12, 55.78),
        "geometry": "causeway",
        "note": "Copenhagen-Malmo; open water with car-quality GPS",
    },
    "great_belt": {
        "bbox": (10.75, 55.28, 11.15, 55.42),
        "geometry": "causeway",
        "note": "18 km fixed link, 254 m pylons",
    },

    # --- supertall in flat terrain: single dominant landmark, saturated coverage
    "dubai_desert": {
        "bbox": (55.05, 24.95, 55.45, 25.35),
        "geometry": "supertall",
        "note": "Burj Khalifa at 828 m across flat desert; visible ~100 km by "
                "curvature alone, haze is the real limit",
    },

    # --- Colorado high roads: high-elevation corridors with named, isolated,
    # well-mapped peaks on both sides.
    "us24_leadville_poncha": {
        "bbox": (-106.55, 38.45, -105.90, 39.35),
        "geometry": "intermontane_basin",
        "note": "US-24/US-285 Leadville to Poncha Springs; Mt Massive massif, "
                "Mt Elbert, the Collegiate Peaks",
    },
    "co82_twinlakes_carbondale": {
        "bbox": (-107.35, 38.95, -106.25, 39.55),
        "geometry": "pass_road",
        "note": "CO-82 Twin Lakes over Independence Pass through Aspen to "
                "Carbondale; Elbert/Massive plus the Maroon Bells",
    },
    "trail_ridge_road": {
        "bbox": (-105.95, 40.15, -105.40, 40.55),
        "geometry": "pass_road",
        "note": "Trail Ridge Road, Estes Park to Grand Lake, topping 3,713 m; "
                "the highest continuous paved road in the collection",
    },

    # --- coastal: islands and headlands as landmarks, dense road coverage
    "vancouver_georgia": {
        "bbox": (-123.35, 49.15, -122.80, 49.45),
        "geometry": "coastal",
        "note": "North Shore mountains, Baker at 85 km across the strait",
    },
}


def region_names(geometry: str = None) -> list[str]:
    if geometry is None:
        return list(REGIONS)
    return [name for name, cfg in REGIONS.items() if cfg["geometry"] == geometry]


def geometries() -> list[str]:
    return sorted({cfg["geometry"] for cfg in REGIONS.values()})
