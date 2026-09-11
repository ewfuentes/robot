#!/usr/bin/env python3
"""Registry of far-field trajectories to collect from Mapillary.

Each entry is keyed by a seed image pKey taken from a Mapillary app URL. The
seed's sequence is usually only a fragment of the trip; seed_to_trajectory.py
expands it to the full capture session.

Two dicts, one contract:

  TRAJECTORIES          entries worth collecting (nothing here is rejected or
                        a duplicate — the orchestrator iterates it directly).
  REJECTED_TRAJECTORIES entries that were tried and rejected, or that resolve
                        to the same trip as a kept entry. Kept ONLY so a newly
                        found seed can be checked against every seed we have
                        ever screened (see `known_seed_pkeys`); the collection
                        selectors never touch it. Full per-dataset reasons live
                        in archive/bad_trajectories/README on the data root.

Fields on a TRAJECTORIES entry:
  seed_pkey  Mapillary image id from the app URL (`pKey=`). Note the URL's
             lat/lng is the map viewport, NOT the image position -- resolve the
             pKey against the API before believing where a capture is.
  user       creator username, for cross-checking the resolved seed
  pano       True if the capture is equirectangular (360). Verified against the
             API, not assumed: Mapillary reports panos as either "spherical" or
             "equirectangular".
  osm        Geofabrik extract(s) covering the area, relative to the Geofabrik
             root. UK sub-regions use
             europe/united-kingdom/england/<county>. May be a list when a
             trajectory's far-field spans a national border (e.g. a Channel
             crossing sees both the English and French coasts); each extract
             becomes its own source feather and they are merged. National
             extracts are fine; prefer consistent snapshot dates across the
             list over minimising file size.
  landmark_buffer_km  Optional per-trajectory override of the landmark bbox
             buffer. Needed where one shore is far from the track (see
             folkestone_dover).
  enc_state  NOAA ENC catalog state code, or None where NOAA has no coverage.
             NOAA charts are US-only, so other regions use OSM-only catalogs.
  note       what the user picked it for
"""

TRAJECTORIES = {
    # ── UK ────────────────────────────────────────────────────────────────────
    "folkestone_dover": {
        "seed_pkey": "298475668560052", "user": "jg360", "pano": True,
        # A Channel crossing sees both coasts, and each national extract holds
        # only its own side. The track starts off Dunkirk, so the French
        # coastline is the nearer far-field for the first half of it.
        # National extracts keep source snapshots consistent across the bbox.
        # Belgium covers the far-field coast east of the track.
        "osm": ["europe/united-kingdom-latest.osm.pbf",
                "europe/france-latest.osm.pbf",
                "europe/belgium-latest.osm.pbf"],
        # The wider buffer includes the visible English shore west of the track.
        "landmark_buffer_km": 45.0,
        "enc_state": None,
        "note": "leaving big harbor, cranes; 7680x3840. TRIMMED 2026-08-17 "
                "(ekf): kept only frames 0-104 (first 7 s of timelapse); "
                "originals in trimmed_frames/. The 45 km landmark buffer and "
                "pinhole faces predate the trim",
    },
    "portsmouth_navalbase": {
        "seed_pkey": "244941107405664", "user": "southglos", "pano": False,
        "osm": "europe/united-kingdom-latest.osm.pbf", "enc_state": None,
        "note": "Portsmouth naval base, good harbor view",
    },
    "london_thames": {
        "seed_pkey": "2968055646776738", "user": "agatefilm", "pano": False,
        "osm": "europe/united-kingdom-latest.osm.pbf", "enc_state": None,
        "note": "River Thames. PARKED in datasets/unvetted/ 2026-08-17 (ekf): "
                "needs stabilization before use (same bucket as harima_b's "
                "per-frame yaw stabiliser idea)",
    },
    # ── US west ───────────────────────────────────────────────────────────────
    "seattle": {
        "seed_pkey": "210586253970126", "user": "adonis", "pano": False,
        "osm": "north-america/us/washington-latest.osm.pbf", "enc_state": "WA",
        "note": "Seattle waterfront",
    },
    # ── US east / gulf ────────────────────────────────────────────────────────
    "nyc_inner_harbor": {
        "seed_pkey": "921534178622574", "user": "cartolab", "pano": False,
        # The harbour straddles the Hudson, so New Jersey is half the far-field
        # (Bayonne, Jersey City, the Newark cranes).
        "osm": ["north-america/us/new-york-latest.osm.pbf",
                "north-america/us/new-jersey-latest.osm.pbf"],
        "enc_state": "NY",
        "note": "NYC inner harbor",
    },
    "nyc_east_river": {
        "seed_pkey": "299652321778444", "user": "daalso", "pano": False,
        # The harbour straddles the Hudson, so New Jersey is half the far-field
        # (Bayonne, Jersey City, the Newark cranes).
        "osm": ["north-america/us/new-york-latest.osm.pbf",
                "north-america/us/new-jersey-latest.osm.pbf"],
        "enc_state": "NY",
        "note": "NYC East River, denser",
    },
    "mississippi_rural": {
        "seed_pkey": "505532990600082", "user": "jasc", "pano": False,
        "osm": "north-america/us/louisiana-latest.osm.pbf", "enc_state": "LA",
        "note": "rural Mississippi river",
    },
    # ── Japan ─────────────────────────────────────────────────────────────────
    "tokyo_bay": {
        "seed_pkey": "854502538833794", "user": "komr", "pano": False,
        "osm": "asia/japan/kanto-latest.osm.pbf", "enc_state": None,
        "note": "Tokyo harbor",
    },
    "harima_b_pano": {
        "seed_pkey": "220187402940452", "user": "harimawood", "pano": True,
        # 94.8 km along the Inland Sea, so the buffered bbox spans three
        # regional extracts; kansai alone misses 32.3% (~1945 km2) of the
        # mappable area, most of it the Shikoku shore being looked at.
        "osm": ["asia/japan/kansai-latest.osm.pbf",
                "asia/japan/chugoku-latest.osm.pbf",
                "asia/japan/shikoku-latest.osm.pbf"],
        # At the default 25 km the bbox's south edge lands at 34.318, which cuts
        # through Shikoku just inland of Takamatsu: the plot's edge-hugging check
        # caught Shikoku contributing 6% of features with 52% of them on the rim,
        # i.e. clipped rather than covered. 50 km reaches 34.093, past the coastal
        # band and into the hills that are actually visible across the water.
        # All three extracts still cover 100% of the wider bbox.
        "landmark_buffer_km": 50.0,
        "enc_state": None,
        "note": "Seto inland sea, 360, longer",
    },
    "fukuyama_yasunari": {
        "seed_pkey": "954915730811637", "user": "yasunari", "pano": False,
        # Inland Sea again: 25 km from the Honshu shore reaches Shikoku, which
        # chugoku alone misses by 27%.
        "osm": ["asia/japan/chugoku-latest.osm.pbf",
                "asia/japan/shikoku-latest.osm.pbf"],
        "enc_state": None,
        "note": "Seto inland sea coast",
    },
    "fukuoka_yumechan_a": {
        "seed_pkey": "1876253562987066", "user": "yumechan", "pano": True,
        "osm": "asia/japan/kyushu-latest.osm.pbf", "enc_state": None,
        "note": "Kyushu coast, 360 (spherical)",
    },
    "kumamoto_yumechan_b": {
        "seed_pkey": "1135003251890947", "user": "yumechan", "pano": True,
        "osm": "asia/japan/kyushu-latest.osm.pbf", "enc_state": None,
        "note": "Kyushu coast, 360 (spherical), longest pano seed",
    },
    "kagoshima_matoken": {
        "seed_pkey": "299908538370198", "user": "matoken", "pano": False,
        "osm": "asia/japan/kyushu-latest.osm.pbf", "enc_state": None,
        "note": "Kagoshima bay",
    },
    # ── North Africa ──────────────────────────────────────────────────────────
    "tangier_morocco": {
        "seed_pkey": "173425018000298", "user": "sashazykov", "pano": False,
        # Across the Strait of Gibraltar: the bbox reaches 36.050 N and Tarifa
        # sits at 36.01 N, ~25 km off the Moroccan shore, so the Spanish side is
        # inside the bbox and is exactly the kind of far-field landmass these
        # datasets exist for. Morocco alone leaves 14.7% of the mappable area
        # uncovered.
        "osm": ["africa/morocco-latest.osm.pbf",
                "europe/spain/andalucia-latest.osm.pbf"],
        "enc_state": None,
        "note": "Tangier, northern Africa. TRIMMED 2026-08-17 (ekf): kept "
                "only frames 0-299 (first 20 s of timelapse); originals in "
                "trimmed_frames/",
    },

    # ---- 2026-08-17: first non-water batch, from systematic discovery ----
    # Found by discover_tracks (coverage vector tiles), viewshed-scored where
    # noted, and QC'd by qc_candidates; survivors of the timelapse triage.

    "flevoland_polder": {
        "seed_pkey": "616672887545650", "pano": True,
        # Turbines, spires and water towers at 10-25 km on a dead-flat polder.
        # QC: 3.4 m/frame, 296 img/km, 0% backtrack.
        "osm": ["europe/netherlands-latest.osm.pbf"],
        "enc_state": None,
        "note": "Flevoland central polder, 2025 360 capture",
    },
    "innsbruck_inn_valley": {
        "seed_pkey": "894556789737848", "pano": True,
        # Steep two-sided valley: short sightlines, extreme elevation angles.
        # QC: 3.3 m/frame, 298 img/km, 0% backtrack. The 25 km landmark buffer
        # likely crosses into Italy/Germany -- let the coverage check say.
        "osm": ["europe/austria-latest.osm.pbf"],
        "enc_state": None,
        "note": "Inn valley east of Innsbruck, 2026 360 capture",
    },

    # ---- 2026-08-17 batch 2, from ekf's link list. QC numbers below are for
    # the SEED SEQUENCE ONLY; stage 1 expands to the full session. Camera type
    # and position verified against the API (the app URL's lat/lng is the
    # viewport).

    "mt_washington_auto_road": {
        "seed_pkey": "1384459916191837", "user": "Lanka6359", "pano": True,
        # Summit is ~20 km from the Maine line, so the 25 km landmark buffer
        # crosses it.
        "osm": ["north-america/us/new-hampshire-latest.osm.pbf",
                "north-america/us/maine-latest.osm.pbf"],
        "enc_state": None,
        "note": "Mt Washington Auto Road, 2025 360. QC: 3.1 m, 0% backtrack. "
                "Resolved short: session has 12 min and 1 h recording gaps, "
                "so only the 2.46 km seed fragment chains (397 frames)",
    },
    "franconia_notch": {
        "seed_pkey": "1112949346609064", "user": "quickness805", "pano": True,
        # ~28 km from the Vermont line; seed image has no SfM geometry
        # (computed_geometry missing) but raw GPS resolves fine.
        "osm": ["north-america/us/new-hampshire-latest.osm.pbf",
                "north-america/us/vermont-latest.osm.pbf"],
        "enc_state": None,
        "note": "Franconia Notch drive-through, 2024 360. QC: 12.1 km, 3.7 m, "
                "progress 1.00",
    },
    "zermatt_ski_b": {
        "seed_pkey": "1239370008117058", "user": "MattWithHuskies", "pano": True,
        # Same skier, same day as the rejected zermatt_ski_a, 4.5 h later and
        # ~8 km north. Lift rides exceed the 300 s stitch gap so the chains
        # stay separate; verify no manifest overlap after stage 1 anyway.
        "osm": ["europe/switzerland-latest.osm.pbf",
                "europe/italy/nord-ovest-latest.osm.pbf"],
        "enc_state": None,
        "note": "Zermatt skiing 2025-12-18 afternoon run, 360. QC: 2.6 m, "
                "turn_p90 43 deg (carving shows but passes)",
    },
    "miami_beach": {
        "seed_pkey": "2624480597852174", "user": "microsoft", "pano": True,
        "osm": ["north-america/us/florida-latest.osm.pbf"],
        "enc_state": "FL",
        "note": "Miami Beach bridge/causeway, 2015 Microsoft streetside-era "
                "360. KEPT at triage 2026-08-17 but trimmed to the 300-frame "
                "JFK Causeway / Biscayne Bay crossing (original frames "
                "1530-1829); the rig has parallax ghosting at stitch seams "
                "on near objects, tolerable only over open water",
    },
    "friesland_workum": {
        "seed_pkey": "1172339501601224", "user": "thewizard", "pano": True,
        "osm": ["europe/netherlands-latest.osm.pbf"],
        "enc_state": None,
        "note": "Friesland flats near Workum / IJsselmeer shore, 2025 360. "
                "KEPT at triage 2026-08-17, trimmed to frames 412+ (898 m GPS "
                "jump). WORLD-LOCKED pixels: compass_angle is course-derived "
                "(MAD 0.95 deg vs course) while SfM computed_compass_angle "
                "sits near-constant 185-195 deg as the course swings -- the "
                "lock drifts ~9 deg over the run. Per-frame yaw must come "
                "from SfM once its column convention is sun-verified; "
                "unwind-to-north candidate. Do NOT run FOE mount calibration",
    },
    "norway_halden": {
        "seed_pkey": "1742726803676770", "user": "thewizard", "pano": True,
        # ~10 km from the Swedish border.
        "osm": ["europe/norway-latest.osm.pbf",
                "europe/sweden-latest.osm.pbf"],
        "enc_state": None,
        "note": "Ostfold near Halden, captured 2026-08-08, 360. QC: 2.4 m",
    },
}

# Screened and NOT collectable: rejected at triage/QC/resolve, or resolving to
# the same trip as a kept entry. One line each; kept for seed-dedup checks so a
# "new" link that resolves to one of these is recognised immediately.
REJECTED_TRAJECTORIES = {
    # duplicates ---------------------------------------------------------------
    "anglesey_menai": {
        "seed_pkey": "2862391380743838", "user": "jg360", "pano": True,
        "duplicate_of": "folkestone_dover",
        "reason": "duplicate of folkestone_dover: URL viewport (Anglesey) was "
                  "misleading; the API puts the image mid Dover Strait, same "
                  "jg360 22-sequence chain",
    },
    "baltimore_b": {
        "seed_pkey": "488382708878671", "user": "talllguy", "pano": False,
        "duplicate_of": "baltimore_a",
        "reason": "duplicate of baltimore_a: both seeds resolve to the same "
                  "668-image talllguy trajectory (screening 2026-08-12)",
    },
    # rejected -----------------------------------------------------------------
    "sf_goldengate": {
        "seed_pkey": "848453164774719", "user": "ohickman", "pano": False,
        "reason": "2026-08-13 triage: camera rotated vs travel, not fixed",
    },
    "sf_bay_pano": {
        "seed_pkey": "790110021645719", "user": "nmixter", "pano": True,
        "reason": "2026-08-13 triage: handheld, unrecoverable",
    },
    "baltimore_a": {
        "seed_pkey": "1913812445450700", "user": "talllguy", "pano": False,
        "reason": "2026-08-13 triage: handheld, camera rotates mid-run",
    },
    "miura_sagami": {
        "seed_pkey": "3743584372544950", "user": "mitz", "pano": False,
        "reason": "2026-08-13 triage: handheld, pointing everywhere",
    },
    "harima_a": {
        "seed_pkey": "324280255948112", "user": "harimawood", "pano": True,
        "reason": "2026-08-13 triage: bad GPS + operator moves",
    },
    "kurashiki_pano_dense": {
        "seed_pkey": "817957859131973", "user": "potaro67v", "pano": True,
        "reason": "2026-08-17 dataset review (ekf): removed",
    },
    "nagasaki_tometome": {
        "seed_pkey": "1096897184165999", "user": "tometome", "pano": False,
        "reason": "2026-08-13 triage: no consistent heading",
    },
    "great_belt_nyborg": {
        "seed_pkey": "1347075487028197", "pano": True,
        "reason": "2026-08-17 timelapse triage (ekf)",
    },
    "trail_ridge_road": {
        "seed_pkey": "3281289785497201", "pano": True,
        "reason": "2026-08-17 timelapse triage (ekf)",
    },
    "us24_poncha_springs": {
        "seed_pkey": "957814683411464", "pano": True,
        "reason": "2026-08-17 timelapse triage (ekf)",
    },
    "mt_washington_cog": {
        "seed_pkey": "1265508078198228", "user": "mapillary01730", "pano": True,
        "reason": "2026-08-17 QC + resolve: only the last 1 km / 8 min of the "
                  "approach exists (x4 parallel cameras), GPS is summit "
                  "multipath (progress_p10 0.27, backtrack 7%)",
    },
    "cape_ann_offshore": {
        "seed_pkey": "3039135899625408", "user": "Lanka6359", "pano": False,
        "reason": "2026-08-17 timelapse triage (ekf, batch 2)",
    },
    "zermatt_ski_a": {
        "seed_pkey": "839647148978622", "user": "MattWithHuskies", "pano": True,
        "reason": "2026-08-17 timelapse triage (ekf, batch 2); sibling "
                  "zermatt_ski_b kept",
    },
    "zermatt_ski_c": {
        "seed_pkey": "740637905170074", "user": "MattWithHuskies", "pano": True,
        "reason": "2026-08-17 QC: turn_p90 88 deg, progress_p10 0.49 -- carve "
                  "turns overwhelm the track",
    },
    "gaustatoppen_summit": {
        "seed_pkey": "1613946866921004", "user": "thewizard", "pano": True,
        "reason": "2026-08-17 resolve: 0.18 km / 32 frames, summit walk only; "
                  "no ascent exists on Mapillary in 360",
    },
    "salt_lake_wasatch": {
        "seed_pkey": "1011036954634042", "user": "flug32", "pano": True,
        "reason": "2026-08-17 timelapse triage (ekf, batch 2)",
    },
    "co_i70_clear_creek": {
        "seed_pkey": "1537459424671022", "user": "flug32", "pano": True,
        "reason": "2026-08-17 timelapse triage (ekf, batch 2)",
    },
}

# The pilot set exercises one panoramic and one perspective trajectory.
PILOT = ["folkestone_dover", "nyc_east_river"]


def collectable() -> dict:
    """Registry entries worth collecting.

    The function makes callers state that rejected trajectories are excluded.
    """
    bad = [k for k, v in TRAJECTORIES.items()
           if v.get("rejected") or v.get("duplicate_of")]
    if bad:
        raise ValueError(
            f"TRAJECTORIES contains rejected/duplicate entries {bad}; move "
            f"them to REJECTED_TRAJECTORIES with a one-line reason.")
    return dict(TRAJECTORIES)


def known_seed_pkeys() -> dict:
    """Every seed pKey ever screened -> its registry name (dedup check).

    Includes REJECTED_TRAJECTORIES on purpose: a newly found link that resolves
    to a seed here has already been screened, whatever its URL says.
    """
    out = {}
    for name, cfg in {**TRAJECTORIES, **REJECTED_TRAJECTORIES}.items():
        out[cfg["seed_pkey"]] = name
    return out


def pano_names() -> list:
    return [k for k, v in collectable().items() if v["pano"]]


def perspective_names() -> list:
    return [k for k, v in collectable().items() if not v["pano"]]
