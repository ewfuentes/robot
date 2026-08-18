#!/usr/bin/env python3
"""Registry of far-field water/coastal trajectories to collect from Mapillary.

Each entry is keyed by a seed image pKey taken from a Mapillary app URL. The
seed's sequence is usually only a fragment of the trip; seed_to_trajectory.py
expands it to the full capture session.

Fields:
  seed_pkey  Mapillary image id from the app URL (`pKey=`). Note the URL's
             lat/lng is the map viewport, NOT the image position -- resolve the
             pKey against the API before believing where a capture is.
  user       creator username, for cross-checking the resolved seed
  pano       True if the capture is equirectangular (360). Verified against the
             API, not assumed: only 8 of these 23 are panoramic, and Mapillary
             reports panos as either "spherical" or "equirectangular".
  osm        Geofabrik extract covering the area, relative to the Geofabrik
             root. NOTE Geofabrik renamed great-britain -> united-kingdom, and
             the old paths 301-redirect to a broken URL, so UK sub-regions are
             europe/united-kingdom/england/<county>. May be a list when a trajectory's far-field spans a national
             border (e.g. a Channel crossing sees both the English and French
             coasts); each extract becomes its own source feather and they are
             merged. National extracts are fine; prefer consistent snapshot
             dates across the list over minimising file size.
  landmark_buffer_km  Optional per-trajectory override of the landmark bbox
             buffer. Needed where one shore is far from the track (see
             folkestone_dover).
  osm_reference  Optional. A larger parent extract a sub-extract substitutes for;
             stage 4 fails if the chosen set loses coverage relative to it. Only
             needed if you deliberately pick a sub-extract.
  enc_state  NOAA ENC catalog state code, or None where NOAA has no coverage.
             NOAA charts are US-only, so UK/Japan/Morocco get OSM-only catalogs
             and are measurably thinner in fixed navaids (beacons 4%, buoys 20%,
             lights 27% present in OSM for the Boston comparison).
  note       what the user picked it for
"""

TRAJECTORIES = {
    # ── UK ────────────────────────────────────────────────────────────────────
    # DUPLICATE of folkestone_dover, and not in Anglesey at all. The name came
    # from the app URL's lat/lng (53.2273, -4.1111), which is the map *viewport
    # centre*, not the image position -- the API puts this image at
    # (51.0224, 2.1732), mid Dover Strait. Its sequence is part of the same jg360
    # crossing, so it resolves to the same 22-sequence / 33.8 km chain.
    # Checked all 22 seeds against the API afterwards; this was the only one
    # whose URL coordinates disagreed with the image.
    "anglesey_menai": {
        "seed_pkey": "2862391380743838", "user": "jg360", "pano": True,
        "osm": "europe/united-kingdom-latest.osm.pbf", "enc_state": None,
        "note": "same trip as folkestone_dover (URL viewport was misleading)",
        "duplicate_of": "folkestone_dover",
    },
    "folkestone_dover": {
        "seed_pkey": "298475668560052", "user": "jg360", "pano": True,
        # A Channel crossing sees both coasts, and each national extract holds
        # only its own side. The track starts off Dunkirk, so the French
        # coastline is the nearer far-field for the first half of it.
        # National extracts are fine now: with the tags-as-dict schema and the
        # bounded node index, whole France (4.7 GB) extracts in 1:37 at 3.0 GB,
        # versus 28 GB and climbing before. Sub-extracts were only ever an
        # OOM workaround, and mixing them invites mismatched snapshot dates --
        # france-250101 against nord-pas-de-calais-260812 differs by 19 months
        # of mapping, ~44k features on this bbox.
        # Belgium is in for the same reason at the other end: the 45 km buffer
        # reaches 2.832 E, and the Belgian coast starts at ~2.55 E, so the strip
        # from the French border past Nieuwpoort sits inside the bbox 15-40 km
        # east of the track. Without it 14.3% of the bbox's mappable land has no
        # extract behind it.
        "osm": ["europe/united-kingdom-latest.osm.pbf",
                "europe/france-latest.osm.pbf",
                "europe/belgium-latest.osm.pbf"],
        # The stitched chain covers only the Dunkirk half of the crossing, so the
        # English shore is 31-40 km west of the track's western end (Dover 31.4,
        # Folkestone 40.4). The default 25 km buffer reaches 1.4007 E and misses
        # both, leaving a catalog that is ~99.6% French with nothing on the coast
        # the ferry is steaming toward. Chalk cliffs that tall are visible across
        # the Strait, so they belong in the catalog.
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
    "sf_goldengate": {
        "rejected": "2026-08-13 triage: camera rotated vs travel, not fixed",
        "seed_pkey": "848453164774719", "user": "ohickman", "pano": False,
        "osm": "north-america/us/california/norcal-latest.osm.pbf", "enc_state": "CA",
        "note": "Golden Gate / SF harbor, foggy",
    },
    "sf_bay_pano": {
        "rejected": "2026-08-13 triage: handheld, unrecoverable",
        "seed_pkey": "790110021645719", "user": "nmixter", "pano": True,
        "osm": "north-america/us/california/norcal-latest.osm.pbf", "enc_state": "CA",
        "note": "SF bay from a different ship, 360",
    },
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
    "baltimore_a": {
        "rejected": "2026-08-13 triage: handheld, camera rotates mid-run",
        "seed_pkey": "1913812445450700", "user": "talllguy", "pano": False,
        "osm": "north-america/us/maryland-latest.osm.pbf", "enc_state": "MD",
        "note": "Baltimore harbor",
    },
    # DUPLICATE of baltimore_a: both seeds resolve to the same 668-image,
    # 2.81 km talllguy trajectory (screening 2026-08-12). Kept for provenance
    # but excluded from the collection selectors below.
    "baltimore_b": {
        "seed_pkey": "488382708878671", "user": "talllguy", "pano": False,
        "osm": "north-america/us/maryland-latest.osm.pbf", "enc_state": "MD",
        "note": "Baltimore harbor, second seed -- same trip as baltimore_a",
        "duplicate_of": "baltimore_a",
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
    "miura_sagami": {
        "rejected": "2026-08-13 triage: handheld, pointing everywhere",
        "seed_pkey": "3743584372544950", "user": "mitz", "pano": False,
        "osm": "asia/japan/kanto-latest.osm.pbf", "enc_state": None,
        "note": "Miura / Sagami bay",
    },
    "harima_a": {
        "rejected": "2026-08-13 triage: bad GPS + operator moves",
        "seed_pkey": "324280255948112", "user": "harimawood", "pano": True,
        "osm": "asia/japan/kansai-latest.osm.pbf", "enc_state": None,
        "note": "Seto inland sea, 360",
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
    "kurashiki_pano_dense": {
        "rejected": "2026-08-17 dataset review (ekf): removed",
        "seed_pkey": "817957859131973", "user": "potaro67v", "pano": True,
        # The Inland Sea is narrow: a 25 km buffer from the Honshu shore reaches
        # across to Shikoku, so chugoku alone misses 46% of the mappable area.
        "osm": ["asia/japan/chugoku-latest.osm.pbf",
                "asia/japan/shikoku-latest.osm.pbf"],
        "enc_state": None,
        "note": "dense 360 capture",
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
    "nagasaki_tometome": {
        "rejected": "2026-08-13 triage: no consistent heading",
        "seed_pkey": "1096897184165999", "user": "tometome", "pano": False,
        "osm": "asia/japan/kyushu-latest.osm.pbf", "enc_state": None,
        "note": "Nagasaki area",
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
    # noted, and QC'd by qc_candidates: all five seeds pass GPS-consistency
    # and density gates (3.2-7.3 m/frame, <=1% backtrack). ekf picked these
    # five from the Sightline Scout board.

    "great_belt_nyborg": {
        "rejected": "2026-08-17 timelapse triage (ekf)",
        "seed_pkey": "1347075487028197", "pano": True,
        # Funen shore facing the Great Belt bridge (254 m pylons) across open
        # water. QC: 3.7 m/frame, 282 img/km, 0.9% backtrack, progress 1.00.
        "osm": ["europe/denmark-latest.osm.pbf"],
        "enc_state": None,
        "note": "Great Belt, Funen coast near Nyborg, 2026 360 capture",
    },
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
        # likely crosses into Italy/Germany -- let stage 4's coverage check say.
        "osm": ["europe/austria-latest.osm.pbf"],
        "enc_state": None,
        "note": "Inn valley east of Innsbruck, 2026 360 capture",
    },
    "trail_ridge_road": {
        "rejected": "2026-08-17 timelapse triage (ekf)",
        "seed_pkey": "3281289785497201", "pano": True,
        # Viewshed score 2.38, grazing 0.47 deg. QC: 3.3 m/frame, 305 img/km.
        "osm": ["north-america/us/colorado-latest.osm.pbf"],
        "enc_state": None,
        "note": "Trail Ridge Road, Estes Park side, 2023 360 capture",
    },
    "us24_poncha_springs": {
        "rejected": "2026-08-17 timelapse triage (ekf)",
        "seed_pkey": "957814683411464", "pano": True,
        # Collegiate Peaks corridor, sightlines to 53 km. QC: 4.1 m/frame,
        # 242 img/km, 0% backtrack, progress 1.00 over the 20 km seed fragment.
        "osm": ["north-america/us/colorado-latest.osm.pbf"],
        "enc_state": None,
        "note": "US-24/US-285 Poncha Springs approach, 2026 360 capture",
    },

    # ---- 2026-08-17 batch 2, from ekf's link list. Collected into
    # datasets/unvetted/ pending timelapse triage. QC numbers below are for the
    # SEED SEQUENCE ONLY; stage 1 expands to the full session. Camera type and
    # position verified against the API (the app URL's lat/lng is the viewport).

    "mt_washington_cog": {
        # The session (2025-09-02) is 4 cameras recording the SAME 8 minutes
        # in parallel -- only the final ~1 km of the summit approach exists,
        # no ride-up. Checked the whole route's history: nobody has captured
        # the full Cog ride on Mapillary. On top of that the fragment's GPS
        # is summit multipath (median step 3.1 m, p99 49.9 m).
        "rejected": "2026-08-17 QC + resolve: only the last 1 km / 8 min of "
                    "the approach exists (x4 parallel cameras), and its GPS "
                    "oscillates (progress_p10 0.27, backtrack 7%)",
        "seed_pkey": "1265508078198228", "user": "mapillary01730", "pano": True,
        "osm": ["north-america/us/new-hampshire-latest.osm.pbf"],
        "enc_state": None,
        "note": "Mt Washington Cog summit approach",
    },
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
    "cape_ann_offshore": {
        "rejected": "2026-08-17 timelapse triage (ekf, batch 2)",
        "seed_pkey": "3039135899625408", "user": "Lanka6359", "pano": False,
        "osm": ["north-america/us/massachusetts-latest.osm.pbf"],
        "enc_state": "MA",
        "note": "boat off Cape Ann, 2026, PERSPECTIVE. QC: 5.8 km, 8.0 m, "
                "progress 0.90",
    },
    "zermatt_ski_a": {
        "rejected": "2026-08-17 timelapse triage (ekf, batch 2); sibling "
                    "zermatt_ski_b kept",
        "seed_pkey": "839647148978622", "user": "MattWithHuskies", "pano": True,
        # Italian border ~8 km south of the ski area.
        "osm": ["europe/switzerland-latest.osm.pbf",
                "europe/italy/nord-ovest-latest.osm.pbf"],
        "enc_state": None,
        "note": "Zermatt skiing 2025-12-18 morning run, 360. QC: 2.8 m, "
                "turn_p90 10 deg",
    },
    "zermatt_ski_b": {
        "seed_pkey": "1239370008117058", "user": "MattWithHuskies", "pano": True,
        # Same skier, same day as zermatt_ski_a, 4.5 h later and ~8 km north.
        # Lift rides exceed the 300 s stitch gap so the chains stay separate;
        # verify no manifest overlap after stage 1 anyway.
        "osm": ["europe/switzerland-latest.osm.pbf",
                "europe/italy/nord-ovest-latest.osm.pbf"],
        "enc_state": None,
        "note": "Zermatt skiing 2025-12-18 afternoon run, 360. QC: 2.6 m, "
                "turn_p90 43 deg (carving shows but passes)",
    },
    "zermatt_ski_c": {
        "rejected": "2026-08-17 QC: turn_p90 88 deg, progress_p10 0.49 -- "
                    "carve turns overwhelm the track",
        "seed_pkey": "740637905170074", "user": "MattWithHuskies", "pano": True,
        "osm": ["europe/switzerland-latest.osm.pbf",
                "europe/italy/nord-ovest-latest.osm.pbf"],
        "enc_state": None,
        "note": "Zermatt skiing, middle seed of the three",
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
    "gaustatoppen_summit": {
        # Stage 1 expanded to nothing: the whole session is 205 images /
        # 0.18 km of summit walking (32 frames after spacing). Checked the
        # 5 km box for alternatives: thewizard's fragment is the best 360
        # there; the only route coverage is jopparn2's 2024-07-03 perspective
        # hike (seed 1529588764293212, QC WARN: 8.5 m spacing, progress 0.73,
        # fragmented) -- collect that only if a marginal hike is wanted.
        "rejected": "2026-08-17 resolve: 0.18 km / 32 frames, summit walk "
                    "only; no ascent exists on Mapillary in 360",
        "seed_pkey": "1613946866921004", "user": "thewizard", "pano": True,
        "osm": ["europe/norway-latest.osm.pbf"],
        "enc_state": None,
        "note": "Gaustatoppen summit walk, Telemark, captured 2026-08-11",
    },
    "norway_halden": {
        "seed_pkey": "1742726803676770", "user": "thewizard", "pano": True,
        # ~10 km from the Swedish border.
        "osm": ["europe/norway-latest.osm.pbf",
                "europe/sweden-latest.osm.pbf"],
        "enc_state": None,
        "note": "Ostfold near Halden, captured 2026-08-08, 360. QC: 2.4 m",
    },
    "salt_lake_wasatch": {
        "rejected": "2026-08-17 timelapse triage (ekf, batch 2)",
        "seed_pkey": "1011036954634042", "user": "flug32", "pano": True,
        "osm": ["north-america/us/utah-latest.osm.pbf"],
        "enc_state": None,  # NOAA ENC is coastal/Great Lakes only
        "note": "Wasatch Front near Farmington, 2026 360. QC WARN: 64.8 km "
                "(longest in batch) at 13.2 m spacing, progress 0.98 -- "
                "spacing in the warn band like nyc_east_river",
    },
    "co_i70_clear_creek": {
        "rejected": "2026-08-17 timelapse triage (ekf, batch 2)",
        "seed_pkey": "1537459424671022", "user": "flug32", "pano": True,
        "osm": ["north-america/us/colorado-latest.osm.pbf"],
        "enc_state": None,
        "note": "I-70 Clear Creek corridor west of Georgetown, 2026 360. "
                "QC: 21.1 km, 4.6 m, progress 1.00",
    },
}

# The three chosen to validate the tooling end to end before the batch: one
# panoramic, one perspective, one Japanese capture.
PILOT = ["folkestone_dover", "nyc_east_river", "kurashiki_pano_dense"]


def collectable():
    """Registry entries worth collecting.

    Skips seeds that duplicate another and seeds rejected at triage — the
    rejected datasets live in archive/bad_trajectories/ (see its README for
    per-dataset reasons), and without this skip a `--trajectories all` run
    would silently re-collect them into datasets/.
    """
    return {k: v for k, v in TRAJECTORIES.items()
            if not v.get("duplicate_of") and not v.get("rejected")}


def pano_names():
    return [k for k, v in collectable().items() if v["pano"]]


def perspective_names():
    return [k for k, v in collectable().items() if not v["pano"]]
