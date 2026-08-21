"""The one buffered-bbox helper for collection tooling.

There were three `bbox_from_dataset`-shaped functions on the checkpoint branch,
two of them incompatible with this one:

  * `run_farfield_collection.bbox_from_dataset` — trajectory GPS from the
    converted dataset, expanded by a metric buffer with the longitude scale
    corrected by cos(mid latitude), returned as (west, south, east, north).
  * `extract_landmarks_historical.compute_bbox_from_dataset` and
    `extract_landmarks_from_enc.bbox_from_dataset_path` — read VIGOR
    *satellite* metadata / satellite_bbox.json and pad by a FRACTIONAL 10% of
    the bbox extent. Those answer "what area do the satellite tiles span",
    not "what could the trajectory see", and a fractional pad of a small bbox
    is metres where the far-field needs tens of kilometres.

The first definition won, because it is the one the collection pipeline's
consumers (Overpass/ENC extraction and the pbf coverage gate) were validated
against: the buffer is a physical sighting range, so it must be metric, and
the WSEN order matches Mapillary, the extractors' --bbox, and pbf_coverage.
The satellite-metadata variants stay with their own tools; they are not this
function. The only change from the original: degree-per-km scaling now comes
from `geometry.METERS_PER_DEG_LAT` instead of a restated 111.0.
"""

import csv
import math
from pathlib import Path

from experimental.overhead_matching.swag.farfield import geometry

KM_PER_DEG_LAT = geometry.METERS_PER_DEG_LAT / 1000.0


def padded_bbox_wsen(lats, lons, buffer_km: float):
    """(west, south, east, north) around points, padded by buffer_km.

    The longitude pad is scaled by cos(mid latitude) so the buffer is metric
    in both axes; at high latitude an unscaled pad would be far too narrow.
    """
    lats, lons = list(lats), list(lons)
    if not lats or len(lats) != len(lons):
        raise ValueError("padded_bbox_wsen needs equal, non-empty lat/lon lists")
    mid = (min(lats) + max(lats)) / 2.0
    dlat = buffer_km / KM_PER_DEG_LAT
    dlng = buffer_km / max(1e-6, KM_PER_DEG_LAT * math.cos(math.radians(mid)))
    return (min(lons) - dlng, min(lats) - dlat,
            max(lons) + dlng, max(lats) + dlat)


def bbox_from_dataset(dataset_dir: Path, buffer_km: float):
    """(west, south, east, north) around a converted dataset's trajectory.

    Reads pano_id_mapping.csv (written by mapillary_to_vigor; one row per kept
    frame) rather than raw sidecars, so the bbox describes exactly the frames
    the dataset ships.
    """
    lats, lons = [], []
    with open(Path(dataset_dir) / "pano_id_mapping.csv") as f:
        for row in csv.DictReader(f):
            lats.append(float(row["lat"]))
            lons.append(float(row["lon"]))
    return padded_bbox_wsen(lats, lons, buffer_km)
