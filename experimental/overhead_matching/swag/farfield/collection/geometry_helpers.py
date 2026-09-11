"""Shared buffered-bbox helpers for collection tooling.

The bbox is derived from trajectory GPS and expanded by a metric sighting
range, with longitude scaled by cosine of the midpoint latitude. Results use
(west, south, east, north), matching Mapillary, the extractors, and the PBF
coverage gate. Degree scaling comes from `geometry.METERS_PER_DEG_LAT`.
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
