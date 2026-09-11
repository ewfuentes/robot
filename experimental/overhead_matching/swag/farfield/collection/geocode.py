import requests
from experimental.overhead_matching.swag.farfield.collection.models import BBox

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def geocode_city(name: str) -> tuple[BBox, str]:
    """Geocode a city name to a bounding box via Nominatim.
    Returns (bbox, display_name).
    """
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": name, "format": "json", "limit": 1},
        headers={"User-Agent": "mapillary-pano-downloader/1.0"},
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"Could not geocode: {name}")

    r = results[0]
    bb = r["boundingbox"]  # [south, north, west, east] as strings
    bbox = BBox(
        west=float(bb[2]),
        south=float(bb[0]),
        east=float(bb[3]),
        north=float(bb[1]),
    )
    return bbox, r.get("display_name", name)
