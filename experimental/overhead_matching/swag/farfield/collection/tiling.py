import math

from experimental.overhead_matching.swag.farfield.collection.api import (
    MapillaryQueryTooLarge,
)
from experimental.overhead_matching.swag.farfield.collection.models import BBox

TILE_SIZE = 0.005
MIN_TILE_SIZE = 0.0001
API_CAP = 200

# The /images endpoint rejects any bbox whose area exceeds this, with an
# HTTP 500 whose body is an MLYApiException reading "Bounding box area is too
# large. Maximum allowed area is 0.010 square degrees". Verified empirically:
# a 0.100 x 0.100 tile is accepted, 0.110 x 0.110 is not.
MAX_BBOX_AREA_SQ_DEG = 0.010

# Tile size for sparse scans (e.g. filtered by creator_username), where the
# 200-result API cap is not the binding constraint. 0.09^2 = 0.0081 sq deg,
# just under the area limit, and ~324x fewer requests than TILE_SIZE for the
# same ground area. TILE_SIZE stays small because dense city-wide pano scans
# do hit the result cap and rely on adaptive_subdivide.
SCAN_TILE_SIZE = 0.09


def assert_legal_bbox(bbox: BBox) -> None:
    """Raise if a bbox would be rejected by the API's area limit."""
    area = abs(bbox.width * bbox.height)
    if area > MAX_BBOX_AREA_SQ_DEG:
        raise ValueError(
            f"bbox {bbox.to_string()} has area {area:.5f} sq deg, over the API "
            f"limit of {MAX_BBOX_AREA_SQ_DEG}. Tile it with generate_tiles() "
            f"(tile_size <= {MAX_BBOX_AREA_SQ_DEG ** 0.5:.3f})."
        )


def generate_tiles(bbox: BBox, tile_size: float = TILE_SIZE) -> list[BBox]:
    tiles = []
    lat = bbox.south
    while lat < bbox.north:
        lng = bbox.west
        while lng < bbox.east:
            tile = BBox(
                west=lng,
                south=lat,
                east=min(lng + tile_size, bbox.east),
                north=min(lat + tile_size, bbox.north),
            )
            tiles.append(tile)
            lng += tile_size
        lat += tile_size
    return tiles


def adaptive_subdivide(bbox: BBox, query_fn, depth: int = 0) -> list:
    """Query a tile, subdividing into quadrants whenever the tile is too big.

    Two distinct conditions force a split:
      * the response hit the API's 200-result cap, so there may be more; and
      * the request was rejected outright with MapillaryQueryTooLarge (bbox
        area over 0.010 sq deg, or too much data in a dense area). A rejected
        tile returns no results at all, so this must subdivide rather than
        treat the tile as empty — otherwise dense areas silently vanish.
    """
    too_large = False
    results = []
    try:
        results = query_fn(bbox)
    except MapillaryQueryTooLarge as e:
        too_large = True
        reason = str(e)

    if not too_large and len(results) < API_CAP:
        return results

    at_min_size = bbox.width < MIN_TILE_SIZE or bbox.height < MIN_TILE_SIZE
    if at_min_size:
        if too_large:
            # Cannot split further and cannot fetch: losing data here would be
            # invisible downstream, so fail loudly instead.
            raise RuntimeError(
                f"tile {bbox.to_string()} still rejected at minimum size "
                f"({MIN_TILE_SIZE}): {reason}"
            )
        print(f"  Warning: tile {bbox.to_string()} hit cap at min size, returning {len(results)} results")
        return results

    why = reason if too_large else f"hit {API_CAP} cap"
    print(f"  Tile {bbox.to_string()} {why}, subdividing (depth={depth+1})")
    all_results = []
    for quad in bbox.quadrants():
        all_results.extend(adaptive_subdivide(quad, query_fn, depth + 1))
    return all_results
