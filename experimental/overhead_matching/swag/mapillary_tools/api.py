import os
import threading
import time
import requests
from pathlib import Path

from experimental.overhead_matching.swag.mapillary_tools.models import BBox, PanoImage, PanoSequence

BASE_URL = "https://graph.mapillary.com"

SEARCH_FIELDS = (
    "id,computed_geometry,geometry,compass_angle,computed_compass_angle,"
    "captured_at,camera_type,height,width"
)

SEQUENCE_SEARCH_FIELDS = SEARCH_FIELDS + ",sequence"

# Adds the fields needed to convert a non-equirectangular capture: the lens
# model (camera_parameters = [focal_normalized, k1, k2]) and is_pano. Kept
# separate from SEARCH_FIELDS so the cheap bulk scans stay cheap.
FULL_SEARCH_FIELDS = SEQUENCE_SEARCH_FIELDS + ",camera_parameters,is_pano"

# Single-image lookup used to resolve a seed pKey to its sequence + creator.
IMAGE_DETAIL_FIELDS = FULL_SEARCH_FIELDS + ",creator"

ENTITY_FIELDS = "id,thumb_2048_url,thumb_original_url,width,height"


class MapillaryQueryTooLarge(RuntimeError):
    """The query exceeded a hard per-request limit and must be split.

    Mapillary reports both of these as HTTP 500 with an MLYApiException body,
    which is indistinguishable from a transient server error by status code
    alone — but retrying can never succeed:

      * "Bounding box area is too large. Maximum allowed area is 0.010 square
        degrees" — purely geometric, triggered above 0.010 sq deg.
      * "Please reduce the amount of data you're asking for" — result volume,
        triggered in dense areas even for a bbox far under the area limit, and
        NOT avoided by adding a creator_username filter.

    Callers should subdivide the bbox (see tiling.adaptive_subdivide) rather
    than back off and retry.
    """


# Substrings identifying the two permanent, split-to-fix errors above.
_TOO_LARGE_MARKERS = (
    "bounding box area is too large",
    "reduce the amount of data",
)


class RateLimiter:
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.interval = 60.0 / max_per_minute
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._last + self.interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


class MapillaryClient:
    def __init__(self, token: str = None):
        if token is None:
            token = self._read_token()
        self.token = token
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"OAuth {self.token}"
        self._search_limiter = RateLimiter(9000)
        self._entity_limiter = RateLimiter(55000)

    # The token is a secret and this file is checked in, so it is never read
    # from anywhere inside the repo: env var first (CI, one-off shells), then
    # a user config file. Both hold the raw "MLY|..." string.
    TOKEN_ENV = "MLY_TOKEN"
    TOKEN_PATH = Path.home() / ".config" / "mapillary" / "token"

    @classmethod
    def _read_token(cls) -> str:
        token = os.environ.get(cls.TOKEN_ENV, "").strip()
        if token:
            return token
        if cls.TOKEN_PATH.exists():
            for line in cls.TOKEN_PATH.read_text().splitlines():
                line = line.strip()
                if line.startswith("MLY|"):
                    return line
        raise RuntimeError(
            f"No Mapillary token: set ${cls.TOKEN_ENV} or put the MLY|... "
            f"line in {cls.TOKEN_PATH} (chmod 600)")

    def _request(self, url: str, params: dict, limiter: RateLimiter, max_retries: int = 5, timeout: int = 60) -> dict:
        for attempt in range(max_retries):
            limiter.acquire()
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
            except requests.exceptions.Timeout:
                wait = 2 ** attempt
                print(f"  Request timeout, retrying in {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503):
                # Some 500s are permanent "your query is too big" errors that no
                # amount of retrying will fix. Surface those immediately so the
                # caller can subdivide instead of sleeping through 5 backoffs.
                message = self._error_message(resp)
                if any(m in message.lower() for m in _TOO_LARGE_MARKERS):
                    raise MapillaryQueryTooLarge(message)
                wait = 2 ** attempt
                suffix = f" ({message})" if message else ""
                print(f"  HTTP {resp.status_code}{suffix}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"Max retries exceeded for {url}")

    @staticmethod
    def _error_message(resp) -> str:
        try:
            return resp.json().get("error", {}).get("message", "") or ""
        except Exception:
            return (resp.text or "")[:200]

    def search_panos(self, bbox: BBox, after_ms: int = None, before_ms: int = None) -> list[PanoImage]:
        params = {
            "fields": SEARCH_FIELDS,
            "is_pano": "true",
            "bbox": bbox.to_string(),
            "limit": 200,
        }
        if after_ms is not None:
            params["start_time"] = after_ms
        if before_ms is not None:
            params["end_time"] = before_ms

        data = self._request(f"{BASE_URL}/images", params, self._search_limiter)
        results = data.get("data", [])
        return [PanoImage.from_api(item) for item in results]

    def get_image_url(self, image_id: str, max_width: int = None) -> str:
        """Best download URL for an image.

        With max_width set, prefer the 2048 thumbnail when the original is
        wider than the cap and 2048 still satisfies it — downloading a 7680px
        original only to shrink it wastes bandwidth and disk. Falls back to the
        original whenever the thumbnail would lose detail we asked to keep.
        """
        params = {"fields": ENTITY_FIELDS}
        data = self._request(f"{BASE_URL}/{image_id}", params, self._entity_limiter)
        original = data.get("thumb_original_url") or ""
        thumb = data.get("thumb_2048_url") or ""
        if max_width is not None and thumb:
            width = data.get("width") or 0
            if width > max_width and max_width <= 2048:
                return thumb
        return original or thumb

    def get_image_detail(self, image_id: str) -> dict:
        """Full metadata for one image, including creator and lens model."""
        params = {"fields": IMAGE_DETAIL_FIELDS}
        return self._request(f"{BASE_URL}/{image_id}", params, self._entity_limiter)

    def search_images(self, bbox: BBox, creator_username: str = None,
                      is_pano: bool = None, after_ms: int = None,
                      before_ms: int = None, fields: str = None) -> list[PanoImage]:
        """Search one bbox tile, optionally scoped to a creator.

        The bbox must be within the API's 0.010 sq-deg area limit; callers are
        expected to have tiled already (see mapillary_lib.tiling). Leaving
        is_pano unset returns perspective captures too, which the pano-only
        helpers below would silently drop.
        """
        params = {
            "fields": fields or FULL_SEARCH_FIELDS,
            "bbox": bbox.to_string(),
            "limit": 200,
        }
        if creator_username:
            params["creator_username"] = creator_username
        if is_pano is not None:
            params["is_pano"] = "true" if is_pano else "false"
        if after_ms is not None:
            params["start_time"] = after_ms
        if before_ms is not None:
            params["end_time"] = before_ms
        data = self._request(f"{BASE_URL}/images", params, self._search_limiter)
        return [PanoImage.from_api(item) for item in data.get("data", [])]

    def search_panos_with_sequences(self, bbox: BBox, after_ms: int = None, before_ms: int = None) -> list[PanoImage]:
        """Like search_panos but includes sequence field."""
        params = {
            "fields": SEQUENCE_SEARCH_FIELDS,
            "is_pano": "true",
            "bbox": bbox.to_string(),
            "limit": 200,
        }
        if after_ms is not None:
            params["start_time"] = after_ms
        if before_ms is not None:
            params["end_time"] = before_ms

        data = self._request(f"{BASE_URL}/images", params, self._search_limiter)
        results = data.get("data", [])
        return [PanoImage.from_api(item) for item in results]

    def get_sequence_image_ids(self, sequence_id: str) -> list[str]:
        """GET /image_ids?sequence_id=XXX -> ordered list of image IDs."""
        params = {"sequence_id": sequence_id}
        data = self._request(f"{BASE_URL}/image_ids", params, self._search_limiter)
        return [str(x["id"]) for x in data.get("data", [])]

    def batch_get_images(self, image_ids: list[str], fields: str = None) -> list[PanoImage]:
        """Batch fetch metadata for images, in chunks of 50.
        The batch endpoint returns {id: {fields...}, ...} not {"data": [...]}.
        """
        all_images = []
        chunk_size = 50
        for i in range(0, len(image_ids), chunk_size):
            chunk = image_ids[i:i + chunk_size]
            ids_str = ",".join(chunk)
            params = {"ids": ids_str, "fields": fields or FULL_SEARCH_FIELDS}
            data = self._request(BASE_URL, params, self._search_limiter)
            for img_id in chunk:
                if img_id in data:
                    all_images.append(PanoImage.from_api(data[img_id]))
        return all_images

    def get_full_sequence(self, sequence_id: str) -> PanoSequence:
        """Fetch complete sequence: get IDs, batch-fetch metadata, compute length.

        Image order comes from get_sequence_image_ids() which returns the
        authoritative spatial sequence order. Do NOT re-sort by captured_at
        (timestamps have 1-second resolution causing ties that scramble order).
        """
        image_ids = self.get_sequence_image_ids(sequence_id)
        if not image_ids:
            return PanoSequence(id=sequence_id)
        images = self.batch_get_images(image_ids)
        # Re-order to match the authoritative sequence order from image_ids
        img_by_id = {img.id: img for img in images}
        images = [img_by_id[iid] for iid in image_ids if iid in img_by_id]
        # Ensure sequence_id is set on all images
        for img in images:
            img.sequence_id = sequence_id
        seq = PanoSequence(id=sequence_id, images=images)
        seq.compute_length()
        return seq

    def search_images_by_user(self, username: str, is_pano: bool = True,
                               after_ms: int = None, before_ms: int = None) -> list[PanoImage]:
        """Fetch all images by a username, with cursor-based pagination."""
        params = {
            "fields": SEQUENCE_SEARCH_FIELDS,
            "creator_username": username,
            "limit": 2000,
        }
        if is_pano:
            params["is_pano"] = "true"
        if after_ms is not None:
            params["start_time"] = after_ms
        if before_ms is not None:
            params["end_time"] = before_ms

        all_images = []
        url = f"{BASE_URL}/images"
        page = 0
        while True:
            page += 1
            data = self._request(url, params, self._search_limiter)
            results = data.get("data", [])
            all_images.extend(PanoImage.from_api(item) for item in results)
            print(f"  Page {page}: got {len(results)} images ({len(all_images)} total)", flush=True)

            # Cursor-based pagination
            paging = data.get("paging", {})
            cursors = paging.get("cursors", {})
            after_cursor = cursors.get("after")
            if not after_cursor or len(results) == 0:
                break
            params = dict(params)  # copy to avoid mutating
            params["after"] = after_cursor

        return all_images

    def download_image(self, url: str) -> bytes:
        self._entity_limiter.acquire()
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.content
