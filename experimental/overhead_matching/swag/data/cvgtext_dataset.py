"""CVG-Text dataset adapter for retrieval evaluation.

Ships panorama/text queries and satellite/OSM galleries from the CVG-Text
benchmark (ICCV 2025, yejy53/CVG-Text) in the minimal shape expected by
`retrieval_metrics.compute_top_k_metrics`. Queries and gallery entries are
paired by filename — each pano filename `LAT,LON_YYYY-MM_<panoid>_dSSS_zZ.png`
appears both under `data/query/<City>-ground/` and under `reference/<City>-<kind>/`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_FILENAME_RE = re.compile(
    r"^(?P<lat>-?[\d.]+),(?P<lon>-?[\d.]+)_"
    r"(?P<date>\d{4}-\d{2})_"
    r"(?P<pano_id>.+)_"
    r"d(?P<yaw>\d+)_"
    r"z(?P<zoom>\d+)"
    r"\.(?:png|jpg|jpeg)$"
)

CITIES = ("Brisbane", "NewYork", "Tokyo")
GALLERY_KINDS = ("satellite", "OSM")


def _parse_filename(fn: str) -> dict:
    m = _FILENAME_RE.match(fn)
    if m is None:
        raise ValueError(f"Filename does not match expected CVG-Text pattern: {fn!r}")
    g = m.groupdict()
    return {
        "pano_id": g["pano_id"],
        "lat": float(g["lat"]),
        "lon": float(g["lon"]),
        "date": g["date"],
        "yaw_deg": float(g["yaw"]),
        "zoom": int(g["zoom"]),
    }


@dataclass
class CVGTextDataset:
    """Minimal CVG-Text loader sufficient for text/image-to-gallery retrieval eval.

    Attributes:
        root: CVG-Text dataset root (contains `data/`, `reference/`, `annotation/`).
        city: one of {"Brisbane", "NewYork", "Tokyo"}.
        split: "test" (the 1000-sample pano test split).
        gallery_kind: "satellite" or "OSM" — which reference gallery to retrieve against.
        query_view: "ground" (panoramas) or "photos-ground" (single-view). We default to
            "ground" — the paper's single-view variant is out of scope here.
    """

    root: Path
    city: str
    split: str = "test"
    gallery_kind: str = "satellite"
    query_view: str = "ground"

    def __post_init__(self):
        self.root = Path(self.root)
        if self.city not in CITIES:
            raise ValueError(f"city must be one of {CITIES}, got {self.city!r}")
        if self.gallery_kind not in GALLERY_KINDS:
            raise ValueError(
                f"gallery_kind must be one of {GALLERY_KINDS}, got {self.gallery_kind!r}"
            )
        if self.query_view not in ("ground", "photos-ground"):
            raise ValueError(f"query_view must be ground|photos-ground, got {self.query_view!r}")

        city_suffix = "" if self.query_view == "ground" else "-photos"
        annotation_path = self.root / "annotation" / f"{self.city}{city_suffix}" / f"{self.split}.json"
        query_dir = self.root / "data" / "query" / f"{self.city}{city_suffix}-ground"
        gallery_dir = self.root / "reference" / f"{self.city}{city_suffix}-{self.gallery_kind}"

        with open(annotation_path) as f:
            self._annotations: dict[str, str] = json.load(f)

        # Build gallery metadata by enumerating the reference directory.
        gallery_rows = []
        for path in sorted(gallery_dir.iterdir()):
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            parsed = _parse_filename(path.name)
            gallery_rows.append({
                "filename": path.name,
                "path": str(path),
                **parsed,
            })
        self._satellite_metadata = pd.DataFrame(gallery_rows).reset_index(drop=True)

        filename_to_gallery_idx = {
            fn: idx for idx, fn in enumerate(self._satellite_metadata["filename"])
        }

        # Build query (panorama) metadata and pair each to its matching gallery row.
        query_rows = []
        missing = []
        for fn, text in self._annotations.items():
            if fn not in filename_to_gallery_idx:
                missing.append(fn)
                continue
            parsed = _parse_filename(fn)
            query_rows.append({
                "filename": fn,
                "path": str(query_dir / fn),
                "text": text,
                "positive_satellite_idxs": [filename_to_gallery_idx[fn]],
                "semipositive_satellite_idxs": [],
                **parsed,
            })
        if missing:
            raise RuntimeError(
                f"{len(missing)} query filenames missing from gallery {gallery_dir}. "
                f"First 3: {missing[:3]}"
            )
        self._panorama_metadata = pd.DataFrame(query_rows).reset_index(drop=True)

    @property
    def num_queries(self) -> int:
        return len(self._panorama_metadata)

    @property
    def num_gallery(self) -> int:
        return len(self._satellite_metadata)

    @property
    def texts(self) -> list[str]:
        return self._panorama_metadata["text"].tolist()

    @property
    def query_image_paths(self) -> list[str]:
        return self._panorama_metadata["path"].tolist()

    @property
    def gallery_image_paths(self) -> list[str]:
        return self._satellite_metadata["path"].tolist()
