from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CityPbf:
    pbf_filename: str
    region_label: str  # used as the .mbtiles stem


CITY_TO_PBF: dict[str, CityPbf] = {
    # VIGOR original cities (under <vigor-root>/<City>/).
    "Boston":       CityPbf("massachusetts-260101.osm.pbf", "massachusetts_260101_Boston"),
    "Chicago":      CityPbf("illinois-200101.osm.pbf",      "illinois"),
    "NewYork":      CityPbf("new-york-200101.osm.pbf",      "new_york"),
    "Seattle":      CityPbf("washington-200101.osm.pbf",    "washington"),
    "SanFrancisco": CityPbf("california-200101.osm.pbf",    "california"),
    # VIGOR mapillary cities (under <vigor-root>/mapillary/<City>/). Distinct
    # mbtiles stems per city since several share a state-level PBF but cover
    # disjoint bboxes.
    "Framingham":             CityPbf("massachusetts-260101.osm.pbf", "massachusetts_260101_Framingham"),
    "Middletown":             CityPbf("connecticut-250101.osm.pbf",   "connecticut_250101_Middletown"),
    "Gap":                    CityPbf("france-250101.osm.pbf",        "france_250101_Gap"),
    "SanFrancisco_mapillary": CityPbf("norcal-220101.osm.pbf",        "norcal_220101_SanFrancisco_mapillary"),
    "MiamiBeach":             CityPbf("florida-220101.osm.pbf",       "florida_220101_MiamiBeach"),
    "post_hurricane_ian":     CityPbf("florida-220101.osm.pbf",       "florida_220101_post_hurricane_ian"),
    "post_hurricane_ian_sw":  CityPbf("florida-220101.osm.pbf",       "florida_220101_post_hurricane_ian_sw"),
    "Norway":                 CityPbf("norway-251201.osm.pbf",        "norway_251201_Norway"),
}


def cities() -> list[str]:
    return list(CITY_TO_PBF.keys())


def resolve_pbf(city: str, dumps_dirs: Path | list[Path]) -> Path:
    """Return the first existing PBF path for `city` across `dumps_dirs`.

    `dumps_dirs` accepts a single Path (legacy) or a list searched in order.
    Falls back to `<first_dir>/<filename>` if none exist, so the caller's
    error message points at a concrete path.
    """
    if city not in CITY_TO_PBF:
        raise KeyError(f"unknown city {city!r}; known: {sorted(CITY_TO_PBF)}")
    fname = CITY_TO_PBF[city].pbf_filename
    dirs = [dumps_dirs] if isinstance(dumps_dirs, Path) else list(dumps_dirs)
    if not dirs:
        raise ValueError("dumps_dirs must contain at least one path")
    for d in dirs:
        candidate = d / fname
        if candidate.exists():
            return candidate
    return dirs[0] / fname


def region_label(city: str) -> str:
    return CITY_TO_PBF[city].region_label
