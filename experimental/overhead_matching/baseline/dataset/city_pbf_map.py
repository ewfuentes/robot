from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CityPbf:
    pbf_filename: str
    region_label: str  # used as the .mbtiles stem


CITY_TO_PBF: dict[str, CityPbf] = {
    "Boston":       CityPbf("massachusetts-200101.osm.pbf", "massachusetts"),
    "Chicago":      CityPbf("illinois-200101.osm.pbf",      "illinois"),
    "NewYork":      CityPbf("new-york-200101.osm.pbf",      "new_york"),
    "Seattle":      CityPbf("washington-200101.osm.pbf",    "washington"),
    "SanFrancisco": CityPbf("california-200101.osm.pbf",    "california"),
}


def cities() -> list[str]:
    return list(CITY_TO_PBF.keys())


def resolve_pbf(city: str, dumps_dir: Path) -> Path:
    if city not in CITY_TO_PBF:
        raise KeyError(f"unknown city {city!r}; known: {sorted(CITY_TO_PBF)}")
    return dumps_dir / CITY_TO_PBF[city].pbf_filename


def region_label(city: str) -> str:
    return CITY_TO_PBF[city].region_label
