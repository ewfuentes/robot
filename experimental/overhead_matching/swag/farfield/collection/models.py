"""Data model for Mapillary images, sequences, and bounding boxes.

`PanoSequence.compute_length` uses the shared `farfield.geometry.haversine_m`.
"""

from dataclasses import dataclass, field
from typing import Optional

from experimental.overhead_matching.swag.farfield.geometry import haversine_m


@dataclass
class BBox:
    west: float
    south: float
    east: float
    north: float

    @property
    def width(self) -> float:
        return self.east - self.west

    @property
    def height(self) -> float:
        return self.north - self.south

    def to_string(self) -> str:
        return f"{self.west},{self.south},{self.east},{self.north}"

    @classmethod
    def from_string(cls, s: str) -> "BBox":
        parts = [float(x) for x in s.split(",")]
        return cls(west=parts[0], south=parts[1], east=parts[2], north=parts[3])

    def to_dict(self) -> dict:
        return {"west": self.west, "south": self.south, "east": self.east, "north": self.north}

    @classmethod
    def from_dict(cls, d: dict) -> "BBox":
        return cls(west=d["west"], south=d["south"], east=d["east"], north=d["north"])

    def quadrants(self) -> list["BBox"]:
        mid_lng = (self.west + self.east) / 2
        mid_lat = (self.south + self.north) / 2
        return [
            BBox(self.west, self.south, mid_lng, mid_lat),
            BBox(mid_lng, self.south, self.east, mid_lat),
            BBox(self.west, mid_lat, mid_lng, self.north),
            BBox(mid_lng, mid_lat, self.east, self.north),
        ]


# camera_type values that mean "the image is already an equirectangular
# projection". Mapillary uses both spellings; verified by downloading one of
# each and confirming both are 2:1 equirectangular panoramas (not, say,
# side-by-side dual fisheye, which would also be 2:1). No reprojection is
# needed for either — they differ only in the string.
EQUIRECT_CAMERA_TYPES = ("spherical", "equirectangular")


@dataclass
class PanoImage:
    id: str
    lat: float
    lng: float
    compass_angle: float
    computed_compass_angle: float
    captured_at: int
    camera_type: str
    height: int
    width: int
    sequence_id: str = ""
    downloaded: bool = False
    # Far-field additions. camera_parameters is [focal_normalized, k1, k2] for
    # perspective/fisheye captures and is absent for equirectangular ones.
    camera_parameters: Optional[list] = None
    is_pano: Optional[bool] = None
    creator_username: str = ""
    # Which coordinate source lat/lng came from: "computed" (SfM-refined) or
    # "raw" (as-uploaded GPS). Recorded because the fallback is silent and the
    # distinction matters for boat tracks, where SfM often has no solution.
    geometry_source: str = ""

    @property
    def is_equirectangular(self) -> bool:
        if self.camera_type:
            return self.camera_type in EQUIRECT_CAMERA_TYPES
        return bool(self.is_pano)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "lat": self.lat,
            "lng": self.lng,
            "compass_angle": self.compass_angle,
            "computed_compass_angle": self.computed_compass_angle,
            "captured_at": self.captured_at,
            "camera_type": self.camera_type,
            "height": self.height,
            "width": self.width,
            "downloaded": self.downloaded,
        }
        if self.sequence_id:
            d["sequence_id"] = self.sequence_id
        if self.camera_parameters is not None:
            d["camera_parameters"] = self.camera_parameters
        if self.is_pano is not None:
            d["is_pano"] = self.is_pano
        if self.creator_username:
            d["creator_username"] = self.creator_username
        if self.geometry_source:
            d["geometry_source"] = self.geometry_source
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PanoImage":
        return cls(
            id=d["id"],
            lat=d["lat"],
            lng=d["lng"],
            compass_angle=d.get("compass_angle", 0.0),
            computed_compass_angle=d.get("computed_compass_angle", 0.0),
            captured_at=d.get("captured_at", 0),
            camera_type=d.get("camera_type", ""),
            height=d.get("height", 0),
            width=d.get("width", 0),
            sequence_id=d.get("sequence_id", ""),
            downloaded=d.get("downloaded", False),
            camera_parameters=d.get("camera_parameters"),
            is_pano=d.get("is_pano"),
            creator_username=d.get("creator_username", ""),
            geometry_source=d.get("geometry_source", ""),
        )

    @classmethod
    def from_api(cls, data: dict) -> "PanoImage":
        computed = data.get("computed_geometry")
        raw = data.get("geometry")
        chosen = computed or raw or {}
        geom = chosen.get("coordinates", [0, 0])
        creator = data.get("creator") or {}
        return cls(
            id=str(data["id"]),
            lat=geom[1],
            lng=geom[0],
            compass_angle=data.get("compass_angle", 0.0),
            computed_compass_angle=data.get("computed_compass_angle", 0.0),
            captured_at=data.get("captured_at", 0),
            camera_type=data.get("camera_type", ""),
            height=data.get("height", 0),
            width=data.get("width", 0),
            sequence_id=data.get("sequence", ""),
            downloaded=False,
            camera_parameters=data.get("camera_parameters"),
            is_pano=data.get("is_pano"),
            creator_username=creator.get("username", "") if isinstance(creator, dict) else "",
            geometry_source=("computed" if computed else "raw" if raw else ""),
        )


@dataclass
class PanoSequence:
    id: str
    images: list[PanoImage] = field(default_factory=list)
    length_km: float = 0.0

    @property
    def start_time(self) -> int:
        if not self.images:
            return 0
        return min(img.captured_at for img in self.images)

    @property
    def end_time(self) -> int:
        if not self.images:
            return 0
        return max(img.captured_at for img in self.images)

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def camera_types(self) -> list[str]:
        return sorted(set(img.camera_type for img in self.images if img.camera_type))

    @property
    def min_width(self) -> int:
        if not self.images:
            return 0
        return min(img.width for img in self.images)

    @property
    def min_height(self) -> int:
        if not self.images:
            return 0
        return min(img.height for img in self.images)

    def compute_length(self) -> float:
        if len(self.images) < 2:
            self.length_km = 0.0
            return self.length_km
        total_m = 0.0
        for i in range(len(self.images) - 1):
            a, b = self.images[i], self.images[i + 1]
            total_m += haversine_m(a.lat, a.lng, b.lat, b.lng)
        self.length_km = total_m / 1000.0
        return self.length_km

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "length_km": round(self.length_km, 3),
            "image_count": self.image_count,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "camera_types": self.camera_types,
            "min_width": self.min_width,
            "min_height": self.min_height,
            "images": [img.to_dict() for img in self.images],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PanoSequence":
        images = [PanoImage.from_dict(img) for img in d.get("images", [])]
        seq = cls(
            id=d["id"],
            images=images,
            length_km=d.get("length_km", 0.0),
        )
        return seq
