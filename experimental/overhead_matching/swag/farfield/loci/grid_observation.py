"""LOCI observation likelihoods on the far-field localization grid.

The landmark matrix binds its source catalog and spatial lattice. This reader
validates that lineage and maps regular lattice columns onto filter cells; it
does not open or depend on satellite imagery.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import common.torch.load_torch_deps  # noqa: F401  (must precede torch)
import numpy as np
import torch

from common.gps import web_mercator
from experimental.overhead_matching.swag.data import vigor_dataset as vd
from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    SafaPlusNormalizedLandmarkAggregator,
    _load_similarity_matrix,
)
from experimental.overhead_matching.swag.filter.histogram_belief import (
    CellToPatchMapping,
    segment_max,
)


_REGION_KIND = "loci_regions"
_OSM_KIND = "loci_osm_landmarks"
_LANDMARK_KIND = "loci_correspondence_scores"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _only_upstream(manifest: artifact.ArtifactManifest, kind: str) \
        -> artifact.ArtifactRef:
    matches = [ref for ref in manifest.upstreams if ref.kind == kind]
    if len(matches) != 1:
        raise ValueError(
            f"{manifest.kind} must have exactly one {kind} upstream")
    return matches[0]


def _manifest_ref(path: Path, kind: str, required_outputs: Iterable[str] = ()) \
        -> tuple[artifact.ArtifactManifest, artifact.ArtifactRef]:
    manifest = artifact.load_manifest(path)
    if manifest.kind != kind:
        raise ValueError(
            f"artifact at {path} is {manifest.kind!r}, expected {kind!r}")
    missing = sorted(set(required_outputs) - set(manifest.declared_outputs))
    if missing:
        raise ValueError(f"{kind} manifest does not declare {missing}")
    reference = artifact.ArtifactRef(
        path=str(path.resolve()),
        kind=manifest.kind,
        dataset=manifest.dataset,
        version=manifest.version,
        manifest_digest=artifact.manifest_digest_of_document(
            manifest.to_dict()),
        content_digest=manifest.content_digest,
    )
    return manifest, reference


def _ordered_strings_sha256(values: Iterable[str], count: int) -> str:
    """Streaming twin of Vigor's ordered-string identity hash."""
    digest = hashlib.sha256(b"swag_ordered_strings/v1\0")
    digest.update(count.to_bytes(8, "big"))
    observed = 0
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        observed += 1
    if observed != count:
        raise ValueError(
            f"ordered identity expected {count} values, got {observed}")
    return digest.hexdigest()


def _pixel_to_lat_lon(x: float, y: float, zoom: int) -> tuple[float, float]:
    scale = 256 * (2 ** zoom)
    lon = x / scale * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(
        math.pi * (1.0 - 2.0 * y / scale))))
    return lat, lon


def _lattice_identity(grid: dict) \
        -> tuple[tuple[int, ...], tuple[int, ...], str]:
    if grid.get("schema") != "loci_web_mercator_grid/v1":
        raise ValueError("invalid LOCI Web-Mercator grid schema")
    zoom = grid.get("zoom")
    shape = grid.get("shape_xy")
    minimum = grid.get("min_pixel_xy")
    last = grid.get("last_center_pixel_xy")
    stride = grid.get("stride_px")
    source = grid.get("source_px")
    if type(zoom) is not int or zoom <= 0:
        raise ValueError("LOCI grid zoom must be a positive integer")
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)):
        raise ValueError("LOCI grid shape_xy must contain two positive integers")
    if (not isinstance(minimum, list) or len(minimum) != 2
            or not all(math.isfinite(float(value)) for value in minimum)):
        raise ValueError("LOCI grid min_pixel_xy must contain two finite numbers")
    if (not isinstance(last, list) or len(last) != 2
            or not all(math.isfinite(float(value)) for value in last)):
        raise ValueError(
            "LOCI grid last_center_pixel_xy must contain two finite numbers")
    if (isinstance(stride, bool) or not isinstance(stride, (int, float))
            or not math.isfinite(stride) or stride <= 0.0):
        raise ValueError("LOCI grid stride_px must be finite and positive")
    if (isinstance(source, bool) or not isinstance(source, (int, float))
            or not math.isfinite(source) or source <= 0.0):
        raise ValueError("LOCI grid source_px must be finite and positive")
    n_x, n_y = shape
    if grid.get("n_patches") != n_x * n_y:
        raise ValueError("LOCI grid n_patches disagrees with shape_xy")
    expected_last = (
        minimum[0] + (n_x - 1) * stride,
        minimum[1] + (n_y - 1) * stride,
    )
    if not all(math.isclose(float(actual), float(expected), abs_tol=1e-6)
               for actual, expected in zip(last, expected_last, strict=True)):
        raise ValueError("LOCI grid last center disagrees with its regular lattice")

    lon_by_x = []
    for x_index in range(n_x):
        _, lon = _pixel_to_lat_lon(
            minimum[0] + x_index * stride, minimum[1], zoom)
        lon_by_x.append(f"{lon:.8f}")
    lat_by_y = []
    for y_index in range(n_y):
        lat, _ = _pixel_to_lat_lon(
            minimum[0], minimum[1] + y_index * stride, zoom)
        lat_by_y.append(f"{lat:.8f}")
    if len(set(lon_by_x)) != n_x or len(set(lat_by_y)) != n_y:
        raise ValueError("formatted LOCI lattice coordinates are not unique")

    x_order = sorted(range(n_x), key=lon_by_x.__getitem__)
    y_order = sorted(range(n_y), key=lat_by_y.__getitem__)
    x_rank = [0] * n_x
    y_rank = [0] * n_y
    for rank, index in enumerate(x_order):
        x_rank[index] = rank
    for rank, index in enumerate(y_order):
        y_rank[index] = rank

    filenames = (
        f"satellite_{lat_by_y[y_index]}_{lon_by_x[x_index]}.jpg"
        for y_index in y_order
        for x_index in x_order
    )
    return (
        tuple(x_rank), tuple(y_rank),
        _ordered_strings_sha256(filenames, n_x * n_y),
    )


def _matrix_identity(sidecar: dict, path: Path) -> dict:
    identity = sidecar.get("matrix_identity")
    required = {
        "schema", "panorama_count", "panorama_ids_sha256",
        "satellite_count", "satellite_filenames_sha256",
    }
    if not isinstance(identity, dict) or set(identity) != required:
        raise ValueError(f"invalid matrix_identity in {path}")
    if identity["schema"] != vd.SIMILARITY_MATRIX_IDENTITY_SCHEMA:
        raise ValueError(f"unsupported matrix identity schema in {path}")
    if (type(identity["panorama_count"]) is not int
            or identity["panorama_count"] <= 0
            or type(identity["satellite_count"]) is not int
            or identity["satellite_count"] <= 0):
        raise ValueError(f"invalid matrix identity counts in {path}")
    for key in ("panorama_ids_sha256", "satellite_filenames_sha256"):
        value = identity[key]
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError(f"invalid {key} in {path}")
    return identity


def _panorama_id(filename: str) -> str:
    fields = Path(filename).stem.split(",")
    if len(fields) != 4 or not fields[0]:
        raise ValueError(f"invalid panorama filename {filename!r}")
    return fields[0]


def _build_regular_lattice_mapping(
        grid, cell_x_px: np.ndarray, cell_y_px: np.ndarray,
        x_rank: tuple[int, ...], y_rank: tuple[int, ...],
        device: str | torch.device) -> tuple[CellToPatchMapping, torch.Tensor]:
    """Map cells to overlapping patch columns in O(cells * overlap)."""
    n_x, n_y = grid["shape_xy"]
    min_x, min_y = grid["min_pixel_xy"]
    stride = float(grid["stride_px"])
    half = float(grid["source_px"]) / 2.0
    x = np.asarray(cell_x_px, dtype=np.float64).reshape(-1)
    y = np.asarray(cell_y_px, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("grid cell pixel coordinates must be finite and paired")

    # Strict inequalities reproduce HistogramBelief's abs(delta) < half rule.
    x_lo = np.floor((x - half - min_x) / stride).astype(np.int64) + 1
    x_hi = np.ceil((x + half - min_x) / stride).astype(np.int64) - 1
    y_lo = np.floor((y - half - min_y) / stride).astype(np.int64) + 1
    y_hi = np.ceil((y + half - min_y) / stride).astype(np.int64) - 1
    x_lo = np.maximum(x_lo, 0)
    x_hi = np.minimum(x_hi, n_x - 1)
    y_lo = np.maximum(y_lo, 0)
    y_hi = np.minimum(y_hi, n_y - 1)
    x_count = np.maximum(x_hi - x_lo + 1, 0)
    y_count = np.maximum(y_hi - y_lo + 1, 0)
    max_x = max(1, int(x_count.max(initial=0)))
    max_y = max(1, int(y_count.max(initial=0)))

    dx = np.arange(max_x, dtype=np.int64)[None, None, :]
    dy = np.arange(max_y, dtype=np.int64)[None, :, None]
    candidate_x = x_lo[:, None, None] + dx
    candidate_y = y_lo[:, None, None] + dy
    valid = ((dx < x_count[:, None, None])
             & (dy < y_count[:, None, None]))
    safe_x = np.clip(candidate_x, 0, n_x - 1)
    safe_y = np.clip(candidate_y, 0, n_y - 1)
    columns = (
        np.asarray(y_rank, dtype=np.int64)[safe_y] * n_x
        + np.asarray(x_rank, dtype=np.int64)[safe_x]
    )
    columns = np.broadcast_to(columns, valid.shape)
    counts = valid.reshape(len(x), -1).sum(axis=1, dtype=np.int64)
    patch_indices = columns[valid]
    segment_ids = np.repeat(np.arange(len(x), dtype=np.int64), counts)
    offsets = np.empty(len(x) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    mapping = CellToPatchMapping(
        patch_indices=torch.from_numpy(patch_indices).to(device),
        cell_offsets=torch.from_numpy(offsets).to(device),
        segment_ids=torch.from_numpy(segment_ids).to(device),
    )
    support = torch.from_numpy(counts > 0).to(device)
    return mapping, support



@dataclass(frozen=True)
class LociArtifacts:
    landmark_matrix_path: Path
    landmark_matrix_ref: artifact.ArtifactRef
    region_ref: artifact.ArtifactRef
    osm_ref: artifact.ArtifactRef
    panorama_ids: tuple[str, ...]
    matrix_identity: dict
    grid: dict
    x_matrix_rank: tuple[int, ...]
    y_matrix_rank: tuple[int, ...]

    @classmethod
    def load(cls, landmark_matrix_path: Path) -> "LociArtifacts":
        landmark_matrix_path = Path(landmark_matrix_path)
        sidecar_path = landmark_matrix_path.with_suffix(".json")
        for path in (landmark_matrix_path, sidecar_path):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"LOCI matrix input is not a regular file: {path}")

        matrix_manifest, matrix_ref = _manifest_ref(
            landmark_matrix_path.parent, _LANDMARK_KIND,
            (landmark_matrix_path.name, sidecar_path.name))
        region_ref = _only_upstream(matrix_manifest, _REGION_KIND)
        osm_ref = _only_upstream(matrix_manifest, _OSM_KIND)
        sidecar = _read_json(sidecar_path)
        matrix_identity = _matrix_identity(sidecar, sidecar_path)

        panorama_filenames = sidecar.get("panorama_filenames")
        if (not isinstance(panorama_filenames, list)
                or not all(isinstance(name, str)
                           for name in panorama_filenames)):
            raise ValueError(
                "landmark similarity sidecar has no panorama filenames")
        panorama_ids = tuple(_panorama_id(name)
                             for name in panorama_filenames)
        if len(panorama_ids) != len(set(panorama_ids)):
            raise ValueError(
                "landmark similarity sidecar has duplicate panorama IDs")
        if (len(panorama_ids) != matrix_identity["panorama_count"]
                or _ordered_strings_sha256(panorama_ids, len(panorama_ids))
                != matrix_identity["panorama_ids_sha256"]):
            raise ValueError(
                "panorama filenames disagree with matrix row identity")

        region_path = Path(region_ref.path)
        live_region = artifact.open_artifact(
            region_path, expected_kind=_REGION_KIND,
            expected_dataset=region_ref.dataset,
            expected_version=region_ref.version)
        if live_region != region_ref:
            raise ValueError("live LOCI region differs from matrix lineage")
        region_plan = _read_json(region_path / "region.json")
        if region_plan.get("schema") != "loci_region/v1":
            raise ValueError("invalid LOCI region document schema")
        grid = region_plan.get("grid")

        osm_manifest, live_osm = _manifest_ref(
            Path(osm_ref.path), _OSM_KIND)
        if live_osm != osm_ref:
            raise ValueError("live LOCI catalog differs from matrix lineage")
        if _only_upstream(osm_manifest, _REGION_KIND) != live_region:
            raise ValueError(
                "LOCI catalog and matrix use different regions")

        x_rank, y_rank, filename_digest = _lattice_identity(grid)
        if (grid["n_patches"] != matrix_identity["satellite_count"]
                or filename_digest
                != matrix_identity["satellite_filenames_sha256"]):
            raise ValueError(
                "LOCI lattice disagrees with matrix column identity: "
                f"region=({grid['n_patches']}, {filename_digest}), "
                f"matrix=({matrix_identity['satellite_count']}, "
                f"{matrix_identity['satellite_filenames_sha256']})")
        return cls(
            landmark_matrix_path=landmark_matrix_path,
            landmark_matrix_ref=matrix_ref,
            region_ref=live_region,
            osm_ref=live_osm,
            panorama_ids=panorama_ids,
            matrix_identity=dict(matrix_identity),
            grid=dict(grid),
            x_matrix_rank=x_rank,
            y_matrix_rank=y_rank,
        )

    def provenance(self) -> dict:
        return {
            "landmark_matrix": self.landmark_matrix_ref.to_dict(),
            "region": self.region_ref.to_dict(),
            "osm_landmarks": self.osm_ref.to_dict(),
            "matrix_identity": dict(self.matrix_identity),
            "matrix_file": str(self.landmark_matrix_path.resolve()),
            "satellite": None,
        }

    def bind(self, grid, frame, *, landmark_sigma: float,
             device: str | torch.device,
             landmark_use_raw_residual: bool = False) -> "LociGridObservation":
        east, north = grid.centers()
        cell_east, cell_north = np.meshgrid(east, north)
        lat, lon = frame.latlon_from_enu(
            cell_east.reshape(-1), cell_north.reshape(-1))
        cell_y, cell_x = web_mercator.latlon_to_pixel_coords(
            lat, lon, self.grid["zoom"])
        mapping, support = _build_regular_lattice_mapping(
            self.grid, cell_x, cell_y,
            self.x_matrix_rank, self.y_matrix_rank, device)

        landmark_matrix = _load_similarity_matrix(
            self.landmark_matrix_path, mmap=True)
        expected_shape = (
            self.matrix_identity["panorama_count"],
            self.matrix_identity["satellite_count"],
        )
        if tuple(landmark_matrix.shape) != expected_shape:
            raise ValueError(
                "LOCI matrix tensor shape disagrees with ordered identity")
        aggregator = SafaPlusNormalizedLandmarkAggregator(
            image_similarity_matrix=None,
            landmark_similarity_matrix=landmark_matrix,
            panorama_metadata={"pano_id": self.panorama_ids},
            image_sigma=1.0,
            landmark_sigma=landmark_sigma,
            landmark_use_raw_residual=landmark_use_raw_residual,
            device=torch.device(device),
        )
        return LociGridObservation(
            aggregator=aggregator,
            mapping=mapping,
            support_mask=support.reshape(grid.n_north, grid.n_east),
            n_north=grid.n_north,
            n_east=grid.n_east,
            n_patches=expected_shape[1],
        )


@dataclass(frozen=True)
class LociGridObservation:
    aggregator: SafaPlusNormalizedLandmarkAggregator
    mapping: CellToPatchMapping
    support_mask: torch.Tensor
    n_north: int
    n_east: int
    n_patches: int

    def log_likelihood(self, pano_id: str) -> torch.Tensor:
        patch_log_likelihood = self.aggregator(pano_id)
        if patch_log_likelihood.numel() != self.n_patches:
            raise ValueError("LOCI aggregator returned the wrong patch count")
        cell_log_likelihood = segment_max(
            patch_log_likelihood[self.mapping.patch_indices],
            self.mapping.cell_offsets,
            self.mapping.segment_ids,
        )
        result = cell_log_likelihood.reshape(self.n_north, self.n_east)
        return torch.where(self.support_mask, result, torch.zeros_like(result))
