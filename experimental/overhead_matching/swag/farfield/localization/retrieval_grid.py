"""CrossLocate schema-0.7 scores on the existing causal localization grid.

Raw artifacts stay read-only. Only scores.npy is extracted into an explicitly
chosen local cache; a single frame is evaluated at a time. Position lookup is
nearest-neighbor, yaw lookup circular-linear, followed by temperature softmax
and a uniform outlier floor over the declared filter support.
"""

import csv
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import common.torch.load_torch_deps  # noqa: F401
import numpy as np
from pyproj import CRS, Proj, Transformer
from scipy.spatial import cKDTree
from scipy.special import logsumexp
import torch

from experimental.overhead_matching.swag.farfield import artifact


@lru_cache(maxsize=16)
def _archive_digest(path, size, mtime_ns, ctime_ns):
    # Inputs are immutable within a batch; stat changes invalidate this cache.
    return artifact.sha256_file(Path(path))


def _scores_cache(source, cache_dir):
    source, cache_dir = source.resolve(), Path(cache_dir).resolve()
    if cache_dir.is_relative_to(source.parent):
        raise ValueError("retrieval cache must be outside the raw artifact directory")
    stat = source.stat()
    digest = _archive_digest(str(source), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / (digest + ".scores.npy")
    with zipfile.ZipFile(source) as archive:
        member = archive.getinfo("scores.npy")
        if not target.exists():
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as output:
                    temporary = Path(output.name)
                    with archive.open(member) as stream:
                        shutil.copyfileobj(stream, output, length=8 * 1024**2)
                os.replace(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        if target.stat().st_size != member.file_size:
            raise ValueError("retrieval score cache has the wrong size")
    return np.load(target, mmap_mode="r", allow_pickle=False), digest


class RetrievalGridObservation:
    def __init__(self, retrieval_dir, frames_csv, cache_dir, data, grid,
                 n_heading, *, render_crs, temperature, outlier_epsilon,
                 device):
        if type(n_heading) is not int or n_heading <= 0:
            raise ValueError("filter heading count must be a positive integer")
        if (not math.isfinite(temperature) or temperature <= 0
                or not math.isfinite(outlier_epsilon)
                or not 0 < outlier_epsilon < 1):
            raise ValueError("need finite temperature > 0 and 0 < outlier_epsilon < 1")
        crs = CRS.from_user_input(render_crs)
        if not crs.is_projected or any(a.unit_name != "metre" for a in crs.axis_info):
            raise ValueError("render CRS must be projected in metres")
        root = Path(retrieval_dir)
        self.meta = json.loads((root / "retrieval_meta.json").read_text())
        if (self.meta.get("schema_version") != "0.7"
                or self.meta.get("dataset") != data.artifact_ref.dataset):
            raise ValueError("retrieval schema or dataset differs from localization inputs")
        counts = [self.meta.get(k) for k in ("n_keyframes", "n_nodes", "n_heading_bins")]
        if any(type(v) is not int or v <= 0 for v in counts):
            raise ValueError("retrieval dimensions must be positive integers")
        k, l, h = counts
        spacing = float(self.meta["node_spacing_m"])
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("retrieval node spacing must be finite and positive")
        with np.load(root / "retrieval_fields.npz", allow_pickle=False) as fields:
            lat, lon = fields["lat_deg"], fields["lon_deg"]
            indices, ids = fields["keyframe_idx"], fields["pano_ids"]
        if (lat.shape != (l,) or lon.shape != (l,)
                or not np.isfinite(lat).all() or not np.isfinite(lon).all()
                or (np.abs(lat) > 90).any() or (np.abs(lon) > 180).any()):
            raise ValueError("invalid retrieval node coordinates")
        if (k != data.n_keyframes or indices.dtype.kind not in "iu"
                or not np.array_equal(indices, np.arange(k))
                or ids.shape != (k,) or ids.dtype.kind not in "US"):
            raise ValueError("retrieval keyframe identity differs from parent trajectory")
        self.panorama_ids = tuple(str(p) for p in ids)
        if len(set(self.panorama_ids)) != k or not all(self.panorama_ids):
            raise ValueError("retrieval panorama IDs must be nonempty and unique")
        with Path(frames_csv).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if ([int(row["idx"]) for row in rows] != list(range(k))
                or tuple(Path(row["frame_file"]).stem for row in rows)
                != self.panorama_ids):
            raise ValueError("frames CSV order differs from retrieval panorama identity")
        east, north = data.frame.enu_from_latlon(
            np.array([float(r["latitude"]) for r in rows]),
            np.array([float(r["longitude"]) for r in rows]))
        error = np.hypot(east - np.array([p.east_m for p in data.truth]),
                         north - np.array([p.north_m for p in data.truth]))
        if not np.isfinite(error).all() or (error > 1.0).any():
            raise ValueError("frames CSV positions differ from parent trajectory by >1 m")
        self._rows = {p: i for i, p in enumerate(self.panorama_ids)}

        # Check the supplied CRS against the producer's metric lattice rather
        # than silently treating projected-grid north as true north.
        x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
        for coordinate in (x, y):
            steps = (coordinate - coordinate.min()) / spacing
            if np.max(np.abs(steps - np.rint(steps))) * spacing > 0.1:
                raise ValueError("render CRS does not match the declared regular lattice")
        cell_e, cell_n = np.meshgrid(*grid.centers())
        cell_lat, cell_lon = data.frame.latlon_from_enu(cell_e.ravel(), cell_n.ravel())
        prior = data.meta.prior_region
        if prior is None:
            raise ValueError("retrieval requires the declared catalog prior")
        w, s, e, n = prior.bbox_wsen
        inside = ((cell_lon >= w) & (cell_lon <= e)
                  & (cell_lat >= s) & (cell_lat <= n))
        node_e, node_n = data.frame.enu_from_latlon(lat, lon)
        distance, nodes = cKDTree(np.column_stack([node_e, node_n])).query(
            np.column_stack([cell_e.ravel(), cell_n.ravel()]))
        supported = inside & (distance <= 0.75 * spacing)
        if not supported.any():
            raise ValueError("retrieval lattice supports no declared filter cells")
        self._nodes = nodes
        self._inside, self._supported = inside, supported
        self._shape = (n_heading, grid.n_north, grid.n_east)
        self.region_mask = torch.as_tensor(inside.reshape(self._shape[1:]), device=device)
        self.support_mask = torch.as_tensor(supported.reshape(self._shape[1:]), device=device)
        convergence = np.asarray(Proj(crs).get_factors(lon[nodes], lat[nodes]).meridian_convergence)
        mount = float(data.meta.nominal_forward["bearing_camera_cw_deg"])
        if not math.isfinite(mount) or not np.isfinite(convergence).all():
            raise ValueError("heading calibration and convergence must be finite")
        # true nominal = true camera + mount; true camera = grid camera + convergence.
        heading = (np.arange(n_heading)[:, None] * 360 / n_heading
                   - mount - convergence[None, :]) % 360
        position = heading * h / 360
        self._lo = np.floor(position).astype(np.int64) % h
        self._hi = (self._lo + 1) % h
        self._fraction = position - np.floor(position)
        self.temperature, self.epsilon, self.device = temperature, outlier_epsilon, device
        self.scores, digest = _scores_cache(root / "retrieval_fields.npz", cache_dir)
        if self.scores.shape != (k, l, h) or self.scores.dtype not in (np.float16, np.float32):
            raise ValueError("retrieval scores shape or dtype differs from metadata")
        self._provenance = {
            "fields_sha256": digest,
            "meta_sha256": artifact.sha256_file(root / "retrieval_meta.json"),
            "frames_csv_sha256": artifact.sha256_file(Path(frames_csv)),
            "scorer": self.meta["scorer"],
            "db_manifest_sha256": self.meta["db_manifest_sha256"],
            "render_crs": crs.to_string(),
            "render_crs_authority": "explicit_operator_supplied_lattice_checked",
            "heading_conversion": "true_nominal_minus_mount_minus_meridian_convergence",
            "forward_camera_cw_deg": mount,
            "node_spacing_m": spacing,
            "temperature": temperature, "outlier_epsilon": outlier_epsilon,
            "calibration_frozen": False,
            "normalization": "softmax_over_supported_filter_states_plus_uniform_catalog_floor",
            "position_lookup": "nearest_node_within_0.75_spacing",
            "heading_lookup": "circular_linear_score_interpolation",
        }

    def provenance(self):
        return dict(self._provenance)

    def log_likelihood(self, pano_id):
        scores = self.scores[self._rows[pano_id]]
        if not np.isfinite(scores).all():
            raise ValueError(f"nonfinite retrieval scores for {pano_id}")
        interpolated = ((1 - self._fraction) * scores[self._nodes, self._lo]
                        + self._fraction * scores[self._nodes, self._hi])
        supported = np.broadcast_to(self._supported, interpolated.shape)
        signal = np.full(interpolated.shape, -np.inf)
        selected = interpolated[supported]
        selected = (selected - selected.max()) / self.temperature
        signal[supported] = math.log1p(-self.epsilon) + selected - logsumexp(selected)
        floor = math.log(self.epsilon) - math.log(self._inside.sum() * self._shape[0])
        result = np.logaddexp(signal, floor)
        result[:, ~self._inside] = -np.inf
        return torch.as_tensor(result.reshape(self._shape), device=self.device)
