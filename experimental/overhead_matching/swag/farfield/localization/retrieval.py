"""Retrieval score fields as a filter observation source (CLD-3, plan §5.5).

A retrieval baseline (CrossLocate-Depth) scores a whole candidate lattice
jointly in location and heading once per keyframe. This module turns those
dense score fields into the observation likelihood the particle filter
multiplies into its belief — the retrieval counterpart of the bearing
measurement path, sharing the motion model, resampling, mode tracking, and
metrics unchanged.

Likelihood (plan §5.5): with S the raw score field of one keyframe over the
discrete support X (lattice nodes x heading bins),

    K(x) = exp((S(x) - max S) / temperature)
    L(x) = (1 - epsilon) * K(x) / sum(K) + epsilon / |X|

The epsilon floor is what lets one bad retrieval NOT irreversibly delete the
true hypothesis: every pose keeps at least uniform mass. temperature and
epsilon are calibration parameters frozen on validation regions; until a
validation region is declared they are provisional and the run must say so
(`RetrievalConfig.calibration_frozen`).

Particle lookup: nearest lattice node in position (the lattice spacing is a
DECLARED quantization floor, reported with results per §5.5), linear circular
interpolation across the two adjacent heading bins. Particles outside the
lattice support receive only the epsilon floor.

Artifact contract (produced by the retrieval baseline's scoring stage):
  retrieval_meta.json    RetrievalFieldsMeta — provenance, calibration-free
  retrieval_fields.npz   lat_deg (L,), lon_deg (L,), scores (K, L, N) f16,
                         keyframe_idx (K,), pano_ids (K,) str
Node positions are lat/lon so the artifact is anchor-free; they are converted
into the run's ENU frame at load time.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import msgspec
import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.localization import structs


class RetrievalFieldsMeta(msgspec.Struct):
    """Sidecar for retrieval_fields.npz: what scored what, against what."""
    schema_version: str
    dataset: str
    n_keyframes: int
    n_nodes: int
    n_heading_bins: int
    node_spacing_m: float
    db_dir: str
    db_manifest_sha256: str
    scorer: str  # e.g. "dem_baseline.crosslocate_net@<weights sha>"


@dataclass
class ScoreFields:
    """Decoded fields in the run's ENU frame."""
    meta: RetrievalFieldsMeta
    east_m: np.ndarray  # (L,) node east
    north_m: np.ndarray  # (L,) node north
    scores: np.ndarray  # (K, L, N) float32
    keyframe_idx: np.ndarray  # (K,) int
    pano_ids: list


def load_fields(retrieval_dir: Path,
                frame: geo.RegionFrame) -> ScoreFields:
    retrieval_dir = Path(retrieval_dir)
    meta = msgspec.json.decode(
        (retrieval_dir / "retrieval_meta.json").read_bytes(),
        type=RetrievalFieldsMeta)
    data = np.load(retrieval_dir / "retrieval_fields.npz")
    east_m, north_m = frame.enu_from_latlon(data["lat_deg"],
                                            data["lon_deg"])
    scores = data["scores"].astype(np.float32)
    if scores.shape != (len(data["keyframe_idx"]), len(east_m),
                        meta.n_heading_bins):
        raise ValueError(
            f"retrieval fields shape {scores.shape} disagrees with meta "
            f"({len(data['keyframe_idx'])} keyframes, {len(east_m)} nodes, "
            f"{meta.n_heading_bins} heading bins)")
    return ScoreFields(
        meta=meta,
        east_m=np.asarray(east_m, dtype=np.float64),
        north_m=np.asarray(north_m, dtype=np.float64),
        scores=scores,
        keyframe_idx=data["keyframe_idx"].astype(np.int64),
        pano_ids=[str(p) for p in data["pano_ids"]])


def measurements_from_fields(fields: ScoreFields) -> list:
    """One RetrievalMeasurement per scored keyframe, in keyframe order."""
    order = np.argsort(fields.keyframe_idx)
    return [
        structs.RetrievalMeasurement(
            keyframe_idx=int(fields.keyframe_idx[i]),
            field_idx=int(i),
            pano_id=fields.pano_ids[i])
        for i in order]


class RetrievalEngine:
    """Per-particle log-likelihood evaluation over loaded score fields.

    Nearest-node position lookup uses a uniform-bin hash built from the node
    coordinates; nodes need not form a complete rectangle (water masks and
    no-data drop cells), only share one spacing.
    """

    def __init__(self, fields: ScoreFields,
                 config: structs.RetrievalConfig):
        if not 0.0 < config.outlier_epsilon < 1.0:
            raise ValueError(f"outlier_epsilon must be in (0, 1), got "
                             f"{config.outlier_epsilon}")
        if config.temperature <= 0.0:
            raise ValueError(
                f"temperature must be positive, got {config.temperature}")
        self.fields = fields
        self.config = config
        spacing = fields.meta.node_spacing_m
        if spacing <= 0.0:
            raise ValueError("node_spacing_m must be positive")
        self._spacing = spacing
        self._east0 = float(fields.east_m.min())
        self._north0 = float(fields.north_m.min())
        cols = np.rint((fields.east_m - self._east0) / spacing).astype(int)
        rows = np.rint((fields.north_m - self._north0) / spacing).astype(int)
        snapped_e = self._east0 + cols * spacing
        snapped_n = self._north0 + rows * spacing
        offset = np.hypot(fields.east_m - snapped_e,
                          fields.north_m - snapped_n)
        # The equirectangular ENU conversion of a metric-CRS lattice bends a
        # perfect grid slightly; tolerate that, refuse an irregular lattice.
        if offset.max() > 0.25 * spacing:
            raise ValueError(
                f"lattice nodes deviate up to {offset.max():.1f} m from a "
                f"regular {spacing:.0f} m grid; nearest-node lookup would "
                "be wrong")
        self._n_rows = int(rows.max()) + 1
        self._n_cols = int(cols.max()) + 1
        self._node_of_cell = np.full(self._n_rows * self._n_cols, -1,
                                     dtype=np.int64)
        self._node_of_cell[rows * self._n_cols + cols] = np.arange(
            len(fields.east_m))

        n_bins = fields.meta.n_heading_bins
        self._heading_spacing_rad = 2.0 * math.pi / n_bins
        # Per-field normalizers over the whole discrete support (log space).
        flat = fields.scores.reshape(fields.scores.shape[0], -1)
        self._score_max = flat.max(axis=1)  # (K,)
        shifted = (flat - self._score_max[:, None]) / config.temperature
        self._log_norm = np.log(
            np.exp(shifted).sum(axis=1))  # (K,) log sum K(x)
        n_cells = flat.shape[1]
        self._log_floor = math.log(config.outlier_epsilon) - math.log(n_cells)
        self._log_signal_coeff = math.log1p(-config.outlier_epsilon)

    @property
    def quantization_floor_m(self) -> float:
        """The declared position quantization of this observation source."""
        return self._spacing

    def _node_indices(self, east_m, north_m):
        """Nearest-node index per particle; -1 outside the support."""
        cols = np.rint((np.asarray(east_m) - self._east0)
                       / self._spacing).astype(int)
        rows = np.rint((np.asarray(north_m) - self._north0)
                       / self._spacing).astype(int)
        inside = ((cols >= 0) & (cols < self._n_cols)
                  & (rows >= 0) & (rows < self._n_rows))
        cells = np.where(inside, rows * self._n_cols + cols, 0)
        nodes = np.where(inside, self._node_of_cell[cells], -1)
        return nodes

    def log_likelihood(self, field_idx: int, east_m, north_m,
                       heading_rad) -> np.ndarray:
        """log L(x) of plan §5.5 at arbitrary poses, vectorized."""
        scores = self.fields.scores[field_idx]  # (L, N)
        nodes = self._node_indices(east_m, north_m)
        safe_nodes = np.maximum(nodes, 0)

        # Circular linear interpolation between the two adjacent heading
        # bins (bin b covers heading b * spacing, compass CW).
        heading = np.asarray(heading_rad) % (2.0 * math.pi)
        position = heading / self._heading_spacing_rad
        lo = np.floor(position).astype(int) % self.fields.meta.n_heading_bins
        hi = (lo + 1) % self.fields.meta.n_heading_bins
        frac = (position - np.floor(position)).astype(np.float32)
        interp = ((1.0 - frac) * scores[safe_nodes, lo]
                  + frac * scores[safe_nodes, hi])

        log_signal = (self._log_signal_coeff
                      + (interp - self._score_max[field_idx])
                      / self.config.temperature
                      - self._log_norm[field_idx])
        out = np.logaddexp(log_signal, self._log_floor)
        # Outside the declared support only the uniform floor applies: the
        # retrieval never scored those poses, so it cannot endorse them —
        # but it must not delete them either.
        return np.where(nodes >= 0, out, self._log_floor)

    def update(self, belief, meas: "structs.RetrievalMeasurement") -> list:
        """Multiply the field's likelihood into the belief. No associations
        are produced: retrieval is a pose-scored observation, not an
        identity-resolved one."""
        belief.log_weight += self.log_likelihood(
            meas.field_idx, belief.east_m, belief.north_m,
            belief.heading_rad)
        return []


def write_fields(retrieval_dir: Path, meta: RetrievalFieldsMeta,
                 lat_deg: np.ndarray, lon_deg: np.ndarray,
                 scores: np.ndarray, keyframe_idx: np.ndarray,
                 pano_ids: list) -> None:
    """Producer-side writer, kept beside the reader so the artifact contract
    has exactly one home."""
    retrieval_dir = Path(retrieval_dir)
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    if len({len(lat_deg), len(lon_deg), scores.shape[1]}) != 1:
        raise ValueError("node arrays and scores disagree on node count")
    if not (len(keyframe_idx) == scores.shape[0] == len(pano_ids)):
        raise ValueError("keyframe arrays and scores disagree on count")
    if len(set(int(k) for k in keyframe_idx)) != len(keyframe_idx):
        raise ValueError("duplicate keyframe_idx: one field per keyframe")
    (retrieval_dir / "retrieval_meta.json").write_bytes(
        msgspec.json.encode(meta))
    np.savez_compressed(
        retrieval_dir / "retrieval_fields.npz",
        lat_deg=np.asarray(lat_deg, dtype=np.float64),
        lon_deg=np.asarray(lon_deg, dtype=np.float64),
        scores=np.asarray(scores, dtype=np.float16),
        keyframe_idx=np.asarray(keyframe_idx, dtype=np.int64),
        pano_ids=np.asarray([str(p) for p in pano_ids]))


def copy_into_run(retrieval_dir: Path, run_dir: Path) -> None:
    """Preserve the consumed fields inside the run directory (replay
    surface): tier-1 for retrieval runs includes the dense fields."""
    import shutil
    run_dir = Path(run_dir)
    for name in ("retrieval_meta.json", "retrieval_fields.npz"):
        shutil.copyfile(Path(retrieval_dir) / name, run_dir / name)


def describe(fields: ScoreFields) -> str:
    return (f"retrieval fields: {fields.scores.shape[0]} keyframes x "
            f"{fields.scores.shape[1]} nodes x "
            f"{fields.meta.n_heading_bins} heading bins, "
            f"{fields.meta.node_spacing_m:.0f} m spacing "
            f"(scorer {fields.meta.scorer})")
