"""Tag-bundle landmark extractors that share the `simple_v1_v5` encoder.

Two extractors:
  - `OSMTagBundleExtractor`: satellite-side. Reads OSM `pruned_props` for each
    landmark from the vigor dataset, encodes the bundle of (key, value-text-embedding)
    pairs into a single per-landmark vector via `TagBundleEncoder`, and emits
    those vectors as feature tokens with the same satellite landmark-position
    scheme used by `SemanticLandmarkExtractor`.
  - `PanoramaTagBundleExtractor`: panorama-side. Reads `primary_tag` and
    `additional_tags` from the panov2_tuned_prompt pickle indexed by `pano_id`,
    encodes the same way, and emits per-landmark tokens with the same yaw-based
    position encoding used by `PanoramaSemanticLandmarkExtractor`.

Both extractors instantiate their own `TagBundleEncoder`; weight-sharing between
sat and pano is intentionally NOT done in v1 (sat sees OSM-style tags, pano sees
vision-extracted tags — different distributions).
"""

from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.model.additional_panorama_extractors import (
    extract_yaw_angles_from_bboxes, load_v2_pickle, yaw_angles_to_binary_vector,
)
from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    TAG_KEY_TO_IDX, TagBundleEncoder, TagBundleEncoderConfig as _DataclassEncoderConfig,
)
from experimental.overhead_matching.swag.model.semantic_landmark_extractor import (
    compute_landmark_sat_positions,
)
from experimental.overhead_matching.swag.model.swag_config_types import (
    ExtractorDataRequirement,
    OSMTagBundleExtractorConfig,
    PanoramaTagBundleExtractorConfig,
)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ExtractorOutput, ModelInput,
)


def _load_text_embeddings(path: Path) -> dict[str, torch.Tensor]:
    """Load tag-value → 768d text embeddings pickle, returning float32 tensors."""
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    out: dict[str, torch.Tensor] = {}
    for k, v in data.items():
        t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
        out[k] = t.to(torch.float32)
    return out


def _encoder_from_config(cfg) -> TagBundleEncoder:
    """Build a TagBundleEncoder from a msgspec struct, dataclass, or dict.

    `TagBundleEncoder` itself only accepts the dataclass form, so we convert.
    """
    if isinstance(cfg, _DataclassEncoderConfig):
        return TagBundleEncoder(cfg)
    if hasattr(cfg, "__struct_fields__"):  # msgspec.Struct
        kwargs = {f: getattr(cfg, f) for f in cfg.__struct_fields__}
    else:
        kwargs = dict(cfg)
    return TagBundleEncoder(_DataclassEncoderConfig(**kwargs))


def _bundle_to_tensors(
    tags,
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int,
) -> tuple[list[int], list[torch.Tensor]]:
    """Filter to known keys, look up value text embeddings.

    `tags` may be a dict[str, str] OR a frozenset/set of (key, value) tuples
    (which is how `prune_landmark` returns OSM properties in vigor_dataset).
    Keys not in `TAG_KEY_TO_IDX` are skipped (out of fixed vocabulary). Values
    not in `text_embeddings` raise — a missing-embedding is silent data corruption
    and we want to surface it loudly.

    Returns (key_indices, text_embedding_list). Both lists are parallel and may
    be empty (landmark with no recognized tags).
    """
    if hasattr(tags, "items"):
        kv_iter = tags.items()
    else:
        kv_iter = iter(tags)

    key_indices: list[int] = []
    text_embs: list[torch.Tensor] = []
    for k, v in kv_iter:
        if k not in TAG_KEY_TO_IDX:
            continue
        emb = text_embeddings.get(v)
        if emb is None:
            raise KeyError(
                f"No text embedding for tag value {v!r} (key={k!r}). "
                f"The text-embeddings pickle is missing entries that appear in "
                f"the landmark data — re-generate it so every tag value seen at "
                f"runtime has a precomputed embedding.")
        if emb.shape[0] != text_input_dim:
            raise ValueError(
                f"Text embedding for key {k!r} value {v!r} has dim {emb.shape[0]}, "
                f"expected {text_input_dim}.")
        key_indices.append(TAG_KEY_TO_IDX[k])
        text_embs.append(emb)
    return key_indices, text_embs


def _stack_landmark_tag_tensors(
    per_landmark: list[tuple[list[int], list[torch.Tensor]]],
    text_input_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a list of (key_indices, text_embs) per landmark into batched tensors.

    Args:
        per_landmark: list of length L, where each element is a (key_indices,
            text_embs) tuple from `_bundle_to_tensors`.
        text_input_dim: dim of each text embedding.
        device: target device.

    Returns:
        (key_indices_tensor, text_embeddings_tensor, tag_mask) with shapes
        (L, T_max), (L, T_max, text_input_dim), (L, T_max) respectively. T_max
        is at least 1 so the encoder doesn't choke on landmarks with zero tags.
    """
    if not per_landmark:
        return (torch.zeros(0, 1, dtype=torch.long, device=device),
                torch.zeros(0, 1, text_input_dim, device=device),
                torch.zeros(0, 1, dtype=torch.bool, device=device))

    t_max = max((len(ki) for ki, _ in per_landmark), default=0)
    t_max = max(t_max, 1)
    L = len(per_landmark)
    key_t = torch.zeros(L, t_max, dtype=torch.long)
    text_t = torch.zeros(L, t_max, text_input_dim, dtype=torch.float32)
    mask_t = torch.zeros(L, t_max, dtype=torch.bool)
    for i, (key_indices, text_embs) in enumerate(per_landmark):
        for j, (ki, te) in enumerate(zip(key_indices, text_embs)):
            key_t[i, j] = ki
            text_t[i, j] = te
            mask_t[i, j] = True
    return key_t.to(device), text_t.to(device), mask_t.to(device)


class OSMTagBundleExtractor(torch.nn.Module):
    """Satellite-side OSM tag-bundle extractor.

    For each landmark in `model_input.metadata[i]["landmarks"]`, encodes its
    `pruned_props` (key=value pairs) into a single vector via `TagBundleEncoder`.
    Position encoding matches `SemanticLandmarkExtractor` for landmarks of the
    configured `landmark_type` (point / linestring / polygon / multipolygon).
    """

    def __init__(self, config: OSMTagBundleExtractorConfig):
        super().__init__()
        self.config = config
        self._embedding_path = Path(config.tag_text_embedding_path).expanduser()
        self._encoder = _encoder_from_config(config.encoder)
        self._text_embeddings: dict[str, torch.Tensor] | None = None
        self._files_loaded = False

    def load_files(self):
        self._text_embeddings = _load_text_embeddings(self._embedding_path)
        self._files_loaded = True

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        if not self._files_loaded:
            self.load_files()

        device = model_input.image.device
        text_input_dim = self.config.encoder.text_input_dim
        landmark_type = self.config.landmark_type.lower()

        landmark_keep_masks = [
            torch.tensor(
                [lm["geometry"].geom_type.lower() == landmark_type
                 for lm in batch_item["landmarks"]],
                dtype=torch.bool)
            for batch_item in model_input.metadata
        ]
        valid_counts = [int(m.sum().item()) for m in landmark_keep_masks]
        max_landmarks = max(valid_counts) if valid_counts else 0
        batch_size = len(model_input.metadata)

        repr_dim = self._encoder.repr_dim
        features = torch.zeros(batch_size, max_landmarks, repr_dim, device=device)
        positions = torch.zeros(batch_size, max_landmarks, 2, 2, device=device)
        mask = torch.ones(batch_size, max_landmarks, dtype=torch.bool, device=device)

        # Collect tag bundles once; encode per-batch-item to keep the encoder
        # batched but mask handling straightforward.
        for i, batch_item in enumerate(model_input.metadata):
            keep = landmark_keep_masks[i]
            if int(keep.sum().item()) == 0:
                continue

            sat_positions = compute_landmark_sat_positions(
                batch_item, landmark_mask=keep
            )  # (num_kept, 2, 2)
            positions[i, : sat_positions.shape[0]] = sat_positions.to(device)

            per_landmark: list[tuple[list[int], list[torch.Tensor]]] = []
            for keep_flag, landmark in zip(keep.tolist(), batch_item["landmarks"]):
                if not keep_flag:
                    continue
                props = landmark.get("pruned_props", {})
                per_landmark.append(_bundle_to_tensors(
                    props, self._text_embeddings, text_input_dim))

            key_t, text_t, tag_mask_t = _stack_landmark_tag_tensors(
                per_landmark, text_input_dim, device)
            if key_t.shape[0] == 0:
                continue
            landmark_reprs = self._encoder(key_t, text_t, tag_mask_t)  # (L, repr_dim)
            num_kept = landmark_reprs.shape[0]
            features[i, :num_kept] = landmark_reprs
            mask[i, :num_kept] = False

        return ExtractorOutput(features=features, positions=positions, mask=mask)

    @property
    def output_dim(self) -> int:
        return self._encoder.repr_dim

    @property
    def num_position_outputs(self) -> int:
        return 2

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        return [ExtractorDataRequirement.LANDMARKS]


class PanoramaTagBundleExtractor(torch.nn.Module):
    """Panorama-side tag-bundle extractor over panov2_tuned_prompt landmarks.

    Loads the per-city panov2 v2.0 pickle once at first forward, indexes by
    `pano_id`, and for each panorama in the batch encodes its visible landmarks'
    primary + additional OSM tags into per-landmark vectors via `TagBundleEncoder`.
    Positions use the same yaw-binary-vector scheme as
    `PanoramaSemanticLandmarkExtractor`.
    """

    def __init__(self, config: PanoramaTagBundleExtractorConfig):
        super().__init__()
        self.config = config
        self._text_embedding_path = Path(config.tag_text_embedding_path).expanduser()
        self._panov2_root = Path(config.panov2_root).expanduser()
        self._encoder = _encoder_from_config(config.encoder)
        self._text_embeddings: dict[str, torch.Tensor] | None = None
        self._panorama_landmarks: dict[str, list[dict]] | None = None
        self._files_loaded = False
        self._found_pano_ids: set[str] = set()
        self._missing_pano_ids: set[str] = set()

    def load_files(self):
        # Text-value embeddings (shared across cities)
        self._text_embeddings = _load_text_embeddings(self._text_embedding_path)

        # Per-city panov2 pickle structure
        panov2_root = self._panov2_root
        if not panov2_root.exists():
            raise FileNotFoundError(f"Panov2 root does not exist: {panov2_root}")
        city_dirs = [d for d in panov2_root.iterdir() if d.is_dir()]
        if not city_dirs:
            raise FileNotFoundError(f"No city directories in {panov2_root}")

        self._panorama_landmarks = {}
        for city_dir in city_dirs:
            pickle_path = city_dir / "embeddings" / "embeddings.pkl"
            data = load_v2_pickle(pickle_path)
            if data is None:
                raise FileNotFoundError(f"v2.0 pickle not found at {pickle_path}")
            for pano_key, pano_data in data.get("panoramas", {}).items():
                pano_id = pano_key.split(",")[0]
                landmarks = pano_data.get("landmarks", [])
                self._panorama_landmarks[pano_id] = landmarks
        self._files_loaded = True

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        if not self._files_loaded:
            self.load_files()
        for item in model_input.metadata:
            if "pano_id" not in item:
                raise ValueError(
                    "PanoramaTagBundleExtractor requires panorama metadata with 'pano_id'")

        device = model_input.image.device
        text_input_dim = self.config.encoder.text_input_dim

        per_batch_landmarks: list[list[dict]] = []
        for item in model_input.metadata:
            pano_id = item["pano_id"]
            if pano_id in self._panorama_landmarks:
                self._found_pano_ids.add(pano_id)
                per_batch_landmarks.append(
                    sorted(self._panorama_landmarks[pano_id],
                           key=lambda x: x.get("landmark_idx", 0)))
            else:
                self._missing_pano_ids.add(pano_id)
                per_batch_landmarks.append([])

        max_landmarks = max((len(lms) for lms in per_batch_landmarks), default=0)
        batch_size = len(model_input.metadata)
        repr_dim = self._encoder.repr_dim
        features = torch.zeros(batch_size, max_landmarks, repr_dim, device=device)
        positions = torch.zeros(batch_size, max_landmarks, 2, 2, device=device)
        mask = torch.ones(batch_size, max_landmarks, dtype=torch.bool, device=device)

        for i, landmarks in enumerate(per_batch_landmarks):
            if not landmarks:
                continue
            per_landmark: list[tuple[list[int], list[torch.Tensor]]] = []
            yaw_vectors: list[list[int]] = []
            for landmark in landmarks:
                tags = self._collect_panov2_tags(landmark)
                per_landmark.append(_bundle_to_tensors(
                    tags, self._text_embeddings, text_input_dim))
                yaw_angles = extract_yaw_angles_from_bboxes(
                    landmark.get("bounding_boxes", []))
                yaw_vectors.append(yaw_angles_to_binary_vector(yaw_angles))

            key_t, text_t, tag_mask_t = _stack_landmark_tag_tensors(
                per_landmark, text_input_dim, device)
            if key_t.shape[0] == 0:
                continue
            landmark_reprs = self._encoder(key_t, text_t, tag_mask_t)
            num_kept = landmark_reprs.shape[0]
            features[i, :num_kept] = landmark_reprs
            mask[i, :num_kept] = False

            for j, yv in enumerate(yaw_vectors):
                positions[i, j, 0, 0] = float(yv[0])
                positions[i, j, 0, 1] = float(yv[1])
                positions[i, j, 1, 0] = float(yv[2])
                positions[i, j, 1, 1] = float(yv[3])

        return ExtractorOutput(features=features, positions=positions, mask=mask)

    @staticmethod
    def _collect_panov2_tags(landmark: dict) -> dict[str, str]:
        """Flatten a panov2 landmark's primary + additional tags into a dict."""
        out: dict[str, str] = {}
        primary = landmark.get("primary_tag")
        if isinstance(primary, dict):
            k, v = primary.get("key"), primary.get("value")
            if k is not None and v is not None:
                out[str(k)] = str(v)
        for extra in landmark.get("additional_tags", []) or []:
            if not isinstance(extra, dict):
                continue
            k, v = extra.get("key"), extra.get("value")
            if k is not None and v is not None:
                out[str(k)] = str(v)
        return out

    @property
    def output_dim(self) -> int:
        return self._encoder.repr_dim

    @property
    def num_position_outputs(self) -> int:
        return 2

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        # The panov2 pickle is loaded directly; no vigor LANDMARKS dependency.
        return []
