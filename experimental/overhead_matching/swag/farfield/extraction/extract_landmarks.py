"""Build typed pinhole-image and frame-landmark artifacts.

The immutable build recipe owns every value that can shape pixels, prompts,
predictions, execution selection, or cost approval.  The command line names
only the dataset, exact input/output directories, the exact build recipe, and
the transport settings that must agree with that recipe.

Publication has two independent transactions.  ``pinhole_images`` is
published first, then used as an immutable upstream of the LLM request set and
the final ``frame_landmarks`` artifact.  Consequently a crash between the two
publications is safe: a rerun validates and reuses the completed pinhole
artifact.  Provider traffic, request snapshots, and append-only attempts live
beside (never inside) the final artifact under a request-set fingerprint.

There is no partial-result mode.  ``frame_landmarks`` contains exactly one
root ``predictions.jsonl`` record for every panorama, in panorama order, and
is published only after exactly one schema-valid successful response exists
for every request.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.farfield import (
    artifact,
    artifact_recipe,
    build_config,
    dataset as dataset_lib,
    llm_lifecycle,
    paths as paths_lib,
    publication,
    provenance,
)
from experimental.overhead_matching.swag.farfield.extraction import (
    llm_cost,
    panorama_to_pinhole,
    prompts,
    vertex_batch_manager as vbm,
)


GENERATOR = ("//experimental/overhead_matching/swag/farfield/extraction:"
             "extract_landmarks")
LEGACY_ADOPTION_GENERATOR = (
    "//experimental/overhead_matching/swag/farfield/extraction:"
    "legacy_extraction_adoption")
LEGACY_ADOPTION_CONFIG_SCHEMA = (
    "farfield.legacy_extraction_adopted_artifact/v1")
LEGACY_ADOPTION_PROOF_SCHEMA = (
    "farfield.legacy_extraction_publication_proof/v1")
LEGACY_ADOPTION_REPORT_SCHEMA = (
    "farfield.legacy_extraction_adoption_report/v1")
PREDICTIONS_NAME = dataset_lib.PREDICTIONS_NAME
ATTEMPTS_DIR_NAME = llm_lifecycle.ATTEMPTS_DIR_NAME
REQUEST_ARTIFACT_DIR = "requests"
RESULT_ARTIFACT_DIR = "results"
WORK_SUFFIX = ".llm-work"
FACE_FOV_RAD = math.pi / 2.0
NUM_WORKERS = 8
MAX_REQUESTS_PER_BATCH = 10_000

EXTRACTION_CONFIG_KEYS = (
    "artifacts.frame_landmarks_version",
    "artifacts.pinhole_images_version",
    "extraction.model",
    "extraction.prompt_type",
    "extraction.pinhole_resolution",
    "extraction.media_resolution",
    "extraction.thinking_level",
    "execution.llm_transport",
    "execution.batch_gcs_prefix",
    "execution.approve_cost",
    "cost.limit_usd",
)

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_RAW_SHARD_RE = re.compile(r"raw_(\d+)\.jsonl\Z")
_PENDING_SHARD_RE = re.compile(r"pending_(\d+)\.jsonl\Z")
_REF_IDENTITY_KEYS = frozenset({
    "kind", "dataset", "version", "manifest_digest", "content_digest",
})


@dataclass(frozen=True)
class ExtractionContext:
    """Validated immutable recipe plus the exact dataset inputs it binds."""

    document: dict[str, Any]
    selected: dict[str, Any]
    orchestration: dict[str, Any]
    metadata: dict[str, Any]
    frames: tuple[dataset_lib.Frame, ...]
    dataset_base: Path
    panorama_dir: Path
    input_digests: dict[str, str]

    @property
    def stems(self) -> tuple[str, ...]:
        return tuple(frame.pano_stem for frame in self.frames)

    @property
    def pinhole_version(self) -> str:
        return self.selected["artifacts.pinhole_images_version"]

    @property
    def frame_version(self) -> str:
        return self.selected["artifacts.frame_landmarks_version"]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _strict_json_loads(text: str, where: str) -> Any:
    try:
        return json.loads(
            text, object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeError, ValueError) as error:
        raise ValueError(f"{where}: invalid strict JSON: {error}") from error


def _ref_identity(ref: artifact.ArtifactRef) -> dict[str, str]:
    return {
        "kind": ref.kind,
        "dataset": ref.dataset,
        "version": ref.version,
        "manifest_digest": ref.manifest_digest,
        "content_digest": ref.content_digest,
    }


def _validate_ref_identity(value: Any, *, kind: str, dataset: str,
                           where: str) -> dict[str, str]:
    if not isinstance(value, dict) or frozenset(value) != _REF_IDENTITY_KEYS:
        raise ValueError(f"{where} is not an exact artifact identity")
    if value["kind"] != kind or value["dataset"] != dataset:
        raise ValueError(f"{where} names the wrong artifact kind or dataset")
    if (not isinstance(value["version"], str) or not value["version"]
            or not isinstance(value["manifest_digest"], str)
            or not isinstance(value["content_digest"], str)
            or not _DIGEST_RE.fullmatch(value["manifest_digest"])
            or not _DIGEST_RE.fullmatch(value["content_digest"])):
        raise ValueError(f"{where} contains an invalid version or digest")
    return value


def legacy_adoption_marker(report_sha256: str) -> dict[str, str]:
    """Return the exact immutable marker for specialized adopted artifacts."""
    if (not isinstance(report_sha256, str)
            or not _DIGEST_RE.fullmatch(report_sha256)):
        raise ValueError(
            "legacy adoption report digest must be a lowercase SHA-256")
    return {
        "legacy_adoption_schema": LEGACY_ADOPTION_CONFIG_SCHEMA,
        "legacy_adoption_report_sha256": report_sha256,
    }


def _validate_schema(value: Any, schema: Mapping[str, Any],
                     where: str = "prediction") -> None:
    """Strict subset of JSON Schema used by ``prompts.response_schema``.

    The provider schema is composed only of objects, arrays, strings, enums,
    integers, and scalar bounds.  Exact object keys are required here even if
    a provider happens to ignore ``additionalProperties``: unknown model
    fields must never leak into the canonical consumer artifact.
    """
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{where} must be an object")
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            raise ValueError(f"{where} has an invalid object schema")
        required = set(schema.get("required", ()))
        expected = set(properties)
        actual = set(value)
        if actual != expected or required != expected:
            raise ValueError(
                f"{where} must have exact keys {sorted(expected)}; "
                f"found {sorted(actual)}")
        for key, child_schema in properties.items():
            _validate_schema(value[key], child_schema, f"{where}.{key}")
    elif schema_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{where} must be an array")
        item_schema = schema.get("items")
        if not isinstance(item_schema, Mapping):
            raise ValueError(f"{where} has an invalid array schema")
        for index, item in enumerate(value):
            _validate_schema(item, item_schema, f"{where}[{index}]")
    elif schema_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"{where} must be a string")
    elif schema_type == "integer":
        if type(value) is not int:
            raise ValueError(f"{where} must be an integer")
    elif schema_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{where} must be finite")
    elif schema_type == "boolean":
        if type(value) is not bool:
            raise ValueError(f"{where} must be boolean")
    elif schema_type == "null":
        if value is not None:
            raise ValueError(f"{where} must be null")
    elif schema_type is not None:
        raise ValueError(f"{where} has unsupported schema type {schema_type!r}")

    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{where} must be one of {schema['enum']!r}")
    if "minimum" in schema and value < schema["minimum"]:
        raise ValueError(f"{where} must be >= {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        raise ValueError(f"{where} must be <= {schema['maximum']}")


def _provider_prediction_to_canonical(value: Any) -> dict[str, Any]:
    """Validate the provider schema and normalize yaw strings for ingest."""
    _validate_schema(value, prompts.response_schema())
    # JSON round-trip detaches the returned payload before normalization.
    prediction = json.loads(artifact.canonical_json_bytes(value))
    for landmark_index, landmark in enumerate(prediction["landmarks"]):
        boxes = landmark["bounding_boxes"]
        if not boxes:
            raise ValueError(
                f"prediction.landmarks[{landmark_index}].bounding_boxes must "
                "not be empty")
        for box_index, box in enumerate(boxes):
            where = (f"prediction.landmarks[{landmark_index}]."
                     f"bounding_boxes[{box_index}]")
            yaw = box["yaw_angle"]
            if yaw not in ("0", "90", "180", "270"):
                raise ValueError(
                    f"{where}.yaw_angle must be one of '0', '90', '180', "
                    "'270'")
            if box["xmin"] >= box["xmax"] or box["ymin"] >= box["ymax"]:
                raise ValueError(f"{where} must have positive width and height")
            box["yaw_angle"] = int(yaw)
    return prediction


def _validate_canonical_prediction(value: Any) -> dict[str, Any]:
    """Validate the exact prediction shape consumed by ``dataset.run_ingest``."""
    try:
        prediction = json.loads(artifact.canonical_json_bytes(value))
    except (artifact.ArtifactError, TypeError, ValueError) as error:
        raise ValueError(f"prediction is not finite JSON: {error}") from error
    if not isinstance(prediction, dict) or set(prediction) != {
            "location_type", "landmarks"}:
        raise ValueError("prediction must have exact keys location_type, landmarks")
    schema_view = json.loads(artifact.canonical_json_bytes(prediction))
    landmarks = schema_view.get("landmarks")
    if not isinstance(landmarks, list):
        raise ValueError("prediction.landmarks must be an array")
    for landmark_index, landmark in enumerate(landmarks):
        boxes = landmark.get("bounding_boxes") if isinstance(landmark, dict) \
            else None
        if not isinstance(boxes, list) or not boxes:
            raise ValueError(
                f"prediction.landmarks[{landmark_index}].bounding_boxes must "
                "be a non-empty array")
        for box_index, box in enumerate(boxes):
            where = (f"prediction.landmarks[{landmark_index}]."
                     f"bounding_boxes[{box_index}]")
            if not isinstance(box, dict):
                raise ValueError(f"{where} must be an object")
            yaw = box.get("yaw_angle")
            if isinstance(yaw, bool) or type(yaw) is not int or yaw not in (
                    0, 90, 180, 270):
                raise ValueError(f"{where}.yaw_angle must be 0, 90, 180, or 270")
            if (isinstance(box.get("xmin"), bool)
                    or isinstance(box.get("ymin"), bool)
                    or isinstance(box.get("xmax"), bool)
                    or isinstance(box.get("ymax"), bool)):
                raise ValueError(f"{where} coordinates must be numeric")
            try:
                if box["xmin"] >= box["xmax"] or box["ymin"] >= box["ymax"]:
                    raise ValueError(
                        f"{where} must have positive width and height")
            except (KeyError, TypeError) as error:
                raise ValueError(f"{where} has invalid coordinates") from error
            box["yaw_angle"] = str(yaw)
    _validate_schema(schema_view, prompts.response_schema())
    return prediction


def validate_response(key: str, response: Mapping[str, Any]) -> dict[str, Any]:
    """LLM lifecycle validator for one extraction response."""
    del key
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("response must contain exactly one candidate")
    candidate = candidates[0]
    content = candidate.get("content") if isinstance(candidate, Mapping) else None
    parts = content.get("parts") if isinstance(content, Mapping) else None
    if not isinstance(parts, list) or len(parts) != 1:
        raise ValueError("response candidate must contain exactly one part")
    part = parts[0]
    text = part.get("text") if isinstance(part, Mapping) else None
    if not isinstance(text, str) or not text:
        raise ValueError("response part must contain non-empty JSON text")
    payload = _strict_json_loads(text, "response text")
    return _provider_prediction_to_canonical(payload)


def _selected_values(document: dict[str, Any]) -> dict[str, Any]:
    return {key: build_config.value(document, key)
            for key in EXTRACTION_CONFIG_KEYS}


def _validate_selected(selected: dict[str, Any]) -> None:
    specs = {
        "artifacts.frame_landmarks_version": build_config.ValueSpec(
            (str,), nonempty=True),
        "artifacts.pinhole_images_version": build_config.ValueSpec(
            (str,), nonempty=True),
        "extraction.model": build_config.ValueSpec((str,), nonempty=True),
        "extraction.prompt_type": build_config.ValueSpec(
            (str,), choices=prompts.PROMPT_TYPES),
        "extraction.pinhole_resolution": build_config.ValueSpec(
            (int,), minimum=1),
        "extraction.media_resolution": build_config.ValueSpec(
            (str,), choices=prompts.MEDIA_RESOLUTIONS),
        "extraction.thinking_level": build_config.ValueSpec(
            (str,), choices=prompts.THINKING_LEVELS),
        "execution.llm_transport": build_config.ValueSpec(
            (str,), choices=("batch", "on_demand")),
        "execution.batch_gcs_prefix": build_config.ValueSpec(
            (str,), allow_none=True, nonempty=True),
        "execution.approve_cost": build_config.ValueSpec((bool,)),
        "cost.limit_usd": build_config.ValueSpec(
            (int, float), minimum=0.0),
    }
    for key, spec in specs.items():
        spec.validate(key, selected[key])
    prefix = selected["execution.batch_gcs_prefix"]
    if selected["execution.llm_transport"] == "batch":
        if not isinstance(prefix, str) or not prefix.startswith("gs://"):
            raise build_config.InvalidConfigValue(
                "execution.batch_gcs_prefix must be a gs:// URI in batch mode")
    elif prefix is not None:
        raise build_config.InvalidConfigValue(
            "execution.batch_gcs_prefix must be null in on_demand mode")


def load_artifact_validation_context(
        *, build_config_path: Path, dataset: str,
        dataset_base: Path) -> ExtractionContext:
    """Load the immutable extraction context without transport CLI state.

    Completed-artifact readers and stage-reuse authorization deliberately use
    this same context constructor as the producer.  That keeps pixel, prompt,
    request, and prediction validation bound to one implementation.
    """
    config_path = Path(build_config_path)
    if config_path.name != build_config.BUILD_CONFIG_NAME:
        raise ValueError(
            f"--build_config must name {build_config.BUILD_CONFIG_NAME}")
    document = build_config.load(config_path.parent)
    if config_path.resolve() != (
            config_path.parent / build_config.BUILD_CONFIG_NAME).resolve():
        raise ValueError("--build_config does not name the loaded recipe")
    if document["dataset"] != dataset:
        raise ValueError(
            f"build config dataset {document['dataset']!r} does not match "
            f"--dataset {dataset!r}")
    dataset_base = Path(dataset_base).resolve()
    recorded_base = document["inputs"].get("dataset_base")
    if recorded_base is None or Path(recorded_base).resolve() != dataset_base:
        raise ValueError(
            "--dataset_base does not match build_config inputs.dataset_base")

    selected = _selected_values(document)
    _validate_selected(selected)
    actual_digest = artifact.sha256_json(selected)
    orchestration = {
        "schema": "farfield_pipeline_stage/v1",
        "stage": "extract",
        "config_digest": actual_digest,
    }
    metadata = dataset_lib.load_metadata(dataset_base)
    dataset_lib.require_camera_frame_panoramas(metadata, dataset_base)
    if metadata["dataset_name"] != dataset:
        raise ValueError(
            "pipeline_metadata.json dataset_name does not match --dataset")
    frames = tuple(dataset_lib.load_frames(dataset_base))
    if not frames:
        raise ValueError(f"no panoramas under {dataset_base / 'panorama'}")
    panorama_dir = dataset_base / "panorama"
    try:
        source_digests = paths_lib.dataset_source_digests(dataset_base)
    except paths_lib.MissingInput as exc:
        raise ValueError(str(exc)) from exc
    mismatched_sources = [
        key for key in paths_lib.DATASET_SOURCE_DIGEST_KEYS
        if document["inputs"].get(key) != source_digests[key]
    ]
    if mismatched_sources:
        raise ValueError(
            "dataset source bytes differ from the immutable build recipe: "
            f"{mismatched_sources}")
    input_digests = {
        "build_identity": document["build_identity"],
        "orchestration_config": actual_digest,
        "panorama_directory": source_digests[
            paths_lib.DATASET_PANORAMA_SHA256],
        "pipeline_metadata": source_digests[
            paths_lib.DATASET_PIPELINE_METADATA_SHA256],
        "frames_gps": source_digests[paths_lib.DATASET_FRAMES_GPS_SHA256],
    }
    return ExtractionContext(
        document=document,
        selected=selected,
        orchestration=orchestration,
        metadata=metadata,
        frames=frames,
        dataset_base=dataset_base,
        panorama_dir=panorama_dir,
        input_digests=input_digests,
    )


def load_context(args: argparse.Namespace) -> ExtractionContext:
    """Load the exact recipe and validate its dataset/stage binding."""
    context = load_artifact_validation_context(
        build_config_path=args.build_config, dataset=args.dataset,
        dataset_base=args.dataset_base)
    selected = context.selected
    actual_digest = context.orchestration["config_digest"]
    if args.orchestration_config_digest != actual_digest:
        raise ValueError(
            "--orchestration_config_digest does not match the immutable "
            "extraction/execution/cost recipe")
    expected_online = selected["execution.llm_transport"] == "on_demand"
    if bool(args.online) != expected_online:
        raise ValueError(
            "--online selection disagrees with execution.llm_transport")
    expected_prefix = selected["execution.batch_gcs_prefix"]
    if ((expected_online and args.gcs_prefix is not None)
            or (not expected_online and args.gcs_prefix != expected_prefix)):
        raise ValueError(
            "--gcs_prefix disagrees with execution.batch_gcs_prefix")
    if bool(args.approve_cost) != selected["execution.approve_cost"]:
        raise ValueError(
            "--approve_cost disagrees with execution.approve_cost")
    if args.cost_limit != float(selected["cost.limit_usd"]):
        raise ValueError("--cost_limit disagrees with cost.limit_usd")
    if args.parallel < 1 or args.poll_interval < 1:
        raise ValueError("--parallel and --poll_interval must be positive")
    args.model = selected["extraction.model"]

    pinhole_output = Path(args.pinhole_output_dir).resolve()
    frame_output = Path(args.output_dir).resolve()
    if (pinhole_output == frame_output
            or pinhole_output in frame_output.parents
            or frame_output in pinhole_output.parents):
        raise ValueError(
            "--pinhole_output_dir and --output_dir must be disjoint")
    return context


def _pinhole_outputs(context: ExtractionContext) -> tuple[str, ...]:
    return paths_lib.pinhole_declared_outputs(context.stems)


def _pinhole_config(context: ExtractionContext) -> dict[str, Any]:
    source_digests = {
        key: context.document["inputs"][key]
        for key in paths_lib.DATASET_SOURCE_DIGEST_KEYS
    }
    return paths_lib.pinhole_manifest_config(
        source_digests,
        resolution=context.selected["extraction.pinhole_resolution"],
        panorama_keys=context.stems)


def _validate_manifest(path: Path, *, upstreams: Sequence[artifact.ArtifactRef],
                       config: Mapping[str, Any],
                       declared_outputs: Sequence[str]) -> None:
    manifest = artifact.load_manifest(path)
    if manifest.upstreams != tuple(upstreams):
        raise ValueError(f"{path} has different upstream artifact identities")
    if dict(manifest.config) != dict(config):
        raise ValueError(f"{path} has a different immutable configuration")
    if manifest.declared_outputs != tuple(sorted(declared_outputs)):
        raise ValueError(f"{path} has different declared outputs")


def _decode_pinhole_face(path: Path, resolution: int) -> np.ndarray:
    """Decode one required JPEG and enforce its exact pixel dimensions."""
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "JPEG":
                raise ValueError(f"{path} is not a JPEG image")
            if image.size != (resolution, resolution):
                raise ValueError(
                    f"{path} has dimensions {image.size[0]}x{image.size[1]}; "
                    f"expected {resolution}x{resolution}")
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except OSError as error:
        raise ValueError(f"{path} is not a decodable JPEG: {error}") from error


def _sample_frame_indices(n_frames: int) -> tuple[int, ...]:
    """Deterministic first/middle/last sample, without duplicate indices."""
    return tuple(sorted({0, n_frames // 2, n_frames - 1}))


def _expected_decoded_face(
        panorama: np.ndarray, resolution: int, yaw_deg: int) -> np.ndarray:
    projected = panorama_to_pinhole.reproject_pinhole(
        panorama,
        (resolution, resolution),
        (FACE_FOV_RAD, FACE_FOV_RAD),
        yaw=math.radians(yaw_deg),
        pitch=0.0,
    )
    pixels = np.clip(projected * 255.0, 0, 255).astype(np.uint8)
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="JPEG")
    encoded.seek(0)
    with Image.open(encoded) as image:
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def validate_pinhole_images(path: Path, context: ExtractionContext) -> None:
    """Validate every face and reproduce all faces for three fixed panoramas."""
    path = Path(path)
    resolution = context.selected["extraction.pinhole_resolution"]
    for relative in _pinhole_outputs(context):
        _decode_pinhole_face(path / relative, resolution)

    for index in _sample_frame_indices(len(context.frames)):
        frame = context.frames[index]
        panorama_path = context.panorama_dir / f"{frame.pano_stem}.jpg"
        try:
            with Image.open(panorama_path) as image:
                image.load()
                panorama = np.asarray(image, dtype=np.float32) / 255.0
        except OSError as error:
            raise ValueError(
                f"{panorama_path} cannot reproduce pinhole validation "
                f"samples: {error}") from error
        if panorama.ndim != 3:
            raise ValueError(
                f"{panorama_path} must decode to a multi-channel image")
        for face, yaw_deg in zip(prompts.PINHOLE_FACES, prompts.FACE_YAWS):
            face_path = path / frame.pano_stem / f"{face}.jpg"
            actual = _decode_pinhole_face(face_path, resolution)
            expected = _expected_decoded_face(
                panorama, resolution, yaw_deg)
            if not np.array_equal(actual, expected):
                difference = np.abs(
                    actual.astype(np.int16) - expected.astype(np.int16))
                raise ValueError(
                    f"{face_path} does not reproduce its source panorama "
                    f"(maximum channel difference {int(difference.max())})")


def validate_existing_pinhole_artifact(
        args: argparse.Namespace,
        context: ExtractionContext) -> artifact.ArtifactRef:
    """Run the producer's complete pixel and manifest reuse validation."""
    destination = Path(args.pinhole_output_dir)
    validate_pinhole_images(destination, context)
    ref = artifact.open_artifact(
        destination,
        expected_kind=paths_lib.PINHOLE_IMAGES,
        expected_dataset=args.dataset,
        expected_version=context.pinhole_version)
    _validate_manifest(
        destination, upstreams=(), config=_pinhole_config(context),
        declared_outputs=_pinhole_outputs(context))
    return ref


def ensure_pinhole_artifact(
        args: argparse.Namespace, context: ExtractionContext,
        *, arguments: Sequence[str] = ()) -> artifact.ArtifactRef:
    """Validate and reuse, or transactionally publish, pinhole faces."""
    destination = Path(args.pinhole_output_dir)
    outputs = _pinhole_outputs(context)
    config = _pinhole_config(context)
    if destination.exists() or destination.is_symlink():
        return validate_existing_pinhole_artifact(args, context)

    with publication.published_artifact(
            destination,
            kind=paths_lib.PINHOLE_IMAGES,
            dataset=args.dataset,
            version=context.pinhole_version,
            generator=GENERATOR,
            git_commit=provenance.git_commit(),
            arguments=arguments,
            upstreams=(),
            config=config,
            declared_outputs=outputs) as builder:
        resolution = context.selected["extraction.pinhole_resolution"]
        panorama_to_pinhole.process_panoramas(
            context.panorama_dir,
            builder.path,
            FACE_FOV_RAD,
            FACE_FOV_RAD,
            resolution,
            resolution,
            num_workers=NUM_WORKERS,
        )
        validate_pinhole_images(builder.path, context)
    assert builder.artifact_ref is not None
    return builder.artifact_ref


def _request_media_settings(context: ExtractionContext) -> dict[str, Any]:
    return {
        "prompt_type": context.selected["extraction.prompt_type"],
        "pinhole_resolution": context.selected["extraction.pinhole_resolution"],
        "media_resolution": context.selected["extraction.media_resolution"],
        "thinking_level": context.selected["extraction.thinking_level"],
        "face_order": list(prompts.PINHOLE_FACES),
    }


def _request_set_matches(
        request_set: llm_lifecycle.RequestSet,
        context: ExtractionContext,
        pinhole_ref: artifact.ArtifactRef) -> bool:
    value = request_set.to_dict()
    if (value["stage"] != "frame_landmark_extraction"
            or value["model"] != context.selected["extraction.model"]
            or value["system_prompt"] != prompts.SYSTEM_PROMPTS[
                context.selected["extraction.prompt_type"]]
            or value["response_schema"] != prompts.response_schema()
            or value["media_settings"] != _request_media_settings(context)
            or value["input_digests"] != context.input_digests
            or request_set.upstreams != (pinhole_ref,)):
        return False
    units = value["units"]
    if [unit["key"] for unit in units] != list(context.stems):
        return False
    for unit in units:
        if unit["metadata"] != {"panorama_stem": unit["key"]}:
            return False
        request = unit["request"]
        try:
            contents = request["contents"]
            parts = contents[0]["parts"]
            instruction = request["systemInstruction"]["parts"][0]["text"]
            generation = request["generationConfig"]
        except (KeyError, IndexError, TypeError):
            return False
        if (not isinstance(contents, list) or len(contents) != 1
                or not isinstance(parts, list) or len(parts) != 5
                or sum(isinstance(part, dict) and "inline_data" in part
                       for part in parts) != 4
                or parts[-1] != {"text": prompts.USER_PROMPT}
                or instruction != value["system_prompt"]
                or generation.get("responseSchema") != value["response_schema"]
                or generation.get("responseMimeType") != "application/json"
                or generation.get("thinkingConfig") != {
                    "thinkingLevel": context.selected[
                        "extraction.thinking_level"]}):
            return False
    return True


def build_request_set(
        context: ExtractionContext,
        pinhole_ref: artifact.ArtifactRef,
        work_root: Path) -> llm_lifecycle.RequestSet:
    """Construct the complete deterministic workload from the pinhole artifact."""
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=".request-build-", dir=work_root) as temporary:
        files = prompts.write_requests(
            Path(pinhole_ref.path),
            context.panorama_dir,
            Path(temporary),
            prompt_type=context.selected["extraction.prompt_type"],
            media_resolution=context.selected["extraction.media_resolution"],
            thinking_level=context.selected["extraction.thinking_level"],
            num_workers=NUM_WORKERS,
        )
        records = []
        for path in sorted(files):
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        raise ValueError(f"{path}:{line_number}: blank request")
                    value = _strict_json_loads(
                        line, f"{path}:{line_number}")
                    if not isinstance(value, dict) or set(value) != {
                            "key", "request"}:
                        raise ValueError(
                            f"{path}:{line_number}: request record must have "
                            "exact keys key, request")
                    records.append(value)
    keys = [record["key"] for record in records]
    if keys != list(context.stems):
        raise ValueError(
            "generated request keys do not exactly cover panoramas in order: "
            f"expected {list(context.stems)}, found {keys}")
    units = tuple(llm_lifecycle.RequestUnit(
        key=record["key"], request=record["request"],
        metadata={"panorama_stem": record["key"]})
        for record in records)
    return llm_lifecycle.RequestSet.create(
        stage="frame_landmark_extraction",
        model=context.selected["extraction.model"],
        system_prompt=prompts.SYSTEM_PROMPTS[
            context.selected["extraction.prompt_type"]],
        response_schema=prompts.response_schema(),
        media_settings=_request_media_settings(context),
        input_digests=context.input_digests,
        upstreams=(pinhole_ref,),
        units=units,
    )


def _validate_recorded_request_payloads(
        request_path: Path, context: ExtractionContext,
        pinhole_ref: artifact.ArtifactRef,
        fingerprint: str) -> llm_lifecycle.RequestSet:
    """Rebuild pixels-to-request bytes and require their exact snapshot."""
    request_set_path = request_path / llm_lifecycle.REQUEST_SET_NAME
    recorded = llm_lifecycle.load_request_set(request_set_path)
    with tempfile.TemporaryDirectory(
            prefix=".request-validation-") as temporary:
        expected = build_request_set(context, pinhole_ref, Path(temporary))
    if (recorded.fingerprint != fingerprint
            or recorded != expected
            or not _request_set_matches(recorded, context, pinhole_ref)):
        raise ValueError(
            "request set differs from the reproduced extraction workload")
    expected_request_set_bytes = (
        artifact.canonical_json_bytes(recorded.to_dict()) + b"\n")
    if request_set_path.read_bytes() != expected_request_set_bytes:
        raise ValueError("request_set.json is not canonical")
    if (request_path / llm_lifecycle.REQUESTS_NAME).read_bytes() \
            != llm_lifecycle.transport_requests_bytes(recorded):
        raise ValueError("requests.jsonl differs from its typed request set")
    return recorded


def _request_version(context: ExtractionContext, fingerprint: str) -> str:
    return f"{context.frame_version}.requests.{fingerprint[:16]}"


def _result_version(context: ExtractionContext, fingerprint: str) -> str:
    return f"{context.frame_version}.results.{fingerprint[:16]}"


def _request_config(context: ExtractionContext,
                    request_set: llm_lifecycle.RequestSet) -> dict[str, Any]:
    return {
        "orchestration": context.orchestration,
        "build_identity": context.document["build_identity"],
        "purpose": "frame_landmark_extraction",
        "n_expected": len(request_set.units),
        "request_set_fingerprint": request_set.fingerprint,
    }


def _validate_request_artifact(
        path: Path, request_set: llm_lifecycle.RequestSet,
        context: ExtractionContext,
        pinhole_ref: artifact.ArtifactRef) -> artifact.ArtifactRef:
    ref = artifact.open_artifact(
        path,
        expected_kind=llm_lifecycle.REQUEST_ARTIFACT_KIND,
        expected_dataset=context.document["dataset"],
        expected_version=_request_version(context, request_set.fingerprint))
    _validate_manifest(
        path,
        upstreams=(pinhole_ref,),
        config=_request_config(context, request_set),
        declared_outputs=(llm_lifecycle.REQUEST_SET_NAME,
                          llm_lifecycle.REQUESTS_NAME))
    recorded = llm_lifecycle.load_request_set(
        path / llm_lifecycle.REQUEST_SET_NAME)
    if recorded.fingerprint != request_set.fingerprint:
        raise ValueError(f"{path} contains a different request set")
    return ref


def ensure_request_artifact(
        args: argparse.Namespace, context: ExtractionContext,
        pinhole_ref: artifact.ArtifactRef,
        *, arguments: Sequence[str] = (),
    ) -> tuple[llm_lifecycle.RequestSet, artifact.ArtifactRef, Path]:
    """Find the exact fingerprint-scoped workload, or construct it once."""
    work_root = Path(args.output_dir).with_name(
        Path(args.output_dir).name + WORK_SUFFIX)
    work_root.mkdir(parents=True, exist_ok=True)
    candidates = []
    for work_dir in sorted(work_root.iterdir()):
        if not work_dir.is_dir() or not _DIGEST_RE.fullmatch(work_dir.name):
            continue
        request_dir = work_dir / REQUEST_ARTIFACT_DIR
        if not request_dir.exists():
            continue
        # A corrupt fingerprint-scoped snapshot is a hard error, not a reason
        # to silently build another workload beside it.
        artifact.open_artifact(
            request_dir,
            expected_kind=llm_lifecycle.REQUEST_ARTIFACT_KIND,
            expected_dataset=args.dataset)
        request_set = llm_lifecycle.load_request_set(
            request_dir / llm_lifecycle.REQUEST_SET_NAME)
        if work_dir.name != request_set.fingerprint:
            raise ValueError(
                f"work directory {work_dir} does not match its request "
                "fingerprint")
        if _request_set_matches(request_set, context, pinhole_ref):
            candidates.append((request_set, request_dir, work_dir))
    if len(candidates) > 1:
        raise ValueError(
            "multiple request fingerprints claim the same immutable workload")
    if candidates:
        request_set, request_dir, work_dir = candidates[0]
        ref = _validate_request_artifact(
            request_dir, request_set, context, pinhole_ref)
        return request_set, ref, work_dir

    request_set = build_request_set(context, pinhole_ref, work_root)
    work_dir = work_root / request_set.fingerprint
    request_dir = work_dir / REQUEST_ARTIFACT_DIR
    ref = llm_lifecycle.publish_request_set(
        request_dir,
        request_set=request_set,
        dataset=args.dataset,
        version=_request_version(context, request_set.fingerprint),
        generator=GENERATOR,
        git_commit=provenance.git_commit(),
        arguments=arguments,
        extra_config={
            key: value for key, value in _request_config(
                context, request_set).items()
            if key != "request_set_fingerprint"
        },
    )
    # The lifecycle helper adds request_set_fingerprint itself.
    _validate_request_artifact(
        request_dir, request_set, context, pinhole_ref)
    return request_set, ref, work_dir


def _normalize_transport_record(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: transport record must be an object")
    if set(value) == {"key", "response"}:
        if not isinstance(value["key"], str) or not value["key"]:
            raise ValueError(f"{where}: key must be non-empty")
        if not isinstance(value["response"], dict):
            raise ValueError(f"{where}: response must be an object")
        return value
    if set(value) == {"key", "error"}:
        if (not isinstance(value["key"], str) or not value["key"]
                or value["error"] is None):
            raise ValueError(f"{where}: invalid error record")
        return value
    if set(value) == {"key", "request", "response", "error"}:
        key = value["key"]
        if not isinstance(key, str) or not key:
            raise ValueError(f"{where}: key must be non-empty")
        if value["error"] is not None:
            return {"key": key, "error": value["error"]}
        if not isinstance(value["response"], dict):
            raise ValueError(f"{where}: successful response must be an object")
        return {"key": key, "response": value["response"]}
    raise ValueError(
        f"{where}: unrecognized transport record keys {sorted(value)}")


def normalize_transport_shard(raw_path: Path) -> Path:
    """Normalize batch and on-demand files to the lifecycle boundary."""
    records = []
    sources = [raw_path, raw_path.with_suffix(".errors.jsonl")]
    for source in sources:
        if not source.exists():
            continue
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ValueError(f"{source}:{line_number}: blank record")
                value = _strict_json_loads(
                    line, f"{source}:{line_number}")
                records.append(_normalize_transport_record(
                    value, f"{source}:{line_number}"))
    normalized = raw_path.with_name(
        raw_path.name.replace("raw_", "normalized_", 1))
    artifact.atomic_write_file(
        normalized,
        b"".join(artifact.canonical_json_bytes(record) + b"\n"
                 for record in records))
    return normalized


def import_transport_history(
        work_dir: Path, request_set: llm_lifecycle.RequestSet) -> Path:
    """Idempotently import every completed or interrupted transport shard."""
    attempts_dir = work_dir / ATTEMPTS_DIR_NAME
    transport_dir = work_dir / "transport"
    if not transport_dir.exists():
        return attempts_dir
    for raw_path in sorted(transport_dir.iterdir()):
        if not raw_path.is_file() or not _RAW_SHARD_RE.fullmatch(raw_path.name):
            continue
        normalized = normalize_transport_shard(raw_path)
        llm_lifecycle.import_transport_results(
            normalized, attempts_dir, request_set)
    return attempts_dir


def _attempts(path: Path) -> tuple[llm_lifecycle.Attempt, ...]:
    return llm_lifecycle.load_attempts(path) if path.exists() else ()


def pending_units(
        request_set: llm_lifecycle.RequestSet,
        attempts: Sequence[llm_lifecycle.Attempt],
    ) -> tuple[llm_lifecycle.RequestUnit, ...]:
    """Units without a valid response; duplicate valid responses fail closed."""
    expected = {unit.key for unit in request_set.units}
    valid = {key: 0 for key in expected}
    for attempt in attempts:
        if attempt.request_set_fingerprint != request_set.fingerprint:
            raise llm_lifecycle.LlmLifecycleError(
                f"attempt {attempt.attempt_id!r} targets another request set")
        if attempt.key not in expected:
            raise llm_lifecycle.LlmLifecycleError(
                f"attempt {attempt.attempt_id!r} has unknown key")
        if attempt.response is None:
            continue
        try:
            # Attempt values are recursively frozen by llm_lifecycle.  Its
            # public serializer is the supported thaw boundary and produces
            # the ordinary dict/list response shape the stage validator sees
            # during canonical compilation.
            response_value = attempt.to_dict()["response"]
            assert response_value is not None
            validate_response(attempt.key, response_value)
        except Exception:
            continue
        valid[attempt.key] += 1
    duplicates = sorted(key for key, count in valid.items() if count > 1)
    if duplicates:
        raise llm_lifecycle.IncompleteCoverageError(
            f"duplicate valid extraction responses for keys {duplicates}")
    return tuple(unit for unit in request_set.units if valid[unit.key] == 0)


def _request_record_bytes(unit: llm_lifecycle.RequestUnit) -> bytes:
    value = unit.to_dict()
    return artifact.canonical_json_bytes({
        "key": unit.key, "request": value["request"]}) + b"\n"


def _request_chunks(
        units: Sequence[llm_lifecycle.RequestUnit]) -> tuple[bytes, ...]:
    chunks = []
    current = bytearray()
    count = 0
    for unit in units:
        line = _request_record_bytes(unit)
        if len(line) > prompts.MAX_BATCH_FILE_SIZE_GCP:
            raise ValueError(f"request {unit.key!r} exceeds provider file limit")
        if current and (len(current) + len(line)
                        > prompts.MAX_BATCH_FILE_SIZE_GCP
                        or count >= MAX_REQUESTS_PER_BATCH):
            chunks.append(bytes(current))
            current = bytearray()
            count = 0
        current.extend(line)
        count += 1
    if current:
        chunks.append(bytes(current))
    return tuple(chunks)


def _total_estimate(paths: Sequence[Path], model: str) -> llm_cost.Estimate:
    _, _, rate_label = llm_cost.rates_for(model)
    total = llm_cost.Estimate(model=model, rate_label=rate_label)
    for path in paths:
        part = llm_cost.estimate_jsonl(path, model=model)
        for field in ("n_requests", "prompt_tokens", "output_tokens",
                      "n_images", "text_chars", "n_large_prompts",
                      "usd_on_demand"):
            setattr(total, field, getattr(total, field) + getattr(part, field))
    total.usd_batch = total.usd_on_demand * llm_cost.BATCH_MULTIPLIER
    return total


def execute_pending(
        args: argparse.Namespace,
        request_set: llm_lifecycle.RequestSet,
        units: Sequence[llm_lifecycle.RequestUnit],
        work_dir: Path,
        attempts_dir: Path) -> None:
    """Execute one resumable attempt for every currently invalid unit."""
    if not units:
        return
    transport_dir = work_dir / "transport"
    transport_dir.mkdir(parents=True, exist_ok=True)
    indices = [int(match.group(1))
               for path in transport_dir.iterdir()
               for pattern in (_PENDING_SHARD_RE, _RAW_SHARD_RE)
               for match in [pattern.fullmatch(path.name)]
               if match]
    next_index = max(indices, default=-1) + 1
    pending_paths = []
    shards = []
    for offset, content in enumerate(_request_chunks(units)):
        index = next_index + offset
        pending_path = transport_dir / f"pending_{index:04d}.jsonl"
        raw_path = transport_dir / f"raw_{index:04d}.jsonl"
        artifact.atomic_create_file(pending_path, content)
        pending_paths.append(pending_path)
        shards.append((index, pending_path, raw_path))

    total = _total_estimate(pending_paths, args.model)
    llm_cost.enforce_limit(
        total,
        limit_usd=args.cost_limit,
        label=(f"{args.dataset} frame-landmark extraction "
               f"({len(units)} pending panoramas)"),
        online=args.online,
        approved=args.approve_cost,
    )
    # The full pending workload was gated once above.  Per-shard VBM checks
    # are marked approved so splitting for provider limits cannot turn one
    # configured approval into several interactive decisions.
    execution = argparse.Namespace(**{
        **vars(args), "approve_cost": True,
    })
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
    for index, pending_path, raw_path in shards:
        try:
            vbm.run_requests(
                execution,
                pending_path,
                raw_path,
                tag=(f"{args.dataset}_{request_set.fingerprint[:12]}_"
                     f"{index:04d}"),
            )
        finally:
            # Preserve useful prefix results even when the provider command
            # raises or is interrupted; the next invocation retries only the
            # keys that still lack one valid response.
            if raw_path.exists() or raw_path.with_suffix(
                    ".errors.jsonl").exists():
                normalized = normalize_transport_shard(raw_path)
                llm_lifecycle.import_transport_results(
                    normalized, attempts_dir, request_set)


def _result_config(
        context: ExtractionContext,
        request_set: llm_lifecycle.RequestSet) -> dict[str, Any]:
    return {
        "orchestration": context.orchestration,
        "build_identity": context.document["build_identity"],
        "purpose": "frame_landmark_extraction",
        "request_set_fingerprint": request_set.fingerprint,
        "n_expected": len(request_set.units),
        "n_successful": len(request_set.units),
        "coverage": "complete",
    }


def ensure_result_artifact(
        args: argparse.Namespace,
        context: ExtractionContext,
        request_set: llm_lifecycle.RequestSet,
        request_ref: artifact.ArtifactRef,
        results: Sequence[llm_lifecycle.CanonicalResult],
        work_dir: Path,
        *, arguments: Sequence[str] = (),
    ) -> artifact.ArtifactRef:
    result_dir = work_dir / RESULT_ARTIFACT_DIR
    expected_bytes = llm_lifecycle.canonical_results_bytes(
        request_set, results)
    if result_dir.exists() or result_dir.is_symlink():
        ref = artifact.open_artifact(
            result_dir,
            expected_kind=llm_lifecycle.RESULT_ARTIFACT_KIND,
            expected_dataset=args.dataset,
            expected_version=_result_version(
                context, request_set.fingerprint))
        _validate_manifest(
            result_dir,
            upstreams=(request_ref,),
            config=_result_config(context, request_set),
            declared_outputs=(llm_lifecycle.CANONICAL_RESULTS_NAME,))
        if (result_dir / llm_lifecycle.CANONICAL_RESULTS_NAME).read_bytes() \
                != expected_bytes:
            raise ValueError(
                f"{result_dir} contains different canonical results")
        return ref
    ref = llm_lifecycle.publish_canonical_results(
        result_dir,
        request_set=request_set,
        request_artifact=request_ref,
        results=results,
        dataset=args.dataset,
        version=_result_version(context, request_set.fingerprint),
        generator=GENERATOR,
        git_commit=provenance.git_commit(),
        arguments=arguments,
        extra_config={
            "orchestration": context.orchestration,
            "build_identity": context.document["build_identity"],
            "purpose": "frame_landmark_extraction",
        },
    )
    _validate_manifest(
        result_dir,
        upstreams=(request_ref,),
        config=_result_config(context, request_set),
        declared_outputs=(llm_lifecycle.CANONICAL_RESULTS_NAME,))
    return ref


def predictions_bytes(
        request_set: llm_lifecycle.RequestSet,
        results: Sequence[llm_lifecycle.CanonicalResult]) -> bytes:
    if [unit.key for unit in request_set.units] != [
            result.key for result in results]:
        raise llm_lifecycle.IncompleteCoverageError(
            "canonical extraction results do not match request order")
    return b"".join(
        artifact.canonical_json_bytes({
            "key": result.key,
            "prediction": _validate_canonical_prediction(result.result),
        }) + b"\n"
        for result in results)


def _frame_config(
        context: ExtractionContext,
        request_set: llm_lifecycle.RequestSet,
        request_ref: artifact.ArtifactRef,
        result_ref: artifact.ArtifactRef) -> dict[str, Any]:
    return {
        "orchestration": context.orchestration,
        "build_identity": context.document["build_identity"],
        "selected_config": context.selected,
        "prompt_sha256": prompts.prompt_sha256(
            context.selected["extraction.prompt_type"]),
        "response_schema_sha256": artifact.sha256_json(
            prompts.response_schema()),
        "request_set_fingerprint": request_set.fingerprint,
        "request_artifact": _ref_identity(request_ref),
        "canonical_results_artifact": _ref_identity(result_ref),
        "coverage": "complete",
        "n_expected": len(request_set.units),
        "n_successful": len(request_set.units),
    }


def _read_predictions(path: Path, expected_keys: Sequence[str]) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank record")
            value = _strict_json_loads(line, f"{path}:{line_number}")
            if not isinstance(value, dict) or set(value) != {
                    "key", "prediction"}:
                raise ValueError(
                    f"{path}:{line_number}: expected exact keys key, prediction")
            value["prediction"] = _validate_canonical_prediction(
                value["prediction"])
            records.append(value)
    keys = [record["key"] for record in records]
    if keys != list(expected_keys) or len(keys) != len(set(keys)):
        raise ValueError(
            f"{path} must exactly cover panorama order {list(expected_keys)}; "
            f"found {keys}")
    return records


def _open_exact_frame_upstream(
        ref: artifact.ArtifactRef, *, kind: str, dataset: str,
        where: str) -> tuple[Path, artifact.ArtifactManifest]:
    path = Path(ref.path)
    try:
        reopened = artifact.open_artifact(
            path, expected_kind=kind, expected_dataset=dataset,
            expected_version=ref.version)
        manifest = artifact.load_manifest(path)
    except (artifact.ArtifactError, OSError) as error:
        raise ValueError(
            f"{where} does not reopen as its exact typed artifact: {error}") \
            from error
    if reopened != ref:
        raise ValueError(f"{where} changed identity after frame publication")
    return path, manifest


def _validate_adoption_report(
        report: Any, *, dataset: str,
        request_set: llm_lifecycle.RequestSet,
        pinhole_ref: artifact.ArtifactRef,
        canonical_bytes: bytes,
        prediction_bytes: bytes,
        results: Sequence[llm_lifecycle.CanonicalResult]) -> None:
    expected_keys = {
        "schema", "dataset", "status", "provider_calls", "spec_sha256",
        "request_set", "pinhole_images", "request_sources",
        "result_sources", "empty_error_sidecars", "attempts",
        "normalization_ledger", "attempt_summary", "sanitation_ledger",
        "canonical_outputs", "publication_plan",
    }
    if not isinstance(report, dict) or set(report) != expected_keys:
        raise ValueError("legacy adoption report has an invalid exact shape")
    try:
        request_report = report["request_set"]
        pinhole_report = report["pinhole_images"]
        attempt_summary = report["attempt_summary"]
        canonical = report["canonical_outputs"]
        publication_plan = report["publication_plan"]
        anchors_ok = (
            report["schema"] == LEGACY_ADOPTION_REPORT_SCHEMA
            and report["dataset"] == dataset
            and report["status"] == "ready_for_explicit_publication"
            and report["provider_calls"] == 0
            and set(request_report) == {
                "fingerprint", "model", "prompt_sha256",
                "response_schema_sha256", "n_expected", "coverage"}
            and request_report["fingerprint"] == request_set.fingerprint
            and request_report["model"] == request_set.model
            and request_report["prompt_sha256"] == hashlib.sha256(
                request_set.system_prompt.encode("utf-8")).hexdigest()
            and request_report["response_schema_sha256"]
                == artifact.sha256_json(
                    request_set.to_dict()["response_schema"])
            and request_report["n_expected"] == len(request_set.units)
            and request_report["coverage"] == "complete"
            and artifact.records_same_artifact(
                pinhole_report["artifact_ref"], pinhole_ref)
            and pinhole_report["content_digest"]
                == pinhole_ref.content_digest
            and pinhole_report["request_media_match"] == "complete"
            and set(attempt_summary) == {
                "n_total", "n_valid", "n_failed_or_invalid",
                "raw_provenance"}
            and attempt_summary["raw_provenance"]
                == "complete_by_source_and_line_digest"
            and set(canonical) == {
                "n_results", "canonical_results_sha256",
                "predictions_sha256", "coverage"}
            and canonical["n_results"] == len(results)
            and canonical["canonical_results_sha256"]
                == hashlib.sha256(canonical_bytes).hexdigest()
            and canonical["predictions_sha256"]
                == hashlib.sha256(prediction_bytes).hexdigest()
            and canonical["coverage"]
                == "complete_unique_in_request_order"
            and set(publication_plan) == {
                "mode", "requires_explicit_write_authorization",
                "provider_calls", "normal_reader_compatibility_fallback",
                "raw_source_policy", "artifacts"}
            and publication_plan["mode"]
                == "transactional_typed_adoption"
            and publication_plan[
                "requires_explicit_write_authorization"] is True
            and publication_plan["provider_calls"] == 0
            and publication_plan[
                "normal_reader_compatibility_fallback"] is False
            and publication_plan["raw_source_policy"]
                == "retain_verbatim_by_reported_sha256"
            and [item.get("kind")
                 for item in publication_plan["artifacts"]] == [
                    paths_lib.PINHOLE_IMAGES,
                    llm_lifecycle.REQUEST_ARTIFACT_KIND,
                    llm_lifecycle.RESULT_ARTIFACT_KIND,
                    paths_lib.FRAME_LANDMARKS,
                ]
            and isinstance(report["normalization_ledger"], list)
            and isinstance(report["sanitation_ledger"], list))
    except (AttributeError, KeyError, TypeError):
        anchors_ok = False
    if not anchors_ok:
        raise ValueError(
            "legacy adoption report lacks its zero-provider raw-source proof")

    def require_source_digests(records: Any, where: str) -> None:
        if not isinstance(records, list) or not records:
            raise ValueError(f"legacy adoption report has no {where}")
        for index, record in enumerate(records):
            digest = record.get("sha256") if isinstance(record, dict) else None
            if (not isinstance(record, dict)
                    or not isinstance(digest, str)
                    or not _DIGEST_RE.fullmatch(digest)
                    or not isinstance(record.get("path"), str)
                    or not record["path"]):
                raise ValueError(
                    f"legacy adoption report {where}[{index}] lacks raw bytes "
                    "provenance")

    require_source_digests(report["request_sources"], "request_sources")
    require_source_digests(report["result_sources"], "result_sources")
    attempts = report["attempts"]
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("legacy adoption report has no preserved attempts")
    attempt_ids = set()
    for index, attempt in enumerate(attempts):
        raw_line_digest = (attempt.get("raw_line_sha256")
                           if isinstance(attempt, dict) else None)
        if (not isinstance(attempt, dict)
                or not isinstance(attempt.get("attempt_id"), str)
                or not attempt["attempt_id"]
                or not isinstance(raw_line_digest, str)
                or not _DIGEST_RE.fullmatch(raw_line_digest)):
            raise ValueError(
                f"legacy adoption report attempts[{index}] lacks raw-line "
                "provenance")
        attempt_ids.add(attempt["attempt_id"])
    if (len(attempt_ids) != len(attempts)
            or attempt_summary["n_total"] != len(attempts)
            or attempt_summary["n_valid"] != len(results)
            or attempt_summary["n_failed_or_invalid"]
                != len(attempts) - len(results)):
        raise ValueError(
            "legacy adoption attempt summary is internally inconsistent")
    if any(result.attempt_id not in attempt_ids for result in results):
        raise ValueError(
            "canonical results are not bound to preserved legacy attempts")


def _validate_existing_adopted_frame_artifact(
        args: argparse.Namespace,
        context: ExtractionContext,
        pinhole_ref: artifact.ArtifactRef,
        destination: Path,
        frame_ref: artifact.ArtifactRef,
        frame_manifest: artifact.ArtifactManifest,
        result_ref: artifact.ArtifactRef,
        frame_config: dict[str, Any]) -> artifact.ArtifactRef:
    expected_frame_keys = {
        "purpose", "orchestration", "build_identity", "prompt_sha256",
        "response_schema_sha256", "request_set_fingerprint",
        "request_artifact", "canonical_results_artifact", "coverage",
        "n_expected", "n_successful", "legacy_adoption_schema",
        "legacy_adoption_report_sha256",
    }
    if set(frame_config) != expected_frame_keys:
        raise ValueError(
            f"{destination} has an invalid adopted-frame config shape")
    report_digest = frame_config["legacy_adoption_report_sha256"]
    marker = legacy_adoption_marker(report_digest)
    fingerprint = frame_config["request_set_fingerprint"]
    if (frame_config["legacy_adoption_schema"]
            != LEGACY_ADOPTION_CONFIG_SCHEMA
            or frame_config["purpose"] != "frame_landmark_extraction"
            or frame_config["orchestration"] != context.orchestration
            or frame_config["build_identity"]
                != context.document["build_identity"]
            or frame_config["prompt_sha256"] != prompts.prompt_sha256(
                context.selected["extraction.prompt_type"])
            or frame_config["response_schema_sha256"]
                != artifact.sha256_json(prompts.response_schema())
            or frame_config["coverage"] != "complete"
            or frame_config["n_expected"] != len(context.frames)
            or frame_config["n_successful"] != len(context.frames)
            or not isinstance(fingerprint, str)
            or not _DIGEST_RE.fullmatch(fingerprint)):
        raise ValueError(
            f"{destination} has a different adopted extraction contract")

    request_identity = _validate_ref_identity(
        frame_config["request_artifact"],
        kind=llm_lifecycle.REQUEST_ARTIFACT_KIND,
        dataset=args.dataset,
        where="manifest.config.request_artifact")
    result_identity = _validate_ref_identity(
        frame_config["canonical_results_artifact"],
        kind=llm_lifecycle.RESULT_ARTIFACT_KIND,
        dataset=args.dataset,
        where="manifest.config.canonical_results_artifact")
    if result_identity != _ref_identity(result_ref):
        raise ValueError(
            "adopted frame config and canonical-result upstream differ")
    if (request_identity["version"] != _request_version(context, fingerprint)
            or result_identity["version"] != _result_version(
                context, fingerprint)):
        raise ValueError(
            "adopted LLM artifact versions do not match the request "
            "fingerprint")

    result_path, result_manifest = _open_exact_frame_upstream(
        result_ref, kind=llm_lifecycle.RESULT_ARTIFACT_KIND,
        dataset=args.dataset, where="adopted canonical-result upstream")
    if (len(result_manifest.upstreams) != 1
            or result_manifest.upstreams[0].kind
                != llm_lifecycle.REQUEST_ARTIFACT_KIND
            or result_manifest.upstreams[0].dataset != args.dataset):
        raise ValueError(
            "adopted canonical result must bind exactly one typed request")
    request_ref = result_manifest.upstreams[0]
    if request_identity != _ref_identity(request_ref):
        raise ValueError(
            "adopted frame config and result request upstream differ")
    request_path, request_manifest = _open_exact_frame_upstream(
        request_ref, kind=llm_lifecycle.REQUEST_ARTIFACT_KIND,
        dataset=args.dataset, where="adopted request upstream")

    if any(manifest.generator != LEGACY_ADOPTION_GENERATOR
           for manifest in (
               request_manifest, result_manifest, frame_manifest)):
        raise ValueError(
            "adopted lineage was not published by the adoption generator")
    if (len({manifest.git_commit for manifest in (
                request_manifest, result_manifest, frame_manifest)}) != 1
            or len({manifest.arguments for manifest in (
                request_manifest, result_manifest, frame_manifest)}) != 1):
        raise ValueError(
            "adopted lineage does not share one publisher invocation")

    request_set = _validate_recorded_request_payloads(
        request_path, context, pinhole_ref, fingerprint)

    request_config = dict(request_manifest.config)
    expected_request_keys = {
        "purpose", "orchestration", "build_identity",
        "request_set_fingerprint", "n_expected", "legacy_adoption_schema",
        "legacy_adoption_report_sha256", "legacy_adoption",
    }
    if set(request_config) != expected_request_keys:
        raise ValueError("adopted request manifest has an invalid config shape")
    proof = request_config["legacy_adoption"]
    if (not isinstance(proof, dict)
            or set(proof) != {
                "schema", "verification_report_sha256",
                "verification_report"}
            or proof["schema"] != LEGACY_ADOPTION_PROOF_SCHEMA
            or proof["verification_report_sha256"] != report_digest
            or artifact.sha256_json(proof["verification_report"])
                != report_digest):
        raise ValueError(
            "adopted request manifest has an invalid verification proof")
    expected_request_config = {
        "purpose": "frame_landmark_extraction",
        "orchestration": context.orchestration,
        "build_identity": context.document["build_identity"],
        "request_set_fingerprint": fingerprint,
        **marker,
        "n_expected": len(request_set.units),
        "legacy_adoption": proof,
    }
    _validate_manifest(
        request_path, upstreams=(pinhole_ref,),
        config=expected_request_config,
        declared_outputs=(
            llm_lifecycle.REQUEST_SET_NAME, llm_lifecycle.REQUESTS_NAME))

    canonical_path = result_path / llm_lifecycle.CANONICAL_RESULTS_NAME
    results = llm_lifecycle.load_canonical_results(
        canonical_path, request_set)
    canonical_bytes = llm_lifecycle.canonical_results_bytes(
        request_set, results)
    if canonical_path.read_bytes() != canonical_bytes:
        raise ValueError(
            "adopted canonical-result payload is not exact canonical JSONL")
    expected_result_config = {
        "purpose": "frame_landmark_extraction",
        "orchestration": context.orchestration,
        "build_identity": context.document["build_identity"],
        "request_set_fingerprint": fingerprint,
        "n_expected": len(request_set.units),
        "n_successful": len(results),
        "coverage": "complete",
        **marker,
    }
    _validate_manifest(
        result_path, upstreams=(request_ref,),
        config=expected_result_config,
        declared_outputs=(llm_lifecycle.CANONICAL_RESULTS_NAME,))

    expected_predictions = predictions_bytes(request_set, results)
    prediction_path = destination / PREDICTIONS_NAME
    if prediction_path.read_bytes() != expected_predictions:
        raise ValueError(
            "adopted frame predictions differ from canonical results")
    _read_predictions(prediction_path, context.stems)
    expected_frame_config = {
        "purpose": "frame_landmark_extraction",
        "orchestration": context.orchestration,
        "build_identity": context.document["build_identity"],
        "prompt_sha256": prompts.prompt_sha256(
            context.selected["extraction.prompt_type"]),
        "response_schema_sha256": artifact.sha256_json(
            prompts.response_schema()),
        "request_set_fingerprint": fingerprint,
        "request_artifact": _ref_identity(request_ref),
        "canonical_results_artifact": _ref_identity(result_ref),
        "coverage": "complete",
        "n_expected": len(request_set.units),
        "n_successful": len(results),
        **marker,
    }
    _validate_manifest(
        destination, upstreams=(pinhole_ref, result_ref),
        config=expected_frame_config,
        declared_outputs=(PREDICTIONS_NAME,))
    _validate_adoption_report(
        proof["verification_report"], dataset=args.dataset,
        request_set=request_set, pinhole_ref=pinhole_ref,
        canonical_bytes=canonical_bytes,
        prediction_bytes=expected_predictions, results=results)
    return frame_ref


def validate_existing_frame_artifact(
        args: argparse.Namespace,
        context: ExtractionContext,
        pinhole_ref: artifact.ArtifactRef) -> artifact.ArtifactRef:
    destination = Path(args.output_dir)
    ref = artifact.open_artifact(
        destination,
        expected_kind=paths_lib.FRAME_LANDMARKS,
        expected_dataset=args.dataset,
        expected_version=context.frame_version)
    manifest = artifact.load_manifest(destination)
    if manifest.declared_outputs != (PREDICTIONS_NAME,):
        raise ValueError(
            f"{destination} must declare only {PREDICTIONS_NAME}")
    if len(manifest.upstreams) != 2 or manifest.upstreams[0] != pinhole_ref:
        raise ValueError(
            f"{destination} does not bind the exact pinhole artifact once")
    result_ref = manifest.upstreams[1]
    if (result_ref.kind != llm_lifecycle.RESULT_ARTIFACT_KIND
            or result_ref.dataset != args.dataset):
        raise ValueError(
            f"{destination} does not bind a canonical LLM result artifact")
    config = dict(manifest.config)
    if (config.get("legacy_adoption_schema")
            == LEGACY_ADOPTION_CONFIG_SCHEMA):
        return _validate_existing_adopted_frame_artifact(
            args, context, pinhole_ref, destination, ref, manifest,
            result_ref, config)
    expected_keys = {
        "orchestration", "build_identity", "selected_config",
        "prompt_sha256", "response_schema_sha256",
        "request_set_fingerprint", "request_artifact",
        "canonical_results_artifact", "coverage", "n_expected",
        "n_successful",
    }
    if set(config) != expected_keys:
        raise ValueError(f"{destination} has an invalid manifest config shape")
    if (config["orchestration"] != context.orchestration
            or config["build_identity"] != context.document["build_identity"]
            or config["selected_config"] != context.selected
            or config["prompt_sha256"] != prompts.prompt_sha256(
                context.selected["extraction.prompt_type"])
            or config["response_schema_sha256"] != artifact.sha256_json(
                prompts.response_schema())
            or config["coverage"] != "complete"
            or config["n_expected"] != len(context.frames)
            or config["n_successful"] != len(context.frames)
            or not isinstance(config["request_set_fingerprint"], str)
            or not _DIGEST_RE.fullmatch(config["request_set_fingerprint"])):
        raise ValueError(f"{destination} has a different extraction contract")
    request_identity = _validate_ref_identity(
        config["request_artifact"],
        kind=llm_lifecycle.REQUEST_ARTIFACT_KIND,
        dataset=args.dataset,
        where="manifest.config.request_artifact")
    result_identity = _validate_ref_identity(
        config["canonical_results_artifact"],
        kind=llm_lifecycle.RESULT_ARTIFACT_KIND,
        dataset=args.dataset,
        where="manifest.config.canonical_results_artifact")
    if result_identity != _ref_identity(result_ref):
        raise ValueError(
            "manifest config and upstream canonical-result identities differ")
    fingerprint = config["request_set_fingerprint"]
    if (request_identity["version"] != _request_version(context, fingerprint)
            or result_identity["version"] != _result_version(
                context, fingerprint)):
        raise ValueError(
            "manifest LLM artifact versions do not match its request "
            "fingerprint")

    result_path, result_manifest = _open_exact_frame_upstream(
        result_ref, kind=llm_lifecycle.RESULT_ARTIFACT_KIND,
        dataset=args.dataset, where="canonical-result upstream")
    if (len(result_manifest.upstreams) != 1
            or result_manifest.upstreams[0].kind
                != llm_lifecycle.REQUEST_ARTIFACT_KIND
            or result_manifest.upstreams[0].dataset != args.dataset):
        raise ValueError(
            "canonical result must bind exactly one typed request")
    request_ref = result_manifest.upstreams[0]
    if request_identity != _ref_identity(request_ref):
        raise ValueError(
            "frame config and result request upstream identities differ")
    request_path, request_manifest = _open_exact_frame_upstream(
        request_ref, kind=llm_lifecycle.REQUEST_ARTIFACT_KIND,
        dataset=args.dataset, where="request upstream")
    if any(item.generator != GENERATOR for item in (
            request_manifest, result_manifest, manifest)):
        raise ValueError(
            "normal extraction lineage was not published by its producer")

    request_set = _validate_recorded_request_payloads(
        request_path, context, pinhole_ref, fingerprint)
    _validate_manifest(
        request_path, upstreams=(pinhole_ref,),
        config=_request_config(context, request_set),
        declared_outputs=(
            llm_lifecycle.REQUEST_SET_NAME, llm_lifecycle.REQUESTS_NAME))

    canonical_path = result_path / llm_lifecycle.CANONICAL_RESULTS_NAME
    results = llm_lifecycle.load_canonical_results(
        canonical_path, request_set)
    canonical_bytes = llm_lifecycle.canonical_results_bytes(
        request_set, results)
    if canonical_path.read_bytes() != canonical_bytes:
        raise ValueError(
            "canonical-result payload is not exact canonical JSONL")
    _validate_manifest(
        result_path, upstreams=(request_ref,),
        config=_result_config(context, request_set),
        declared_outputs=(llm_lifecycle.CANONICAL_RESULTS_NAME,))

    prediction_path = destination / PREDICTIONS_NAME
    expected_predictions = predictions_bytes(request_set, results)
    if prediction_path.read_bytes() != expected_predictions:
        raise ValueError(
            "frame predictions differ from canonical results")
    _read_predictions(prediction_path, context.stems)
    _validate_manifest(
        destination, upstreams=(pinhole_ref, result_ref),
        config=_frame_config(
            context, request_set, request_ref, result_ref),
        declared_outputs=(PREDICTIONS_NAME,))
    return ref


def publish_frame_artifact(
        args: argparse.Namespace,
        context: ExtractionContext,
        pinhole_ref: artifact.ArtifactRef,
        request_set: llm_lifecycle.RequestSet,
        request_ref: artifact.ArtifactRef,
        result_ref: artifact.ArtifactRef,
        results: Sequence[llm_lifecycle.CanonicalResult],
        *, arguments: Sequence[str] = (),
    ) -> artifact.ArtifactRef:
    destination = Path(args.output_dir)
    content = predictions_bytes(request_set, results)
    config = _frame_config(
        context, request_set, request_ref, result_ref)
    with publication.published_artifact(
            destination,
            kind=paths_lib.FRAME_LANDMARKS,
            dataset=args.dataset,
            version=context.frame_version,
            generator=GENERATOR,
            git_commit=provenance.git_commit(),
            arguments=arguments,
            upstreams=(pinhole_ref, result_ref),
            config=config,
            artifact_identity=getattr(args, "artifact_identity", None),
            recipe=artifact_recipe.load(
                getattr(args, "artifact_recipe", None)),
            declared_outputs=(PREDICTIONS_NAME,)) as builder:
        artifact.atomic_write_file(
            builder.output_path(PREDICTIONS_NAME), content)
    assert builder.artifact_ref is not None
    return builder.artifact_ref


def prepare_adoption(
        args: argparse.Namespace, *, arguments: Sequence[str] = (),
        ) -> tuple[artifact.ArtifactRef, artifact.ArtifactRef]:
    """Publish only the deterministic PINHOLE and REQUEST prerequisites.

    This explicit mode succeeds before cost enforcement or provider execution.
    It is resumable, but refuses to prepare beside an already-published final
    frame artifact.
    """
    context = load_context(args)
    destination = Path(args.output_dir)
    if destination.exists() or destination.is_symlink():
        raise ValueError(
            "--prepare_adoption requires an unpublished frame output")
    pinhole_ref = ensure_pinhole_artifact(
        args, context, arguments=arguments)
    _, request_ref, _ = ensure_request_artifact(
        args, context, pinhole_ref, arguments=arguments)
    return pinhole_ref, request_ref


def run(args: argparse.Namespace, *, arguments: Sequence[str] = ()) \
        -> tuple[artifact.ArtifactRef, artifact.ArtifactRef]:
    """Execute the complete extraction producer contract."""
    context = load_context(args)
    pinhole_ref = ensure_pinhole_artifact(
        args, context, arguments=arguments)
    destination = Path(args.output_dir)
    if destination.exists() or destination.is_symlink():
        return pinhole_ref, validate_existing_frame_artifact(
            args, context, pinhole_ref)

    request_set, request_ref, work_dir = ensure_request_artifact(
        args, context, pinhole_ref, arguments=arguments)
    attempts_dir = import_transport_history(work_dir, request_set)
    pending = pending_units(request_set, _attempts(attempts_dir))
    execute_pending(
        args, request_set, pending, work_dir, attempts_dir)
    attempts = _attempts(attempts_dir)
    results = llm_lifecycle.compile_canonical_results(
        request_set, attempts, validate_response)
    result_ref = ensure_result_artifact(
        args, context, request_set, request_ref, results, work_dir,
        arguments=arguments)
    frame_ref = publish_frame_artifact(
        args, context, pinhole_ref, request_set, request_ref, result_ref,
        results, arguments=arguments)
    return pinhole_ref, frame_ref


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_base", type=Path, required=True)
    parser.add_argument("--pinhole_output_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--build_config", type=Path, required=True)
    parser.add_argument("--orchestration_config_digest", required=True)
    # The identity the orchestrator computed for this stage's artifact; see
    # `pipeline.stage_identity_flags`. Optional so a producer stays runnable
    # by hand -- the artifact is then honestly unattributed.
    parser.add_argument("--artifact_identity", default=None)
    parser.add_argument("--artifact_recipe", default=None,
                        help="path to the resolved stage config and build "
                             "inputs this artifact should record, written by "
                             "`pipeline run`")
    parser.add_argument(
        "--prepare_adoption", action="store_true",
        help=("publish only deterministic pinhole and request artifacts; "
              "make no provider call and do not enter cost enforcement"))

    execution = parser.add_argument_group("model transport")
    execution.add_argument("--online", action="store_true")
    execution.add_argument("--gcs_prefix", default=None)
    execution.add_argument("--parallel", type=int, default=8)
    execution.add_argument("--poll_interval", type=int, default=120)
    execution.add_argument("--cost_limit", type=float, required=True)
    execution.add_argument("--approve_cost", action="store_true")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    args = make_parser().parse_args()
    if args.prepare_adoption:
        pinhole_ref, request_ref = prepare_adoption(
            args, arguments=tuple(sys.argv))
        print(f"pinhole_images: {pinhole_ref.path}")
        print(f"llm_requests: {request_ref.path}")
    else:
        pinhole_ref, frame_ref = run(args, arguments=tuple(sys.argv))
        print(f"pinhole_images: {pinhole_ref.path}")
        print(f"frame_landmarks: {frame_ref.path}")


if __name__ == "__main__":
    main()
