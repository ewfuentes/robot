"""Build and canonically finish a source-bound semantic audit in one workflow.

The producer validates exactly one completed ``object_tracks`` artifact and
publishes a content-addressed request snapshot in mutable orchestration state.
Provider output is imported into atomically created immutable attempt shards.
A completed ``semantic_audits`` artifact is published only after every
expected request
has exactly one schema-valid successful response.

There is no run-directory compatibility mode.  Source and destination
artifact directories are explicit, and neither operation mutates a published
artifact.
"""

import argparse
import base64
import dataclasses
import hashlib
import html
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.farfield import (
    artifact,
    artifact_recipe,
    build_config,
    dataset,
    llm_lifecycle as llm,
    paths as paths_lib,
    provenance,
    publication,
)
from experimental.overhead_matching.swag.farfield.extraction import (
    vertex_batch_manager as vbm,
)
from experimental.overhead_matching.swag.farfield.tracking import (
    keyframe_viewer as kv,
    semantic_audit as sa,
)
from experimental.overhead_matching.swag.farfield.viewers import (
    page as page_lib,
)

GENERATOR = ("//experimental/overhead_matching/swag/farfield/tracking"
             ":audit_requests")
REQUEST_ARTIFACT_KIND = "semantic_audit_requests"
AUDIT_META_SCHEMA = "farfield_semantic_audit_meta/v2"
AUDIT_META_NAME = "audit_meta.json"
SETTINGS_NAME = "settings.json"
ATTEMPTS_DIR_NAME = llm.ATTEMPTS_DIR_NAME
AUDIT_CONFIG_KEYS = (
    "artifacts.semantic_audits_version",
    "audit.model",
    "audit.min_supports",
    "audit.thinking_level",
    "audit.max_support_chips",
    "audit.max_context_chips",
    "audit.max_description_samples",
    "audit.chip_height_px",
    "ingest.fov_deg",
    "ingest.seam_gap_norm",
    "ingest.seam_min_y_iou",
    "execution.llm_transport",
    "execution.batch_gcs_prefix",
    "execution.approve_cost",
    "cost.limit_usd",
)

# Pages go through the one shared farfield.viewers.page helper (one CSS, a
# provenance footer on every page); only the classes specific to this viewer
# live here.
EXTRA_STYLE = """
pre{white-space:pre-wrap;font-size:12.5px;max-width:1100px}
.chip{display:inline-block;background:#222;margin:3px;padding:4px;
border-radius:4px;vertical-align:top;max-width:340px;font-size:12px}
.chip img{height:200px;display:block;border-radius:3px}
h2{margin-top:40px;border-top:1px solid #333;padding-top:16px}
"""


def render_all_chips(dossiers, frames_by_idx, dataset_base, chips_dir,
                     chip_height_px, fov_deg) -> dict:
    """Render every selected chip, decoding each pano once.
    Returns {(track_id, t, is_context): chip_path}."""
    chips_dir.mkdir(parents=True, exist_ok=True)
    by_keyframe = defaultdict(list)
    for d in dossiers:
        for e in d["chip_entries"]:
            by_keyframe[e["keyframe"]].append((d, e))

    probe = Image.open(
        dataset_base / "panorama"
        / f"{frames_by_idx[min(frames_by_idx)].pano_stem}.jpg")
    pano_w, pano_h = probe.size

    paths = {}
    for keyframe in sorted(by_keyframe):
        frame = frames_by_idx.get(keyframe)
        if frame is None:
            continue
        pano = np.asarray(Image.open(
            dataset_base / "panorama" / f"{frame.pano_stem}.jpg"))
        for d, e in by_keyframe[keyframe]:
            det_box, mask_box = sa.chip_boxes_for_entry(
                e, e["obs"], pano_w, pano_h, fov_deg)
            out = chips_dir / (f"T{d['track_id']}_t{e['t']:04d}"
                               f"{'_ctx' if e['is_context'] else ''}.jpg")
            sa.render_chip(pano, det_box, mask_box, out, chip_height_px)
            paths[(d["track_id"], e["t"], e["is_context"])] = out
    return paths


def write_preview(out_dir, dossiers, chip_paths, texts):
    preview = out_dir / "preview"
    preview.mkdir(exist_ok=True)
    parts = [
        "<p>Exact prompt text + images per request. System prompt shown "
        "once.</p>",
        "<h2>system prompt (all requests)</h2>",
        f"<pre>{html.escape(sa.SYSTEM_PROMPT)}</pre>"]
    for d in dossiers:
        # The id anchors the review page's "request preview" links.
        parts.append(f"<h2 id='T{d['track_id']}'>T{d['track_id']}</h2>")
        parts.append(f"<pre>{html.escape(texts[d['track_id']])}</pre>")
        for i, e in enumerate(d["chip_entries"], 1):
            path = chip_paths.get((d["track_id"], e["t"], e["is_context"]))
            if path is None:
                continue
            rel = f"../chips/{path.name}"
            caption = html.escape(sa.chip_caption(e, i))
            parts.append(f"<div class='chip'><img src='{rel}' "
                         f"loading='lazy'>{caption}</div>")
    artifact.atomic_write_file(
        preview / "index.html",
        page_lib.page(
            f"semantic audit requests: {len(dossiers)} tracks",
            "\n".join(parts), generator=GENERATOR,
            extra_style=EXTRA_STYLE).encode("utf-8"))
    return preview / "index.html"


def settings_record(args, paths, source_path, source_artifact,
                    ingest_params, audit_config: dict,
                    n_tracks_total: int, n_eligible: int,
                    n_requests: int) -> dict:
    """Return every setting that can change a request or its eligibility."""
    settings = {
        "generator": GENERATOR,
        "git_commit": provenance.git_commit(),
        "argv": list(sys.argv),
        "dataset": paths.dataset,
        "model": args.model,
        "thinking_level": args.thinking_level,
        "min_supports": args.min_supports,
        "system_prompt_sha256": hashlib.sha256(
            sa.SYSTEM_PROMPT.encode()).hexdigest(),
        "audit_config": audit_config,
        "classifier": source_artifact["config"],
        "ingest": dataclasses.asdict(ingest_params),
        "source_tracks_file": source_path.name,
        "n_tracks_total": n_tracks_total,
        "n_eligible": n_eligible,
        "n_requests": n_requests,
    }
    return settings


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _load_json(path: Path):
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value!r}")))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot read strict JSON from {path}: {error}") \
            from error


def source_artifact_id(ref: artifact.ArtifactRef) -> str:
    """Stable human-readable identity retained in audit join metadata."""
    return (f"{paths_lib.OBJECT_TRACKS}:{ref.dataset}:{ref.version}"
            f"@sha256:{ref.content_digest}")


def load_source_tracks(tracks_dir: Path, dataset_name: str):
    """Validate one immutable tracks artifact and its one canonical payload."""
    tracks_dir = Path(tracks_dir)
    source_ref = artifact.open_artifact(
        tracks_dir, expected_kind=paths_lib.OBJECT_TRACKS,
        expected_dataset=dataset_name)
    candidates = sorted(tracks_dir.glob("tracks_*.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"{tracks_dir} must contain exactly one tracks_*.json file; "
            f"found {len(candidates)}")
    source_path = candidates[0]
    source = _load_json(source_path)
    if not isinstance(source, dict):
        raise ValueError(f"{source_path} must contain a JSON object")
    tracks = source.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError(f"{source_path}: tracks must be a list")
    seen = set()
    for index, track in enumerate(tracks):
        if not isinstance(track, dict):
            raise ValueError(f"{source_path}: tracks[{index}] must be an object")
        track_id = track.get("track_id")
        if (isinstance(track_id, bool) or not isinstance(track_id, int)
                or track_id < 0):
            raise ValueError(
                f"{source_path}: tracks[{index}].track_id must be a "
                "nonnegative integer")
        if track_id in seen:
            raise ValueError(f"{source_path}: duplicate track_id {track_id}")
        seen.add(track_id)
    range_value = source.get("range")
    if not isinstance(range_value, dict) or not isinstance(
            range_value.get("name"), str) or not range_value["name"]:
        raise ValueError(f"{source_path}: range.name must be non-empty")
    # This also rejects an absent or stale classifier configuration.
    kv.recorded_config(source)
    return source_ref, source_path, source, tracks, range_value["name"]


def _chip_output_names(dossiers) -> list[str]:
    names = []
    for dossier in dossiers:
        for entry in dossier["chip_entries"]:
            suffix = "_ctx" if entry["is_context"] else ""
            names.append(
                f"chips/T{dossier['track_id']}_t{entry['t']:04d}{suffix}.jpg")
    if len(names) != len(set(names)):
        raise ValueError("semantic-audit chip identities are not unique")
    return names


def _panorama_digest(dossiers, frames_by_idx, dataset_base: Path) -> str:
    paths = set()
    for dossier in dossiers:
        for entry in dossier["chip_entries"]:
            frame = frames_by_idx.get(entry["keyframe"])
            if frame is None:
                raise ValueError(
                    f"no panorama frame for keyframe {entry['keyframe']}")
            paths.add(
                Path(dataset_base) / "panorama" / f"{frame.pano_stem}.jpg")
    return artifact.sha256_json({
        path.name: artifact.sha256_file(path)
        for path in sorted(paths, key=lambda item: item.name)
    })


def _request_meta(dossier, range_name, source_track, chip_paths) -> dict:
    return {
        "track_id": dossier["track_id"],
        "source_track_sha256": artifact.sha256_json(source_track),
        "range": range_name,
        "birth_keyframe": dossier["birth_keyframe"],
        "n_supports": dossier["n_supports"],
        "support_obs_by_t": {
            str(entry["t"]): entry["obs"].obs_id
            for entry in dossier["supports"]
        },
        "chips": [
            f"chips/{chip_paths[(dossier['track_id'], entry['t'],
                                  entry['is_context'])].name}"
            for entry in dossier["chip_entries"]
        ],
    }


def build_request_artifact(args, paths, source_ref, source_path,
                           source, tracks, range_name,
                           ingest_result) -> artifact.ArtifactRef:
    """Publish the exact request set, media, and join metadata atomically."""
    cfg = sa.AuditConfig(
        min_supports=args.min_supports,
        max_support_chips=args.max_support_chips,
        max_context_chips=args.max_context_chips,
        max_description_samples=args.max_description_samples,
        chip_height_px=args.chip_height_px,
        thinking_level=args.thinking_level,
        classifier=kv.recorded_config(source))
    observations = {item.obs_id: item
                    for item in ingest_result.observations}
    frames_by_idx = {frame.frame_idx: frame for frame in ingest_result.frames}
    dossiers = []
    by_track_id = {}
    for track in tracks:
        by_track_id[track["track_id"]] = track
        if not track.get("records"):
            raise ValueError(
                f"source track {track['track_id']} has no records")
        missing_observations = {
            support.get("obs_id")
            for record in track["records"]
            for support in record.get("supports", [])
            if support.get("obs_id") not in observations
        }
        if missing_observations:
            raise ValueError(
                f"source track {track['track_id']} refers to observations "
                f"outside its bound frame_landmarks artifact: "
                f"{sorted(missing_observations, key=str)}")
        dossier = sa.build_dossier(track, observations, cfg)
        if dossier["n_supports"] >= cfg.min_supports:
            dossiers.append(dossier)
    dossiers.sort(key=lambda item: (-item["n_supports"], item["track_id"]))
    if not dossiers:
        raise ValueError("no source tracks satisfy the recorded audit gate")
    print(f"{len(dossiers)} eligible tracks (>= {cfg.min_supports} supports) "
          f"of {len(tracks)} total")

    audit_config = {key: value for key, value in dataclasses.asdict(cfg).items()
                    if key != "classifier"}
    settings = settings_record(
        args, paths, source_path, source, args.ingest_params,
        audit_config, len(tracks), len(dossiers), len(dossiers))
    chip_names = _chip_output_names(dossiers)
    declared = sorted([
        AUDIT_META_NAME,
        SETTINGS_NAME,
        llm.REQUEST_SET_NAME,
        llm.REQUESTS_NAME,
        "preview/index.html",
        *chip_names,
    ])
    with artifact.ArtifactDirectoryBuilder(
            args.output_dir,
            kind=REQUEST_ARTIFACT_KIND,
            dataset=paths.dataset,
            version=args.output_version,
            generator=GENERATOR,
            git_commit=settings["git_commit"],
            arguments=sys.argv[1:],
            upstreams=(source_ref, ingest_result.frame_landmarks_ref),
            config={
                "phase": "requests",
                "settings": settings,
                "build_identity": getattr(args, "build_identity", None),
                "orchestration": getattr(args, "orchestration", None),
                "resolved_stage_config": getattr(
                    args, "resolved_stage_config", None),
            },
            declared_outputs=declared) as builder:
        out_dir = builder.staging_dir
        chip_paths = render_all_chips(
            dossiers, frames_by_idx, paths.dataset_base, out_dir / "chips",
            args.chip_height_px, args.fov_deg)
        if len(chip_paths) != len(chip_names):
            raise ValueError(
                "not every declared audit chip could be rendered")
        units = []
        meta_requests = {}
        texts = {}
        for dossier in dossiers:
            track_id = dossier["track_id"]
            key = f"T{track_id}"
            text = sa.render_dossier_text(dossier)
            texts[track_id] = text
            chips = []
            for index, entry in enumerate(dossier["chip_entries"], 1):
                chip_path = chip_paths[
                    (track_id, entry["t"], entry["is_context"])]
                chips.append((sa.chip_caption(entry, index),
                              base64.b64encode(chip_path.read_bytes()).decode()))
            request = sa.build_request(key, text, chips, cfg)["request"]
            request_meta = _request_meta(
                dossier, range_name, by_track_id[track_id], chip_paths)
            meta_requests[key] = request_meta
            units.append(llm.RequestUnit(key, request, request_meta))

        request_set = llm.RequestSet.create(
            stage="semantic_audit",
            model=args.model,
            system_prompt=sa.SYSTEM_PROMPT,
            response_schema=sa.get_provider_audit_schema(),
            media_settings={
                **audit_config,
                "fov_deg": args.fov_deg,
                "seam_gap_norm": args.seam_gap_norm,
                "seam_min_y_iou": args.seam_min_y_iou,
            },
            input_digests={
                "source_tracks_file": artifact.sha256_file(source_path),
                "frame_landmarks": ingest_result.frame_landmarks_ref.content_digest,
                "panorama_subset": _panorama_digest(
                    dossiers, frames_by_idx, paths.dataset_base),
            },
            upstreams=(source_ref, ingest_result.frame_landmarks_ref),
            units=units)
        audit_meta = {
            "schema": AUDIT_META_SCHEMA,
            "source_tracks": {
                "artifact_id": source_artifact_id(source_ref),
                "file": source_path.name,
                "sha256": artifact.sha256_file(source_path),
            },
            "requests": meta_requests,
        }
        artifact.atomic_write_json(
            builder.output_path(llm.REQUEST_SET_NAME), request_set.to_dict())
        request_lines = b"".join(
            artifact.canonical_json_bytes({
                "key": unit.key, "request": unit.to_dict()["request"]})
            + b"\n" for unit in request_set.units)
        artifact.atomic_write_file(
            builder.output_path(llm.REQUESTS_NAME), request_lines)
        artifact.atomic_write_json(
            builder.output_path(AUDIT_META_NAME), audit_meta)
        artifact.atomic_write_json(
            builder.output_path(SETTINGS_NAME), settings)
        write_preview(out_dir, dossiers, chip_paths, texts)
    assert builder.artifact_ref is not None
    return builder.artifact_ref


def _validate_weight(value, label):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0.0 <= value <= 1.0):
        raise ValueError(f"{label} must be within [0, 1]")


def _require_exact_response_shape(raw, canonical, path="audit"):
    """Reject fields Pydantic would otherwise ignore as unknown data."""
    if isinstance(canonical, dict):
        if not isinstance(raw, dict) or set(raw) != set(canonical):
            raw_keys = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
            raise ValueError(
                f"{path} fields must be exactly {sorted(canonical)}; "
                f"found {raw_keys}")
        for key, value in canonical.items():
            _require_exact_response_shape(raw[key], value, f"{path}.{key}")
    elif isinstance(canonical, list):
        if not isinstance(raw, list) or len(raw) != len(canonical):
            raise ValueError(f"{path} must preserve the response list shape")
        for index, value in enumerate(canonical):
            _require_exact_response_shape(
                raw[index], value, f"{path}[{index}]")


def _validate_audit_response(key: str, response: dict) -> dict:
    """Strict provider response -> deterministic canonical TrackAudit."""
    try:
        text = response["candidates"][0]["content"]["parts"][0]["text"]
        raw = json.loads(
            text, object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value!r}")))
        provider = sa.ProviderTrackAudit.model_validate(raw)
        provider_value = provider.model_dump()
        _require_exact_response_shape(raw, provider_value)
        audit_value = sa.canonicalize_provider_audit(provider)
    except Exception as error:
        message = f"{key}: invalid ProviderTrackAudit response: {error}"
        raise ValueError(message) from error
    if (audit_value["verdict"] in ("keep", "keep_partial")
            and not audit_value["valid_segments"]):
        raise ValueError(
            f"{key}: accepted decision requires at least one valid segment")
    for index, item in enumerate(audit_value["primary_object"]["tags"]):
        _validate_weight(item["weight"], f"{key} tag {index}")
    for index, item in enumerate(
            audit_value["primary_object"]["name_candidates"]):
        _validate_weight(item["weight"], f"{key} name {index}")
    for secondary_index, secondary in enumerate(
            audit_value["secondary_objects"]):
        for tag_index, item in enumerate(secondary["tags"]):
            _validate_weight(
                item["weight"],
                f"{key} secondary {secondary_index} tag {tag_index}")
    return audit_value


def compile_audit_results(request_set: llm.RequestSet, attempts):
    """Require exactly one valid response for every audit request."""
    return llm.compile_canonical_results(
        request_set, attempts, _validate_audit_response)


def _load_request_bundle(request_dir: Path, tracks_dir: Path,
                         dataset_name: str):
    request_ref = artifact.open_artifact(
        request_dir, expected_kind=REQUEST_ARTIFACT_KIND,
        expected_dataset=dataset_name)
    request_set = llm.load_request_set(
        Path(request_dir) / llm.REQUEST_SET_NAME)
    meta = _load_json(Path(request_dir) / AUDIT_META_NAME)
    source_ref, source_path, source, tracks, range_name = load_source_tracks(
        tracks_dir, dataset_name)
    if source_ref not in request_set.upstreams:
        raise ValueError("request snapshot is not bound to this tracks artifact")
    if set(meta) != {"schema", "source_tracks", "requests"}:
        raise ValueError("audit_meta.json has missing or unknown top-level keys")
    if meta["schema"] != AUDIT_META_SCHEMA:
        raise ValueError(f"audit_meta.json schema must be {AUDIT_META_SCHEMA}")
    expected_source = {
        "artifact_id": source_artifact_id(source_ref),
        "file": source_path.name,
        "sha256": artifact.sha256_file(source_path),
    }
    if meta["source_tracks"] != expected_source:
        raise ValueError("audit_meta.json source_tracks binding is stale")
    expected_keys = [unit.key for unit in request_set.units]
    if set(meta["requests"]) != set(expected_keys):
        raise ValueError("audit metadata does not exactly cover the request set")
    tracks_by_id = {track["track_id"]: track for track in tracks}
    for unit in request_set.units:
        request_meta = meta["requests"][unit.key]
        if request_meta != unit.to_dict()["metadata"]:
            raise ValueError(f"{unit.key}: request metadata was changed")
        track = tracks_by_id.get(request_meta.get("track_id"))
        if track is None or request_meta.get(
                "source_track_sha256") != artifact.sha256_json(track):
            raise ValueError(f"{unit.key}: source-track binding is stale")
    return (request_ref, request_set, meta, source_ref, source_path,
            source, tracks, range_name)


def publish_audit_results(destination: Path, *, request_dir: Path,
                          tracks_dir: Path, attempts_dir: Path,
                          dataset_name: str, version: str,
                          arguments=(), manifest_config: dict | None = None,
                          git_commit: str | None = None,
                          artifact_identity: str | None = None,
                          recipe: dict | None = None,
                          ) -> artifact.ArtifactRef:
    """Publish only a complete, unique, validated semantic-audit result."""
    (request_ref, request_set, meta, source_ref, _, _, _,
     _) = _load_request_bundle(request_dir, tracks_dir, dataset_name)
    results = compile_audit_results(
        request_set, llm.load_attempts(attempts_dir))
    expected = [unit.key for unit in request_set.units]
    if [item.key for item in results] != expected:
        raise llm.IncompleteCoverageError(
            "canonical audit result order does not cover the request set")
    result_lines = []
    for item in results:
        # audit_io consumes the normal provider boundary; transport-only
        # metadata and failed attempts remain outside the immutable artifact.
        response = {
            "candidates": [{"content": {"parts": [{
                "text": artifact.canonical_json_bytes(item.result).decode(
                    "utf-8")
            }]}}]
        }
        result_lines.append(artifact.canonical_json_bytes({
            "key": item.key, "response": response}) + b"\n")
    request_manifest = artifact.load_manifest(request_dir)
    frame_refs = [
        ref for ref in request_set.upstreams
        if ref.kind == paths_lib.FRAME_LANDMARKS]
    if len(frame_refs) != 1:
        raise ValueError(
            "audit request set must bind exactly one frame_landmarks artifact")
    declared_outputs = sorted(
        [*request_manifest.declared_outputs, "results.jsonl"])
    with publication.published_artifact(
            destination,
            kind=paths_lib.SEMANTIC_AUDITS,
            dataset=dataset_name,
            version=version,
            generator=GENERATOR,
            git_commit=(provenance.git_commit()
                        if git_commit is None else git_commit),
            arguments=arguments,
            upstreams=(source_ref, frame_refs[0], request_ref),
            artifact_identity=artifact_identity,
            recipe=recipe,
            config={
                "phase": "canonical_results",
                "request_set_fingerprint": request_set.fingerprint,
                "n_expected": len(expected),
                "n_successful": len(results),
                "coverage": "complete",
                **(manifest_config or {}),
            },
            declared_outputs=declared_outputs) as builder:
        # The final semantic audit remains directly reviewable and its
        # artifact-relative chip references stay valid after publication.
        for relative in request_manifest.declared_outputs:
            output = builder.output_path(relative)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(request_dir) / relative, output)
        artifact.atomic_write_file(
            builder.output_path("results.jsonl"), b"".join(result_lines))
    assert builder.artifact_ref is not None
    return builder.artifact_ref


def audit_work_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    return output_dir.with_name(output_dir.name + ".llm-work")


def load_audit_config(args):
    """Resolve and type-check the exact audit/execution recipe once."""
    config_path = Path(args.build_config)
    if config_path.name != build_config.BUILD_CONFIG_NAME \
            or not config_path.is_file():
        raise ValueError(
            f"--build_config must name an existing "
            f"{build_config.BUILD_CONFIG_NAME}")
    document = build_config.load(config_path.parent)
    if document["dataset"] != args.dataset:
        raise ValueError("--dataset disagrees with build_config")
    recorded_base = document["inputs"].get("dataset_base")
    if recorded_base is None or Path(recorded_base).resolve() != Path(
            args.dataset_base).resolve():
        raise ValueError("--dataset_base disagrees with build_config")
    selected = {key: build_config.value(document, key)
                for key in AUDIT_CONFIG_KEYS}
    actual_digest = artifact.sha256_json(selected)
    if args.orchestration_config_digest != actual_digest:
        raise ValueError(
            "--orchestration_config_digest does not match the immutable "
            "audit/ingest/execution/cost recipe")
    specs = {
        "artifacts.semantic_audits_version": build_config.ValueSpec(
            (str,), nonempty=True),
        "audit.model": build_config.ValueSpec((str,), nonempty=True),
        "audit.min_supports": build_config.ValueSpec((int,), minimum=1),
        "audit.thinking_level": build_config.ValueSpec((str,), nonempty=True),
        "audit.max_support_chips": build_config.ValueSpec((int,), minimum=1),
        "audit.max_context_chips": build_config.ValueSpec((int,), minimum=0),
        "audit.max_description_samples": build_config.ValueSpec(
            (int,), minimum=1),
        "audit.chip_height_px": build_config.ValueSpec((int,), minimum=1),
        "ingest.fov_deg": build_config.ValueSpec(
            (int, float), minimum=0.0, maximum=180.0),
        "ingest.seam_gap_norm": build_config.ValueSpec(
            (int, float), minimum=0.0),
        "ingest.seam_min_y_iou": build_config.ValueSpec(
            (int, float), minimum=0.0, maximum=1.0),
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
    for key in ("artifacts.frame_landmarks_version",
                "artifacts.object_tracks_version"):
        build_config.ValueSpec((str,), nonempty=True).validate(
            key, build_config.value(document, key))
    transport = selected["execution.llm_transport"]
    prefix = selected["execution.batch_gcs_prefix"]
    if transport == "batch" and (not isinstance(prefix, str)
                                  or not prefix.startswith("gs://")):
        raise build_config.InvalidConfigValue(
            "execution.batch_gcs_prefix must be a gs:// URI for batch mode")
    if transport == "on_demand" and prefix is not None:
        raise build_config.InvalidConfigValue(
            "execution.batch_gcs_prefix must be null for on_demand mode")
    orchestration = {
        "schema": "farfield_pipeline_stage/v1",
        "stage": "audit",
        "config_digest": actual_digest,
    }
    return document, selected, orchestration


def open_prefix_inputs(args, document):
    """Open the exact TRACK/FRAME artifacts this build's recipe names.

    A stage opens the inputs it was told to open, at the versions the recipe
    records, and `open_artifact` proves each is the artifact it claims to be.
    Whether they are the right GENERATION is the orchestrator's question,
    answered by `artifact_identity`; a stage re-deciding it here is how the
    build-identity check and its human-attested exceptions ended up threaded
    through every producer.
    """
    source_ref = artifact.open_artifact(
        args.tracks_dir, expected_kind=paths_lib.OBJECT_TRACKS,
        expected_dataset=args.dataset,
        expected_version=build_config.value(
            document, "artifacts.object_tracks_version"))
    frame_ref = artifact.open_artifact(
        args.frame_landmarks_dir,
        expected_kind=paths_lib.FRAME_LANDMARKS,
        expected_dataset=args.dataset,
        expected_version=build_config.value(
            document, "artifacts.frame_landmarks_version"))
    return source_ref, frame_ref


def validate_execution_args(args, selected) -> None:
    if args.parallel < 1:
        raise ValueError("--parallel must be positive")
    if args.poll_interval < 1:
        raise ValueError("--poll_interval must be positive")
    expected_online = selected["execution.llm_transport"] == "on_demand"
    if bool(args.online) != expected_online:
        raise ValueError(
            "--online disagrees with execution.llm_transport")
    expected_prefix = selected["execution.batch_gcs_prefix"]
    if ((expected_online and args.gcs_prefix is not None)
            or (not expected_online and args.gcs_prefix != expected_prefix)):
        raise ValueError(
            "--gcs_prefix disagrees with execution.batch_gcs_prefix")
    if bool(args.approve_cost) != selected["execution.approve_cost"]:
        raise ValueError(
            "--approve_cost disagrees with execution.approve_cost")
    expected_limit = float(selected["cost.limit_usd"])
    if args.cost_limit is not None and args.cost_limit != expected_limit:
        raise ValueError("--cost_limit disagrees with cost.limit_usd")
    args.cost_limit = expected_limit
    args.model = selected["audit.model"]


def _request_args(args, selected, document, orchestration,
                  destination: Path, version: str):
    result = argparse.Namespace(**vars(args))
    result.output_dir = destination
    result.output_version = version
    result.model = selected["audit.model"]
    result.min_supports = selected["audit.min_supports"]
    result.thinking_level = selected["audit.thinking_level"]
    result.max_support_chips = selected["audit.max_support_chips"]
    result.max_context_chips = selected["audit.max_context_chips"]
    result.max_description_samples = selected[
        "audit.max_description_samples"]
    result.chip_height_px = selected["audit.chip_height_px"]
    result.fov_deg = selected["ingest.fov_deg"]
    result.seam_gap_norm = selected["ingest.seam_gap_norm"]
    result.seam_min_y_iou = selected["ingest.seam_min_y_iou"]
    result.ingest_params = dataset.IngestParams(
        fov_deg=result.fov_deg,
        seam_gap_norm=result.seam_gap_norm,
        seam_min_y_iou=result.seam_min_y_iou)
    result.build_identity = document["build_identity"]
    result.orchestration = orchestration
    result.resolved_stage_config = selected
    return result


def _prepare_request_artifact(args, selected, document, orchestration,
                              source, ingest_result, work_dir: Path,
                              request_version: str) -> Path:
    """Build once; on resume reconstruct and demand byte-identical inputs."""
    request_dir = work_dir / "requests"
    paths = argparse.Namespace(
        dataset=args.dataset, dataset_base=Path(args.dataset_base))

    def build_at(destination):
        request_args = _request_args(
            args, selected, document, orchestration,
            destination, request_version)
        return build_request_artifact(
            request_args, paths, *source, ingest_result)

    if not request_dir.exists() and not request_dir.is_symlink():
        build_at(request_dir)
        return request_dir
    artifact.open_artifact(
        request_dir, expected_kind=REQUEST_ARTIFACT_KIND,
        expected_dataset=args.dataset, expected_version=request_version)
    current_request_set = llm.load_request_set(
        request_dir / llm.REQUEST_SET_NAME)
    current_config = artifact.load_manifest(request_dir).config
    with tempfile.TemporaryDirectory(dir=work_dir) as temporary:
        candidate_dir = Path(temporary) / "requests"
        build_at(candidate_dir)
        candidate_request_set = llm.load_request_set(
            candidate_dir / llm.REQUEST_SET_NAME)
        candidate_config = artifact.load_manifest(candidate_dir).config
        stable_config_keys = (
            "build_identity", "orchestration", "resolved_stage_config")
        if (candidate_request_set.fingerprint
                != current_request_set.fingerprint
                or any(candidate_config.get(key) != current_config.get(key)
                       for key in stable_config_keys)):
            raise ValueError(
                "recorded semantic-audit request snapshot differs from the "
                "current tracks, frame landmarks, panoramas, prompt, schema, "
                "model, or resolved config; choose a new artifact version")
    return request_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_base", type=Path, required=True)
    parser.add_argument("--tracks_dir", type=Path, required=True)
    parser.add_argument("--frame_landmarks_dir", type=Path, required=True)
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
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--gcs_prefix")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--poll_interval", type=int, default=120)
    parser.add_argument("--cost_limit", type=float)
    parser.add_argument("--approve_cost", action="store_true")
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--submit", action="store_true")
    phase.add_argument("--build_only", action="store_true")
    phase.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()
    try:
        document, selected, orchestration = load_audit_config(args)
        validate_execution_args(args, selected)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    output_version = selected["artifacts.semantic_audits_version"]
    if args.output_dir.name != output_version:
        parser.error(
            f"--output_dir must end in configured version {output_version!r}")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(
            f"completed semantic_audits artifact already exists: "
            f"{args.output_dir}")
    try:
        source_ref, frame_ref = open_prefix_inputs(args, document)
        source = load_source_tracks(args.tracks_dir, args.dataset)
        if source[0] != source_ref:
            raise ValueError(
                "track loader did not retain the exact authorized ref")
        ingest_params = dataset.IngestParams(
            fov_deg=selected["ingest.fov_deg"],
            seam_gap_norm=selected["ingest.seam_gap_norm"],
            seam_min_y_iou=selected["ingest.seam_min_y_iou"])
        ingest_result = dataset.run_ingest(
            args.dataset_base, args.frame_landmarks_dir, ingest_params)
        if ingest_result.frame_landmarks_ref != frame_ref:
            raise ValueError(
                "frame_landmarks identity disagrees with build_config")
        work_dir = audit_work_dir(args.output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        request_version = output_version + ".requests"
        if args.aggregate_only and not (work_dir / "requests").is_dir():
            raise ValueError(
                "--aggregate_only requires an existing immutable request "
                "snapshot")
        request_dir = _prepare_request_artifact(
            args, selected, document, orchestration, source, ingest_result,
            work_dir, request_version)
        request_ref, request_set, _, _, _, _, _, _ = _load_request_bundle(
            request_dir, args.tracks_dir, args.dataset)
    except (artifact.ArtifactError, llm.LlmLifecycleError,
            OSError, ValueError) as error:
        raise SystemExit(f"invalid semantic-audit work state: {error}") \
            from error

    if args.build_only:
        print(f"published immutable audit requests: {request_dir}")
        return
    attempts_dir = work_dir / ATTEMPTS_DIR_NAME
    for transport in vbm.completed_submission_results(work_dir):
        imported = llm.import_transport_results(
            transport, attempts_dir, request_set)
        if imported:
            print(f"preserved {imported} new provider attempt(s)")
    attempts = (llm.load_attempts(attempts_dir)
                if attempts_dir.exists() else ())
    if args.submit:
        pending = llm.pending_request_keys(
            request_set, attempts, _validate_audit_response)
        if pending:
            round_index, pending_path, transport_path = (
                vbm.next_submission_paths(work_dir))
            artifact.atomic_create_file(
                pending_path,
                llm.transport_requests_bytes(request_set, pending))
            vbm.run_requests(
                args, pending_path, transport_path,
                tag=(f"{args.dataset}_semantic_audit_{output_version}_"
                     f"r{round_index:04d}"))
            for completed in vbm.completed_submission_results(work_dir):
                imported = llm.import_transport_results(
                    completed, attempts_dir, request_set)
                if imported:
                    print(f"preserved {imported} new provider attempt(s)")
        else:
            print("all audit requests already have a validated success")
    if not attempts_dir.exists():
        raise SystemExit(
            f"no bound audit attempts at {attempts_dir}; run --submit")
    ref = publish_audit_results(
        args.output_dir,
        request_dir=request_dir,
        tracks_dir=args.tracks_dir,
        attempts_dir=attempts_dir,
        dataset_name=args.dataset,
        version=output_version,
        arguments=sys.argv[1:],
        manifest_config={
            "build_identity": document["build_identity"],
            "orchestration": orchestration,
            "resolved_stage_config": selected,
        },
        git_commit=provenance.git_commit(),
        artifact_identity=getattr(args, "artifact_identity", None),
        recipe=artifact_recipe.load(getattr(args, "artifact_recipe", None)))
    print(f"published complete semantic audit: {ref.path}")


if __name__ == "__main__":
    main()
