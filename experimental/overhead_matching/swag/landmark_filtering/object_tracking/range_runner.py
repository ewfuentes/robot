"""Shared M3 range runner: wires ingest + heading + video + SAM backend into
TrackBuilder over a keyframe range. Used by m3_build_tracks (board renderer)
and m3_track_viewer (per-track media capture) so the tracking loop exists
exactly once."""

import dataclasses
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    pano_geometry as pg,
    track_builder as tb,
    viz_common as vc,
)
from experimental.overhead_matching.swag.landmark_filtering.object_tracking.perf_profile import (
    PROFILE,
)

CELL = 168
ACTION_COLORS = {
    "birth": (60, 255, 60),
    "reanchor_clean": (60, 220, 60),
    "continue_mask": (70, 160, 255),
    "unsupported": (245, 170, 40),
    "mask_dead": (230, 60, 60),
}
CLOSE_COLOR = (230, 60, 60)


class BoardRenderer:
    """Collects one cell image per (track, keyframe) as the builder steps."""

    def __init__(self, font):
        self.cells = {}  # (track_id, keyframe) -> PIL.Image
        self.font = font

    def snap(self, builder, keyframe, keyframe_crop_fn):
        """Render cells for every track that has a record at `keyframe`.
        keyframe_crop_fn(origin, size) -> np.ndarray crop of the keyframe."""
        for track in builder.tracks:
            if not track.records or track.records[-1]["keyframe"] != keyframe:
                continue
            rec = track.records[-1]
            if rec["action"] == "birth":
                mask, origin = track.birth_mask, track.birth_origin
            else:
                mask, origin = track.last_mask, track.last_origin
            if mask is None:
                continue
            crop = keyframe_crop_fn(origin, mask.shape[0])
            img = crop.copy()
            if mask.any():
                overlay = np.zeros_like(img)
                overlay[mask] = (255, 60, 60)
                img = (0.65 * img + 0.35 * overlay).astype(np.uint8)
            cell = Image.fromarray(img).resize((CELL, CELL), Image.BILINEAR)
            draw = ImageDraw.Draw(cell)
            color = ACTION_COLORS.get(rec["action"], (200, 200, 200))
            if track.status == "closed" and track.records[-1] is rec:
                color = CLOSE_COLOR
            draw.rectangle([0, 0, CELL - 1, CELL - 1], outline=color, width=3)
            caption = rec["action"]
            if rec.get("supports"):
                best = rec["supports"][0]
                if best["class"] != "none":
                    caption += f" {best['class']}:{best['iou']:.2f}"
            vc.draw_caption(draw, caption, self.font, xy=(3, 3))
            self.cells[(track.track_id, keyframe)] = cell

    def compose(self, builder, k_start, k_end, font):
        keyframes = list(range(k_start, k_end + 1))
        tracks = [t for t in builder.tracks if t.records]
        tracks.sort(key=lambda t: (t.birth_keyframe, t.track_id))
        gutter = 300
        width = gutter + CELL * len(keyframes)
        board = Image.new("RGB", (width, 24 + CELL * len(tracks)), (18, 18, 18))
        draw = ImageDraw.Draw(board)
        for col, k in enumerate(keyframes):
            vc.draw_caption(draw, f"f{k:04d}", font,
                            xy=(gutter + col * CELL + 4, 4))
        for row, track in enumerate(tracks):
            y = 24 + row * CELL
            status = (track.close_reason if track.status == "closed"
                      else "alive")
            vc.draw_caption(draw, f"T{track.track_id} {track.modal_label()}",
                            font, xy=(4, y + 4))
            vc.draw_caption(draw, f"  b=f{track.birth_keyframe:04d} {status}",
                            font, xy=(4, y + 22))
            for col, k in enumerate(keyframes):
                cell = self.cells.get((track.track_id, k))
                if cell is not None:
                    board.paste(cell, (gutter + col * CELL, y))
        return board


def track_artifact(builder, builder_cfg, range_name, k_start, k_end):
    return {
        "range": {"name": range_name, "k_start": k_start, "k_end": k_end},
        "config": dataclasses.asdict(builder_cfg),
        "tracks": [{
            "track_id": t.track_id,
            "birth_obs_id": t.birth_obs_id,
            "birth_keyframe": t.birth_keyframe,
            "status": t.status,
            "close_reason": t.close_reason,
            "end_keyframe": t.end_keyframe,
            "last_keyframe": t.last_keyframe,
            "modal_label": t.modal_label(),
            "n_supported_keyframes": sum(
                1 for r in t.records
                if any(s["class"] in tb.SUPPORT_CLASSES
                       for s in r.get("supports", []))),
            "records": t.records,
        } for t in builder.tracks],
        "rejected_births": builder.rejected_births,
        "track_overlaps": builder.track_overlaps,
    }


def run_range(range_name, k_start, k_end, builder_cfg, backend, provider,
              model, result, obs_by_frame, det_pano_boxes, pano_w, pano_h,
              dataset_base: Path, renderer: BoardRenderer | None = None,
              on_interval=None, log=print):
    """Run TrackBuilder over [k_start, k_end]. Returns (builder, artifact)."""
    frames_by_idx = {f.frame_idx: f for f in result.frames}
    builder = tb.TrackBuilder(backend, builder_cfg, pano_w, pano_h,
                              on_interval=on_interval)

    def keyframe_crop_fn_for(pano_img):
        def crop_fn(origin, size):
            crop, _ = pg.extract_window(pano_img, origin[0], origin[1],
                                        size, size)
            return crop
        return crop_fn

    dets0 = obs_by_frame.get(k_start, [])
    builder.seed_unassigned(k_start, dets0, det_pano_boxes)

    for k in range(k_start, k_end):
        t0 = frames_by_idx[k].time_s
        t1 = frames_by_idx[k + 1].time_s
        with PROFILE.phase("video_decode"):
            frames = list(provider.frames_between(t0, t1))
        PROFILE.items["video_decode"] += len(frames)
        track_crops = {}

        def crops_fn(track, size, _frames=frames, _t0=t0, _cache=track_crops):
            key = (track.track_id, size)
            if key not in _cache:
              with PROFILE.phase("window_crops", items=len(_frames)):
                crops, origins = [], []
                for _, t, frame_rgb in _frames:
                    az0, _el = pg.direction_from_pano_px(
                        track.center_x, track.center_y, pano_w, pano_h)
                    az_w = az0 - model.delta(t, _t0)
                    wx, _ = pg.pano_px_from_direction(az_w, 0.0, pano_w, pano_h)
                    crop, y0 = pg.extract_window(
                        frame_rgb, wx - size / 2.0,
                        track.center_y - size / 2.0, size, size)
                    crops.append(crop)
                    origins.append((wx - size / 2.0, y0))
                _cache[key] = (crops, origins)
            return _cache[key]

        dets_next = obs_by_frame.get(k + 1, [])
        n_alive = len(builder.alive_tracks())
        with PROFILE.phase("builder_step_total", items=max(n_alive, 1)):
            builder.step(k, crops_fn, dets_next, det_pano_boxes)

        if renderer is not None:
          with PROFILE.phase("board_render"):
            if k == k_start:
                pano0 = np.asarray(Image.open(
                    dataset_base / "panorama"
                    / f"{frames_by_idx[k].pano_stem}.jpg"))
                renderer.snap(builder, k, keyframe_crop_fn_for(pano0))
            pano1 = np.asarray(Image.open(
                dataset_base / "panorama"
                / f"{frames_by_idx[k + 1].pano_stem}.jpg"))
            renderer.snap(builder, k + 1, keyframe_crop_fn_for(pano1))
        log(f"  [{range_name}] f{k:04d}->f{k + 1:04d}: {n_alive} alive, "
            f"{len(dets_next)} dets, {len(builder.tracks)} total tracks")

    artifact = track_artifact(builder, builder_cfg, range_name, k_start, k_end)
    PROFILE.report(log=log, label=f"range {range_name}")
    return builder, artifact


def load_context(dataset_base, landmark_base, video_path, checkpoint,
                 preview_size=None):
    """Load everything a range run needs. Returns a dict of shared state.

    `video_path=None` means the dataset has no source video; the keyframe
    panoramas themselves become the tracking substrate (KeyframeProvider), so
    SAM2 propagates across the keyframe baseline with no intermediates.
    """
    from experimental.overhead_matching.swag.landmark_filtering import ingest
    from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
        heading as heading_mod,
        sam_backend,
        video_frames,
    )
    from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
        IngestConfig,
    )
    result = ingest.run_ingest(dataset_base, landmark_base, IngestConfig())
    probe = Image.open(
        dataset_base / "panorama" / f"{result.frames[0].pano_stem}.jpg")
    pano_w, pano_h = probe.size
    obs_by_frame, obs_by_id, det_pano_boxes = {}, {}, {}
    for obs in result.observations:
        obs_by_frame.setdefault(obs.frame_idx, []).append(obs)
        obs_by_id[obs.obs_id] = obs
        det_pano_boxes[obs.obs_id] = pg.pano_bbox_for_observation(
            obs.boxes, pano_w, pano_h)
    model = heading_mod.heading_model_from_positions(
        [f.x_m for f in result.frames], [f.y_m for f in result.frames],
        [f.time_s for f in result.frames])
    if video_path is None:
        provider = video_frames.KeyframeProvider(
            [f.time_s for f in result.frames],
            [dataset_base / "panorama" / f"{f.pano_stem}.jpg"
             for f in result.frames])
    else:
        provider = video_frames.VideoFrameProvider(video_path)
    return {
        "result": result, "pano_w": pano_w, "pano_h": pano_h,
        "obs_by_frame": obs_by_frame, "obs_by_id": obs_by_id,
        "det_pano_boxes": det_pano_boxes, "model": model,
        "provider": provider,
        "backend": sam_backend.Sam2Backend(checkpoint,
                                          preview_size=preview_size),
    }


def write_artifact(artifact, out_dir: Path, range_name: str):
    (out_dir / f"tracks_{range_name}.json").write_text(
        json.dumps(artifact, indent=1))
