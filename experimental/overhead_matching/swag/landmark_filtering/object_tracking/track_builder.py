"""Track builder: chains SAM interval propagation across keyframes into
mask-anchored tracks, with VLM detections attached as evidence.

Design (see m2_sam_tracking.py for the single-interval spike this grew from):
- Identity lives in the SAM masklet. Detections vote on semantics and
  reliability; they only redefine geometry on a clean 1:1 re-anchor.
- Association uses asymmetric containment, not Hungarian 1:1: a detection can
  be a clean continuation, a superset (merge-shaped), a child (split-shaped),
  or weak evidence. Splits/merges are labeled, not acted on, until data shows
  forking is needed.
- Forward-only. Bad starts are handled by birth gating (mask health at the
  prompt frame) plus graceful starvation: every future detection is itself a
  candidate seed, so a landmark whose box is misplaced at one keyframe founds
  its track at the next well-placed one.
- Stop rules: mask death, detection starvation with a long patience
  (default 15 keyframes - detections flicker), and a drift alarm for the
  "semantically matching detection keeps landing next to the mask but never
  on it" signature of a slid mask.
"""

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    pano_geometry as pg,
)
from experimental.overhead_matching.swag.landmark_filtering.object_tracking.perf_profile import (
    PROFILE,
)


@dataclass
class TrackBuilderConfig:
    # Base (minimum) tracking-window size. Windows grow per track with the
    # object's current extent: near-field objects (the fort at closest
    # approach measured 1361 px against a fixed 1024 window) otherwise get
    # clipped at the window edge, and clipping is a runaway - a clipped
    # mask's centroid is biased away from the object center, so the next
    # window re-centers wrong and clips more (T14 eroded 46k -> 805 px^2).
    window_px: int = 1024
    window_extent_factor: float = 2.0   # window >= factor * object extent
    window_quantum: int = 512           # round window sizes up to multiples
    window_max_px: int = 3072
    # Support classification (thresholds anchored on M2 measurements).
    clean_iou: float = 0.45
    # Coherence floor for merge_superset: a superset detection must itself
    # be meaningfully filled by the mask. A mask speck inside a giant box
    # scores inter/mask 1.0 by construction (T172: a dying island-remnant
    # mask was kept alive 50 keyframes by successive OTHER islands' boxes at
    # inter/box 0.01-0.08, stealing the track across three islands). Below
    # the floor the detection is classified "context": recorded for Level-2
    # merge/occlusion evidence but not counted as support. Genuine
    # granularity supersets (fort) measure inter/box 0.16-0.40.
    superset_min_inter_over_box: float = 0.10
    # Containment guard: a clean re-anchor may not discard more than
    # (1 - this) of the current mask. Re-anchoring is an identity rewrite;
    # a box that only covers part of the mask is truncating it (T0/f0011:
    # a displaced box scored iou 0.46 against a spread mask with
    # inter/mask 0.63 and stole the track onto the neighboring building;
    # every healthy re-anchor measured >= 0.77).
    reanchor_min_inter_over_mask: float = 0.75
    superset_inter_over_mask: float = 0.7
    superset_max_inter_over_box: float = 0.4
    child_inter_over_box: float = 0.7
    child_max_inter_over_mask: float = 0.4
    weak_min_iou: float = 0.15
    weak_min_containment: float = 0.5
    # A containment-only weak support must show SOME mutual agreement: a
    # giant occluding/neighboring box whose edge happens to cover half the
    # mask has inter/box ~ 0 and is not evidence (T185/f0268: a foreground
    # island's box covered 0.52 of a background island's mask with iou 0.00,
    # polluting votes and suppressing the occluder's own track seed).
    weak_min_complement: float = 0.10
    # Birth gating: mask health at the prompt frame.
    birth_min_coverage: float = 0.15  # mask&box / box area
    birth_max_spill: float = 0.5      # mask outside box / mask area
    birth_min_dominant_cc: float = 0.6  # largest component / mask area
    # Stop rules. Two-tier patience: detections flicker on established
    # tracks (a crane survived a 5-keyframe gap), but tracks that were never
    # supported after birth are one-shot VLM detections - in the full-leg1
    # run 138/294 tracks had zero post-birth support and each zombie-
    # propagated 15 keyframes (~1/3 of total runtime) before starving.
    patience_keyframes: int = 15
    patience_unsupported_keyframes: int = 3
    min_mask_area_px: int = 20
    drift_gate_px: float = 150.0
    drift_patience: int = 3


SUPPORT_PRIORITY = ["continue_clean", "merge_superset", "split_child", "weak",
                    "context"]
# Classes that count as support: vote on semantics, reset patience, extend
# end_keyframe, claim their detection, and feed window sizing. "context"
# (contained-but-incoherent) and "none" do none of those.
SUPPORT_CLASSES = frozenset(
    ("continue_clean", "merge_superset", "split_child", "weak"))


def window_size_for_extent(extent_px: float, cfg: TrackBuilderConfig) -> int:
    """Quantized per-track window size covering an object of this width."""
    want = extent_px * cfg.window_extent_factor
    if want <= cfg.window_px:
        return cfg.window_px
    quantized = math.ceil(want / cfg.window_quantum) * cfg.window_quantum
    return int(min(cfg.window_max_px, quantized))


def mask_box_metrics(mask: np.ndarray, box) -> dict:
    """IoU and containment coefficients between a bool mask and a box, in the
    mask's pixel frame."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h, w = mask.shape
    box_mask = np.zeros_like(mask, dtype=bool)
    box_mask[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
    inter = float(np.logical_and(mask, box_mask).sum())
    mask_area = float(mask.sum())
    box_area = float(box_mask.sum())
    union = mask_area + box_area - inter
    return {
        "iou": inter / union if union else 0.0,
        "inter_over_mask": inter / mask_area if mask_area else 0.0,
        "inter_over_box": inter / box_area if box_area else 0.0,
    }


def classify_support(metrics: dict, cfg: TrackBuilderConfig) -> str:
    iou = metrics["iou"]
    iom = metrics["inter_over_mask"]
    iob = metrics["inter_over_box"]
    if iou >= cfg.clean_iou:
        if iom >= cfg.reanchor_min_inter_over_mask:
            return "continue_clean"
        return "weak"  # would truncate the mask; support without re-anchor
    if iom >= cfg.superset_inter_over_mask and iob <= cfg.superset_max_inter_over_box:
        if iob < cfg.superset_min_inter_over_box:
            return "context"
        return "merge_superset"
    if iob >= cfg.child_inter_over_box and iom <= cfg.child_max_inter_over_mask:
        return "split_child"
    if (iou >= cfg.weak_min_iou
            or (iom >= cfg.weak_min_containment
                and iob >= cfg.weak_min_complement)
            or (iob >= cfg.weak_min_containment
                and iom >= cfg.weak_min_complement)):
        return "weak"
    return "none"


def mask_health(mask: np.ndarray, prompt_box, cfg: TrackBuilderConfig) -> dict:
    """Birth-gate stats of a prompt-frame mask against its founding box."""
    import scipy.ndimage
    area = float(mask.sum())
    if area < cfg.min_mask_area_px:
        return {"ok": False, "reason": "empty", "area": area}
    labels, n = scipy.ndimage.label(mask)
    largest = max(np.bincount(labels.ravel())[1:]) if n else 0
    dominant = largest / area
    metrics = mask_box_metrics(mask, prompt_box)
    spill = 1.0 - metrics["inter_over_mask"]
    coverage = metrics["inter_over_box"]
    dominant = float(dominant)
    # bool(): numpy comparison results are np.bool_, which json rejects.
    ok = bool(dominant >= cfg.birth_min_dominant_cc
              and spill <= cfg.birth_max_spill
              and coverage >= cfg.birth_min_coverage)
    reason = "" if ok else (
        "fragmented" if dominant < cfg.birth_min_dominant_cc else
        "spill" if spill > cfg.birth_max_spill else "coverage")
    return {"ok": ok, "reason": reason, "area": area, "n_components": int(n),
            "dominant_cc_frac": round(dominant, 3),
            "spill_frac": round(spill, 3), "coverage": round(coverage, 3)}


@dataclass
class Track:
    track_id: int
    birth_obs_id: str
    birth_keyframe: int
    status: str = "alive"  # alive | closed
    close_reason: str = ""
    end_keyframe: int | None = None       # last supported keyframe
    last_keyframe: int | None = None      # last keyframe propagated to
    # Prompt for the NEXT interval: exactly one of box/mask set.
    prompt_box: list | None = None        # window px at interval frame 0
    prompt_mask: np.ndarray | None = None
    # Window placement/size for the next interval (pano px center).
    center_x: float = 0.0
    center_y: float = 0.0
    window_px: int = 0
    unsupported_streak: int = 0
    drift_streak: int = 0
    ever_supported: bool = False  # any support after the founding detection
    tag_votes: Counter = field(default_factory=Counter)
    name_votes: Counter = field(default_factory=Counter)
    records: list = field(default_factory=list)
    # Transient handoff to the caller for rendering (not serialized).
    last_mask: np.ndarray | None = None
    last_origin: tuple | None = None
    birth_mask: np.ndarray | None = None
    birth_origin: tuple | None = None

    def modal_label(self) -> str:
        tag = self.tag_votes.most_common(1)
        name = self.name_votes.most_common(1)
        label = tag[0][0] if tag else "?"
        if name and name[0][0]:
            label += f" '{name[0][0]}'"
        return label


class TrackBuilder:
    """Steps tracks keyframe-by-keyframe. Rendering/IO stay outside; the
    caller supplies per-interval window crops via a callback so this class
    is testable with fakes."""

    def __init__(self, backend, cfg: TrackBuilderConfig, pano_w: int,
                 pano_h: int, on_interval=None):
        self.backend = backend
        self.cfg = cfg
        self.pano_w = pano_w
        self.pano_h = pano_h
        self.tracks: list[Track] = []
        self.rejected_births: list[dict] = []
        # Pairwise mask overlap between co-alive tracks, per keyframe.
        # Consolidation's merge evidence - masks only exist here, so this
        # is recorded during the run and shipped in the artifact.
        self.track_overlaps: list[dict] = []
        self._next_id = 0
        # Optional media hook: on_interval(track, keyframe, crops, origins,
        # masks), called after the track's record for this interval is
        # written (so track.records[-1] describes the outcome).
        self.on_interval = on_interval

    def alive_tracks(self):
        return [t for t in self.tracks if t.status == "alive"]

    def seed(self, obs, pano_box, keyframe: int):
        """Register a detection as a new track seed; it is birth-gated when
        its first interval propagates."""
        cx = (pano_box[0] + pano_box[2]) / 2.0
        cy = (pano_box[1] + pano_box[3]) / 2.0
        track = Track(
            track_id=self._next_id, birth_obs_id=obs.obs_id,
            birth_keyframe=keyframe, center_x=cx, center_y=cy,
            window_px=window_size_for_extent(pano_box[2] - pano_box[0],
                                             self.cfg))
        self._next_id += 1
        track.prompt_box = None  # set per-interval once window origin known
        track._birth_pano_box = pano_box  # noqa: SLF001 - internal handoff
        self._vote(track, obs)
        self.tracks.append(track)
        return track

    def _vote(self, track: Track, obs):
        track.tag_votes[f"{obs.primary_tag_key}={obs.primary_tag_value}"] += 1
        tags = dict(tuple(t) for t in obs.additional_tags)
        track.name_votes[tags.get("name", "")] += 1

    def window_origin(self, track: Track, size: int):
        return track.center_x - size / 2.0, track.center_y - size / 2.0

    def step(self, keyframe: int, crops_fn, detections: list,
             det_pano_boxes: dict):
        """Advance all alive tracks across the interval keyframe->keyframe+1.

        crops_fn(track, size) -> (crops, origins): per-frame window crops
        (first frame = keyframe, last = keyframe+1) and their (x0, y0) pano
        origins, heading-compensated by the caller.
        detections: observations at keyframe+1; det_pano_boxes maps obs_id to
        unwrapped pano bbox.
        """
        cfg = self.cfg
        # Plan every track's prompt first, propagate them together, then apply
        # the outcomes in the same order a per-track loop would have. Nothing in
        # the planning phase reads another track's post-propagation state, and
        # cross-track bookkeeping (`_record_track_overlaps`) already runs after
        # the loop, so the split is behaviour-preserving -- what it buys is one
        # batched image-encoder pass per frame instead of one per track (see
        # sam_backend.propagate_batch).
        plans = []
        for track in self.alive_tracks():
            if track.birth_keyframe > keyframe:
                continue  # seeded at a future keyframe (shouldn't happen)
            crops, origins = crops_fn(track, track.window_px or cfg.window_px)
            is_birth = track.last_keyframe is None
            if is_birth:
                track.prompt_box = self._box_in_window(
                    track._birth_pano_box, origins[0])
                plans.append((track, crops, origins, True,
                              track.prompt_box, None))
            elif track.prompt_box is not None:
                box = self._box_in_window(track._reanchor_pano_box, origins[0])
                plans.append((track, crops, origins, False, box, None))
            else:
                mask = self._mask_in_window(track, origins[0],
                                            crops[0].shape[0])
                if mask.sum() < cfg.min_mask_area_px:
                    self._close(track, keyframe, "mask_lost_in_window")
                    continue
                plans.append((track, crops, origins, False, None, mask))

        if not plans:
            with PROFILE.phase("track_overlaps"):
                self._record_track_overlaps(keyframe + 1)
            return

        with PROFILE.phase("propagate_batch", items=len(plans)):
            batched = self.backend.propagate_batch(
                [(crops, box, mask) for _, crops, _, _, box, mask in plans])
        # Small per-frame copies the backend made on the GPU while it was
        # preparing encoder input; the media hook uses them instead of walking
        # the full-size crops again on the CPU.
        previews = getattr(self.backend, "last_previews", None) or []

        for index, ((track, crops, origins, is_birth, prompt_box, _),
                    masks) in enumerate(zip(plans, batched)):
            frame_previews = previews[index] if index < len(previews) else None
            if is_birth:
                with PROFILE.phase("apply_mask_health", items=1):
                    health = mask_health(masks[0], prompt_box, cfg)
                track.birth_mask = masks[0]
                track.birth_origin = origins[0]
                track.records.append({
                    "keyframe": keyframe, "action": "birth",
                    "window_origin": [round(origins[0][0], 1), origins[0][1]],
                    "window_px": int(crops[0].shape[0]),
                    "health": health,
                })
                if not health["ok"]:
                    self._close(track, keyframe, f"birth_{health['reason']}")
                    self.rejected_births.append({
                        "obs_id": track.birth_obs_id, "keyframe": keyframe,
                        "health": health})
                    if self.on_interval:
                        with PROFILE.phase("media_on_interval", items=1):
                            self.on_interval(track, keyframe, crops, origins,
                                             masks, frame_previews)
                    continue
                track.end_keyframe = keyframe

            final = masks[-1]
            origin_last = origins[-1]
            track.last_keyframe = keyframe + 1
            track.last_mask = final
            track.last_origin = origin_last
            if final.sum() < cfg.min_mask_area_px:
                self._record_step(track, keyframe + 1, final, origin_last,
                                  [], "mask_dead")
                self._close(track, keyframe + 1, "mask_dead")
                if self.on_interval:
                    with PROFILE.phase("media_on_interval", items=1):
                        self.on_interval(track, keyframe, crops, origins, masks,
                                     frame_previews)
                continue

            with PROFILE.phase("score_detections", items=len(detections)):
                supports = self._score_detections(final, origin_last,
                                                  detections, det_pano_boxes)
            with PROFILE.phase("update_track", items=1):
                self._update_track(track, keyframe + 1, final, origin_last,
                                   supports, detections)
            if self.on_interval:
                with PROFILE.phase("media_on_interval", items=1):
                    self.on_interval(track, keyframe, crops, origins, masks,
                                     frame_previews)

        with PROFILE.phase("track_overlaps"):
            self._record_track_overlaps(keyframe + 1)
        self.seed_unassigned(keyframe + 1, detections, det_pano_boxes)

    def seed_unassigned(self, keyframe, detections, det_pano_boxes):
        """Detections that supported no track become new seeds. Also used to
        bootstrap the first keyframe of a run (no tracks -> all seed)."""
        claimed = set()
        for track in self.alive_tracks():
            if track.records and track.records[-1].get("keyframe") == keyframe:
                for s in track.records[-1].get("supports", []):
                    if s["class"] in SUPPORT_CLASSES:
                        claimed.add(s["obs_id"])
        for obs in detections:
            if obs.obs_id not in claimed:
                self.seed(obs, det_pano_boxes[obs.obs_id], keyframe)

    # -- internals ---------------------------------------------------------

    def _box_in_window(self, pano_box, origin):
        x0, y0 = origin
        rel_x = pg.signed_x_offset(pano_box[0], x0, self.pano_w)
        return [rel_x, pano_box[1] - y0,
                rel_x + (pano_box[2] - pano_box[0]), pano_box[3] - y0]

    def _mask_in_window(self, track: Track, new_origin, new_size: int):
        """Translate the track's stored mask into the (possibly resized)
        new window frame."""
        mask = track.prompt_mask
        old_x0, old_y0 = track._mask_origin  # noqa: SLF001
        dx = int(round(pg.signed_x_offset(old_x0, new_origin[0], self.pano_w)))
        dy = int(round(old_y0 - new_origin[1]))
        old_h, old_w = mask.shape
        out = np.zeros((new_size, new_size), dtype=bool)
        # new[r, c] = old[r - dy, c - dx]
        r0, r1 = max(0, dy), min(new_size, old_h + dy)
        c0, c1 = max(0, dx), min(new_size, old_w + dx)
        if r1 > r0 and c1 > c0:
            out[r0:r1, c0:c1] = mask[r0 - dy:r1 - dy, c0 - dx:c1 - dx]
        return out

    def _score_detections(self, mask, origin, detections, det_pano_boxes):
        window_px = mask.shape[1]
        supports = []
        for obs in detections:
            box = self._box_in_window(det_pano_boxes[obs.obs_id], origin)
            if box[2] < 0 or box[0] > window_px:
                continue
            metrics = mask_box_metrics(mask, box)
            cls = classify_support(metrics, self.cfg)
            supports.append({"obs_id": obs.obs_id, "class": cls,
                             "box_window": [round(v, 1) for v in box],
                             **{k: round(v, 3) for k, v in metrics.items()},
                             "_obs": obs})
        supports.sort(key=lambda s: (
            SUPPORT_PRIORITY.index(s["class"])
            if s["class"] in SUPPORT_PRIORITY else 99, -s["iou"]))
        return supports

    def _update_track(self, track, keyframe, mask, origin, supports,
                      detections):
        cfg = self.cfg
        supporting = [s for s in supports if s["class"] in SUPPORT_CLASSES]
        best = supporting[0] if supporting else None

        # Mask centroid in pano coords -> next window center.
        ys, xs = np.nonzero(mask)
        cx_w, cy_w = float(xs.mean()), float(ys.mean())
        track.center_x = (origin[0] + cx_w) % self.pano_w
        track.center_y = origin[1] + cy_w

        action = "continue_mask"
        track.prompt_box = None
        track.prompt_mask = mask
        track._mask_origin = origin  # noqa: SLF001

        if best is not None and best["class"] == "continue_clean":
            # Clean 1:1 -> re-anchor geometry on the detection (drift reset).
            pano_box = self._window_box_to_pano(best["box_window"], origin)
            track._reanchor_pano_box = pano_box  # noqa: SLF001
            track.prompt_box = pano_box  # marker; window box computed later
            track.prompt_mask = None
            track.center_x = (pano_box[0] + pano_box[2]) / 2.0 % self.pano_w
            track.center_y = (pano_box[1] + pano_box[3]) / 2.0
            action = "reanchor_clean"

        # Adapt the next window to the object's current extent (mask width
        # and any supporting detection's width - superset boxes reveal the
        # true extent even while the mask is eroded).
        ys, xs = np.nonzero(mask)
        extent = float(xs.max() - xs.min()) if len(xs) else 0.0
        for s in supporting:
            extent = max(extent, s["box_window"][2] - s["box_window"][0])
        track.window_px = window_size_for_extent(extent, cfg)

        if supporting:
            track.unsupported_streak = 0
            track.drift_streak = 0
            track.end_keyframe = keyframe
            track.ever_supported = True
            for s in supporting:
                self._vote(track, s["_obs"])
        else:
            track.unsupported_streak += 1
            if self._near_miss(track, mask, origin, supports):
                track.drift_streak += 1
            else:
                track.drift_streak = 0
            action = "unsupported"

        self._record_step(track, keyframe, mask, origin, supports, action)

        patience = (cfg.patience_keyframes if track.ever_supported
                    else cfg.patience_unsupported_keyframes)
        if track.drift_streak >= cfg.drift_patience:
            self._close(track, keyframe, "drift_alarm")
        elif track.unsupported_streak >= patience:
            self._close(track, keyframe, "starved")

    def _near_miss(self, track, mask, origin, supports) -> bool:
        """A detection matching the track's modal tag landed near the mask
        centroid but classified none -> evidence the mask slid off."""
        modal = track.tag_votes.most_common(1)
        if not modal:
            return False
        modal_tag = modal[0][0]
        ys, xs = np.nonzero(mask)
        cx, cy = float(xs.mean()), float(ys.mean())
        for s in supports:
            if s["class"] != "none":
                continue
            obs = s["_obs"]
            if f"{obs.primary_tag_key}={obs.primary_tag_value}" != modal_tag:
                continue
            bx = (s["box_window"][0] + s["box_window"][2]) / 2.0
            by = (s["box_window"][1] + s["box_window"][3]) / 2.0
            if math.hypot(bx - cx, by - cy) <= self.cfg.drift_gate_px:
                return True
        return False

    def _window_box_to_pano(self, box_window, origin):
        x0, y0 = origin
        return [x0 + box_window[0], y0 + box_window[1],
                x0 + box_window[2], y0 + box_window[3]]

    def _record_track_overlaps(self, keyframe: int):
        """Exact mask overlap for every co-alive track pair this keyframe."""
        live = [t for t in self.alive_tracks()
                if t.records and t.records[-1].get("keyframe") == keyframe
                and t.last_mask is not None and t.last_mask.any()]
        for i, ta in enumerate(live):
            for tb in live[i + 1:]:
                overlap = self._pair_overlap(ta, tb)
                if overlap is not None:
                    self.track_overlaps.append({
                        "keyframe": keyframe,
                        "track_a": ta.track_id, "track_b": tb.track_id,
                        **overlap})

    def _pair_overlap(self, ta: Track, tb: Track):
        ma, (ax0, ay0) = ta.last_mask, ta.last_origin
        mb, (bx0, by0) = tb.last_mask, tb.last_origin
        dx = int(round(pg.signed_x_offset(bx0, ax0, self.pano_w)))
        dy = int(round(by0 - ay0))
        # B's pixel (r, c) sits at A-window coords (r + dy, c + dx).
        ah, aw = ma.shape
        bh, bw = mb.shape
        r0, r1 = max(0, dy), min(ah, bh + dy)
        c0, c1 = max(0, dx), min(aw, bw + dx)
        if r1 <= r0 or c1 <= c0:
            return None
        inter = int(np.logical_and(
            ma[r0:r1, c0:c1], mb[r0 - dy:r1 - dy, c0 - dx:c1 - dx]).sum())
        if inter == 0:
            return None
        a_area, b_area = int(ma.sum()), int(mb.sum())
        return {
            "iou": round(inter / (a_area + b_area - inter), 3),
            "inter_over_min": round(inter / max(1, min(a_area, b_area)), 3),
        }

    def _record_step(self, track, keyframe, mask, origin, supports, action):
        ys, xs = np.nonzero(mask)
        bbox = ([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                if len(xs) else None)
        track.records.append({
            "keyframe": keyframe,
            "action": action,
            "window_origin": [round(origin[0], 1), origin[1]],
            "window_px": int(mask.shape[1]),
            "mask_area": int(mask.sum()),
            "mask_bbox_window": bbox,
            "supports": [{k: v for k, v in s.items() if k != "_obs"}
                         for s in supports],
        })

    def _close(self, track, keyframe, reason):
        track.status = "closed"
        track.close_reason = reason
        if track.end_keyframe is None:
            track.end_keyframe = keyframe
        track.prompt_mask = None
