"""Conservative temporal persistence for panoramic person masks.

The public entry point, :func:`bridge_one_frame_gap`, examines a three-frame
window.  It can promote a low-confidence mask in the middle frame, or synthesize
a mask-shaped (never rectangular) fill when accepted endpoint masks agree after
motion compensation.  Any evidence which fails the conservative gates is
returned as a review flag instead of being silently discarded.

All masks and optical flow fields live at the detector/flow resolution.  A
caller rendering at another resolution should resize masks with nearest-neighbor
interpolation.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import cv2
import numpy as np


@dataclasses.dataclass(frozen=True)
class PersistenceConfig:
    """Thresholds and Farneback settings for one-frame-gap consensus."""

    horizontal_padding_fraction: float = 0.125
    vertical_padding_fraction: float = 0.125
    farneback_pyr_scale: float = 0.5
    farneback_levels: int = 4
    farneback_window_size: int = 25
    farneback_iterations: int = 4
    farneback_poly_n: int = 7
    farneback_poly_sigma: float = 1.5
    max_cycle_error_px: float = 1.75
    min_cycle_valid_fraction: float = 0.85
    min_endpoint_iou: float = 0.45
    min_endpoint_coverage: float = 0.65
    max_endpoint_area_ratio: float = 1.8
    max_component_photometric_error: float = 0.22
    max_scene_histogram_distance: float = 0.72
    min_mask_pixels: int = 9
    max_fill_fraction: float = 0.25
    direct_match_iou: float = 0.35
    direct_match_coverage: float = 0.70
    direct_proposal_coverage_threshold: float = 0.75
    min_uncovered_consensus_depth_px: float = 3.5
    candidate_match_iou: float = 0.35
    candidate_match_coverage: float = 0.65
    candidate_endpoint_coverage: float = 0.55
    max_candidate_area_ratio: float = 2.0


@dataclasses.dataclass(frozen=True)
class GapFlows:
    """Dense flow fields for a previous/middle/next frame window.

    Each vector maps a pixel in the named source image to the corresponding
    pixel in the named target image.  Horizontal vector values should use the
    shortest local displacement across the panorama seam.
    """

    previous_to_middle: np.ndarray
    middle_to_previous: np.ndarray
    next_to_middle: np.ndarray
    middle_to_next: np.ndarray


@dataclasses.dataclass(frozen=True)
class AcceptedFill:
    """A temporal mask accepted for the middle frame."""

    mode: str
    mask: np.ndarray
    metrics: Mapping[str, float]

    def metadata(self) -> dict:
        return {
            "mode": self.mode,
            "pixel_count": int(np.count_nonzero(self.mask)),
            "metrics": dict(self.metrics),
        }


@dataclasses.dataclass(frozen=True)
class ReviewFlag:
    """Suspicious middle-frame evidence rejected by one or more gates."""

    reasons: tuple[str, ...]
    mask: np.ndarray
    metrics: Mapping[str, float]

    def metadata(self) -> dict:
        return {
            "reasons": list(self.reasons),
            "pixel_count": int(np.count_nonzero(self.mask)),
            "metrics": dict(self.metrics),
        }


@dataclasses.dataclass(frozen=True)
class PersistenceResult:
    """Output of :func:`bridge_one_frame_gap`.

    ``accepted_mask`` always contains every input middle accepted pixel.
    ``temporal_fill_mask`` contains only newly accepted temporal pixels, so a
    review renderer can color direct and persistent detections differently.
    """

    accepted_mask: np.ndarray
    temporal_fill_mask: np.ndarray
    fills: tuple[AcceptedFill, ...]
    review_flags: tuple[ReviewFlag, ...]
    metrics: Mapping[str, float]

    @property
    def review_required(self) -> bool:
        return bool(self.review_flags)

    def metadata(self) -> dict:
        return {
            "accepted_pixel_count": int(np.count_nonzero(self.accepted_mask)),
            "temporal_fill_pixel_count": int(
                np.count_nonzero(self.temporal_fill_mask)),
            "fills": [fill.metadata() for fill in self.fills],
            "review_flags": [flag.metadata() for flag in self.review_flags],
            "metrics": dict(self.metrics),
        }


@dataclasses.dataclass(frozen=True)
class _WarpedMask:
    raw_mask: np.ndarray
    valid_mask: np.ndarray
    cycle_error: np.ndarray
    photometric_error: np.ndarray


def _as_gray_u8(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[2] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim != 2:
        raise ValueError("frames must be HxW grayscale or HxWx3 BGR arrays")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    array = np.asarray(array, dtype=np.float32)
    if array.size and float(np.nanmax(array)) <= 1.0:
        array = array * 255.0
    return np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0).clip(
        0, 255).astype(np.uint8)


def _as_mask(mask: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return np.ascontiguousarray(array != 0)


def _pad_panorama(image: np.ndarray, config: PersistenceConfig) -> tuple[
        np.ndarray, int, int]:
    height, width = image.shape
    pad_x = min(width - 1, max(2, int(round(
        width * config.horizontal_padding_fraction)))) if width > 1 else 0
    pad_y = min(height - 1, max(2, int(round(
        height * config.vertical_padding_fraction)))) if height > 1 else 0
    padded = np.pad(image, ((0, 0), (pad_x, pad_x)), mode="wrap")
    vertical_mode = "reflect" if height > 1 else "edge"
    padded = np.pad(padded, ((pad_y, pad_y), (0, 0)), mode=vertical_mode)
    return np.ascontiguousarray(padded), pad_y, pad_x


def _estimate_flow(source: np.ndarray, target: np.ndarray,
                   config: PersistenceConfig) -> np.ndarray:
    source_padded, pad_y, pad_x = _pad_panorama(source, config)
    target_padded, _, _ = _pad_panorama(target, config)
    flow = cv2.calcOpticalFlowFarneback(
        source_padded, target_padded, None,
        config.farneback_pyr_scale,
        config.farneback_levels,
        config.farneback_window_size,
        config.farneback_iterations,
        config.farneback_poly_n,
        config.farneback_poly_sigma,
        0,
    )
    height, width = source.shape
    return np.ascontiguousarray(
        flow[pad_y:pad_y + height, pad_x:pad_x + width],
        dtype=np.float32,
    )


def estimate_gap_flows(previous_frame: np.ndarray, middle_frame: np.ndarray,
                       next_frame: np.ndarray,
                       config: PersistenceConfig | None = None) -> GapFlows:
    """Estimate panorama-aware forward/backward flows for a frame triplet."""
    config = config or PersistenceConfig()
    previous = _as_gray_u8(previous_frame)
    middle = _as_gray_u8(middle_frame)
    following = _as_gray_u8(next_frame)
    if previous.shape != middle.shape or following.shape != middle.shape:
        raise ValueError("all three frames must have the same dimensions")
    return GapFlows(
        previous_to_middle=_estimate_flow(previous, middle, config),
        middle_to_previous=_estimate_flow(middle, previous, config),
        next_to_middle=_estimate_flow(following, middle, config),
        middle_to_next=_estimate_flow(middle, following, config),
    )


def _validate_flows(flows: GapFlows, shape: tuple[int, int]) -> GapFlows:
    expected = (*shape, 2)
    fields = {}
    for field in dataclasses.fields(flows):
        value = np.asarray(getattr(flows, field.name), dtype=np.float32)
        if value.shape != expected:
            raise ValueError(
                f"{field.name} must have shape {expected}, got {value.shape}")
        fields[field.name] = np.ascontiguousarray(value)
    return GapFlows(**fields)


def _remap_panorama(values: np.ndarray, map_x: np.ndarray,
                    map_y: np.ndarray, interpolation: int) -> np.ndarray:
    height, width = values.shape[:2]
    wrapped_x = np.mod(map_x, width).astype(np.float32)
    bounded_y = np.clip(map_y, 0, height - 1).astype(np.float32)
    return cv2.remap(
        values, wrapped_x, bounded_y, interpolation=interpolation,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _warp_source_mask_to_target(
        source_mask: np.ndarray, source_gray: np.ndarray,
        target_gray: np.ndarray, source_to_target: np.ndarray,
        target_to_source: np.ndarray,
        config: PersistenceConfig) -> _WarpedMask:
    height, width = source_mask.shape
    grid_y, grid_x = np.indices((height, width), dtype=np.float32)
    source_x = grid_x + target_to_source[..., 0]
    source_y = grid_y + target_to_source[..., 1]
    vertical_valid = (source_y >= 0.0) & (source_y <= height - 1)
    finite = (np.isfinite(source_x) & np.isfinite(source_y)
              & np.isfinite(target_to_source).all(axis=2))

    sampled_mask = _remap_panorama(
        source_mask.astype(np.uint8), source_x, source_y,
        cv2.INTER_NEAREST) != 0
    raw_mask = sampled_mask & vertical_valid & finite

    reverse = _remap_panorama(
        source_to_target, source_x, source_y, cv2.INTER_LINEAR)
    cycle_x = target_to_source[..., 0] + reverse[..., 0]
    cycle_y = target_to_source[..., 1] + reverse[..., 1]
    cycle_error = np.hypot(cycle_x, cycle_y)
    cycle_valid = (finite & np.isfinite(reverse).all(axis=2)
                   & vertical_valid
                   & (cycle_error <= config.max_cycle_error_px))

    sampled_gray = _remap_panorama(
        source_gray, source_x, source_y, cv2.INTER_LINEAR)
    photometric_error = (
        np.abs(sampled_gray.astype(np.float32)
               - target_gray.astype(np.float32)) / 255.0)
    return _WarpedMask(
        raw_mask=raw_mask,
        valid_mask=cycle_valid,
        cycle_error=cycle_error.astype(np.float32),
        photometric_error=photometric_error,
    )


def _histogram_distance(first: np.ndarray, second: np.ndarray) -> float:
    histograms = []
    for image in (first, second):
        histogram = cv2.calcHist([image], [0], None, [32], [0, 256])
        cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
        histograms.append(histogram)
    return float(cv2.compareHist(
        histograms[0], histograms[1], cv2.HISTCMP_BHATTACHARYYA))


def _components(mask: np.ndarray, minimum_pixels: int = 1) -> list[np.ndarray]:
    """Return 8-connected components on a cylinder, without polar wrapping."""
    if not np.any(mask):
        return []
    # Put the cylindrical cut through the least occupied column.  A person
    # crossing x=0 is then one component, while y=0 and y=H-1 remain distinct.
    cut = int(np.argmin(np.count_nonzero(mask, axis=0)))
    rolled = np.roll(mask, -cut, axis=1).astype(np.uint8)
    count, labels = cv2.connectedComponents(rolled, connectivity=8)
    components = []
    for label in range(1, count):
        component = np.roll(labels == label, cut, axis=1)
        if int(np.count_nonzero(component)) >= minimum_pixels:
            components.append(component)
    return components


def _areas(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, float]:
    area_a = float(np.count_nonzero(mask_a))
    area_b = float(np.count_nonzero(mask_b))
    intersection = float(np.count_nonzero(mask_a & mask_b))
    union = area_a + area_b - intersection
    minimum = min(area_a, area_b)
    maximum = max(area_a, area_b)
    return {
        "area_a": area_a,
        "area_b": area_b,
        "intersection": intersection,
        "iou": intersection / union if union else 0.0,
        "minimum_coverage": intersection / minimum if minimum else 0.0,
        "area_ratio": maximum / minimum if minimum else float("inf"),
    }


def _median_on(values: np.ndarray, mask: np.ndarray,
               default: float = 1.0) -> float:
    selected = values[mask]
    return float(np.median(selected)) if selected.size else default


def _matching_component(mask: np.ndarray, components: list[np.ndarray],
                        minimum_iou: float,
                        minimum_coverage: float) -> tuple[int | None, dict]:
    best_index = None
    best_metrics = {}
    best_score = -1.0
    for index, component in enumerate(components):
        metrics = _areas(mask, component)
        score = metrics["iou"] + metrics["minimum_coverage"]
        if (metrics["iou"] >= minimum_iou
                and metrics["minimum_coverage"] >= minimum_coverage
                and score > best_score):
            best_index = index
            best_metrics = metrics
            best_score = score
    return best_index, best_metrics


def _scene_cut_reasons(metrics: Mapping[str, float],
                       config: PersistenceConfig) -> tuple[str, ...]:
    reasons = []
    if metrics["previous_scene_histogram_distance"] > (
            config.max_scene_histogram_distance):
        reasons.append("scene_cut_previous_to_middle")
    if metrics["next_scene_histogram_distance"] > (
            config.max_scene_histogram_distance):
        reasons.append("scene_cut_middle_to_next")
    return tuple(reasons)


def _distance_from_mask_on_cylinder(mask: np.ndarray) -> np.ndarray:
    """Measure distance from ``mask`` while treating panorama x as periodic."""
    if not np.any(mask):
        return np.full(mask.shape, np.inf, dtype=np.float32)
    width = mask.shape[1]
    tiled_background = np.tile(~mask, (1, 3)).astype(np.uint8)
    tiled_distance = cv2.distanceTransform(
        tiled_background, cv2.DIST_L2, cv2.DIST_MASK_5)
    return np.ascontiguousarray(tiled_distance[:, width:2 * width])


def _uncovered_consensus_review_flag(
        previous_component: np.ndarray,
        next_component: np.ndarray,
        middle_mask: np.ndarray,
        direct_proposal_coverage: float,
        config: PersistenceConfig) -> ReviewFlag | None:
    """Return meaningful endpoint consensus omitted by a direct detection.

    Thin contour differences and small fragments remain suppressed.  A
    mask-sized consensus component which reaches appreciably beyond the direct
    mask is retained as review evidence instead of disappearing behind the
    direct-coverage shortcut.
    """
    consensus = previous_component & next_component
    consensus_area = int(np.count_nonzero(consensus))
    if consensus_area < config.min_mask_pixels:
        return None

    uncovered = consensus & ~middle_mask
    if int(np.count_nonzero(uncovered)) < config.min_mask_pixels:
        return None
    distance_from_direct = _distance_from_mask_on_cylinder(middle_mask)
    evidence = np.zeros_like(uncovered)
    max_depth = 0.0
    component_count = 0
    for component in _components(uncovered, config.min_mask_pixels):
        # Use the maximum rather than median depth for the gate: a missing
        # head or limb can taper where it meets an otherwise good body mask.
        component_depth = float(np.max(distance_from_direct[component]))
        if component_depth < config.min_uncovered_consensus_depth_px:
            continue
        evidence |= component
        max_depth = max(max_depth, component_depth)
        component_count += 1

    evidence_area = int(np.count_nonzero(evidence))
    if not evidence_area:
        return None
    return ReviewFlag(
        reasons=("uncovered_endpoint_consensus",),
        mask=evidence,
        metrics={
            "direct_proposal_coverage": direct_proposal_coverage,
            "endpoint_consensus_pixel_count": float(consensus_area),
            "uncovered_consensus_pixel_count": float(evidence_area),
            "uncovered_consensus_fraction": evidence_area / consensus_area,
            "uncovered_consensus_component_count": float(component_count),
            "max_uncovered_consensus_depth_px": max_depth,
        },
    )


def bridge_one_frame_gap(
        previous_frame: np.ndarray,
        middle_frame: np.ndarray,
        next_frame: np.ndarray,
        previous_accepted_mask: np.ndarray,
        middle_accepted_mask: np.ndarray,
        next_accepted_mask: np.ndarray,
        middle_candidate_mask: np.ndarray | None = None,
        *,
        config: PersistenceConfig | None = None,
        flows: GapFlows | None = None) -> PersistenceResult:
    """Conservatively bridge missed person masks in one middle frame.

    A candidate is promoted only after both endpoints support it.  With no
    matching candidate, the union of the two motion-compensated endpoint masks
    is accepted only when flow-cycle, overlap, area, appearance, and scene-cut
    gates all pass.  Existing middle detections are never removed.

    Supplying ``flows`` is useful when a caller caches optical flow for adjacent
    windows.  Otherwise panorama-aware Farneback flow is estimated internally.
    """
    config = config or PersistenceConfig()
    previous = _as_gray_u8(previous_frame)
    middle = _as_gray_u8(middle_frame)
    following = _as_gray_u8(next_frame)
    if previous.shape != middle.shape or following.shape != middle.shape:
        raise ValueError("all three frames must have the same dimensions")
    shape = middle.shape
    previous_mask = _as_mask(
        previous_accepted_mask, shape, "previous_accepted_mask")
    middle_mask = _as_mask(
        middle_accepted_mask, shape, "middle_accepted_mask")
    next_mask = _as_mask(next_accepted_mask, shape, "next_accepted_mask")
    candidate_mask = (np.zeros(shape, dtype=bool)
                      if middle_candidate_mask is None else
                      _as_mask(middle_candidate_mask, shape,
                               "middle_candidate_mask"))

    if flows is None:
        flows = estimate_gap_flows(previous, middle, following, config)
    flows = _validate_flows(flows, shape)

    previous_warp = _warp_source_mask_to_target(
        previous_mask, previous, middle,
        flows.previous_to_middle, flows.middle_to_previous, config)
    next_warp = _warp_source_mask_to_target(
        next_mask, following, middle,
        flows.next_to_middle, flows.middle_to_next, config)

    metrics = {
        "previous_scene_histogram_distance": _histogram_distance(
            previous, middle),
        "next_scene_histogram_distance": _histogram_distance(
            middle, following),
        "previous_warp_pixel_count": float(np.count_nonzero(
            previous_warp.raw_mask)),
        "next_warp_pixel_count": float(np.count_nonzero(
            next_warp.raw_mask)),
        "max_direct_proposal_coverage": 0.0,
        "direct_coverage_short_circuit_count": 0.0,
    }
    scene_reasons = _scene_cut_reasons(metrics, config)
    review_flags: list[ReviewFlag] = []
    fills: list[AcceptedFill] = []
    temporal_fill = np.zeros(shape, dtype=bool)

    # Endpoint remnants below ``min_mask_pixels`` are intentionally discarded:
    # a one-pixel flow artifact should not make an otherwise quiet frame need
    # review.  Middle-frame candidates are different evidence.  Even a tiny
    # low-confidence detection can be a real missed face/person, so preserve
    # its uncovered pixels for a human while keeping it out of the components
    # which are eligible for temporal promotion.
    candidate_components = []
    eligible_candidate_mask = np.zeros(shape, dtype=bool)
    for candidate in _components(candidate_mask):
        candidate_pixel_count = int(np.count_nonzero(candidate))
        if candidate_pixel_count >= config.min_mask_pixels:
            candidate_components.append(candidate)
            eligible_candidate_mask |= candidate
            continue
        uncovered = candidate & ~middle_mask
        uncovered_pixel_count = int(np.count_nonzero(uncovered))
        if uncovered_pixel_count:
            review_flags.append(ReviewFlag(
                reasons=("candidate_too_small",),
                mask=uncovered,
                metrics={
                    "pixel_count": float(candidate_pixel_count),
                    "uncovered_pixel_count": float(uncovered_pixel_count),
                    "min_mask_pixels": float(config.min_mask_pixels),
                },
            ))

    if scene_reasons:
        evidence = (previous_warp.raw_mask | next_warp.raw_mask
                    | eligible_candidate_mask) & ~middle_mask
        if np.any(evidence):
            review_flags.append(ReviewFlag(
                reasons=scene_reasons,
                mask=evidence,
                metrics=dict(metrics),
            ))
        return PersistenceResult(
            accepted_mask=middle_mask.copy(),
            temporal_fill_mask=temporal_fill,
            fills=(),
            review_flags=tuple(review_flags),
            metrics=metrics,
        )

    previous_components = _components(
        previous_warp.raw_mask, config.min_mask_pixels)
    next_components = _components(
        next_warp.raw_mask, config.min_mask_pixels)
    direct_components = _components(middle_mask, config.min_mask_pixels)
    used_previous: set[int] = set()
    used_next: set[int] = set()
    used_candidates: set[int] = set()

    pair_options = []
    for previous_index, previous_component in enumerate(previous_components):
        for next_index, next_component in enumerate(next_components):
            raw_metrics = _areas(previous_component, next_component)
            if raw_metrics["intersection"]:
                pair_options.append((
                    raw_metrics["iou"] + raw_metrics["minimum_coverage"],
                    previous_index, next_index,
                ))
    pair_options.sort(reverse=True)

    for _, previous_index, next_index in pair_options:
        if previous_index in used_previous or next_index in used_next:
            continue
        used_previous.add(previous_index)
        used_next.add(next_index)
        previous_raw = previous_components[previous_index]
        next_raw = next_components[next_index]
        raw_proposal = previous_raw | next_raw
        raw_proposal_area = max(1, int(np.count_nonzero(raw_proposal)))
        raw_direct_coverage = float(np.count_nonzero(
            raw_proposal & middle_mask)) / raw_proposal_area
        metrics["max_direct_proposal_coverage"] = max(
            metrics["max_direct_proposal_coverage"], raw_direct_coverage)

        # Component matching below catches one strong direct instance.  This
        # union-level gate also catches a direct mask split into fragments and
        # small endpoint boundary jitter: if most of the temporal proposal is
        # already private, its uncovered edge is not a meaningful missed
        # region and should not become a temporal fill.
        if raw_direct_coverage >= (
                config.direct_proposal_coverage_threshold):
            metrics["direct_coverage_short_circuit_count"] += 1.0
            partial_flag = _uncovered_consensus_review_flag(
                previous_raw, next_raw, middle_mask,
                raw_direct_coverage, config)
            if partial_flag is not None:
                review_flags.append(partial_flag)
            for candidate_index, candidate in enumerate(candidate_components):
                if np.any(candidate & raw_proposal):
                    used_candidates.add(candidate_index)
            continue

        # A strong direct detection is authoritative for this middle frame.
        # Check it against the ungated motion-compensated evidence before flow
        # cycle and appearance gates: those gates exist to decide whether to
        # add a missing mask, and must not create review noise around a person
        # which is already covered by a direct mask.
        direct_index, _ = _matching_component(
            raw_proposal, direct_components,
            config.direct_match_iou, config.direct_match_coverage)
        if direct_index is not None:
            direct_component = direct_components[direct_index]
            partial_flag = _uncovered_consensus_review_flag(
                previous_raw, next_raw, middle_mask,
                raw_direct_coverage, config)
            if partial_flag is not None:
                review_flags.append(partial_flag)
            for candidate_index, candidate in enumerate(candidate_components):
                if np.any(candidate & direct_component):
                    used_candidates.add(candidate_index)
            continue

        previous_valid = previous_raw & previous_warp.valid_mask
        next_valid = next_raw & next_warp.valid_mask
        endpoint_metrics = _areas(previous_valid, next_valid)
        previous_raw_area = max(1, int(np.count_nonzero(previous_raw)))
        next_raw_area = max(1, int(np.count_nonzero(next_raw)))
        pair_metrics = {
            "endpoint_iou": endpoint_metrics["iou"],
            "endpoint_minimum_coverage": endpoint_metrics[
                "minimum_coverage"],
            "endpoint_area_ratio": endpoint_metrics["area_ratio"],
            "previous_cycle_valid_fraction": float(np.count_nonzero(
                previous_valid)) / previous_raw_area,
            "next_cycle_valid_fraction": float(np.count_nonzero(
                next_valid)) / next_raw_area,
            "previous_median_cycle_error_px": _median_on(
                previous_warp.cycle_error, previous_raw),
            "next_median_cycle_error_px": _median_on(
                next_warp.cycle_error, next_raw),
            "previous_median_photometric_error": _median_on(
                previous_warp.photometric_error, previous_valid),
            "next_median_photometric_error": _median_on(
                next_warp.photometric_error, next_valid),
        }
        proposal = previous_valid | next_valid
        pair_metrics["proposal_pixel_count"] = float(
            np.count_nonzero(proposal))
        pair_metrics["proposal_frame_fraction"] = float(
            np.count_nonzero(proposal)) / proposal.size
        proposal_area = max(1, int(np.count_nonzero(proposal)))
        direct_proposal_coverage = float(np.count_nonzero(
            proposal & middle_mask)) / proposal_area
        pair_metrics["direct_proposal_coverage"] = (
            direct_proposal_coverage)
        metrics["max_direct_proposal_coverage"] = max(
            metrics["max_direct_proposal_coverage"],
            direct_proposal_coverage)

        if direct_proposal_coverage >= (
                config.direct_proposal_coverage_threshold):
            metrics["direct_coverage_short_circuit_count"] += 1.0
            partial_flag = _uncovered_consensus_review_flag(
                previous_valid, next_valid, middle_mask,
                direct_proposal_coverage, config)
            if partial_flag is not None:
                review_flags.append(partial_flag)
            for candidate_index, candidate in enumerate(candidate_components):
                if np.any(candidate & proposal):
                    used_candidates.add(candidate_index)
            continue

        reasons = []
        if endpoint_metrics["iou"] < config.min_endpoint_iou:
            reasons.append("endpoint_iou_below_threshold")
        if endpoint_metrics["minimum_coverage"] < (
                config.min_endpoint_coverage):
            reasons.append("endpoint_overlap_below_threshold")
        if endpoint_metrics["area_ratio"] > config.max_endpoint_area_ratio:
            reasons.append("endpoint_area_ratio_too_large")
        if pair_metrics["previous_cycle_valid_fraction"] < (
                config.min_cycle_valid_fraction):
            reasons.append("previous_flow_cycle_invalid")
        if pair_metrics["next_cycle_valid_fraction"] < (
                config.min_cycle_valid_fraction):
            reasons.append("next_flow_cycle_invalid")
        if pair_metrics["previous_median_photometric_error"] > (
                config.max_component_photometric_error):
            reasons.append("previous_appearance_mismatch")
        if pair_metrics["next_median_photometric_error"] > (
                config.max_component_photometric_error):
            reasons.append("next_appearance_mismatch")
        if pair_metrics["proposal_pixel_count"] < config.min_mask_pixels:
            reasons.append("proposal_too_small")
        if pair_metrics["proposal_frame_fraction"] > config.max_fill_fraction:
            reasons.append("proposal_too_large")
        if reasons:
            review_flags.append(ReviewFlag(
                reasons=tuple(reasons),
                mask=(previous_raw | next_raw) & ~middle_mask,
                metrics=pair_metrics,
            ))
            continue

        direct_index, _ = _matching_component(
            proposal, direct_components,
            config.direct_match_iou, config.direct_match_coverage)
        if direct_index is not None:
            partial_flag = _uncovered_consensus_review_flag(
                previous_valid, next_valid, middle_mask,
                direct_proposal_coverage, config)
            if partial_flag is not None:
                review_flags.append(partial_flag)
            for candidate_index, candidate in enumerate(candidate_components):
                if np.any(candidate & direct_components[direct_index]):
                    used_candidates.add(candidate_index)
            continue

        candidate_index, candidate_metrics = _matching_component(
            proposal, candidate_components,
            config.candidate_match_iou, config.candidate_match_coverage)
        chosen_mask = proposal
        mode = "synthesized_endpoint_consensus"
        if candidate_index is not None:
            candidate = candidate_components[candidate_index]
            proposal_area = max(1, int(np.count_nonzero(proposal)))
            endpoint_coverage = float(np.count_nonzero(
                candidate & proposal)) / proposal_area
            area_ratio = _areas(candidate, proposal)["area_ratio"]
            if (endpoint_coverage >= config.candidate_endpoint_coverage
                    and area_ratio <= config.max_candidate_area_ratio):
                chosen_mask = candidate
                mode = "promoted_middle_candidate"
                used_candidates.add(candidate_index)
                pair_metrics.update({
                    "candidate_iou": candidate_metrics["iou"],
                    "candidate_minimum_coverage": candidate_metrics[
                        "minimum_coverage"],
                    "candidate_endpoint_coverage": endpoint_coverage,
                    "candidate_area_ratio": area_ratio,
                })

        new_pixels = chosen_mask & ~middle_mask
        if not np.any(new_pixels):
            continue
        temporal_fill |= new_pixels
        fills.append(AcceptedFill(
            mode=mode,
            mask=new_pixels,
            metrics=dict(pair_metrics),
        ))

    for index, component in enumerate(previous_components):
        if index not in used_previous:
            review_flags.append(ReviewFlag(
                reasons=("endpoint_disagreement",),
                mask=component & ~middle_mask,
                metrics={"supported_endpoint": 0.0,
                         "pixel_count": float(np.count_nonzero(component))},
            ))
    for index, component in enumerate(next_components):
        if index not in used_next:
            review_flags.append(ReviewFlag(
                reasons=("endpoint_disagreement",),
                mask=component & ~middle_mask,
                metrics={"supported_endpoint": 2.0,
                         "pixel_count": float(np.count_nonzero(component))},
            ))

    for index, candidate in enumerate(candidate_components):
        if index in used_candidates:
            continue
        direct_index, _ = _matching_component(
            candidate, direct_components,
            config.direct_match_iou, config.direct_match_coverage)
        if direct_index is None:
            review_flags.append(ReviewFlag(
                reasons=("unconfirmed_middle_candidate",),
                mask=candidate & ~middle_mask,
                metrics={"pixel_count": float(np.count_nonzero(candidate))},
            ))

    return PersistenceResult(
        accepted_mask=middle_mask | temporal_fill,
        temporal_fill_mask=temporal_fill,
        fills=tuple(fills),
        review_flags=tuple(review_flags),
        metrics=metrics,
    )
