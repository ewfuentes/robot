"""Track consolidation: group tracks that observe the same physical object.

The decisive signal is geometric and needs no ego-motion: two tracks alive at
the SAME keyframe are in the same camera frame, so their mask positions can
be compared directly. Tracks whose masks sit apart at a shared keyframe
cannot be one object, no matter how well their names agree - on
boston_harbor leg1 that constraint rejects 4 of 7 name-agreeing groups
(three tracks each claiming 'Custom House Tower' were pairwise co-visible
10-24 deg apart).

So merging is constraint satisfaction, not similarity search:

  co-visible + coincident masks -> duplicate, MERGE (geometry alone)
  co-visible + one inside other -> parent/child LINK, never a merge
  co-visible + partial overlap  -> AMBIGUOUS: geometry declines to decide,
                                   emitted for adjudication (see below)
  co-visible + disjoint masks   -> CANNOT-LINK (hard, beats any name match)
  never co-visible              -> handoff PROPOSAL only; needs ego-motion
                                   to verify, so it is never auto-merged

Under-merging is cheap (two tracklets of one object both match the same map
feature and the filter's data association copes). Over-merging is poison: it
welds two objects into one landmark with a bimodal bearing. Every ambiguous
case therefore resolves to "don't merge".

Semantics are deliberately NOT used to merge. They ride along on the output
so the matcher can use them, and name disagreement within a merged group is
reported rather than resolved - only a map can settle it.
"""

import math
from dataclasses import dataclass, field
from itertools import combinations

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    pano_geometry as pg,
)

DUPLICATE = "duplicate"
PARENT_CHILD = "parent_child"
# Meaningful overlap, but not coincident enough to merge nor separate enough
# to rule out. Measured on leg1, this band genuinely mixes true duplicates
# (Tobin Bridge iou 0.39, Commonwealth Pier 0.25) with genuinely different
# adjacent buildings (Boston Harbor Hotel vs One International Place, 0.23),
# and no threshold on iou, angular separation, or separation normalized by
# object width splits them. So geometry declines to decide: these neither
# merge nor assert a cannot-link, and are emitted for adjudication. This is
# where semantic review earns its keep - unlike the clean cases, where it
# would only add noise.
AMBIGUOUS = "ambiguous"
DISTINCT = "distinct"
DISJOINT = "disjoint"


@dataclass
class MergeConfig:
    # Mask-bbox IoU at shared keyframes above which two tracks are the same
    # object. Scale-free, so it works for point-like towers and islands
    # alike (a fixed angle would not).
    duplicate_min_iou: float = 0.50
    # Asymmetric containment: the smaller mask lies inside the larger, and
    # the larger is meaningfully bigger. Fort-on-island, not duplication.
    child_min_containment: float = 0.70
    child_max_area_frac: float = 0.50
    # Below this iou the masks share almost nothing and a cannot-link is
    # safe to assert; between here and duplicate_min_iou is the AMBIGUOUS
    # band that geometry cannot resolve.
    ambiguous_min_iou: float = 0.05
    # Minimum shared keyframes before a co-visible verdict is trusted.
    min_covisible_keyframes: int = 3
    # Handoff proposals: max keyframe gap between one track ending and the
    # next beginning for the pair to be worth ego-motion verification.
    handoff_max_gap: int = 30


@dataclass
class PairStats:
    track_a: int
    track_b: int
    verdict: str
    n_covisible: int
    median_iou: float
    median_sep_deg: float
    # Containment of the smaller mask in the larger, median over shared
    # keyframes; only meaningful for PARENT_CHILD.
    median_containment: float = 0.0
    parent: int | None = None
    child: int | None = None
    gap_keyframes: int | None = None


@dataclass
class MergedLandmark:
    landmark_id: str
    track_ids: list
    n_supports: int
    n_supported_keyframes: int
    keyframe_span: tuple
    name_votes: dict = field(default_factory=dict)
    tag_votes: dict = field(default_factory=dict)
    name_contested: bool = False
    child_of: list = field(default_factory=list)
    parent_of: list = field(default_factory=list)
    handoff_proposals: list = field(default_factory=list)
    review_pairs: list = field(default_factory=list)
    merge_conflicts: list = field(default_factory=list)


def mask_boxes_by_keyframe(track: dict, valid_segments=None) -> dict:
    """keyframe -> mask bbox in pano coords, restricted to valid segments.

    Segments come from the semantic audit (relative time indices); applying
    them BEFORE co-visibility matters, or a track's own drifted tail makes it
    look separated from its earlier self.
    """
    birth = track["birth_keyframe"]
    spans = None
    if valid_segments:
        spans = [(birth + s["start_t"], birth + s["end_t"])
                 for s in valid_segments]
    out = {}
    for rec in track["records"]:
        mb = rec.get("mask_bbox_window")
        if mb is None:
            continue
        kf = rec["keyframe"]
        if spans is not None and not any(a <= kf <= b for a, b in spans):
            continue
        ox, oy = rec["window_origin"]
        out[kf] = (ox + mb[0], oy + mb[1], ox + mb[2], oy + mb[3])
    return out


def _wrapped_dx(xa: float, xb: float, pano_w: int) -> float:
    """Signed x offset from xa to xb, shortest way around the wrap."""
    return (xb - xa + pano_w / 2.0) % pano_w - pano_w / 2.0


def box_overlap(box_a, box_b, pano_w: int):
    """(iou, containment_of_smaller, area_frac_smaller_over_larger) for two
    pano bboxes, wrap-safe in x."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    # Re-anchor b next to a across the wrap.
    bx0_rel = ax0 + _wrapped_dx(ax0, bx0, pano_w)
    bx1_rel = bx0_rel + (bx1 - bx0)
    ix = max(0.0, min(ax1, bx1_rel) - max(ax0, bx0_rel))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    area_a = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1e-9, (bx1 - bx0) * (by1 - by0))
    union = area_a + area_b - inter
    smaller, larger = min(area_a, area_b), max(area_a, area_b)
    return (inter / union if union > 0 else 0.0,
            inter / smaller if smaller > 0 else 0.0,
            smaller / larger)


def angular_separation_deg(box_a, box_b, pano_w: int) -> float:
    ca = (box_a[0] + box_a[2]) / 2.0
    cb = (box_b[0] + box_b[2]) / 2.0
    return abs(_wrapped_dx(ca, cb, pano_w)) / pano_w * 360.0


def _median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def compare_pair(track_a: dict, boxes_a: dict, track_b: dict, boxes_b: dict,
                 pano_w: int, cfg: MergeConfig) -> PairStats:
    """Geometric verdict for one track pair."""
    shared = sorted(set(boxes_a) & set(boxes_b))
    tid_a, tid_b = track_a["track_id"], track_b["track_id"]

    if not shared:
        span_a = (min(boxes_a), max(boxes_a)) if boxes_a else None
        span_b = (min(boxes_b), max(boxes_b)) if boxes_b else None
        gap = None
        if span_a and span_b:
            gap = max(span_a[0], span_b[0]) - min(span_a[1], span_b[1]) - 1
        return PairStats(tid_a, tid_b, DISJOINT, 0, 0.0, 0.0,
                         gap_keyframes=gap)

    ious, seps, contains, area_fracs = [], [], [], []
    for kf in shared:
        iou, containment, area_frac = box_overlap(boxes_a[kf], boxes_b[kf],
                                                  pano_w)
        ious.append(iou)
        contains.append(containment)
        area_fracs.append(area_frac)
        seps.append(angular_separation_deg(boxes_a[kf], boxes_b[kf], pano_w))

    med_iou, med_sep = _median(ious), _median(seps)
    med_contain, med_area = _median(contains), _median(area_fracs)

    # Too few shared keyframes to trust either way: refuse to merge, but do
    # not assert a hard cannot-link on such thin evidence.
    if len(shared) < cfg.min_covisible_keyframes:
        return PairStats(tid_a, tid_b, DISJOINT, len(shared), med_iou,
                         med_sep, med_contain)

    if med_iou >= cfg.duplicate_min_iou:
        return PairStats(tid_a, tid_b, DUPLICATE, len(shared), med_iou,
                         med_sep, med_contain)

    if (med_contain >= cfg.child_min_containment
            and med_area <= cfg.child_max_area_frac):
        area_a = ((boxes_a[shared[0]][2] - boxes_a[shared[0]][0])
                  * (boxes_a[shared[0]][3] - boxes_a[shared[0]][1]))
        area_b = ((boxes_b[shared[0]][2] - boxes_b[shared[0]][0])
                  * (boxes_b[shared[0]][3] - boxes_b[shared[0]][1]))
        parent, child = (tid_a, tid_b) if area_a >= area_b else (tid_b, tid_a)
        return PairStats(tid_a, tid_b, PARENT_CHILD, len(shared), med_iou,
                         med_sep, med_contain, parent=parent, child=child)

    if med_iou >= cfg.ambiguous_min_iou:
        return PairStats(tid_a, tid_b, AMBIGUOUS, len(shared), med_iou,
                         med_sep, med_contain)

    return PairStats(tid_a, tid_b, DISTINCT, len(shared), med_iou, med_sep,
                     med_contain)


def cluster(pair_stats, track_ids, cfg: MergeConfig):
    """Connected components over DUPLICATE edges, with CANNOT-LINK enforced.

    A component that contains a DISTINCT pair is contradictory (A~B, B~C, but
    A and C provably differ). Rather than silently keeping a bad weld, the
    weakest duplicate edge in the component is dropped and the component
    re-split; every dropped edge is reported as a conflict.
    """
    cannot = {(min(p.track_a, p.track_b), max(p.track_a, p.track_b))
              for p in pair_stats if p.verdict == DISTINCT}
    dup_edges = sorted(
        (p for p in pair_stats if p.verdict == DUPLICATE),
        key=lambda p: -p.median_iou)

    conflicts = []
    while True:
        parent = {t: t for t in track_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for p in dup_edges:
            ra, rb = find(p.track_a), find(p.track_b)
            if ra != rb:
                parent[ra] = rb

        groups = {}
        for t in track_ids:
            groups.setdefault(find(t), []).append(t)

        bad_edge = None
        for members in groups.values():
            violated = [(a, b) for a, b in combinations(sorted(members), 2)
                        if (a, b) in cannot]
            if violated:
                # Drop the weakest duplicate edge inside this component.
                inside = [p for p in dup_edges
                          if p.track_a in members and p.track_b in members]
                if inside:
                    bad_edge = min(inside, key=lambda p: p.median_iou)
                    conflicts.append({
                        "dropped_edge": (bad_edge.track_a, bad_edge.track_b),
                        "median_iou": bad_edge.median_iou,
                        "contradicted_by": violated})
                break
        if bad_edge is None:
            return groups, conflicts
        dup_edges.remove(bad_edge)


def merge_tracks(tracks, dossiers, evidences, audits, pano_w: int,
                 cfg: MergeConfig):
    """Full consolidation pass.

    tracks/dossiers/evidences: {track_id: ...}. audits: {track_id: audit dict}
    or {} when audits are unavailable (then whole tracks are used).
    Returns (landmarks, pair_stats).
    """
    boxes = {}
    for tid, track in tracks.items():
        audit = audits.get(tid) or {}
        boxes[tid] = mask_boxes_by_keyframe(track, audit.get("valid_segments"))

    track_ids = sorted(tracks)
    stats = []
    for tid_a, tid_b in combinations(track_ids, 2):
        if not boxes[tid_a] or not boxes[tid_b]:
            continue
        stats.append(compare_pair(tracks[tid_a], boxes[tid_a], tracks[tid_b],
                                  boxes[tid_b], pano_w, cfg))

    groups, conflicts = cluster(stats, track_ids, cfg)

    by_pair = {(min(p.track_a, p.track_b), max(p.track_a, p.track_b)): p
               for p in stats}
    group_of = {t: root for root, members in groups.items() for t in members}

    landmarks = {}
    for root, members in groups.items():
        members = sorted(members)
        name_votes, tag_votes = {}, {}
        n_sup = n_supkf = 0
        lo = hi = None
        for tid in members:
            ev = evidences[tid]
            n_sup += ev["n_supports"]
            n_supkf += ev["n_supported_keyframes"]
            for name, count in ev["name_votes"].items():
                name_votes[name] = name_votes.get(name, 0) + count
            for tag, count in ev["tag_votes"].items():
                tag_votes[tag] = tag_votes.get(tag, 0) + count
            if boxes[tid]:
                lo = min(lo, min(boxes[tid])) if lo is not None \
                    else min(boxes[tid])
                hi = max(hi, max(boxes[tid])) if hi is not None \
                    else max(boxes[tid])
        total_named = sum(name_votes.values())
        top = max(name_votes.values()) if name_votes else 0
        runner = sorted(name_votes.values(), reverse=True)[1] \
            if len(name_votes) > 1 else 0
        landmarks[root] = MergedLandmark(
            landmark_id="L" + "_".join(f"T{t}" for t in members),
            track_ids=members,
            n_supports=n_sup,
            n_supported_keyframes=n_supkf,
            keyframe_span=(lo, hi),
            name_votes=dict(sorted(name_votes.items(),
                                   key=lambda kv: -kv[1])),
            tag_votes=dict(sorted(tag_votes.items(), key=lambda kv: -kv[1])),
            name_contested=bool(
                total_named and (top / total_named < 0.5 or top < 2 * runner)),
            merge_conflicts=[c for c in conflicts
                             if c["dropped_edge"][0] in members
                             or c["dropped_edge"][1] in members])

    # Parent/child links and handoff proposals, expressed between landmarks.
    for p in stats:
        la = landmarks[group_of[p.track_a]]
        lb = landmarks[group_of[p.track_b]]
        if la is lb:
            continue
        if p.verdict == AMBIGUOUS:
            entry = {"with": lb.landmark_id, "median_iou": p.median_iou,
                     "median_sep_deg": p.median_sep_deg,
                     "n_covisible": p.n_covisible,
                     "status": "geometry_inconclusive"}
            if entry not in la.review_pairs:
                la.review_pairs.append(entry)
        elif p.verdict == PARENT_CHILD:
            parent_l = landmarks[group_of[p.parent]]
            child_l = landmarks[group_of[p.child]]
            if child_l.landmark_id not in parent_l.parent_of:
                parent_l.parent_of.append(child_l.landmark_id)
            if parent_l.landmark_id not in child_l.child_of:
                child_l.child_of.append(parent_l.landmark_id)
        elif (p.verdict == DISJOINT and p.gap_keyframes is not None
              and 0 <= p.gap_keyframes <= cfg.handoff_max_gap):
            shared_names = set(la.name_votes) & set(lb.name_votes)
            shared_tags = set(la.tag_votes) & set(lb.tag_votes)
            if shared_names or shared_tags:
                proposal = {
                    "with": lb.landmark_id,
                    "gap_keyframes": p.gap_keyframes,
                    "shared_names": sorted(shared_names),
                    "shared_tags": sorted(shared_tags),
                    "status": "needs_ego_motion_check"}
                if proposal not in la.handoff_proposals:
                    la.handoff_proposals.append(proposal)

    return list(landmarks.values()), stats


def bearing_series(track: dict, pano_w: int, valid_segments=None):
    """[(keyframe, az_cw_deg, angular_width_deg)] from the tracked mask.

    Camera-frame azimuth (pano_geometry convention, CW positive). The
    localization filter wants BODY-frame bearings, which differ by the fixed
    camera-to-body mount offset only - no per-frame heading is involved.
    Angular width is the mask's own extent, the basis for the measurement's
    concentration.
    """
    out = []
    for kf, box in sorted(mask_boxes_by_keyframe(track,
                                                 valid_segments).items()):
        centre = (box[0] + box[2]) / 2.0
        # Single source of truth for the pano<->direction convention.
        az, _ = pg.direction_from_pano_px(centre % pano_w, 0.0, pano_w, 1)
        width = (box[2] - box[0]) / pano_w * 360.0
        out.append((kf, az, width))
    return out


def fuse_bearings(series, epoch_keyframes: int, bearing_sigma_deg: float):
    """Fuse a bearing series into sparse per-epoch measurements.

    Returns [(anchor_keyframe, az_cw_deg, kappa)]. The filter consumes one
    fused bearing per tracklet per information epoch rather than one per
    keyframe: consecutive bearings on one object are strongly correlated
    (same mask, same tracker), so treating them as independent would
    overcount evidence.

    kappa combines the per-observation concentration with the object's own
    angular width - an extended object's centroid is not a point bearing -
    and does NOT grow with the number of fused keyframes, which is the
    conservative choice while the correlation is unmodelled.
    """
    if not series:
        return []
    fused = []
    epoch = max(1, epoch_keyframes)
    start_kf = series[0][0]
    bucket = []

    def flush(bucket):
        if not bucket:
            return
        # Circular mean of the bucket's azimuths.
        sin_sum = sum(math.sin(math.radians(a)) for _, a, _ in bucket)
        cos_sum = sum(math.cos(math.radians(a)) for _, a, _ in bucket)
        mean_az = math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0
        mean_width = sum(w for _, _, w in bucket) / len(bucket)
        anchor = bucket[len(bucket) // 2][0]
        # Width contributes a centroid ambiguity of about a quarter-width.
        sigma = math.hypot(bearing_sigma_deg, mean_width / 4.0)
        kappa = 1.0 / math.radians(sigma) ** 2
        fused.append((anchor, mean_az, kappa))

    for entry in series:
        if entry[0] - start_kf >= epoch:
            flush(bucket)
            bucket = []
            start_kf = entry[0]
        bucket.append(entry)
    flush(bucket)
    return fused
