"""Per-track semantic audit: dossier + VLM request construction.

Level-1 canonicalization. For each track built by M3, assemble a "dossier"
(procedurally generated evidence summary + representative image chips) and
turn it into a Vertex AI request. The model adjudicates ONLY semantics; all
geometry (support classification, context screening) is decided by
track_builder rules before the model ever sees the track.

Design constraints (from the r002 hand review):
- The dossier contains NO dataset / run / location identifiers and no
  hand-written interpretation - every line must be derivable from the track
  artifact alone. Keyframes are presented as relative time indices t0..tN.
- Temporal structure is conveyed by run-length-encoded sequences, not by
  narrative ("X dominates early") which cannot be generated procedurally.
- Tag votes cover ALL tags (primary and additional), with per-detector-
  confidence counts instead of an opaque weighted score.
- The model outputs categorical judgments and matching keys, never scalar
  reliability scores - those are computed downstream from geometry.

The request JSONL format matches vertex_batch_manager (`run-online` /
batch submission): {"key", "request": {contents, systemInstruction,
generationConfig}} with inline_data image parts.
"""

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    pano_geometry as pg,
    track_builder as tb,
)

# Tags that are reported as additional_tags but are not identity evidence;
# they get their own dossier sections instead of rows in the tag table.
NON_IDENTITY_TAG_KEYS = ("name", "distance_estimate")

CONFIDENCE_ORDER = ("high", "medium", "low")

DETECTION_COLOR = (40, 200, 70)   # green
MASK_COLOR = (255, 60, 60)        # red


@dataclass
class AuditConfig:
    # 2 supports = 3 detections counting the birth. The bar was 3 during the
    # hand review, where the point was to study tracks with plenty of
    # evidence; for production that threw away 22 auditable tracks for no
    # reason. The audit works on three detections, it just has less to
    # reconcile - and a thin track is where its judgement matters most,
    # because vote-counting has almost nothing to work with.
    min_supports: int = 2
    max_support_chips: int = 6
    max_context_chips: int = 2
    max_description_samples: int = 10
    chip_height_px: int = 320
    thinking_level: str = "HIGH"
    classifier: tb.TrackBuilderConfig = None

    def __post_init__(self):
        if self.classifier is None:
            self.classifier = tb.TrackBuilderConfig()


# ---------------------------------------------------------------------------
# response schema (pydantic -> Gemini responseSchema)
# ---------------------------------------------------------------------------
# No Optional fields: Gemini structured output handles required-everything
# schemas most reliably (same convention as semantic_landmark_extractor).
# "none"/""/[] are the explicit empty sentinels.

class WeightedTag(BaseModel):
    tag: str          # "key=value", extraction vocabulary
    weight: float     # 0..1, relative belief this is the object's OSM tag


class Segment(BaseModel):
    start_t: int
    end_t: int


class StrikeVote(BaseModel):
    t: int
    reason: str


class NameCandidate(BaseModel):
    name: str
    weight: float     # 0..1 belief this names the tracked object
    # Only two values are reachable. The auditor is shown the detector's names
    # and forbidden from inventing new ones, so every name it endorses was
    # reported by some detection; all it can add is visual corroboration. A
    # third value "read_from_images" existed and occurred zero times in 73
    # candidates across 105 tracks, because it describes an impossible state.
    basis: Literal["reported_by_detections", "both"]


class PrimaryObject(BaseModel):
    tags: list[WeightedTag]
    # Every name the evidence raises, weighted - NOT a single verdict. The
    # matcher resolves names against the map; the auditor cannot (it has no
    # map, and a name asserted many times can still be a dense-skyline
    # misidentification). Empty list when no name was ever reported.
    name_candidates: list[NameCandidate]
    name_aliases: list[str]       # alternates for the SAME structure only
    description: str              # one sentence, intrinsic properties only
    distinctive_features: list[str]
    extent: Literal["point_like", "small_extended", "large_extended"]


class SecondaryObject(BaseModel):
    tags: list[WeightedTag]
    name: str
    description: str
    ts: list[int]                 # time indices of the detections describing it
    relation: Literal["part_of_primary", "contains_primary", "occluder",
                      "adjacent", "background"]
    worth_own_landmark: bool


class TrackAudit(BaseModel):
    landmark_kind: Literal[
        "fixed_structure", "navigation_aid", "terrain", "vegetation",
        "vessel_or_vehicle", "transient_phenomenon", "mixed_or_unclear"]
    single_object: bool
    valid_segments: list[Segment]  # time spans where the mask stays on the
                                   # primary object; whole lifetime if clean
    verdict: Literal["keep", "keep_partial", "drop"]
    drop_reason: Literal["none", "dynamic_object", "not_a_physical_landmark",
                         "identity_broken", "insufficient_evidence"]
    primary_object: PrimaryObject
    strike_votes: list[StrikeVote]
    secondary_objects: list[SecondaryObject]
    confidence: Literal["low", "medium", "high"]
    unresolved: str               # "" if nothing remains ambiguous


def _resolve_refs(schema, defs=None):
    """Recursively inline $ref definitions (mirrors semantic_landmark_extractor)."""
    if defs is None:
        defs = schema.get("$defs", {}) or schema.get("definitions", {})
    if isinstance(schema, dict):
        if "$ref" in schema:
            name = schema["$ref"].split("/")[-1]
            return _resolve_refs(defs[name], defs) if name in defs else schema
        return {k: _resolve_refs(v, defs) for k, v in schema.items()
                if k not in ("$defs", "definitions")}
    if isinstance(schema, list):
        return [_resolve_refs(item, defs) for item in schema]
    return schema


def _require_all(schema):
    """Mark every object property required and strip titles (in place)."""
    if isinstance(schema, dict):
        schema.pop("title", None)
        if schema.get("type") == "object" and "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
        for value in schema.values():
            if isinstance(value, dict):
                _require_all(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _require_all(item)
    return schema


def get_audit_schema() -> dict:
    return _require_all(_resolve_refs(TrackAudit.model_json_schema()))


# ---------------------------------------------------------------------------
# dossier construction (pure, deterministic)
# ---------------------------------------------------------------------------

def effective_class(support: dict, cfg: AuditConfig) -> str:
    """Recompute the support class under the current classifier so old
    artifacts are audited under present rules (same idea as m4)."""
    return tb.classify_support(
        {"iou": support["iou"], "inter_over_mask": support["inter_over_mask"],
         "inter_over_box": support["inter_over_box"]}, cfg.classifier)


def collect_evidence(track: dict, obs_by_id: dict, cfg: AuditConfig):
    """Split a track's recorded associations into supports and context
    entries, joined with their observations, in time order.

    Returns (supports, context) where each entry is a dict with:
    t (relative keyframe), keyframe, obs, support (raw record), rec.
    """
    birth = track["birth_keyframe"]
    supports, context = [], []
    for rec in track["records"]:
        for s in rec.get("supports", []):
            obs = obs_by_id.get(s["obs_id"])
            if obs is None:
                continue
            entry = {"t": rec["keyframe"] - birth, "keyframe": rec["keyframe"],
                     "obs": obs, "support": s, "rec": rec}
            eff = effective_class(s, cfg)
            if eff in tb.SUPPORT_CLASSES:
                supports.append(entry)
            elif eff == "context":
                context.append(entry)
            # eff == "none": rejected outright, not evidence
    return supports, context


def _obs_tags(obs):
    """All key=value tags of an observation: (primary, {key: value})."""
    extra = dict(tuple(t) for t in obs.additional_tags)
    primary = f"{obs.primary_tag_key}={obs.primary_tag_value}"
    return primary, extra


def run_length_encode(values: list) -> list[tuple]:
    """[(value, count), ...] preserving order."""
    runs = []
    for v in values:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    return [tuple(r) for r in runs]


def tag_vote_table(supports) -> list[dict]:
    """Rows: tag, total, per-confidence counts, as_primary/as_additional.

    Covers ALL identity tags (primary and additional), excluding
    NON_IDENTITY_TAG_KEYS which get their own sections.
    """
    rows = {}

    def bump(tag, conf, role):
        row = rows.setdefault(tag, {"tag": tag, "total": 0, "as_primary": 0,
                                    "as_additional": 0,
                                    **{c: 0 for c in CONFIDENCE_ORDER}})
        row["total"] += 1
        row[role] += 1
        if conf in row:
            row[conf] += 1

    for e in supports:
        primary, extra = _obs_tags(e["obs"])
        conf = e["obs"].confidence
        bump(primary, conf, "as_primary")
        for key, value in extra.items():
            if key not in NON_IDENTITY_TAG_KEYS:
                bump(f"{key}={value}", conf, "as_additional")
    return sorted(rows.values(), key=lambda r: (-r["total"], r["tag"]))


def sample_descriptions(supports, max_samples: int) -> list:
    """Deterministic sample of support entries whose descriptions are quoted.

    Even temporal stride including first and last, then any primary tag not
    yet represented gets its highest-confidence instance appended.
    """
    if len(supports) <= max_samples:
        picked = list(supports)
    else:
        idx = np.round(np.linspace(0, len(supports) - 1, max_samples))
        picked = [supports[int(i)] for i in idx]

    have_tags = {_obs_tags(e["obs"])[0] for e in picked}
    conf_rank = {c: i for i, c in enumerate(CONFIDENCE_ORDER)}
    for tag in dict.fromkeys(_obs_tags(e["obs"])[0] for e in supports):
        if tag in have_tags:
            continue
        best = min((e for e in supports if _obs_tags(e["obs"])[0] == tag),
                   key=lambda e: (conf_rank.get(e["obs"].confidence, 9), e["t"]))
        picked.append(best)
    return sorted(picked, key=lambda e: e["t"])


def select_chip_entries(supports, context, cfg: AuditConfig):
    """Deterministic chip pick: first + last support, then the highest-IoU
    support of each primary-tag run (longest runs first), then the most
    speck-like context boxes. Returns entries sorted by t, tagged with
    entry["is_context"].
    """
    picked = {}

    def add(e):
        picked.setdefault(e["t"], e)

    if supports:
        add(supports[0])
        add(supports[-1])
        runs = []
        start = 0
        tags = [_obs_tags(e["obs"])[0] for e in supports]
        for i in range(1, len(supports) + 1):
            if i == len(supports) or tags[i] != tags[start]:
                runs.append(supports[start:i])
                start = i
        runs.sort(key=lambda r: (-len(r), r[0]["t"]))
        for run in runs:
            if len(picked) >= cfg.max_support_chips:
                break
            add(max(run, key=lambda e: (e["support"]["iou"], -e["t"])))
        for e in sorted(supports, key=lambda e: (-e["support"]["iou"], e["t"])):
            if len(picked) >= cfg.max_support_chips:
                break
            add(e)

    for e in picked.values():
        e["is_context"] = False

    # Context chips are chosen one per DISTINCT primary tag before doubling
    # up on any tag: the model is asked to judge each context group, and a
    # group it never saw an image of gets judged from text alone (an f0180
    # "egg-shaped digester tanks" box that is really a low seawall was
    # promoted to its own landmark that way). Within a tag, prefer the
    # largest mask fill - the most interpretable view of that group.
    ctx_by_tag = {}
    for e in context:
        tag = _obs_tags(e["obs"])[0]
        best = ctx_by_tag.get(tag)
        if best is None or (e["support"]["inter_over_box"]
                            > best["support"]["inter_over_box"]):
            ctx_by_tag[tag] = e
    ctx_order = sorted(ctx_by_tag.values(),
                       key=lambda e: (-len([c for c in context
                                            if _obs_tags(c["obs"])[0]
                                            == _obs_tags(e["obs"])[0]]),
                                      e["t"]))
    for e in ctx_order[:cfg.max_context_chips]:
        e = dict(e, is_context=True)
        picked.setdefault(("ctx", e["t"]), e)

    return sorted(picked.values(), key=lambda e: e["t"])


def _close_reason_text(track: dict) -> str:
    birth = track["birth_keyframe"]
    reasons = {
        "starved": "no detections were associated for many consecutive "
                   "keyframes",
        "drift_alarm": "a geometric drift alarm fired (detections repeatedly "
                       "landed near, but not on, the tracked mask)",
        "mask_dead": "the tracked mask was lost",
    }
    if track["status"] == "alive":
        return "the track was still active when the recording ended"
    reason = reasons.get(track["close_reason"], track["close_reason"])
    return (f"the track was closed at t{track['end_keyframe'] - birth} "
            f"because {reason}")


def build_dossier(track: dict, obs_by_id: dict, cfg: AuditConfig) -> dict:
    """All dossier content, structured. Deterministic, artifact-only."""
    supports, context = collect_evidence(track, obs_by_id, cfg)
    birth = track["birth_keyframe"]
    lifetime = track["end_keyframe"] - birth + 1

    # Names are counted with their detector confidence. Every identity *tag*
    # reaches the auditor split high/medium/low via tag_vote_table, but `name`
    # is in NON_IDENTITY_TAG_KEYS and so was excluded from it - the name, the
    # single most consequential piece of evidence, used to arrive as a bare
    # count. The extraction prompt explicitly delegates naming doubt to
    # `confidence` ("express your certainty through the confidence field
    # rather than by staying silent"), so dropping it discarded exactly the
    # signal the extractor was told to send.
    name_votes = {}
    name_confidence = {}
    for e in supports:
        _, extra = _obs_tags(e["obs"])
        name = extra.get("name", "")
        if name:
            name_votes[name] = name_votes.get(name, 0) + 1
            by_conf = name_confidence.setdefault(
                name, {c: 0 for c in CONFIDENCE_ORDER})
            if e["obs"].confidence in by_conf:
                by_conf[e["obs"].confidence] += 1

    reanchors = sum(1 for r in track["records"]
                    if r["action"] == "reanchor_clean")
    supported_ts = {e["t"] for e in supports}
    gap_keyframes = sum(
        1 for r in track["records"]
        if r["keyframe"] - birth < lifetime
        and (r["keyframe"] - birth) not in supported_ts)

    return {
        "track_id": track["track_id"],
        "birth_keyframe": birth,
        "lifetime": lifetime,
        "n_supports": len(supports),
        "n_gap_keyframes": gap_keyframes,
        "n_reanchors": reanchors,
        "drift_alarm": track["close_reason"] == "drift_alarm",
        "close_text": _close_reason_text(track),
        "tag_table": tag_vote_table(supports),
        "name_votes": sorted(name_votes.items(), key=lambda kv: -kv[1]),
        "name_confidence": name_confidence,
        "primary_tag_rle": run_length_encode(
            [_obs_tags(e["obs"])[0] for e in supports]),
        "distance_rle": run_length_encode(
            [_obs_tags(e["obs"])[1].get("distance_estimate", "unreported")
             for e in supports]),
        "description_samples": sample_descriptions(
            supports, cfg.max_description_samples),
        "context": context,
        "chip_entries": select_chip_entries(supports, context, cfg),
        "supports": supports,
    }


def build_evidence(track: dict, dossier: dict, pano_w: int) -> dict:
    """Procedural evidence weight for one track - how much this track's
    semantics should be trusted, and how much geometry it offers.

    Computed from the artifact only; the model never produces or influences
    these numbers. Downstream (matching, SLAM) must be able to prefer a
    68-support track over a 4-support one, so every count that distinguishes
    them travels with the record.
    """
    supports = dossier["supports"]
    ious = sorted(s["support"]["iou"] for s in supports)
    box_px = sorted(s["support"]["box_window"][2] - s["support"]["box_window"][0]
                    for s in supports)

    def median(values):
        return values[len(values) // 2] if values else 0.0

    name_votes = dict(dossier["name_votes"])
    total_named = sum(name_votes.values())
    ranked = sorted(name_votes.items(), key=lambda kv: -kv[1])
    top_n = ranked[0][1] if ranked else 0
    runner_n = ranked[1][1] if len(ranked) > 1 else 0

    tag_rows = dossier["tag_table"]
    tag_total = sum(r["total"] for r in tag_rows)

    # Camera-frame azimuth swept by the mask centre. NOT a world bearing
    # (ego rotation is folded in) - a coarse proxy for viewpoint diversity
    # until the localization stage supplies heading.
    centres = []
    for rec in track["records"]:
        mb = rec.get("mask_bbox_window")
        if mb is None:
            continue
        ox = rec["window_origin"][0]
        centres.append(((ox + (mb[0] + mb[2]) / 2) % pano_w) / pano_w * 360.0)
    span = 0.0
    if len(centres) > 1:
        unwrapped = [centres[0]]
        for c in centres[1:]:
            d = (c - unwrapped[-1] + 180.0) % 360.0 - 180.0
            unwrapped.append(unwrapped[-1] + d)
        span = max(unwrapped) - min(unwrapped)

    return {
        "n_supports": len(supports),
        "n_supported_keyframes": track["n_supported_keyframes"],
        "lifetime_keyframes": dossier["lifetime"],
        "support_density": (len(supports) / dossier["lifetime"]
                            if dossier["lifetime"] else 0.0),
        "n_context_only": len(dossier["context"]),
        "n_reanchors": dossier["n_reanchors"],
        "drift_alarm": dossier["drift_alarm"],
        "close_reason": track["close_reason"],
        "median_iou": median(ious),
        "median_box_px": median(box_px),
        "max_box_px": box_px[-1] if box_px else 0,
        "camera_azimuth_span_deg": span,
        "tag_votes": {r["tag"]: r["total"] for r in tag_rows},
        "n_distinct_tags": len(tag_rows),
        "tag_top_share": (tag_rows[0]["total"] / tag_total
                          if tag_total else 0.0),
        "confidence_counts": {
            c: sum(r[c] for r in tag_rows) for c in CONFIDENCE_ORDER},
        "name_votes": name_votes,
        "n_named_supports": total_named,
        "n_distinct_names": len(name_votes),
        "name_top_share": top_n / total_named if total_named else 0.0,
        "name_margin": top_n / runner_n if runner_n else float(top_n),
        # True when the name evidence is split enough that no single name
        # should be treated as this track's identity downstream.
        "name_contested": bool(
            total_named and (top_n / total_named < 0.5 or top_n < 2 * runner_n)),
    }


# ---------------------------------------------------------------------------
# prompt rendering
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """<role>
You audit object tracks from a robot's landmark pipeline and produce one
canonical landmark record per track.
</role>

<provenance>
A camera on a moving robot produced landmark detections at each
keyframe: a bounding box, an OpenStreetMap-style tag (key=value), a detector
confidence (high/medium/low), and a one-sentence description. Separately, a
mask tracker propagated the object's mask between keyframes, and detections
were associated to the track PURELY GEOMETRICALLY (box-to-mask overlap).
Semantic labels played no role in the association. Your job is to audit the
semantics now.

Downstream, your output is matched by tag + description against map databases
(e.g., OpenStreetMap), and the track's bearing measurements are
then used for localization. A wrong canonical tag poisons map matching; an
overly timid answer discards a usable landmark.
</provenance>

<image_convention>
GREEN box = the detector's bounding box for that keyframe's detection.
RED box = the tracked mask's bounding box at the same keyframe.
Crops are taken around the union of the two boxes with margin. The object may
be small in the frame; judge only what is visible.
</image_convention>

<calibration>
- Detectors waffle between related tags for the same physical object
  (e.g. water_tower / lighthouse / tower, or several building=* values).
  This is common and is NOT by itself evidence of multiple objects.
- Track identity errors are visible in the IMAGES: the red mask box sitting
  on clearly different physical objects at different times.
- If a description disagrees with its image, trust the image.
- Never invent a name the evidence does not support. Only keep a name if it
  was reported by detections and is consistent with the images.
- name_aliases must be alternate names for the SAME physical structure. If a
  reported name belongs to a DIFFERENT nearby structure whose box grazed the
  track, that is contamination: strike it or record it as a secondary object,
  never as an alias.
- Time indices t0..tN are keyframes since the track started.
</calibration>

<output_semantics>
- landmark_kind: what the tracked object fundamentally is. Movable things
  (vessels, vehicles) and transient phenomena (wakes, glare, clouds, shadows)
  are not landmarks: verdict=drop with drop_reason=dynamic_object or
  not_a_physical_landmark.
- single_object / valid_segments: if the mask visibly switches objects,
  report the time spans that belong to the PRIMARY object (the one supported
  by most evidence) as valid_segments and use verdict=keep_partial. If no
  usable span remains, verdict=drop with drop_reason=identity_broken.
- primary_object.tags: weighted distribution over OSM-style tags for the
  primary object. Include every tag the map might plausibly index this object
  under; weights express your relative belief and need not sum to 1.
- primary_object.name_candidates: list EVERY name the evidence raises for the
  primary object, each weighted. Do NOT collapse to one name and do NOT drop
  the names you disbelieve - a later stage resolves names against a map you
  cannot see, so a name you rank low may still win, and one you rank high may
  be geometrically impossible. Weight by how well each name fits what you can
  SEE (basis="read_from_images" or "both" outranks a name that is merely
  frequent in the detection text). Never rank a name above one with more
  supporting detections unless the images justify it - say why in unresolved
  if you do. Dense skylines produce confident wrong names at every range, so
  a single track collecting many different building names is expected; report
  the spread rather than hiding it behind a winner.
  CALIBRATE THE WEIGHT AGAINST THE SUPPORT. The dossier gives each name's
  detection count and the detector's own confidence in that name, against the
  track's total detection count. A name asserted by one detection out of
  dozens is worth listing, but it is not worth a high weight: reserve weights
  above 0.7 for names carried by a substantial share of the detections or
  corroborated by what you can see in the images, and keep a name reported
  once, or reported at `medium` confidence, below 0.3. A downstream matcher
  treats a high-weighted name as near-decisive and will place this object at
  the map row of that name however far away it is, so an over-weighted name
  is worse than no name at all.
- primary_object.description: one sentence, intrinsic properties only (shape,
  color, material, count), reusable to re-identify the object from a
  different viewpoint. Never mention image position, direction, distance,
  lighting, or weather.
- primary_object.extent: point_like (a tower, a chimney), small_extended
  (a building, a pier), large_extended (an island, a bridge span, a skyline).
- strike_votes: detections whose semantics describe a DIFFERENT object than
  the primary (contaminants). Their votes are removed and re-routed; do not
  strike mere tag variance on the same object.
- secondary_objects: coherent groups of detections that describe a different
  physical object (a fort ON the tracked island, an occluder in front of it,
  a background cluster inside a context box). Set worth_own_landmark=true
  ONLY for a discrete physical object a map could plausibly index (a fort, a
  tank cluster, a specific building) - never generic scenery such as a city
  skyline, tree line, generic shoreline, or horizon; record those with
  worth_own_landmark=false. All tags everywhere must be OSM-style key=value
  from the same vocabulary the detections use. Set worth_own_landmark=true
  only for an object you can SEE in one of the provided images; if a group is
  described only in text, keep it with worth_own_landmark=false and say so in
  its description - a detection's tag and description can be wrong, and
  without an image you cannot check it.
- unresolved: what you could not settle from the evidence, "" if nothing.
</output_semantics>
"""

USER_PROMPT_HEADER = "TRACK EVIDENCE\n"

QUESTIONS_TEXT = """Audit this track following the system instructions:
identify what the tracked object is, whether the track stays on one object,
which detections (if any) describe other objects, and produce the canonical
landmark record. Answer with the JSON schema.
"""


def _format_tag_table(rows) -> str:
    lines = ["tag votes from all associated detections (every tag, by "
             "detector confidence)",
             f"  {'tag':<40} total  high  med  low  as_primary  as_addl"]
    for r in rows:
        lines.append(
            f"  {r['tag']:<40} {r['total']:>5} {r['high']:>5} "
            f"{r['medium']:>4} {r['low']:>4} {r['as_primary']:>11} "
            f"{r['as_additional']:>8}")
    return "\n".join(lines)


def _format_rle(runs, quote=False) -> str:
    def fmt(v):
        return f"'{v}'" if quote else str(v)
    return " | ".join(f"{fmt(v)} x{n}" for v, n in runs)


def render_dossier_text(dossier: dict) -> str:
    """The user-prompt text block. Facts only; no identifiers beyond
    relative time; no interpretation."""
    parts = [USER_PROMPT_HEADER]
    parts.append(
        f"lifetime: {dossier['lifetime']} keyframes "
        f"(t0..t{dossier['lifetime'] - 1}); {dossier['close_text']}")
    parts.append(
        f"associated detections: {dossier['n_supports']}; keyframes in the "
        f"lifetime with no associated detection: {dossier['n_gap_keyframes']}")
    parts.append(
        f"geometry events: {dossier['n_reanchors']} re-anchors; drift alarm: "
        f"{'yes' if dossier['drift_alarm'] else 'no'}")

    parts.append("")
    parts.append(_format_tag_table(dossier["tag_table"]))

    parts.append("")
    if dossier["name_votes"]:
        conf = dossier.get("name_confidence") or {}

        def _one(name, n):
            by = conf.get(name)
            if not by:
                return f"'{name}' x{n}"
            split = ", ".join(f"{by[c]} {c}" for c in CONFIDENCE_ORDER if by[c])
            return f"'{name}' x{n} ({split})" if split else f"'{name}' x{n}"

        parts.append("names reported, with the detector's confidence in each "
                     "name: " + ", ".join(
                         _one(name, n) for name, n in dossier["name_votes"]))
    else:
        parts.append("names reported: (none)")

    parts.append("")
    parts.append("primary tag in time order (run-length encoded):")
    parts.append("  " + _format_rle(dossier["primary_tag_rle"]))
    parts.append("")
    parts.append("detector distance_estimate in time order (run-length "
                 "encoded):")
    parts.append("  " + _format_rle(dossier["distance_rle"]))

    parts.append("")
    parts.append("detection descriptions (verbatim, time order, sampled):")
    for e in dossier["description_samples"]:
        primary, _ = _obs_tags(e["obs"])
        parts.append(f"  [t{e['t']}] {primary} ({e['obs'].confidence}) "
                     f"\"{e['obs'].description}\"")

    if dossier["context"]:
        parts.append("")
        parts.append(
            "context detections (their box CONTAINS the tracked mask but the "
            "mask fills too little of it to count as the same object; "
            "excluded from all counts above; listed as merge/occlusion "
            "evidence):")
        for e in dossier["context"]:
            primary, _ = _obs_tags(e["obs"])
            frac = e["support"]["inter_over_box"]
            parts.append(
                f"  [t{e['t']}] {primary} ({e['obs'].confidence}) mask fills "
                f"{frac:.0%} of the box \"{e['obs'].description}\"")

    n_chips = len(dossier["chip_entries"])
    parts.append("")
    parts.append(f"{n_chips} images follow in time order; captions give the "
                 "time index and the detection they show.")
    parts.append("")
    parts.append(QUESTIONS_TEXT)
    return "\n".join(parts)


def chip_caption(entry, index: int) -> str:
    primary, _ = _obs_tags(entry["obs"])
    s = entry["support"]
    if entry.get("is_context"):
        return (f"[IMAGE {index}] t{entry['t']} CONTEXT detection: {primary} "
                f"({entry['obs'].confidence}); the tracked mask (red) fills "
                f"{s['inter_over_box']:.0%} of this box")
    return (f"[IMAGE {index}] t{entry['t']} detection: {primary} "
            f"({entry['obs'].confidence}), iou={s['iou']:.2f}")


# ---------------------------------------------------------------------------
# chip rendering
# ---------------------------------------------------------------------------

def render_chip(pano: np.ndarray, det_box, mask_box, out_path,
                chip_height: int):
    """Crop around the union of detection box and mask bbox (wrap-safe);
    detection box green, mask bbox red."""
    pano_w = pano.shape[1]
    x0, y0, x1, y1 = det_box
    ux0, uy0, ux1, uy1 = x0, y0, x1, y1
    mask_rel = None
    if mask_box is not None:
        dx = pg.signed_x_offset(mask_box[0], x0, pano_w)
        mxa = x0 + dx
        mask_rel = (mxa, mask_box[1], mxa + (mask_box[2] - mask_box[0]),
                    mask_box[3])
        ux0, uy0 = min(ux0, mask_rel[0]), min(uy0, mask_rel[1])
        ux1, uy1 = max(ux1, mask_rel[2]), max(uy1, mask_rel[3])
    w, h = ux1 - ux0, uy1 - uy0
    mx, my = max(30, 0.25 * w), max(30, 0.25 * h)
    cw, ch = int(w + 2 * mx), int(h + 2 * my)
    crop, cy0 = pg.extract_window(pano, ux0 - mx, uy0 - my, cw, ch)
    img = Image.fromarray(crop)
    draw = ImageDraw.Draw(img)
    line_w = max(2, int(ch / 130))
    cx0 = pg.signed_x_offset(ux0, ux0 - mx, pano_w)  # = mx, wrap-safe
    draw.rectangle([cx0 + (x0 - ux0), y0 - cy0, cx0 + (x1 - ux0), y1 - cy0],
                   outline=DETECTION_COLOR, width=line_w)
    if mask_rel is not None:
        draw.rectangle([cx0 + (mask_rel[0] - ux0), mask_rel[1] - cy0,
                        cx0 + (mask_rel[2] - ux0), mask_rel[3] - cy0],
                       outline=MASK_COLOR, width=line_w)
    scale = chip_height / img.height
    img = img.resize((max(80, int(img.width * scale)), chip_height),
                     Image.BILINEAR)
    img.save(out_path, quality=88)


def chip_boxes_for_entry(entry, obs, pano_w, pano_h):
    """(det_box, mask_box) in pano coordinates for a chip entry."""
    det_box = pg.pano_bbox_for_observation(obs.boxes, pano_w, pano_h)
    rec = entry["rec"]
    mask_box = None
    mb = rec.get("mask_bbox_window")
    if mb is not None:
        ox, oy = rec["window_origin"]
        mask_box = (ox + mb[0], oy + mb[1], ox + mb[2], oy + mb[3])
    return det_box, mask_box


# ---------------------------------------------------------------------------
# request assembly
# ---------------------------------------------------------------------------

def build_request(track_key: str, dossier_text: str,
                  chips: list[tuple[str, str]], cfg: AuditConfig) -> dict:
    """Vertex request record. chips = [(caption, base64_jpeg), ...] in the
    order referenced by the dossier text; captions are interleaved as text
    parts before each image."""
    parts = [{"text": dossier_text}]
    for caption, b64 in chips:
        parts.append({"text": caption})
        parts.append({"inline_data": {"mime_type": "image/jpeg",
                                      "data": b64}})
    return {
        "key": track_key,
        "request": {
            "contents": [{"parts": parts, "role": "user"}],
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": get_audit_schema(),
                "thinkingConfig": {"thinkingLevel": cfg.thinking_level},
            },
        },
    }


def upgrade_legacy_audit(data: dict) -> dict:
    """Migrate pre-name_candidates audit payloads in place-ish.

    Results produced before names became a weighted candidate list carry a
    single `name`. Those runs are still worth consuming (valid_segments and
    strike/secondary evidence are unaffected), so the old single name is
    lifted into a lone candidate rather than discarded. The weight is left
    at 1.0 because the old schema recorded no alternative to weigh it
    against - `name_contested` from build_evidence is the honest signal for
    those records, not this weight.
    """
    primary = data.get("primary_object")
    if not isinstance(primary, dict) or "name_candidates" in primary:
        return data
    name = primary.pop("name", "")
    primary["name_candidates"] = (
        [{"name": name, "weight": 1.0, "basis": "reported_by_detections"}]
        if name else [])
    return data


def parse_result_line(record: dict) -> tuple[str, dict | None, str | None]:
    """(track_key, TrackAudit dict or None, error or None) from one result
    JSONL line written by vertex_batch_manager."""
    key = record.get("key", "?")
    if record.get("error"):
        return key, None, record["error"]
    try:
        text = record["response"]["candidates"][0]["content"]["parts"][0]["text"]
        audit = TrackAudit.model_validate(
            upgrade_legacy_audit(json.loads(text)))
        return key, audit.model_dump(), None
    except Exception as e:  # noqa: BLE001 - surfaced to caller per line
        return key, None, f"{type(e).__name__}: {e}"
