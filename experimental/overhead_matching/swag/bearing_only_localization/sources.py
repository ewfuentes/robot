"""Tracklet payload from the object-track run that produced a localization run.

§7.1 keeps raw imagery out of the run log on purpose — "tracklets carry
frame/crop *references*", and a bounded cache materializes crops on demand —
which is what keeps run directories portable and small. The cost is that the
run log alone cannot answer the tracker half of "did the tracker, the matcher,
or the filter get this wrong?": it has bearings and concentrations, but not the
thing that was looked at.

This module reads that back from the object-track run directory, which holds
what the localization export dropped:

  merged/landmarks.json    per-tracklet name/tag votes, supporting keyframe
                           span, constituent source track ids, and the handoff
                           proposals that record where the merger was unsure
  merged/measurements.json tracklet_id -> source_track_id, which is what names
                           the crop files
  thumbs/<stem>_T<id>.jpg  one crop per source track — the tracker's evidence
  matching/request_meta    chunk key -> the tracklet ids that chunk scored
  matching/results.jsonl   the matcher's own verdicts, including the
                           no-match confidence it asserted and the uniqueness
                           it claimed

The matcher's stated verdict is worth having next to the derived log-LR,
because they answer different questions. A table entry says what the filter
was told; `no_match_rate` says what the matcher believed. A tracklet whose
matcher declared no-match on every chunk it appeared in is a different failure
from one whose matcher was confident and wrong, and only the second is worth
arguing with.

Everything degrades: a missing sources directory, a missing crop, an
unparseable prompt each drop one field rather than failing the build, because
the inspector is still useful with only some of this and useless if a stale
path takes the whole viewer down.
"""

import base64
import dataclasses
import json
import re
from pathlib import Path

# Crops are already small (~8 KB each). The budget exists so that a run with an
# unexpectedly large thumbnail set cannot silently balloon a self-contained
# page; what does not fit is reported rather than dropped quietly.
DEFAULT_THUMBNAIL_BUDGET_BYTES = 3 * 1024 * 1024
_ENTRY_RE = re.compile(r"^\s*(\d+)\.\s", re.MULTILINE)


@dataclasses.dataclass
class TrackletSource:
    """What the tracker and the matcher saw, for one tracklet."""
    tracklet_id: str
    track_ids: tuple = ()
    n_supports: int = 0
    n_supported_keyframes: int = 0
    keyframe_span: tuple | None = None
    name_votes: dict = dataclasses.field(default_factory=dict)
    tag_votes: dict = dataclasses.field(default_factory=dict)
    name_contested: bool = False
    # Merger uncertainty: tracklets this one might be a continuation of. A
    # populated list is a live hypothesis that this "tracklet" is two objects.
    handoff_proposals: tuple = ()
    # The observed payload as the matcher was shown it.
    description: str | None = None
    features: tuple = ()
    unresolved: str | None = None
    # The matcher's own verdict, aggregated over the chunks that scored it.
    n_matcher_chunks: int = 0
    no_match_rate: float | None = None
    median_uniqueness: float | None = None
    thumbnail_data_uri: str | None = None

    @property
    def best_name(self) -> str | None:
        if not self.name_votes:
            return None
        return max(self.name_votes, key=self.name_votes.get)

    @property
    def best_tags(self) -> list:
        return sorted(self.tag_votes, key=self.tag_votes.get, reverse=True)[:4]


@dataclasses.dataclass
class SourceBundle:
    tracklets: dict
    sources_dir: Path | None
    # Everything that could not be loaded, so a thin inspector panel is
    # explained rather than mysterious.
    notes: tuple = ()

    def get(self, tracklet_id: str) -> TrackletSource | None:
        return self.tracklets.get(tracklet_id)


def _read_json(path: Path):
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None


def _parse_set1_entries(prompt: str) -> dict:
    """Split a chunk prompt's "Set 1" block into per-index payload text.

    The prompt is generated, so its shape is stable, but it is still prose
    being reverse-engineered: every field here is optional and a parse miss
    costs one panel rather than the build.
    """
    start = prompt.find("Set 1")
    if start < 0:
        return {}
    end = prompt.find("Set 2", start)
    block = prompt[start:end if end > 0 else len(prompt)]
    entries = {}
    matches = list(_ENTRY_RE.finditer(block))
    for i, match in enumerate(matches):
        stop = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        entries[int(match.group(1))] = block[match.end():stop]
    return entries


def _field(text: str, label: str) -> str | None:
    match = re.search(rf"^\s*{label}:\s*(.+?)$", text, re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip().strip('"') or None


def _load_matcher_verdicts(matching_dir: Path, notes: list) -> dict:
    """Per-tracklet aggregate of what the matcher itself asserted."""
    meta = _read_json(matching_dir / "request_meta.json")
    results_path = matching_dir / "results.jsonl"
    if not meta or not results_path.exists():
        notes.append("matching/: no request_meta.json or results.jsonl, so "
                     "the matcher's own verdicts are unavailable")
        return {}

    per_tracklet: dict[str, list] = {}
    payloads: dict[str, dict] = {}
    for line in results_path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        entry = meta.get(row.get("key"))
        if not entry:
            continue
        keys = entry.get("batch_keys") or []

        prompt = _extract_prompt(row.get("request"))
        if prompt and entry.get("chunk_index") == 0:
            for index, text in _parse_set1_entries(prompt).items():
                if index < len(keys):
                    payloads.setdefault(keys[index], {
                        "description": _field(text, "description"),
                        "features": tuple(
                            f.strip() for f in
                            (_field(text, "features") or "").split(";")
                            if f.strip()),
                        "unresolved": _field(text, "unresolved"),
                    })

        for match in _extract_matches(row.get("response")):
            index = match.get("set_1_id")
            if not isinstance(index, int) or index >= len(keys):
                continue
            per_tracklet.setdefault(keys[index], []).append(match)

    verdicts = {}
    for tracklet_id, matches in per_tracklet.items():
        no_match = [m.get("no_match_confidence") for m in matches
                    if isinstance(m.get("no_match_confidence"), (int, float))]
        unique = sorted(m["uniqueness_score"] for m in matches
                        if isinstance(m.get("uniqueness_score"), (int, float)))
        verdicts[tracklet_id] = {
            "n_matcher_chunks": len(matches),
            "no_match_rate": (sum(no_match) / len(no_match)
                              if no_match else None),
            "median_uniqueness": (float(unique[len(unique) // 2])
                                  if unique else None),
            **payloads.get(tracklet_id, {}),
        }
    for tracklet_id, payload in payloads.items():
        verdicts.setdefault(tracklet_id, {}).update(payload)
    return verdicts


def _extract_prompt(request) -> str | None:
    try:
        return request["contents"][0]["parts"][0]["text"]
    except (TypeError, KeyError, IndexError):
        return None


def _extract_matches(response) -> list:
    """The `matches` array out of a Vertex response envelope."""
    try:
        text = response["candidates"][0]["content"]["parts"][0]["text"]
    except (TypeError, KeyError, IndexError):
        return []
    try:
        return json.loads(text).get("matches", []) or []
    except ValueError:
        # Truncated or fenced JSON: recover the array if it is intact.
        start, end = text.find("["), text.rfind("]")
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                return []
        return []


def _thumbnail_index(sources_dir: Path) -> dict:
    """source track id -> crop path, from whatever stem the run used."""
    index = {}
    thumbs = sources_dir / "thumbs"
    if not thumbs.is_dir():
        return index
    for path in thumbs.glob("*_T*.jp*g"):
        suffix = path.stem.rsplit("_T", 1)[-1]
        if suffix.isdigit():
            index.setdefault(int(suffix), path)
    return index


def load(sources_dir: Path | None, tracklet_ids,
         embed_thumbnails: bool = True,
         thumbnail_budget_bytes: int = DEFAULT_THUMBNAIL_BUDGET_BYTES
         ) -> SourceBundle:
    """Load payload for `tracklet_ids`. Missing pieces become notes."""
    wanted = set(tracklet_ids)
    if sources_dir is None:
        return SourceBundle({}, None,
                            ("no sources directory: the tracklet inspector "
                             "shows bearings and LLRs but no crops or "
                             "matcher payload",))
    sources_dir = Path(sources_dir)
    notes: list[str] = []
    if not sources_dir.is_dir():
        return SourceBundle({}, sources_dir,
                            (f"sources directory {sources_dir} does not "
                             f"exist",))

    out = {tracklet_id: TrackletSource(tracklet_id=tracklet_id)
           for tracklet_id in sorted(wanted)}

    merged = _read_json(sources_dir / "merged" / "landmarks.json") or []
    if not merged:
        notes.append("merged/landmarks.json missing or unreadable: no "
                     "name/tag votes or handoff proposals")
    for row in merged:
        tracklet_id = row.get("landmark_id")
        if tracklet_id not in out:
            continue
        span = row.get("keyframe_span")
        out[tracklet_id] = dataclasses.replace(
            out[tracklet_id],
            track_ids=tuple(row.get("track_ids") or ()),
            n_supports=row.get("n_supports", 0),
            n_supported_keyframes=row.get("n_supported_keyframes", 0),
            keyframe_span=tuple(span) if span else None,
            name_votes=row.get("name_votes") or {},
            tag_votes=row.get("tag_votes") or {},
            name_contested=bool(row.get("name_contested")),
            handoff_proposals=tuple(
                (p.get("with"), p.get("gap_keyframes"), p.get("status"))
                for p in (row.get("handoff_proposals") or [])))

    # Tracklet ids that never made it into merged/landmarks.json — usually
    # aliases created after merging (the export records these).
    unresolved = [tid for tid, source in out.items() if not source.track_ids]
    if unresolved and merged:
        measurements = _read_json(
            sources_dir / "merged" / "measurements.json") or []
        fallback: dict[str, set] = {}
        for row in measurements:
            tracklet_id = row.get("tracklet_id")
            if tracklet_id in out and row.get("source_track_id") is not None:
                fallback.setdefault(tracklet_id, set()).add(
                    int(row["source_track_id"]))
        for tracklet_id, track_ids in fallback.items():
            if not out[tracklet_id].track_ids:
                out[tracklet_id] = dataclasses.replace(
                    out[tracklet_id], track_ids=tuple(sorted(track_ids)))
        still_missing = [tid for tid in unresolved if not out[tid].track_ids]
        if still_missing:
            notes.append(
                f"{len(still_missing)} tracklet(s) have no source track in "
                f"this run ({', '.join(sorted(still_missing)[:4])}"
                + (", ..." if len(still_missing) > 4 else "")
                + "): likely aliased after merging, so they have no crop")

    verdicts = _load_matcher_verdicts(sources_dir / "matching", notes)
    for tracklet_id, verdict in verdicts.items():
        if tracklet_id in out:
            out[tracklet_id] = dataclasses.replace(
                out[tracklet_id],
                **{k: v for k, v in verdict.items() if v is not None})

    if embed_thumbnails:
        index = _thumbnail_index(sources_dir)
        if not index:
            notes.append("thumbs/ holds no crops: the tracker panel will have "
                         "no imagery")
        spent, skipped = 0, []
        for tracklet_id in sorted(out):
            source = out[tracklet_id]
            path = next((index[t] for t in source.track_ids if t in index),
                        None)
            if path is None:
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if spent + len(raw) > thumbnail_budget_bytes:
                skipped.append(tracklet_id)
                continue
            spent += len(raw)
            suffix = "jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "png"
            out[tracklet_id] = dataclasses.replace(
                source, thumbnail_data_uri=(
                    f"data:image/{suffix};base64,"
                    + base64.b64encode(raw).decode("ascii")))
        if skipped:
            notes.append(
                f"thumbnail budget of {thumbnail_budget_bytes // 1024} KB "
                f"reached: {len(skipped)} crop(s) not embedded "
                f"({', '.join(skipped[:4])}"
                + (", ..." if len(skipped) > 4 else "") + ")")

    return SourceBundle(out, sources_dir, tuple(notes))


def guess_sources_dir(export_dir: Path | None) -> Path | None:
    """An export lives inside its object-track run, so its parent is the
    sources directory. Returns None rather than guessing wildly."""
    if export_dir is None:
        return None
    export_dir = Path(export_dir)
    parent = export_dir.parent
    if (parent / "merged").is_dir() or (parent / "thumbs").is_dir():
        return parent
    return None
