"""Report where the data root disagrees with the paper roster.

For every sequence in `table_common.DATASET_GROUPS`, every pinned lane is
opened from its manifest and checked against the roster: it exists and is
complete, its upstream refs name the roster's sibling versions, and its LLM
settings agree with `LLM_STANDARD` (or, while a standard is undecided, with
each other). Catalog clip and reported prior area are shown beside the
group's region policy. Output is Markdown; nothing is modified.
"""

import argparse
import math
import sys
from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.paper.table_common import (
    DATASET_GROUPS,
    DEFAULT_FARFIELD_ROOT,
    LLM_STANDARD,
    SEQUENCE_ARTIFACTS,
    SEQUENCE_LANES,
    DatasetGroup,
    emit_table,
    read_json_object,
)

_LLM_KEYS = {
    "frame_landmarks": ("extraction.model", "extraction.prompt_type"),
    "semantic_audits": ("audit.model",),
    "landmark_matches": ("matching.model",),
}


def _stage_config(manifest: artifact.ArtifactManifest) -> dict:
    recipe = manifest.recipe if isinstance(manifest.recipe, dict) else {}
    return recipe.get("stage_config") or {}


def _bbox_area_km2(wsen) -> float:
    west, south, east, north = wsen
    mid = math.radians((south + north) / 2.0)
    return (east - west) * 111.32 * math.cos(mid) * (north - south) * 111.32


def _catalog_clip(manifest: artifact.ArtifactManifest) -> str:
    config = manifest.config
    region = config.get("region_bbox_wsen")
    if region and config.get("region_source") != "clip_bbox_wsen":
        return (f"region {config.get('region_source')} "
                f"{_bbox_area_km2(region):,.0f} km²")
    plan = config.get("clip_plan")
    if isinstance(plan, dict) and plan.get("bbox_wsen"):
        policy = plan.get("policy") or {}
        return (f"clip plan {policy.get('resolved_area_km2', 0):,.0f} km²"
                f" ({policy.get('minimum_area_km2', '?')} min)")
    if config.get("clip_km"):
        return f"legacy clip_km={config['clip_km']}"
    if config.get("bbox_wsen"):
        return f"fetch bbox {_bbox_area_km2(config['bbox_wsen']):,.0f} km²"
    return "NO CLIP"


def _region_area_km2(manifest: artifact.ArtifactManifest,
                     group: DatasetGroup) -> float | None:
    """The region the group's policy declares, as the catalog records it."""
    config = manifest.config
    plan = config.get("clip_plan")
    if group.region_policy == "area625" and isinstance(plan, dict):
        return float((plan.get("policy") or {}).get("resolved_area_km2", 0)) or None
    if group.region_policy == "fetch_bbox":
        bbox = config.get("region_bbox_wsen") or config.get("bbox_wsen")
        return _bbox_area_km2(bbox) if bbox else None
    return None


def _prior_area_km2(root: Path, group: DatasetGroup, sequence: str) -> str:
    if group.run_spec is None:
        return "no runs"
    experiment, pattern = group.run_spec
    areas = set()
    for run_dir in (root / "runs" / experiment).glob(pattern):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = read_json_object(manifest_path)
        if manifest.get("dataset") != sequence:
            continue
        init = (manifest["config"]["localization_run_contract"]
                ["filter_config"]["init"])
        areas.add(round((init["east_max_m"] - init["east_min_m"])
                        * (init["north_max_m"] - init["north_min_m"]) / 1e6))
    if not areas:
        return "no runs"
    return " / ".join(f"{area:,}" for area in sorted(areas))


def check_sequence(root: Path, group: DatasetGroup, sequence: str,
                   llm_seen: dict[str, dict[str, set[str]]]) -> list[list[str]]:
    rows = []
    region_km2: list[float | None] = [None]
    pins = {"catalogs": group.catalog_version, **SEQUENCE_ARTIFACTS[sequence]}
    dataset_dir = root / "datasets" / sequence
    rows.append(["dataset", "", "OK" if (dataset_dir / "pipeline_metadata.json").is_file()
                 else "MISSING", str(dataset_dir) if not dataset_dir.is_dir() else ""])
    for kind in ("catalogs", *SEQUENCE_LANES):
        version = pins.get(kind)
        if version is None:
            rows.append([kind, "—", "UNPINNED", ""])
            continue
        path = root / "artifacts" / kind / sequence / version
        try:
            manifest = artifact.load_manifest(path)
        except artifact.ArtifactValidationError as exc:
            status = "MISSING" if not path.is_dir() else "INVALID"
            rows.append([kind, version, status, str(exc).splitlines()[0][:120]])
            continue
        notes = []
        for upstream in manifest.upstreams:
            if upstream.kind == kind:  # own lineage (e.g. trim -> full catalog)
                notes.append(f"from {upstream.version}")
                continue
            expected = pins.get(upstream.kind)
            if expected is not None and upstream.version != expected:
                notes.append(f"upstream {upstream.kind}={upstream.version}"
                             f" ≠ roster {expected}")
        stage_config = _stage_config(manifest)
        for key in _LLM_KEYS.get(kind, ()):
            value = stage_config.get(key)
            if value is None:
                notes.append(f"{key} not recorded")
                continue
            llm_seen.setdefault(key, {}).setdefault(str(value), set()).add(sequence)
            standard = LLM_STANDARD.get(key)
            if standard is not None and value != standard:
                notes.append(f"{key}={value} ≠ standard {standard}")
        if kind == "catalogs":
            clip = _catalog_clip(manifest)
            clipped = clip.startswith("clip plan")
            if clipped != (group.region_policy == "area625"):
                clip += f" ≠ region policy {group.region_policy}"
            notes.append(clip)
            notes.append(f"rows_out={manifest.config.get('rows_out', manifest.config.get('rows', '?'))}")
            region_km2[0] = _region_area_km2(manifest, group)
        rows.append([kind, version, "MISMATCH" if any("≠" in n for n in notes) else "OK",
                     "; ".join(notes)])
    prior = _prior_area_km2(root, group, sequence)
    status = ""
    if region_km2[0] is not None and prior[0].isdigit():
        largest = max(float(part.replace(",", "")) for part in prior.split(" / "))
        if abs(largest - region_km2[0]) > 0.02 * region_km2[0]:
            status = "MISMATCH"
            prior += f" ≠ region {region_km2[0]:,.0f}"
    rows.append(["uniform prior km²", "", status, prior])
    return rows


def _markdown(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"
              for row in rows]
    return "\n".join(lines)


def render_report(root: Path) -> str:
    sections = ["# Paper roster check", "",
                f"Generated from `paper/table_common.py` against `{root}`. "
                "Do not edit; change the roster and rerun `paper:check_roster`.", ""]
    llm_seen: dict[str, dict[str, set[str]]] = {}
    problems = 0
    for group in DATASET_GROUPS:
        sections.append(f"## {group.display_name} — scope `{group.catalog_scope}`, "
                        f"region `{group.region_policy}`, "
                        f"runs `{group.run_spec[0] if group.run_spec else 'none'}`")
        for sequence in group.sequences:
            rows = check_sequence(root, group, sequence, llm_seen)
            problems += sum(row[2] not in ("OK", "") for row in rows)
            sections += [f"### {sequence}", "",
                         _markdown(["lane", "version", "status", "notes"], rows), ""]
    sections += ["## LLM settings across sequences", ""]
    llm_rows = []
    for key in LLM_STANDARD:
        values = llm_seen.get(key, {})
        for value, sequences in sorted(values.items()):
            llm_rows.append([key, value, str(len(sequences)),
                             ", ".join(sorted(sequences))])
        if len(values) > 1 and LLM_STANDARD[key] is None:
            problems += 1
    sections += [_markdown(["setting", "value", "n", "sequences"], llm_rows), "",
                 f"**{problems} problem(s).**"]
    return "\n".join(sections)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farfield_root", type=Path, default=DEFAULT_FARFIELD_ROOT)
    parser.add_argument("--output", type=Path, default=None,
                        help="write the Markdown report here instead of stdout")
    args = parser.parse_args(argv)
    report = render_report(args.farfield_root)
    emit_table(report, args.output)
    return 0 if report.endswith("**0 problem(s).**") else 1


if __name__ == "__main__":
    sys.exit(main())
