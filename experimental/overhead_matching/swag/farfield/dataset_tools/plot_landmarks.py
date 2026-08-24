"""Publish coverage diagnostics for one full typed landmark catalog.

This is collection stage 8. It replaces the checkpoint-era plotter that
searched dataset-local landmark sources and wrote a PNG into the frozen
dataset. The port has explicit inputs and one immutable output:

* the frozen dataset supplies the validated trajectory;
* a complete full CATALOGS artifact supplies landmarks and extraction
  provenance;
* cached Geofabrik .poly files supply the source clip boundaries already
  checked by collection stage 5;
* a transactional catalog_coverage artifact publishes the PNG, a
  machine-readable report, and a review page.

No matching result or trimmed catalog is involved. This diagnostic asks
whether source extraction covered the requested area before semantic trimming.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box

from experimental.overhead_matching.swag.farfield import (
    artifact,
    dataset,
    geometry,
    paths as paths_lib,
    publication,
    provenance,
)
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.collection import pbf_coverage
from experimental.overhead_matching.swag.farfield.viewers import page


GENERATOR = (
    "//experimental/overhead_matching/swag/farfield/dataset_tools:"
    "plot_landmarks"
)
PAYLOAD_SCHEMA = "farfield_catalog_coverage/v1"
OUTPUTS = (
    "coverage_report.json",
    "index.html",
    "landmark_coverage.png",
)
_FULL_CONFIG_KEYS = frozenset({
    "schema",
    "bbox_wsen",
    "osm_specs",
    "enc_state",
    "enc_cells",
    "enc_available",
    "dedupe_tolerance_m",
    "node_margin_deg",
    "selected_source_feather",
    "selected_source_sha256",
    "rows",
    "source_coverage",
})
_SOURCE_COVERAGE_KEYS = frozenset({
    "schema",
    "status",
    "message",
    "details",
    "reference_specs",
})
_ANALYSIS_KEYS = frozenset({
    "grid_cells",
    "max_empty_run",
    "empty_fraction_warning",
    "far_range_km",
    "min_far_fraction",
    "max_track_samples",
})


class CoverageError(ValueError):
    """A source or result cannot satisfy the stage-8 diagnostic contract."""


def _exact_keys(value: Any, expected: frozenset[str], where: str) -> dict:
    if not isinstance(value, dict):
        raise CoverageError(f"{where} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise CoverageError(
            f"{where} has missing={missing}, unknown={unknown}")
    return value


def _finite(value: Any, where: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CoverageError(f"{where} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise CoverageError(f"{where} must be finite")
    if minimum is not None and result < minimum:
        raise CoverageError(f"{where} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise CoverageError(f"{where} must be <= {maximum}")
    return result


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CoverageError(f"{where} must be a positive integer")
    return value


def validate_analysis_config(config: dict) -> dict:
    config = dict(_exact_keys(config, _ANALYSIS_KEYS, "coverage config"))
    config["grid_cells"] = _positive_int(
        config["grid_cells"], "coverage grid_cells")
    config["max_empty_run"] = _positive_int(
        config["max_empty_run"], "coverage max_empty_run")
    if config["max_empty_run"] > config["grid_cells"]:
        raise CoverageError(
            "coverage max_empty_run cannot exceed grid_cells")
    config["empty_fraction_warning"] = _finite(
        config["empty_fraction_warning"],
        "coverage empty_fraction_warning", minimum=0.0, maximum=1.0)
    config["far_range_km"] = _finite(
        config["far_range_km"], "coverage far_range_km", minimum=0.0)
    config["min_far_fraction"] = _finite(
        config["min_far_fraction"], "coverage min_far_fraction",
        minimum=0.0, maximum=1.0)
    config["max_track_samples"] = _positive_int(
        config["max_track_samples"], "coverage max_track_samples")
    return config


def _bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise CoverageError(
            "full catalog config bbox_wsen must contain four values")
    west, south, east, north = (
        _finite(item, f"full catalog bbox_wsen[{index}]")
        for index, item in enumerate(value)
    )
    if not (-180.0 <= west < east <= 180.0
            and -90.0 <= south < north <= 90.0):
        raise CoverageError(
            "full catalog bbox_wsen must be ordered W,S,E,N WGS84 bounds")
    return west, south, east, north


def validate_full_catalog_config(
        manifest: artifact.ArtifactManifest,
) -> tuple[dict, tuple[float, float, float, float]]:
    config = dict(_exact_keys(
        manifest.config, _FULL_CONFIG_KEYS, "full catalog manifest config"))
    if config["schema"] != schema.FULL_ARTIFACT_SCHEMA:
        raise CoverageError(
            "catalog is not a stage-5 full catalog artifact")
    bbox = _bbox(config["bbox_wsen"])
    specs = config["osm_specs"]
    if (not isinstance(specs, list) or not specs
            or not all(isinstance(spec, str) and spec for spec in specs)):
        raise CoverageError(
            "full catalog osm_specs must be a non-empty string list")
    cells = config["enc_cells"]
    if (not isinstance(cells, list)
            or not all(isinstance(cell, str) and cell for cell in cells)):
        raise CoverageError(
            "full catalog enc_cells must be a string list")
    if config["enc_state"] is not None and (
            not isinstance(config["enc_state"], str)
            or not config["enc_state"]):
        raise CoverageError("full catalog enc_state must be null or non-empty")
    if type(config["enc_available"]) is not bool:
        raise CoverageError("full catalog enc_available must be boolean")
    _finite(config["dedupe_tolerance_m"],
            "full catalog dedupe_tolerance_m", minimum=0.0)
    node_margin = _finite(
        config["node_margin_deg"], "full catalog node_margin_deg")
    if node_margin < 0.0 and node_margin != -1.0:
        raise CoverageError(
            "full catalog node_margin_deg must be -1 or nonnegative")
    if (not isinstance(config["selected_source_feather"], str)
            or not config["selected_source_feather"]):
        raise CoverageError(
            "full catalog selected_source_feather must be non-empty")
    digest = config["selected_source_sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef"
                   for character in digest)):
        raise CoverageError(
            "full catalog selected_source_sha256 must be lowercase SHA-256")
    if (isinstance(config["rows"], bool)
            or not isinstance(config["rows"], int)
            or config["rows"] < 0):
        raise CoverageError("full catalog rows must be nonnegative integer")
    coverage = _exact_keys(
        config["source_coverage"], _SOURCE_COVERAGE_KEYS,
        "full catalog source_coverage")
    if coverage["schema"] != "farfield_catalog_source_coverage/v1":
        raise CoverageError("unsupported full catalog source_coverage schema")
    if coverage["status"] not in ("passed", "skipped_by_operator"):
        raise CoverageError("invalid full catalog source_coverage status")
    if not isinstance(coverage["message"], str) or not coverage["message"]:
        raise CoverageError(
            "full catalog source_coverage message must be non-empty")
    if not isinstance(coverage["details"], list):
        raise CoverageError(
            "full catalog source_coverage details must be a list")
    if (not isinstance(coverage["reference_specs"], list)
            or not all(isinstance(spec, str) and spec
                       for spec in coverage["reference_specs"])):
        raise CoverageError(
            "full catalog source_coverage reference_specs must be a string "
            "list")
    return config, bbox


def _dataset_inputs(dataset_name: str, dataset_dir: Path) -> dict:
    dataset_dir = Path(dataset_dir)
    metadata = dataset.load_metadata(dataset_dir)
    if metadata["dataset_name"] != dataset_name:
        raise CoverageError(
            "dataset metadata identity disagrees with --dataset: "
            f"{metadata['dataset_name']!r} != {dataset_name!r}")
    frames = dataset.load_frames(dataset_dir)
    if not frames:
        raise CoverageError(f"{dataset_dir} has no validated frames")
    try:
        source_digests = {
            "pipeline_metadata_sha256": artifact.sha256_file(
                dataset_dir / "pipeline_metadata.json"),
            "frames_gps_sha256": artifact.sha256_file(
                dataset_dir / "frames_gps.csv"),
            "panorama_index_sha256": artifact.sha256_json([
                {
                    "pano_id": frame.pano_id,
                    "pano_stem": frame.pano_stem,
                    "lat": frame.lat,
                    "lon": frame.lon,
                }
                for frame in frames
            ]),
        }
    except artifact.ArtifactError as error:
        raise CoverageError(f"invalid frozen dataset source: {error}") from error
    return {
        "metadata": metadata,
        "frames": frames,
        "source_digests": source_digests,
    }


def _clip_boundaries(specs: list[str], poly_cache_dir: Path) -> tuple[list, dict]:
    poly_cache_dir = Path(poly_cache_dir)
    if poly_cache_dir.is_symlink() or not poly_cache_dir.is_dir():
        raise CoverageError(
            f"poly cache must be a regular directory: {poly_cache_dir}")
    boundaries = []
    digests = {}
    for spec in specs:
        name = pbf_coverage.poly_url_for(spec).rsplit("/", 1)[-1]
        path = poly_cache_dir / name
        try:
            digest = artifact.sha256_file(path)
            boundary = pbf_coverage.parse_poly(path)
        except (artifact.ArtifactError, OSError, ValueError) as error:
            raise CoverageError(
                f"invalid cached Geofabrik boundary for {spec}: {error}"
            ) from error
        if boundary is None or boundary.is_empty or not boundary.is_valid:
            raise CoverageError(
                f"cached Geofabrik boundary for {spec} is empty or invalid")
        boundaries.append((spec, boundary))
        digests[spec] = digest
    return boundaries, digests


def load_inputs(dataset_name: str, dataset_dir: Path, catalog_dir: Path,
                poly_cache_dir: Path) -> dict:
    dataset_inputs = _dataset_inputs(dataset_name, dataset_dir)
    try:
        catalog_ref = artifact.open_artifact(
            catalog_dir, expected_kind=paths_lib.CATALOGS,
            expected_dataset=dataset_name)
        catalog_manifest = artifact.load_manifest(catalog_dir)
    except artifact.ArtifactError as error:
        raise CoverageError(f"invalid full catalog artifact: {error}") from error
    if catalog_manifest.declared_outputs != ("catalog.feather",):
        raise CoverageError(
            "full catalog must declare exactly catalog.feather")
    full_config, bbox = validate_full_catalog_config(catalog_manifest)
    try:
        catalog = schema.read_frame(Path(catalog_dir) / "catalog.feather")
    except (OSError, schema.CatalogSchemaError) as error:
        raise CoverageError(f"invalid full catalog payload: {error}") from error
    payload_digest = artifact.sha256_file(
        Path(catalog_dir) / "catalog.feather")
    if payload_digest != full_config["selected_source_sha256"]:
        raise CoverageError(
            "full catalog payload does not match the selected source digest")
    if len(catalog) != full_config["rows"]:
        raise CoverageError(
            "catalog row count disagrees with its manifest: "
            f"{len(catalog)} != {full_config['rows']}")
    boundaries, boundary_digests = _clip_boundaries(
        full_config["osm_specs"], poly_cache_dir)
    dataset_inputs["source_digests"]["geofabrik_poly_sha256"] = (
        boundary_digests)
    return {
        **dataset_inputs,
        "catalog": catalog,
        "catalog_ref": catalog_ref,
        "catalog_manifest": catalog_manifest,
        "full_config": full_config,
        "bbox": bbox,
        "boundaries": boundaries,
        "dataset_dir": Path(dataset_dir),
        "poly_cache_dir": Path(poly_cache_dir),
    }


def verify_inputs_unchanged(resolved: dict) -> None:
    """Recheck every external identity immediately before publication."""
    current_ref = artifact.open_artifact(
        resolved["catalog_ref"].path,
        expected_kind=paths_lib.CATALOGS,
        expected_dataset=resolved["catalog_ref"].dataset,
        expected_version=resolved["catalog_ref"].version,
    )
    if current_ref != resolved["catalog_ref"]:
        raise CoverageError(
            "full catalog artifact changed during coverage analysis")
    try:
        current_dataset = _dataset_inputs(
            resolved["catalog_ref"].dataset, resolved["dataset_dir"])
    except (CoverageError, dataset.ContractViolation) as error:
        raise CoverageError(
            f"frozen dataset changed during coverage analysis: {error}"
        ) from error
    _, current_poly_digests = _clip_boundaries(
        resolved["full_config"]["osm_specs"], resolved["poly_cache_dir"])
    current_digests = dict(current_dataset["source_digests"])
    current_digests["geofabrik_poly_sha256"] = current_poly_digests
    if current_digests != resolved["source_digests"]:
        raise CoverageError(
            "frozen dataset or Geofabrik boundary changed during coverage "
            "analysis")


def _point_table(catalog, bbox_wsen) -> dict:
    west, south, east, north = bbox_wsen
    clip = box(west, south, east, north)
    frame = catalog.to_crs("EPSG:4326")
    tags = schema.tag_dicts(frame)
    lon = []
    lat = []
    sources = []
    kept_tags = []
    outside = 0
    for position, geometry_value in enumerate(frame.geometry):
        clipped = geometry_value.intersection(clip)
        if clipped.is_empty:
            outside += 1
            continue
        point = clipped.representative_point()
        lon.append(float(point.x))
        lat.append(float(point.y))
        sources.append(frame.iloc[position]["landmark_type"])
        kept_tags.append(tags[position])
    return {
        "lon": np.asarray(lon, dtype=np.float64),
        "lat": np.asarray(lat, dtype=np.float64),
        "sources": sources,
        "tags": kept_tags,
        "outside_bbox": outside,
    }


def _sample_track(frames: list[dataset.Frame], maximum: int) -> tuple[np.ndarray,
                                                                      np.ndarray]:
    count = min(len(frames), maximum)
    indices = np.unique(np.linspace(
        0, len(frames) - 1, count, dtype=np.int64))
    return (
        np.asarray([frames[index].lon for index in indices],
                   dtype=np.float64),
        np.asarray([frames[index].lat for index in indices],
                   dtype=np.float64),
    )


def _nearest_track_distances_km(points: dict, frames: list[dataset.Frame],
                                maximum_track_samples: int) -> np.ndarray:
    if not len(points["lon"]):
        return np.empty(0, dtype=np.float64)
    track_lon, track_lat = _sample_track(frames, maximum_track_samples)
    anchor_lat = float(np.mean(track_lat))
    anchor_lon = float(np.mean(track_lon))
    region = geometry.RegionFrame(anchor_lat, anchor_lon)
    track_e, track_n = region.enu_from_latlon(track_lat, track_lon)
    point_e, point_n = region.enu_from_latlon(points["lat"], points["lon"])
    distances = np.empty(len(point_e), dtype=np.float64)
    for start in range(0, len(point_e), 4096):
        stop = min(start + 4096, len(point_e))
        delta_e = point_e[start:stop, None] - track_e[None, :]
        delta_n = point_n[start:stop, None] - track_n[None, :]
        distances[start:stop] = np.sqrt(
            np.min(delta_e * delta_e + delta_n * delta_n, axis=1))
    return distances / 1000.0


def _best_empty_run(occupied: np.ndarray) -> int:
    run = best = 0
    for value in occupied:
        run = 0 if value else run + 1
        best = max(best, run)
    return best


def analyze(resolved: dict, analysis_config: dict) -> dict:
    config = validate_analysis_config(analysis_config)
    bbox_wsen = resolved["bbox"]
    west, south, east, north = bbox_wsen
    points = _point_table(resolved["catalog"], bbox_wsen)
    distances = _nearest_track_distances_km(
        points, resolved["frames"], config["max_track_samples"])
    histogram, lon_edges, lat_edges = np.histogram2d(
        points["lon"], points["lat"], bins=config["grid_cells"],
        range=[[west, east], [south, north]])
    occupied = histogram > 0
    findings = []

    def finding(severity: str, code: str, message: str, **metrics) -> None:
        findings.append({
            "severity": severity,
            "code": code,
            "message": message,
            "metrics": metrics,
        })

    source_coverage = resolved["full_config"]["source_coverage"]
    finding(
        ("ok" if source_coverage["status"] == "passed" else "warning"),
        "source_extract_coverage",
        source_coverage["message"],
        status=source_coverage["status"],
    )

    if not len(points["lon"]):
        finding("error", "no_landmarks",
                "the full catalog has no geometry intersecting its bbox")
    else:
        for axis, label, edges in (
                (0, "longitude", lon_edges),
                (1, "latitude", lat_edges)):
            bands = occupied.any(axis=1 - axis)
            populated = np.flatnonzero(bands)
            interior = bands[populated[0]:populated[-1] + 1]
            longest = _best_empty_run(interior)
            degrees = abs(float(edges[1] - edges[0]))
            cell_km = degrees * 111.0 * (
                math.cos(math.radians((south + north) / 2.0))
                if axis == 0 else 1.0)
            severity = (
                "error" if longest >= config["max_empty_run"] else "ok")
            finding(
                severity, f"interior_{label}_gap",
                f"largest interior {label} gap is {longest} grid bands",
                empty_bands=longest,
                approximate_km=round(longest * cell_km, 3),
            )

    empty_fraction = float(1.0 - occupied.mean())
    finding(
        ("warning" if empty_fraction >= config["empty_fraction_warning"]
         else "ok"),
        "empty_grid_fraction",
        f"{100.0 * empty_fraction:.1f}% of the bbox grid is empty",
        fraction=round(empty_fraction, 8),
    )

    far_fraction = (
        float(np.mean(distances > config["far_range_km"]))
        if len(distances) else 0.0)
    finding(
        ("warning" if far_fraction < config["min_far_fraction"] else "ok"),
        "far_range_tail",
        f"{100.0 * far_fraction:.1f}% of landmarks are farther than "
        f"{config['far_range_km']:g} km from the trajectory",
        fraction=round(far_fraction, 8),
        range_km=config["far_range_km"],
        median_km=(round(float(np.median(distances)), 6)
                   if len(distances) else None),
        maximum_km=(round(float(np.max(distances)), 6)
                    if len(distances) else None),
    )

    source_counts = dict(sorted(Counter(points["sources"]).items()))
    report = {
        "schema": PAYLOAD_SCHEMA,
        "passed": not any(item["severity"] == "error"
                          for item in findings),
        "dataset": resolved["catalog_ref"].dataset,
        "catalog": resolved["catalog_ref"].to_dict(),
        "dataset_source_digests": resolved["source_digests"],
        "analysis_config": config,
        "catalog_build": {
            "bbox_wsen": list(bbox_wsen),
            "osm_specs": list(resolved["full_config"]["osm_specs"]),
            "enc_state": resolved["full_config"]["enc_state"],
            "enc_cells": list(resolved["full_config"]["enc_cells"]),
            "enc_available": resolved["full_config"]["enc_available"],
            "source_coverage": source_coverage,
        },
        "summary": {
            "frames": len(resolved["frames"]),
            "catalog_rows": len(resolved["catalog"]),
            "plotted_landmarks": len(points["lon"]),
            "geometries_outside_bbox": points["outside_bbox"],
            "landmarks_by_source": source_counts,
        },
        "findings": findings,
    }
    return {
        "report": report,
        "points": points,
        "distances_km": distances,
        "histogram": histogram,
        "lon_edges": lon_edges,
        "lat_edges": lat_edges,
    }


def render_figure(resolved: dict, analysis: dict, destination: Path) -> None:
    report = analysis["report"]
    points = analysis["points"]
    distances = analysis["distances_km"]
    west, south, east, north = resolved["bbox"]
    track_lon = np.asarray([frame.lon for frame in resolved["frames"]])
    track_lat = np.asarray([frame.lat for frame in resolved["frames"]])

    figure, axes = plt.subplots(2, 2, figsize=(17, 12))
    figure.suptitle(
        f"{report['dataset']} — full catalog coverage "
        f"({report['summary']['plotted_landmarks']:,} landmarks, "
        f"{report['summary']['frames']:,} frames)",
        fontsize=14,
    )

    axis = axes[0][0]
    for source in sorted(set(points["sources"])):
        selected = np.asarray(
            [value == source for value in points["sources"]], dtype=bool)
        axis.scatter(points["lon"][selected], points["lat"][selected],
                     s=1.5, alpha=0.3,
                     label=f"{source} ({int(selected.sum()):,})")
    axis.plot(track_lon, track_lat, "-", color="red", lw=2.0,
              label="trajectory", zorder=5)
    axis.plot(track_lon[0], track_lat[0], "o", color="lime", ms=8,
              mec="black", zorder=6, label="start")
    axis.add_patch(plt.Rectangle(
        (west, south), east - west, north - south,
        fill=False, ec="black", ls="--", lw=1.5, label="request bbox"))
    for spec, boundary in resolved["boundaries"]:
        clipped = boundary.intersection(box(west, south, east, north))
        geometries = (
            list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped])
        for polygon in geometries:
            if polygon.is_empty or not hasattr(polygon, "exterior"):
                continue
            x, y = polygon.exterior.xy
            axis.plot(x, y, color="purple", alpha=0.65, lw=1.0,
                      label=(f"Geofabrik: {spec.rsplit('/', 1)[-1]}"
                             if spec else "Geofabrik"))
    axis.set_xlim(west, east)
    axis.set_ylim(south, north)
    axis.set_aspect(1.0 / math.cos(math.radians((south + north) / 2.0)))
    axis.set_title("full catalog, trajectory, and cached extract boundaries")
    axis.set_xlabel("longitude")
    axis.set_ylabel("latitude")
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), loc="best", fontsize=7)

    axis = axes[0][1]
    density = np.log10(analysis["histogram"].T + 1.0)
    image = axis.imshow(
        density, origin="lower", aspect="auto",
        extent=[west, east, south, north], cmap="viridis")
    axis.plot(track_lon, track_lat, "-", color="red", lw=1.3)
    axis.set_title(
        f"log10 density on a "
        f"{report['analysis_config']['grid_cells']}×"
        f"{report['analysis_config']['grid_cells']} grid")
    axis.set_xlabel("longitude")
    axis.set_ylabel("latitude")
    figure.colorbar(image, ax=axis, label="log10(count + 1)")

    axis = axes[1][0]
    if len(distances):
        axis.hist(distances, bins=60, color="steelblue")
        axis.axvline(
            report["analysis_config"]["far_range_km"],
            color="red", ls="--", label="far-range threshold")
        axis.set_yscale("log")
        axis.legend(fontsize=8)
    else:
        axis.text(0.5, 0.5, "no landmarks", ha="center", va="center",
                  transform=axis.transAxes)
    axis.set_title("landmark distance from trajectory")
    axis.set_xlabel("kilometres")
    axis.set_ylabel("count")

    axis = axes[1][1]
    tag_name = None
    tag_counts = Counter()
    for candidate in ("man_made", "building", "natural", "place",
                      "seamark:type"):
        candidate_counts = Counter(
            tag_record[candidate] for tag_record in points["tags"]
            if candidate in tag_record)
        if candidate_counts:
            tag_name = candidate
            tag_counts = candidate_counts
            break
    if tag_name is None:
        tag_name = "landmark_type"
        tag_counts = Counter(points["sources"])
    common = tag_counts.most_common(14)
    if common:
        labels = [str(key) for key, _ in reversed(common)]
        values = [value for _, value in reversed(common)]
        axis.barh(labels, values, color="darkorange")
        axis.set_xscale("log")
    else:
        axis.text(0.5, 0.5, "no classes", ha="center", va="center",
                  transform=axis.transAxes)
    axis.set_title(f"most common {tag_name} values")
    axis.set_xlabel("count")

    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure.savefig(destination, dpi=110)
    plt.close(figure)


def render_page(report: dict) -> str:
    status_class = "ok" if report["passed"] else "bad"
    status = "PASS" if report["passed"] else "FAIL"
    rows = []
    for finding in report["findings"]:
        css = {
            "error": "bad",
            "warning": "warn",
            "ok": "ok",
        }[finding["severity"]]
        rows.append([
            f'<span class="{css}">{page.esc(finding["severity"])}</span>',
            page.esc(finding["code"]),
            page.esc(finding["message"]),
        ])
    body = (
        f'<p class="{status_class}"><strong>{status}</strong></p>'
        '<p><a href="coverage_report.json">machine-readable report</a></p>'
        '<p><img src="landmark_coverage.png" alt="landmark coverage" '
        'style="max-width:100%;height:auto"></p>'
        '<h2>Findings</h2>'
        + page.table(["severity", "check", "result"], rows)
    )
    return page.page(
        f"{report['dataset']} catalog coverage",
        body,
        generator=GENERATOR,
    )


def publish(resolved: dict, output_dir: Path, analysis_config: dict,
            *, arguments: tuple[str, ...] = ()) -> tuple[artifact.ArtifactRef,
                                                         dict]:
    analysis = analyze(resolved, analysis_config)
    verify_inputs_unchanged(resolved)
    report = analysis["report"]
    output_dir = Path(output_dir)
    manifest_config = {
        "schema": PAYLOAD_SCHEMA,
        "analysis": report["analysis_config"],
        "dataset_source_digests": report["dataset_source_digests"],
        "catalog_build": report["catalog_build"],
    }
    with publication.published_artifact(
            output_dir,
            kind=paths_lib.CATALOG_COVERAGE,
            dataset=resolved["catalog_ref"].dataset,
            version=output_dir.name,
            generator=GENERATOR,
            git_commit=provenance.git_commit(),
            arguments=arguments,
            upstreams=(resolved["catalog_ref"],),
            config=manifest_config,
            declared_outputs=OUTPUTS) as builder:
        render_figure(
            resolved, analysis,
            builder.output_path("landmark_coverage.png"))
        artifact.atomic_write_json(
            builder.output_path("coverage_report.json"), report)
        artifact.atomic_write_file(
            builder.output_path("index.html"),
            render_page(report).encode("utf-8"))
    return builder.artifact_ref, report


def cli(argv=None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_dir", required=True, type=Path)
    parser.add_argument("--catalog_dir", required=True, type=Path)
    parser.add_argument("--poly_cache_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--grid_cells", required=True, type=int)
    parser.add_argument("--max_empty_run", required=True, type=int)
    parser.add_argument("--empty_fraction_warning", required=True, type=float)
    parser.add_argument("--far_range_km", required=True, type=float)
    parser.add_argument("--min_far_fraction", required=True, type=float)
    parser.add_argument("--max_track_samples", required=True, type=int)
    args = parser.parse_args(raw_argv)
    try:
        resolved = load_inputs(
            args.dataset, args.dataset_dir, args.catalog_dir,
            args.poly_cache_dir)
        _, report = publish(
            resolved,
            args.output_dir,
            {
                "grid_cells": args.grid_cells,
                "max_empty_run": args.max_empty_run,
                "empty_fraction_warning": args.empty_fraction_warning,
                "far_range_km": args.far_range_km,
                "min_far_fraction": args.min_far_fraction,
                "max_track_samples": args.max_track_samples,
            },
            arguments=tuple(raw_argv),
        )
    except (CoverageError, artifact.ArtifactError,
            dataset.ContractViolation, publication.PublicationValidationError,
            schema.CatalogSchemaError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(cli())
