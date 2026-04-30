"""Patches the OSM Bright GL style to read from local MBTiles + local glyphs.

The upstream style targets MapTiler's hosted services. We swap three URLs:

    sources.openmaptiles.url -> mbtiles:///abs/path/<region>.mbtiles
    glyphs                   -> file:///abs/path/{fontstack}/{range}.pbf
    sprite                   -> removed; not shipped in the v1.11 release zip,
                                 only published to github-pages. Layers with
                                 icon-image references quietly skip rendering.
"""
import json
import os
from pathlib import Path


def _runfiles_dir() -> Path:
    runfiles = os.environ.get("RUNFILES_DIR")
    if runfiles:
        return Path(runfiles)
    # __file__ is a symlink into the runfiles tree; do NOT resolve symlinks
    # or we'll end up in the source repo, which has no .runfiles ancestor.
    here = Path(os.path.abspath(__file__))
    for parent in here.parents:
        if parent.name.endswith(".runfiles"):
            return parent
    raise RuntimeError("could not locate runfiles tree; invoke via `bazel run` or `bazel test`")


def _resolve_external(repo_name: str) -> Path:
    """Locate the runfiles root of an external repo (e.g. @osm_bright_gl_style)."""
    candidates = [
        _runfiles_dir() / repo_name,
        _runfiles_dir() / "_main" / "external" / repo_name,
        _runfiles_dir().parent / repo_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"could not locate runfiles for @{repo_name}; tried: {[str(c) for c in candidates]}"
    )


def osm_bright_dir() -> Path:
    return _resolve_external("osm_bright_gl_style")


def fonts_dir() -> Path:
    return _resolve_external("openmaptiles_fonts")


def render_style(mbtiles_path: Path, drop_text: bool = True) -> str:
    """Patch the OSM Bright style for offline rendering against a local MBTiles.

    drop_text=True (default) removes all `symbol` layers — these are MapLibre's
    text/icon renderers. Removing them gives a label-free render that's a)
    deterministic across runs (label placement is otherwise an output of a
    label-collision algorithm), b) free of confounding read-the-text signal
    that wouldn't appear in satellite imagery.
    """
    style_path = osm_bright_dir() / "style.json"
    fonts_root = fonts_dir()

    with open(style_path) as f:
        style = json.load(f)

    style["sources"] = {
        "openmaptiles": {
            "type": "vector",
            "url": f"mbtiles://{Path(mbtiles_path).resolve()}",
        }
    }
    style["glyphs"] = f"file://{fonts_root.resolve()}/{{fontstack}}/{{range}}.pbf"
    style.pop("sprite", None)

    if drop_text:
        style["layers"] = [l for l in style["layers"] if l.get("type") != "symbol"]

    return json.dumps(style)
