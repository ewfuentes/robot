"""Patches the OSM Bright GL style to read from local MBTiles + local glyphs.

The upstream style targets MapTiler's hosted services. We swap three URLs:

    sources.openmaptiles.url -> mbtiles:///abs/path/<region>.mbtiles
    glyphs                   -> file:///abs/path/{fontstack}/{range}.pbf
    sprite                   -> removed; not shipped in the v1.11 release zip,
                                 only published to github-pages. Layers with
                                 icon-image references quietly skip rendering.
"""
import json
from pathlib import Path


def _runfiles():
    from python.runfiles import Runfiles  # @rules_python//python/runfiles
    return Runfiles.Create()


def osm_bright_dir() -> Path:
    return Path(_runfiles().Rlocation("osm_bright_gl_style/style.json")).parent


def fonts_dir() -> Path:
    # Probe a known font file inside the archive; its parent.parent is the dir.
    pbf = _runfiles().Rlocation("openmaptiles_fonts/Noto Sans Regular/0-255.pbf")
    return Path(pbf).parent.parent


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
