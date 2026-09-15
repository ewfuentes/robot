"""Static viewer for a depth-render database: click through lattice
locations and see each 12-view rendered depth ring plus where it sits on the
terrain.

The database stores descriptors only, so the selected locations are
re-rendered here from the surface + the manifest's recorded render config —
what you see is bit-identical to what was embedded.

    bazel run //experimental/overhead_matching/swag/farfield/dem_baseline:export_render_viewer -- \
        --db_dir /data/farfield_matching/artifacts/depth_render_db/mount_washington/v1_dev100m \
        --n_locations 24

Output: the sibling `<db_dir>.viewer/viewer.html` — the farfield sibling-page
convention (RUN_SIBLING_PAGES in viewers/indexes.py), so the immutable
artifact version directory is never written into.
"""

import argparse
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

import numpy as np

from experimental.overhead_matching.swag.farfield.dem_baseline import (
    depth_render,
    render_db,
    terrain,
    viz,
)
from experimental.overhead_matching.swag.farfield.viewers import page

GENERATOR = "dem_baseline.export_render_viewer"


def location_page_name(index: int) -> str:
    return f"loc_{index:04d}.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="default: the <db_dir>.viewer sibling")
    parser.add_argument("--n_locations", type=int, default=24,
                        help="evenly spread over the lattice")
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="explicit lattice indices instead")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    db = render_db.load_database(args.db_dir)
    manifest = db["manifest"]
    out_dir = args.out_dir or args.db_dir.with_name(
        args.db_dir.name + ".viewer")
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)

    hf = terrain.HeightField.load(Path(manifest["height_field"]))
    config = depth_render.RenderConfig(**manifest["render_config"])
    tt = depth_render.TerrainTensor.from_height_field(hf, device=args.device)
    mapper = viz.MapRenderer(hf)
    bounds = tuple(manifest["lattice"]["bounds_xy"])

    n_total = len(db["x_m"])
    if args.indices:
        indices = [i for i in args.indices if 0 <= i < n_total]
    else:
        indices = sorted(set(
            int(i) for i in
            np.linspace(0, n_total - 1, min(args.n_locations, n_total))))

    viz.depth_colorbar().save(images / "colorbar.png")
    yaws = config.yaw_degrees()

    for k, loc in enumerate(indices):
        x, y = float(db["x_m"][loc]), float(db["y_m"][loc])
        ring = depth_render.render_ring(tt, config, x, y)
        entries = []
        for v in range(config.n_yaw):
            name = f"loc{loc:06d}_yaw{int(yaws[v]):03d}.png"
            viz.depth_image(ring.depth_m[v].cpu().numpy()).save(images / name)
            entries.append((f"images/{name}",
                            f"yaw {int(yaws[v])}&deg; &middot; cov "
                            f"{ring.coverage[v]:.2f}"))
        map_name = f"map_loc{loc:06d}.png"
        mapper.render(images / map_name, bounds_xy=bounds, markers=[
            (x, y, {"marker": "o", "s": 40, "facecolors": "none",
                    "edgecolors": "#e0a04a", "linewidths": 1.6}, "location"),
        ])
        lat, lon = terrain.latlon_from_utm(x, y, hf.crs)
        body = (
            viz.nav_html(k, len(indices), location_page_name)
            + '<div class="pane"><div>'
            + f"<h2>depth ring &middot; lattice index {loc}</h2>"
            + viz.thumb_strip(entries)
            + viz.colorbar_html("images/colorbar.png")
            + '</div><div class="mapimg">'
            + f"<h2>location</h2><img src='images/{map_name}'>"
            + f'<div class="muted">({x:.0f} E, {y:.0f} N) {page.esc(hf.crs)}'
            + f" &middot; {lat:.5f}, {lon:.5f} &middot; observer z "
            + f"{ring.observer_z_m:.1f} m</div></div></div>")
        (out_dir / location_page_name(k)).write_text(page.page(
            f"render {loc} — {args.db_dir.name}", body,
            generator=GENERATOR, extra_style=viz.VIEWER_STYLE,
            crumbs=[("viewer", "viewer.html"), (f"loc {loc}", None)]))

    overview_name = "map_overview.png"
    mapper.render(images / overview_name, bounds_xy=bounds, markers=[
        (float(db["x_m"][loc]), float(db["y_m"][loc]),
         {"marker": "o", "s": 14, "color": "#e0a04a"}, "rendered")
        for loc in indices])
    rows = [[
        f'<a href="{location_page_name(k)}">{location_page_name(k)}</a>',
        page.esc(loc),
        f"{db['x_m'][loc]:.0f}, {db['y_m'][loc]:.0f}",
        f"{db['coverage'][loc].mean():.3f}",
    ] for k, loc in enumerate(indices)]
    body = (
        f"<p>{n_total} lattice locations at "
        f"{manifest['lattice']['spacing_m']:.0f} m spacing; "
        f"{len(indices)} re-rendered for viewing. Render range "
        f"{manifest['render_config']['max_range_m']:.0f} m, observer "
        f"{manifest['render_config']['observer_height_m']} m.</p>"
        + f"<div class='mapimg'><img src='images/{overview_name}'></div>"
        + page.table(["page", "lattice idx", "easting, northing (m)",
                      "mean coverage"], rows))
    (out_dir / "viewer.html").write_text(page.page(
        f"depth render viewer — {args.db_dir.name}", body,
        generator=GENERATOR, extra_style=viz.VIEWER_STYLE))
    print(f"wrote {out_dir}/viewer.html ({len(indices)} location pages)")


if __name__ == "__main__":
    main()
