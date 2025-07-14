import argparse
from dataclasses import dataclass
from pathlib import Path
import osmnx as ox
import pandas as pd

from experimental.overhead_matching.swag.data import vigor_dataset as vd
from common.gps import web_mercator


@dataclass
class Options:
    tags: dict[str, str | list[str]]
    only_nodes: bool = True


to_retreive = {
    "bus_stop": Options({"highway": "bus_stop"}),
    "t_stop": Options({"railway": "station"}),
    "restaurants": Options({"amenity": ["restaurant", "cafe", "bar", "fast_food", "pub"]}),
    "grocery_store": Options({"shop": ["supermarket", "convenience", "greengrocer", "deli"]}),
    "places_of_worship": Options({"amenity": "place_of_worship"}),
    "school": Options({"amenity": ["school", "college", "university"]}),
    # "residential": Options({"building": True}, only_nodes=False),
    # "parks": Options({"leisure": ["park", "garden", "nature_reserve"]}, only_nodes=False),
}


def main(dataset_path: Path, zoom_level: int, output_path: Path | None, show_extracted_landmarks: bool):
    sat_metadata = vd.load_satellite_metadata(dataset_path / "satellite", zoom_level)

    min_yx_pixel = sat_metadata[["web_mercator_y", "web_mercator_x"]].min().to_numpy()
    max_yx_pixel = sat_metadata[["web_mercator_y", "web_mercator_x"]].max().to_numpy()

    top, left = web_mercator.pixel_coords_to_latlon(*min_yx_pixel, zoom_level)
    bottom, right = web_mercator.pixel_coords_to_latlon(*max_yx_pixel, zoom_level)
    bbox = [left, bottom, right, top]

    dfs = []

    def only_nodes(df):
        return df[df.geometry.geom_type == "Point"]

    for type, opts in to_retreive.items():
        df = ox.features_from_bbox(bbox, opts.tags)
        if opts.only_nodes:
            df = only_nodes(df)
        df["landmark_type"] = type
        dfs.append(df)
    landmark_df = pd.concat(dfs)

    if show_extracted_landmarks:
        import matplotlib.pyplot as plt
        G = ox.graph.graph_from_bbox(bbox, network_type="drive")
        _fig, _ax = ox.plot.plot_graph(G, show=False)
        for t, group in landmark_df.groupby("landmark_type"):
            group.plot(ax=_ax, label=t)

        plt.legend()
        plt.tight_layout()
        plt.show(block=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        landmark_df.to_file(output_path, driver="GeoJSON")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path")
    parser.add_argument("--output_path")
    parser.add_argument("--show_extracted_landmarks", action='store_true')
    parser.add_argument("--zoom_level", type=int, default=20)
    args = parser.parse_args()

    main(Path(args.dataset_path), args.zoom_level, Path(args.output_path) if args.output_path else None, args.show_extracted_landmarks)
