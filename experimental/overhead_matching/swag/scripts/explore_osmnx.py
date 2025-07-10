import marimo

__generated_with = "0.11.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import osmnx as ox
    import networkx as nx
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import pandas as pd
    from dataclasses import dataclass
    return dataclass, gpd, mo, nx, ox, pd, plt


@app.cell
def _():
    def only_nodes(df):
        return df[df.geometry.geom_type == "Point"]
    return (only_nodes,)


@app.cell
def _(dataclass):
    @dataclass
    class Options:
        tags: dict[str, str | list[str]]
        only_nodes: bool = True

    to_retreive = {
        "bus_stops": Options({"highway": "bus_stop"}),
        "t_stops": Options({"railway": "station"}),
        "restuarants": Options({"amenity": ["restaurant", "cafe", "bar", "fast_food", "pub"]}),
        "grocery_store": Options({"shop": ["supermarket", "convenience", "greengrocer", "deli"]}),
        "places_of_worship": Options({"amenity": "place_of_worship"}),
        "schools": Options({"amenity": ["school", "college", "university"]}),
        # "residential": Options({"building": True}, only_nodes=False),
        # "parks": Options({"leisure": ["park", "garden", "nature_reserve"]}, only_nodes=False),
    }
    return Options, to_retreive


@app.cell
def _(only_nodes, ox, pd, to_retreive):
    G = ox.graph.graph_from_place("Cambridge, Massachusetts, USA", network_type="drive")
    _landmarks = []
    for _name, _opts in to_retreive.items():
        _features = ox.features_from_place("Cambridge, Massachusetts, USA", tags=_opts.tags)
        if _opts.only_nodes:
            _features = only_nodes(_features)
        _features["type"] = _name
        _landmarks.append(_features)
    
    landmarks_df = pd.concat(_landmarks)
    return G, landmarks_df


@app.cell
def _(G, landmarks_df, mo, ox, plt):

    plt.figure()
    _fig, _ax = ox.plot.plot_graph(G)
    for t, group in landmarks_df.groupby("type"):
        group.plot(ax=_ax, label=t)

    plt.legend()
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return group, t


@app.cell
def _(landmarks_df):
    landmarks_df
    return


@app.cell
def _(landmarks_df):
    landmarks_df["type"]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
