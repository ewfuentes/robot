from dataclasses import dataclass
import matplotlib as mpl
import matplotlib.pyplot as plt
import supervision as sv
import pandas as pd
import seaborn as sns
import numpy as np
import networkx as nx
import itertools
from typing import Tuple

import experimental.overhead_matching.grounding_sam as gs


@dataclass
class OverheadMatchingInput:
    overhead: np.ndarray
    ego: np.ndarray


def _node_x_distance(node_x: float, others: pd.Series, wrap_around_width: None | int):
    if wrap_around_width is not None:
        a = np.minimum(node_x, others)
        b = np.maximum(node_x, others)
        dx = np.minimum(b - a, a + wrap_around_width - b)
    else:
        dx = others - node_x

    return dx


def _compute_nodes(
    image: np.ndarray, model: gs.GroundingSam, classes: list[str], debug=False
) -> pd.DataFrame:
    result = model.detect_queries(image, classes)

    out = pd.DataFrame(
        result["dino_results"]["boxes"],
        columns=[f"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"],
    )
    out.loc[:, "class"] = result["dino_results"]["labels"]
    out.loc[:, "center_x"] = (out["bbox_x1"] + out["bbox_x2"]) / 2.0
    out.loc[:, "center_y"] = (out["bbox_y1"] + out["bbox_y2"]) / 2.0

    if debug:
        detections = sv.Detections(
            xyxy=out[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].values,
            class_id=np.array(out.index),
        )

        box_annotator = sv.BoxAnnotator()
        ann = box_annotator.annotate(scene=image.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        ann = label_annotator.annotate(
            scene=ann,
            detections=detections,
            labels=[str(x) for x in out["class"].values],
        )

        plt.figure()
        plt.imshow(ann)

    return out


def _compute_ego_nodes(image, model: gs.GroundingSam, classes: list[str], debug: bool):
    doubled_image = np.concatenate(
        [
            image,
            image,
        ],
        axis=1,
    )

    # Compute nodes from the first panorama
    nodes_in_image = _compute_nodes(doubled_image, model, classes, debug)

    image_half_width = image.shape[1] // 2
    center_x = nodes_in_image["center_x"]
    mask = np.logical_and(
        center_x >= image_half_width, center_x < 2 * image.shape[1] - image_half_width
    )

    center_nodes_in_image = nodes_in_image[mask].copy()
    center_nodes_in_image["center_x"] %= image.shape[1]

    return center_nodes_in_image.reset_index(drop=True)


def _compute_covisibility_window_size(
    nodes: pd.DataFrame, ego_width: None | int
) -> float:
    all_dists = []
    for t in nodes.itertuples():
        other_nodes = nodes.loc[nodes.index != t.Index, :]
        dx = _node_x_distance(t.center_x, other_nodes["center_x"], ego_width)
        dy = other_nodes["center_y"] - t.center_y
        all_dists.append(np.min((dx * dx + dy * dy) ** 0.5))

    mean = np.mean(all_dists)
    std = np.std(all_dists)
    return mean + 5 * std


def _get_cliques(
    nodes: pd.DataFrame, ego_width: None | int, window_size: float
) -> np.ndarray:
    G = nx.Graph()
    for t in nodes.itertuples():
        other_nodes = nodes.loc[nodes.index > t.Index, :]
        dx = _node_x_distance(t.center_x, other_nodes["center_x"], ego_width)
        dy = other_nodes["center_y"] - t.center_y
        mask = np.logical_and(np.abs(dx) < window_size, np.abs(dy) < window_size)

        neighbor_nodes = other_nodes[mask]

        for i in neighbor_nodes.index:
            G.add_edge(t.Index, i)

    cliques = list(nx.find_cliques(G))
    out = np.zeros((len(nodes), len(cliques)))
    for j, c in enumerate(cliques):
        for i in c:
            out[i, j] = 1

    return out


def _compute_edges(nodes, ego_width: None | int):
    # Compute the visibility window size
    window_size = _compute_covisibility_window_size(nodes, ego_width)

    # Find the cliques
    # The clique matrix is a nxc matrix where n is the number of nodes and c is
    # the number of cliques. A 1 in position i, j indicates that the ith node is
    # part of the jth clique.
    clique_matrix = _get_cliques(nodes, ego_width=ego_width, window_size=window_size)
    return clique_matrix


def _visualize_graph(image, nodes, cliques):

    plt.figure()

    plt.imshow(image)

    adj = cliques @ cliques.T

    for i, j in itertools.product(range(len(nodes)), repeat=2):
        if j <= i or adj[i, j] == 0:
            continue
        xs = [nodes.loc[i, "center_x"], nodes.loc[j, "center_x"]]
        ys = [nodes.loc[i, "center_y"], nodes.loc[j, "center_y"]]
        plt.plot(xs, ys, "b", alpha=0.5)

    sns.scatterplot(nodes, x="center_x", y="center_y", hue="class")


def _preprocess_overhead_cliques(overhead_cliques):
    # In the paper, they grow the cliques with the nodes of nearby cliques
    # if they share a large number of nodes in common. We skip this step for now
    return overhead_cliques


def _compute_class_graph(
    nodes: pd.DataFrame,
    cliques: np.ndarray,
    classes: list[str],
    compute_per_clique: bool,
):

    # The adjacency matrix for the graph from the clique matrix
    adj = np.minimum(cliques @ cliques.T, 1.0)

    # In the case of the ego perspective, we want to compute a single class matrix
    # using all of the nodes. For the overhead view, we want to compute a class graph
    # on a per clique basis.
    if not compute_per_clique:
        cliques = np.ones((cliques.shape[0], 1))

    num_nodes, num_cliques = cliques.shape
    num_classes = len(classes)

    out = np.zeros((num_cliques, num_classes, num_classes))

    num_edges = 0
    for i, j in itertools.product(range(num_nodes), repeat=2):
        if j <= i or adj[i, j] == 0:
            continue
        num_edges += 1
        edge_present_in_clique = np.logical_and(cliques[i, :], cliques[j, :])
        class_i = classes.index(nodes.loc[i, "class"])
        class_j = classes.index(nodes.loc[j, "class"])

        out[edge_present_in_clique, class_i, class_j] += 1
        if class_i != class_j:
            out[edge_present_in_clique, class_j, class_i] += 1
    out /= num_edges
    return out


def _match_graphs(overhead_nodes, overhead_cliques, ego_nodes, ego_cliques, classes):
    overhead_cliques = _preprocess_overhead_cliques(overhead_cliques)

    ego_class_graphs = _compute_class_graph(
        ego_nodes, ego_cliques, classes, compute_per_clique=False
    )
    overhead_class_graphs = _compute_class_graph(
        overhead_nodes, overhead_cliques, classes, compute_per_clique=True
    )

    # The observation likelihood P(Z | L) is the probability of making the ego
    # observation at a given location, where a location is represented as a
    # clique in the overhead graph.
    observation_likelihood_num = np.sum(
        ego_class_graphs * overhead_class_graphs, axis=(-1, -2)
    )
    perfect_ego_alignment = np.sum(ego_class_graphs * ego_class_graphs, axis=(-1, -2))
    perfect_overhead_alignment = np.sum(
        ego_class_graphs * ego_class_graphs, axis=(-1, -2)
    )
    observation_likelihood_den = np.sqrt(
        perfect_ego_alignment * perfect_overhead_alignment
    )
    observation_likelihood = observation_likelihood_num / observation_likelihood_den

    normalizer = np.sum(observation_likelihood)
    if normalizer == 0:
        return observation_likelihood

    return observation_likelihood / normalizer


def _visualize_matches(image, nodes, cliques, clique_probabilities):
    plt.figure()
    plt.imshow(image)

    clique_centroids = []
    for i, clique_mask in enumerate(cliques.T.astype(bool)):
        clique_centroids.append(
            nodes.loc[clique_mask, ["center_x", "center_y"]].mean().values
        )

    clique_centroids = np.stack(clique_centroids, axis=0)
    plt.scatter(clique_centroids[:, 0], clique_centroids[:, 1], c=clique_probabilities)
    plt.colorbar()


def estimate_overhead_transform(
        inputs: OverheadMatchingInput, model: gs.GroundingSam, debug=False
        ) -> Tuple[float, float] | str:
    classes = ["tree", "house"]
    ego_width = inputs.ego.shape[1]

    # Detect nodes in overhead
    overhead_nodes = _compute_nodes(inputs.overhead, model, classes, debug)

    if len(overhead_nodes) == 0:
        return 'No nodes detected in overhead'

    if len(overhead_nodes) == 1:
        return 'Cannot find cliques with a single node in overhead'

    # compute edge in overhead
    overhead_cliques = _compute_edges(overhead_nodes, ego_width=None)

    if overhead_cliques.shape[1] == 0:
        return f'Invalid overhead cliques shape: {overhead_cliques.shape}'

    if debug:
        _visualize_graph(inputs.overhead, overhead_nodes, overhead_cliques)

    # detect nodes in ego
    ego_nodes = _compute_ego_nodes(inputs.ego, model, classes, debug)

    if len(ego_nodes) == 0:
        return 'No nodes detected in ego'

    if len(ego_nodes) == 1:
        return 'Cannot find cliques with a single node in ego'

    # detect edges in ego
    ego_cliques = _compute_edges(ego_nodes, ego_width=ego_width)

    if ego_cliques.shape[1] == 0:
        return f'Invalid ego cliques shape: {ego_cliques.shape}'

    if debug:
        _visualize_graph(inputs.ego, ego_nodes, ego_cliques)

    # perform matching
    overhead_clique_probabilities = _match_graphs(
        overhead_nodes, overhead_cliques, ego_nodes, ego_cliques, classes
    )

    if debug:
        _visualize_matches(
            inputs.overhead,
            overhead_nodes,
            overhead_cliques,
            overhead_clique_probabilities,
        )

    argmax_idx = np.argmax(overhead_clique_probabilities)
    argmax_clique = overhead_cliques[:, argmax_idx].astype(bool)
    clique_center = overhead_nodes.loc[
        argmax_clique, ["center_x", "center_y"]].mean().values.tolist()

    return tuple(clique_center)
