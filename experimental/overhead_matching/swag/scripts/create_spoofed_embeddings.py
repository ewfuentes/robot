import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    #!/usr/bin/env python3
    """
    Create spoofed landmark embeddings based on correspondences.

    This script creates "perfect" embeddings where all landmarks in a correspondence
    group share the same embedding, enabling investigation of landmark uniqueness
    for patch identification.
    """

    import argparse
    import base64
    import collections
    import hashlib
    import json
    import pickle
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    from collections import defaultdict

    import common.torch.load_torch_deps  # Must be imported before torch
    import torch
    import experimental.overhead_matching.swag.model.semantic_landmark_utils as slu
    import experimental.overhead_matching.swag.data.vigor_dataset as vd


    return (
        Path,
        argparse,
        base64,
        collections,
        dataclass,
        defaultdict,
        hashlib,
        json,
        np,
        pickle,
        torch,
        tqdm,
        vd,
    )


@app.cell
def _(Path, argparse):
    parser = argparse.ArgumentParser(
        description="Create spoofed landmark embeddings based on correspondences"
    )
    parser.add_argument(
        "--correspondence_path",
        type=Path,
        default=Path("/data/overhead_matching/datasets/landmark_correspondence/v4_minimal_full/"),
        help="Path to correspondence data directory"
    )
    parser.add_argument(
        "--pano_embedding_path",
        type=Path,
        default=Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v1/"),
        help="Path to panorama embeddings directory"
    )
    parser.add_argument(
        "--osm_embedding_path",
        type=Path,
        default=Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/v3_no_addresses/"),
        help="Path to OSM embeddings directory"
    )
    parser.add_argument(
        "--vigor_dataset_path",
        type=Path,
        default=Path("/data/overhead_matching/datasets/VIGOR/Chicago/"),
        help="Path to VIGOR dataset (for landmark metadata)"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/spoofed_perfect_correspondence/"),
        help="Output path for spoofed embeddings"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=1536,
        help="Dimension of embeddings"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    for k in vars(args):
        print(k, vars(args)[k])
    return (args,)


@app.cell
def _(args, vd):
    dataset = vd.VigorDataset(config=vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images = False,
        should_load_landmarks= True,
        landmark_version= "v3",
    ), dataset_path=args.vigor_dataset_path)
    return (dataset,)


@app.cell
def _(base64, dataclass, hashlib, json):
    @dataclass
    class GroupStatistics:
        """Statistics about correspondence groups."""
        num_groups: int
        group_sizes: list[int]
        pano_only_groups: int
        osm_only_groups: int
        mixed_groups: int
        singleton_groups: int
        largest_group_size: int
        mean_group_size: float




    def custom_id_from_props(props: dict) -> str:
        """Generate custom_id for OSM landmarks (same as in semantic_landmark_extractor.py)."""
        json_props = json.dumps(dict(props), sort_keys=True)
        custom_id = base64.b64encode(
            hashlib.sha256(json_props.encode('utf-8')).digest()
        ).decode('utf-8')
        return custom_id


    def get_city_from_coordinates(lat: float, lon: float) -> str:
        """Determine city based on lat/lon coordinates."""
        # Chicago: ~41-42°N, Seattle: ~47-48°N
        if lat < 45.0:
            return "Chicago"
        else:
            return "Seattle"
    return (custom_id_from_props,)


@app.cell
def _(Path, json):

    def parse_correspondence_responses(response_file: Path) -> dict[str, dict]:
        """
        Parse correspondence response JSONL files.

        Returns:
            Dict mapping custom_id (panorama ID) -> correspondence data
        """
        print(f"Loading correspondences from {response_file}")
        with response_file.open() as f:
            correspondences = json.load(f)


        return correspondences

    return (parse_correspondence_responses,)


@app.cell
def _(Path, parse_correspondence_responses):

    # Use minimal_heading.json for all Chicago panoramas (25,478 panos)
    correspondences = parse_correspondence_responses(Path("/tmp/minimal_heading.json"))
    # correspondences = parse_correspondence_responses(Path("/tmp/medium_heading.json"))  # Only 1000 panos
    print(len(correspondences))
    return (correspondences,)


@app.cell
def _(dataset):
    dataset._landmark_metadata
    return


@app.cell
def _(dataset):
    len(dataset._landmark_metadata["pruned_props"].unique())
    return


@app.cell
def _():
    return


@app.cell
def _(dataset):
    all_props = {}
    for _props in dataset._landmark_metadata["pruned_props"].unique():
        for x in _props:
            tag = x[0]
            if tag not in all_props:
                all_props[tag] = set()
            all_props[tag] = all_props[tag].union({x[1]})
    # print(all_props)
    for _k,_v in sorted(all_props.items(), key=lambda x: len(x[1])):
        print(_k, len(_v),
              list(_v)[:min(len(_v), 5)])
    return


@app.cell
def _(dataset):
    keys_to_drop = [
        "fixme",
        "contact:facebook",
        "contact:youtube",
        "old_name",
        "source",
        "branch",
        "brand:wikidata",
        "addr:housenumber",
        "ref",
        "phone",
        "wikidata",
        "wikipedia",
        "height",
        "gnis:feature_id",
        "name:etymology:wikidata",
        "addr:street",
        "addr:unit",
        "turn:lanes",
        "email",
        "divvy:id",
        "start_date",
        "railway:track_ref",
        "operator:wikidata",
        "country",
        "capacity",
        "generator:solar:modules",
        "ref:nrhp",
        "maxheight",
        "destination:lanes",
    ]


    def trim_bad_props(props):
        out = []
        for k,v in props:
            if k not in keys_to_drop:
                out.append((k,v))
        return frozenset(out)

    dataset._landmark_metadata["new_pruned_props"] = dataset._landmark_metadata["pruned_props"].apply(trim_bad_props)
    # print(len(dataset._landmark_metadata["new_pruned_props"].unique()))
    len(dataset._landmark_metadata["new_pruned_props"].unique())
    return


@app.cell
def _(dataset):
    dataset._landmark_metadata["new_pruned_props"].value_counts()
    return


@app.cell
def _(correspondences, dataset):
    def get_nodes_from_correspondences(correspondences):
        panorama_nodes = []
        osm_nodes = dataset._landmark_metadata["id"].tolist()
        osm_index_map = {_k: _v for _v, _k in enumerate(osm_nodes)}
        assert len(osm_index_map) == len(osm_nodes)
        osm_edges = []
        # add edges between all OSM nodes that have the same "new_pruned_props"
        for name, group in dataset._landmark_metadata.groupby("new_pruned_props"):
            ids = [osm_index_map[x] for x in group["id"]]
            if len(ids) > 1:
                osm_edges.extend([(ids[0], x) for x in ids[1:]])

        edges = []
        for pano_id in correspondences.keys():
            pano_correspondences = correspondences[pano_id]
            pano_ids = [(pano_id, _idx, x) for _idx, x in enumerate(pano_correspondences["pano"])]
            zero_pano_index_number = len(panorama_nodes)
            panorama_nodes.extend(pano_ids)
            for match in pano_correspondences["matches"]["matches"]:
                pano_local_idx = match["set_1_id"] - 1 # zero indexed
                for set_2_idx in match["set_2_matches"]:
                    if set_2_idx - 1 >= len(pano_correspondences["osm"]):
                        continue
                    osm_landmark = pano_correspondences["osm"][set_2_idx - 1]
                    for osm_id in osm_landmark["ids"]:
                        str_id = str((osm_id[0], osm_id[1]))
                        edges.append((pano_local_idx + zero_pano_index_number, osm_index_map[str_id]))

        # combine nodes into single list
        split_idx = len(panorama_nodes)
        all_nodes = panorama_nodes + osm_nodes
        edges = [(x, y+split_idx) for x,y in edges]
        osm_edges = [(x+split_idx, y + split_idx) for x,y in osm_edges]
        return all_nodes, edges + osm_edges, split_idx


    nodes, edges, split_idx = get_nodes_from_correspondences(correspondences=correspondences)
    return edges, nodes, split_idx


@app.cell
def _(edges, nodes, split_idx):
    print(len(nodes), len(edges), split_idx, len(nodes)-split_idx)
    return


@app.cell
def _(correspondences):
    pano_id = "eICZd2TZWjHa1ibb55eD3g"
    # pano_id = next(iter(correspondences.keys()))
    correspondences[pano_id]
    return


@app.cell
def _(collections, edges, split_idx):
    # Calculate degree for each node
    degree_count = collections.defaultdict(int)
    for _e1, _e2 in edges:
        degree_count[_e1] += 1
        degree_count[_e2] += 1
    # Separate pano and OSM nodes
    pano_node_degrees = [(idx, deg) for idx, deg in degree_count.items() if idx < split_idx]
    osm_node_degrees = [(idx, deg) for idx, deg in degree_count.items() if idx >= split_idx]

    # Sort by degree (descending)
    pano_node_degrees.sort(key=lambda x: x[1], reverse=True)
    osm_node_degrees.sort(key=lambda x: x[1], reverse=True)

    return osm_node_degrees, pano_node_degrees


@app.cell
def _(np, osm_node_degrees, pano_node_degrees):
    # Degree statistics
    print("=== DEGREE STATISTICS ===")
    print(f"\nPanorama nodes:")
    print(f"  Total nodes with edges: {len(pano_node_degrees)}")
    if pano_node_degrees:
        _pano_degrees = [_d for _, _d in pano_node_degrees]
        print(f"  Mean degree: {np.mean(_pano_degrees):.2f}")
        print(f"  Median degree: {np.median(_pano_degrees):.2f}")
        print(f"  Max degree: {max(_pano_degrees)}")
        print(f"  90th percentile: {np.percentile(_pano_degrees, 90):.2f}")

    print(f"\nOSM nodes:")
    print(f"  Total nodes with edges: {len(osm_node_degrees)}")
    if osm_node_degrees:
        _osm_degrees = [_d for _, _d in osm_node_degrees]
        print(f"  Mean degree: {np.mean(_osm_degrees):.2f}")
        print(f"  Median degree: {np.median(_osm_degrees):.2f}")
        print(f"  Max degree: {max(_osm_degrees)}")
        print(f"  90th percentile: {np.percentile(_osm_degrees, 90):.2f}")
    return


@app.cell
def _(dataset, nodes, osm_node_degrees):
    # Top 20 highest degree OSM nodes
    print("\n=== TOP 20 HIGHEST DEGREE OSM LANDMARKS ===")
    for _i, (_node_idx, _degree) in enumerate(osm_node_degrees[:20]):
        _osm_id = nodes[_node_idx]
        print(f"{_i+1}. Node {_node_idx}: degree {_degree}, OSM ID: {_osm_id},\n OSM: {dataset._landmark_metadata[dataset._landmark_metadata.id == str(_osm_id)].pruned_props.values}")
    return


@app.cell
def _(nodes, pano_node_degrees):
    # Top 20 highest degree panorama nodes
    print("\n=== TOP 20 HIGHEST DEGREE PANORAMA LANDMARKS ===")
    for _i, (_node_idx, _degree) in enumerate(pano_node_degrees[:20]):
        _pano_data = nodes[_node_idx]
        print(f"{_i+1}. Node {_node_idx}: degree {_degree}")
        print(f"    Pano ID: {_pano_data[0]}")
        print(f"    Index: {_pano_data[1]}")
        print(f"    Description: {_pano_data[2]}")
    return


@app.cell
def _(edges):
    # Build adjacency lists for easy lookup
    adjacency_list = {}
    for _e1, _e2 in edges:
        if _e1 not in adjacency_list:
            adjacency_list[_e1] = []
        if _e2 not in adjacency_list:
            adjacency_list[_e2] = []
        adjacency_list[_e1].append(_e2)
        adjacency_list[_e2].append(_e1)
    return (adjacency_list,)


@app.cell
def _(adjacency_list, nodes, osm_node_degrees, split_idx):
    # Show what the highest degree OSM node is connected to
    if osm_node_degrees:
        _top_osm_node_idx, _top_osm_degree = osm_node_degrees[0]
        print(f"\n=== CONNECTIONS FOR HIGHEST DEGREE OSM NODE ===")
        print(f"Node {_top_osm_node_idx}: {nodes[_top_osm_node_idx]}")
        print(f"Degree: {_top_osm_degree}")
        print(f"\nConnected to {len(adjacency_list[_top_osm_node_idx])} nodes:")

        # Sample some connections
        _connections = adjacency_list[_top_osm_node_idx]
        print(f"\nFirst 50 connections:")
        for _i, _neighbor_idx in enumerate(_connections[:50]):
            if _neighbor_idx < split_idx:
                _pano_data = nodes[_neighbor_idx]
                print(f"  {_i+1}. PANO Node {_neighbor_idx}: {_pano_data[0]}, idx={_pano_data[1]}, desc={_pano_data[2]}")
            else:
                print(f"  {_i+1}. OSM Node {_neighbor_idx}: {nodes[_neighbor_idx]}")
    return


@app.function
# Function to trace edge back to correspondence
def trace_edge_to_correspondence(edge, nodes, correspondences, split_idx):
    """Trace an edge back to the correspondence it came from."""
    e1, e2 = edge

    # Determine which is pano and which is OSM
    if e1 < split_idx and e2 >= split_idx:
        pano_idx, osm_idx = e1, e2
    elif e2 < split_idx and e1 >= split_idx:
        pano_idx, osm_idx = e2, e1
    else:
        return None  # Both same type, shouldn't happen

    pano_data = nodes[pano_idx]
    osm_id = nodes[osm_idx]

    pano_id = pano_data[0]
    pano_landmark_idx = pano_data[1]

    # Find the correspondence
    if pano_id in correspondences:
        corr = correspondences[pano_id]

        # Find which OSM landmark in the correspondence matches
        for osm_landmark in corr["osm"]:
            for osm_id_tuple in osm_landmark["ids"]:
                if str((osm_id_tuple[0], osm_id_tuple[1])) == osm_id:
                    return {
                        "pano_id": pano_id,
                        "pano_landmark_idx": pano_landmark_idx,
                        "pano_description": pano_data[2],
                        "osm_id": osm_id,
                        "osm_tags": osm_landmark.get("tags", ""),
                    }

    return None


@app.cell
def _(correspondences, edges, nodes, osm_node_degrees, split_idx):
    # Trace edges from highest degree OSM node back to correspondences
    if osm_node_degrees:
        _top_osm_node_idx, _ = osm_node_degrees[0]
        print(f"\n=== TRACE EDGES FOR HIGHEST DEGREE OSM NODE ===")
        print(f"OSM Node {_top_osm_node_idx}: {nodes[_top_osm_node_idx]}")

        # Find all edges involving this node
        _relevant_edges = [(_e1, _e2) for _e1, _e2 in edges if _e1 == _top_osm_node_idx or _e2 == _top_osm_node_idx]

        print(f"\nTracing first 20 edges back to correspondences:")
        for _i, _edge in enumerate(_relevant_edges[:20]):
            _trace_info = trace_edge_to_correspondence(_edge, nodes, correspondences, split_idx)
            if _trace_info:
                print(f"\n{_i+1}. Edge: {_edge}")
                print(f"   Pano: {_trace_info['pano_id']}")
                print(f"   Pano landmark #{_trace_info['pano_landmark_idx']}: {_trace_info['pano_description']}")
                print(f"   OSM ID: {_trace_info['osm_id']}")
                print(f"   OSM tags: {_trace_info['osm_tags'][:100]}...")
    return


@app.cell
def _(correspondences, nodes, osm_node_degrees):
    # Lookup function to get OSM landmark details from correspondence
    def _get_osm_details(_osm_node_idx, _nodes, _correspondences):
        """Get full OSM landmark details including tags."""
        _osm_id = _nodes[_osm_node_idx]

        # Search through all correspondences to find this OSM landmark
        for _pano_id, _corr in _correspondences.items():
            for _osm_landmark in _corr["osm"]:
                for _osm_id_tuple in _osm_landmark["ids"]:
                    if str((_osm_id_tuple[0], _osm_id_tuple[1])) == _osm_id:
                        return {
                            "osm_id": _osm_id,
                            "tags": _osm_landmark.get("tags", ""),
                            "example_pano": _pano_id,
                        }
        return None

    # Show details for top OSM nodes
    print("\n=== DETAILS FOR TOP 10 HIGHEST DEGREE OSM NODES ===")
    for _i, (_node_idx, _degree) in enumerate(osm_node_degrees[:10]):
        _details = _get_osm_details(_node_idx, nodes, correspondences)
        if _details:
            print(f"\n{_i+1}. Node {_node_idx}, Degree {_degree}")
            print(f"   OSM ID: {_details['osm_id']}")
            print(f"   Tags: {_details['tags']}")
            print(f"   Example panorama: {_details['example_pano']}")
    return


@app.cell
def _(
    adjacency_list,
    dataset,
    edges,
    json,
    nodes,
    osm_node_degrees,
    pano_node_degrees,
    split_idx,
    tqdm,
):
    # Export graph data for visualization

    print("Exporting graph data for visualization...")

    # Pre-build lookup dictionaries for fast access (avoid repeated DataFrame filtering)
    print("Building lookup dictionaries...")
    _pano_lookup = {}
    for _, _row in tqdm(list(dataset._panorama_metadata.iterrows()), desc="Building pano lookup"):
        _pano_lookup[_row.pano_id] = {
            "lat": float(_row.lat),
            "lon": float(_row.lon)
        }

    _osm_lookup = {}
    for _, _row in tqdm(list(dataset._landmark_metadata.iterrows()), desc="Building OSM lookup"):
        # Handle both Point and Polygon geometries
        _geom = _row.geometry
        if _geom.geom_type == 'Point':
            _lat, _lon = _geom.y, _geom.x
        else:
            # Use centroid for Polygons, LineStrings, etc.
            _centroid = _geom.centroid
            _lat, _lon = _centroid.y, _centroid.x

        _osm_lookup[_row.id] = {
            "lat": float(_lat),
            "lon": float(_lon),
            "tags": str(_row.pruned_props) if hasattr(_row, 'pruned_props') else ""
        }

    print(f"Built lookups: {len(_pano_lookup)} panos, {len(_osm_lookup)} OSM landmarks")

    # Build node data with lat/lon using fast dictionary lookups
    _node_data = []
    for _idx, _node in tqdm(enumerate(nodes), total=len(nodes), desc="Processing nodes"):
        _node_dict = {
            "id": _idx,
            "degree": len(adjacency_list.get(_idx, [])),
        }

        if _idx < split_idx:
            # Panorama node
            _pano_id, _pano_idx, _desc = _node
            _node_dict.update({
                "type": "pano",
                "pano_id": _pano_id,
                "pano_index": _pano_idx,
                "description": _desc,
            })
            # Fast lookup
            if _pano_id in _pano_lookup:
                _node_dict.update(_pano_lookup[_pano_id])
        else:
            # OSM node
            _osm_id = _node
            _node_dict.update({
                "type": "osm",
                "osm_id": _osm_id,
            })
            # Fast lookup
            if _osm_id in _osm_lookup:
                _node_dict.update(_osm_lookup[_osm_id])

        _node_data.append(_node_dict)

    print(f"Processed all {len(_node_data)} nodes")

    # Build edge data (only include edges where both nodes have lat/lon)
    _edge_data = []
    for _e1, _e2 in edges:
        _edge_data.append({
            "source": int(_e1),
            "target": int(_e2),
        })

    # Build degree rankings
    _pano_degree_ranks = {int(_idx): _rank for _rank, (_idx, _deg) in enumerate(pano_node_degrees)}
    _osm_degree_ranks = {int(_idx): _rank for _rank, (_idx, _deg) in enumerate(osm_node_degrees)}

    # Export to JSON
    _export_data = {
        "nodes": _node_data,
        "edges": _edge_data,
        "split_idx": split_idx,
        "pano_degree_ranks": _pano_degree_ranks,
        "osm_degree_ranks": _osm_degree_ranks,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "pano_nodes": split_idx,
            "osm_nodes": len(nodes) - split_idx,
        }
    }

    with open("/tmp/graph_export.json", "w") as _f:
        json.dump(_export_data, _f)

    print(f"Exported {len(_node_data)} nodes and {len(_edge_data)} edges to /tmp/graph_export.json")
    print(f"File size: {len(json.dumps(_export_data)) / 1024 / 1024:.1f} MB")
    return


@app.cell
def _():
    # Instructions for viewing the visualization
    print("\n" + "="*60)
    print("GRAPH VISUALIZATION WEB APP")
    print("="*60)
    print("\nTo view the interactive visualization:")
    print("1. Run the following command in a terminal:")
    print("   cd /tmp && python3 -m http.server 8000")
    print("\n2. Open your browser to:")
    print("   http://localhost:8000/graph_viz/")
    print("\n3. Use the controls to filter by degree and explore!")
    print("="*60)
    return


@app.cell
def _(Path, pickle):
    # Load existing embeddings to use as source for fake embeddings
    print("Loading existing embeddings...")

    # Load OSM embeddings
    _osm_emb_path = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/v3_no_addresses/embeddings/embeddings.pkl")
    with open(_osm_emb_path, 'rb') as _f:
        osm_embeddings_tensor, osm_landmark_id_to_idx = pickle.load(_f)
    print(f"Loaded OSM embeddings: {osm_embeddings_tensor.shape}")

    # Load pano embeddings
    _pano_emb_path = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v1/Chicago/embeddings/embeddings.pkl")
    with open(_pano_emb_path, 'rb') as _f:
        pano_embeddings_tensor, pano_landmark_id_to_idx = pickle.load(_f)
    print(f"Loaded pano embeddings: {pano_embeddings_tensor.shape}")

    _embedding_dim = osm_embeddings_tensor.shape[1]
    print(f"Embedding dimension: {_embedding_dim}")

    return (
        osm_embeddings_tensor,
        osm_landmark_id_to_idx,
        pano_embeddings_tensor,
        pano_landmark_id_to_idx,
    )


@app.cell
def _(osm_landmark_id_to_idx):
    print(next(iter(osm_landmark_id_to_idx.items())))
    return


@app.cell
def _(dataset, np):
    # Create OSM landmark groups based on new_pruned_props
    # Each group maps to one randomly selected member's embedding
    print("\nCreating OSM landmark groups...")

    osm_groups_by_props = {}
    osm_group_to_osm_id = {}  # Maps group -> selected OSM ID from that group

    # Group OSM landmarks by new_pruned_props
    for _name, _group in dataset._landmark_metadata.groupby("new_pruned_props"):
        _osm_ids = _group["id"].tolist()
        osm_groups_by_props[_name] = _osm_ids
        # Randomly select one OSM member from this group
        _selected_osm_id = _osm_ids[np.random.randint(0, len(_osm_ids))]
        osm_group_to_osm_id[_name] = _selected_osm_id

    print(f"Created {len(osm_groups_by_props)} OSM groups based on new_pruned_props")

    # Create mapping from OSM ID to group
    osm_id_to_group = {}
    for _group_props, _osm_ids in osm_groups_by_props.items():
        for _osm_id in _osm_ids:
            osm_id_to_group[_osm_id] = _group_props

    # Count singleton OSM groups
    osm_singleton_groups = sum(1 for _osm_ids in osm_groups_by_props.values() if len(_osm_ids) == 1)
    print(f"OSM groups that are singletons: {osm_singleton_groups} / {len(osm_groups_by_props)} ({osm_singleton_groups / len(osm_groups_by_props) * 100:.1f}%)")

    # Show max group size
    _max_group_size = max(len(_osm_ids) for _osm_ids in osm_groups_by_props.values())
    print(f"Max OSM group size: {_max_group_size}")

    return (
        osm_group_to_osm_id,
        osm_groups_by_props,
        osm_id_to_group,
        osm_singleton_groups,
    )


@app.cell
def _(adjacency_list, nodes, np, osm_id_to_group, split_idx):
    # For each pano landmark, find which OSM groups it connects to
    print("\nProcessing pano landmarks...")

    pano_to_osm_groups = {}  # Maps pano node index -> set of OSM group props
    pano_tiebreaks = 0  # Count of pano landmarks connecting to multiple groups

    for _pano_idx in range(split_idx):
        # Get all neighbors (should be OSM nodes)
        _neighbors = adjacency_list.get(_pano_idx, [])

        # Filter to only OSM neighbors
        _osm_neighbors = [_n for _n in _neighbors if _n >= split_idx]
        if len(_neighbors) != len(_osm_neighbors):
            raise RuntimeError(f"{_neighbors}, {_osm_neighbors}")

        if not _osm_neighbors:
            continue

        # Get the OSM IDs and their groups
        _osm_groups_connected = set()
        for _osm_idx in _osm_neighbors:
            _osm_id = nodes[_osm_idx]
            if _osm_id in osm_id_to_group:
                _group_props = osm_id_to_group[_osm_id]
                _osm_groups_connected.add(_group_props)

        pano_to_osm_groups[_pano_idx] = _osm_groups_connected

        if len(_osm_groups_connected) > 1:
            pano_tiebreaks += 1

    print(f"Pano landmarks with edges: {len(pano_to_osm_groups)}")
    print(f"Pano landmarks requiring tiebreak (connected to multiple OSM groups): {pano_tiebreaks} ({pano_tiebreaks / max(len(pano_to_osm_groups), 1) * 100:.1f}%)")

    # Select random group for each pano landmark
    pano_to_selected_group = {}
    for _pano_idx, _group_set in pano_to_osm_groups.items():
        if _group_set:
            # Randomly pick one group
            _selected_group = list(_group_set)[np.random.randint(0, len(_group_set))]
            pano_to_selected_group[_pano_idx] = _selected_group

    return pano_tiebreaks, pano_to_osm_groups, pano_to_selected_group


@app.cell
def _(pano_to_selected_group):
    print(len(pano_to_selected_group))
    next(iter(pano_to_selected_group.items()))
    return


@app.cell
def _(osm_id_to_group):
    print(len(osm_id_to_group))
    next(iter(osm_id_to_group.items())) # new pruned props, to 
    return


@app.cell
def _(nodes, osm_id_to_group, pano_to_selected_group, split_idx):
    # Test: Verify association logic with a few examples
    print("\n=== TESTING ASSOCIATION LOGIC ===")

    # Test a pano node
    _test_pano_idx = 0
    if _test_pano_idx in pano_to_selected_group:
        _pano_node = nodes[_test_pano_idx]
        _selected_group = pano_to_selected_group[_test_pano_idx]
        print(f"\nTest Pano Node {_test_pano_idx}:")
        print(f"  Pano ID: {_pano_node[0]}")
        print(f"  Description: {_pano_node[2][:100]}...")
        print(f"  Selected group (first 3 props): {list(_selected_group)[:3]}")

    # Test an OSM node
    _test_osm_idx = split_idx
    _osm_id = nodes[_test_osm_idx]
    if _osm_id in osm_id_to_group:
        _osm_group = osm_id_to_group[_osm_id]
        print(f"\nTest OSM Node {_test_osm_idx}:")
        print(f"  OSM ID: {_osm_id}")
        print(f"  Group (first 3 props): {list(_osm_group)[:3]}")

    print("\n=== TEST COMPLETE ===")
    return


@app.cell
def _(torch):
    def generate_normal_vector(dim):
        # _v = torch.rand(dim)-0.5
        # return (_v / _v.norm(dim=0))
        return torch.rand(dim)
    generate_normal_vector(10)
    return (generate_normal_vector,)


@app.cell
def _(
    custom_id_from_props,
    dataset,
    generate_normal_vector,
    osm_embeddings_tensor,
    osm_group_to_osm_id,
    osm_landmark_id_to_idx,
    tqdm,
):

    # Create spoofed embedding tensors and mappings
    print("\n=== CREATING SPOOFED EMBEDDINGS ===")


    # Pre-create embeddings for each OSM group (both semantic and random)
    print("\nPre-creating embeddings for OSM groups...")
    group_to_semantic_emb = {}  # Group -> semantic embedding from a member
    group_to_random_emb = {}    # Group -> random embedding

    for _group_props, _selected_osm_id in tqdm(osm_group_to_osm_id.items(), desc="Creating group embeddings"):
        # Semantic: Use embedding from selected member
        _selected_custom_id = custom_id_from_props(
            dataset._landmark_metadata[dataset._landmark_metadata["id"] == _selected_osm_id].iloc[0]["pruned_props"]
        )
        _emb_idx = osm_landmark_id_to_idx[_selected_custom_id]
        group_to_semantic_emb[_group_props] = osm_embeddings_tensor[_emb_idx]

        # Random: Generate random embedding for this group
        group_to_random_emb[_group_props] = generate_normal_vector(1536)
    return group_to_random_emb, group_to_semantic_emb


@app.cell
def _(
    defaultdict,
    group_to_random_emb,
    group_to_semantic_emb,
    osm_embeddings_tensor,
    tqdm,
):
    random_emb_dict = defaultdict(set)
    semantic_emb_dict = defaultdict(set)
    direct_emb_dict = defaultdict(set)
    for _k, _v in tqdm(group_to_random_emb.items()):
        _key = tuple(_v.tolist())
        random_emb_dict[_key].add(_k)
    for _k, _v in tqdm(group_to_semantic_emb.items()):
        _key = tuple(_v.tolist())
        semantic_emb_dict[_key].add(_k)
    for _v in tqdm(osm_embeddings_tensor):
        _key = tuple(_v.tolist())
        direct_emb_dict[_key].add(1)
    print(len(random_emb_dict))
    print(len(semantic_emb_dict))
    print(len(direct_emb_dict))
    return


@app.cell
def _(
    custom_id_from_props,
    dataset,
    group_to_random_emb,
    group_to_semantic_emb,
    osm_id_to_group,
    torch,
    tqdm,
):

    # Process OSM landmarks - build both semantic and random versions
    print("\nProcessing OSM landmarks...")
    spoofed_osm_embeddings = []
    spoofed_osm_id_to_idx = {}  # custom_id to index of embedding vector in tensor
    random_osm_embeddings = []
    random_osm_id_to_idx = {}

    # First pass: Build mapping from custom_id to group_props
    # (handles case where same custom_id might belong to multiple groups)
    print("  Building custom_id to group mapping...")
    custom_id_to_group_props = {} # custom ID -> new_pruned_props
    group_conflicts = 0

    for _idx, _row in tqdm(list(dataset._landmark_metadata.iterrows()), desc="  Mapping custom_ids"):
        _osm_id = _row["id"]
        _pruned_props = _row["pruned_props"]
        _custom_id = custom_id_from_props(_pruned_props)

        if _osm_id not in osm_id_to_group:
            raise RuntimeError(f"{_osm_id} not found in osm_id_to_groups")

        _group_props = osm_id_to_group[_osm_id]

        if _custom_id in custom_id_to_group_props:
            # Verify they're in the same group
            if custom_id_to_group_props[_custom_id] != _group_props:
                group_conflicts += 1
                if group_conflicts <= 5:
                    print(f"  Warning: custom_id {_custom_id[:20]}... appears in multiple groups!")
        else:
            custom_id_to_group_props[_custom_id] = _group_props

    if group_conflicts > 0:
        print(f"  Found {group_conflicts} custom_ids appearing in multiple groups")


    # Second pass: Process each unique custom_id once
    print(f"  Processing {len(custom_id_to_group_props)} unique custom_ids...")
    for _custom_id, _group_props in tqdm(custom_id_to_group_props.items(), desc="  Creating embeddings"):
        # Semantic version: Use group's semantic embedding
        spoofed_osm_id_to_idx[_custom_id] = len(spoofed_osm_embeddings)
        spoofed_osm_embeddings.append(group_to_semantic_emb[_group_props])

        # Random version: Use group's random embedding
        random_osm_id_to_idx[_custom_id] = len(random_osm_embeddings)
        random_osm_embeddings.append(group_to_random_emb[_group_props])

    spoofed_osm_embeddings_tensor = torch.stack(spoofed_osm_embeddings)
    random_osm_embeddings_tensor = torch.stack(random_osm_embeddings)
    print(f"Created {len(spoofed_osm_embeddings)} OSM embeddings (semantic and random)")
    print(f"  Unique custom_ids: {len(spoofed_osm_id_to_idx)}")
    return (
        random_osm_embeddings_tensor,
        random_osm_id_to_idx,
        spoofed_osm_embeddings_tensor,
        spoofed_osm_id_to_idx,
    )


@app.cell
def _(
    dataset,
    generate_normal_vector,
    group_to_random_emb,
    group_to_semantic_emb,
    nodes,
    pano_embeddings_tensor,
    pano_landmark_id_to_idx,
    pano_to_selected_group,
    split_idx,
    torch,
    tqdm,
):

    # Build pano lookup for lat/lon
    _pano_lookup = {}
    for _, _row in dataset._panorama_metadata.iterrows():
        _pano_lookup[_row.pano_id] = {
            "lat": float(_row.lat),
            "lon": float(_row.lon)
        }
    # Process pano landmarks - build both semantic and random versions
    print("\nProcessing pano landmarks...")
    spoofed_pano_embeddings = []
    spoofed_pano_id_to_idx = {}
    random_pano_embeddings = []
    random_pano_id_to_idx = {}
    pano_landmarks_with_edges = 0
    pano_landmarks_without_edges = 0

    for _pano_idx in tqdm(range(split_idx), desc="Pano landmarks"):
        _pano_node = nodes[_pano_idx]
        _pano_id, _landmark_idx, _description = _pano_node

        # Get lat/lon for this pano
        if _pano_id not in _pano_lookup:
            print(f"Warning: Pano {_pano_id} not in metadata")
            continue
        _lat = _pano_lookup[_pano_id]["lat"]
        _lon = _pano_lookup[_pano_id]["lon"]
        _custom_id = f"{_pano_id},{_lat:.6f},{_lon:.6f},__landmark_{_landmark_idx}"

        # Check if this pano has edges to OSM
        if _pano_idx in pano_to_selected_group:
            # Has edges - use group's embedding
            _selected_group = pano_to_selected_group[_pano_idx]
            _semantic_emb = group_to_semantic_emb[_selected_group]
            _random_emb = group_to_random_emb[_selected_group]
            pano_landmarks_with_edges += 1
        else:
            if _custom_id not in pano_landmark_id_to_idx:
                raise RuntimeError(f"Original pano embedding not found for {_custom_id}")
            _emb_idx = pano_landmark_id_to_idx[_custom_id]
            _original_emb = pano_embeddings_tensor[_emb_idx]
            _semantic_emb = _original_emb
            _random_emb = generate_normal_vector(1536)
            pano_landmarks_without_edges += 1

        # Add to both versions
        spoofed_pano_id_to_idx[_custom_id] = len(spoofed_pano_embeddings)
        spoofed_pano_embeddings.append(_semantic_emb)

        random_pano_id_to_idx[_custom_id] = len(random_pano_embeddings)
        random_pano_embeddings.append(_random_emb)

    spoofed_pano_embeddings_tensor = torch.stack(spoofed_pano_embeddings)
    random_pano_embeddings_tensor = torch.stack(random_pano_embeddings)
    print(f"Created {len(spoofed_pano_embeddings)} pano embeddings (semantic and random)")
    print(f"Pano landmarks with OSM edges: {pano_landmarks_with_edges}")
    print(f"Pano landmarks without OSM edges (keep original): {pano_landmarks_without_edges}")

    return (
        pano_landmarks_without_edges,
        random_pano_embeddings_tensor,
        random_pano_id_to_idx,
        spoofed_pano_embeddings_tensor,
        spoofed_pano_id_to_idx,
    )


@app.cell
def _(
    defaultdict,
    generate_normal_vector,
    spoofed_osm_embeddings_tensor,
    spoofed_pano_embeddings_tensor,
    torch,
    tqdm,
):
    # create "semantic" embeddings, void of semantics
    all_spoofed_embeddings = torch.cat([spoofed_pano_embeddings_tensor, spoofed_osm_embeddings_tensor], dim=0)
    no_semantic_pano_embeddings_tensor = torch.ones_like(spoofed_pano_embeddings_tensor) * torch.nan
    no_semantic_osm_embeddings_tensor = torch.ones_like(spoofed_osm_embeddings_tensor) * torch.nan
    _hash = defaultdict(set)
    for _i in tqdm(range(all_spoofed_embeddings.shape[0])):
        _vector = all_spoofed_embeddings[_i]
        _key = tuple(_vector.tolist())
        _hash[_key].add(_i)

    for _key, _indexes in _hash.items():
        _new_vector = generate_normal_vector(1536)
        _sorted_indexes = sorted(list(_indexes))
        _pano_idxs= [x for x in _sorted_indexes if x < no_semantic_pano_embeddings_tensor.shape[0]]
        _osm_idxs = [x - no_semantic_pano_embeddings_tensor.shape[0] for x in _sorted_indexes if x >= no_semantic_pano_embeddings_tensor.shape[0]]
        no_semantic_pano_embeddings_tensor[_pano_idxs] = _new_vector
        no_semantic_osm_embeddings_tensor[_osm_idxs] = _new_vector

    assert not torch.any(torch.isnan(no_semantic_osm_embeddings_tensor))
    assert not torch.any(torch.isnan(no_semantic_pano_embeddings_tensor))


    return (
        no_semantic_osm_embeddings_tensor,
        no_semantic_pano_embeddings_tensor,
    )


@app.cell
def _(spoofed_pano_id_to_idx):
    next(iter(spoofed_pano_id_to_idx.items()))
    return


@app.cell
def _(random_pano_id_to_idx):
    next(iter(random_pano_id_to_idx.items()))
    return


@app.cell
def _(random_pano_id_to_idx, spoofed_pano_id_to_idx):
    spoofed_pano_id_to_idx == random_pano_id_to_idx
    return


@app.cell
def _(
    Path,
    no_semantic_osm_embeddings_tensor,
    no_semantic_pano_embeddings_tensor,
    pickle,
    random_osm_embeddings_tensor,
    random_osm_id_to_idx,
    random_pano_embeddings_tensor,
    random_pano_id_to_idx,
    spoofed_osm_embeddings_tensor,
    spoofed_osm_id_to_idx,
    spoofed_pano_embeddings_tensor,
    spoofed_pano_id_to_idx,
):
     # Save spoofed embeddings to pickle files
    print("\n=== SAVING SPOOFED EMBEDDINGS ===")

    # Base directory
    base_dir = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/")

    # Create output directories for real embeddings version
    osm_real_dir = base_dir / "sat_spoof/embeddings"
    pano_real_dir = base_dir / "pano_spoof/Chicago/embeddings"
    osm_real_dir.mkdir(parents=True, exist_ok=True)
    pano_real_dir.mkdir(parents=True, exist_ok=True)

    # Save OSM embeddings (real version)
    osm_real_path = osm_real_dir / "embeddings.pkl"
    with open(osm_real_path, 'wb') as f:
        pickle.dump((spoofed_osm_embeddings_tensor, spoofed_osm_id_to_idx), f)
    print(f"Saved OSM embeddings (real) to {osm_real_path}")
    print(f"  Tensor shape: {spoofed_osm_embeddings_tensor.shape}")
    print(f"  Number of custom IDs: {len(spoofed_osm_id_to_idx)}")

    # Save pano embeddings (real version)
    pano_real_path = pano_real_dir / "embeddings.pkl"
    with open(pano_real_path, 'wb') as f:
        pickle.dump((spoofed_pano_embeddings_tensor, spoofed_pano_id_to_idx), f)
    print(f"Saved pano embeddings (real) to {pano_real_path}")
    print(f"  Tensor shape: {spoofed_pano_embeddings_tensor.shape}")
    print(f"  Number of custom IDs: {len(spoofed_pano_id_to_idx)}")

    # Create output directories for random U(0,1) version
    osm_random_dir = base_dir / "sat_spoof_random/embeddings"
    pano_random_dir = base_dir / "pano_spoof_random/Chicago/embeddings"
    osm_random_dir.mkdir(parents=True, exist_ok=True)
    pano_random_dir.mkdir(parents=True, exist_ok=True)

    # Save OSM embeddings (random version)
    osm_random_path = osm_random_dir / "embeddings.pkl"
    with open(osm_random_path, 'wb') as f:
        pickle.dump((random_osm_embeddings_tensor, random_osm_id_to_idx), f)
    print(f"\nSaved OSM embeddings (random) to {osm_random_path}")
    print(f"  Tensor shape: {random_osm_embeddings_tensor.shape}")
    print(f"  Number of custom IDs: {len(random_osm_id_to_idx)}")

    # Save pano embeddings (random version)
    pano_random_path = pano_random_dir / "embeddings.pkl"
    with open(pano_random_path, 'wb') as f:
        pickle.dump((random_pano_embeddings_tensor, random_pano_id_to_idx), f)
    print(f"Saved pano embeddings (random) to {pano_random_path}")
    print(f"  Tensor shape: {random_pano_embeddings_tensor.shape}")
    print(f"  Number of custom IDs: {len(random_pano_id_to_idx)}")

    # Create output directories for no semantic version
    osm_no_semantic_dir = base_dir / "sat_spoof_no_semantic/embeddings"
    pano_no_semantic_dir = base_dir / "pano_spoof_no_semantic/Chicago/embeddings"
    osm_no_semantic_dir.mkdir(parents=True, exist_ok=True)
    pano_no_semantic_dir.mkdir(parents=True, exist_ok=True)

    # Save OSM embeddings (no_semantic version)
    osm_no_semantic_path = osm_no_semantic_dir / "embeddings.pkl"
    with open(osm_no_semantic_path, 'wb') as f:
        pickle.dump((no_semantic_osm_embeddings_tensor, spoofed_osm_id_to_idx), f)
    print(f"\nSaved OSM embeddings (no_semantic) to {osm_no_semantic_path}")
    print(f"  Tensor shape: {no_semantic_osm_embeddings_tensor.shape}")
    print(f"  Number of custom IDs: {len(spoofed_osm_id_to_idx)}")

    # Save pano embeddings (no_semantic version)
    pano_no_semantic_path = pano_no_semantic_dir / "embeddings.pkl"
    with open(pano_no_semantic_path, 'wb') as f:
        pickle.dump((no_semantic_pano_embeddings_tensor, spoofed_pano_id_to_idx), f)
    print(f"Saved pano embeddings (no_semantic) to {pano_no_semantic_path}")
    print(f"  Tensor shape: {no_semantic_pano_embeddings_tensor.shape}")
    print(f"  Number of custom IDs: {len(spoofed_pano_id_to_idx)}")

    print("\n✓ All embeddings saved successfully!")
    return


@app.cell
def _(
    osm_groups_by_props,
    osm_singleton_groups,
    pano_landmarks_with_embeddings,
    pano_landmarks_without_edges,
    pano_tiebreaks,
    pano_to_osm_groups,
    split_idx,
):
    # Print final statistics
    print("\n" + "="*60)
    print("SPOOFED EMBEDDINGS STATISTICS")
    print("="*60)

    print("\n--- OSM Landmarks ---")
    print(f"Total OSM groups (by new_pruned_props): {len(osm_groups_by_props)}")
    print(f"OSM singleton groups: {osm_singleton_groups} ({osm_singleton_groups / len(osm_groups_by_props) * 100:.1f}%)")

    # Group size distribution
    _group_sizes = [len(ids) for ids in osm_groups_by_props.values()]
    print(f"OSM group sizes - min: {min(_group_sizes)}, max: {max(_group_sizes)}, mean: {sum(_group_sizes) / len(_group_sizes):.1f}")

    print("\n--- Pano Landmarks ---")
    print(f"Total pano landmarks: {split_idx}")
    print(f"Pano landmarks with OSM edges: {pano_landmarks_with_embeddings} ({pano_landmarks_with_embeddings / split_idx * 100:.1f}%)")
    print(f"Pano landmarks without OSM edges (singletons): {pano_landmarks_without_edges} ({pano_landmarks_without_edges / split_idx * 100:.1f}%)")
    print(f"Pano landmarks requiring tiebreak: {pano_tiebreaks} ({pano_tiebreaks / max(len(pano_to_osm_groups), 1) * 100:.1f}%)")

    # Distribution of number of groups per pano
    _groups_per_pano = [len(groups) for groups in pano_to_osm_groups.values()]
    if _groups_per_pano:
        print(f"OSM groups per pano - min: {min(_groups_per_pano)}, max: {max(_groups_per_pano)}, mean: {sum(_groups_per_pano) / len(_groups_per_pano):.1f}")

    print("="*60)
    return


if __name__ == "__main__":
    app.run()
