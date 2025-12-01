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
    import marimo as mo
    from tqdm import tqdm
    from collections import defaultdict

    import common.torch.load_torch_deps  # Must be imported before torch
    import torch
    import experimental.overhead_matching.swag.model.semantic_landmark_utils as slu
    import experimental.overhead_matching.swag.data.vigor_dataset as vd


    return (
        Path,
        collections,
        defaultdict,
        json,
        mo,
        np,
        pickle,
        slu,
        torch,
        tqdm,
        vd,
    )


@app.cell
def _(Path, vd):
    dataset = vd.VigorDataset(config=vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images = False,
        should_load_landmarks= True,
        landmark_version= "v3",
    ), dataset_path=Path("/data/overhead_matching/datasets/VIGOR/Chicago/"))
    return (dataset,)


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
    print(len(correspondences))
    return (correspondences,)


@app.cell
def _(dataset):
    len(dataset._landmark_metadata["pruned_props"].unique())
    return


@app.cell
def _(dataset):
    # Print low instance count tag keys
    _all_props = {}
    for _props in dataset._landmark_metadata["pruned_props"].unique():
        for _x in _props:
            _tag = _x[0]
            if _tag not in _all_props:
                _all_props[_tag] = set()
            _all_props[_tag] = _all_props[_tag].union({_x[1]})
    # print(all_props)
    for _k,_v in sorted(_all_props.items(), key=lambda x: len(x[1])):
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
    len(dataset._landmark_metadata["new_pruned_props"].unique())
    return (trim_bad_props,)


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
    # pano_id = "eICZd2TZWjHa1ibb55eD3g"
    pano_id = next(iter(correspondences.keys()))
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


@app.cell(disabled=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Create Spoofed Embeddings From Correspondences""")
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

    return osm_group_to_osm_id, osm_groups_by_props, osm_id_to_group


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

    return (pano_to_selected_group,)


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
        _v = torch.rand(dim)-0.5
        return (_v / _v.norm(dim=0))
    generate_normal_vector(10)
    return (generate_normal_vector,)


@app.cell
def _(
    dataset,
    generate_normal_vector,
    osm_embeddings_tensor,
    osm_group_to_osm_id,
    osm_landmark_id_to_idx,
    slu,
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
        _selected_custom_id = slu.custom_id_from_props(
            dataset._landmark_metadata[dataset._landmark_metadata["id"] == _selected_osm_id].iloc[0]["pruned_props"]
        )
        _emb_idx = osm_landmark_id_to_idx[_selected_custom_id]
        group_to_semantic_emb[_group_props] = osm_embeddings_tensor[_emb_idx]

        # Random: Generate random embedding for this group
        group_to_random_emb[_group_props] = generate_normal_vector(1536)
    return group_to_random_emb, group_to_semantic_emb


@app.cell
def _(defaultdict, group_to_random_emb, group_to_semantic_emb, tqdm):
    random_emb_dict = defaultdict(set)
    semantic_emb_dict = defaultdict(set)
    for _k, _v in tqdm(group_to_random_emb.items()):
        _key = tuple(_v.tolist())
        random_emb_dict[_key].add(_k)
    for _k, _v in tqdm(group_to_semantic_emb.items()):
        _key = tuple(_v.tolist())
        semantic_emb_dict[_key].add(_k)
    print(len(random_emb_dict))
    print(len(semantic_emb_dict))
    return


@app.cell
def _(
    dataset,
    group_to_random_emb,
    group_to_semantic_emb,
    osm_id_to_group,
    slu,
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
        _custom_id = slu.custom_id_from_props(_pruned_props)

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

    assert spoofed_pano_id_to_idx == random_pano_id_to_idx
    spoofed_pano_embeddings_tensor = torch.stack(spoofed_pano_embeddings)
    random_pano_embeddings_tensor = torch.stack(random_pano_embeddings)
    print(f"Created {len(spoofed_pano_embeddings)} pano embeddings (semantic and random)")
    print(f"Pano landmarks with OSM edges: {pano_landmarks_with_edges}")
    print(f"Pano landmarks without OSM edges (keep original): {pano_landmarks_without_edges}")

    return (
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
def _(Path, json, random_pano_id_to_idx, spoofed_pano_id_to_idx):
    # Generate panorama_metadata.jsonl for Chicago training embeddings
    print("\n=== GENERATING CHICAGO TRAINING PANO METADATA ===")

    _base_dir = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/")

    # Versions to generate metadata for
    _chicago_training_versions = [
        ("pano_spoof", spoofed_pano_id_to_idx),
        ("pano_spoof_random", random_pano_id_to_idx),
        ("pano_spoof_no_semantic", spoofed_pano_id_to_idx),  # Uses same id_to_idx as semantic
    ]

    for _version_name, _id_to_idx in _chicago_training_versions:
        print(f"\nGenerating metadata for {_version_name}/Chicago...")

        # Parse custom_ids and create metadata entries
        _metadata_entries = []
        for _custom_id in _id_to_idx.keys():
            # Parse custom_id format: "pano_id,lat,lon,__landmark_N"
            _parts = _custom_id.split(',')
            if len(_parts) != 4:
                print(f"Warning: Skipping malformed custom_id: {_custom_id}")
                continue

            _pano_id = _parts[0]
            _lat = _parts[1]
            _lon = _parts[2]
            _landmark_part = _parts[3]

            # Extract landmark_idx from "__landmark_N"
            if not _landmark_part.startswith('__landmark_'):
                print(f"Warning: Unexpected landmark format in {_custom_id}")
                continue
            _landmark_idx = int(_landmark_part.split('_')[-1])

            # Create metadata entry
            _entry = {
                "panorama_id": f"{_pano_id},{_lat},{_lon},",
                "landmark_idx": _landmark_idx,
                "custom_id": _custom_id,
                "yaw_angles": [0.0, 0.0, 0.0, 0.0]  # Default: present at all angles
            }
            _metadata_entries.append(_entry)

        # Sort by panorama_id then landmark_idx
        _metadata_entries.sort(key=lambda x: (x["panorama_id"], x["landmark_idx"]))

        # Write to file
        _output_dir = _base_dir / _version_name / "Chicago" / "embedding_requests"
        _output_dir.mkdir(parents=True, exist_ok=True)
        _output_file = _output_dir / "panorama_metadata.jsonl"

        with open(_output_file, 'w') as _f:
            for _entry in _metadata_entries:
                _f.write(json.dumps(_entry) + '\n')

        print(f"  ✓ Wrote {len(_metadata_entries)} entries to {_output_file}")

    print("\n✓ Chicago training pano metadata generated successfully!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Create Seattle Validation Set Embeddings""")
    return


@app.cell
def _(Path, vd):
    # Load Seattle dataset
    print("Loading Seattle dataset...")
    seattle_dataset = vd.VigorDataset(config=vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images = False,
        should_load_landmarks= True,
        landmark_version= "v4_202001",
        factor=0.3 
    ), dataset_path=Path("/data/overhead_matching/datasets/VIGOR/Seattle/"))
    print(f"Loaded {len(seattle_dataset._landmark_metadata)} Seattle OSM landmarks")
    print(f"Loaded {len(seattle_dataset._panorama_metadata)} Seattle panoramas")
    return (seattle_dataset,)


@app.cell
def _(Path, parse_correspondence_responses):
    # Load Seattle correspondences
    print("\nLoading Seattle correspondences...")
    seattle_correspondences = parse_correspondence_responses(Path("/tmp/minimal_historical_seattle.json"))
    print(f"Loaded correspondences for {len(seattle_correspondences)} Seattle panoramas")
    return (seattle_correspondences,)


@app.cell
def _(Path, pickle):
    # Load Seattle pano embeddings
    print("\nLoading Seattle pano embeddings...")
    _seattle_pano_emb_path = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v1/Seattle_validation/embeddings/embeddings.pkl")
    with open(_seattle_pano_emb_path, 'rb') as _f:
        seattle_pano_embeddings_tensor, seattle_pano_landmark_id_to_idx = pickle.load(_f)
    print(f"Loaded Seattle pano embeddings: {seattle_pano_embeddings_tensor.shape}")

    return seattle_pano_embeddings_tensor, seattle_pano_landmark_id_to_idx


@app.cell
def _(Path, pickle):
    # Load Seattle OSM embeddings
    print("\nLoading Seattle OSM embeddings...")
    _seattle_osm_emb_path = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/v4_202001_no_addresses/embeddings/embeddings.pkl")
    with open(_seattle_osm_emb_path, 'rb') as _f:
        seattle_osm_embeddings_tensor, seattle_osm_landmark_id_to_idx = pickle.load(_f)
    print(f"Loaded Seattle OSM embeddings: {seattle_osm_embeddings_tensor.shape}")

    return seattle_osm_embeddings_tensor, seattle_osm_landmark_id_to_idx


@app.cell
def _(seattle_dataset, trim_bad_props):
    # Apply same trimming to Seattle OSM landmarks
    print("\nProcessing Seattle OSM landmarks...")
    seattle_dataset._landmark_metadata["new_pruned_props"] = seattle_dataset._landmark_metadata["pruned_props"].apply(trim_bad_props)
    print(f"Seattle unique new_pruned_props groups: {len(seattle_dataset._landmark_metadata['new_pruned_props'].unique())}")
    return


@app.cell
def _(osm_groups_by_props, seattle_dataset):
    # Match Seattle OSM landmarks to Chicago groups or create Seattle-only groups
    print("\nMatching Seattle OSM to Chicago groups...")

    seattle_osm_groups_by_props = {}  # Seattle OSM groups by new_pruned_props
    seattle_osm_id_to_group = {}  # Maps Seattle OSM ID -> group props

    chicago_group_props = set(osm_groups_by_props.keys())
    matched_to_chicago = 0
    seattle_only_groups = 0

    # Group Seattle OSM by new_pruned_props
    for _name, _group in seattle_dataset._landmark_metadata.groupby("new_pruned_props"):
        _seattle_osm_ids = _group["id"].tolist()
        seattle_osm_groups_by_props[_name] = _seattle_osm_ids

        # Check if this matches a Chicago group
        if _name in chicago_group_props:
            matched_to_chicago += len(_seattle_osm_ids)
        else:
            seattle_only_groups += 1

        # Map each Seattle OSM ID to its group
        for _osm_id in _seattle_osm_ids:
            seattle_osm_id_to_group[_osm_id] = _name

    print(f"Created {len(seattle_osm_groups_by_props)} Seattle OSM groups")
    print(f"  Matched to existing Chicago groups: {matched_to_chicago} landmarks")
    print(f"  Seattle-only groups: {seattle_only_groups} groups")

    # Identify which groups are Seattle-only (not in Chicago)
    seattle_only_group_props = set(seattle_osm_groups_by_props.keys()) - chicago_group_props
    print(f"  Total Seattle-only group props: {len(seattle_only_group_props)}")

    return (
        seattle_only_group_props,
        seattle_osm_groups_by_props,
        seattle_osm_id_to_group,
    )


@app.cell
def _(seattle_correspondences, seattle_dataset):
    # Build Seattle correspondence graph (similar to Chicago version)
    print("\nBuilding Seattle correspondence graph...")

    def get_seattle_nodes_from_correspondences(correspondences, dataset):
        panorama_nodes = []
        osm_nodes = dataset._landmark_metadata["id"].tolist()
        osm_index_map = {_k: _v for _v, _k in enumerate(osm_nodes)}
        assert len(osm_index_map) == len(osm_nodes)

        osm_edges = []
        # Add edges between all Seattle OSM nodes that have the same "new_pruned_props"
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

        # Combine nodes into single list
        split_idx = len(panorama_nodes)
        all_nodes = panorama_nodes + osm_nodes
        edges = [(x, y+split_idx) for x,y in edges]
        osm_edges = [(x+split_idx, y + split_idx) for x,y in osm_edges]
        return all_nodes, edges + osm_edges, split_idx

    seattle_nodes, seattle_edges, seattle_split_idx = get_seattle_nodes_from_correspondences(
        correspondences=seattle_correspondences,
        dataset=seattle_dataset
    )

    print(f"Seattle graph: {len(seattle_nodes)} nodes, {len(seattle_edges)} edges")
    print(f"  Pano nodes: {seattle_split_idx}")
    print(f"  OSM nodes: {len(seattle_nodes) - seattle_split_idx}")

    return seattle_edges, seattle_nodes, seattle_split_idx


@app.cell
def _(
    np,
    seattle_edges,
    seattle_nodes,
    seattle_osm_id_to_group,
    seattle_split_idx,
):
    # Build adjacency list for Seattle graph
    seattle_adjacency_list = {}
    for _e1, _e2 in seattle_edges:
        if _e1 not in seattle_adjacency_list:
            seattle_adjacency_list[_e1] = []
        if _e2 not in seattle_adjacency_list:
            seattle_adjacency_list[_e2] = []
        seattle_adjacency_list[_e1].append(_e2)
        seattle_adjacency_list[_e2].append(_e1)

    # Associate Seattle pano landmarks with groups
    print("\nAssociating Seattle pano landmarks with groups...")
    seattle_pano_to_osm_groups = {}  # Maps pano node index -> set of OSM group props
    seattle_pano_tiebreaks = 0  # Count of pano landmarks connecting to multiple groups

    for _pano_idx in range(seattle_split_idx):
        # Get all neighbors (should be OSM nodes)
        _neighbors = seattle_adjacency_list.get(_pano_idx, [])

        # Filter to only OSM neighbors
        _osm_neighbors = [_n for _n in _neighbors if _n >= seattle_split_idx]
        if len(_neighbors) != len(_osm_neighbors) and len(_neighbors) > 0:
            raise RuntimeError(f"Pano node has non-OSM neighbors: {_neighbors}, {_osm_neighbors}")

        if not _osm_neighbors:
            continue

        # Get the OSM IDs and their groups
        _osm_groups_connected = set()
        for _osm_idx in _osm_neighbors:
            _osm_id = seattle_nodes[_osm_idx]
            if _osm_id in seattle_osm_id_to_group:
                _group_props = seattle_osm_id_to_group[_osm_id]
                _osm_groups_connected.add(_group_props)

        seattle_pano_to_osm_groups[_pano_idx] = _osm_groups_connected

        if len(_osm_groups_connected) > 1:
            seattle_pano_tiebreaks += 1

    print(f"Seattle pano landmarks with edges: {len(seattle_pano_to_osm_groups)}")
    print(f"Seattle pano landmarks requiring tiebreak: {seattle_pano_tiebreaks} ({seattle_pano_tiebreaks / max(len(seattle_pano_to_osm_groups), 1) * 100:.1f}%)")

    # Select random group for each Seattle pano landmark
    seattle_pano_to_selected_group = {}
    for _pano_idx, _group_set in seattle_pano_to_osm_groups.items():
        if _group_set:
            # Randomly pick one group
            _selected_group = list(_group_set)[np.random.randint(0, len(_group_set))]
            seattle_pano_to_selected_group[_pano_idx] = _selected_group

    return (seattle_pano_to_selected_group,)


@app.cell
def _(
    generate_normal_vector,
    np,
    seattle_dataset,
    seattle_only_group_props,
    seattle_osm_embeddings_tensor,
    seattle_osm_groups_by_props,
    seattle_osm_landmark_id_to_idx,
    slu,
    tqdm,
):
    # Generate embeddings for Seattle-only groups
    print("\nGenerating embeddings for Seattle-only groups...")

    seattle_group_to_semantic_emb = {}
    seattle_group_to_random_emb = {}

    for _group_props in tqdm(seattle_only_group_props, desc="Seattle-only groups"):
        _seattle_osm_ids = seattle_osm_groups_by_props[_group_props]

        # Randomly select one Seattle OSM member from this group
        _selected_osm_id = _seattle_osm_ids[np.random.randint(0, len(_seattle_osm_ids))]

        # Get the original embedding for semantic version (from Seattle OSM embeddings)
        _selected_custom_id = slu.custom_id_from_props(
            seattle_dataset._landmark_metadata[seattle_dataset._landmark_metadata["id"] == _selected_osm_id].iloc[0]["pruned_props"]
        )
        _emb_idx = seattle_osm_landmark_id_to_idx[_selected_custom_id]
        seattle_group_to_semantic_emb[_group_props] = seattle_osm_embeddings_tensor[_emb_idx]

        # Generate random embedding
        seattle_group_to_random_emb[_group_props] = generate_normal_vector(1536)

    print(f"Created embeddings for {len(seattle_only_group_props)} Seattle-only groups")

    return seattle_group_to_random_emb, seattle_group_to_semantic_emb


@app.cell
def _(
    dataset,
    group_to_random_emb,
    group_to_semantic_emb,
    osm_id_to_group,
    seattle_dataset,
    seattle_group_to_random_emb,
    seattle_group_to_semantic_emb,
    seattle_osm_id_to_group,
    slu,
    torch,
    tqdm,
):
    # Create combined OSM embeddings (Chicago + Seattle)
    print("\n=== CREATING COMBINED OSM EMBEDDINGS ===")

    combined_osm_embeddings_semantic = []
    combined_osm_id_to_idx_semantic = {}
    combined_osm_embeddings_random = []
    combined_osm_id_to_idx_random = {}

    # First: Process all Chicago OSM landmarks
    print("Processing Chicago OSM landmarks...")
    chicago_custom_id_to_group = {}
    chicago_group_conflicts = 0

    for _idx, _row in tqdm(list(dataset._landmark_metadata.iterrows()), desc="  Chicago OSM"):
        _osm_id = _row["id"]
        _pruned_props = _row["pruned_props"]
        _custom_id = slu.custom_id_from_props(_pruned_props)

        if _osm_id not in osm_id_to_group:
            raise RuntimeError(f"Chicago {_osm_id} not found in osm_id_to_group")

        _group_props = osm_id_to_group[_osm_id]

        if _custom_id in chicago_custom_id_to_group:
            if chicago_custom_id_to_group[_custom_id] != _group_props:
                chicago_group_conflicts += 1
        else:
            chicago_custom_id_to_group[_custom_id] = _group_props

    if chicago_group_conflicts > 0:
        print(f"  Found {chicago_group_conflicts} Chicago custom_ids in multiple groups")

    # Add Chicago embeddings
    for _custom_id, _group_props in tqdm(chicago_custom_id_to_group.items(), desc="  Adding Chicago embeddings"):
        # Semantic version
        combined_osm_id_to_idx_semantic[_custom_id] = len(combined_osm_embeddings_semantic)
        combined_osm_embeddings_semantic.append(group_to_semantic_emb[_group_props])

        # Random version
        combined_osm_id_to_idx_random[_custom_id] = len(combined_osm_embeddings_random)
        combined_osm_embeddings_random.append(group_to_random_emb[_group_props])

    print(f"Added {len(chicago_custom_id_to_group)} Chicago OSM embeddings")

    # Second: Process all Seattle OSM landmarks
    print("\nProcessing Seattle OSM landmarks...")
    seattle_custom_id_to_group = {}
    seattle_group_conflicts = 0

    for _idx, _row in tqdm(list(seattle_dataset._landmark_metadata.iterrows()), desc="  Seattle OSM"):
        _osm_id = _row["id"]
        _pruned_props = _row["pruned_props"]
        _custom_id = slu.custom_id_from_props(_pruned_props)

        if _osm_id not in seattle_osm_id_to_group:
            raise RuntimeError(f"Seattle {_osm_id} not found in seattle_osm_id_to_group")

        _group_props = seattle_osm_id_to_group[_osm_id]

        if _custom_id in seattle_custom_id_to_group:
            if seattle_custom_id_to_group[_custom_id] != _group_props:
                seattle_group_conflicts += 1
        else:
            seattle_custom_id_to_group[_custom_id] = _group_props

    if seattle_group_conflicts > 0:
        print(f"  Found {seattle_group_conflicts} Seattle custom_ids in multiple groups")

    # Add Seattle embeddings
    for _custom_id, _group_props in tqdm(seattle_custom_id_to_group.items(), desc="  Adding Seattle embeddings"):
        # Check if this group is Chicago or Seattle-only
        if _group_props in group_to_semantic_emb:
            # Chicago group - use Chicago embeddings
            _semantic_emb = group_to_semantic_emb[_group_props]
            _random_emb = group_to_random_emb[_group_props]
        else:
            # Seattle-only group
            _semantic_emb = seattle_group_to_semantic_emb[_group_props]
            _random_emb = seattle_group_to_random_emb[_group_props]

        # Semantic version
        combined_osm_id_to_idx_semantic[_custom_id] = len(combined_osm_embeddings_semantic)
        combined_osm_embeddings_semantic.append(_semantic_emb)

        # Random version
        combined_osm_id_to_idx_random[_custom_id] = len(combined_osm_embeddings_random)
        combined_osm_embeddings_random.append(_random_emb)

    print(f"Added {len(seattle_custom_id_to_group)} Seattle OSM embeddings")

    # Convert to tensors
    combined_osm_embeddings_tensor_semantic = torch.stack(combined_osm_embeddings_semantic)
    combined_osm_embeddings_tensor_random = torch.stack(combined_osm_embeddings_random)

    print(f"\nCombined OSM embeddings:")
    print(f"  Semantic: {combined_osm_embeddings_tensor_semantic.shape}")
    print(f"  Random: {combined_osm_embeddings_tensor_random.shape}")
    print(f"  Total unique custom_ids: {len(combined_osm_id_to_idx_semantic)}")

    return (
        combined_osm_embeddings_tensor_random,
        combined_osm_embeddings_tensor_semantic,
        combined_osm_id_to_idx_random,
        combined_osm_id_to_idx_semantic,
    )


@app.cell
def _(
    generate_normal_vector,
    group_to_random_emb,
    group_to_semantic_emb,
    seattle_dataset,
    seattle_group_to_random_emb,
    seattle_group_to_semantic_emb,
    seattle_nodes,
    seattle_pano_embeddings_tensor,
    seattle_pano_landmark_id_to_idx,
    seattle_pano_to_selected_group,
    seattle_split_idx,
    torch,
    tqdm,
):
    # Create Seattle pano embeddings
    print("\n=== CREATING SEATTLE PANO EMBEDDINGS ===")

    # Build Seattle pano lookup for lat/lon
    _seattle_pano_lookup = {}
    for _, _row in seattle_dataset._panorama_metadata.iterrows():
        _seattle_pano_lookup[_row.pano_id] = {
            "lat": float(_row.lat),
            "lon": float(_row.lon)
        }

    seattle_pano_embeddings_semantic = []
    seattle_pano_id_to_idx_semantic = {}
    seattle_pano_embeddings_random = []
    seattle_pano_id_to_idx_random = {}
    seattle_pano_with_edges = 0
    seattle_pano_without_edges = 0

    for _pano_idx in tqdm(range(seattle_split_idx), desc="Seattle pano landmarks"):
        _pano_node = seattle_nodes[_pano_idx]
        _pano_id, _landmark_idx, _description = _pano_node

        # Get lat/lon for this pano
        if _pano_id not in _seattle_pano_lookup:
            print(f"Warning: Seattle pano {_pano_id} not in metadata")
            continue
        _lat = _seattle_pano_lookup[_pano_id]["lat"]
        _lon = _seattle_pano_lookup[_pano_id]["lon"]
        _custom_id = f"{_pano_id},{_lat:.6f},{_lon:.6f},__landmark_{_landmark_idx}"

        # Check if this pano has edges to OSM groups
        if _pano_idx in seattle_pano_to_selected_group:
            # Has edges - use group's embedding
            _selected_group = seattle_pano_to_selected_group[_pano_idx]

            # Check if Chicago or Seattle-only group
            if _selected_group in group_to_semantic_emb:
                # Chicago group
                _semantic_emb = group_to_semantic_emb[_selected_group]
                _random_emb = group_to_random_emb[_selected_group]
            else:
                # Seattle-only group
                _semantic_emb = seattle_group_to_semantic_emb[_selected_group]
                _random_emb = seattle_group_to_random_emb[_selected_group]

            seattle_pano_with_edges += 1
        else:
            # No edges - use original or generate random
            if _custom_id not in seattle_pano_landmark_id_to_idx:
                print(f"Warning: Seattle pano embedding not found for {_custom_id}")
                continue
            _emb_idx = seattle_pano_landmark_id_to_idx[_custom_id]
            _original_emb = seattle_pano_embeddings_tensor[_emb_idx]
            _semantic_emb = _original_emb
            _random_emb = generate_normal_vector(1536)
            seattle_pano_without_edges += 1

        # Add to both versions
        seattle_pano_id_to_idx_semantic[_custom_id] = len(seattle_pano_embeddings_semantic)
        seattle_pano_embeddings_semantic.append(_semantic_emb)

        seattle_pano_id_to_idx_random[_custom_id] = len(seattle_pano_embeddings_random)
        seattle_pano_embeddings_random.append(_random_emb)

    assert seattle_pano_id_to_idx_semantic == seattle_pano_id_to_idx_random

    seattle_pano_embeddings_tensor_semantic = torch.stack(seattle_pano_embeddings_semantic)
    seattle_pano_embeddings_tensor_random = torch.stack(seattle_pano_embeddings_random)

    print(f"Created {len(seattle_pano_embeddings_semantic)} Seattle pano embeddings")
    print(f"  With OSM edges: {seattle_pano_with_edges}")
    print(f"  Without OSM edges: {seattle_pano_without_edges}")

    return (
        seattle_pano_embeddings_tensor_random,
        seattle_pano_embeddings_tensor_semantic,
        seattle_pano_id_to_idx_random,
        seattle_pano_id_to_idx_semantic,
    )


@app.cell
def _(
    combined_osm_embeddings_tensor_semantic,
    defaultdict,
    generate_normal_vector,
    seattle_pano_embeddings_tensor_semantic,
    spoofed_pano_embeddings_tensor,
    torch,
    tqdm,
):
    # Create "no_semantic" versions for validation set
    print("\n=== CREATING NO_SEMANTIC VERSIONS ===")

    # Combine all spoofed embeddings (Chicago pano + combined OSM + Seattle pano)
    all_chicago_pano = spoofed_pano_embeddings_tensor
    all_combined_osm = combined_osm_embeddings_tensor_semantic
    all_seattle_pano = seattle_pano_embeddings_tensor_semantic

    # Hash all unique embeddings across all three sources
    _hash = defaultdict(set)

    print("Hashing Chicago pano embeddings...")
    for _i in tqdm(range(all_chicago_pano.shape[0])):
        _vector = all_chicago_pano[_i]
        _key = tuple(_vector.tolist())
        _hash[_key].add(('chicago_pano', _i))

    print("Hashing combined OSM embeddings...")
    for _i in tqdm(range(all_combined_osm.shape[0])):
        _vector = all_combined_osm[_i]
        _key = tuple(_vector.tolist())
        _hash[_key].add(('combined_osm', _i))

    print("Hashing Seattle pano embeddings...")
    for _i in tqdm(range(all_seattle_pano.shape[0])):
        _vector = all_seattle_pano[_i]
        _key = tuple(_vector.tolist())
        _hash[_key].add(('seattle_pano', _i))

    print(f"Found {len(_hash)} unique embedding groups")

    # Create no_semantic versions
    combined_osm_no_semantic = torch.ones_like(combined_osm_embeddings_tensor_semantic) * torch.nan
    chicago_pano_no_semantic = torch.ones_like(spoofed_pano_embeddings_tensor) * torch.nan
    seattle_pano_no_semantic = torch.ones_like(seattle_pano_embeddings_tensor_semantic) * torch.nan

    print("Assigning new random embeddings to each group...")
    for _key, _indexes in tqdm(_hash.items(), desc="Assigning embeddings"):
        _new_vector = generate_normal_vector(1536)

        for _source, _idx in _indexes:
            if _source == 'chicago_pano':
                chicago_pano_no_semantic[_idx] = _new_vector
            elif _source == 'combined_osm':
                combined_osm_no_semantic[_idx] = _new_vector
            elif _source == 'seattle_pano':
                seattle_pano_no_semantic[_idx] = _new_vector

    # Verify no NaNs
    assert not torch.any(torch.isnan(combined_osm_no_semantic))
    assert not torch.any(torch.isnan(chicago_pano_no_semantic))
    assert not torch.any(torch.isnan(seattle_pano_no_semantic))

    print("✓ No_semantic versions created successfully")

    return (
        chicago_pano_no_semantic,
        combined_osm_no_semantic,
        seattle_pano_no_semantic,
    )


@app.cell
def _(
    Path,
    chicago_pano_no_semantic,
    combined_osm_embeddings_tensor_random,
    combined_osm_embeddings_tensor_semantic,
    combined_osm_id_to_idx_random,
    combined_osm_id_to_idx_semantic,
    combined_osm_no_semantic,
    pickle,
    random_pano_embeddings_tensor,
    random_pano_id_to_idx,
    seattle_pano_embeddings_tensor_random,
    seattle_pano_embeddings_tensor_semantic,
    seattle_pano_id_to_idx_random,
    seattle_pano_id_to_idx_semantic,
    seattle_pano_no_semantic,
    spoofed_pano_embeddings_tensor,
    spoofed_pano_id_to_idx,
):
    # Save validation embeddings
    print("\n=== SAVING VALIDATION EMBEDDINGS ===")

    _val_base_dir = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/")

    # ========== SEMANTIC VERSION ==========
    print("\n1. Saving semantic version...")

    # OSM (combined Chicago + Seattle)
    _val_osm_semantic_dir = _val_base_dir / "sat_spoof_validation/embeddings"
    _val_osm_semantic_dir.mkdir(parents=True, exist_ok=True)
    _val_osm_semantic_path = _val_osm_semantic_dir / "embeddings.pkl"
    with open(_val_osm_semantic_path, 'wb') as _val_f:
        pickle.dump((combined_osm_embeddings_tensor_semantic, combined_osm_id_to_idx_semantic), _val_f)
    print(f"  ✓ OSM semantic: {_val_osm_semantic_path}")
    print(f"    Shape: {combined_osm_embeddings_tensor_semantic.shape}")

    # Chicago Pano
    _val_chicago_pano_semantic_dir = _val_base_dir / "pano_spoof_validation/Chicago/embeddings"
    _val_chicago_pano_semantic_dir.mkdir(parents=True, exist_ok=True)
    _val_chicago_pano_semantic_path = _val_chicago_pano_semantic_dir / "embeddings.pkl"
    with open(_val_chicago_pano_semantic_path, 'wb') as _val_f:
        pickle.dump((spoofed_pano_embeddings_tensor, spoofed_pano_id_to_idx), _val_f)
    print(f"  ✓ Chicago pano semantic: {_val_chicago_pano_semantic_path}")
    print(f"    Shape: {spoofed_pano_embeddings_tensor.shape}")

    # Seattle Pano
    _val_seattle_pano_semantic_dir = _val_base_dir / "pano_spoof_validation/Seattle/embeddings"
    _val_seattle_pano_semantic_dir.mkdir(parents=True, exist_ok=True)
    _val_seattle_pano_semantic_path = _val_seattle_pano_semantic_dir / "embeddings.pkl"
    with open(_val_seattle_pano_semantic_path, 'wb') as _val_f:
        pickle.dump((seattle_pano_embeddings_tensor_semantic, seattle_pano_id_to_idx_semantic), _val_f)
    print(f"  ✓ Seattle pano semantic: {_val_seattle_pano_semantic_path}")
    print(f"    Shape: {seattle_pano_embeddings_tensor_semantic.shape}")

    # ========== RANDOM VERSION ==========
    print("\n2. Saving random version...")

    # OSM (combined Chicago + Seattle)
    _val_osm_random_dir = _val_base_dir / "sat_spoof_random_validation/embeddings"
    _val_osm_random_dir.mkdir(parents=True, exist_ok=True)
    _val_osm_random_path = _val_osm_random_dir / "embeddings.pkl"
    with open(_val_osm_random_path, 'wb') as _val_f:
        pickle.dump((combined_osm_embeddings_tensor_random, combined_osm_id_to_idx_random), _val_f)
    print(f"  ✓ OSM random: {_val_osm_random_path}")
    print(f"    Shape: {combined_osm_embeddings_tensor_random.shape}")

    # Chicago Pano
    _val_chicago_pano_random_dir = _val_base_dir / "pano_spoof_random_validation/Chicago/embeddings"
    _val_chicago_pano_random_dir.mkdir(parents=True, exist_ok=True)
    _val_chicago_pano_random_path = _val_chicago_pano_random_dir / "embeddings.pkl"
    with open(_val_chicago_pano_random_path, 'wb') as _val_f:
        pickle.dump((random_pano_embeddings_tensor, random_pano_id_to_idx), _val_f)
    print(f"  ✓ Chicago pano random: {_val_chicago_pano_random_path}")
    print(f"    Shape: {random_pano_embeddings_tensor.shape}")

    # Seattle Pano
    _val_seattle_pano_random_dir = _val_base_dir / "pano_spoof_random_validation/Seattle/embeddings"
    _val_seattle_pano_random_dir.mkdir(parents=True, exist_ok=True)
    _val_seattle_pano_random_path = _val_seattle_pano_random_dir / "embeddings.pkl"
    with open(_val_seattle_pano_random_path, 'wb') as _val_f:
        pickle.dump((seattle_pano_embeddings_tensor_random, seattle_pano_id_to_idx_random), _val_f)
    print(f"  ✓ Seattle pano random: {_val_seattle_pano_random_path}")
    print(f"    Shape: {seattle_pano_embeddings_tensor_random.shape}")

    # ========== NO_SEMANTIC VERSION ==========
    print("\n3. Saving no_semantic version...")

    # OSM (combined Chicago + Seattle)
    _val_osm_no_semantic_dir = _val_base_dir / "sat_spoof_no_semantic_validation/embeddings"
    _val_osm_no_semantic_dir.mkdir(parents=True, exist_ok=True)
    _val_osm_no_semantic_path = _val_osm_no_semantic_dir / "embeddings.pkl"
    with open(_val_osm_no_semantic_path, 'wb') as _val_f:
        pickle.dump((combined_osm_no_semantic, combined_osm_id_to_idx_semantic), _val_f)
    print(f"  ✓ OSM no_semantic: {_val_osm_no_semantic_path}")
    print(f"    Shape: {combined_osm_no_semantic.shape}")

    # Chicago Pano
    _val_chicago_pano_no_semantic_dir = _val_base_dir / "pano_spoof_no_semantic_validation/Chicago/embeddings"
    _val_chicago_pano_no_semantic_dir.mkdir(parents=True, exist_ok=True)
    _val_chicago_pano_no_semantic_path = _val_chicago_pano_no_semantic_dir / "embeddings.pkl"
    with open(_val_chicago_pano_no_semantic_path, 'wb') as _val_f:
        pickle.dump((chicago_pano_no_semantic, spoofed_pano_id_to_idx), _val_f)
    print(f"  ✓ Chicago pano no_semantic: {_val_chicago_pano_no_semantic_path}")
    print(f"    Shape: {chicago_pano_no_semantic.shape}")

    # Seattle Pano
    _val_seattle_pano_no_semantic_dir = _val_base_dir / "pano_spoof_no_semantic_validation/Seattle/embeddings"
    _val_seattle_pano_no_semantic_dir.mkdir(parents=True, exist_ok=True)
    _val_seattle_pano_no_semantic_path = _val_seattle_pano_no_semantic_dir / "embeddings.pkl"
    with open(_val_seattle_pano_no_semantic_path, 'wb') as _val_f:
        pickle.dump((seattle_pano_no_semantic, seattle_pano_id_to_idx_semantic), _val_f)
    print(f"  ✓ Seattle pano no_semantic: {_val_seattle_pano_no_semantic_path}")
    print(f"    Shape: {seattle_pano_no_semantic.shape}")

    print("\n" + "="*60)
    print("✓ ALL VALIDATION EMBEDDINGS SAVED SUCCESSFULLY!")
    print("="*60)
    print("\nSummary:")
    print(f"  - Combined OSM embeddings (Chicago + Seattle): {len(combined_osm_id_to_idx_semantic)} unique IDs")
    print(f"  - Chicago pano embeddings: {len(spoofed_pano_id_to_idx)} unique IDs")
    print(f"  - Seattle pano embeddings: {len(seattle_pano_id_to_idx_semantic)} unique IDs")
    print(f"\nThree versions created: semantic, random, no_semantic")
    print(f"Base directory: {_val_base_dir}")

    return


@app.cell
def _(
    Path,
    json,
    random_pano_id_to_idx,
    seattle_pano_id_to_idx_random,
    seattle_pano_id_to_idx_semantic,
    spoofed_pano_id_to_idx,
):
    # Generate panorama_metadata.jsonl for validation embeddings
    print("\n=== GENERATING VALIDATION PANO METADATA ===")

    _val_meta_base_dir = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/")

    # Validation versions with their Chicago and Seattle id_to_idx mappings
    _validation_versions = [
        ("pano_spoof_validation", spoofed_pano_id_to_idx, seattle_pano_id_to_idx_semantic),
        ("pano_spoof_random_validation", random_pano_id_to_idx, seattle_pano_id_to_idx_random),
        ("pano_spoof_no_semantic_validation", spoofed_pano_id_to_idx, seattle_pano_id_to_idx_semantic),  # Uses same as semantic
    ]

    for _version_name, _chicago_id_to_idx, _seattle_id_to_idx in _validation_versions:
        print(f"\nGenerating metadata for {_version_name}...")

        # Process Chicago
        print(f"  Processing Chicago...")
        _chicago_metadata_entries = []
        for _custom_id in _chicago_id_to_idx.keys():
            # Parse custom_id format: "pano_id,lat,lon,__landmark_N"
            _parts = _custom_id.split(',')
            if len(_parts) != 4:
                continue

            _pano_id = _parts[0]
            _lat = _parts[1]
            _lon = _parts[2]
            _landmark_part = _parts[3]

            if not _landmark_part.startswith('__landmark_'):
                continue
            _landmark_idx = int(_landmark_part.split('_')[-1])

            _entry = {
                "panorama_id": f"{_pano_id},{_lat},{_lon},",
                "landmark_idx": _landmark_idx,
                "custom_id": _custom_id,
                "yaw_angles": [0.0, 0.0, 0.0, 0.0]
            }
            _chicago_metadata_entries.append(_entry)

        _chicago_metadata_entries.sort(key=lambda x: (x["panorama_id"], x["landmark_idx"]))

        _chicago_output_dir = _val_meta_base_dir / _version_name / "Chicago" / "embedding_requests"
        _chicago_output_dir.mkdir(parents=True, exist_ok=True)
        _chicago_output_file = _chicago_output_dir / "panorama_metadata.jsonl"

        with open(_chicago_output_file, 'w') as _f:
            for _entry in _chicago_metadata_entries:
                _f.write(json.dumps(_entry) + '\n')

        print(f"    ✓ Chicago: {len(_chicago_metadata_entries)} entries")

        # Process Seattle
        print(f"  Processing Seattle...")
        _seattle_metadata_entries = []
        for _custom_id in _seattle_id_to_idx.keys():
            _parts = _custom_id.split(',')
            if len(_parts) != 4:
                continue

            _pano_id = _parts[0]
            _lat = _parts[1]
            _lon = _parts[2]
            _landmark_part = _parts[3]

            if not _landmark_part.startswith('__landmark_'):
                continue
            _landmark_idx = int(_landmark_part.split('_')[-1])

            _entry = {
                "panorama_id": f"{_pano_id},{_lat},{_lon},",
                "landmark_idx": _landmark_idx,
                "custom_id": _custom_id,
                "yaw_angles": [0.0, 0.0, 0.0, 0.0]
            }
            _seattle_metadata_entries.append(_entry)

        _seattle_metadata_entries.sort(key=lambda x: (x["panorama_id"], x["landmark_idx"]))

        _seattle_output_dir = _val_meta_base_dir / _version_name / "Seattle" / "embedding_requests"
        _seattle_output_dir.mkdir(parents=True, exist_ok=True)
        _seattle_output_file = _seattle_output_dir / "panorama_metadata.jsonl"

        with open(_seattle_output_file, 'w') as _f:
            for _entry in _seattle_metadata_entries:
                _f.write(json.dumps(_entry) + '\n')

        print(f"    ✓ Seattle: {len(_seattle_metadata_entries)} entries")

    print("\n✓ Validation pano metadata generated successfully!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Create Independent Seattle Embeddings""")
    return


@app.cell
def _(np, seattle_dataset):
    # Create independent Seattle OSM groups (not matched to Chicago)
    print("\n=== CREATING INDEPENDENT SEATTLE GROUPS ===")

    independent_seattle_osm_groups_by_props = {}
    independent_seattle_osm_id_to_group = {}
    independent_seattle_group_to_osm_id = {}

    # Group Seattle OSM by new_pruned_props independently
    for _name, _group in seattle_dataset._landmark_metadata.groupby("new_pruned_props"):
        _osm_ids = _group["id"].tolist()
        independent_seattle_osm_groups_by_props[_name] = _osm_ids

        # Randomly select one OSM member from this group
        _selected_osm_id = _osm_ids[np.random.randint(0, len(_osm_ids))]
        independent_seattle_group_to_osm_id[_name] = _selected_osm_id

        # Map each OSM ID to its group
        for _osm_id in _osm_ids:
            independent_seattle_osm_id_to_group[_osm_id] = _name

    print(f"Created {len(independent_seattle_osm_groups_by_props)} independent Seattle OSM groups")

    # Count singleton groups
    _singleton_groups = sum(1 for _osm_ids in independent_seattle_osm_groups_by_props.values() if len(_osm_ids) == 1)
    print(f"Singleton groups: {_singleton_groups} / {len(independent_seattle_osm_groups_by_props)} ({_singleton_groups / len(independent_seattle_osm_groups_by_props) * 100:.1f}%)")

    # Show max group size
    _max_group_size = max(len(_osm_ids) for _osm_ids in independent_seattle_osm_groups_by_props.values())
    print(f"Max group size: {_max_group_size}")

    return (
        independent_seattle_group_to_osm_id,
        independent_seattle_osm_groups_by_props,
        independent_seattle_osm_id_to_group,
    )


@app.cell
def _(
    independent_seattle_osm_id_to_group,
    np,
    seattle_adjacency_list,
    seattle_nodes,
    seattle_split_idx,
):
    # Associate independent Seattle pano landmarks with independent Seattle groups
    print("\nAssociating independent Seattle pano with groups...")

    independent_seattle_pano_to_osm_groups = {}
    independent_seattle_pano_tiebreaks = 0

    for _pano_idx in range(seattle_split_idx):
        # Get all neighbors (should be OSM nodes)
        _neighbors = seattle_adjacency_list.get(_pano_idx, [])

        # Filter to only OSM neighbors
        _osm_neighbors = [_n for _n in _neighbors if _n >= seattle_split_idx]
        if len(_neighbors) != len(_osm_neighbors) and len(_neighbors) > 0:
            raise RuntimeError(f"Pano node has non-OSM neighbors: {_neighbors}, {_osm_neighbors}")

        if not _osm_neighbors:
            continue

        # Get the OSM IDs and their independent groups
        _osm_groups_connected = set()
        for _osm_idx in _osm_neighbors:
            _osm_id = seattle_nodes[_osm_idx]
            if _osm_id in independent_seattle_osm_id_to_group:
                _group_props = independent_seattle_osm_id_to_group[_osm_id]
                _osm_groups_connected.add(_group_props)

        independent_seattle_pano_to_osm_groups[_pano_idx] = _osm_groups_connected

        if len(_osm_groups_connected) > 1:
            independent_seattle_pano_tiebreaks += 1

    print(f"Independent Seattle pano landmarks with edges: {len(independent_seattle_pano_to_osm_groups)}")
    print(f"Requiring tiebreak: {independent_seattle_pano_tiebreaks} ({independent_seattle_pano_tiebreaks / max(len(independent_seattle_pano_to_osm_groups), 1) * 100:.1f}%)")

    # Select random group for each pano landmark
    independent_seattle_pano_to_selected_group = {}
    for _pano_idx, _group_set in independent_seattle_pano_to_osm_groups.items():
        if _group_set:
            # Randomly pick one group
            _selected_group = list(_group_set)[np.random.randint(0, len(_group_set))]
            independent_seattle_pano_to_selected_group[_pano_idx] = _selected_group

    return (independent_seattle_pano_to_selected_group,)


@app.cell
def _(
    generate_normal_vector,
    independent_seattle_group_to_osm_id,
    independent_seattle_osm_groups_by_props,
    seattle_dataset,
    seattle_osm_embeddings_tensor,
    seattle_osm_landmark_id_to_idx,
    slu,
    tqdm,
):
    # Generate embeddings for independent Seattle groups
    print("\nGenerating embeddings for independent Seattle groups...")

    independent_seattle_group_to_semantic_emb = {}
    independent_seattle_group_to_random_emb = {}

    for _group_props in tqdm(independent_seattle_osm_groups_by_props.keys(), desc="Independent Seattle groups"):
        # Get the selected Seattle OSM member for this group
        _selected_osm_id = independent_seattle_group_to_osm_id[_group_props]

        # Get the original embedding for semantic version (from Seattle OSM embeddings)
        _selected_custom_id = slu.custom_id_from_props(
            seattle_dataset._landmark_metadata[seattle_dataset._landmark_metadata["id"] == _selected_osm_id].iloc[0]["pruned_props"]
        )
        _emb_idx = seattle_osm_landmark_id_to_idx[_selected_custom_id]
        independent_seattle_group_to_semantic_emb[_group_props] = seattle_osm_embeddings_tensor[_emb_idx]

        # Generate random embedding
        independent_seattle_group_to_random_emb[_group_props] = generate_normal_vector(1536)

    print(f"Created embeddings for {len(independent_seattle_osm_groups_by_props)} independent Seattle groups")

    return (
        independent_seattle_group_to_random_emb,
        independent_seattle_group_to_semantic_emb,
    )


@app.cell
def _(
    dataset,
    group_to_random_emb,
    group_to_semantic_emb,
    independent_seattle_group_to_random_emb,
    independent_seattle_group_to_semantic_emb,
    independent_seattle_osm_id_to_group,
    osm_id_to_group,
    seattle_dataset,
    slu,
    torch,
    tqdm,
):
    # Create independent combined OSM embeddings (Chicago + independent Seattle)
    print("\n=== CREATING INDEPENDENT COMBINED OSM EMBEDDINGS ===")

    _indep_combined_osm_embeddings_semantic = []
    _indep_combined_osm_id_to_idx_semantic = {}
    _indep_combined_osm_embeddings_random = []
    _indep_combined_osm_id_to_idx_random = {}

    # First: Process all Chicago OSM landmarks
    print("Processing Chicago OSM landmarks...")
    _indep_chicago_custom_id_to_group = {}
    _indep_chicago_group_conflicts = 0

    for _indep_idx, _indep_row in tqdm(list(dataset._landmark_metadata.iterrows()), desc="  Chicago OSM"):
        _indep_osm_id = _indep_row["id"]
        _indep_pruned_props = _indep_row["pruned_props"]
        _indep_custom_id = slu.custom_id_from_props(_indep_pruned_props)

        if _indep_osm_id not in osm_id_to_group:
            raise RuntimeError(f"Chicago {_indep_osm_id} not found in osm_id_to_group")

        _indep_group_props = osm_id_to_group[_indep_osm_id]

        if _indep_custom_id in _indep_chicago_custom_id_to_group:
            if _indep_chicago_custom_id_to_group[_indep_custom_id] != _indep_group_props:
                _indep_chicago_group_conflicts += 1
        else:
            _indep_chicago_custom_id_to_group[_indep_custom_id] = _indep_group_props

    if _indep_chicago_group_conflicts > 0:
        print(f"  Found {_indep_chicago_group_conflicts} Chicago custom_ids in multiple groups")

    # Add Chicago embeddings
    for _indep_custom_id, _indep_group_props in tqdm(_indep_chicago_custom_id_to_group.items(), desc="  Adding Chicago embeddings"):
        # Semantic version
        _indep_combined_osm_id_to_idx_semantic[_indep_custom_id] = len(_indep_combined_osm_embeddings_semantic)
        _indep_combined_osm_embeddings_semantic.append(group_to_semantic_emb[_indep_group_props])

        # Random version
        _indep_combined_osm_id_to_idx_random[_indep_custom_id] = len(_indep_combined_osm_embeddings_random)
        _indep_combined_osm_embeddings_random.append(group_to_random_emb[_indep_group_props])

    print(f"Added {len(_indep_chicago_custom_id_to_group)} Chicago OSM embeddings")

    # Second: Process all Seattle OSM landmarks (with independent groups)
    print("\nProcessing independent Seattle OSM landmarks...")
    _indep_seattle_custom_id_to_group = {}
    _indep_seattle_group_conflicts = 0

    for _indep_idx, _indep_row in tqdm(list(seattle_dataset._landmark_metadata.iterrows()), desc="  Seattle OSM"):
        _indep_osm_id = _indep_row["id"]
        _indep_pruned_props = _indep_row["pruned_props"]
        _indep_custom_id = slu.custom_id_from_props(_indep_pruned_props)

        if _indep_osm_id not in independent_seattle_osm_id_to_group:
            raise RuntimeError(f"Seattle {_indep_osm_id} not found in independent_seattle_osm_id_to_group")

        _indep_group_props = independent_seattle_osm_id_to_group[_indep_osm_id]

        if _indep_custom_id in _indep_seattle_custom_id_to_group:
            if _indep_seattle_custom_id_to_group[_indep_custom_id] != _indep_group_props:
                _indep_seattle_group_conflicts += 1
        else:
            _indep_seattle_custom_id_to_group[_indep_custom_id] = _indep_group_props

    if _indep_seattle_group_conflicts > 0:
        print(f"  Found {_indep_seattle_group_conflicts} Seattle custom_ids in multiple groups")

    # Add independent Seattle embeddings
    for _indep_custom_id, _indep_group_props in tqdm(_indep_seattle_custom_id_to_group.items(), desc="  Adding Seattle embeddings"):
        # Use independent Seattle group embeddings
        _indep_semantic_emb = independent_seattle_group_to_semantic_emb[_indep_group_props]
        _indep_random_emb = independent_seattle_group_to_random_emb[_indep_group_props]

        # Semantic version
        _indep_combined_osm_id_to_idx_semantic[_indep_custom_id] = len(_indep_combined_osm_embeddings_semantic)
        _indep_combined_osm_embeddings_semantic.append(_indep_semantic_emb)

        # Random version
        _indep_combined_osm_id_to_idx_random[_indep_custom_id] = len(_indep_combined_osm_embeddings_random)
        _indep_combined_osm_embeddings_random.append(_indep_random_emb)

    print(f"Added {len(_indep_seattle_custom_id_to_group)} independent Seattle OSM embeddings")

    # Convert to tensors
    independent_combined_osm_embeddings_tensor_semantic = torch.stack(_indep_combined_osm_embeddings_semantic)
    independent_combined_osm_embeddings_tensor_random = torch.stack(_indep_combined_osm_embeddings_random)

    # Export without underscore prefix for marimo
    independent_combined_osm_id_to_idx_semantic = _indep_combined_osm_id_to_idx_semantic
    independent_combined_osm_id_to_idx_random = _indep_combined_osm_id_to_idx_random

    print(f"\nIndependent combined OSM embeddings:")
    print(f"  Semantic: {independent_combined_osm_embeddings_tensor_semantic.shape}")
    print(f"  Random: {independent_combined_osm_embeddings_tensor_random.shape}")
    print(f"  Total unique custom_ids: {len(independent_combined_osm_id_to_idx_semantic)}")

    return (
        independent_combined_osm_embeddings_tensor_random,
        independent_combined_osm_embeddings_tensor_semantic,
        independent_combined_osm_id_to_idx_random,
        independent_combined_osm_id_to_idx_semantic,
    )


@app.cell
def _(
    generate_normal_vector,
    independent_seattle_group_to_random_emb,
    independent_seattle_group_to_semantic_emb,
    independent_seattle_pano_to_selected_group,
    seattle_dataset,
    seattle_nodes,
    seattle_pano_embeddings_tensor,
    seattle_pano_landmark_id_to_idx,
    seattle_split_idx,
    torch,
    tqdm,
):
    # Create independent Seattle pano embeddings
    print("\n=== CREATING INDEPENDENT SEATTLE PANO EMBEDDINGS ===")

    # Build Seattle pano lookup for lat/lon
    _indep_seattle_pano_lookup = {}
    for _, _indep_row in seattle_dataset._panorama_metadata.iterrows():
        _indep_seattle_pano_lookup[_indep_row.pano_id] = {
            "lat": float(_indep_row.lat),
            "lon": float(_indep_row.lon)
        }

    _indep_seattle_pano_embeddings_semantic = []
    _indep_seattle_pano_id_to_idx_semantic = {}
    _indep_seattle_pano_embeddings_random = []
    _indep_seattle_pano_id_to_idx_random = {}
    _indep_seattle_pano_with_edges = 0
    _indep_seattle_pano_without_edges = 0

    for _indep_pano_idx in tqdm(range(seattle_split_idx), desc="Independent Seattle pano"):
        _indep_pano_node = seattle_nodes[_indep_pano_idx]
        _indep_pano_id, _indep_landmark_idx, _indep_description = _indep_pano_node

        # Get lat/lon for this pano
        if _indep_pano_id not in _indep_seattle_pano_lookup:
            print(f"Warning: Seattle pano {_indep_pano_id} not in metadata")
            continue
        _indep_lat = _indep_seattle_pano_lookup[_indep_pano_id]["lat"]
        _indep_lon = _indep_seattle_pano_lookup[_indep_pano_id]["lon"]
        _indep_custom_id = f"{_indep_pano_id},{_indep_lat:.6f},{_indep_lon:.6f},__landmark_{_indep_landmark_idx}"

        # Check if this pano has edges to independent Seattle OSM groups
        if _indep_pano_idx in independent_seattle_pano_to_selected_group:
            # Has edges - use independent group's embedding
            _indep_selected_group = independent_seattle_pano_to_selected_group[_indep_pano_idx]
            _indep_semantic_emb = independent_seattle_group_to_semantic_emb[_indep_selected_group]
            _indep_random_emb = independent_seattle_group_to_random_emb[_indep_selected_group]
            _indep_seattle_pano_with_edges += 1
        else:
            # No edges - use original or generate random
            if _indep_custom_id not in seattle_pano_landmark_id_to_idx:
                print(f"Warning: Seattle pano embedding not found for {_indep_custom_id}")
                continue
            _indep_emb_idx = seattle_pano_landmark_id_to_idx[_indep_custom_id]
            _indep_original_emb = seattle_pano_embeddings_tensor[_indep_emb_idx]
            _indep_semantic_emb = _indep_original_emb
            _indep_random_emb = generate_normal_vector(1536)
            _indep_seattle_pano_without_edges += 1

        # Add to both versions
        _indep_seattle_pano_id_to_idx_semantic[_indep_custom_id] = len(_indep_seattle_pano_embeddings_semantic)
        _indep_seattle_pano_embeddings_semantic.append(_indep_semantic_emb)

        _indep_seattle_pano_id_to_idx_random[_indep_custom_id] = len(_indep_seattle_pano_embeddings_random)
        _indep_seattle_pano_embeddings_random.append(_indep_random_emb)

    assert _indep_seattle_pano_id_to_idx_semantic == _indep_seattle_pano_id_to_idx_random

    independent_seattle_pano_embeddings_tensor_semantic = torch.stack(_indep_seattle_pano_embeddings_semantic)
    independent_seattle_pano_embeddings_tensor_random = torch.stack(_indep_seattle_pano_embeddings_random)

    # Export without underscore prefix for marimo
    independent_seattle_pano_id_to_idx_semantic = _indep_seattle_pano_id_to_idx_semantic
    independent_seattle_pano_id_to_idx_random = _indep_seattle_pano_id_to_idx_random

    print(f"Created {len(_indep_seattle_pano_embeddings_semantic)} independent Seattle pano embeddings")
    print(f"  With OSM edges: {_indep_seattle_pano_with_edges}")
    print(f"  Without OSM edges: {_indep_seattle_pano_without_edges}")

    return (
        independent_seattle_pano_embeddings_tensor_random,
        independent_seattle_pano_embeddings_tensor_semantic,
        independent_seattle_pano_id_to_idx_random,
        independent_seattle_pano_id_to_idx_semantic,
    )


@app.cell
def _(
    defaultdict,
    generate_normal_vector,
    independent_combined_osm_embeddings_tensor_semantic,
    independent_seattle_pano_embeddings_tensor_semantic,
    spoofed_pano_embeddings_tensor,
    torch,
    tqdm,
):
    # Create independent no_semantic versions
    print("\n=== CREATING INDEPENDENT NO_SEMANTIC VERSIONS ===")

    # Combine all independent spoofed embeddings (Chicago pano + independent combined OSM + independent Seattle pano)
    _indep_all_chicago_pano = spoofed_pano_embeddings_tensor
    _indep_all_combined_osm = independent_combined_osm_embeddings_tensor_semantic
    _indep_all_seattle_pano = independent_seattle_pano_embeddings_tensor_semantic

    # Hash all unique embeddings across all three sources
    _indep_hash = defaultdict(set)

    print("Hashing Chicago pano embeddings...")
    for _indep_i in tqdm(range(_indep_all_chicago_pano.shape[0])):
        _indep_vector = _indep_all_chicago_pano[_indep_i]
        _indep_key = tuple(_indep_vector.tolist())
        _indep_hash[_indep_key].add(('chicago_pano', _indep_i))

    print("Hashing independent combined OSM embeddings...")
    for _indep_i in tqdm(range(_indep_all_combined_osm.shape[0])):
        _indep_vector = _indep_all_combined_osm[_indep_i]
        _indep_key = tuple(_indep_vector.tolist())
        _indep_hash[_indep_key].add(('combined_osm', _indep_i))

    print("Hashing independent Seattle pano embeddings...")
    for _indep_i in tqdm(range(_indep_all_seattle_pano.shape[0])):
        _indep_vector = _indep_all_seattle_pano[_indep_i]
        _indep_key = tuple(_indep_vector.tolist())
        _indep_hash[_indep_key].add(('seattle_pano', _indep_i))

    print(f"Found {len(_indep_hash)} unique embedding groups")

    # Create no_semantic versions
    _indep_combined_osm_no_semantic = torch.ones_like(independent_combined_osm_embeddings_tensor_semantic) * torch.nan
    _indep_chicago_pano_no_semantic = torch.ones_like(spoofed_pano_embeddings_tensor) * torch.nan
    _indep_seattle_pano_no_semantic = torch.ones_like(independent_seattle_pano_embeddings_tensor_semantic) * torch.nan

    print("Assigning new random embeddings to each group...")
    for _indep_key, _indep_indexes in tqdm(_indep_hash.items(), desc="Assigning embeddings"):
        _indep_new_vector = generate_normal_vector(1536)

        for _indep_source, _indep_idx in _indep_indexes:
            if _indep_source == 'chicago_pano':
                _indep_chicago_pano_no_semantic[_indep_idx] = _indep_new_vector
            elif _indep_source == 'combined_osm':
                _indep_combined_osm_no_semantic[_indep_idx] = _indep_new_vector
            elif _indep_source == 'seattle_pano':
                _indep_seattle_pano_no_semantic[_indep_idx] = _indep_new_vector

    # Verify no NaNs
    assert not torch.any(torch.isnan(_indep_combined_osm_no_semantic))
    assert not torch.any(torch.isnan(_indep_chicago_pano_no_semantic))
    assert not torch.any(torch.isnan(_indep_seattle_pano_no_semantic))

    # Export without underscore prefix for marimo
    independent_chicago_pano_no_semantic = _indep_chicago_pano_no_semantic
    independent_combined_osm_no_semantic = _indep_combined_osm_no_semantic
    independent_seattle_pano_no_semantic = _indep_seattle_pano_no_semantic

    print("✓ Independent no_semantic versions created successfully")

    return (
        independent_chicago_pano_no_semantic,
        independent_combined_osm_no_semantic,
        independent_seattle_pano_no_semantic,
    )


@app.cell
def _(
    Path,
    independent_chicago_pano_no_semantic,
    independent_combined_osm_embeddings_tensor_random,
    independent_combined_osm_embeddings_tensor_semantic,
    independent_combined_osm_id_to_idx_random,
    independent_combined_osm_id_to_idx_semantic,
    independent_combined_osm_no_semantic,
    independent_seattle_pano_embeddings_tensor_random,
    independent_seattle_pano_embeddings_tensor_semantic,
    independent_seattle_pano_id_to_idx_random,
    independent_seattle_pano_id_to_idx_semantic,
    independent_seattle_pano_no_semantic,
    pickle,
    random_pano_embeddings_tensor,
    random_pano_id_to_idx,
    spoofed_pano_embeddings_tensor,
    spoofed_pano_id_to_idx,
):
    # Save independent embeddings
    print("\n=== SAVING INDEPENDENT EMBEDDINGS ===")

    _indep_save_base_dir = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/")

    # ========== SEMANTIC VERSION ==========
    print("\n1. Saving independent semantic version...")

    # OSM (combined Chicago + independent Seattle)
    _indep_save_osm_semantic_dir = _indep_save_base_dir / "sat_spoof_independent_validation/embeddings"
    _indep_save_osm_semantic_dir.mkdir(parents=True, exist_ok=True)
    _indep_save_osm_semantic_path = _indep_save_osm_semantic_dir / "embeddings.pkl"
    with open(_indep_save_osm_semantic_path, 'wb') as _indep_save_f:
        pickle.dump((independent_combined_osm_embeddings_tensor_semantic, independent_combined_osm_id_to_idx_semantic), _indep_save_f)
    print(f"  ✓ OSM semantic: {_indep_save_osm_semantic_path}")
    print(f"    Shape: {independent_combined_osm_embeddings_tensor_semantic.shape}")

    # Chicago Pano
    _indep_save_chicago_pano_semantic_dir = _indep_save_base_dir / "pano_spoof_independent_validation/Chicago/embeddings"
    _indep_save_chicago_pano_semantic_dir.mkdir(parents=True, exist_ok=True)
    _indep_save_chicago_pano_semantic_path = _indep_save_chicago_pano_semantic_dir / "embeddings.pkl"
    with open(_indep_save_chicago_pano_semantic_path, 'wb') as _indep_save_f:
        pickle.dump((spoofed_pano_embeddings_tensor, spoofed_pano_id_to_idx), _indep_save_f)
    print(f"  ✓ Chicago pano semantic: {_indep_save_chicago_pano_semantic_path}")
    print(f"    Shape: {spoofed_pano_embeddings_tensor.shape}")

    # Seattle Pano
    _indep_save_seattle_pano_semantic_dir = _indep_save_base_dir / "pano_spoof_independent_validation/Seattle/embeddings"
    _indep_save_seattle_pano_semantic_dir.mkdir(parents=True, exist_ok=True)
    _indep_save_seattle_pano_semantic_path = _indep_save_seattle_pano_semantic_dir / "embeddings.pkl"
    with open(_indep_save_seattle_pano_semantic_path, 'wb') as _indep_save_f:
        pickle.dump((independent_seattle_pano_embeddings_tensor_semantic, independent_seattle_pano_id_to_idx_semantic), _indep_save_f)
    print(f"  ✓ Seattle pano semantic: {_indep_save_seattle_pano_semantic_path}")
    print(f"    Shape: {independent_seattle_pano_embeddings_tensor_semantic.shape}")

    # ========== RANDOM VERSION ==========
    print("\n2. Saving independent random version...")

    # OSM (combined Chicago + independent Seattle)
    _indep_save_osm_random_dir = _indep_save_base_dir / "sat_spoof_random_independent_validation/embeddings"
    _indep_save_osm_random_dir.mkdir(parents=True, exist_ok=True)
    _indep_save_osm_random_path = _indep_save_osm_random_dir / "embeddings.pkl"
    with open(_indep_save_osm_random_path, 'wb') as _indep_save_f:
        pickle.dump((independent_combined_osm_embeddings_tensor_random, independent_combined_osm_id_to_idx_random), _indep_save_f)
    print(f"  ✓ OSM random: {_indep_save_osm_random_path}")
    print(f"    Shape: {independent_combined_osm_embeddings_tensor_random.shape}")

    # Chicago Pano
    _indep_save_chicago_pano_random_dir = _indep_save_base_dir / "pano_spoof_random_independent_validation/Chicago/embeddings"
    _indep_save_chicago_pano_random_dir.mkdir(parents=True, exist_ok=True)
    _indep_save_chicago_pano_random_path = _indep_save_chicago_pano_random_dir / "embeddings.pkl"
    with open(_indep_save_chicago_pano_random_path, 'wb') as _indep_save_f:
        pickle.dump((random_pano_embeddings_tensor, random_pano_id_to_idx), _indep_save_f)
    print(f"  ✓ Chicago pano random: {_indep_save_chicago_pano_random_path}")
    print(f"    Shape: {random_pano_embeddings_tensor.shape}")

    # Seattle Pano
    _indep_save_seattle_pano_random_dir = _indep_save_base_dir / "pano_spoof_random_independent_validation/Seattle/embeddings"
    _indep_save_seattle_pano_random_dir.mkdir(parents=True, exist_ok=True)
    _indep_save_seattle_pano_random_path = _indep_save_seattle_pano_random_dir / "embeddings.pkl"
    with open(_indep_save_seattle_pano_random_path, 'wb') as _indep_save_f:
        pickle.dump((independent_seattle_pano_embeddings_tensor_random, independent_seattle_pano_id_to_idx_random), _indep_save_f)
    print(f"  ✓ Seattle pano random: {_indep_save_seattle_pano_random_path}")
    print(f"    Shape: {independent_seattle_pano_embeddings_tensor_random.shape}")

    # ========== NO_SEMANTIC VERSION ==========
    print("\n3. Saving independent no_semantic version...")

    # OSM (combined Chicago + independent Seattle)
    _indep_save_osm_no_semantic_dir = _indep_save_base_dir / "sat_spoof_no_semantic_independent_validation/embeddings"
    _indep_save_osm_no_semantic_dir.mkdir(parents=True, exist_ok=True)
    _indep_save_osm_no_semantic_path = _indep_save_osm_no_semantic_dir / "embeddings.pkl"
    with open(_indep_save_osm_no_semantic_path, 'wb') as _indep_save_f:
        pickle.dump((independent_combined_osm_no_semantic, independent_combined_osm_id_to_idx_semantic), _indep_save_f)
    print(f"  ✓ OSM no_semantic: {_indep_save_osm_no_semantic_path}")
    print(f"    Shape: {independent_combined_osm_no_semantic.shape}")

    # Chicago Pano
    _indep_save_chicago_pano_no_semantic_dir = _indep_save_base_dir / "pano_spoof_no_semantic_independent_validation/Chicago/embeddings"
    _indep_save_chicago_pano_no_semantic_dir.mkdir(parents=True, exist_ok=True)
    _indep_save_chicago_pano_no_semantic_path = _indep_save_chicago_pano_no_semantic_dir / "embeddings.pkl"
    with open(_indep_save_chicago_pano_no_semantic_path, 'wb') as _indep_save_f:
        pickle.dump((independent_chicago_pano_no_semantic, spoofed_pano_id_to_idx), _indep_save_f)
    print(f"  ✓ Chicago pano no_semantic: {_indep_save_chicago_pano_no_semantic_path}")
    print(f"    Shape: {independent_chicago_pano_no_semantic.shape}")

    # Seattle Pano
    _indep_save_seattle_pano_no_semantic_dir = _indep_save_base_dir / "pano_spoof_no_semantic_independent_validation/Seattle/embeddings"
    _indep_save_seattle_pano_no_semantic_dir.mkdir(parents=True, exist_ok=True)
    _indep_save_seattle_pano_no_semantic_path = _indep_save_seattle_pano_no_semantic_dir / "embeddings.pkl"
    with open(_indep_save_seattle_pano_no_semantic_path, 'wb') as _indep_save_f:
        pickle.dump((independent_seattle_pano_no_semantic, independent_seattle_pano_id_to_idx_semantic), _indep_save_f)
    print(f"  ✓ Seattle pano no_semantic: {_indep_save_seattle_pano_no_semantic_path}")
    print(f"    Shape: {independent_seattle_pano_no_semantic.shape}")

    print("\n" + "="*60)
    print("✓ ALL INDEPENDENT EMBEDDINGS SAVED SUCCESSFULLY!")
    print("="*60)
    print("\nSummary:")
    print(f"  - Independent combined OSM embeddings (Chicago + Seattle): {len(independent_combined_osm_id_to_idx_semantic)} unique IDs")
    print(f"  - Chicago pano embeddings: {len(spoofed_pano_id_to_idx)} unique IDs")
    print(f"  - Independent Seattle pano embeddings: {len(independent_seattle_pano_id_to_idx_semantic)} unique IDs")
    print(f"\nThree versions created: semantic, random, no_semantic")
    print(f"Base directory: {_indep_save_base_dir}")

    return


@app.cell
def _(
    Path,
    independent_seattle_pano_id_to_idx_random,
    independent_seattle_pano_id_to_idx_semantic,
    json,
    random_pano_id_to_idx,
    spoofed_pano_id_to_idx,
):
    # Generate panorama_metadata.jsonl for independent validation embeddings
    print("\n=== GENERATING INDEPENDENT VALIDATION PANO METADATA ===")

    _indep_meta_base_dir = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/")

    # Independent validation versions with their Chicago and Seattle id_to_idx mappings
    _independent_validation_versions = [
        ("pano_spoof_independent_validation", spoofed_pano_id_to_idx, independent_seattle_pano_id_to_idx_semantic),
        ("pano_spoof_random_independent_validation", random_pano_id_to_idx, independent_seattle_pano_id_to_idx_random),
        ("pano_spoof_no_semantic_independent_validation", spoofed_pano_id_to_idx, independent_seattle_pano_id_to_idx_semantic),
    ]

    for _version_name, _chicago_id_to_idx, _seattle_id_to_idx in _independent_validation_versions:
        print(f"\nGenerating metadata for {_version_name}...")

        # Process Chicago
        print(f"  Processing Chicago...")
        _chicago_metadata_entries = []
        for _custom_id in _chicago_id_to_idx.keys():
            _parts = _custom_id.split(',')
            if len(_parts) != 4:
                continue

            _pano_id = _parts[0]
            _lat = _parts[1]
            _lon = _parts[2]
            _landmark_part = _parts[3]

            if not _landmark_part.startswith('__landmark_'):
                continue
            _landmark_idx = int(_landmark_part.split('_')[-1])

            _entry = {
                "panorama_id": f"{_pano_id},{_lat},{_lon},",
                "landmark_idx": _landmark_idx,
                "custom_id": _custom_id,
                "yaw_angles": [0.0, 0.0, 0.0, 0.0]
            }
            _chicago_metadata_entries.append(_entry)

        _chicago_metadata_entries.sort(key=lambda x: (x["panorama_id"], x["landmark_idx"]))

        _chicago_output_dir = _indep_meta_base_dir / _version_name / "Chicago" / "embedding_requests"
        _chicago_output_dir.mkdir(parents=True, exist_ok=True)
        _chicago_output_file = _chicago_output_dir / "panorama_metadata.jsonl"

        with open(_chicago_output_file, 'w') as _f:
            for _entry in _chicago_metadata_entries:
                _f.write(json.dumps(_entry) + '\n')

        print(f"    ✓ Chicago: {len(_chicago_metadata_entries)} entries")

        # Process Seattle
        print(f"  Processing Seattle...")
        _seattle_metadata_entries = []
        for _custom_id in _seattle_id_to_idx.keys():
            _parts = _custom_id.split(',')
            if len(_parts) != 4:
                continue

            _pano_id = _parts[0]
            _lat = _parts[1]
            _lon = _parts[2]
            _landmark_part = _parts[3]

            if not _landmark_part.startswith('__landmark_'):
                continue
            _landmark_idx = int(_landmark_part.split('_')[-1])

            _entry = {
                "panorama_id": f"{_pano_id},{_lat},{_lon},",
                "landmark_idx": _landmark_idx,
                "custom_id": _custom_id,
                "yaw_angles": [0.0, 0.0, 0.0, 0.0]
            }
            _seattle_metadata_entries.append(_entry)

        _seattle_metadata_entries.sort(key=lambda x: (x["panorama_id"], x["landmark_idx"]))

        _seattle_output_dir = _indep_meta_base_dir / _version_name / "Seattle" / "embedding_requests"
        _seattle_output_dir.mkdir(parents=True, exist_ok=True)
        _seattle_output_file = _seattle_output_dir / "panorama_metadata.jsonl"

        with open(_seattle_output_file, 'w') as _f:
            for _entry in _seattle_metadata_entries:
                _f.write(json.dumps(_entry) + '\n')

        print(f"    ✓ Seattle: {len(_seattle_metadata_entries)} entries")

    print("\n✓ Independent validation pano metadata generated successfully!")
    return


if __name__ == "__main__":
    app.run()
