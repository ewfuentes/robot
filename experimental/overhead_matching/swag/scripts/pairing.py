from dataclasses import dataclass
from enum import StrEnum
from typing import Union
import common.torch.load_torch_deps
import torch


@dataclass
class Pairs:
    positive_pairs: list[tuple[int, int]]
    negative_pairs: list[tuple[int, int]]
    semipositive_pairs: list[tuple[int, int]]


class PairingType(StrEnum):
    PAIRS = "pairs"
    ANCHOR_SETS = "anchor_sets"


@dataclass
class PositiveAnchorSets:
    anchor: list[int]
    positive: list[set[int]]
    semipositive: list[set[int]]


PairingDataType = Union[Pairs, PositiveAnchorSets]


def create_pairs(panorama_metadata, satellite_metadata) -> Pairs:
    # Generate an exhaustive set of triplets where the anchor is a panorama
    # TODO consider creating triplets where a satellite patch is the anchor
    out = Pairs(positive_pairs=[], negative_pairs=[], semipositive_pairs=[])
    for batch_pano_idx in range(len(panorama_metadata)):
        # batch_pano_idx is the index of the panorama in the batch
        # pano_idx is the index of the panorama in the dataset
        pano_idx = panorama_metadata[batch_pano_idx]['index']

        for batch_sat_idx in range(len(satellite_metadata)):
            # batch_sat_idx is the index of the satellite image in the batch
            curr_sat_metadata = satellite_metadata[batch_sat_idx]
            if pano_idx in curr_sat_metadata["positive_panorama_idxs"]:
                out.positive_pairs.append((batch_pano_idx, batch_sat_idx))
            elif pano_idx in curr_sat_metadata["semipositive_panorama_idxs"]:
                out.semipositive_pairs.append((batch_pano_idx, batch_sat_idx))
            else:
                out.negative_pairs.append((batch_pano_idx, batch_sat_idx))
    return out


def create_anchors(panorama_metadata,
                   satellite_metadata,
                   use_pano_as_anchor: bool) -> PositiveAnchorSets:
    name_of_non_anchor = "satellite" if use_pano_as_anchor else "panorama"
    anchor_metadata = panorama_metadata if use_pano_as_anchor else satellite_metadata
    non_anchor_metadata = satellite_metadata if use_pano_as_anchor else panorama_metadata

    non_anchor_dataset_to_batch_map = {
        non_anchor_metadata[i]['index']: i for i in range(len(non_anchor_metadata))}
    assert len(non_anchor_dataset_to_batch_map) == len(non_anchor_metadata)
    non_anchor_dataset_indices = set(non_anchor_dataset_to_batch_map.keys())
    out = PositiveAnchorSets(anchor=[], positive=[], semipositive=[])

    for batch_anchor_idx, _ in enumerate(anchor_metadata):
        # batch_anchor_idx is the index of the anchor in the batch

        # get the set of positive/semi-positive dataset indices
        anchor_dataset_positives = set(
            anchor_metadata[batch_anchor_idx][f"positive_{name_of_non_anchor}_idxs"])
        anchor_dataset_semipositives = set(
            anchor_metadata[batch_anchor_idx][f"semipositive_{name_of_non_anchor}_idxs"])
        # get the set of dataset indices for positive/semipositive members in the batch
        positive_non_anchor_dataset_in_batch = non_anchor_dataset_indices.intersection(
            anchor_dataset_positives)
        semipositive_non_anchor_dataset_in_batch = non_anchor_dataset_indices.intersection(
            anchor_dataset_semipositives)
        # map back to batch indicies from dataset indices
        positive_batch_indices = set(
            map(lambda x: non_anchor_dataset_to_batch_map[x], positive_non_anchor_dataset_in_batch))
        semipositive_batch_indices = set(
            map(lambda x: non_anchor_dataset_to_batch_map[x], semipositive_non_anchor_dataset_in_batch))

        out.anchor.append(batch_anchor_idx)
        out.positive.append(positive_batch_indices)
        out.semipositive.append(semipositive_batch_indices)

    return out


def collapse_anchors_to_torch(anchor_set: PositiveAnchorSets) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given an anchor set, return two torch.long tensors:
    anchor_index: torch.long, index into the anchor batch
    pos_semipos_index: torch.long, index into the non_anchor positives/semi-positives.

    Note that to collapse the map from anchor -> {set of pos/semipos}, anchor indexes are replicated 
    """

    anchor_index_tensor = torch.tensor(anchor_set.anchor, dtype=torch.long)
    flattened_pos_semipos = []
    anchor_repeat = torch.zeros_like(anchor_index_tensor)
    for i, (pos, semipos) in enumerate(zip(anchor_set.positive, anchor_set.semipositive)):
        anchor_repeat[i] = len(pos) + len(semipos)
        flattened_pos_semipos += list(pos) + list(semipos)

    out_anchor_set = torch.repeat_interleave(
        anchor_index_tensor, anchor_repeat, dim=0, output_size=anchor_repeat.sum().item())
    out_flattened_pos_semipos = torch.tensor(flattened_pos_semipos)
    assert out_anchor_set.shape == out_flattened_pos_semipos.shape
    return out_anchor_set, out_flattened_pos_semipos
