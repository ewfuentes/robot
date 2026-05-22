from dataclasses import dataclass


@dataclass
class Pairs:
    positive_pairs: list[tuple[int, int]]
    negative_pairs: list[tuple[int, int]]
    semipositive_pairs: list[tuple[int, int]]


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
