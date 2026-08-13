"""Mode tracking: cluster the belief and follow those clusters over time.

Design doc §5.1/§5.6. Modes are the unit of explanation in the visualizer,
and the doc is explicit that mode bookkeeping is a first-class filter output
rather than post-hoc analysis — "mode A believes tracklet 7 is Graves Light;
mode B believes it's Boston Light" is the sentence the whole debugging story
is built around, and it cannot be reconstructed from a summary later.

Two design choices worth stating:

**Grid connected components, not a distance-based clusterer.** Binning
particles into (east, north, heading) cells and joining occupied neighbours
is O(N), needs no extra dependency, and — the deciding factor — is exactly
deterministic. A clusterer with iteration order or random seeding would put
a nondeterminism inside the filter's output and break the §3.8 replay
contract.

**Identity by particle lineage, not by centroid matching.** Every particle
carries the mode id it belonged to last keyframe, so a cluster's identity is
whichever ancestor holds the most weight inside it. Centroid nearest-neighbour
matching guesses; lineage knows. It also makes merges and splits fall out for
free: two ancestors inside one cluster IS a merge, one ancestor spread across
two clusters IS a split, and a cluster with no ancestor is a birth — which,
for particles injected by the mixture proposal, carries the provenance §5.5
attaches to them.
"""

import dataclasses
import math

import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    structs,
)

UNASSIGNED = -1


@dataclasses.dataclass
class ModeAssignment:
    mode_id: np.ndarray  # per particle; UNASSIGNED for outliers
    modes: list  # list[structs.ModeRecord], heaviest first
    events: list  # list[structs.ModeEvent]


def _cell_keys(belief, config: structs.ModeConfig):
    heading_deg = np.degrees(belief.heading_rad) % 360.0
    return np.stack([
        np.floor(belief.east_m / config.cell_size_m).astype(np.int64),
        np.floor(belief.north_m / config.cell_size_m).astype(np.int64),
        np.floor(heading_deg / config.heading_cell_deg).astype(np.int64),
    ], axis=1)


def _connected_components(cells: np.ndarray, n_heading_bins: int):
    """Label occupied cells by 26-connectivity, wrapping the heading axis.

    Cells are visited in sorted order so the labelling — and therefore every
    downstream mode id — is independent of particle order.
    """
    index_of = {tuple(cell): i for i, cell in enumerate(cells)}
    labels = np.full(len(cells), -1, dtype=np.int64)
    offsets = [(de, dn, dh)
               for de in (-1, 0, 1) for dn in (-1, 0, 1) for dh in (-1, 0, 1)
               if not (de == 0 and dn == 0 and dh == 0)]
    next_label = 0
    for start in range(len(cells)):
        if labels[start] != -1:
            continue
        labels[start] = next_label
        stack = [start]
        while stack:
            current = stack.pop()
            east, north, heading = cells[current]
            for de, dn, dh in offsets:
                neighbour = (east + de, north + dn,
                             (heading + dh) % n_heading_bins)
                found = index_of.get(neighbour)
                if found is not None and labels[found] == -1:
                    labels[found] = next_label
                    stack.append(found)
        next_label += 1
    return labels, next_label


def cluster(belief, config: structs.ModeConfig):
    """Weighted clustering of a belief into candidate modes.

    Returns (cluster_id per particle, n_clusters); UNASSIGNED marks particles
    in cells too light to be part of any mode.
    """
    weights = belief.normalized_weights()
    keys = _cell_keys(belief, config)
    n_heading_bins = int(math.ceil(360.0 / config.heading_cell_deg))

    unique_cells, inverse = np.unique(keys, axis=0, return_inverse=True)
    cell_weight = np.bincount(inverse, weights=weights,
                              minlength=len(unique_cells))
    dense = cell_weight >= config.min_cell_weight
    if not np.any(dense):
        return np.full(belief.n, UNASSIGNED, dtype=np.int64), 0

    dense_indices = np.nonzero(dense)[0]
    labels, n_labels = _connected_components(unique_cells[dense_indices],
                                             n_heading_bins)
    cell_label = np.full(len(unique_cells), UNASSIGNED, dtype=np.int64)
    cell_label[dense_indices] = labels
    return cell_label[inverse], n_labels


def _circular_mean_deg(heading_rad, weights):
    total = weights.sum()
    if total <= 0.0:
        return 0.0
    return math.degrees(math.atan2(
        float(weights @ np.sin(heading_rad)) / total,
        float(weights @ np.cos(heading_rad)) / total)) % 360.0


def _circular_std_deg(heading_rad, weights):
    total = weights.sum()
    if total <= 0.0:
        return 0.0
    resultant = math.hypot(float(weights @ np.sin(heading_rad)) / total,
                           float(weights @ np.cos(heading_rad)) / total)
    resultant = min(max(resultant, 1e-15), 1.0 - 1e-15)
    return math.degrees(math.sqrt(-2.0 * math.log(resultant)))


class ModeTracker:
    """Stateful across keyframes: owns mode ids, births, deaths and merges."""

    def __init__(self, config: structs.ModeConfig):
        self.config = config
        self._next_mode_id = 0
        self._previous = {}  # mode_id -> structs.ModeRecord
        self._birth = {}  # mode_id -> (keyframe_idx, provenance dict)

    def update(self, belief, keyframe_idx: int,
               proposal_events=()) -> ModeAssignment:
        cluster_id, n_clusters = cluster(belief, self.config)
        weights = belief.normalized_weights()

        # Cluster -> weight, dropping anything too light to call a mode.
        candidates = []
        for label in range(n_clusters):
            member = cluster_id == label
            weight = float(weights[member].sum())
            if weight >= self.config.min_mode_weight:
                candidates.append((weight, label, member))
        candidates.sort(key=lambda item: (-item[0], item[1]))

        # Lineage: the heaviest ancestor inside a cluster claims it. Heavier
        # clusters choose first, so a split leaves the id with the dominant
        # child and mints a new one for the other.
        events = []
        claimed = {}
        assignments = []
        for weight, label, member in candidates:
            ancestors = {}
            for ancestor in np.unique(belief.mode_id[member]):
                if int(ancestor) == UNASSIGNED:
                    continue
                ancestors[int(ancestor)] = float(
                    weights[member & (belief.mode_id == ancestor)].sum())
            inherited = None
            for ancestor, _ in sorted(ancestors.items(),
                                      key=lambda kv: (-kv[1], kv[0])):
                if ancestor not in claimed:
                    inherited = ancestor
                    break
            if inherited is None:
                inherited = self._next_mode_id
                self._next_mode_id += 1
                provenance = self._provenance(belief, member, proposal_events)
                self._birth[inherited] = (keyframe_idx, provenance)
                events.append(structs.ModeEvent(
                    keyframe_idx=keyframe_idx, kind="birth",
                    mode_id=inherited,
                    parent_mode_ids=sorted(ancestors),
                    detail=provenance))
            elif len(ancestors) > 1:
                events.append(structs.ModeEvent(
                    keyframe_idx=keyframe_idx, kind="merge",
                    mode_id=inherited,
                    parent_mode_ids=sorted(ancestors),
                    detail={}))
            claimed[inherited] = True
            assignments.append((inherited, weight, member, sorted(ancestors)))

        mode_id = np.full(belief.n, UNASSIGNED, dtype=np.int64)
        modes = []
        for assigned_id, weight, member, ancestors in assignments:
            mode_id[member] = assigned_id
            member_weights = weights[member]
            birth_keyframe, provenance = self._birth.get(
                assigned_id, (keyframe_idx, {}))
            modes.append(structs.ModeRecord(
                mode_id=assigned_id,
                weight=weight,
                n_particles=int(member.sum()),
                mean_east_m=float(member_weights @ belief.east_m[member]
                                  / member_weights.sum()),
                mean_north_m=float(member_weights @ belief.north_m[member]
                                   / member_weights.sum()),
                mean_heading_deg=_circular_mean_deg(
                    belief.heading_rad[member], member_weights),
                position_std_m=self._position_std(belief, member,
                                                  member_weights),
                heading_std_deg=_circular_std_deg(belief.heading_rad[member],
                                                  member_weights),
                birth_keyframe_idx=birth_keyframe,
                parent_mode_ids=ancestors,
                provenance=provenance))

        for previous_id in self._previous:
            if previous_id not in claimed:
                events.append(structs.ModeEvent(
                    keyframe_idx=keyframe_idx, kind="death",
                    mode_id=previous_id, parent_mode_ids=[], detail={}))
        self._previous = {record.mode_id: record for record in modes}
        return ModeAssignment(mode_id=mode_id, modes=modes, events=events)

    @staticmethod
    def _position_std(belief, member, member_weights):
        total = member_weights.sum()
        east = belief.east_m[member]
        north = belief.north_m[member]
        mean_east = float(member_weights @ east) / total
        mean_north = float(member_weights @ north) / total
        variance = float(member_weights @ (np.square(east - mean_east)
                                           + np.square(north - mean_north)))
        return math.sqrt(max(0.5 * variance / total, 0.0))

    @staticmethod
    def _provenance(belief, member, proposal_events):
        """What produced a newborn mode (§5.5 [CONTRACT]).

        Modes inherit provenance from their founding particles, which is what
        makes "where did this wrong mode come from" a one-click question.
        """
        event_ids = belief.proposal_event_id[member]
        from_proposal = event_ids >= 0
        if not np.any(from_proposal):
            return {"source": "motion"}
        values, counts = np.unique(event_ids[from_proposal],
                                   return_counts=True)
        event_id = int(values[int(np.argmax(counts))])
        hypotheses = belief.proposal_hypothesis[member][
            event_ids == event_id]
        provenance = {"source": "proposal", "proposal_event_id": event_id}
        if hypotheses.size:
            values, counts = np.unique(hypotheses, return_counts=True)
            hypothesis = int(values[int(np.argmax(counts))])
            provenance["hypothesis_index"] = hypothesis
            for event in proposal_events:
                if event.event_id != event_id:
                    continue
                provenance["trigger"] = event.trigger
                if hypothesis < len(event.hypothesis_landmark_ids):
                    provenance["landmark_ids"] = ",".join(
                        event.hypothesis_landmark_ids[hypothesis])
                    provenance["tracklet_ids"] = ",".join(
                        event.hypothesis_tracklet_ids[hypothesis])
        return provenance
