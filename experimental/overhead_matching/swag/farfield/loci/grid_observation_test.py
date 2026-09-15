import unittest

import common.torch.load_torch_deps  # noqa: F401
import numpy as np
import torch

from experimental.overhead_matching.swag.farfield.loci.grid_observation import (
    LociGridObservation,
    _build_regular_lattice_mapping,
)
from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    SafaPlusNormalizedLandmarkAggregator,
)


class GridObservationTest(unittest.TestCase):
    def test_arithmetic_overlap_uses_matrix_order_and_exact_loci_fusion(self):
        lattice = {
            "shape_xy": [3, 2],
            "min_pixel_xy": [0.0, 0.0],
            "stride_px": 1.0,
            "source_px": 2.0,
        }
        # Lattice-index -> lexicographically sorted matrix-column rank.
        mapping, support = _build_regular_lattice_mapping(
            lattice,
            cell_x_px=np.asarray([0.5, 2.0, 9.0]),
            cell_y_px=np.asarray([0.5, 1.0, 9.0]),
            x_rank=(2, 0, 1),
            y_rank=(1, 0),
            device="cpu",
        )
        self.assertEqual(mapping.cell_offsets.tolist(), [0, 4, 5, 5])
        self.assertEqual(mapping.patch_indices.tolist(), [5, 3, 2, 0, 1])
        self.assertEqual(support.tolist(), [True, True, False])

        landmark = torch.tensor([[0.2, 1.0, float("nan"), 0.4, 0.6, 0.8]])
        aggregator = SafaPlusNormalizedLandmarkAggregator(
            None, landmark, {"pano_id": ["p0"]},
            image_sigma=0.2, landmark_sigma=0.4, device=torch.device("cpu"))
        patch_log_likelihood = aggregator("p0")
        observation = LociGridObservation(
            aggregator=aggregator,
            mapping=mapping,
            support_mask=support.reshape(1, 3),
            n_north=1,
            n_east=3,
            n_patches=6,
        )
        actual = observation.log_likelihood("p0")[0]
        expected = torch.stack([
            patch_log_likelihood[[5, 3, 2, 0]].max(),
            patch_log_likelihood[1],
            torch.tensor(-torch.inf),
        ])
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.exp()[2].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
