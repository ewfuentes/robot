
import unittest

import common.torch.load_torch_deps
import torch

import experimental.overhead_matching.learned.model.pose_optimizer as po


class PoseOptimizerTest(unittest.TestCase):
    def test_two_object_test(self):
        # Setup
        # There is a robot at (-2, 0) with a heading of 0 that observes two landmarks
        # A, at (0, -3), and B, at (0, 4).
        # Given object positions in the robot frame and in the world frame,
        # we want to solve for the robot position with respect to the world frame
        #
        #           +Y
        #           ▲
        #           │
        #           │
        #           │
        #           B
        #           │
        # ◄─────R───┼─────────► +X
        #           │
        #           A
        #           │
        #           │
        #           │
        #           ▼

        obj_in_world = torch.tensor([[
            [0.0, -3.0],  # A in world
            [0.0, 4.0]]   # B in world
        ])

        obj_in_robot = torch.tensor([[
            [2.0, -3.0],  # A in robot
            [2.0, 4.0]]   # B in robot
        ])

        association = torch.tensor([[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]]
        ])

        pose_optimizer = po.PoseOptimizerLayer()

        # Action

        solution = pose_optimizer(association, obj_in_world, obj_in_robot)

        print(solution)

        # Verification
        self.assertEqual(solution.shape, (1, 4))
        self.assertAlmostEqual(solution[0, 0].item(), 2.0)
        self.assertAlmostEqual(solution[0, 1].item(), 0)
        self.assertAlmostEqual(solution[0, 2].item(), 1.0)
        self.assertAlmostEqual(solution[0, 3].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
