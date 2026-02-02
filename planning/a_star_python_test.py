import unittest
import numpy as np

from planning import a_star_python


class AStarPythonTest(unittest.TestCase):

    def test_simple_linear_path(self):
        """Test A* on a simple linear graph: 0 -> 1 -> 2 -> 3"""
        # Graph: 0 -- 1 -- 2 -- 3
        neighbors = [
            [1],      # 0 -> 1
            [0, 2],   # 1 -> 0, 2
            [1, 3],   # 2 -> 1, 3
            [2],      # 3 -> 2
        ]
        edge_costs = [
            [1.0],        # 0 -> 1: cost 1
            [1.0, 1.0],   # 1 -> 0: cost 1, 1 -> 2: cost 1
            [1.0, 1.0],   # 2 -> 1: cost 1, 2 -> 3: cost 1
            [1.0],        # 3 -> 2: cost 1
        ]
        # Heuristics: straight-line distance to goal (node 3)
        heuristics = np.array([3.0, 2.0, 1.0, 0.0])

        result = a_star_python.find_path(neighbors, edge_costs, heuristics, 0, 3)

        self.assertIsNotNone(result)
        self.assertEqual(result.states, [0, 1, 2, 3])
        self.assertAlmostEqual(result.cost, 3.0)

    def test_shortest_path_not_fewest_hops(self):
        """Test A* finds shortest path by cost, not fewest edges.

        Graph structure:
              1
            / | \
           2  |  4
          /   |   \
         0----3----5

        Direct path 0->3->5 has 2 hops but cost 10+10=20.
        Path 0->1->4->5 has 3 hops but cost 1+1+1=3.
        A* should find the cheaper 3-hop path.
        """
        neighbors = [
            [1, 3],      # 0 -> 1, 3
            [0, 2, 3, 4],  # 1 -> 0, 2, 3, 4
            [1],         # 2 -> 1
            [0, 1, 5],   # 3 -> 0, 1, 5
            [1, 5],      # 4 -> 1, 5
            [3, 4],      # 5 -> 3, 4
        ]
        edge_costs = [
            [1.0, 10.0],        # 0->1: cheap, 0->3: expensive
            [1.0, 1.0, 1.0, 1.0],  # all cheap from 1
            [1.0],
            [10.0, 1.0, 10.0],  # 3->0, 3->1, 3->5 (3->5 expensive)
            [1.0, 1.0],         # 4->1, 4->5 cheap
            [10.0, 1.0],        # 5->3 expensive, 5->4 cheap
        ]
        # Heuristics to goal node 5 (underestimates are fine)
        heuristics = np.array([2.0, 1.5, 2.5, 1.0, 0.5, 0.0])

        result = a_star_python.find_path(neighbors, edge_costs, heuristics, 0, 5)

        self.assertIsNotNone(result)
        # Should take the cheap path: 0 -> 1 -> 4 -> 5 (cost 3)
        # NOT the direct path: 0 -> 3 -> 5 (cost 20)
        self.assertEqual(result.states, [0, 1, 4, 5])
        self.assertAlmostEqual(result.cost, 3.0)

    def test_grid_with_obstacle(self):
        """Test A* navigates around an obstacle in a grid.

        Grid (5x3):
          0 - 1 - 2 - 3 - 4
          |   |   X   |   |
          5 - 6 - 7 - 8 - 9
          |   |   |   |   |
         10 -11 -12 -13 -14

        X marks a blocked connection (no edge between 2-7).
        Path from 0 to 4: naive would go 0->1->2->3->4
        But if 2-7 were the goal, we'd need to go around.

        Test: find path from 2 to 12 when 2-7 is blocked.
        Must go: 2->1->6->7->12 or 2->3->8->7->12 (both cost 4)
        """
        # Build 5x3 grid
        def idx(row, col):
            return row * 5 + col

        neighbors = [[] for _ in range(15)]
        edge_costs = [[] for _ in range(15)]

        for row in range(3):
            for col in range(5):
                node = idx(row, col)
                # Right neighbor
                if col < 4:
                    right = idx(row, col + 1)
                    # Block connection between 2 and 7 (node 2 at row 0, col 2)
                    # Actually let's block 2-7 connection differently
                    neighbors[node].append(right)
                    edge_costs[node].append(1.0)
                # Left neighbor
                if col > 0:
                    left = idx(row, col - 1)
                    neighbors[node].append(left)
                    edge_costs[node].append(1.0)
                # Down neighbor
                if row < 2:
                    down = idx(row + 1, col)
                    # Block the direct path from node 2 (0,2) to node 7 (1,2)
                    if not (node == 2 and down == 7):
                        neighbors[node].append(down)
                        edge_costs[node].append(1.0)
                # Up neighbor
                if row > 0:
                    up = idx(row - 1, col)
                    # Block the reverse direction too
                    if not (node == 7 and up == 2):
                        neighbors[node].append(up)
                        edge_costs[node].append(1.0)

        # Heuristics: Manhattan distance to goal (node 12 at row 2, col 2)
        goal_row, goal_col = 2, 2
        heuristics = np.array([
            abs(r - goal_row) + abs(c - goal_col)
            for r in range(3) for c in range(5)
        ], dtype=float)

        result = a_star_python.find_path(neighbors, edge_costs, heuristics, 2, 12)

        self.assertIsNotNone(result)
        # Path should go around the blocked edge
        # Either 2->1->6->11->12 or 2->3->8->13->12 or through 7 via different route
        # The cost should be 4 (4 edges)
        self.assertEqual(result.states[0], 2)
        self.assertEqual(result.states[-1], 12)
        self.assertAlmostEqual(result.cost, 4.0)
        # Verify 7 is NOT immediately after 2 in the path
        for i, state in enumerate(result.states):
            if state == 2 and i + 1 < len(result.states):
                self.assertNotEqual(result.states[i + 1], 7,
                    "Path should not go directly from 2 to 7 (blocked)")

    def test_no_path_exists(self):
        """Test A* returns None when no path exists."""
        # Two disconnected components: {0, 1} and {2, 3}
        neighbors = [
            [1],   # 0 -> 1
            [0],   # 1 -> 0
            [3],   # 2 -> 3
            [2],   # 3 -> 2
        ]
        edge_costs = [
            [1.0],
            [1.0],
            [1.0],
            [1.0],
        ]
        heuristics = np.array([2.0, 2.0, 1.0, 0.0])

        result = a_star_python.find_path(neighbors, edge_costs, heuristics, 0, 3)

        self.assertIsNone(result)

    def test_max_expansion_limit(self):
        """Test A* respects the max_expanded limit."""
        # Long linear path
        n = 100
        neighbors = [[i + 1] if i < n - 1 else [] for i in range(n)]
        neighbors = [[i - 1] + neighbors[i] if i > 0 else neighbors[i] for i in range(n)]
        edge_costs = [[1.0] * len(neighbors[i]) for i in range(n)]
        heuristics = np.array([float(n - 1 - i) for i in range(n)])

        # With very low expansion limit, should fail to find path
        result = a_star_python.find_path(neighbors, edge_costs, heuristics, 0, n - 1, max_expanded=5)
        self.assertIsNone(result)

        # With sufficient limit, should find path
        result = a_star_python.find_path(neighbors, edge_costs, heuristics, 0, n - 1, max_expanded=200)
        self.assertIsNotNone(result)
        self.assertEqual(result.states[0], 0)
        self.assertEqual(result.states[-1], n - 1)

    def test_weighted_diamond_graph(self):
        """Test A* on a diamond graph where the optimal path is non-obvious.

        Graph:
              0
             /|\
            / | \
           1  2  3
            \ | /
             \|/
              4

        Edge costs designed so that going through node 2 (middle) looks
        attractive by heuristic but is actually more expensive.

        Costs: 0->1: 1, 0->2: 1, 0->3: 1
               1->4: 1, 2->4: 5, 3->4: 1

        Heuristics suggest 2 is closest to goal (h=1), but actual path
        through 2 costs 6, while paths through 1 or 3 cost only 2.
        """
        neighbors = [
            [1, 2, 3],  # 0 -> 1, 2, 3
            [0, 4],     # 1 -> 0, 4
            [0, 4],     # 2 -> 0, 4
            [0, 4],     # 3 -> 0, 4
            [1, 2, 3],  # 4 -> 1, 2, 3
        ]
        edge_costs = [
            [1.0, 1.0, 1.0],  # 0->1, 0->2, 0->3 all cost 1
            [1.0, 1.0],       # 1->0: 1, 1->4: 1
            [1.0, 5.0],       # 2->0: 1, 2->4: 5 (expensive!)
            [1.0, 1.0],       # 3->0: 1, 3->4: 1
            [1.0, 5.0, 1.0],  # 4->1, 4->2, 4->3
        ]
        # Deceptive heuristics: node 2 appears closer to goal
        heuristics = np.array([2.0, 1.5, 1.0, 1.5, 0.0])

        result = a_star_python.find_path(neighbors, edge_costs, heuristics, 0, 4)

        self.assertIsNotNone(result)
        # Should NOT go through node 2 (cost would be 6)
        # Should go through 1 or 3 (cost is 2)
        self.assertNotIn(2, result.states, "Should avoid expensive node 2")
        self.assertAlmostEqual(result.cost, 2.0)
        self.assertEqual(result.states[0], 0)
        self.assertEqual(result.states[-1], 4)


if __name__ == "__main__":
    unittest.main()
