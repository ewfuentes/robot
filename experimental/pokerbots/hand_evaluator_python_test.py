import unittest

import experimental.pokerbots.hand_evaluator_python as hep


class HandEvaluatorPythonTest(unittest.TestCase):
    def test_monte_carlo_sim(self):
        # Setup
        player_hand = "AsAc"
        board = "AdAhKs"
        opponent_hand = ""

        # Action
        result = hep.evaluate_expected_strength(player_hand, opponent_hand, board, 0.05)

        # Verification
        print("Player Equity:", result.strength)
        print("hands evaluated:", result.num_evaluations)

    def test_hand_potential(self):
        # Setup
        player_hand = "AsAc"
        board = "KsQcJd"

        # Action
        hep.evaluate_strength_potential(player_hand, board, timeout_s=0.005)

        # Verification

    def test_hand_potential_flush_draw(self):
        # Setup
        player_hand = "AsAc"
        board = "KcQcJc"

        # Action
        hep.evaluate_strength_potential(player_hand, board, timeout_s=0.005)

        # Verification


if __name__ == "__main__":
    unittest.main()
