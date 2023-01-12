
import unittest

import experimental.pokerbots.hand_evaluator_python as hep

class HandEvaluatorPythonTest(unittest.TestCase):
    def test_monte_carlo_sim(self):
        # Setup
        player_hand = 'AsAc'
        board = 'AdAhKs'
        opponent_hand = ''

        # Action
        result = hep.evaluate_hand(player_hand, opponent_hand, board, 0.05)

        # Verification
        print('Player Equity:', result.equity)
        print('hands evaluated:', result.num_evaluations)



if __name__ == "__main__":
    unittest.main();
