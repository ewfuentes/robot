import unittest

from python_skeleton.skeleton.states import (
    GameState,
    RoundState,
    TerminalState,
    NUM_ROUNDS,
    STARTING_STACK,
    BIG_BLIND,
    SMALL_BLIND,
)
from experimental.pokerbots.pokerbot import Pokerbot


class PokerbotTest(unittest.TestCase):

    def test_compute_infoset_id(self):
        # Setup
        player_num = 1
        game_state = GameState(bankroll=0, game_clock=30, round_num=0)
        prev_state = RoundState(
            button=0,
            street=0,
            pips=[1, 2],
            stacks=[399, 398],
            hands=[[], ["5d", "As"]],
            deck=[],
            previous_state=None,
        )
        curr_state = RoundState(
            button=1,
            street=0,
            pips=[4, 2],
            stacks=[396, 398],
            hands=[[], ["5d", "As"]],
            deck=[],
            previous_state=prev_state,
        )

        # Action
        p = Pokerbot()
        p.handle_new_round(game_state, prev_state, player_num)
        p.get_action(game_state, curr_state, player_num)

        # Verification


if __name__ == "__main__":
    unittest.main()
