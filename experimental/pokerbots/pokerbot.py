from typing import Union

from python_skeleton.skeleton import bot
from python_skeleton.skeleton.states import (
    GameState,
    RoundState,
    TerminalState,
    NUM_ROUNDS,
    STARTING_STACK,
    BIG_BLIND,
    SMALL_BLIND,
)
from python_skeleton.skeleton.actions import (
    FoldAction,
    CheckAction,
    CallAction,
    RaiseAction,
)

from python_skeleton.skeleton import runner
from common.python.pybind_example_python import add as pybind_add
import experimental.pokerbots.hand_evaluator_python as hep
import random

def calc_strength(hole, board):
    '''

    '''
    result = hep.evaluate_hand(''.join(hole), 'random', ''.join(board))
    print(f'evaluated {result.num_evaluations} hands')
    return result.equity

class Pokerbot(bot.Bot):
    """First pass pokerbot."""


    def __init__(self):
        """Init."""
        print("Calling pybind add: ", pybind_add(1, 2))
        pass

    def handle_new_round(
        self, game_state: GameState, round_state: RoundState, active
    ) -> None:
        """Handle New Round."""
        print("******************* New Round")
        print("game_state:", game_state)
        print("round_state:", round_state)
        print("active:", active)
        pass

    def handle_round_over(
        self, game_state: GameState, terminal_state: TerminalState, active
    ):
        """Handle Round Over."""
        print("############ End Round")
        print("game_state:", game_state)
        print("terminal_state:", terminal_state)
        print("active:", active)
        pass

    def get_action(
        self, game_state: GameState, round_state: RoundState, active
    ) -> Union[FoldAction, CallAction, CheckAction, RaiseAction]:
        """Get action."""
        legal_actions = (
            round_state.legal_actions()
        )  # the actions you are allowed to take
        street = (
            round_state.street
        )  # int representing pre-flop, flop, turn, or river respectively
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.deck[:street]  # the board cards
        my_pip = round_state.pips[
            active
        ]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[
            1 - active
        ]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[
            1 - active
        ]  # the number of chips your opponent has remaining
        continue_cost = (
            opp_pip - my_pip
        )  # the number of chips needed to stay in the pot
        my_contribution = (
            STARTING_STACK - my_stack
        )  # the number of chips you have contributed to the pot
        opp_contribution = (
            STARTING_STACK - opp_stack
        )  # the number of chips your opponent has contributed to the pot

        (
            min_raise,
            max_raise,
        ) = (
            round_state.raise_bounds()
        )  # the smallest and largest numbers of chips for a legal bet/raise
        my_action = None

        pot_total = my_contribution + opp_contribution

        if street < 3:
            raise_amount = int(
                my_pip + continue_cost + 0.4 * (pot_total + continue_cost)
            )
        else:
            raise_amount = int(
                my_pip + continue_cost + 0.75 * (pot_total + continue_cost)
            )

        raise_amount = max([min_raise, raise_amount])
        raise_cost = raise_amount - my_pip

        if RaiseAction in legal_actions and (raise_cost <= my_stack):
            temp_action = RaiseAction(raise_amount)
        elif CallAction in legal_actions and (continue_cost <= my_stack):
            temp_action = CallAction()
        elif CheckAction in legal_actions:
            temp_action = CheckAction()
        else:
            temp_action = FoldAction()

        strength = calc_strength(my_cards, board_cards)

        if continue_cost > 0:
            scary = 0
            if continue_cost > 6:
                scary = 0.15
            if continue_cost > 12:
                scary = 0.25
            if continue_cost > 50:
                scary = 0.35
            strength = max([0, strength - scary])
            pot_odds = continue_cost / (pot_total + continue_cost)
            if strength > pot_odds:
                if random.random() < strength and strength > 0.5:
                    my_action = temp_action
                else:
                    my_action = CallAction()
            else:
                my_action = FoldAction()
        else:
            if random.random() < strength:
                my_action = temp_action
            else:
                my_action = CheckAction()
        return my_action


if __name__ == "__main__":
    runner.run_bot(Pokerbot(), runner.parse_args())
    pass
