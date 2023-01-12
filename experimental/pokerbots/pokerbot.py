from typing import Union
import pickle

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
import experimental.pokerbots.hand_evaluator_python as hep
import random

import os

if os.environ["RUNFILES_DIR"]:
    os.chdir(os.path.join(os.environ["RUNFILES_DIR"], "__main__"))


def card_key(card):
    """Rank a card based on rank and suit."""
    rank = card[0]
    suit = card[1]
    rank_index = "23456789TJQKA".find(rank)
    suit_index = "shcd".find(suit)
    return rank_index * 4 + suit_index


class Pokerbot(bot.Bot):
    """First pass pokerbot."""

    def calc_strength(self, hole: str, board: str, street: int):
        """Compute hand strength or effective hand strength post flop."""
        if street == 0:
            if self.preflop_equity is None:
                # This is the preflop round, just compute hand strength
                key = "".join(sorted(hole, key=card_key))
                print("Equity:", self._preflop_equities[key])
                self.preflop_equity = self._preflop_equities[key]
            return self.preflop_equity
        else:
            # This is a post flop round, compute effective hand strength
            TIME_LIMIT_S = 0.02
            result = hep.evaluate_hand_potential(
                "".join(hole), "".join(board), TIME_LIMIT_S
            )
            print(f"evaluated {result.num_evaluations} hands")
            return result.equity

    def __init__(self):
        """Init."""
        with open("experimental/pokerbots/preflop_equities.p", "rb") as file_in:
            self._preflop_equities = pickle.load(file_in)

    def handle_new_round(
        self, game_state: GameState, round_state: RoundState, active
    ) -> None:
        """Handle New Round."""
        print(
            f"******************* New Round {game_state.round_num} Player: {active} Clock Remaning: {game_state.game_clock}"
        )
        self.preflop_equity = None
        # print("game_state:", game_state)
        # print("round_state:", round_state)
        # print("active:", active)
        # pass

    def handle_round_over(
        self, game_state: GameState, terminal_state: TerminalState, active
    ):
        """Handle Round Over."""
        print(f"############ End Round deltas: {terminal_state.deltas}")
        print(
            f"Hands: {terminal_state.previous_state.hands} Board: {terminal_state.previous_state.deck}"
        )
        # print("game_state:", game_state)
        # print("terminal_state:", terminal_state)
        # print("active:", active)
        # pass

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

        strength = self.calc_strength(my_cards, board_cards, street)

        if continue_cost > 0:
            pot_odds = continue_cost / (pot_total + continue_cost)
            if strength > pot_odds:
                if random.random() < strength:
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
