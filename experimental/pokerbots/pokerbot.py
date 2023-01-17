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
import learning.min_regret_strategy_pb2
import random

import os

if os.environ["RUNFILES_DIR"]:
    os.chdir(os.path.join(os.environ["RUNFILES_DIR"], "__main__"))


def compute_opponent_action(current_state: RoundState, player_num: int):
    print('Current state:', current_state)
    prev_state = current_state.previous_state
    if prev_state is None:
        # This is the first action
        return None

    other_player = 1 - player_num
    # detect fold
    # detect check
    other_player_pip_change = current_state.pips[other_player] - prev_state.pips[other_player]
    if other_player_pip_change == 0:
        return 'Check'
    elif current_state.pips[other_player] == current_state.pips[player_num]:
        return 'Call'

    # detect call
    pip_diff = current_state.pips[other_player] - current_state.pips[player_num]
    previous_pot = 2 * max(prev_state.pips)
    raise_fraction = pip_diff / previous_pot

    print(f'Raised {pip_diff} on {previous_pot}. Frac: {raise_fraction}')

    if raise_fraction < 4.0:
        return 'RaisePot'
    else:
        return 'AllIn'


def compute_betting_round(round_state: RoundState):
    street = round_state.street
    if street < 3:
        return street

    return None

def card_idx(card_str):

    suits = 'shcd'
    ranks = '23456789TJQKA'
    rank_idx = ranks.find(card_str[0])
    suit_idx = suits.find(card_str[1])
    out = rank_idx * 4 + suit_idx
    print(card_str, rank_idx, suit_idx, out)
    return out

def compute_infoset_id(action_queue, betting_round, hand, board_cards):
    infoset_id = 0
    action_dict = {
        'Fold': 1,
        'Check': 2,
        'Call': 3,
        'Raise': 4,
        'RaisePot': 5,
        'AllIn:': 6,
    }
    for action in action_queue:
        infoset_id = (infoset_id << 4) | action_dict[action]

    infoset_id = (infoset_id << 8) | betting_round

    hand = sorted(hand, key=card_idx)
    # Compute the infoset id
    if betting_round == 0:
        # encode the hole card ranks and suits
        print('current hand:', hand)
        infoset_id = infoset_id << 16
        infoset_id = infoset_id | (1 << (card_idx(hand[0]) // 4))
        infoset_id = infoset_id | (1 << (card_idx(hand[1]) // 4))

        infoset_id = infoset_id << 16
        lower_suit = card_idx(hand[0]) % 4
        higher_suit = card_idx(hand[1]) % 4
        is_lower_red = (lower_suit % 2) == 1
        is_higher_red = (higher_suit % 2) == 1
        is_suit_equal = lower_suit == higher_suit
        print('suits', [lower_suit, higher_suit], f'is lower red {is_lower_red} is higher red: {is_higher_red} is suit equal: {is_suit_equal}')

        if is_suit_equal:
            # suited red or black
            infoset_id = infoset_id | (1 if is_higher_red else 0)
        elif is_higher_red == is_lower_red:
            # off suit, but both black or red
            infoset_id = infoset_id | (3 if is_higher_red else 2)
        else:
            # off suit of different colors
            infoset_id = infoset_id | (5 if is_higher_red else 4)

    else:
        # compute the hand potential and encode it into the infoset id
        assert False, 'not implemented yet'
        pass

    return infoset_id


class Pokerbot(bot.Bot):
    """First pass pokerbot."""

    def calc_strength(self, hole: str, board: str, street: int):
        """Compute hand strength or effective hand strength post flop."""
        if street == 0:
            if self.preflop_equity is None:
                # This is the preflop round, just compute hand strength
                key = "".join(sorted(hole, key=card_idx))
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
        with open('experimental/pokerbots/pokerbot_checkpoint_10000.pb', 'rb') as file_in:
            strategy = learning.min_regret_strategy_pb2.MinRegretStrategy()
            strategy.ParseFromString(file_in.read())
        self._strategy = {x.id_num: x for x in strategy.infoset_counts}


    def handle_new_round(
        self, game_state: GameState, round_state: RoundState, active
    ) -> None:
        """Handle New Round."""
        print(
            f"******************* New Round {game_state.round_num} Player: {active} Clock Remaning: {game_state.game_clock}"
        )
        self._action_queue = []

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
        maybe_action = compute_opponent_action(round_state, active)
        if maybe_action:
            print('Detected:', maybe_action)
            self._action_queue.append(maybe_action)
        betting_round = compute_betting_round(round_state)

        # look up the strategy
        infoset_id = compute_infoset_id(self._action_queue, betting_round, my_cards, board_cards)

        print(hex(infoset_id), infoset_id)

        if infoset_id in self._strategy:
            print('Found infoset!')
            print(self._strategy[infoset_id])
        else:
            print('could not find strategy!')


        return my_action


if __name__ == "__main__":
    runner.run_bot(Pokerbot(), runner.parse_args())
    pass
