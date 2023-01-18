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
import numpy as np

import os

if os.environ["RUNFILES_DIR"]:
    os.chdir(os.path.join(os.environ["RUNFILES_DIR"], "__main__"))


def compute_opponent_action(current_state: RoundState, player_num: int):
    print(current_state, player_num)
    prev_state = current_state.previous_state
    if prev_state is None:
        # This is the first action
        return None

    other_player = 1 - player_num
    # detect fold
    # detect check
    is_new_street = current_state.street != prev_state.street
    if is_new_street and player_num == 1:
        # Player 1 starts every betting round after the first
        return None
    pip_change = current_state.pips[other_player] - prev_state.pips[other_player]
    if pip_change == 0:
        return "Check"
    elif current_state.pips[other_player] == current_state.pips[player_num]:
        return "Call"

    # detect call
    pip_diff = current_state.pips[other_player] - current_state.pips[player_num]
    previous_pot = 2 * STARTING_STACK - sum(prev_state.stacks)
    raise_fraction = pip_diff / previous_pot

    if raise_fraction < 4.0:
        return "RaisePot"
    else:
        return "AllIn"


def compute_betting_round(round_state: RoundState):
    if len(round_state.deck) == 0:
        return 0
    betting_round = len(round_state.deck) - 2
    if betting_round < 3:
        return betting_round

    last_card = round_state.deck[-1]
    if last_card[1] in "sc":
        return 4
    return 3


def card_idx(card_str):

    suits = "shcd"
    ranks = "23456789TJQKA"
    rank_idx = ranks.find(card_str[0])
    suit_idx = suits.find(card_str[1])
    out = rank_idx * 4 + suit_idx
    return out


def compute_infoset_id(action_queue, betting_round, hand, board_cards):
    infoset_id = 0
    action_dict = {
        "Fold": 1,
        "Check": 2,
        "Call": 3,
        "Raise": 4,
        "RaisePot": 5,
        "AllIn": 6,
    }
    for action in action_queue:
        infoset_id = (infoset_id << 4) | action_dict[action]

    infoset_id = (infoset_id << 8) | betting_round

    hand = sorted(hand, key=card_idx)
    # Compute the infoset id
    if betting_round == 0:
        # encode the hole card ranks and suits
        infoset_id = infoset_id << 16
        infoset_id = infoset_id | (1 << (card_idx(hand[0]) // 4))
        infoset_id = infoset_id | (1 << (card_idx(hand[1]) // 4))

        infoset_id = infoset_id << 16
        lower_suit = card_idx(hand[0]) % 4
        higher_suit = card_idx(hand[1]) % 4
        is_lower_red = (lower_suit % 2) == 1
        is_higher_red = (higher_suit % 2) == 1
        is_suit_equal = lower_suit == higher_suit

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
        TIMEOUT_S = 0.02
        result = hep.evaluate_strength_potential(
            "".join(hand), "".join(board_cards), TIMEOUT_S
        )

        if result.strength < 0.2:
            hand_strength_bin = 0
        elif result.strength < 0.4:
            hand_strength_bin = 1
        elif result.strength < 0.6:
            hand_strength_bin = 2
        elif result.strength < 0.8:
            hand_strength_bin = 3
        else:
            hand_strength_bin = 4

        if result.negative_potential < 0.1:
            negative_potential_bin = 0
        elif result.negative_potential < 0.2:
            negative_potential_bin = 1
        else:
            negative_potential_bin = 2

        if result.positive_potential < 0.1:
            positive_potential_bin = 0
        elif result.positive_potential < 0.2:
            positive_potential_bin = 1
        else:
            positive_potential_bin = 2

        infoset_id = infoset_id << 32
        infoset_id = (
            infoset_id
            | (hand_strength_bin << 8)
            | (negative_potential_bin << 4)
            | (positive_potential_bin)
        )

    return infoset_id


class Pokerbot(bot.Bot):
    """First pass pokerbot."""

    def calc_strength(self, hole: str, board: str, street: int):
        """Compute hand strength or effective hand strength post flop."""
        if street == 0:
            if self.preflop_equity is None:
                # This is the preflop round, just compute hand strength
                key = "".join(sorted(hole, key=card_idx))
                self.preflop_equity = self._preflop_equities[key]
            return self.preflop_equity
        else:
            # This is a post flop round, compute effective hand strength
            TIME_LIMIT_S = 0.02
            result = hep.evaluate_hand_potential(
                "".join(hole), "".join(board), TIME_LIMIT_S
            )
            return result.equity

    def __init__(self):
        """Init."""
        with open(
            "experimental/pokerbots/pokerbot_checkpoint_001270000.pb", "rb"
        ) as file_in:
            strategy = learning.min_regret_strategy_pb2.MinRegretStrategy()
            strategy.ParseFromString(file_in.read())
        self._strategy = {x.id_num: x for x in strategy.infoset_counts}
        self._rng = np.random.default_rng()
        self._round_state = {}

    def handle_new_round(
        self, game_state: GameState, round_state: RoundState, active
    ) -> None:
        """Handle New Round."""
        print(
            f"******************* New Round {game_state.round_num} Player: {active} Clock Remaining: {game_state.game_clock}"
        )
        self._round_state = {
            'action_queue': [],
            'street': 0
        }

    def handle_round_over(
        self, game_state: GameState, terminal_state: TerminalState, active
    ):
        """Handle Round Over."""
        print(
            f"Hands: {terminal_state.previous_state.hands} Board: {terminal_state.previous_state.deck}"
        )
        print(f"############ End Round deltas: {terminal_state.deltas}")

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

        if (
            self._round_state['street'] != round_state.street
        ):
            print('Street', round_state.street)
            self._round_data['action_queue'].clear()

        pot_total = my_contribution + opp_contribution
        maybe_action = compute_opponent_action(round_state, active)
        if maybe_action:
            print("Detected:", maybe_action)
            self._action_queue.append(maybe_action)
        print("Action Queue:", self._action_queue)
        betting_round = compute_betting_round(round_state)

        # look up the strategy
        infoset_id = compute_infoset_id(
            self._action_queue, betting_round, my_cards, board_cards
        )

        print(hex(infoset_id), infoset_id)

        actions = ["Fold", "Check", "Call", "RaisePot", "AllIn"]
        if infoset_id in self._strategy:
            strategy_sum = np.array(self._strategy[infoset_id].strategy_sum)
            strategy = strategy_sum / np.sum(strategy_sum)

        else:
            print("could not find strategy! uniform over valid actions")
            strategy = np.zeros(len(actions))
            if FoldAction in legal_actions:
                strategy[0] = 1.0
            if CheckAction in legal_actions:
                strategy[1] = 1.0
            if RaiseAction in legal_actions:
                if max_raise > pot_total:
                    strategy[3] = 1
                strategy[4] = 1
            strategy = strategy / np.sum(strategy)

        print('Strategy', strategy)
        action_idx = self._rng.choice(len(actions), p=strategy)
        action = actions[action_idx]
        self._action_queue.append(action)

        action_dict = {
            "Fold": FoldAction(),
            "Check": CheckAction(),
            "Call": CallAction(),
            "RaisePot": RaiseAction(amount=max(min_raise, pot_total)),
            "AllIn": RaiseAction(amount=max_raise),
        }
        if action == "Fold" and continue_cost == 0:
            action = "Check"
        print("Selected action", action, action_dict[action])

        self._round_state['street'] = round_state.street
        return action_dict[action]


if __name__ == "__main__":
    runner.run_bot(Pokerbot(), runner.parse_args())
    pass
