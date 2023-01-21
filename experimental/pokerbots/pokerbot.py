from typing import Union
import os

from python_skeleton.skeleton import bot
from python_skeleton.skeleton.states import (
    GameState,
    RoundState,
    TerminalState,
    STARTING_STACK,
    NUM_ROUNDS,
)
from python_skeleton.skeleton.actions import (
    FoldAction,
    CheckAction,
    CallAction,
    RaiseAction,
)

from python_skeleton.skeleton import runner
import experimental.pokerbots.hand_evaluator_python as hep
import experimental.pokerbots.bin_centers_pb2
import learning.min_regret_strategy_pb2
import numpy as np

np.set_printoptions(precision=3)

runfiles_dir = os.environ.get("RUNFILES_DIR")
if runfiles_dir:
    os.chdir(os.path.join(runfiles_dir, "__main__"))


def compute_opponent_action(current_state: RoundState, player_num: int):
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

    if current_state.stacks[other_player] == 0:
        return "AllIn"

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
        return 3
    return 2


def card_idx(card_str):

    suits = "shcd"
    ranks = "23456789TJQKA"
    rank_idx = ranks.find(card_str[0])
    suit_idx = suits.find(card_str[1])
    out = rank_idx * 4 + suit_idx
    return out


def compute_infoset_id(
    action_queue, betting_round, hand, board_cards, per_turn_bin_centers
):
    infoset_id = 0
    action_dict = {
        "Fold": 1,
        "Check": 2,
        "Call": 3,
        "Raise": 4,
        "RaisePot": 5,
        "AllIn": 6,
    }
    for action in action_queue[::-1]:
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

        return [infoset_id]

    else:
        # compute the hand potential and encode it into the infoset id
        TIMEOUT_S = 0.02
        HAND_LIMIT = 1000
        MAX_ADDITIONAL_CARDS = 7
        result = hep.evaluate_strength_potential(
            "".join(hand),
            "".join(board_cards),
            MAX_ADDITIONAL_CARDS,
            TIMEOUT_S,
            HAND_LIMIT,
        )

        if betting_round == 1:
            bin_centers = per_turn_bin_centers.flop_centers
        elif betting_round == 2:
            bin_centers = per_turn_bin_centers.turn_centers
        else:
            bin_centers = per_turn_bin_centers.river_centers

        def dist_to_bin(bin_center, strength_potential):
            d_strength = bin_center.strength - strength_potential.strength
            d_neg_pot = (
                bin_center.negative_potential - strength_potential.negative_potential
            )
            d_pos_pot = (
                bin_center.positive_potential - strength_potential.positive_potential
            )
            return (
                d_strength * d_strength + d_neg_pot * d_neg_pot + d_pos_pot * d_pos_pot
            )

        nearest_bucket_idx = sorted(
            range(len(bin_centers)),
            key=lambda idx: dist_to_bin(bin_centers[idx], result),
        )

        infoset_id = infoset_id << 32
        return [(infoset_id | idx) for idx in nearest_bucket_idx]


class Pokerbot(bot.Bot):
    """First pass pokerbot."""

    def __init__(self):
        """Init."""
        with open(
            "experimental/pokerbots/pokerbot_checkpoint_031070000.pb", "rb"
        ) as file_in:
            strategy = learning.min_regret_strategy_pb2.MinRegretStrategy()
            strategy.ParseFromString(file_in.read())
        self._strategy = {x.id_num: x for x in strategy.infoset_counts}

        with open("experimental/pokerbots/bin_centers.pb", "rb") as file_in:
            self._per_turn_bin_centers = (
                experimental.pokerbots.bin_centers_pb2.PerTurnBinCenters()
            )
            self._per_turn_bin_centers.ParseFromString(file_in.read())

        self._rng = np.random.default_rng()
        self._round_state = {}
        self._should_check_fold = False

    def handle_new_round(
        self, game_state: GameState, round_state: RoundState, active
    ) -> None:
        """Handle New Round."""
        rounds_remaining = NUM_ROUNDS - game_state.round_num
        check_fold_cost_per_round = 1.5
        cost_to_check_fold = rounds_remaining * check_fold_cost_per_round + 10
        if cost_to_check_fold < game_state.bankroll:
            self._should_check_fold = True
        print(
            f"*** Round {game_state.round_num} Player: {active} BankRoll: {game_state.bankroll} Clock: {game_state.game_clock} check fold: {self._should_check_fold}"
        )

        self._round_state = {"action_queue": [], "street": 0, "have_gone_all_in": False}

    def handle_round_over(
        self, game_state: GameState, terminal_state: TerminalState, active
    ):
        """Handle Round Over."""
        print(
            f"Hands: {terminal_state.previous_state.hands} Board: {terminal_state.previous_state.deck}"
        )
        print(f"### End Round deltas: {terminal_state.deltas}")

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

        if self._round_state["have_gone_all_in"]:
            return CheckAction()

        if self._round_state["street"] != round_state.street:
            print("Street", round_state.street)
            self._round_state["action_queue"].clear()

        pot_total = my_contribution + opp_contribution
        maybe_action = compute_opponent_action(round_state, active)
        if maybe_action:
            self._round_state["action_queue"].append(maybe_action)
        betting_round = compute_betting_round(round_state)

        # Get nearest infoset ids
        infoset_ids = compute_infoset_id(
            self._round_state["action_queue"],
            betting_round,
            my_cards,
            board_cards,
            self._per_turn_bin_centers,
        )

        actions = ["Fold", "Check", "Call", "RaisePot", "AllIn"]
        strategy = None
        ids_skipped = 0
        for infoset_id in infoset_ids:
            if infoset_id not in self._strategy:
                ids_skipped += 1
                continue
            strategy_sum = np.array(self._strategy[infoset_id].strategy_sum)
            partition = np.sum(strategy_sum)
            if partition == 0.0:
                ids_skipped += 1
                continue
            print("id:", hex(infoset_id))
            if ids_skipped:
                print("Skipped", ids_skipped, "ids")
            strategy = strategy_sum / np.sum(strategy_sum)
            break

        if strategy is None:
            print("Could not find strategy!")
            strategy = np.zeros(len(actions))
            if FoldAction in legal_actions:
                strategy[0] = 0.5
            if CheckAction in legal_actions:
                strategy[1] = 1.0
            if CallAction in legal_actions:
                strategy[2] = 1.0
            if RaiseAction in legal_actions:
                if max_raise > pot_total:
                    strategy[3] = 0.5
                strategy[4] = 0.01
            strategy = strategy / np.sum(strategy)

        # Purify strategy
        strategy[strategy < 0.1] = 0.0
        strategy = strategy / np.sum(strategy)
        print(strategy)
        action_idx = self._rng.choice(len(actions), p=strategy)
        action = actions[action_idx]

        action_dict = {
            "Fold": FoldAction(),
            "Check": CheckAction(),
            "Call": CallAction(),
            "RaisePot": RaiseAction(amount=max(min_raise, pot_total)),
            "AllIn": RaiseAction(amount=max_raise),
        }
        if action == "Fold" and continue_cost == 0:
            action = "Check"
        if action == "RaisePot" and action_dict["RaisePot"].amount > max_raise:
            action = "AllIn"
        if self._should_check_fold:
            if FoldAction in legal_actions:
                action = "Fold"
            if CheckAction in legal_actions:
                action = "Check"

        print(action)

        if action == "AllIn":
            self._round_state["have_gone_all_in"] = True

        self._round_state["action_queue"].append(action)
        self._round_state["street"] = round_state.street
        print("")

        return action_dict[action]


if __name__ == "__main__":
    runner.run_bot(Pokerbot(), runner.parse_args())
    pass
