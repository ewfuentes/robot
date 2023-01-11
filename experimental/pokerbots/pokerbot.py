from typing import Union

import sys

print(sys.version)

from python_skeleton.skeleton import bot
from python_skeleton.skeleton.states import GameState, RoundState, TerminalState
from python_skeleton.skeleton.actions import (
    FoldAction,
    CheckAction,
    CallAction,
    RaiseAction,
)
from python_skeleton.skeleton import runner


class Pokerbot(bot.Bot):
    """First pass pokerbot."""

    def __init__(self):
        """Init."""
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
        legal_actions = round_state.legal_actions()
        if CheckAction in legal_actions:
            return CheckAction()
        return CallAction()


if __name__ == "__main__":
    runner.run_bot(Pokerbot(), runner.parse_args())
    pass
