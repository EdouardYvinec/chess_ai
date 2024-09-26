import random
from typing import Optional

import chess

from chess_ai.player.utils import ChessGame


class RandomChessGame(ChessGame):
    def __init__(self, board: Optional[chess.Board] = None) -> None:
        super().__init__(board)

    def next_move(self) -> chess.Move:
        all_legal_moves = self.legal_moves()
        move = random.choice(all_legal_moves)
        return move


def play_a_random_game() -> None:
    RandomChessGame().auto_play(game_name="random", max_iterations=1000, save_game=True)
