import random
from typing import Optional

import chess
from stockfish import Stockfish

from chess_ai.player.utils import ChessGame


class StockfishChessGame(ChessGame):
    def __init__(self, board: Optional[chess.Board] = None, elo: int = 2800, play_best: bool = False) -> None:
        super().__init__(board)
        self.cpt_move = 0
        self.play_best = play_best
        self.elo = elo
        self.stockfish = Stockfish()
        self.stockfish.update_engine_parameters({"Hash": 2048, "UCI_Elo": elo, "UCI_LimitStrength": "true"})

    def next_move(self) -> chess.Move:
        if self.play_best and (self.cpt_move % 2 == 0):
            move_str = self.stockfish.get_best_move()
        else:
            possible_moves = self.stockfish.get_top_moves(3)
            move_str = random.choice(possible_moves)["Move"]
        self.stockfish.make_moves_from_current_position([move_str])
        move = chess.Move.from_uci(move_str)
        self.cpt_move += 1
        return move


def play_a_stockfish_game(elo: int = 2800) -> None:
    StockfishChessGame(elo=elo).auto_play(game_name=f"stockfish (elo={elo})", max_iterations=500, save_game=True)
