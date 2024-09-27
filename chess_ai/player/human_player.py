import os
import random
from typing import Dict, Optional

import cairosvg
import chess
import chess.svg as svg
import cv2
import numpy
from chess import SQUARE_NAMES, Board, Move, Square
from PIL import Image
from stockfish import Stockfish

from chess_ai.player.utils import REPO_ROOT_DIR, ChessGame


class StopGame(Exception):
    pass


class HumanChessGame(ChessGame):
    img_dim = 1080
    margin = 43

    def __init__(self, opponent: str, board: Optional[Board] = None) -> None:
        super().__init__(board)
        self.mouseX = 0
        self.mouseY = 0
        self.clicked = False
        self.opponent = opponent
        self.initialize_opponent(opponent=opponent)

    def initialize_opponent(self, opponent: str) -> None:
        if opponent == "random":
            self.next_move_to_call = self.random_next_move
        elif "stockfish" in opponent:
            self.elo = int(opponent.split()[-1][1:-1])
            self.stockfish = Stockfish()
            self.stockfish.update_engine_parameters({"UCI_Elo": self.elo, "UCI_LimitStrength": "true"})
            self.next_move_to_call = self.stockfish_next_move
        else:
            raise NotImplementedError(f"opponent {opponent} not supported yet")

    def stockfish_next_move(self) -> chess.Move:
        possible_moves = self.stockfish.get_top_moves(3)
        move_str = random.choice(possible_moves)["Move"]
        self.stockfish.make_moves_from_current_position([move_str])
        move = chess.Move.from_uci(move_str)
        return move

    def random_next_move(self) -> Move:
        all_legal_moves = self.legal_moves()  # debug
        move = random.choice(all_legal_moves)  # debug
        return move

    def next_move(self) -> Move:
        return self.next_move_to_call()

    def get_png_board(self, last_move: Optional[Move] = None, fill: Optional[Dict[Square, str]] = None) -> Image.Image:
        if fill is None:
            fill = {}
        img_path = os.path.join(REPO_ROOT_DIR, "img", "tmp.png")
        svg_str = svg.board(self.board, fill=fill, lastmove=last_move)
        cairosvg.svg2png(
            bytestring=svg_str.encode(), write_to=img_path, output_width=self.img_dim, output_height=self.img_dim
        )
        img = Image.open(img_path)
        return img

    def coords_to_board_position(self) -> str:
        x = min(max(0, self.mouseX - self.margin), self.img_dim - 2 * self.margin)
        y = min(max(0, self.mouseY - self.margin), self.img_dim - 2 * self.margin)
        row = chr(97 + int(x * 8 / (self.img_dim - 2 * self.margin)))
        col = 8 - int(y * 8 / (self.img_dim - 2 * self.margin))
        return f"{row}{col}"

    def get_next_human_move(self, last_move: Optional[Move] = None) -> Move:
        # select piece
        while not self.clicked:
            key = cv2.waitKey(1)
            if key in [27, 113]:
                raise StopGame
            if self.clicked:
                self.clicked = False
                break
        square1 = self.coords_to_board_position()
        self.show_board(fill={SQUARE_NAMES.index(square1.lower()): "#cc0000cc"}, last_move=last_move)
        # select square
        while not self.clicked:
            key = cv2.waitKey(1)
            if key in [27, 113]:
                raise StopGame
            if self.clicked:
                self.clicked = False
                break
        square2 = self.coords_to_board_position()
        self.show_board(last_move=last_move)
        # handle pawn promotion
        piece = str(self.board.piece_at(chess.parse_square(square1))).lower()
        if piece == "p" and "8" in square2:
            move = Move.from_uci(uci=f"{square1}{square2}q")
        else:
            move = Move.from_uci(uci=f"{square1}{square2}")
        return move

    def play_next_human_move(self, last_move: Optional[Move] = None) -> None:
        illegal_move_was_played = True
        while illegal_move_was_played:
            illegal_move_was_played = False
            try:
                move = self.get_next_human_move(last_move=last_move)
            except (ValueError, chess.InvalidMoveError):
                illegal_move_was_played = True
                continue
            try:
                self.play(move)
            except (ValueError, chess.InvalidMoveError):
                illegal_move_was_played = True
        if "stockfish" in self.opponent:
            self.stockfish.make_moves_from_current_position([str(move)])

    def show_board(
        self, fill: Dict[Square, str] = {}, game_ended: bool = False, last_move: Optional[Move] = None
    ) -> None:
        if game_ended:
            cv2.imshow("chess mini app", numpy.array(self.get_png_board(last_move=last_move, fill=fill)))
        else:
            cv2.imshow(
                "chess mini app",
                cv2.cvtColor(numpy.array(self.get_png_board(last_move=last_move, fill=fill)), cv2.COLOR_BGR2RGB),
            )

    def run(self) -> None:
        def mouse_coordinates(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.clicked = True
                self.mouseX, self.mouseY = x, y

        try:
            last_move = None
            cv2.namedWindow("chess mini app")
            cv2.setMouseCallback("chess mini app", mouse_coordinates)
            while not self.game_ended():
                self.show_board(last_move=last_move)
                self.play_next_human_move(last_move=last_move)
                if self.game_ended():
                    break
                last_move = self.next_move()
                self.play(last_move)
        except StopGame:
            return None
        print(self.board.outcome())
        self.show_board(game_ended=True)
        cv2.waitKey(0)
