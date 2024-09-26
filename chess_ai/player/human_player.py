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

from chess_ai.player.utils import ChessGame


class HumanChessGame(ChessGame):

    def __init__(self, board: Optional[Board] = None) -> None:
        super().__init__(board)
        self.mouseX = 0
        self.mouseY = 0
        self.clicked = False

    def next_move(self) -> Move:
        all_legal_moves = self.legal_moves()  # debug
        move = random.choice(all_legal_moves)  # debug
        return move

    def get_png_board(self, fill: Optional[Dict[Square, str]] = None) -> Image.Image:
        if fill is None:
            fill = {}
        img_path = os.path.join(os.getcwd(), "img", "tmp.png")
        svg_str = svg.board(self.board, fill=fill)
        cairosvg.svg2png(bytestring=svg_str.encode(), write_to=img_path, output_width=1080, output_height=1080)
        img = Image.open(img_path)
        return img

    def coords_to_board_position(self) -> str:
        row = chr(97 + int(self.mouseX * 8 / 1080))
        col = 8 - int(self.mouseY * 8 / 1080)
        return f"{row}{col}"

    def get_next_human_move(self) -> Move:
        # select piece
        while not self.clicked:
            cv2.waitKey(1)
            if self.clicked:
                self.clicked = False
                break
        square1 = self.coords_to_board_position()
        self.show_board(fill={SQUARE_NAMES.index(square1.lower()): "#cc0000cc"})
        # select square
        while not self.clicked:
            cv2.waitKey(1)
            if self.clicked:
                self.clicked = False
                break
        square2 = self.coords_to_board_position()
        self.show_board()
        # handle pawn promotion
        piece = str(self.board.piece_at(chess.parse_square(square1))).lower()
        if piece == "p" and "8" in square2:
            move = Move.from_uci(uci=f"{square1}{square2}q")
        else:
            move = Move.from_uci(uci=f"{square1}{square2}")
        return move

    def play_next_human_move(self) -> None:
        illegal_move_was_played = True
        while illegal_move_was_played:
            illegal_move_was_played = False
            try:
                move = self.get_next_human_move()
            except (ValueError, chess.InvalidMoveError):
                illegal_move_was_played = True
                continue
            try:
                self.play(move)
            except (ValueError, chess.InvalidMoveError):
                print(f"illegal move: {move}")
                illegal_move_was_played = True
        print(move)

    def show_board(self, fill: Dict[Square, str] = {}) -> None:
        cv2.imshow("chess mini app", numpy.array(self.get_png_board(fill=fill)))

    def run(self) -> None:
        def mouse_coordinates(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.clicked = True
                self.mouseX, self.mouseY = x, y

        cv2.namedWindow("chess mini app")
        cv2.setMouseCallback("chess mini app", mouse_coordinates)
        while not self.game_ended():
            self.show_board()
            self.play_next_human_move()
            if self.game_ended():
                break
            move = self.next_move()
            self.play(move)
        print(self.board.outcome())
        self.show_board()
        cv2.waitKey(0)
