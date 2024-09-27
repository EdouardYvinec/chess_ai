import contextlib
import glob
import os
import pathlib
import shutil
from typing import List, Optional

import cairosvg
import chess
import chess.svg as svg
from PIL import Image

REPO_ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.resolve())


class ChessGame:
    def __init__(self, board: Optional[chess.Board] = None) -> None:
        self.board = board if board is not None else chess.Board()
        self.img_folder = os.path.join(REPO_ROOT_DIR, "img")
        os.makedirs(self.img_folder, exist_ok=True)

    def game_ended(self) -> bool:
        return (
            self.board.is_stalemate()
            or self.board.is_checkmate()
            or self.board.is_variant_draw()
            or self.board.is_insufficient_material()
        )

    def play(self, move: chess.Move) -> None:
        if move in self.board.legal_moves:
            self.board.push(move)
        else:
            raise ValueError("illegal move")

    def legal_moves(self) -> List[chess.Move]:
        return list(self.board.legal_moves)

    def render_and_save_board(self, path: str) -> None:
        svg_str = svg.board(self.board)
        cairosvg.svg2png(
            bytestring=svg_str.encode(),
            write_to=os.path.join(self.img_folder, path),
            output_width=1080,
            output_height=1080,
        )

    def next_move(self) -> chess.Move:
        raise NotImplementedError

    def auto_play(self, game_name: str, max_iterations: int = -1, save_game: bool = False) -> None:
        cpt = 0
        while not self.game_ended():
            print(f"\rprogress (max={max_iterations}): {str(cpt).rjust(5)}", end="")
            move = self.next_move()
            self.play(move)
            if save_game:
                self.render_and_save_board(path=str(cpt).zfill(5) + ".png")
            cpt += 1
            if cpt == max_iterations:
                break
        print(f"\rprogress (max={max_iterations}): {str(cpt).rjust(5)}")
        print(self.board.outcome())
        self.reset(game_name, save_game)

    def save_to_gif(self, path: str = "game.gif") -> None:
        fp_in = os.path.join(self.img_folder, "*.png")
        with contextlib.ExitStack() as stack:
            all_files = glob.glob(fp_in)
            imgs = (stack.enter_context(Image.open(f)) for f in sorted(all_files))
            img = next(imgs)
            # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
            img.save(fp=path, format="GIF", append_images=imgs, save_all=True, duration=5 * len(all_files), loop=0)

    def reset(self, game_name: str, save_game: bool = False) -> None:
        if save_game:
            self.save_to_gif(game_name + ".gif")
        self.board = chess.Board()
        shutil.rmtree(self.img_folder)
        os.makedirs(self.img_folder, exist_ok=True)
