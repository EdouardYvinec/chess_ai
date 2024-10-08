import argparse
import atexit
import os
import shutil

from chess_ai.player.human_player import HumanChessGame
from chess_ai.player.random_player import play_a_random_game
from chess_ai.player.select import OpponentSelector
from chess_ai.player.stockfish_player import play_a_stockfish_game
from chess_ai.player.supervised_model_player import play_a_supervised_model_game
from chess_ai.player.utils import REPO_ROOT_DIR


def remove_cache_folders(current_repo: str = REPO_ROOT_DIR) -> None:
    """Removes all __pycache__ directories in the module.

    Args:
        current_repo: current folder to clean recursively
    """
    new_refs = [current_repo + "/" + elem for elem in os.listdir(current_repo)]
    for elem in new_refs:
        if os.path.isdir(elem):
            if "__pycache__" in elem:
                shutil.rmtree(elem)
            else:
                remove_cache_folders(current_repo=elem)


def main() -> None:
    atexit.register(remove_cache_folders)
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true", default=False)
    parser.add_argument("--supervised", action="store_true", default=False)
    parser.add_argument("--stockfish", type=int, default=-1)
    args = parser.parse_args()
    if args.supervised:
        play_a_supervised_model_game(
            num_games=1_000, max_iterations=100, batch_size=32, num_epochs=50, embedding_dim=64, num_head=8
        )
    elif args.random:
        play_a_random_game()
    elif args.stockfish >= 100:
        play_a_stockfish_game(elo=args.stockfish)
    else:
        opponent = OpponentSelector().select()
        if opponent != "":
            HumanChessGame(opponent=opponent).run()


if __name__ == "__main__":
    main()
