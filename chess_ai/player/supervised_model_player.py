import os
from typing import Optional

import chess
import torch

from chess_ai.player.utils import ChessGame
from chess_ai.supervised.train import ALL_PIECES, ALL_UCI, SupervisedModel, SupervisedModelTrainer


class SupervisedModelChessGame(ChessGame):
    def __init__(self, agent: SupervisedModel, board: Optional[chess.Board] = None) -> None:
        super().__init__(board)
        self.agent = agent

    def next_move(self) -> chess.Move:
        moves = self.legal_moves()
        legal_moves = self.agent.convert_legal_moves(moves).unsqueeze(0).cuda()
        board = self.agent.convert_board(str(self.board)).unsqueeze(0).cuda()
        prediction = int(self.agent(legal_moves, board).argmin(dim=-1)[0])
        return moves[prediction]


def get_agent(
    num_games: int = 50,
    max_iterations: int = 5,
    batch_size: int = 32,
    num_epochs: int = 1,
    embedding_dim: int = 8,
    num_head: int = 2,
    model_folder: str = os.path.join(os.getcwd(), "model"),
) -> SupervisedModel:
    model_path = os.path.join(model_folder, "agent.pt")
    if not os.path.exists(model_path):
        agent_trainer = SupervisedModelTrainer(model_folder=model_folder, num_games=num_games)
        agent = agent_trainer.train(
            max_iterations=max_iterations,
            batch_size=batch_size,
            num_epochs=num_epochs,
            embedding_dim=embedding_dim,
            num_head=num_head,
        )
        torch.save(agent.state_dict(), model_path)
    else:
        agent = SupervisedModel(
            moves_vocab_size=len(ALL_UCI),
            squares_vocab_size=len(ALL_PIECES),
            embedding_dim=embedding_dim,
            num_head=num_head,
        ).cuda()
        agent.load_state_dict(torch.load(model_path, weights_only=True))
    return agent


def play_a_supervised_model_game(
    model_folder: str = os.path.join(os.getcwd(), "model"),
    num_games: int = 50,
    max_iterations: int = 5,
    batch_size: int = 32,
    num_epochs: int = 1,
    embedding_dim: int = 64,
    num_head: int = 8,
) -> None:
    agent = get_agent(
        num_games=num_games,
        max_iterations=max_iterations,
        batch_size=batch_size,
        model_folder=model_folder,
        num_epochs=num_epochs,
        embedding_dim=embedding_dim,
        num_head=num_head,
    )
    SupervisedModelChessGame(agent=agent).auto_play(game_name="supervised_model", max_iterations=1000, save_game=True)
