import glob
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from chess import PIECE_SYMBOLS, SQUARE_NAMES, Move
from loadingpy import PyBar
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Dataset

from chess_ai.player.stockfish_player import StockfishChessGame
from chess_ai.supervised.transformer_utils import CrossAttentionModel, PositionalEncoding, SelfAttentionModel

ALL_UCI: List[str] = []
for piece_symbol in PIECE_SYMBOLS:
    if piece_symbol is not None:
        for square_name in SQUARE_NAMES:
            ALL_UCI.append(piece_symbol.upper() + "@" + square_name)
for square_name1 in SQUARE_NAMES:
    for square_name2 in SQUARE_NAMES:
        ALL_UCI.append(square_name1 + square_name2)
        for piece_symbol in PIECE_SYMBOLS:
            if piece_symbol is not None:
                ALL_UCI.append(square_name1 + square_name2 + piece_symbol)

ALL_PIECES: List[str] = [".", "p", "n", "b", "r", "q", "k", "P", "N", "B", "R", "Q", "K"]


@dataclass
class ChessMoveSample:
    ground_truth: Move
    legal_moves: List[Move]
    board: str


class Word2Vec(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.proj = nn.Linear(in_features=embedding_dim, out_features=32)
        self.act = nn.ReLU()
        self.head = nn.Linear(in_features=32, out_features=vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        x = self.proj(x)
        x = self.act(x)
        x = self.head(x)
        return x


class SupervisedModel(nn.Module):
    def __init__(self, moves_vocab_size: int, squares_vocab_size: int, embedding_dim: int, num_head: int) -> None:
        super().__init__()
        self.uci_embed = nn.Embedding(num_embeddings=moves_vocab_size, embedding_dim=embedding_dim)
        self.squares_embed = nn.Embedding(num_embeddings=squares_vocab_size, embedding_dim=embedding_dim)

        self.cross_att = CrossAttentionModel(embedding_dim=embedding_dim, num_head=num_head)
        self.self_att = SelfAttentionModel(embedding_dim=embedding_dim, num_head=num_head)

        self.pos = PositionalEncoding(d_model=embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim).cuda())

    def convert_move(self, move: str) -> Tensor:
        return torch.tensor(ALL_UCI.index(move))

    def convert_legal_moves(self, legal_moves: List[Move]) -> Tensor:
        output = torch.stack([self.convert_move(move.uci()) for move in legal_moves])
        return output

    def convert_board(self, board: str) -> Tensor:
        output = torch.stack([torch.tensor(ALL_PIECES.index(piece)) for piece in board.split()])
        return output

    def add_cls_token(self, moves_embeddings: Tensor) -> Tensor:
        return torch.concat([self.cls_token.repeat([moves_embeddings.shape[0], 1, 1]), moves_embeddings], dim=1)

    def forward(self, legal_moves: Tensor, board: Tensor) -> Tensor:
        moves_embeddings: Tensor = self.uci_embed(legal_moves)
        board_embeddings: Tensor = self.squares_embed(board)

        board_embeddings = self.pos(board_embeddings)
        _moves_embeddings = self.add_cls_token(moves_embeddings)
        x = self.cross_att(_moves_embeddings, board_embeddings)
        x = self.self_att(x)[:, 0].unsqueeze(1)

        return self.head(x, moves_embeddings)

    def head(self, prediction: Tensor, moves_embeddings: Tensor) -> Tensor:
        distance = torch.norm(moves_embeddings - prediction, dim=-1)
        return distance


class SupervisedModelDataset(Dataset):
    def __init__(self, data_folder: str, agent: SupervisedModel) -> None:
        super().__init__()
        self.all_files = glob.glob(os.path.join(data_folder, "*.pt"))
        self.agent = agent

    @staticmethod
    def collate_fn(x: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        pad_size = max([e["legal_moves"].shape[0] for e in x])
        output: Dict[str, Tensor] = {
            "board": torch.stack([e["board"] for e in x]),
            "legal_moves": torch.stack(
                [F.pad(e["legal_moves"], pad=[0, pad_size - e["legal_moves"].shape[0]]) for e in x]
            ),
            "ground_truth": torch.stack([e["ground_truth"] for e in x]),
        }
        return output

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        data: ChessMoveSample = torch.load(self.all_files[index], weights_only=False)
        board = self.agent.convert_board(data.board)
        ground_truth = torch.tensor(data.legal_moves.index(data.ground_truth)).long()
        legal_moves = self.agent.convert_legal_moves(data.legal_moves)
        return {"board": board, "legal_moves": legal_moves, "ground_truth": ground_truth}


class SupervisedModelTrainer:
    def __init__(
        self,
        model_folder: str = os.path.join(os.getcwd(), "model"),
        data_folder: str = os.path.join(os.getcwd(), "train_set"),
        num_games: int = 50,
    ) -> None:
        self.data_folder = data_folder
        self.model_folder = model_folder
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)
        self.num_games = num_games

    def observe_a_game(self, max_iterations: int = 5) -> List[ChessMoveSample]:
        data: List[ChessMoveSample] = []
        stockfish_game = StockfishChessGame(play_best=True)
        cpt = 0
        while not stockfish_game.game_ended():
            move = stockfish_game.next_move()
            if cpt % 2 == 0:
                data.append(
                    ChessMoveSample(
                        ground_truth=move, legal_moves=stockfish_game.legal_moves(), board=str(stockfish_game.board)
                    )
                )
            stockfish_game.play(move)
            cpt += 1
            if cpt == max_iterations:
                break
        return data

    def get_num_data_samples(self) -> int:
        all_files = glob.glob(os.path.join(self.data_folder, "*.pt"))
        cpt = len(all_files)
        return cpt

    def save_game_actions(self, data: List[ChessMoveSample]) -> None:
        cpt = self.get_num_data_samples()
        for idx, sample in enumerate(data):
            torch.save(sample, os.path.join(self.data_folder, str(cpt + idx).zfill(6) + ".pt"))

    def create_training_set(self, max_iterations: int = 5) -> None:
        if self.get_num_data_samples() == 0:
            for _ in PyBar(range(self.num_games), base_str="extract training set"):
                data = self.observe_a_game(max_iterations=max_iterations)
                self.save_game_actions(data=data)

    def train_word2vec(self, vocab_size: int, embedding_dim: int) -> Word2Vec:
        batch_size = 64
        word2vec = Word2Vec(vocab_size=vocab_size, embedding_dim=embedding_dim).cuda()
        optimizer = Adam(params=word2vec.parameters())
        loss_fn = nn.CrossEntropyLoss()
        pbar = PyBar(range(10_000), base_str="train word2vec")
        for _ in pbar:
            optimizer.zero_grad()
            x = torch.randint(low=0, high=vocab_size, size=(batch_size,)).cuda()
            y = word2vec(x)
            loss: Tensor = loss_fn(y, x)
            loss.backward()
            pbar.monitoring = f"{loss:.5f}".rjust(8)
            optimizer.step()
        return word2vec

    def get_word2vec(self, vocab_size: int, embedding_dim: int, name: str) -> Tensor:
        embedding_path = os.path.join(self.model_folder, name)
        if not os.path.exists(embedding_path):
            word2vec = self.train_word2vec(vocab_size=vocab_size, embedding_dim=embedding_dim)
            torch.save(word2vec.embed.weight.data, embedding_path)
            return word2vec.embed.weight.data
        return torch.load(embedding_path, weights_only=False)

    def get_agent(self, num_head: int, embedding_dim: int) -> SupervisedModel:
        uci_embed_weight = self.get_word2vec(
            vocab_size=len(ALL_UCI), embedding_dim=embedding_dim, name="uci_embedding.pt"
        )
        squares_embed_weight = self.get_word2vec(
            vocab_size=len(ALL_PIECES), embedding_dim=embedding_dim, name="squares_embedding.pt"
        )
        agent = SupervisedModel(
            moves_vocab_size=len(ALL_UCI),
            squares_vocab_size=len(ALL_PIECES),
            embedding_dim=embedding_dim,
            num_head=num_head,
        )
        agent.uci_embed.weight.data = uci_embed_weight
        agent.squares_embed.weight.data = squares_embed_weight
        agent.uci_embed.requires_grad_(False)
        agent.squares_embed.requires_grad_(False)
        return agent.cuda()

    def train(
        self,
        max_iterations: int = 5,
        batch_size: int = 32,
        num_epochs: int = 1,
        embedding_dim: int = 8,
        num_head: int = 2,
    ) -> SupervisedModel:
        self.create_training_set(max_iterations=max_iterations)
        print(f"we have {self.get_num_data_samples()} training samples.")
        agent = self.get_agent(embedding_dim=embedding_dim, num_head=num_head)
        dataloader = DataLoader(
            SupervisedModelDataset(data_folder=self.data_folder, agent=agent),
            batch_size=batch_size,
            collate_fn=SupervisedModelDataset.collate_fn,
        )
        optimizer = Adam(params=[p for p in agent.parameters() if p.requires_grad], lr=5e-3)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            pbar = PyBar(dataloader, base_str=f"epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                optimizer.zero_grad()
                prediction: Tensor = agent(batch["legal_moves"].cuda(), batch["board"].cuda())
                ground_truth = batch["ground_truth"].cuda()
                loss: Tensor = loss_fn(-prediction, ground_truth)
                loss.backward()
                pbar.monitoring = f"{loss:.5f}".rjust(8)
                optimizer.step()
        raise ValueError  # debug
        return agent
