import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CrossAttentionModel(nn.Module):
    def __init__(self, embedding_dim: int, num_head: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_head = num_head
        self.ln1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.ln2 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.q_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.k_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.v_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.out_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.up_proj = nn.Linear(in_features=embedding_dim, out_features=4 * embedding_dim)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(in_features=4 * embedding_dim, out_features=embedding_dim)

    def mhca(self, moves_embeddings: Tensor, board_embeddings: Tensor) -> Tensor:
        batch_size = moves_embeddings.size(0)
        moves_seq_length = moves_embeddings.size(1)
        board_seq_length = board_embeddings.size(1)

        q = self.q_proj(moves_embeddings)
        k = self.k_proj(board_embeddings)
        v = self.v_proj(board_embeddings)

        q = q.view(batch_size, moves_seq_length, self.num_head, self.embedding_dim // self.num_head)
        k = k.view(batch_size, board_seq_length, self.num_head, self.embedding_dim // self.num_head)
        v = v.view(batch_size, board_seq_length, self.num_head, self.embedding_dim // self.num_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k_adjusted = k.transpose(-1, -2)
        product = torch.matmul(q, k_adjusted)
        product = product / math.sqrt(self.embedding_dim)
        scores = F.softmax(product, dim=-1)

        scores = torch.matmul(scores, v)
        concat = scores.transpose(1, 2)
        concat = concat.reshape(batch_size, moves_seq_length, self.embedding_dim)
        output = self.out_proj(concat)
        return output

    def ffn(self, x: Tensor) -> Tensor:
        x = self.up_proj(x)
        x = self.act(x)
        x = self.down_proj(x)
        return x

    def forward(self, moves_embeddings: Tensor, board_embeddings: Tensor) -> Tensor:
        y = self.mhca(moves_embeddings, board_embeddings)
        x = moves_embeddings + y
        x = self.ln1(x)
        y = self.ffn(x)
        x = x + y
        x = self.ln2(x)
        return x


class SelfAttentionModel(nn.Module):
    def __init__(self, embedding_dim: int, num_head: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_head = num_head
        self.ln1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.ln2 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.q_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.k_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.v_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.out_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.up_proj = nn.Linear(in_features=embedding_dim, out_features=4 * embedding_dim)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(in_features=4 * embedding_dim, out_features=embedding_dim)

    def mhsa(self, moves_embeddings: Tensor) -> Tensor:
        batch_size = moves_embeddings.size(0)
        moves_seq_length = moves_embeddings.size(1)

        q = self.q_proj(moves_embeddings)
        k = self.k_proj(moves_embeddings)
        v = self.v_proj(moves_embeddings)

        q = q.view(batch_size, moves_seq_length, self.num_head, self.embedding_dim // self.num_head)
        k = k.view(batch_size, moves_seq_length, self.num_head, self.embedding_dim // self.num_head)
        v = v.view(batch_size, moves_seq_length, self.num_head, self.embedding_dim // self.num_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k_adjusted = k.transpose(-1, -2)
        product = torch.matmul(q, k_adjusted)
        product = product / math.sqrt(self.embedding_dim)
        scores = F.softmax(product, dim=-1)

        scores = torch.matmul(scores, v)
        concat = scores.transpose(1, 2)
        concat = concat.reshape(batch_size, moves_seq_length, self.embedding_dim)
        output = self.out_proj(concat)
        return output

    def ffn(self, x: Tensor) -> Tensor:
        x = self.up_proj(x)
        x = self.act(x)
        x = self.down_proj(x)
        return x

    def forward(self, moves_embeddings: Tensor) -> Tensor:
        y = self.mhsa(moves_embeddings)
        x = moves_embeddings + y
        x = self.ln1(x)
        y = self.ffn(x)
        x = x + y
        x = self.ln2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5_000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = pe.cuda()

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x.transpose(0, 1)
        x = x + self.pe[: x.size(0)]
        x = x.transpose(0, 1)
        return x
