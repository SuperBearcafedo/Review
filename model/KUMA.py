from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.kuma_layers import UMA


class InvertedEmbedding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.projection(x.transpose(1, 2)))


class KoopmanDynamicModule(nn.Module):
    def __init__(self, d_model: int, decay_ratio: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.rank = d_model // decay_ratio
        self.proj_u = nn.Linear(d_model, self.rank * d_model)
        self.proj_v = nn.Linear(d_model, self.rank * d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, tokens, dimensions = x.shape
        basis = self.proj_u(x).view(batch_size, tokens, dimensions, self.rank)
        coordinates = self.proj_v(x).view(batch_size, tokens, dimensions, self.rank)
        weights = (coordinates * x.unsqueeze(-1)).sum(dim=2) / (dimensions**0.5)
        weights = torch.softmax(weights, dim=-1)
        dominant = (weights.unsqueeze(2) * basis).sum(dim=-1)
        dominant = self.dropout(self.activation(dominant))
        return self.norm(dominant + x)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x) + x
        residual = x = self.norm1(x)
        x = self.activation(self.conv1(x.transpose(1, 2)))
        x = self.conv2(x).transpose(1, 2)
        return self.norm2(residual + x)


class Encoder(nn.Module):
    def __init__(self, layers: list[EncoderLayer], d_model: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.embedding = InvertedEmbedding(configs.seq_len, configs.d_model, configs.dropout)
        self.koopman = KoopmanDynamicModule(configs.d_model, decay_ratio=4, dropout=configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    attention=UMA(configs.d_model, configs.enc_in),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            configs.d_model,
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            centered = x - means
            stdev = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = centered / stdev

        tokens = self.embedding(x)
        koopman_dynamics = self.koopman(tokens)
        residual_dynamics = tokens - koopman_dynamics
        residual_dynamics = self.encoder(residual_dynamics)
        output = self.projector(residual_dynamics + koopman_dynamics).transpose(1, 2)

        if self.use_norm:
            output = output * stdev + means
        return output[:, -self.pred_len :, :]
