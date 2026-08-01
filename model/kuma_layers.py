from __future__ import annotations

import torch
import torch.nn as nn


class RoPE1D(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.rotary_dim = (dim // 2) * 2
        half_dim = self.rotary_dim // 2
        frequencies = 1.0 / (base ** (torch.arange(half_dim).float() / half_dim))
        self.register_buffer("frequencies", frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, length, _ = x.shape
        positions = torch.arange(length, device=x.device, dtype=self.frequencies.dtype)
        angles = torch.einsum("i,j->ij", positions, self.frequencies)
        cosine = torch.cat((angles, angles), dim=-1).cos()[None, :, 0::2]
        sine = torch.cat((angles, angles), dim=-1).sin()[None, :, 0::2]

        rotary = x[:, :, : self.rotary_dim]
        first, second = rotary[..., 0::2], rotary[..., 1::2]
        rotated = torch.stack(
            (first * cosine - second * sine, first * sine + second * cosine),
            dim=-1,
        ).flatten(-2)
        return torch.cat((rotated, x[:, :, self.rotary_dim :]), dim=-1)


class LinearAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.usable_channels = (channels // num_heads) * num_heads
        if self.usable_channels == 0:
            raise ValueError("The number of channels must be at least the number of heads.")
        self.qk = nn.Linear(channels, channels * 2)
        self.elu = nn.ELU()
        self.rope = RoPE1D(self.usable_channels)
        self.local_enhancement = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, tokens = x.shape
        head_dim = self.usable_channels // self.num_heads
        x_tokens = x.transpose(1, 2)
        query, key = self.qk(x_tokens).chunk(2, dim=-1)
        query = self.elu(query[..., : self.usable_channels]) + 1.0
        key = self.elu(key[..., : self.usable_channels]) + 1.0

        query_rope = self.rope(query).view(
            batch_size, tokens, self.num_heads, head_dim
        ).transpose(1, 2)
        key_rope = self.rope(key).view(
            batch_size, tokens, self.num_heads, head_dim
        ).transpose(1, 2)
        query = query.view(batch_size, tokens, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, tokens, self.num_heads, head_dim).transpose(1, 2)
        value = x_tokens[..., : self.usable_channels].view(
            batch_size, tokens, self.num_heads, head_dim
        ).transpose(1, 2)

        normalizer = 1.0 / (
            query @ key.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6
        )
        key_value = (key_rope.transpose(-2, -1) * (tokens**-0.5)) @ (
            value * (tokens**-0.5)
        )
        attended = (query_rope @ key_value * normalizer).transpose(1, 2)
        attended = attended.reshape(batch_size, tokens, self.usable_channels)

        if channels > self.usable_channels:
            attended = torch.cat((attended, x_tokens[..., self.usable_channels :]), dim=-1)
        return attended.transpose(1, 2) + self.local_enhancement(x)


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, output_dim: int, channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.scale = d_model**-0.5
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear_attention = LinearAttention(channels, num_heads=4)
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.sigmoid(self.query(x) * self.key(x) * self.scale)
        value = self.value(x)
        filtered = self.dropout(weights * value + value)
        attended = self.linear_attention(x)
        return self.output_projection(self.norm(x + filtered + attended))


class LinearUp(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, lower: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(lower)) * self.projection(lower) + skip


class UMA(nn.Module):
    def __init__(self, d_model: int, channels: int) -> None:
        super().__init__()
        self.level1 = AttentionBlock(d_model, d_model, channels)
        self.level2 = AttentionBlock(d_model, d_model // 2, channels)
        self.bottom = AttentionBlock(d_model // 2, d_model // 4, channels)
        self.up2 = LinearUp(d_model // 4, d_model // 2)
        self.up1 = LinearUp(d_model // 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        level1 = self.level1(x)
        level2 = self.level2(level1)
        bottom = self.bottom(level2)
        return self.up1(self.up2(bottom, level2), level1)
