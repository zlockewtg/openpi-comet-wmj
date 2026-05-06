"""State-history prefix-token projection for PyTorch OpenPI."""

from __future__ import annotations

import torch
from torch import nn


class StateHistoryTokenProjector(nn.Module):
    """Projects a fixed proprio/state history window into learned prefix tokens."""

    def __init__(
        self,
        embed_dim: int,
        *,
        hidden_dim: int = 256,
        window: int = 32,
        state_dim: int = 23,
        num_tokens: int = 32,
    ) -> None:
        super().__init__()
        if num_tokens != window:
            raise ValueError(f"Expected one token per history frame; got num_tokens={num_tokens}, window={window}")
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.window = window
        self.state_dim = state_dim
        self.num_tokens = num_tokens

        self.projector = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, window, embed_dim))
        nn.init.normal_(self.temporal_pos_embedding, mean=0.0, std=0.02)

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        if state_history.ndim != 3:
            raise ValueError(
                f"Expected state_history to be rank-3 [B, H, D], got shape {tuple(state_history.shape)}"
            )
        if state_history.shape[1] != self.window or state_history.shape[2] != self.state_dim:
            raise ValueError(
                f"Expected state_history shape [B, {self.window}, {self.state_dim}], "
                f"got {tuple(state_history.shape)}"
            )
        return self.projector(state_history) + self.temporal_pos_embedding
