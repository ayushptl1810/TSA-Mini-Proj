from typing import List

import torch
import torch.nn as nn


class VariableAttention(nn.Module):
    """
    Context-dependent per-variable gating.
    Down-weights variables with low observation rates (e.g. Cholesterol at 0.2%)
    and up-weights informative ones.

    Input:  X (B, T, D), mask (B, T, D)
    Output: X_gated (B, T, D)
    """

    def __init__(self, n_vars: int, hidden: int = 16):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(n_vars, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_vars),
            nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        obs_rate = mask.mean(dim=1)                   # (B, D)
        attn_w   = self.attn(obs_rate).unsqueeze(1)   # (B, 1, D)
        return X * attn_w                             # (B, T, D)


class HorizonConditionedHead(nn.Module):
    """
    Single prediction head shared across all horizons, conditioned on a
    learned horizon embedding.

    Shares representation across horizons (smoother interpolation) and
    encodes the monotone risk ordering: 6h ⊂ 12h ⊂ 24h.
    """

    def __init__(self, input_dim: int, horizons: List[int], embed_dim: int = 8,
                 dropout: float = 0.3):
        super().__init__()
        self.horizons       = horizons
        self.horizon_embed  = nn.Embedding(len(horizons), embed_dim)
        self.horizon_to_idx = {h: i for i, h in enumerate(horizons)}
        self.net = nn.Sequential(
            nn.Linear(input_dim + embed_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        hidden: (B, T, H) or (B, H)
        returns: (B, T) or (B,) probability
        """
        idx     = self.horizon_to_idx[horizon]
        emb     = self.horizon_embed(torch.tensor(idx, device=hidden.device))  # (E,)
        squeeze = hidden.dim() == 2
        if squeeze:
            hidden = hidden.unsqueeze(1)  # (B, 1, H)
        emb_exp = emb.unsqueeze(0).unsqueeze(0).expand(hidden.shape[0], hidden.shape[1], -1)
        out     = self.net(torch.cat([hidden, emb_exp], dim=-1)).squeeze(-1)   # (B, T)
        return out.squeeze(1) if squeeze else out
