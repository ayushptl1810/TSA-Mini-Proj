import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config
from src.models.attention import VariableAttention, HorizonConditionedHead


class GRUDCell(nn.Module):
    """
    Single GRU-D recurrent cell.
    Input at each step: x (D), mask (D), delta (D)
    The GRU cell receives (x̃ || mask) of size 2D.
    """

    def __init__(self, input_size: int, hidden_size: int, x_mean: torch.Tensor):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.gru_cell    = nn.GRUCell(input_size * 2, hidden_size)
        self.W_gamma_x   = nn.Linear(input_size, input_size, bias=True)
        self.W_gamma_h   = nn.Linear(input_size, hidden_size, bias=True)
        self.register_buffer('x_mean', x_mean.clone())

    def forward(self, x, mask, delta, h, x_last):
        gamma_x    = torch.exp(-F.relu(self.W_gamma_x(delta)))
        gamma_h    = torch.exp(-F.relu(self.W_gamma_h(delta)))
        x_hat      = gamma_x * x_last + (1 - gamma_x) * self.x_mean.unsqueeze(0)
        x_tilde    = mask * x + (1 - mask) * x_hat
        h_tilde    = gamma_h * h
        h_new      = self.gru_cell(torch.cat([x_tilde, mask], dim=-1), h_tilde)
        x_last_new = mask * x + (1 - mask) * x_last
        return h_new, x_last_new


class GRUD(nn.Module):
    """Multi-layer GRU-D. Returns hidden_seq (B, T, H) and h_final (B, H)."""

    def __init__(self, input_size, hidden_size, n_layers, x_mean, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        dummy            = torch.zeros(hidden_size)
        self.cells = nn.ModuleList([
            GRUDCell(
                input_size if l == 0 else hidden_size,
                hidden_size,
                x_mean if l == 0 else dummy,
            )
            for l in range(n_layers)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, X, mask, delta):
        B, T, D = X.shape
        device  = X.device
        hs      = [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.n_layers)]
        x_lasts = [torch.zeros(B, self.cells[l].input_size, device=device) for l in range(self.n_layers)]
        seq = []
        for t in range(T):
            inp_x, inp_m, inp_d = X[:, t], mask[:, t], delta[:, t]
            for l, cell in enumerate(self.cells):
                h_new, x_last_new = cell(inp_x, inp_m, inp_d, hs[l], x_lasts[l])
                if l < self.n_layers - 1:
                    h_new = self.drop(h_new)
                hs[l], x_lasts[l] = h_new, x_last_new
                inp_x = h_new
                inp_m = torch.ones(B, self.hidden_size, device=device)
                inp_d = torch.zeros(B, self.hidden_size, device=device)
            seq.append(hs[-1].unsqueeze(1))
        return torch.cat(seq, dim=1), hs[-1]


class GRUDSurvivalModel(nn.Module):
    """GRU-D with variable attention + horizon-conditioned shared prediction head."""

    def __init__(self, cfg: Config, x_mean: torch.Tensor):
        super().__init__()
        self.cfg      = cfg
        self.var_attn = VariableAttention(cfg.n_vars)
        self.encoder  = GRUD(
            input_size=cfg.n_vars, hidden_size=cfg.grud_hidden,
            n_layers=cfg.grud_layers, x_mean=x_mean, dropout=cfg.grud_dropout,
        )
        H = cfg.grud_hidden
        self.horizon_head = HorizonConditionedHead(H, cfg.horizons, embed_dim=8,
                                                    dropout=cfg.grud_dropout)
        self.global_head = nn.Sequential(
            nn.Linear(H, H // 2), nn.ReLU(),
            nn.Dropout(cfg.grud_dropout),
            nn.Linear(H // 2, 1), nn.Sigmoid(),
        )

    def forward(self, batch: dict) -> dict:
        X, mask, delta       = batch['X'], batch['mask'], batch['delta']
        X_gated              = self.var_attn(X, mask)
        hidden_seq, h_final  = self.encoder(X_gated, mask, delta)
        outputs = {}
        for h in self.cfg.horizons:
            outputs[f'p_die_{h}h'] = self.horizon_head(hidden_seq, h)  # (B, T)
        outputs['p_global'] = self.global_head(h_final).squeeze(-1)    # (B,)
        return outputs
