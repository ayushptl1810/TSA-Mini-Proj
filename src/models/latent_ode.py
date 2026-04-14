import torch
import torch.nn as nn
import torchdiffeq

from src.config import Config
from src.models.attention import HorizonConditionedHead


class ODEFunc(nn.Module):
    """
    f(t, z) = dz/dt — time-conditional dynamics.
    Concatenates a sinusoidal time embedding so the ODE can capture
    circadian rhythms and clinical workflow patterns.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64, time_embed_dim: int = 8):
        super().__init__()
        self.latent_dim     = latent_dim
        self.time_embed_dim = time_embed_dim
        self.time_proj      = nn.Linear(time_embed_dim, time_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.nfe = 0

    def _time_embed(self, t: torch.Tensor, batch_size: int, device) -> torch.Tensor:
        E      = self.time_embed_dim
        t_f    = t.to(torch.float32)
        steps  = torch.arange(1, E // 2 + 1, device=device, dtype=torch.float32)
        pi     = torch.tensor(3.141592653589793, device=device, dtype=torch.float32)
        pos    = t_f * steps * pi
        emb    = torch.cat([torch.sin(pos), torch.cos(pos)], dim=-1)
        return self.time_proj(emb).unsqueeze(0).expand(batch_size, -1)

    def forward(self, t, z):
        self.nfe += 1
        t_emb = self._time_embed(t, z.shape[0], z.device)
        return self.net(torch.cat([z, t_emb], dim=-1))


class RecognitionRNN(nn.Module):
    """Backward-pass encoder → (z0_mean, z0_logvar)."""

    def __init__(self, obs_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn_cell   = nn.GRUCell(obs_dim * 2, hidden_dim)
        self.to_z0      = nn.Linear(hidden_dim, latent_dim * 2)

    def forward(self, X, mask):
        B, T, D = X.shape
        h = torch.zeros(B, self.hidden_dim, device=X.device, dtype=torch.float32)
        for t in reversed(range(T)):
            inp     = torch.cat([X[:, t], mask[:, t]], dim=-1)
            h_new   = self.rnn_cell(inp, h)
            any_obs = (mask[:, t].sum(-1, keepdim=True) > 0).to(torch.float32)
            h       = any_obs * h_new + (1.0 - any_obs) * h
        z0_mean, z0_logvar = self.to_z0(h).chunk(2, dim=-1)
        return z0_mean, z0_logvar


class DeepHitHead(nn.Module):
    def __init__(self, latent_dim: int, n_times: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, z_traj):
        hazard   = self.net(z_traj).squeeze(-1)                   # (B, T)
        log_1mh  = torch.log(1 - hazard + 1e-7)
        log_surv = torch.cumsum(log_1mh, dim=-1)
        log_surv = torch.cat([
            torch.zeros(hazard.shape[0], 1, device=hazard.device),
            log_surv[:, :-1],
        ], dim=-1)
        return hazard, torch.exp(log_surv)


class LatentODESurvival(nn.Module):
    """
    Latent ODE + DeepHit survival model.

    - Time-conditional ODEFunc (circadian / workflow dynamics)
    - dopri5 adaptive solver (accurate continuous trajectories)
    - p_global = 1 - S(T-1)  (all-cause mortality probability)
    - Shared horizon-conditioned head
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg       = cfg
        L              = cfg.latent_dim
        self.encoder   = RecognitionRNN(cfg.n_vars, cfg.enc_hidden, L)
        self.ode_func  = ODEFunc(L, cfg.ode_hidden)
        self.surv_head = DeepHitHead(L, cfg.n_hours)
        self.horizon_head = HorizonConditionedHead(L, cfg.horizons, embed_dim=8)
        self.register_buffer(
            'time_pts',
            torch.linspace(0, 1, cfg.n_hours, dtype=torch.float32),
        )

    def reparameterise(self, mean, logvar):
        if self.training:
            return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return mean

    def forward(self, batch: dict) -> dict:
        X, mask = batch['X'], batch['mask']
        z0_mean, z0_logvar = self.encoder(X, mask)
        z0     = self.reparameterise(z0_mean, z0_logvar).to(device=z0_mean.device, dtype=torch.float32)
        t_pts  = self.time_pts.to(device=z0.device, dtype=torch.float32)

        self.ode_func.nfe = 0
        z_traj = torchdiffeq.odeint(
            self.ode_func, z0, t_pts,
            method='dopri5',
            rtol=1e-3,
            atol=1e-4,
            options={'dtype': torch.float32},
        ).permute(1, 0, 2)   # (B, T, L)

        hazard, survival = self.surv_head(z_traj)

        outputs = {
            'hazard'    : hazard,
            'survival'  : survival,
            'z0_mean'   : z0_mean,
            'z0_logvar' : z0_logvar,
            'nfe'       : self.ode_func.nfe,
            'p_global'  : (1 - survival[:, -1]),
        }
        for h in self.cfg.horizons:
            outputs[f'p_die_{h}h'] = self.horizon_head(z_traj, h)
        return outputs
