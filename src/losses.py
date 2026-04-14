from typing import List, Tuple

import torch
import torch.nn.functional as F


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.85,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Focal loss — handles class imbalance (~14% positive)."""
    bce   = F.binary_cross_entropy(pred, target, reduction='none')
    pt    = torch.where(target == 1, pred, 1 - pred)
    focal = alpha * torch.where(target == 1, (1 - pt) ** gamma, torch.tensor(1.0)) * bce
    return focal.mean() if reduction == 'mean' else focal


def windowed_survival_loss(
    outputs: dict,
    wlabels: torch.Tensor,   # (B, n_horizons, T)
    mask: torch.Tensor,      # (B, T, D)
    horizons: List[int],
    gamma: float = 2.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Sum of focal losses across horizons.
    Loss is computed only at timesteps where at least one variable was observed.
    """
    total    = torch.tensor(0.0, device=wlabels.device)
    log      = {}
    obs_at_t = (mask.sum(dim=-1) > 0).float()  # (B, T)

    for i, h in enumerate(horizons):
        key = f'p_die_{h}h'
        if key not in outputs:
            continue
        pred = outputs[key]        # (B, T)
        tgt  = wlabels[:, i, :]   # (B, T)
        l_pt = focal_loss(pred, tgt, gamma=gamma, reduction='none')
        l_pt = (l_pt * obs_at_t).sum() / (obs_at_t.sum() + 1e-6)
        total = total + l_pt
        log[f'loss_horizon_{h}h'] = l_pt.item()

    return total, log


def deephit_loss(
    hazard: torch.Tensor,       # (B, T)
    survival: torch.Tensor,     # (B, T)
    y: torch.Tensor,            # (B,) binary
    event_time: torch.Tensor,   # (B,) long — last obs hour (used when y=1)
    alpha: float = 0.2,
    sigma: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """
    L_total = L_nll + alpha * L_rank

    Event times:
      - Dead   (y=1): event at last_obs_hour  → -log [S(t_d) * h(t_d)]
      - Alive  (y=0): censored at T-1         → -log S(T-1)
    """
    EPS = 1e-7
    B   = y.shape[0]
    dev = survival.device

    # For dead patients use actual event time; for alive use final timestep
    t_idx  = torch.where(y.bool(), event_time.to(dev), torch.full((B,), survival.shape[1] - 1, device=dev))
    arange = torch.arange(B, device=dev)

    surv_at_t = survival[arange, t_idx]
    haz_at_t  = hazard[arange, t_idx]

    log_p_event  = torch.log(surv_at_t * haz_at_t + EPS)
    log_p_censor = torch.log(surv_at_t + EPS)
    l_nll = -(y * log_p_event + (1 - y) * log_p_censor).mean()

    # Ranking loss: higher risk for dead vs alive at their respective event times
    risk    = 1 - surv_at_t
    diff    = risk.unsqueeze(1) - risk.unsqueeze(0)
    eta_ij  = y.unsqueeze(1) * (1 - y.unsqueeze(0))
    l_rank  = (eta_ij * torch.exp(-diff / sigma)).sum()
    n_pairs = eta_ij.sum().clamp(min=1)
    l_rank  = l_rank / n_pairs

    total = l_nll + alpha * l_rank
    return total, {'loss_nll': l_nll.item(), 'loss_rank': l_rank.item()}


def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1).mean()
