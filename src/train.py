from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.config import Config
from src.losses import focal_loss, windowed_survival_loss, deephit_loss, kl_divergence


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = 'max', delta: float = 1e-4):
        self.patience   = patience
        self.mode       = mode
        self.delta      = delta
        self.best       = -np.inf if mode == 'max' else np.inf
        self.counter    = 0
        self.best_state = None

    def step(self, metric: float, model: nn.Module) -> bool:
        improved = (
            metric > self.best + self.delta if self.mode == 'max'
            else metric < self.best - self.delta
        )
        if improved:
            self.best       = metric
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def load_best(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


def _to_device(batch: dict, device) -> dict:
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            # Force float32 for all data tensors fed to model, plus mask/delta
            if torch.is_floating_point(v) or k in ['mask', 'delta']:
                v = v.to(torch.float32)
            new_batch[k] = v
        elif isinstance(v, dict):
            # Handle windowed labels dict
            new_batch[k] = {tk: tv.to(device).to(torch.float32) for tk, tv in v.items()}
        else:
            new_batch[k] = v
    return new_batch


def train_grud_epoch(model, loader, optim, cfg: Config, device) -> dict:
    model.train()
    total, n = 0.0, 0
    logs = {}
    for batch in loader:
        batch  = _to_device(batch, device)
        optim.zero_grad()
        out    = model(batch)
        loss_w, log_w = windowed_survival_loss(out, batch['wlabels'], batch['mask'], cfg.horizons)
        loss_g = focal_loss(out['p_global'], batch['y'])
        loss   = loss_w + 0.3 * loss_g
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        optim.step()
        total += loss.item()
        n     += 1
        for k, v in log_w.items():
            logs[k] = logs.get(k, 0) + v
    return {'train_loss': total / n, **{k: v / n for k, v in logs.items()}}


def train_ode_epoch(model, loader, optim, cfg: Config, device,
                    kl_weight: float = 1e-3) -> dict:
    model.train()
    total, n  = 0.0, 0
    logs      = {}
    nfe_total = 0
    for batch in loader:
        batch  = _to_device(batch, device)
        optim.zero_grad()
        out    = model(batch)
        loss_dh, log_dh = deephit_loss(
            out['hazard'], out['survival'], batch['y'],
            alpha=cfg.deephit_alpha, sigma=cfg.deephit_sigma,
        )
        loss_w, log_w = windowed_survival_loss(out, batch['wlabels'], batch['mask'], cfg.horizons)
        loss_kl = kl_divergence(out['z0_mean'], out['z0_logvar'])
        loss    = loss_dh + loss_w + kl_weight * loss_kl
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        optim.step()
        total     += loss.item()
        nfe_total += out.get('nfe', 0)
        n         += 1
        for k, v in {**log_dh, **log_w}.items():
            logs[k] = logs.get(k, 0) + v
    return {
        'train_loss': total / n,
        'train_kl'  : loss_kl.item(),
        'avg_nfe'   : nfe_total / n,
        **{k: v / n for k, v in logs.items()},
    }


def kl_anneal(epoch: int, warmup: int = 20, max_weight: float = 1e-3) -> float:
    return min(epoch / warmup, 1.0) * max_weight
