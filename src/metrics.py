from typing import List

import math
import numpy as np
import torch

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def compute_metrics(y_true, y_score, prefix='') -> dict:
    """AUROC, AUPRC, Brier score. y_true must be binary {0, 1}."""
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return {f'{prefix}auroc': float('nan'), f'{prefix}auprc': float('nan'),
                f'{prefix}brier': float('nan')}
    return {
        f'{prefix}auroc': roc_auc_score(y_true, y_score),
        f'{prefix}auprc': average_precision_score(y_true, y_score),
        f'{prefix}brier': brier_score_loss(y_true, y_score),
    }


def c_index(risk_scores, event) -> float:
    """Harrell's C-index (vectorised)."""
    rs, ev  = np.array(risk_scores), np.array(event)
    dead    = np.where(ev == 1)[0]
    alive   = np.where(ev == 0)[0]
    if len(dead) == 0 or len(alive) == 0:
        return 0.5
    ri = rs[dead][:, None]
    rj = rs[alive][None, :]
    concordant = (ri > rj).sum()
    discordant = (ri < rj).sum()
    total      = concordant + discordant
    return float(concordant / total) if total > 0 else 0.5


def integrated_brier_score(
    survival_matrix: np.ndarray,
    y: np.ndarray,
    event_times: np.ndarray | None = None,
):
    """
    Integrated Brier Score using the proper time-varying event indicator I(T_i <= t).

    BS(t) = mean_i [ S_i(t)^2 * I(T_i <= t)  +  (1 - S_i(t))^2 * I(T_i > t) ]

    event_times: (N,) int array — last_obs_hour for dead patients, T-1 for alive.
                 If None, treats all dead patients as having event at T-1 (legacy).
    """
    N, T = survival_matrix.shape
    if event_times is None:
        # Fallback: dead patients die at T-1, alive patients survive past T-1
        event_times = np.where(y == 1, T - 1, T - 1)

    bs_per_t = []
    for t in range(T):
        # Died by time t (event observed, and event was before or at t)
        died_by_t       = ((y == 1) & (event_times <= t)).astype(float)
        # Still at risk at time t (either alive patient, or dead but event after t)
        at_risk_past_t  = 1.0 - died_by_t

        S_t  = survival_matrix[:, t]
        bs_t = np.mean(S_t ** 2 * died_by_t + (1 - S_t) ** 2 * at_risk_past_t)
        bs_per_t.append(bs_t)
    return float(np.mean(bs_per_t)), np.array(bs_per_t)


@torch.no_grad()
def evaluate_model(model, loader, horizons: List[int], device, model_type='grud') -> dict:
    """
    Compute:
      - global AUROC / AUPRC / Brier
      - per-horizon windowed AUROC / AUPRC  (soft labels binarised at >= 0.5)
      - fixed-time AUROC at t ∈ {6, 12, 24, 36}
      - C-index  (survival models)
      - Integrated Brier Score (survival models)
    """
    model.eval()
    all_y          = []
    all_p_global   = []
    all_event_times = []
    horizon_preds  = {h: [] for h in horizons}
    horizon_true   = {h: [] for h in horizons}
    survival_risk  = []
    survival_mats  = []
    QUERY_TIMES    = [6, 12, 24, 36]
    qt_preds       = {t: {h: [] for h in horizons} for t in QUERY_TIMES}
    qt_true        = {t: {h: [] for h in horizons} for t in QUERY_TIMES}

    for batch in loader:
        batch  = {
            k: v.to(device).to(torch.float32) if v.is_floating_point() else v.to(device)
            for k, v in batch.items()
        }
        out    = model(batch)
        mask_t = (batch['mask'].sum(-1) > 0)

        all_y.extend(batch['y'].cpu().numpy())

        if 'last_obs_hour' in batch:
            all_event_times.extend(batch['last_obs_hour'].cpu().numpy())

        if 'p_global' in out:
            all_p_global.extend(out['p_global'].cpu().numpy())

        if 'survival' in out:
            surv_np = out['survival'].cpu().numpy()
            survival_risk.extend(1 - surv_np[:, -1])
            survival_mats.append(surv_np)

        for i_h, h in enumerate(horizons):
            key = f'p_die_{h}h'
            if key not in out:
                continue
            pred  = out[key].cpu()
            tgt   = batch['wlabels'][:, i_h, :].cpu()
            valid = mask_t.cpu()
            horizon_preds[h].extend(pred[valid].numpy())
            horizon_true[h].extend(tgt[valid].numpy())
            for qt in QUERY_TIMES:
                if qt < pred.shape[1]:
                    qt_preds[qt][h].extend(pred[:, qt].numpy())
                    qt_true[qt][h].extend(tgt[:, qt].numpy())

    metrics = {}

    if all_p_global:
        metrics.update(compute_metrics(
            (np.array(all_y) >= 0.5).astype(int),
            np.array(all_p_global), 'global_',
        ))

    for h in horizons:
        if horizon_preds[h]:
            yt_bin = (np.array(horizon_true[h]) >= 0.5).astype(int)
            yp     = np.array(horizon_preds[h])
            metrics.update(compute_metrics(yt_bin, yp, f'h{h}_'))

    for qt in QUERY_TIMES:
        for h in horizons:
            if qt_preds[qt][h]:
                yt_bin = (np.array(qt_true[qt][h]) >= 0.5).astype(int)
                yp     = np.array(qt_preds[qt][h])
                if len(np.unique(yt_bin)) > 1:
                    metrics[f'ft_t{qt}_h{h}_auroc'] = roc_auc_score(yt_bin, yp)

    if survival_risk:
        metrics['c_index'] = c_index(np.array(survival_risk), np.array(all_y))

    if survival_mats:
        S    = np.vstack(survival_mats)
        y_np = np.array(all_y)
        # event_times: dead → last_obs_hour, alive → T-1
        if all_event_times:
            et   = np.array(all_event_times)
            et_full = np.where(y_np == 1, et, S.shape[1] - 1)
        else:
            et_full = None
        ibs, _ = integrated_brier_score(S, y_np, et_full)
        metrics['IBS'] = ibs

    return metrics
