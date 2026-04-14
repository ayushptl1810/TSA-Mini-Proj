"""
All post-training plots and comparison tables.
"""
import math
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.calibration import calibration_curve

from src.metrics import integrated_brier_score
from src.config import PLOTS_DIR
plt.switch_backend('Agg')


# ── comparison tables ─────────────────────────────────────────────────────────

def print_metric_table(grud_metrics: dict, ode_metrics: dict):
    metrics_to_show = [
        ('Global AUROC',                  'global_auroc'),
        ('Global AUPRC',                  'global_auprc'),
        ('Global Brier',                  'global_brier'),
        ('6h Risk AUROC (windowed)',       'h6_auroc'),
        ('6h Risk AUPRC (windowed)',       'h6_auprc'),
        ('6h Risk Brier (windowed)',       'h6_brier'),
        ('Fixed t=6,  6h Risk AUROC',     'ft_t6_h6_auroc'),
        ('Fixed t=12, 6h Risk AUROC',     'ft_t12_h6_auroc'),
        ('Fixed t=24, 6h Risk AUROC',     'ft_t24_h6_auroc'),
        ('Fixed t=36, 6h Risk AUROC',     'ft_t36_h6_auroc'),
        ('C-index',                       'c_index'),
        ('IBS',                           'IBS'),
    ]
    rows = []
    for label, key in metrics_to_show:
        g = grud_metrics.get(key, float('nan'))
        o = ode_metrics.get(key, float('nan'))
        rows.append({'Metric': label, 'GRU-D': g, 'Latent ODE + DeepHit': o})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format='{:.4f}'.format))
    print('\nFixed-time AUROC = AUROC at a specific ICU hour, not aggregated over all timesteps.')


def print_full_comparison(lr_metrics: dict, grud_metrics: dict, ode_metrics: dict):
    rows = [
        ('LR (first-24h mean)',  lr_metrics.get('lr_auroc', np.nan),       lr_metrics.get('lr_auprc', np.nan)),
        ('GRU-D',                grud_metrics.get('global_auroc', np.nan),  grud_metrics.get('global_auprc', np.nan)),
        ('Latent ODE + DeepHit', ode_metrics.get('global_auroc', np.nan),   ode_metrics.get('global_auprc', np.nan)),
    ]
    df = pd.DataFrame(rows, columns=['Model', 'AUROC', 'AUPRC'])
    print('\nFull comparison (global mortality prediction):')
    print(df.to_string(index=False, float_format='{:.4f}'.format))
    print('\n→ LR uses only first 24h — fair comparison with no outcome leakage.')


# ── calibration curves ────────────────────────────────────────────────────────

@torch.no_grad()
def _get_global_preds(model, loader, device):
    model.eval()
    all_p, all_y = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out   = model(batch)
        if 'p_global' in out:
            all_p.extend(out['p_global'].cpu().numpy())
            all_y.extend(batch['y'].cpu().numpy())
    return np.array(all_y), np.array(all_p)


def plot_calibration(grud_model, ode_model, test_loader, device,
                     y_test_lr, lr_probs, save_path=PLOTS_DIR / 'calibration_curves.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Calibration Curves — Global Mortality Prediction', fontsize=12)

    models_to_plot = [('LR (static)', y_test_lr, lr_probs)]
    gy, gp = _get_global_preds(grud_model, test_loader, device)
    if len(gp) > 0:
        models_to_plot.append(('GRU-D', gy, gp))
    oy, op = _get_global_preds(ode_model, test_loader, device)
    if len(op) > 0:
        models_to_plot.append(('Latent ODE', oy, op))

    for ax, (name, y_t, y_p) in zip(axes, models_to_plot):
        if len(np.unique(y_t)) < 2:
            ax.text(0.5, 0.5, 'N/A', ha='center')
            continue
        frac_pos, mean_pred = calibration_curve(y_t, y_p, n_bins=10, strategy='quantile')
        ax.plot(mean_pred, frac_pos, 'o-', color='steelblue', lw=2, label=name)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── IBS + mean survival curves ────────────────────────────────────────────────

@torch.no_grad()
def _collect_survival_matrix(model, loader, device):
    model.eval()
    surv_list, y_list, et_list = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out   = model(batch)
        if 'survival' in out:
            surv_list.append(out['survival'].cpu().numpy())
            y_list.extend(batch['y'].cpu().numpy())
            if 'last_obs_hour' in batch:
                et_list.extend(batch['last_obs_hour'].cpu().numpy())
    if not surv_list:
        return None, None, None
    S = np.vstack(surv_list)
    y = np.array(y_list)
    # event_times: dead → last_obs_hour, alive → T-1
    if et_list:
        et      = np.array(et_list)
        et_full = np.where(y == 1, et, S.shape[1] - 1)
    else:
        et_full = None
    return S, y, et_full


def plot_ibs_and_survival(ode_model, test_loader, device, n_hours: int = 48,
                           save_path=PLOTS_DIR / 'ibs_survival_mean.png'):
    S_test, y_s, et_full = _collect_survival_matrix(ode_model, test_loader, device)
    if S_test is None:
        return

    ibs_score, bs_by_t = integrated_brier_score(S_test, y_s, et_full)
    hours = np.arange(n_hours)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(hours, bs_by_t, color='tomato', lw=2)
    null_bs = y_s.mean() * (1 - y_s.mean())
    axes[0].axhline(null_bs, color='gray', ls='--', lw=1,
                    label=f'Null model BS = {null_bs:.4f}')
    axes[0].set_xlabel('Hour in ICU')
    axes[0].set_ylabel('Brier Score')
    axes[0].set_title(f'Brier Score over Time  (IBS={ibs_score:.4f})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    dead_s  = S_test[y_s == 1].mean(axis=0)
    alive_s = S_test[y_s == 0].mean(axis=0)
    axes[1].plot(hours, dead_s,  color='crimson',   lw=2, label='Mean S(t) — Died')
    axes[1].plot(hours, alive_s, color='steelblue', lw=2, label='Mean S(t) — Survived')
    axes[1].fill_between(hours, dead_s,  alpha=0.15, color='crimson')
    axes[1].fill_between(hours, alive_s, alpha=0.10, color='steelblue')
    axes[1].set_xlabel('Hour in ICU')
    axes[1].set_ylabel('S(t)')
    axes[1].set_title('Mean Survival Curves by Outcome')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'IBS = {ibs_score:.4f}  (null model = {null_bs:.4f})')


# ── training curves ───────────────────────────────────────────────────────────

def plot_training_curves(grud_history: list, ode_history: list,
                          save_path=PLOTS_DIR / 'training_curves.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for hist, label, color in [
        (grud_history, 'GRU-D',              'steelblue'),
        (ode_history,  'Latent ODE+DeepHit', 'tomato'),
    ]:
        epochs     = [r['epoch'] for r in hist]
        train_loss = [r['train_loss'] for r in hist]
        val_auroc  = [r.get('val_h6_auroc', float('nan')) for r in hist]
        axes[0].plot(epochs, train_loss, label=label, color=color, lw=2)
        axes[1].plot(epochs, val_auroc,  label=label, color=color, lw=2)

    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title('Validation AUROC (12h horizon)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUROC')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── per-patient survival curves ───────────────────────────────────────────────

@torch.no_grad()
def plot_survival_curves(model, loader, device, n_hours: int = 48, n_samples: int = 8,
                          title='Latent ODE Survival Curves', save_path=PLOTS_DIR / 'survival_curves.png'):
    model.eval()
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    out   = model(batch)

    if 'survival' not in out:
        print('Model does not output survival curves.')
        return

    survival = out['survival'][:n_samples].cpu().numpy()
    y_true   = batch['y'][:n_samples].cpu().numpy()
    hours    = np.arange(n_hours)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    fig.suptitle(title, fontsize=13)

    for i, ax in enumerate(axes.flat):
        if i >= n_samples:
            break
        color = 'crimson' if y_true[i] == 1 else 'steelblue'
        label = 'Died' if y_true[i] == 1 else 'Survived'
        ax.plot(hours, survival[i], color=color, lw=2)
        ax.fill_between(hours, 0, survival[i], alpha=0.15, color=color)
        ax.set_ylim(0, 1)
        ax.set_title(f'Patient {i + 1} — {label}', fontsize=9, color=color)
        ax.set_xlabel('Hour', fontsize=8)
        ax.set_ylabel('S(t)', fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── windowed AUROC bar chart ──────────────────────────────────────────────────

def plot_windowed_auroc(grud_test_metrics: dict, ode_test_metrics: dict, horizons: List[int],
                         save_path=PLOTS_DIR / 'windowed_auroc.png'):
    import numpy as np
    bar_width = 0.35
    x = np.arange(len(horizons))

    grud_aurocs = [grud_test_metrics.get(f'h{h}_auroc', 0) for h in horizons]
    ode_aurocs  = [ode_test_metrics.get(f'h{h}_auroc',  0) for h in horizons]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - bar_width / 2, grud_aurocs, bar_width, label='GRU-D',              color='steelblue', alpha=0.85)
    ax.bar(x + bar_width / 2, ode_aurocs,  bar_width, label='Latent ODE+DeepHit', color='tomato',    alpha=0.85)
    ax.axhline(0.5, color='gray', lw=1, ls='--', label='Random')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}h horizon' for h in horizons])
    ax.set_ylabel('AUROC')
    ax.set_title('Windowed Mortality Prediction — Test AUROC by Horizon')
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
