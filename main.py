"""
PhysioNet 2012 — ICU Mortality Survival Prediction
GRU-D + Latent ODE with DeepHit Survival Head

Usage:
    python main.py
"""
import math
from dataclasses import asdict

import numpy as np
import torch
import wandb

from src.config import CFG, DEVICE, load_env
from src.data.dataset import load_tensors, compute_delta, get_splits, build_windowed_labels, make_loaders
from src.data.analysis import (
    plot_obs_density, plot_acf_pacf, run_adf_tests,
    plot_missingness, plot_windowed_labels,
)
from src.models.grud import GRUDSurvivalModel
from src.models.latent_ode import LatentODESurvival
from src.metrics import evaluate_model
from src.train import EarlyStopping, train_grud_epoch, train_ode_epoch, kl_anneal
from src.baselines import train_lr_baseline
from src.evaluate import (
    print_metric_table, print_full_comparison,
    plot_calibration, plot_ibs_and_survival,
    plot_training_curves, plot_survival_curves, plot_windowed_auroc,
)


def main():
    load_env()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X_all, mask_all, y_all, record_ids, var_names, X_means, X_stds = load_tensors()
    delta_all = compute_delta(mask_all)
    print(f'delta shape: {delta_all.shape}')

    train_idx, val_idx, test_idx = get_splits(record_ids)

    # ── 2. TSA analysis ───────────────────────────────────────────────────────
    plot_obs_density(mask_all, train_idx, var_names)
    plot_acf_pacf(X_all, mask_all, train_idx, var_names)
    run_adf_tests(X_all, mask_all, y_all, train_idx, var_names)
    plot_missingness(mask_all, delta_all, y_all, train_idx)

    # ── 3. Windowed labels ────────────────────────────────────────────────────
    windowed_labels = build_windowed_labels(y_all, mask_all, CFG.horizons, CFG.n_hours)
    plot_windowed_labels(windowed_labels, y_all, CFG.horizons, CFG.n_hours)

    # ── 4. DataLoaders ────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = make_loaders(
        CFG, X_all, mask_all, delta_all, y_all, windowed_labels,
        train_idx, val_idx, test_idx,
    )
    print(f'Train batches: {len(train_loader)}  Val: {len(val_loader)}  Test: {len(test_loader)}')

    # ── 5. Train GRU-D ────────────────────────────────────────────────────────
    x_mean_tensor = torch.from_numpy(X_means.astype(np.float32)).to(DEVICE)
    grud_model    = GRUDSurvivalModel(CFG, x_mean_tensor).to(DEVICE)
    grud_optim    = torch.optim.AdamW(grud_model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    grud_sched    = torch.optim.lr_scheduler.CosineAnnealingLR(grud_optim, T_max=CFG.n_epochs, eta_min=1e-5)
    grud_es       = EarlyStopping(patience=CFG.patience, mode='max')

    grud_run = wandb.init(project=CFG.project, entity=CFG.entity or None,
                          name='GRU-D', config=asdict(CFG), tags=['gru-d'], reinit=True)
    wandb.watch(grud_model, log='gradients', log_freq=50)
    print(f'\nTraining GRU-D on {DEVICE}  ({sum(p.numel() for p in grud_model.parameters()):,} params)')

    grud_history = []
    for epoch in range(1, CFG.n_epochs + 1):
        train_log   = train_grud_epoch(grud_model, train_loader, grud_optim, CFG, DEVICE)
        val_metrics = evaluate_model(grud_model, val_loader, CFG.horizons, DEVICE, 'grud')
        grud_sched.step()

        monitor = val_metrics.get('h6_auroc', val_metrics.get('global_auroc', 0.0))
        log     = {'epoch': epoch, 'lr': grud_sched.get_last_lr()[0],
                   **train_log, **{f'val_{k}': v for k, v in val_metrics.items()}}
        wandb.log(log)
        grud_history.append(log)

        if epoch % 5 == 0 or epoch == 1:
            print(f'E{epoch:03d} | loss={train_log["train_loss"]:.4f}'
                  f' | val_h6_auroc={val_metrics.get("h6_auroc", float("nan")):.4f}'
                  f' | val_global_auroc={val_metrics.get("global_auroc", float("nan")):.4f}')

        if grud_es.step(monitor, grud_model):
            print(f'Early stopping at epoch {epoch}')
            break

    grud_es.load_best(grud_model)
    print('Loaded best GRU-D checkpoint.')
    
    # Save the best GRU-D model
    torch.save(grud_model.state_dict(), 'models/grud_best.pth')
    print('Saved GRU-D weights to models/grud_best.pth')

    grud_test_metrics = evaluate_model(grud_model, test_loader, CFG.horizons, DEVICE, 'grud')
    print('\n── GRU-D Test Results ─────────────────────────────────────────────────')
    for k, v in sorted(grud_test_metrics.items()):
        if not math.isnan(v):
            print(f'  {k:<30} {v:.4f}')
    wandb.log({f'test_{k}': v for k, v in grud_test_metrics.items()})
    grud_run.finish()

    # ── 6. Train Latent ODE ───────────────────────────────────────────────────
    ode_model = LatentODESurvival(CFG).to(DEVICE).float()
    ode_optim = torch.optim.AdamW(ode_model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    ode_sched = torch.optim.lr_scheduler.CosineAnnealingLR(ode_optim, T_max=CFG.n_epochs, eta_min=1e-5)
    ode_es    = EarlyStopping(patience=CFG.patience, mode='max')

    ode_run = wandb.init(project=CFG.project, entity=CFG.entity or None,
                         name='Latent-ODE-DeepHit', config=asdict(CFG),
                         tags=['latent-ode', 'deephit'], reinit=True)
    wandb.watch(ode_model, log='gradients', log_freq=50)
    print(f'\nTraining Latent ODE on {DEVICE}  ({sum(p.numel() for p in ode_model.parameters()):,} params)')

    ode_history = []
    for epoch in range(1, CFG.n_epochs + 1):
        kl_w        = kl_anneal(epoch)
        train_log   = train_ode_epoch(ode_model, train_loader, ode_optim, CFG, DEVICE, kl_w)
        val_metrics = evaluate_model(ode_model, val_loader, CFG.horizons, DEVICE, 'ode')
        ode_sched.step()

        monitor = val_metrics.get('h6_auroc', val_metrics.get('c_index', 0.0))
        log     = {'epoch': epoch, 'kl_weight': kl_w,
                   'lr': ode_sched.get_last_lr()[0],
                   **train_log, **{f'val_{k}': v for k, v in val_metrics.items()}}
        wandb.log(log)
        ode_history.append(log)

        if epoch % 5 == 0 or epoch == 1:
            ibs_str = f", val_IBS={val_metrics.get('IBS', float('nan')):.4f}" if 'IBS' in val_metrics else ''
            print(f'E{epoch:03d} | loss={train_log["train_loss"]:.4f}'
                  f' | kl={train_log.get("train_kl", 0):.4f}'
                  f' | val_h6_auroc={val_metrics.get("h6_auroc", float("nan")):.4f}'
                  f' | c_index={val_metrics.get("c_index", float("nan")):.4f}'
                  f'{ibs_str}')

        if ode_es.step(monitor, ode_model):
            print(f'Early stopping at epoch {epoch}')
            break

    ode_es.load_best(ode_model)
    print('Loaded best Latent ODE checkpoint.')
    
    # Save the best Latent ODE model
    torch.save(ode_model.state_dict(), 'models/latent_ode_best.pth')
    print('Saved Latent ODE weights to models/latent_ode_best.pth')

    ode_test_metrics = evaluate_model(ode_model, test_loader, CFG.horizons, DEVICE, 'ode')
    print('\n── Latent ODE Test Results ────────────────────────────────────────────')
    for k, v in sorted(ode_test_metrics.items()):
        if not math.isnan(v):
            print(f'  {k:<30} {v:.4f}')
    wandb.log({f'test_{k}': v for k, v in ode_test_metrics.items()})
    ode_run.finish()

    # ── 7. LR baseline ───────────────────────────────────────────────────────
    _, _, lr_probs, y_test_lr, lr_metrics = train_lr_baseline(
        X_all, mask_all, y_all, train_idx, test_idx
    )

    # ── 8. Results & plots ────────────────────────────────────────────────────
    print('\n── Full Metric Table ──────────────────────────────────────────────────')
    print_metric_table(grud_test_metrics, ode_test_metrics)
    print_full_comparison(lr_metrics, grud_test_metrics, ode_test_metrics)

    plot_calibration(grud_model, ode_model, test_loader, DEVICE, y_test_lr, lr_probs)
    plot_ibs_and_survival(ode_model, test_loader, DEVICE, CFG.n_hours)
    plot_training_curves(grud_history, ode_history)
    plot_survival_curves(ode_model, test_loader, DEVICE, CFG.n_hours)
    plot_windowed_auroc(grud_test_metrics, ode_test_metrics, CFG.horizons)


if __name__ == '__main__':
    main()
