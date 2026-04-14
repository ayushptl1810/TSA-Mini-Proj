from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import Config, TENSOR_NPZ, PIVOTED_CSV


def load_tensors():
    """Load pre-built NPZ tensors and return raw arrays + metadata."""
    data = np.load(TENSOR_NPZ)
    X_all      = data['X']           # (N, 48, 36) normalised, forward-filled
    mask_all   = data['mask']        # (N, 48, 36) binary observation mask
    y_all      = data['y']           # (N,) binary: in-hospital death
    record_ids = data['record_ids']  # (N,)
    var_names  = data['var_names']   # (36,)
    X_means    = data['means']       # (36,) training-set means
    X_stds     = data['stds']        # (36,)

    N, T, D = X_all.shape
    print(f'Patients : {N}')
    print(f'Hours    : {T}')
    print(f'Variables: {D}  → {list(var_names)}')
    print(f'Mortality: {y_all.mean():.1%}  ({int(y_all.sum())} deaths)')
    print(f'Obs rate : {mask_all.mean():.1%}')

    return X_all, mask_all, y_all, record_ids, var_names, X_means, X_stds


def compute_delta(mask: np.ndarray) -> np.ndarray:
    """mask: (N, T, D) → delta: (N, T, D) hours since last observation."""
    N, T, D = mask.shape
    delta = np.zeros_like(mask, dtype=np.float32)
    for t in range(1, T):
        prev_obs = mask[:, t - 1, :]
        delta[:, t, :] = np.where(prev_obs == 1, 1.0, delta[:, t - 1, :] + 1.0)
    return delta


def get_splits(record_ids: np.ndarray):
    """Return train/val/test index lists based on original PhysioNet set labels."""
    piv = pd.read_csv(PIVOTED_CSV, usecols=['RecordID', 'set']).drop_duplicates('RecordID')
    rid_to_set = dict(zip(piv['RecordID'], piv['set']))

    train_idx = [i for i, r in enumerate(record_ids) if rid_to_set.get(r, 'C') == 'A']
    val_idx   = [i for i, r in enumerate(record_ids) if rid_to_set.get(r, 'C') == 'B']
    test_idx  = [i for i, r in enumerate(record_ids) if rid_to_set.get(r, 'C') == 'C']

    print(f'Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}')
    return train_idx, val_idx, test_idx


def build_windowed_labels(
    y: np.ndarray,
    mask: np.ndarray,
    horizons: List[int],
    n_hours: int = 48,
    ramp_factor: float = 2.0,
) -> Dict[int, np.ndarray]:
    """
    Soft-ramp windowed labels.

    For dead patients, T_death = last observed hour:
      - t >= T_death - X                           → 1.0  (danger zone)
      - T_death - X*ramp ≤ t < T_death - X         → linear ramp 0→1
      - t < T_death - X*ramp                        → 0.0
    Alive patients → all zeros.
    """
    N, T, D = mask.shape

    last_obs_hour = np.array([
        int(np.where(mask[i].sum(axis=1) > 0)[0][-1])
        if mask[i].sum() > 0 else T - 1
        for i in range(N)
    ])

    windowed = {}
    for X in horizons:
        labels = np.zeros((N, T), dtype=np.float32)
        for i in range(N):
            if y[i] != 1:
                continue
            T_d          = last_obs_hour[i]
            t_full       = T_d - X
            t_ramp_start = T_d - int(X * ramp_factor)
            for t in range(T):
                if t >= t_full:
                    labels[i, t] = 1.0
                elif t >= t_ramp_start and t_full > t_ramp_start:
                    labels[i, t] = (t - t_ramp_start) / (t_full - t_ramp_start)
        windowed[X] = labels

    return windowed


class ICUDataset(Dataset):
    """
    Multivariate ICU time-series dataset.

    Each item:
      X       : (T, D) normalised values
      mask    : (T, D) observation mask
      delta   : (T, D) hours since last observation
      y       : scalar binary label
      wlabels : (n_horizons, T) windowed labels
      time    : (T,) hour indices
    """

    def __init__(
        self,
        indices: List[int],
        X: np.ndarray,
        mask: np.ndarray,
        delta: np.ndarray,
        y: np.ndarray,
        windowed_labels: Dict[int, np.ndarray],
        horizons: List[int],
    ):
        self.idx      = indices
        self.X        = torch.from_numpy(X[indices]).float()
        self.mask     = torch.from_numpy(mask[indices]).float()
        self.delta    = torch.from_numpy(delta[indices]).float()
        self.y        = torch.from_numpy(y[indices]).float()
        self.horizons = horizons
        stacked       = np.stack([windowed_labels[h][indices] for h in horizons], axis=1)
        self.wlabels  = torch.from_numpy(stacked).float()
        self.time     = torch.arange(X.shape[1]).float()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return {
            'X'      : self.X[i],
            'mask'   : self.mask[i],
            'delta'  : self.delta[i],
            'y'      : self.y[i],
            'wlabels': self.wlabels[i],
            'time'   : self.time,
        }


def make_loaders(
    cfg: Config,
    X, mask, delta, y, windowed_labels,
    train_idx, val_idx, test_idx,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    common = dict(X=X, mask=mask, delta=delta, y=y,
                  windowed_labels=windowed_labels, horizons=cfg.horizons)
    ds_train = ICUDataset(train_idx, **common)
    ds_val   = ICUDataset(val_idx,   **common)
    ds_test  = ICUDataset(test_idx,  **common)

    loader_kw = dict(batch_size=cfg.batch_size, pin_memory=False, num_workers=0)
    return (
        DataLoader(ds_train, shuffle=True,  **loader_kw),
        DataLoader(ds_val,   shuffle=False, **loader_kw),
        DataLoader(ds_test,  shuffle=False, **loader_kw),
    )
