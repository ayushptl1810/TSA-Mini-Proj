import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.metrics import compute_metrics


def compute_lr_features(X: np.ndarray, mask: np.ndarray, max_hours: int = 24) -> np.ndarray:
    """(N, T, D) → (N, D) mean of observed values in first max_hours hours only.
    Capped to avoid leaking post-deterioration signal into the static baseline."""
    X_w    = X[:, :max_hours, :]
    mask_w = mask[:, :max_hours, :]
    obs_sum   = (X_w * mask_w).sum(axis=1)
    obs_count = mask_w.sum(axis=1).clip(min=1)
    return obs_sum / obs_count


def train_lr_baseline(X_all, mask_all, y_all, train_idx, test_idx, max_hours: int = 24):
    X_train = compute_lr_features(X_all[train_idx], mask_all[train_idx], max_hours)
    X_test  = compute_lr_features(X_all[test_idx],  mask_all[test_idx],  max_hours)
    y_train = y_all[train_idx]
    y_test  = y_all[test_idx]

    scaler   = StandardScaler().fit(X_train)
    lr_model = LogisticRegression(class_weight='balanced', max_iter=2000, C=0.1)
    lr_model.fit(scaler.transform(X_train), y_train)
    lr_probs = lr_model.predict_proba(scaler.transform(X_test))[:, 1]

    metrics = compute_metrics(y_test, lr_probs, 'lr_')
    print(f'Logistic Regression baseline (first {max_hours}h mean features — no leakage):')
    for k, v in sorted(metrics.items()):
        print(f'  {k:<25} {v:.4f}')

    return lr_model, scaler, lr_probs, y_test, metrics
