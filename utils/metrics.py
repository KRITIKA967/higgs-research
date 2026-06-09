"""
utils/metrics.py
----------------
All evaluation metrics for the HIGGS benchmark.
Metrics: AUC, F1, Signal Significance (Z = S/sqrt(S+B)).
Accuracy excluded — meaningless for imbalanced classification.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def compute_auc(y_true, y_pred_proba):
    """ROC AUC score. Range: 0.5 (random) to 1.0 (perfect)."""
    return roc_auc_score(y_true, y_pred_proba)


def compute_f1(y_true, y_pred):
    """F1 score on minority class (signal, pos_label=1)."""
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)


def compute_signal_significance(y_true, y_pred_proba, threshold=0.3):
    """
    Physics discovery metric: Z = S / sqrt(S + B)

    S = true positives  (predicted signal AND truly signal)
    B = false positives (predicted signal BUT truly background)

    threshold=0.3 instead of 0.5 — at severe imbalance, model
    suppresses signal probabilities, lower threshold captures more.

    Z >= 5 is the particle physics discovery standard (5-sigma).

    Parameters
    ----------
    y_true       : array of true binary labels
    y_pred_proba : array of predicted probabilities for class 1
    threshold    : float — decision boundary (default 0.3)

    Returns
    -------
    float — Z score
    """
    preds = (y_pred_proba >= threshold).astype(int)
    TP    = int(((preds == 1) & (y_true == 1)).sum())
    FP    = int(((preds == 1) & (y_true == 0)).sum())
    return TP / np.sqrt(TP + FP) if (TP + FP) > 0 else 0.0


def compute_all_metrics(y_true, y_pred_proba, y_pred, train_time):
    """
    Compute all 3 metrics + training time.

    Returns
    -------
    dict — AUC, F1, Signal_Significance, Train_Time_sec
    """
    return {
        'AUC':                compute_auc(y_true, y_pred_proba),
        'F1':                 compute_f1(y_true, y_pred),
        'Signal_Significance': compute_signal_significance(y_true, y_pred_proba),
        'Train_Time_sec':     round(train_time, 2)
    }


def print_metrics(metrics_dict, label=""):
    """Pretty-print metrics dict."""
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}AUC={metrics_dict['AUC']:.4f} | "
          f"F1={metrics_dict['F1']:.4f} | "
          f"Signal_Sig={metrics_dict['Signal_Significance']:.4f} | "
          f"Train_Time={metrics_dict['Train_Time_sec']}s")
