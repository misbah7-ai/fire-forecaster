"""
Bootstrap confidence intervals -- used everywhere a metric is reported. No closed-form
approximation: every ROC-AUC in this repo is accompanied by a percentile bootstrap CI, because
the single chronological hold-out is small (n~122) and its CI is wide (~+/-0.08-0.10). Reporting
a point estimate without the CI is how the previous framing over-claimed.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def bootstrap_metric_ci(y_true, y_score, metric="roc_auc", n_boot=2000, seed=42, alpha=0.05):
    """Percentile bootstrap CI for a ranking metric. Resamples rows with replacement; skips
    resamples that end up single-class (metric undefined). Returns (point, lo, hi, n_valid)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    fn = roc_auc_score if metric == "roc_auc" else average_precision_score
    point = float(fn(y_true, y_score))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():  # single class -> metric undefined
            continue
        vals.append(fn(yt, y_score[idx]))
    vals = np.asarray(vals)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi), int(len(vals))


def ci_str(point, lo, hi):
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"
