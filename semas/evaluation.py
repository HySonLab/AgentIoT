"""Honest evaluation utilities.

Two invariants enforced here, both violated by the previous implementation:
  1. Decision thresholds are calibrated on the VALIDATION split only.
     (`create_predictions` previously ran precision_recall_curve on y_test.)
  2. Latency is reported end-to-end per sample with an explicitly defined
     boundary: raw feature vector in -> binary decision out, wall-clock,
     averaged over the test set. Training/adaptation time is reported
     separately, never mixed into inference latency.
"""

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calibrate_threshold(y_val: np.ndarray, scores_val: np.ndarray) -> float:
    """Best-F1 threshold on the validation split.

    Degenerate thresholds at/below the minimum score (which classify every
    sample - including edge-filtered zeros - as anomalous) are excluded.
    """
    prec, rec, thresholds = precision_recall_curve(y_val, scores_val)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    lo = float(np.min(scores_val))
    valid = thresholds > lo
    if not valid.any():
        return float(np.median(scores_val[scores_val > lo])) if (scores_val > lo).any() \
            else float(np.median(scores_val))
    f1_valid = f1[:-1][valid]
    return float(thresholds[valid][np.argmax(f1_valid)])


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    out = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "anomaly_rate": float(np.mean(y_true)),
    }
    out["roc_auc"] = (
        float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")
    )
    return out


def rul_metrics(rul_true: np.ndarray, rul_pred: np.ndarray) -> dict:
    err = rul_pred - rul_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        # NASA scoring function (Saxena et al. 2008): asymmetric penalty
        "nasa_score": float(
            np.sum(np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1))
        ),
        "n": int(len(rul_true)),
    }


@dataclass
class LatencyReport:
    per_sample_ms: float
    batch_ms: float
    n_samples: int
    boundary: str

    def as_dict(self) -> dict:
        return {
            "per_sample_ms": self.per_sample_ms,
            "batch_ms": self.batch_ms,
            "n_samples": self.n_samples,
            "boundary": self.boundary,
        }


def measure_latency(fn, X, boundary: str, repeats: int = 3) -> LatencyReport:
    """Wall-clock latency of `fn(X)` (feature vectors -> decisions).

    Reports both total batch time and amortized per-sample time; the
    manuscript must state which one is quoted and what `boundary` covers.
    """
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(X)
        times.append(time.perf_counter() - t0)
    batch_s = float(np.median(times))
    n = len(X)
    return LatencyReport(
        per_sample_ms=batch_s / n * 1000.0,
        batch_ms=batch_s * 1000.0,
        n_samples=n,
        boundary=boundary,
    )
