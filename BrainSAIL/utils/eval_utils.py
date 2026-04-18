"""Evaluation metrics and per-class threshold optimization.

Provides macro-averaged classification metrics and a per-class threshold
search strategy tuned for the clinical setting of Brain-OHN, where
sensitivity (recall) is prioritized over precision.

Example:
    >>> f1, f2, bacc = compute_metrics(labels, probs, threshold=0.5)
    >>> thresholds = find_optimal_thresholds(labels, probs)
    >>> f1, f2, bacc = compute_metrics(labels, probs, threshold=thresholds)
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, fbeta_score


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float | list | np.ndarray = 0.5,
) -> tuple[float, float, float]:
    """Compute macro F1, macro F2, and macro Balanced Accuracy.

    Metrics are computed per-class in binary mode to avoid sklearn format
    errors with multi-label arrays, then averaged (macro).

    Args:
        labels (np.ndarray): Ground-truth binary array of shape ``[N, C]``.
        probs (np.ndarray): Predicted probabilities of shape ``[N, C]``.
        threshold (float | list | np.ndarray): Decision threshold(s).
            A scalar applies the same threshold to all classes; a list/array
            of length C applies per-class thresholds. Defaults to ``0.5``.

    Returns:
        tuple[float, float, float]:
            - **macro_f1** — Macro-averaged F1 score.
            - **macro_f2** — Macro-averaged F2 score (recall-weighted).
            - **macro_bacc** — Macro-averaged Balanced Accuracy.
    """
    labels = np.atleast_2d(labels)
    probs = np.atleast_2d(probs)

    if isinstance(threshold, (list, np.ndarray)):
        preds = (probs >= np.array(threshold).reshape(1, -1)).astype(int)
    else:
        preds = (probs >= threshold).astype(int)

    f1s, f2s, baccs = [], [], []
    for i in range(labels.shape[1]):
        y_true = labels[:, i].astype(int).ravel()
        y_pred = preds[:, i].astype(int).ravel()

        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        f2s.append(fbeta_score(y_true, y_pred, beta=2, zero_division=0))

        # Skip BACC for classes with only one active label value
        if y_true.sum() > 0 and (1 - y_true).sum() > 0:
            baccs.append(balanced_accuracy_score(y_true, y_pred))

    return (
        float(np.mean(f1s)),
        float(np.mean(f2s)),
        float(np.mean(baccs)) if baccs else 0.0,
    )


def find_optimal_thresholds(
    labels: np.ndarray,
    probs: np.ndarray,
) -> list[float]:
    """Find per-class decision thresholds on a validation set.

    Searches over 100 linearly spaced values and 9 quantiles of the predicted
    probability distribution, selecting the threshold that maximizes:

        score = 0.7 * F1 + 0.3 * Balanced Accuracy

    The 0.7 / 0.3 weighting reflects a clinical preference for recall
    (detecting true positives) over avoiding false alarms.

    Args:
        labels (np.ndarray): Ground-truth binary array of shape ``[N, C]``.
        probs (np.ndarray): Predicted probabilities of shape ``[N, C]``.

    Returns:
        list[float]: Optimal threshold for each class, length ``C``.
    """
    n_classes = labels.shape[1]
    best_thresholds = [0.5] * n_classes

    for i in range(n_classes):
        y_true = labels[:, i]
        y_prob = probs[:, i]

        # Skip degenerate classes
        if y_true.sum() == 0 or (1 - y_true).sum() == 0:
            continue

        quantiles = np.quantile(y_prob, np.linspace(0.1, 0.9, 9))
        linear = np.linspace(
            max(1e-3, float(y_prob.min())),
            min(0.999, float(y_prob.max())),
            100,
        )
        candidates = np.unique(np.concatenate([linear, quantiles]))

        best_score, best_th = -1.0, 0.5
        for th in candidates:
            preds = (y_prob >= th).astype(int)
            score = (
                0.7 * f1_score(y_true, preds, zero_division=0)
                + 0.3 * balanced_accuracy_score(y_true, preds)
            )
            if score > best_score:
                best_score, best_th = score, th

        best_thresholds[i] = float(best_th)

    return best_thresholds


def evaluate_clinical_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, float]]:
    """Compute sensitivity, specificity, PPV, and NPV per class.

    Args:
        labels (np.ndarray): Ground-truth binary array of shape ``[N, C]``.
        probs (np.ndarray): Predicted probabilities of shape ``[N, C]``.
        thresholds (list[float]): Per-class decision thresholds, length ``C``.

    Returns:
        list[dict[str, float]]: One dictionary per class containing keys
            ``"sensitivity"``, ``"specificity"``, ``"ppv"``, and ``"npv"``.
    """
    metrics = []
    for i in range(labels.shape[1]):
        y_true = labels[:, i]
        preds = (probs[:, i] >= thresholds[i]).astype(int)

        tp = int(np.sum((preds == 1) & (y_true == 1)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))

        metrics.append({
            "sensitivity": tp / (tp + fn + 1e-9),
            "specificity": tn / (tn + fp + 1e-9),
            "ppv":         tp / (tp + fp + 1e-9),
            "npv":         tn / (tn + fn + 1e-9),
        })
    return metrics
