"""Train-only threshold selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import train_test_split

from lexical_ambiguity.evaluation.metrics import checked_arrays


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    threshold: float
    macro_f1: float
    accuracy: float
    roc_auc: float


def threshold_candidates(scores: ArrayLike) -> NDArray[np.float64]:
    values = np.unique(np.asarray(scores, dtype=np.float64))
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("threshold scores must be a non-empty finite vector")
    lower = np.nextafter(values[0], -np.inf)
    upper = np.nextafter(values[-1], np.inf)
    if len(values) == 1:
        return np.array([lower, upper], dtype=np.float64)
    midpoints = values[:-1] + (values[1:] - values[:-1]) / 2.0
    return np.concatenate(([lower], midpoints, [upper]))


def select_threshold(scores: ArrayLike, labels: ArrayLike) -> ThresholdSelection:
    score_array, label_array = checked_arrays(scores, labels)
    order = np.argsort(score_array, kind="stable")
    sorted_scores = score_array[order]
    sorted_labels = label_array[order]
    values, starts, counts = np.unique(
        sorted_scores, return_index=True, return_counts=True
    )
    positive_count = int(np.count_nonzero(label_array))
    negative_count = len(label_array) - positive_count

    # AUC stays the same as the cutoff moves. Give tied scores their average
    # rank so they do not win or lose just because of row order.
    ranks = np.empty(len(score_array), dtype=np.float64)
    for start, count in zip(starts, counts, strict=True):
        ranks[order[start : start + count]] = start + (count + 1) / 2.0
    roc_auc = (
        float(
            (ranks[label_array].sum() - positive_count * (positive_count + 1) / 2.0)
            / (positive_count * negative_count)
        )
        if positive_count and negative_count
        else float("nan")
    )

    best_key: tuple[float, float, float, float] | None = None
    best: ThresholdSelection | None = None
    true_positive = positive_count
    false_positive = negative_count
    false_negative = 0
    true_negative = 0

    def consider(threshold: float) -> None:
        nonlocal best_key, best
        accuracy = (true_positive + true_negative) / len(label_array)
        positive_denominator = 2 * true_positive + false_positive + false_negative
        negative_denominator = 2 * true_negative + false_positive + false_negative
        positive_f1 = (
            2.0 * true_positive / positive_denominator if positive_denominator else 0.0
        )
        negative_f1 = (
            2.0 * true_negative / negative_denominator if negative_denominator else 0.0
        )
        macro_f1 = (positive_f1 + negative_f1) / 2.0
        key = (
            macro_f1,
            accuracy,
            -abs(float(threshold) - 0.5),
            -float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = ThresholdSelection(
                threshold=float(threshold),
                macro_f1=macro_f1,
                accuracy=accuracy,
                roc_auc=roc_auc,
            )

    consider(float(np.nextafter(values[0], -np.inf)))
    for group_index, (start, count) in enumerate(zip(starts, counts, strict=True)):
        group = sorted_labels[start : start + count]
        moved_positive = int(np.count_nonzero(group))
        moved_negative = int(count) - moved_positive
        true_positive -= moved_positive
        false_negative += moved_positive
        false_positive -= moved_negative
        true_negative += moved_negative
        threshold = (
            float(values[group_index] + (values[group_index + 1] - values[group_index]) / 2.0)
            if group_index + 1 < len(values)
            else float(np.nextafter(values[-1], np.inf))
        )
        consider(threshold)
    assert best is not None  # checked_arrays rejects empty input
    return best


def make_tuning_split(
    labels: ArrayLike, *, holdout_fraction: float, seed: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    label_array = np.asarray(labels).astype(bool)
    if label_array.ndim != 1 or len(label_array) < 4:
        raise ValueError("at least four one-dimensional labels are required")
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between 0 and 0.5")
    indices = np.arange(len(label_array), dtype=np.int64)
    train_indices, holdout_indices = train_test_split(
        indices,
        test_size=holdout_fraction,
        random_state=seed,
        stratify=label_array,
    )
    return np.sort(train_indices), np.sort(holdout_indices)
