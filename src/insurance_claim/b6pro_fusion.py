"""B6pro nested discrete fusion helpers (pre-registered rules; no continuous search)."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RULE_NAMES = ("mean", "mean_2_1", "power2", "power3", "max", "rank_mean")


def apply_rule(rule: str, arms: list[np.ndarray]) -> np.ndarray:
    stacked = np.vstack(arms)
    if rule == "mean":
        return stacked.mean(axis=0)
    if rule == "mean_2_1":
        # Prefer first arm (main) with 2:1 vs equal-rest mean of others.
        if len(arms) == 1:
            return arms[0]
        if len(arms) == 2:
            return (2.0 * arms[0] + arms[1]) / 3.0
        rest = stacked[1:].mean(axis=0)
        return (2.0 * arms[0] + rest) / 3.0
    if rule == "power2":
        return np.sqrt(np.mean(np.square(stacked), axis=0))
    if rule == "power3":
        return np.cbrt(np.mean(np.power(stacked, 3), axis=0))
    if rule == "max":
        return stacked.max(axis=0)
    if rule == "rank_mean":
        ranks = np.vstack([rankdata(a) for a in arms])
        return ranks.mean(axis=0)
    raise ValueError(f"unknown rule: {rule}")


def score_rules(y: np.ndarray, arms: list[np.ndarray]) -> dict[str, float]:
    return {r: float(roc_auc_score(y, apply_rule(r, arms))) for r in RULE_NAMES}


def nested_select_rule(
    y: np.ndarray,
    arms: list[np.ndarray],
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Select fusion rule by nested SKF on train indices; build nested OOF."""
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    nested = np.zeros(len(y), dtype=float)
    fold_rules: list[str] = []
    for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
        y_tr = y[tr_idx]
        arm_tr = [a[tr_idx] for a in arms]
        scores = score_rules(y_tr, arm_tr)
        rule = max(scores, key=scores.get)
        fold_rules.append(rule)
        nested[va_idx] = apply_rule(rule, [a[va_idx] for a in arms])
    # Majority / consistency for full-data application
    from collections import Counter

    majority = Counter(fold_rules).most_common(1)[0][0]
    full_scores = score_rules(y, arms)
    return {
        "nested_oof": nested,
        "nested_oof_auc": float(roc_auc_score(y, nested)),
        "fold_rules": fold_rules,
        "selected_rule": majority,
        "consistent_all_folds": len(set(fold_rules)) == 1,
        "full_data_scores": full_scores,
        "full_selected_oof": apply_rule(majority, arms),
        "full_selected_auc": float(full_scores[majority]),
    }


def apply_rule_test(rule: str, test_arms: list[np.ndarray]) -> np.ndarray:
    return apply_rule(rule, test_arms)
