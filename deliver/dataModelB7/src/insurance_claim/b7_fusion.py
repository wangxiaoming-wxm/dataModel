"""B7 fusion utilities: pre-registered discrete rules + nested selection.

Inherited from V10 honesty protocol; no continuous OOF weight search.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Pre-registered before looking at full-data leaderboard among rules.
FUSION_RULES = ("mean", "mean_2_1", "power2", "power3", "max", "rank_mean")


def fuse_pair(a: np.ndarray, b: np.ndarray, rule: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if rule == "mean":
        return 0.5 * a + 0.5 * b
    if rule == "mean_2_1":
        return (2.0 * a + b) / 3.0
    if rule == "power2":
        return np.sqrt(0.5 * (a**2 + b**2))
    if rule == "power3":
        return (0.5 * (a**3 + b**3)) ** (1.0 / 3.0)
    if rule == "max":
        return np.maximum(a, b)
    if rule == "rank_mean":
        # Keep as ranks for OOF scoring; for test, caller may rescale.
        return 0.5 * (rankdata(a) + rankdata(b))
    raise ValueError(f"unknown fusion rule: {rule}")


def all_pair_fusions(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    return {r: fuse_pair(a, b, r) for r in FUSION_RULES}


def nested_select_pair(
    a: np.ndarray,
    b: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
    rules: Iterable[str] = FUSION_RULES,
) -> dict:
    """Honest nested rule selection on train OOF only."""
    rules = tuple(rules)
    y = np.asarray(y)
    nested = np.zeros(len(y), dtype=float)
    chosen: list[str] = []
    fold_rows = []
    for fold, (tr, va) in enumerate(
        StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(a, y)
    ):
        scores = {
            r: float(roc_auc_score(y[tr], fuse_pair(a[tr], b[tr], r))) for r in rules
        }
        name = max(scores, key=scores.get)
        chosen.append(name)
        nested[va] = fuse_pair(a[va], b[va], name)
        fold_rows.append({"fold": fold, "chosen": name, "scores": scores})
    pick = Counter(chosen).most_common(1)[0][0]
    full_scores = {
        r: float(roc_auc_score(y, fuse_pair(a, b, r))) for r in rules
    }
    return {
        "nested_rule_votes": chosen,
        "selected_rule": pick,
        "nested_oof_auc": float(roc_auc_score(y, nested)),
        "nested_oof": nested,
        "full_data_scores": full_scores,
        "full_data_selected_auc": full_scores[pick],
        "fold_rows": fold_rows,
        "consistent_all_folds": len(set(chosen)) == 1,
    }


def fuse_three_mean(arms: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(arms), axis=0)


def fuse_three_max(arms: list[np.ndarray]) -> np.ndarray:
    out = arms[0].astype(float).copy()
    for a in arms[1:]:
        out = np.maximum(out, a)
    return out
