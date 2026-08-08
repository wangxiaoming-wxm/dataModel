"""Independently recompute B7 closest / fuse0 nested AUCs from committed OOF.

Usage (repo root):
  PYTHONPATH=src python3 scripts/b7_recompute_closest.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
CLAIMED_CLOSEST = 0.7027049552615718
CLAIMED_FUSE0_PAIR = 0.7022093156561012
B6_POOLED = 0.6989746962571622
PUBLIC_LB = 0.70722  # submission_b7_closest_honest.csv


def nested_select_pair(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> tuple[float, list[str]]:
    nested = np.zeros(len(y))
    votes: list[str] = []
    rules = ("mean", "mean_2_1", "power2", "power3", "max", "rank_mean")
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(a, y):
        scores = {}
        for name in rules:
            if name == "mean":
                pred = 0.5 * (a[tr] + b[tr])
            elif name == "mean_2_1":
                pred = (2 * a[tr] + b[tr]) / 3
            elif name == "power2":
                pred = np.sqrt(0.5 * (a[tr] ** 2 + b[tr] ** 2))
            elif name == "power3":
                pred = (0.5 * (a[tr] ** 3 + b[tr] ** 3)) ** (1 / 3)
            elif name == "max":
                pred = np.maximum(a[tr], b[tr])
            else:
                from scipy.stats import rankdata

                pred = 0.5 * (rankdata(a[tr]) + rankdata(b[tr]))
            scores[name] = float(roc_auc_score(y[tr], pred))
        name = max(scores, key=scores.get)
        votes.append(name)
        if name == "mean":
            nested[va] = 0.5 * (a[va] + b[va])
        elif name == "mean_2_1":
            nested[va] = (2 * a[va] + b[va]) / 3
        elif name == "power2":
            nested[va] = np.sqrt(0.5 * (a[va] ** 2 + b[va] ** 2))
        elif name == "power3":
            nested[va] = (0.5 * (a[va] ** 3 + b[va] ** 3)) ** (1 / 3)
        elif name == "max":
            nested[va] = np.maximum(a[va], b[va])
        else:
            from scipy.stats import rankdata

            nested[va] = 0.5 * (rankdata(a[va]) + rankdata(b[va]))
    return float(roc_auc_score(y, nested)), votes


def main() -> None:
    b6 = np.load(ROOT / "artifacts/b6_frozen/predictions.npz")
    plus_npz = np.load(ROOT / "reference/v10/oof_plus_h2_10.npz")
    y = b6["y"]
    gap, gap_bag = b6["oof_gap"], b6["oof_gap_bag"]
    eq = 0.5 * (gap + gap_bag)
    plus = plus_npz["oof"]

    eq_auc = float(roc_auc_score(y, eq))
    max3 = np.maximum(np.maximum(gap, gap_bag), plus)
    max3_auc = float(roc_auc_score(y, max3))
    pair_nested, votes = nested_select_pair(eq, plus, y)

    rng = np.random.RandomState(42)
    plus_shuf = plus.copy()
    rng.shuffle(plus_shuf)
    shuf_max3 = float(roc_auc_score(y, np.maximum(np.maximum(gap, gap_bag), plus_shuf)))

    cross = {}
    closest_path = ROOT / "artifacts/b7_closest/predictions.npz"
    if closest_path.exists():
        c = np.load(closest_path)
        cross["closest_oof_matches_max3"] = bool(np.allclose(c["oof"], max3))
        cross["closest_arms_match_sources"] = bool(
            np.allclose(c["gap"], gap)
            and np.allclose(c["gap_bag"], gap_bag)
            and np.allclose(c["plus"], plus)
        )
        cross["closest_claimed_abs_err"] = abs(
            float(roc_auc_score(c["y"], c["oof"])) - CLAIMED_CLOSEST
        )

    out = {
        "b6_equal_auc": eq_auc,
        "b6_equal_abs_err_vs_frozen": abs(eq_auc - B6_POOLED),
        "closest_max3_auc": max3_auc,
        "closest_abs_err_vs_claimed": abs(max3_auc - CLAIMED_CLOSEST),
        "fuse0_pair_nested_auc": pair_nested,
        "fuse0_pair_nested_votes": votes,
        "fuse0_pair_abs_err_vs_claimed": abs(pair_nested - CLAIMED_FUSE0_PAIR),
        "shuffled_plus_max3_auc": shuf_max3,
        "shuffled_plus_max3_pass_lt_0_66": shuf_max3 < 0.66,
        "public_leaderboard_auc": PUBLIC_LB,
        "public_leaderboard_file": "submissions/submission_b7_closest_honest.csv",
        "cross_check_closest_npz": cross,
        "sources": {
            "b6": "artifacts/b6_frozen/predictions.npz",
            "plus": "reference/v10/oof_plus_h2_10.npz",
            "closest": "artifacts/b7_closest/predictions.npz",
        },
        "pass_recompute_lt_1e-8": bool(
            abs(max3_auc - CLAIMED_CLOSEST) < 1e-8
            and abs(pair_nested - CLAIMED_FUSE0_PAIR) < 1e-8
            and abs(eq_auc - B6_POOLED) < 1e-8
        ),
    }
    audit = ROOT / "artifacts/b7_audit"
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "recompute_closest.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    if not out["pass_recompute_lt_1e-8"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
