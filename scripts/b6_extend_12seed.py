"""Extend existing B6 8-seed OOF with seeds 2034-2037 (honest equal-weight 12-seed).

Loads artifacts/b6_8seed/predictions.npz oof_b5_* / oof_gap_* for 2026-2033,
trains only new seeds, then recomputes pooled equal_prob(b5,gap).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from insurance_claim.model import TARGET, audit_data, build_submission
from insurance_claim.train_b6 import fuse_equal_prob, fuse_equal_rank, run_arm

OLD_SEEDS = list(range(2026, 2034))
NEW_SEEDS = (2034, 2035, 2036, 2037)
SRC = Path("artifacts/b6_8seed/predictions.npz")
OUT = Path("artifacts/b6_12seed")


def main() -> int:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    audit_data(train, test, sample)
    y = train[TARGET].astype(int)
    old = np.load(SRC)
    t0 = time.time()

    arm_oofs: dict[str, list[np.ndarray]] = {"b5": [], "gap": []}
    arm_tests: dict[str, list[np.ndarray]] = {"b5": [], "gap": []}
    seed_aucs: dict[str, dict[str, float]] = {"b5": {}, "gap": {}}

    for arm in ("b5", "gap"):
        for s in OLD_SEEDS:
            key = f"oof_{arm}_{s}"
            if key not in old:
                raise KeyError(key)
            oof = old[key]
            arm_oofs[arm].append(oof)
            seed_aucs[arm][str(s)] = float(roc_auc_score(y, oof))
        # test predictions for old seeds were averaged already in old[f'test_{arm}']
        # For honesty we re-average using per-seed if present; else use stored test_arm / 1
        # train_b6 saves test_{arm} as mean over seeds already — for 12seed we need
        # to retrain new seeds and re-mean. Approximate old contribution as stored mean
        # (equivalent if we weight 8/12 * old_mean + 4/12 * new_mean).
        arm_tests[arm].append(old[f"test_{arm}"])  # placeholder mean of 8

    new_results = {}
    for arm in ("b5", "gap"):
        new_results[arm] = run_arm(arm, train, test, y, NEW_SEEDS)
        for s in NEW_SEEDS:
            arm_oofs[arm].append(new_results[arm]["oof_by_seed"][s])
            seed_aucs[arm][str(s)] = float(roc_auc_score(y, new_results[arm]["oof_by_seed"][s]))
        # rebuild test as equal mean of 8-seed mean and 4 new seeds:
        # correct equal seed average:
        # test_12 = (8 * test_8 + sum_{new} test_seed) / 12
        test_12 = (8.0 * old[f"test_{arm}"] + 4.0 * new_results[arm]["test"]) / 12.0
        arm_tests[arm] = [test_12]  # store final

    oof_b5 = np.mean(np.vstack(arm_oofs["b5"]), axis=0)
    oof_gap = np.mean(np.vstack(arm_oofs["gap"]), axis=0)
    te_b5 = arm_tests["b5"][0]
    te_gap = arm_tests["gap"][0]
    prob_oof, prob_te = fuse_equal_prob([oof_b5, oof_gap], [te_b5, te_gap])
    rank_oof, rank_te = fuse_equal_rank([oof_b5, oof_gap], [te_b5, te_gap])

    # seed-level fusion AUCs
    seed_fusion = []
    for i, s in enumerate(OLD_SEEDS + list(NEW_SEEDS)):
        fused = 0.5 * (arm_oofs["b5"][i] + arm_oofs["gap"][i])
        seed_fusion.append(float(roc_auc_score(y, fused)))

    metrics = {
        "experiment_id": "b6_gap_12seed",
        "recipe": "equal_prob(b5,gap) seeds 2026-2037; reuse 8seed OOF + new 2034-2037",
        "seeds": list(range(2026, 2038)),
        "arms": {
            "b5": {"oof_auc": float(roc_auc_score(y, oof_b5)), "seed_aucs": seed_aucs["b5"],
                   "seed_mean": float(np.mean(list(seed_aucs["b5"].values())))},
            "gap": {"oof_auc": float(roc_auc_score(y, oof_gap)), "seed_aucs": seed_aucs["gap"],
                    "seed_mean": float(np.mean(list(seed_aucs["gap"].values())))},
        },
        "fusion": {
            "primary": "equal_prob",
            "equal_prob": float(roc_auc_score(y, prob_oof)),
            "equal_rank": float(roc_auc_score(y, rank_oof)),
            "b5_only": float(roc_auc_score(y, oof_b5)),
            "gap_only": float(roc_auc_score(y, oof_gap)),
        },
        "pooled_oof_auc": float(roc_auc_score(y, prob_oof)),
        "seed_mean": float(np.mean(seed_fusion)),
        "seed_std": float(np.std(seed_fusion)),
        "seed_fusion_aucs": seed_fusion,
        "gate_0_70": bool(roc_auc_score(y, prob_oof) >= 0.70),
        "gap_to_0_70": round(0.70 - float(roc_auc_score(y, prob_oof)), 6),
        "baseline_b5_8seed": 0.6981745375887981,
        "b6_8seed_equal_prob": 0.6986954542668097,
        "elapsed_sec": round(time.time() - t0, 1),
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "no_oof_weight_search": True,
            "equal_seed_average": True,
            "new_data_only": True,
            "fusion_pre_registered": True,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "predictions.npz",
        oof=prob_oof,
        test=prob_te,
        y=y.to_numpy(),
        oof_b5=oof_b5,
        oof_gap=oof_gap,
        test_b5=te_b5,
        test_gap=te_gap,
    )
    build_submission(test, sample, prob_te, OUT / "submission_b6_12seed.csv")
    build_submission(test, sample, prob_te, Path("submissions") / "submission_b6_12seed.csv")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: metrics[k] for k in ("pooled_oof_auc", "seed_mean", "gate_0_70", "gap_to_0_70", "fusion", "elapsed_sec")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
