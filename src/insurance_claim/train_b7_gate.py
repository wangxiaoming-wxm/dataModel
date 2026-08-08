"""B7 nested gate: learn when to trust plus vs B6 (honest nested OOF).

Stage-1 arms frozen: equal_prob(B6 gap, gap_bag) and V10 plus.
Gate model predicts P(plus closer to label) using fold-local residual cats.
Final blend: soft gate * plus + (1-gate) * b6, fused with stage1 via nested rules.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b7_fusion import FUSION_RULES, fuse_pair, nested_select_pair
from insurance_claim.model import TARGET, build_submission
from insurance_claim.train_b7_residual import build_residual_frame

THREAD = 8
GATE_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=800,
    learning_rate=0.04,
    depth=4,
    l2_leaf_reg=16,
    random_strength=0.8,
    od_type="Iter",
    od_wait=80,
    verbose=False,
    thread_count=THREAD,
    allow_writing_files=False,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b7_gate"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument(
        "--mode",
        choices=("soft", "hard", "max_gate"),
        default="soft",
        help="soft=blend; hard=pick; max_gate=max(b6, soft)",
    )
    args = ap.parse_args()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train[TARGET].astype(int).to_numpy()
    feats = train.drop(columns=[TARGET])

    b6 = np.load("artifacts/b6_gapbag_8seed/predictions.npz")
    plus = np.load("reference/v10/oof_plus_h2_10.npz")
    eq = 0.5 * (b6["oof_gap"] + b6["oof_gap_bag"])
    plus_oof = plus["oof"]
    te_eq = 0.5 * (b6["test_gap"] + b6["test_gap_bag"])
    te_plus = np.load("reference/v10/test_plus_h2_10.npy")
    stage1 = np.maximum(eq, plus_oof)
    te_stage1 = np.maximum(te_eq, te_plus)

    # which arm is closer (for gate labels)
    plus_better = (np.abs(plus_oof - y) < np.abs(eq - y)).astype(int)
    print(
        f"plus_better_rate={plus_better.mean():.4f} magic_auc={roc_auc_score(y, np.where(plus_better, plus_oof, eq)):.6f}",
        flush=True,
    )

    t0 = time.time()
    oofs, tests, folds = [], [], []
    for seed in args.seeds:
        oof_gate = np.zeros(len(train))
        oof_blend = np.zeros(len(train))
        pte_gate = np.zeros(len(test))
        pte_blend = np.zeros(len(test))
        for fold, (a, b) in enumerate(
            StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(feats, y)
        ):
            Xtr, Xva = feats.iloc[a].reset_index(drop=True), feats.iloc[b].reset_index(drop=True)
            tr, va, te, cats = build_residual_frame(Xtr, Xva, test.copy())
            # arm scores as features (OOF-safe on train indices)
            for df, idx, is_te in (
                (tr, a, False),
                (va, b, False),
                (te, None, True),
            ):
                if is_te:
                    df["p_b6"] = te_eq
                    df["p_plus"] = te_plus
                    df["p_max"] = te_stage1
                    df["p_diff"] = te_plus - te_eq
                    df["p_absdiff"] = np.abs(te_plus - te_eq)
                else:
                    df["p_b6"] = eq[idx]
                    df["p_plus"] = plus_oof[idx]
                    df["p_max"] = stage1[idx]
                    df["p_diff"] = plus_oof[idx] - eq[idx]
                    df["p_absdiff"] = np.abs(plus_oof[idx] - eq[idx])

            y_gate = plus_better[a]
            y_gate_va = plus_better[b]
            # need both classes in fold
            if y_gate.min() == y_gate.max():
                g_va = np.full(len(b), float(y_gate.mean()))
                g_te = np.full(len(test), float(y_gate.mean()))
                best = -1
            else:
                p = dict(GATE_PARAMS)
                p["random_seed"] = seed + fold
                m = CatBoostClassifier(**p)
                m.fit(
                    tr,
                    y_gate,
                    eval_set=(va, y_gate_va),
                    cat_features=cats,
                    use_best_model=True,
                )
                g_va = m.predict_proba(va)[:, 1]
                g_te = m.predict_proba(te)[:, 1]
                best = int(m.get_best_iteration() or -1)

            oof_gate[b] = g_va
            if args.mode == "soft":
                blend_va = g_va * plus_oof[b] + (1.0 - g_va) * eq[b]
                blend_te = g_te * te_plus + (1.0 - g_te) * te_eq
            elif args.mode == "hard":
                pick = (g_va >= 0.5).astype(float)
                blend_va = pick * plus_oof[b] + (1.0 - pick) * eq[b]
                pick_te = (g_te >= 0.5).astype(float)
                blend_te = pick_te * te_plus + (1.0 - pick_te) * te_eq
            else:  # max_gate
                soft = g_va * plus_oof[b] + (1.0 - g_va) * eq[b]
                blend_va = np.maximum(stage1[b], soft)
                soft_te = g_te * te_plus + (1.0 - g_te) * te_eq
                blend_te = np.maximum(te_stage1, soft_te)

            oof_blend[b] = blend_va
            pte_gate += g_te / args.folds
            pte_blend += blend_te / args.folds
            auc = float(roc_auc_score(y[b], blend_va))
            gate_auc = float(roc_auc_score(y_gate_va, g_va)) if y_gate_va.min() != y_gate_va.max() else float("nan")
            folds.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "blend_auc": auc,
                    "gate_auc": gate_auc,
                    "best": best,
                }
            )
            print(
                f"gate seed={seed} fold={fold} blend={auc:.5f} gate_auc={gate_auc:.5f} best={best}",
                flush=True,
            )
        print(
            f"gate seed={seed} blend_OOF={roc_auc_score(y, oof_blend):.6f} gate_OOF={roc_auc_score(plus_better, oof_gate):.6f}",
            flush=True,
        )
        oofs.append(oof_blend)
        tests.append(pte_blend)

    stage2 = np.mean(np.vstack(oofs), 0)
    te2 = np.mean(np.vstack(tests), 0)

    # nested fuse stage1 (max b6/plus) with gated blend
    nested = nested_select_pair(stage1, stage2, y)
    rule = nested["selected_rule"]
    oof_final = fuse_pair(stage1, stage2, rule)
    if rule == "rank_mean":
        from scipy.stats import rankdata

        te_final = 0.5 * (rankdata(te_stage1) + rankdata(te2))
        te_final = (te_final - te_final.min()) / (te_final.max() - te_final.min() + 1e-12)
    else:
        te_final = fuse_pair(te_stage1, te2, rule)

    # also score raw stage2 and max(stage1, stage2)
    metrics = {
        "experiment_id": f"b7_gate_{args.mode}",
        "mode": args.mode,
        "stage1_auc": float(roc_auc_score(y, stage1)),
        "stage2_auc": float(roc_auc_score(y, stage2)),
        "corr_s1_s2": float(np.corrcoef(stage1, stage2)[0, 1]),
        "nested": {
            "selected_rule": rule,
            "nested_oof_auc": nested["nested_oof_auc"],
            "votes": nested["nested_rule_votes"],
            "full_scores": {
                r: float(roc_auc_score(y, fuse_pair(stage1, stage2, r))) for r in FUSION_RULES
            },
        },
        "direct_stage2_auc": float(roc_auc_score(y, stage2)),
        "max_s1_s2_auc": float(roc_auc_score(y, np.maximum(stage1, stage2))),
        "pooled_selected_auc": float(roc_auc_score(y, oof_final)),
        "gate_0_71": bool(nested["nested_oof_auc"] >= 0.71),
        "gap_to_0_71": round(0.71 - nested["nested_oof_auc"], 6),
        "seeds": list(args.seeds),
        "n_splits": args.folds,
        "elapsed_sec": round(time.time() - t0, 1),
        "folds": folds,
        "protocol_declaration": {
            "stage1_frozen_b6_plus": True,
            "fusion_rules_preregistered": list(FUSION_RULES),
            "rule_selection": "nested_5fold",
            "no_global_te": True,
            "gate_label": "plus_closer_than_b6",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        oof=oof_final,
        test=te_final,
        y=y,
        stage1=stage1,
        stage2=stage2,
        nested_oof=nested["nested_oof"],
    )
    build_submission(test, sample, te_final, args.output_dir / "submission_b7.csv")
    build_submission(test, sample, te_final, Path("submissions") / f"submission_b7_gate_{args.mode}.csv")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in (
                    "stage1_auc",
                    "stage2_auc",
                    "corr_s1_s2",
                    "nested",
                    "max_s1_s2_auc",
                    "gate_0_71",
                    "gap_to_0_71",
                    "elapsed_sec",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
