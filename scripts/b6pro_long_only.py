#!/usr/bin/env python3
"""Long-only specialist (train on days>=3000) patched onto B7 max3; nested fuse.

Key insight: B7 max3 long-slice AUC≈0.663 while long is ~79% of claims.
A long-only CatBoost (even weaker solo) can lift nested fusion via max/mean rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule
from insurance_claim.b6pro_long_features import build_long_aging, build_long_keepx
from insurance_claim.model import IDENTIFIER, TARGET
from insurance_claim.train_b6 import PARAMS_GAP_BAG, build_gap

B7_FLOOR = 0.7027049552615718
GATE = 0.71


def write_submission(sample: pd.DataFrame, test: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    pred = np.asarray(pred, dtype=float)
    assert len(pred) == len(test)
    out = sample.copy()
    # sample has id + label column
    label_col = [c for c in out.columns if c != IDENTIFIER][0]
    out[label_col] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def run_long_only(builder, features, y, test, days, seeds, params, min_days=3000.0):
    mask = days >= min_days
    idx = np.where(mask)[0]
    oof_acc = np.zeros(len(y), dtype=float)
    test_acc = np.zeros(len(test), dtype=float)
    for seed in seeds:
        oof = np.zeros(len(y), dtype=float)
        pte = np.zeros(len(test), dtype=float)
        Xl = features.iloc[idx].reset_index(drop=True)
        yl = y.iloc[idx].reset_index(drop=True)
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xl, yl)):
            gtr, gva = idx[tr], idx[va]
            trd, vad, ted, cats = builder(
                features.iloc[gtr].reset_index(drop=True),
                features.iloc[gva].reset_index(drop=True),
                test.copy(),
            )
            model = CatBoostClassifier(**{**params, "random_seed": int(seed + fold), "thread_count": 4})
            model.fit(
                trd,
                y.iloc[gtr],
                eval_set=(vad, y.iloc[gva]),
                cat_features=cats,
                use_best_model=True,
            )
            oof[gva] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5.0
            print(
                f"seed={seed} fold={fold} long_auc={roc_auc_score(y.iloc[gva], oof[gva]):.5f}",
                flush=True,
            )
        print(
            f"seed={seed} longOOF={roc_auc_score(y.to_numpy()[mask], oof[mask]):.6f}",
            flush=True,
        )
        oof_acc += oof
        test_acc += pte
    return oof_acc / len(seeds), test_acc / len(seeds), mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029])
    ap.add_argument("--builder", choices=["gap", "aging", "keepx"], default="gap")
    ap.add_argument("--min-days", type=float, default=3000.0)
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_long_only"))
    args = ap.parse_args()

    builders = {"gap": build_gap, "aging": build_long_aging, "keepx": build_long_keepx}
    builder = builders[args.builder]

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train[TARGET].astype(int)
    features = train.drop(columns=[TARGET])
    days = features["days"].to_numpy(dtype=float)
    days_te = test["days"].to_numpy(dtype=float)

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    tmax = np.maximum.reduce([fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]])

    params = {**PARAMS_GAP_BAG}
    oof_lo, test_lo, mask = run_long_only(
        builder, features, y, test, days, args.seeds, params, min_days=args.min_days
    )
    mask_te = days_te >= args.min_days

    patch = max3.copy()
    patch[mask] = oof_lo[mask]
    tpatch = tmax.copy()
    tpatch[mask_te] = test_lo[mask_te]

    meanL = max3.copy()
    meanL[mask] = 0.5 * (max3[mask] + oof_lo[mask])
    tmeanL = tmax.copy()
    tmeanL[mask_te] = 0.5 * (tmax[mask_te] + test_lo[mask_te])

    print("max3", roc_auc_score(y, max3), "long", roc_auc_score(y.to_numpy()[mask], max3[mask]), flush=True)
    print(
        "long_only slice",
        roc_auc_score(y.to_numpy()[mask], oof_lo[mask]),
        flush=True,
    )
    print("patch", roc_auc_score(y, patch), flush=True)
    print("meanL", roc_auc_score(y, meanL), flush=True)

    cands = {
        "b7_3arm": ([b7["gap"], b7["gap_bag"], b7["plus"]], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]]),
        "b7+patch": (
            [b7["gap"], b7["gap_bag"], b7["plus"], patch],
            [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], tpatch],
        ),
        "b7+meanL": (
            [b7["gap"], b7["gap_bag"], b7["plus"], meanL],
            [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], tmeanL],
        ),
        "max3×patch": ([max3, patch], [tmax, tpatch]),
        "max3×meanL": ([max3, meanL], [tmax, tmeanL]),
        "b7+lo_raw": (
            [b7["gap"], b7["gap_bag"], b7["plus"], oof_lo],
            [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], test_lo],
        ),
    }
    results = {}
    best_name, best_res, best_arms = None, None, None
    for name, (oof_arms, te_arms) in cands.items():
        res = nested_select_rule(y.to_numpy(), oof_arms)
        results[name] = res
        print(
            f"{name}: nested={res['nested_oof_auc']:.8f} rule={res['selected_rule']} "
            f"folds={res['fold_rules']} full_max={res['full_data_scores'].get('max')}",
            flush=True,
        )
        if best_res is None or res["nested_oof_auc"] > best_res["nested_oof_auc"]:
            best_name, best_res, best_arms = name, res, (oof_arms, te_arms)

    deliver_auc = best_res["nested_oof_auc"]
    deliver_oof = best_res["nested_oof"]
    deliver_test = apply_rule(best_res["selected_rule"], best_arms[1])
    if deliver_auc + 1e-12 < B7_FLOOR:
        print("FALLBACK B7", flush=True)
        best_name = "b7_fallback"
        deliver_auc = float(roc_auc_score(y, max3))
        deliver_oof = max3
        deliver_test = tmax

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y=y.to_numpy(),
        oof=deliver_oof,
        test=deliver_test,
        oof_long_only=oof_lo,
        test_long_only=test_lo,
        oof_patch=patch,
        test_patch=tpatch,
        oof_meanL=meanL,
        test_meanL=tmeanL,
        max3=max3,
    )
    write_submission(sample, test, deliver_test, args.output_dir / "submission_b6pro.csv")
    metrics = {
        "experiment_id": f"b6pro_long_only_{args.builder}",
        "builder": args.builder,
        "min_days": args.min_days,
        "seeds": args.seeds,
        "best_fusion": best_name,
        "nested_oof_auc": deliver_auc,
        "selected_rule": best_res["selected_rule"],
        "fold_rules": best_res["fold_rules"],
        "full_data_scores": best_res["full_data_scores"],
        "all_candidate_nested": {k: float(v["nested_oof_auc"]) for k, v in results.items()},
        "long_only_slice_auc": float(roc_auc_score(y.to_numpy()[mask], oof_lo[mask])),
        "baseline_max3": B7_FLOOR,
        "gate_0_71": bool(deliver_auc >= GATE),
        "gap_to_0_71": float(GATE - deliver_auc),
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "long_only_specialist": True,
            "b7_floor_enforced": True,
            "no_oof_weight_search": True,
            "new_data_only": True,
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"GATE={'PASS' if metrics['gate_0_71'] else 'FAIL'} nested={deliver_auc:.8f}", flush=True)
    return 0 if metrics["gate_0_71"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
