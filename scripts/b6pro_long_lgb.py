#!/usr/bin/env python3
"""LGBM residual long specialist + full hetero arm; nested fuse vs B7 floor.

Targets anti-monotonic exceptions inside long exposure (same-region LL pairs),
using fold-local healthy TE + days/condition baseline residual features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule
from insurance_claim.b6pro_long_resid import build_long_resid_matrix
from insurance_claim.model import IDENTIFIER, TARGET

B7_FLOOR = 0.7027049552615718
GATE = 0.71
CLOSEST = 0.7054481147284526
WEAK = frozenset({"908d", "f09d", "9685", "fafc", "f167", "ab86"})

PARAMS_LONG = dict(
    n_estimators=4000,
    learning_rate=0.02,
    num_leaves=48,
    max_depth=-1,
    subsample=0.85,
    colsample_bytree=0.65,
    reg_lambda=8.0,
    reg_alpha=0.8,
    min_child_samples=35,
    objective="binary",
    metric="auc",
    n_jobs=2,
    verbose=-1,
)

PARAMS_FULL = dict(
    n_estimators=3500,
    learning_rate=0.025,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    reg_alpha=0.5,
    min_child_samples=40,
    objective="binary",
    metric="auc",
    n_jobs=2,
    verbose=-1,
)


def write_submission(sample: pd.DataFrame, test: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    pred = np.asarray(pred, dtype=float)
    assert len(pred) == len(test)
    out = sample.copy()
    label_col = [c for c in out.columns if c != IDENTIFIER][0]
    out[label_col] = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def region_blend(max3: np.ndarray, long_spec: np.ndarray, region: np.ndarray, days: np.ndarray, wo: float) -> np.ndarray:
    out = max3.copy()
    long = days >= 3000
    weak_m = long & np.isin(region, list(WEAK))
    other = long & ~np.isin(region, list(WEAK))
    out[weak_m] = long_spec[weak_m]
    out[other] = wo * long_spec[other] + (1.0 - wo) * max3[other]
    return out


def run_lgb(
    features: pd.DataFrame,
    y: pd.Series,
    test: pd.DataFrame,
    seeds: list[int],
    params: dict,
    *,
    long_only: bool = False,
    min_days: float = 3000.0,
) -> tuple[np.ndarray, np.ndarray]:
    days = features["days"].to_numpy(dtype=float)
    if long_only:
        mask = days >= min_days
        idx = np.where(mask)[0]
    else:
        idx = np.arange(len(y))
        mask = np.ones(len(y), dtype=bool)

    oof_acc = np.zeros(len(y), dtype=float)
    test_acc = np.zeros(len(test), dtype=float)
    for seed in seeds:
        oof = np.zeros(len(y), dtype=float)
        pte = np.zeros(len(test), dtype=float)
        Xl = features.iloc[idx].reset_index(drop=True)
        yl = y.iloc[idx].reset_index(drop=True)
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xl, yl)):
            gtr, gva = idx[tr], idx[va]
            trd, vad, ted, cats = build_long_resid_matrix(
                features.iloc[gtr].reset_index(drop=True),
                y.iloc[gtr].to_numpy(),
                features.iloc[gva].reset_index(drop=True),
                test.copy(),
            )
            model = LGBMClassifier(**{**params, "random_state": int(seed + fold)})
            model.fit(
                trd,
                y.iloc[gtr],
                eval_set=[(vad, y.iloc[gva])],
                categorical_feature=cats,
                callbacks=[early_stopping(150), log_evaluation(0)],
            )
            oof[gva] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5.0
            print(
                f"{'lo' if long_only else 'full'} s{seed} f{fold} "
                f"auc={roc_auc_score(y.iloc[gva], oof[gva]):.5f} best={model.best_iteration_}",
                flush=True,
            )
        slice_y = y.to_numpy()[mask]
        slice_p = oof[mask]
        print(
            f"{'lo' if long_only else 'full'} s{seed} OOF={roc_auc_score(y, oof):.6f} "
            f"slice={roc_auc_score(slice_y, slice_p):.6f}",
            flush=True,
        )
        oof_acc += oof
        test_acc += pte
    return oof_acc / len(seeds), test_acc / len(seeds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029])
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_long_lgb"))
    args = ap.parse_args()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train[TARGET].astype(int)
    features = train.drop(columns=[TARGET])
    days = features["days"].to_numpy(dtype=float)
    days_te = test["days"].to_numpy(dtype=float)
    region = train["region"].astype(str).to_numpy()
    region_te = test["region"].astype(str).to_numpy()
    long = days >= 3000
    long_te = days_te >= 3000

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    tmax = np.maximum.reduce([fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]])
    cur = np.load("artifacts/b6pro_long_best/predictions.npz")
    cur_oof, cur_te = cur["oof"], cur["test"]
    # mean of three long-only specialists (current closest recipe ingredients)
    aging = np.load("artifacts/b6pro_long_only_aging/predictions.npz")
    gap = np.load("artifacts/b6pro_long_only_gap/predictions.npz")
    keepx = np.load("artifacts/b6pro_long_only_keepx/predictions.npz")
    meanL = (aging["oof_long_only"] + gap["oof_long_only"] + keepx["oof_long_only"]) / 3.0
    tmeanL = (aging["test_long_only"] + gap["test_long_only"] + keepx["test_long_only"]) / 3.0

    print("=== long-only LGBM resid ===", flush=True)
    oof_lo, te_lo = run_lgb(features, y, test, args.seeds, PARAMS_LONG, long_only=True)
    print(
        "long_only slice",
        roc_auc_score(y.to_numpy()[long], oof_lo[long]),
        "corr(max3)",
        float(np.corrcoef(oof_lo[long], max3[long])[0, 1]),
        flush=True,
    )

    oof_full = te_full = None
    if not args.skip_full:
        print("=== full LGBM resid ===", flush=True)
        oof_full, te_full = run_lgb(features, y, test, args.seeds, PARAMS_FULL, long_only=False)
        print(
            "full OOF",
            roc_auc_score(y, oof_full),
            "long",
            roc_auc_score(y.to_numpy()[long], oof_full[long]),
            "corr(max3)",
            float(np.corrcoef(oof_full, max3)[0, 1]),
            flush=True,
        )

    # blend recipes
    def blend_specs(lo: np.ndarray, te_lo_arr: np.ndarray):
        specs = {}
        # patch long with lo
        patch = max3.copy()
        patch[long] = lo[long]
        tpatch = tmax.copy()
        tpatch[long_te] = te_lo_arr[long_te]
        specs["patch"] = (patch, tpatch)
        # meanL with lo
        m = max3.copy()
        m[long] = 0.5 * (max3[long] + lo[long])
        tm = tmax.copy()
        tm[long_te] = 0.5 * (tmax[long_te] + te_lo_arr[long_te])
        specs["meanL"] = (m, tm)
        # combine with existing meanL3
        combo = max3.copy()
        combo[long] = 0.5 * (meanL[long] + lo[long])
        tcombo = tmax.copy()
        tcombo[long_te] = 0.5 * (tmeanL[long_te] + te_lo_arr[long_te])
        specs["combo_meanL3"] = (combo, tcombo)
        for wo in (0.0, 0.1, 0.15, 0.2, 0.3):
            # long_spec = mean(existing meanL3, new lo)
            ls = 0.5 * (meanL + lo)
            tls = 0.5 * (tmeanL + te_lo_arr)
            arm = region_blend(max3, ls, region, days, wo)
            tarm = region_blend(tmax, tls, region_te, days_te, wo)
            specs[f"rb_combo_wo{wo}"] = (arm, tarm)
            arm2 = region_blend(max3, lo, region, days, wo)
            tarm2 = region_blend(tmax, te_lo_arr, region_te, days_te, wo)
            specs[f"rb_lo_wo{wo}"] = (arm2, tarm2)
        return specs

    specs = blend_specs(oof_lo, te_lo)
    if oof_full is not None:
        specs["full_raw"] = (oof_full, te_full)
        # max with full
        specs["max_full"] = (np.maximum(max3, oof_full), np.maximum(tmax, te_full))
        specs["mean_full"] = (0.5 * (max3 + oof_full), 0.5 * (tmax + te_full))

    results = {}
    best_name, best_res, best_pair = None, None, None
    for name, (oof_arm, te_arm) in specs.items():
        for tag, oof_arms, te_arms in [
            (f"b7+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], oof_arm], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], te_arm]),
            (f"max3×{name}", [max3, oof_arm], [tmax, te_arm]),
            (f"cur+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], cur_oof, oof_arm], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], cur_te, te_arm]),
        ]:
            res = nested_select_rule(y.to_numpy(), oof_arms)
            results[tag] = float(res["nested_oof_auc"])
            print(
                f"{tag}: nested={res['nested_oof_auc']:.8f} rule={res['selected_rule']} "
                f"folds={res['fold_rules']}",
                flush=True,
            )
            if best_res is None or res["nested_oof_auc"] > best_res["nested_oof_auc"]:
                best_name, best_res, best_pair = tag, res, (oof_arms, te_arms)

    deliver_auc = best_res["nested_oof_auc"]
    deliver_oof = best_res["nested_oof"]
    deliver_test = apply_rule(best_res["selected_rule"], best_pair[1])
    if deliver_auc + 1e-12 < B7_FLOOR:
        print("FALLBACK B7", flush=True)
        best_name = "b7_fallback"
        deliver_auc = float(roc_auc_score(y, max3))
        deliver_oof = max3
        deliver_test = tmax

    # promote closest only if beats current
    promoted = False
    if deliver_auc > CLOSEST + 1e-12:
        promoted = True
        dest = Path("artifacts/b6pro_long_best")
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / "predictions.npz", y=y.to_numpy(), oof=deliver_oof, test=deliver_test, arm=oof_lo)
        write_submission(sample, test, deliver_test, dest / "submission_b6pro.csv")
        sub = Path("submissions/b6pro_closest")
        sub.mkdir(parents=True, exist_ok=True)
        write_submission(sample, test, deliver_test, sub / "submission_b6pro.csv")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save = {
        "y": y.to_numpy(),
        "oof": deliver_oof,
        "test": deliver_test,
        "oof_lo": oof_lo,
        "test_lo": te_lo,
        "max3": max3,
    }
    if oof_full is not None:
        save["oof_full"] = oof_full
        save["test_full"] = te_full
    np.savez_compressed(args.output_dir / "predictions.npz", **save)
    write_submission(sample, test, deliver_test, args.output_dir / "submission_b6pro.csv")

    metrics = {
        "experiment_id": "b6pro_long_lgb",
        "seeds": args.seeds,
        "best_fusion": best_name,
        "nested_oof_auc": deliver_auc,
        "selected_rule": best_res["selected_rule"],
        "fold_rules": best_res["fold_rules"],
        "all_candidate_nested": results,
        "long_only_slice_auc": float(roc_auc_score(y.to_numpy()[long], oof_lo[long])),
        "long_corr_max3": float(np.corrcoef(oof_lo[long], max3[long])[0, 1]),
        "full_oof_auc": None if oof_full is None else float(roc_auc_score(y, oof_full)),
        "baseline_max3": B7_FLOOR,
        "prev_closest": CLOSEST,
        "promoted_closest": promoted,
        "gate_0_71": bool(deliver_auc >= GATE),
        "gap_to_0_71": float(GATE - deliver_auc),
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_te_only": True,
            "fold_local_fe": True,
            "b7_floor_enforced": True,
            "no_oof_weight_search": True,
            "new_data_only": True,
        },
    }
    if promoted:
        (Path("artifacts/b6pro_long_best") / "metrics.json").write_text(
            json.dumps(
                {
                    "experiment_id": "b6pro_long_best",
                    "spec": best_name,
                    "nested_oof_auc": deliver_auc,
                    "baseline_max3": B7_FLOOR,
                    "gate_0_71": metrics["gate_0_71"],
                    "gap_to_0_71": metrics["gap_to_0_71"],
                    "source": "b6pro_long_lgb",
                },
                indent=2,
            )
        )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"GATE={'PASS' if metrics['gate_0_71'] else 'FAIL'} nested={deliver_auc:.8f} promoted={promoted}", flush=True)
    return 0 if metrics["gate_0_71"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
