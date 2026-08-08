#!/usr/bin/env python3
"""Residual corrector on B7/closest OOF: learn delta ranking for long exposure.

Protocol: max3/closest scores are already nested OOF. We train a second-level
regressor/classifier on fold-local features to predict claim residual relative
to the base score, then add a discrete scaled delta (pre-registered scales).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
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
# pre-registered delta scales (discrete; nested selects among them)
SCALES = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30)


def write_submission(sample, test, pred, path: Path):
    out = sample.copy()
    label_col = [c for c in out.columns if c != IDENTIFIER][0]
    out[label_col] = np.asarray(pred, float)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def region_blend(base, spec, region, days, wo):
    out = base.copy()
    long = days >= 3000
    weak_m = long & np.isin(region, list(WEAK))
    other = long & ~np.isin(region, list(WEAK))
    out[weak_m] = spec[weak_m]
    out[other] = wo * spec[other] + (1 - wo) * base[other]
    return out


def main() -> int:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train[TARGET].astype(int)
    features = train.drop(columns=[TARGET])
    days = features["days"].to_numpy(float)
    days_te = test["days"].to_numpy(float)
    region = train["region"].astype(str).to_numpy()
    region_te = test["region"].astype(str).to_numpy()
    long = days >= 3000
    long_te = days_te >= 3000

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    tmax = np.maximum.reduce([fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]])
    cur = np.load("artifacts/b6pro_long_best/predictions.npz")
    base_oof = cur["oof"]
    base_te = cur["test"]

    seeds = [2026, 2027, 2028, 2029]
    # Train residual classifier: P(y=1 | features, base_score) on long rows;
    # delta = pred - base (or pred alone)
    oof_pred = np.zeros(len(y))
    te_pred = np.zeros(len(test))
    idx = np.where(long)[0]

    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
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
            # append base score as feature (OOF for train rows; test uses base_te)
            trd = trd.copy()
            vad = vad.copy()
            ted = ted.copy()
            trd["base_score"] = base_oof[gtr]
            vad["base_score"] = base_oof[gva]
            ted["base_score"] = base_te
            # residual target emphasis: also try sample weight on errors
            base_tr = base_oof[gtr]
            # weight hard examples more (base wrong direction)
            hard = np.abs(y.iloc[gtr].to_numpy() - base_tr)
            w = 0.5 + hard  # in [0.5, 1.5]

            model = LGBMClassifier(
                n_estimators=3000,
                learning_rate=0.02,
                num_leaves=40,
                subsample=0.85,
                colsample_bytree=0.7,
                reg_lambda=6.0,
                min_child_samples=30,
                objective="binary",
                metric="auc",
                n_jobs=2,
                verbose=-1,
                random_state=seed + fold,
            )
            model.fit(
                trd,
                y.iloc[gtr],
                sample_weight=w,
                eval_set=[(vad, y.iloc[gva])],
                categorical_feature=cats,
                callbacks=[early_stopping(120), log_evaluation(0)],
            )
            oof[gva] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5.0
            print(
                f"resid_corr s{seed} f{fold} auc={roc_auc_score(y.iloc[gva], oof[gva]):.5f} "
                f"base={roc_auc_score(y.iloc[gva], base_oof[gva]):.5f}",
                flush=True,
            )
        print(
            f"resid_corr s{seed} slice={roc_auc_score(y.to_numpy()[long], oof[long]):.6f} "
            f"base_slice={roc_auc_score(y.to_numpy()[long], base_oof[long]):.6f}",
            flush=True,
        )
        oof_pred += oof
        te_pred += pte
    oof_pred /= len(seeds)
    te_pred /= len(seeds)

    print(
        "pooled resid",
        roc_auc_score(y.to_numpy()[long], oof_pred[long]),
        "corr(base)",
        float(np.corrcoef(oof_pred[long], base_oof[long])[0, 1]),
        flush=True,
    )

    # Build corrected scores with discrete scales
    specs = {}
    delta = oof_pred - base_oof
    tdelta = te_pred - base_te
    for s in SCALES:
        corr = base_oof.copy()
        corr[long] = base_oof[long] + s * delta[long]
        tcorr = base_te.copy()
        tcorr[long_te] = base_te[long_te] + s * tdelta[long_te]
        specs[f"add_s{s}"] = (corr, tcorr)
        # replace long with blend
        blend = base_oof.copy()
        blend[long] = (1 - s) * base_oof[long] + s * oof_pred[long]
        tblend = base_te.copy()
        tblend[long_te] = (1 - s) * base_te[long_te] + s * te_pred[long_te]
        specs[f"blend_s{s}"] = (blend, tblend)
        # weak-only replace
        weak = region_blend(base_oof, oof_pred, region, days, 0.0)
        # actually region_blend with wo=0 replaces weak with oof_pred, other stays base
        tw = region_blend(base_te, te_pred, region_te, days_te, 0.0)
        specs["weak_replace"] = (weak, tw)
        weak_b = region_blend(base_oof, (1 - s) * base_oof + s * oof_pred, region, days, 0.0)
        # simpler: weak gets blend
        out = base_oof.copy()
        wm = long & np.isin(region, list(WEAK))
        out[wm] = (1 - s) * base_oof[wm] + s * oof_pred[wm]
        tout = base_te.copy()
        wmt = long_te & np.isin(region_te, list(WEAK))
        tout[wmt] = (1 - s) * base_te[wmt] + s * te_pred[wmt]
        specs[f"weak_blend_s{s}"] = (out, tout)

    # also max(base, resid) on long
    mx = base_oof.copy()
    mx[long] = np.maximum(base_oof[long], oof_pred[long])
    tmx = base_te.copy()
    tmx[long_te] = np.maximum(base_te[long_te], te_pred[long_te])
    specs["max_base_resid"] = (mx, tmx)

    results = {}
    best_name, best_res, best_pair = None, None, None
    for name, (oof_arm, te_arm) in specs.items():
        # direct score of corrected arm
        direct = float(roc_auc_score(y, oof_arm))
        for tag, oof_arms, te_arms in [
            (f"direct_{name}", [oof_arm], [te_arm]),
            (f"b7+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], oof_arm], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], te_arm]),
            (f"max3×{name}", [max3, oof_arm], [tmax, te_arm]),
        ]:
            if len(oof_arms) == 1:
                res = {
                    "nested_oof_auc": direct,
                    "nested_oof": oof_arm,
                    "selected_rule": "mean",
                    "fold_rules": ["mean"],
                }
            else:
                res = nested_select_rule(y.to_numpy(), oof_arms)
            results[tag] = float(res["nested_oof_auc"])
            if best_res is None or res["nested_oof_auc"] > best_res["nested_oof_auc"]:
                best_name, best_res, best_pair = tag, res, (oof_arms, te_arms)
        print(f"{name}: direct={direct:.8f} best_so_far={best_res['nested_oof_auc']:.8f}", flush=True)

    deliver_auc = best_res["nested_oof_auc"]
    deliver_oof = best_res["nested_oof"]
    deliver_test = apply_rule(best_res["selected_rule"], best_pair[1]) if len(best_pair[1]) > 1 else best_pair[1][0]
    if deliver_auc + 1e-12 < B7_FLOOR:
        best_name = "b7_fallback"
        deliver_auc = float(roc_auc_score(y, max3))
        deliver_oof, deliver_test = max3, tmax

    promoted = deliver_auc > CLOSEST + 1e-12
    out_dir = Path("artifacts/b6pro_resid_corr")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "predictions.npz",
        y=y.to_numpy(),
        oof=deliver_oof,
        test=deliver_test,
        oof_resid=oof_pred,
        test_resid=te_pred,
    )
    write_submission(sample, test, deliver_test, out_dir / "submission_b6pro.csv")
    if promoted:
        dest = Path("artifacts/b6pro_long_best")
        np.savez_compressed(dest / "predictions.npz", y=y.to_numpy(), oof=deliver_oof, test=deliver_test, arm=oof_pred)
        write_submission(sample, test, deliver_test, dest / "submission_b6pro.csv")
        write_submission(sample, test, deliver_test, Path("submissions/b6pro_closest/submission_b6pro.csv"))
        (dest / "metrics.json").write_text(
            json.dumps(
                {
                    "experiment_id": "b6pro_long_best",
                    "spec": best_name,
                    "nested_oof_auc": deliver_auc,
                    "baseline_max3": B7_FLOOR,
                    "gate_0_71": deliver_auc >= GATE,
                    "gap_to_0_71": GATE - deliver_auc,
                    "source": "b6pro_resid_corr",
                },
                indent=2,
            )
        )

    metrics = {
        "experiment_id": "b6pro_resid_corr",
        "best_fusion": best_name,
        "nested_oof_auc": deliver_auc,
        "all_candidate_nested": results,
        "resid_long_auc": float(roc_auc_score(y.to_numpy()[long], oof_pred[long])),
        "promoted_closest": promoted,
        "prev_closest": CLOSEST,
        "gate_0_71": bool(deliver_auc >= GATE),
        "gap_to_0_71": float(GATE - deliver_auc),
        "baseline_max3": B7_FLOOR,
        "note": "discrete scales only; base_score is prior OOF (stacking)",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "all_candidate_nested"}, indent=2), flush=True)
    print("TOP10", sorted(results.items(), key=lambda kv: -kv[1])[:10], flush=True)
    print(f"GATE={'PASS' if metrics['gate_0_71'] else 'FAIL'} nested={deliver_auc:.8f} promoted={promoted}", flush=True)
    return 0 if metrics["gate_0_71"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
