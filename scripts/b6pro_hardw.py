#!/usr/bin/env python3
"""Hard-residual weighted CatBoost: upweight anti-monotonic / days-baseline errors.

Business: LL wrong pairs are anti-monotonic in days/condition. A days-isotonic
baseline identifies residual risk; sample weights force keepx trees to fit those.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule
from insurance_claim.b6pro_long_features import build_long_keepx
from insurance_claim.train_b6 import PARAMS_GAP_BAG

B7_FLOOR = 0.7027049552615718
GATE = 0.71
CLOSEST = float(json.load(open("artifacts/b6pro_long_best/metrics.json"))["nested_oof_auc"])
WEAK = frozenset({"908d", "f09d", "9685", "fafc", "f167", "ab86"})
PARAMS = {**PARAMS_GAP_BAG, "thread_count": 4, "iterations": 3500, "od_wait": 160, "l2_leaf_reg": 10}


def days_baseline_weights(days_tr, y_tr, days_all_for_pred=None):
    """Isotonic days→y on train; return residual-based sample weights for train rows."""
    days_tr = np.asarray(days_tr, float)
    y_tr = np.asarray(y_tr, float)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(days_tr, y_tr)
    pred = iso.predict(days_tr)
    resid = np.abs(y_tr - pred)
    # weights in ~[1, 4]
    w = 1.0 + 3.0 * (resid / (resid.max() + 1e-6))
    return w, iso


def main() -> int:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train["label"].astype(int)
    features = train.drop(columns=["label"])
    days = features["days"].to_numpy(float)
    days_te = test["days"].to_numpy(float)
    region = train["region"].astype(str).to_numpy()
    region_te = test["region"].astype(str).to_numpy()
    long = days >= 3000

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    tmax = np.maximum.reduce([fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]])
    cur = np.load("artifacts/b6pro_long_best/predictions.npz")
    fk = np.load("artifacts/b6pro_full_keepx/predictions.npz")
    aging = np.load("artifacts/b6pro_long_only_aging/predictions.npz")
    gap = np.load("artifacts/b6pro_long_only_gap/predictions.npz")
    keepx = np.load("artifacts/b6pro_long_only_keepx/predictions.npz")
    meanL3 = (aging["oof_long_only"] + gap["oof_long_only"] + keepx["oof_long_only"]) / 3.0
    tmeanL3 = (aging["test_long_only"] + gap["test_long_only"] + keepx["test_long_only"]) / 3.0

    seeds = [2026, 2027, 2028, 2029]
    oof_acc = np.zeros(len(y))
    te_acc = np.zeros(len(test))
    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(features, y)):
            w, iso = days_baseline_weights(days[tr], y.iloc[tr].to_numpy())
            # extra boost for long rows that are residual-hard
            long_tr = days[tr] >= 3000
            w = w.copy()
            w[long_tr] *= 1.25
            trd, vad, ted, cats = build_long_keepx(
                features.iloc[tr].reset_index(drop=True),
                features.iloc[va].reset_index(drop=True),
                test.copy(),
            )
            # append isotonic baseline as feature
            trd = trd.copy()
            vad = vad.copy()
            ted = ted.copy()
            trd["iso_days"] = iso.predict(days[tr])
            vad["iso_days"] = iso.predict(days[va])
            ted["iso_days"] = iso.predict(days_te)
            model = CatBoostClassifier(**{**PARAMS, "random_seed": seed + fold})
            model.fit(
                trd,
                y.iloc[tr],
                sample_weight=w,
                eval_set=(vad, y.iloc[va]),
                cat_features=cats,
                use_best_model=True,
            )
            oof[va] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5.0
            print(
                f"hardw s{seed} f{fold} {roc_auc_score(y.iloc[va], oof[va]):.5f} "
                f"long={roc_auc_score(y.iloc[va][days[va]>=3000], oof[va][days[va]>=3000]):.5f}",
                flush=True,
            )
        print(
            f"hardw s{seed} OOF={roc_auc_score(y, oof):.6f} "
            f"long={roc_auc_score(y.to_numpy()[long], oof[long]):.6f}",
            flush=True,
        )
        oof_acc += oof
        te_acc += pte
    oof_h, te_h = oof_acc / len(seeds), te_acc / len(seeds)
    print(
        "pooled",
        roc_auc_score(y, oof_h),
        "long",
        roc_auc_score(y.to_numpy()[long], oof_h[long]),
        "corr(max3)",
        float(np.corrcoef(oof_h, max3)[0, 1]),
        "corr(kx)",
        float(np.corrcoef(oof_h, fk["oof_k"])[0, 1]),
        flush=True,
    )

    def rb(base, spec, reg, d, wo):
        out = base.copy()
        longm = d >= 3000
        weak = longm & np.isin(reg, list(WEAK))
        other = longm & ~np.isin(reg, list(WEAK))
        out[weak] = spec[weak]
        out[other] = wo * spec[other] + (1 - wo) * base[other]
        return out

    arms = {
        "raw": (oof_h, te_h),
        "mean_m3": (0.5 * (max3 + oof_h), 0.5 * (tmax + te_h)),
        "max_m3": (np.maximum(max3, oof_h), np.maximum(tmax, te_h)),
        "mean_kx": (0.5 * (fk["oof_k"] + oof_h), 0.5 * (fk["te_k"] + te_h)),
        "mix_mL": (0.5 * (meanL3 + oof_h), 0.5 * (tmeanL3 + te_h)),
    }
    for wo in (0.0, 0.15, 0.2):
        mix = 0.5 * (meanL3 + oof_h)
        tmix = 0.5 * (tmeanL3 + te_h)
        arms[f"rb_mix_w{wo}"] = (
            rb(max3, mix, region, days, wo),
            rb(tmax, tmix, region_te, days_te, wo),
        )
        arms[f"rb_h_w{wo}"] = (
            rb(max3, oof_h, region, days, wo),
            rb(tmax, te_h, region_te, days_te, wo),
        )

    results = {}
    best_name, best_res, best_pair = None, None, None
    for name, (oa, ta) in arms.items():
        for tag, oof_arms, te_arms in [
            (f"b7+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], oa], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], ta]),
            (f"cur+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], cur["oof"], oa], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], cur["test"], ta]),
            (f"b7+kx+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], fk["oof_k"], oa], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], fk["te_k"], ta]),
            (f"max3×{name}", [max3, oa], [tmax, ta]),
        ]:
            res = nested_select_rule(y.to_numpy(), oof_arms)
            results[tag] = float(res["nested_oof_auc"])
            print(f"{tag}: {res['nested_oof_auc']:.8f}", flush=True)
            if best_res is None or res["nested_oof_auc"] > best_res["nested_oof_auc"]:
                best_name, best_res, best_pair = tag, res, (oof_arms, te_arms)

    deliver = best_res["nested_oof_auc"]
    deliver_oof = best_res["nested_oof"]
    deliver_test = apply_rule(best_res["selected_rule"], best_pair[1])
    if deliver < B7_FLOOR:
        deliver = float(roc_auc_score(y, max3))
        deliver_oof, deliver_test = max3, tmax
        best_name = "b7_fallback"
    promoted = deliver > CLOSEST + 1e-12
    out = Path("artifacts/b6pro_hardw")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", y=y.to_numpy(), oof=deliver_oof, test=deliver_test, oof_h=oof_h, te_h=te_h)
    lab = [c for c in sample.columns if c != "id"][0]
    sub = sample.copy()
    sub[lab] = deliver_test
    sub.to_csv(out / "submission_b6pro.csv", index=False)
    if promoted:
        dest = Path("artifacts/b6pro_long_best")
        np.savez_compressed(dest / "predictions.npz", y=y.to_numpy(), oof=deliver_oof, test=deliver_test, arm=oof_h)
        sub.to_csv(dest / "submission_b6pro.csv", index=False)
        sub.to_csv("submissions/b6pro_closest/submission_b6pro.csv", index=False)
        (dest / "metrics.json").write_text(
            json.dumps(
                {
                    "experiment_id": "b6pro_long_best",
                    "spec": best_name,
                    "nested_oof_auc": deliver,
                    "baseline_max3": B7_FLOOR,
                    "gate_0_71": deliver >= GATE,
                    "gap_to_0_71": GATE - deliver,
                    "source": "b6pro_hardw",
                },
                indent=2,
            )
        )
    metrics = {
        "best": best_name,
        "nested": deliver,
        "hardw": float(roc_auc_score(y, oof_h)),
        "hardw_long": float(roc_auc_score(y.to_numpy()[long], oof_h[long])),
        "promoted": promoted,
        "gate": deliver >= GATE,
        "top": sorted(results.items(), key=lambda kv: -kv[1])[:15],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"GATE={'PASS' if deliver >= GATE else 'FAIL'} nested={deliver:.8f} promoted={promoted}", flush=True)
    return 0 if deliver >= GATE else 2


if __name__ == "__main__":
    raise SystemExit(main())
