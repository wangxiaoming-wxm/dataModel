#!/usr/bin/env python3
"""CatBoost long residual specialist: native string crosses + fold-local healthy TE.

Complements existing aging/gap/keepx long-only arms with:
- days_fixed business windows
- ratio×region / t3_sfx×code×days cats
- fold-local credibility TE numerics (healthy crosses only)
- x0-x18 latents (keepx-style) for anti-monotonic residual ranking
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
from insurance_claim.b6pro_long_features import build_long_keepx, build_long_aging
from insurance_claim.b6pro_long_resid import (
    DAYS_FIXED_EDGES,
    DAYS_FIXED_LABELS,
    _days_fixed,
    _key_frame,
    _oof_te_map,
    _apply_te,
    fit_resid_edges,
)
from insurance_claim.model import IDENTIFIER, TARGET
from insurance_claim.train_b6 import PARAMS_GAP_BAG

B7_FLOOR = 0.7027049552615718
GATE = 0.71
CLOSEST = 0.7054481147284526
WEAK = frozenset({"908d", "f09d", "9685", "fafc", "f167", "ab86"})

EXTRA_CATS = (
    "resid_days_fixed",
    "resid_region_days5",
    "resid_ratio_region",
    "resid_car_days5",
    "resid_wpair_days5",
    "resid_t3sfx_code_days5",
    "resid_ultra_long",
    "resid_mid_long",
)

PARAMS = {
    **PARAMS_GAP_BAG,
    "depth": 7,
    "l2_leaf_reg": 12,
    "learning_rate": 0.03,
    "iterations": 3000,
    "od_wait": 150,
    "thread_count": 2,
}


def write_submission(sample: pd.DataFrame, test: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample.copy()
    label_col = [c for c in out.columns if c != IDENTIFIER][0]
    out[label_col] = np.asarray(pred, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def build_long_resid_cb(
    X_tr: pd.DataFrame, y_tr, X_va: pd.DataFrame, X_te: pd.DataFrame
):
    """keepx base + residual business cats + fold-local TE numerics."""
    tr, va, te, cats = build_long_keepx(X_tr, X_va, X_te)
    edges = fit_resid_edges(X_tr)
    ktr = _key_frame(X_tr, edges)
    kva = _key_frame(X_va, edges)
    kte = _key_frame(X_te, edges)
    y_arr = np.asarray(y_tr, dtype=float)
    gmean = float(np.mean(y_arr))
    te_specs = {
        "te_region_days5": "region_days5",
        "te_ratio_region": "ratio_region",
        "te_car_days5": "car_days5",
        "te_wpair_days5": "wpair_days5",
        "te_t3sfx_code_days5": "t3sfx_code_days5",
        "te_days_fixed": "days_fixed",
    }
    maps = {n: _oof_te_map(ktr[c], y_arr) for n, c in te_specs.items()}

    def attach(base: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
        out = base.copy()
        out["resid_days_fixed"] = keys["days_fixed"].astype(str)
        out["resid_region_days5"] = keys["region_days5"].astype(str)
        out["resid_ratio_region"] = keys["ratio_region"].astype(str)
        out["resid_car_days5"] = keys["car_days5"].astype(str)
        out["resid_wpair_days5"] = keys["wpair_days5"].astype(str)
        out["resid_t3sfx_code_days5"] = keys["t3sfx_code_days5"].astype(str)
        days = pd.to_numeric(out["days"] if "days" in out.columns else keys.index.to_series(), errors="coerce")
        # prefer raw days from keys length match via base
        if "days" in base.columns:
            days = pd.to_numeric(base["days"], errors="coerce")
        else:
            # recover from long_log_days if needed
            days = pd.Series(np.nan, index=base.index)
        out["resid_ultra_long"] = ((pd.to_numeric(keys.get("days_fixed", pd.Series("__NA__")), errors="coerce") * 0) + 0).astype(str)
        # use days_fixed label instead
        out["resid_ultra_long"] = keys["days_fixed"].eq("d10k_p").astype(int).astype(str)
        out["resid_mid_long"] = keys["days_fixed"].eq("d5k_7k").astype(int).astype(str)
        for n, c in te_specs.items():
            out[n] = _apply_te(keys[c], maps[n], gmean)
        return out

    tr = attach(tr, ktr)
    va = attach(va, kva).reindex(columns=tr.columns)
    te = attach(te, kte).reindex(columns=tr.columns)
    cats = list(dict.fromkeys(list(cats) + list(EXTRA_CATS)))
    for c in cats:
        for d in (tr, va, te):
            d[c] = d[c].astype(str).fillna("__MISSING__")
    for c in tr.columns:
        if c in cats:
            continue
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
        med = float(tr[c].median()) if tr[c].notna().any() else 0.0
        tr[c] = tr[c].fillna(med)
        va[c] = pd.to_numeric(va[c], errors="coerce").fillna(med)
        te[c] = pd.to_numeric(te[c], errors="coerce").fillna(med)
    return tr, va, te, cats


def region_blend(max3, long_spec, region, days, wo):
    out = max3.copy()
    long = days >= 3000
    weak_m = long & np.isin(region, list(WEAK))
    other = long & ~np.isin(region, list(WEAK))
    out[weak_m] = long_spec[weak_m]
    out[other] = wo * long_spec[other] + (1.0 - wo) * max3[other]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029])
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_long_resid_cb"))
    args = ap.parse_args()

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
    idx = np.where(long)[0]

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    tmax = np.maximum.reduce([fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]])
    cur = np.load("artifacts/b6pro_long_best/predictions.npz")
    aging = np.load("artifacts/b6pro_long_only_aging/predictions.npz")
    gap = np.load("artifacts/b6pro_long_only_gap/predictions.npz")
    keepx = np.load("artifacts/b6pro_long_only_keepx/predictions.npz")
    meanL3 = (aging["oof_long_only"] + gap["oof_long_only"] + keepx["oof_long_only"]) / 3.0
    tmeanL3 = (aging["test_long_only"] + gap["test_long_only"] + keepx["test_long_only"]) / 3.0

    oof_acc = np.zeros(len(y))
    test_acc = np.zeros(len(test))
    for seed in args.seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        Xl = features.iloc[idx].reset_index(drop=True)
        yl = y.iloc[idx].reset_index(drop=True)
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xl, yl)):
            gtr, gva = idx[tr], idx[va]
            trd, vad, ted, cats = build_long_resid_cb(
                features.iloc[gtr].reset_index(drop=True),
                y.iloc[gtr],
                features.iloc[gva].reset_index(drop=True),
                test.copy(),
            )
            model = CatBoostClassifier(**{**PARAMS, "random_seed": int(seed + fold)})
            model.fit(trd, y.iloc[gtr], eval_set=(vad, y.iloc[gva]), cat_features=cats, use_best_model=True)
            oof[gva] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5.0
            print(
                f"resid_cb s{seed} f{fold} auc={roc_auc_score(y.iloc[gva], oof[gva]):.5f}",
                flush=True,
            )
        print(
            f"resid_cb s{seed} slice={roc_auc_score(y.to_numpy()[long], oof[long]):.6f}",
            flush=True,
        )
        oof_acc += oof
        test_acc += pte
    oof_lo = oof_acc / len(args.seeds)
    te_lo = test_acc / len(args.seeds)
    print(
        "pooled slice",
        roc_auc_score(y.to_numpy()[long], oof_lo[long]),
        "corr(max3)",
        float(np.corrcoef(oof_lo[long], max3[long])[0, 1]),
        "corr(meanL3)",
        float(np.corrcoef(oof_lo[long], meanL3[long])[0, 1]),
        flush=True,
    )

    # 4-specialist mean
    meanL4 = (meanL3 * 3 + oof_lo) / 4.0
    tmeanL4 = (tmeanL3 * 3 + te_lo) / 4.0
    # equal mean of 4
    mean_eq = (aging["oof_long_only"] + gap["oof_long_only"] + keepx["oof_long_only"] + oof_lo) / 4.0
    tmean_eq = (aging["test_long_only"] + gap["test_long_only"] + keepx["test_long_only"] + te_lo) / 4.0

    specs = {}
    for wo in (0.0, 0.1, 0.15, 0.2, 0.25):
        for sn, ls, tls in [
            ("m3", meanL3, tmeanL3),
            ("m4", mean_eq, tmean_eq),
            ("lo", oof_lo, te_lo),
            ("mix", 0.5 * (meanL3 + oof_lo), 0.5 * (tmeanL3 + te_lo)),
        ]:
            arm = region_blend(max3, ls, region, days, wo)
            tarm = region_blend(tmax, tls, region_te, days_te, wo)
            specs[f"rb_{sn}_wo{wo}"] = (arm, tarm)

    results = {}
    best_name, best_res, best_pair = None, None, None
    for name, (oof_arm, te_arm) in specs.items():
        for tag, oof_arms, te_arms in [
            (f"b7+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], oof_arm], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], te_arm]),
            (f"cur+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], cur["oof"], oof_arm], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], cur["test"], te_arm]),
        ]:
            res = nested_select_rule(y.to_numpy(), oof_arms)
            results[tag] = float(res["nested_oof_auc"])
            print(f"{tag}: nested={res['nested_oof_auc']:.8f} rule={res['selected_rule']}", flush=True)
            if best_res is None or res["nested_oof_auc"] > best_res["nested_oof_auc"]:
                best_name, best_res, best_pair = tag, res, (oof_arms, te_arms)

    deliver_auc = best_res["nested_oof_auc"]
    deliver_oof = best_res["nested_oof"]
    deliver_test = apply_rule(best_res["selected_rule"], best_pair[1])
    if deliver_auc + 1e-12 < B7_FLOOR:
        best_name = "b7_fallback"
        deliver_auc = float(roc_auc_score(y, max3))
        deliver_oof, deliver_test = max3, tmax

    promoted = deliver_auc > CLOSEST + 1e-12
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y=y.to_numpy(),
        oof=deliver_oof,
        test=deliver_test,
        oof_lo=oof_lo,
        test_lo=te_lo,
        meanL4=mean_eq,
    )
    write_submission(sample, test, deliver_test, args.output_dir / "submission_b6pro.csv")
    if promoted:
        dest = Path("artifacts/b6pro_long_best")
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / "predictions.npz", y=y.to_numpy(), oof=deliver_oof, test=deliver_test, arm=oof_arm)
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
                    "source": "b6pro_long_resid_cb",
                },
                indent=2,
            )
        )

    metrics = {
        "experiment_id": "b6pro_long_resid_cb",
        "best_fusion": best_name,
        "nested_oof_auc": deliver_auc,
        "selected_rule": best_res["selected_rule"],
        "all_candidate_nested": results,
        "long_only_slice_auc": float(roc_auc_score(y.to_numpy()[long], oof_lo[long])),
        "prev_closest": CLOSEST,
        "promoted_closest": promoted,
        "gate_0_71": bool(deliver_auc >= GATE),
        "gap_to_0_71": float(GATE - deliver_auc),
        "baseline_max3": B7_FLOOR,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"GATE={'PASS' if metrics['gate_0_71'] else 'FAIL'} nested={deliver_auc:.8f} promoted={promoted}", flush=True)
    return 0 if metrics["gate_0_71"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
