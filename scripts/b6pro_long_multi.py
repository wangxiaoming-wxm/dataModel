#!/usr/bin/env python3
"""Multi-threshold / extra-seed long-only specialists fused with B7."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule
from insurance_claim.train_b6 import PARAMS_GAP_BAG, build_gap

B7_FLOOR = 0.7027049552615718
GATE = 0.71


def train_slice(features, y, test, days, min_days, seeds, params):
    mask = days >= min_days
    idx = np.where(mask)[0]
    oof_acc = np.zeros(len(y))
    te_acc = np.zeros(len(test))
    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        Xl = features.iloc[idx]
        yl = y.iloc[idx]
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xl, yl)):
            gtr, gva = idx[tr], idx[va]
            trd, vad, ted, cats = build_gap(
                features.iloc[gtr].reset_index(drop=True),
                features.iloc[gva].reset_index(drop=True),
                test.copy(),
            )
            model = CatBoostClassifier(**{**params, "random_seed": int(seed + fold), "thread_count": 4})
            model.fit(
                trd, y.iloc[gtr], eval_set=(vad, y.iloc[gva]), cat_features=cats, use_best_model=True
            )
            oof[gva] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5.0
            print(
                f"d{int(min_days)} seed={seed} fold={fold} auc={roc_auc_score(y.iloc[gva], oof[gva]):.5f}",
                flush=True,
            )
        print(
            f"d{int(min_days)} seed={seed} sliceOOF={roc_auc_score(y.to_numpy()[mask], oof[mask]):.6f}",
            flush=True,
        )
        oof_acc += oof
        te_acc += pte
    return oof_acc / len(seeds), te_acc / len(seeds), mask


def make_meanL(base, arm, mask, w=0.7):
    out = base.copy()
    out[mask] = w * arm[mask] + (1 - w) * base[mask]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_long_multi"))
    args = ap.parse_args()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train["label"].astype(int)
    features = train.drop(columns=["label"])
    days = features["days"].to_numpy(dtype=float)
    days_te = test["days"].to_numpy(dtype=float)

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    tmax = np.maximum.reduce([fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]])

    prev = np.load("artifacts/b6pro_long_only_gap/predictions.npz")
    oof_lo3, te_lo3 = prev["oof_long_only"], prev["test_long_only"]

    params = dict(PARAMS_GAP_BAG)
    print("=== extra seeds 3k ===", flush=True)
    oof_extra, te_extra, mask3 = train_slice(
        features, y, test, days, 3000, [2030, 2031, 2032, 2033], params
    )
    oof_8 = (oof_lo3 + oof_extra) / 2
    te_8 = (te_lo3 + te_extra) / 2
    print("8seed long slice", roc_auc_score(y.to_numpy()[mask3], oof_8[mask3]), flush=True)

    print("=== 7k ===", flush=True)
    oof_7, te_7, mask7 = train_slice(features, y, test, days, 7000, [2026, 2027, 2028, 2029], params)
    print("=== 10k ===", flush=True)
    oof_10, te_10, mask10 = train_slice(features, y, test, days, 10000, [2026, 2027], params)

    meanL3 = make_meanL(max3, oof_8, mask3)
    meanL7 = make_meanL(max3, oof_7, mask7)
    meanL10 = make_meanL(max3, oof_10, mask10)
    casc = max3.copy()
    casc[mask3] = 0.7 * oof_8[mask3] + 0.3 * casc[mask3]
    casc[mask7] = 0.7 * oof_7[mask7] + 0.3 * casc[mask7]
    casc[mask10] = 0.7 * oof_10[mask10] + 0.3 * casc[mask10]

    arms = {
        "meanL3": meanL3,
        "meanL7": meanL7,
        "meanL10": meanL10,
        "casc": casc,
        "lo8": oof_8,
    }
    t_meanL3 = make_meanL(tmax, te_8, days_te >= 3000)
    t_meanL7 = make_meanL(tmax, te_7, days_te >= 7000)
    t_meanL10 = make_meanL(tmax, te_10, days_te >= 10000)
    t_casc = tmax.copy()
    t_casc[days_te >= 3000] = 0.7 * te_8[days_te >= 3000] + 0.3 * t_casc[days_te >= 3000]
    t_casc[days_te >= 7000] = 0.7 * te_7[days_te >= 7000] + 0.3 * t_casc[days_te >= 7000]
    t_casc[days_te >= 10000] = 0.7 * te_10[days_te >= 10000] + 0.3 * t_casc[days_te >= 10000]
    t_arms = {
        "meanL3": t_meanL3,
        "meanL7": t_meanL7,
        "meanL10": t_meanL10,
        "casc": t_casc,
        "lo8": te_8,
    }

    cands = []
    for n, a in arms.items():
        cands.append((f"b7+{n}", [b7["gap"], b7["gap_bag"], b7["plus"], a], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], t_arms[n]]))
    for a, b in combinations(["meanL3", "meanL7", "casc"], 2):
        cands.append(
            (
                f"b7+{a}+{b}",
                [b7["gap"], b7["gap_bag"], b7["plus"], arms[a], arms[b]],
                [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], t_arms[a], t_arms[b]],
            )
        )
    cands.append(
        (
            "b7+all",
            [b7["gap"], b7["gap_bag"], b7["plus"], meanL3, meanL7, meanL10, casc],
            [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], t_meanL3, t_meanL7, t_meanL10, t_casc],
        )
    )

    best_name, best_res, best_te = None, None, None
    results = {}
    for name, oof_arms, te in cands:
        res = nested_select_rule(y.to_numpy(), oof_arms)
        results[name] = res["nested_oof_auc"]
        print(f"{name}: {res['nested_oof_auc']:.8f} {res['selected_rule']}", flush=True)
        if best_res is None or res["nested_oof_auc"] > best_res["nested_oof_auc"]:
            best_name, best_res, best_te = name, res, te

    deliver_auc = best_res["nested_oof_auc"]
    deliver_oof = best_res["nested_oof"]
    deliver_test = apply_rule(best_res["selected_rule"], best_te)
    if deliver_auc + 1e-12 < B7_FLOOR:
        best_name, deliver_auc, deliver_oof, deliver_test = "b7_fallback", B7_FLOOR, max3, tmax

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y=y.to_numpy(),
        oof=deliver_oof,
        test=deliver_test,
        oof_lo8=oof_8,
        oof_lo7=oof_7,
        oof_lo10=oof_10,
        meanL3=meanL3,
        casc=casc,
        max3=max3,
    )
    sub = sample.copy()
    sub[sub.columns[1]] = deliver_test
    sub.to_csv(args.output_dir / "submission_b6pro.csv", index=False)
    metrics = {
        "experiment_id": "b6pro_long_multi",
        "best_fusion": best_name,
        "nested_oof_auc": float(deliver_auc),
        "baseline_max3": B7_FLOOR,
        "gate_0_71": bool(deliver_auc >= GATE),
        "gap_to_0_71": float(GATE - deliver_auc),
        "all_candidate_nested": results,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    return 0 if metrics["gate_0_71"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
