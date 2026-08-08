#!/usr/bin/env python3
"""Kitchen-sink CatBoost: B5+gap + x0-18 + plus-style cats; aim strong solo then fuse."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule
from insurance_claim.b6pro_plus import build_plus_gap
from insurance_claim.model import build_submission
from insurance_claim.train_b6 import build_gap
from insurance_claim.v10_plus.plus_features import build_plus, parse_frame

TARGET = 0.71
PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=2000,
    learning_rate=0.02,
    depth=7,
    l2_leaf_reg=16,
    random_strength=1.0,
    bagging_temperature=0.8,
    od_type="Iter",
    od_wait=150,
    verbose=False,
    thread_count=4,
    allow_writing_files=False,
)


def build_sink(X_tr, X_va, X_te):
    # gap view
    gtr, gva, gte, gcats = build_gap(X_tr, X_va, X_te)
    # plus view on parsed
    Ptr = parse_frame(X_tr.copy())
    Pva = parse_frame(X_va.copy())
    Pte = parse_frame(X_te.copy())
    for c in ("id", "x19"):
        for d in (Ptr, Pva, Pte):
            if c in d.columns:
                d.drop(columns=[c], inplace=True)
    ptr, pva, pte, pcats = build_plus(Ptr, Pva, Pte)

    def merge(a, b):
        out = pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=1)
        return out.loc[:, ~out.columns.duplicated()]

    tr, va, te = merge(gtr, ptr), merge(gva, pva), merge(gte, pte)
    va = va.reindex(columns=tr.columns)
    te = te.reindex(columns=tr.columns)
    cats = []
    for c in tr.columns:
        if c in gcats or c in pcats:
            cats.append(c)
        elif not pd.api.types.is_numeric_dtype(tr[c]):
            cats.append(c)
    cats = list(dict.fromkeys(cats))
    for c in cats:
        for d in (tr, va, te):
            d[c] = d[c].astype(str).fillna("__NA__")
    for c in tr.columns:
        if c in cats:
            continue
        for d, src in ((tr, tr), (va, tr), (te, tr)):
            d[c] = pd.to_numeric(d[c], errors="coerce")
            med = float(src[c].median()) if src[c].notna().any() else 0.0
            d[c] = d[c].fillna(med)
    return tr, va, te, cats


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train["label"].astype(int)
    features = train.drop(columns=["label"])
    seeds = [2026, 2027, 2028, 2029]
    oofs, tests = [], []
    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(features, y)):
            trd, vad, ted, cats = build_sink(
                features.iloc[tr].reset_index(drop=True),
                features.iloc[va].reset_index(drop=True),
                test.copy(),
            )
            p = dict(PARAMS)
            p["random_seed"] = seed + fold
            model = CatBoostClassifier(**p)
            model.fit(trd, y.iloc[tr], eval_set=(vad, y.iloc[va]), cat_features=cats, use_best_model=True)
            oof[va] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5
            print(f"sink seed={seed} fold={fold} auc={roc_auc_score(y.iloc[va], oof[va]):.5f} n={trd.shape[1]}", flush=True)
        print(f"sink seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
        oofs.append(oof)
        tests.append(pte)
    oof = np.mean(oofs, 0)
    te = np.mean(tests, 0)
    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    print("sink solo", roc_auc_score(y, oof), "corr", np.corrcoef(oof, 0.5 * (b7["gap"] + b7["gap_bag"]))[0, 1])
    fused = nested_select_rule(y.to_numpy(), [b7["gap"], b7["gap_bag"], b7["plus"], oof])
    print("nested", fused["nested_oof_auc"], fused["selected_rule"])
    print("full", fused["full_data_scores"])
    tp = apply_rule(fused["selected_rule"], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], te])
    out = Path("artifacts/b6pro_sink")
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "predictions.npz", y=y.to_numpy(), oof=fused["nested_oof"], test=tp, oof_sink=oof, test_sink=te)
    build_submission(test, sample, tp, out / "submission_b6pro.csv")
    metrics = {
        "experiment_id": "b6pro_sink",
        "oof_auc": float(roc_auc_score(y, oof)),
        "nested_oof_auc": fused["nested_oof_auc"],
        "selected_rule": fused["selected_rule"],
        "full_data_scores": fused["full_data_scores"],
        "baseline_max3": 0.7027049552615718,
        "gate_0_71": fused["nested_oof_auc"] >= TARGET,
        "gap_to_0_71": round(TARGET - fused["nested_oof_auc"], 6),
        "public_b7_signal": {"local": 0.702704955, "public": 0.707},
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "no_oof_weight_search": True,
            "new_data_only": True,
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ["oof_auc", "nested_oof_auc", "gate_0_71", "gap_to_0_71"]}, indent=2))


if __name__ == "__main__":
    main()
