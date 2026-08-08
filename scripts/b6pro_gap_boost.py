#!/usr/bin/env python3
"""Gap+residual-mining cats arm (fold-local) + Ordered gap arm; fuse vs B7 max3."""

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
from insurance_claim.train_b6 import PARAMS_B5, build_gap
from insurance_claim.model import build_submission

TARGET = 0.71


def add_resid_cats(df: pd.DataFrame, edges_days, edges_cond, edges_ratio) -> pd.DataFrame:
    out = df.copy()
    days = pd.to_numeric(out.get("days"), errors="coerce")
    cond = pd.to_numeric(out.get("condition"), errors="coerce")
    # crude ratio proxy if present
    if "V" in out and "cc" in out:
        ratio = pd.to_numeric(out["V"], errors="coerce") / pd.to_numeric(out["cc"], errors="coerce").replace(0, np.nan)
    else:
        ratio = pd.Series(np.nan, index=out.index)
    d5 = pd.cut(days, bins=[-np.inf, *edges_days, np.inf], labels=False).astype("float").astype(str)
    c5 = pd.cut(cond, bins=[-np.inf, *edges_cond, np.inf], labels=False).astype("float").astype(str)
    r5 = pd.cut(ratio, bins=[-np.inf, *edges_ratio, np.inf], labels=False).astype("float").astype(str)
    region = out.get("region", pd.Series("__NA__", index=out.index)).astype(str)
    source = out.get("source", pd.Series("__NA__", index=out.index)).astype(str)
    age = out.get("age_range", pd.Series(-1, index=out.index)).astype(str)
    liv = out.get("livability", pd.Series(-1, index=out.index)).astype(str)
    # parse car from source
    car = source.str.extract(r"CAR_(\d+)")[0].fillna("__NA__")
    out["resid_region_car_d5"] = region + "|" + car + "|" + d5
    out["resid_cond5_source"] = c5 + "|" + source
    out["resid_ratio5_region"] = r5 + "|" + region
    out["resid_age_r5"] = age + "|" + r5
    out["resid_liv_d5"] = liv + "|" + d5
    return out


def build_gap_resid(X_tr, X_va, X_te):
    tr, va, te, cats = build_gap(X_tr, X_va, X_te)
    days = pd.to_numeric(X_tr["days"], errors="coerce")
    cond = pd.to_numeric(X_tr["condition"], errors="coerce")
    ratio = pd.to_numeric(X_tr["V"], errors="coerce") / pd.to_numeric(X_tr["cc"], errors="coerce").replace(0, np.nan)
    edges_days = np.unique(np.nanquantile(days, np.linspace(0, 1, 6)))[1:-1]
    edges_cond = np.unique(np.nanquantile(cond.dropna(), np.linspace(0, 1, 6)))[1:-1] if cond.notna().any() else np.array([0.0])
    edges_ratio = np.unique(np.nanquantile(ratio.replace([np.inf, -np.inf], np.nan).dropna(), np.linspace(0, 1, 6)))[1:-1]
    if len(edges_ratio) < 1:
        edges_ratio = np.array([1.0])

    def merge(base, raw):
        extra = add_resid_cats(raw.reset_index(drop=True), edges_days, edges_cond, edges_ratio)
        cols = [c for c in extra.columns if c.startswith("resid_")]
        out = pd.concat([base.reset_index(drop=True), extra[cols]], axis=1)
        return out.loc[:, ~out.columns.duplicated()]

    tr = merge(tr, X_tr)
    va = merge(va, X_va).reindex(columns=tr.columns)
    te = merge(te, X_te).reindex(columns=tr.columns)
    resid_cols = [c for c in tr.columns if c.startswith("resid_")]
    cats = list(dict.fromkeys(list(cats) + resid_cols))
    for c in resid_cols:
        for df in (tr, va, te):
            df[c] = df[c].astype(str).fillna("__NA__")
    return tr, va, te, cats


def run_arm(builder, name, features, y, test, seeds, params):
    oofs, tests = [], []
    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(features, y)):
            Xtr = features.iloc[tr].reset_index(drop=True)
            Xva = features.iloc[va].reset_index(drop=True)
            trd, vad, ted, cats = builder(Xtr, Xva, test.copy())
            p = dict(params)
            p["random_seed"] = seed + fold
            model = CatBoostClassifier(**p)
            model.fit(trd, y.iloc[tr], eval_set=(vad, y.iloc[va]), cat_features=cats, use_best_model=True)
            oof[va] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5
            print(f"{name} seed={seed} fold={fold} auc={roc_auc_score(y.iloc[va], oof[va]):.5f}", flush=True)
        print(f"{name} seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
        oofs.append(oof)
        tests.append(pte)
    return np.mean(oofs, 0), np.mean(tests, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["resid", "ordered", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029])
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_gap_resid"))
    args = ap.parse_args()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train["label"].astype(int)
    features = train.drop(columns=["label"])
    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")

    arms_oof = [b7["gap"], b7["gap_bag"], b7["plus"]]
    arms_te = [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]]
    names = ["gap", "gap_bag", "plus"]

    if args.mode in ("resid", "both"):
        params = {**PARAMS_B5, "thread_count": 4}
        oof, te = run_arm(build_gap_resid, "gap_resid", features, y, test, args.seeds, params)
        print("gap_resid solo", roc_auc_score(y, oof), "corr", np.corrcoef(oof, b7["gap"])[0, 1])
        arms_oof.append(oof)
        arms_te.append(te)
        names.append("gap_resid")

    if args.mode in ("ordered", "both"):
        params = {**PARAMS_B5, "thread_count": 4, "boosting_type": "Ordered", "iterations": 1000}
        # Ordered is slow — fewer seeds if both
        seeds = args.seeds[:2] if args.mode == "both" else args.seeds
        oof, te = run_arm(build_gap, "gap_ordered", features, y, test, seeds, params)
        print("gap_ordered solo", roc_auc_score(y, oof), "corr", np.corrcoef(oof, b7["gap"])[0, 1])
        arms_oof.append(oof)
        arms_te.append(te)
        names.append("gap_ordered")

    fused = nested_select_rule(y.to_numpy(), arms_oof)
    print("arm aucs", {n: float(roc_auc_score(y, a)) for n, a in zip(names, arms_oof)})
    print("nested", fused["nested_oof_auc"], fused["selected_rule"], fused["full_data_scores"])

    tp = apply_rule(fused["selected_rule"], arms_te)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y=y.to_numpy(),
        oof=fused["nested_oof"],
        test=tp,
        **{f"oof_{n}": a for n, a in zip(names, arms_oof)},
    )
    build_submission(test, sample, tp, args.output_dir / "submission_b6pro.csv")
    metrics = {
        "experiment_id": f"b6pro_gap_{args.mode}",
        "arm_aucs": {n: float(roc_auc_score(y, a)) for n, a in zip(names, arms_oof)},
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
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ["nested_oof_auc", "gate_0_71", "gap_to_0_71", "arm_aucs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
