"""B7 high-recall / class-weighted arm aimed at FN positives.

Heterogeneous from B6 (Balanced weights + deeper trees + residual cats).
Fuses with frozen max(B6, plus) via nested pre-registered rules.
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

from insurance_claim.b6_gap_features import GAP_CAT_COLS, add_gap_cats, fit_gap_edges
from insurance_claim.b7_fusion import FUSION_RULES, fuse_pair, nested_select_pair
from insurance_claim.model import TARGET, build_submission
from insurance_claim.train_b6 import enrich
from insurance_claim.v10_plus.plus_features import parse_frame

THREAD = 4

PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1400,
    learning_rate=0.025,
    depth=7,
    l2_leaf_reg=8,
    random_strength=1.0,
    auto_class_weights="Balanced",
    od_type="Iter",
    od_wait=120,
    verbose=False,
    thread_count=THREAD,
    allow_writing_files=False,
)


def build_recall_frame(X_tr, X_va, X_te):
    """Plus-lite numerics + gap cats + FN-oriented crosses."""
    edges = fit_gap_edges(X_tr)

    def one(raw, ref_edges):
        d = parse_frame(raw)
        # enrich for gap
        enriched = enrich(d)
        g = add_gap_cats(enriched, ref_edges)
        out = pd.DataFrame(index=d.index)
        for c in [
            "days",
            "condition",
            "cc",
            "V",
            "max_g",
            "x20",
            "month_n",
            "t3_num",
            "car",
            "version_n",
            "grades_ord",
            "age_range8",
            "w1",
            "w2",
            "t1",
            "t2",
            "r2",
            "c2",
        ]:
            out[c] = pd.to_numeric(d[c], errors="coerce")
        for i in range(19):
            out[f"x{i}"] = pd.to_numeric(d[f"x{i}"], errors="coerce")
        out["ratio"] = out["condition"] / (out["days"].abs() + 1.0)
        out["log_days"] = np.log1p(out["days"].clip(lower=0))
        out["region"] = d["region"].astype(str)
        out["source"] = d["source"].astype(str)
        out["code"] = d["code"].astype(str)
        out["version"] = d["version"].astype(str)
        out["month"] = d["month"].astype(str)
        out["livability"] = d["livability"].astype(str)
        out["t3_unit"] = d["t3_unit"].astype(str)
        out["grades"] = d["grades"].astype(str)
        # gap cats
        for c in GAP_CAT_COLS:
            if c in g.columns:
                out[c] = g[c].astype(str)
        # FN-oriented: region×month, source×code, car×days_bin
        d5 = pd.qcut(out["days"].rank(method="first"), 5, labels=False, duplicates="drop")
        out["d5"] = d5.astype(str)
        out["region_month"] = out["region"] + "|" + out["month"]
        out["source_code"] = out["source"] + "|" + out["code"]
        out["car_d5"] = out["car"].fillna(-1).astype(int).astype(str) + "|" + out["d5"]
        out["region_d5"] = out["region"] + "|" + out["d5"]
        out["t3_unit_code"] = out["t3_unit"] + "|" + out["code"]
        return out

    tr, va, te = one(X_tr, edges), one(X_va, edges), one(X_te, edges)
    cats = [
        "region",
        "source",
        "code",
        "version",
        "month",
        "livability",
        "t3_unit",
        "grades",
        "d5",
        "region_month",
        "source_code",
        "car_d5",
        "region_d5",
        "t3_unit_code",
    ] + [c for c in GAP_CAT_COLS if c in tr.columns]
    cats = list(dict.fromkeys(cats))
    for c in cats:
        for d in (tr, va, te):
            d[c] = d[c].astype(str).fillna("__MISSING__")
    for c in tr.columns:
        if c in cats:
            continue
        med = float(pd.to_numeric(tr[c], errors="coerce").median())
        for d in (tr, va, te):
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(med)
    return tr, va, te, cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b7_recall"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029])
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train[TARGET].astype(int)
    feats = train.drop(columns=[TARGET])

    b6 = np.load("artifacts/b6_gapbag_8seed/predictions.npz")
    eq = 0.5 * (b6["oof_gap"] + b6["oof_gap_bag"])
    plus = np.load("reference/v10/oof_plus_h2_10.npz")["oof"]
    stage1 = np.maximum(eq, plus)
    te_stage1 = np.maximum(
        0.5 * (b6["test_gap"] + b6["test_gap_bag"]),
        np.load("reference/v10/test_plus_h2_10.npy"),
    )

    t0 = time.time()
    oofs, tests, folds = [], [], []
    for seed in args.seeds:
        oof = np.zeros(len(train))
        pte = np.zeros(len(test))
        for fold, (a, b) in enumerate(
            StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(feats, y)
        ):
            Xtr, Xva = feats.iloc[a].reset_index(drop=True), feats.iloc[b].reset_index(drop=True)
            ytr, yva = y.iloc[a].reset_index(drop=True), y.iloc[b].reset_index(drop=True)
            tr, va, te, cats = build_recall_frame(Xtr, Xva, test.copy())
            p = dict(PARAMS)
            p["random_seed"] = seed + fold
            m = CatBoostClassifier(**p)
            m.fit(tr, ytr, eval_set=(va, yva), cat_features=cats, use_best_model=True)
            oof[b] = m.predict_proba(va)[:, 1]
            pte += m.predict_proba(te)[:, 1] / args.folds
            auc = float(roc_auc_score(yva, oof[b]))
            folds.append({"seed": seed, "fold": fold, "auc": auc, "best": int(m.get_best_iteration() or -1)})
            print(f"recall seed={seed} fold={fold} auc={auc:.5f} best={m.get_best_iteration()} n={tr.shape[1]}", flush=True)
        print(f"recall seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
        oofs.append(oof)
        tests.append(pte)

    stage2 = np.mean(np.vstack(oofs), 0)
    te2 = np.mean(np.vstack(tests), 0)
    nested = nested_select_pair(stage1, stage2, y.to_numpy())
    rule = nested["selected_rule"]
    oof_final = fuse_pair(stage1, stage2, rule)
    if rule == "rank_mean":
        from scipy.stats import rankdata

        te_final = 0.5 * (rankdata(te_stage1) + rankdata(te2))
        te_final = (te_final - te_final.min()) / (te_final.max() - te_final.min() + 1e-12)
    else:
        te_final = fuse_pair(te_stage1, te2, rule)

    metrics = {
        "experiment_id": "b7_recall_balanced",
        "stage1_auc": float(roc_auc_score(y, stage1)),
        "stage2_auc": float(roc_auc_score(y, stage2)),
        "corr_s1_s2": float(np.corrcoef(stage1, stage2)[0, 1]),
        "corr_b6": float(np.corrcoef(eq, stage2)[0, 1]),
        "corr_plus": float(np.corrcoef(plus, stage2)[0, 1]),
        "nested": {
            "selected_rule": rule,
            "nested_oof_auc": nested["nested_oof_auc"],
            "votes": nested["nested_rule_votes"],
            "full_scores": {
                r: float(roc_auc_score(y, fuse_pair(stage1, stage2, r))) for r in FUSION_RULES
            },
        },
        "max_s1_s2": float(roc_auc_score(y, np.maximum(stage1, stage2))),
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
            "auto_class_weights": "Balanced",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        oof=oof_final,
        test=te_final,
        y=y.to_numpy(),
        stage1=stage1,
        stage2=stage2,
        nested_oof=nested["nested_oof"],
    )
    build_submission(test, sample, te_final, args.output_dir / "submission_b7.csv")
    build_submission(test, sample, te_final, Path("submissions") / "submission_b7_recall.csv")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in (
                    "stage1_auc",
                    "stage2_auc",
                    "corr_s1_s2",
                    "corr_b6",
                    "corr_plus",
                    "nested",
                    "max_s1_s2",
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
