"""B7 residual corrector: nested second-stage model on fusion residuals.

Stage-1 = max(B6_equal, plus_v10) [frozen].
Stage-2 = CatBoost on residual-oriented cats/nums, OOF nested.
Final fusion among pre-registered rules on (stage1, stage2).
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

THREAD = 8
PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1200,
    learning_rate=0.03,
    depth=5,
    l2_leaf_reg=12,
    random_strength=0.8,
    od_type="Iter",
    od_wait=100,
    verbose=False,
    thread_count=THREAD,
    allow_writing_files=False,
)


def build_residual_frame(X_tr, X_va, X_te):
    def fe(df, ref):
        out = pd.DataFrame(index=df.index)
        days = pd.to_numeric(df["days"], errors="coerce")
        cond = pd.to_numeric(df["condition"], errors="coerce")
        ratio = cond / (days.abs() + 1.0)
        out["days"] = days
        out["condition"] = cond
        out["ratio"] = ratio
        out["log_days"] = np.log1p(days.clip(lower=0))
        out["log_cond"] = np.log1p(cond.clip(lower=0))
        # fold-local qbins from ref
        for name, series_ref, series in [
            ("d5", pd.to_numeric(ref["days"], errors="coerce"), days),
            ("c5", pd.to_numeric(ref["condition"], errors="coerce"), cond),
            ("r5", pd.to_numeric(ref["condition"], errors="coerce") / (pd.to_numeric(ref["days"], errors="coerce").abs() + 1), ratio),
        ]:
            edges = np.unique(series_ref.dropna().quantile(np.linspace(0, 1, 6)).to_numpy())[1:-1]
            out[name] = pd.Series(np.searchsorted(edges, series.fillna(series_ref.median()).to_numpy(), side="right"), index=df.index).astype(str)
        out["region"] = df["region"].astype(str)
        out["source"] = df["source"].astype(str)
        out["code"] = df["code"].astype(str)
        out["version"] = df["version"].astype(str)
        out["livability"] = df["livability"].astype(str)
        w1 = pd.to_numeric(df["w1"], errors="coerce").fillna(-1).astype(int)
        w2 = pd.to_numeric(df["w2"], errors="coerce").fillna(-1).astype(int)
        out["w_pair"] = w1.astype(str) + "_" + w2.astype(str)
        t3 = df["t3"].astype(str).str.extract(r"([A-Za-z]+)$")[0].fillna("__N__")
        out["t3_sfx"] = t3
        age = pd.to_numeric(df["age_range"], errors="coerce").clip(upper=8).fillna(-1).astype(int)
        out["age_c"] = age.astype(str)
        car = df["source"].astype(str).str.extract(r"(CAR_\d+)")[0].fillna("__NA__")
        out["car"] = car
        # crosses highlighted by residual mining
        out["cond5_source"] = out["c5"] + "|" + out["source"]
        out["ratio5_region"] = out["r5"] + "|" + out["region"]
        out["age_r5"] = out["age_c"] + "|" + out["r5"]
        out["liv_d5"] = out["livability"] + "|" + out["d5"]
        out["w_d5"] = out["w_pair"] + "|" + out["d5"]
        out["t3_code_d5"] = out["t3_sfx"] + "|" + out["code"] + "|" + out["d5"]
        # latent summary (plus-like diversity)
        xs = df[[f"x{i}" for i in range(19)]].apply(pd.to_numeric, errors="coerce")
        out["x_mean"] = xs.mean(axis=1)
        out["x_std"] = xs.std(axis=1)
        out["x19"] = df["x19"].astype(str) if "x19" in df.columns else "__NA__"
        out["x20"] = pd.to_numeric(df["x20"], errors="coerce")
        return out

    tr, va, te = fe(X_tr, X_tr), fe(X_va, X_tr), fe(X_te, X_tr)
    cats = [
        "d5", "c5", "r5", "region", "source", "code", "version", "livability",
        "w_pair", "t3_sfx", "age_c", "car", "cond5_source", "ratio5_region",
        "age_r5", "liv_d5", "w_d5", "t3_code_d5", "x19",
    ]
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
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b7_resid"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029])
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train[TARGET].astype(int)
    feats = train.drop(columns=[TARGET])

    b6 = np.load("artifacts/b6_gapbag_8seed/predictions.npz")
    plus = np.load("reference/v10/oof_plus_h2_10.npz")
    eq = 0.5 * (b6["oof_gap"] + b6["oof_gap_bag"])
    stage1 = np.maximum(eq, plus["oof"])
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
            tr, va, te, cats = build_residual_frame(Xtr, Xva, test.copy())
            # optionally include stage1 score as feature (fold-safe: use OOF stage1)
            tr = tr.copy(); va = va.copy(); te = te.copy()
            tr["stage1"] = stage1[a]
            va["stage1"] = stage1[b]
            te["stage1"] = te_stage1
            p = dict(PARAMS)
            p["random_seed"] = seed + fold
            m = CatBoostClassifier(**p)
            m.fit(tr, ytr, eval_set=(va, yva), cat_features=cats, use_best_model=True)
            oof[b] = m.predict_proba(va)[:, 1]
            pte += m.predict_proba(te)[:, 1] / args.folds
            auc = float(roc_auc_score(yva, oof[b]))
            folds.append({"seed": seed, "fold": fold, "auc": auc, "best": int(m.get_best_iteration() or -1)})
            print(f"resid seed={seed} fold={fold} auc={auc:.5f} best={m.get_best_iteration()}", flush=True)
        print(f"resid seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
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
        "experiment_id": "b7_residual_corrector",
        "stage1_auc": float(roc_auc_score(y, stage1)),
        "stage2_auc": float(roc_auc_score(y, stage2)),
        "corr_s1_s2": float(np.corrcoef(stage1, stage2)[0, 1]),
        "nested": {
            "selected_rule": rule,
            "nested_oof_auc": nested["nested_oof_auc"],
            "votes": nested["nested_rule_votes"],
            "full_scores": {r: float(roc_auc_score(y, fuse_pair(stage1, stage2, r))) for r in FUSION_RULES},
        },
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
    build_submission(test, sample, te_final, Path("submissions") / "submission_b7_resid.csv")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ("stage1_auc", "stage2_auc", "corr_s1_s2", "nested", "gate_0_71", "gap_to_0_71", "elapsed_sec")}, indent=2))


if __name__ == "__main__":
    main()
