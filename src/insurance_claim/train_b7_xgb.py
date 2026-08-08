"""B7 heterogeneous XGBoost arm on plus-style numeric + residual cats (encoded).

Goal: near-strength OOF with lower corr to B6/plus than CatBoost clones.
Fold-local FE; no global TE.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from insurance_claim.b7_fusion import nested_select_pair, fuse_pair, FUSION_RULES
from insurance_claim.model import TARGET, build_submission
from insurance_claim.v10_plus.plus_features import parse_frame

THREAD = 4


def build_xgb_matrix(X_tr, X_va, X_te, y_tr=None):
    """Numeric matrix with fold-local target-free encodings."""
    def fe(df, ref):
        d = parse_frame(df)
        out = pd.DataFrame(index=d.index)
        for c in [
            "days", "condition", "cc", "V", "max_g", "x20",
            "month_n", "t3_num", "car", "version_n", "grades_ord", "age_range8",
        ]:
            out[c] = pd.to_numeric(d[c], errors="coerce")
        for i in range(19):
            out[f"x{i}"] = pd.to_numeric(d[f"x{i}"], errors="coerce")
        out["ratio"] = out["condition"] / (out["days"].abs() + 1.0)
        out["log_days"] = np.log1p(out["days"].clip(lower=0))
        out["w1"] = pd.to_numeric(d["w1"], errors="coerce")
        out["w2"] = pd.to_numeric(d["w2"], errors="coerce")
        out["w_both"] = ((out["w1"] == 1) & (out["w2"] == 1)).astype(float)
        out["t1"] = pd.to_numeric(d["t1"], errors="coerce")
        out["t2"] = pd.to_numeric(d["t2"], errors="coerce")
        out["r2"] = pd.to_numeric(d["r2"], errors="coerce")
        out["c2"] = pd.to_numeric(d["c2"], errors="coerce")
        # frequency encode cats from ref
        for col in ["region", "source", "code", "version", "livability", "t3_unit"]:
            vc = ref[col].astype(str).value_counts(normalize=True)
            out[f"{col}_freq"] = d[col].astype(str).map(vc).fillna(0.0)
        # fold-local qbin indices
        days = out["days"]
        cond = out["condition"]
        ratio = out["ratio"]
        for name, sref, s in [
            ("d5", pd.to_numeric(ref["days"], errors="coerce"), days),
            ("c5", pd.to_numeric(ref["condition"], errors="coerce"), cond),
            (
                "r5",
                pd.to_numeric(ref["condition"], errors="coerce")
                / (pd.to_numeric(ref["days"], errors="coerce").abs() + 1),
                ratio,
            ),
        ]:
            edges = np.unique(sref.dropna().quantile(np.linspace(0, 1, 6)).to_numpy())[1:-1]
            out[name] = np.searchsorted(edges, s.fillna(sref.median()).to_numpy(), side="right")
        # interactions
        out["region_d5"] = (
            pd.factorize(d["region"].astype(str) + "|" + out["d5"].astype(str))[0]
        )
        # Actually use freq of cross from ref
        cross = ref["region"].astype(str) + "|" + pd.Series(
            np.searchsorted(
                np.unique(pd.to_numeric(ref["days"], errors="coerce").dropna().quantile(np.linspace(0, 1, 6)).to_numpy())[1:-1],
                pd.to_numeric(ref["days"], errors="coerce").fillna(pd.to_numeric(ref["days"], errors="coerce").median()).to_numpy(),
                side="right",
            ).astype(str),
            index=ref.index,
        )
        vc = cross.value_counts(normalize=True)
        key = d["region"].astype(str) + "|" + out["d5"].astype(str)
        out["region_d5_freq"] = key.map(vc).fillna(0.0)
        out["cond5_source_freq"] = (
            out["c5"].astype(str) + "|" + d["source"].astype(str)
        ).map(
            (
                pd.Series(
                    np.searchsorted(
                        np.unique(
                            pd.to_numeric(ref["condition"], errors="coerce")
                            .dropna()
                            .quantile(np.linspace(0, 1, 6))
                            .to_numpy()
                        )[1:-1],
                        pd.to_numeric(ref["condition"], errors="coerce")
                        .fillna(pd.to_numeric(ref["condition"], errors="coerce").median())
                        .to_numpy(),
                        side="right",
                    ).astype(str),
                    index=ref.index,
                )
                + "|"
                + ref["source"].astype(str)
            ).value_counts(normalize=True)
        ).fillna(0.0)
        return out

    ref = parse_frame(X_tr)
    tr = fe(X_tr, ref)
    va = fe(X_va, ref)
    te = fe(X_te, ref)
    cols = tr.columns.tolist()
    for df in (tr, va, te):
        for c in cols:
            med = float(pd.to_numeric(tr[c], errors="coerce").median())
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(med)
    return tr[cols].to_numpy(), va[cols].to_numpy(), te[cols].to_numpy(), cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b7_xgb"))
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
            tr, va, te, cols = build_xgb_matrix(Xtr, Xva, test.copy())
            m = XGBClassifier(
                n_estimators=1200,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.85,
                colsample_bytree=0.7,
                reg_lambda=8.0,
                min_child_weight=5,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                n_jobs=THREAD,
                random_state=seed + fold,
                early_stopping_rounds=80,
            )
            m.fit(tr, ytr, eval_set=[(va, yva)], verbose=False)
            oof[b] = m.predict_proba(va)[:, 1]
            pte += m.predict_proba(te)[:, 1] / args.folds
            auc = float(roc_auc_score(yva, oof[b]))
            folds.append({"seed": seed, "fold": fold, "auc": auc, "best": int(m.best_iteration)})
            print(f"xgb seed={seed} fold={fold} auc={auc:.5f} best={m.best_iteration} n={len(cols)}", flush=True)
        print(f"xgb seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
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
        "experiment_id": "b7_xgb_hetero",
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
            "hetero_model": "xgboost",
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
    build_submission(test, sample, te_final, Path("submissions") / "submission_b7_xgb.csv")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ("stage1_auc", "stage2_auc", "corr_s1_s2", "corr_b6", "corr_plus", "nested", "max_s1_s2", "gate_0_71", "gap_to_0_71", "elapsed_sec")}, indent=2))


if __name__ == "__main__":
    main()
