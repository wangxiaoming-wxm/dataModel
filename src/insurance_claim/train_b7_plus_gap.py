"""B7 enhanced plus: V10 root_plus + B6 gap cats + optional x19_cat (fold-local, no TE)."""

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
from insurance_claim.model import TARGET, audit_data, build_submission
from insurance_claim.train_b5_focus import enrich
from insurance_claim.v10_plus.plus_features import build_plus

THREAD_COUNT = 8
SEEDS_DEFAULT = (2026, 2027, 2028, 2029)

PARAMS = {
    "h2": dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2500,
        learning_rate=0.02,
        depth=7,
        l2_leaf_reg=20,
        random_strength=1.0,
        od_type="Iter",
        od_wait=150,
        verbose=False,
        thread_count=THREAD_COUNT,
        allow_writing_files=False,
    ),
    "h3": dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=3000,
        learning_rate=0.015,
        depth=8,
        l2_leaf_reg=30,
        random_strength=2.0,
        od_type="Iter",
        od_wait=200,
        verbose=False,
        thread_count=THREAD_COUNT,
        allow_writing_files=False,
    ),
}


def build_plus_gap(X_tr, X_va, X_te, with_x19: bool = True):
    tr, va, te, cats = build_plus(X_tr, X_va, X_te)
    edges = fit_gap_edges(X_tr)

    def gap_part(raw):
        g = add_gap_cats(enrich(raw), edges)
        cols = list(GAP_CAT_COLS)
        if with_x19 and "x19" in raw.columns:
            g = g.copy()
            g["gap_x19_cat"] = raw["x19"].astype(str)
            cols = cols + ["gap_x19_cat"]
        return g.loc[:, [c for c in cols if c in g.columns]]

    gtr, gva, gte = gap_part(X_tr), gap_part(X_va), gap_part(X_te)

    def merge(base, extra):
        out = pd.concat([base.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        return out.loc[:, ~out.columns.duplicated()]

    tr = merge(tr, gtr)
    va = merge(va, gva).reindex(columns=tr.columns)
    te = merge(te, gte).reindex(columns=tr.columns)
    extra_cats = [c for c in tr.columns if c.startswith("gap_")]
    cats = list(dict.fromkeys(list(cats) + extra_cats))
    for c in cats:
        for d in (tr, va, te):
            if c in d.columns:
                d[c] = d[c].astype(str).fillna("__MISSING__")
    return tr, va, te, cats


def run(train, test, y, seeds, n_splits, params, builder):
    feats = train.drop(columns=[TARGET])
    oof_by_seed, test_by_seed, folds = {}, {}, []
    for seed in seeds:
        oof = np.zeros(len(train))
        pte = np.zeros(len(test))
        for fold, (a, b) in enumerate(
            StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(feats, y)
        ):
            Xtr, Xva = feats.iloc[a].reset_index(drop=True), feats.iloc[b].reset_index(drop=True)
            ytr, yva = y.iloc[a].reset_index(drop=True), y.iloc[b].reset_index(drop=True)
            tr, va, te, cats = builder(Xtr, Xva, test.copy())
            p = dict(params)
            p["random_seed"] = seed + fold
            m = CatBoostClassifier(**p)
            m.fit(tr, ytr, eval_set=(va, yva), cat_features=cats, use_best_model=True)
            oof[b] = m.predict_proba(va)[:, 1]
            pte += m.predict_proba(te)[:, 1] / n_splits
            auc = float(roc_auc_score(yva, oof[b]))
            folds.append({"seed": seed, "fold": fold, "valid_auc": auc, "best": int(m.get_best_iteration() or -1), "n": int(tr.shape[1])})
            print(f"seed={seed} fold={fold} auc={auc:.5f} best={m.get_best_iteration()} n={tr.shape[1]}", flush=True)
        print(f"seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
        oof_by_seed[seed] = oof
        test_by_seed[seed] = pte
    oof = np.mean(np.vstack(list(oof_by_seed.values())), 0)
    te = np.mean(np.vstack(list(test_by_seed.values())), 0)
    return oof, te, oof_by_seed, test_by_seed, folds, float(roc_auc_score(y, oof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b7_plus_gap"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    ap.add_argument("--folds", type=int, default=5)  # screen with 5; full later 10
    ap.add_argument("--config", choices=["h2", "h3"], default="h2")
    ap.add_argument("--builder", choices=["plus", "plus_gap"], default="plus_gap")
    args = ap.parse_args()
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    audit_data(train, test, sample)
    y = train[TARGET].astype(int)
    builder = build_plus if args.builder == "plus" else build_plus_gap
    t0 = time.time()
    oof, te, oof_by_seed, test_by_seed, folds, auc = run(
        train, test, y, tuple(args.seeds), args.folds, PARAMS[args.config], builder
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        oof=oof,
        test=te,
        y=y.to_numpy(),
        **{f"oof_{s}": oof_by_seed[s] for s in args.seeds},
    )
    build_submission(test, sample, te, args.output_dir / "submission.csv")
    # fuse vs B6
    b6 = np.load("artifacts/b6_gapbag_8seed/predictions.npz")
    eq = 0.5 * (b6["oof_gap"] + b6["oof_gap_bag"])
    metrics = {
        "builder": args.builder,
        "config": args.config,
        "oof_auc": auc,
        "seed_aucs": {str(s): float(roc_auc_score(y, oof_by_seed[s])) for s in args.seeds},
        "corr_b6": float(np.corrcoef(oof, eq)[0, 1]),
        "max_with_b6": float(roc_auc_score(y, np.maximum(eq, oof))),
        "mean_with_b6": float(roc_auc_score(y, 0.5 * (eq + oof))),
        "ref_plus_v10": 0.6886170674774439,
        "ref_max_b6_v10plus": 0.7022093156561012,
        "elapsed_sec": round(time.time() - t0, 1),
        "folds": folds,
        "seeds": list(args.seeds),
        "n_splits": args.folds,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ("oof_auc", "corr_b6", "max_with_b6", "mean_with_b6", "elapsed_sec")}, indent=2))


if __name__ == "__main__":
    main()
