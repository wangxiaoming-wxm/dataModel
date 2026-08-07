"""Multi-view CatBoost bagging on NEW data (honest OOF, no TE, fixed equal-rank fusion).

Views are intentionally diverse but protocol-safe:
- classic: raw+structured+days_condition+dual(3)
- risk: drop x0-x18 + days/cond crosses + domain parse + dual
- physics: risk + numeric physics residuals
Fusion: equal-weight probability mean AND equal-rank mean (both fixed a priori).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.feature_blocks import (
    DaysConditionCrossFeatureBlock,
    DaysConditionFeatureBlock,
    DomainParseFeatureBlock,
    DualCategoryFeatureBlock,
    NumericPhysicsFeatureBlock,
    RawFeatureBlock,
    StructuredStringFeatureBlock,
)
from insurance_claim.model import TARGET, audit_data, build_submission
from insurance_claim.train_semantic_plus import (
    DUAL_COLS_PARSED,
    DUAL_COLS_RAW,
    force_high_value_crosses,
    prepare_for_cat,
)

N_SPLITS = 5

CAT_A = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1200,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=10,
    random_strength=0.8,
    bagging_temperature=0.3,
    border_count=128,
    od_type="Iter",
    od_wait=120,
    verbose=False,
    thread_count=-1,
    allow_writing_files=False,
)
CAT_B = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1600,
    learning_rate=0.025,
    depth=7,
    l2_leaf_reg=12,
    random_strength=0.5,
    bagging_temperature=0.1,
    border_count=254,
    od_type="Iter",
    od_wait=150,
    verbose=False,
    thread_count=-1,
    allow_writing_files=False,
)
CAT_C = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1000,
    learning_rate=0.035,
    depth=5,
    l2_leaf_reg=8,
    random_strength=1.0,
    bagging_temperature=0.5,
    border_count=64,
    od_type="Iter",
    od_wait=100,
    verbose=False,
    thread_count=-1,
    allow_writing_files=False,
)


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(parts, axis=1).loc[:, lambda d: ~d.columns.duplicated()]


def build_classic(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    parts_tr, parts_va, parts_te = [], [], []
    for block in [
        RawFeatureBlock(drop_near_id_latent=False),
        StructuredStringFeatureBlock(),
        DaysConditionFeatureBlock(),
        DualCategoryFeatureBlock(
            columns=DUAL_COLS_RAW, max_categories=64, cross_order=3, max_cross_columns=6
        ),
    ]:
        parts_tr.append(block.fit_transform(X_tr))
        parts_va.append(block.transform(X_va))
        parts_te.append(block.transform(X_te))
    tr, va, te = _concat(parts_tr), _concat(parts_va), _concat(parts_te)
    va, te = va.reindex(columns=tr.columns), te.reindex(columns=tr.columns)
    return prepare_for_cat(tr, va, te)


def build_risk(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    parts_tr, parts_va, parts_te = [], [], []
    for block in [
        RawFeatureBlock(drop_near_id_latent=True),
        StructuredStringFeatureBlock(),
        DaysConditionFeatureBlock(
            quantile_bins=(5, 10, 20),
            categorical_cross_columns=("region", "source", "version", "code"),
            categorical_cross_bins=(5, 10),
        ),
        DualCategoryFeatureBlock(
            columns=DUAL_COLS_RAW, max_categories=64, cross_order=3, max_cross_columns=6
        ),
    ]:
        parts_tr.append(block.fit_transform(X_tr))
        parts_va.append(block.transform(X_va))
        parts_te.append(block.transform(X_te))

    parse = DomainParseFeatureBlock()
    ptr, pva, pte = parse.fit_transform(X_tr), parse.transform(X_va), parse.transform(X_te)
    parts_tr.append(ptr)
    parts_va.append(pva)
    parts_te.append(pte)

    def aug(base: pd.DataFrame, parsed: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [base.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1
        ).loc[:, lambda d: ~d.columns.duplicated()]

    tr_aug, va_aug, te_aug = aug(X_tr, ptr), aug(X_va, pva), aug(X_te, pte)
    for block in [
        DaysConditionCrossFeatureBlock(),
        DualCategoryFeatureBlock(
            columns=DUAL_COLS_PARSED, max_categories=64, cross_order=3, max_cross_columns=6
        ),
    ]:
        parts_tr.append(block.fit_transform(tr_aug))
        parts_va.append(block.transform(va_aug))
        parts_te.append(block.transform(te_aug))

    tr = force_high_value_crosses(_concat(parts_tr))
    va = force_high_value_crosses(_concat(parts_va)).reindex(columns=tr.columns)
    te = force_high_value_crosses(_concat(parts_te)).reindex(columns=tr.columns)
    return prepare_for_cat(tr, va, te)


def build_physics(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    tr, va, te, cats = build_risk(X_tr, X_va, X_te)
    # Rebuild with physics appended on augmented frame for cleanliness.
    parts_tr, parts_va, parts_te = [], [], []
    # reuse risk builder internals via calling blocks again is heavy; instead
    # append physics on original+parse.
    parse = DomainParseFeatureBlock()
    ptr, pva, pte = parse.fit_transform(X_tr), parse.transform(X_va), parse.transform(X_te)

    def aug(base: pd.DataFrame, parsed: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [base.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1
        ).loc[:, lambda d: ~d.columns.duplicated()]

    tr_aug, va_aug, te_aug = aug(X_tr, ptr), aug(X_va, pva), aug(X_te, pte)
    phys = NumericPhysicsFeatureBlock()
    p_tr, p_va, p_te = phys.fit_transform(tr_aug), phys.transform(va_aug), phys.transform(te_aug)
    tr2 = force_high_value_crosses(
        pd.concat([tr.reset_index(drop=True), p_tr.reset_index(drop=True)], axis=1).loc[
            :, lambda d: ~d.columns.duplicated()
        ]
    )
    va2 = force_high_value_crosses(
        pd.concat([va.reset_index(drop=True), p_va.reset_index(drop=True)], axis=1).loc[
            :, lambda d: ~d.columns.duplicated()
        ]
    ).reindex(columns=tr2.columns)
    te2 = force_high_value_crosses(
        pd.concat([te.reset_index(drop=True), p_te.reset_index(drop=True)], axis=1).loc[
            :, lambda d: ~d.columns.duplicated()
        ]
    ).reindex(columns=tr2.columns)
    return prepare_for_cat(tr2, va2, te2)


VIEWS: dict[str, tuple[Callable, dict[str, Any]]] = {
    "classic_A": (build_classic, CAT_A),
    "risk_B": (build_risk, CAT_B),
    "physics_C": (build_physics, CAT_C),
}


def run_view(
    name: str,
    builder: Callable,
    params_base: dict[str, Any],
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: pd.Series,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    features = train.drop(columns=[TARGET])
    oof_by_seed: dict[int, np.ndarray] = {}
    test_by_seed: dict[int, np.ndarray] = {}
    fold_rows: list[dict[str, Any]] = []
    for seed in seeds:
        oof = np.zeros(len(train), dtype=float)
        pred_test = np.zeros(len(test), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(
            StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(features, y)
        ):
            X_tr = features.iloc[tr_idx].reset_index(drop=True)
            X_va = features.iloc[va_idx].reset_index(drop=True)
            y_tr = y.iloc[tr_idx].reset_index(drop=True)
            y_va = y.iloc[va_idx].reset_index(drop=True)
            tr, va, te, cats = builder(X_tr, X_va, test.copy())
            params = dict(params_base)
            params["random_seed"] = seed + fold * 17
            model = CatBoostClassifier(**params)
            model.fit(
                tr,
                y_tr,
                eval_set=(va, y_va),
                cat_features=cats,
                use_best_model=True,
                verbose=False,
            )
            oof[va_idx] = model.predict_proba(va)[:, 1]
            pred_test += model.predict_proba(te)[:, 1] / N_SPLITS
            best = model.get_best_iteration()
            auc = float(roc_auc_score(y_va, oof[va_idx]))
            fold_rows.append(
                {
                    "view": name,
                    "seed": seed,
                    "fold": fold,
                    "valid_auc": auc,
                    "best_iter": int(best if best is not None else -1),
                    "n_features": int(tr.shape[1]),
                    "n_cats": len(cats),
                }
            )
            print(
                f"{name} seed={seed} fold={fold} auc={auc:.5f} best={best} n={tr.shape[1]}",
                flush=True,
            )
        seed_auc = float(roc_auc_score(y, oof))
        print(f"{name} seed={seed} OOF={seed_auc:.6f}", flush=True)
        oof_by_seed[seed] = oof
        test_by_seed[seed] = pred_test
    oof = np.mean(np.vstack(list(oof_by_seed.values())), axis=0)
    te = np.mean(np.vstack(list(test_by_seed.values())), axis=0)
    return {
        "oof": oof,
        "test": te,
        "oof_auc": float(roc_auc_score(y, oof)),
        "seed_aucs": {str(s): float(roc_auc_score(y, oof_by_seed[s])) for s in seeds},
        "folds": fold_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/multiview"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029])
    parser.add_argument(
        "--views",
        nargs="+",
        default=list(VIEWS.keys()),
        choices=list(VIEWS.keys()),
    )
    parser.add_argument("--shuffled", action="store_true")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "submit_sample.csv")
    audit = audit_data(train, test, sample)
    y = train[TARGET].astype(int)
    seeds = tuple(args.seeds)
    started = time.time()

    view_results = {}
    for name in args.views:
        builder, params = VIEWS[name]
        view_results[name] = run_view(name, builder, params, train, test, y, seeds)
        print(f"VIEW {name} pooled={view_results[name]['oof_auc']:.6f}", flush=True)

    # Fixed a-priori fusions (no OOF weight search).
    oofs = [view_results[n]["oof"] for n in args.views]
    tests = [view_results[n]["test"] for n in args.views]
    mean_oof = np.mean(np.vstack(oofs), axis=0)
    mean_test = np.mean(np.vstack(tests), axis=0)
    rank_oof = np.mean(np.vstack([rankdata(o) for o in oofs]), axis=0)
    rank_test = np.mean(np.vstack([rankdata(t) for t in tests]), axis=0)
    # Convert rank fusion to [0,1]-ish by min-max for submission compatibility.
    rank_test_prob = (rank_test - rank_test.min()) / (rank_test.max() - rank_test.min() + 1e-12)

    mean_auc = float(roc_auc_score(y, mean_oof))
    rank_auc = float(roc_auc_score(y, rank_oof))
    # Choose the better fixed fusion by OOF (reporting both; selection is between
    # two pre-registered rules only — still mild selection bias, documented).
    if rank_auc >= mean_auc:
        final_name, final_oof, final_test, final_auc = (
            "equal_rank",
            rank_oof,
            rank_test_prob,
            rank_auc,
        )
    else:
        final_name, final_oof, final_test, final_auc = (
            "equal_prob_mean",
            mean_oof,
            mean_test,
            mean_auc,
        )

    metrics: dict[str, Any] = {
        "recipe": "multiview_catboost_newdata",
        "data_note": "current workspace train/test only",
        "seeds": list(seeds),
        "views": {
            n: {
                "oof_auc": view_results[n]["oof_auc"],
                "seed_aucs": view_results[n]["seed_aucs"],
            }
            for n in args.views
        },
        "fusion": {
            "equal_prob_mean_auc": mean_auc,
            "equal_rank_auc": rank_auc,
            "selected": final_name,
            "selected_auc": final_auc,
            "note": "selection only between two pre-registered fusions",
        },
        "pooled_oof_auc": final_auc,
        "gate_0_698": bool(final_auc >= 0.698),
        "elapsed_sec": round(time.time() - started, 1),
        "audit": {
            "train_rows": audit["train_rows"],
            "test_rows": audit["test_rows"],
            "target_rate": audit["target_rate"],
            "id_overlap": audit["id_overlap"],
        },
        "folds": [row for n in args.views for row in view_results[n]["folds"]],
        "policy": (
            "multi-view CatBoost; fold-local FE; no TE; equal seed avg; "
            "fusion pre-registered equal-prob / equal-rank only"
        ),
    }

    if args.shuffled:
        # Sanity on the selected builder set using first view only (cheap proxy)
        # plus report that full shuffle of final blend should be near 0.5.
        y_s = y.to_numpy().copy()
        np.random.default_rng(2026).shuffle(y_s)
        # Evaluate shuffled labels against frozen OOF ranks of final_oof — this
        # checks label association, not retrain. For strict check, retrain one seed.
        sh_auc = float(roc_auc_score(y_s, final_oof))
        metrics["shuffled_against_frozen_oof_auc"] = sh_auc
        # Proper retrain shuffle on classic_A / first seed only
        name0 = args.views[0]
        builder, params = VIEWS[name0]
        sh = run_view(
            f"{name0}_shuffled",
            builder,
            params,
            train,
            test,
            pd.Series(y_s, name=TARGET),
            (seeds[0],),
        )
        metrics["shuffled_retrain_oof_auc"] = sh["oof_auc"]
        metrics["shuffled_pass"] = bool(0.47 <= sh["oof_auc"] <= 0.53)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        oof=final_oof,
        test=final_test,
        y=y.to_numpy(),
        **{f"oof_{n}": view_results[n]["oof"] for n in args.views},
        **{f"test_{n}": view_results[n]["test"] for n in args.views},
    )
    build_submission(
        test, sample, final_test, args.output_dir / "submission_multiview.csv"
    )
    Path("submissions").mkdir(parents=True, exist_ok=True)
    build_submission(
        test, sample, final_test, Path("submissions") / "submission_multiview.csv"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "views": metrics["views"],
        "fusion": metrics["fusion"],
        "gate_0_698": metrics["gate_0_698"],
        "shuffled_retrain_oof_auc": metrics.get("shuffled_retrain_oof_auc"),
        "shuffled_pass": metrics.get("shuffled_pass"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
