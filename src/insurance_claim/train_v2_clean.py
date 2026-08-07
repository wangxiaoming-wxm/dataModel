"""Improved CatBoost recipe: cleaner dual cols + optional nested low-card TE.

Fixes:
- Remove near-unique ``condition`` from dual-category crosses
- Remove ``livability`` from dual (≈ region R²≈0.99)
- Prefer region/source/version/age/t3/code/month for semantic triples
- Optional nested (inner-fold) TE only for low-card keys

Protocol remains honest: outer-fold local FE; TE nested inside train fold;
no OOF weight search; equal seed average.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
from insurance_claim.train_lean_business import add_business_crosses, fit_edges
from insurance_claim.train_semantic_plus import force_high_value_crosses, prepare_for_cat

N_SPLITS = 5
SEEDS_DEFAULT = (2026, 2027, 2028, 2029)

# Clean dual columns (no near-unique floats, no region-collinear livability).
DUAL_CLEAN = ["region", "source", "version", "age_range", "t3", "code", "month", "grades"]
DUAL_PARSED = ["region", "car_id", "version", "age_range", "t3_sfx", "code", "ver_era", "month"]

CAT_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1400,
    learning_rate=0.028,
    depth=6,
    l2_leaf_reg=12,
    random_strength=0.6,
    bagging_temperature=0.2,
    border_count=128,
    od_type="Iter",
    od_wait=140,
    verbose=False,
    thread_count=-1,
    allow_writing_files=False,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="/workspace").decode().strip()
    except Exception:
        return "unknown"


def nested_target_encode(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_train: pd.Series,
    keys: list[str],
    n_inner: int = 5,
    seed: int = 0,
    prior: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict nested TE: for train rows use inner-fold OOF TE; valid/test use full train map."""
    global_mean = float(y_train.mean())
    tr_te = pd.DataFrame(index=train_frame.index)
    va_te = pd.DataFrame(index=valid_frame.index)
    te_te = pd.DataFrame(index=test_frame.index)

    for key in keys:
        if key not in train_frame.columns:
            continue
        col_name = f"nte__{key}"
        oof = np.full(len(train_frame), np.nan, dtype=float)
        inner = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
        values = train_frame[key].astype(str).fillna("__MISSING__")
        for inner_tr, inner_va in inner.split(train_frame, y_train):
            tmp = pd.DataFrame({"k": values.iloc[inner_tr], "y": y_train.iloc[inner_tr]})
            stats = tmp.groupby("k")["y"].agg(["sum", "count"])
            mapping = (stats["sum"] + prior * global_mean) / (stats["count"] + prior)
            oof[inner_va] = values.iloc[inner_va].map(mapping).fillna(global_mean).to_numpy(dtype=float)
        tr_te[col_name] = oof
        full = pd.DataFrame({"k": values, "y": y_train})
        stats = full.groupby("k")["y"].agg(["sum", "count"])
        mapping = (stats["sum"] + prior * global_mean) / (stats["count"] + prior)
        va_te[col_name] = (
            valid_frame[key].astype(str).fillna("__MISSING__").map(mapping).fillna(global_mean).astype(float)
        )
        te_te[col_name] = (
            test_frame[key].astype(str).fillna("__MISSING__").map(mapping).fillna(global_mean).astype(float)
        )
    return tr_te, va_te, te_te


def build_v2(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: pd.Series | None = None,
    with_nested_te: bool = False,
    te_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    parts_tr, parts_va, parts_te = [], [], []
    for block in [
        RawFeatureBlock(drop_near_id_latent=True),
        StructuredStringFeatureBlock(columns=["t3", "source", "version", "month", "grades", "code"]),
        DaysConditionFeatureBlock(
            quantile_bins=(5, 10, 20),
            categorical_cross_columns=("region", "source", "version", "code"),
            categorical_cross_bins=(5, 10),
        ),
        DualCategoryFeatureBlock(
            columns=DUAL_CLEAN, max_categories=64, cross_order=3, max_cross_columns=6
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
        DaysConditionCrossFeatureBlock(with_t3_sfx=True, with_code=True),
        DualCategoryFeatureBlock(
            columns=DUAL_PARSED, max_categories=64, cross_order=3, max_cross_columns=6
        ),
        NumericPhysicsFeatureBlock(),
    ]:
        parts_tr.append(block.fit_transform(tr_aug))
        parts_va.append(block.transform(va_aug))
        parts_te.append(block.transform(te_aug))

    tr = force_high_value_crosses(pd.concat(parts_tr, axis=1).loc[:, lambda d: ~d.columns.duplicated()])
    va = force_high_value_crosses(pd.concat(parts_va, axis=1).loc[:, lambda d: ~d.columns.duplicated()]).reindex(
        columns=tr.columns
    )
    te = force_high_value_crosses(pd.concat(parts_te, axis=1).loc[:, lambda d: ~d.columns.duplicated()]).reindex(
        columns=tr.columns
    )

    # Business high-support crosses from raw fold edges.
    edges = fit_edges(X_tr)

    def with_raw(fe: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
        keep = [
            c
            for c in ["days", "condition", "region", "source", "code", "w1", "w2", "age_range", "version"]
            if c in raw.columns
        ]
        base = pd.concat([fe.reset_index(drop=True), raw[keep].reset_index(drop=True)], axis=1)
        return base.loc[:, ~base.columns.duplicated()]

    tr = add_business_crosses(with_raw(tr, X_tr), edges)
    va = add_business_crosses(with_raw(va, X_va), edges)
    te = add_business_crosses(with_raw(te, X_te), edges)
    va = va.reindex(columns=tr.columns)
    te = te.reindex(columns=tr.columns)

    if with_nested_te and y_tr is not None:
        # Low-card keys only (+ stable biz crosses already created).
        te_keys = [
            "region",
            "biz_days5",
            "biz_d5_c5",
            "biz_region_d5",
            "biz_car_d5",
            "source",
            "version",
            "biz_t3sfx_code_d5",
        ]
        # Ensure keys exist on frames used for TE mapping.
        tr_te, va_te, te_te = nested_target_encode(
            tr, va, te, y_tr.reset_index(drop=True), te_keys, n_inner=5, seed=te_seed, prior=20.0
        )
        tr = pd.concat([tr.reset_index(drop=True), tr_te.reset_index(drop=True)], axis=1)
        va = pd.concat([va.reset_index(drop=True), va_te.reset_index(drop=True)], axis=1)
        te = pd.concat([te.reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)
        va = va.reindex(columns=tr.columns)
        te = te.reindex(columns=tr.columns)

    return prepare_for_cat(tr, va, te)


def run_seeds(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seeds: tuple[int, ...],
    with_nested_te: bool = False,
    y_override: np.ndarray | None = None,
) -> dict[str, Any]:
    y = (
        pd.Series(y_override, name=TARGET).astype(int)
        if y_override is not None
        else train[TARGET].astype(int)
    )
    features = train.drop(columns=[TARGET])
    oof_by_seed: dict[int, np.ndarray] = {}
    test_by_seed: dict[int, np.ndarray] = {}
    fold_rows: list[dict[str, Any]] = []
    started = time.time()

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
            tr, va, te, cats = build_v2(
                X_tr, X_va, test.copy(), y_tr=y_tr, with_nested_te=with_nested_te, te_seed=seed + fold
            )
            params = dict(CAT_PARAMS)
            params["random_seed"] = seed + fold * 23
            model = CatBoostClassifier(**params)
            model.fit(
                tr, y_tr, eval_set=(va, y_va), cat_features=cats, use_best_model=True, verbose=False
            )
            oof[va_idx] = model.predict_proba(va)[:, 1]
            pred_test += model.predict_proba(te)[:, 1] / N_SPLITS
            best = model.get_best_iteration()
            auc = float(roc_auc_score(y_va, oof[va_idx]))
            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "valid_auc": auc,
                    "best_iter": int(best if best is not None else -1),
                    "n_features": int(tr.shape[1]),
                    "n_cats": len(cats),
                }
            )
            print(
                f"v2{'+nte' if with_nested_te else ''} seed={seed} fold={fold} "
                f"auc={auc:.5f} best={best} n={tr.shape[1]}",
                flush=True,
            )
        seed_auc = float(roc_auc_score(y, oof))
        print(f"v2 seed={seed} OOF={seed_auc:.6f}", flush=True)
        oof_by_seed[seed] = oof
        test_by_seed[seed] = pred_test

    oof = np.mean(np.vstack([oof_by_seed[s] for s in seeds]), axis=0)
    te = np.mean(np.vstack([test_by_seed[s] for s in seeds]), axis=0)
    seed_aucs = {str(s): float(roc_auc_score(y, oof_by_seed[s])) for s in seeds}
    fold_aucs = [r["valid_auc"] for r in fold_rows]
    metrics = {
        "experiment_id": f"v2_clean_dual{'_nte' if with_nested_te else ''}",
        "recipe": "v2_clean_dual_business_physics",
        "with_nested_te": with_nested_te,
        "git_commit": _git_commit(),
        "seeds": list(seeds),
        "cv_scheme": "StratifiedKFold",
        "n_splits": N_SPLITS,
        "pooled_oof_auc": float(roc_auc_score(y, oof)),
        "seed_aucs": seed_aucs,
        "seed_mean": float(np.mean(list(seed_aucs.values()))),
        "seed_std": float(np.std(list(seed_aucs.values()))),
        "fold_auc_min": float(np.min(fold_aucs)),
        "fold_auc_max": float(np.max(fold_aucs)),
        "fold_auc_range": float(np.max(fold_aucs) - np.min(fold_aucs)),
        "pred_mean": float(te.mean()),
        "elapsed_sec": round(time.time() - started, 1),
        "folds": fold_rows,
        "target_encoding": "nested_inner_fold_low_card" if with_nested_te else "none",
        "fusion": "equal_seed_probability_mean",
        "policy": "clean dual cols; business crosses; fold-local; equal seed avg; no OOF weight search",
    }
    metrics["gate_0_698"] = bool(metrics["pooled_oof_auc"] >= 0.698)
    print(
        f"V2 POOLED={metrics['pooled_oof_auc']:.6f} gate={'PASS' if metrics['gate_0_698'] else 'FAIL'}",
        flush=True,
    )
    return {"metrics": metrics, "oof": oof, "test": te, "y": y.to_numpy()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v2_clean"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    parser.add_argument("--nested-te", action="store_true")
    parser.add_argument("--shuffled", action="store_true")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "submit_sample.csv")
    audit = audit_data(train, test, sample)
    result = run_seeds(train, test, tuple(args.seeds), with_nested_te=args.nested_te)
    metrics = result["metrics"]
    metrics["data_sha256"] = {
        "train": _sha256(args.data_dir / "train.csv"),
        "test": _sha256(args.data_dir / "test.csv"),
        "submit": _sha256(args.data_dir / "submit_sample.csv"),
    }
    metrics["audit"] = {
        "train_rows": audit["train_rows"],
        "test_rows": audit["test_rows"],
        "target_rate": audit["target_rate"],
        "id_overlap": audit["id_overlap"],
    }
    metrics["protocol_declaration"] = {
        "no_test_labels": True,
        "fold_local_fe": True,
        "nested_te_only": bool(args.nested_te),
        "no_oof_weight_search": True,
        "equal_seed_average": True,
        "new_data_only": True,
    }

    if args.shuffled:
        shuffled = train[TARGET].to_numpy().copy()
        np.random.default_rng(2026).shuffle(shuffled)
        sh = run_seeds(
            train, test, (args.seeds[0],), with_nested_te=args.nested_te, y_override=shuffled
        )
        metrics["shuffled_oof_auc"] = sh["metrics"]["pooled_oof_auc"]
        metrics["shuffled_pass"] = bool(0.47 <= metrics["shuffled_oof_auc"] <= 0.53)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz", oof=result["oof"], test=result["test"], y=result["y"]
    )
    build_submission(test, sample, result["test"], args.output_dir / "submission_v2.csv")
    Path("submissions").mkdir(exist_ok=True)
    build_submission(test, sample, result["test"], Path("submissions") / "submission_v2_clean.csv")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "pooled_oof_auc": metrics["pooled_oof_auc"],
        "seed_aucs": metrics["seed_aucs"],
        "gate_0_698": metrics["gate_0_698"],
        "shuffled_oof_auc": metrics.get("shuffled_oof_auc"),
        "shuffled_pass": metrics.get("shuffled_pass"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
