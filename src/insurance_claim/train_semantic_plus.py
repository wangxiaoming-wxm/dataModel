"""CatBoost + extreme semantic feature engineering on NEW competition data.

Protocol (honest local OOF):
- Fold-local feature blocks only
- No target encoding
- No OOF weight search / no test labels
- CatBoost native categoricals + string semantic crosses
- Equal average across seeds
- Optional shuffled-label sanity check
"""

from __future__ import annotations

import argparse
import json
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

N_SPLITS = 5
SEEDS_DEFAULT = (2026, 2027, 2028, 2029)

# Stage-1 dual cross on raw semantic fields (proven backbone).
DUAL_COLS_RAW = [
    "region",
    "source",
    "version",
    "age_range",
    "month",
    "livability",
    "condition",
    "t3",
]

# Stage-2 dual cross on parsed tokens (car / powertrain / era).
DUAL_COLS_PARSED = [
    "region",
    "car_id",
    "version",
    "age_range",
    "month",
    "t3_sfx",
    "code",
    "ver_era",
]

CAT_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1200,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=10,
    random_strength=0.7,
    bagging_temperature=0.2,
    border_count=128,
    od_type="Iter",
    od_wait=120,
    verbose=False,
    thread_count=-1,
    allow_writing_files=False,
)


def force_high_value_crosses(frame: pd.DataFrame) -> pd.DataFrame:
    """Explicit business-critical crosses (string cats for CatBoost)."""
    out = frame.copy()

    def col(*names: str) -> pd.Series | None:
        for name in names:
            if name in out.columns:
                return out[name].astype(str)
        return None

    region = col("region", "region__category")
    car = col("car_token", "src_car", "car_id")
    version = col("version", "version__category")
    t3_sfx = col("t3_sfx")
    code = col("code", "code__category")
    days_q5 = col("days_q5", "days__bin_5")
    days_q10 = col("days_q10", "days__bin_10")
    ver_era = col("ver_era")
    grades = col("grades_token", "grades", "grades__category")

    def put(name: str, series: pd.Series) -> None:
        out[name] = series.astype(str)

    if region is not None and version is not None:
        put("force__region_x_version", region + "|" + version)
    if region is not None and car is not None:
        put("force__region_x_car", region + "|" + car)
    if car is not None and version is not None:
        put("force__car_x_version", car + "|" + version)
    if t3_sfx is not None and code is not None:
        put("force__t3sfx_x_code", t3_sfx + "|" + code)
    if days_q10 is not None and region is not None and version is not None:
        put("force__d10_region_version", days_q10 + "|" + region + "|" + version)
    if days_q5 is not None and car is not None and t3_sfx is not None:
        put("force__d5_car_t3sfx", days_q5 + "|" + car + "|" + t3_sfx)
    if ver_era is not None and region is not None:
        put("force__verera_x_region", ver_era + "|" + region)
    if code is not None and grades is not None:
        put("force__code_x_grades", code + "|" + grades)
    if car is not None and code is not None:
        put("force__car_x_code", car + "|" + code)
    return out


def build_features(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    *,
    drop_latent_raw: bool = True,
    with_physics: bool = True,
    with_parsed_dual: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    parts_tr, parts_va, parts_te = [], [], []

    stage1 = [
        RawFeatureBlock(drop_near_id_latent=drop_latent_raw),
        StructuredStringFeatureBlock(),
        DaysConditionFeatureBlock(
            quantile_bins=(5, 10, 20),
            categorical_cross_columns=("region", "source", "version", "code"),
            categorical_cross_bins=(5, 10),
        ),
        DualCategoryFeatureBlock(
            columns=DUAL_COLS_RAW,
            max_categories=64,
            cross_order=3,
            max_cross_columns=6,
        ),
    ]
    for block in stage1:
        parts_tr.append(block.fit_transform(X_tr))
        parts_va.append(block.transform(X_va))
        parts_te.append(block.transform(X_te))

    parse = DomainParseFeatureBlock()
    ptr = parse.fit_transform(X_tr)
    pva = parse.transform(X_va)
    pte = parse.transform(X_te)
    parts_tr.append(ptr)
    parts_va.append(pva)
    parts_te.append(pte)

    def aug(base: pd.DataFrame, parsed: pd.DataFrame) -> pd.DataFrame:
        return (
            pd.concat([base.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
            .loc[:, lambda d: ~d.columns.duplicated()]
        )

    tr_aug, va_aug, te_aug = aug(X_tr, ptr), aug(X_va, pva), aug(X_te, pte)

    stage2: list[Any] = [
        DaysConditionCrossFeatureBlock(
            days_bins=5,
            condition_bins=5,
            days_bins_hi=10,
            with_region=True,
            with_source_car=True,
            with_version=True,
            with_t3_sfx=True,
            with_code=True,
        )
    ]
    if with_parsed_dual:
        stage2.append(
            DualCategoryFeatureBlock(
                columns=DUAL_COLS_PARSED,
                max_categories=64,
                cross_order=3,
                max_cross_columns=6,
            )
        )
    if with_physics:
        stage2.append(NumericPhysicsFeatureBlock())

    for block in stage2:
        parts_tr.append(block.fit_transform(tr_aug))
        parts_va.append(block.transform(va_aug))
        parts_te.append(block.transform(te_aug))

    tr = force_high_value_crosses(
        pd.concat(parts_tr, axis=1).loc[:, lambda d: ~d.columns.duplicated()]
    )
    va = force_high_value_crosses(
        pd.concat(parts_va, axis=1).loc[:, lambda d: ~d.columns.duplicated()]
    ).reindex(columns=tr.columns)
    te = force_high_value_crosses(
        pd.concat(parts_te, axis=1).loc[:, lambda d: ~d.columns.duplicated()]
    ).reindex(columns=tr.columns)

    return prepare_for_cat(tr, va, te)


def prepare_for_cat(
    tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    def is_cat(col: str, series: pd.Series) -> bool:
        if not pd.api.types.is_numeric_dtype(series):
            return True
        name = str(col)
        return (
            name.endswith(
                (
                    "__category",
                    "__category_cross",
                    "__prefix",
                    "__suffix",
                    "__pattern",
                    "__bin",
                    "__alpha",
                )
            )
            or "__bin_" in name
            or name.endswith(("_bin", "__bin"))
            or "days_condition__bin" in name
            or name.startswith("force__")
            or name
            in {
                "t3_sfx",
                "t3_bin",
                "t3_key",
                "car_id",
                "eng_id",
                "car_token",
                "ver_era",
                "grades_token",
                "car_code_key",
                "t3sfx_code_key",
                "ver_era_region_key",
                "car_ver_key",
                "code_grades_key",
                "t3sfx_car_key",
                "days_q5",
                "days_q10",
                "cond_q5",
                "days_win",
                "src_car",
                "V_bin",
                "car_prefix",
                "eng_prefix",
            }
        )

    cat_names = [column for column in tr.columns if is_cat(column, tr[column])]
    tr, va, te = tr.copy(), va.copy(), te.copy()
    for column in cat_names:
        tr[column] = tr[column].astype(str).fillna("__MISSING__")
        va[column] = va[column].astype(str).fillna("__MISSING__")
        te[column] = te[column].astype(str).fillna("__MISSING__")
    for column in tr.columns:
        if column in cat_names:
            continue
        tr[column] = pd.to_numeric(tr[column], errors="coerce")
        median = float(tr[column].median()) if tr[column].notna().any() else 0.0
        tr[column] = tr[column].fillna(median)
        va[column] = pd.to_numeric(va[column], errors="coerce").fillna(median)
        te[column] = pd.to_numeric(te[column], errors="coerce").fillna(median)
    return tr, va, te, cat_names


def run_seeds(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seeds: tuple[int, ...],
    y_override: np.ndarray | None = None,
    cat_params: dict[str, Any] | None = None,
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
    params_base = dict(CAT_PARAMS)
    if cat_params:
        params_base.update(cat_params)

    for seed in seeds:
        folds = list(
            StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(features, y)
        )
        oof = np.zeros(len(train), dtype=float)
        pred_test = np.zeros(len(test), dtype=float)
        for fold, (train_idx, valid_idx) in enumerate(folds):
            X_tr = features.iloc[train_idx].reset_index(drop=True)
            y_tr = y.iloc[train_idx].reset_index(drop=True)
            X_va = features.iloc[valid_idx].reset_index(drop=True)
            y_va = y.iloc[valid_idx].reset_index(drop=True)
            tr_fe, va_fe, te_fe, cat_names = build_features(X_tr, X_va, test.copy())
            params = dict(params_base)
            params["random_seed"] = seed + fold
            model = CatBoostClassifier(**params)
            model.fit(
                tr_fe,
                y_tr,
                eval_set=(va_fe, y_va),
                cat_features=cat_names,
                use_best_model=True,
                verbose=False,
            )
            valid_pred = model.predict_proba(va_fe)[:, 1]
            test_pred = model.predict_proba(te_fe)[:, 1]
            oof[valid_idx] = valid_pred
            pred_test += test_pred / N_SPLITS
            best_iteration = model.get_best_iteration()
            valid_auc = float(roc_auc_score(y_va, valid_pred))
            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "valid_auc": valid_auc,
                    "best_iter": int(best_iteration if best_iteration is not None else -1),
                    "n_features": int(tr_fe.shape[1]),
                    "n_cats": len(cat_names),
                }
            )
            print(
                f"seed={seed} fold={fold} auc={valid_auc:.5f} "
                f"best={best_iteration} n_feat={tr_fe.shape[1]} n_cat={len(cat_names)}",
                flush=True,
            )
        seed_auc = float(roc_auc_score(y, oof))
        oof_by_seed[seed] = oof
        test_by_seed[seed] = pred_test
        print(f"seed={seed} OOF={seed_auc:.5f}", flush=True)

    oof_pool = np.mean(np.vstack([oof_by_seed[seed] for seed in seeds]), axis=0)
    test_pool = np.mean(np.vstack([test_by_seed[seed] for seed in seeds]), axis=0)
    seed_aucs = {str(seed): float(roc_auc_score(y, oof_by_seed[seed])) for seed in seeds}
    metrics = {
        "recipe": "catboost_semantic_plus_newdata",
        "data_note": "trained on current workspace train/test only; prior data obsolete",
        "seeds": list(seeds),
        "pooled_oof_auc": float(roc_auc_score(y, oof_pool)),
        "seed_aucs": seed_aucs,
        "seed_mean": float(np.mean(list(seed_aucs.values()))),
        "seed_std": float(np.std(list(seed_aucs.values()))),
        "pred_mean": float(test_pool.mean()),
        "elapsed_sec": round(time.time() - started, 1),
        "folds": fold_rows,
        "cat_params": {k: v for k, v in params_base.items() if k != "verbose"},
        "policy": (
            "CatBoost semantic+domain crosses; fold-local FE; drop raw x0-x18; "
            "no TE; equal seed average; no OOF weight search"
        ),
        "gate_0_698": None,
    }
    metrics["gate_0_698"] = bool(metrics["pooled_oof_auc"] >= 0.698)
    print(
        f"POOLED OOF={metrics['pooled_oof_auc']:.6f} "
        f"seed_mean={metrics['seed_mean']:.6f}±{metrics['seed_std']:.6f} "
        f"gate_0.698={'PASS' if metrics['gate_0_698'] else 'FAIL'}",
        flush=True,
    )
    return {"metrics": metrics, "oof": oof_pool, "test": test_pool, "y": y.to_numpy()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/cat_semantic_plus")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    parser.add_argument("--shuffled", action="store_true")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "submit_sample.csv")
    audit = audit_data(train, test, sample)

    overrides = {}
    if args.iterations is not None:
        overrides["iterations"] = args.iterations
    if args.depth is not None:
        overrides["depth"] = args.depth
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate

    result = run_seeds(train, test, tuple(args.seeds), cat_params=overrides or None)
    metrics = result["metrics"]
    metrics["audit"] = {
        "train_rows": audit["train_rows"],
        "test_rows": audit["test_rows"],
        "target_rate": audit["target_rate"],
        "id_overlap": audit["id_overlap"],
    }

    if args.shuffled:
        shuffled = train[TARGET].to_numpy().copy()
        np.random.default_rng(2026).shuffle(shuffled)
        shuffled_result = run_seeds(
            train, test, (args.seeds[0],), y_override=shuffled, cat_params=overrides or None
        )
        metrics["shuffled_oof_auc"] = shuffled_result["metrics"]["pooled_oof_auc"]
        metrics["shuffled_pass"] = bool(0.47 <= metrics["shuffled_oof_auc"] <= 0.53)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        oof=result["oof"],
        test=result["test"],
        y=result["y"],
    )
    build_submission(
        test, sample, result["test"], args.output_dir / "submission_cat_semantic_plus.csv"
    )
    final_dir = Path("submissions")
    final_dir.mkdir(parents=True, exist_ok=True)
    build_submission(
        test, sample, result["test"], final_dir / "submission_catboost_semantic_plus.csv"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pooled_oof_auc": metrics["pooled_oof_auc"],
                "seed_aucs": metrics["seed_aucs"],
                "gate_0_698": metrics["gate_0_698"],
                "shuffled_oof_auc": metrics.get("shuffled_oof_auc"),
                "shuffled_pass": metrics.get("shuffled_pass"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
