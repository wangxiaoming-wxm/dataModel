"""Lean business-priority CatBoost on NEW data.

Focus on high-support semantic crosses recommended by product+EDA:
region×days5, days5×cond5, car×days5, t3sfx×code×days5, w_pair×days5,
age_coarse×days5 — avoid sparse region×version×days10 TE-like cells.

Protocol: fold-local FE, no TE, equal multi-seed average.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    DaysConditionFeatureBlock,
    DomainParseFeatureBlock,
    DualCategoryFeatureBlock,
    RawFeatureBlock,
    StructuredStringFeatureBlock,
)
from insurance_claim.model import TARGET, audit_data, build_submission
from insurance_claim.train_semantic_plus import prepare_for_cat

N_SPLITS = 5
SEEDS_DEFAULT = (2026, 2027, 2028, 2029, 2030, 2031)
_SOURCE_RE = re.compile(r"^([A-Za-z]+)_?(\d+)\|([A-Za-z]+)_?(\d+)$")

CAT_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1500,
    learning_rate=0.028,
    depth=6,
    l2_leaf_reg=12,
    random_strength=0.6,
    bagging_temperature=0.25,
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
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="/workspace")
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def add_business_crosses(frame: pd.DataFrame, edges: dict[str, np.ndarray]) -> pd.DataFrame:
    out = frame.copy()
    days = pd.to_numeric(out.get("days"), errors="coerce")
    cond = pd.to_numeric(out.get("condition"), errors="coerce")

    def bin_series(values: pd.Series, edge_key: str, prefix: str) -> pd.Series:
        edges_ = edges[edge_key]
        numeric = values.to_numpy(dtype=float)
        codes = np.full(len(numeric), -1, dtype=np.int16)
        valid = np.isfinite(numeric)
        if edges_.size:
            codes[valid] = np.searchsorted(edges_, numeric[valid], side="right").astype(np.int16)
        else:
            codes[valid] = 0
        return pd.Series(codes, index=values.index).astype(str).radd(prefix + "_")

    d5 = bin_series(days, "days5", "d5")
    c5 = bin_series(cond, "cond5", "c5")
    out["biz_days5"] = d5
    out["biz_cond5"] = c5
    out["biz_d5_c5"] = (d5 + "|" + c5).astype(str)

    region = out["region"].astype(str) if "region" in out else pd.Series("__NA__", index=out.index)
    out["biz_region_d5"] = (region + "|" + d5).astype(str)
    out["biz_region_c5"] = (region + "|" + c5).astype(str)

    # car from source or car_token
    if "car_token" in out:
        car = out["car_token"].astype(str)
    elif "source" in out:
        def parse_car(v: object) -> str:
            m = _SOURCE_RE.match(str(v))
            return f"{m.group(1)}_{m.group(2)}" if m else "__NA__"

        car = out["source"].map(parse_car).astype(str)
    else:
        car = pd.Series("__NA__", index=out.index)
    out["biz_car"] = car
    out["biz_car_d5"] = (car + "|" + d5).astype(str)

    t3_sfx = out["t3_sfx"].astype(str) if "t3_sfx" in out else pd.Series("__NONE__", index=out.index)
    code = out["code"].astype(str) if "code" in out else pd.Series("__NA__", index=out.index)
    out["biz_t3sfx_code"] = (t3_sfx + "|" + code).astype(str)
    out["biz_t3sfx_code_d5"] = (t3_sfx + "|" + code + "|" + d5).astype(str)

    w1 = pd.to_numeric(out.get("w1"), errors="coerce").fillna(-1).astype(int)
    w2 = pd.to_numeric(out.get("w2"), errors="coerce").fillna(-1).astype(int)
    w_pair = w1.astype(str) + "_" + w2.astype(str)
    out["biz_w_pair"] = w_pair
    out["biz_w_d5"] = (w_pair + "|" + d5).astype(str)

    age = pd.to_numeric(out.get("age_range"), errors="coerce").fillna(-1)
    age_coarse = age.clip(upper=8).astype(int).astype(str)
    age_coarse = age_coarse.where(age >= 0, "__NA__")
    out["biz_age_coarse"] = age_coarse
    out["biz_age_d5"] = (age_coarse.astype(str) + "|" + d5).astype(str)

    version = out["version"].astype(str) if "version" in out else pd.Series("__NA__", index=out.index)
    out["biz_version_d5"] = (version + "|" + d5).astype(str)
    out["biz_car_region"] = (car + "|" + region).astype(str)
    return out


def fit_edges(X_tr: pd.DataFrame) -> dict[str, np.ndarray]:
    def edges(series: pd.Series, bins: int) -> np.ndarray:
        values = pd.to_numeric(series, errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            return np.array([], dtype=float)
        e = np.unique(finite.quantile(np.linspace(0, 1, bins + 1)).to_numpy(dtype=float))
        return e[1:-1] if len(e) > 1 else np.array([], dtype=float)

    return {
        "days5": edges(X_tr["days"], 5),
        "cond5": edges(X_tr["condition"], 5),
    }


def build_lean(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    parts_tr, parts_va, parts_te = [], [], []
    # Keep only stable raw fields: drop near-id latents.
    dual_cols = [
        "region",
        "source",
        "version",
        "age_range",
        "month",
        "code",
        "t3",
        "grades",
    ]
    for block in [
        RawFeatureBlock(drop_near_id_latent=True),
        StructuredStringFeatureBlock(columns=["t3", "source", "version", "month", "grades"]),
        DaysConditionFeatureBlock(
            quantile_bins=(5, 10),
            categorical_cross_columns=("region", "source", "code"),
            categorical_cross_bins=(5,),
            include_single_axis_crosses=True,
        ),
        DualCategoryFeatureBlock(
            columns=dual_cols, max_categories=64, cross_order=2, max_cross_columns=5
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

    tr = pd.concat(parts_tr, axis=1).loc[:, lambda d: ~d.columns.duplicated()]
    va = pd.concat(parts_va, axis=1).loc[:, lambda d: ~d.columns.duplicated()].reindex(columns=tr.columns)
    te = pd.concat(parts_te, axis=1).loc[:, lambda d: ~d.columns.duplicated()].reindex(columns=tr.columns)

    # Business crosses use fold-local edges from raw train fold.
    edges = fit_edges(X_tr)
    # Need original columns on frames — merge from X_* for biz function.
    def with_raw(fe: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in ["days", "condition", "region", "source", "code", "w1", "w2", "age_range", "version"] if c in raw.columns]
        base = pd.concat([fe.reset_index(drop=True), raw[keep].reset_index(drop=True)], axis=1)
        return base.loc[:, ~base.columns.duplicated()]

    tr = add_business_crosses(with_raw(tr, X_tr), edges)
    va = add_business_crosses(with_raw(va, X_va), edges)
    te = add_business_crosses(with_raw(te, X_te), edges)
    va = va.reindex(columns=tr.columns)
    te = te.reindex(columns=tr.columns)
    return prepare_for_cat(tr, va, te)


def run_seeds(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seeds: tuple[int, ...],
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
            tr, va, te, cats = build_lean(X_tr, X_va, test.copy())
            params = dict(CAT_PARAMS)
            params["random_seed"] = seed + fold * 19
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
                f"lean seed={seed} fold={fold} auc={auc:.5f} best={best} n={tr.shape[1]}",
                flush=True,
            )
        seed_auc = float(roc_auc_score(y, oof))
        print(f"lean seed={seed} OOF={seed_auc:.6f}", flush=True)
        oof_by_seed[seed] = oof
        test_by_seed[seed] = pred_test

    oof = np.mean(np.vstack([oof_by_seed[s] for s in seeds]), axis=0)
    te = np.mean(np.vstack([test_by_seed[s] for s in seeds]), axis=0)
    seed_aucs = {str(s): float(roc_auc_score(y, oof_by_seed[s])) for s in seeds}
    fold_aucs = [r["valid_auc"] for r in fold_rows]
    metrics = {
        "experiment_id": "lean_business_catboost_newdata",
        "recipe": "lean_business_semantic",
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
        "target_encoding": "none",
        "fusion": "equal_seed_probability_mean",
        "policy": "lean business crosses; fold-local; no TE; no OOF weight search",
    }
    metrics["gate_0_698"] = bool(metrics["pooled_oof_auc"] >= 0.698)
    print(
        f"LEAN POOLED={metrics['pooled_oof_auc']:.6f} "
        f"seed_mean={metrics['seed_mean']:.6f}±{metrics['seed_std']:.6f} "
        f"gate={'PASS' if metrics['gate_0_698'] else 'FAIL'}",
        flush=True,
    )
    return {"metrics": metrics, "oof": oof, "test": te, "y": y.to_numpy()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lean_business"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    parser.add_argument("--shuffled", action="store_true")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "submit_sample.csv")
    audit = audit_data(train, test, sample)
    result = run_seeds(train, test, tuple(args.seeds))
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
        "no_global_te": True,
        "fold_local_fe": True,
        "no_oof_weight_search": True,
        "equal_seed_average": True,
        "new_data_only": True,
    }

    if args.shuffled:
        shuffled = train[TARGET].to_numpy().copy()
        np.random.default_rng(2026).shuffle(shuffled)
        sh = run_seeds(train, test, (args.seeds[0],), y_override=shuffled)
        metrics["shuffled_oof_auc"] = sh["metrics"]["pooled_oof_auc"]
        metrics["shuffled_pass"] = bool(0.47 <= metrics["shuffled_oof_auc"] <= 0.53)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz", oof=result["oof"], test=result["test"], y=result["y"]
    )
    build_submission(test, sample, result["test"], args.output_dir / "submission_lean.csv")
    Path("submissions").mkdir(exist_ok=True)
    build_submission(test, sample, result["test"], Path("submissions") / "submission_lean_business.csv")
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
