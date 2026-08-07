"""B5 focus recipe: x19/x20 as cats + days×semantic crosses (best 1-seed ~0.6902).

Enrichment drops near-unique x0..x18, keeps x19/x20 as string categories,
crosses days/condition bins with region/source/x19/x20/age_range, and uses
dual-category triples on the focused set.

Protocol: fold-local FE, no TE, equal multi-seed average, optional shuffled check.
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
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.feature_blocks import (
    DaysConditionFeatureBlock,
    DualCategoryFeatureBlock,
    RawFeatureBlock,
    StructuredStringFeatureBlock,
)
from insurance_claim.model import TARGET, audit_data, build_submission

N_SPLITS = 5
SEEDS_DEFAULT = (2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033)
NOISE_X = [f"x{i}" for i in range(19)]

CAT_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1400,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=10,
    random_strength=0.7,
    od_type="Iter",
    od_wait=150,
    verbose=False,
    thread_count=-1,
    allow_writing_files=False,
)

DUAL_FOCUS = [
    "region",
    "source",
    "x19_cat",
    "x20_cat",
    "age_range",
    "livability",
    "version",
    "month",
]


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


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    drop = [c for c in NOISE_X if c in out.columns]
    out = out.drop(columns=drop, errors="ignore")
    src = out["source"].astype(str)
    out["source_car"] = src.str.extract(r"CAR_(\d+)", expand=False).fillna("__NA__")
    out["source_eng"] = src.str.extract(r"ENG_(\d+)", expand=False).fillna("__NA__")
    t3 = out["t3"].astype(str)
    parsed = t3.str.extract(r"^(-?\d+(?:\.\d+)?)([A-Za-z])$")
    out["t3_value"] = pd.to_numeric(parsed[0], errors="coerce")
    out["t3_kind"] = parsed[1].fillna("__NA__")
    out["x19_cat"] = out["x19"].astype(str)
    out["x20_cat"] = out["x20"].astype(str)
    out["condition_missing"] = out["condition"].isna().astype(int)
    out["grades_n"] = out["grades"].map({"s": 1.0, "ss": 2.0, "sss": 3.0})
    out["month_n"] = pd.to_numeric(out["month"].astype(str).str.removeprefix("M"), errors="coerce")
    out["version_n"] = pd.to_numeric(out["version"].astype(str).str.removeprefix("v"), errors="coerce")
    days = pd.to_numeric(out["days"], errors="coerce")
    cond = pd.to_numeric(out["condition"], errors="coerce")
    out["cond_over_days"] = cond / (days.abs() + 1.0)
    out["days_x_cond"] = days * cond
    out["log_days"] = np.log1p(days.clip(lower=0))
    out["log_cc"] = np.log1p(pd.to_numeric(out["cc"], errors="coerce").clip(lower=0))
    out["log_max_g"] = np.log1p(pd.to_numeric(out["max_g"], errors="coerce").clip(lower=0))
    out["log_V"] = np.log1p(pd.to_numeric(out["V"], errors="coerce").clip(lower=0))
    return out


def prepare_for_cat(
    tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    def is_cat(col: str, series: pd.Series) -> bool:
        if not pd.api.types.is_numeric_dtype(series):
            return True
        name = str(col)
        return (
            name.endswith(("__category", "__category_cross", "__prefix", "__suffix", "__pattern"))
            or "__bin_" in name
            or name.endswith(("_bin", "__bin"))
            or "days_condition__bin" in name
            or name in {"source_car", "source_eng", "t3_kind", "x19_cat", "x20_cat"}
        )

    cats = [c for c in tr.columns if is_cat(c, tr[c])]
    tr, va, te = tr.copy(), va.copy(), te.copy()
    for c in cats:
        tr[c] = tr[c].astype(str).fillna("__MISSING__")
        va[c] = va[c].astype(str).fillna("__MISSING__")
        te[c] = te[c].astype(str).fillna("__MISSING__")
    for c in tr.columns:
        if c in cats:
            continue
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
        med = float(tr[c].median()) if tr[c].notna().any() else 0.0
        tr[c] = tr[c].fillna(med)
        va[c] = pd.to_numeric(va[c], errors="coerce").fillna(med)
        te[c] = pd.to_numeric(te[c], errors="coerce").fillna(med)
    return tr, va, te, cats


def build_b5(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    X_tr, X_va, X_te = enrich(X_tr), enrich(X_va), enrich(X_te)
    parts_tr, parts_va, parts_te = [], [], []
    for block in [
        RawFeatureBlock(drop_near_id_latent=False),  # already dropped in enrich
        StructuredStringFeatureBlock(columns=["source", "t3", "region"]),
        DaysConditionFeatureBlock(
            quantile_bins=(5, 10, 20),
            categorical_cross_columns=("region", "source", "x19_cat", "x20_cat", "age_range"),
            categorical_cross_bins=(10,),
        ),
        DualCategoryFeatureBlock(
            columns=DUAL_FOCUS, max_categories=128, cross_order=3, max_cross_columns=6
        ),
    ]:
        parts_tr.append(block.fit_transform(X_tr))
        parts_va.append(block.transform(X_va))
        parts_te.append(block.transform(X_te))
    tr = pd.concat(parts_tr, axis=1).loc[:, lambda d: ~d.columns.duplicated()]
    va = pd.concat(parts_va, axis=1).loc[:, lambda d: ~d.columns.duplicated()].reindex(columns=tr.columns)
    te = pd.concat(parts_te, axis=1).loc[:, lambda d: ~d.columns.duplicated()].reindex(columns=tr.columns)
    return prepare_for_cat(tr, va, te)


def build_b1(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Secondary diverse view: order-2 dual + denser days crosses."""
    X_tr, X_va, X_te = enrich(X_tr), enrich(X_va), enrich(X_te)
    dual = [
        "region",
        "source",
        "version",
        "age_range",
        "month",
        "livability",
        "x19_cat",
        "x20_cat",
        "t3_kind",
        "code",
    ]
    parts_tr, parts_va, parts_te = [], [], []
    for block in [
        RawFeatureBlock(drop_near_id_latent=False),
        StructuredStringFeatureBlock(columns=["source", "t3", "version", "month", "grades", "code", "region"]),
        DaysConditionFeatureBlock(
            quantile_bins=(5, 10, 20, 40),
            categorical_cross_columns=("region", "source", "x19_cat", "x20_cat"),
            categorical_cross_bins=(10, 20),
        ),
        DualCategoryFeatureBlock(columns=dual, max_categories=128, cross_order=2, max_cross_columns=8),
    ]:
        parts_tr.append(block.fit_transform(X_tr))
        parts_va.append(block.transform(X_va))
        parts_te.append(block.transform(X_te))
    tr = pd.concat(parts_tr, axis=1).loc[:, lambda d: ~d.columns.duplicated()]
    va = pd.concat(parts_va, axis=1).loc[:, lambda d: ~d.columns.duplicated()].reindex(columns=tr.columns)
    te = pd.concat(parts_te, axis=1).loc[:, lambda d: ~d.columns.duplicated()].reindex(columns=tr.columns)
    return prepare_for_cat(tr, va, te)


VIEWS = {
    "b5": (build_b5, CAT_PARAMS),
    "b1": (
        build_b1,
        {**CAT_PARAMS, "iterations": 1200, "learning_rate": 0.03},
    ),
}


def run_view(
    name: str,
    builder,
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
            params["random_seed"] = seed + fold
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
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/b5_focus"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    parser.add_argument("--views", nargs="+", default=["b5", "b1"], choices=list(VIEWS))
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

    oofs = [view_results[n]["oof"] for n in args.views]
    tests = [view_results[n]["test"] for n in args.views]
    mean_oof = np.mean(np.vstack(oofs), axis=0)
    mean_test = np.mean(np.vstack(tests), axis=0)
    rank_oof = np.mean(np.vstack([rankdata(o) for o in oofs]), axis=0)
    rank_test = np.mean(np.vstack([rankdata(t) for t in tests]), axis=0)
    rank_test_prob = (rank_test - rank_test.min()) / (rank_test.max() - rank_test.min() + 1e-12)
    mean_auc = float(roc_auc_score(y, mean_oof))
    rank_auc = float(roc_auc_score(y, rank_oof))
    # Prefer b5 alone if fusion does not improve (pre-registered comparison of 3 rules).
    b5_auc = view_results.get("b5", {"oof_auc": -1})["oof_auc"]
    candidates = {
        "b5_only": (b5_auc, view_results["b5"]["oof"] if "b5" in view_results else mean_oof,
                    view_results["b5"]["test"] if "b5" in view_results else mean_test),
        "equal_prob_mean": (mean_auc, mean_oof, mean_test),
        "equal_rank": (rank_auc, rank_oof, rank_test_prob),
    }
    # If b5 not in views, drop it.
    if "b5" not in args.views:
        candidates.pop("b5_only")
    selected = max(candidates.items(), key=lambda kv: kv[1][0])
    final_name = selected[0]
    final_auc, final_oof, final_test = selected[1]

    fold_aucs = [r["valid_auc"] for n in args.views for r in view_results[n]["folds"]]
    metrics: dict[str, Any] = {
        "experiment_id": "b5_focus_multiseed",
        "recipe": "b5_focus_x19x20_days_cross",
        "git_commit": _git_commit(),
        "data_sha256": {
            "train": _sha256(args.data_dir / "train.csv"),
            "test": _sha256(args.data_dir / "test.csv"),
            "submit": _sha256(args.data_dir / "submit_sample.csv"),
        },
        "seeds": list(seeds),
        "cv_scheme": "StratifiedKFold",
        "n_splits": N_SPLITS,
        "views": {n: {"oof_auc": view_results[n]["oof_auc"], "seed_aucs": view_results[n]["seed_aucs"]} for n in args.views},
        "fusion": {
            "candidates": {k: float(v[0]) for k, v in candidates.items()},
            "selected": final_name,
            "selected_auc": float(final_auc),
            "note": "selection among pre-registered fusion rules only",
        },
        "pooled_oof_auc": float(final_auc),
        "seed_mean": float(np.mean([view_results[n]["oof_auc"] for n in args.views])),
        "fold_auc_min": float(np.min(fold_aucs)),
        "fold_auc_max": float(np.max(fold_aucs)),
        "fold_auc_range": float(np.max(fold_aucs) - np.min(fold_aucs)),
        "gate_0_698": bool(final_auc >= 0.698),
        "elapsed_sec": round(time.time() - started, 1),
        "folds": [row for n in args.views for row in view_results[n]["folds"]],
        "target_encoding": "none",
        "policy": "B5 focus + optional B1; fold-local; no TE; equal seed avg; pre-registered fusion",
        "audit": {
            "train_rows": audit["train_rows"],
            "test_rows": audit["test_rows"],
            "target_rate": audit["target_rate"],
            "id_overlap": audit["id_overlap"],
        },
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "no_oof_weight_search": True,
            "equal_seed_average": True,
            "new_data_only": True,
        },
    }

    if args.shuffled:
        y_s = y.to_numpy().copy()
        np.random.default_rng(2026).shuffle(y_s)
        sh = run_view("b5_shuffled", build_b5, CAT_PARAMS, train, test, pd.Series(y_s, name=TARGET), (seeds[0],))
        metrics["shuffled_oof_auc"] = sh["oof_auc"]
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
    build_submission(test, sample, final_test, args.output_dir / "submission_b5.csv")
    Path("submissions").mkdir(exist_ok=True)
    build_submission(test, sample, final_test, Path("submissions") / "submission_b5_focus.csv")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "views": metrics["views"],
                "fusion": metrics["fusion"],
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
