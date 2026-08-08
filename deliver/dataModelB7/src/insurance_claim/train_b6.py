"""B6: B5 main + gap / biz / fixed arms; pre-registered equal fusion toward honest OOF 0.70.

Protocol (honest local OOF on NEW data only):
- StratifiedKFold >= 5
- >=8 seeds equal-weight for delivery claims
- No global TE; fold-local FE only
- Fusion: pre-registered equal_prob (primary) / equal_rank (reported)
- No OOF weight search / no test pseudo-labels
- Optional shuffled-label sanity check

B5 freeze: reuses build_b5 / CAT_PARAMS from train_b5_focus; does not mutate
submissions/b5_frozen or artifacts/b5_frozen.

Primary pre-registered path (post data-mining):
  equal_prob(b5, gap)  — gap = B5 FE + mining P0/P1 cat crosses
Secondary disclosure arms: biz (lean business FE), fixed (no early stop),
  parse, lossguide (weak-arm filter may drop them before fusion when enabled).
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

from insurance_claim.b6_gap_features import GAP_CAT_COLS, add_gap_cats, fit_gap_edges
from insurance_claim.feature_blocks import DomainParseFeatureBlock
from insurance_claim.model import TARGET, audit_data, build_submission
from insurance_claim.train_b5_focus import CAT_PARAMS, N_SPLITS, build_b5, enrich
from insurance_claim.train_lean_business import build_lean

SEEDS_DEFAULT = tuple(range(2026, 2038))  # 12 seeds
ARMS_ALL = ("b5", "gap", "gap_bag", "bag_hot", "biz", "fixed", "parse", "lossguide")
ARMS_DEFAULT = ("gap", "gap_bag", "bag_hot")
FUSE_DEFAULT = ("gap", "gap_bag", "bag_hot")  # hotter bagging adds residual diversity

# Pre-registered weak-arm drop thresholds vs b5 seed-mean (not continuous weight search).
WEAK_DELTA = {
    "lossguide": -0.008,
    "parse": -0.008,
    "biz": -0.008,
    "gap": -0.010,
    "gap_bag": -0.010,
    "bag_hot": -0.010,
    "fixed": -0.015,
}

THREAD_COUNT = 8  # fixed for determinism (avoid thread_count=-1 nondeterminism)

PARAMS_B5 = {**dict(CAT_PARAMS), "thread_count": THREAD_COUNT}

PARAMS_GAP_BAG = {
    **PARAMS_B5,
    "bagging_temperature": 1.0,
    "random_strength": 1.2,
}

# Hotter Bayesian bagging (1-seed screen: solo 0.69257; mean with gap+gap_bag → 0.69358).
PARAMS_BAG_HOT = {
    **PARAMS_B5,
    "bagging_temperature": 2.0,
    "random_strength": 1.5,
}

PARAMS_LOSSGUIDE = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1400,
    learning_rate=0.03,
    depth=6,
    grow_policy="Lossguide",
    max_leaves=31,
    l2_leaf_reg=10,
    random_strength=0.7,
    od_type="Iter",
    od_wait=150,
    verbose=False,
    thread_count=THREAD_COUNT,
    allow_writing_files=False,
)

# Median B5 best_iter ≈ 408 → fixed 400, no early stopping.
PARAMS_FIXED = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=400,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=10,
    random_strength=0.7,
    verbose=False,
    thread_count=THREAD_COUNT,
    allow_writing_files=False,
)

PARAMS_BIZ = dict(
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
    thread_count=THREAD_COUNT,
    allow_writing_files=False,
)

PARSE_CAT_EXTRA = {
    "t3_sfx",
    "t3_bin",
    "t3_key",
    "car_prefix",
    "car_id",
    "eng_prefix",
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
}


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


def build_gap(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """B5 features + fold-local mining gap categorical crosses."""
    tr_b5, va_b5, te_b5, _ = build_b5(X_tr, X_va, X_te)
    edges = fit_gap_edges(X_tr)

    # Gap cats need raw-ish columns (days/condition/w1/w2/t3/code/...).
    # Apply on enrich(raw) so numeric fields exist, then take only GAP_CAT_COLS.
    def gap_only(raw: pd.DataFrame) -> pd.DataFrame:
        enriched = enrich(raw)
        with_gap = add_gap_cats(enriched, edges)
        return with_gap.loc[:, list(GAP_CAT_COLS)].copy()

    gtr, gva, gte = gap_only(X_tr), gap_only(X_va), gap_only(X_te)

    def merge(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
        out = pd.concat([base.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        return out.loc[:, ~out.columns.duplicated()]

    tr = merge(tr_b5, gtr)
    va = merge(va_b5, gva).reindex(columns=tr.columns)
    te = merge(te_b5, gte).reindex(columns=tr.columns)

    # Force gap cols as cats + re-prepare.
    def is_cat(col: str, series: pd.Series) -> bool:
        if col in GAP_CAT_COLS:
            return True
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


def build_parse(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """B5 features + fold-local DomainParse tokens."""
    tr_b5, va_b5, te_b5, _ = build_b5(X_tr, X_va, X_te)
    parse = DomainParseFeatureBlock()
    ptr = parse.fit_transform(X_tr)
    pva = parse.transform(X_va)
    pte = parse.transform(X_te)

    def merge(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
        out = pd.concat([base.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        return out.loc[:, ~out.columns.duplicated()]

    tr = merge(tr_b5, ptr)
    va = merge(va_b5, pva).reindex(columns=tr.columns)
    te = merge(te_b5, pte).reindex(columns=tr.columns)

    def is_cat(col: str, series: pd.Series) -> bool:
        if col in PARSE_CAT_EXTRA:
            return True
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


def arm_spec(name: str) -> tuple[Any, dict[str, Any], bool]:
    """Return (builder, params, use_best_model)."""
    if name == "b5":
        return build_b5, PARAMS_B5, True
    if name == "gap":
        return build_gap, PARAMS_B5, True
    if name == "gap_bag":
        return build_gap, PARAMS_GAP_BAG, True
    if name == "bag_hot":
        return build_gap, PARAMS_BAG_HOT, True
    if name == "biz":
        return build_lean, PARAMS_BIZ, True
    if name == "lossguide":
        return build_b5, PARAMS_LOSSGUIDE, True
    if name == "fixed":
        return build_b5, PARAMS_FIXED, False
    if name == "parse":
        return build_parse, PARAMS_B5, True
    raise ValueError(f"unknown arm: {name}")


def run_arm(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: pd.Series,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    builder, params_base, use_best = arm_spec(name)
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
            fit_kw: dict[str, Any] = dict(
                cat_features=cats,
                verbose=False,
            )
            if use_best:
                model.fit(tr, y_tr, eval_set=(va, y_va), use_best_model=True, **fit_kw)
                best = model.get_best_iteration()
            else:
                model.fit(tr, y_tr, **fit_kw)
                best = params.get("iterations", -1)
            oof[va_idx] = model.predict_proba(va)[:, 1]
            pred_test += model.predict_proba(te)[:, 1] / N_SPLITS
            auc = float(roc_auc_score(y_va, oof[va_idx]))
            fold_rows.append(
                {
                    "arm": name,
                    "seed": seed,
                    "fold": fold,
                    "valid_auc": auc,
                    "best_iter": int(best if best is not None else -1),
                    "n_features": int(tr.shape[1]),
                    "n_cats": len(cats),
                    "use_best_model": use_best,
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
        "oof_by_seed": oof_by_seed,
        "test_by_seed": test_by_seed,
        "oof_auc": float(roc_auc_score(y, oof)),
        "seed_aucs": {str(s): float(roc_auc_score(y, oof_by_seed[s])) for s in seeds},
        "folds": fold_rows,
    }


def select_arms(
    arm_results: dict[str, dict[str, Any]],
    requested: list[str],
    fuse_arms: list[str] | None,
) -> list[str]:
    """Select fusion arms.

    If fuse_arms is provided, use that pre-registered set (intersected with
    available results). Otherwise apply weak-arm filter vs b5.
    """
    if fuse_arms:
        kept = [n for n in fuse_arms if n in arm_results]
        return kept if kept else list(requested)

    if "b5" not in arm_results:
        return list(requested)
    b5_mean = float(np.mean(list(arm_results["b5"]["seed_aucs"].values())))
    kept = ["b5"]
    for name in requested:
        if name == "b5":
            continue
        if name not in arm_results:
            continue
        arm_mean = float(np.mean(list(arm_results[name]["seed_aucs"].values())))
        delta = arm_mean - b5_mean
        thr = WEAK_DELTA.get(name, -0.008)
        if delta < thr:
            continue
        kept.append(name)
    if len(kept) < 2:
        others = [n for n in requested if n != "b5" and n in arm_results]
        if others:
            best = max(others, key=lambda n: arm_results[n]["oof_auc"])
            kept = ["b5", best]
    return kept


def fuse_equal_prob(oofs: list[np.ndarray], tests: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(np.vstack(oofs), axis=0), np.mean(np.vstack(tests), axis=0)


def fuse_equal_rank(oofs: list[np.ndarray], tests: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rank_oof = np.mean(np.vstack([rankdata(o) for o in oofs]), axis=0)
    rank_test = np.mean(np.vstack([rankdata(t) for t in tests]), axis=0)
    rank_test_prob = (rank_test - rank_test.min()) / (rank_test.max() - rank_test.min() + 1e-12)
    return rank_oof, rank_test_prob


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/b6_run"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    parser.add_argument("--arms", nargs="+", default=list(ARMS_DEFAULT), choices=list(ARMS_ALL))
    parser.add_argument(
        "--fuse-arms",
        nargs="+",
        default=list(FUSE_DEFAULT),
        choices=list(ARMS_ALL),
        help="Pre-registered fusion arm set (default: b5 gap).",
    )
    parser.add_argument(
        "--fusion",
        choices=["equal_prob", "equal_rank"],
        default="equal_prob",
        help="Pre-registered fusion rule used for final submission (default equal_prob).",
    )
    parser.add_argument(
        "--weak-filter",
        action="store_true",
        help="Ignore --fuse-arms and apply weak-arm delta filter vs b5.",
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

    arm_results: dict[str, dict[str, Any]] = {}
    for name in args.arms:
        arm_results[name] = run_arm(name, train, test, y, seeds)
        print(f"ARM {name} pooled={arm_results[name]['oof_auc']:.6f}", flush=True)

    if args.weak_filter:
        kept = select_arms(arm_results, list(args.arms), fuse_arms=None)
        weak_note = f"weak_filter kept={kept}"
    else:
        # Only fuse arms that were actually trained.
        fuse_req = [n for n in args.fuse_arms if n in args.arms]
        if not fuse_req:
            fuse_req = list(args.arms)
        kept = select_arms(arm_results, list(args.arms), fuse_arms=fuse_req)
        weak_note = f"preregistered_fuse={kept}"

    oofs = [arm_results[n]["oof"] for n in kept]
    tests = [arm_results[n]["test"] for n in kept]
    if len(kept) == 1:
        prob_oof, prob_test = oofs[0], tests[0]
        rank_oof, rank_test = rankdata(oofs[0]), tests[0]
    else:
        prob_oof, prob_test = fuse_equal_prob(oofs, tests)
        rank_oof, rank_test = fuse_equal_rank(oofs, tests)
    prob_auc = float(roc_auc_score(y, prob_oof))
    rank_auc = float(roc_auc_score(y, rank_oof))

    if args.fusion == "equal_prob":
        final_name, final_auc, final_oof, final_test = "equal_prob", prob_auc, prob_oof, prob_test
    else:
        final_name, final_auc, final_oof, final_test = "equal_rank", rank_auc, rank_oof, rank_test

    b5_auc = arm_results["b5"]["oof_auc"] if "b5" in arm_results else float("nan")
    gap_auc = arm_results["gap"]["oof_auc"] if "gap" in arm_results else float("nan")
    fold_aucs = [r["valid_auc"] for n in kept for r in arm_results[n]["folds"]]
    seed_means = {
        n: float(np.mean(list(arm_results[n]["seed_aucs"].values()))) for n in args.arms
    }
    # Seed-level fusion AUCs (equal_prob of kept arms per seed, then mean).
    seed_fusion_aucs = []
    if all(n in arm_results for n in kept):
        for s in seeds:
            arrs = [arm_results[n]["oof_by_seed"][s] for n in kept]
            fused = np.mean(np.vstack(arrs), axis=0) if len(arrs) > 1 else arrs[0]
            seed_fusion_aucs.append(float(roc_auc_score(y, fused)))
    seed_mean_final = float(np.mean(seed_fusion_aucs)) if seed_fusion_aucs else float("nan")
    seed_std_final = float(np.std(seed_fusion_aucs)) if seed_fusion_aucs else float("nan")

    metrics: dict[str, Any] = {
        "experiment_id": "b6_gap_multiview",
        "recipe": "b5 + mining-gap cats; optional biz/fixed/parse/lossguide; pre-registered equal fusion",
        "git_commit": _git_commit(),
        "data_sha256": {
            "train": _sha256(args.data_dir / "train.csv"),
            "test": _sha256(args.data_dir / "test.csv"),
            "submit": _sha256(args.data_dir / "submit_sample.csv"),
        },
        "baseline_b5_8seed": 0.6981745375887981,
        "gap_to_0_70_baseline": round(0.70 - 0.6981745375887981, 6),
        "seeds": list(seeds),
        "cv_scheme": "StratifiedKFold",
        "n_splits": N_SPLITS,
        "thread_count": THREAD_COUNT,
        "arms_requested": list(args.arms),
        "arms_kept": kept,
        "fuse_arms_preregistered": list(args.fuse_arms),
        "weak_arm_filter": weak_note,
        "arms": {
            n: {
                "oof_auc": arm_results[n]["oof_auc"],
                "seed_aucs": arm_results[n]["seed_aucs"],
                "seed_mean": seed_means[n],
            }
            for n in args.arms
        },
        "fusion": {
            "primary": final_name,
            "equal_prob": prob_auc,
            "equal_rank": rank_auc,
            "b5_only": b5_auc,
            "gap_only": gap_auc,
            "delta_vs_b5_only": float(final_auc - b5_auc) if np.isfinite(b5_auc) else None,
            "note": "primary fusion pre-registered; equal_rank reported only",
        },
        "pooled_oof_auc": float(final_auc),
        "seed_mean": seed_mean_final,
        "seed_std": seed_std_final,
        "seed_fusion_aucs": seed_fusion_aucs,
        "fold_auc_min": float(np.min(fold_aucs)) if fold_aucs else None,
        "fold_auc_max": float(np.max(fold_aucs)) if fold_aucs else None,
        "fold_auc_range": float(np.max(fold_aucs) - np.min(fold_aucs)) if fold_aucs else None,
        "gate_0_70": bool(final_auc >= 0.70),
        "gap_to_0_70": round(0.70 - float(final_auc), 6),
        "early_stopping_on_valid": any(arm_spec(n)[2] for n in kept),
        "elapsed_sec": round(time.time() - started, 1),
        "folds": [row for n in args.arms for row in arm_results[n]["folds"]],
        "target_encoding": "none",
        "policy": "B6 gap-first; fold-local; no TE; equal seed avg; pre-registered fusion",
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
            "fusion_pre_registered": True,
            "b5_freeze_untampered": True,
            "early_stopping_on_valid": any(arm_spec(n)[2] for n in kept),
        },
    }

    if args.shuffled:
        y_s = y.to_numpy().copy()
        np.random.default_rng(2026).shuffle(y_s)
        sh = run_arm("b5", train, test, pd.Series(y_s, name=TARGET), (seeds[0],))
        metrics["shuffled_oof_auc"] = sh["oof_auc"]
        metrics["shuffled_pass"] = bool(0.47 <= sh["oof_auc"] <= 0.53)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_kw: dict[str, Any] = {
        "oof": final_oof,
        "test": final_test,
        "y": y.to_numpy(),
        "oof_equal_prob": prob_oof,
        "test_equal_prob": prob_test,
        "oof_equal_rank": rank_oof,
        "test_equal_rank": rank_test,
    }
    for n in args.arms:
        save_kw[f"oof_{n}"] = arm_results[n]["oof"]
        save_kw[f"test_{n}"] = arm_results[n]["test"]
        for s, arr in arm_results[n]["oof_by_seed"].items():
            save_kw[f"oof_{n}_{s}"] = arr
    np.savez_compressed(args.output_dir / "predictions.npz", **save_kw)
    build_submission(test, sample, final_test, args.output_dir / "submission_b6.csv")
    Path("submissions").mkdir(exist_ok=True)
    build_submission(test, sample, final_test, Path("submissions") / "submission_b6.csv")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "arms": metrics["arms"],
                "arms_kept": kept,
                "fusion": metrics["fusion"],
                "pooled_oof_auc": metrics["pooled_oof_auc"],
                "seed_mean": metrics["seed_mean"],
                "seed_std": metrics["seed_std"],
                "gate_0_70": metrics["gate_0_70"],
                "gap_to_0_70": metrics["gap_to_0_70"],
                "shuffled_oof_auc": metrics.get("shuffled_oof_auc"),
                "shuffled_pass": metrics.get("shuffled_pass"),
                "elapsed_sec": metrics["elapsed_sec"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
