#!/usr/bin/env python3
"""Fold-local KNN claim-rate arm for long residual ranking.

Within each region (fallback: global), kNN in (log_days, condition, ratio, age)
space → credibility-smoothed neighbor claim rate. CatBoost then uses KNN score
+ keepx features. Targets same-region LL pair failures (acc≈0.63).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule
from insurance_claim.b6pro_long_features import build_long_keepx
from insurance_claim.model import IDENTIFIER, TARGET
from insurance_claim.train_b6 import PARAMS_GAP_BAG

B7_FLOOR = 0.7027049552615718
GATE = 0.71
CLOSEST = 0.7054481147284526
WEAK = frozenset({"908d", "f09d", "9685", "fafc", "f167", "ab86"})

PARAMS = {**PARAMS_GAP_BAG, "thread_count": 2, "iterations": 2800, "od_wait": 140}


def knn_rates(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_query: pd.DataFrame,
    *,
    k: int = 64,
    prior: float = 8.0,
) -> np.ndarray:
    """Region-conditional kNN claim rate; fold-fit scaler/neighbors only on X_tr."""
    y_tr = np.asarray(y_tr, dtype=float)
    gmean = float(y_tr.mean())
    cols = []
    for c, transform in [
        ("days", lambda s: np.log1p(np.clip(pd.to_numeric(s, errors="coerce"), 0, None))),
        ("condition", lambda s: pd.to_numeric(s, errors="coerce")),
        ("age_range", lambda s: pd.to_numeric(s, errors="coerce")),
        ("x20", lambda s: pd.to_numeric(s, errors="coerce")),
        ("livability", lambda s: pd.to_numeric(s, errors="coerce")),
    ]:
        if c in X_tr.columns:
            cols.append((c, transform))

    def mat(df: pd.DataFrame) -> np.ndarray:
        blocks = []
        for c, transform in cols:
            v = transform(df[c]).to_numpy(dtype=float)
            blocks.append(v)
        days = pd.to_numeric(df["days"], errors="coerce").to_numpy(float)
        cond = pd.to_numeric(df["condition"], errors="coerce").to_numpy(float)
        blocks.append(cond / (np.abs(days) + 1.0))
        M = np.column_stack(blocks)
        # impute col median from train later
        return M

    tr_m = mat(X_tr)
    qu_m = mat(X_query)
    med = np.nanmedian(tr_m, axis=0)
    tr_m = np.where(np.isfinite(tr_m), tr_m, med)
    qu_m = np.where(np.isfinite(qu_m), qu_m, med)
    scaler = StandardScaler()
    tr_s = scaler.fit_transform(tr_m)
    qu_s = scaler.transform(qu_m)

    region_tr = X_tr["region"].astype(str).to_numpy() if "region" in X_tr.columns else np.array(["__ALL__"] * len(X_tr))
    region_qu = X_query["region"].astype(str).to_numpy() if "region" in X_query.columns else np.array(["__ALL__"] * len(X_query))

    out = np.full(len(X_query), gmean, dtype=float)
    # global fallback NN
    k_global = min(k, len(X_tr))
    nn_g = NearestNeighbors(n_neighbors=k_global, algorithm="auto")
    nn_g.fit(tr_s)
    dist_g, idx_g = nn_g.kneighbors(qu_s)
    # distance weights
    w_g = 1.0 / (dist_g + 1e-3)
    w_g = w_g / w_g.sum(axis=1, keepdims=True)
    global_rate = (w_g * y_tr[idx_g]).sum(axis=1)
    global_rate = (global_rate * k_global + prior * gmean) / (k_global + prior)

    for r in np.unique(region_qu):
        q_idx = np.where(region_qu == r)[0]
        tr_idx = np.where(region_tr == r)[0]
        if len(tr_idx) < max(15, k // 4):
            out[q_idx] = global_rate[q_idx]
            continue
        kk = min(k, len(tr_idx))
        nn = NearestNeighbors(n_neighbors=kk, algorithm="auto")
        nn.fit(tr_s[tr_idx])
        dist, ii = nn.kneighbors(qu_s[q_idx])
        w = 1.0 / (dist + 1e-3)
        w = w / w.sum(axis=1, keepdims=True)
        local = (w * y_tr[tr_idx][ii]).sum(axis=1)
        local = (local * kk + prior * gmean) / (kk + prior)
        # blend local with global by region size credibility
        cred = len(tr_idx) / (len(tr_idx) + 40.0)
        out[q_idx] = cred * local + (1.0 - cred) * global_rate[q_idx]
    return out


def build_knn_keepx(X_tr, y_tr, X_va, X_te):
    tr, va, te, cats = build_long_keepx(X_tr, X_va, X_te)
    knn_tr = knn_rates(X_tr, np.asarray(y_tr, float), X_tr)
    knn_va = knn_rates(X_tr, np.asarray(y_tr, float), X_va)
    knn_te = knn_rates(X_tr, np.asarray(y_tr, float), X_te)
    # self-NN on train is optimistic; replace knn_tr with leave-one-out approx via k+1
    # cheap fix: recompute with k and drop self by using NearestNeighbors k+1
    # For training features, use CV-free smoothed: fit on tr but take 2nd..k+1 neighbor
    knn_tr = knn_rates_loo(X_tr, np.asarray(y_tr, float), k=64)
    for df, val in ((tr, knn_tr), (va, knn_va), (te, knn_te)):
        df["knn_claim_rate"] = val
        df["knn_minus_base"] = val - float(np.mean(y_tr))
    return tr, va, te, cats


def knn_rates_loo(X_tr: pd.DataFrame, y_tr: np.ndarray, k: int = 64, prior: float = 8.0) -> np.ndarray:
    """Leave-one-out style: k+1 neighbors, drop self (distance~0)."""
    y_tr = np.asarray(y_tr, dtype=float)
    gmean = float(y_tr.mean())
    days = np.log1p(np.clip(pd.to_numeric(X_tr["days"], errors="coerce").to_numpy(float), 0, None))
    cond = pd.to_numeric(X_tr["condition"], errors="coerce").to_numpy(float)
    age = pd.to_numeric(X_tr.get("age_range"), errors="coerce").to_numpy(float) if "age_range" in X_tr.columns else np.zeros(len(X_tr))
    x20 = pd.to_numeric(X_tr.get("x20"), errors="coerce").to_numpy(float) if "x20" in X_tr.columns else np.zeros(len(X_tr))
    ratio = cond / (np.expm1(days) + 1.0)
    M = np.column_stack([days, cond, age, x20, ratio])
    med = np.nanmedian(M, axis=0)
    M = np.where(np.isfinite(M), M, med)
    M = StandardScaler().fit_transform(M)
    region = X_tr["region"].astype(str).to_numpy()
    out = np.full(len(X_tr), gmean)
    for r in np.unique(region):
        idx = np.where(region == r)[0]
        if len(idx) < 20:
            # global LOO
            continue
        kk = min(k + 1, len(idx))
        nn = NearestNeighbors(n_neighbors=kk)
        nn.fit(M[idx])
        dist, ii = nn.kneighbors(M[idx])
        # drop first neighbor (self)
        ii = ii[:, 1:]
        dist = dist[:, 1:]
        w = 1.0 / (dist + 1e-3)
        w = w / w.sum(axis=1, keepdims=True)
        local = (w * y_tr[idx][ii]).sum(axis=1)
        n_eff = kk - 1
        out[idx] = (local * n_eff + prior * gmean) / (n_eff + prior)
    # fill remaining with global LOO
    need = np.where(out == gmean)[0]
    if len(need):
        kk = min(k + 1, len(X_tr))
        nn = NearestNeighbors(n_neighbors=kk)
        nn.fit(M)
        dist, ii = nn.kneighbors(M[need])
        ii = ii[:, 1:]
        dist = dist[:, 1:]
        w = 1.0 / (dist + 1e-3)
        w = w / w.sum(axis=1, keepdims=True)
        local = (w * y_tr[ii]).sum(axis=1)
        out[need] = (local * (kk - 1) + prior * gmean) / (kk - 1 + prior)
    return out


def region_blend(max3, long_spec, region, days, wo):
    out = max3.copy()
    long = days >= 3000
    weak_m = long & np.isin(region, list(WEAK))
    other = long & ~np.isin(region, list(WEAK))
    out[weak_m] = long_spec[weak_m]
    out[other] = wo * long_spec[other] + (1 - wo) * max3[other]
    return out


def write_submission(sample, test, pred, path: Path):
    out = sample.copy()
    label_col = [c for c in out.columns if c != IDENTIFIER][0]
    out[label_col] = np.asarray(pred, float)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> int:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train[TARGET].astype(int)
    features = train.drop(columns=[TARGET])
    days = features["days"].to_numpy(float)
    days_te = test["days"].to_numpy(float)
    region = train["region"].astype(str).to_numpy()
    region_te = test["region"].astype(str).to_numpy()
    long = days >= 3000
    idx = np.where(long)[0]

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    tmax = np.maximum.reduce([fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]])
    cur = np.load("artifacts/b6pro_long_best/predictions.npz")
    aging = np.load("artifacts/b6pro_long_only_aging/predictions.npz")
    gap = np.load("artifacts/b6pro_long_only_gap/predictions.npz")
    keepx = np.load("artifacts/b6pro_long_only_keepx/predictions.npz")
    meanL3 = (aging["oof_long_only"] + gap["oof_long_only"] + keepx["oof_long_only"]) / 3.0
    tmeanL3 = (aging["test_long_only"] + gap["test_long_only"] + keepx["test_long_only"]) / 3.0

    seeds = [2026, 2027, 2028, 2029]
    oof_acc = np.zeros(len(y))
    te_acc = np.zeros(len(test))
    # Also pure knn OOF for diagnostics
    knn_oof = np.zeros(len(y))

    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        knn_seed = np.zeros(len(y))
        Xl = features.iloc[idx].reset_index(drop=True)
        yl = y.iloc[idx].reset_index(drop=True)
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xl, yl)):
            gtr, gva = idx[tr], idx[va]
            # pure knn diagnostic on long fold
            knn_seed[gva] = knn_rates(
                features.iloc[gtr].reset_index(drop=True),
                y.iloc[gtr].to_numpy(),
                features.iloc[gva].reset_index(drop=True),
            )
            trd, vad, ted, cats = build_knn_keepx(
                features.iloc[gtr].reset_index(drop=True),
                y.iloc[gtr],
                features.iloc[gva].reset_index(drop=True),
                test.copy(),
            )
            model = CatBoostClassifier(**{**PARAMS, "random_seed": seed + fold})
            model.fit(trd, y.iloc[gtr], eval_set=(vad, y.iloc[gva]), cat_features=cats, use_best_model=True)
            oof[gva] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5.0
            print(
                f"knn_cb s{seed} f{fold} auc={roc_auc_score(y.iloc[gva], oof[gva]):.5f} "
                f"knn={roc_auc_score(y.iloc[gva], knn_seed[gva]):.5f}",
                flush=True,
            )
        print(
            f"knn_cb s{seed} slice={roc_auc_score(y.to_numpy()[long], oof[long]):.6f} "
            f"knn_slice={roc_auc_score(y.to_numpy()[long], knn_seed[long]):.6f}",
            flush=True,
        )
        oof_acc += oof
        te_acc += pte
        knn_oof += knn_seed
    oof_lo = oof_acc / len(seeds)
    te_lo = te_acc / len(seeds)
    knn_oof /= len(seeds)
    print(
        "pooled slice",
        roc_auc_score(y.to_numpy()[long], oof_lo[long]),
        "knn",
        roc_auc_score(y.to_numpy()[long], knn_oof[long]),
        "corr(max3)",
        float(np.corrcoef(oof_lo[long], max3[long])[0, 1]),
        flush=True,
    )

    specs = {}
    mean_mix = 0.5 * (meanL3 + oof_lo)
    tmean_mix = 0.5 * (tmeanL3 + te_lo)
    mean4 = (aging["oof_long_only"] + gap["oof_long_only"] + keepx["oof_long_only"] + oof_lo) / 4.0
    tmean4 = (aging["test_long_only"] + gap["test_long_only"] + keepx["test_long_only"] + te_lo) / 4.0
    for wo in (0.0, 0.1, 0.15, 0.2):
        for sn, ls, tls in [
            ("knn", oof_lo, te_lo),
            ("mix", mean_mix, tmean_mix),
            ("m4", mean4, tmean4),
            ("m3", meanL3, tmeanL3),
        ]:
            specs[f"rb_{sn}_wo{wo}"] = (
                region_blend(max3, ls, region, days, wo),
                region_blend(tmax, tls, region_te, days_te, wo),
            )
    # also pure knn as long_spec
    specs["rb_pureknn_wo0"] = (
        region_blend(max3, knn_oof, region, days, 0.0),
        region_blend(tmax, knn_oof[:0], region_te, days_te, 0.0) if False else region_blend(tmax, np.full(len(test), float(y.mean())), region_te, days_te, 0.0),
    )
    # fix pureknn test: compute knn on full train once for test delivery only after selection
    knn_te_full = knn_rates(features, y.to_numpy(), test)
    specs["rb_pureknn_wo0"] = (
        region_blend(max3, knn_oof, region, days, 0.0),
        region_blend(tmax, knn_te_full, region_te, days_te, 0.0),
    )

    results = {}
    best_name, best_res, best_pair = None, None, None
    for name, (oof_arm, te_arm) in specs.items():
        for tag, oof_arms, te_arms in [
            (f"b7+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], oof_arm], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], te_arm]),
            (f"cur+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], cur["oof"], oof_arm], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], cur["test"], te_arm]),
        ]:
            res = nested_select_rule(y.to_numpy(), oof_arms)
            results[tag] = float(res["nested_oof_auc"])
            print(f"{tag}: nested={res['nested_oof_auc']:.8f}", flush=True)
            if best_res is None or res["nested_oof_auc"] > best_res["nested_oof_auc"]:
                best_name, best_res, best_pair = tag, res, (oof_arms, te_arms)

    deliver_auc = best_res["nested_oof_auc"]
    deliver_oof = best_res["nested_oof"]
    deliver_test = apply_rule(best_res["selected_rule"], best_pair[1])
    if deliver_auc + 1e-12 < B7_FLOOR:
        best_name = "b7_fallback"
        deliver_auc = float(roc_auc_score(y, max3))
        deliver_oof, deliver_test = max3, tmax

    promoted = deliver_auc > CLOSEST + 1e-12
    out_dir = Path("artifacts/b6pro_long_knn")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "predictions.npz",
        y=y.to_numpy(),
        oof=deliver_oof,
        test=deliver_test,
        oof_lo=oof_lo,
        test_lo=te_lo,
        knn_oof=knn_oof,
    )
    write_submission(sample, test, deliver_test, out_dir / "submission_b6pro.csv")
    if promoted:
        dest = Path("artifacts/b6pro_long_best")
        np.savez_compressed(dest / "predictions.npz", y=y.to_numpy(), oof=deliver_oof, test=deliver_test, arm=oof_lo)
        write_submission(sample, test, deliver_test, dest / "submission_b6pro.csv")
        write_submission(sample, test, deliver_test, Path("submissions/b6pro_closest/submission_b6pro.csv"))
        (dest / "metrics.json").write_text(
            json.dumps(
                {
                    "experiment_id": "b6pro_long_best",
                    "spec": best_name,
                    "nested_oof_auc": deliver_auc,
                    "baseline_max3": B7_FLOOR,
                    "gate_0_71": deliver_auc >= GATE,
                    "gap_to_0_71": GATE - deliver_auc,
                    "source": "b6pro_long_knn",
                },
                indent=2,
            )
        )

    metrics = {
        "experiment_id": "b6pro_long_knn",
        "best_fusion": best_name,
        "nested_oof_auc": deliver_auc,
        "all_candidate_nested": results,
        "long_only_slice_auc": float(roc_auc_score(y.to_numpy()[long], oof_lo[long])),
        "knn_slice_auc": float(roc_auc_score(y.to_numpy()[long], knn_oof[long])),
        "promoted_closest": promoted,
        "prev_closest": CLOSEST,
        "gate_0_71": bool(deliver_auc >= GATE),
        "gap_to_0_71": float(GATE - deliver_auc),
        "baseline_max3": B7_FLOOR,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"GATE={'PASS' if metrics['gate_0_71'] else 'FAIL'} nested={deliver_auc:.8f} promoted={promoted}", flush=True)
    return 0 if metrics["gate_0_71"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
