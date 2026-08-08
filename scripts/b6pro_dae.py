#!/usr/bin/env python3
"""Unsupervised DAE on train+test numerics → fold-local classifier (protocol-safe).

Uses only features (no test labels). Encoder embeddings may capture structure
that trees miss for long residual ranking.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule
from insurance_claim.b6pro_long_features import build_long_keepx
from insurance_claim.train_b6 import PARAMS_GAP_BAG

B7_FLOOR = 0.7027049552615718
GATE = 0.71
CLOSEST = float(json.load(open("artifacts/b6pro_long_best/metrics.json"))["nested_oof_auc"])
WEAK = frozenset({"908d", "f09d", "9685", "fafc", "f167", "ab86"})
PARAMS = {**PARAMS_GAP_BAG, "thread_count": 4, "iterations": 2800, "od_wait": 140}


class DAE(nn.Module):
    def __init__(self, d_in: int, d_lat: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Linear(128, d_lat),
        )
        self.dec = nn.Sequential(
            nn.Linear(d_lat, 128),
            nn.ReLU(),
            nn.Linear(128, d_in),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = []
    for c in df.columns:
        if c in ("id", "label"):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.5:
            cols.append(s.to_numpy(dtype=float))
    # also parse car hash-ish from source length etc skip
    M = np.column_stack(cols) if cols else np.zeros((len(df), 1))
    med = np.nanmedian(M, axis=0)
    return np.where(np.isfinite(M), M, med)


def train_dae(X: np.ndarray, seed: int = 0, epochs: int = 80, d_lat: int = 32) -> tuple[DAE, StandardScaler]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)
    torch.manual_seed(seed)
    model = DAE(Xs.shape[1], d_lat)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    model.train()
    n = len(Xs)
    bs = 512
    for ep in range(epochs):
        perm = np.random.default_rng(seed + ep).permutation(n)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xb = torch.from_numpy(Xs[idx])
            noise = torch.randn_like(xb) * 0.1
            recon, _ = model(xb + noise)
            loss = loss_fn(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        if (ep + 1) % 20 == 0:
            print(f"  dae ep{ep+1} loss={total/n:.5f}", flush=True)
    model.eval()
    return model, scaler


def embed(model: DAE, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    Xs = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        z = model.enc(torch.from_numpy(Xs)).numpy()
    return z


def main() -> int:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train["label"].astype(int)
    features = train.drop(columns=["label"])
    days = features["days"].to_numpy(float)
    days_te = test["days"].to_numpy(float)
    region = train["region"].astype(str).to_numpy()
    region_te = test["region"].astype(str).to_numpy()
    long = days >= 3000

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    tmax = np.maximum.reduce([fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]])
    cur = np.load("artifacts/b6pro_long_best/predictions.npz")
    fk = np.load("artifacts/b6pro_full_keepx/predictions.npz")
    aging = np.load("artifacts/b6pro_long_only_aging/predictions.npz")
    gap = np.load("artifacts/b6pro_long_only_gap/predictions.npz")
    keepx = np.load("artifacts/b6pro_long_only_keepx/predictions.npz")
    meanL3 = (aging["oof_long_only"] + gap["oof_long_only"] + keepx["oof_long_only"]) / 3.0
    tmeanL3 = (aging["test_long_only"] + gap["test_long_only"] + keepx["test_long_only"]) / 3.0

    # Unsupervised DAE on train+test (no labels)
    print("=== train DAE on train+test numerics ===", flush=True)
    X_all = numeric_matrix(pd.concat([features, test], axis=0, ignore_index=True))
    dae, scaler = train_dae(X_all, seed=2026, epochs=100, d_lat=40)
    Z_tr = embed(dae, scaler, numeric_matrix(features))
    Z_te = embed(dae, scaler, numeric_matrix(test))
    print("embed shapes", Z_tr.shape, Z_te.shape, flush=True)

    seeds = [2026, 2027, 2028, 2029]
    oof_acc = np.zeros(len(y))
    te_acc = np.zeros(len(test))
    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(features, y)):
            trd, vad, ted, cats = build_long_keepx(
                features.iloc[tr].reset_index(drop=True),
                features.iloc[va].reset_index(drop=True),
                test.copy(),
            )
            # attach DAE embeddings
            for i in range(Z_tr.shape[1]):
                trd[f"dae_{i}"] = Z_tr[tr, i]
                vad[f"dae_{i}"] = Z_tr[va, i]
                ted[f"dae_{i}"] = Z_te[:, i]
            model = CatBoostClassifier(**{**PARAMS, "random_seed": seed + fold})
            model.fit(trd, y.iloc[tr], eval_set=(vad, y.iloc[va]), cat_features=cats, use_best_model=True)
            oof[va] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5.0
            print(f"dae s{seed} f{fold} {roc_auc_score(y.iloc[va], oof[va]):.5f}", flush=True)
        print(
            f"dae s{seed} OOF={roc_auc_score(y, oof):.6f} long={roc_auc_score(y.to_numpy()[long], oof[long]):.6f}",
            flush=True,
        )
        oof_acc += oof
        te_acc += pte
    oof_d, te_d = oof_acc / len(seeds), te_acc / len(seeds)
    print(
        "pooled",
        roc_auc_score(y, oof_d),
        "long",
        roc_auc_score(y.to_numpy()[long], oof_d[long]),
        "corr(max3)",
        float(np.corrcoef(oof_d, max3)[0, 1]),
        flush=True,
    )

    def rb(base, spec, reg, d, wo):
        out = base.copy()
        longm = d >= 3000
        weak = longm & np.isin(reg, list(WEAK))
        other = longm & ~np.isin(reg, list(WEAK))
        out[weak] = spec[weak]
        out[other] = wo * spec[other] + (1 - wo) * base[other]
        return out

    arms = {
        "raw": (oof_d, te_d),
        "mean_m3": (0.5 * (max3 + oof_d), 0.5 * (tmax + te_d)),
        "mean_kx": (0.5 * (fk["oof_k"] + oof_d), 0.5 * (fk["te_k"] + te_d)),
        "max_m3": (np.maximum(max3, oof_d), np.maximum(tmax, te_d)),
    }
    for wo in (0.0, 0.15, 0.2):
        mix = 0.5 * (meanL3 + oof_d)
        tmix = 0.5 * (tmeanL3 + te_d)
        arms[f"rb_mix_w{wo}"] = (
            rb(max3, mix, region, days, wo),
            rb(tmax, tmix, region_te, days_te, wo),
        )

    results = {}
    best_name, best_res, best_pair = None, None, None
    for name, (oa, ta) in arms.items():
        for tag, oof_arms, te_arms in [
            (f"b7+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], oa], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], ta]),
            (f"cur+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], cur["oof"], oa], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], cur["test"], ta]),
            (f"b7+kx+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], fk["oof_k"], oa], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], fk["te_k"], ta]),
        ]:
            res = nested_select_rule(y.to_numpy(), oof_arms)
            results[tag] = float(res["nested_oof_auc"])
            print(f"{tag}: {res['nested_oof_auc']:.8f}", flush=True)
            if best_res is None or res["nested_oof_auc"] > best_res["nested_oof_auc"]:
                best_name, best_res, best_pair = tag, res, (oof_arms, te_arms)

    deliver = best_res["nested_oof_auc"]
    deliver_oof = best_res["nested_oof"]
    deliver_test = apply_rule(best_res["selected_rule"], best_pair[1])
    if deliver < B7_FLOOR:
        deliver = float(roc_auc_score(y, max3))
        deliver_oof, deliver_test = max3, tmax
        best_name = "b7_fallback"
    promoted = deliver > CLOSEST + 1e-12
    out = Path("artifacts/b6pro_dae")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", y=y.to_numpy(), oof=deliver_oof, test=deliver_test, oof_d=oof_d, te_d=te_d)
    lab = [c for c in sample.columns if c != "id"][0]
    sub = sample.copy()
    sub[lab] = deliver_test
    sub.to_csv(out / "submission_b6pro.csv", index=False)
    if promoted:
        dest = Path("artifacts/b6pro_long_best")
        np.savez_compressed(dest / "predictions.npz", y=y.to_numpy(), oof=deliver_oof, test=deliver_test, arm=oof_d)
        sub.to_csv(dest / "submission_b6pro.csv", index=False)
        sub.to_csv("submissions/b6pro_closest/submission_b6pro.csv", index=False)
        (dest / "metrics.json").write_text(
            json.dumps(
                {
                    "experiment_id": "b6pro_long_best",
                    "spec": best_name,
                    "nested_oof_auc": deliver,
                    "baseline_max3": B7_FLOOR,
                    "gate_0_71": deliver >= GATE,
                    "gap_to_0_71": GATE - deliver,
                    "source": "b6pro_dae",
                },
                indent=2,
            )
        )
    metrics = {
        "best": best_name,
        "nested": deliver,
        "dae": float(roc_auc_score(y, oof_d)),
        "dae_long": float(roc_auc_score(y.to_numpy()[long], oof_d[long])),
        "promoted": promoted,
        "gate": deliver >= GATE,
        "top": sorted(results.items(), key=lambda kv: -kv[1])[:12],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"GATE={'PASS' if deliver >= GATE else 'FAIL'} nested={deliver:.8f} promoted={promoted}", flush=True)
    return 0 if deliver >= GATE else 2


if __name__ == "__main__":
    raise SystemExit(main())
