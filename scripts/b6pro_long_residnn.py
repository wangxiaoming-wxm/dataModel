#!/usr/bin/env python3
"""Long residual NN (x0-x18 + hard-example weights) fused with B7 / current closest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from insurance_claim.b6pro_fusion import nested_select_rule

WEAK = frozenset({"908d", "f09d", "9685", "fafc", "f167", "ab86"})
B7_FLOOR = 0.7027049552615718
GATE = 0.71


class Net(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def blend(base, arm, mask, reg, wo=0.15):
    out = base.copy()
    weak = mask & np.isin(reg, list(WEAK))
    other = mask & ~np.isin(reg, list(WEAK))
    out[weak] = arm[weak]
    out[other] = wo * arm[other] + (1 - wo) * base[other]
    return out


def main() -> int:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y = train["label"].to_numpy(int)
    days = train["days"].to_numpy(float)
    long = days >= 3000
    region = train["region"].astype(str).to_numpy()

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    tmax = np.maximum.reduce([fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]])
    cur_arm = np.load("artifacts/b6pro_long_best/predictions.npz")["arm"]

    lo_a = np.load("artifacts/b6pro_long_only_aging/predictions.npz")
    lo_g = np.load("artifacts/b6pro_long_only_gap/predictions.npz")
    lo_k = np.load("artifacts/b6pro_long_only_keepx/predictions.npz")
    base = np.mean([lo_a["oof_long_only"], lo_g["oof_long_only"], lo_k["oof_long_only"]], 0)
    tbase = np.mean([lo_a["test_long_only"], lo_g["test_long_only"], lo_k["test_long_only"]], 0)

    cols = [f"x{i}" for i in range(21)] + [
        "days",
        "condition",
        "cc",
        "V",
        "max_g",
        "livability",
        "age_range",
        "w1",
        "w2",
    ]
    X = train[cols].apply(pd.to_numeric, errors="coerce")
    Xte = test[[c for c in cols if c in test.columns]].apply(pd.to_numeric, errors="coerce")
    for c in X.columns:
        if c not in Xte.columns:
            Xte[c] = 0.0
    Xte = Xte[X.columns]
    X["log_days"] = np.log1p(X["days"].clip(lower=0))
    Xte["log_days"] = np.log1p(Xte["days"].clip(lower=0))
    X["ratio"] = X["condition"] / (X["days"].abs() + 1)
    Xte["ratio"] = Xte["condition"] / (Xte["days"].abs() + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    oof = np.zeros(len(y))
    pte = np.zeros(len(test))
    idx = np.where(long)[0]

    for seed in [0, 1]:
        oof_s = np.zeros(len(y))
        pte_s = np.zeros(len(test))
        for fold, (tr, va) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(idx)), y[idx])
        ):
            gtr, gva = idx[tr], idx[va]

            def mat(ii, is_test=False):
                if is_test:
                    M = Xte.copy()
                    M["max3"] = tmax
                    M["base"] = tbase
                else:
                    M = X.iloc[ii].copy()
                    M["max3"] = max3[ii]
                    M["base"] = base[ii]
                return M.to_numpy(dtype=np.float32)

            Xtr = mat(gtr)
            Xva = mat(gva)
            Xtee = mat(None, True)
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(np.nan_to_num(Xtr))
            Xva = scaler.transform(np.nan_to_num(Xva))
            Xtee = scaler.transform(np.nan_to_num(Xtee))
            model = Net(Xtr.shape[1]).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            hard = np.abs(y[gtr] - max3[gtr])
            w = 0.5 + 2.0 * hard
            w = w / w.mean()
            xt = torch.tensor(Xtr, device=device)
            yt = torch.tensor(y[gtr].astype(np.float32), device=device)
            wt = torch.tensor(w.astype(np.float32), device=device)
            xv = torch.tensor(Xva, device=device)
            best_state, best_auc, bad = None, -1.0, 0
            for ep in range(80):
                model.train()
                perm = torch.randperm(len(xt))
                for i in range(0, len(xt), 256):
                    j = perm[i : i + 256]
                    opt.zero_grad()
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        model(xt[j]), yt[j], weight=wt[j]
                    )
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    pv = torch.sigmoid(model(xv)).cpu().numpy()
                auc = roc_auc_score(y[gva], pv)
                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= 12:
                        break
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                oof_s[gva] = torch.sigmoid(model(xv)).cpu().numpy()
                pte_s += (
                    torch.sigmoid(model(torch.tensor(Xtee, device=device))).cpu().numpy() / 5.0
                )
            print(f"seed{seed} fold{fold} auc={best_auc:.5f}", flush=True)
        print("seed", seed, "long", roc_auc_score(y[long], oof_s[long]), flush=True)
        oof += oof_s / 2
        pte += pte_s / 2

    print(
        "residnn long",
        roc_auc_score(y[long], oof[long]),
        "all",
        roc_auc_score(y, oof),
        flush=True,
    )
    results = {}
    best_auc = -1.0
    for wo in [0.15, 0.3, 0.5]:
        rb = blend(max3, oof, long, region, wo)
        res = nested_select_rule(y, [b7["gap"], b7["gap_bag"], b7["plus"], rb])
        results[f"resid_wo{wo}"] = res["nested_oof_auc"]
        print("resid region", wo, res["nested_oof_auc"], flush=True)
        best_auc = max(best_auc, res["nested_oof_auc"])
    for a in [0.3, 0.5, 0.7]:
        mix = cur_arm.copy()
        mix[long] = a * oof[long] + (1 - a) * cur_arm[long]
        res = nested_select_rule(y, [b7["gap"], b7["gap_bag"], b7["plus"], mix])
        results[f"mix{a}"] = res["nested_oof_auc"]
        print("mix cur", a, res["nested_oof_auc"], flush=True)
        best_auc = max(best_auc, res["nested_oof_auc"])
    res = nested_select_rule(y, [b7["gap"], b7["gap_bag"], b7["plus"], cur_arm, oof])
    results["b7_cur_resid"] = res["nested_oof_auc"]
    print("b7+cur+resid", res["nested_oof_auc"], res["selected_rule"], flush=True)
    best_auc = max(best_auc, res["nested_oof_auc"])

    out = Path("artifacts/b6pro_long_residnn")
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "predictions.npz", y=y, oof_resid=oof, test_resid=pte)
    metrics = {
        "resid_long_auc": float(roc_auc_score(y[long], oof[long])),
        "resid_auc": float(roc_auc_score(y, oof)),
        "best_nested": float(best_auc),
        "all": results,
        "baseline_max3": B7_FLOOR,
        "prev_closest": 0.7054481147284526,
        "gate_0_71": bool(best_auc >= GATE),
        "gap_to_0_71": float(GATE - best_auc),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    return 0 if metrics["gate_0_71"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
