#!/usr/bin/env python3
"""Fast torch MLP arm on fold-local numeric+code features; fuse with B7 max3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule
from insurance_claim.ebm_arm import build_ebm_features
from insurance_claim.model import build_submission

TARGET = 0.71


class MLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _to_num(df: pd.DataFrame) -> np.ndarray:
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype("category").cat.codes
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out.to_numpy(dtype=np.float32)


def train_one(Xtr, ytr, Xva, yva, seed: int, epochs: int = 40):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    device = torch.device("cpu")
    model = MLP(Xtr_s.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.tensor(Xtr_s, device=device)
    yt = torch.tensor(ytr.astype(np.float32), device=device)
    xv = torch.tensor(Xva_s, device=device)
    best_state, best_auc, bad = None, -1.0, 0
    bs = 512
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=torch.Generator().manual_seed(seed + ep))
        for i in range(0, len(xt), bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(xv)).cpu().numpy()
        auc = roc_auc_score(yva, pv)
        if auc > best_auc:
            best_auc, best_state, bad = auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, best_auc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027])
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_mlp"))
    args = ap.parse_args()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train["label"].astype(int).to_numpy()
    raw = train.drop(columns=["label"])
    raw_te = test.copy()

    oofs, tests = [], []
    for seed in args.seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(raw, y)):
            Xtr = _to_num(build_ebm_features(raw.iloc[tr].reset_index(drop=True)))
            Xva = _to_num(build_ebm_features(raw.iloc[va].reset_index(drop=True)))
            Xte = _to_num(build_ebm_features(raw_te))
            # align cols by min width
            d = min(Xtr.shape[1], Xva.shape[1], Xte.shape[1])
            Xtr, Xva, Xte = Xtr[:, :d], Xva[:, :d], Xte[:, :d]
            model, scaler, fauc = train_one(Xtr, y[tr], Xva, y[va], seed + fold)
            Xva_s = scaler.transform(Xva)
            Xte_s = scaler.transform(Xte)
            with torch.no_grad():
                oof[va] = torch.sigmoid(model(torch.tensor(Xva_s))).numpy()
                pte += torch.sigmoid(model(torch.tensor(Xte_s))).numpy() / 5
            print(f"mlp seed={seed} fold={fold} auc={fauc:.5f}", flush=True)
        print(f"mlp seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
        oofs.append(oof)
        tests.append(pte)

    oof = np.mean(np.vstack(oofs), 0)
    te = np.mean(np.vstack(tests), 0)
    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    fused = nested_select_rule(y, [b7["gap"], b7["gap_bag"], b7["plus"], oof])
    print(
        "solo",
        roc_auc_score(y, oof),
        "corr",
        np.corrcoef(oof, 0.5 * (b7["gap"] + b7["gap_bag"]))[0, 1],
        "nested",
        fused["nested_oof_auc"],
        fused["selected_rule"],
    )
    tp = apply_rule(fused["selected_rule"], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], te])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "predictions.npz", y=y, oof=fused["nested_oof"], test=tp, oof_mlp=oof, test_mlp=te)
    build_submission(test, sample, tp, args.output_dir / "submission_b6pro.csv")
    metrics = {
        "experiment_id": "b6pro_mlp",
        "oof_auc": float(roc_auc_score(y, oof)),
        "nested_oof_auc": fused["nested_oof_auc"],
        "selected_rule": fused["selected_rule"],
        "full_data_scores": fused["full_data_scores"],
        "baseline_max3": 0.7027049552615718,
        "gate_0_71": fused["nested_oof_auc"] >= TARGET,
        "gap_to_0_71": round(TARGET - fused["nested_oof_auc"], 6),
        "public_b7_signal": {"local": 0.702704955, "public": 0.707},
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "no_oof_weight_search": True,
            "new_data_only": True,
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ["oof_auc", "nested_oof_auc", "gate_0_71", "gap_to_0_71"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
