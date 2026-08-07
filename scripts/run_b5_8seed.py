"""Re-run B5 seeds 2030-2033 and pool with 2026-2029 from artifacts/b5_4s."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.model import build_submission
from insurance_claim.train_b5_focus import CAT_PARAMS, N_SPLITS, build_b5

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("submit_sample.csv")
y = train["label"].astype(int)
feats = train.drop(columns=["label"])
base = np.load("artifacts/b5_4s/predictions.npz")

seeds = (2030, 2031, 2032, 2033)
params = dict(CAT_PARAMS)
params["thread_count"] = -1
oof_by: dict[int, np.ndarray] = {}
test_by: dict[int, np.ndarray] = {}
t0 = time.time()
for seed in seeds:
    oof = np.zeros(len(train))
    te = np.zeros(len(test))
    for fold, (a, b) in enumerate(
        StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(feats, y)
    ):
        Xtr = feats.iloc[a].reset_index(drop=True)
        Xva = feats.iloc[b].reset_index(drop=True)
        ytr = y.iloc[a].reset_index(drop=True)
        yva = y.iloc[b].reset_index(drop=True)
        tr, va, te_fe, cats = build_b5(Xtr, Xva, test.copy())
        p = dict(params)
        p["random_seed"] = seed + fold
        model = CatBoostClassifier(**p)
        model.fit(
            tr, ytr, eval_set=(va, yva), cat_features=cats, use_best_model=True, verbose=False
        )
        oof[b] = model.predict_proba(va)[:, 1]
        te += model.predict_proba(te_fe)[:, 1] / N_SPLITS
        print(f"b5 seed={seed} fold={fold} auc={roc_auc_score(yva, oof[b]):.5f}", flush=True)
    print(f"b5 seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
    oof_by[seed] = oof
    test_by[seed] = te

pooled4 = base["oof_b5"] if "oof_b5" in base.files else base["oof"]
test4 = base["test_b5"] if "test_b5" in base.files else base["test"]
oof8 = (4.0 * pooled4 + np.sum(np.vstack([oof_by[s] for s in seeds]), axis=0)) / 8.0
test8 = (4.0 * test4 + np.sum(np.vstack([test_by[s] for s in seeds]), axis=0)) / 8.0
auc8 = float(roc_auc_score(y, oof8))
auc4 = float(roc_auc_score(y, pooled4))
seed_aucs = {str(s): float(roc_auc_score(y, oof_by[s])) for s in seeds}
metrics = {
    "recipe": "b5_8seed_pool",
    "auc_4seed": auc4,
    "auc_8seed": auc8,
    "extra_seed_aucs": seed_aucs,
    "gate_0_698": bool(auc8 >= 0.698),
    "elapsed_sec": round(time.time() - t0, 1),
}
print(json.dumps(metrics, indent=2), flush=True)
out = Path("artifacts/b5_8seed")
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out / "predictions.npz",
    oof=oof8,
    test=test8,
    y=y.to_numpy(),
    oof_4=pooled4,
    **{f"oof_{s}": oof_by[s] for s in seeds},
)
(out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
build_submission(test, sample, test8, out / "submission_b5_8seed.csv")
Path("submissions").mkdir(exist_ok=True)
build_submission(test, sample, test8, Path("submissions") / "submission_b5_8seed.csv")
status = "PASS" if metrics["gate_0_698"] else f"FAIL {auc8:.6f}"
print("GATE", status, flush=True)
