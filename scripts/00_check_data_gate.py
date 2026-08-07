#!/usr/bin/env python3
"""Independent data-gate check for train/test/submit_sample integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    sub = pd.read_csv(ROOT / "submit_sample.csv")

    feat_train = [c for c in train.columns if c not in ("id", "label")]
    feat_test = [c for c in test.columns if c != "id"]
    train_ids = set(train["id"].astype(str))
    test_ids = set(test["id"].astype(str))
    sub_ids = set(sub["id"].astype(str))

    train_keys = set(pd.util.hash_pandas_object(train[feat_train], index=False))
    test_keys = set(pd.util.hash_pandas_object(test[feat_test], index=False))

    checks = {
        "PASS_no_id_overlap": len(train_ids & test_ids) == 0,
        "PASS_label_only_train": ("label" in train.columns)
        and ("label" not in test.columns),
        "PASS_submit_structure": (
            list(sub.columns) == ["id", "label"]
            and sub_ids == test_ids
            and int(sub["id"].duplicated().sum()) == 0
            and float(sub["label"].min()) >= 0.0
            and float(sub["label"].max()) <= 1.0
        ),
        "PASS_no_dup_ids": int(train["id"].duplicated().sum()) == 0
        and int(test["id"].duplicated().sum()) == 0,
        "PASS_feature_cols_match": feat_train == feat_test,
        "PASS_binary_label": sorted(train["label"].dropna().unique().tolist()) == [0, 1]
        and int(train["label"].isna().sum()) == 0,
    }

    payload = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "submit_rows": int(len(sub)),
        "n_features": len(feat_train),
        "pos_rate": float(train["label"].mean()),
        "id_overlap_train_test": len(train_ids & test_ids),
        "exact_feature_row_overlap": len(train_keys & test_keys),
        "submit_id_order_matches_test": list(sub["id"].astype(str))
        == list(test["id"].astype(str)),
        "file_sha256": {
            "train.csv": sha256(ROOT / "train.csv"),
            "test.csv": sha256(ROOT / "test.csv"),
            "submit_sample.csv": sha256(ROOT / "submit_sample.csv"),
        },
        "checks": checks,
        "DATA_GATE": "PASS" if all(checks.values()) else "FAIL",
    }

    out = ROOT / "artifacts" / "data_gate" / "new_data_integrity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Keep human protocol JSON as source of truth if already richer; still print gate.
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["DATA_GATE"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
