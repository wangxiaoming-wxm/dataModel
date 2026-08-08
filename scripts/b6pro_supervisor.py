#!/usr/bin/env python3
"""Independent B6pro supervisor: audit artifacts only; never invent scores."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

TARGET = 0.715
BASELINE_B6 = 0.6989746962571622


def scan(artifacts: Path) -> list[dict]:
    rows = []
    for p in sorted(artifacts.glob("b6pro*/metrics.json")):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            rows.append({"path": str(p), "error": str(e)})
            continue
        nested = m.get("nested_oof_auc")
        pooled = m.get("pooled_oof_auc")
        row = {
            "path": str(p),
            "experiment_id": m.get("experiment_id"),
            "nested_oof_auc": nested,
            "pooled_oof_auc": pooled,
            "gate_0_715": m.get("gate_0_715"),
            "gap_to_0_715": m.get("gap_to_0_715"),
            "selected_rule": (m.get("fusion") or {}).get("selected_rule"),
            "protocol_id": m.get("protocol_id"),
            "reference_bootstrap": (m.get("protocol_declaration") or {}).get(
                "reference_plus_bootstrap"
            ),
        }
        pred = p.parent / "predictions.npz"
        if pred.exists() and nested is not None:
            z = np.load(pred)
            if "oof" in z.files and "y" in z.files:
                recomputed = float(roc_auc_score(z["y"], z["oof"]))
                row["recomputed_auc"] = recomputed
                row["recompute_delta"] = abs(recomputed - float(nested))
                row["recompute_ok"] = row["recompute_delta"] < 1e-8
        rows.append(row)
    return rows


def closest(rows: list[dict]) -> dict | None:
    cands = [r for r in rows if isinstance(r.get("nested_oof_auc"), (int, float))]
    # Prefer self-trained (not reference bootstrap) when available
    self_trained = [r for r in cands if not r.get("reference_bootstrap")]
    pool = self_trained or cands
    if not pool:
        return None
    return max(pool, key=lambda r: float(r["nested_oof_auc"]))


def write_status(out: Path, rows: list[dict], closest_row: dict | None) -> dict:
    best = float(closest_row["nested_oof_auc"]) if closest_row else None
    gate = bool(best is not None and best >= TARGET)
    status = {
        "protocol_id": "IA-AUC715-B6PRO-v1",
        "role": "independent_supervisor",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "baseline_b6": BASELINE_B6,
        "candidates": rows,
        "closest_honest": closest_row,
        "closest_honest_nested_oof_auc": best,
        "gate_0_715": gate,
        "verdict": "PASS_CANDIDATE" if gate else "REJECT_WAITING",
        "deliver_0_715_allowed": False,  # human/final packet still required
        "note": "Supervisor does not invent scores; scans artifacts only.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/b6pro_audit/waiting_status.json"))
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=120)
    args = ap.parse_args()

    while True:
        rows = scan(args.artifacts)
        closest_row = closest(rows)
        status = write_status(args.out, rows, closest_row)
        print(
            json.dumps(
                {
                    "updated_at": status["updated_at"],
                    "closest_honest_nested_oof_auc": status["closest_honest_nested_oof_auc"],
                    "gate_0_715": status["gate_0_715"],
                    "verdict": status["verdict"],
                    "candidates_n": len(rows),
                },
                indent=2,
            ),
            flush=True,
        )
        if not args.loop:
            break
        if status["gate_0_715"]:
            print("TARGET MET on disk — supervisor remains for final packet check", flush=True)
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
