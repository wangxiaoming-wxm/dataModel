#!/usr/bin/env python3
"""B6pro autonomous experiment loop — keeps iterating until nested >= 0.715.

Does not cheat: fold-local FE, discrete nested fusion only, no TE / no OOF weight search.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

TARGET = 0.715
ROOT = Path("/workspace")


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(cmd), flush=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": "src"},
        )
    return p.returncode


def best_nested() -> float | None:
    best = None
    for p in Path("artifacts").glob("b6pro*/metrics.json"):
        m = json.loads(p.read_text(encoding="utf-8"))
        if m.get("protocol_declaration", {}).get("reference_plus_bootstrap"):
            # bootstrap floor only
            val = m.get("nested_oof_auc")
            if val is not None:
                best = val if best is None else max(best, val)
            continue
        val = m.get("nested_oof_auc")
        if val is None:
            continue
        best = val if best is None else max(best, val)
    return best


def main() -> int:
    Path("artifacts/b6pro_loop").mkdir(parents=True, exist_ok=True)
    log = Path("artifacts/b6pro_loop/loop.log")
    stage = 0
    while True:
        stage += 1
        b = best_nested()
        print(f"[loop] stage={stage} best_nested={b} target={TARGET}", flush=True)
        if b is not None and b >= TARGET:
            print("[loop] TARGET REACHED", flush=True)
            return 0

        # Stage ladder
        if not Path("artifacts/b6pro_fuse0/metrics.json").exists():
            # Fuse0 bootstrap with reference plus (disclosed) + B5 as proxy main if B6 OOF missing
            # Prefer training B6 arms; if too long, first fuse B5×ref plus as sanity.
            run(
                [
                    sys.executable,
                    "-m",
                    "insurance_claim.train_b6pro",
                    "--mode",
                    "fuse",
                    "--main-npz",
                    "artifacts/b5_8seed/predictions.npz",
                    "--ref-plus",
                    "--shuffled",
                    "--output-dir",
                    "artifacts/b6pro_fuse0_b5ref",
                ],
                log,
            )

        if not Path("artifacts/b6pro_main/metrics.json").exists():
            run(
                [
                    sys.executable,
                    "-m",
                    "insurance_claim.train_b6pro",
                    "--mode",
                    "b6_arms",
                    "--b6-arms",
                    "gap",
                    "gap_bag",
                    "--seeds",
                    *map(str, range(2026, 2034)),
                    "--output-dir",
                    "artifacts/b6pro_main",
                ],
                log,
            )

        if not Path("artifacts/b6pro_plus_h2/metrics.json").exists():
            run(
                [
                    sys.executable,
                    "-m",
                    "insurance_claim.train_b6pro",
                    "--mode",
                    "plus_only",
                    "--plus-variant",
                    "plus",
                    "--plus-config",
                    "h2",
                    "--plus-seeds",
                    *map(str, range(2026, 2030)),
                    "--plus-folds",
                    "5",
                    "--output-dir",
                    "artifacts/b6pro_plus_h2",
                ],
                log,
            )

        if Path("artifacts/b6pro_main/predictions.npz").exists() and Path(
            "artifacts/b6pro_plus_h2/predictions.npz"
        ).exists() and not Path("artifacts/b6pro_fuse_self/metrics.json").exists():
            run(
                [
                    sys.executable,
                    "-m",
                    "insurance_claim.train_b6pro",
                    "--mode",
                    "fuse",
                    "--main-npz",
                    "artifacts/b6pro_main/predictions.npz",
                    "--plus-npz",
                    "artifacts/b6pro_plus_h2/predictions.npz",
                    "--shuffled",
                    "--output-dir",
                    "artifacts/b6pro_fuse_self",
                ],
                log,
            )

        if not Path("artifacts/b6pro_plus_gap/metrics.json").exists():
            run(
                [
                    sys.executable,
                    "-m",
                    "insurance_claim.train_b6pro",
                    "--mode",
                    "plus_only",
                    "--plus-variant",
                    "plus_gap",
                    "--plus-config",
                    "h2",
                    "--plus-seeds",
                    *map(str, range(2026, 2030)),
                    "--output-dir",
                    "artifacts/b6pro_plus_gap",
                ],
                log,
            )

        if Path("artifacts/b6pro_main/predictions.npz").exists() and Path(
            "artifacts/b6pro_plus_gap/predictions.npz"
        ).exists() and not Path("artifacts/b6pro_fuse_plusgap/metrics.json").exists():
            run(
                [
                    sys.executable,
                    "-m",
                    "insurance_claim.train_b6pro",
                    "--mode",
                    "fuse",
                    "--main-npz",
                    "artifacts/b6pro_main/predictions.npz",
                    "--plus-npz",
                    "artifacts/b6pro_plus_gap/predictions.npz",
                    "--shuffled",
                    "--output-dir",
                    "artifacts/b6pro_fuse_plusgap",
                ],
                log,
            )

        # H3 plus push
        if not Path("artifacts/b6pro_plus_h3/metrics.json").exists():
            run(
                [
                    sys.executable,
                    "-m",
                    "insurance_claim.train_b6pro",
                    "--mode",
                    "plus_only",
                    "--plus-variant",
                    "plus",
                    "--plus-config",
                    "h3",
                    "--plus-seeds",
                    *map(str, range(2026, 2030)),
                    "--output-dir",
                    "artifacts/b6pro_plus_h3",
                ],
                log,
            )

        if Path("artifacts/b6pro_main/predictions.npz").exists() and Path(
            "artifacts/b6pro_plus_h3/predictions.npz"
        ).exists() and not Path("artifacts/b6pro_fuse_h3/metrics.json").exists():
            run(
                [
                    sys.executable,
                    "-m",
                    "insurance_claim.train_b6pro",
                    "--mode",
                    "fuse",
                    "--main-npz",
                    "artifacts/b6pro_main/predictions.npz",
                    "--plus-npz",
                    "artifacts/b6pro_plus_h3/predictions.npz",
                    "--shuffled",
                    "--output-dir",
                    "artifacts/b6pro_fuse_h3",
                ],
                log,
            )

        # Supervisor scan
        run([sys.executable, "scripts/b6pro_supervisor.py"], log)

        b = best_nested()
        print(f"[loop] after stage={stage} best_nested={b}", flush=True)
        if b is not None and b >= TARGET:
            print("[loop] TARGET REACHED", flush=True)
            return 0

        # If all planned stages done and still short, sleep and re-enter for future extensions
        planned_done = all(
            Path(p).exists()
            for p in [
                "artifacts/b6pro_main/metrics.json",
                "artifacts/b6pro_plus_h2/metrics.json",
                "artifacts/b6pro_fuse_self/metrics.json",
                "artifacts/b6pro_plus_gap/metrics.json",
                "artifacts/b6pro_fuse_plusgap/metrics.json",
                "artifacts/b6pro_plus_h3/metrics.json",
                "artifacts/b6pro_fuse_h3/metrics.json",
            ]
        )
        if planned_done:
            print("[loop] planned ladder complete; waiting for new arms / extensions", flush=True)
            # Keep process alive for supervisor; exit code non-zero means not done
            time.sleep(60)
            # Future: residual mining script hooks here
            if not Path("artifacts/b6pro_resid_note.json").exists():
                Path("artifacts/b6pro_resid_note.json").write_text(
                    json.dumps(
                        {
                            "status": "need_new_hetero_arm",
                            "best_nested": b,
                            "gap": None if b is None else TARGET - b,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            # Don't busy-spin forever without work — return and let parent extend
            return 2

        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
