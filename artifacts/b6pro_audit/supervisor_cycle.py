#!/usr/bin/env python3
"""One supervisor cycle: scan, update waiting_status, append SUPERVISOR_LOG, exit.

Exit 0 if gate_0_715 true (candidate on disk >= 0.715).
Exit 2 if still waiting.
Exit 3 if cheating / frozen tamper detected.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/workspace")
LOG = ROOT / "artifacts/b6pro_audit/SUPERVISOR_LOG.md"
STATUS = ROOT / "artifacts/b6pro_audit/waiting_status.json"
BASELINE = ROOT / "artifacts/b6pro_audit/b6_frozen_integrity_baseline.json"
FAIL_NOTE = ROOT / "artifacts/b6pro_audit/FAIL_CHEATING_NOTE.md"


def frozen_ok() -> tuple[bool, str]:
    if not BASELINE.exists():
        return True, "no baseline yet"
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    for rel, meta in base["files"].items():
        p = Path(rel)
        if not p.exists():
            return False, f"missing {rel}"
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        if h != meta["sha256_16"] or p.stat().st_size != meta["size"]:
            return False, f"hash/size mismatch {rel}"
    return True, "OK"


def progress_hint() -> str:
    log = ROOT / "artifacts/b6pro_loop/loop.log"
    if not log.exists():
        return "no loop.log"
    lines = [ln for ln in log.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    return lines[-1][:120] if lines else "empty loop.log"


def next_cycle() -> int:
    if not LOG.exists():
        return 1
    n = 0
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if line.startswith("| ") and not line.startswith("| cycle") and not line.startswith("|---"):
            parts = line.split("|")
            if len(parts) > 1 and parts[1].strip().isdigit():
                n = max(n, int(parts[1].strip()))
    return n + 1


def append_log(cycle: int, status: dict, notes: str) -> None:
    ch = status.get("closest_honest") or {}
    exp = ch.get("experiment_id") or "none"
    nested = status.get("closest_honest_nested_oof_auc")
    nested_s = f"{nested:.16f}" if isinstance(nested, (int, float)) else "n/a"
    recompute = "yes" if ch.get("recompute_ok") else ("n/a" if not ch else "NO")
    ref = ch.get("reference_bootstrap")
    ref_s = "**true**" if ref else ("false" if ref is False else "n/a")
    rule = ch.get("selected_rule") or "n/a"
    verdict = status.get("verdict")
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = (
        f"| {cycle} | {utc} | {exp} | {nested_s} | {recompute} | {ref_s} | {rule} | "
        f"{verdict} | {notes} |\n"
    )
    if not LOG.exists():
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(
            "# B6pro Independent Supervisor Log\n\n"
            "Protocol: `IA-AUC715-B6PRO-v1` · Target nested_oof_auc ≥ 0.715  \n"
            "Role: audit-only (no training/tuning). Scores from on-disk artifacts only.\n\n"
            "| cycle | utc | closest_exp | nested_oof_auc | recomputed_ok | ref_bootstrap | selected_rule | verdict | notes |\n"
            "|------:|-----|-------------|---------------:|:-------------:|:-------------:|---------------|---------|-------|\n",
            encoding="utf-8",
        )
    text = LOG.read_text(encoding="utf-8")
    if f"| {cycle} |" in text:
        return
    LOG.write_text(text + row, encoding="utf-8")


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run(
        [sys.executable, "scripts/b6pro_supervisor.py"],
        cwd=str(ROOT),
        env=env,
        check=False,
    )
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    cycle = next_cycle()
    fok, fmsg = frozen_ok()
    hint = progress_hint()

    if not fok:
        FAIL_NOTE.write_text(
            "# FAIL — b6_frozen tampering detected\n\n"
            f"Detected at {datetime.now(timezone.utc).isoformat()}.\n"
            f"Detail: {fmsg}\n"
            "Integrity hashes diverge from `b6_frozen_integrity_baseline.json`.\n"
            "Verdict: REJECT. deliver_0_715_allowed remains false.\n",
            encoding="utf-8",
        )
        notes = f"FAIL: b6_frozen {fmsg}"
        append_log(cycle, status, notes)
        print(json.dumps({"cycle": cycle, "exit": 3, "notes": notes}, indent=2))
        return 3

    best = status.get("closest_honest_nested_oof_auc")
    gate = bool(status.get("gate_0_715"))
    ch = status.get("closest_honest") or {}
    if gate:
        notes = (
            f"PASS_CANDIDATE nested={best}; ref={ch.get('reference_bootstrap')}; "
            f"recompute_ok={ch.get('recompute_ok')}; rule={ch.get('selected_rule')}; "
            f"progress={hint}"
        )
        code = 0
    else:
        notes = f"below 0.715; frozen OK; progress={hint}"
        code = 2

    append_log(cycle, status, notes)
    print(
        json.dumps(
            {
                "cycle": cycle,
                "closest": best,
                "verdict": status.get("verdict"),
                "gate": gate,
                "deliver": status.get("deliver_0_715_allowed"),
                "frozen_ok": fok,
                "n_candidates": len(status.get("candidates") or []),
                "exit": code,
            },
            indent=2,
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
