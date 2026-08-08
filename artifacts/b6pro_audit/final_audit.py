#!/usr/bin/env python3
"""Deep A–F audit when a candidate claims nested_oof_auc >= 0.715.

Writes docs/supervision/B6PRO_FINAL_AUDIT_OPINION.md and updates waiting_status.
Never invents scores. Audit-only.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path("/workspace")
THRESH = json.loads((ROOT / "docs/supervision/B6PRO_AUDIT_THRESHOLDS.json").read_text(encoding="utf-8"))
STATUS = ROOT / "artifacts/b6pro_audit/waiting_status.json"
OPINION = ROOT / "docs/supervision/B6PRO_FINAL_AUDIT_OPINION.md"
BASELINE = ROOT / "artifacts/b6pro_audit/b6_frozen_integrity_baseline.json"
TARGET = float(THRESH["target_nested_oof_auc"])


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def recompute(pred: Path, nested: float) -> dict:
    z = np.load(pred)
    assert "oof" in z.files and "y" in z.files
    auc = float(roc_auc_score(z["y"], z["oof"]))
    delta = abs(auc - float(nested))
    return {"recomputed_auc": auc, "delta": delta, "ok": delta < 1e-8, "n": int(len(z["y"]))}


def frozen_ok() -> bool:
    if not BASELINE.exists():
        return False
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    for rel, meta in base["files"].items():
        p = Path(rel)
        if not p.exists():
            return False
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        if h != meta["sha256_16"] or p.stat().st_size != meta["size"]:
            return False
    return True


def audit_candidate(metrics_path: Path) -> dict:
    m = load_metrics(metrics_path)
    pred = metrics_path.parent / "predictions.npz"
    nested = m.get("nested_oof_auc")
    decl = m.get("protocol_declaration") or {}
    fusion = m.get("fusion") or {}
    seeds = m.get("seeds") or []
    checks = {}

    # A score
    checks["A_nested_ge_target"] = isinstance(nested, (int, float)) and float(nested) >= TARGET
    rc = recompute(pred, float(nested)) if pred.exists() and nested is not None else {"ok": False, "delta": None}
    checks["A_recompute_ok"] = bool(rc.get("ok"))

    # B CV
    checks["B_seeds_ge_8"] = len(seeds) >= int(THRESH["accept"]["min_n_seeds_main_arm"])
    checks["B_equal_seed"] = decl.get("equal_seed_average") is True
    # SKF>=5 inferred from fold_rules length or declaration
    fold_rules = fusion.get("fold_rules") or []
    checks["B_skf_ge_5"] = len(fold_rules) >= int(THRESH["accept"]["min_n_splits"]) or decl.get("rule_selection_nested") is True

    # C shuffle
    hard = THRESH["accept"]["shuffled_oof_auc_hard_band"]
    shuf = m.get("shuffled_oof_auc")
    shuf_plus_max = m.get("shuffled_plus_max_auc")
    # Prefer explicit label-shuffle if present; else note max-collapse separately
    if isinstance(shuf, (int, float)):
        checks["C_shuffled_in_hard_band"] = hard[0] <= float(shuf) <= hard[1]
        checks["C_shuffled_value"] = float(shuf)
    else:
        checks["C_shuffled_in_hard_band"] = None  # unknown / not reported
        checks["C_shuffled_value"] = None
    if isinstance(shuf_plus_max, (int, float)):
        checks["C_shuffle_collapse_max"] = float(shuf_plus_max)
        # For max rule, collapse should drop meaningfully below nested; soft gate <0.66 used by trainee is weaker than protocol
        checks["C_max_collapse_meaningful"] = float(shuf_plus_max) < float(nested) - 0.02

    # D protocol declaration
    required_true = [
        "no_test_labels",
        "no_global_te",
        "fold_local_fe",
        "no_oof_weight_search",
        "fusion_rules_preregistered",
        "rule_selection_nested",
        "equal_seed_average",
        "new_data_only",
        "b6_freeze_untampered",
    ]
    checks["D_declaration_all_true"] = all(decl.get(k) is True for k in required_true)
    checks["D_reference_bootstrap"] = bool(decl.get("reference_plus_bootstrap"))
    checks["D_no_reference_for_final"] = not checks["D_reference_bootstrap"]

    # E fusion
    rules = fusion.get("rules_preregistered") or []
    checks["E_rules_preregistered"] = set(THRESH["accept"]["fusion_rules_preregistered"]).issubset(set(rules)) or decl.get("fusion_rules_preregistered") is True
    checks["E_nested_rule_selection"] = fusion.get("rule_selection") == "nested_5fold" or decl.get("rule_selection_nested") is True
    checks["E_selected_rule"] = fusion.get("selected_rule")
    prereg = set(THRESH["accept"]["fusion_rules_preregistered"])
    selected = fusion.get("selected_rule")
    checks["E_selected_rule_preregistered"] = selected in prereg if selected else False
    used = set(fusion.get("rules_used") or [])
    checks["E_no_extra_rules_in_search"] = (not used) or used.issubset(prereg)
    arms = m.get("arm_names") or []
    checks["D_uses_ref_arm"] = any("ref" in str(a).lower() for a in arms) or bool(
        (m.get("plus") or {}).get("reference_bootstrap")
    )

    # F not single seed packaging
    checks["F_multi_seed"] = len(seeds) >= 8

    checks["frozen_integrity"] = frozen_ok()

    hard_fail = []
    if not checks["A_nested_ge_target"]:
        hard_fail.append("A nested < 0.715")
    if not checks["A_recompute_ok"]:
        hard_fail.append("A recompute mismatch")
    if not checks["B_seeds_ge_8"]:
        hard_fail.append("B seeds < 8")
    if not checks["D_declaration_all_true"]:
        hard_fail.append("D protocol_declaration incomplete")
    if checks["D_reference_bootstrap"]:
        hard_fail.append("D reference_plus_bootstrap=true (not eligible for final PASS)")
    if checks["D_uses_ref_arm"] and not checks["D_reference_bootstrap"]:
        hard_fail.append("D uses ref arm without reference_plus_bootstrap disclosure")
    if not checks["E_nested_rule_selection"]:
        hard_fail.append("E rule selection not nested")
    if not checks["E_selected_rule_preregistered"]:
        hard_fail.append(f"E selected_rule={selected!r} not in preregistered set")
    if not checks["E_no_extra_rules_in_search"]:
        hard_fail.append("E fusion searched non-preregistered rules")
    if not checks["frozen_integrity"]:
        hard_fail.append("frozen integrity fail")
    if checks["C_shuffled_in_hard_band"] is False:
        hard_fail.append("C shuffled outside hard band")

    selected = fusion.get("selected_rule")
    conditional = False
    if selected == "max" and checks["A_nested_ge_target"] and not hard_fail:
        # borderline max → CONDITIONAL unless collapse clearly documented
        if not checks.get("C_max_collapse_meaningful"):
            conditional = True
            hard_fail.append("max rule without clear shuffle-collapse → CONDITIONAL")

    if hard_fail and any(x.startswith("D reference") or x.startswith("A ") or x.startswith("frozen") for x in hard_fail):
        verdict = "REJECT"
        deliver = False
    elif conditional or (hard_fail and selected == "max"):
        verdict = "CONDITIONAL"
        deliver = False
    elif not hard_fail:
        verdict = "PASS"
        deliver = True
    else:
        verdict = "REJECT"
        deliver = False

    return {
        "metrics_path": str(metrics_path),
        "experiment_id": m.get("experiment_id"),
        "nested_oof_auc": nested,
        "recompute": rc,
        "checks": checks,
        "hard_fail": hard_fail,
        "verdict": verdict,
        "deliver_0_715_allowed": deliver,
        "protocol_declaration": decl,
        "selected_rule": selected,
    }


def write_opinion(result: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# B6PRO Final Audit Opinion",
        "",
        f"- Protocol: `{THRESH['protocol_id']}`",
        f"- Auditor role: independent supervisor (no training/tuning)",
        f"- Written at: `{now}`",
        f"- Candidate: `{result['experiment_id']}` @ `{result['metrics_path']}`",
        f"- nested_oof_auc (reported): `{result['nested_oof_auc']}`",
        f"- recomputed AUC: `{result['recompute'].get('recomputed_auc')}` delta=`{result['recompute'].get('delta')}` ok=`{result['recompute'].get('ok')}`",
        f"- selected_rule: `{result['selected_rule']}`",
        f"- Verdict: **{result['verdict']}**",
        f"- deliver_0_715_allowed: `{result['deliver_0_715_allowed']}`",
        "",
        "## Checks (A–F)",
        "",
        "```json",
        json.dumps(result["checks"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Hard fail / blockers",
        "",
    ]
    if result["hard_fail"]:
        for h in result["hard_fail"]:
            lines.append(f"- {h}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Protocol declaration (as recorded)",
            "",
            "```json",
            json.dumps(result["protocol_declaration"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Notes",
            "",
            "- Scores read only from on-disk `metrics.json` / `predictions.npz`.",
            "- Final PASS requires nested_oof_auc ≥ 0.715, recompute match <1e-8,",
            "  protocol_declaration all required trues, no reference_plus_bootstrap,",
            "  nested discrete fusion, and intact b6_frozen trees.",
            "- max-rule borderline without clear shuffle-collapse → CONDITIONAL at most.",
            "",
        ]
    )
    OPINION.parent.mkdir(parents=True, exist_ok=True)
    OPINION.write_text("\n".join(lines), encoding="utf-8")


def update_status(result: dict) -> None:
    if STATUS.exists():
        status = json.loads(STATUS.read_text(encoding="utf-8"))
    else:
        status = {}
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    status["final_audit"] = {
        "verdict": result["verdict"],
        "deliver_0_715_allowed": result["deliver_0_715_allowed"],
        "experiment_id": result["experiment_id"],
        "nested_oof_auc": result["nested_oof_auc"],
        "hard_fail": result["hard_fail"],
    }
    status["deliver_0_715_allowed"] = bool(result["deliver_0_715_allowed"])
    if result["deliver_0_715_allowed"]:
        status["verdict"] = "PASS"
    elif result["verdict"] == "CONDITIONAL":
        status["verdict"] = "CONDITIONAL"
    else:
        status["verdict"] = status.get("verdict") or "REJECT_WAITING"
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    cands = []
    for c in status.get("candidates") or []:
        nested = c.get("nested_oof_auc")
        if isinstance(nested, (int, float)) and nested >= TARGET:
            cands.append(c)
    if not cands:
        print(json.dumps({"error": "no candidate >= 0.715"}, indent=2))
        return 2
    # Prefer non-reference bootstrap
    cands.sort(key=lambda c: (bool(c.get("reference_bootstrap")), -float(c["nested_oof_auc"])))
    best = cands[0]
    metrics_path = Path(best["path"])
    result = audit_candidate(metrics_path)
    write_opinion(result)
    update_status(result)
    print(json.dumps({k: result[k] for k in ["verdict", "deliver_0_715_allowed", "nested_oof_auc", "hard_fail", "experiment_id"]}, indent=2))
    return 0 if result["deliver_0_715_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
