# B7 Closest-Honest Draft (NOT a final PASS)

**Protocol:** IA-AUC710-B7-v1  
**Status:** DRAFT / WAITING — **no nested_oof_auc ≥ 0.71** observed; **do not deliver 0.71**.  
**Auditor role:** independent supervisor only; scores taken from on-disk `artifacts/b7_*/metrics.json` (not invented).  
**Checked branch:** `cursor/b7-push-auc071-a5f5`  
**B6 freeze check:** `artifacts/b7_audit/b6_freeze_check.json` → **PASS** (pooled still **0.6989746962571622** vs `origin/cursor/b6-push-auc070-a5f5`).

---

## Provisional closest honest nested

| Field | Value |
|---|---|
| `closest_honest_nested_oof_auc` | **0.7022093156561012** |
| Canonical source | `artifacts/b7_fuse0_b6/metrics.json` (`b7_fuse0_nested`) |
| Tie (same nested) | `artifacts/b7_gate_soft/metrics.json` (`b7_gate_soft`) — nested equals stage1 max; no lift |
| `gate_0_71` | `false` |
| Gap to 0.71 | ≈ **0.007791** |
| Nested rule | `max` on all 5 folds |
| Non-max full-data peaks (fuse0) | best non-max `power3` ≈ 0.70025 (all non-max **< 0.71**) |
| `shuffled_plus_max_auc` (fuse0) | 0.6483687530800213 (`pass` under max-collapse `< 0.66`) |

### Observed nested ranking (disk metrics only)

| nested_oof_auc | path | experiment_id | note |
|---|---|---|---|
| **0.7022093156561012** | `b7_fuse0_b6` | `b7_fuse0_nested` | closest; max |
| 0.7022093156561012 | `b7_gate_soft` | `b7_gate_soft` | tie; max ≡ stage1 |
| 0.7021806550384172 | `b7_fuse0_gapbag` | `b7_fuse0_nested` | max |
| 0.6983995433408249 | `b7_xgb` | `b7_xgb_hetero` | mean_2_1 |
| 0.6970980726530736 | `b7_resid` | `b7_residual_corrector` | mean_2_1 |
| 0.6969451165093915 | `b7_stack` | `b7_nested_logistic_stack` | nested logistic |

Arm-only disclosures without nested main score (not authority):  
`b7_plus_s1` oof≈0.6790; `b7_plus_gap_s1` oof≈0.6809.

In progress / no metrics yet: `b7_plus_full`, `b7_eda` (mining/smoke only).

---

## Why not 0.71 (provisional REJECT-0.71 skeleton)

```text
verdict: REJECT   # provisional until main process signals closeout or a ≥0.71 packet appears
claimed_or_attempted: honest nested_oof_auc >= 0.71
closest_honest_nested_oof_auc: 0.7022093156561012
why_not_0.71: best observed nested_oof_auc is 0.7022093156561012 (< 0.71)
red_lines_hit: []   # no freeze tamper; score shortfall is the gate failure
b6_freeze_check: PASS
deliver_0_71_allowed: false
```

**Gate A FAIL** — no candidate meets `nested_oof_auc ≥ 0.71`. Final PASS opinion (`B7_FINAL_AUDIT_OPINION.md`) is **withheld**.

---

## Red-line checklist (so far; incomplete packet ≠ PASS)

| Check | Status |
|---|---|
| SKF / nested folds ≥ 5 | Observed nested uses 5 folds (fuse0 / gate / resid / xgb) |
| Seeds (main arm ≥ 8 where B6-class) | B6 equal arm inherits frozen 8-seed baseline; several B7 arms use 2–4 seeds — disclose-only |
| Fold-local FE / no global TE | Declared on several packets (`no_global_te` / `fold_local_fe`) — not independently re-proven this cycle |
| No continuous OOF weight search | Discrete V10 six-rule nested selection declared on fuse/gate/resid/xgb |
| Shuffled ≈ chance / max collapse | fuse0 reports `shuffled_plus_max_auc≈0.648` (pass); not present on all packets |
| B6 untampered | **PASS** (hashes + pooled 0.6989746962571622) |
| Packet complete for PASS (§5) | **NO** — missing e.g. full §5.5 keys / `protocol_id` / git binding on most candidates; max-only peak would be CONDITIONAL even if ≥0.71 |

---

## Auditor stance

1. Keep `waiting_status.json` until nested ≥ 0.71 **and** protocol packet is complete, **or** main process signals closest-honest closeout.  
2. On closeout below 0.71: promote this draft to a signed **REJECT-0.71 / certify closest honest** opinion.  
3. On nested ≥ 0.71 with complete packet: write `B7_FINAL_AUDIT_OPINION.md` only after re-running freeze check + red-line review (max dependency → CONDITIONAL / `deliver_0_71_allowed=false` by default).  
4. Do not invent scores; do not modify training code or B6 frozen trees.

---

*Draft updated after scan cycle observing resid/xgb/gate_soft. Not a delivery authorization.*
