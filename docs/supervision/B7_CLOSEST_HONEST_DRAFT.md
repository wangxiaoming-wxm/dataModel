# B7 Closest-Honest Draft (NOT a final PASS)

**Protocol:** IA-AUC710-B7-v1  
**Status:** DRAFT / WAITING — **no nested_oof_auc ≥ 0.71** observed; **do not deliver 0.71**.  
**Auditor role:** independent supervisor only; scores taken from on-disk `artifacts/b7_*/metrics.json` (not invented).  
**Checked branch:** `cursor/b7-push-auc071-a5f5`  
**B6 freeze check:** `artifacts/b7_audit/b6_freeze_check.json` → **PASS** (pooled still **0.6989746962571622** vs `origin/cursor/b6-push-auc070-a5f5`).  
**Rescan:** after plus_mine / lgb / ebm (scan cycle recorded in `waiting_status.json`).

---

## Provisional closest honest nested

| Field | Value |
|---|---|
| `closest_honest_nested_oof_auc` | **0.7022093156561012** |
| Canonical source | `artifacts/b7_fuse0_b6/metrics.json` (`b7_fuse0_nested`) |
| Tie (same nested, no lift) | `artifacts/b7_gate_soft/metrics.json` — nested ≡ stage1 max |
| `gate_0_71` | `false` |
| Gap to 0.71 | ≈ **0.007791** |
| Nested rule | `max` on all 5 folds |
| Non-max full-data peaks (fuse0) | best non-max `power3` ≈ 0.70025 (all non-max **< 0.71**) |
| `shuffled_plus_max_auc` (fuse0) | 0.6483687530800213 (`pass` under max-collapse `< 0.66`) |

**Verdict on closest:** fuse0 **remains** the closest honest nested after plus_mine / lgb / ebm wave — no new candidate beat **0.7022093156561012**.

### Observed nested ranking (disk metrics only)

| nested_oof_auc | path | experiment_id | note |
|---|---|---|---|
| **0.7022093156561012** | `b7_fuse0_b6` | `b7_fuse0_nested` | **closest**; max |
| 0.7022093156561012 | `b7_gate_soft` | `b7_gate_soft` | tie; no lift vs stage1 |
| 0.7021806550384172 | `b7_fuse0_gapbag` | `b7_fuse0_nested` | max |
| 0.7012312223196153 | `b7_fuse_plusmine` | `b7_fuse_b6_plus2` | below fuse0 |
| 0.6998994987576419 | `b7_lgb_gap` | `b7_lgb_gap` | below fuse0 |
| 0.6983995433408249 | `b7_xgb` | `b7_xgb_hetero` | below fuse0 |
| 0.6970980726530736 | `b7_resid` | `b7_residual_corrector` | below fuse0 |
| 0.6969451165093915 | `b7_stack` | `b7_nested_logistic_stack` | below fuse0 |
| 0.6951199426946871 | `b7_ebm` | `b7_ebm` | below fuse0 |

### Major negatives vs fuse0 closest (this wave)

| Arm / run | Observed | vs closest |
|---|---|---|
| **resid** (`b7_residual_corrector`) | nested **0.6970980726530736** | −0.00511; stage2 alone 0.66821 |
| **gate_soft** | nested **0.7022093156561012** | tie only; stage2 0.69629 — no gain past fuse0 max |
| **plus_mine** (`b7_plus_mine_h2`) | arm oof **0.6860946843311607** (no nested main) | arm well below; fuse with B6 (`b7_fuse_plusmine`) nested **0.7012312223196153** still < fuse0 |
| **lgb** (`b7_lgb_gap`) | nested **0.6998994987576419** | −0.00231; stage2 0.66842 |
| **ebm** (`b7_ebm`) | nested **0.6951199426946871** | −0.00709; stage2 0.64424 |

Other arm-only disclosures (not nested authority): `b7_plus_s1` oof≈0.6790; `b7_plus_gap_s1` oof≈0.6809.

In progress / no metrics yet: `b7_eda` (npy screens only), `b7_plus_full`, `b7_plus_h3`, `b7_hybrid`.

---

## Why not 0.71 (provisional REJECT-0.71 skeleton)

```text
verdict: REJECT   # provisional until main process signals closeout or a ≥0.71 packet appears
claimed_or_attempted: honest nested_oof_auc >= 0.71
closest_honest_nested_oof_auc: 0.7022093156561012
why_not_0.71: best observed nested_oof_auc is 0.7022093156561012 (< 0.71); resid/gate/plus_mine/lgb/ebm did not improve
red_lines_hit: []   # no freeze tamper; score shortfall is the gate failure
b6_freeze_check: PASS
deliver_0_71_allowed: false
```

**Gate A FAIL** — no candidate meets `nested_oof_auc ≥ 0.71`. Final PASS opinion (`B7_FINAL_AUDIT_OPINION.md`) is **withheld**.

---

## Red-line checklist (so far; incomplete packet ≠ PASS)

| Check | Status |
|---|---|
| SKF / nested folds ≥ 5 | Observed on fuse0 / gate / resid / xgb / lgb / ebm / fuse_plusmine |
| Seeds (main arm ≥ 8 where B6-class) | B6 equal arm inherits frozen 8-seed baseline; several B7 arms use fewer seeds — disclose-only |
| Fold-local FE / no global TE | Declared on several packets — not independently re-proven this cycle |
| No continuous OOF weight search | Discrete nested rule selection declared on nested candidates |
| Shuffled ≈ chance / max collapse | fuse0 reports `shuffled_plus_max_auc≈0.648` (pass); not present on all packets |
| B6 untampered | **PASS** (hashes + pooled 0.6989746962571622) |
| Packet complete for PASS (§5) | **NO** — missing full §5.5 / `protocol_id` / git binding on most candidates |

---

## Auditor stance

1. Keep `waiting_status.json` until nested ≥ 0.71 **and** protocol packet is complete, **or** main process signals closest-honest closeout.  
2. On closeout below 0.71: promote this draft to a signed **REJECT-0.71 / certify closest honest** opinion (fuse0 **0.7022093156561012**).  
3. Do not invent scores; do not modify training code or B6 frozen trees.

---

*Draft updated after plus_mine / lgb / ebm rescan. fuse0 remains closest. Not a delivery authorization.*
