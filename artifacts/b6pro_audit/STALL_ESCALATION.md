# Supervisor escalation — stall / retry storm

**Time:** 2026-08-07T13:26Z  
**Protocol:** IA-AUC715-B6PRO-v1  
**Branch:** `cursor/b6pro-auc0715-100c`

## Observation

- Closest nested on disk remains **0.7022093156561012** (`b6pro_fuse_main_refplus`, `reference_plus_bootstrap=true`) across many consecutive supervisor cycles (≥30+).
- Best **disclosed self-trained** clean packet still ~**0.70009–0.70089** (`fuse_plusgap` / `fuse_h3`).
- Gap to 0.715 ≈ **0.0128**.
- `scripts/b6pro_loop.py` (tee2) is in an extension **retry storm**: stages 15–18+ re-run identical fuse_npzs / fuse0 commands, sleep 120s, repeat. No new ≥0.715 artifact.
- `deliver_0_715_allowed` remains **false**. Verdict: **REJECT_WAITING**.

## Integrity

- `b6_frozen` hashes unchanged vs baseline.
- No authorized PASS packet.
- Watched risks (not yet FAIL-for-delivery, scores still <0.715):
  - non-preregistered fusion rules (`median`, `min`, extended set)
  - ref arms without `reference_plus_bootstrap` disclosure
  - diagnostic oracle-ceiling probe (see `ORACLE_PROBE_WATCH.md`)

## Action

Supervisor does **not** write training/tuning code. Continues audit loop; will run A–F deep audit + `B6PRO_FINAL_AUDIT_OPINION.md` only if an honest nested_oof_auc ≥ 0.715 appears on disk with complete packet.

**Escalation condition met:** no score progress across consecutive checkpoints; trainee loop stalled in identical retries.
