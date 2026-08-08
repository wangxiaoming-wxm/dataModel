# B6pro Supervisor Rejects

## 2026-08-08T10:20:41Z

- Branch checked: `cursor/b6pro-auc0715-100c`
- Current only approved submission: `submissions/b6pro_closest/submission_SUBMIT_THIS.csv`
- Approved B7 SHA256: `5c9ccfdaaed914c92e153cb9ba2b0fb4e066b462b09406184d3f4ed1e75c1de8`
- Protocol checked: `artifacts/b6pro_audit/ANTI_OVERFIT_PROTOCOL.json` is active and keeps `direct_nodays_ultra_patch_s2027` on the public-LB denylist.
- STATUS checked: `docs/b6pro/STATUS.md` no longer claims GATE PASS; it points to B7 fallback and states the 0.710071 local run is invalidated.

### REJECT / DO NOT PROMOTE

1. `direct_nodays_ultra_patch_s2027`
   - Evidence: `artifacts/b6pro_long_best/metrics.json`
   - Local nested OOF: `0.7100714803766324`
   - Reason: public LB `0.70208` is below verified B7 public `0.707`; candidate depends on single seed 2027 and ultra small-slice alpha search.
   - Required action: keep quarantined; do not promote or submit.

2. Historical repackages of the same invalidated 0.710071 chain
   - Evidence: `artifacts/b6pro_nodays_ultra/metrics.json`, `artifacts/b6pro_honest_blend/metrics.json`
   - Current state observed: `gate=false` with quarantine notes.
   - Reason: these artifacts remain local >0.71 candidates tied to the invalidated single-seed / ultra-alpha chain.
   - Required action: reject promote unless a future candidate passes the full anti-overfit protocol with multi-seed, repeated outer CV, B7-relative stability, and auditor PASS.

### submission_b6pro.csv warning check

- Found many non-`SUBMIT_THIS` files named `submission_b6pro.csv`, including `submissions/b6pro_closest/submission_b6pro.csv` with a different SHA256 from B7.
- No current docs/audit recommendation was found that treats a non-`SUBMIT_THIS` `submission_b6pro.csv` as the recommended submission.
- Warning remains active: any future recommendation of `submission_b6pro.csv` instead of `submissions/b6pro_closest/submission_SUBMIT_THIS.csv` must be rejected immediately.
