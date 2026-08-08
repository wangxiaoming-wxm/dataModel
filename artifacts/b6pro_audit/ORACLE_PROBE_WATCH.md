# Watch note — oracle / OOF transform probe

Detected at 2026-08-07T11:43:38.273179+00:00 in a parallel shell (not supervisor).

Observed diagnostic code that:
1. Computes label-conditioned `oracle` blend of main/plus (`np.where(y==1, max, min)`) and prints AUC ceiling.
2. Sweeps plus transforms (rank/sq/sqrt/stretch) via `nested_select_rule` on full OOF.
3. Sweeps delta thresholds for `np.where(delta>t, plus, main)`.

**Policy:** Diagnostic probes alone are not a FAIL. Any **delivered** metrics/predictions that use label-conditioned selection, continuous OOF weight/threshold search, or post-hoc non-preregistered transforms as the reported nested score → **FAIL** (cheating / OOF search).

Current status: no such packet has been authorized. deliver_0_715_allowed remains false.
