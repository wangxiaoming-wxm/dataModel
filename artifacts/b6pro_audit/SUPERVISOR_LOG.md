# B6pro Independent Supervisor Log

Protocol: `IA-AUC715-B6PRO-v1` · Target nested_oof_auc ≥ 0.715  
Role: audit-only (no training/tuning). Scores from on-disk artifacts only.

| cycle | utc | closest_exp | nested_oof_auc | recomputed_ok | ref_bootstrap | selected_rule | verdict | notes |
|------:|-----|-------------|---------------:|:-------------:|:-------------:|---------------|---------|-------|
| 1 | 2026-08-07T10:27:37Z | b6pro_fuse_plus_h2 (fuse0_b5ref) | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | gap=0.013685; ref bootstrap not eligible for final PASS; shuffled_plus_max_auc=0.6466 outside [0.47,0.53] hard band if treated as label-shuffle gate; train_b6pro b6_arms → b6pro_main in flight |
| 2 | 2026-08-07T10:30:32Z | b6pro_fuse_plus_h2 (fuse0_b5ref) | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | no new candidates; b6pro_main still empty (b6_arms training); deliver_0_715_allowed=false |

