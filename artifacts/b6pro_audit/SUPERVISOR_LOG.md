# B6pro Independent Supervisor Log

Protocol: `IA-AUC715-B6PRO-v1` · Target nested_oof_auc ≥ 0.715  
Role: audit-only (no training/tuning). Scores from on-disk artifacts only.

| cycle | utc | closest_exp | nested_oof_auc | recomputed_ok | ref_bootstrap | selected_rule | verdict | notes |
|------:|-----|-------------|---------------:|:-------------:|:-------------:|---------------|---------|-------|
| 1 | 2026-08-07T10:27:37Z | b6pro_fuse_plus_h2 (fuse0_b5ref) | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | gap=0.013685; ref bootstrap not eligible for final PASS; shuffled_plus_max_auc=0.6466 outside [0.47,0.53] hard band if treated as label-shuffle gate; train_b6pro b6_arms → b6pro_main in flight |
| 2 | 2026-08-07T10:30:32Z | b6pro_fuse_plus_h2 (fuse0_b5ref) | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | no new candidates; b6pro_main still empty (b6_arms training); deliver_0_715_allowed=false |
| 3 | 2026-08-07T10:33:51Z | b6pro_fuse_plus_h2 (fuse0_b5ref) | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | still only fuse0; b6_arms train ~6min CPU-hot; no PASS packet; branch=cursor/b6pro-auc0715-100c |
| 4 | 2026-08-07T10:37:03Z | b6pro_fuse_plus_h2 (fuse0_b5ref) | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | gap seeds 2026–2028 mid-train (~0.69 OOF/seed); no new metrics.json; REJECT_WAITING holds |
| 5 | 2026-08-07T10:40:15Z | b6pro_fuse_plus_h2 (fuse0_b5ref) | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | gap seed 2029 in progress; closest still fuse0@0.7013; deliver denied |
| 6 | 2026-08-07T10:43:32Z | b6pro_fuse_plus_h2 (fuse0_b5ref) | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | gap seed 2030; b6_frozen integrity OK; still REJECT_WAITING |
| 7 | 2026-08-07T10:46:46Z | b6pro_fuse_plus_h2 (fuse0_b5ref) | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | gap 2030 done; train still running (likely 2031+); no ≥0.715 candidate |

| 8 | 2026-08-07T10:50:44Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap seed=2031 OOF=0.690211 |
| 9 | 2026-08-07T10:53:15Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap seed=2032 fold=2 auc=0.67982 best=141 n=181 |
| 10 | 2026-08-07T10:55:47Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap seed=2033 fold=2 auc=0.70667 best=353 n=181 |
| 11 | 2026-08-07T10:56:12Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap seed=2033 fold=2 auc=0.70667 best=353 n=181 |
| 12 | 2026-08-07T10:56:28Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap seed=2033 fold=3 auc=0.68959 best=360 n=181 |
| 13 | 2026-08-07T10:58:18Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap_bag seed=2026 fold=0 auc=0.70541 best=525 n=181 |
| 14 | 2026-08-07T11:00:49Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap_bag seed=2026 OOF=0.693119 |
| 15 | 2026-08-07T11:03:20Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap_bag seed=2027 fold=2 auc=0.70326 best=422 n=181 |
| 16 | 2026-08-07T11:05:51Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap_bag seed=2028 fold=0 auc=0.66800 best=709 n=181 |
| 17 | 2026-08-07T11:06:34Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap_bag seed=2028 fold=1 auc=0.71485 best=799 n=181 |
| 18 | 2026-08-07T11:08:22Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap_bag seed=2029 fold=0 auc=0.69406 best=289 n=181 |
| 19 | 2026-08-07T11:10:53Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap_bag seed=2029 fold=3 auc=0.69187 best=466 n=181 |
| 20 | 2026-08-07T11:13:24Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap_bag seed=2030 fold=2 auc=0.68693 best=521 n=181 |
| 21 | 2026-08-07T11:15:55Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap_bag seed=2031 fold=0 auc=0.69960 best=379 n=181 |
| 22 | 2026-08-07T11:18:26Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap_bag seed=2031 OOF=0.690849 |
| 23 | 2026-08-07T11:20:57Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap_bag seed=2032 fold=3 auc=0.68197 best=346 n=181 |
| 24 | 2026-08-07T11:22:24Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap_bag seed=2033 fold=0 auc=0.70734 best=496 n=181 |
| 25 | 2026-08-07T11:23:28Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=gap_bag seed=2033 fold=2 auc=0.70018 best=273 n=181 |
| 26 | 2026-08-07T11:25:59Z | b6pro_fuse_plus_h2 | 0.7013149650619108 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=plus seed=2027 fold=1 auc=0.68021 best=583 n=80 |
| 27 | 2026-08-07T11:27:39Z | b6pro_multifuse_equal_b6_plus_ref | 0.7022093156561012 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress=plus seed=2029 fold=0 auc=0.68345 best=305 n=80 |

### Yellow flag @ 2026-08-07T11:27:59Z
- `b6pro_fuse_main_refplus` nested=0.7022093156561012 (recompute OK) but `reference_plus_bootstrap=true` → ineligible for final PASS.
- Fusion `rules_used` includes geom_mean/min/median/mean_3_1_1 beyond preregistered set in B6PRO_AUDIT_THRESHOLDS.json — if a future ≥0.715 packet relies on non-preregistered rules, treat as FAIL (post-hoc rule expansion).
- Selected rule here is still `max` (preregistered); score remains <0.715 → REJECT_WAITING.
- Self-trained `b6pro_main` equal_prob=0.6989746962571622 matches B6 baseline exactly; nested_oof_auc=null (arms only).
| 28 | 2026-08-07T11:28:30Z | b6pro_multifuse_equal_b6_plus_ref | 0.7022093156561012 | yes | **true** | max | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus_gap config=h2 === |
| 29 | 2026-08-07T11:31:01Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress=plus_gap seed=2027 fold=0 auc=0.66474 best=478 n=99 |
| 30 | 2026-08-07T11:33:32Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress=plus_gap seed=2028 fold=2 auc=0.67691 best=429 n=99 |
| 31 | 2026-08-07T11:36:03Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress=plus_gap seed=2029 fold=3 auc=0.67602 best=657 n=99 |
| 32 | 2026-08-07T11:38:35Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress=plus seed=2026 fold=3 auc=0.67457 best=1516 n=80 |
| 33 | 2026-08-07T11:41:06Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 34 | 2026-08-07T11:43:06Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 35 | 2026-08-07T11:43:37Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |

### Closest classes @ 2026-08-07T11:44:01Z
- best_self_trained: {'dir': 'b6pro_fuse_plusgap', 'nested': 0.7000905695422023, 'ref_decl': False, 'uses_ref_arm': False, 'expanded_rules': False, 'rule': 'max', 'recompute_ok': True}
- best_ref_or_mixed: {'dir': 'b6pro_fuse_main_refplus', 'nested': 0.7022093156561012, 'ref_decl': True, 'uses_ref_arm': True, 'expanded_rules': True, 'rule': 'max', 'recompute_ok': True}
- all_candidates: [
  {
    "dir": "b6pro_fuse0_b5ref",
    "nested": 0.7013149650619108,
    "ref_decl": true,
    "uses_ref_arm": true,
    "expanded_rules": false,
    "rule": "max",
    "recompute_ok": true
  },
  {
    "dir": "b6pro_fuse3_ref_self",
    "nested": 0.7009217772130575,
    "ref_decl": null,
    "uses_ref_arm": true,
    "expanded_rules": true,
    "rule": "max",
    "recompute_ok": true
  },
  {
    "dir": "b6pro_fuse_main_refplus",
    "nested": 0.7022093156561012,
    "ref_decl": true,
    "uses_ref_arm": true,
    "expanded_rules": true,
    "rule": "max",
    "recompute_ok": true
  },
  {
    "dir": "b6pro_fuse_plusgap",
    "nested": 0.7000905695422023,
    "ref_decl": false,
    "uses_ref_arm": false,
    "expanded_rules": false,
    "rule": "max",
    "recompute_ok": true
  },
  {
    "dir": "b6pro_fuse_self",
    "nested": 0.7004410650126306,
    "ref_decl": null,
    "uses_ref_arm": false,
    "expanded_rules": true,
    "rule": "max",
    "recompute_ok": true
  }
]
- deliver_0_715_allowed=false (all < 0.715)
- main b6pro_loop pane appears idle; plus_h2_10f training in train2
| 36 | 2026-08-07T11:46:08Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 37 | 2026-08-07T11:48:39Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 38 | 2026-08-07T11:51:10Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 39 | 2026-08-07T11:53:41Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 40 | 2026-08-07T11:56:12Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 41 | 2026-08-07T11:58:43Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 42 | 2026-08-07T11:59:13Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 43 | 2026-08-07T12:01:14Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 44 | 2026-08-07T12:03:45Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 45 | 2026-08-07T12:06:16Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 46 | 2026-08-07T12:08:48Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 47 | 2026-08-07T12:11:19Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 48 | 2026-08-07T12:13:50Z | b6pro_multifuse_equal_b6_plus_ref_plus5 | 0.7009217772130575 | yes | n/a | max | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 49 | 2026-08-07T12:14:49Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |

### Protocol risk @ 2026-08-07T12:15:00Z
- `b6pro_fuse3_ultra` nested=0.7020793228371782 recomputed OK; selected_rule=**median**.
- `median` is **not** in B6PRO_AUDIT_THRESHOLDS preregistered rules (mean, mean_2_1, power2, power3, max, rank_mean).
- Packet also mixes `plus_ref` without `reference_plus_bootstrap=true` disclosure.
- If this score were ≥0.715, verdict would be **REJECT** (post-hoc rule expansion + undisclosed ref arm), not PASS.
- Current: still <0.715 → REJECT_WAITING; deliver_0_715_allowed=false.
| 50 | 2026-08-07T12:16:21Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 51 | 2026-08-07T12:18:52Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 52 | 2026-08-07T12:21:23Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 53 | 2026-08-07T12:23:54Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 54 | 2026-08-07T12:26:26Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 === |
| 55 | 2026-08-07T12:28:57Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress==== train plus variant=plus config=h3 xf=prob === |
| 56 | 2026-08-07T12:30:33Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=plus seed=2026 fold=0 auc=0.68671 best=896 n=80 xf=prob |
| 57 | 2026-08-07T12:31:29Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=plus seed=2026 fold=1 auc=0.67862 best=995 n=80 xf=prob |
| 58 | 2026-08-07T12:34:00Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=plus seed=2026 fold=3 auc=0.67457 best=1516 n=80 xf=prob |
| 59 | 2026-08-07T12:36:31Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=plus seed=2027 fold=1 auc=0.68070 best=986 n=80 xf=prob |
| 60 | 2026-08-07T12:39:02Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=plus seed=2027 fold=3 auc=0.68137 best=607 n=80 xf=prob |
| 61 | 2026-08-07T12:41:34Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=plus seed=2028 fold=1 auc=0.70152 best=708 n=80 xf=prob |
| 62 | 2026-08-07T12:44:05Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=plus seed=2028 OOF=0.678099 |
| 63 | 2026-08-07T12:45:48Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=plus seed=2029 fold=1 auc=0.69963 best=597 n=80 xf=prob |
| 64 | 2026-08-07T12:46:37Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=plus seed=2029 fold=2 auc=0.68106 best=744 n=80 xf=prob |
| 65 | 2026-08-07T12:49:08Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=xgb seed=2029 fold=1 auc=0.66518 best=65 |
| 66 | 2026-08-07T12:51:39Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
| 67 | 2026-08-07T12:54:11Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
| 68 | 2026-08-07T12:56:42Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
| 69 | 2026-08-07T12:59:13Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
| 70 | 2026-08-07T13:01:44Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
| 71 | 2026-08-07T13:04:15Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
| 72 | 2026-08-07T13:06:04Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
| 73 | 2026-08-07T13:06:46Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
| 74 | 2026-08-07T13:09:17Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
| 75 | 2026-08-07T13:11:48Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
| 76 | 2026-08-07T13:14:19Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
| 77 | 2026-08-07T13:16:50Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
| 78 | 2026-08-07T13:19:21Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
| 79 | 2026-08-07T13:21:53Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
| 80 | 2026-08-07T13:24:24Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
| 81 | 2026-08-07T13:26:19Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
| 82 | 2026-08-07T13:26:55Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
| 83 | 2026-08-07T13:29:26Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=$ /usr/bin/python3 scripts/b6pro_fuse_npzs.py --arms equal_b6=artifacts/b6pro_main/predictions.npz:oof_main:test_main pl |
| 84 | 2026-08-07T13:31:57Z | b6pro_multifuse_equal_b6_plus_ref_ultra | 0.7020793228371782 | yes | n/a | median | REJECT_WAITING | below 0.715; frozen OK; progress=} |
