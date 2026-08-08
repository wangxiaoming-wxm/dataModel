# B7 独立监督终审意见（IA-AUC710-B7-v1）

> 复核官：独立监督者（不参与训练抬分）  
> 对象：`cursor/b7-push-auc071-a5f5`  
> 时间：2026-08-07

---

## 裁决

| 项 | 结论 |
|---|---|
| **门禁 nested ≥ 0.71** | **REJECT** |
| **Closest honest nested** | **0.7027049552615718** |
| 配方 | `max(gap, gap_bag, plus_v10)` |
| 距 0.71 | **0.007295** |
| B6 冻结完整性 | **PASS** |
| shuffled_plus → max3 | **≈0.644 PASS** |
| 连续 OOF 搜权 | **未发现**（预注册 max 族 / 嵌套 5/5） |
| 全局 TE | **未发现** |

**不得宣称诚实本地 AUC ≥ 0.71。**  
允许交付并披露 **closest honest 0.702705**，并同时列出二臂 fuse0 **0.702209** 作为更保守副报。

---

## 证据摘要

1. `artifacts/b7_closest/metrics.json`：nested=pooled=0.702704955；gate_0_71=false  
2. `artifacts/b7_fuse0_b6/metrics.json`：pair nested max=0.702209316  
3. `artifacts/b7_audit/b6_freeze_check.json`：verdict PASS；B6 pooled 仍为 0.6989746962571622  
4. 阴性簇（resid/gate/plus_mine/lgb/ebm/hybrid/plus_h3）均 **未超过** closest，详见 `docs/B7_FINAL_REPORT.md`

---

## 签字

独立监督者：**REJECT-0.71**；认证 closest honest **0.702704955**。
