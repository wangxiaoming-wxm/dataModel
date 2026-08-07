# 车险索赔 AUC · B6pro 独立复核协议（IA-AUC715-B6PRO-v1）

> **角色**：独立监督者 / 复核官。**不参与**写模型、调参、抬分或提交包装。  
> **监督对象**：分支 `cursor/b6pro-auc0715-100c` 冲刺诚实本地 OOF **≥ 0.715**。  
> **基线**：B6 closest honest pooled **0.69897470**（`equal_prob(gap,gap_bag)`）；B6 对 0.70 为 REJECT。  
> **用户硬约束**：不过拟合、不作弊；不达 **0.715** 不得停止宣称达标；未达标只认证 closest honest。  
> **继承**：IA-AUC698-v1 / IA-AUC700-B6-v1 全部红线；冲突时以更严为准。

---

## 0. 冻结与数据

| 冻结 | 要求 |
|---|---|
| B5 / B6 冻结树 | 不得篡改 `*/b5_frozen/`、`*/b6_frozen/` |
| 其他分支 | 不得改写 `cursor/b6-push-*`、`cursor/b7-*`、`cursor/claim-*` 等远程分支内容 |
| 数据 SHA | 与 IA-AUC698-v1 一致（train/test/submit） |

---

## 1. 主门槛（宣称 ≥ 0.715）

| # | 门槛 | 判定 |
|---|---|---|
| A | **nested_oof_auc ≥ 0.715**（或预注册的 pooled 主口径且 ≥0.715） | 主报告分 |
| B | CV：SKF≥5；主臂 seeds≥8 等权 | 缺一 FAIL |
| C | shuffled ∈ [0.47, 0.53]（推荐 [0.48,0.52]）；max 融合须另报 shuffle-collapse | 带外 FAIL |
| D | 协议声明全 true；折内 FE；无全局 TE；无 OOF 连续搜权 | 任一 false FAIL |
| E | 融合规则预注册 + 嵌套选规则（若用离散规则集） | 看完 OOF 搜权 → FAIL |
| F | 非单折/单 seed 包装 | FAIL |

**硬失败**：测集标签/伪标签、第三方旧包冒充自研、全量 fit 再 OOF 编码、公开榜回流仍称盲测。

若用 **elementwise max** 且刚过线：默认最多 **CONDITIONAL**，须 shuffle-collapse 与固定 iter 对照。

---

## 2. 监督者工作方式

1. 扫描 `artifacts/b6pro_*/metrics.json`；**禁止编造分数**。  
2. 复算 `predictions.npz` 的 `roc_auc_score`（误差 < 1e-8）。  
3. 检查 B6 冻结完整性。  
4. nested < 0.715 → 维持 REJECT / WAITING，更新 closest honest。  
5. 仅当 A–F 全过才写 `B6PRO_FINAL_AUDIT_OPINION.md` 且 `deliver_0_715_allowed=true`。

状态文件：`artifacts/b6pro_audit/waiting_status.json`

---

## 3. 合格带阈值（机器可读）

见 `docs/supervision/B6PRO_AUDIT_THRESHOLDS.json`。
