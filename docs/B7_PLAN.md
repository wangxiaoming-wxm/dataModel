# B7 计划（冲诚实本地 OOF ≥ 0.71）

## 0. 硬约束

| 项 | 约定 |
|---|---|
| B6 | **冻结不动**（`docs/b6_frozen/`、`submissions/b6_frozen/`、`artifacts/b6_frozen/`） |
| B5 | 继续冻结（既有 `*/b5_frozen/`） |
| 数据 | 仅新 `train.csv` / `test.csv` |
| 协议 | SKF≥5；seeds≥8（主臂）；折内 FE；**无外置/全局 TE**；禁止连续 OOF 搜权 |
| 融合 | **预注册离散规则** + **嵌套折选规则**（继承 V10 诚实口径） |
| 目标 | 诚实本地 **≥0.71**；否则报告 closest honest |

基线：

| 来源 | 口径 | 值 |
|---|---|---:|
| B6 | equal_prob(gap,gap_bag) | 0.698975 |
| V10 | nested max(B5-8,plus) | 0.701315 |
| V10 | max(B5-12,plus) | 0.701751 |
| V10 | 公开榜 | 0.70570 |
| 快检 | max(B6, plus_v10) | **≈0.70221** |

缺口：`0.71 − 0.70221 ≈ 0.0078`。

---

## 1. 主路径

```text
Arm_A = B6-class strong arm (gap / gap_bag；可 8–12 seed)
Arm_B = V10 plus (keep x0–x18 root_plus；H2；可增强)
Arm_C = 新挖掘残差臂（可选；须近强度且与 A/B 相关更低）

B7 = nested_select( pre-registered rules on (A,B[,C]) )
```

预注册规则（开跑前写死）：`mean, mean_2_1, power2, power3, max, rank_mean`  
嵌套：`StratifiedKFold(5, shuffle=True, random_state=42)` 在 train 段选规则 → 主报 **nested OOF**；提交用全量选定规则。

---

## 2. 正向继承

**来自 B6：** gap 猫交叉（ratio×geo、t3_sfx×code×days、w_pair、days_fixed…）；`gap_bag`（bagging_temperature=1.0）。  
**来自 V10：** plus 异构（保留 latent x0–x18、root 交叉、w_conflict/t3_num_z/x20_grid）；**max 嵌套融合**（过 0.70 的关键杠杆）；plus 与 B5/B6 corr≈0.91–0.92。

---

## 3. 新挖掘方向（冲 +0.008）

1. **残差切片**：在 `max(B6,plus)` 仍错分 / midband 上找未吃满交叉  
2. **plus 增强**：折内修正 `t3_num_z`；注入 B6 健康缺口猫（不破坏异构）  
3. **第三臂**：Bernoulli/RSM 或 LGBM-on-plus-numeric，目标 corr(A)<0.93 且 OOF≥0.685  
4. **主臂加种**：gap_bag 扩到 12 seed（边际小，作配套）

禁止：高 gap TE、`t3_full` 稀疏三阶、OOF 连续搜权、测集伪标签。

---

## 4. 独立监督

协议：`docs/supervision/B7_AUDIT_PROTOCOL.md`（IA-AUC710-B7-v1）  
仅当 nested≥0.71 且红线全过 → PASS；否则 closest honest。
