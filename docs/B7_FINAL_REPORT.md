# B7 最终报告（冲诚实 nested ≥ 0.71）

> 日期：2026-08-07  
> 分支：`cursor/b7-push-auc071-a5f5`  
> **门禁：未达标**（`nested_oof_auc ≥ 0.71` → **REJECT**）  
> **Closest honest：** `max(gap, gap_bag, plus_v10)` = **0.702704955**（距 0.71 ≈ **0.007295**）  
> B6 冻结：**未改动**（独立监督冻结核验 PASS）

---

## 1. 结论

在严格诚实口径（折内 FE、无全局 TE、预注册离散融合、嵌套选规则、shuffled 崩盘）下，**未能达到 0.71**。

权威交付分：

| 口径 | 值 | 用途 |
|---|---:|---|
| **Closest honest nested** | **0.702704955** | 主报（三臂 `max`；5/5 嵌套一致） |
| Pair fuse0 `max(B6_equal, plus)` | 0.702209316 | 协议最小二臂口径 |
| B6 frozen equal | 0.698974696 | 冻结基线 |
| V10 nested | 0.701314965 | 参考 |
| 距 0.71 | **0.007295** | — |

提交：`submissions/submission_b7_closest_honest.csv`

**公开榜 AUC：0.70722**（同一文件；相对本地 closest ≈ +0.0045）

---

## 2. 配方（closest）

```text
Arm_gap     = B6 gap CatBoost 8seed（冻结）
Arm_gap_bag = B6 gap_bag（bagging_temperature=1.0）8seed（冻结）
Arm_plus    = V10 plus H2 10fold×4seed（reference/v10，保留 x0–x18）

B7_closest = elementwise_max(gap, gap_bag, plus)
```

- 嵌套 `StratifiedKFold(5, shuffle=True, random_state=42)` 在 max 族候选中 **5/5 选 max3**
- `shuffled_plus → max3` AUC ≈ **0.644**（PASS，增益依赖真实 plus 排序）
- 二臂 `max(equal, plus)` = 0.702209 仍可作为更保守的副报

---

## 3. 已挖掘 / 已否定路径

| 路径 | nested / 关键 | 结论 |
|---|---:|---|
| residual corrector | 0.6971 | 负；叠层乐观偏置风险 |
| soft gate（学何时信 plus） | 0.702209 | 凸组合无法超过 max |
| nested logistic stack | 0.6969 | 负 |
| plus_mine（+gap/FN 交叉）10×4 | solo 0.686；max 0.6997 | 弱于 V10 plus，同质 |
| plus H3 / bag 集成 | 0.7017 | 未超 fuse0 |
| plus PCA+groupstats | 0.6836 | 负 |
| hybrid gap+x0–18 | 0.7017 | corr(B6)≈0.987，无新信息 |
| LGB on gap | 0.6999 | 臂弱 |
| EBM | 0.6951 | 臂弱 |
| XGB hetero | 0.6984 | 臂弱 |
| Balanced / midband 加权 | ≤0.688 | 破坏主排序 |
| 残差 TE 校正 | 0.686 | 负 |
| RSM/Langevin 多样性 | max3≈0.7023 | 无增益 |

**误差结构：** 阈值 0.5 下错误几乎全是 FN；「每行选更优臂」魔术上界 ≈0.76，但可学习门控未能兑现。

**敏感性：** 在 corr≈0.92 下，仅把异构臂从 0.689 提到 0.695 通常不够到 0.71；需要 **近强度且真正互补** 的新信号（本地缺口约 +0.007–0.008）。

---

## 4. 协议与监督

- 协议：`docs/supervision/B7_AUDIT_PROTOCOL.md`（IA-AUC710-B7-v1）
- B6 冻结：`artifacts/b7_audit/b6_freeze_check.json` → PASS
- 实验日志：`docs/B7_EXPERIMENT_LOG.md`
- 终审意见：`docs/supervision/B7_FINAL_AUDIT_OPINION.md`（REJECT-0.71 / certify closest）

---

## 5. 公开榜

| 文件 | 公开榜 AUC |
|---|---:|
| `submissions/submission_b7_closest_honest.csv` | **0.70722** |
| V10（历史参考） | 0.70570 |

- **本地诚实主报仍为 0.7027**；公开 0.70722 分列披露，勿把公开榜当 CV
- B7 closest 公开高于 V10 历史公开 0.70570
- **不要**用未过门禁的 0.71 叙事包装

---

## 6. 若继续冲 0.71（后续方向）

1. 寻找 **corr≲0.90 且 solo≳0.695** 的第三/第四异构源（表神经网络 / 不同损失排序目标）
2. 真正嵌套重算 stage1 的残差栈（避免 OOF 叠层泄漏）
3. 避免再堆与 B6 corr>0.98 的 CatBoost 同质变体
