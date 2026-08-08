# B7 实验日志（冲 nested ≥ 0.71）

## 当前权威地板

| 配方 | nested OOF | 备注 |
|---|---:|---|
| fuse0 `max(B6_equal, plus_v10)` | **0.702209** | 5/5 选 max；距 0.71 ≈ 0.0078 |
| disclosure `max3(gap,gap_bag,plus)` | 0.702705 | 非主报 nested 选三臂 |

B6 冻结未动（独立监督 PASS）。

## 已试路径（诚实口径）

| 实验 | nested / 关键分 | 结论 |
|---|---:|---|
| residual corrector（stage1 作特征） | 0.6971 | 负；stage2 弱且有 CV 叠层乐观偏置风险 |
| soft gate（学何时信 plus） | 0.702209 | 凸组合无法点式超过 max；无增益 |
| nested logistic stack | 0.6969 | 负 |
| XGB hetero | 0.6984 | 臂太弱（~0.655） |
| gap Balanced / plus Balanced | max≤0.688 | 负；破坏主排序 |
| plus_mine 5fold 1seed | solo 0.680 | 略好于 plus_base5，待 10×4 全量 |
| 校准 / disagree_max / softmax | ≤0.7017 | 均未超 fuse0 |

## 误差结构

- 阈值 0.5 下错误几乎全是 **FN**（漏检正例）
- magic「每行选更优臂」上界 ≈ **0.762** → 理论有空间，但门控未能兑现
- 敏感性：需 **近强度且互补** 的新臂；仅把 plus 提到 0.695（corr≈0.92）通常不够到 0.71

## 进行中

- `plus_mine` H2：**10fold × 4seed**（主攻更强异构 plus）
- `lgb_gap`：LightGBM on B6 gap FE（异构模型）

## 下一步

1. 吃满 plus_mine / LGB 结果并 nested 融合
2. 若仍 <0.71：继续挖第三近强度臂（corr≲0.93）或 FN 专用排序目标
3. 独立监督仅在 nested≥0.71 时 PASS；否则 closest honest


## 续：2026-08-07 下午迭代

| 实验 | nested / 关键 | 结论 |
|---|---:|---|
| plus_mine 10×4 | solo **0.68609** | 弱于 V10 plus 0.68862；corr(plus)=0.982 同质 |
| max(B6, plus_mine) | 0.69965 | 负于 fuse0 |
| mean(plus,plus_mine)→plus2 × B6 max | 0.70123 | 仍低于 fuse0 **0.70221** |
| lgb_gap 4seed | 0.69990 | 臂弱 ~0.668 |
| ebm 2seed | 0.69512 | 臂弱 ~0.644 |
| nested residual TE | corr_model 0.686 | 负 |
| gap_v2 1seed | 0.691 | 高相关，不抬 max |

**仍权威：** fuse0 nested max(B6, plus_v10) = **0.702209**（距 0.71 = 0.007791）

进行中：hybrid(gap+x0–18)、plus H3/bag 集成。

## 终局

- **REJECT 0.71**
- Closest honest nested **0.702704955** = max(gap, gap_bag, plus_v10)
- 提交：`submissions/submission_b7_closest_honest.csv`
- 终审：`docs/supervision/B7_FINAL_AUDIT_OPINION.md`
- 总报告：`docs/B7_FINAL_REPORT.md`
