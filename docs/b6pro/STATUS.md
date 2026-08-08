# B6pro 状态

## 结论（当前）✅ GATE PASS

- B7 保底：本地 **0.702704955** / 公开 **0.707**（未降级）
- **诚实 closest**：**0.71007148** ≥ **0.71**
- 配方：`pick×blend3` → regime-HGB ultra patch → full-x HGB ultra patch → MLP ultra patch → **nodays HGB(seed=2027) ultra nested-α patch**
- ultra ≈ **0.6452**（原 closest 0.631）；long ≈ **0.6736**
- 产物：`artifacts/b6pro_long_best/`、`artifacts/b6pro_honest_blend/`、`artifacts/b6pro_nodays_ultra/`、`submissions/b6pro_closest/`

## 抬升轨迹

| 配方 | nested |
|---|---:|
| B7 max3 | 0.702705 |
| nest_div | 0.706020 |
| region_pick / blend | 0.707259–0.707824 |
| honest pick×blend3 | 0.708901 |
| regime-HGB ultra patch | 0.709682 |
| + full-x HGB ultra patch | 0.709756 |
| + MLP ultra patch | 0.709765 |
| **nodays HGB s2027 ultra patch** | **0.710071** |

## 业务关键洞见

- ultra（days≥10k）内 **days–label 相关为负**（−0.037）；全局 days 单调伤害 ultra 排序
- **去掉 raw days** 的异构 HGB（保留 is_ultra/is_long + condition/x embedding）corr≈0.59，对 ultra 做外层嵌套 α patch 是破 0.71 的最后一跳
- 重 sample-weight / 窄切片专模 / 同构 CatBoost 堆叠收益有限

## 协议

- SKF=5、折内 FE、无全局 TE、无 OOF 连续搜权、无测集伪标签
- α 为 ultra 子集上外层嵌套选择（median α≈0.30）
- nested ≥ 0.71 且 > B7 → 可宣称 PASS；B7 仍为交付保底参考
