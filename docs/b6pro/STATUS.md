# B6pro 状态

## 结论（当前）

- B7 保底：本地 **0.702704955** / 公开 **0.707**
- **新 closest**：**0.70547643**（nodays keepx mean + closest 嵌套）
- 相对 B7 **+0.00277**；距 0.71 缺口 ≈ **0.00452**
- 产物：`artifacts/b6pro_long_best/`、`submissions/b6pro_closest/`

## 本轮

- LGBM/resid_corr/plus_gap2/纯KNN：未超
- nodays(drop raw days, keepx full)：solo 0.6963，融合抬到 **0.705476**（微升）
- 主杠杆仍是 long 内排序（0.668→0.68 可过门）与/或 short 抬升

## 协议

nested < B7 则 fallback；未达 0.71 不宣称 PASS。
