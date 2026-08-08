# B6pro 状态

## 结论（当前）

- B7 保底：本地 **0.702704955** / 公开 **0.707**
- **新 closest**：**0.70573437**（direct_logit_gap+gap_bag+plus+kx8+cur_C3.0 / b6pro_nest_stack）
- 距 0.71 缺口 ≈ **0.00427**
- 产物：`artifacts/b6pro_long_best/`、`submissions/b6pro_closest/`

## 本轮

- 嵌套 logit stack（gap/gap_bag/plus/kx8/cur）抬到 **0.70573**
- DAE/hardw/XGB/f09d 等未超
- 主缺口仍约 0.0043；继续异构与业务残差

## 协议

未达 0.71 不宣称 PASS；nested < B7 则 fallback。
