# B6pro 状态

## 结论（当前）

- B7 保底：`max(gap,gap_bag,plus_v10)` 本地 **0.702704955** / 公开 **0.707**
- **新 closest（诚实 nested）**：**0.70419869**（`b7+meanL_w0.7`）
  - 配方：`elementwise_max(gap, gap_bag, plus, meanL)`  
    `meanL`：短暴露=max3；长暴露(`days≥3000`)=`0.7*long_only_gap + 0.3*max3`
  - 相对 B7 **+0.00149**；距 0.71 缺口 ≈ **0.00580**
- 产物：`artifacts/b6pro_long_blend/`、`artifacts/b6pro_long_only_gap/`
- days≥5k 专臂：未超过 B7（回退保底）

## 继续方向

1. aging builder long-only / 8seed+多阈值 multi
2. 寻找能把 long 切片 AUC 从 0.66 抬向 0.70 的异构源
