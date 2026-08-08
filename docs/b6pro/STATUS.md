# B6pro 状态

## 结论（当前）

- B7 保底：本地 **0.702704955** / 公开 **0.707**
- **新 closest（诚实 nested）**：**0.70509254**
  - 配方：`max(gap, gap_bag, plus, region_meanL_aging)`
  - 长暴露专臂：aging 特征 + days≥3000 only 训练（4 seed）
  - 弱区域长暴露用 long_only；其余长暴露 0.5 混合；短暴露 max3
  - 弱区域预注册：`908d,f09d,9685,fafc,f167,ab86`
  - 相对 B7 **+0.00239**；距 0.71 缺口 ≈ **0.00491**
- 产物：`artifacts/b6pro_long_region_aging/`、`submissions/b6pro_closest/`

## 继续方向

多阈值/keepx/更强异构臂；目标 long 切片 AUC↑。
