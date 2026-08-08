# B6pro 状态

## 结论（当前）

- B7 保底：本地 **0.702704955** / 公开 **0.707**
- **新 closest（诚实 nested）**：**0.70472075**
  - 配方：`max(gap, gap_bag, plus, region_meanL)`
  - `region_meanL`：短暴露=max3；弱区域长暴露=`long_only`；其余长暴露=`0.5*(max3+long_only)`
  - 弱区域（切片审计预注册）：`908d,f09d,9685,fafc,f167,ab86`
  - 相对 B7 **+0.00202**；距 0.71 缺口 ≈ **0.00528**
- 产物：`artifacts/b6pro_long_region/`、`submissions/b6pro_closest/`

## 继续方向

aging / multi-threshold / 更强 long 专臂，目标把 long 切片 AUC 从 ~0.66 抬向 0.70。
