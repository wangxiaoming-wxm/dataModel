# B6pro 状态

## 结论（当前）

- B7 保底：本地 **0.702704955** / 公开 **0.707**
- **新 closest**：**0.70558281**（cur+rb_mix_m3_kx_w0.2 / quick_fuse_fullkx）
- 距 0.71 缺口 ≈ **0.00442**
- 产物：`artifacts/b6pro_long_best/`、`submissions/b6pro_closest/`

## 本轮要点

- 灵敏度：long 0.668→0.675 或 f09d-long 0.60→0.65 可过 0.71
- f09d 区内单变量上限≈0.54，纯区域专模失败（0.52）
- full keepx/aging ≈0.696；nodays/fullkx 微抬 closest
- 窄切片专模（10k+/f09d-only）严重欠拟合，增益仍靠全局异构+融合

## 协议

未达 0.71 不宣称 PASS；nested < B7 则 fallback。
