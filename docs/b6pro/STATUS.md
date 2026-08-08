# B6pro 状态

## 结论（当前）

- B7 保底：本地 **0.702704955** / 公开 **0.707**
- **新 closest**：**0.70544811**（mean(aging,gap,keepx) + region blend wo=0.15）
- 相对 B7 **+0.00274**；距 0.71 缺口 ≈ **0.00455**
- 产物：`artifacts/b6pro_long_best/`、`submissions/b6pro_closest/`

## 已尝试未破 0.71

同质 CatBoost 变体、门控、MLP/EBM/FLAML、多阈值 5k/7k/10k、8seed aging、RSM 等。
主杠杆仍是提升 long 切片内排序（当前 best long AUC 仍远低于 0.70）。
