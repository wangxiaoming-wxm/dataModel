# B6pro 状态

## 结论（当前）

- B7 保底：`max(gap,gap_bag,plus_v10)` 本地 **0.702704955** / 公开 **0.707**
- **新 closest（诚实 nested）**：**0.70409783**（`b7+meanL`，long-only gap 专臂）
  - 配方：`elementwise_max(gap, gap_bag, plus, meanL)`  
    其中 `meanL` = 短暴露用 max3，长暴露(`days≥3000`)用 `0.5*(max3 + long_only_gap)`
  - 相对 B7 **+0.00139**；距 0.71 缺口 ≈ **0.00590**
- 产物：`artifacts/b6pro_long_only_gap/`

## 业务洞见（继续抬分）

- 长暴露占行数 ~66%、索赔 ~79%，但 max3 在该切片 AUC≈**0.663**（整体拖累主因）
- 灵敏度：若长暴露内排序提到 ~0.70（保持 max3 边际），整体可到 ~0.715
- 已验证：long-only 专臂 solo 弱于 max3，但与 B7 max 融合可稳定抬 nested

## 继续方向

1. 更多 seed / aging·keepx builder 的 long-only
2. 多阈值专臂（5k/7k/10k）与 B7 嵌套 max
3. 寻找 solo≳0.695 且 corr(max3)≲0.90 的真正异构源
