# B6pro 状态

## 结论（当前）

- B7 保底：本地 **0.702704955** / 公开 **0.707**
- **新 closest（诚实 nested）**：**0.70543543**
  - 配方：`max(gap, gap_bag, plus, region_meanL)`
  - long 专臂 = mean(aging, gap, keepx) long-only OOF
  - 弱区域长暴露：100% 专臂；其余长暴露：0.2 专臂 + 0.8 max3
  - 弱区域预注册：`908d,f09d,9685,fafc,f167,ab86`
  - 相对 B7 **+0.00273**；距 0.71 缺口 ≈ **0.00456**
- 产物：`artifacts/b6pro_long_region_keepx/`、`submissions/b6pro_closest/`

## 进行中

FLAML 异构臂、8seed/多阈值 long_multi。
