# B6pro 状态

## 结论（当前）

- B7 保底：本地 **0.702704955** / 公开 **0.707**
- **新 closest**：**0.70782351**（per-region nested helper+α pick_all / b6pro_region_pick）
- 距 0.71 缺口 ≈ **0.00218**
- 产物：`artifacts/b6pro_long_best/`、`submissions/b6pro_closest/`

## 抬升轨迹

| 配方 | nested |
|---|---:|
| B7 max3 | 0.702705 |
| nest_stack | 0.705734 |
| nest_div EBM+FLAML | 0.706020 |
| seq_patch 弱区 lm | 0.706808 |
| region_blend lm_all | 0.707596 |
| **region_pick helper+α** | **0.707824** |

## 进行中

- `b6pro_weak_weight` / `b6pro_iso_resid` / `b6pro_lgb_weakw`

## 协议

未达 0.71 不宣称 PASS；nested < B7 则 fallback。
