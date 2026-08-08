# B6pro 状态

## 结论（当前）

- B7 保底：本地 **0.702704955** / 公开 **0.707**
- **诚实 closest**：**0.70890082**（honest nested α pick+blend3）
- 距 0.71 缺口 ≈ **0.00110**
- long ≈ 0.6716；f09d-long 仍约 0.607
- 产物：`artifacts/b6pro_long_best/`、`submissions/b6pro_closest/`

## 抬升轨迹

| 配方 | nested |
|---|---:|
| B7 max3 | 0.702705 |
| nest_div | 0.706020 |
| region_pick | 0.707824 |
| post+pick logit | 0.708285 |
| **honest pick×blend3** | **0.708901** |

## 进行中

- `b6pro_weak_weight` / `b6pro_iso_resid`（CatBoost）

## 协议

未达 0.71 不宣称 PASS；nested < B7 则 fallback。α/权使用外层嵌套选择。
