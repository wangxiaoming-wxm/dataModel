# B6pro 状态

## 结论（当前）

- B7 保底：本地 **0.702704955** / 公开 **0.707**
- **新 closest**：**0.70828452**（logit post+pick C=0.05）
- 距 0.71 缺口 ≈ **0.00172**
- 产物：`artifacts/b6pro_long_best/`、`submissions/b6pro_closest/`

## 抬升轨迹

| 配方 | nested |
|---|---:|
| B7 max3 | 0.702705 |
| nest_div EBM+FLAML | 0.706020 |
| region_pick | 0.707824 |
| nest+cur logit | 0.707969 |
| **post+pick logit** | **0.708285** |

## 协议

未达 0.71 不宣称 PASS；nested < B7 则 fallback。
