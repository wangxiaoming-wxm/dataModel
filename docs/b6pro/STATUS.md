# B6pro 状态

## 结论（当前）

- B7 保底：本地 **0.702704955** / 公开 **0.707**
- **新 closest**：**0.70680850**（seq_patch_lm：nest_div + 弱区 lgb/mlp 嵌套 α / b6pro_f09d_multi）
- 距 0.71 缺口 ≈ **0.00319**
- 产物：`artifacts/b6pro_long_best/`、`submissions/b6pro_closest/`

## 抬升轨迹

| 配方 | nested |
|---|---:|
| B7 max3 | 0.702705 |
| nest_stack | 0.705734 |
| nest_div EBM+FLAML | 0.706020 |
| f09d score×lgb | 0.706402 |
| **seq_patch 弱区 lm** | **0.706808** |

## 进行中

- `b6pro_weak_weight`（CatBoost 弱区加权）
- `b6pro_iso_resid`（days-isotonic 残差）
- `b6pro_lgb_weakw`（LGBM 弱区加权）

## 协议

未达 0.71 不宣称 PASS；nested < B7 则 fallback。
