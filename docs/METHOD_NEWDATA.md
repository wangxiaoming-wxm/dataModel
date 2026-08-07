# 新数据车险索赔建模（诚实 OOF）

## 最终结果

| 项 | 值 |
|---|---:|
| **pooled OOF（8 seeds）** | **0.69817454** |
| seed mean ± std | 0.68977 ± 0.00181 |
| shuffled retrain | 0.50757（PASS） |
| gate ≥ 0.698 | **PASS** |

提交：`submissions/submission_b5_8seed.csv`  
详情：`docs/FINAL_B5_8SEED.md` · `artifacts/b5_8seed/metrics.json`

## 数据

train 14930 / test 6398；正例率 ≈0.1002；**旧数据与旧 OOF 全部作废**。

## 配方摘要

B5 focus：丢 `x0..x18`；`x19/x20` 作类别并与 days/condition 语义交叉；CatBoost 原生类别三阶；无 TE；8 seeds 等权平均。

## 复现

```bash
PYTHONPATH=src python3 -m insurance_claim.train_b5_focus --views b5 --seeds 2026 2027 2028 2029
PYTHONPATH=src python3 scripts/run_b5_8seed.py
```
