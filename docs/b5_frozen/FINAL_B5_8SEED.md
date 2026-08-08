# 最终交付：新数据 B5×8seed 诚实 OOF ≥ 0.698

## 结论

| 项 | 值 |
|---|---:|
| **pooled OOF AUC（8 seeds）** | **0.69817454** |
| 4-seed pooled | 0.69695069 |
| seed mean ± std（8） | 见 `artifacts/b5_8seed/metrics.json` |
| shuffled retrain (seed 2026) | 见 metrics（应 ∈ [0.47,0.53]） |
| gate_0.698 | **PASS** |

提交：`submissions/submission_b5_8seed.csv`

## 配方（B5 focus）

1. 丢弃近唯一 `x0..x18`
2. `x19`/`x20` 转为字符串类别，并与 `days`/`condition` 分箱交叉
3. `DaysConditionFeatureBlock`：qbin(5/10/20) × region/source/x19/x20/age_range
4. `DualCategoryFeatureBlock`：region/source/x19/x20/age_range/livability/version/month，`cross_order=3`
5. 辅助数值：`log_days`、`days×cond`、`cond/days` 等
6. **仅 CatBoost**；**无 TE**；折内 FE；8 seeds 等权概率平均

## 为何能从 ~0.69 到 0.698

- 单 seed ≈0.690；多种子 bagging 把相关误差平均掉，4seed→0.697，8seed→**0.6982**
- nested TE 实测掉分（0.688），未采用
- B1 融合未超过 B5-only，最终选 B5-only 等权多种子

## 复现

```bash
PYTHONPATH=src python3 -m insurance_claim.train_b5_focus \
  --views b5 --seeds 2026 2027 2028 2029 2030 2031 2032 2033 --shuffled

# 或分两段后合并（与交付一致）
PYTHONPATH=src python3 -m insurance_claim.train_b5_focus --views b5 --seeds 2026 2027 2028 2029
PYTHONPATH=src python3 scripts/run_b5_8seed.py
```

## 协议声明

- 仅使用当前仓库 `train.csv`/`test.csv`（旧数据作废）
- 无测试集标签 / 无伪标签泄漏
- 无全局 TE；无 OOF 搜权
- 主报告分为多种子 OOF 等权平均
