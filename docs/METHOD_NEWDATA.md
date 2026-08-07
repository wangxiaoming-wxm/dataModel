# 新数据车险索赔建模进展（诚实 OOF）

## 数据绑定

| 项 | 值 |
|---|---|
| train / test | 14930 / 6398 |
| 正例率 | ≈0.10020 |
| 旧数据/旧 OOF | **全部作废**，仅参考代码逻辑 |
| 指标 | `roc_auc_score(y_true, y_pred)` |

数据门禁与审核协议：`docs/supervision/INDEPENDENT_AUDIT_PROTOCOL.md`。

## 特征工程主线

1. **暴露轴**：`days` / `condition` 分位箱 + 积/比 + log1p  
2. **语义解析**：`source→car/eng`，`t3→num/sfx`，`version→era`  
3. **业务交叉（字符串，交给 CatBoost）**：`region×days`、`days×cond`、`car×days`、`cond×source`、`x19/x20×days`  
4. **丢弃**：近行唯一 `x0..x18` 原值（噪声）；**保留** `x19/x20` 并转为类别  
5. **禁止**：全局 TE、测集伪标签、OOF 搜权、旧预测包

## 消融结论（1-seed=2026，Stratified 5-fold）

| 配方 | OOF |
|---|---:|
| A0 classic semantic | 0.6843 |
| physics / risk | 0.6893 / 0.6883 |
| risk+physics mean | 0.6900 |
| **B5 focus（x19/x20 cat + days 交叉）** | **0.69021** |
| B1 order-2 denser days | 0.68971 |
| v2 clean dual（去 condition/livability） | 0.6834 ↓ |

抬分关键：**把 `x19/x20` 当类别并与 days/condition/source 交叉**，同时丢掉近唯一 `x*`。

## 协议

- `StratifiedKFold(5)` × ≥3 seeds，等权概率平均为主报告分  
- 折内 fit 特征块；无 TE 或仅严格 nested TE  
- shuffled-label 复训应 ∈ [0.47, 0.53]  
- 融合仅在预注册规则（单模型 / 等权均值 / 等权 rank）中选择  

## 复现

```bash
# 当前最佳配方多 seed
PYTHONPATH=src python3 -m insurance_claim.train_b5_focus \
  --views b5 b1 --seeds 2026 2027 2028 2029 --shuffled

# 早期 semantic+
PYTHONPATH=src python3 -m insurance_claim.train_semantic_plus --seeds 2026
```

提交：`submissions/submission_b5_focus.csv`（训练完成后生成）。

## 状态

持续迭代中：1-seed 天花板约 **0.690**，目标诚实 pooled ≥ **0.698**（差额约 0.008）。下步靠多种子 bagging + 低相关异构臂等权融合，不做泄漏型 TE。
