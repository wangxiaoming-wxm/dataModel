# 保险索赔概率预测

> 当前状态：`SUBMIT_GATE_0.72 = FAIL`。仓库不保留可误交的正式
> `submission.csv`；最新证据见 `docs/RESEARCH_GATE_20260806.md`。

这是一个面向 AUC 指标的、可复现且防泄漏的训练流程。模型只使用
`train.csv` 中除 `id`、`label` 外的字段，不读取任何测试标签，也不根据公开榜单
反馈调节模型或集成权重。

## 验证策略

- 重复分层 5 折交叉验证（默认 2 次，共 10 个模型/算法）
- CatBoost 处理原生类别特征，XGBoost 处理数值和语义拆分特征
- 固定 50/50 概率平均，不在同一份 OOF 结果上搜索集成权重
- 每折独立早停，报告每折、每次重复和整体均值/标准差
- `id` 永不进入特征；运行前强制检查 ID 重叠、重复和提交顺序

训练测试对抗验证 AUC 为 0.491，未发现明显分布漂移。训练集正例率约 10%，
因此使用分层切分保证每折类别比例稳定。

## 运行

```bash
python3 -m pip install -e ".[dev]"
PYTHONPATH=src python3 -m insurance_claim.model \
  --data-dir . \
  --output-dir artifacts
```

该命令属于历史基线复现，会产生研究预测；它没有通过当前 0.72 提交门禁。
输出：

- `artifacts/submission.csv`：研究预测，**禁止作为正式提交**
- `artifacts/audit_report.json`：数据边界与泄漏审计
- `artifacts/cv_metrics.json`：完整交叉验证证据

## 测试

```bash
PYTHONPATH=src python3 -m pytest
```

测试覆盖数据审计、特征工程、概率边界、提交顺序和完整的小样本训练流程。
