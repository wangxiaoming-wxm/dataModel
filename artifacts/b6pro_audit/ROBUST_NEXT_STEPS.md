# 稳健下一实验（摘要）

1. 回退 B7 为唯一安全提交（公开 0.707）
2. 禁止：单 seed 过线、ultra 小样本嵌套 α 唯一抬分、多层同标签 OOF 堆叠后宣称 PASS
3. 优先：≥4–8 seed 异构臂 + 预注册离散融合（nested_select_rule）+ ≥3 组外层 SKF random_state
4. 参考臂：冻结 gap/gap_bag + 自训 keepx/plus/xgb/EBM；勿再 patch 已作废 closest
