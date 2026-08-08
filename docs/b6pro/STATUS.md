# B6pro 状态

## 结论（当前）

- B7 保底：本地 **0.702704955** / 公开 **0.707**
- **新 closest**：**0.70602038**（direct_logit_base+ebm+flaml nestedC / b6pro_nest_div）
- 距 0.71 缺口 ≈ **0.00398**
- 产物：`artifacts/b6pro_long_best/`、`submissions/b6pro_closest/`

## 本轮

- 嵌套 logit 加入低相关 EBM+FLAML 臂，抬到 **0.70602**
- resid_cb2 / f09d-only 专模未超（切片专模欠拟合 ~0.52）
- 灵敏度：f09d-long 0.60→0.64 ⇒ 整体≈0.711；主攻弱区域加权全量训练
- 进行中：`b6pro_weak_weight`、`b6pro_ordered_kx`、`b6pro_iso_resid`

## 协议

未达 0.71 不宣称 PASS；nested < B7 则 fallback。
