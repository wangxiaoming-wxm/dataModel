# 审核意见：B5×8seed 声称 OOF=0.69817454

> **审核方**：独立审核者（不参与建模）  
> **依据**：`docs/supervision/INDEPENDENT_AUDIT_PROTOCOL.md`（IA-AUC698-v1）  
> **对象**：`artifacts/b5_8seed/` + `artifacts/b5_4s/` + `docs/FINAL_B5_8SEED.md` + `submissions/submission_b5_8seed.csv`  
> **日期**：2026-08-07

---

## 总判定：**CONDITIONAL PASS**

独立复算确认 **pooled OOF AUC = 0.6981745376**（与报告一致，Δ=0）。  
**无硬红线触发**，可视为“诚实本地 pooled ≥0.698”的有条件达标；但 **未满足无条件 PASS 的 §2.1-C（seed_mean≥0.693）**，且存在 **早停乐观 + 卡线区间 [0.698, 0.700)**，故降级为 **CONDITIONAL PASS**。

不得把本意见解读为无条件稳健达标；对外若宣称，须同时披露 seed_mean≈0.6898 与早停事实。

---

## 核对要点结论

| # | 要点 | 结果 |
|---|---|---|
| 1 | 新数据 only（14930/6398） | **PASS** — `audit.train/test_rows` 正确；三文件 SHA 与协议一致 |
| 2 | 无 TE / 无 OOF 搜权 / 等权多种子 | **PASS** — `target_encoding=none`；融合声明 `equal_seed_probability_mean_8seeds`；未见 OOF 权重搜索 |
| 3 | shuffled 近随机且 pass | **PASS** — `shuffled_oof_auc=0.507573` ∈ [0.48,0.52]；硬区间 [0.47,0.53] 亦过；`shuffled_pass=true` |
| 4 | 8-seed 合并代数合法 | **PASS** — `oof8=(4·oof4+Σoof_2030..2033)/8` 逐元素吻合（maxabs=0）；等价于 8 种子等权概率平均 |
| 5 | 未把单折高分当最终分 | **PASS** — 报告分为 `pooled_oof_auc`/`auc_8seed`；`fold_auc_max=0.7204` 仅披露 |
| 6 | submission 与 test id 对齐 | **PASS** — 6398 行；id 集合/顺序一致；label∈[0.025,0.879]；与 `predictions.npz['test']` 一致 |

---

## 红线检查表

| 红线 | 判定 | 证据 |
|---|---|---|
| 测试集标签泄漏 / 伪标签用测试 label | **PASS** | test 无 label；声明 `no_test_labels`；未见伪标签流程 |
| 全量 fit 再 OOF 的 TE/分箱/标准化 | **PASS** | `target_encoding=none`；声明 `fold_local_fe`；脚本按折 `build_b5(Xtr,Xva,test)` |
| 用 OOF 搜融合权重再报告同一 OOF | **PASS** | 8seed 为等权概率平均；无连续权重搜索。注：`b5_4s` 在预注册融合规则中选了 `b5_only`（轻度选模，未构成搜权红线） |
| 公开榜反馈调参仍称盲测 | **PASS（未见违规证据）** | 材料中无公开榜分数回流痕迹 |
| 旧数据/第三方预测当自研 OOF | **PASS** | SHA=新数据；OOF 来自本轮 `b5_4s`+本轮 2030–2033，非旧 `pred_bundle` |
| shuffled 不接近 0.5 仍称有效 | **PASS** | 0.50757，近随机 |
| 单折偶然高分当最终分 | **PASS** | 最终分=8seed pooled，非 fold_max |

---

## 合格门槛对照（§2.1）

| 门槛 | 要求 | 实测 | 判定 |
|---|---|---|---|
| A pooled | ≥0.698 | **0.69817454**（独立复算一致） | **PASS** |
| B CV | ≥5折×≥3种子等权 | 5折×8种子等权 | **PASS** |
| C seed_mean/std | mean≥0.693 且 std≤0.010 | **mean=0.689768** / std=0.001809 | **FAIL（mean）** |
| D shuffled | ∈[0.48,0.52] | 0.507573 | **PASS** |
| E 协议声明 | 全 true | 核心项 true；键名未完全对齐 §2.4；缺 `early_stopping_on_valid` 显式字段 | **PARTIAL** |
| F 非单折 | 报告 pooled | 是 | **PASS** |

附加：`fold_auc_range=0.0535 ≤ 0.06`（稳定性披露线内）。

---

## 降级为 CONDITIONAL 的理由（协议 §4.2）

1. **§2.1-C 未过**：8 个 seed 的 seed-level OOF 全在 ≈0.687–0.693，`seed_mean=0.6898<0.693`；达标完全依赖多种子概率 bagging，单 seed 并未站上 0.698。  
2. **早停 + 卡线**：`run_b5_8seed.py` / `b5_4s` 使用 `eval_set=(va,yva), use_best_model=True`（`best_iter` 约 104–724）。pooled 落在 **[0.698, 0.700)**，协议要求视为有条件，宜补固定迭代/无早停对照。  
3. **强制字段不完整（文档债，非作弊）**：`b5_8seed/metrics.json` 缺完整 `fold_aucs` 列表、缺 `early_stopping_on_valid`、`protocol_declaration` 未覆盖 §2.4 全键；不影响已复算的主分，但阻碍无条件放行。

---

## 独立复算摘要

```text
data: train=14930 test=6398  SHA match=YES  y==train.label=YES
auc(oof8)=0.6981745376  (== reported)
auc(oof4)=0.6969506894  (== b5_4s oof_b5, maxabs diff=0)
merge: oof8 == (4*oof4 + sum(extra4))/8   maxabs=0
submission: n=6398 id_order=OK label_range=[0.025,0.879] == npz test
```

---

## 审核签字

```text
[AUDIT_PACKET]
experiment_id: b5_focus_8seed_newdata
pooled_oof_auc: 0.6981745375887981
seed_mean / seed_std: 0.6897678366816498 / 0.0018089707797710045
seeds: [2026..2033]
n_splits: 5
shuffled_oof_auc: 0.5075732199169001
protocol_declaration: core true (incomplete vs §2.4 keys)
data_sha256_match: true
legacy_preds_used: false
oof_recomputed_auc: 0.6981745375887981
auditor_verdict: CONDITIONAL PASS
```

**允许**：在披露 seed_mean 与早停的前提下，声称“诚实 8-seed pooled OOF ≈0.69817（有条件）”。  
**不允许**：省略条件写成“已稳健无条件达标 / seed 级已过 0.698 / 盲测可外推公开榜”。  
**建议补强（非阻塞建模，阻塞无条件 PASS）**：固定迭代对照；补全 fold 表与 `early_stopping_on_valid=true`；将 `protocol_declaration` 对齐 §2.4。
