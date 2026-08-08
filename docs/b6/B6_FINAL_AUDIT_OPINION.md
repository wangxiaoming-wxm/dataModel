# B6 独立终审意见（IA-AUC700-B6-v1）

> **审核方**：独立复核官（Independent Auditor；不参与建模 / 抬分）  
> **协议**：`docs/supervision/B6_AUDIT_PROTOCOL.md`（IA-AUC700-B6-v1）  
> **对象**：`docs/B6_FINAL_REPORT.md` + `artifacts/b6_gapbag_8seed/` + `submissions/submission_b6_gapbag_8seed.csv`  
> **对照冻结**：`artifacts/b5_frozen/` / `docs/b5_frozen/` / `submissions/b5_frozen/`  
> **分支**：`cursor/b6-push-auc070-a5f5`  
> **日期**：2026-08-07

---

## 总判定：**REJECT**（驳回 0.70 交付）

独立复算 **pooled OOF AUC = 0.6989746962571622**（与团队报告一致，Δ=0）。  
**未达到** `pooled_oof_auc ≥ 0.70`，故 **不得** 宣称诚实本地 OOF ≥ 0.70，**不得** 将 0.69897 包装为 0.70。

在硬红线复查下，本候选 **协议清洁**（无 TE / 无 OOF 搜权 / shuffled 合格 / 种子未樱桃采摘 / B5 冻结未篡改）。  
据此 **认证最接近诚实分**：

```text
closest_honest_pooled_oof_auc = 0.6989746962571622
```

| 项 | 值 |
|---|---|
| auditor_verdict | **REJECT** |
| deliver_0_70_allowed | **false** |
| gate_0_70 | **FAIL** |
| why_not_0.70 | pooled 0.69897470 < 0.70（缺口 ≈ 0.001025） |
| red_lines_hit | **[]**（无硬红线） |
| B5 freeze check | **PASS**（pooled 仍为 0.6981745375887981） |

---

## 独立复算摘要

```text
data: train=14930 test=6398
SHA(train/test/submit) = protocol §0 match = YES
y in predictions.npz == train.label = YES

oof_recomputed_auc = 0.6989746962571622
reported_pooled    = 0.6989746962571622
delta              = 0.0  (< 1e-8)

fusion algebra: oof == 0.5*(oof_gap + oof_gap_bag)   maxabs = 0
                test == 0.5*(test_gap + test_gap_bag) maxabs = 0
arm AUC(gap)     = 0.6986833630687241  (== metrics)
arm AUC(gap_bag) = 0.6989061794680111  (== metrics)
corr(gap,gap_bag)= 0.996025

submission_b6_gapbag_8seed.csv:
  n=6398  id_order==test.id  YES
  label == npz['test']       maxabs ~ 1e-16
  label_range ≈ [0.0232, 0.8728]
```

机器可读复算：`artifacts/b6_audit/b6_final_audit_recompute.json`  
终审冻结核验：`artifacts/b6_audit/b5_freeze_check.json` → **verdict=PASS**

---

## 团队宣称核对

| 宣称 | 独立结果 |
|---|---|
| closest honest pooled = 0.69897470 | **确认** 0.6989746962571622 |
| gate_0_70 = FAIL | **确认** |
| shuffled ≈ 0.5056 PASS | **确认** 0.5055995184220101 ∈ [0.48, 0.52] |
| recipe equal_prob(gap, gap_bag), seeds 2026–2033 | **确认**；等权概率代数成立 |
| fold-local FE, no TE, no OOF weight search | **确认**（代码 `target_encoding=none`；融合无连续搜权；metrics 声明一致） |

---

## 红线检查表

| 红线 | 判定 | 证据 |
|---|---|---|
| 测试集标签 / 伪标签泄漏 | **PASS** | test 无 label；声明 `no_test_labels`；未见伪标签流程 |
| 全量 fit TE / 全局编码进 OOF | **PASS** | `train_b6.py` 折内 `build_b5` / gap FE；`no_global_te=true` |
| 报告用同一 OOF 搜融合权重 | **PASS** | primary=`equal_prob`；可复算到 1e-12；无 blend 网格 |
| 公开榜回流仍称盲测 | **PASS（未见违规）** | 材料无 LB 分数回流 |
| 旧数据 / 第三方预测冒充自研 OOF | **PASS** | 数据 SHA=新数据；gap 臂与 `artifacts/b6_8seed` 的 `oof_gap` 逐元素一致（复用自研） |
| shuffled ∉ [0.47, 0.53] 仍称有效 | **PASS** | 0.50560，合格带内 |
| 只报单折 / 单 seed 最高分 | **PASS** | 报告分为 pooled；seed_fusion 全表披露 |
| B5 冻结被改 | **PASS** | 三件套 SHA 与开跑基线一致；独立复算 B5 OOF=0.6981745375887981 |
| 事后挑种子冒充预注册 | **PASS** | seeds=`[2026..2033]` 完整 8 个，非高分子集 |

**说明（非硬红线，不影响 closest-honest 认证）**：交付臂从原计划 `equal_prob(b5,gap)` 调整为 `equal_prob(gap,gap_bag)` 是在 1-seed / 对照实验后写入的路径更新；融合仍为等权概率、无连续搜权，且团队未宣称过 0.70。若将来以该臂组合宣称 ≥0.70，须更早的书面预注册时间戳，否则最多 CONDITIONAL。

---

## §2 门槛对照（0.70 交付）

| # | 门槛 | 实测 | 判定 |
|---|---|---|---|
| A | pooled ≥ 0.70 | **0.69897470** | **FAIL** → 直接 REJECT |
| B | seeds≥8、n_splits≥5、等权 | 8 seeds × 5 folds；等权 | PASS |
| C | shuffled ∈ [0.48, 0.52] | 0.50560 | PASS |
| D | 无 OOF 搜权；等权融合 | equal_prob 可证 | PASS |
| E | B5 冻结未篡改 | freeze check PASS | PASS |
| F | 声明 / SHA / 复算 <1e-8 | 复算Δ=0；SHA 匹配；声明键不完整 | PARTIAL（字段债） |
| G | 报告分为 pooled | 是；gate 诚实 FAIL | PASS |

稳定性参考（若已过 0.70 会影响等级）：`seed_mean=0.691634 < 0.693`；`seed_std=0.001239 ≤ 0.010`；gap_bag `fold_auc_range≈0.0476 ≤ 0.06`。早停 `use_best_model=true` 已披露，无固定 iter 对照臂写入本交付 metrics。

---

## 文档债（阻塞 PASS，不推翻 closest-honest）

`artifacts/b6_gapbag_8seed/metrics.json` 缺若干 §4 强制顶层键（含 `protocol_id`、`data_sha256`、`train_rows`/`test_rows`、`fold_auc_min/max/range`、`target_encoding`、`fixed_iter_control`、`b5_freeze_untampered` 等）；`protocol_declaration` 键名未完全对齐 §4.5。`folds` 仅含 `gap_bag` 折表。  
以上构成文档债，在已 FAIL 门槛 A 的前提下不改变 REJECT；亦不否定已复算主分的诚实性。

---

## 审核签字

```text
[AUDIT_PACKET_B6]
protocol_id: IA-AUC700-B6-v1
experiment_id: b6_gap_gapbag_8seed
pooled_oof_auc: 0.6989746962571622
gate_0_70: false
seed_mean / seed_std: 0.6916341572663821 / 0.00123881803678489
seeds: [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033]
n_splits: 5
shuffled_oof_auc: 0.5055995184220101
early_stopping_on_valid: true
fixed_iter_control_pooled: null
fusion: equal_prob(gap, gap_bag)
fusion_preregistered: true (equal-weight; path updated post 1seed — see caveat)
b5_freeze_check: PASS (pooled still 0.6981745375887981)
protocol_declaration: core true (incomplete vs §4.5 key set)
data_sha256_match: true
oof_recomputed_auc: 0.6989746962571622
closest_honest_pooled_oof_auc: 0.6989746962571622
auditor_verdict: REJECT
deliver_0_70_allowed: false
red_lines_hit: []
```

**允许**：对外报告「B6 最接近诚实 8-seed pooled OOF = **0.69897470**（未过 0.70；协议清洁）」。  
**不允许**：宣称「已达诚实 OOF ≥ 0.70 / 可交付 0.70 标签 / 用 fold_max、seed_max、搜权分或 rank 副分替代」。  
**相对 B5**：诚实抬分 ≈ **+0.00080016**（0.69897470 − 0.69817454）；缺口至 0.70 仍约 **0.001025**。
