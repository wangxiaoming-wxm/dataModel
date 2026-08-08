# 车险索赔 AUC 竞赛 · B6 独立复核协议（IA-AUC700-B6-v1）

> **角色**：独立复核官（Independent Auditor）。**不参与**写模型、调参、抬分代码或提交包装。  
> **监督对象**：分支 `cursor/b6-push-auc070-a5f5` 上冲刺诚实本地 OOF **≥ 0.70** 的实验与交付。  
> **基线**：B5 已冻结 — pooled **0.69817454**（分支 `cursor/claim-auc698-council-a5f5`）；B5 意见为 **CONDITIONAL PASS**（见 `docs/supervision/FINAL_AUDIT_OPINION.md`）。  
> **用户硬约束**：不存在过拟合与作弊才能交付 **0.70**；否则只报告**最接近的诚实分**，不得包装为达标。  
> **效力**：主进程宣称 B6 达标前必须按本协议自检；任一红线 FAIL → **驳回**。终审由本复核官在结果就绪后另行签字。

**继承**：本协议继承 `docs/supervision/INDEPENDENT_AUDIT_PROTOCOL.md`（IA-AUC698-v1）全部红线与数据门禁；下列为 **相对 B5 的加严增量**。冲突时以更严条款为准。

---

## 0. B5 冻结基线（不可篡改）

| 项 | 冻结值 |
|---|---|
| experiment_id | `b5_focus_8seed_newdata` |
| pooled_oof_auc / auc_8seed | **0.6981745375887981** |
| seeds | `[2026..2033]`（8 个，等权概率平均） |
| shuffled_oof_auc | **0.5075732199169001** ∈ [0.48, 0.52] |
| seed_mean / seed_std | **0.68976784** / **0.00180897** |
| fusion | `equal_seed_probability_mean_8seeds` |
| 冻结目录 | `artifacts/b5_frozen/`、`submissions/b5_frozen/`、`docs/b5_frozen/` |

**B6 不得修改**上述冻结文件内容或语义；允许只读引用。开跑前与终审前均须复跑冻结核验：

```bash
# 产物：artifacts/b6_audit/b5_freeze_check.json → verdict 必须为 PASS
```

任一冻结 SHA/分数漂移 → B6 交付 **一票否决**（视为基线被污染）。

数据 SHA（与 IA-AUC698-v1 一致）：

- `train.csv` = `494a61073a0438f692914c4868db31df1171e662348e0024e06b120d08d44f28`
- `test.csv` = `d6ffd26bd4873fa09f6fac361f59170a880e88e331a01d7a6356bd9184ce55ec`
- `submit_sample.csv` = `83cb0263cc5729f61d0e05c68d673dc3f21b41c24bad68afa35159859054c4bf`

---

## 1. 相对 B5 的增量审核点

B5 已在卡线区间 **[0.698, 0.700)** 获得有条件放行；B6 冲 **0.70** 时，下列风险从“披露即可”升级为**必须过关**。

### 1.1 多种子 bagging 是否虚高

| 审核点 | 要求 | 驳回信号 |
|---|---|---|
| 种子数量 | **seeds ≥ 8**（完整列表落盘；禁止只报最好 seed） | `<8` 仍宣称 0.70 |
| 加权方式 | **等权概率平均**（或事先书面预注册的等权 rank 平均，二选一写死） | 看完 OOF 后加权 / 丢弃低分 seed |
| bagging 贡献披露 | 必须同时报告 `seed_mean`、`seed_std`、`pooled_oof_auc` | 只报 pooled、隐瞒 seed_mean≪0.70 |
| 虚高判定 | 若 `pooled≥0.70` 但 `seed_mean < 0.693`，视为**强依赖 bagging** | 不得写成“单模型已稳过 0.70”；最多 CONDITIONAL，且须证明非选 seed 偏置 |
| 扩展种子 | 若由 8→12+，须披露新增 seed 列表为**预注册**，禁止“试很多 seed 后只留下高分” | 事后挑选种子集合 |

**B5 教训**：8-seed pooled≈0.69817，而 seed_mean≈0.6898。B6 若仅靠再加种子把 pooled 推过 0.70，复核官将重点审查种子预注册与等权完整性。

### 1.2 早停乐观偏差

| 审核点 | 要求 | 驳回 / 降级信号 |
|---|---|---|
| 声明 | `early_stopping_on_valid` 必须显式 `true/false` | 缺字段 → 不得无条件 PASS |
| 卡线加严 | 若使用 `eval_set` + `use_best_model`，且 pooled ∈ **[0.700, 0.705)** | 默认 **CONDITIONAL**：须提供固定 iteration / 关闭 OD 对照 |
| 对照臂 | 对照臂须同一特征块、同一 seeds（或至少同一主 seed 集）、仅改早停策略 | 对照换配方再报“无早停也过” → 无效 |
| 乐观幅度 | 报告 `best_iter` 分布或固定 iter 对照的 ΔAUC | 无披露且刚过 0.70 → CONDITIONAL 或 REJECT（视完整度） |

**性质重申**：折内 valid 早停 ≠ 标签泄漏，但会使 OOF **轻度乐观**。B5 已因此降级；B6 不得在无对照时把“刚过 0.70”包装为稳健达标。

### 1.3 预注册融合是否被破坏

`docs/B6_PLAN.md` 预注册方向摘要：

1. B5 主臂保留 + 异构臂（Lossguide / 解析增强 / 物理残差）等权或等权 rank（**开跑前二选一写死**）  
2. 种子扩展（≥8，目标 12）等权 bagging  
3. 早停乐观对照：固定 iteration 无 OD  
4. 禁止：全局 TE、OOF 网格搜权、旧预测包、测集伪标签  

| 审核点 | 要求 | 红线 |
|---|---|---|
| 融合规则时间戳 | 融合类型（等权概率 / 等权 rank）、臂名单须在看 OOF 前写入 metrics/`protocol_declaration` 或计划文档 | 用报告用的同一 OOF 搜权重 / 网格融合 |
| 臂选择 | 允许预注册臂的等权融合；**禁止**“多臂全跑完后按 OOF 挑幸存臂再报同一 OOF” | 选模偏置冒充预注册 |
| B5 主臂 | 可保留 B5 配方；不得改写 B5 冻结产物来“更新基线” | 篡改 `b5_frozen` |
| 异构臂 | 新臂须独立 OOF 数组可抽检；融合公式可复算到 1e-12 | 缺数组 / 公式与声明不符 |
| 旧产物 | 旧数据 / 第三方 `oof_*.npy` / 旧提交不可并入自研 OOF | 立即 FAIL |

### 1.4 其它相对 B5 的加严项

- **目标分**：主门槛从 0.698 → **0.70**（`pooled_oof_auc ≥ 0.70`）。  
- **报告纪律**：未过 0.70 时，权威分仍为诚实 pooled；可并列 “closest_honest_pooled”，**禁止**用 fold_max / seed_max / 搜权分填 0.70。  
- **字段完整**：B5 曾缺完整 `fold_aucs`、`early_stopping_on_valid`、§2.4 全键；B6 **缺一不得无条件 PASS**。

---

## 2. 达标门槛（宣称诚实 OOF ≥ 0.70）

全部满足才进入合格候选：

| # | 门槛 | 硬性 |
|---|---|---|
| A | **`pooled_oof_auc ≥ 0.70`**（多种子等权概率平均后的全样本 AUC；若预注册为等权 rank，须在声明中写死且全程一致） | 是 |
| B | **`len(seeds) ≥ 8`**，分层 **`n_splits ≥ 5`**，种子等权 | 是 |
| C | **`shuffled_oof_auc ∈ [0.48, 0.52]`**（合格带）；硬失败带外：**∉ [0.47, 0.53]** | 是 |
| D | **禁止 OOF 搜权**：`no_oof_weight_search_on_reported_oof=true`；融合为预注册等权（或等权 rank） | 是 |
| E | **B5 冻结文件未被篡改**：`artifacts/b6_audit/b5_freeze_check.json` → `verdict=PASS`；冻结 metrics 中 pooled 仍为 **0.6981745375887981** | 是 |
| F | 协议声明（§4）全部 `true`；数据 SHA 与 §0 一致；独立复算 OOF 误差 **< 1e-8** | 是 |
| G | 报告分 = pooled（或预注册 seed_mean 且不得高于 pooled 包装）；禁止 fold_max / seed_max | 是 |

**稳定性参考线（影响合格等级，非单独红线）：**

| 项 | 期望 | 偏离处理 |
|---|---|---|
| `seed_mean` | ≥ 0.693 | 低于则最多 CONDITIONAL（强 bagging 依赖） |
| `seed_std` | ≤ 0.010 | 过大 → CONDITIONAL，要求更多 seeds 或证明非泄漏 |
| `fold_auc_range` | ≤ 0.06 | 超出 → CONDITIONAL |

**硬失败（直接 REJECT，不论 pooled 多高）：**

- shuffled ∉ [0.47, 0.53] 仍称有效  
- 测试集标签 / 伪标签泄漏  
- 全量 fit 再 OOF 的 TE/分箱/标准化/词表  
- **在报告用的同一 OOF 上搜融合权重**  
- 公开榜回流调参仍称盲测本地分  
- 旧数据/第三方预测当自研 OOF  
- 只报单折/单 seed 最高分  
- **B5 冻结文件被改**（SHA 或分数漂移）  
- 事后挑选种子/臂集合冒充预注册  

---

## 3. 合格 / 有条件合格 / 驳回规则

### 3.1 PASS（合格）

同时满足：

1. §2 门槛 A–G 全部通过；  
2. `seed_mean ≥ 0.693` 且 `seed_std ≤ 0.010`；  
3. `fold_auc_range ≤ 0.06`；  
4. 若 `early_stopping_on_valid=true`：须有固定 iter / 无 OD 对照，且对照 pooled 仍 **≥ 0.70**，或主实验本身为固定 iter；  
5. 强制 metrics 字段（§4）齐全；融合规则与 B6 计划预注册一致；  
6. 独立复算与 submission 对齐抽检通过。  

→ **批准**宣称：“诚实本地多种子 pooled OOF ≥ 0.70”。

### 3.2 CONDITIONAL（有条件合格）

`pooled ≥ 0.70` 且无硬红线，但出现以下之一：

- `seed_mean < 0.693`（达标主要靠多种子 bagging）；  
- 早停开启且 pooled ∈ **[0.700, 0.705)**，缺固定 iter 对照；  
- `seed_std > 0.010` 或 `fold_auc_range > 0.06`；  
- 强制字段部分缺失（文档债），但不影响已复算主分与红线；  
- 预注册融合表述含糊，但等权可证且无搜权证据。  

→ **不得**写成无条件稳健达标；对外须披露条件项。  
→ 若用户要求“不存在过拟合与作弊才能交付 0.70”，**CONDITIONAL 默认不构成可交付 0.70**；复核官应同时给出 **closest_honest_pooled** 与条件清单，由用户决定是否接受有条件声明。

### 3.3 REJECT（驳回）

- `pooled_oof_auc < 0.70`；或  
- 任一硬红线；或  
- 用作弊/泄漏手段把分数做上 0.70。  

**驳回时的强制输出：**

```text
verdict: REJECT
claimed_or_attempted: ...
closest_honest_pooled_oof_auc: <独立复算的合法 pooled>
why_not_0.70: <一条主因>
red_lines_hit: [...]
```

不得用 CONDITIONAL 话术掩盖 REJECT。

### 3.4 分数权威序（写死）

```text
权威分 = pooled_oof_auc（预注册等权规则下的全样本 AUC）
参考分 = seed_mean ± seed_std
对照分 = fixed_iter_pooled（若有）
禁止当分 = max(fold_aucs), max(seed_aucs), OOF搜权后分数, 公开榜, 旧数据OOF, 被篡改的B5分
未达标时交付分 = closest_honest_pooled_oof_auc
```

---

## 4. 强制 metrics 字段清单

B6 每次候选交付的 `metrics.json`（或等价）**必须**包含下列字段；缺关键项 → 不得 PASS。

### 4.1 身份与数据绑定

| 字段 | 要求 |
|---|---|
| `experiment_id` | 唯一；建议前缀 `b6_` |
| `git_commit` | 训练代码 commit |
| `git_branch` | 应为 `cursor/b6-push-auc070-a5f5` 或其记录的实验子提交 |
| `data_sha256` | train/test/submit 三 SHA，须与 §0 一致 |
| `train_rows` / `test_rows` | 14930 / 6398 |
| `pos_rate` | train label 均值 |
| `feature_list_hash` 或完整特征清单 | 可复现 |
| `b5_freeze_ref` | 引用冻结 metrics SHA 或 `pooled=0.6981745375887981` 声明 |
| `b5_freeze_untampered` | `true`（并附 `artifacts/b6_audit/b5_freeze_check.json`） |
| `protocol_id` | `IA-AUC700-B6-v1` |
| `protocol_declaration` | 见 §4.5，逐键布尔 |

### 4.2 CV / 种子 / 主分

| 字段 | 要求 |
|---|---|
| `cv_scheme` | 分层 K 折，推荐 `StratifiedKFold` |
| `n_splits` | ≥ 5 |
| `seeds` | **≥ 8** 的完整列表（预注册） |
| `n_seeds` | `len(seeds)` |
| `seed_aucs` | 每 seed 的 seed-level OOF AUC |
| `seed_mean` / `seed_std` | 等权 |
| `pooled_oof_auc` | **主报告分** |
| `fold_aucs` | 每 seed × 每 fold 的 valid AUC（完整表或嵌套结构） |
| `fold_auc_min` / `fold_auc_max` / `fold_auc_range` | 必填 |
| `gate_0_70` | `pooled_oof_auc >= 0.70` |

### 4.3 泄漏控制与训练细节

| 字段 | 要求 |
|---|---|
| `target_encoding` | `none` 或严格 nested 描述 |
| `binning_scaler_pca_freq` | 折内 fit 声明 |
| `cat_vocab_fit_scope` | 折内 |
| `early_stopping_on_valid` | **显式** true/false |
| `fixed_iter_control` | 若早停为 true：对照结果对象或路径；否则可 `null` |
| `pseudo_label` | 必须 `false` |
| `external_predictions_used` | 必须 `false`（自研） |
| `blend_weights` / `fusion` | 预注册等权或等权 rank；禁止 OOF 搜权结果 |
| `fusion_preregistered` | `true` |
| `arms` | 融合臂名单与各臂 OOF 引用 |
| `public_lb_feedback_used` | 盲测声明时必须 `false` |

### 4.4 诚实性对照与复现工件

| 字段 | 要求 |
|---|---|
| `shuffled_oof_auc` | 至少 1 个正式 seed，同流水线打乱 label |
| `shuffled_pass` | 是否 ∈ [0.48, 0.52] |
| `pred_mean_oof` / `pred_mean_test` | 披露 |
| `train_valid_gap_median` | 若可得则必填 |
| 工件 | `predictions.npz`（`oof`,`y`,`test`）、`metrics.json`、`submission.csv`、一行复现命令 |

### 4.5 协议声明（须逐键 true/false）

```text
PROTOCOL_DECLARATION:
  no_test_label_leak: true/false
  no_test_pseudo_label: true/false
  fold_local_encoding_only: true/false
  no_oof_weight_search_on_reported_oof: true/false
  no_public_lb_tuning_claimed_as_blind: true/false
  no_legacy_or_thirdparty_preds_as_self_oof: true/false
  shuffled_near_chance: true/false
  reported_score_is_pooled_multiseed: true/false
  equal_seed_average: true/false
  fusion_preregistered: true/false
  b5_freeze_untampered: true/false
  seeds_preregistered_no_cherry_pick: true/false
  early_stopping_disclosed: true/false
```

全部为 `true` 才允许进入 PASS 候选。

---

## 5. 复核流程（主进程 → 复核官）

1. **开跑前**：确认 `b5_freeze_check.json` PASS；融合规则写入计划/metrics 草稿。  
2. **实验中**：不向复核官索要抬分建议；复核官不提供建模代码。  
3. **宣称达标时**：提交 `[AUDIT_PACKET_B6]`（§6）+ 完整 artifacts。  
4. **终审**：复核官独立复算 OOF、核对冻结 SHA、核对预注册融合、出具 PASS / CONDITIONAL / REJECT。  
5. **未达标**：只发布 `closest_honest_pooled_oof_auc` 与红线/缺口说明。

---

## 6. 审核签字格式

```text
[AUDIT_PACKET_B6]
protocol_id: IA-AUC700-B6-v1
experiment_id: ...
pooled_oof_auc: ...
gate_0_70: true/false
seed_mean / seed_std: ... / ...
seeds: [...]   # n>=8
n_splits: ...
shuffled_oof_auc: ...
early_stopping_on_valid: true/false
fixed_iter_control_pooled: ... | null
fusion: ...
fusion_preregistered: true/false
b5_freeze_check: PASS/FAIL (pooled still 0.6981745375887981)
protocol_declaration: all true?
data_sha256_match: true/false
oof_recomputed_auc: ...
closest_honest_pooled_oof_auc: ...
auditor_verdict: PASS | CONDITIONAL | REJECT
deliver_0_70_allowed: true/false
```

`deliver_0_70_allowed=true` **仅当** `auditor_verdict=PASS`（用户“无过拟合无作弊才交付”默认解释下，CONDITIONAL 不自动放行交付标签）。

---

## 7. 开跑前冻结核验（本轮已执行）

独立复核官已用 Python 核对 B5 冻结文件存在性、SHA 一致性、分数未改，并独立复算 OOF。

机器可读结果：`artifacts/b6_audit/b5_freeze_check.json`

期望摘要（开跑时）：

```text
verdict: PASS
expected_pooled_oof_auc: 0.6981745375887981
actual_pooled_oof_auc:   0.6981745375887981
oof_recomputed_auc:      0.6981745375887981
freeze_files_present: true
scores_unaltered: true
```

终审前须**重跑**同一核验；若 FAIL → 中止 B6 0.70 交付审查。

---

## 8. 复核官立场

1. 本文件是 **B6 放行标准**，不是抬分方案。  
2. 默认怀疑任何“刚好 ≥0.70”且缺 shuffled / 多种子完整表 / 早停对照 / 预注册融合证明的结果。  
3. B5 的 CONDITIONAL PASS **不自动**延伸为 B6 PASS。  
4. 复核官不修改模型代码；只接受 / 有条件接受 / 驳回，并在驳回时报告最接近的诚实分。  
5. 版本：`IA-AUC700-B6-v1`；数据或冻结基线变更后必须升版并重跑 §7。

---

**协议生效**：本文档合入仓库后，后续所有“本地 OOF≥0.70”声明均以本协议裁决；主进程达标后须再提请本复核官**终审**。
