# 车险索赔 AUC · B7 独立监督协议（IA-AUC710-B7-v1）

> **角色**：独立监督者 / 复核官。**不参与**写模型、调参、抬分代码或提交包装。  
> **对象**：分支 `cursor/b7-push-auc071-a5f5` 冲刺诚实本地 OOF **≥ 0.71**。  
> **冻结**：B6 closest honest pooled **0.6989746962571622**；B5 **0.6981745375887981**。B7 **不得篡改** B5/B6 冻结目录。  
> **用户约束**：仅无过拟合/作弊时可交付 0.71；否则只报 **closest honest nested**。  
> **效力**：主进程宣称 B7 达标前必须按本协议自检；任一硬红线 FAIL → **驳回**。终审由本复核官在结果就绪后另行签字。

**继承**：本协议继承 `docs/supervision/B6_AUDIT_PROTOCOL.md`（IA-AUC700-B6-v1）与 `INDEPENDENT_AUDIT_PROTOCOL.md`（IA-AUC698-v1）全部红线与数据门禁；下列为 **相对 B6 / V10 的加严增量**。冲突时以更严条款为准。

---

## 0. B6 / B5 冻结基线（不可篡改）

| 项 | 冻结值 |
|---|---|
| B6 experiment_id | `b6_gap_gapbag_8seed` |
| B6 closest honest `pooled_oof_auc` | **0.6989746962571622** |
| B6 fusion | `equal_prob(gap, gap_bag)` × seeds `[2026..2033]` |
| B6 shuffled_oof_auc | **0.5055995184220101** ∈ [0.48, 0.52] |
| B6 冻结目录 | `artifacts/b6_frozen/`、`submissions/b6_frozen/`、`docs/b6_frozen/` |
| B5 pooled | **0.6981745375887981**（`*/b5_frozen/` 仍不可改） |

**B7 不得修改**上述冻结文件内容或语义；允许只读引用。开跑前与终审前均须复跑冻结核验：

```bash
# 产物：artifacts/b7_audit/b6_freeze_check.json → verdict 必须为 PASS
```

对照分支：`origin/cursor/b6-push-auc070-a5f5`（submission / metrics_summary 字节一致）。  
任一冻结 SHA/分数漂移 → B7 交付 **一票否决**。

数据 SHA（与 IA-AUC698-v1 / B6 一致）：

- `train.csv` = `494a61073a0438f692914c4868db31df1171e662348e0024e06b120d08d44f28`
- `test.csv` = `d6ffd26bd4873fa09f6fac361f59170a880e88e331a01d7a6356bd9184ce55ec`
- `submit_sample.csv` = `83cb0263cc5729f61d0e05c68d673dc3f21b41c24bad68afa35159859054c4bf`

---

## 1. 分数口径（相对 B6 的关键变更）

| 口径 | 要求 |
|---|---|
| **主报** | **`nested_oof_auc`**（嵌套折上选定预注册离散融合规则后的 OOF AUC） |
| 副报 | full-data 选定规则下的 pooled；各臂 OOF；公开榜另列且不得冒充盲测本地分 |
| 门禁 | **`nested_oof_auc ≥ 0.71`** 才可主张达标 |
| seeds | 主臂（B6-class）**≥ 8**；plus 臂若沿用 V10 预注册（如 4×10fold）须完整披露，不得事后改口径 |
| 权威序 | `nested_oof_auc` ≫ full-data 选定规则分 ≫ 单臂 / fold_max / seed_max（后两者禁止当分） |

未过 0.71 时：权威交付分 = **`closest_honest_nested_oof_auc`**（独立复算合法 nested），禁止用 full-data max / 搜权分 / LB 包装。

---

## 2. 达标门槛（宣称诚实 nested OOF ≥ 0.71）

全部满足才进入合格候选：

| # | 门槛 | 硬性 |
|---|---|---|
| A | **`nested_oof_auc ≥ 0.71`** | 是 |
| B | 主臂 `len(seeds) ≥ 8`（或书面预注册的 plus 口径已披露），分层 **`n_splits ≥ 5`**（嵌套选规则默认 `StratifiedKFold(5, shuffle=True, random_state=42)`，须与计划一致） | 是 |
| C | **`shuffled_oof_auc ∈ [0.48, 0.52]`**（合格带）；硬失败：**∉ [0.47, 0.53]`**。若主报依赖 `max` 规则，另须 **`shuffled_plus`（或等价打乱异构臂）后 max 明显崩盘（`< 0.66`）** | 是 |
| D | **禁止连续 OOF 搜权**：融合为**开跑前预注册的离散规则集** + **嵌套折选规则**；`no_oof_weight_search_on_reported_oof=true` | 是 |
| E | **B6（及 B5）冻结未篡改**：`artifacts/b7_audit/b6_freeze_check.json` → `verdict=PASS`；B6 pooled 仍为 **0.6989746962571622** | 是 |
| F | 协议声明（§5.5）全部相关键为 `true`；数据 SHA 与 §0 一致；独立复算 nested 误差 **< 1e-8** | 是 |
| G | 报告分 = **nested_oof_auc**；禁止 fold_max / seed_max / full-data 挑最高规则冒充 nested | 是 |

**稳定性参考线（影响合格等级，非单独红线）：**

| 项 | 期望 | 偏离处理 |
|---|---|---|
| 嵌套折规则一致性 | 多数折选中同一规则；`consistent_all_folds` 披露 | 折间规则剧烈抖动 → CONDITIONAL |
| 主臂 seed_mean / seed_std | 沿用 B6 参考：seed_mean 过低或 seed_std>0.010 | 最多 CONDITIONAL |
| `fold_auc_range`（nested 折） | ≤ 0.06 | 超出 → CONDITIONAL |
| max 依赖 | 见 §3.2 | CONDITIONAL 或叙事绑定 |

**硬失败（直接 REJECT，不论 nested 多高）：**

- shuffled ∉ [0.47, 0.53] 仍称有效  
- 测试集标签 / 伪标签泄漏  
- 全量 fit 再 OOF 的 TE/分箱/标准化/词表（基座臂）  
- **在报告用的同一 OOF 上连续搜融合权重**（含未嵌套的 grid / 无折外选的 logistic 系数）  
- 公开榜回流调参仍称盲测本地分  
- 旧数据/第三方预测冒充自研 OOF（只读引用已冻结 B5/B6/V10 预注册臂须披露来源）  
- 只报单折/单 seed/单规则最高分  
- **B5 或 B6 冻结文件被改**  
- **伪造 nested**（用 full-data 选规则分数冒充 nested）  
- 事后发明规则集 / 臂集合冒充预注册  

---

## 3. 嵌套融合 · max 依赖 · stacking 红线

### 3.1 嵌套离散融合（允许的主路径）

- 规则集合须**开跑前**写入 `docs/B7_PLAN.md` 与/或 `protocol_declaration`（默认允许 V10 六规则：`mean, mean_2_1, power2, power3, max, rank_mean`）。  
- **允许**：在嵌套折的 train 段上，从预注册离散规则中选一条，应用到该折 valid，拼成 **nested OOF**；提交可用全量数据上按同一嵌套程序选定的规则。  
- **禁止**：连续权重网格；看完 **full OOF** 再发明新规则或挑最高规则当作 nested；用 full-data 分数替换 nested。  
- 必须落盘：每折选定规则、`nested_oof_auc`、full-data 各规则分数（仅披露，不得升格为主报）。

### 3.2 max 依赖（CONDITIONAL 触发）

- 若 **`nested_oof_auc ≥ 0.71` 仅在 `max`（或实质等价的逐点取 max）上成立**，而预注册非 max 规则均 `< 0.71`：  
  - 须证明 nested **稳定选中** max（折间一致或可解释）；  
  - 须披露 max 叙事与 `shuffled_*_max` 崩盘证据（`< 0.66`）；  
  - 监督者默认标 **CONDITIONAL PASS**（叙事绑定 max），**不得**写成与规则无关的稳健达标。  
- 用户“无过拟合无作弊才交付 0.71”默认解释下：**CONDITIONAL 不自动构成可交付 0.71**；须同时给出 `closest_honest_nested` 与条件清单。

### 3.3 Stacking / 元学习器规则

| 允许 | 禁止 |
|---|---|
| 预注册的**离散**规则嵌套选择 | 在报告 OOF 上拟合 **连续** blend 权重并直接报该 OOF |
| **折内嵌套** meta（仅用该折 train 拟合 meta，预测 valid）且 meta 族/超参开跑前写死 | 用全量 OOF 拟合 Logistic/线性/GBM meta 再报同一 OOF |
| 基座臂 OOF 作为特征，但 TE/编码仍须折内 | 全局 TE；或把 stacking 当作绕过“禁止连续搜权”的后门 |
| 披露 meta 系数 / 选中规则仅作透明性 | 多组 meta 超参按同一 OOF 网格挑选后只报最好 |

若 stacking 的报告分 **不是** 严格嵌套 OOF，则该分 **不得** 进入门槛 A；最多作探索臂披露。

---

## 4. 合格 / 有条件合格 / 驳回

### 4.1 PASS

同时满足：§2 门槛 A–G；无硬红线；nested 非“仅靠 max 才过线”（或 max 虽选中但至少一条非 max 预注册规则的 nested/对照亦 ≥0.71）；强制 metrics 字段齐全；冻结核验 PASS；独立复算通过。

→ 批准宣称：“诚实本地嵌套 OOF ≥ 0.71”。

### 4.2 CONDITIONAL

`nested ≥ 0.71` 且无硬红线，但出现：强 max 依赖；嵌套折规则剧烈不一致；主臂强 bagging / 早停卡线缺对照；强制字段部分缺失但不影响已复算主分；预注册表述含糊但离散嵌套可证。

→ 对外须披露条件项；默认 **`deliver_0_71_allowed=false`**。

### 4.3 REJECT

- `nested_oof_auc < 0.71`；或任一硬红线；或作弊/泄漏抬分。  

驳回时强制输出：

```text
verdict: REJECT
claimed_or_attempted: ...
closest_honest_nested_oof_auc: <独立复算的合法 nested>
why_not_0.71: <一条主因>
red_lines_hit: [...]
b6_freeze_check: PASS/FAIL
```

不得用 CONDITIONAL 话术掩盖 REJECT。

### 4.4 分数权威序（写死）

```text
权威分 = nested_oof_auc（预注册离散规则 + 嵌套折选）
参考分 = full_data_selected_rule_auc；各臂 OOF；seed_mean±std
禁止当分 = max(fold), max(seed), full-data 挑最高规则, 非嵌套连续搜权/stacking, 公开榜, 旧数据OOF, 被篡改的B5/B6分
未达标时交付分 = closest_honest_nested_oof_auc
```

---

## 5. 强制 metrics 字段清单

B7 候选交付的 `artifacts/b7_*/metrics.json`（或等价）**必须**包含下列关键项；缺关键项 → 不得 PASS。

### 5.1 身份与冻结绑定

| 字段 | 要求 |
|---|---|
| `experiment_id` | 唯一；建议前缀 `b7_` |
| `git_commit` / `git_branch` | 训练代码绑定；分支应为 `cursor/b7-push-auc071-a5f5` 或其记录的实验提交 |
| `data_sha256` | train/test(/submit) 与 §0 一致 |
| `protocol_id` | `IA-AUC710-B7-v1` |
| `b6_freeze_ref` | 引用 B6 pooled=`0.6989746962571622` 或冻结 metrics SHA |
| `b6_freeze_untampered` | `true`（并附 `artifacts/b7_audit/b6_freeze_check.json`） |
| `protocol_declaration` | 见 §5.5 |

### 5.2 嵌套主分

| 字段 | 要求 |
|---|---|
| `nested_oof_auc` | **主报告分** |
| `nested.selected_rule` / 每折 `votes` | 必填 |
| `nested.consistent_all_folds` | 布尔；须披露 |
| `nested.full_data_scores` | 各预注册规则 full-data 分（披露） |
| `gate_0_71` | `nested_oof_auc >= 0.71` |
| `arms` / `arm_aucs` | 融合臂名单与各臂 OOF |
| `fusion_rules_preregistered` | 完整列表 |
| `rule_selection` | 如 `nested_5fold` |

### 5.3 泄漏控制与诚实对照

| 字段 | 要求 |
|---|---|
| `shuffled_oof_auc` 或臂级 shuffled | 合格带判定 |
| 若用 max：`shuffled_plus_max_auc` / `shuffled_plus_max_pass` | max `< 0.66` 才算对照过关 |
| `early_stopping_on_valid` | 基座臂须显式披露 |
| `pseudo_label` | `false` |
| `no_oof_weight_search_on_reported_oof` | `true` |
| stacking 时：`nested_meta=true` 与 meta 超参预注册证明 | 否则 stacking 分不得作主报 |

### 5.4 工件

`predictions`（oof/y/test 或可复算等价物）、`metrics.json`、`submission.csv`、一行复现命令。

### 5.5 协议声明（须逐键 true/false）

```text
PROTOCOL_DECLARATION:
  no_test_label_leak: true/false
  no_test_pseudo_label: true/false
  fold_local_encoding_only: true/false
  no_oof_weight_search_on_reported_oof: true/false
  no_continuous_fusion_grid: true/false
  fusion_rules_preregistered: true/false
  rule_selection_is_nested: true/false
  reported_score_is_nested_oof: true/false
  no_public_lb_tuning_claimed_as_blind: true/false
  no_legacy_or_thirdparty_preds_as_self_oof: true/false
  shuffled_near_chance: true/false
  b5_freeze_untampered: true/false
  b6_freeze_untampered: true/false
  seeds_preregistered_no_cherry_pick: true/false
  early_stopping_disclosed: true/false
  max_dependency_disclosed: true/false   # 若过线仅靠 max 必须 true
  stacking_is_nested_or_unused: true/false
```

全部为 `true` 才允许进入 PASS 候选（不适用键须在审计包中显式 N/A 并说明）。

---

## 6. 复核流程

1. **开跑前**：`b6_freeze_check.json` PASS；规则集写入计划/`protocol_declaration`。  
2. **实验中**：复核官不提供抬分代码；不发明分数。  
3. **等待态**：若尚无 `artifacts/b7_*/metrics.json` 满足 `nested_oof_auc ≥ 0.71`，保持 `artifacts/b7_audit/waiting_status.json`，**不进行终审放行**。  
4. **宣称达标时**：提交 `[AUDIT_PACKET_B7]`（§7）+ 完整 artifacts。  
5. **终审**：独立复算 nested、核对 B6/B5 冻结 SHA、核对预注册与 stacking 合法性 → PASS / CONDITIONAL / REJECT。  
6. **未达标**：只认证 `closest_honest_nested_oof_auc`。

---

## 7. 审核签字格式

```text
[AUDIT_PACKET_B7]
protocol_id: IA-AUC710-B7-v1
experiment_id: ...
nested_oof_auc: ...
gate_0_71: true/false
selected_rule / per_fold_votes: ...
consistent_all_folds: true/false
max_dependency: true/false
shuffled_oof_auc / shuffled_plus_max_auc: ...
fusion_rules_preregistered: [...]
rule_selection: nested_...
stacking: unused | nested_meta (params=...)
b6_freeze_check: PASS/FAIL (pooled still 0.6989746962571622)
b5_freeze_untampered: true/false
protocol_declaration: all true?
data_sha256_match: true/false
oof_recomputed_nested_auc: ...
closest_honest_nested_oof_auc: ...
auditor_verdict: PASS | CONDITIONAL | REJECT
deliver_0_71_allowed: true/false
```

`deliver_0_71_allowed=true` **仅当** `auditor_verdict=PASS`。

---

## 8. 开跑前冻结核验

机器可读结果：`artifacts/b7_audit/b6_freeze_check.json`

期望摘要：

```text
verdict: PASS
expected_closest_honest_pooled_oof_auc: 0.6989746962571622
actual_pooled_oof_auc:                 0.6989746962571622
freeze_files_present: true
scores_unaltered: true
compared_against: origin/cursor/b6-push-auc070-a5f5
```

终审前须**重跑**同一核验；若 FAIL → 中止 B7 0.71 交付审查。

等待态产物：`artifacts/b7_audit/waiting_status.json`（无 nested≥0.71 候选时不得伪造终审意见）。

---

## 9. 复核官立场

1. 本文件是 **B7 放行标准**，不是抬分方案。  
2. 默认怀疑任何“刚好 ≥0.71”且缺 nested 折表 / shuffled(+max 崩盘) / 预注册证明 / 冻结核验的结果。  
3. B6 的 closest-honest 认证与 REJECT(0.70) **不自动**延伸为 B7 PASS。  
4. 强依赖 `max` 或非嵌套 stacking 不得无条件放行。  
5. 复核官不修改模型代码；只接受 / 有条件接受 / 驳回，并在驳回时报告最接近的诚实 nested 分。  
6. 版本：`IA-AUC710-B7-v1`；冻结基线或数据变更后必须升版并重跑 §8。

---

**协议生效**：本文档合入仓库后，后续所有“本地 OOF≥0.71”声明均以本协议裁决；主进程达标后须再提请本复核官**终审**。
