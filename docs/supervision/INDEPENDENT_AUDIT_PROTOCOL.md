# 车险索赔 AUC 竞赛 · 独立审核协议（IA-AUC698-v1）

> **角色**：独立审核者（Independent Auditor）。不参与建模实现、不代写调参、不代写提交。  
> **监督对象**：后续团队在 `/workspace` 用**新数据** `train.csv` / `test.csv` 训练，目标本地诚实 OOF AUC ≥ **0.698**。  
> **效力**：主 Agent / 建模方每次宣称达标或晋级提交前，必须按本协议自检并接受复核；任一红线 FAIL → **驳回**，不得包装为“盲测本地分”。

---

## 0. 数据基本事实（本轮已核对）

| 项 | 值 |
|---|---|
| train 行数 | **14930** |
| test 行数 | **6398** |
| submit_sample 行数 | **6398** |
| 特征列数（除 id/label） | **43** |
| 正样本率 | **≈0.10020**（1496 / 14930） |
| label 位置 | **仅 train**；test **无** label |
| train∩test id 重叠 | **0** |
| 精确特征行跨集重叠 | **0** |
| submit 结构 | `id,label`；id 集合/顺序与 test 一致；占位 label 全为 0.5 |
| DATA_GATE | **PASS** |

文件 SHA-256：

- `train.csv` = `494a61073a0438f692914c4868db31df1171e662348e0024e06b120d08d44f28`
- `test.csv` = `d6ffd26bd4873fa09f6fac361f59170a880e88e331a01d7a6356bd9184ce55ec`
- `submit_sample.csv` = `83cb0263cc5729f61d0e05c68d673dc3f21b41c24bad68afa35159859054c4bf`

机器可读核对：`artifacts/data_gate/new_data_integrity.json`  
复跑脚本：`python3 scripts/00_check_data_gate.py`

### 旧分支警示（必须继承，禁止遗忘）

来源：`cursor/s06-v2-audit-agent-a-1b4a`、`cursor/push-oof072-from-069-20260806-dd73`、`cursor/cb-semantic-oof069-20260806-v1-dd73` 等。

1. **旧数据 ≠ 新数据**：旧轮 train≈21328 / test≈10000；SHA 不同。旧 OOF、旧提交、旧 `.npy` **一律不可**当作本轮“自研 OOF”。
2. **TE / WOE 非折内污染**：历史上“全量 fit 再 OOF”曾导致污染率 100%；公开侧出现本地~0.713 → 公开~0.698（高估≈0.015）先例。
3. **第三方预测冒充自研**：曾用第三名 / MiniMax 包 `oof_*.npy` 融合；训练脚本缺失、shuffled 缺失 → 不得宣称自研诚实分。
4. **OOF 搜权 / 选模偏置**：在同一 OOF 上搜融合权重或挑“幸存正增益臂”，再报告该 OOF → 信息泄漏；等权可消搜权，**消不掉选谁进融合的偏置**。
5. **单折偶然高分**：旧 CatBoost 语义日志中曾出现单折 valid≈0.725，而 seed/pooled 明显更低；禁止把单折/单 seed 当最终分。
6. **holdout / 公开榜回流调参**：用外层 holdout 或公开榜反馈调参后，仍宣称“盲测本地分” → 违规。
7. **train–valid gap / holdout 落差**：旧案 LGB gap 中位≈0.29、holdout 低于 OOF≈0.033 → 判定明显过拟合；本协议继续要求披露 gap。

---

## 1. 强制验证清单（每次实验必须报告的字段）

建模方每次实验产物（`metrics.json` 或等价报告）**必须**包含下列字段；缺一不可宣称达标。

### 1.1 身份与数据绑定

| 字段 | 要求 |
|---|---|
| `experiment_id` | 唯一实验名 |
| `git_commit` | 训练代码 commit |
| `data_sha256` | train/test/submit 三文件 SHA，须与本协议第 0 节一致（或显式声明数据更新并重跑门禁） |
| `train_rows` / `test_rows` | 须为 14930 / 6398（除非数据更新） |
| `pos_rate` | train label 均值 |
| `feature_list_hash` 或完整特征清单 | 可复现 |
| `protocol_declaration` | 见 §2.4，逐条布尔声明 |

### 1.2 CV 协议

| 字段 | 要求 |
|---|---|
| `cv_scheme` | 必须为分层 K 折，推荐 `StratifiedKFold` |
| `n_splits` | ≥ **5** |
| `seeds` | ≥ **3** 个固定种子列表（完整报告，禁止只报最好 seed） |
| `seed_aucs` | 每个 seed 的 **seed-level OOF AUC**（该 seed 下全样本 OOF） |
| `seed_mean` / `seed_std` | 多种子等权均值与标准差 |
| `pooled_oof_auc` | 多种子 OOF 概率**等权平均**后，对全 train 计算的 AUC（**主报告分**） |
| `fold_aucs` | 每 seed × 每 fold 的 valid AUC（禁止只报最高折） |
| `fold_auc_min` / `fold_auc_max` / `fold_auc_range` | 披露折间波动 |

### 1.3 预处理 / 泄漏控制声明

| 字段 | 要求 |
|---|---|
| `target_encoding` | `none` 或描述；若使用，必须证明**严格折内嵌套** |
| `binning_scaler_pca_freq` | 一律仅在**当前折训练集** fit；valid/test 只 transform |
| `cat_vocab_fit_scope` | 类别词表 / 频率 / 交叉组合映射的 fit 范围 |
| `early_stopping_on_valid` | 是否用折内 valid 早停（若是，必须披露；见 §3） |
| `pseudo_label` | 必须为 `false`；禁止用 test 伪标签（尤其禁止任何 test label） |
| `external_predictions_used` | 是否引入旧数据/第三方 OOF/test 预测；若 true → **不得**计为自研 OOF |
| `blend_weights` | 必须为**预注册等权**或非 OOF 目标外验证得到；禁止“在报告用的同一 OOF 上搜权” |
| `public_lb_feedback_used` | 是否看过公开榜后调参；若 true → 不得宣称盲测本地分 |

### 1.4 诚实性对照实验

| 字段 | 要求 |
|---|---|
| `shuffled_oof_auc` | 至少 1 个 seed、相同流水线、打乱 train label 后的 OOF AUC |
| `shuffled_pass` | 是否落入允许区间（见 §2） |
| `pred_mean_oof` / `pred_mean_test` | 预测均值；相对 pos_rate 的偏离需披露 |
| `train_valid_gap_median`（若可得） | 训练折 AUC − 验证折 AUC 的中位数 |

### 1.5 复现工件

必须落盘并可被独立重算：

- `oof_pred.npy` 或 `predictions.npz`（含 `oof`, `y`, 建议含 `test`）
- `metrics.json`（含本清单全部强制字段）
- `submission.csv`（若产出提交）
- 训练入口命令一行可复现

审核方抽检：用保存的 `oof` 与 `y` **独立** `roc_auc_score` 复算，误差须 < 1e-8。

---

## 2. 合格门槛（宣称本地 OOF ≥ 0.698 时）

### 2.1 主门槛（全部满足才可 PASS）

| # | 门槛 | 判定 |
|---|---|---|
| A | **pooled OOF AUC ≥ 0.698** | 多种子 OOF 概率等权平均后的全样本 AUC |
| B | CV：**≥ 5 折分层** × **≥ 3 种子** | 种子等权；禁止只取最好 seed |
| C | **seed_mean ≥ 0.693** 且 **seed_std ≤ 0.010** | 防止单种子虚高；std 过大则 RISK |
| D | **shuffled OOF AUC ∈ [0.48, 0.52]** | 推荐目标带；宽松硬失败带见下 |
| E | **协议声明全部为 true**（§2.4） | 任一 false → FAIL |
| F | **非单折偶然**：`fold_auc_max` 不得单独作为报告分；报告分必须是 pooled（或事先注册的 seed_mean，且不得高于 pooled 包装） |

**硬失败（直接 FAIL，不论 pooled 多高）：**

- `shuffled_oof_auc` ∉ **[0.47, 0.53]** 仍宣称模型有效  
- 存在测试集标签泄漏 / 伪标签用测试 label  
- 使用旧数据预测文件或第三方提交当“自研 OOF”  
- 全量数据 fit 再 OOF 的 TE/分箱/标准化/词表（非折内）  
- 用 OOF 分数搜融合权重后再报告**同一** OOF  
- 公开榜反馈调参后仍宣称“盲测本地分”  
- 只报告单折 / 单 seed 最高分当作最终分  

### 2.2 seed / fold 稳定性附加线

| 等级 | 条件 | 处理 |
|---|---|---|
| PASS | 满足 §2.1，且 `fold_auc_range = max−min ≤ 0.06`（按各 fold valid AUC） | 可宣称达标 |
| RISK | pooled≥0.698，但 `fold_auc_range > 0.06` 或 `seed_std > 0.010` | 有条件：须补跑至 ≥5 seeds 或证明非泄漏 |
| FAIL | 任一红线；或仅 1 seed / <5 折 | 驳回 |

### 2.3 shuffled 细则

- 打乱对象：仅 `train.label`；特征与 id **不变**。  
- 流水线：与正式实验**同一**特征块、同一模型超参、同一折数；至少跑 **1 个正式 seed**（推荐与主实验 seeds[0] 相同）。  
- **合格带**：[0.48, 0.52]  
- **硬失败带外**：<0.47 或 >0.53 → FAIL（标签泄漏/目标泄漏/实现错误嫌疑）。  
- 禁止：打乱后关闭交叉特征 / 降复杂度 / 换模型再报 shuffled。

### 2.4 协议声明（每次实验必须逐条写出 true/false）

```text
PROTOCOL_DECLARATION:
  no_test_label_leak: true/false
  no_test_pseudo_label: true/false
  fold_local_encoding_only: true/false          # TE/分箱/标准化/词表/频率均折内
  no_oof_weight_search_on_reported_oof: true/false
  no_public_lb_tuning_claimed_as_blind: true/false
  no_legacy_or_thirdparty_preds_as_self_oof: true/false
  shuffled_near_chance: true/false
  reported_score_is_pooled_multiseed: true/false
  equal_seed_average: true/false
```

全部为 `true` 才允许进入“达标候选”；审核方有权抽查代码与数组证伪。

---

## 3. CatBoost 语义交叉方案 · 特有风险点

针对拟采用 / 已存在的配方：`raw + structured_string + days_condition + dual_category(cross_order=3, max_cross_columns=6)` + CatBoost 多种子等权（参考旧分支 `cat_semantic`，旧数据 pooled≈0.691，**不迁移到本轮**）。

### 3.1 高基数三阶交叉记忆

- `dual_category` 将多列拼成 `A|B|C` 字符串交叉；`cross_order=3` 时组合爆炸，稀有组合易被叶子记住。  
- **风险**：OOF 仍可能偏乐观（尤其早停见下）；公开泛化可能低于本地。  
- **要求**：报告 `n_features` / `n_cats`；披露交叉列名单；禁止在全量 train 上先 fit 词表再切分。

### 3.2 词表 / 频率必须折内

- `DualCategoryFeatureBlock` 的 `vocabularies_` / `frequencies_` 必须只在**当前折训练集** `fit`；valid/test 未见类 → `-1` / 0。  
- **红线**：先全量 fit `dual_category` 再 KFold = 与历史 TE 污染同类。  
- **抽查**：对 valid 中人为制造未见水平，确认映射为未知码而非改写词表。

### 3.3 早停用折内 valid（温和乐观偏差）

- 现有 `CatBoostClassifier.fit(..., eval_set=(va, y_va), use_best_model=True)` 使验证折参与模型选择。  
- **性质**：不是标签泄漏，但会使 **OOF 略偏乐观**。  
- **要求**：必须在 `early_stopping_on_valid=true` 声明；若 pooled 仅略高于 0.698（例如 <0.700），审核方可要求：固定迭代数重跑，或 nested/无 early-stopping 对照。

### 3.4 数值列中位数填充的折内性

- `prepare_for_cat` 用训练集中位数填 valid/test；必须按折计算，禁止用全量 train 中位数预计算后注入各折。

### 3.5 禁止把语义交叉当“隐式 TE”

- 交叉特征本身不含 label，但若再叠加目标编码 / 按 label 筛选交叉列 / 用 OOF 重要性删列再重报同一 OOF → 升级为选模泄漏。  
- **要求**：特征块与超参须**预注册**；基于 OOF 的特征筛选若发生，须另开 holdout 或 nested，不得污染报告分。

### 3.6 多种子等权 vs 伪提升

- 允许 seeds 等权平均提升稳定性。  
- **禁止**：看完各 seed OOF 后加权或丢弃低分 seed 再报“pooled”。  
- 旧日志教训：单折可到 ~0.72 而 pooled ~0.69；本轮同样禁止以单折冲击 0.698 叙事。

### 3.7 旧产物隔离

- 旧 `artifacts/pred_bundle/oof_v1_3rd.npy`、`oof_v7_lgbm.npy`、旧 `submission_catboost_069.csv` 等：**只读对照，不可并入本轮自研 OOF**。  
- 本轮任何融合若混入上述文件 → `no_legacy_or_thirdparty_preds_as_self_oof=false` → FAIL。

---

## 4. 建议的「可接受逼近 0.698」判定规则

### 4.1 正式达标（ACCEPT / PASS）

同时满足：

1. **≥5 折分层** × **≥3 种子等权**（推荐默认 seeds：`2026,2027,2028`；冲刺可用 4 seeds：`2026..2029`）。  
2. **`pooled_oof_auc ≥ 0.698`**（主指标）。  
3. **`seed_mean ≥ 0.693`** 且 **`seed_std ≤ 0.010`**。  
4. **`shuffled_oof_auc ∈ [0.48, 0.52]`**（且硬区间 [0.47,0.53] 内）。  
5. §2.4 协议声明全 true；数据 SHA 与本协议一致。  
6. 独立复算 OOF AUC 与报告一致（误差 < 1e-8）。  
7. 未使用旧数据/第三方预测；无 OOF 搜权；无公开榜回流宣称盲测。

→ 审核结论：**批准宣称本地诚实 OOF ≥ 0.698**。

### 4.2 有条件接近（CONDITIONAL / RISK）

出现以下之一，同时 pooled 仍 ≥0.698、无硬红线：

- 仅 3 seeds 且 `seed_std > 0.008`；或 `fold_auc_range > 0.06`  
- 仅依赖 early-stopping，且 pooled ∈ **[0.698, 0.700)**  
- CatBoost 语义三阶 + 高 `n_cats`，缺固定迭代对照  

→ **不得**无条件写成“已稳健达标”；须补：≥5 seeds，或无 early-stopping / 固定 iter 对照，或独立 15% holdout（holdout **禁止**用于调参，只用于一次确认；确认后若再调参则作废盲测声明）。

### 4.3 明确不接受（REJECT）

- pooled < 0.698  
- 或任一 §2.1 硬失败红线  
- 或把 seed_max / fold_max / 搜权后 OOF / 旧文件分数 当作 0.698 证据  

### 4.4 报告分数优先级（写死）

```text
权威分 = pooled_oof_auc（多种子等权平均概率 → 全量 AUC）
参考分 = seed_mean ± seed_std
禁止当分 = max(fold_aucs), max(seed_aucs), 搜权后OOF, 公开榜分, 旧数据OOF
```

### 4.5 审核签字格式（主 Agent 每次交付时粘贴）

```text
[AUDIT_PACKET]
experiment_id: ...
pooled_oof_auc: ...
seed_mean / seed_std: ... / ...
seeds: [...]
n_splits: ...
shuffled_oof_auc: ...
protocol_declaration: all true?
data_sha256_match: true/false
legacy_preds_used: false
oof_recomputed_auc: ...
auditor_verdict: PASS | CONDITIONAL | REJECT
```

---

## 5. 作弊 / 过拟合红线速查表

| 红线 | 审核动作 |
|---|---|
| 测试集标签泄漏 / 伪标签用测试 label | **立即 FAIL** |
| 全量 fit 再 OOF 的 TE/分箱/标准化/词表 | **立即 FAIL** |
| 用 OOF 搜融合权重再报告同一 OOF | **立即 FAIL** |
| 公开榜反馈调参后仍称盲测本地分 | **立即 FAIL**；分数降级为“榜后分” |
| 旧数据预测 / 第三方提交当自研 OOF | **立即 FAIL** |
| shuffled AUC 不接近 0.5 仍称有效 | **立即 FAIL** |
| 单折偶然高分当最终分 | **立即 FAIL** |
| 早停乐观 + 卡线 0.698 | **CONDITIONAL**，要求对照 |
| 语义三阶高基数无稳定性披露 | **CONDITIONAL** |

---

## 6. 数据门禁核对结果（本轮）

独立审核者已用 Python 核对，结论摘要：

```text
DATA_GATE: PASS
- train/test id 重叠: 0
- label 仅在 train: PASS（test 无 label 列）
- submit_sample: 列=[id,label]，行数=6398，id 与 test 集合及顺序一致，label 占位全 0.5
- 无重复 id；特征列 train/test 对齐（43 列）
- 二分类 label {0,1}，无缺失；pos_rate≈0.10020
- 精确特征行跨集重叠: 0
- 与旧分支数据 SHA/规模均不同 → 旧 OOF 证据隔离
```

详情见 `artifacts/data_gate/new_data_integrity.json`。

---

## 7. 审核者立场（给主 Agent）

1. 本协议是**放行标准**，不是建模建议书；达标路径由建模方自行选择，但必须可审计。  
2. 默认怀疑：任何“刚好 0.698+”且缺 shuffled / 多种子 / 折内编码证明的结果。  
3. CatBoost 语义交叉可以尝试，但必须按 §3 披露特有风险；**旧数据 0.691 不构成本轮进度**。  
4. 审核者不修改模型代码以提高分数；只接受/驳回/降级声明。  
5. 版本：`IA-AUC698-v1`；数据更新后须重跑 §0 门禁并升版协议附录。

---

**协议生效**：本文档合入仓库后，后续主 Agent 所有“本地 OOF≥0.698”声明均以本协议裁决。
