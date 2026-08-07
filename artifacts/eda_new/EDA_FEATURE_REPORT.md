# 新数据深度 EDA 与特征挖掘报告（车险索赔二分类）

> **硬约束声明**：本报告全部数字来自 `/workspace/train.csv` 与 `/workspace/test.csv` 实测；旧数据/旧 OOF 全部作废。  
> 目标：支撑本地诚实 AUC ≥ 0.698 的特征工程。  
> TE 仅作诊断上界，不代表最终模型必须用 TE。

---

## 0. 数据概况

| 项 | 值 |
|---|---|
| train | 14,930 × 45（含 `label`） |
| test | 6,398 × 44 |
| 正样本率 | **10.020%**（1,496 / 14,930） |
| 唯一缺失列 | `condition`：train 144（0.965%），test 65 |
| 近唯一数值 | `days`(14928)、多数 `x*`、`cc`(12285)、`max_g`(14921) |
| 低基数语义 | `region`20 / `source`11 / `version`19 / `month`13 / `livability`22 / `t3`163 / `grades`3 / `code`4 |

---

## 1. 单变量 AUC / Mutual Info / 索赔率分层

### 1.1 核心结论

- **最强单变量是 `days`**（AUC **0.5932**），远超其余字段。
- **地理/宜居必须当离散类别**：`livability` 数值 AUC≈0.495，但 OOF-TE **0.5426**；`region` OOF-TE **0.5408**。
- **`condition` 方向为负**：原始 AUC 0.4676 ⇒ abs **0.5324**（越高越安全）。
- **禁止对近唯一 `x0–x18` 做 TE**（几乎一行一码）。

### 1.2 单变量排行（节选，完整见 `artifacts/eda_new/univariate_auc_mi.csv`）

| 列 | 类型 | nunique | 主指标 | MI |
|---|---|---:|---:|---:|
| days | num | 14928 | AUC **0.5932** | 0.00694 |
| condition | num | 14714 | AUC abs **0.5324**（原始 0.4676） | 0.00323 |
| region | cat | 20 | OOF-TE **0.5408** | 0.00224 |
| livability | 伪连续/离散22 | 22 | OOF-TE **0.5426** / 数值AUC 0.495 | 0.00232 |
| t3 | cat | 163 | OOF-TE **0.5258**（insample 乐观 0.601） | 0.00667 |
| age_range | num/ord | 10 | AUC **0.5229** / OOF-TE 0.5195 | 0.00074 |
| V | num | 337 | AUC **0.5205** | 0.00439 |
| w2 | bin | 2 | AUC **0.5184** | 0.00026 |
| source | cat | 11 | OOF-TE **0.5176** | 0.00083 |
| x1 / x5 / x10 | near-unique | ~1492x | AUC 0.528 / 0.525 / 0.522 | ~0 |
| version / month / grades / code | cat | 19/13/3/4 | OOF-TE ≈0.499–0.501 | 弱 |

### 1.3 `days` 十分位索赔率（单调上升后略回落）

| days 区间 | n | claim_rate |
|---|---:|---:|
| (24.4, 722] | 1493 | **6.03%** |
| (722, 1255] | 1493 | **4.89%** |
| (1255, 2446] | 1493 | 6.43% |
| (2446, 3837] | 1493 | 9.04% |
| (3837, 5144] | 1493 | 9.71% |
| (5144, 6990] | 1493 | 10.58% |
| (6990, 8839] | 1493 | **13.93%** |
| (8839, 9464] | 1493 | **14.27%** ← 峰值 |
| (9464, 10160] | 1493 | 13.60% |
| (10160, 11781] | 1493 | 11.72% |

### 1.4 `condition` 十分位（非单调 + 两端极值）

| condition | claim_rate |
|---|---:|
| 最低十分位 (~0–0.11) | **14.67%** |
| 次低 (0.11–0.21) | 7.37% |
| 中段 | ~9–11% |
| 最高十分位 (>1.52) | **6.56%** |
| missing (n=144) | 9.03%（接近全局） |

### 1.5 关键类别索赔率

**region（高→低）**：`f09d` 11.91% … `2a36` 5.75% … `c1f5` **3.72%**（跨度 ~8pp）。  
**source**：`CAR_10|ENG_651` **15.63%** → `CAR_8|ENG_843` **5.62%**。  
**livability（离散）**：0.375→12.18%，0.116→11.91%；`0.000` 仅 **3.72%**（与 region=`c1f5` 同向）。  
**age_range**：1→8.36%，8→**16.46%**（9/10 样本极少，需合并）。

---

## 2. 风险面与 Train/Test 分布漂移

### 2.1 数值 PSI（越低越稳）

| 列 | PSI | train mean / p50 | test mean / p50 |
|---|---:|---|---|
| days | **0.00084** | 5426 / 5144 | 5448 / 5168 |
| condition | 0.00347 | 0.746 / 0.637 | 0.741 / 0.613 |
| livability | 0.00092 | 0.253 / 0.116 | 0.253 / 0.116 |

→ **几乎无漂移**，days/condition 风险面可安全迁移到 test。

### 2.2 类别重叠与 cat-PSI

| 列 | cat-PSI | 重叠 | 备注 |
|---|---:|---:|---|
| region | 0.0028 | 20/20 = 100% | 无 test-only |
| source | 0.0011 | 11/11 | 无 test-only |
| version | 0.0055 | 19/19 | 稳 |
| month | max‖Δp‖=0.0047 | 全覆盖 | 稳 |
| t3 | 0.0161 | 162/163 | train-only:`5.02P`（极稀） |

### 2.3 days × condition 风险面（5×5，摘录）

- 高 days × 低 condition：索赔率可到 **~15%**。  
- 低 days × 高 condition：可低至 **~3.2%**。  
- 交叉 OOF-TE：`days_q10×cond_q10` = **0.6180**（相对 days 单轴 +0.025）。

### 2.4 region 风险与 days/condition 均值

高赔 region（如 `f09d`）并非单纯 days 更长；region 与 livability **非双射**（49 个 region×liv 组合），两者宜同时保留。

---

## 3. source / t3 / version / month 解析结构信号

### 3.1 `source = CAR_{id}|ENG_{id}`

- **CAR↔ENG 一一对应**（11 对双射）⇒ `car_id` / `eng_id` / 原 `source` 信息等价。  
- 解析后 OOF-TE 与 source 相同：**0.5176**。  
- 推荐保留：`car_id`（或 `source` 原串）+ `CAR_*` / `ENG_*` token（给树模型/交叉用）；不必同时堆三份。

### 3.2 `t3 = {num}{suffix}`，suffix∈{E,M,P}

| suffix | n | claim | num 均值 |
|---|---:|---:|---:|
| E | 6193 | **10.74%** | 4.85 |
| M | 313 | 10.22% | 4.51 |
| P | 8424 | **9.48%** | 4.87 |

- `t3` 全串 OOF-TE **0.5258**（163 水平，均值计数≈92，可用但需平滑/低频合并）。  
- `t3_num` 数值 AUC **0.5133**；`t3_suffix` OOF **0.5079**。  
- **结构价值**：suffix 作稳定三分类；num 作连续/分箱；全串作高基数类别（CatBoost 原生或低频→OTHER）。

### 3.3 `version = v{N}`

- 单变量几乎无效：OOF-TE **0.4992**，`version_num` AUC 0.5028。  
- 但水平索赔有起伏（v7 13.4%、v11 13.6%、v8 7.1%、v16 7.0%）⇒ **不适合单独 TE，适合作为交叉第三轴的弱调节器**（见 §4；三阶带 version 多数稀疏且 AUC 回落）。

### 3.4 `month = M{N}`

- OOF-TE **0.5004**；主体 M1/M2（占 ~84%）。  
- 稀有月（M5–M12，4.35% 行）索赔 8.92% vs 主体 10.07% ⇒ 可做 `month_rare` 标志，但增益预期很小。  
- **交叉中 month 常引入稀疏且 lift≤0**（`days_q10×month` lift −0.003）。

### 3.5 其它解析

- `grades`（s/ss/sss）：OOF **0.4960**，弱。  
- `code` A/B/C/D：OOF **0.4979**；D 与 `CAR_7`/source 尾部相关嫌疑，谨慎。  
- `x19`（16 值）≈ **source 的细粒度代理**（每 source 1–2 个 x19），OOF-TE 0.5170；与 source 冗余，可留作数值或丢其一。

---

## 4. 语义交叉（2/3 阶）与泄漏 TE 上界（诊断）

协议：5-fold Stratified OOF + Laplace 平滑 TE（α=20/30），**仅诊断**。

### 4.1 二阶：必做 / 推荐 / 慎用

**S 级（高 AUC + 正 lift + 不太稀）**

| 交叉 | OOF-TE | lift vs best base | mean_count |
|---|---:|---:|---:|
| days_q10 × cond_q10 | **0.6180** | +0.0249 | 149 |
| days_q5 × cond_q5 | **0.6130** | +0.0200 | 597 |
| days_q10 × source/car | **0.6003** | +0.0072 | 136 |
| days_q10 × region | **0.6002** | +0.0071 | 75 |
| days_q10 × livability | 0.5995 | +0.0064 | 68（略稀） |
| days_q10 × t3_suf | 0.5972 | +0.0041 | 498 |
| cond_q10 × source/car | 0.5945 | +0.0416 | 136 |
| cond_q10 × region | 0.5693 | +0.0163 | 75 |
| age_range × region | 0.5569 | +0.0161 | 89 |
| region × t3_suf | 0.5510 | +0.0101 | 253 |
| t3_suf × livability | 0.5532 | +0.0106 | 237 |

**物理比替代**：`condition/(days+1)` 十分位 OOF-TE **0.5956**；`days_q5 × ratio_q5` **0.6183**（与 days×cond 同级）。

**不推荐作主交叉**：`days×version`、`days×month`、`days×grades`、`days×age_range`（lift≤0 或负）。

### 4.2 三阶

| 交叉 | OOF-TE | 稀疏？ | 建议 |
|---|---:|---|---|
| days_q5 × cond_q5 × source/car | **0.6213** | 否（mean≈54） | **首选三阶** |
| days_q5 × t3_suf × region | 0.6071 | 否 | 推荐 |
| days_q5 × livability × region | 0.6022 | 否（mean≈108） | 推荐 |
| days_q5 × grades × region | 0.6012 | 否 | 可试 |
| days_q5 × cond_q5 × region | 0.6054 | 边界（495 水平） | 可用 q5 控制基数 |
| days_q5 × region × source | 0.6134 | **是**（890 水平，mean≈17） | 仅 Cat 原生交叉/强正则 |
| 含 version 的多数三阶 | ≤0.58 | 常稀疏 | 降优先级 |

**诊断上界解读**：最好的平滑 OOF-TE 交叉约 **0.62**。最终 ≥0.698 必须靠 **树模型非线性 + 多特征堆叠**，不能指望单 TE。

---

## 5. x0–x20：丢弃 / 残差化 / 保留；PCA / 行统计

### 5.1 画像

- `x0–x18`：近唯一连续（nunique≈14907–14930），**禁止 TE**。  
- `x19`：16 值，source 细代理，可当低基数类别/数值。  
- `x20`：125 值，AUC abs 0.5106，弱。  
- 高相关簇：`x1–x5–x9–x13–x17`（corr 0.75–0.86），及 `x10–x11–x15`；与 `x19` 也中高相关。

### 5.2 建议动作

| 动作 | 列 | 依据 |
|---|---|---|
| **KEEP_raw（优先）** | x1, x5, x9(残差后↑), x10, x14, x0 | auc_abs 0.517–0.528；残差化后仍≈0.52+ |
| **KEEP 低优** | x2, x13, x15, x17, x7 | 0.510–0.516 |
| **KEEP 特殊** | **x19** | 低基数；或与 source 二选一防冗余 |
| **DROP 候选** | x3,x4,x6,x8,x12,x16,x18 | auc_abs≲0.507 且近唯一 |
| **DROP/弱留** | x20 | 弱；可进 PCA/行统计 |
| **禁止** | 任何 near-unique x* 的 TE | 必过拟合 |

残差化（对 days/condition/liv/age/car/version/month OLS）：多数列残差 AUC 变化 <0.01；`x9` 残差后 0.516→**0.524** 略升。可选对强相关簇做 **1 个代表 + 残差**，非必须。

### 5.3 PCA / 行统计增益

| 方法 | OOF/AUC |
|---|---|
| PCA PC1–10 + LogReg OOF | **0.5218**（PC1 单轴≈0.519） |
| emb_std（x0–x17） | abs **0.5256** |
| emb_mean | 0.5234 |
| x_std / x_range / x_l2 | 0.520 / 0.520 / 0.519 |

→ PCA/行统计有 **弱正增益（~0.52）**，可作为 `latent_compress` 小块，**不能替代 days×语义主线**。

---

## 6. 缺失、稀有类别、冲突指示

### 6.1 缺失

- 仅 `condition` 缺失；缺失索赔率 9.03% ≈ 全局；`condition_missing` AUC≈0.499。  
- 仍建议：`condition__missing` 指示 + 中位数填充（树模型可直接吃 NaN）。

### 6.2 冲突 / 相等指示

| 指示 | 发生率 | 相等组索赔 | 不等组索赔 | AUC |
|---|---:|---:|---:|---:|
| w1==w2 | 9.36% | 10.66% | 9.95% | 0.503 |
| t1==t2 | 9.36% | ≈10.02% | ≈10.02% | 0.500 |
| r1==r2 | 81.0% | 9.91% | 10.49% | ~0.495 |
| c1==c2 | 81.4% | 9.92% | 10.45% | ~0.496 |

- **w1/w2 本身有方向**：w2=1→10.56%，w2=0→9.14%（AUC 0.518）；w1 相反（AUC 0.483≡flip 0.517）。  
- `w1_eq_w2` 增益很小；保留 **原始 w1/w2** 优于冲突标志。  
- t1/t2/r1/r2/c1/c2：单变量弱，可作原始二元进入模型，不必强交叉。

### 6.3 稀有类别

| 列 | 水平数 | count&lt;50 水平数 | 稀有行占比 |
|---|---:|---:|---:|
| t3 | 163 | 若干尾部 | 建议 &lt;30→OTHER |
| version | 19 | 0（最小~198） | 可全留 |
| month | 13 | M5–M12 较小 | 可 `month_bucket={M0-4, rare}` |
| age_range | 10 | 9+10 共17行 | **合并为 8+** |
| region/source | 20/11 | 无极端稀有 | 全留 |

---

## 7. 极致特征工程清单（按块）

> 设计原则：以 **days 风险面 × 车型/地理语义** 为主轴；livability/region 当离散；近唯一 x* 禁 TE；version/month 弱单轴、慎交叉。

### A. `domain_parse`（结构解析）

| 特征 | 公式 | 动机 |
|---|---|---|
| `car_id` | `extract(source, r'CAR_(\d+)')` | 车型主轴；与 source 双射，供交叉 |
| `eng_id` | `extract(source, r'ENG_(\d+)')` | 引擎码；与 car 双射，二选一即可 |
| `source_raw` | `source` 原串类别 | CatBoost 原生类别 |
| `t3_num` | `to_float(extract(t3, r'[-+]?\d+\.?\d*'))` | 排量/规格数值弱信号 |
| `t3_suffix` | `extract(t3, r'[A-Za-z]+$')` ∈ {E,M,P} | 稳定三分类，交叉友好 |
| `t3_raw` | 原串；`count&lt;30→OTHER` | 保留细粒度，控稀疏 |
| `version_num` | `extract(version, r'(\d+)')` | 序数；弱单轴 |
| `version_raw` | 原串类别 | 供弱交叉 |
| `month_num` | `extract(month, r'(\d+)')` | 序数 |
| `month_rare` | `1[month ∉ {M0..M4}]` | 压缩尾部月 |
| `grades_s_count` | `count('s' in grades)` | 弱序 |
| `age_range_clip` | `min(age_range, 8)` | 合并 9/10 |

### B. `days_condition`（主风险面）

| 特征 | 公式 | 动机 |
|---|---|---|
| `days`, `log1p(days)` | `log1p(max(days,0))` | 主信号 AUC0.593 |
| `days_q5`, `days_q10`, `days_q20` | fold 内分位数分箱 | 风险分层 + 交叉原子 |
| `condition_filled` | `fillna(median_fold)` | 处理 1% 缺失 |
| `condition_missing` | `isna(condition)` | 完整性指示 |
| `neg_condition` / 原值 | 保留原值即可（树可学反向） | abs AUC0.532 |
| `cond_q5/10` | 分位数分箱 | 与 days 交叉 |
| `days_cond_prod` | `days * condition_filled` | AUC0.548 |
| `cond_over_days` | `condition / (|days|+1)` | **AUC abs 0.6048**，强物理比 |
| `ratio_q5/10` | 对 `cond_over_days` 分箱 | OOF-TE≈0.596 |
| `V_times_days` | `V * days` | AUC0.5965，弱增补 |
| `days_q{k}__X__cond_q{k}` | 字符串交叉 | TE上界0.613–0.618 |

### C. `dual_category`（语义双表示 + 交叉）

对列：`region, livability, source/car_id, t3_suffix, t3_raw, age_range_clip, version, month_bucket, grades, code`：

| 特征 | 公式 | 动机 |
|---|---|---|
| `{col}__category` | 字符串类别 | Cat 原生 |
| `{col}__freq` | fold 内频率 | 稀有度 |
| **2阶交叉（优先）** | `days_qk × {region,source,livability,t3_suf}` | lift 明确，TE上界~0.60 |
| | `cond_qk × {source,region,t3_suf}` | 条件风险面×车型/地 |
| | `age_range × region` | +0.016 lift |
| | `region × t3_suf`, `t3_suf × livability` | 地理×规格 |
| **3阶交叉（优先）** | `days_q5 × cond_q5 × source` | **上界 0.621** |
| | `days_q5 × livability × region` | 上界0.602，不稀 |
| | `days_q5 × t3_suf × region` | 上界0.607 |
| **降优** | 含 `version`/`month` 的高阶 | 稀疏或 lift 负 |
| **禁止** | 对 x0–x18、cc、max_g、days 原值 TE | 近唯一泄漏 |

### D. `numeric_physics`（数值物理）

| 特征 | 公式 | 动机 |
|---|---|---|
| `cc`, `log1p(cc)`, `cc/(|days|+1)` | — | cc 弱；比值吃 days 尺度 |
| `V`, `V*days`, `V/days` | — | V*days 略超 days |
| `max_g`, `log1p(max_g)`, `max_g/days` | — | 近唯一，**禁 TE**，仅连续 |
| `livability` **类别优先** | 当 cat，勿当纯连续 | TE0.543 vs 数值0.495 |
| `w1`,`w2`,`r1`,`c2` 等二元 | 原值 | w2 AUC0.518 |
| 可选 `w1_ne_w2` | `w1!=w2` | 极弱，可省略 |

### E. `latent_compress`（x* 压缩）

| 特征 | 公式 | 动机 |
|---|---|---|
| 保留 raw：`x1,x5,x9,x10,x14,x0,x19` | 原值 | 最强匿名信号 |
| 可选 raw：`x2,x13,x15,x17` | 原值 | 弱保留 |
| 丢弃默认：`x3,x4,x6,x8,x12,x16,x18,x20` | — | ~噪声；可进 PCA |
| `emb_mean/std/l2` | 对 x0–x17 | abs AUC~0.52–0.526 |
| `pca_1..8` | StandardScaler+PCA(fold内) | LogReg OOF~0.52 |
| **禁止** x* Target Encoding | — | 近唯一必泄漏 |

---

## 8. 冲刺 ≥0.698 的特征优先级（执行序）

1. **底座**：`days` 多尺度 + `condition` + `cond/(|days|+1)` + `days×cond` 分箱交叉。  
2. **语义**：`region`、`livability`(cat)、`source/car`、`t3_suffix`+低频合并的 `t3`。  
3. **交叉**：二阶 days/cond×{region,source,liv,t3_suf}；三阶 `days_q5×cond_q5×source` 与 `days_q5×liv×region`。  
4. **弱特征**：age_range（clip）、w1/w2、V、cc；version/month 仅低阶。  
5. **匿名**：精选 x* raw + 可选 emb 行统计/PCA；**绝不 TE**。  
6. **模型侧**：优先 CatBoost/LightGBM 吃原生类别交叉；TE 仅诊断，最终可不用。

---

## 9. 产物路径

- `artifacts/eda_new/eda_report.json` — 全量数值  
- `artifacts/eda_new/univariate_auc_mi.csv`  
- `artifacts/eda_new/parsed_feature_auc.csv`  
- `artifacts/eda_new/cross2_te_upperbound.csv`  
- `artifacts/eda_new/cross3_te_upperbound.csv`  
- `artifacts/eda_new/x_features_analysis.csv`  
- `artifacts/eda_new/run_deep_eda.py` — 可复现脚本  

---

*报告生成：仅基于新 train/test 实测；诊断 TE 上界最高约 0.621（三阶 days×cond×source）。*
