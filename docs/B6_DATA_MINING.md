# B6 数据挖掘洞见（新 train.csv / test.csv）

> **结论先行：** 相对冻结 B5 pooled OOF **0.69817454**，新数据上仍有可吃信号。  
> 主抬分口：**(1) `cond/days` 比分箱交叉**、**(2) `t3_sfx×code×days`**、**(3) `w_pair×days`**、**(4) 固定天数窗**。  
> 1-seed CatBoost 探针（fold-local FE、无 TE）：B5 **0.68800** → B5+缺口猫特征 **0.69108**（**+0.00308**）。按 B5 的 1seed→8seed 抬升量级外推，**诚实冲 0.70 可执行**。  
> TE 仅作诊断上界；最终推荐 **CatBoost 原生字符串交叉**，避免全局/泄漏 TE。

---

## 0. 数据与基线（实测）

| 项 | 值 |
|---|---|
| train / test | 14,930 × 45 / 6,398 × 44 |
| 索赔率 | **10.020%**（1496/14930） |
| B5 冻结 8seed pooled OOF | **0.69817454** |
| 本报告重算 B5 OOF AUC | **0.69817454**（`artifacts/b5_8seed/predictions.npz`） |
| 协议 | 仅新 CSV；折内 FE；TE=5-fold OOF + prior10 平滑（诊断） |

**B5 已吃：** 丢 `x0–x18`；`x19/x20` 类别；`days×region/source/x19/x20/age_range`；`days×cond` 联合分箱；dual_category³=`region/source/x19/x20/age_range/livability/version/month`；数值 `cond_over_days` / `days×cond` / `log_days`。

**B5 未吃满（本报告焦点）：** `ratio` 的**类别交叉**、`t3_sfx`/`code` 进 days 交叉、`w_pair`、`age_coarse`、固定天数窗、`version×days`、条款对 `t/c/r_pair×days`。

复现：

```bash
python3 scripts/b6_data_mining.py
# → artifacts/b6_eda/*.csv + mining_summary.json
```

---

## 1. B5 尚未吃满的信号

### 1.1 `condition/days` 比（数值已有，交叉未满）

| 特征 | AUC / OOF-TE | mean_count | vs B5 OOF corr | 解读 |
|---|---:|---:|---:|---|
| `cond_over_days`（数值） | AUC_abs **0.6048** | — | −0.169 | **比单轴 days(0.593) 更强**；B5 enrich 已含数值，但树未必充分切开比×地理 |
| `ratio_q5` | OOF-TE **0.5915** | 2488 | 0.498 | 五分位索赔率 **13.93%→4.56%**（单调） |
| `ratio_q5×region` | **0.6056** | 128 | 0.478 | 健康；row_lt20=1.6%；B5 midband AUC **0.540** |
| `ratio_q5×source` | **0.6008** | 226 | 0.508 | 健康 |
| `ratio_1k_q5×region` | **0.6056** | 128 | 0.478 | 与 ratio×region 几乎等价 |

B5 的 `days_q5×cond_q5` OOF-TE=**0.6121** 仍是最上界，但 **ratio 交叉是不同切割**：强调「单位暴露车况强度」，在 B5 分数 midband(0.08–0.25) 上仍有分离力。  
**推荐：** 保留数值比 + 新增 `ratio_q5` / `ratio_q5|region|source` 为 **cat 字符串**（勿 TE）。

### 1.2 `t3_sfx × code × days`（三阶里最干净）

| 交叉 | OOF-TE | mean_count | n\<20 行占比 | gap(leaky−OOF) |
|---|---:|---:|---:|---:|
| `t3_sfx×code×days_q5` | **0.6019** | **426.6** | **0.000** | 0.017 |
| `t3_sfx×code×days_fixed` | 0.5968 | 304.7 | 0.000 | 0.026 |
| `t3_sfx×code` | 0.5115 | 2133 | 0.000 | 0.014 |
| `t3_sfx×days_q5` | 0.5979 | 995 | 0.000 | 0.008 |

`t3_sfx×code` 主效应弱，但 ×days 后上界跳到 ~0.60 且 **35 格、零小样本**。B5 dual 列**不含** `t3_sfx`/`code`，days 交叉也不含 —— **明确缺口**。  
索赔率（n≥20）：`E|B` 10.88%、`E|C` 10.89% vs `P|B` 8.01%。

### 1.3 `age_coarse`（合并 age≥8）

| 交叉 | OOF-TE | mean_count | corr(B5 OOF) |
|---|---:|---:|---:|
| `age_coarse` | 0.5210 | 1866 | **0.081** |
| `age_coarse×days_q5` | 0.5941 | 373 | 0.438 |
| `age_raw×days_q5`（B5 已有） | 0.5947 | 325 | 0.438 |

与 B5 的 `age_raw×days` 几乎同界；**单变量 age_coarse 与 B5 OOF 低相关（0.08）**，适合异构臂主效应，主臂用 coarse 替代 raw 防 9/10 稀档即可。索赔：age8=**16.57%** vs age1=8.36%。

### 1.4 `w_pair = w1_w2`（条款开关）

| 交叉 | OOF-TE | mean_count | corr(B5 OOF) |
|---|---:|---:|---:|
| `w_pair` | 0.5135 | 3733 | **0.065** |
| `w_pair×days_q5` | **0.5960** | 747 | 0.495 |
| `w_pair×days_fixed` | 0.5956 | 533 | 0.488 |
| `w_pair×ratio_q5` | 0.5992 | 622 | 0.508 |

四态索赔率：`1_0` **8.98%**（最低）vs `1_1` 11.0% / `0_1` 10.52%。B5 **完全未用** w1/w2。  
**主臂必加 `w_pair` + `w_pair|days_bin`；低相关单变量适合异构臂。**

### 1.5 固定天数窗（非分位数）

窗：`≤700 / 700–2500 / 2500–5k / 5k–7k / 7k–9k / 9k–10k / >10k`（贴齐索赔率拐点）。

| 交叉 | OOF-TE | mean_count | row_lt20 | gap |
|---|---:|---:|---:|---:|
| `days_fixed` | 0.5916 | 2133 | 0 | 0.009 |
| `days_fixed×cond_q5` | **0.6029** | 355 | 0.004 | 0.024 |
| `days_fixed×source` | 0.6008 | 194 | 0.004 | 0.031 |
| `days_fixed×region` | 0.6006 | 107 | 0.026 | 0.050 |

索赔率：`d9k_10k` **14.08%**、`d7k_9k` 13.91% vs `d700_2500` 5.53%、`d0_700` 6.37%。  
与 qbin 相关但边界不同 → 给树第二种暴露离散化；`×region` gap 偏大，**优先 `days_fixed` 本体 + `×cond/source`，region 用 CatBoost 原生而非 TE。**

### 1.6 其它缺口（次优先）

| 交叉 | OOF-TE | mean_count | 备注 |
|---|---:|---:|---|
| `code×days_q5` | 0.6024 | 747 | 零稀疏；code 未进 B5 days 交叉 |
| `version×days_q5` | 0.5762 | 157 | gap 0.045；dual 有 version 但无 ×days |
| `cond_q5×source` | 0.5693 | 226 | **corr(B5)=0.297**（异构友好）；midband AUC 0.538 |
| `t_pair×days_q5` | 0.5941 | 747 | 条款对；零稀疏 |
| `r_pair×days_q5` | 0.5956 | 747 | 同上 |
| `c_pair×days_q5` | 0.5912 | 747 | 弱于 w/t |

---

## 2. 高 OOF-TE 上界且不稀疏的新增交叉

筛选：`sparse_risk=False`、`mean_count≥50`、`row_share_lt20≤5%`、`auc_oof_te≥0.55` → **40** 条（见 `healthy_high_upperbound.csv`）。

**Top（相对 B5 增量视角，非单纯重复 days×region）：**

| 优先级 | 交叉 | OOF-TE | mean_count | gap | 建议用法 |
|---|---|---:|---:|---:|---|
| P0 | `ratio_q5×region` | 0.6056 | 128 | 0.037 | CatBoost cat 字符串 |
| P0 | `days_fixed×cond_q5` | 0.6029 | 355 | 0.024 | 固定窗×车况 |
| P0 | `t3_sfx×code×days_q5` | 0.6019 | **427** | 0.017 | **最干净三阶** |
| P0 | `code×days_q5` | 0.6024 | 747 | 0.010 | 或由上式覆盖 |
| P0 | `ratio_q5×source` | 0.6008 | 226 | 0.025 | 与 region 二选一或都留 |
| P1 | `w_pair×days_q5` | 0.5960 | 747 | 0.011 | 条款×暴露 |
| P1 | `w_pair×ratio_q5` | 0.5992 | 622 | 0.014 | 条款×车况强度 |
| P1 | `days_fixed×source` | 0.6008 | 194 | 0.031 | 固定窗×车型 |
| P1 | `t3_sfx×days_q5` | 0.5979 | 995 | 0.008 | 轻量后备 |
| P2 | `age_coarse×days_q5` | 0.5941 | 373 | 0.019 | 用 coarse 替换 raw |
| P2 | `version×days_q5` | 0.5762 | 157 | 0.045 | 仅原生 cat，禁止 TE |
| P2 | `cond_q5×source` | 0.5693 | 226 | 0.031 | 异构臂 |

> 上界 0.60–0.61 **低于** B5 已有 `days×cond`(0.612)，但方向正交；CatBoost 探针显示堆叠仍有 **+0.003** 诚实增益。

---

## 3. 与 B5 低相关、适合异构臂的特征块

以候选 OOF-TE 分数与 B5 8seed OOF 的 Pearson corr 衡量冗余（越低越适合第二臂）。

### 3.1 低相关「条款 / 人龄 / 车况侧」块（推荐异构臂 A）

| 块 | 代表特征 | corr(B5 OOF) | 单变量/交叉强度 |
|---|---|---:|---|
| 条款开关 | `w_pair` | **0.065** | 弱单变量；×days→0.596 |
| 人龄粗分 | `age_coarse` | **0.081** | 0.521；高龄高赔 |
| 车况×动力 | `cond_q5×source` | **0.297** | 0.569；midband 仍有效 |
| 条款×暴露 | `t_pair×days_q5` | 0.462 | 0.594 |
| ratio 本体 | `ratio_q5` | 0.498 | 0.592；midband 0.523 |

**异构臂构图建议（与 B5 主臂等权/rank 融合）：**

1. **保留** days/condition 连续 + log，但 **弱化** dual³ 的 region×source×version（与 B5 高度同构）。  
2. **强化** `w_pair`、`t_pair`/`r_pair`、`age_coarse`、`cond_bin×source/car`、`ratio_q*`。  
3. 训练差异：`grow_policy=Lossguide` 或更深/更强 L2（B6_PLAN 预注册），制造误差不相关。  
4. 同模型硬塞全部缺口猫特征时，OOF corr(B5,B5+gap)=**0.977**（探针）——说明 **单模型堆特征不够异构**，必须第二臂改特征哲学或生长策略。

### 3.2 不适合当「异构」的块

- `days_q*×region/source/x19`：与 B5 同构，corr~0.47–0.48 且已在主臂。  
- `x19/x20` 再交叉：B5 已吃；x19≈V 共线。  
- `livability×*`：对 region R²≈0.99，伪异构。

---

## 4. 「不要做」的过拟合交叉清单

判定：`gap(leaky−OOF)≥0.04` 或 `row_share_lt20≥5%` 或高基数低 mean_count。完整表：`overfit_cross_blacklist.csv`。

| 交叉 | OOF-TE | gap | mean_count | row_lt20 | 判定 |
|---|---:|---:|---:|---:|---|
| `region×car×version` | 0.525 | **0.215** | 7.7 | **41%** | 禁止 TE；dual 已够，勿再显式三阶展开 |
| `t3_full×code×days_q5` | 0.573 | **0.183** | 12.4 | 31% | 禁止；用 `t3_sfx` 代替全串 |
| `region×days_q5×version` | 0.570 | **0.165** | 10.6 | 35% | 禁止 TE；解释力虚高 |
| `t3_full×days_q5` | 0.572 | **0.148** | 19.4 | 19% | 禁止 |
| `car×version×days_q5` | 0.561 | **0.142** | 16.8 | 27% | 禁止 |
| `t3_full×code` | 0.523 | **0.100** | 58 | 6.6% | 禁止 TE |
| `source×version` / `car×version` | 0.511 | **0.080** | 74 | 6.5% | 仅原生 cat |
| `age_raw×version` | 0.512 | 0.074 | 93 | 2.9% | 稀龄×版本，合并或丢 |
| `month×version` | 0.516 | 0.062 | 69 | 5.0% | 低优先 |
| `t3_sfx×region×days_q5` | 0.606 | 0.061 | 56 | 6.2% | 上界诱人但稀疏；勿 TE |
| `days_fixed×region` | 0.601 | 0.050 | 107 | 2.6% | gap 偏大；仅原生 |
| `B5:days_q5×x20` | 0.599 | **0.103** | **29** | — | 已在 B5；勿再 TE 加码 |

**总原则：** 高基数 `t3` 全串、任意 `*×version` 细交叉、region×days×version 三阶 —— **只许 CatBoost 有序目标统计在树内学，禁止显式全局/OOF-TE 喂入。**

---

## 5. B6 可执行特征清单（按优先级）

### P0 — 主臂增量（相对 B5 直接加，CatBoost 原生 cat）

1. **`ratio_q5` / `ratio_q10`**（fold 内分位）+ 字符串交叉 **`ratio_q5|region`**、**`ratio_q5|source`**  
2. **`t3_sfx|code|days_q5`**（及组件 `t3_sfx`、`code` 进入 dual 候选列）  
3. **`w_pair`** + **`w_pair|days_q5`**（必要时 `w_pair|ratio_q5`）  
4. **`days_fixed`** + **`days_fixed|cond_q5`**、**`days_fixed|source`**（region 交叉谨慎）  
5. dual_category 列扩展：在 max_cross≤6 下加入 **`t3_sfx`/`code`/`w_pair`**，可换出冗余的 `month` 或与 source 共线的 `x19_cat`

### P1 — 稳健替换 / 轻量增强

6. **`age_coarse`** 替换 `age_range` 参与交叉（合并 ≥8）  
7. **`code|days_q5`**（若未做 P0-2 的完整三阶）  
8. **`version|days_q5`** 仅作原生 cat（不做 TE）  
9. **`t_pair|days_q5`** / **`r_pair|days_q5`**（条款侧补充）

### P2 — 异构臂专用（低相关块）

10. 臂特征中心：`w_pair`、`age_coarse`、`cond_bin×source`、`ratio_q*`、条款对；弱化 region×source×version 三阶  
11. 算法异构：Lossguide / 不同 depth-l2 / 固定 iteration 无 OD（见 B6_PLAN）  
12. 与 B5 主臂 **等权概率或等权 rank** 融合（禁止 OOF 搜权）

### 明确不做

- 全局 TE；`t3_full×code(*×days)`；`region×days×version`；`car×version(*×days)`  
- 恢复 `x0–x18`；`region×livability` 细交叉；稀有 `month×version`  
- 用泄漏 TE 或测集伪标签冲数

### 诚实抬分证据（新数据实测）

| 实验 | pooled OOF AUC | 说明 |
|---|---:|---|
| B5 冻结 8seed | **0.69817** | 基线 |
| 本环境 1seed B5（iter≤900） | 0.68800 | 探针对照 |
| 1seed B5 + P0/P1 缺口猫特征 | **0.69108** | **+0.00308** |
| 两 OOF 等权混合 | 0.69064 | corr=0.977，单模型已吃大部分 |

外推：B5 历史约 1seed~0.690 → 8seed **0.6982**（+~0.008）。若缺口特征在多种子下保持 ~+0.002–0.003，则 **8seed 目标带约 0.700–0.701**。需正式 B6 多 seed 复验；本报告不替代最终训练交付。

---

## 6. 产物索引

| 路径 | 内容 |
|---|---|
| `docs/B6_DATA_MINING.md` | 本报告 |
| `scripts/b6_data_mining.py` | 可复现挖掘脚本 |
| `artifacts/b6_eda/cross_oof_te_upperbound.csv` | 全部交叉 OOF-TE / 稀疏 / vs B5 corr |
| `artifacts/b6_eda/healthy_high_upperbound.csv` | 健康高上界子集 |
| `artifacts/b6_eda/overfit_cross_blacklist.csv` | 过拟合黑名单 |
| `artifacts/b6_eda/hetero_candidates_by_b5_corr.csv` | 按与 B5 相关排序 |
| `artifacts/b6_eda/numeric_auc_vs_b5.csv` | 数值 AUC / 残差相关 |
| `artifacts/b6_eda/claim_rate_slices.csv` | 关键切片索赔率 |
| `artifacts/b6_eda/catboost_gap_probe.json` | 1seed CatBoost 增量探针 |
| `artifacts/b6_eda/mining_summary.json` | 汇总 JSON |

**未改动**任何 `docs/b5_frozen/`、`artifacts/b5_frozen/`、B5 训练语义。
