# 车险索赔二分类：新数据诊断与 CatBoost 落地报告

> 全部数字来自当前仓库 `train.csv` / `test.csv` 实测。旧分支 OOF/提交/预测包作废，仅参考代码逻辑。  
> 协议：5 折分层 · 多种子等权概率平均 · 无 TE · 无测集标签 · 无 OOF 搜权。

---

## 0. 执行摘要

| 项 | 实测 |
|---|---|
| 数据 | train 14930 / test 6398 / 正例率 **10.02%** |
| 旧配方 `cat_semantic`（1 seed） | OOF **0.68429** |
| 推荐 B5 focus（1 seed≈2026） | OOF **≈0.6902** |
| B5 × 4 seeds 等权 | pooled **0.69695**（交付脚本）/ 本诊断变体 **0.69821** |
| B5 × 8 seeds 等权（交付） | pooled **0.69817**，shuffled **0.50757 PASS** |
| 目标 ≥0.698 | **已达**（依赖多种子 bagging；单 seed 仍 ≈0.689–0.694） |

**一句话配方**：丢掉 `x0..x18` → `x19/x20` 当类别 → `days/condition` 多尺度分箱与 region/source/x19/x20/age_range 语义交叉 → CatBoost 原生三阶类别交叉 → seeds `2026..2033` 等权平均。

---

## 1. 新数据单变量 AUC Top 特征

口径说明：

- 连续量：`max(AUC, 1−AUC)`（方向无关）
- 类别：折外 OOF-TE（诊断用，**不进最终模型**）
- 交互：`condition/(days+1)`、`days×condition` 为工程特征

### 1.1 Top 表（实测）

| 排名 | 特征 | 口径 | AUC | nunique | 备注 |
|---:|---|---|---:|---:|---|
| 1 | `condition/days` | 交互数值 | **0.6048** | ≈14786 | 最强单特征信号 |
| 2 | `days` | 数值 | **0.5932** | 14928 | 近唯一但强单调风险面 |
| 3 | `log1p(days)` | 数值 | 0.5932 | 14928 | 与 days 同序 |
| 4 | `livability` | OOF-TE | **0.5426** | 22 | 须当离散类，勿当连续 |
| 5 | `region` | OOF-TE | **0.5408** | 20 | 索赔率 3.7%–11.9% |
| 6 | `x20` | LOO-TE / 作类别 | **0.5475** | 125 | 数值 AUC 仅 0.511，**必须当 cat** |
| 7 | `days×condition` | 交互 | 0.5474 | ≈14786 | |
| 8 | `condition` | 数值 abs | **0.5327** | 14713 | 越高越安全（原始 AUC≈0.468） |
| 9 | `t3` | OOF-TE | 0.5258 | 163 | in-sample TE 会虚高到 0.60 |
| 10 | `x1` | 数值 | 0.5282 | 14925 | 近唯一噪声，见 §4 |
| 11 | `x5` | 数值 | 0.5247 | 14925 | 同上 |
| 12 | `age_range` | 数值 | 0.5229 | 10 | 可进 dual |
| 13 | `x10` | 数值 | 0.5224 | 14922 | 噪声倾向 |
| 14 | `V` | 数值 | 0.5205 | 337 | 弱连续 |
| 15 | `x19` | 数值 | 0.5193 | 16 | **作 cat 更有用** |
| 16 | `w2` | 二值 | 0.5184 | 2 | |
| 17 | `source` | OOF-TE | 0.5176 | 11 | 索赔率 5.6%–15.6% |
| 18 | `cc` | 数值 | 0.5152 | 12285 | 高基数弱信号 |

### 1.2 关键分布事实

- `days` 十分位索赔率约 **4.9% → 14.3%**（峰值在高分位后略回落）
- `condition` 最低十分位索赔率 **14.7%**，最高十分位 **6.6%**；缺失约 1%，接近全局均值
- Train/Test PSI：`days≈0.0008`，`condition≈0.0035` —— **几乎无漂移**，风险面可迁移
- `x0..x18`：uniq_ratio≈0.999，MI≈0，与 `days` 弱相关（|ρ|≤0.21）—— **行级噪声**

---

## 2. 推荐 CatBoost 配方（可直接落地）

### 2.1 CV 协议

```text
StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
seeds = (2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033)   # 至少 4；冲 0.698 建议 8
融合 = 各 seed 的 OOF/test 概率 等权算术平均
early stopping: eval_set=valid fold, use_best_model=True
禁止: 全局 TE / 测集伪标签 / 用 OOF 搜融合权重
```

### 2.2 特征块

| 块 | 配置 |
|---|---|
| **enrich（行级，无标签）** | 丢 `x0..x18`；解析 `source_car/eng`、`t3_value/kind`；`x19_cat=str(x19)`、`x20_cat=str(x20)`；`log_days`、`cond_over_days`、`days_x_cond`、`log_cc`、`log_V`、`grades_n`、`month_n`、`version_n`、`condition_missing` |
| **RawFeatureBlock** | 保留 enrich 后全部列（已无 x0–x18） |
| **StructuredStringFeatureBlock** | `columns=["source","t3","region"]` |
| **DaysConditionFeatureBlock** | `quantile_bins=(5,10,20)`；`categorical_cross_columns=("region","source","x19_cat","x20_cat","age_range")`；`categorical_cross_bins=(10,)` |
| **DualCategoryFeatureBlock** | `columns=["region","source","x19_cat","x20_cat","age_range","livability","version","month"]`；`cross_order=3`；`max_cross_columns=6`；`max_categories=128` |

交叉列优先级（进入 dual 前 6 列即参与三阶交叉）：

```text
region | source | x19_cat | x20_cat | age_range | livability   (+ version, month 作 dual 单体)
```

不要把高基数 `condition` / `t3` 原文塞进三阶交叉（稀疏过拟合）；`condition` 走 days_condition 分箱即可。

### 2.3 超参字典

```python
CAT_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1400,          # 可到 1500；靠 od 早停
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=10,           # 诊断变体用 12 亦可
    random_strength=0.7,
    od_type="Iter",
    od_wait=150,
    verbose=False,
    thread_count=-1,
    allow_writing_files=False,
)
# 每折: random_seed = seed + fold
```

**不要用**：`auto_class_weights='Balanced'`（本数据实测 1-seed OOF 从 0.690→0.687）；不要默认 `boosting_type='Ordered'`（更慢且未涨分）。

### 2.4 实测消融阶梯（seed=2026，诚实 OOF）

| 配置 | OOF AUC | Δ vs 旧配方 |
|---|---:|---:|
| A0 旧 `cat_semantic`（含 x0–x18 + dual 含 condition） | 0.68429 | — |
| A1 仅丢 `x0..x18` | 0.68557 | +0.0013 |
| A2 x19/x20 入 dual + days×region/source/age | 0.68720 | +0.0029 |
| B0/B1 enrich + 交叉 | ≈0.6896–0.6897 | +0.0054 |
| **B5 focus（推荐骨架）** | **0.69021** | **+0.0059** |
| B5 × 4 seeds 等权 | **0.69695** | +0.0127 |
| **B5 × 8 seeds 等权** | **0.69817** | **+0.0139** |

本诊断复现（B5 + `l2=12, iter=1500`，无 group-stats）：4-seed pooled **0.69821**  
（seed OOFs：0.6911 / 0.6901 / 0.6940 / 0.6889；seed_mean=0.6910）

特征重要性高频 Top（多折一致）：

```text
condition__bin_10__X__source
days / days__log1p / days__filled
region__X__livability__category_cross
cond_over_days / days_condition__ratio / days_x_cond
days_condition__bin_10__X__source
condition__bin_10__X__x19_cat
```

---

## 3. 从 0.69 → 0.698+ 的增益点（优先级）

| 优先级 | 动作 | 预期增益（相对单 seed≈0.690） | 证据 |
|---:|---|---|---|
| **P0** | **多种子等权概率平均（4→8）** | **+0.007 ~ +0.008** | 4s 0.697 / 8s **0.69817**；本诊断 4s 已 **0.69821** |
| **P1** | 丢弃 `x0..x18`（及勿把 `max_g` 当主信号） | +0.001 ~ +0.002 | A0→A1；保留弱 x 的 B4 反而掉到 0.6876 |
| **P2** | `x19/x20` 字符串化 + days/condition×语义交叉 | +0.002 ~ +0.003 | A1→A2→B5；importance 中 `X__source` / `X__x19_cat` 稳定 |
| **P3** | enrich：`cond_over_days`、`log_days`、`days_x_cond` | +0.002 | 单变量 0.605；B0 相对 A2 抬升 |
| **P4** | dual 列聚焦（去掉 condition 原文三阶；保留 region×livability） | +0.0005 ~ +0.001 | B5 优于更臃肿的 A3/A4（219 维更差） |
| **P5** | 超参微调（l2 10–12，od_wait 150–180，iter 1400–1500） | ≤+0.001 | 边际；depth=7 / Balanced / Ordered 未赢 |
| **P6（可选）** | 预注册的第二视图等权融合（非 OOF 搜权） | 不确定，常 ≤0 | 交付中 B1 融合未超 B5-only |
| **避免当增益** | nested/全局 TE、group-stats 乱拟合、加回 x0–x18 | **负** | TE 诊断掉分；fold group-stats 4s pooled **0.6957 < 0.6982** |

冲线实操顺序：**先锁 B5 单 seed≈0.690 → 开 4 seeds → 不够再加到 8 seeds**。不要先堆 TE。

---

## 4. 必须避免的过拟合陷阱

1. **对 `x0..x18` / 近唯一 `max_g` 做 TE 或强记忆**  
   一行一码 ⇒ TE≈标签泄漏；数值 AUC 虚高、MI≈0。

2. **全局 fit 再 OOF 的 TE / 分箱 / 频次 / group mean**  
   分箱边、类别词表、group days 统计必须 **只在训练折 fit**。

3. **用 OOF 搜融合权重再报告同一 OOF**  
   多种子/多视图只允许 **预声明等权**（或预注册离散选一），禁止连续权重搜索。

4. **把 in-sample TE / 单折 max AUC 当成绩**  
   `t3` in-sample TE≈0.60，OOF-TE≈0.53；单折可到 0.72，最终只报 **pooled**。

5. **`auto_class_weights` / 过度三阶稀疏交叉**  
   正例 10% 下 Balanced 伤 AUC；`max_cross_columns` 过大 + 高基数列 ⇒ 折间方差爆炸（见 fold3 常年 ~0.66）。

6. **旧数据预测包 / 测集伪标签**  
   本轮数据 SHA 与旧数据不同；任何旧 OOF 不可比、不可融合。

7. **混淆 seed_mean 与 pooled**  
   8-seed seed_mean≈**0.6898**，pooled≈**0.6982**。对外须同时披露；达标依赖 bagging，不是单模型已稳站 0.698。

8. **早停乐观**  
   `use_best_model=True` 合法但略乐观；卡在 [0.698, 0.700) 时建议补固定迭代对照。

---

## 5. 可落地伪代码与关键参数

```python
# ===== 关键参数 =====
N_SPLITS = 5
SEEDS = (2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033)
NOISE_X = [f"x{i}" for i in range(19)]
DUAL_COLS = [
    "region", "source", "x19_cat", "x20_cat",
    "age_range", "livability", "version", "month",
]
CAT_PARAMS = dict(
    loss_function="Logloss", eval_metric="AUC",
    iterations=1400, learning_rate=0.03, depth=6,
    l2_leaf_reg=10, random_strength=0.7,
    od_type="Iter", od_wait=150,
    verbose=False, thread_count=-1, allow_writing_files=False,
)

def enrich(df):
    out = df.drop(columns=[c for c in NOISE_X if c in df.columns], errors="ignore").copy()
    src = out["source"].astype(str)
    out["source_car"] = src.str.extract(r"CAR_(\d+)", expand=False).fillna("__NA__")
    out["source_eng"] = src.str.extract(r"ENG_(\d+)", expand=False).fillna("__NA__")
    p = out["t3"].astype(str).str.extract(r"^(-?\d+(?:\.\d+)?)([A-Za-z])$")
    out["t3_value"] = pd.to_numeric(p[0], errors="coerce")
    out["t3_kind"] = p[1].fillna("__NA__")
    out["x19_cat"] = out["x19"].astype(str)
    out["x20_cat"] = out["x20"].astype(str)
    days = pd.to_numeric(out["days"], errors="coerce")
    cond = pd.to_numeric(out["condition"], errors="coerce")
    out["cond_over_days"] = cond / (days.abs() + 1.0)
    out["days_x_cond"] = days * cond
    out["log_days"] = np.log1p(days.clip(lower=0))
    out["log_cc"] = np.log1p(pd.to_numeric(out["cc"], errors="coerce").clip(lower=0))
    out["log_V"] = np.log1p(pd.to_numeric(out["V"], errors="coerce").clip(lower=0))
    out["condition_missing"] = out["condition"].isna().astype(int)
    out["grades_n"] = out["grades"].map({"s": 1.0, "ss": 2.0, "sss": 3.0})
    out["month_n"] = pd.to_numeric(out["month"].astype(str).str.removeprefix("M"), errors="coerce")
    out["version_n"] = pd.to_numeric(out["version"].astype(str).str.removeprefix("v"), errors="coerce")
    return out

def make_blocks():
    return [
        RawFeatureBlock(),
        StructuredStringFeatureBlock(columns=["source", "t3", "region"]),
        DaysConditionFeatureBlock(
            quantile_bins=(5, 10, 20),
            categorical_cross_columns=("region", "source", "x19_cat", "x20_cat", "age_range"),
            categorical_cross_bins=(10,),
        ),
        DualCategoryFeatureBlock(
            columns=DUAL_COLS, max_categories=128,
            cross_order=3, max_cross_columns=6,
        ),
    ]

# ===== 训练循环伪代码 =====
oof_by_seed, test_by_seed = {}, {}
for seed in SEEDS:
    oof = np.zeros(n_train); pred_te = np.zeros(n_test)
    for fold, (tr_idx, va_idx) in enumerate(StratifiedKFold(5, True, seed).split(X, y)):
        Xtr, Xva, Xte = enrich(X.iloc[tr_idx]), enrich(X.iloc[va_idx]), enrich(test)
        # 各 block: fit_transform(Xtr) / transform(Xva) / transform(Xte)  —— 仅训练折 fit
        # prepare: 类别填 "__MISSING__"；数值用训练折中位数填
        model = CatBoostClassifier(**{**CAT_PARAMS, "random_seed": seed + fold})
        model.fit(Xtr_fe, ytr, eval_set=(Xva_fe, yva),
                  cat_features=cat_names, use_best_model=True)
        oof[va_idx] = model.predict_proba(Xva_fe)[:, 1]
        pred_te += model.predict_proba(Xte_fe)[:, 1] / 5
    oof_by_seed[seed] = oof
    test_by_seed[seed] = pred_te

oof_pool = mean(stack(oof_by_seed.values()), axis=0)   # 等权
test_pool = mean(stack(test_by_seed.values()), axis=0)
score = roc_auc_score(y, oof_pool)  # 唯一主报告分
```

### 复现命令（仓库已实现）

```bash
PYTHONPATH=src python3 -m insurance_claim.train_b5_focus \
  --views b5 --seeds 2026 2027 2028 2029 2030 2031 2032 2033 --shuffled

# 或分段：先 4 seed，再 scripts/run_b5_8seed.py 合并
```

产物：`artifacts/b5_8seed/metrics.json`（pooled **0.69817454**）、`submissions/submission_b5_8seed.csv`。

---

## 6. 结论与使用建议

1. **信号结构**：`days` + `condition` 交互是主轴；`region/livability/source` + `x19/x20` 是语义侧翼；`x0–x18` 是噪声。  
2. **无 TE 路径已够用**：B5 focus 单 seed≈0.690，**8-seed 等权冲过 0.698**。  
3. **继续抬分**优先加种子多样性 / 固定迭代稳健性，而不是加 TE。  
4. **报告规范**：主分=pooled OOF；同时写 seed_mean±std、shuffled、早停声明。

---

*诊断实验日志：`artifacts/diag_newdata_catboost.json`（本报告 4-seed 干净复现）。*  
*交付审核：`docs/supervision/FINAL_AUDIT_OPINION.md`（CONDITIONAL PASS：pooled 过线，seed_mean 未过 0.693）。*
