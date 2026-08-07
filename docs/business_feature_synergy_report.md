# 车险理赔业务 × 特征协同报告（新 train.csv）

**结论先行：** 新数据基线索赔率 **10.02%**（1496/14930）。最强单变量是 **`days`（暴露/车龄代理）**：十分位 CramerV≈0.108，高三分位 OR≈1.40 vs 低三分位 OR≈0.57。业务上优先做 **「地理 × 暴露」「动力源 × 暴露」「车况 × 暴露」** 的原生类别交叉，用 CatBoost 吃字符串交叉，而不是粗糙全局 Target Encoding。本地冲 0.698+ 的主路径是：**语义交叉 + 折叠内分箱 + 多 seed 等权**，不是堆高 cardinality TE。

---

## 1. 字段业务语义归类

| 业务角色 | 字段 | 新数据证据 | 产品解读 |
|---|---|---|---|
| **暴露时长 / 风险时间** | `days` | \|corr\| 最高 0.100；十分位索赔率 4.9%→14.3%；三分位 OR 0.57 / 1.07 / 1.40 | 最像「已行驶/已承保时长」或强相关的车龄代理。风险随天数近似单调上升，是定价暴露因子。 |
| **车况 / 使用损耗** | `condition`, `x20` | condition 缺失 144；最低十分位索赔率 14.5%（OR 1.53），最高十分位 6.5%（OR 0.62）；`x20`↔condition corr≈0.52 | 低 condition 高赔——更像「车况差/损耗高」而非「分越高越好」。`x20` 为车况相关衍生。 |
| **车型动力 / 排量价值簇** | `cc`, `V`, `max_g`, `x19`, `source`(CAR\|ENG), `code`, `grades`, `t3` | cc↔V 0.94；V↔max_g 0.90；x19↔V 0.997；`code` 几乎决定 CAR 族（A→CAR_1/4，D→CAR_7）；grades 与 code 强耦合 | 同一「动力/价位」簇的多视角编码。`source`=`CAR_*\|ENG_*` 是车型+引擎组合；`t3`=`数值+后缀(P/E/M)` 像排量档/动力标定。 |
| **保单条款 / 开关位** | `t1,t2`, `r1,r2`, `c1,c2`, `w1,w2`, `month` | 多为 0/1；w1/w2 有显著差（w1=0 率 10.5% vs 1 为 9.2%；w2 反向）；t1/t2、c1/c2 单变量弱 | 更像条款开关、批单标志、续保/附加险。`month` 高度偏斜（M2 9184、M1 3433），像进件月或保单月，单独弱（CramerV 0.026）。 |
| **地理 / 宜居** | `region`, `livability` | region 20 档，高赔 `f09d` 11.9%（OR 1.21）vs 低赔 `c1f5` 3.7%（OR 0.35）；**livability 对 region 的 R²≈0.990** | `livability` 基本是区域属性嵌入，不是独立驾驶行为。地理主信号在 `region`，livability 作连续微调即可。 |
| **渠道 / 产品版本** | `version`, `source` 的渠道侧, `grades` | version 19 档，v11 率 13.6%（OR 1.42）vs v16 6.97%（OR 0.67）；与 CAR 交叉后差异更大 | 产品/核保规则版本或渠道包。版本风险差存在，但单档 n≈200–300，**TE 极不稳定**。 |
| **被保险人/标的年龄段** | `age_range` | 1→8.4%，8→16.5%（OR 1.77）；与 days 均值同向上升但 R²(days~age) 仅 0.02 | 年龄段独立于 days，二者应交叉而非互相替代。注意 age≥9 样本极稀（14/3）。 |
| **匿名连续嵌入** | `x0`–`x17`, `x18` | x0–x16 近似零均值小方差；与 label 的 \|corr\| 多数 <0.03 | 更像脱敏 embedding / 残差特征，适合当数值主效应，不宜硬分箱交叉。 |

### 1.1 关键结构洞见（建模必用）

1. **`code` ⊃ 车型族**：A={CAR_1,CAR_4}，B={CAR_2,3,5,9}，C={CAR_0,6,8,10}，D={CAR_7}。`code×source` 信息冗余，交叉时二选一或拆 `car`/`eng`。
2. **`grades` 与 `code` 强绑定**（D 几乎全是 `s`；A 几乎全是 `ss`），单独 grades 信号弱。
3. **`t3` 后缀三态**：P(8424, 9.48%) / E(6193, 10.74%) / M(313, 10.22%)。M 与 code=D / CAR_7 对齐——小众动力包。
4. **days 的地理异质性极强**：同为「高 days vs 低 days」，`c1f5` OR≈6.3、`90fc`≈4.3，而 `abb2`≈0.42、`ab86`≈0.80——**暴露斜率因区域而异**，这是 region×days 必须做的业务理由。

---

## 2. 业务假设的数据验证（均来自新 train.csv）

| 假设 | 验证结果 | 对特征的含义 |
|---|---|---|
| 暴露越长越易赔 | days 十分位率：~4.9% → ~14.3%；CramerV=0.108，p≈6e-33 | `days` 连续 + log1p + 分位 bin 全保留 |
| 车况越差越易赔 | condition 最低/最高十分位 OR 1.53 / 0.62；CramerV=0.073 | 与 days 乘积/比值 + 联合分箱 |
| 地理有稳定风险差 | region 高/低（n≥30）OR 1.21 vs 0.35；全档 n≥79，无极小单元 | region 可作稳定类别主效应 |
| 宜居是地理衍生 | livability R²(region)=0.990 | 避免 region×livability 细交叉（信息重复） |
| 车型有风险差 | CAR_10 率 15.6%（OR 1.66）vs CAR_8 5.6%（OR 0.53） | `car`/`eng`/`source` 作类别；优先 × days |
| 版本有风险差但不稳 | v11 OR 1.42 vs v16 OR 0.67；单档 n~230 | 给 CatBoost 当 cat，不做全局 TE |
| 条款开关弱但可交互 | w2=1 vs 0：10.56% vs 9.14%（p≈0.006）；t1/t2/c* 单变量不显著 | 用 `w1_w2` 对 × days/version |
| 年龄单调偏高龄高赔 | age1 8.4% → age8 16.5%；age9/10 过稀 | `age_coarse`（合并 8+）× days |

**region×days_bin5 业务可读单元（n≥80）：**

- 高风险：`8a21×D4` 17.8%，`f09d×D4` 16.9%，`9685×D4` 16.4%
- 低风险：`f167×D1` 1.2%，`2a36×D4` 2.6%，`f09d×D1` 4.1%

**days×condition 联合（n≥80）：** `D5×C2` 17.6% vs `D1×C2` 2.7%——「高暴露 + 中低车况」是清晰高赔象限。

---

## 3. 推荐的 12 个语义交叉（按业务解释力 × 可泛化性）

| # | 交叉 | 业务故事 | 新数据支撑 | 用法建议 |
|---|---|---|---|---|
| 1 | **`region × days_bin5`** | 区域风险价格曲线：同一地区随暴露抬升的斜率不同 | CramerV 0.147；OOF-TE AUC 0.600；n\<20 仅 9 格，行占比 0.9% | **第一优先**。fold 内分位；CatBoost 字符串交叉 |
| 2 | **`days_bin5 × condition_bin5`** | 暴露 × 车况：老车况差的联合损失 | CramerV 0.131；25 格全 n≥50；span 0.15 | 数值积/比 + 类别联合 bin |
| 3 | **`source/car × days_bin5`** | 车型动力在不同车龄/里程下的索赔曲线 | CramerV 0.137；mass≥50 = 99% | 拆 `car`/`eng` 更稳；优于 source×version TE |
| 4 | **`t3_sfx × code × days_bin5`** | 动力标定 × 车型条款族 × 暴露 | 35 格，**零小样本**；span 0.19；CramerV 0.128 | 三阶里最干净的一档 |
| 5 | **`w1_w2 × days_bin5`** | 条款开关在不同暴露下的相对风险 | 20 格全充足；CramerV 0.115 | 把 w1/w2 合成 4 态再交叉 |
| 6 | **`age_coarse × days_bin5`** | 人龄 × 车暴露（两套时钟） | CramerV 0.114；合并后小样本可控 | age_range 原始 × version 易碎，用 coarse |
| 7 | **`days_bin5 × version`** | 产品版本的暴露定价曲线 | CramerV 0.131；span 0.24 | CatBoost cat；勿全局 TE |
| 8 | **`region × car`** | 区域车型结构（进口/本地车型占比） | CramerV 0.133；注意部分车×区 n\<20 | 可进 dual_category 二阶 |
| 9 | **`car × version`** | 车型在特定产品包下的核保结果 | n≥50 时 CAR_0×v17 19.3% vs CAR_1×v8 3.1% | **只做原生类别**；作 TE 时 leaky 0.59 / OOF 0.51 |
| 10 | **`t3_sfx × code`** | 动力后缀 × 车型族（条款/车系） | 7 格全充足；主效应弱但结构清晰 | 轻量交叉，配合 days 三阶 |
| 11 | **`region × condition_bin5`** | 区域道路/气候对车况索赔的放大 | CramerV 0.112 | 次于 region×days，可作补充 |
| 12 | **`livability_bin × days_bin5`** | 宜居档的暴露曲线（实质近 region） | CramerV 0.116；与 region 共线 | 有 region 时降权；无 region 时顶替 |

可选增强（仅 CatBoost 原生、限制 max_cross）：`region × days5 × version` 解释力极强（CramerV 0.32），但 **90% 格子 n\<20**——见第 4 节。

---

## 4. 「泄漏 TE 高但易过拟合」的交叉

用 **全局 TE AUC（泄漏）− 5-fold OOF TE AUC（诚实）** 衡量虚高：

| 交叉 | 泄漏 TE AUC | OOF TE AUC | Gap | 小样本 | 判定 |
|---|---:|---:|---:|---|---|
| `t3 × code` | 0.624 | 0.523 | **0.101** | 256 格，n\<20: 89 | 高基数细则；TE 死路 |
| `source×version` / `car×version` | 0.592 | 0.511 | **0.081** | 203 格，n\<20: 116（行占比 6.5%） | 单元过碎，OOF 近随机 |
| `region×days_bin10` | 0.659 | 0.597 | 0.062 | 200 格，n\<20: 64 | 比 bin5 虚高更多；优先 bin5 |
| `region×days5×version` | CramerV 0.32 | — | — | **90% n\<20**，mass≥50 仅 48% | 三阶爆炸；TE 必过拟合 |
| `car×version×days5` | CramerV 0.27 | — | — | 86% n\<20 | 同上 |
| `age_range × version` | — | — | — | 161 格，n\<20: 61；age9/10 极稀 | 合并年龄或丢弃 |
| `month × version` | — | — | — | month 稀有月 × version 更碎；month 单变量 p=0.64 | 低优先级 |

**相对健康（可 TE 或原生 cat）：** `region×days_bin5` 泄漏 0.638 / OOF **0.600**，gap 仅 0.038，且 n\<20 行占比 \<1%。

---

## 5. 为何风控/定价更吃 CatBoost 原生类别 + 字符串交叉，而不是粗糙 TE

1. **信度与 credibility：** 定价要求单元有足够暴露。粗糙 TE 把 `CAR_0×v17`（n=57、11 赔）的 19.3% 当成「真纯保费」，无部分池化；CatBoost 的 ordered TS / 贝叶斯平滑等价于 **credibility weighting**——小单元向先验收缩。
2. **泄漏 ≠ 可保信号：** `car×version` 泄漏 AUC 0.59，诚实 OOF 0.51。核保若按泄漏 TE 调费，会把噪声当成风险差异，导致 **逆向选择 + 不稳定费率**。
3. **组合爆炸与可解释审批：** 三阶 `region×days×version` 1400+ 格无法进费率表；CatBoost 在树分裂里学交互，事后可用 SHAP/partial dependence 抽 **主费率因子**（region、days 分位、car），符合监管可解释诉求。
4. **新水平泛化：** 测试集新 version/region 上，全局 TE 只能回退全局均值；原生类别 + 未知水平处理（fold vocab → -1 / `__MISSING__`）与 **区域主效应 + 暴露斜率** 仍可外推。
5. **与现有 recipe 对齐：** `raw + structured_string(t3/source 拆前后缀) + days_condition(分箱交叉) + dual_category(cross_order=3)`，无 TE、fold-local——正是冲本地 AUC 0.698+ 的风控友好路径。

---

## 6. 冲本地 AUC 0.698+ 的特征协同清单

**必做**

1. `days`：连续 + `log1p` + fold 内 qcut(5/10)  
2. `condition`：填补 + 分箱 + `days*condition` / `condition/(days+1)`  
3. 字符串解析：`source→car/eng`，`t3→t3_num/t3_sfx`  
4. 语义交叉进 CatBoost cat：`region|days_bin5`，`days_bin5|cond_bin5`，`car|days_bin5`，`t3_sfx|code|days_bin5`，`w_pair|days_bin5`  
5. dual_category 列优先：`region, source/car, version, age_range(或 coarse), t3_sfx, days_bin`；`cross_order=2~3` 但 **max_cross_columns 控在 5–6**，避免三阶全展开

**慎做 / 不做**

- 全局 TE on `t3×code`、`car×version`、任意三阶高基交叉  
- `region×livability` 细交叉（共线）  
- `month` 稀有月单独高权  
- `age_range` 9–10 不合并就交叉 version  

**数值侧**

- `cc/V/max_g/x19` 共线，留 1–2 个代表 + 分位即可  
- `x0–x17` 当连续主效应，不参与高基交叉  

---

## 7. 复现

```bash
python scripts/business_feature_synergy_audit.py
# → artifacts/business_feature_synergy/audit_numbers.json
# → artifacts/business_feature_synergy/audit_summary.md
```

所有索赔率、OR、CramerV、TE gap 均由该脚本从 **新** `train.csv` 现算。
