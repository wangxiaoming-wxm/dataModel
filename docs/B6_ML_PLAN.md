# B6 抬分方案（冲诚实 pooled OOF ≥ 0.70）

## 0. 硬约束与基线

| 项 | 约定 |
|---|---|
| B5 冻结 | 不改 `submissions/b5_frozen/`、`artifacts/b5_frozen/`、B5 交付语义 |
| 数据 | 仅 `/workspace/train.csv` + `test.csv`（新数据）；旧预测包作废 |
| 协议 | StratifiedKFold≥5；≥4 seeds 等权；无全局 TE；折内 FE；须 shuffled |
| 融合 | **仅预注册** `equal_prob` / `equal_rank`；**禁止** OOF 网格搜权 |
| 基线 | B5×8seed pooled **0.69817454**；seed_mean≈0.6898；shuffled≈0.5076 |

缺口：`0.70 − 0.69817 ≈ 0.00183`。4→8 seed 仅 +0.00122，单靠再加 4 个种子预计不够；需要 **异构臂多样性** 且臂强度不能明显弱于 B5（否则等权融合会拖分）。

已知负结果（勿重踩）：

- nested TE / group-stats：掉分
- B1 等权融合未超 B5-only
- 弱 Lossguide（4s OOF≈0.6818）与 B5 等权 → 0.6936 **<** B5

---

## 1. 策略总览

**主路径（数据挖掘后更新）**：B5 主臂 + **gap 缺口猫特征臂** → **预注册 equal-prob(b5, gap)** → 种子 ≥8（目标 12）。  
`fixed` 作早停对照臂（披露）；`biz`/`parse`/`lossguide` 为筛选臂，不进入默认融合。

```text
B6 = equal_prob( Arm_B5, Arm_Gap )
     × seeds(2026..2037) 等权
```

依据：1-seed CatBoost 探针 B5 0.688 → B5+gap 0.691（+0.003）；lossguide≈0.68 会拖分。  
副报告：同臂集合上的 `equal_rank`（不参与选型，只披露）。

---

## 2. 臂定义（预注册）

### Arm A — `b5`（主臂，复用 `build_b5`）

- FE：丢 `x0..x18`；`x19/x20` 作 cat；days/condition 语义交叉；dual order-3
- 参数：`iterations=1400, lr=0.03, depth=6, l2=10, od_type=Iter, od_wait=150, use_best_model=True, thread_count=8`
- 角色：锚点；强度天花板

### Arm B — `gap`（主抬分臂，B5 + 挖掘缺口猫特征）

- FE：`build_b5` + 折内分箱的 P0/P1 猫交叉（`ratio_q5×region/source`、`t3_sfx×code×days5`、`w_pair×days5`、`days_fixed×cond5/source`、`age_coarse` 等）
- 参数：与 B5 相同；**无 TE**
- 角色：吃满 B5 未覆盖的暴露/条款/车况强度切割

### Arm C — `fixed`（固定迭代稳健对照）

- FE：同 B5；`iterations=400`；无 OD / `use_best_model=False`
- 角色：对冲早停乐观（审计要求）

### Arm D/E/F — 筛选臂（默认不融合）

- `biz`：lean business FE（`build_lean`）
- `parse`：B5 + DomainParse
- `lossguide`：同 B5 FE + Lossguide（历史弱，默认剔除）

---

## 3. 种子与融合协议

```text
N_SPLITS = 5
SEEDS_12 = (2026, 2027, ..., 2037)   # 默认 12；快速实验可用 2–4
融合主规则（预注册）: equal_prob = mean(arm_oof) 再对 seeds 等权
副规则（预注册披露）: equal_rank = mean(rankdata(arm_oof))
禁止: 连续权重、按 OOF 网格选权、测集伪标签、全局 TE
```

实现落点：`src/insurance_claim/train_b6.py`  
产物：`artifacts/b6_<tag>/metrics.json` + `predictions.npz` + submission。

---

## 4. 实验阶梯（先快后全）

| 阶段 | 配置 | 目的 |
|---|---|---|
| Q1 快速 | 2 seeds × 4 arms + shuffled(1 seed) | 验证臂强度与融合 Δ |
| Q2 扩展 | 若 Q1 融合 ≥ B5_same_seeds + 0.0005，扩到 4–8 seeds | 估 0.70 可达性 |
| F 全量 | 12 seeds × 存活臂 + shuffled | 正式 B6 数字 |

**弱臂剔除规则（预注册，非搜权）**：

1. 若 `lossguide` 相对 `b5` 同种子 mean Δ < −0.008 → 从融合集合移除 `lossguide`
2. 若 `parse` 同理 → 移除 `parse`
3. `fixed` 允许更低（稳健臂），仅当 Δ < −0.015 才移除
4. 剔除后融合集合仍至少含 `b5` + 1 个异构臂；否则退化为 **B5×12seed** 对照

---

## 5. 期望与风险

| 来源 | 粗估 Δ vs B5×8 | 依据 |
|---|---:|---|
| 8→12 seed bagging | +0.0004 ~ +0.0008 | 4→8 已 +0.00122，边际递减 |
| 近强度异构等权 | +0.0005 ~ +0.0015 | 残差去相关；弱臂会变负 |
| 合计 | 有望逼近/越过 0.70 | 需实测 |

风险：异构臂过弱 → 等权拖分（已在旧 Lossguide 验证）。缓解：同 FE + 近参 + 弱臂剔除规则。

---

## 6. 交付清单

1. `docs/B6_ML_PLAN.md`（本文件）
2. `src/insurance_claim/train_b6.py`
3. `artifacts/b6_*/metrics.json`（真实数字，含 shuffled）
4. 不改动任何 B5 冻结路径

## 7. 复现命令

```bash
# 快速 2-seed
PYTHONPATH=src python3 -m insurance_claim.train_b6 \
  --seeds 2026 2027 --arms b5 lossguide fixed parse --shuffled \
  --output-dir artifacts/b6_q2seed

# 全量 12-seed（耗时长）
PYTHONPATH=src python3 -m insurance_claim.train_b6 \
  --seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 \
  --arms b5 lossguide fixed parse --shuffled \
  --output-dir artifacts/b6_12seed
```


## 8. 实测进度（诚实数字，勿改写）

| 实验 | pooled OOF | 备注 |
|---|---:|---|
| B5×8seed 冻结 | 0.69817454 | 基线 |
| B6 1-seed equal_prob(b5,gap) | 0.69221373 | gap 0.69184 > b5 0.69055 |
| B6×8seed equal_prob(b5,gap) | 0.69869545 | shuffled 0.5062 PASS |
| B6×12seed equal_prob(b5,gap) | 0.69867217 | 8→12 未抬分 |
| gap_bag 1seed | **0.693119** | bagging_temperature=1.0, random_strength=1.2 |
| mean(gap,gap_bag) 1seed | **0.693376** | |
| **B6×8seed equal_prob(gap,gap_bag)** | **0.69897470** | gap_bag_only 0.69891；shuffled 0.5056 PASS |
| corr(gap,gap_bag) @8seed | 0.996 | 多样性仍不足 |
| LGB / lossguide / main | ≤0.683 | 弱臂，不进融合 |
| 距 0.70（当前最佳） | **0.001025** | 继续筛 Bernoulli/MVS/hot-bag 异构 |
