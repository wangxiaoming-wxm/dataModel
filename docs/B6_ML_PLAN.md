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

**主路径**：B5 主臂保留 + 三条近强度异构臂 → **预注册 equal-prob** 融合 → 种子扩到 **12**。

```text
B6 = equal_prob( Arm_B5, Arm_Lossguide, Arm_Fixed, Arm_Parse )
     × seeds(2026..2037) 等权
```

副报告：同臂集合上的 `equal_rank`（不参与选型，只披露）。  
若某臂单 seed 明显弱于 B5（Δ < −0.008），该臂在 **后续扩种子前** 可按预注册规则剔除（见 §4），不是 OOF 搜权。

---

## 2. 四臂定义（预注册）

### Arm A — `b5`（主臂，复用 `build_b5`）

- FE：丢 `x0..x18`；`x19/x20` 作 cat；days/condition 语义交叉；dual order-3
- 参数：`iterations=1400, lr=0.03, depth=6, l2=10, od_type=Iter, od_wait=150, use_best_model=True`
- 角色：锚点；强度天花板

### Arm B — `lossguide`（同 FE，叶向生长）

- FE：与 B5 **完全相同**（`build_b5`）
- 参数差异：
  - `grow_policy="Lossguide"`
  - `max_leaves=31`
  - `depth=6`（限制树深；`depth=0` 实测无效）
  - 其余与 B5 对齐（含 od 早停）
- 角色：改变分裂顺序，制造预测残差相关但不等同的 OOF
- 失败门槛：单 seed OOF < B5_same_seed − 0.008 → 标记弱臂

### Arm C — `fixed`（固定迭代稳健臂）

- FE：同 B5
- 参数：去掉 `od_type/od_wait`；`iterations=400`（B5 best_iter 中位数≈408）；`use_best_model=False`
- 角色：对冲早停乐观与折间 early-stop 噪声；略欠拟合但偏差结构不同

### Arm D — `parse`（解析增强臂）

- FE：`build_b5` 输出 **再拼接** 折内 `DomainParseFeatureBlock`（t3/source/version 解析键、car/eng、ver_era、业务 key）
- 参数：与 B5 相同（含 od）
- 角色：在 B5 骨架上注入更细的业务 token，不引入 TE

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
