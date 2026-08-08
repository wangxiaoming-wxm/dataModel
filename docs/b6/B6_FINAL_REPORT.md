# B6 交付报告（诚实逼近 0.70）

## 结论

| 项 | 值 |
|---|---|
| **B6 最接近诚实 pooled OOF** | **0.69897470** |
| 距目标 0.70 | **0.001025** |
| gate_0_70 | **FAIL**（未达标） |
| shuffled | **0.50560 PASS** |
| B5 冻结（不可变） | **0.69817454** |
| 相对 B5 诚实抬分 | **+0.00080** |

**不可将 0.69897 包装为 0.70。** 按用户约定：未达标则交付**最接近的诚实分**。

---

## 最优配方（预注册）

```text
equal_prob(gap, gap_bag) × seeds[2026..2033]
```

| 臂 | FE | 参数差异 | 8seed pooled | seed_mean |
|---|---|---|---:|---:|
| gap | B5 + mining P0/P1 猫交叉 | 标准 CatBoost early-stop | 0.698683 | 0.69025 |
| gap_bag | 同 gap | `bagging_temperature=1.0`, `random_strength=1.2` | **0.698906** | 0.69098 |
| **融合** | — | 等权概率 | **0.698975** | 0.69163 |

协议：折内 FE、无全局 TE、无 OOF 搜权、新数据 only、`thread_count=8`。

提交：`submissions/submission_b6_gapbag_8seed.csv`  
指标：`artifacts/b6_gapbag_8seed/metrics.json`

---

## 路径里程碑

| 阶段 | pooled OOF |
|---|---:|
| B5 冻结 8seed | 0.698175 |
| B6 equal_prob(b5,gap) 8seed | 0.698695 |
| B6 gap 12seed | 0.698712 |
| **B6 equal_prob(gap, gap_bag) 8seed** | **0.698975** |

---

## 负结果（勿重踩）

- Lossguide / biz / lean 弱臂等权会拖分
- 改写 dual（去掉 month/livability、塞 code/w_pair）→ 1seed 掉到 0.683
- 扩充 gap 猫特征（gapv2）→ 1seed −0.0003
- LGBM 同 FE → 0.659，无用
- 8→12 seed 对 b5+gap 融合无抬分（甚至略降）
- 三臂 b5+gap+gap_bag 不如 gap+gap_bag（相关 >0.994）

---

## 为何未到 0.70

剩余缺口 ~0.001。当前最强臂高度同质（corr(gap, gap_bag)≈0.996），等权融合边际仅 +0.0003。在**禁止 TE / 禁止 OOF 搜权 / 禁止测集伪标签**约束下，未找到近强度且低相关的第二哲学臂。再加种子的边际已证明不足。

---

## 独立复核要点

- B5 冻结三件套未改
- shuffled ∈ [0.47, 0.53]
- 融合规则预注册为 equal_prob(gap, gap_bag)
- 宣称：**最接近诚实分 0.69897，未过 0.70 门**
