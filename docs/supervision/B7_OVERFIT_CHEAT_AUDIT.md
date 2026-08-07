# B7 过拟合 / 作弊独立检测意见

> 角色：**独立检测 Agent**（不参与训练抬分）  
> 协议：`IA-AUC710-B7-v1`  
> 对象分支：`cursor/b7-push-auc071-a5f5`  
> 对照冻结：`origin/cursor/b6-push-auc070-a5f5`  
> 机器可读：`artifacts/b7_audit/overfit_cheat_status.json`

---

## 总裁决

| 项 | 结论 |
|---|---|
| **总裁决** | **CONDITIONAL** |
| 是否作弊（硬红线） | **否** |
| 是否过拟合（崩盘式 / 可驳回） | **否**（仅轻度乐观偏置） |
| closest 分数是否可采信 | **是**（数值复算误差 = 0；口径须标 CONDITIONAL） |
| 可否交付诚实 0.71 | **否**（`nested < 0.71`） |
| 相对草稿 `B7_FINAL_AUDIT_OPINION.md` | **基本维持** REJECT-0.71 + 认证 closest；**加严** max3 升格为 CONDITIONAL |

未发现可触发 `REJECT_CHEAT` / `REJECT_OVERFIT` 的硬证据。  
因 **max3 由 fuse0 disclosure 升格为 closest 主报** + **强 max 依赖**，总裁决为 **CONDITIONAL**（非 CLEAN）。

---

## 红线 checklist

| # | 红线 | 裁决 | 证据要点 |
|---|---|---|---|
| 1 | 全局 / 外置 Target Encoding | **PASS** | 交付路径 = 冻结 B6 OOF + `reference/v10` plus；无全局 TE |
| 2 | 全量 fit FE 再 CV | **PASS** | B6 折内 FE；closest 只复用冻结 OOF |
| 3 | 连续 OOF 搜权 / 全量 OOF 挑权重 | **PASS** | 离散 `FUSION_RULES` + nested；stack 连续 meta 未作主报且更低 |
| 4 | test 伪标签 / test 标签 | **PASS** | `data_gate.json`：test 无 label；SHA 与协议一致 |
| 5 | 嵌套规则选择诚实性（pair） | **PASS** | 独立复算 pair nested = **0.7022093156561012**，5/5 `max` |
| 6 | max3 事后扩规则 / disclosure 冒充 nested 主报 | **CONDITIONAL** | `b7_fuse0.py` 将 max3 标为 `three_arm_disclosure`；实验日志曾写「非主报」；后升格 closest |
| 7 | OOF 叠层泄漏作主报 | **PASS** | resid 用 stage1 OOF 作特征但分数更低；closest 无 stage2 叠层 |
| 8 | B6 冻结完整性 | **PASS** | pooled 仍 = `0.6989746962571622`；字节对照 B6 分支一致 |
| 9 | 数据门禁 SHA | **PASS** | train/test/submit_sample 与协议 §0 一致 |

详表：`artifacts/b7_audit/redline_checklist.json`。

---

## Closest 分数是否可采信

**可采信（数值）**，附条件：

| 口径 | AUC | 地位 |
|---|---:|---|
| closest `max(gap, gap_bag, plus_v10)` | **0.7027049552615718** | 数值正确；**CONDITIONAL** 叙事 |
| 协议最稳妥副报 fuse0 `max(equal_b6, plus)` | **0.7022093156561012** | 预注册六规则 + nested，无升格争议 |

复算（误差 **0** &lt; 1e-8）：

```text
roc_auc(y, max(gap,gap_bag,plus)) == 0.7027049552615718
oof ≡ max3；arms ≡ b6_gapbag_8seed + reference/v10 plus
nested_select_pair(equal_b6, plus) == 0.7022093156561012
max-family nested 5/5 → max3
```

产物：`artifacts/b7_audit/closest_recompute.json`。

---

## 过拟合检测

| 检测 | 结果 | 解读 |
|---|---|---|
| 复算 closest OOF | 误差 0 | 分数未伪造 |
| `shuffled_plus → max3` | **0.6442427188098837** &lt; 0.66 | 崩盘达标；增益依赖真实 plus 排序 |
| 打乱 y × max3 预测 | ≈0.496 | 近随机 |
| B6 `seed_std` | 0.00124 | 稳定 |
| max3 / pair fold AUC range | ≈0.026 / 0.027 ≤ 0.06 | 可接受 |
| V10 公开榜 0.70570 vs 本地 nested 0.7027 | 本地更低 | **不**自动判作弊；披露 LB≠盲测本地 |
| early stopping on valid | B6 臂有；已披露 | 已知轻度 OOF 乐观偏置 |

**结论（是否过拟合？）**  
存在 **轻度乐观偏置**（强依赖 elementwise `max` + valid early stopping），但 **不是** 标签泄漏 / 伪标签导致的虚假高分；对照实验崩盘正常 → **不构成 `REJECT_OVERFIT`**。

---

## 阴性实验交叉核验

抽查 `b7_resid` / `b7_gate_soft` / `b7_plus_mine` / `b7_hybrid` / `b7_lgb_gap` / `b7_ebm`（及 plus_h3、stack、fuse_plusmine）metrics：

- **无一** 超过 closest **0.702704955**
- 若干实验的 `stage1_auc=0.702209` 只是 fuse0 地板回显，非更高主报
- 未发现用更高分阴臂冒充主报

证据：`artifacts/b7_audit/negatives_crosscheck.json`。

---

## 明确回答用户两个问题

1. **是否过拟合？**  
   **轻度有（叙事/口径层面），非严重过拟合。** shuffled 崩盘、复算一致；不可据此驳回分数本身，但不可把 max3 写成与规则无关的稳健 0.70+。

2. **是否作弊？**  
   **否。** 未发现全局 TE、test 标签、连续 OOF 搜权主报、伪造 nested、篡改 B6 冻结。  
   **唯一 CONDITIONAL 项**：max3 先作 disclosure、后升格 closest（对照 `B7_PLAN` / `b7_fuse0.py` / 实验日志）。

---

## 复现命令

```bash
# 数据门禁
sha256sum train.csv test.csv submit_sample.csv

# closest 复算
PYTHONPATH=src python3 -c "
import numpy as np
from sklearn.metrics import roc_auc_score
z=np.load('artifacts/b7_closest/predictions.npz')
print(roc_auc_score(z['y'], np.maximum(np.maximum(z['gap'], z['gap_bag']), z['plus'])))
"

# pair nested 复算
PYTHONPATH=src python3 -c "
import numpy as np
from insurance_claim.b7_fusion import nested_select_pair
z=np.load('artifacts/b7_closest/predictions.npz')
eq=0.5*(z['gap']+z['gap_bag'])
print(nested_select_pair(eq, z['plus'], z['y'])['nested_oof_auc'])
"

# shuffled plus → max3（seed=42）
PYTHONPATH=src python3 -c "
import numpy as np
from sklearn.metrics import roc_auc_score
z=np.load('artifacts/b7_closest/predictions.npz')
p=z['plus'].copy(); np.random.RandomState(42).shuffle(p)
print(roc_auc_score(z['y'], np.maximum(np.maximum(z['gap'], z['gap_bag']), p)))
"
```

---

## 签字

```text
[AUDIT_PACKET_B7_OVERFIT_CHEAT]
protocol_id: IA-AUC710-B7-v1
auditor_verdict: CONDITIONAL
is_cheat: false
is_overfit_reject: false
closest_honest_nested_oof_auc: 0.7027049552615718
pair_fuse0_nested_oof_auc: 0.7022093156561012
gate_0_71: false
deliver_0_71_allowed: false
b6_freeze_check: PASS
data_sha256_match: true
max3_dependency: CONDITIONAL
hard_redlines_failed: []
```

独立检测 Agent — 2026-08-07
