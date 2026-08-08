# V10 交付说明（详细版）

> 日期：2026-08-07（含公开榜回写）  
> **公开榜 AUC：0.70570**（文件 `submissions/submission_v10.csv`）  
> **V9 未改动**（`submissions/submission.csv` ≡ `sub_v9_champion_4seed.csv`）  
> 相关审核：`13` / `14`；排序：`15`；总报告：`07`；索引：`INDEX.md`

---

## 1. 结论一览

| 口径 | 值 | 用途 |
|------|-----|------|
| **公开榜 AUC** | **0.70570** | 已提交成绩 |
| **本地嵌套主口径** | **0.701314965** | 对外/答辩诚实主报 |
| **本地冻结池化** | **0.701750796** | `max(B5-12, plus)`；与提交 test 对应 |
| 门禁本地 ≥0.70 | **通过**（嵌套与池化均过） |
| 选定标签 | `b5_12__max__plus` | 见 `outputs/v10/meta_v10.json` |
| 嵌套选定规则 | **`max`**（5/5 折） | `fuse_v10.py` |

绝对路径：

```
/Volumes/pssd/app/ml/正式比赛/20260807-cursor/submissions/submission_v10.csv
/Volumes/pssd/app/ml/正式比赛/20260807-cursor/outputs/v10/predictions_v10.npz
/Volumes/pssd/app/ml/正式比赛/20260807-cursor/outputs/v10/meta_v10.json
```

---

## 2. 设计目标与约束

**目标：** 在 B5（本地 ~0.698）之上，产出**诚实**本地 AUC > 0.70，且不破坏 V9 产物。

**硬约束：**

1. 仅新数据（与 B5 仓库 train/test MD5 一致）  
2. 选定臂 **无外置 TE**  
3. 折内 FE（fit 仅 train fold）  
4. 融合仅用**预注册离散规则** + 嵌套选择（禁止连续搜权）  
5. 不改 `submissions/submission.csv`（V9）

---

## 3. 双臂配方

### 3.1 臂 A — B5 focus（主排序锚）

| 项 | 内容 |
|----|------|
| 代码 | `external_review/b5_repo/src/insurance_claim/train_b5_focus.py` |
| 特征策略 | 丢 `x0..x18`；`x19`/`x20`→字符串类别；days/condition 分位交叉；dual order=3（聚焦列） |
| 模型 | CatBoost Logloss/AUC，`1400/0.03/d6/l2=10/rs=0.7/od_wait=150` |
| CV | StratifiedKFold **5**；折内 `build_b5`；`use_best_model` |
| B5-8 可复核 | pooled **0.698174538**；seed_mean≈**0.6898** |
| 本机扩种 | seeds **2034–2037** → 与 B5-8 合成 **12-seed** 等权池 **0.698640571** |

> **冻结说明：** 提交对应 12-seed。若磁盘另有 `partial_b5_extra_s2038+.npz`，**未进入**当前 `meta_v10` / `submission_v10`，重跑 `fuse_v10.py` 时亦以「已写入冻结的 2034–2037」为准，避免 silent 漂移。

### 3.2 臂 B — plus / root_plus（异构）

| 项 | 内容 |
|----|------|
| 特征代码 | `src/v10/plus_features.py`（自 `20260807-codex/push698` 接入并落盘） |
| 策略要点 | **保留 x0–x18**，去掉 x19；root 交叉 + `w_conflict` / `t3_num_z` / `x20_grid` / latent 行统计等 |
| TE | **不使用**（holdout 上 te20/50/100 均弱于 root_plus） |
| 模型 | H2：`2500/0.02/d7/l2=20/rs=1.0/od_wait=150` |
| CV | StratifiedKFold **10** × seeds 2026–2029 |
| pooled | **0.688617067** |
| 与 B5 相关 | pearson≈**0.915**，spearman≈**0.948**（可融合，但非低相关） |

OOF/test 冻结：`outputs/v10/oof_plus_h2_10.npz`、`test_plus_h2_10.npy`。

### 3.3 融合协议

预注册集合（固定 6 类，无连续权重）：

| 规则 | 公式（示意） | B5-8×plus AUC | B5-12×plus AUC |
|------|--------------|---------------|----------------|
| mean | 0.5(a+b) | 0.698871 | 0.698922 |
| mean_2_1 | (2a+b)/3 | 0.699448 | 0.699670 |
| power2 | RMS | 0.699401 | 0.699515 |
| power3 | L3 均值 | 0.699790 | 0.699964 |
| **max** | **elementwise max** | **0.701315** | **0.701751** |
| rank_mean | 秩平均 | 0.696106 | 0.696277 |

**嵌套选择：** `StratifiedKFold(5, shuffle=True, random_state=42)`  
每折在 train 段上比较 6 规则 → **五折全部选择 `max`**  
→ 嵌套 OOF = **0.701314965** ≡ full-data `max(B5-8, plus)`  

提交：`max(B5-12_pool, plus)` → OOF **0.701750796**；test 同步 `max`。

**Sanity：** 打乱 plus 行后再 `max` 与 B5 → AUC 崩至 ~0.64；说明增益依赖 plus 真实排序，非空壳公式。

---

## 4. 阴性 / 未采用路径

| 尝试 | 结果 | 结论 |
|------|------|------|
| b5plus（密交叉 + t3/code） | seed≈0.687 | 差于 B5，已停 |
| b5_phys（折内 x19 残差等） | 0.69028 vs B5 0.69021 | 无增益 |
| b5_t3cross（仅 days×t3_kind） | 0.69023 | 无增益 |
| 仅扩 B5 seed | 12-seed 0.69864 |  alone 不足 0.70 |
| V9×B5 等权 | < B5 | 相关过高 |
| 嵌套 logistic 堆叠 | 最高≈0.6994 | 未过 0.70 |
| mean/power 族融合 | 均 <0.700 | 嵌套未选中 |

---

## 5. 公开榜结果与解读

| 项目 | 值 |
|------|-----|
| 提交文件 | `submissions/submission_v10.csv` |
| **公开榜 AUC** | **0.70570** |
| 相对嵌套本地 | **+0.004385** |
| 相对冻结池化 | **+0.003949** |

**解读（更新过时叙事）：**

1. 早期文档曾写「公开榜可能低于本地 pooled」——对 **本枪 V10 已不适用**；实际为**上浮**。  
2. 上浮支持：异构臂 + `max` 捕捉的高分排序在公开集上有效，而非纯本地过拟合幻觉。  
3. 仍须披露：`max` 不利校准；双臂早停使本地 OOF 轻度乐观；**不得**把 0.70570 说成本地 CV。  
4. 独立审核「有条件通过」结论**不因公开榜上浮而改为无条件通过**——协议风险（规则依赖 `max`）仍在，但「上榜必回撤」的先验应下调。

---

## 6. 诚实口径（必读）

**应主报：**

1. 本地：**嵌套 0.7013**（B5-8 × plus，`max`）  
2. 旁注：冻结提交对应池化 **0.7018**（B5-12 × plus）  
3. 公开榜：**0.70570**（与本地分列）  
4. 结构：B5 focus 多种子 + plus(keep-x, 10 折) + 嵌套选定 `max`  
5. 披露：验证集早停；`max` 利 AUC、不利概率校准  

**不应这么说：**

- 「单模 / 单 seed 已稳 0.70 / 0.705」  
- 「无条件诚实 CV = 0.7018」或把公开榜当作 CV  
- 「所有预注册融合都过 0.70」（mean/power 未过）  
- 「V9 已替换」（V9 文件仍在且未改）

---

## 7. 产物清单

| 路径 | 说明 |
|------|------|
| `submissions/submission_v10.csv` | 公开榜提交 |
| `outputs/v10/predictions_v10.npz` | `oof/test/y` + 分臂 OOF |
| `outputs/v10/predictions_v10_max_b5_plus.npz` | `max(B5-8,plus)` 平行冻结 |
| `outputs/v10/meta_v10.json` | 选定项、候选表、公开榜 |
| `outputs/v10/oof_plus_h2_10.npz` / `test_plus_h2_10.npy` | plus 臂 |
| `outputs/v10/partial_b5_extra_s2034..2037.npz` | B5-12 的 extra seeds |
| `outputs/v10/ablation_archive/` | 阴性/未冻结产物归档 |
| `src/v10/fuse_v10.py` | 融合与嵌套选择 |
| `src/v10/plus_features.py` / `train_plus_arm.py` | plus 特征与训练 |
| `src/v10/train_v10.py` / `train_v10_batch.py` | 主训练入口 |
| `src/v10/builders_v10.py` | 消融构建器 |
| `src/v10/experimental/` | 非主路径冒烟脚本 |

---

## 8. 复现

```bash
cd /Volumes/pssd/app/ml/正式比赛/20260807-cursor

# 仅融合（推荐；依赖已冻结 OOF）
python3 -u src/v10/fuse_v10.py

# 可选：重训 B5 extra（耗时；勿在未更新 meta 时混入未冻结 seed）
python3 -u src/v10/train_v10_batch.py --tag b5_extra --seeds 2034 2035 2036 2037
```

校验示例：

```python
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
y = pd.read_csv("/Volumes/pssd/app/ml/正式比赛/data/train.csv")["label"].astype(int)
z = np.load("outputs/v10/predictions_v10.npz")
assert abs(roc_auc_score(y, z["oof"]) - 0.7017507955311727) < 1e-12
sub = pd.read_csv("submissions/submission_v10.csv")
assert np.allclose(sub["label"], z["test"])
```

---

## 9. 与 B5 / V9 关系

| | V9 | B5-8 | V10 |
|--|----|------|-----|
| 本地主报 | 0.679 | 0.6982 | **0.7013** |
| 公开榜 | — | — | **0.70570** |
| 管道 | 全量 FE→CV 轻度流程问题 | 折内 FE 最干净 | 双臂+嵌套 `max` |
| 角色 | 历史基线 | 稳妥备枪 / 答辩单臂 | **当前公开榜主交** |
