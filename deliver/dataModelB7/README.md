# dataModelB7 — 车险索赔二分类 · B7 完整交付

本仓库是 **B7 方案**的干净交付包：比赛说明、数据、可复现代码、冻结 OOF、提交文件与训练流程。  
已去掉 B5/B6 实验废稿、冲突文档与无关 ablation 产物。

| 项 | 内容 |
|---|---|
| 任务 | 车险索赔 **二分类**（是否索赔） |
| 主指标 | ROC-AUC（诚实本地 OOF / nested） |
| B7 配方 | `max(B6_gap, B6_gap_bag, V10_plus)` |
| 本地权威分 | **0.702704955**（未达 0.71 门禁） |
| **公开榜** | **0.70722**（同一提交文件） |
| 提交文件 | `submissions/submission_b7_closest_honest.csv` |

---

## 1. 比赛是干什么的

根据保单相关特征，预测样本是否发生索赔（`label∈{0,1}`）。  
训练集约 1.5 万行，测试集约 6.4 千行；提交格式为 `id,label`（`label` 为预测概率）。

权威本地口径强调：

- 折内特征工程（fold-local FE）
- 无全局 / 外置 Target Encoding
- 融合仅用**预注册离散规则**（本交付 closest 为三臂 elementwise `max`）
- 禁止用 test 标签或伪标签抬本地分

---

## 2. 数据（训练 / 测试 / 提交模板）

全部位于 `data/`：

| 文件 | 行数 | 说明 |
|---|---:|---|
| `data/train.csv` | 14930 | 含 `label` |
| `data/test.csv` | 6398 | 无标签 |
| `data/submit_sample.csv` | 6398 | 提交模板 `id,label` |
| `data/SHA256SUMS.txt` | — | 校验和 |

```bash
cd data && sha256sum -c SHA256SUMS.txt
```

期望 SHA256：

- `train.csv` = `494a61073a0438f692914c4868db31df1171e662348e0024e06b120d08d44f28`
- `test.csv` = `d6ffd26bd4873fa09f6fac361f59170a880e88e331a01d7a6356bd9184ce55ec`
- `submit_sample.csv` = `83cb0263cc5729f61d0e05c68d673dc3f21b41c24bad68afa35159859054c4bf`

---

## 3. 环境

Python ≥ 3.10，建议：

```bash
python3 -m pip install -r requirements.txt
```

---

## 4. 一键复现本地 AUC（推荐，秒级）

不重新训练，直接用仓库内冻结 OOF 复算 B7 closest / fuse0 / B6：

```bash
PYTHONPATH=src python3 scripts/b7_recompute_closest.py
# 写出 artifacts/b7_audit/recompute_closest.json
# 关键字段 pass_recompute_lt_1e-8 应为 true
```

期望：

| 口径 | AUC |
|---|---:|
| B7 closest `max(gap,gap_bag,plus)` | **0.7027049552615718** |
| fuse0 pair nested `max(equal,plus)` | **0.7022093156561012** |
| B6 equal `0.5*(gap+gap_bag)` | **0.6989746962571622** |
| **公开榜**（同上提交） | **0.70722** |

生成提交（与已交付文件一致）：

```bash
PYTHONPATH=src python3 scripts/fuse_b7_closest.py
# → submissions/submission_b7_closest_honest.csv
```

---

## 5. B7 方案（完整）

### 5.1 三臂

1. **B6 gap**：在 B5 特征上加入 gap 挖掘类别交叉（ratio×geo、t3_sfx×code×days、w_pair…）；CatBoost；8 seeds `2026–2033`；5-fold  
2. **B6 gap_bag**：同 FE，`bagging_temperature=1.0` 等；同多种子  
3. **V10 plus**：保留 latent `x0–x18` 的 root_plus / H2；**10-fold × 4 seeds**；冻结于 `reference/v10/`

### 5.2 融合

```text
oof  = max(oof_gap, oof_gap_bag, oof_plus)
test = max(test_gap, test_gap_bag, test_plus)
```

嵌套折上 max 族候选 **5/5** 选三臂 `max`。  
更保守的二臂口径：`max(0.5*(gap+gap_bag), plus)` = 0.702209。

### 5.3 训练过程（可选重训）

> 权威交付分以**冻结 OOF 复算**为准。重训用于验证流程，因 early stopping / 线程等，数值可能有微小差异。

**A. 重训 B6 gap + gap_bag（约 0.5–1h，视 CPU）**

```bash
# 在仓库根目录；训练脚本默认读根目录 csv，先建软链
ln -sfn data/train.csv train.csv
ln -sfn data/test.csv test.csv
ln -sfn data/submit_sample.csv submit_sample.csv

PYTHONPATH=src python3 -m insurance_claim.train_b6 \
  --arms gap gap_bag \
  --fuse-arms gap gap_bag \
  --output-dir artifacts/b6_retrain
```

**B. plus 臂**  
本交付使用已冻结的 V10 plus（`reference/v10/oof_plus_h2_10.npz`）。  
若需重训 plus（耗时长）：

```bash
PYTHONPATH=src python3 -m insurance_claim.train_b7_plus \
  --output-dir artifacts/b7_plus_retrain \
  --seeds 2026 2027 2028 2029 \
  --folds 10
```

**C. 融合**  
用 `scripts/fuse_b7_closest.py`（默认读冻结路径；可用参数指向重训产物）。

---

## 6. 仓库结构

```text
data/                      比赛数据 + SHA256
src/insurance_claim/       B6/B7 训练与特征代码
scripts/
  b7_recompute_closest.py  独立复算本地 AUC
  fuse_b7_closest.py       从 OOF 生成提交
artifacts/
  b6_frozen/               B6 gap/gap_bag 冻结 OOF
  b7_closest/              B7 closest 打包 OOF + metrics
  b7_fuse0_b6/             二臂 fuse0 对照
reference/v10/             V10 plus 冻结 OOF/test
submissions/               应提交 CSV
docs/                      比赛/方案/审计（仅权威终稿）
```

---

## 7. 文档索引（已去冲突）

| 文档 | 用途 |
|---|---|
| `docs/B7_FINAL_REPORT.md` | B7 方案与结果终稿 |
| `docs/LESSONS.md` | 实战经验：有效/有坑/浪费时间/弱模型 |
| `docs/TRAINING.md` | 训练与复现步骤 |
| `docs/supervision/B7_RECOMPUTE.md` | 独立复算说明 |
| `docs/supervision/B7_FINAL_AUDIT_OPINION.md` | 门禁终审（REJECT 0.71） |
| `docs/supervision/B7_OVERFIT_CHEAT_AUDIT.md` | 过拟合/作弊检测 |
| `docs/b6/B6_FINAL_REPORT.md` | B6 冻结基线 |

**不要**再引用旧仓库里的 B5「最终方案」、B6/B7 计划草稿或实验日志中与终稿冲突的表述；以本仓库 `*_FINAL_*` 为准。

---

## 8. 许可与来源

代码与实验来自车险索赔建模迭代（B5→B6→V10→B7）。  
本地诚实主报 **0.702705**；同一提交 `submission_b7_closest_honest.csv` 的**公开榜 AUC = 0.70722**（约 +0.0045）。公开分勿冒充 CV。
