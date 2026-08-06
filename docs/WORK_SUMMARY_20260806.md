# 保险索赔 AUC 竞赛工作总结（2026-08-06）

## 1. 一句话结论

**`SUBMIT_GATE_0.72 = FAIL`（尚无约 0.72 的可信公开证据）。**  
若必须交 1–2 枪：优先交 V1 公开锚副本；条件第二枪交 V1+V7 固定等权 rank。  
**不交** MiniMax TE 搜权融合、Grok、V3、V4_quick、sem_plus / EBM / RealMLP / TabM。

---

## 2. 代码地址与分支（建议你拉取的版本）

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/wangxiaoming-wxm/dataModel |
| 分支 | `cursor/insurance-auc-model-dd73` |
| 基线 | `main` |
| 本文档对应提交 | `818645a`（及之后若有小补丁） |
| 克隆示例 | `git clone -b cursor/insurance-auc-model-dd73 https://github.com/wangxiaoming-wxm/dataModel.git` |

PR 描述已登记待人工批准创建（草稿），标题含「保险索赔 AUC：防泄漏流水线、门禁证据与最终提交候选」。

---

## 3. 最终建议你提交的文件名

目录：`final_candidates/`（相对仓库根目录）

| 优先级 | 文件名 | 本地 OOF | 公开预期 | SHA-256（文件） |
|---|---|---:|---|---|
| **第 1 枪（首选）** | `final_candidates/submission_1_safe_v1_anchor.csv` | 0.69295958 | ≈ **0.70236** | `3ce0e65e75ecae15ed06744eea00c2859c40c8ddb89c319197ae3529e4857245` |
| **第 2 枪（条件）** | `final_candidates/submission_2_upside_v1_v7_equal_rank.csv` | 0.70028365 | ≈ 0.70–0.71（无公开锚） | `31ed3c097162ed65f1ff6942df1d21791ea1959d77566838e66c5b42fe424a90` |

### 使用方式

1. 检出分支 `cursor/insurance-auc-model-dd73`。
2. 把上表文件原样上传到竞赛提交入口（不要改名内容；平台若要求固定文件名可仅改下载后的文件名，勿改 `label`）。
3. **只交 1 枪 → 只交 `submission_1_safe_v1_anchor.csv`。**
4. **交 2 枪 → 再交 `submission_2_upside_v1_v7_equal_rank.csv`。**

### 关键风险披露

- `submission_1` 的**预测值与已知第三名（公开约 0.70236）逐值相同**（maxabs=0）。  
  已知第三名 CSV SHA 为 `49ece575…`；本文件因浮点序列化写法不同，CSV SHA 为 `3ce0e65e…`，但**语义撞车**，可能被平台/对手识别为同方案。
- `submission_2`：固定 50/50 等权 rank（**无 OOF 搜权**），相对 V1 本地 +0.00732，bootstrap 95% CI `[0.00336, 0.01128]`；V7 缺完整训练源码 / 多 seed / 完整 shuffled，公开无锚。
- 二者均**未达到**门禁要求的约 0.72 可信证据；公开合理预期约 **0.70–0.71**。

机器可读元数据：`final_candidates/FINAL_CANDIDATES.json`。  
构建脚本：`scripts/build_final_candidates.py`。

---

## 4. 任务背景与硬约束

| 项 | 内容 |
|---|---|
| 任务 | 预测投保人一年内是否索赔（二分类） |
| 指标 | ROC-AUC |
| 数据 | `train.csv`（21328×45，正例约 10%）、`test.csv`（10000）、`submit_sample.csv` |
| 公开目标口述 | 冲前三约需 0.72+；#1≈0.7597，#2/#3≈0.72 |
| 硬约束 | 不过拟合、不作弊、不欠拟合；惜枪——无较可信达约 0.72 的证据时不建议正式冲榜提交 |
| 门禁要点 | 禁伪标签/半监督；禁分箱 TE；禁高基数交互 TE；集成禁 OOF 权重寻优；OOF>0.70 须多折切分 + shuffled |

交接来源：`main` 上的交接包与后续外部方案包（Grok / MiniMax）。

---

## 5. 全部工作时间线

### 5.1 本仓库早期基线

- 实现防泄漏流水线：`src/insurance_claim/`（CatBoost + XGBoost，嵌套早停，固定 50/50）。
- 严格嵌套 CV 约 **0.64873**，弱于交接 V1，后续以交接方案为续研主线。

### 5.2 交接复核（已验证事实）

| 对象 | 本地 OOF | 公开 | 备注 |
|---|---:|---:|---|
| V1 / 第三名语义 4-seed | **0.69295958** | ≈ **0.70236** | 预测与第三名相同；第三名文件 SHA `49ece575…` |
| V3（0.5×V1+0.5×V5 TE） | 诚实曾约 0.71343 | **0.69827** | TE/dropx 拖累，高估约 0.015 |
| V4_quick | 0.6921 | 未交 | 期望公开约 0.70；用户明确不交 |
| sem_plus seed2026 | 0.68268 | — | 相对 V1 −0.010，已止损 |

### 5.3 异构臂筛选（均未达 0.72）

| 新臂 | 结果 | 决策 |
|---|---|---|
| sem_plus | 相对 V1 −0.01028，CI 全负 | 淘汰 |
| EBM | OOF 0.64922；与 V1 等权 0.67725 | 淘汰 |
| RealMLP | 前两折约 0.600 | 提前淘汰 |
| TabM | 首折 0.57451 | 提前淘汰 |

证据：`docs/RESEARCH_GATE_20260806.md`、`artifacts/research_gate_20260806.json`、`scripts/rebuild_research_gate.py`。  
结论：`SUBMIT_GATE_0.72 = FAIL`；已删除易误交的旧 `artifacts/submission.csv`。

### 5.4 工程加固（审查后）

- 缓存绑定数据 / 标签 / schema / 配置 / 模型族 / 依赖 / 源码版本；NPZ 原子写入；family 隔离。
- RealMLP 显式早停；shuffled 双侧 `0.47–0.53`。
- 测试约 23 passed，覆盖率约 89%。

### 5.5 外部方案审计（独立子代理 + 本代理交叉）

| 包 | 诚实结论 | 是否提交 |
|---|---|---|
| Grok | 诚实 OOF **0.66165**；TE 非真正外层折内、无平滑、含高基数/交互目标统计风险；弱于 V1 | **不交** |
| MiniMax 主推 `0.55×V5+0.45×V3` | 报告 OOF 0.72081，但权重来自 **OOF 网格搜索（违反 R07）**；代数上放大 V5（≈0.775×V5）；与 V3 公开翻车路径同类 | **不交** |

### 5.6 最终候选生成与独立复核

- 生成 `final_candidates/` 两文件 + `FINAL_CANDIDATES.json` + README。
- 独立复核：格式对齐 `submit_sample.csv`；safe 与第三名预测 maxabs=0；构建脚本可字节级复现；门禁仍 FAIL。
- 已 `git commit` + `git push` 至分支 `cursor/insurance-auc-model-dd73`。

---

## 6. 明确不采用清单

| 候选 | 原因 |
|---|---|
| MiniMax `0.55×V5+0.45×V3` 及同包其他 TE 重混 | OOF 搜权（R07）；加重已翻车的 V5/TE；无合格新臂 |
| Grok `submission.csv` | OOF 0.66165；TE/分箱/高基数交互风险；证据不全 |
| V3 | 公开 0.69827，低于 V1 公开锚 |
| V4_quick | 无充分 OOF/公开/稳定性支撑占用名额 |
| sem_plus / EBM / RealMLP / TabM | 筛选阶段失败 |
| 任何 OOF 权重寻优集成 | 违反门禁 R07 |

---

## 7. 关键路径速查

```
仓库根/
├── final_candidates/
│   ├── submission_1_safe_v1_anchor.csv      ← 第 1 枪
│   ├── submission_2_upside_v1_v7_equal_rank.csv  ← 第 2 枪（条件）
│   ├── FINAL_CANDIDATES.json
│   └── README.md
├── scripts/
│   ├── build_final_candidates.py
│   └── rebuild_research_gate.py
├── docs/
│   ├── WORK_SUMMARY_20260806.md            ← 本文档
│   └── RESEARCH_GATE_20260806.md
├── src/insurance_claim/                    ← 防泄漏基线流水线
└── artifacts/research_gate_20260806.json
```

---

## 8. 推荐提交策略（操作清单）

1. 打开 https://github.com/wangxiaoming-wxm/dataModel/tree/cursor/insurance-auc-model-dd73/final_candidates  
2. 下载 `submission_1_safe_v1_anchor.csv` → **正式提交第 1 枪**。  
3. 若还有名额且接受更高风险：下载 `submission_2_upside_v1_v7_equal_rank.csv` → **第 2 枪**。  
4. 不要提交 MiniMax / Grok / V3 / V4_quick。  
5. 心理预期：公开约 **0.70–0.71**，**不是** 0.72 冲前三证据；第 1 枪存在与已知第三名撞车风险。

---

## 9. 门禁状态（最终）

| 门禁 | 状态 |
|---|---|
| `SUBMIT_GATE_0.72` | **FAIL** |
| 是否建议以冲前三为目标正式提交 | **否**（若仅消耗名额保底/试探，见第 8 节） |
| 是否已准备 1–2 个风险最优文件 | **是**（见第 3 节） |

---

*文档生成日期：2026-08-06。以分支 `cursor/insurance-auc-model-dd73` 上内容为准。*
