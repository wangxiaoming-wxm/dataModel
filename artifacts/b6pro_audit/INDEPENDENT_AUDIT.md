# B6pro 独立审计报告

- 审计角色：独立监察员 / auditor
- 审计范围：只核实现有仓库证据、公开榜反馈与 promote 风险；不训练、不调参、不刷本地分。
- 审计分支：`cursor/b6pro-auc0715-100c`
- 结论时间：2026-08-08 UTC

## 一句话结论

当前 B6pro 的本地 `0.710071` 不是可 promote 的诚实 0.71。它已经被公开榜 `0.70208` 反证，且低于已验证 B7 公开 `0.707`。在新的反过拟合门禁执行前，唯一安全提交是 B7：

- `reference/b7_closest/submission_b7_closest_honest.csv`
- `submissions/b6pro_closest/submission_SUBMIT_THIS.csv`

两者与 `submissions/b6pro_closest/submission_B7_SAFE_public0707.csv` 的 SHA256 完全一致：

```text
5c9ccfdaaed914c92e153cb9ba2b0fb4e066b462b09406184d3f4ed1e75c1de8
```

## 1. PASS 宣称核查

### `artifacts/b6pro_long_best/metrics.json`

核查结果：主入口已不再宣称 PASS。

证据：

- `spec`: `direct_nodays_ultra_patch_s2027`
- `nested_oof_auc`: `0.7100714803766324`
- `gate_0_71`: `false`
- `public_lb_observed`: `0.70208`
- `overfit_invalidated`: `true`
- `note`: 明确写明该本地 0.710071 已失效，公开榜低于 B7，禁止提交，应使用 B7 SAFE。

审计判断：该文件已更正为“记录历史本地分 + 标记失效”，不是 PASS 入口。但它仍保留 `nested_oof_auc > 0.71` 和负的 `gap_to_0_71`，后续自动脚本不得只看分数字段，必须同时检查 `overfit_invalidated != true`、`gate_0_71 == true` 和本协议门禁。

### `docs/b6pro/STATUS.md`

核查结果：状态文档已不再宣称 PASS。

证据：

- 明确写明“过拟合版本本地宣称 0.710071，提交后公开榜 0.70208”。
- 明确写明“低于 B7 公开 0.707，该版本作废，禁止再交”。
- 当前可提交指向 `submissions/b6pro_closest/submission_SUBMIT_THIS.csv` 和 `reference/b7_closest/submission_b7_closest_honest.csv`。
- 已提出多 seed、重复外层 CV、禁止依赖 n<2k 切片高 alpha patch 等条件。

审计判断：`STATUS.md` 的方向正确，无需本次修改。

### 二级指标中的残留风险

虽然用户点名的两个主入口已更正，但链路内仍有历史本地过线记录，不能被后续脚本误用：

- `artifacts/b6pro_nodays_ultra/metrics.json`: `nested=0.7100714803766324`, `gate=true`, `alpha=0.30000000000000004`
- `artifacts/b6pro_honest_blend/metrics.json`: `best=direct_nodays_ultra_patch_s2027`, `nested=0.7100714803766324`, `gate=true`

审计判断：这些是已被公开榜反证的历史本地 gate，不得作为 promote 依据。主入口已隔离，但任何候选扫描器如果递归读取所有 `metrics.json`，必须把这些文件加入 quarantine / denylist，或强制读取 `ANTI_OVERFIT_PROTOCOL.json` 的失效规则。

## 2. 过拟合配方链路核查

### A. pick x blend3 / 多切片挑选

证据：

- `artifacts/b6pro_region_blend3/metrics.json`
  - `nested`: `0.7075959692821868`
  - `best`: `direct_lm_all_i0.001`
  - `chosen` 对多个 region 选择不同 alpha 和模型。
- `artifacts/b6pro_region_pick/metrics.json`
  - `nested`: `0.707823512693071`
  - `best`: `direct_pick_all_i0.001`
  - `chosen` 覆盖 15 个 region，每个 region 在 `mlp/ebm/xgb/flaml/resid/lm/...` 等臂和 alpha 间挑选。
- `artifacts/b6pro_post_stack3/metrics.json`
  - top 中出现 `honest_pick_blend3`、`honest_pick_blend2`、`mean_pick+blend2+ps2`、`logit_pick+blend2+ps2_C0.05` 等后续组合。

审计判断：该链路存在明显的多重比较风险。虽然部分步骤带有“honest/nested”字样，但同一 OOF 标签面上连续做 region 级 pick、blend、post-stack、top 候选排序，会把局部噪声逐层固化。它不能独立证明泛化提升。

### B. regime / ultra patch

证据：

- `artifacts/b6pro_loop/regime.log`
  - 4 seeds 的 regime helper OOF 约 `0.686853` 到 `0.692154`，solo 并不强。
  - `patch_ultra`、`patch_long`、`patch_midcond` 多个切片 patch 同时尝试。
- `artifacts/b6pro_loop/hgb_regime.log`
  - `patch_ultra 0.708951...`，alpha 中位数 `0.3`，fold alpha 为 `[0.1, 0.3, 0.3, 0.3, 0.1]`。
  - 同时尝试 `patch_long`、`patch_midcond`、`patch_ultra_or_mid`、`seq_ultra_mid`。
- `artifacts/b6pro_hgb_regime/metrics.json`
  - 记录了 `nested=0.7096815267988718`、`best=s26_27_d6`，仍低于 0.71。

审计判断：regime 系列说明团队在公开失败前后持续围绕 ultra/long/midcond 切片寻找局部 lift。即便单个 helper 使用外层切分，多个 helper、多个切片、多个 alpha、多个 seed 范围和多个结构被串行尝试后，必须按整个搜索过程而不是单次脚本来评估过拟合风险。

### C. xall / 全特征

核查结果：未检出独立命名为 `xall` 的脚本或工件。

相关证据：

- `scripts/b6pro_nodays_ultra.py` 使用 `x0` 到 `x20` 全部 embedding / 数值列，并额外加入 `is_ultra`、`is_long`、`cond`、`invc`。

审计判断：不能凭仓库证据断言存在独立 `xall` 配方。但可以确认 nodays ultra helper 使用了全量 `x0..x20` 特征并与 ultra patch 绑定；后续报告不得把“xall 已验证有效”作为事实宣称。

### D. MLP / 非预注册融合规则

证据：

- `artifacts/b6pro_mlp/metrics.json`
  - `oof_auc`: `0.6104292355417136`
  - `nested_oof_auc`: `0.69807288196045`
  - `selected_rule`: `mean_3_1_1`
  - `full_data_scores` 中同时比较 `mean`、`mean_2_1`、`power2`、`power3`、`max`、`rank_mean`、`geom_mean`、`min`、`median`、`mean_3_1_1`。
- `docs/supervision/B6PRO_AUDIT_THRESHOLDS.json` 预注册规则只有 `mean`、`mean_2_1`、`power2`、`power3`、`max`、`rank_mean`。

审计判断：MLP 本身没有接近 0.71，且 `mean_3_1_1` 不在预注册规则内。任何依赖非预注册规则或事后扩展规则的提升，均不得 PASS。

### E. nodays ultra patch / 单 seed 2027

证据：

- `scripts/b6pro_nodays_ultra.py`
  - 明确写死 `seed = 2027`。
  - `spec` 写入 `direct_nodays_ultra_patch_s2027`。
  - 使用 ultra 子集 `days >= 10000` 做 alpha patch。
  - alpha 网格为 `np.linspace(0, 1, 21)`。
  - alpha 选择只在 ultra 子集内做 5 折，最终用 fold alpha 中位数应用到 test ultra。
- `artifacts/b6pro_nodays_ultra/metrics.json`
  - `best`: `s2027`
  - `nested`: `0.7100714803766324`
  - `alpha`: `0.30000000000000004`
- 公开榜反馈：
  - 本地 `0.7100714803766324`
  - 公开榜 `0.70208`
  - B7 公开 `0.707`

审计判断：这是本次失败的核心链路。单 seed 2027 + ultra 小样本 alpha patch + 前置多轮候选筛选共同形成过拟合。公开榜已经给出反证，不能再以“nested OOF 已过 0.71”提交。

## 3. ultra 切片 n 与嵌套 alpha 风险

从 `train.csv` / `test.csv` 直接核查：

```text
train n = 14930
train ultra(days >= 10000) n = 1806
train ultra positives = 216
train ultra negatives = 1590
test ultra(days >= 10000) n = 805
train long(days >= 3000) n = 9787
test long(days >= 3000) n = 4247
```

审计判断：

1. ultra 训练样本只有 1806，其中正样本 216。对 AUC 排序来说，单折内正样本更少，alpha 的方差很高。
2. `scripts/b6pro_nodays_ultra.py` 在该切片上比较 21 个 alpha，并且这一步发生在前面已经经历 region pick、blend、post-stack、regime patch 等多轮选择之后。
3. 单次 ultra 内层“nested alpha”不能抵消跨脚本、跨切片、跨 seed、跨候选的全局多重比较。
4. alpha 中位数 `0.3` 被用于 test ultra，但没有重复外层 CV 证明 `0.3` 在不同 outer split、不同 helper seed、不同候选冻结点上稳定。

结论：ultra alpha 属于小样本上过寻优的高风险 patch。除非按新协议通过重复外层、multi-seed、稳定性和 B7 相对检查，否则不得 promote。

## 4. 公开榜诚实性判断

本地到公开的偏差：

```text
B6pro overfit local = 0.7100714803766324
B6pro public        = 0.70208
local - public      = 0.007991480376632332
B7 local            = 0.7027049552615718
B7 public           = 0.707
```

审计判断：

- B7 是“本地略低、公开略高”的已验证安全信号。
- B6pro nodays ultra 是“本地过线、公开低于 B7”的反向信号。
- 因此当前不能把 B6pro 0.710071 解释为 leaderboard variance；它应被视为 promote 失败和过拟合证据。

## 5. 当前主要风险清单

1. **单 seed 过线风险**：最终过线配方硬编码 seed 2027，缺少等权多 seed 和 seed 间稳定性。
2. **小样本切片过寻优风险**：ultra 仅 n=1806 / 正样本 216，却在 21 个 alpha 上择优，并继承前置搜索偏差。
3. **多层 OOF 堆叠风险**：region pick、blend3、post-stack、ctx-stack、regime patch 等在同一标签面上串行筛选。
4. **规则事后扩展风险**：MLP 和若干 fusion 记录中出现非预注册规则，如 `mean_3_1_1`、`median`、`geom_mean`、`min`。
5. **指标扫描误用风险**：二级 artifacts 仍有 `gate=true` 的历史失败记录，自动 promote 若只看 `nested` 和 `gate` 会误判。
6. **公开榜反证未入硬门禁风险**：必须把 public 0.70208 < B7 public 0.707 写入 denylist，而不是作为备注。
7. **相对 B7 不稳定风险**：B6pro 与 B7 的本地/公开关系反向，任何新候选必须证明相对 B7 的稳定提升，而不是只在少数行产生分布漂移。

## 6. promote 门禁条文摘要

详见 `artifacts/b6pro_audit/ANTI_OVERFIT_PROTOCOL.json`。摘要如下：

1. **B7 fallback 固定**：未全量通过时，唯一可提交文件为 `reference/b7_closest/submission_b7_closest_honest.csv` 或其同哈希拷贝 `submissions/b6pro_closest/submission_SUBMIT_THIS.csv`。
2. **禁止单 seed 过线**：任何候选不得因单个 seed 或单个 outer split 达到 0.71 而 promote。
3. **主模型至少 4 个等权 seed**：所有进入最终提交的新增训练臂必须报告 per-seed OOF、均值、标准差、最差 seed。
4. **重复外层 CV**：至少 3 组不同 outer `StratifiedKFold random_state`，每组完整重跑选择流程；报告均值、最差组、组间方差。
5. **alpha / 权重预注册**：alpha 网格、切片定义、融合规则必须在运行前写入 manifest；未预注册的规则或切片搜索只能进入探索日志，不能 promote。
6. **ultra 小样本限制**：n<2000 的切片 patch 不得单独提供最终过线贡献；必须证明 overall、non-ultra、ultra 三者均不退化，且 lift 不主要来自高 alpha 小切片。
7. **B7 相对稳定性检查**：新候选在每个 outer repeat 上必须相对 B7 提升稳定，且 test 分数与 B7 的相关性、均值、标准差、最大单行偏移必须报告。
8. **公开榜反证 denylist**：`direct_nodays_ultra_patch_s2027`、`b6pro_nodays_ultra`、历史 `0.710071` 本地 PASS 均标记为 invalidated，不得重新包装提交。
9. **自动 promote 必须读协议**：不得只读任一 `metrics.json` 的 `gate=true`；必须同时检查 denylist、重复外层、多 seed、相对 B7 和审计签名。

## 7. 最终审计意见

**REJECT / QUARANTINE B6pro 0.710071。**

当前没有证据证明 B6pro 已不过拟合地真正达到 0.71。当前唯一安全提交是 B7：

```text
reference/b7_closest/submission_b7_closest_honest.csv
submissions/b6pro_closest/submission_SUBMIT_THIS.csv
```

任何新 B6pro 候选必须按 `ANTI_OVERFIT_PROTOCOL.json` 重新产出完整证据包后，才允许进入 promote 审核。
