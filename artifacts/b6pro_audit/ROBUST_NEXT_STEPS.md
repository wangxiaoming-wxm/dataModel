# B6pro 稳健下一实验（审计摘要）

**日期：** 2026-08-08  
**保底：** B7 本地 nested **0.702705** / 公开 **0.707**  
**已作废：** nodays single-seed ultra patch 本地 **0.710071** → 公开 **0.70208**

---

## 1. B7 为何能泛化

**配方（预注册、无连续搜权）：**

```text
elementwise_max(gap_8seed, gap_bag_8seed, plus_v10_h2_10)
```

嵌套选规则在 max 族上折折一致选中 `max`（`nested_select_rule`，SKF outer rs=42）。

| 臂 | 强度 (OOF) | 特征路径 | 产物 |
|---|---:|---|---|
| **gap** | 0.69868 | B5 enrich + 折内 `fit_gap_edges` → `GAP_CAT_COLS`（days/cond/ratio 分位 × region/source/code/…） | `artifacts/b6pro_frozen/arms/main_8seed.npz`；代码 `src/insurance_claim/train_b6.py::build_gap` + `b6_gap_features.py` |
| **gap_bag** | 0.69891 | 同 gap 特征；`bagging_temperature=1.0`, `random_strength=1.2` | 同上 `oof_gap_bag*` |
| **plus** | 0.68862 | V10 `root_plus` H2：keep x0–x18 + 折内 winsor/clip/业务交叉；**无 TE**；4 seed × 10 折参考 OOF | `reference/v10/oof_plus_h2_10.npz`；冻结副本 `artifacts/b6pro_frozen/arms/plus_ref_h2_10.npz`；代码 `v10_plus/plus_features.py` |

**结构要点：** 8-seed CatBoost 主臂 + 异构 plus（corr≈0.92，非 0.996 同质）+ **全局 elementwise max**（非小众切片高 α）。公开 − 本地 ≈ **+0.004**，非虚高。

复现：`docs/b6pro/FROZEN.md`；提交：`submissions/b6pro_closest/submission_SUBMIT_THIS.csv` ≡ `reference/b7_closest/`。

---

## 2. 本地高分但疑过拟合（勿再 promote / 勿交）

| 本地 nested | 配方 | 过拟合机制 |
|---:|---|---|
| **0.710071** | `scripts/b6pro_nodays_ultra.py` → `artifacts/b6pro_nodays_ultra/` / `b6pro_long_best` | **单 seed 2027**；仅在 **n=1806 ultra** 上外层嵌套搜 α（median≈0.3）；叠在已抬高的 closest 链上；公开 0.70208 |
| **0.7078–0.7097** | `b6pro_region_pick` / `region_blend2–3` / `post_stack*` / `hgb_regime` | **按 region 樱桃采摘** helper+α；或 **多层同标签 OOF 堆叠**；promoted 本地虚高，无公开验证 |
| **0.706–0.709** | `f09d_*` / `nest_div` / `nest_stack` / `ultra_weight` / `soft_ultra` 链 | 弱区/ ultra 连续或准连续 α；小样本切片抬整体 |
| **0.705+ keepx/long_*** | long_region / nodays_keepx / resid_* | 相对 B7 有边际，但多为 **单层 patch 叠 closest**；未过「≥4 seed + 重复外层」门 |

**红线（`docs/b6pro/STATUS.md`）：** 禁止单 seed 过线；禁止仅靠 n&lt;2k 高 α patch；未达稳健门 → fallback B7。

---

## 3. 下一实验方向（优先序）

1. **多 seed CatBoost 异构臂（≥4，建议 8）**  
   - 自训 plus_pro / keepx / lossguide / bag_hot，与冻结 gap/gap_bag **并列成臂**，勿先 patch closest。  
   - 入口：`PYTHONPATH=src python3 -m insurance_claim.train_b6pro`；`scripts/b6pro_fuse_npzs.py`；参考臂 `artifacts/b6pro_keepx8/`、`b6pro_xgb*`、`b6pro_hybrid8/`。

2. **预注册离散融合 + 嵌套选规则（禁连续 α）**  
   - 规则集写死：`mean, mean_2_1, power2, power3, max, rank_mean`（≥3 臂可扩 EXT，但须开跑前登记）。  
   - 实现：`src/insurance_claim/b6pro_fusion.py`；勿用 `nodays_ultra` 式 `linspace` α。

3. **重复外层嵌套（≥3 组 SKF `random_state`）**  
   - 报告 mean±std；单次过 0.71 不算可交付。  
   - 可薄封装：对同一组臂 OOF 循环 `nested_select_rule(..., random_state∈{42,43,44})`。

4. **异构哲学臂（非同质 bagging）**  
   - XGB / EBM / RealMLP-TabM：`scripts/b6pro_tabm.py`、`b6pro_tabpfn.py`、已有 `artifacts/b6pro_ebm/`、`b6pro_xgb_full/`。  
   - 目标：corr(main, hetero) ≪ 0.99，solo AUC 接近 0.69+，再 `nested_select` 进 max3/max4。

5. **明确避免**  
   - 单 seed；仅 ultra 高 α；region_pick 式 per-region 搜权；在已过拟合 closest 上再叠 patch。

---

## 4. 建议路径速查

| 用途 | 路径 |
|---|---|
| B7 SAFE 提交 | `submissions/b6pro_closest/submission_SUBMIT_THIS.csv` |
| B7 OOF/metrics | `reference/b7_closest/{predictions.npz,metrics.json}` |
| 冻结三臂 | `artifacts/b6pro_frozen/{predictions.npz,FREEZE.json,arms/}` |
| 公开失败记录 | `artifacts/b6pro_audit/PUBLIC_LB_FAIL.json` |
| 融合工具 | `src/insurance_claim/b6pro_fusion.py`，`scripts/b6pro_fuse_npzs.py` |
| 稳健协议 | `docs/b6pro/STATUS.md`，`docs/B6PRO_PLAN.md`，`docs/supervision/B6PRO_AUDIT_PROTOCOL.md` |
| 作废 closest | `artifacts/b6pro_long_best/`（`overfit_invalidated`） |

**验收：** 外层 nested ≥0.71 **且** ≥4-seed 等权臂 **且** ≥3 外层 seed 均值过线 **且** 非仅 ultra/小众切片贡献 → 才可替换 B7；否则交 B7。
