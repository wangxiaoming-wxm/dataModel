# B6pro 计划（诚实本地 OOF ≥ 0.715）

## 0. 硬约束

| 项 | 约定 |
|---|---|
| 基线分支 | 自 `origin/cursor/b6-push-auc070-a5f5` 拉出 **新分支** `cursor/b6pro-auc0715-100c` |
| 其他分支 | **不修改** B5/B6/B7 远程分支与其冻结交付 |
| B6 冻结 | `docs/b6_frozen/`、`artifacts/b6_frozen/`、`submissions/b6_frozen/` 只读引用 |
| 数据 | 仅新 `train.csv` / `test.csv` |
| 协议 | SKF≥5；主臂 seeds≥8；折内 FE；**无全局/外置 TE**；禁止连续 OOF 搜权 |
| 融合 | **预注册离散规则** + **嵌套折选**（继承 V10 诚实口径） |
| 目标 | 诚实本地 **≥ 0.715**；未达标只报 closest honest，不得包装 |

基线：

| 来源 | 口径 | 值 |
|---|---|---:|
| B6 | equal_prob(gap, gap_bag) ×8 | **0.69897470** |
| V10 参考 | nested max(B5-8, plus) | 0.70131497 |
| 公开榜（V10，仅背景） | public | 0.70570 |

缺口：`0.715 − 0.69897 ≈ 0.016`。同质 bagging 已耗尽；必须走 **异构臂 + 嵌套离散融合**，并继续挖残差。

---

## 1. 主路径（预注册）

```text
Arm_A = B6-class：gap / gap_bag（可 equal_prob 后再与异构融合）
Arm_B = plus_pro：保留 x0–x18 的 root_plus（H2+）；可注入健康 gap 猫交叉
Arm_C = 第三异构（可选）：XGB / Bernoulli-RSM / residual；须近强度且相关更低

B6pro = nested_select( pre-registered rules on (A,B[,C]) )
```

预注册离散规则（开跑前写死）：

**两臂核心集（V10 继承）：** `mean, mean_2_1, power2, power3, max, rank_mean`

**多臂扩展集（≥3 臂时启用，开跑前写死）：** 上述 + `geom_mean, min, median, mean_3_1_1`

嵌套：`StratifiedKFold(5, shuffle=True, random_state=42)` 在 train 段选规则 → 主报 **nested_oof_auc**。

---

## 2. 抬分阶梯

1. **Fuse0**：`nested_max(B6_equal, reference_plus)` 建立 ~0.702 地板（披露；正式交付须自训 plus）
2. **Plus 自训**：本分支重训 plus_pro（H2，≥4 seeds，5 或 10 折）
3. **Plus+gap 注入**：在 plus 上折内加入 B6 健康缺口猫（不破坏异构）
4. **第三臂**：低相关近强度源；等权/嵌套进融合
5. **残差切片挖掘**：在 max(A,B) 错分/midband 上找未吃满交叉 → 新臂
6. 循环直至 nested ≥ **0.715** 且红线全过

禁止：高 gap TE、测集伪标签、OOF 连续搜权、旧数据冒充、只报单折/单 seed。

---

## 3. 独立监督

协议：`docs/supervision/B6PRO_AUDIT_PROTOCOL.md`（IA-AUC715-B6PRO-v1）  
监督者不参与写抬分代码；仅审计 metrics / 红线；未达标不得签字 PASS。

---

## 4. 复现入口

```bash
# 自训 plus + 与 B6 臂嵌套融合（示例）
PYTHONPATH=src python3 -m insurance_claim.train_b6pro \
  --mode full --seeds 2026 2027 2028 2029 2030 2031 2032 2033 \
  --output-dir artifacts/b6pro_run

# 仅嵌套融合已有 OOF
PYTHONPATH=src python3 -m insurance_claim.train_b6pro --mode fuse \
  --main-npz artifacts/b6pro_main/predictions.npz \
  --plus-npz artifacts/b6pro_plus/predictions.npz \
  --output-dir artifacts/b6pro_fuse
```
