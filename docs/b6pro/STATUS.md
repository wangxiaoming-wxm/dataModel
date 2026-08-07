# B6pro 状态（已冻结 closest）

## 结论

- 软目标本地诚实 nested **0.705**：**未达到**
- 硬目标 **0.715**：**未达到**
- **冻结 closest honest nested OOF：`0.7027049552615718`**

## 冻结方案

| 项 | 值 |
|---|---|
| 路径 | `artifacts/b6pro_frozen/` |
| 提交 | `submissions/b6pro_frozen/submission_b6pro.csv` |
| 口径 | nested SKF(5) 选规则 → 主报 nested OOF |
| 规则 | **max**（五折一致） |
| 臂 | `gap_8seed` (0.698683) + `gap_bag_8seed` (0.698906) + `reference_plus_h2_10` (0.688617) |
| 相对 equal(main)×plus | +0.000496（0.702209 → **0.702705**） |
| 距 0.705 | ≈0.002295 |

## 披露

1. plus 为 V10 参考臂 bootstrap；自训 plus 更弱，正式若禁 bootstrap 需自训持平后再报。
2. `max` 利 AUC、不利概率校准。
3. 折内 early stopping。
4. 公开榜 0.70570（V10）**不是**本地 CV；本地诚实从未到 0.705。

## 已试未超过 closest 的方向（摘）

12-seed gap/gap_bag、bag_hot、hybrid、ordered、EBM、dropout、stack/meta-stack、regime、ultra/plus 变体等；最高仍卡在 ~0.7027。

## 监督

协议 `IA-AUC715-B6PRO-v1`：未达 0.715 → 不得 PASS；本冻结仅认证 **closest honest**。
