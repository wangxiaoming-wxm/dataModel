# B6pro 状态

## 结论（当前）

- B7 保底：本地 **0.702704955** / 公开 **0.707**
- **诚实 closest**：**0.70975596**（orig pick×blend3 → regime-HGB ultra patch → full-x HGB ultra patch）
- 距 0.71 缺口 ≈ **0.000244**
- ultra ≈ **0.6398**（原 0.631）；long 仍是主瓶颈
- 产物：`artifacts/b6pro_long_best/`、`artifacts/b6pro_honest_blend/`、`submissions/b6pro_closest/`

## 抬升轨迹

| 配方 | nested |
|---|---:|
| B7 max3 | 0.702705 |
| nest_div | 0.706020 |
| region_pick | 0.707824 |
| honest pick×blend3 | 0.708901 |
| HGB regime ultra patch (s26/27) | 0.709682 |
| **+ full-x HGB ultra patch** | **0.709756** |

## 业务关键洞见（本轮）

- ultra（days≥10k）内 **days–label 相关为负**（−0.037），全局 days 单调伤害 ultra 排序
- mid(3–7k) 车况斜率相对 short **翻转**；需 regime-slope 特征（cap/excess days、cond×band）
- 高杠杆：异构 HGB（corr≈0.63）对 ultra 做 **外层嵌套 α patch**，比再堆同构 CatBoost 有效
- 重 sample-weight / 窄切片专模 / 上下文 stack 未超过当前 closest

## 协议

未达 0.71 不宣称 PASS；nested < B7 则 fallback。α/权使用外层嵌套选择。
